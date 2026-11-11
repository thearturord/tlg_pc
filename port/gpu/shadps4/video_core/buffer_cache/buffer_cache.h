// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/half_phase.h"
#include <absl/container/btree_map.h>
#include "native/native_buffer_index.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_ring_mirror.h"
#include "native/native_resources.h"
#include "native/native_readback_policy.h"

#include <condition_variable>
#include <mutex>
#include <vector>

#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "common/hash_u64.h"
#include "common/interval_set.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/page_clock.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace Core {
class NativeExecutor;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class ReadbackBatch;

class BufferCache {
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, Core::NativeExecutor* executor,
                         TextureCache& texture_cache);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the stream buffer (the record side's ring: constants, tables, the indexed rows).
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }
    /// Plan S step P3d: the resolve side's ring. With TLG_SPLIT_BIND=1 every copy `BufferCache` makes of small guest buffers goes into this one, so the resolver never writes
    /// the ring the record side writes; otherwise it is the same ring as before.
    [[nodiscard]] StreamBuffer& CopyRing() noexcept {
        return VideoCore::ResolveCommands::SplitMode() == 1 ? resolve_stream_buffer : stream_buffer;
    }
    [[nodiscard]] const StreamBuffer& CopyRing() const noexcept {
        return VideoCore::ResolveCommands::SplitMode() == 1 ? resolve_stream_buffer : stream_buffer;
    }
    /// Either ring: host written, read by the GPU only (no GPU hazard to track).
    [[nodiscard]] bool IsStreamBuffer(const Buffer* buffer) const noexcept {
        return buffer == &stream_buffer || buffer == &resolve_stream_buffer;
    }
    [[nodiscard]] bool IsStreamHandle(vk::Buffer handle) const noexcept {
        return handle == stream_buffer.Handle() || handle == resolve_stream_buffer.Handle();
    }

    /// The block granularity the shader recompiler's raw address access used (that access is off: TLG_DIRECT_MEMORY_ACCESS)
    u32 GetSparsePageShift() const noexcept {
        return 14;
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// O1 / C6 step 1 (`TLG_STREAM_IN_PASS=0` turns it off): a small read-only request while a render pass is open, whose bytes the GPU did not write, is served
    /// from the stream buffer, so an upload does not cut the pass
    [[nodiscard]] bool StreamWhilePassOpen(VAddr device_addr, u32 size);
    std::pair<const Buffer*, u64> StreamCopy(VAddr device_addr, u32 size);

    /// N5c: uploads the CPU-changed pages of a range the engine announced (before the draw that uses it), outside any render pass. Returns the bytes copied.
    u64 PreSynchronize(VAddr device_addr, u64 size);

    /// Finds a buffer for the specified region.
    /// immutable_hint (N3c): the immutable engine buffer the caller resolved by the engine's id (null: found by the address).
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false,
                                                             Core::NativeResources::ImmutableBuffer* immutable_hint = nullptr,
                                                             bool verify_hint = false, bool known_no_object = false);
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferBody(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer, Core::NativeResources::ImmutableBuffer* immutable_hint,
                                                                 bool verify_hint, bool known_no_object);
    /// Binding memo (raster thread): a read-only request of at most BindMemoMax bytes asked again in the same command buffer, with nothing written to its memory
    /// since and nothing moved, gets the same answer without the lookups and synchronization (TLG_BIND_MEMO, default on; =0 off)
    static constexpr u64 BindMemoMax = 65536;
    /// D4 (`TLG_BIND_MEMO_WIDE`, see BindMemoWideMode in the .cpp): read-only texel buffers and read-only requests up to this size use the memo too
    static constexpr u64 BindMemoWideMax = u64{1} << 17; // 128 KiB: an erase looks back this far in the tree, a 16 MiB bound made every erase walk thousands of entries (measured 1.2 ms per frame)
    static constexpr size_t BindMemoMaxAddresses = 200000; ///< the memo is cleared at a new command buffer when it holds more addresses (stale copy entries are no longer walked away)
    void BindMemoErase(VAddr address, u64 size);
    /// Design review 85: [address, address + size) is wholly in the GPU-written set (a written binding needs no new mark: the address readers already see it).
    [[nodiscard]] bool GpuWrittenContains(VAddr address, u64 size);

    /// Draw recipes (verify stage, raster thread only): the resolved shape of a graphics draw by the identity of its record, with the guest ranges its buffers came from. A recipe
    /// is predicted valid while no write overlapped one of its ranges (the invalidation the bind memo uses), the epochs are the same, and (when it names a copy in the stream
    /// buffer) the command buffer is the same. Nothing is reused yet: the caller resolves as usual and compares.
    struct RecipeEpochs {
        u64 memo{}, range{}, placement{}, immutable{}, image{};
        bool operator==(const RecipeEpochs&) const = default;
    };
    [[nodiscard]] RecipeEpochs RecipeEpochsNow(u64 immutable_epoch, u64 image_generation) const noexcept {
        return {bind_memo_epoch.load(std::memory_order_acquire), range_epoch.load(std::memory_order_acquire), PlacementEpoch(), immutable_epoch, image_generation};
    }
    /// 0 no recipe for the key, 1 predicted valid, 2 predicted invalid (`shape` is the stored one in both of the last cases).
    [[nodiscard]] u32 RecipePredict(u64 key, u64 tick, const RecipeEpochs& now, u64& shape);
    void RecipeStore(u64 key, u64 shape, const boost::container::small_vector<std::pair<VAddr, u32>, 8>& ranges, bool uses_stream, bool reusable, u64 tick, const RecipeEpochs& at_start);
    void RecipeCount(u32 index) noexcept {
        recipe_counts[index].fetch_add(1, std::memory_order_relaxed);
    }
    /// Plan V, step V0 (`TLG_VERSION_CENSUS=1`, shadow only): the same recipes checked against the identity events (native_identity_log.h) that overlap THEIR ranges, instead of the global counters.
    /// Content writes (`BindMemoErase` -> `RecipeEraseRange`) do not make a recipe invalid here; they are counted (`SvValidContentDirty`).
    enum ScopedIndex : u32 {
        SvChecked, SvNoRecipe, SvNotReusable, SvIdentityChanged, SvLogUnknown, SvValid, SvValidSame, SvValidStreamOffsetOnly, SvValidOtherDifference, SvValidContentClean,
        SvValidContentDirty, SvValidUsesStream, SvValidStreamOtherCommandBuffer, SvInvalidButSame, SvInvalidAndDifferent, SvNoImageRanges,
        SvKindBase = 16, // + the kind of the event that changed the recipe (native_identity_log.h), up to 8
        SvNsValid = 24, SvNsInvalid, SvNsNoRecipe, SvNsAll, SvValidPathSwitch, SvValidDifferentSameKinds, SvCountsEnd = 40,
    };
    struct ScopedPrediction {
        u32 state{0}; ///< 0 no recipe, 1 valid, 2 not reusable, 3 identity changed, 4 log overrun
        u64 shape{}, shape_nostream{}, kind_signature{};
        bool content_clean{false}, uses_stream{false}, other_command_buffer{false};
    };
    [[nodiscard]] ScopedPrediction RecipePredictScoped(u64 key, u64 tick);
    void RecipeStoreScoped(u64 key, u64 shape_nostream, u64 kind_signature, const boost::container::small_vector<std::pair<VAddr, u32>, 4>& image_ranges, u64 identity_clock);
    void ScopedCount(u32 index, u64 n = 1) noexcept {
        scoped_counts[index].fetch_add(n, std::memory_order_relaxed);
    }
    void BindMemoBump() noexcept {
        bind_memo_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    [[nodiscard]] nlohmann::json BindMemoReport() const;

    /// The arena range of a copy destination that is downloaded in stream order (an explicit readback): the CPU's pending writes are
    /// uploaded, the range is not marked GPU modified.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForDownload(VAddr device_addr, u32 size);

    /// The largest copy RecordDmaDownload downloads.
    static constexpr u32 MaxDmaDownload = 4_MB;

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);
    /// F1 class 2 static rows: [device_addr, +size) of engine buffer `id` is uploaded from guest memory whole into the buffer's device memory (its placed buffer or the arena), so a row can
    /// name it; the bytes uploaded are returned in `uploaded`. The version of the buffer is not bumped (its content did not change).
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainStatic(u64 id, VAddr device_addr, u32 size, u64& uploaded);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    /// The most a read-only buffer may be to be copied through the stream buffer (and the constant ring of the rasterizer).
    [[nodiscard]] static constexpr u64 StreamThreshold() noexcept {
        return STREAM_THRESHOLD;
    }
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);
    /// E4: memory no engine buffer answers for: changed when announced bytes of it were not uploaded yet
    [[nodiscard]] bool NativeCpuQuery(VAddr addr, size_t size);
    /// What a CPU write of [address, address + size) means for the caches (the harvest's replay; B2: an applied announcement)
    void ApplyCpuWrite(VAddr address, u64 size, bool newer = false, u64 object = 0);
    /// The full apply of an announced range (every step); `ApplyCpuWrite` calls it, or the marks by id for tags 2 and 12 (`TLG_ID_MARKS`)
    void ApplyCpuWriteBody(VAddr address, u64 size, bool newer, u64 object);
    void ApplyCpuWriteFull(VAddr address, u64 size, bool newer, u64 object);
    void ApplyIdMark(VAddr address, u64 size, bool newer, u64 object, u32 mode);
    void IdMarkLean(VAddr address, u64 size, bool newer, u64 id);
    /// TLG_ANNOUNCE_SKIP: may an announced write with the same bytes as the last announcement of these bytes be left unapplied? Not when the GPU wrote them since or an image covers them.
    [[nodiscard]] u32 CanSkipIdenticalWrite(VAddr address, u64 size);
    /// L1: may these announced pieces be applied as the one range [begin, begin + size)?
    [[nodiscard]] bool CanMergeAnnounced(u64 object, VAddr begin, u64 size, const Core::NativeUpdates::AnnouncedPiece* pieces, size_t count);
    /// B9.2: the engine deleted the buffer `id` at [address, address + bytes) (raster thread, in order): its announced bytes go; in native mode also the GPU written
    /// bytes and stream copies of its memory, unless another live buffer shares the memory
    void NoteEngineBufferDelete(u64 id, VAddr address, u64 bytes, bool native);
    /// B9.3: the engine buffer that holds all of [addr, addr + size) (0: none)
    [[nodiscard]] u64 BufferHolding(VAddr addr, size_t size) const;
    /// B9.4: a new engine resource at [address, address + bytes): placed buffers under it move back into the arena
    void NoteEngineResourceCreate(u64 id, VAddr address, u64 bytes, bool texture, u32 usage = 0, u32 cpu_access = 0);
    /// B9.1: [address, address + size) changed on the CPU: each engine buffer it meets keeps its part, the rest waits for the first buffer upload over it. Returns the
    /// parts no buffer holds (B9.3 step 2: what the page tracker still marks)
    void MarkBuffersChanged(VAddr address, u64 size, bool announced, std::vector<std::pair<VAddr, VAddr>>* gaps, const Core::NativeDevice::BufferPart* told = nullptr, bool use_told = false);
    /// R1: changes when an arena is replaced
    [[nodiscard]] u64 PlacementEpoch() const noexcept {
        return placement_epoch.load(std::memory_order_acquire);
    }
    /// R1b: the object's own copy in the stream ring for [device_addr, device_addr + size) when it is current (no copy is made); null when not
    [[nodiscard]] const Buffer* ObjectCopy(const Core::NativeResources::ImmutableBuffer& object, VAddr device_addr, u64& offset) const noexcept {
        if (object.copy_offset == ~u64{0} || object.copy_generation != object.generation.load(std::memory_order_acquire) ||
            object.copy_gpu_writes != object.gpu_writes.load(std::memory_order_acquire) || object.copy_wraps != CopyRing().WrapCount()) {
            return nullptr;
        }
        offset = object.copy_offset + (device_addr - object.begin);
        return &CopyRing();
    }
    /// R1b switch (`TLG_OBJECT_COPIES=0` turns it off)
    static bool ObjectCopiesOn();
    /// N5e: byte precise. True when a shader or a GPU copy wrote bytes of [addr, addr + size) that were not downloaded since (the set `gpu_modified_ranges`), as against
    /// `IsRegionGpuModified`, which answers per 4 KB page and keeps the answer for as long as the page lives.
    [[nodiscard]] bool IsRangeGpuWritten(VAddr addr, size_t size);
    /// F1 1b real (TLG_HOOK_RING_1B=2): the rasterizer's hook ring buffer, bound for frame ring ranges the hook ring holds whole
    void SetHookRing(const Buffer* buffer) noexcept {
        hook_ring = buffer;
    }
    [[nodiscard]] bool IsHookRing(const Buffer* buffer) const noexcept {
        return buffer != nullptr && buffer == hook_ring;
    }
    /// N5h: new contents were written into the range by the CPU (a creation, an initialization, a Map / Unmap): what the GPU wrote there before is gone.
    void SupersedeGpuWritten(VAddr addr, size_t size);

    void SetSideUploadEpoch(u64 epoch) {
        side_upload_epoch = epoch;
    }

    /// True once if downloads for the priority thread were recorded since the last call (the
    /// command buffer must reach the GPU soon).
    [[nodiscard]] bool TakeAsyncRecorded() {
        return async_recorded.exchange(0, std::memory_order_acq_rel) != 0;
    }

    /// A guest memory write by the command processor (WriteData, occlusion results, DMA...).
    /// Drops reusable stream copies it overlaps.
    void NoteCpGuestWrite(VAddr device_addr, u64 size) {
        if (Core::RingMirror::ShadowOn()) {
            Core::RingMirror::NoteHostWrite(device_addr, size);
        }
        BindMemoBump(); // the command processor's writes (any thread): every memo answer is checked again
        if (StreamReuseOn()) {
            InvalidateStreamReuse(device_addr, size);
        }
    }

    /// readback_dma_async: copies [dst, dst + size) of the arena into a private staging buffer; the
    /// priority thread puts the bytes into guest memory once the GPU executed the copy.
    void RecordDmaDownload(const Buffer* arena, VAddr dst, u32 size, bool tracked = false);

    /// Waits until the explicit downloads (RecordDmaDownload with `tracked`) over the range have landed; 0 nothing in flight, 1 waited, 2 timed out.
    u32 WaitForExplicitDownloads(VAddr device_addr, u64 size, u32 timeout_ms);

private:
    const Buffer* hook_ring = nullptr; ///< F1 1b real
    /// Drops the reusable stream copies if a guest memory write overlaps them.
    void InvalidateStreamReuse(VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    /// Bytes the last SynchronizeMemory of this thread copied from guest memory (resource census).
    static inline thread_local u64 last_upload_bytes{};
public:
    /// D4 sizing: the upload counter of this thread, to attribute bytes to the binding kind that caused them.
    static void UploadBytesReset() noexcept { last_upload_bytes = 0; }
    [[nodiscard]] static u64 UploadBytes() noexcept { return last_upload_bytes; }
    /// Plan S step P1d: a synchronization copy decided now and recorded later. The staging bytes are read from guest memory when the copy is decided (as before);
    /// only the copy command (and its barrier work) waits for `RunDeferredUploads`. Active on this thread while a sink is set.
    struct DeferredUpload {
        VideoCore::Buffer* staging;
        const Buffer* arena;
        boost::container::small_vector<vk::BufferCopy, 4> copies;
    };
    using UploadSink = boost::container::small_vector<DeferredUpload, 4>;
    static void SetUploadSink(UploadSink* sink) noexcept { t_upload_sink = sink; }
    void RunDeferredUploads(UploadSink& sink);
    static inline thread_local UploadSink* t_upload_sink{nullptr};
    /// True when the last ObtainBufferForImage of this thread read GPU written memory (the data came from the arena, not from guest memory).
    static inline thread_local bool last_image_source_gpu{};
private:

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    Core::NativeExecutor* executor;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::atomic<u64> cpu_epoch{1}; ///< bumped when a CPU change is applied or announced bytes are uploaded (the upload memo's answers are stale)

    StreamBuffer stream_buffer;
    StreamBuffer resolve_stream_buffer; ///< plan S P3d: see CopyRing
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    GpuRangeSet gpu_modified_ranges;
    /// TLG_GPUWRITE_WHO=1 (diagnostic): the draw that last wrote each GPU written range, by start address (guarded by gpu_ranges_mutex)
    std::map<u64, std::pair<u64, std::string>> gpu_writer_who;
    /// TLG_GPUWRITE_WHO=1: placed blocks dropped while they held GPU written bytes: start -> {end, text} (guarded by gpu_ranges_mutex)
    std::map<u64, std::pair<u64, std::string>> dropped_with_gpu_bytes;
    /// B9.1: the bytes of each engine buffer (by its id) the engine announced since their last upload
    std::unordered_map<u64, StampedRangeSet> object_dirty;
  public:
    /// W3.1 shadow: does engine buffer `id` hold announced bytes in [addr, addr + size) that no upload took yet, or are there bytes no engine buffer holds there
    [[nodiscard]] bool ObjectRangeDirty(u64 id, VAddr addr, u64 size);
    /// S3b.2 real: the same answer for the row of `buffer`, from the per-buffer clock (lock-free); one call in 64 is also asked of the sets, and where the sets say dirty the answer is dirty.
    [[nodiscard]] bool ObjectRangeDirtyClock(u64 id, const Buffer* buffer, VAddr addr, u64 size);
  private:
    StampedRangeSet all_object_dirty; ///< B9.1 audit: the same bytes of all buffers by address
    /// B9.4: the Vulkan buffer of [address, address + size): a placed engine buffer's own when the range is inside one (placing it at its first use), else the arena
    /// (resident). `demote_spanning`: a range over a placed buffer's edge moves that buffer back into the arena first; false: such a range gets nullptr
    const Buffer* BufferFor(VAddr address, u64 size, bool demote_spanning = true, bool written = false);
    /// Design review 85: the engine buffer `id`'s `cpu_dirty` flag after its dirty set changed (`object_dirty_mutex` held; `added`: an add, else a subtraction).
    void OwnDirtyNote(u64 id, bool added);
  public:
    /// Design review 87: the placed block holding the whole of engine buffer `h` becomes its own memory (`own`, `own_new`), with `cpu_dirty` taken from its dirty set;
    /// null when no placed block holds it (then not asked again until the placements change).
    const Buffer* OwnResolve(Core::NativeResources::ImmutableBuffer& h);
    /// Note 117: the placed block that holds [address, address + size) whole, without its slack, or null (raster thread).
    [[nodiscard]] const Buffer* PlacedWhole(VAddr address, u64 size) {
        const PlacedBuffer* entry = PlacedAt(address, size, true);
        return entry != nullptr ? entry->buffer.get() : nullptr;
    }
  private:
    static constexpr u64 PlacedSlack = 64; ///< B9.4: a read may run this far past a placed buffer's end (vertex fetch rounding); the bytes there are not memory's
    struct PlacedBuffer {
        std::vector<u64> ids; ///< the engine buffers it holds (one, or a block the engine laid out next to each other and binds as one range)
        std::vector<u64> users; ///< the resident objects (engine ids) whose binding resolution names this buffer: all are reset when it is freed
        VAddr begin{}, end{}; ///< the memory it holds (the Vulkan buffer starts at begin aligned down to 256: the same offset alignment as the arena)
        std::unique_ptr<Buffer> buffer;
    };
    std::map<VAddr, PlacedBuffer> placed;
    std::atomic<u64> placed_version{1}; ///< bumped on every insert and erase of `placed` (PlacedAt keeps its last answer until then)
    struct BindMemoEntry {
        u32 size;
        const Buffer* buffer;
        u64 offset;
        u64 range_epoch, placement_epoch, wraps, epoch;
        u64 tick; ///< the command buffer the answer was made in (an answer that names a copy in the stream buffer is valid in that one only)
        bool texel{false};
        bool lingered{false}; ///< verify mode only: announced bytes were still pending over the range right after the call that made the entry (they were taken another way)
    };
    struct BindMemo {
        u64 tick{~u64{0}};
        u64 max_size{BindMemoMax}; ///< the largest request ever stored: a write erases the entries that start up to this far before it
        absl::btree_map<VAddr, boost::container::small_vector<BindMemoEntry, 2>> entries; ///< (a B-tree: the lookup per request walked a node tree of 10 000+ addresses, a cache miss per level)
    };
    struct DrawRecipe {
        u64 shape{}, tick{};
        RecipeEpochs epochs{};
        bool valid{false}, uses_stream{false}, reusable{false};
        boost::container::small_vector<std::pair<VAddr, u32>, 8> ranges;
        u64 shape_nostream{}, identity_clock{}, kind_signature{}; ///< V0: the shape with the stream copies' offsets left out; the identity event clock when the draw started
        boost::container::small_vector<std::pair<VAddr, u32>, 4> image_ranges;
    };
    std::array<std::atomic<u64>, SvCountsEnd> scoped_counts{};
    std::unordered_map<u64, DrawRecipe> recipes;
    absl::btree_map<VAddr, boost::container::small_vector<u64, 2>> recipe_index; ///< range start -> keys of the recipes that hold a range starting there
    u64 recipe_max_len{4096};
    std::array<std::atomic<u64>, 24> recipe_counts{};
    void RecipeEraseRange(VAddr address, u64 size);
    std::array<BindMemo, 2> bind_memos; ///< by stream (graphics, side)
    std::atomic<u64> bind_memo_epoch{0}; ///< bumped by writes the raster thread does not see by range (faults, the command processor)
    std::array<std::atomic<u64>, 4> bind_memo_counts{}; ///< hits, misses, entries erased by writes, memo cleared
    std::array<std::atomic<u64>, 32> bind_memo_wide_counts{}; // [0..9] widened class, [10..19] the classic class (same meaning, mode 1 only) ///< widened class: answers used, compared (mode 1), different answer, stream answers, uncached call did work, hits seen
    std::pair<const Buffer*, u64> ObtainBufferUncached(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                       Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object);
    // zero copy (TLG_ZERO_COPY=1, zero-copy-dynamic-buffers.md): counters
    std::array<std::atomic<u64>, 10> zero_copy_counts{};
    std::atomic<bool> zero_copy_unavailable{false};
    /// The engine buffers as the in order create and delete events know them: the raster thread's view. The live table of the game thread is ahead of the raster thread at frames in flight 2
    /// (it has deleted what the raster thread still works on and created what it has not reached). `TLG_ORDERED_VIEW=1`: the buffer cache asks this view (BuffersInView, ResourceAtView,
    /// LookupView); otherwise it asks the live table. `TLG_ORDERED_VIEW_CHECK=1` counts the requests where the two disagree.
    Core::NativeBufferIndex::Ordered ordered; ///< G5b: the raster thread's in-order view of the engine buffers (native/native_buffer_index.h)
    void OrderedViewCheck(VAddr address, u64 size, bool written);
    void BuffersInView(VAddr address, u64 size, std::vector<Core::NativeDevice::BufferPart>& out);
    [[nodiscard]] bool ResourceAtView(VAddr address, Core::NativeDevice::Object& out) const;
    [[nodiscard]] bool LookupView(u64 id, Core::NativeDevice::Object& out) const;

public:
    [[nodiscard]] bool ZeroCopyOn() const noexcept;
    [[nodiscard]] nlohmann::json ZeroCopyReport() const;

private:
    std::unordered_set<u64> place_refused; ///< B9.4: engine buffers a request over their edge moved back into the arena (they stay there until deleted) ///< B9.4: by begin (raster thread; placed_mutex for the readers on other threads)
    mutable std::mutex placed_mutex;
    PlacedBuffer* PlacedAt(VAddr address, u64 size, bool written = true);
    bool PlacedMeets(VAddr address, u64 size) const;
    const Buffer* Place(std::vector<u64> ids, VAddr begin, VAddr end);
    void Demote(VAddr begin, bool by_request);
    void DropPlaced(VAddr begin, const char* why = "?");
    /// B9.4: a resident object's binding resolution names `buffer` from now on (a placed buffer remembers it, so freeing the buffer resets it)
    void NoteResolutionUser(const Buffer* buffer, u64 id);
    std::atomic<u64> b9_epoch{0}; ///< B9.3 step 2: bumped when an engine buffer gets changed bytes (the upload memo's answers are stale)
    StampedRangeSet b9_unattributed; ///< B9.1: announced bytes no engine buffer held when they were announced (a buffer's creation data comes before the buffer is known)
    std::mutex object_dirty_mutex;
public:
    std::atomic<u64> upload_bytes_kept_for_gpu{0}; ///< B5 / B6: uploads that left GPU written bytes out
private:

    struct ExplicitDownload {
        VAddr addr;
        u64 size;
        u64 id;
    };
    std::mutex explicit_mutex;
    std::condition_variable explicit_landed;
    std::vector<ExplicitDownload> explicit_inflight;
    u64 explicit_next_id{1};

    /// Answers of ObtainBuffer for large read-only and for written ranges: the same request in the same command
    /// buffer with the tracker and the range bookkeeping unchanged has nothing to upload and nothing to mark.
    struct ObtainMemo {
        VAddr addr{};
        u32 size{};
        u32 flags{};
        const Buffer* arena{};
        u64 offset{};
        u64 epoch{~0ull};
        u64 tick{~0ull};
        u64 epoch_no_cand{~0ull}; ///< F1 ring-out census: the epoch without the ticks of ring candidate applies
    };
    static constexpr u32 ObtainMemoSlots = 8192;
    std::unique_ptr<ObtainMemo[]> obtain_memo;
    std::atomic<u64> range_epoch{0}; ///< bumped when the GPU modified ranges shrink or the arenas move
    std::atomic<u64> placement_epoch{0}; ///< R1: bumped when an arena is replaced (a binding resolved against the old one is stale)
    /// Guards gpu_modified_ranges.
    mutable std::shared_mutex gpu_ranges_mutex;

    /// readback_stream_reuse_enabled: stream copies of the current command buffer by range. A copy
    /// is only reused while the command buffer, and the stream buffer pass, are the same.
    const bool stream_reuse_enabled;
    bool StreamReuseOn() const {
        const s32 override_mode = VideoCore::ReadbackPolicy::Get().stream_reuse_override.load(std::memory_order_relaxed);
        return override_mode >= 0 ? override_mode != 0 : stream_reuse_enabled;
    }
    /// One set per command stream (0 graphics, 1 asynchronous): a window of the other stream must not drop the copies of this one.
    struct StreamReuseSet {
        tsl::robin_map<u64, u64, Common::U64Hash> map;
        u64 tick{};
        u64 wraps{};
        u32 submit{}; ///< M2: the submission the copies were made for (0: not tracked)
        VAddr min{~0ULL};
        VAddr max{};
    };
    std::array<StreamReuseSet, 2> stream_reuse_sets;
    /// Asynchronous stream: the untracked ring is uploaded once per epoch (one frame's window of work of queue 1, which the graphics stream
    /// covers with one tick), not once per tick of that stream.
    u64 side_upload_epoch{1};

    std::mutex dma_mutex;
    std::vector<std::unique_ptr<Buffer>> dma_pool;
    std::atomic<u32> async_recorded{};

    /// Upload ring ranges already uploaded in the current command buffer (see UntrackedRing).
    struct RingUpload {
        u64 lo{};
        u64 hi{};
        u64 tick{};
    };
    std::array<RingUpload, 256> ring_uploads{};
    /// N2 (`TLG_RING_MIRROR=1`): the parts of the frame ring that the mirror buffer (`ring_mirror_token` of the Buffer) already holds; guarded by object_dirty_mutex
    RangeSet ring_filled;
    RangeSet ring_fill_pending; ///< O2 shadow: the ranges a fill at announce time would have uploaded since the last window sync took them; guarded by object_dirty_mutex
    u64 ring_mirror_token_next{0}, ring_mirror_token_current{0};
};

} // namespace VideoCore
