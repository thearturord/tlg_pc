// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/half_phase.h"
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "native/native_identity_log.h"
#include "native/native_ui_trace.h"
#include "native/native_config.h"
#include "native/native_device.h"
#include "native/native_draw.h"
#include "native/native_n0_census.h"
#include "native/native_readbacks.h"
#include "native/native_registry.h"
#include "native/native_updates.h"
#include "native/native_image_caller.h"
#include "native/native_flight.h"
#include "native/native_prof.h"
#include "native/native_debug_server.h"
#include "native/native_ring_mirror.h"

#include "native/native_resources.h"
#include "native/native_resource_audit.h"
#include "native/native_executor.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "native/native_readback_policy.h"
#include "native/native_uploads.h"
#include "native/native_untracked_ring.h"
#include "native/native_content.h"
#include <nlohmann/json.hpp>
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#include "native/native_split_census.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

namespace NativeIdentity = Core::NativeIdentity;

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, Core::NativeExecutor* executor_,
                         TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, executor{executor_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      resolve_stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      stream_reuse_enabled{EmulatorSettings.IsReadbackStreamReuseEnabled()} {
    resolve_stream_buffer.ring_id = 1;
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
    Core::NativeReadbacks::SetLandingGate(
        [this](VAddr address, u64 size, u32 timeout_ms) { return WaitForExplicitDownloads(address, size, timeout_ms); });
    // B2 cutover: an applied announcement is a CPU write for the caches
    Core::NativeUpdates::SetApply([](void* context, u64 address, u64 size, bool newer, u64 object) { static_cast<BufferCache*>(context)->ApplyCpuWrite(address, size, newer, object); }, this);
    Core::NativeUpdates::SetSkipCheck([](void* context, u64 address, u64 size) -> u32 { return static_cast<BufferCache*>(context)->CanSkipIdenticalWrite(address, size); }, this);
    Core::NativeUpdates::SetMergeCheck([](void* context, u64 object, u64 begin, u64 size, const Core::NativeUpdates::AnnouncedPiece* pieces, size_t count) {
        return static_cast<BufferCache*>(context)->CanMergeAnnounced(object, begin, size, pieces, count);
    }, this);
}

BufferCache::~BufferCache() {
    Core::NativeReadbacks::SetLandingGate({});
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (Core::NativeUpdates::B9TrackerOff()) {
        // a CPU change found without an announcement (a write fault, a decoder writing memory): the engine buffers it meets take it too
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9OtherChanges);
        BindMemoBump(); // D4: an unannounced write (any thread) did not reach the memo before: every answer is checked again
        MarkBuffersChanged(device_addr, size, false, nullptr);
    }
    // E4: no GPU bytes are flushed to guest memory here (the fault readback is gone: no page is ever made unreadable)
    cpu_epoch.fetch_add(1, std::memory_order_acq_rel);
}

void BufferCache::ApplyCpuWrite(VAddr address, u64 size, bool newer, u64 object) {
    Core::SplitCensus::Touch(Core::SplitCensus::BufDirty);
    // N1.2: the announcer named the engine buffer it wrote into. `told_whole`: that buffer holds the whole range, so its part is known without the address walk
    const u32 id_mode = object != 0 ? Core::NativeUpdates::BufferIdMode() : 0u;
    Core::NativeDevice::BufferPart told{};
    bool told_whole = false;
    const bool buffers_only = object == Core::NativeUpdates::kBuffersOnly; // (a block over several engine buffers: no part is named, but no image is there)
    if (buffers_only) {
        Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdToldBlock);
    } else if (id_mode != 0) {
        if (!Core::NativeDevice::BufferPartOf(object, address, size, told)) {
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdUnknown);
        } else if (told.begin == address && told.end == address + size) {
            told_whole = true;
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdToldWhole);
        } else {
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdToldPartial);
        }
    }

    bool n0_ring = false;
    if (Core::N0::On()) {
        const auto& ring = UntrackedRing::Get();
        if (ring.Active()) {
            const auto [lo, hi] = ring.Clip(address, size);
            n0_ring = lo < hi;
        }
    }
    Core::N0::ApplyScope n0_apply{n0_ring, size};
    if (Core::RingMirror::ShadowOn()) {
        Core::RingMirror::NoteAnnounced(address, size);
    }
    {
        TLG_PROF_LEVEL("cpuwrite.bind_memo_erase", 3);
        BindMemoErase(address, size);
    }
    if (newer) {
        // B2: the engine wrote these bytes after anything the GPU wrote there before (applied in order on the raster thread)
        TLG_PROF_LEVEL("cpuwrite.supersede", 3);
        SupersedeGpuWritten(address, size);
    }
    {
        TLG_PROF_LEVEL("cpuwrite.note_resources", 3);
        Core::NativeResources::NoteCpuWrite(address, size);
    }
    // O2 (`TLG_RING_FILL=1`): an announcement wholly inside the frame ring needs no walk of the engine's buffers (none lies in the ring: the shadow counted 0 in 1.8 M announcements) and no dirty set by
    // address: the ring has its own set of the bytes the mirror has not taken yet. The part of the ring the mirror has not filled yet needs no mark at all (the first read uploads it whole).
    bool ring_fast = false;
    if (Core::RingMirror::FillMode() == 1 && Core::RingMirror::Mode() == 1 && !buffers_only && object == 0) {
        const auto& ring = UntrackedRing::Get();
        if (ring.Active()) {
            const auto [lo, hi] = ring.Clip(address, size);
            if (lo == address && hi == address + size) {
                ring_fast = true;
                std::scoped_lock lock{object_dirty_mutex};
                ring_filled.ForEachInRange(lo, hi - lo, [&](u64 a, u64 e) { ring_fill_pending.Add(a, e - a); });
                cpu_epoch.fetch_add(1, std::memory_order_acq_rel);
            }
        }
    }
    if (!ring_fast && Core::NativeUpdates::B9TrackerOff()) {
        TLG_PROF_LEVEL("cpuwrite.mark_buffers", 3);
        // B9.3 step 2: the page tracker only marks what no engine buffer holds
        static thread_local std::vector<std::pair<VAddr, VAddr>> gaps;
        gaps.clear();
        MarkBuffersChanged(address, size, true, &gaps, id_mode != 0 && told_whole ? &told : nullptr, id_mode == 1);
        u64 marked = 0;
        for (const auto& [a, e] : gaps) {
            marked += e - a;
        }
        cpu_epoch.fetch_add(1, std::memory_order_acq_rel);
        if (marked < size) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerMarksSkipped);
        }
    }
    if (Core::RingMirror::FillMode() == 2 && Core::RingMirror::Mode() == 1) {
        // O2 shadow: what a fill at announce time would upload for this announcement
        const auto& ring = UntrackedRing::Get();
        Core::RingMirror::NoteFill(Core::RingMirror::FillApplies);
        const auto [lo, hi] = ring.Active() ? ring.Clip(address, size) : std::pair<VAddr, VAddr>{0, 0};
        if (lo >= hi) {
            Core::RingMirror::NoteFill(Core::RingMirror::FillNotRing);
        } else {
            boost::container::small_vector<std::pair<u64, u64>, 4> filled;
            u64 filled_bytes = 0;
            {
                std::scoped_lock lock{object_dirty_mutex};
                ring_filled.ForEachInRange(lo, hi - lo, [&](u64 a, u64 e) { filled.emplace_back(a, e - a); });
                for (const auto& [a, n] : filled) {
                    ring_fill_pending.Add(a, n);
                    filled_bytes += n;
                }
            }
            Core::RingMirror::NoteFill(Core::RingMirror::FillBytesFilled, filled_bytes);
            Core::RingMirror::NoteFill(Core::RingMirror::FillBytesUnfilled, (hi - lo) - filled_bytes);
            if (filled_bytes != 0) {
                static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
                parts.clear();
                BuffersInView(lo, hi - lo, parts);
                if (!parts.empty()) {
                    Core::RingMirror::NoteFill(Core::RingMirror::FillWithBuffers);
                }
            }
        }
    }
    const u32 b8 = Core::NativeUpdates::B8ImageMode();
    if (b8 != 0) {
        TLG_PROF_LEVEL("cpuwrite.image_index", 3);
        auto& index = texture_cache.EngineIndex();
        if (id_mode == 1 && (told_whole || buffers_only)) {
            // an engine buffer holds the whole range: no engine image or waiting clear lives in a buffer's memory, so none is looked for (the shadow mode counts the cases where the lookup would have changed something)
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdImageSkipped);
        } else if (index.NoteAnnouncedWrite(address, size, newer) == 1 && id_mode == 2 && (told_whole || buffers_only)) {
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdImageFoundOverTold);
        }
    }
}

bool BufferCache::CanMergeAnnounced(u64 object, VAddr begin, u64 size, const Core::NativeUpdates::AnnouncedPiece* pieces, size_t count) {
    // L1: the apply of the merged range does to the gap bytes what the pieces do to their own. Nothing may depend on which bytes exactly were written: the dirty marks of the buffers (the gap
    // bytes are current guest memory, uploading them is right), the resident generations (a page each), the memo (erased wider). What must not widen: bytes the GPU wrote (the apply would
    // supersede them) and images (the apply dirties every image it meets).
    if (!Core::NativeUpdates::B9TrackerOff() || Core::NativeUpdates::B8ImageMode() == 0) {
        Core::NativeUpdates::NoteMergeRefused(0, count);
        return false;
    }
    if (object != 0 && object != Core::NativeUpdates::kBuffersOnly) {
        Core::NativeDevice::BufferPart part{};
        if (!Core::NativeDevice::BufferPartOf(object, begin, size, part) || part.begin != begin || part.end != begin + size) {
            Core::NativeUpdates::NoteMergeRefused(1, count);
            return false; // (the named buffer must hold the whole range, as for one piece)
        }
    }
    if (IsRangeGpuWritten(begin, size)) {
        Core::NativeUpdates::NoteMergeRefused(2, count);
        return false;
    }
    boost::container::small_vector<std::pair<u64, u64>, 16> list;
    for (size_t i = 0; i < count; ++i) {
        list.emplace_back(pieces[i].address, pieces[i].size);
    }
    const bool exact = texture_cache.EngineIndex().MergeIsExact(begin, size, list.data(), list.size());
    if (!exact) {
        Core::NativeUpdates::NoteMergeRefused(3, count);
    }
    return exact;
}

u32 BufferCache::CanSkipIdenticalWrite(VAddr address, u64 size) {
    // the apply does its work in three places, and each has to be a no-op for these bytes: the buffers (only when the engine buffers are tracked by announcements, and the GPU wrote
    // nothing over the range, else the CPU's bytes must win again), the images (none over the range, no clear waiting) and the resident buffers (their stamps stay: same bytes)
    if (!Core::NativeUpdates::B9TrackerOff() || Core::NativeUpdates::B8ImageMode() == 0) {
        return 3;
    }
    if (IsRangeGpuWritten(address, size)) {
        return 1;
    }
    const u32 inert = texture_cache.EngineIndex().AnnouncedWriteIsInert(address, size);
    return inert == 0 ? 0 : inert == 1 ? 4 : 2;
}

void BufferCache::MarkBuffersChanged(VAddr address, u64 size, bool announced, std::vector<std::pair<VAddr, VAddr>>* gaps, const Core::NativeDevice::BufferPart* told, bool use_told) {
    Core::SplitCensus::Touch(Core::SplitCensus::BufDirty);
    // B9.1: each engine buffer the bytes belong to keeps its part until its next upload (an announcement can run over several buffers)
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    parts.clear();
    if (told != nullptr && use_told) {
        parts.push_back(*told); // N1.2: named by the announcer
    } else {
        TLG_PROF_LEVEL("cpuwrite.mark.buffers_in_view", 3); // D2: the address to buffer ids resolution
        BuffersInView(address, size, parts);
        if (told != nullptr) {
            const bool same = parts.size() == 1 && parts[0].id == told->id && parts[0].begin == told->begin && parts[0].end == told->end;
            Core::NativeUpdates::NoteBufferId(same ? Core::NativeUpdates::IdPartsMatch : Core::NativeUpdates::IdPartsDiffer);
        }
    }
    TLG_PROF_NAMED_LEVEL(dirty_scope, "cpuwrite.mark.object_dirty", 3);
    if (!parts.empty()) {
        Core::N0::SetEffect(Core::N0::EffBufferParts);
    }
    std::scoped_lock lock{object_dirty_mutex};
    if (Core::NativeUpdates::LeanDirtyMode() != 2) {
        all_object_dirty.Add(address, size); // audit: every changed byte, by address
    }
    if (announced) {
        Core::NativeUpdates::NoteB9(parts.empty() ? Core::NativeUpdates::B9AnnouncedNoBuffer
                                    : parts.size() == 1 && parts[0].begin == address && parts[0].end == address + size ? Core::NativeUpdates::B9AnnouncedOnBuffer
                                                                                                                      : Core::NativeUpdates::B9AnnouncedClipped);
    }
    for (const auto& part : parts) {
        object_dirty[part.id].Add(part.begin, part.end - part.begin);
    }
    dirty_scope.End();
    TLG_PROF_NAMED_LEVEL(gaps_scope, "cpuwrite.mark.unattributed", 3);
    // the bytes no buffer holds now: the first upload of a buffer over them takes them. (Add of the whole range then Subtract of the buffers' parts, done as the same
    // set algebra without the detour: the parts leave the set, the gaps between them join it; the detour allocated a node per write and freed it again)
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
    boost::container::small_vector<std::pair<VAddr, VAddr>, 4> own_gaps;
    {
        VAddr at = address;
        for (const auto& part : parts) {
            if (part.begin > at) {
                own_gaps.emplace_back(at, part.begin);
            }
            at = std::max(at, part.end);
        }
        if (at < address + size) {
            own_gaps.emplace_back(at, address + size);
        }
    }
    for (const auto& part : parts) {
        b9_unattributed.Subtract(part.begin, part.end - part.begin);
    }
    for (const auto& [a, e] : own_gaps) {
        b9_unattributed.Add(a, e - a);
    }
    if (!parts.empty()) {
        b9_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    if (gaps != nullptr) {
        gaps->insert(gaps->end(), own_gaps.begin(), own_gaps.end());
    }
}

void BufferCache::InvalidateStreamReuse(VAddr device_addr, u64 size) {
    for (auto& set : stream_reuse_sets) {
        if (set.map.empty() || device_addr >= set.max || device_addr + size <= set.min) {
            continue;
        }
        set.map.clear();
        set.min = ~0ULL;
        set.max = 0;
    }
}

void BufferCache::RecordDmaDownload(const Buffer* arena, VAddr dst, u32 size, bool tracked) {
    // The game's own readbacks are small lists and counters; a bigger copy is not one of them.
    if (size == 0 || size > MaxDmaDownload) {
        return;
    }
    const auto staging = staging_pool.Request(size, MemoryType::HostCached, 0, true);
    const vk::BufferCopy region = {
        .srcOffset = arena->Offset(dst),
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, staging.buffer, std::span{&region, 1});
    auto& policy = ReadbackPolicy::Get();
    policy.dma_async_copies.fetch_add(1, std::memory_order_relaxed);
    policy.dma_async_bytes.fetch_add(size, std::memory_order_relaxed);
    async_recorded.store(1, std::memory_order_release);
    // The priority thread runs its operations in the order they were deferred, so a fence deferred
    // after this one is signaled only once the bytes are in guest memory.
    u64 id = 0;
    if (tracked) {
        std::scoped_lock lock{explicit_mutex};
        id = explicit_next_id++;
        explicit_inflight.push_back({dst, size, id});
    }
    scheduler.DeferPriorityOperation(
        [this, staging, dst, size, tracked, id] {
            if (tracked) {
                Core::NativeReadbacks::LandingBegin(dst, size);
            }
            staging.buffer->Invalidate(staging.offset, size);
            memory->TryWriteBacking(std::bit_cast<u8*>(dst), staging.mapped, size);
            staging_pool.FreeDeferred(staging);
            if (tracked) {
                Core::NativeReadbacks::LandingEnd(dst, size);
                {
                    std::scoped_lock lock{explicit_mutex};
                    std::erase_if(explicit_inflight, [id](const ExplicitDownload& download) { return download.id == id; });
                }
                explicit_landed.notify_all();
            }
        },
        scheduler.CurrentTick());
}

u64 BufferCache::PreSynchronize(VAddr device_addr, u64 size) {
    if (size == 0 || !memory->IsValidMapping(device_addr, 1)) {
        return 0;
    }
    size = std::min<u64>(memory->ClampRangeSize(device_addr, size), u64{1} << 26);
    if (size == 0) {
        return 0;
    }
    const auto* arena = BufferFor(device_addr, size, false); // B9.4: a range over placed buffers' edges is left to their own requests
    if (arena == nullptr) {
        return 0;
    }
    last_upload_bytes = 0;
    Core::NativeUploads::SetPreSyncActive(true);
    Core::N0::SetSyncSite(Core::N0::SyncPre);
    SynchronizeMemory(arena, device_addr, static_cast<u32>(size), false, false);
    Core::NativeUploads::SetPreSyncActive(false);
    return last_upload_bytes;
}

static bool BindMemoOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_BIND_MEMO");
        return e == nullptr || *e != '0';
    }();
    return on;
}

namespace {
/// Names the caller of ObtainBuffer for the late read samples when the bind site did not (NativeContent::SetBindContext)
struct BindGuard {
    bool own;
    BindGuard(u64 caller, VAddr address, u64 size, bool written, bool handle) : own(!Core::NativeContent::HasBindContext()) {
        if (own) {
            Core::NativeContent::SetBindContext({"other caller", 0, 0, 0, 0, handle, written, address, size, caller});
        }
    }
    ~BindGuard() {
        if (own) {
            Core::NativeContent::ClearBindContext();
        }
    }
};
} // namespace

/// D4 `TLG_BIND_MEMO_WIDE` (default 0): 1 the memo also answers read-only texel buffers and read-only requests over BindMemoMax (up to BindMemoWideMax), but the uncached answer is
/// used and the two are compared, plus whether the uncached call did any work a memo hit would have skipped (counted); 2 the memo answer is used. Written requests never use
/// the memo: they mark the bytes GPU written, which is state, not a lookup.
static u32 BindMemoWideMode() {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_BIND_MEMO_WIDE");
        return e != nullptr && *e >= '0' && *e <= '2' ? static_cast<u32>(*e - '0') : 0u;
    }();
    return mode;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                        Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object) {
    const auto result = ObtainBufferBody(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
    if (result.first != nullptr) {
        const_cast<Buffer*>(result.first)->NoteResolveUse(); // P3c: asked for by the operation being resolved (every answer, memo hits too)
    }
    return result;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferBody(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                            Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBufResolve};
    Core::SplitCensus::Touch(Core::SplitCensus::BufObtain);
    const BindGuard bind_guard{reinterpret_cast<u64>(__builtin_return_address(0)), device_addr, size, is_written, immutable_hint != nullptr};
    TLG_PROF_LEVEL("buffer.obtain", 2);
    if (is_written) {
        TLG_PROF_LEVEL("buffer.obtain.written_erase", 3);
        BindMemoErase(device_addr, size); // the GPU writes it: answers over this memory are stale
    }
    const u32 wide_mode = BindMemoWideMode();
    const bool wide_class = wide_mode != 0 && !is_written && (is_texel_buffer || size > BindMemoMax) && size <= BindMemoWideMax;
    if (is_written || ((is_texel_buffer || size > BindMemoMax) && !wide_class) || !BindMemoOn()) {
        if (is_written) {
            TLG_PROF_COUNT("buffer.request.written");
        } else if (is_texel_buffer) {
            TLG_PROF_COUNT("buffer.request.texel");
        } else {
            TLG_PROF_COUNT("buffer.request.too_big_for_the_bind_memo");
        }
        return ObtainBufferUncached(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
    }
    TLG_PROF_NAMED_LEVEL(lookup_scope, "buffer.obtain.memo_lookup", 3);
    auto& memo = bind_memos[Vulkan::Scheduler::SideSelected() ? 1 : 0];
    const u64 tick = scheduler.ResolveEpoch();
    if (memo.tick != tick) {
        // a new command buffer: copies in the stream buffer belong to the old one (the game may rewrite their memory between submissions); answers that name
        // an arena or a placed buffer stay valid until a write or a move erases them (the frame has about 13 command buffers). The entries remember their command
        // buffer (`tick`), so no walk over the whole memo is needed here (it visited every node 13 times a frame); a stale copy entry is replaced at its next miss.
        memo.tick = tick;
        if (memo.entries.size() > BindMemoMaxAddresses) {
            memo.entries.clear();
            bind_memo_counts[3].fetch_add(1, std::memory_order_relaxed);
        }
    }
    const u64 epoch = bind_memo_epoch.load(std::memory_order_acquire);
    const u64 ranges = range_epoch.load(std::memory_order_acquire);
    const u64 arenas = PlacementEpoch();
    if (const auto it = memo.entries.find(device_addr); it != memo.entries.end()) {
        for (const auto& e : it->second) {
            if (e.size == size && e.texel == is_texel_buffer && e.epoch == epoch && e.range_epoch == ranges && e.placement_epoch == arenas &&
                (e.buffer != &CopyRing() || (e.wraps == CopyRing().WrapCount() && e.tick == tick))) {
                if (wide_class || wide_mode == 1) {
                    const u32 cb = wide_class ? 0u : 10u; // counter base: widened class or classic class
                    bind_memo_wide_counts[cb + 5].fetch_add(1, std::memory_order_relaxed);
                    if (wide_class) {
                        TLG_PROF_COUNT("buffer.bind_memo.wide_hit");
                    }
                    if (wide_mode == 1) {
                        // verify: the uncached answer decides; the memo's is compared, and so is whether the uncached call did anything a hit would have skipped
                        const auto* list = scheduler.CommandBuffer().List();
                        const u64 commands_before = list != nullptr ? list->Size() : 0;
                        UploadBytesReset();
                        const bool cpu_modified_before = IsRegionCpuModified(device_addr, size);
                        const bool tracker_before = NativeCpuQuery(device_addr, size);
                        const auto answer = ObtainBufferUncached(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
                        const u64 commands_after = list != nullptr ? list->Size() : 0;
                        bind_memo_wide_counts[cb + 1].fetch_add(1, std::memory_order_relaxed);
                        const bool stream = e.buffer == &CopyRing();
                        if (stream) {
                            bind_memo_wide_counts[cb + 3].fetch_add(1, std::memory_order_relaxed);
                        }
                        if (answer.first != e.buffer || (!stream && answer.second != e.offset)) {
                            bind_memo_wide_counts[cb + 2].fetch_add(1, std::memory_order_relaxed);
                            static std::atomic<u32> logged{0};
                            if (logged.fetch_add(1) < 8) {
                                LOG_ERROR(Render, "bind memo (wide) answered {:p}+{:#x}, the uncached answer is {:p}+{:#x} for {:#x}+{:#x} (texel {})", static_cast<const void*>(e.buffer), e.offset,
                                          static_cast<const void*>(answer.first), answer.second, device_addr, size, is_texel_buffer);
                            }
                        }
                        if ((UploadBytes() != 0 || commands_after != commands_before) && e.lingered) {
                            bind_memo_wide_counts[cb + 20].fetch_add(1, std::memory_order_relaxed); // explained: the entry was made with announced bytes still pending over it
                        } else if (UploadBytes() != 0 || commands_after != commands_before) {
                            bind_memo_wide_counts[cb + 4].fetch_add(1, std::memory_order_relaxed);
                            bind_memo_wide_counts[cb + (cpu_modified_before ? 6 : 7)].fetch_add(1, std::memory_order_relaxed);
                            bind_memo_wide_counts[cb + (tracker_before ? 8 : 9)].fetch_add(1, std::memory_order_relaxed);
                            static std::atomic<u32> logged_work{0};
                            if (logged_work.fetch_add(1) < 8) {
                                LOG_ERROR(Render, "bind memo (wide) would have skipped work: {:#x}+{:#x} (texel {}) uploaded {} bytes, {} commands", device_addr, size, is_texel_buffer, UploadBytes(),
                                          commands_after - commands_before);
                            }
                        }
                        return answer;
                    }
                    bind_memo_wide_counts[cb + 0].fetch_add(1, std::memory_order_relaxed);
                }
                bind_memo_counts[0].fetch_add(1, std::memory_order_relaxed);
                TLG_PROF_COUNT("buffer.bind_memo.hit");
                return {e.buffer, e.offset};
            }
        }
    }
    bind_memo_counts[1].fetch_add(1, std::memory_order_relaxed);
    TLG_PROF_COUNT("buffer.bind_memo.miss");
    lookup_scope.End();
    const auto result = ObtainBufferUncached(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
    TLG_PROF_LEVEL("buffer.obtain.memo_insert", 3);
    if (memo.tick == tick) {
        auto& list = memo.entries[device_addr];
        list.erase(std::remove_if(list.begin(), list.end(), [&](const BindMemoEntry& e) { return e.size == size && e.texel == is_texel_buffer; }), list.end());
        // the epochs as they were before the request: anything the request itself changed makes the next one ask again
        list.push_back({size, result.first, result.second, ranges, arenas, result.first == &CopyRing() ? CopyRing().WrapCount() : 0, epoch, tick, is_texel_buffer,
                        BindMemoWideMode() == 1 && IsRegionCpuModified(device_addr, size)});
        memo.max_size = std::max<u64>(memo.max_size, size);
    }
    return result;
}

u32 BufferCache::RecipePredict(u64 key, u64 tick, const RecipeEpochs& now, u64& shape) {
    const auto it = recipes.find(key);
    if (it == recipes.end()) {
        return 0;
    }
    const DrawRecipe& r = it->second;
    shape = r.shape;
    RecipeEpochs a = r.epochs, b = now;
    a.memo = b.memo = 0; // the global memo epoch is bumped by every command processor write: measured as a gate it voids 98 % of the recipes, so it is left out here and the compare shows what that costs
    const bool ok = r.valid && r.reusable && a == b && (!r.uses_stream || r.tick == tick);
    if (!ok) {
        // why not: every reason that applies is counted (a prediction can fail for several)
        recipe_counts[10 + 0].fetch_add(!r.valid, std::memory_order_relaxed);
        recipe_counts[10 + 1].fetch_add(!r.reusable, std::memory_order_relaxed);
        recipe_counts[10 + 2].fetch_add(r.epochs.memo != now.memo, std::memory_order_relaxed);
        recipe_counts[10 + 3].fetch_add(r.epochs.range != now.range, std::memory_order_relaxed);
        recipe_counts[10 + 4].fetch_add(r.epochs.placement != now.placement, std::memory_order_relaxed);
        recipe_counts[10 + 5].fetch_add(r.epochs.immutable != now.immutable, std::memory_order_relaxed);
        recipe_counts[10 + 6].fetch_add(r.epochs.image != now.image, std::memory_order_relaxed);
        recipe_counts[10 + 7].fetch_add(r.uses_stream && r.tick != tick, std::memory_order_relaxed);
    }
    return ok ? 1 : 2;
}

void BufferCache::RecipeStore(u64 key, u64 shape, const boost::container::small_vector<std::pair<VAddr, u32>, 8>& ranges, bool uses_stream, bool reusable, u64 tick, const RecipeEpochs& at_start) {
    if (recipes.size() > 100000) {
        recipes.clear();
        recipe_index.clear();
    }
    DrawRecipe& r = recipes[key];
    const bool same_ranges = r.ranges.size() == ranges.size() && std::equal(r.ranges.begin(), r.ranges.end(), ranges.begin());
    r.shape = shape;
    r.tick = tick;
    r.epochs = at_start;
    r.uses_stream = uses_stream;
    r.reusable = reusable;
    r.valid = true;
    if (!same_ranges) {
        r.ranges = ranges;
        for (const auto& [a, n] : ranges) {
            recipe_max_len = std::max<u64>(recipe_max_len, n);
            auto& list = recipe_index[a];
            if (std::find(list.begin(), list.end(), key) == list.end()) {
                list.push_back(key);
            }
        }
    }
}

BufferCache::ScopedPrediction BufferCache::RecipePredictScoped(u64 key, u64 tick) {
    ScopedPrediction out;
    scoped_counts[SvChecked].fetch_add(1, std::memory_order_relaxed);
    const auto it = recipes.find(key);
    if (it == recipes.end() || it->second.identity_clock == 0) {
        scoped_counts[SvNoRecipe].fetch_add(1, std::memory_order_relaxed);
        return out;
    }
    const DrawRecipe& r = it->second;
    out.shape = r.shape;
    out.shape_nostream = r.shape_nostream;
    out.kind_signature = r.kind_signature;
    out.content_clean = r.valid; // false once a content write overlapped one of its buffer ranges since it was stored
    out.uses_stream = r.uses_stream;
    out.other_command_buffer = r.uses_stream && r.tick != tick;
    if (!r.reusable) {
        scoped_counts[SvNotReusable].fetch_add(1, std::memory_order_relaxed);
        out.state = 2;
        return out;
    }
    NativeIdentity::Kind kind{};
    for (const auto& [address, length] : r.ranges) {
        const auto change = NativeIdentity::ChangedSince(r.identity_clock, address, address + length, &kind);
        if (change == NativeIdentity::Change::Unknown) {
            scoped_counts[SvLogUnknown].fetch_add(1, std::memory_order_relaxed);
            out.state = 4;
            return out;
        }
        if (change == NativeIdentity::Change::Overlaps) {
            scoped_counts[SvIdentityChanged].fetch_add(1, std::memory_order_relaxed);
            scoped_counts[SvKindBase + (kind < 8 ? kind : 7)].fetch_add(1, std::memory_order_relaxed);
            out.state = 3;
            return out;
        }
    }
    for (const auto& [address, length] : r.image_ranges) {
        const auto change = NativeIdentity::ChangedSince(r.identity_clock, address, address + length, &kind);
        if (change == NativeIdentity::Change::Unknown) {
            scoped_counts[SvLogUnknown].fetch_add(1, std::memory_order_relaxed);
            out.state = 4;
            return out;
        }
        if (change == NativeIdentity::Change::Overlaps) {
            scoped_counts[SvIdentityChanged].fetch_add(1, std::memory_order_relaxed);
            scoped_counts[SvKindBase + (kind < 8 ? kind : 7)].fetch_add(1, std::memory_order_relaxed);
            out.state = 3;
            return out;
        }
    }
    if (r.image_ranges.empty()) {
        scoped_counts[SvNoImageRanges].fetch_add(1, std::memory_order_relaxed);
    }
    out.state = 1;
    scoped_counts[SvValid].fetch_add(1, std::memory_order_relaxed);
    return out;
}

void BufferCache::RecipeStoreScoped(u64 key, u64 shape_nostream, u64 kind_signature, const boost::container::small_vector<std::pair<VAddr, u32>, 4>& image_ranges, u64 identity_clock) {
    const auto it = recipes.find(key);
    if (it == recipes.end()) {
        return;
    }
    it->second.shape_nostream = shape_nostream;
    it->second.kind_signature = kind_signature;
    it->second.image_ranges = image_ranges;
    it->second.identity_clock = identity_clock;
}

void BufferCache::RecipeEraseRange(VAddr address, u64 size) {
    if (recipe_index.empty()) {
        return;
    }
    auto it = recipe_index.lower_bound(address >= recipe_max_len ? address - recipe_max_len : 0);
    for (; it != recipe_index.end() && it->first < address + size; ++it) {
        for (const u64 key : it->second) {
            const auto r = recipes.find(key);
            if (r == recipes.end() || !r->second.valid) {
                continue;
            }
            for (const auto& [a, n] : r->second.ranges) {
                if (a < address + size && address < a + n) {
                    Core::N0::SetEffect(Core::N0::EffRecipe);
                    r->second.valid = false;
                    break;
                }
            }
        }
    }
}

void BufferCache::BindMemoErase(VAddr address, u64 size) {
    Core::SplitCensus::Touch(Core::SplitCensus::BufMemoErase);
    RecipeEraseRange(address, size);
    for (auto& memo : bind_memos) {
        if (memo.entries.empty()) {
            continue;
        }
        auto it = memo.entries.lower_bound(address >= memo.max_size ? address - memo.max_size : 0);
        while (it != memo.entries.end() && it->first < address + size) {
            const size_t n0_before = it->second.size();
            it->second.erase(std::remove_if(it->second.begin(), it->second.end(), [&](const BindMemoEntry& e) { return it->first + e.size > address; }), it->second.end());
            if (it->second.size() != n0_before) {
                Core::N0::SetEffect(Core::N0::EffMemo);
            }
            if (it->second.empty()) {
                it = memo.entries.erase(it);
                bind_memo_counts[2].fetch_add(1, std::memory_order_relaxed);
            } else {
                ++it;
            }
        }
    }
}

nlohmann::json BufferCache::BindMemoReport() const {
    return {{"on", BindMemoOn()}, {"hits", bind_memo_counts[0].load()}, {"misses", bind_memo_counts[1].load()}, {"addresses_erased_by_writes", bind_memo_counts[2].load()},
            {"recipes", {{"graphics_draws_checked", recipe_counts[0].load()}, {"no_key", recipe_counts[1].load()}, {"no_recipe_yet", recipe_counts[2].load()},
                         {"predicted_valid_and_same", recipe_counts[3].load()}, {"PREDICTED_VALID_BUT_DIFFERENT_MUST_BE_0", recipe_counts[4].load()},
                         {"predicted_invalid_and_same", recipe_counts[5].load()}, {"predicted_invalid_and_different", recipe_counts[6].load()},
                         {"not_reusable_ranges_or_written", recipe_counts[7].load()}, {"recipes_held", recipes.size()},
                         {"why_not_range_written", recipe_counts[10].load()}, {"why_not_unreusable", recipe_counts[11].load()}, {"why_not_memo_epoch", recipe_counts[12].load()},
                         {"why_not_range_epoch", recipe_counts[13].load()}, {"why_not_placement_epoch", recipe_counts[14].load()}, {"why_not_immutable_epoch", recipe_counts[15].load()},
                         {"why_not_image_generation", recipe_counts[16].load()}, {"why_not_other_command_buffer", recipe_counts[17].load()}}},
            {"versions_v0", {{"draws_checked", scoped_counts[SvChecked].load()}, {"no_recipe_yet", scoped_counts[SvNoRecipe].load()}, {"not_reusable", scoped_counts[SvNotReusable].load()},
                             {"identity_changed", scoped_counts[SvIdentityChanged].load()}, {"log_overrun", scoped_counts[SvLogUnknown].load()},
                             {"identity_changed_by_kind", {{"buffer_placed", scoped_counts[SvKindBase + 1].load()}, {"buffer_dropped", scoped_counts[SvKindBase + 2].load()},
                                                           {"immutable_added", scoped_counts[SvKindBase + 3].load()}, {"immutable_removed", scoped_counts[SvKindBase + 4].load()},
                                                           {"image_registered", scoped_counts[SvKindBase + 5].load()}, {"image_unregistered", scoped_counts[SvKindBase + 6].load()},
                                                           {"gpu_written_state", scoped_counts[SvKindBase + 7].load()}}},
                             {"valid", scoped_counts[SvValid].load()}, {"valid_and_same_shape", scoped_counts[SvValidSame].load()},
                             {"valid_shape_differs_only_in_stream_copy_offsets", scoped_counts[SvValidStreamOffsetOnly].load()},
                             {"VALID_BUT_DIFFERENT_OTHERWISE_MUST_BE_0", scoped_counts[SvValidOtherDifference].load()},
                             {"of_those_a_buffer_changed_kind_ring_stream_other", scoped_counts[SvValidPathSwitch].load()}, {"of_those_same_kinds", scoped_counts[SvValidDifferentSameKinds].load()},
                             {"valid_content_clean", scoped_counts[SvValidContentClean].load()}, {"valid_content_dirty", scoped_counts[SvValidContentDirty].load()},
                             {"valid_uses_stream_copy", scoped_counts[SvValidUsesStream].load()}, {"valid_stream_copy_from_other_command_buffer", scoped_counts[SvValidStreamOtherCommandBuffer].load()},
                             {"invalid_but_shape_same", scoped_counts[SvInvalidButSame].load()}, {"invalid_and_shape_differs", scoped_counts[SvInvalidAndDifferent].load()},
                             {"recipes_without_image_ranges", scoped_counts[SvNoImageRanges].load()},
                             {"resolve_ms_valid_draws", scoped_counts[SvNsValid].load() / 1e6}, {"resolve_ms_invalid_draws", scoped_counts[SvNsInvalid].load() / 1e6},
                             {"resolve_ms_no_recipe_draws", scoped_counts[SvNsNoRecipe].load() / 1e6}, {"resolve_ms_all_checked", scoped_counts[SvNsAll].load() / 1e6},
                             {"identity_events", {{"buffer_placed", NativeIdentity::Count(NativeIdentity::BufferPlaced)}, {"buffer_dropped", NativeIdentity::Count(NativeIdentity::BufferDropped)},
                                                  {"immutable_added", NativeIdentity::Count(NativeIdentity::ImmutableAdded)}, {"immutable_removed", NativeIdentity::Count(NativeIdentity::ImmutableRemoved)},
                                                  {"image_registered", NativeIdentity::Count(NativeIdentity::ImageRegistered)}, {"image_unregistered", NativeIdentity::Count(NativeIdentity::ImageUnregistered)},
                                                  {"gpu_written_state", NativeIdentity::Count(NativeIdentity::GpuWrittenState)}}}}},
            {"wide", {{"mode", BindMemoWideMode()}, {"hits_seen", bind_memo_wide_counts[5].load()}, {"answers_used", bind_memo_wide_counts[0].load()}, {"compared", bind_memo_wide_counts[1].load()},
                      {"DIFFERENT_ANSWER_MUST_BE_0", bind_memo_wide_counts[2].load()}, {"compared_stream_answers", bind_memo_wide_counts[3].load()}, {"UNCACHED_DID_WORK_MUST_BE_0", bind_memo_wide_counts[4].load()},
                      {"work_explained_by_dirt_that_lingered", bind_memo_wide_counts[20].load()}, {"CLASSIC_work_explained_by_dirt_that_lingered", bind_memo_wide_counts[30].load()},
                      {"CLASSIC_hits_seen", bind_memo_wide_counts[15].load()}, {"CLASSIC_DIFFERENT_ANSWER_MUST_BE_0", bind_memo_wide_counts[12].load()}, {"CLASSIC_UNCACHED_DID_WORK_MUST_BE_0", bind_memo_wide_counts[14].load()},
                      {"CLASSIC_work_with_cpu_modified_true_before", bind_memo_wide_counts[16].load()},
                      {"work_with_cpu_modified_true_before", bind_memo_wide_counts[6].load()}, {"work_with_cpu_modified_false_before", bind_memo_wide_counts[7].load()},
                      {"work_with_any_announced_dirty_before", bind_memo_wide_counts[8].load()}, {"work_with_none_announced_before", bind_memo_wide_counts[9].load()}}}};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferUncached(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                                Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object) {
    Core::ResourceAudit::Note(Core::ResourceAudit::BufferUse, device_addr, size, is_written);
    // An immutable engine buffer (see NativeResources::ImmutableBuffer): after one synchronization of the whole buffer it is
    // read straight from the arena.
    // known_no_object (R2): the command processor's lookup by address found no engine object and none was registered since
    TLG_PROF_NAMED_LEVEL(imm_scope, "buffer.part.immutable", 3);
    if (!is_written && !is_texel_buffer && !known_no_object && Core::NativeResources::ImmutableResidencyEnabled()) {
        Core::NativeResources::ImmutableBuffer* immutable = nullptr;
        if (immutable_hint != nullptr) {
            // N3c: the handle of the engine id replaces the lookup by address; it must contain the request
            if (device_addr >= immutable_hint->begin && device_addr + size <= immutable_hint->end && immutable_hint->alive.load(std::memory_order_acquire)) {
                immutable = immutable_hint;
                Core::NativeResources::NoteHandle(0);
                if (verify_hint && Core::NativeResources::FindImmutable(device_addr, size) != immutable) {
                    Core::NativeResources::NoteHandle(4);
                    immutable = Core::NativeResources::FindImmutable(device_addr, size);
                }
            } else {
                Core::NativeResources::NoteHandle(3);
                immutable = Core::NativeResources::FindImmutable(device_addr, size);
                if (!immutable_hint->alive.load(std::memory_order_acquire)) {
                    Core::NativeDraw::LifeStaleBind(immutable_hint->object, immutable != nullptr);
                }
            }
        } else {
            immutable = Core::NativeResources::FindImmutable(device_addr, size);
        }
        if (immutable != nullptr) {
            Core::NativeDraw::LifeBound(immutable->object, immutable->id, immutable == immutable_hint);
        }
        if (immutable != nullptr && immutable->usage != 1 && ObjectCopiesOn()) {
            // R1b (native back end): a default or dynamic engine buffer has its own copy in the stream ring, made once per generation from its exact bytes
            // (no page of a neighbour is uploaded with it, so a page the GPU wrote elsewhere does not matter); bytes of the object the GPU wrote keep it in the arena
            u64 offset = 0;
            if (const Buffer* copy = ObjectCopy(*immutable, device_addr, offset)) {
                Core::NativeContent::SilentResidentUse(1, immutable->begin, device_addr, size, immutable->usage); // W1.5 census
                Core::NativeResources::NoteResident(0);
                TLG_PROF_COUNT("buffer.uncached.object_copy_reused");
                return {copy, offset};
            }
            const u32 whole = static_cast<u32>(immutable->end - immutable->begin);
            if (!IsRangeGpuWritten(immutable->begin, whole)) {
                const u32 generation = immutable->generation.load(std::memory_order_acquire);
                const u32 gpu_writes = immutable->gpu_writes.load(std::memory_order_acquire);
                const auto [data, at] = CopyRing().Map(whole, instance.UniformMinAlignment());
                {
                    Core::N0::CopyScope n0{Core::N0::CopyObject, immutable->begin, whole};
                    memory->CopySparseMemory(immutable->begin, data, whole);
                    n0.Done(data);
                }
                // M2: the bytes of the record's submission; a copy that was corrected must not be reused by a later generation's check, so it records a generation nobody has
                const bool corrected = Core::NativeContent::FrameStoreServe(Core::NativeContent::LateKind::BufferObjectCopy, immutable->begin, data, whole) != 0;
                Core::NativeContent::NoteLateRead(Core::NativeContent::LateKind::BufferObjectCopy, immutable->begin, whole);
                Core::NativeContent::SilentSynced(1, immutable->begin, whole); // W1.5 census
                CopyRing().Commit();
                immutable->copy_offset = at;
                immutable->copy_wraps = CopyRing().WrapCount();
                immutable->copy_generation = corrected ? ~generation : generation;
                immutable->copy_gpu_writes = gpu_writes;
                Core::NativeResources::NoteResident(1);
                Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::StreamCopy, whole);
                TLG_PROF_ADD("buffer.uncached.object_copy_made (sum = bytes)", whole);
                return {&CopyRing(), at + (device_addr - immutable->begin)};
            }
        }
        if (immutable != nullptr && immutable->usage != 1 && IsRegionGpuModified(device_addr, size)) {
            // a resident (mutable usage) buffer the GPU modified: the normal path decides
            Core::NativeResources::NoteResident(4);
            immutable = nullptr;
        }
        if (immutable != nullptr) {
            // a resident buffer is also asked of the tracker: the fault handler marks a page after it bumped the generation (the mark can land after a synchronization that
            // saw nothing), so the generation alone is not enough; the tracker's dirty bit is the safety net of the engine's own knowledge
            if (immutable->synced.load(std::memory_order_acquire) && !Core::NativeResources::ImmutableCheck(*immutable) &&
                (immutable->usage == 1 || !IsRegionCpuModified(device_addr, size))) {
                const auto* arena = BufferFor(device_addr, size); // B9.4: the object's own Vulkan buffer when it is placed
                if (immutable->resident_epoch.load(std::memory_order_relaxed) != PlacementEpoch() && immutable->begin >= arena->cpu_addr &&
                    immutable->end <= arena->cpu_addr + arena->size_bytes) {
                    // R1: the arenas moved since the synchronization; the object is resident in this one now
                    NoteResolutionUser(arena, immutable->id);
                    immutable->resident_buffer.store(arena, std::memory_order_relaxed);
                    immutable->resident_base.store(arena->cpu_addr, std::memory_order_relaxed);
                    immutable->resident_epoch.store(PlacementEpoch(), std::memory_order_release);
                }
                Core::NativeResources::NoteImmutableHit();
                Core::NativeContent::SilentResidentUse(0, immutable->begin, device_addr, size, immutable->usage); // W1.5 census
                if (immutable->usage != 1) {
                    Core::NativeResources::NoteResident(0);
                }
                TLG_PROF_COUNT("buffer.uncached.immutable_resident");
                return {arena, arena->Offset(device_addr)};
            }
            if (StreamWhilePassOpen(device_addr, size)) {
                // O1 / C6 step 1: a small read-only range of a buffer that needs a synchronization while a render pass is open goes through the stream buffer
                // (no copy command, so the pass is not cut); the buffer stays unsynchronized and the next use outside a pass synchronizes it
                Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::StreamCopy, size);
                return StreamCopy(device_addr, size);
            }
            const u32 generation = immutable->generation.load(std::memory_order_acquire);
            const u32 whole = static_cast<u32>(immutable->end - immutable->begin);
            const auto* arena = BufferFor(immutable->begin, immutable->end - immutable->begin);
            last_upload_bytes = 0;
            Core::N0::SetSyncSite(Core::N0::SyncImmutable);
            SynchronizeMemory(arena, immutable->begin, whole, false, false);
            Core::NativeContent::SilentSynced(0, immutable->begin, whole); // W1.5 census
            // R1: the object keeps where it is resident (read by the command processor's binding resolution once `synced` is set)
            NoteResolutionUser(arena, immutable->id);
            immutable->resident_buffer.store(arena, std::memory_order_relaxed);
            immutable->resident_base.store(arena->cpu_addr, std::memory_order_relaxed);
            immutable->resident_epoch.store(PlacementEpoch(), std::memory_order_relaxed);
            Core::NativeResources::ImmutableSynced(*immutable, generation);
            Core::NativeResources::NoteImmutableSync();
            if (immutable->usage != 1) {
                Core::NativeResources::NoteResident(1);
            }
            TLG_PROF_ADD("buffer.uncached.immutable_synced (sum = bytes uploaded)", last_upload_bytes);
            return {arena, arena->Offset(device_addr)};
        }
    }
    imm_scope.End();
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    TLG_PROF_NAMED_LEVEL(stream_scope, "buffer.part.stream", 3);
    if (!is_written && ((size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) || (!is_texel_buffer && StreamWhilePassOpen(device_addr, size)))) {
        // Stream copies are at most STREAM_THRESHOLD bytes, so the size fits in the low 16 bits.
        const u64 reuse_key = (device_addr << 16) | size;
        auto& reuse = stream_reuse_sets[Vulkan::Scheduler::SideSelected() ? 1 : 0];
        if (StreamReuseOn()) {
            const u64 tick = scheduler.ResolveEpoch();
            const u64 wraps = CopyRing().WrapCount();
            // Copies are per submission, with or without the store: the game rewrites constants (camera, view) in place between submissions, a copy reused across them is stale (W0, owner play 2026-11-04:
            // random lines on the whole screen while walking when the store did not serve; this key was tied to the store serving)
            const u32 record_submit = Core::NativeContent::CurrentRecordSubmit();
            if (tick != reuse.tick || wraps != reuse.wraps || record_submit != reuse.submit) {
                reuse.submit = record_submit;
                // A new command buffer: the game may have changed the data since. A wrap of the
                // stream buffer may overwrite earlier copies.
                reuse.tick = tick;
                reuse.wraps = wraps;
                reuse.map.clear();
                reuse.min = ~0ULL;
                reuse.max = 0;
            } else if (const auto it = reuse.map.find(reuse_key); it != reuse.map.end()) {
                Core::NativeResources::NoteBufferRequest(device_addr, size, false,
                                                         Core::NativeResources::BufferPath::StreamReuse, 0);
                TLG_PROF_COUNT("buffer.uncached.stream_reuse_hit");
                return {&CopyRing(), it->second};
            }
        }
        const auto [data, offset] = CopyRing().Map(size, instance.UniformMinAlignment());
        {
            Core::N0::CopyScope n0{Core::N0::CopyStream, device_addr, size};
            memory->CopySparseMemory(device_addr, data, size);
            n0.Done(data);
        }
        Core::NativeContent::FrameStoreServe(Core::NativeContent::LateKind::BufferStreamCopy, device_addr, data, size); // M2
        Core::NativeContent::NoteLateRead(Core::NativeContent::LateKind::BufferStreamCopy, device_addr, size);
        CopyRing().Commit();
        Core::NativeResources::NoteBufferRequest(device_addr, size, false,
                                                 Core::NativeResources::BufferPath::StreamCopy, size);
        if (StreamReuseOn() && CopyRing().WrapCount() == reuse.wraps) {
            reuse.map.emplace(reuse_key, offset);
            reuse.min = std::min<VAddr>(reuse.min, device_addr);
            reuse.max = std::max<VAddr>(reuse.max, device_addr + size);
        }
        TLG_PROF_ADD("buffer.uncached.stream_copy (sum = bytes)", size);
        return {&CopyRing(), offset};
    }
    stream_scope.End();
    TLG_PROF_NAMED_LEVEL(memo_scope, "buffer.part.arena_memo", 3);
    const bool memoizable = !is_texel_buffer;
    ObtainMemo* memo = nullptr;
    u64 epoch_before = 0;
    if (memoizable) {
        if (!obtain_memo) {
            obtain_memo = std::make_unique<ObtainMemo[]>(ObtainMemoSlots);
        }
        const u64 hash = (device_addr * 0x9E3779B97F4A7C15ull) ^ (u64(size) * 0xC2B2AE3D27D4EB4Full) ^ (is_written ? 0x55 : 0);
        memo = &obtain_memo[(hash >> 20) & (ObtainMemoSlots - 1)];
        epoch_before = cpu_epoch.load(std::memory_order_acquire) + range_epoch.load(std::memory_order_acquire) + b9_epoch.load(std::memory_order_acquire);
        if (memo->addr == device_addr && memo->size == size && memo->flags == (is_written ? 1u : 0u) &&
            memo->epoch == epoch_before && memo->tick == scheduler.ResolveEpoch()) {
            Core::NativeResources::NoteBufferRequest(device_addr, size, is_written,
                                                     Core::NativeResources::BufferPath::Memo, 0);
            TLG_PROF_COUNT("buffer.uncached.obtain_memo_hit");
            return {memo->arena, memo->offset};
        }

    }
    memo_scope.End();
    TLG_PROF_NAMED_LEVEL(for_scope, "buffer.part.buffer_for", 3);
    const auto* arena = BufferFor(device_addr, size, true, is_written); // B9.4: a placed engine buffer's own Vulkan buffer, or the arena
    for_scope.End();
    last_upload_bytes = 0;
    Core::N0::SetSyncSite(is_written ? Core::N0::SyncArenaWritten : is_texel_buffer ? Core::N0::SyncArenaTexel : Core::N0::SyncArenaRead);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    TLG_PROF_NAMED_LEVEL(post_scope, "buffer.part.after_sync", 3);
    if (last_upload_bytes != 0) {
        if (Core::NativeContent::CurrentBindIsCopyDestination()) {
            Core::NativeContent::NoteCopyDestinationSync(size); // the copy overwrites the whole range right after: not a read that feeds a draw
        } else {
            Core::NativeContent::NoteLateRead(Core::NativeContent::LateKind::BufferArenaSync, device_addr, size);
        }
    }
    Core::NativeResources::NoteBufferRequest(device_addr, size, is_written,
                                             Core::NativeResources::BufferPath::Arena, last_upload_bytes);
    if (is_written) {
        std::unique_lock lock{gpu_ranges_mutex};
        if (!gpu_modified_ranges.Intersects(device_addr, size)) {
            NativeIdentity::Note(device_addr, device_addr + size, NativeIdentity::GpuWrittenState);
        }
        gpu_modified_ranges.Add(device_addr, size);
        static const bool who_on = [] { const char* e = std::getenv("TLG_GPUWRITE_WHO"); return e && e[0] == '1'; }();
        if (who_on) {
            if (gpu_writer_who.size() > 200000) {
                gpu_writer_who.clear();
            }
            const auto [pgm, sw_stage] = Core::NativeRegistry::BufferWriter();
            gpu_writer_who[device_addr] = {size, fmt::format("shader {:#x} stage {} class {} seq {} tick {}{}", pgm, sw_stage, Core::NativeRegistry::CurrentDrawClass(),
                                                             Core::NativeRegistry::CurrentDrawSeq(), scheduler.CurrentTick(),
                                                             sw_stage >= 100 && sw_stage < 103 ? " source " + Core::NativeDevice::OwnerName(pgm) : std::string{})};
        }
        Core::NativeResources::NoteGpuWritten(device_addr, size); // R1b: copies of engine buffers with these bytes are stale
    }
    TLG_PROF_ADD("buffer.uncached.arena_synchronized (sum = bytes uploaded)", last_upload_bytes);
    const u64 offset = arena->Offset(device_addr);
    if (memo) {
        // the epoch read before the work: an upload this call did makes the entry stale, as it should
        *memo = {device_addr, size, is_written ? 1u : 0u, arena, offset, epoch_before, scheduler.ResolveEpoch()};
    }
    return {arena, offset};
}

u32 BufferCache::WaitForExplicitDownloads(VAddr device_addr, u64 size, u32 timeout_ms) {
    std::unique_lock lock{explicit_mutex};
    const auto overlaps = [&] {
        return std::ranges::any_of(explicit_inflight, [&](const ExplicitDownload& download) {
            return download.addr < device_addr + size && device_addr < download.addr + download.size;
        });
    };
    if (!overlaps()) {
        return 0;
    }
    return explicit_landed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return !overlaps(); }) ? 1 : 2;
}

std::pair<const Buffer*, u64> BufferCache::StreamCopy(VAddr device_addr, u32 size) {
    const auto [data, offset] = CopyRing().Map(size, instance.UniformMinAlignment());
    {
        Core::N0::CopyScope n0{Core::N0::CopyStreamFn, device_addr, size};
        memory->CopySparseMemory(device_addr, data, size);
        n0.Done(data);
    }
    CopyRing().Commit();
    return {&CopyRing(), offset};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForDownload(VAddr device_addr, u32 size) {
    const auto* arena = BufferFor(device_addr, size);
    last_upload_bytes = 0;
    Core::N0::SetSyncSite(Core::N0::SyncDownload);
    SynchronizeMemory(arena, device_addr, size, false, false);
    Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::Arena, last_upload_bytes);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBufResolve};
    Core::SplitCensus::Touch(Core::SplitCensus::BufObtain);
    if (IsRegionGpuModified(device_addr, size)) {
        last_image_source_gpu = true;
        Core::NativeContent::SetBindContext({"image source (GPU modified region)", 0, 0, 0, 0, false, false, device_addr, size, reinterpret_cast<u64>(__builtin_return_address(0))});
        const auto gpu_result = ObtainBuffer(device_addr, size, false);
        Core::NativeContent::ClearBindContext();
        return gpu_result;
    }
    last_image_source_gpu = false;
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    {
        Core::N0::CopyScope n0{Core::N0::CopyForImage, device_addr, staging.size};
        memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
        n0.Done(staging.mapped);
    }
    Core::NativeContent::CheckUploadMemory(device_addr, size); // F2.5: outside the ring, was the memory written after the draw's record?
    Core::NativeContent::OverlayImage(device_addr, size, staging.mapped); // F2.2 swap (TLG_F2_RING_STORE=1): ring bytes as announced, not as the ring holds them now
    staging.Flush();
    return {staging.buffer, staging.offset};
}

const Buffer* BufferCache::BufferFor(VAddr address, u64 size, bool demote_spanning, bool written) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBufResolve};
    Core::SplitCensus::Touch(Core::SplitCensus::BufObtain);
    OrderedViewCheck(address, size, written);
    if (const PlacedBuffer* entry = PlacedAt(address, size, written)) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedServed);
        return entry->buffer.get();
    }
    if (!demote_spanning && PlacedMeets(address, size)) {
        return nullptr; // a range over placed buffers' edges is left to their own requests
    }
    TLG_PROF_LEVEL("buffer.bufferfor.slow_block", 3);
    // the block the request needs: the engine buffers it meets and the placed buffers it meets, grown until no buffer reaches over its edges
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    static thread_local std::vector<u64> textures;
    VAddr lo = address, hi = address + size;
    for (u32 round = 0; round < 4; ++round) {
        parts.clear();
        BuffersInView(lo, hi - lo, parts);
        VAddr new_lo = lo, new_hi = hi;
        for (const auto& part : parts) {
            new_lo = std::min(new_lo, part.object_begin);
            new_hi = std::max(new_hi, part.object_end);
        }
        for (auto it = placed.upper_bound(new_lo) == placed.begin() ? placed.begin() : std::prev(placed.upper_bound(new_lo)); it != placed.end() && it->first < new_hi; ++it) {
            if (it->second.end > new_lo) {
                new_lo = std::min(new_lo, it->second.begin);
                new_hi = std::max(new_hi, it->second.end);
            }
        }
        if (new_lo == lo && new_hi == hi) {
            break;
        }
        lo = new_lo;
        hi = new_hi;
    }
    const bool any_buffer = !parts.empty();
    // members may overlap each other and textures may lie in the block: one Vulkan buffer is one memory range
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.object_begin < b.object_begin; });
    std::vector<u64> ids;
    VAddr last_end = 0;
    bool overlapping = false;
    for (const auto& part : parts) {
        overlapping |= part.object_begin < last_end;
        last_end = std::max(last_end, part.object_end);
        ids.push_back(part.id);
    }
    textures.clear();
    (void)Core::NativeDevice::TexturesIn(lo, hi - lo, textures);
    if (!any_buffer) {
        // E4: memory no engine buffer owns gets a pseudo resource of its own: the frame allocator's ring as a whole (the engine's allocator hook found it),
        // any other range as the request's block (grown over the placed buffers it meets)
        const auto& ring = VideoCore::UntrackedRing::Get();
        if (ring.Active() && address >= ring.Start() && address + size <= ring.End() && !PlacedMeets(ring.Start(), ring.End() - ring.Start())) {
            lo = ring.Start();
            hi = ring.End();
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4PseudoRing);
        }
        ids.push_back((u64{1} << 63) | lo);
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4PseudoPlaced);
    } else if (ids.size() > 1 || PlacedMeets(lo, hi - lo) || lo != parts.front().object_begin || hi != parts.back().object_end) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan, "B9.4: request {:#x}+{:#x} places a block {:#x}+{:#x} of {} engine buffers", address, size, lo, hi - lo, ids.size());
        }
    }
    if (hi - lo > (u64{256} << 20)) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan, "E4: request {:#x}+{:#x} places a block of {} MB", address, size, (hi - lo) >> 20);
        }
    }
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedServed);
    if (overlapping) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BlockAliased);
    }
    if (!textures.empty()) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BlockWithTextures);
    }
    return Place(std::move(ids), lo, hi);
}

bool BufferCache::NativeCpuQuery(VAddr addr, size_t size) {
    // E4: memory no engine buffer answers for: changed when announced bytes of it were not uploaded yet
    std::scoped_lock lock{object_dirty_mutex};
    const u32 lean = Core::NativeUpdates::LeanDirtyMode();
    if (Core::RingMirror::FillMode() == 1 && ring_fill_pending.Intersects(addr, size)) {
        return true;
    }
    if (lean == 2) {
        return b9_unattributed.Intersects(addr, size);
    }
    const bool unattributed = b9_unattributed.Intersects(addr, size);
    const bool answer = all_object_dirty.Intersects(addr, size) || unattributed || (Core::RingMirror::FillMode() == 1 && ring_fill_pending.Intersects(addr, size));
    if (lean == 1) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::LeanChecked);
        if (answer && !unattributed) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::LeanMissing, size);
            Core::RareEvents::Note("d2_lean_cpu_query_missing", fmt::format("{:#x}+{:#x}", addr, size));
        }
    }
    return answer;
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    if (Core::NativeUpdates::B9TrackerOff()) {
        // B9.3 step 2: the tracker is not kept for engine buffers: a range over several of them is answered by theirs, the bytes between them by the tracker
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        BuffersInView(addr, size, parts);
        if (parts.size() > 1 || (parts.size() == 1 && (parts[0].begin != addr || parts[0].end != addr + size))) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuQueries);
            std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
            VAddr at = addr;
            {
                std::scoped_lock lock{object_dirty_mutex};
                if (b9_unattributed.Intersects(addr, size)) {
                    return true;
                }
                for (const auto& part : parts) {
                    if (const auto it = object_dirty.find(part.id); it != object_dirty.end() && it->second.Intersects(part.begin, part.end - part.begin)) {
                        return true;
                    }
                }
            }
            for (const auto& part : parts) {
                if (part.begin > at && NativeCpuQuery(at, part.begin - at)) {
                    return true;
                }
                at = std::max<VAddr>(at, part.end);
            }
            return at < addr + size && NativeCpuQuery(at, addr + size - at);
        }
    }
    // B9.3: a range inside one engine buffer: changed when the buffer holds announced bytes there that were not uploaded
    if (const u32 mode = Core::NativeUpdates::B9QueryMode(); mode != 0 && Core::NativeUpdates::B9BufferMode() != 0) {
        if (const u64 id = BufferHolding(addr, size); id != 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuQueries);
            bool object_says;
            {
                std::scoped_lock lock{object_dirty_mutex};
                const auto it = object_dirty.find(id);
                object_says = (it != object_dirty.end() && it->second.Intersects(addr, size)) || b9_unattributed.Intersects(addr, size);
            }
            return object_says;
        }
    }
    return NativeCpuQuery(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    // E4: the GPU written bytes answer for every range (engine buffers or not): bytes a shader or a GPU copy wrote and nothing superseded or downloaded since
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GpuQueries);
    std::shared_lock lock{gpu_ranges_mutex};
    return gpu_modified_ranges.Intersects(addr, size);
}

void BufferCache::SupersedeGpuWritten(VAddr addr, size_t size) {
    Core::SplitCensus::Touch(Core::SplitCensus::GpuRanges);
    Core::UiTrace::NoteSupersede(addr, size, reinterpret_cast<u64>(__builtin_return_address(0)));
    std::unique_lock lock{gpu_ranges_mutex};
    BindMemoBump(); // GPU written bytes change hands: memo answers are checked again
    if (gpu_modified_ranges.Intersects(addr, size)) {
        Core::N0::SetEffect(Core::N0::EffGpuRanges);
        NativeIdentity::Note(addr, addr + size, NativeIdentity::GpuWrittenState);
    }
    gpu_modified_ranges.Subtract(addr, size);
}

bool BufferCache::IsRangeGpuWritten(VAddr addr, size_t size) {
    Core::SplitCensus::Count(Core::SplitCensus::CtGpuWrittenCheck);
    std::shared_lock lock{gpu_ranges_mutex};
    return gpu_modified_ranges.Intersects(addr, size);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBufResolve};
    Core::SplitCensus::Touch(Core::SplitCensus::BufSync);
    Core::N0::SyncScope n0_sync{size, last_upload_bytes};
    if (arena != nullptr) {
        const_cast<Buffer*>(arena)->NoteResolveUse(); // P3c: whoever synchronizes an arena asks for it in this operation
    }
    TLG_PROF("buffer.synchronize");
    TLG_PROF_NAMED_LEVEL(s_placed, "buffer.sync.placed_walk", 3);
    // B9.4: a placed buffer holds [begin, end) of its block; a read into its slack must not take (and lose) the bytes of the memory after it
    VAddr held_lo = device_addr, held_hi = device_addr + size;
    const u32 sync_id_mode = arena->placed_begin != 0 ? Core::NativeUpdates::SyncByIdMode() : 0u; // N3
    if (sync_id_mode == 1) {
        held_lo = std::max<VAddr>(held_lo, arena->placed_begin);
        held_hi = std::min<VAddr>(held_hi, arena->placed_end);
        if (held_lo >= held_hi) {
            return false;
        }
    } else {
        std::scoped_lock lock{placed_mutex};
        for (auto it = placed.upper_bound(arena->cpu_addr + 255); it != placed.begin();) {
            --it;
            if (it->second.buffer.get() == arena) {
                held_lo = std::max<VAddr>(held_lo, it->second.begin);
                held_hi = std::min<VAddr>(held_hi, it->second.end);
                break;
            }
            if (it->first < arena->cpu_addr) {
                break;
            }
        }
        if (held_lo >= held_hi) {
            return false;
        }
        if (sync_id_mode == 2) {
            const VAddr told_lo = std::max<VAddr>(device_addr, arena->placed_begin), told_hi = std::min<VAddr>(device_addr + size, arena->placed_end);
            Core::NativeUpdates::NoteBufferId(told_lo == held_lo && told_hi == held_hi ? Core::NativeUpdates::IdSyncHeldMatch : Core::NativeUpdates::IdSyncHeldDiffer);
        }
    }
    s_placed.End();
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    u32 n0_cause = 0; // N0 census: why the next piece is uploaded
    const auto add_piece = [&](u64 addr, u64 copy_size) {
        {
            // B9.4: a placed engine buffer takes only the memory it holds (a page range of the tracker, or a read past its end into the slack, can reach beyond)
            const u64 lo = std::max<u64>(addr, held_lo), hi = std::min<u64>(addr + copy_size, held_hi);
            if (lo >= hi) {
                return;
            }
            addr = lo;
            copy_size = hi - lo;
        }
        Core::N0::NoteCause(n0_cause, copy_size);
        copies.emplace_back(total_size_bytes, addr, copy_size);
        total_size_bytes += copy_size;
    };
    // B5 / B6: bytes the GPU wrote (and nothing the CPU wrote since) are the GPU's: a page upload of PS4 memory leaves them out
    static const bool keep_gpu_bytes = Core::NativeUpdates::B2NativeMode() != 0;
    const auto add_copy = [&](u64 addr, u64 copy_size) {
        if (!keep_gpu_bytes) {
            add_piece(addr, copy_size);
            return;
        }
        std::shared_lock lock{gpu_ranges_mutex};
        if (!gpu_modified_ranges.Intersects(addr, copy_size)) {
            lock.unlock();
            add_piece(addr, copy_size);
            return;
        }
        upload_bytes_kept_for_gpu.fetch_add(1, std::memory_order_relaxed);
        boost::container::small_vector<std::pair<u64, u64>, 8> pieces;
        gpu_modified_ranges.ForEachNotInRange(addr, copy_size, [&](u64 a, u64 n) { pieces.emplace_back(a, n); });
        lock.unlock();
        for (const auto& [a, n] : pieces) {
            add_piece(a, n);
        }
    };
    // B9.1: an upload over engine buffers: their announced bytes against the page tracker's ranges (B9.4 step 2: a request may cover several buffers of a block;
    // the bytes between them, which no buffer holds, keep the tracker)
    TLG_PROF_NAMED_LEVEL(s_b9, "buffer.sync.b9_dirty", 3);
    const u32 b9 = Core::NativeUpdates::B9BufferMode();
    u64 b9_object = 0; // the first engine buffer the request meets (0: none)
    boost::container::small_vector<std::pair<u64, u64>, 8> b9_pieces;
    boost::container::small_vector<std::pair<u64, u64>, 4> b9_gaps; // the request's bytes no engine buffer holds
    if (b9 != 0) {
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        if (sync_id_mode == 1 && arena->placed_single_id != 0) {
            // N3: the buffer was placed for exactly this engine buffer: its part of the request is known without the walk of the engine's address table
            parts.push_back({arena->placed_single_id, held_lo, held_hi, 0, 0, arena->placed_begin, arena->placed_end});
            Core::NativeUpdates::NoteBufferId(Core::NativeUpdates::IdSyncToldUsed);
        } else {
            BuffersInView(held_lo, held_hi - held_lo, parts);
            if (sync_id_mode == 2 && arena->placed_single_id != 0) {
                const bool same = parts.size() == 1 && parts[0].id == arena->placed_single_id && parts[0].begin == held_lo && parts[0].end == held_hi;
                Core::NativeUpdates::NoteBufferId(same ? Core::NativeUpdates::IdSyncPartsMatch : Core::NativeUpdates::IdSyncPartsDiffer);
                if (!same) {
                    std::string what;
                    for (const auto& part : parts) {
                        what += fmt::format(" [id {} {:#x}-{:#x} object {:#x}-{:#x}]", part.id, part.begin, part.end, part.object_begin, part.object_end);
                    }
                    Core::RareEvents::Note("n3_sync_parts_differ", fmt::format("placed id {} {:#x}-{:#x}, held {:#x}-{:#x}, address walk gave {} parts:{}", arena->placed_single_id, arena->placed_begin, arena->placed_end, held_lo, held_hi, parts.size(), what));
                }
            }
        }
        if (parts.empty()) {
            // the walk by memory found nothing: the buffer holding the request, found the long way (the request's bytes must not be left to the tracker)
            Core::NativeDevice::Object object;
            if (ResourceAtView(device_addr, object) && object.kind == Core::NativeDevice::Kind::Buffer) {
                const VAddr object_end = object.address + std::max<u64>(object.bytes, 1);
                parts.push_back({object.id, held_lo, std::min<VAddr>(object_end, held_hi), object.usage, object.cpu_access, object.address, object_end});
            }
        }
        if (!parts.empty()) {
            std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
            b9_object = parts.front().id;
            VAddr at = held_lo;
            for (const auto& part : parts) {
                if (part.begin > at) {
                    b9_gaps.emplace_back(at, part.begin - at);
                }
                at = std::max<VAddr>(at, part.end);
            }
            if (at < held_hi) {
                b9_gaps.emplace_back(at, held_hi - at);
            }
            std::scoped_lock lock{object_dirty_mutex};
            for (const auto& part : parts) {
                // (a subtraction only where the walk found bytes: with nothing in the range it changes nothing, and it is the costly half of the call)
                if (const auto it = object_dirty.find(part.id); it != object_dirty.end()) {
                    const size_t before = b9_pieces.size();
                    it->second.ForEachInRange(part.begin, part.end - part.begin, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); }); // (start, end) pairs
                    if (b9_pieces.size() != before) {
                        it->second.Subtract(part.begin, part.end - part.begin);
                    }
                }
                const size_t before_unattributed = b9_pieces.size();
                b9_unattributed.ForEachInRange(part.begin, part.end - part.begin, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); });
                if (b9_pieces.size() != before_unattributed) {
                    b9_unattributed.Subtract(part.begin, part.end - part.begin);
                }
            }
            if (Core::NativeUpdates::LeanDirtyMode() != 2) {
                for (const auto& [a, n] : b9_pieces) {
                    all_object_dirty.Subtract(a, n);
                }
            }
        }
        Core::NativeUpdates::NoteB9(b9_object != 0 ? Core::NativeUpdates::B9SyncByObject : Core::NativeUpdates::B9SyncNoObject);
    }
    s_b9.End();
    TLG_PROF_NAMED_LEVEL(s_ann, "buffer.sync.announced", 3);
    // the parts of [addr, addr + n) between the engine buffers of the request (they keep the tracker's copies in native mode)
    const auto for_each_gap = [&](u64 addr, u64 n, auto&& fn) {
        for (const auto& [g, gn] : b9_gaps) {
            const u64 lo = std::max<u64>(addr, g), hi = std::min<u64>(addr + n, g + gn);
            if (lo < hi) {
                fn(lo, hi - lo);
            }
        }
    };
    const bool b9_native = b9 == 2 && b9_object != 0;
    // E4: memory no engine buffer holds takes the announced bytes (the placement already copied the rest)
    // the announced bytes inside [x, x + xn) of memory no engine buffer holds (a request over unowned memory, or a gap between the buffers of one): uploaded, then no longer announced
    boost::container::small_vector<std::pair<u64, u64>, 8>* announced_sink = nullptr; // N2: the announced pieces a ring window takes
    const auto take_announced = [&](u64 x, u64 xn) {
        boost::container::small_vector<std::pair<u64, u64>, 8> from_announced;
        {
            std::scoped_lock lock{object_dirty_mutex};
            const u32 lean = Core::NativeUpdates::LeanDirtyMode();
            if (lean == 2) {
                b9_unattributed.ForEachInRange(x, xn, [&](u64 lo, u64 e) { from_announced.emplace_back(lo, e - lo); });
            } else {
                all_object_dirty.ForEachInRange(x, xn, [&](u64 lo, u64 e) { from_announced.emplace_back(lo, e - lo); });
                if (lean == 1) {
                    // compare with the unattributed set alone: every byte of the audit set must be in it (missing = stale if the audit set were dropped)
                    boost::container::small_vector<std::pair<u64, u64>, 8> lean_pieces;
                    b9_unattributed.ForEachInRange(x, xn, [&](u64 lo, u64 e) { lean_pieces.emplace_back(lo, e - lo); });
                    const auto covered = [](const auto& list, u64 lo, u64 n) {
                        u64 at = lo;
                        for (const auto& [a, an] : list) { // (merged, sorted, disjoint)
                            if (a + an <= at) {
                                continue;
                            }
                            if (a > at) {
                                break;
                            }
                            at = a + an;
                            if (at >= lo + n) {
                                break;
                            }
                        }
                        return at >= lo + n;
                    };
                    u64 missing = 0, extra = 0;
                    for (const auto& [lo, n] : from_announced) {
                        if (!covered(lean_pieces, lo, n)) {
                            missing += n;
                        }
                    }
                    for (const auto& [lo, n] : lean_pieces) {
                        if (!covered(from_announced, lo, n)) {
                            extra += n;
                        }
                    }
                    Core::NativeUpdates::NoteB9(Core::NativeUpdates::LeanChecked);
                    if (missing != 0) {
                        Core::NativeUpdates::NoteB9(Core::NativeUpdates::LeanMissing, missing);
                        Core::RareEvents::Note("d2_lean_take_announced_missing", fmt::format("{:#x}+{:#x}: {} bytes", x, xn, missing));
                    }
                    if (extra != 0) {
                        Core::NativeUpdates::NoteB9(Core::NativeUpdates::LeanExtra, extra);
                    }
                }
            }
            for (const auto& [lo, n] : from_announced) {
                if (lean != 2) {
                    all_object_dirty.Subtract(lo, n);
                }
                b9_unattributed.Subtract(lo, n);
            }
        }
        for (const auto& [lo, n] : from_announced) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4AnnouncedUploads, n);
            n0_cause = 2;
            if (announced_sink != nullptr) {
                announced_sink->emplace_back(lo, n); // (a ring window merges its pieces before it uploads them)
                continue;
            }
            add_copy(lo, n);
        }
        if (!from_announced.empty()) {
            cpu_epoch.fetch_add(1, std::memory_order_acq_rel); // the upload memo's answers are stale
        }
    };
    const auto announced_ranges = [&](VAddr a, u64 n) {
        if (b9_object == 0) {
            take_announced(a, n);
        } else {
            for_each_gap(a, n, take_announced);
        }
    };
    const auto& ring = UntrackedRing::Get();
    VAddr ring_lo = 0, ring_hi = 0; // the part the untracked ring uploads whole (native mode adds no announced piece there)
    if (ring.Active() && !is_written) {
        const auto [lo, hi] = ring.Clip(device_addr, size);
        if (lo < hi && !IsRegionGpuModified(lo, hi - lo)) {
            ring_lo = lo;
            ring_hi = hi;
            // Untracked upload ring: nobody told us what the CPU wrote, so the requested part is uploaded.
            if (device_addr < lo) {
                announced_ranges(device_addr, lo - device_addr);
            }
            const bool mirror_real = Core::RingMirror::Mode() == 1 && arena->cpu_addr <= ring.Start() && arena->cpu_addr + arena->SizeBytes() >= ring.End();
            if (mirror_real) {
                // N2 real mirror: the arena of the ring window is the ring's own buffer. A part of the ring is uploaded whole the first time a window reads it; after that only the bytes the
                // engine announced for it are uploaded (and nothing at all when none were).
                ring_lo = ring_hi = 0; // (the announced pieces of engine buffers inside the ring are uploaded by the loop below)
                boost::container::small_vector<std::pair<u64, u64>, 8> unfilled, filled_parts, taken;
                {
                    std::scoped_lock lock{object_dirty_mutex};
                    if (arena->ring_mirror_token == 0 || arena->ring_mirror_token != ring_mirror_token_current) {
                        arena->ring_mirror_token = ++ring_mirror_token_next; // another buffer is the mirror now: it holds nothing yet
                        ring_mirror_token_current = arena->ring_mirror_token;
                        ring_filled.Clear();
                        ring_fill_pending.Clear();
                    }
                    ring_filled.ForEachInRange(lo, hi - lo, [&](u64 a, u64 e) { filled_parts.emplace_back(a, e - a); });
                    ring_filled.ForEachNotInRange(lo, hi - lo, [&](u64 a, u64 n) { unfilled.emplace_back(a, n); });
                    for (const auto& [a, n] : unfilled) {
                        ring_filled.Add(a, n);
                        b9_unattributed.Subtract(a, n); // (the marks before the upload are in it)
                        if (Core::NativeUpdates::LeanDirtyMode() != 2) {
                            all_object_dirty.Subtract(a, n);
                        }
                    }
                }
                u64 whole = 0;
                for (const auto& [a, n] : unfilled) {
                    n0_cause = 0;
                    add_copy(a, n);
                    whole += n;
                    Core::NativeUploads::NoteRingUpload(n);
                    Core::Flight::Add(Core::Flight::Upload, 0, a, n, 1); // L1
                }
                if (Core::RingMirror::FillMode() == 1) {
                    // the ring's own pending set: what was announced inside the filled parts of this window since they were last taken
                    std::scoped_lock lock{object_dirty_mutex};
                    for (const auto& [a, n] : filled_parts) {
                        boost::container::small_vector<std::pair<u64, u64>, 8> pend;
                        ring_fill_pending.ForEachInRange(a, n, [&](u64 lo, u64 e) { pend.emplace_back(lo, e - lo); });
                        for (const auto& [lo, len] : pend) {
                            ring_fill_pending.Subtract(lo, len);
                            taken.emplace_back(lo, len);
                            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4AnnouncedUploads, len);
                        }
                    }
                }
                announced_sink = &taken;
                for (const auto& [a, n] : filled_parts) {
                    announced_ranges(a, n);
                }
                announced_sink = nullptr;
                u64 announced_bytes = 0, merged_bytes = 0;
                for (const auto& [a, n] : taken) {
                    announced_bytes += n;
                }
                if (Core::RingMirror::FillMode() == 2) {
                    // O2 shadow: every byte this window takes from the dirty sets must be one a fill at announce time would have uploaded
                    u64 missed = 0;
                    std::scoped_lock lock{object_dirty_mutex};
                    for (const auto& [a, n] : taken) {
                        ring_fill_pending.ForEachNotInRange(a, n, [&](u64, u64 miss) { missed += miss; });
                        ring_fill_pending.Subtract(a, n);
                    }
                    Core::RingMirror::NoteFill(Core::RingMirror::FillTakenBytes, announced_bytes);
                    Core::RingMirror::NoteFill(Core::RingMirror::FillTakenPieces, taken.size());
                    Core::RingMirror::NoteFill(Core::RingMirror::FillMissedBytes, missed);
                }
                {
                    // pieces close together are one copy: the gap bytes are current guest memory (a copy has a fixed cost, the pieces are mostly 100 to 500 bytes)
                    const u64 gap = Core::RingMirror::MergeGap();
                    boost::container::small_vector<std::pair<u64, u64>, 8> sorted(taken.begin(), taken.end());
                    std::sort(sorted.begin(), sorted.end());
                    u64 count_out = 0;
                    for (size_t i = 0; i < sorted.size();) {
                        u64 a = sorted[i].first, e = sorted[i].first + sorted[i].second;
                        size_t j = i + 1;
                        while (j < sorted.size() && sorted[j].first <= e + gap && (sorted[j].first + sorted[j].second) - a <= (u64{1} << 20)) {
                            merged_bytes += sorted[j].first > e ? sorted[j].first - e : 0;
                            e = std::max<u64>(e, sorted[j].first + sorted[j].second);
                            ++j;
                        }
                        n0_cause = 2;
                        add_copy(a, e - a);
                        ++count_out;
                        i = j;
                    }
                    Core::RingMirror::NoteReal(hi - lo, whole, announced_bytes, sorted.size(), count_out, merged_bytes);
                }
                if (Core::RingMirror::AuditOn()) {
                    // the host shadow beside it: what the window holds must equal what the mirror was given; a difference is uploaded now
                    taken.insert(taken.end(), unfilled.begin(), unfilled.end());
                    if (Core::RingMirror::NeedFill(ring.Start(), ring.End())) {
                        Core::RingMirror::Fill(ring.Start(), ring.End());
                    }
                    std::vector<std::pair<u64, u64>> repair;
                    Core::RingMirror::CheckWindow(lo, hi, taken.data(), taken.size(), &repair);
                    for (const auto& [a, n] : repair) {
                        n0_cause = 0;
                        add_copy(a, n);
                    }
                }
                if (hi < device_addr + size) {
                    announced_ranges(hi, device_addr + size - hi);
                }
            } else {
            if (Core::RingMirror::Mode() >= 2) {
                // N2 shadow: a host copy of the ring kept current by the announced bytes alone; the window must equal it (the uploads below are unchanged)
                boost::container::small_vector<std::pair<u64, u64>, 16> taken;
                {
                    std::scoped_lock lock{object_dirty_mutex};
                    if (Core::RingMirror::NeedFill(ring.Start(), ring.End())) {
                        b9_unattributed.Subtract(ring.Start(), ring.End() - ring.Start()); // (the marks before the fill are in the copy)
                        Core::RingMirror::Fill(ring.Start(), ring.End());
                    }
                    b9_unattributed.ForEachInRange(lo, hi - lo, [&](u64 a, u64 e) { taken.emplace_back(a, e - a); });
                    for (const auto& [a, n] : taken) {
                        b9_unattributed.Subtract(a, n);
                    }
                }
                Core::RingMirror::CheckWindow(lo, hi, taken.data(), taken.size());
            }
            // Read again in the same command buffer: the data cannot have changed since the first upload.
            auto& done = ring_uploads[(lo >> 6) & 255];
            const u64 tick = Vulkan::Scheduler::SideSelected() ? (side_upload_epoch | Vulkan::Scheduler::SideTag) : scheduler.ResolveEpoch();
            if (done.tick != tick || done.lo > lo || done.hi < hi) {
                n0_cause = 0;
                add_copy(lo, hi - lo);
                Core::NativeUploads::NoteRingUpload(hi - lo);
                Core::Flight::Add(Core::Flight::Upload, 0, lo, hi - lo, 1); // L1
                done = {lo, hi, tick};
            } else {
                Core::NativeUploads::NoteRingSkipped(hi - lo);
            }
            if (hi < device_addr + size) {
                announced_ranges(hi, device_addr + size - hi);
            }
            }
        } else {
            announced_ranges(device_addr, size);
        }
    } else {
        announced_ranges(device_addr, size);
    }
    if (b9_object != 0) {
        for (size_t i = 0; i < b9_pieces.size(); ++i) {
            auto [a, n] = b9_pieces[i];
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BytesByObject, n);
            if (!b9_native) {
                continue;
            }
            // native: the announced bytes are the upload (outside the part the untracked ring uploads whole)
            n0_cause = 1;
            const u64 end = a + n;
            if (ring_lo < ring_hi && a < ring_hi && ring_lo < end) {
                if (a < ring_lo) {
                    add_copy(a, ring_lo - a);
                }
                if (ring_hi < end) {
                    add_copy(ring_hi, end - ring_hi);
                }
                continue;
            }
            add_copy(a, n);
        }
    }
    s_ann.End();
    TLG_PROF_NAMED_LEVEL(s_up, "buffer.sync.upload", 3);
    Core::SplitCensus::PhaseScope upload_phase{Core::SplitCensus::PhaseUpload};
    last_upload_bytes += total_size_bytes;
    Core::NativeUploads::NoteUploaded(total_size_bytes);
    if (Core::NativeUploads::Enabled()) {
        for (const auto& copy : copies) {
            Core::NativeUploads::NoteUploadedRange(copy.dstOffset, copy.size);
        }
    }
    if (!copies.empty()) {
        // where uploads go: the arena (memory no placed buffer holds), a placed buffer in plain device memory, a placed buffer the CPU can write
        zero_copy_counts[arena->mem_type == MemoryType::DeviceMapped ? 9 : 8].fetch_add(1, std::memory_order_relaxed);
    }
    if (!copies.empty() && arena->mem_type == MemoryType::DeviceMapped && arena->is_device_local && !arena->mapped_data.empty() && arena->CpuAddr() != 0) {
        // zero copy: the GPU is done with this engine buffer (every submission that used it finished): its bytes are written straight into its video memory,
        // with no staging and no copy in the stream (the submission that follows makes the host writes visible). Otherwise the copy below runs as before.
        const bool tick_idle = (arena->last_gpu_tick[0] == 0 || scheduler.IsFree(arena->last_gpu_tick[0])) &&
                               (arena->last_gpu_tick[1] == 0 || scheduler.IsFree(arena->last_gpu_tick[1]));
        // P3c: the GPU being done with what the record side noted is not enough once the resolve side runs ahead: the operations resolved before this one must have been recorded too
        // (TLG_SPLIT_BIND=1 uses both; 2 uses the tick as before and counts where the sequence would have said no)
        static const u32 split_mode = [] {
            const char* e = std::getenv("TLG_SPLIT_BIND");
            return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u;
        }();
        const bool seq_idle = arena->EarlierUsesRetired();
        if (split_mode == 2) {
            static std::atomic<u64> checked{0}, would_refuse{0};
            const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
            if (tick_idle && !seq_idle && would_refuse.fetch_add(1) < 20) {
                LOG_WARNING(Render_Vulkan, "split bind verify: zero copy idle by tick but an earlier operation ({}) is not retired ({})", arena->resolve_prev_seq,
                            VideoCore::OpSequence::retired.load());
            }
            if ((n & (n - 1)) == 0) {
                LOG_WARNING(Render_Vulkan, "split bind verify: {} zero copy idle tests, {} refused by sequence only", n, would_refuse.load());
            }
        }
        const bool idle = tick_idle && (split_mode != 1 || seq_idle);
        if (idle) {
            u8* const base = const_cast<u8*>(arena->mapped_data.data());
            for (const auto& copy : copies) {
                u8* const dst = base + (copy.dstOffset - arena->cpu_addr);
                {
                    Core::N0::CopyScope n0{Core::N0::CopyZeroDirect, copy.dstOffset, copy.size};
                    memory->CopySparseMemory(copy.dstOffset, dst, copy.size);
                    n0.Done(dst);
                }
                // M2b: the announced bytes are those of the submission being worked on (the raster thread applies announcements in order), so the game having run ahead is undone here
                Core::NativeContent::FrameStoreServe(Core::NativeContent::LateKind::BufferArenaSync, copy.dstOffset, dst, copy.size);
            }
            if (!arena->is_coherent) {
                const_cast<Buffer*>(arena)->Flush(0, arena->SizeBytes());
            }
            zero_copy_counts[0].fetch_add(1, std::memory_order_relaxed);
            zero_copy_counts[1].fetch_add(copies.size(), std::memory_order_relaxed);
            zero_copy_counts[2].fetch_add(total_size_bytes, std::memory_order_relaxed);
            copies.clear();
        } else {
            zero_copy_counts[3].fetch_add(1, std::memory_order_relaxed);
            zero_copy_counts[4].fetch_add(total_size_bytes, std::memory_order_relaxed);
        }
    }
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            {
                Core::N0::CopyScope n0{Core::N0::CopyStaging, copy.dstOffset, copy.size};
                memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
                n0.Done(staging.mapped + copy.srcOffset);
            }
            Core::NativeContent::FrameStoreServe(Core::NativeContent::LateKind::BufferArenaSync, copy.dstOffset, staging.mapped + copy.srcOffset, copy.size); // M2b
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        if (ResolveCommands::Deferring()) {
            // plan S P3d: one ordered queue for everything the resolve half would record (image uploads read what this copy writes)
            ResolveCommands::Defer([this, source = staging.buffer, arena, copies = std::move(copies)]() { runtime.CopyBuffer(source, arena, copies); });
        } else if (t_upload_sink != nullptr) {
            t_upload_sink->push_back({staging.buffer, arena, std::move(copies)}); // plan S P1d: the record pass issues the copy
        } else {
            runtime.CopyBuffer(staging.buffer, arena, copies);
        }
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

void BufferCache::RunDeferredUploads(UploadSink& sink) {
    for (auto& u : sink) {
        runtime.CopyBuffer(u.staging, u.arena, u.copies);
    }
    sink.clear();
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    Core::N0::FromImageScope n0_image;
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBufResolve};
    Core::SplitCensus::Touch(Core::SplitCensus::BufSync);
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    n0_image.tiled = true;
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

} // namespace VideoCore
