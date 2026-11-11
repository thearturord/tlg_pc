// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/cb_db_extent.h"
#include "native/native_readback_policy.h"

#include "video_core/amdgpu/regs.h"
#include <deque>
#include <mutex>
#include <vector>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw_tag.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include <span>
#include <string>
#include "native/native_device.h"
#include "native/native_draw.h"
#include "native/native_uploads.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/vk_constant_ring.h"
#include "video_core/renderer_vulkan/vk_host_import.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/texture_cache/texture_cache.h"
#include "native/native_occlusion.h"
#include "native/native_raster_host.h"

namespace Core {
class NativeExecutor;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;
struct FixedState;
struct DrawInputs;

class Rasterizer : public Core::RasterHost {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        Core::NativeExecutor* executor);
    ~Rasterizer() override;

    // ---- Core::RasterHost (native/native_raster_host.h): the functions below with a host name are in vk_rasterizer_host.cpp; the others are the ones of this class ----
    Core::DrawPlanPtr MakeDrawPlan(Core::DrawIndirectParams params, const FixedState& native, u64 exec_key, u64 inputs_key, const Core::NativeDraw::ExecDraw& record,
                                   u32 index_count) override;
    Core::DrawPlanPtr MakeExecDrawPlan(const FixedState& native, u64 exec_key, u64 inputs_key, std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data,
                                       const Core::NativeDraw::ExecDraw& draw, u32 index_count, u32 draw_class) override;
    Core::DrawPlanPtr MakeOperationPlan(const Core::NativeDraw::ExecDraw& draw) override;
    Core::DispatchPlanPtr MakeDispatchPlan(const Core::NativeDispatch::DispatchInputs& cs) override;
    u64 DrawPlanDigest(const DrawPlan& plan) override;
    u64 DispatchPlanDigest(const DispatchPlan& plan) override;
    void RecycleDrawPlan(Core::DrawPlanPtr&& plan) override;
    void RecycleDispatchPlan(Core::DispatchPlanPtr&& plan) override;
    u64 DrawPlanSeq(const DrawPlan& plan) override;
    bool DrawPlanIsDepthOnly(const DrawPlan& plan) override;
    void SetDrawPlanTargets(DrawPlan& plan, const ColorExtents& cb_extent, const DepthExtent& db_extent, const std::array<u64, 18>& target_resources) override;
    const Core::NativeDispatch::DispatchInputs& DispatchPlanInputs(const DispatchPlan& plan) override;
    void TagDispatchPlan(DispatchPlan& plan, u64 seq) override;
    bool MarkersEnabled() const noexcept override;
    void MarkerBegin(std::string_view text) override;
    void MarkerEnd() override;
    IntervalStats PublishIntervalStats() override;

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    /// With the command processor split (cp_split_mode) the pipeline lookup is done by the command
    /// processor thread (PlanDraw / PlanDispatch) and the raster thread gets its result.
    void Draw(bool is_indexed, u32 index_offset = 0, const DrawPlan* plan = nullptr);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset,
                      const DrawPlan* plan = nullptr);

    void DispatchDirect(const DispatchPlan* plan = nullptr);
    void SetExplicitImageReadback(bool on);
    void DispatchIndirect(VAddr address, u32 offset, u32 size,
                          const DispatchPlan* plan = nullptr);

    /// Command processor thread (F3.5): plans a draw from its record: the pipeline is looked up from the record's keys, the shader and the target descriptions; `native` is the
    /// fixed-function state the native path built from the engine's objects.
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanDraw(DrawIndirectParams params, const FixedState& native, u64 exec_key, u64 inputs_key, const Core::NativeDraw::ExecDraw& record,
                                                     u32 index_count = ~0u);
    /// cp_exec_inputs: plans a native draw from its record alone. Null when the render target inputs or the pipeline of the record's keys are not known yet, or the
    /// fixed-function state needs the registers (the caller plans the old way).
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanExecDraw(const FixedState& native, u64 exec_key, u64 inputs_key,
                                                         std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data,
                                                         const Core::NativeDraw::ExecDraw& draw, u32 index_count = ~0u, u32 draw_class = 15);
    /// The plan of a fast clear eliminate record: the operation and the render target it works on, decoded from the record's register pairs.
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanExecOperation(const Core::NativeDraw::ExecDraw& draw);
    /// A hash of what the planning read from guest memory and put into the plan: pipeline, per stage the program, user data and flat user data (N6b plan verification).
    [[nodiscard]] static u64 PlanDigest(const DrawPlan& plan);
    [[nodiscard]] static u64 PlanDigest(const DispatchPlan& plan);
    [[nodiscard]] std::unique_ptr<DispatchPlan> PlanDispatch(const Core::NativeDispatch::DispatchInputs& cs);

    /// Raster thread: hands a finished plan back for reuse. Allocating a plan for every draw on
    /// one thread and freeing it on another made them fight over the heap lock.
    // ---- Constant ring (documents/tlg/pc-port-draw-pipeline-architecture.md, T1). Env TLG_CONST_RING, one level per piece of the work:
    // 1 the front notes the buffers the queued draws and dispatches write (pending GPU writes), counters only; 2 the ring exists; 3 the front copies the
    // small read-only buffers of a plan into the ring (the raster thread does not use them yet); 4 the raster thread binds them.
    [[nodiscard]] u32 ConstRingLevel() const noexcept {
        return const_ring_level;
    }
    /// Front (command processor thread) only: a plan about to be pushed as the operation at queue position `position`; `done` is the position the raster
    /// thread has completed. Notes what the plan writes and, from level 3, copies its small read-only buffers into the ring.
    void FrontQueueOp(DrawPlan* draw, DispatchPlan* dispatch, u64 position, u64 done, u64 seq = 0, bool indexed = false, u32 index_offset = 0);
    /// Raster thread: the plan the next draw or dispatch runs from (for the ring bindings); Stamp after it was recorded.
    void BeginRingUse(const DrawPlan* draw, const DispatchPlan* dispatch) noexcept {
        ring_use.Begin(draw, dispatch);
    }
    void EndRingUse() noexcept;
    using RingStats = Core::NativePlan::RingStats;
    RingStats TakeRingStats() noexcept;
    /// N7.5, in place buffers (`TLG_INPLACE=1`): read-only buffers up to `TLG_INPLACE_CAP` bytes (default 16 KiB) in guest memory are bound where they are, from an import of the
    /// runtime's always read/write view of guest memory (`HostImport`), with no copy and no dirty tracking.
    struct InPlaceStats {
        u64 offered{}, taken{}, bytes{};
        u64 written{}, special{}, too_large{}, not_in_backing{}, gpu_written{}, image_or_meta{};
        u64 imported_bytes{}, imported_chunks{}, failed_chunks{};
    };
    InPlaceStats TakeInPlaceStats() noexcept;
    /// C0 census (read only, counts per ~5 s interval): how every buffer and image binding, and every vertex / index buffer, of the draws and dispatches was resolved
    /// by the consumer. Index = (compute ? 32 : 0) + kind for the shader bindings, 24 + kind for the geometry (graphics only); see `CensusKind` in vk_rasterizer.cpp.
    std::array<u64, 64> TakeConsumerCensus() noexcept;
    /// R2 (command processor thread): resolves the shader buffer bindings of a planned draw or dispatch whose record names a resident immutable engine buffer
    /// (`TLG_RESOLVE_AHEAD=0` turns it off); the raster thread binds them after a stamp check
    void ResolveAhead(DrawPlan* draw_plan, DispatchPlan* dispatch_plan);
    struct ResolveStats {
        std::atomic<u64> planned{}, hits{}, stale{};
        std::array<std::atomic<u64>, 6> stale_why{};
        std::array<std::atomic<u64>, 9> rejected{}; ///< 0 no record entry, 1 null or huge V#, 2 no engine object, 3 not immutable, 4 outside the object, 5 not synchronized, 6 no arena yet, 7 arena moved, 8 clamped
    };
    std::array<u64, 18> TakeResolveStats();
    std::array<std::atomic<u64>, 4> slot_census{}; ///< R4 census: new buffer slots from the stream ring, -, other buffers; 3 slot hits
    std::atomic<u64> image_words_planned{0}; ///< R3 step 2 counter
    /// R1b verify counters (cumulative): copies checked, copies that differed from guest memory
    [[nodiscard]] u64 UploadsKeptForGpu() const noexcept {
        return buffer_cache.upload_bytes_kept_for_gpu.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool LivePlannerOn() const noexcept {
        return runtime.LivePlannerOn();
    }
    [[nodiscard]] u32 B10Mode() const noexcept {
        return runtime.B10Mode();
    }
    [[nodiscard]] Runtime::LivePlanStats ReadLivePlanStats() const {
        return runtime.ReadLivePlanStats();
    }
    [[nodiscard]] bool InPlaceOn() const noexcept {
        return inplace_on;
    }
    void RecyclePlan(std::unique_ptr<DrawPlan>&& plan);
    void RecyclePlan(std::unique_ptr<DispatchPlan>&& plan);

    /// cp_flush_commands: after a draw or dispatch, hands the commands recorded so far to the
    /// encoder thread once there are enough of them.
    void MaybeFlushRecorded();

    /// A guest memory write done by the command processor itself.
    void NoteCpGuestWrite(VAddr addr, u64 size) {
        buffer_cache.NoteCpGuestWrite(addr, size);
    }

    /// Orders all later GPU work after everything recorded so far, for a command stream wait
    /// that passed because its producer (a fence recorded at `producer_tick`) was already
    /// recorded, not because it finished. Returns whether a barrier was recorded.
    bool CommandStreamBarrier(u64 producer_tick);

    [[nodiscard]] u64 CurrentTick() const noexcept;

    struct TextureSetStats {
        u64 hits{}, no_entry{}, generation{}, image_check{}, not_eligible{}, recorded{};
        u64 by_id_predicted{}, by_id_same{}, by_id_DIFFERENT{}, by_id_unchecked{}, by_id_served{}; ///< F2 item 4 (TLG_TEXSET_BY_ID)
    };
    TextureSetStats TakeTextureSetStats() noexcept {
        return std::exchange(texture_set_stats, TextureSetStats{});
    }
    [[nodiscard]] bool TextureSetMemoOn() const noexcept {
        return texture_set_on;
    }

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds, bool staging_readback = false);
    void SetDispatchCaller(u64 caller) {
        dispatch_caller = caller;
    }
    /// A memory fill (a pattern of up to four dwords repeated over `bytes`): when an image of the fill's size lives at the
    /// address the fill is an image clear. SetPendingFill covers the next dispatch.
    bool TryClearImage(VAddr dst, u32 bytes, const u32* pattern, u32 dwords);
    void SetPendingFill(VAddr dst, u32 bytes, const u32* pattern, u32 dwords) {
        pending_fill = {dst, bytes, dwords, {pattern[0], dwords > 1 ? pattern[1] : 0, dwords > 2 ? pattern[2] : 0, dwords > 3 ? pattern[3] : 0}};
    }
    /// The memory copy of two images that live on the GPU with the same layout is an image copy.
    bool TryCopyImages(VAddr dst, VAddr src, u32 num_bytes);
    /// Clears the depth / stencil target the registers describe (native depth clear record).
    void ClearDepthStencilTarget(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);

    /// Native draw path: the tag the game wrote for the draw about to be executed (raster thread only).
    void SetNativeTag(const VideoCore::NativeTag& tag) {
        native_tag = tag;
    }
    /// Anything other than a tagged draw ran: the reuse state is stale.
    void NativeInvalidate() {
        native_last.valid = false;
    }
    struct NativeStats {
        u64 tagged{};
        u64 untagged{};
        u64 dynamic_skipped{};
        u64 bind_skipped{};
    };
    NativeStats TakeNativeStats() {
        return std::exchange(native_stats, {});
    }
    bool IsMapped(VAddr addr, u64 size);
    /// N5h: see BufferCache::SupersedeGpuWritten.
    /// B8.2: the engine deleted a texture or view (raster thread, in order with the GPU work)
    void EngineObjectDeleted(u64 id, VAddr address, u64 bytes, u32 what);
    /// B11: the engine clear whose packets the raster thread is running now (null: none); the record is copied
    void SetEngineClear(const Core::NativeDevice::ClearRecord* record);
    /// Real occlusion queries (native/native_occlusion.h): the game begins / ends the counting for the result block at `address`
    void OcclusionOpen(VAddr address, u32 pairs);
    void OcclusionClose(VAddr address);
    /// B11 step 3: the clear as a Vulkan clear of the target's own images; false when it cannot (the old path runs)
    bool NativeClear(const Core::NativeDevice::ClearRecord& record);
    /// B11 step 3: true when the work at hand belongs to an engine clear that ran natively (it is skipped)
    bool SkipForEngineClear() {
        if (engine_clear_on && engine_clear_skip) {
            Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeSkippedWork);
            return true;
        }
        return false;
    }
    bool engine_clear_on{};
    bool engine_clear_skip{};
    Core::NativeDevice::ClearRecord engine_clear{};
    void SupersedeGpuWritten(VAddr addr, u64 size) {
        buffer_cache.SupersedeGpuWritten(addr, size);
    }
    /// N5c (raster thread): uploads the ranges the engine announced (outside a render pass: it is called in front of a draw or dispatch).
    void PreSynchronize(const std::vector<Core::NativeUploads::Range>& ranges);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish() override;

    /// Raster thread: records the following operations into the asynchronous stream (second Vulkan queue, TLG_ASC_QUEUE=1) or the graphics stream.
    /// Leaving the asynchronous stream submits what it recorded. No effect without the second queue; the first windows stay on the graphics stream
    /// (the inputs shared with the graphics are uploaded there first).
    void SelectStream(bool side);
    /// The asynchronous stream has started (the warm-up is over): queue 1 operations record into command buffers of their own.
    [[nodiscard]] bool SideStreamReady() const noexcept {
        return side_ready.load(std::memory_order_acquire);
    }
    void OnSubmit();

    void ProcessDownloadImages();

    /// Handles readbacks at an end-of-pipe fence and runs `signal` once their data is in guest
    /// memory. With async fences the GPU thread doesn't wait and `signal` runs on another thread;
    /// `must_sync` forces the blocking path (for fences whose signal needs the GPU thread).
    void OnFence(Common::UniqueFunction<void>&& signal, VideoCore::FenceKind kind,
                 bool must_sync = false);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline, const FixedState& fixed);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    /// The render state of a draw that starts or changes a render pass (the work BeginRendering does when the memo below cannot answer).
    RenderState BeginRenderingFull(const GraphicsPipeline* pipeline);
    /// Design review 83: the own memory answer of the binding being compared with the old path (shadow or audit), null when none.
    const VideoCore::Buffer* own_compare{};
    u64 own_compare_offset{}, own_compare_size{};
    u32 own_compare_slot{};
    u32 own_compare_cls{};

    u32 const_ring_level = 0;
    std::unique_ptr<ConstantRing> constant_ring;
    std::unique_ptr<VideoCore::Buffer> hook_ring_buffer; ///< F1 1a (TLG_HOOK_RING=2): the frame ring copy filled at the writes, two lap halves, in GPU-visible memory
    // N7.5 (raster thread)
    bool inplace_on = false;
    u32 inplace_cap = 16384;
    std::unique_ptr<HostImport> host_import;
    std::atomic<u64> inplace_offered{0}, inplace_taken{0}, inplace_bytes{0}, inplace_written{0}, inplace_special{0}, inplace_too_large{0}, inplace_not_in_backing{0}, inplace_gpu_written{0},
        inplace_image_or_meta{0};
    std::array<std::atomic<u64>, 64> consumer_census{};
    void Census(u32 index, u64 n = 1) noexcept {
        consumer_census[index & 63].fetch_add(n, std::memory_order_relaxed);
        if (cover_on) [[unlikely]] {
            NoteCover(index);
        }
    }
    /// A2 census (`TLG_COVER_CENSUS=1`, default off): per draw and dispatch, which kinds of binding it met that the command processor cannot know ahead, and the time of its BindResources.
    /// Mask bits: 1 buffer of an engine handle not resolved ahead, 2 buffer by guest address, 4 image by lookup, 8 image found, 16 storage image, 32 image transit with a barrier, 64 vertex / index by address.
    bool cover_on = [] { const char* e = std::getenv("TLG_COVER_CENSUS"); return e != nullptr && e[0] == '1'; }();
    std::array<std::atomic<u64>, 256> cover_count{}, cover_ns{}, cover_sync_ns{}; ///< [side * 128 + mask]
    /// R1': per buffer binding of a shader stage, [side * 128 + cause * 8 + kind]: bindings, ns until the binding is chosen, ns of it in SynchronizeMemory, calls that copied.
    /// cause: 0 no resolved entry, 1 front found no engine object, 2 epoch moved, 3.. not hit (4 gone, 5 not synchronized, 6 generation moved, 7 arena or epoch, 8 tracker dirty), 9 hit (object copy), 10 hit (arena).
    /// kind: 1 ring, 2 special, 3 null, 4 handle, 5 address read, 6 address written, 7 resolved.
    std::array<std::atomic<u64>, 256> cover_buf_n{}, cover_buf_ns{}, cover_buf_sync{}, cover_buf_copies{};
    /// TLG_COVER_CENSUS: a binding by engine handle (kind 4) split in six segments, [side][0..5] ns and [side][6] count (design review 53)
    std::array<std::array<std::atomic<u64>, 8>, 2> cover_handle_seg{};
    std::array<std::array<std::atomic<u64>, 8>, 2> cover_handle_seg_served{}; ///< the same for the bindings a row served
    /// Rows ahead census (TLG_ROW_AHEAD_CENSUS=1), [graphics, compute]: 0 bindings with an engine id, 1 the front saw none, 2 KEY_DIFFERS, 3 key same and a row served with the version
    /// the front saw, 4 key same, row served, version moved since the plan, 5 key same, no row served, 6 of the key-same: written, 7 of the key-same: texel
    std::array<std::array<std::atomic<u64>, 8>, 2> row_ahead_census{};
    /// Rows ahead shadow (TLG_ROWS_AHEAD=1), [graphics, compute]: see the report names in vk_rasterizer_host.cpp
    std::array<std::array<std::atomic<u64>, 12>, 2> rows_ahead_shadow{};
    /// Rows ahead real (TLG_ROWS_AHEAD=2), [graphics, compute]: 0 bound from the planned row, 1 sampled (full path), 2 SAMPLED_DIFFERENT_MUST_BE_0, 3 sampled same,
    /// 4 refused by the identity walk, 5 planned row stale (epoch or version), 6 static row (full path), 7 no planned row
    std::array<std::array<std::atomic<u64>, 8>, 2> rows_ahead_real{};
    /// design review 66 (TLG_COVER_CENSUS): the identity walks of the rows ahead fast path: [graphics, compute][calls, events stepped, ns]
    std::array<std::array<std::atomic<u64>, 3>, 2> rows_ahead_walk{};
    std::array<std::array<std::atomic<u64>, 8>, 2> rows_ahead_fseg{}; ///< design review 66: [side][6 segments ns, count, of them indexed]
    void NoteFastSegments(u32 side, bool indexed) noexcept;
    /// Design review 67 to 69 (`TLG_BARRIER_GUARD`: 2 real (default since 2026-11-10, owner's play), 1 shadow, 0 off): the barrier query of a read-only buffer binding. A read can only need a barrier against a write the
    /// planner holds since its last barrier; with none (`PlanHasWrites` false) the query is not asked (real) or asked and compared (shadow). The read itself is still recorded at the
    /// finish (`AccessBuffer` over the bound list). One in 64 skipped queries is asked anyway.
    [[nodiscard]] bool BufferBarrierQuery(const VideoCore::Buffer* buffer, u64 offset, u64 size, bool written);
    /// Design review 76: the plan's geometry handle holding [address, address + size) (alive, the plan's epoch current), without the compare mode's lookup by address.
    [[nodiscard]] const Core::NativeResources::ImmutableBuffer* GeometryHandleOfPlan(VAddr address, u64 size) const noexcept;
    /// Note 117 (`TLG_OWN_GEOMETRY`): the own memory of the engine buffer the draw's geometry ids name that holds [address, address + size), when its bytes are current
    /// (immutable: synchronized; default and dynamic: no CPU bytes waiting), else null. `cls` 7 index, 8 vertex.
    [[nodiscard]] const VideoCore::Buffer* OwnGeometry(VAddr address, u64 size, u32 cls);
    /// Note 117 shadow and audit: the old path's index or vertex answer against the own memory's
    void OwnGeometryCompare(const VideoCore::Buffer* own, VAddr address, vk::Buffer bound, u64 offset, u32 cls);
    const VideoCore::Buffer* own_index{}; ///< note 117: the index read's own memory, compared after the old path bound it
    /// [0 read queries, 1 of them with no planned write, 2 SHADOW_NO_WRITES_BUT_BARRIER_MUST_BE_0, 3 real: asked (writes pending), 4 real: skipped, 5 real: sampled, 6 SAMPLED_BARRIER_MUST_BE_0]
    std::array<std::atomic<u64>, 8> barrier_guard{};
    /// Texture census (TLG_TEXTURE_CENSUS=1, design review 58), [graphics, compute][class][calls, ns, images]: classes 0 no image, 1 more than 16 images, 2 memo off, 3 a written image,
    /// 4 a mip fallback image, 5 set hit, 6 miss no entry, 7 miss generation, 8 miss image check, 9 other
    std::array<std::array<std::array<std::atomic<u64>, 3>, 10>, 2> tex_census{};
    /// R1'': handle bindings the front left without an entry (cause 0), by why ResolveAhead skips them: [side * 16 + (1 written, 2 formatted, 4 program-base offset, 8 not a single load)]
    std::array<std::atomic<u64>, 32> cover_skip_n{}, cover_skip_ns{}, cover_skip_sync{};
    /// R1''': formatted (texel) engine-handle bindings: [side * 4 + (0 checked, 1 an image overlaps the range, 2 the range is image metadata, 3 GPU wrote the range)]
    std::array<std::atomic<u64>, 8> cover_texel{};
    /// per draw class (0 z prepass / shadow, 1 g-buffer, 2 forward, 15 untagged): draws, ns in BindResources, ns in the whole Draw() call
    std::array<std::atomic<u64>, 16> cover_class_draws{}, cover_class_bind_ns{}, cover_class_total_ns{};
    /// W1.0 census (`TLG_W1_CENSUS=1` together with `TLG_COVER_CENSUS=1`; rewrite plan section 4): z prepass / shadow draws grouped into passes (the depth target changes or the class changes).
    /// Cumulative counters, published as `consumer_census.w1_census`.
    std::array<std::atomic<u64>, 64> w1_counts{};
    /// W1.1 shadow (`TLG_W1_CENSUS=1` with `TLG_COVER_CENSUS=1`; rewrite plan section 4b): per shadow draw, whether every buffer it bound was a lookup (ring window, resident arena
    /// buffer) or needed work (image, written buffer, address lookup, stream copy, upload), and whether an id-keyed row (engine id -> buffer, base, placement epoch) predicts the
    /// answer of the old path. Published as `consumer_census.w1_shadow`.
    std::array<std::atomic<u64>, 192> w1s_counts{};
    void W1ShadowRow(u64 id, VAddr address, const void* buffer, u64 offset, u32 base) noexcept;
    void W1ShadowBinding(const Core::NativeResources::ImmutableBuffer* hint, u64 record_id, VAddr address, const void* buffer, u64 offset, bool written, u32 why, bool formatted) noexcept;
    void W1ShadowGeometry(const Core::NativeResources::ImmutableBuffer* hint, VAddr address, const void* buffer, u64 offset, bool vertex) noexcept;
    /// W3.0 dispatch census (`TLG_COVER_CENSUS=1`): per compute program the sums in `W3Stat::v` (published as `consumer_census.w3_dispatch`)
    [[nodiscard]] std::vector<std::pair<u64, std::array<u64, 48>>> W3Rows() const;
    [[nodiscard]] bool RowsServe(u64 id, VAddr address, u64 size, bool texel, const VideoCore::Buffer*& buffer, u64& offset, bool written = false) noexcept;
    void RowsLearn(u64 id, VAddr address, u64 size, bool texel, const VideoCore::Buffer* buffer, u64 offset, u64 epoch_pre, u64 clock_pre, bool gpu_pre, u64 version_pre = 0,
                   u64 unattr_pre = 0, bool written = false) noexcept;
    void W31Before(u64 id, VAddr address, u64 size) noexcept;
    void W31After(const void* buffer, u64 offset, VAddr address, bool compute, bool written) noexcept;
    void W1NoteBinding(const Pipeline* pipeline) noexcept;
    void W1EndDraw() noexcept;
    void NoteCover(u32 index) noexcept;
    void CoverFinish(bool compute) noexcept;
    // Front thread only:
    struct PendingWrite {
        VAddr begin, end;
        u64 position;
    };
    std::vector<PendingWrite> pending_gpu_writes;
    VAddr pending_min = ~VAddr{0}, pending_max = 0;
    u32 pending_checks = 0;
    void NotePendingGpuWrite(VAddr address, u64 size, u64 position);
    bool PendingGpuWriteOverlaps(VAddr address, u64 size, u64 done);
    /// The buffers of one stage: notes the written ones (`position`), and from level 3 copies the small read-only ones into the ring.
    /// `owner`: the draw plan the stage belongs to (null for a dispatch); its record bindings give the V#s of a stage planned without flat user data (N3d).
    void FrontStageNotes(StagePlan& stage, u64 position, const DrawPlan* owner);
    void FrontStageRing(StagePlan& stage, u64 done, const DrawPlan* owner, u64 seq);
    /// N5g: the vertex ranges and the index range of a draw into the ring.
    void FrontGeometryRing(DrawPlan& plan, u64 done, bool indexed, u32 index_offset);
    [[nodiscard]] const RingBinding* FindGeometryRing(VAddr address, u64 size, u32 kind) const noexcept;
    // Raster thread:
    Core::NativePlan::RingUse ring_use;
    const RingBinding* FindRingBinding(const Shader::Info& stage, u32 index) const noexcept;

    /// Render state memo (the port's `RenderStateMemo`, tlg_pc docs/parallel_gpu.md): about 90 % of the draws continue the render pass the draw before
    /// them opened. Such a draw gets that draw's render state back instead of redoing the target lookups, the layout transitions and the attachment
    /// descriptions. It only applies while the scheduler still has exactly that pass open (a barrier, copy or dispatch ends the pass), the signature of
    /// the inputs is the same (target ids and views, the pipeline's attachment key, the image registry generation, depth control), nothing clears,
    /// and no target is also sampled by the draw.
    struct BeginSignature {
        std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS + 1> ids{};
        std::array<VideoCore::ImageViewInfo, AmdGpu::NUM_COLOR_BUFFERS + 1> views{};
        u32 mrt_mask = 0;
        u32 num_samples = 0;
        std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples{};
        u64 generation = 0;
        u32 depth_control = 0;
        bool depth_valid = false;
        bool stencil_valid = false;
        bool operator==(const BeginSignature&) const = default;
    };
    BeginSignature MakeBeginSignature(const GraphicsPipeline* pipeline) const;
    struct BeginMemo {
        bool valid = false;
        BeginSignature signature;
        RenderState state;
    } begin_memo;
    void Resolve(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0, const AmdGpu::CbDbExtent& cb1);
    void DepthStencilCopy(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent, bool is_depth, bool is_stencil);
    void EliminateFastClear(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0);
    /// P3e.3: the two halves of the fast clear eliminate. The resolve half looks the target up and decides whether a clear is pending; only then is there a command to record.
    struct FastClearOp {
        VideoCore::Image* image{nullptr};
        VideoCore::SubresourceRange range{};
        vk::ClearValue value{};
        u64 address{0};
    };
    bool ResolveFastClearEliminate(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0, FastClearOp& op);
    void RecordFastClearEliminate(const FastClearOp& op);
    /// Clears the bound depth / stencil target to the clear values of the registers (native replacement of the SDK's clear quad).
    void ClearDepthTarget(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent);

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed,
                            const FixedState& fixed) const;
    void UpdateViewportScissorState(const FixedState& fixed) const;
    void UpdateDepthStencilState(const FixedState& fixed) const;
    void UpdatePrimitiveState(bool is_indexed, const FixedState& fixed) const;
    void UpdateRasterizationState(const FixedState& fixed) const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline, const FixedState& fixed) const;

    bool FilterDraw(const DrawPlan* plan = nullptr);
    const GraphicsPipeline* ResolveDrawPipeline(const DrawPlan* plan,
                                                DrawIndirectParams params = {});
    const ComputePipeline* ResolveDispatchPipeline(const DispatchPlan* plan);

    enum class DrawClass : u8 {
        Normal,
        EliminateFastClear,
        FmaskDecompress,
        Resolve,
        PrimitiveNone,
        DepthStencilCopy,
        SkipShadow, ///< Live switch skip_shadows: a depth-only draw into a shadow map sized target.
        DepthTransition, ///< The SDK's full-screen triangle that decompresses / resummarizes depth: no effect without compression.
        DepthClear, ///< The SDK's full-screen triangle with the depth / stencil clear bits set: a clear of the target.
    };
    struct DrawClassification {
        DrawClass kind{};
        bool depth_copy{};
        bool stencil_copy{};
    };
    /// What is left for a draw that is not such an operation, from the plan only (the shadow skip switch).
    static DrawClassification ClassifyRemaining(const DrawInputs& inputs, const FixedState& fixed);

    void ProcessReadbacksSync();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindBuffersBody(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                         Shader::PushData& push_data);
    void BindBuffersBodyLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding, Shader::PushData& push_data);
    /// Plan S step P1: what the resolve half decided for one buffer binding (the record half turns it into descriptor writes).
    struct BufferBind {
        enum Kind : u8 { Info, Skip, FlatCopy, ClipCopy, LdsMap };
        Kind kind{Info};
        bool has_offset{false};
        bool has_bound{false};
        bool has_query{false}, query_plan_writes{false}, query_written{false}, census_query{false};
        bool has_slot{false};
        u32 slot{0};
        const VideoCore::Buffer* query_buffer{nullptr};
        u64 query_offset{0}, query_size{0};
        u32 adjust{0};
        u32 lds_size{0};
        vk::DescriptorBufferInfo info{};
        const VideoCore::Buffer* bound_buffer{nullptr};
        u64 bound_offset{0};
        u32 bound_size{0};
        vk::AccessFlags2 bound_access{};
        std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
        /// P3e.3: the stage's flattened user data, copied by the resolve half (FlatCopy / Skip of an indexed stage): the record half must not read the shader info, the next draw's plan rewrites it
        boost::container::small_vector<u32, 32> flat;
        /// a barrier query for the record pass: does the buffer range conflict with what the plan holds (with `plan_writes`: only asked when the plan has writes)
        void SetQuery(const VideoCore::Buffer* buffer, u64 offset, u64 size, bool written, bool plan_writes) {
            has_query = true;
            if (buffer != nullptr) {
                const_cast<VideoCore::Buffer*>(buffer)->NoteResolveUse();
            }
            query_buffer = buffer;
            query_offset = offset;
            query_size = size;
            query_written = written;
            query_plan_writes = plan_writes;
        }
        void SetBound(const VideoCore::Buffer* buffer, u64 offset, u64 size, vk::AccessFlags2 access) {
            has_bound = true;
            if (buffer != nullptr) {
                const_cast<VideoCore::Buffer*>(buffer)->NoteResolveUse(); // P3c: the resolve side announces every buffer the operation names, whichever way it found it
            }
            bound_buffer = buffer;
            bound_offset = offset;
            bound_size = static_cast<u32>(size);
            bound_access = access;
        }
    };
    void ResolveBufferBinding(const Shader::Info& stage, u32 buffer_index, u64 alignment, u32 census_base, BufferBind& out);
    void ResolveBufferBindingBody(const Shader::Info& stage, u32 buffer_index, u64 alignment, u32 census_base, BufferBind& out);
    u32 BufferSlotCounted(const vk::DescriptorBufferInfo& info);
    std::vector<u32> verify_slots;
    bool verify_slots_on{false};
    void RecordBufferBindings(const Shader::Info& stage, Shader::Backend::Bindings& binding, Shader::PushData& push_data, BufferBind* binds, u32 count, u64 alignment, u32 census_base);
    static u32 SplitBindMode();
    /// Plan S step P2: what the resolve half decided for one image binding.
    struct TextureBind {
        VideoCore::ImageId image_id{};
        VideoCore::Image* image{nullptr};
        vk::ImageView view{};
        VideoCore::SubresourceRange range{};
        bool null_entry{false}, is_storage{false}, is_depth{false}, is_target{false}, general{false}, feedback{false};
    };
    struct TextureResolved {
        boost::container::small_vector<TextureBind, 16> binds;
        boost::container::small_vector<u32, 8> array_sizes;
        std::array<u32, Shader::NUM_IMAGES> image_keys{};
        bool set_hit{false};
    };
    /// Plan S step P3e.1: everything the resolve half decided for one stage of a draw or dispatch.
    struct StageResolved {
        const Shader::Info* stage{nullptr};
        boost::container::small_vector<BufferBind, 16> buffers;
        TextureResolved textures;
        boost::container::small_vector<vk::Sampler, 8> samplers;
    };
    void BindTexturesLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void ResolveTextures(const Shader::Info& stage, TextureResolved& r);
    void RecordTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding, TextureResolved& r);
    void BindSamplersLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void ResolveSamplers(const Shader::Info& stage, boost::container::small_vector<vk::Sampler, 8>& out);
    void RecordSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding, const vk::Sampler* samplers, u32 count);
    struct VertexBind;
    struct IndexBind;
    void BindVertexBuffersLegacy(const GraphicsPipeline* pipeline);
    void ResolveVertexBuffers(const GraphicsPipeline* pipeline, VertexBind& vb);
    void RecordVertexBuffers(VertexBind& vb, VideoCore::BufferCache::UploadSink& uploads);
    void BindIndexBufferLegacy(u32 index_offset, bool is_indirect);
    void ResolveIndexBuffer(u32 index_offset, bool is_indirect, IndexBind& ib);
    void RecordIndexBuffer(IndexBind& ib, VideoCore::BufferCache::UploadSink& uploads);
    struct VerifyVertexEntry {
        vk::Buffer buffer;
        u64 offset;
        bool stream;
    };
    std::vector<vk::Sampler> verify_samplers;
    std::vector<VerifyVertexEntry> verify_vertex;
    struct VerifyIndex {
        vk::Buffer buffer{};
        u64 offset{};
        u32 size{};
        bool stream{};
    } verify_index;
    void VerifyBufferBindings(const Shader::Info& stage, Shader::Backend::Bindings& binding, Shader::PushData& push_data, u64 alignment, u32 census_base, u32 count);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding);

    /// Texture set memo (documents/tlg/pc-port-draw-pipeline-architecture.md, T3; the port's `TextureSetMemo`): the images a stage's T#s resolve to, remembered by
    /// the raw T# and the use flags of each resource. A hit redoes only the per draw effects (touch, update, bound flags, layout transition, descriptor info).
    /// Raster thread only. Env TLG_TEXTURE_SET_MEMO=1 turns it on.
    struct TextureSetEntry {
        std::array<u32, 8> raw{}; ///< the eight dwords of the T#
        u64 flags{};
        VideoCore::ImageId found{}; ///< what FindImage returned
        VideoCore::ImageId id{};    ///< after the depth redirect (null entry: an empty descriptor)
        vk::ImageView view{};
        const void* backing{};
        VideoCore::SubresourceRange range{};
        bool null_entry{};
        bool storage{}; ///< note 111: a written (storage) binding of the image
        u32 array_key{}; ///< tlg C9: the image's descriptor array (Shader::Indexed::ImageKey of its T#)
        u64 found_uid{}, id_uid{}; ///< F2 item 4: the images' uids (a slot can be reused by another image)
    };
    struct TextureSet {
        static constexpr u32 MaxImages = 32; ///< the entries' capacity; the stages the memo takes are limited by `Limit()`
        /// Texture census (design review 58 to 60): `TLG_TEXSET_MAX` (default 32 since 2026-11-10 after the owner's play; at most 32; =16 the old limit): stages with more images take
        /// the slow path every time
        [[nodiscard]] static u32 Limit() noexcept {
            static const u32 limit = [] {
                const char* e = std::getenv("TLG_TEXSET_MAX");
                const u32 v = e != nullptr ? static_cast<u32>(std::atoi(e)) : MaxImages;
                return v == 0 ? MaxImages : v > MaxImages ? MaxImages : v;
            }();
            return limit;
        }
        u64 key{};
        u64 generation{~0ull};
        u32 count{};
        std::array<TextureSetEntry, MaxImages> entries{};
    };
    static constexpr u32 TextureSetSlots = 4096;
    std::unique_ptr<TextureSet[]> texture_sets;
    bool texture_set_on = false;
    TextureSetStats texture_set_stats;
    /// True when the stage's images were bound from the set (the descriptor writes are done).
    bool BindTexturesFromSet(const Shader::Info& stage, TextureSet& set, const std::array<u32, 8>* raws, const u64* flags, u32 count,
                             Shader::Backend::Bindings& binding);
    /// The set hit's tail (touch and refresh, the per-image checks, transitions, descriptor writes): false when a check fails (nothing bound; touch and refresh are idempotent).
    bool BindTexturesFromEntries(const Shader::Info& stage, const TextureSet& set, u32 count, Shader::Backend::Bindings& binding);
    /// Design review notes 95 and 96 (`TLG_VIEW_TABLE`): after a stage's images were bound by the old path, the view table's answer for each read-only binding of a non-dynamic
    /// engine texture view is compared with it, and the entry is made or remade from it (`first_info`: the stage's first image info; `bound_before`: the bound image list's size
    /// before the stage; `answers`/`answers_ok`: the old path's per-image record and whether it can be bound from the table).
    void ViewTableShadow(const Shader::Info& stage, u32 first_info, size_t bound_before, const TextureSetEntry* answers, const bool* answers_ok);
    /// Note 103 (`TLG_TARGET_TABLE=1`, shadow): target `slot` (0..7 color, 8 depth) of the draw, found by the old path as `image_id` (`old_find`: by FindImage), against the
    /// table's entry for the view the record names; `use` keeps apart what changes the target's description. Returns the entry's key (0: not in the class) for the render pass.
    u64 TargetTableShadow(u32 slot, u64 use, VideoCore::ImageId image_id, const VideoCore::TextureCache::ImageDesc& desc, bool& in_class, bool& served);
    /// Note 107 (`TLG_WRITTEN_NO_IMAGE`): what a written buffer binding of engine buffer `id` (0: none) over [address, address + size) tells the texture cache; `inside`: the
    /// caller knows the range lies in the buffer.
    void InvalidateWrittenBuffer(u64 id, VAddr address, u64 size, bool inside);
    /// Note 104 (`TLG_TARGET_TABLE=2`): target `slot`'s image and description from the table (true), or nothing done (false: the old path finds it).
    bool TargetTableTake(u32 slot, u64 use, VideoCore::ImageId& image_id, VideoCore::TextureCache::ImageDesc& desc);
    /// The use a target is keyed with besides its view: the slot and the bytes its description takes besides the view.
    static u64 TargetUse(u64 slot, const void* bytes, size_t size) noexcept;
    std::unordered_map<u64, VideoCore::TextureCache::ImageDesc> target_descs; ///< note 104: the old path's description of each target table entry
    /// ... at the render pass: the view the old path chose for target `slot`.
    void TargetViewShadow(u32 slot, vk::ImageView view);
    std::array<u64, 9> target_use{}; ///< note 103: the use the draw's targets were looked up with (0: not in the class)
    /// Note 97 (`TLG_VIEW_TABLE=2`): the stage's images bound from the view table (true), or nothing done (false: the old path binds them).
    bool ViewTableBind(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool ResolveTextureSet(const Shader::Info& stage, TextureSet& set, const std::array<u32, 8>* raws, const u64* flags, u32 count, TextureResolved& r);
    /// F2 item 4 (`TLG_TEXSET_BY_ID`, default 2 since 2026-11-10; 0 off, 1 shadow): a texture set whose generation moved is still right when none of its images changed by identity (the texture
    /// cache's event ring: no image registered or unregistered over their memory since the set was recorded). Returns true when the set may be used (real mode, all unchanged).
    bool TextureSetGenerationMiss(const TextureSet& set);
    /// Shadow: after a set was recorded again, its new entries against the ones the by-id check called still right.
    void TextureSetShadowCompare(const TextureSet& set);
    struct TexsetShadowState {
        TextureSet* slot{};
        std::array<TextureSetEntry, TextureSet::MaxImages> entries{};
        u32 count{};
    };
    static TexsetShadowState& TexsetShadow();
    /// Plan S step P3e.2: the plain data a graphics draw carries from its resolve half to its record half (render pass, vertex buffers, index buffer).
    struct DrawJob;
    static bool RenderStateMemoEnabled();
    void ResolveDrawJob(const GraphicsPipeline* pipeline, DrawJob& job);
    RenderState RecordDrawJob(DrawJob& job);
    /// P3e.4: the record half of a whole graphics draw: the stage records, the render pass, vertex and index buffers, the barrier flush, the pipeline's descriptors, the dynamic
    /// state, the draw call and the access notes. Inputs only from `job`; the members it writes are the ones listed under "recorder-owned" below.
    void RecordDraw(DrawJob& job);
    static std::vector<std::unique_ptr<DrawJob>>& DrawJobPool();
    std::unique_ptr<DrawJob> TakeDrawJob();
    void RunDrawJob(std::unique_ptr<DrawJob>&& job);
    void MaybeFlushRecordedBody();
    /// The measuring and the indexed descriptor table that follow the stage records in `BindResources`.
    void FinishBindResources(const Pipeline* pipeline, bool uses_dma, u64 resolve_ns);
    bool BindResources(const Pipeline* pipeline, const FixedState* fixed = nullptr, DrawJob* draw_job = nullptr);
    void BindCensus(const Pipeline* pipeline);
    static void NoteRecordedStateDiagnostics();

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);
    void ResetImageBindings();
    /// Plan S step P2c: what the resolve half decided for the render pass (the state minus the layouts, and the images to transit).
    struct RenderResolved {
        struct Target {
            VideoCore::Image* image{nullptr};
            VideoCore::SubresourceRange range{};
            bool feedback{false};
            vk::ImageLayout new_layout{};
        };
        RenderState state;
        bool feedback_loop{false}; ///< P3e.4b: what `attachment_feedback_loop` was when the resolve half finished (the member is rewritten by the next draw's resolve)
        std::array<Target, AmdGpu::NUM_COLOR_BUFFERS> color{};
        std::array<bool, AmdGpu::NUM_COLOR_BUFFERS> color_used{};
        Target depth{};
    };
    RenderState BeginRenderingLegacy(const GraphicsPipeline* pipeline);
    void ResolveRenderTargets(const GraphicsPipeline* pipeline, RenderResolved& r);
    RenderState RecordRenderTargets(RenderResolved& r);
    void RecordBufferAccess(bool is_compute, u64 op_sequence = 0);

    /// P3e.3: the game's own compute copies and clears of images. The resolve half decides (`Match*`: which images, which values), the record half issues the command (`RunComputeImageOp`).
    struct ComputeImageOp {
        enum Kind : u8 { None, Copy, Clear };
        Kind kind{None};
        VideoCore::Image* source{nullptr};
        VideoCore::Image* target{nullptr};
        VideoCore::SubresourceRange range{};
        vk::ClearValue clear{};
    };
    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool MatchComputeImageCopy(const Pipeline* pipeline, ComputeImageOp& op);
    bool MatchComputeImageClear(const Pipeline* pipeline, ComputeImageOp& op);
    void RunComputeImageOp(const ComputeImageOp& op);

private:
    friend class VideoCore::BufferCache;

    VideoCore::NativeTag native_tag{};
    struct NativeLast {
        bool valid{};
        u64 k2{};
        const GraphicsPipeline* pipeline{};
        u64 tick{};
        u64 seq{}; ///< P3e.4: the call number of that draw (`draw_call_seq`)
    } native_last{};
    u64 draw_call_seq{}; ///< P3e.4: counts every call of `Draw`; only the very next call may reuse `native_last`
    NativeStats native_stats{};

    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<Core::NativeOcclusion::Queries> occlusion;
    std::unique_ptr<Scheduler::SideScope> side_scope;
    u32 side_windows{};
    bool side_requested{};
    std::atomic<bool> side_ready{false};
    u64 side_epoch{1};
    u64 side_last_end_ns{};
    Runtime& runtime;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    Core::NativeExecutor* executor;
    const std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS>* current_cb_extent{}; ///< The extent hints of the targets of the draw being recorded.
    const AmdGpu::CbDbExtent* current_db_extent{};
    static constexpr std::array<u64, 18> kNoTargetResources{};
    const std::array<u64, 18>* current_target_resources{&kNoTargetResources}; ///< E3: engine resource ids behind the draw's color views (0..7) and depth view (8).
    const AmdGpu::BorderColorBuffer* current_bc_base{}; ///< The border colour table of the draw or dispatch being recorded.
    const Core::NativeDispatch::DispatchInputs* current_cs{}; ///< The compute program of the dispatch being recorded (its plan's; an empty one without a plan).
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    const bool readback_batching;
    const bool async_fences;
    u64 recorded_work{};
    u64 last_barrier_work{~0ULL};

    /// rt_lookup_reuse_enabled: the register inputs of the last target lookup of each color slot
    /// (and the depth target, last entry) and the texture cache generation it was made at.
    struct CachedTarget {
        bool valid{};
        u64 generation{};
        u32 size{};
        std::array<u8, 256> inputs{};
    };
    std::array<CachedTarget, AmdGpu::NUM_COLOR_BUFFERS + 1> rt_cache{};
    /// Sampled images (BindTextures): the result of FindImage for a T# and the way the shader uses it, valid while
    /// no image was registered or removed (native_draw_mode >= 2).
    /// The lookup key of a sampled image: the plain description, the shader's use flags and the binding index (with the engine object the record names)
    struct TextureKey {
        Core::NativeDesc::ImageDesc image{};
        u64 flags{};
        u64 index{};
        bool operator==(const TextureKey&) const = default;
    };
    struct CachedTextureLookup {
        bool valid{};
        u64 generation{};
        VideoCore::ImageId image_id{};
        u64 image_uid{}; ///< L1(b): the unique id of that image (the slot index is reused), so the answer can be proven again after the generation moved
        TextureKey key{};
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<CachedTextureLookup, 8192> texture_lookup_cache{}; ///< L1(b): 8192 entries (was 512: 37k collision misses in 90 s)
    /// B3: the image of each engine object (by id and use), raster thread
    struct NativeImage {
        bool valid{};
        u64 id{};
        u64 flags{};
        u32 index{};
        Core::NativeDesc::ImageDesc image_key{}; ///< the plain description the image was found for (the guard that the object still describes the same image)
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::unordered_map<u64, NativeImage> native_images;
    std::unordered_map<u64, std::vector<u64>> native_image_keys; ///< B8.2: the native_images keys of each view id (dropped at the engine's deletion)
public:
    std::array<std::atomic<u64>, 5> b6_downloads{}; ///< B6: copy results downloaded into PS4 memory / left on the GPU (no CPU reader)
    std::atomic<u64> b5_copies_on_gpu{0}, b5_copies_on_cpu{0};
    std::atomic<u64> b9_dma_binds{0}, b9_all_binds{0}; ///< B9.4 census: draws / dispatches whose shaders read memory by a raw address, of all ///< B5: guest copies run on the GPU / as a CPU copy in PS4 memory
    std::array<std::atomic<u64>, 3> b3_stats{}; ///< B3 image bindings: 0 by id, 1 first sight, 2 the object's image changed
private:
    const bool rt_reuse_enabled;
    bool RtReuseOn() const {
        const s32 override_mode = VideoCore::ReadbackPolicy::Get().rt_reuse_override.load(std::memory_order_relaxed);
        return override_mode >= 0 ? override_mode != 0 : rt_reuse_enabled;
    }
    const u32 encoder_debug;
    /// Encoder thread: fences without readbacks wait for all work recorded before them.
    const bool full_eop_fences;
    /// Hand recorded commands to the encoder thread every this many commands (0 = off).
    const u32 flush_commands;
    tsl::robin_map<u64, DrawInputs> exec_inputs; ///< the render target inputs of the draws planned the old way, by the identity of what they come from
    Core::NativePlan::PlanPool plan_pool;
    struct PendingFill {
        VAddr dst{};
        u32 bytes{};
        u32 dwords{};
        std::array<u32, 4> pattern{};
    };
    PendingFill pending_fill{};
    const DrawInputs* draw_inputs{}; ///< of the draw being recorded (Draw / DrawIndirect)
    const DrawPlan* geometry_plan{}; ///< N6.2: the plan of the draw whose vertex and index buffers are being bound (its record names their engine ids)
    std::unique_ptr<DrawPlan> dispatch_binder_plan; ///< B9: scratch plan the binder reads a dispatch's record from
    const DrawPlan* DispatchBinderFor(const DispatchPlan* plan);
    const DrawPlan* binder_plan{};   ///< N3: the plan whose record binds the resources of the draw being recorded (null: from the flat user data)
    /// R3 step 2: the record words of image `index` of `stage` as the command processor found them (the binder plan's bindings), or false when it did not resolve it
    bool PlannedImageWords(const Shader::Info& stage, u32 index, u32 dwords, const u32*& words, u32& binding_index) const;
    const DrawPlan* resolved_draw{};         ///< R2: the draw being bound, when its plan carries resolved bindings
    const DispatchPlan* resolved_dispatch{}; ///< R2: the dispatch being bound
    ResolveStats resolve_stats;
    const DrawPlan* NativeBinderFor(const DrawPlan* plan, u32 draw_class);
    /// kind: 1 buffer, 2 image, 3 sampler descriptor
    const u32* BinderWords(const Shader::Info& stage, u32 offset, u32 dwords, u32 kind, u32* binding_index = nullptr) const;
    /// N6.2: the immutable engine buffer the record's ids name that holds [address, address + size) (null: none, the lookup by address decides)
    Core::NativeResources::ImmutableBuffer* GeometryHandle(VAddr address, u64 size) const;
    /// N3b: the binder's plan has no flat user data for this stage
    bool StageFlatSkipped(const Shader::Info& stage) const;
    u32 draw_class_now{15}; ///< census: the draw class of the draw being bound
    u64 dispatch_caller{}; ///< diagnostics: the game call that made the dispatch being bound
    u8 clip_plane_mask{}; ///< user clip planes of the draw being bound
    /// True if the inputs match the cached lookup; otherwise stores them for the next draw.
    template <typename... Ts>
    bool TargetInputsMatch(CachedTarget& cache, const Ts&... parts);

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    /// tlg C9: the draw's row of slot numbers (the indexed buffers of the stages, in the order of the shaders' resources)
    bool indexed_active{false};
    bool record_indexed_active{false}; ///< P3e.4b: the same, as the record half sees it (the resolve half moves on to the next draw's pipeline before the record half of this one runs)
    /// Puts the slot of one descriptor into the draw's row.
    void NoteIndexedRow(u32 slot);
    /// The dwords of a stage's flat user data go into the row (as many as the shader reserved), zero padded.
    void NoteIndexedFlat(const Shader::Info& stage, const u32* flat, u32 flat_size);
    /// The slot of a sampled image / of a sampler for the draw's row, with the verify against what the legacy binder writes.
    void NoteIndexedImage(const vk::DescriptorImageInfo& info, u32 image_key);
    /// An image bound once per mip level: a run of consecutive slots, the row holds the first.
    void NoteIndexedRun(const vk::DescriptorImageInfo* infos, u32 count, u32 image_key);
    void NoteIndexedSampler(vk::Sampler sampler);
    std::array<u32, 4096> indexed_row{};
    u32 indexed_row_size{0};

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    /// draw recipe verify (graphics draws): the identity of the record, the prediction made at the start of BindResources, the guest ranges the buffers came from
    u64 recipe_key{};
    u32 recipe_prediction{0};
    u64 recipe_stored_shape{};
    bool recipe_active{false}, recipe_reusable{true};
    VideoCore::BufferCache::RecipeEpochs recipe_epochs_at_start{};
    boost::container::small_vector<std::pair<VAddr, u32>, 8> recipe_ranges;
    boost::container::small_vector<std::pair<VAddr, u32>, 4> recipe_image_ranges; ///< V0: the guest ranges of the images this draw bound
    u64 recipe_identity_clock{};
    VideoCore::BufferCache::ScopedPrediction version_prediction{};

    bool attachment_feedback_loop{};
    bool needs_barrier{};
    // Plan S step P3e.4: who owns which member (checked by scratchpad/audit_members.py: each name against the functions that mention it).
    //   recorder-owned (only Record*, RecordDraw, FinishBindResources, the legacy Bind* paths and the non-draw operations behind a drain): needs_barrier, bound_buffers,
    //     set_writes, set_write_index, buffer_infos, image_infos, indexed_row, indexed_row_size, recorded_work, last_barrier_work, native_last, push_data (a copy per draw),
    //     indexed_active (read by Record*: a recorder copy is needed before the thread)
    //   resolver-owned: bound_images, clip_plane_mask, indexed_active (read by Resolve*), cb_descs, db_desc, rt_cache, texture_lookup_cache, native_images, draw_call_seq
    //   carried by the job: push_data's first value, attachment_feedback_loop (written by ResolveRenderTargets, read by UpdateColorBlendingState)
};

} // namespace Vulkan
