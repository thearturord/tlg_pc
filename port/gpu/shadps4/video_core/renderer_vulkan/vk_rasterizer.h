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
    void SetDrawPlanTargets(DrawPlan& plan, const ColorExtents& cb_extent, const DepthExtent& db_extent, const std::array<u64, 9>& target_resources) override;
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

    u32 const_ring_level = 0;
    std::unique_ptr<ConstantRing> constant_ring;
    // N7.5 (raster thread)
    bool inplace_on = false;
    u32 inplace_cap = 16384;
    std::unique_ptr<HostImport> host_import;
    std::atomic<u64> inplace_offered{0}, inplace_taken{0}, inplace_bytes{0}, inplace_written{0}, inplace_special{0}, inplace_too_large{0}, inplace_not_in_backing{0}, inplace_gpu_written{0},
        inplace_image_or_meta{0};
    std::array<std::atomic<u64>, 64> consumer_census{};
    void Census(u32 index, u64 n = 1) noexcept {
        consumer_census[index & 63].fetch_add(n, std::memory_order_relaxed);
    }
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
        u32 array_key{}; ///< tlg C9: the image's descriptor array (Shader::Indexed::ImageKey of its T#)
    };
    struct TextureSet {
        static constexpr u32 MaxImages = 16;
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
    bool ResolveTextureSet(const Shader::Info& stage, TextureSet& set, const std::array<u32, 8>* raws, const u64* flags, u32 count, TextureResolved& r);
    bool BindResources(const Pipeline* pipeline, const FixedState* fixed = nullptr);
    void BindCensus(const Pipeline* pipeline);

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
        std::array<Target, AmdGpu::NUM_COLOR_BUFFERS> color{};
        std::array<bool, AmdGpu::NUM_COLOR_BUFFERS> color_used{};
        Target depth{};
    };
    RenderState BeginRenderingLegacy(const GraphicsPipeline* pipeline);
    void ResolveRenderTargets(const GraphicsPipeline* pipeline, RenderResolved& r);
    RenderState RecordRenderTargets(RenderResolved& r);
    void RecordBufferAccess(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    VideoCore::NativeTag native_tag{};
    struct NativeLast {
        bool valid{};
        u64 k2{};
        const GraphicsPipeline* pipeline{};
        u64 tick{};
    } native_last{};
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
    static constexpr std::array<u64, 9> kNoTargetResources{};
    const std::array<u64, 9>* current_target_resources{&kNoTargetResources}; ///< E3: engine resource ids behind the draw's color views (0..7) and depth view (8).
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
    /// Puts the slot of one descriptor into the draw's row.
    void NoteIndexedRow(u32 slot);
    /// The dwords of a stage's flat user data go into the row (as many as the shader reserved), zero padded.
    void NoteIndexedFlat(const Shader::Info& stage);
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
};

} // namespace Vulkan
