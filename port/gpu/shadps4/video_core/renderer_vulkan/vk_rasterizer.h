// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/readback_policy.h"

#include "video_core/amdgpu/draw_state.h"
#include <mutex>
#include <vector>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/native_draw_tag.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include <span>
#include <string>
#include "core/native_draw.h"
#include "core/native_uploads.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/vk_constant_ring.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;
struct FixedState;
struct DrawInputs;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

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

    /// Command processor thread: looks the pipeline up for `regs`. Returns null for draws that
    /// FilterDraw handles without a pipeline (the raster thread repeats that decision).
    /// `native` is the fixed-function state the native path built from the engine's objects (null: built from `regs`).
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanDraw(const AmdGpu::DrawRegs& regs,
                                                     DrawIndirectParams params,
                                                     const FixedState* native = nullptr,
                                                     u32 native_stale = 0, bool verify = true, u64 exec_key = 0, u64 inputs_key = 0);
    /// cp_exec_inputs: plans a native draw from its record alone. Null when the render target inputs or the pipeline of the record's keys are not known yet, or the
    /// fixed-function state needs the registers (the caller plans the old way).
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanExecDraw(const FixedState& native, u32 native_stale, u64 exec_key, u64 inputs_key,
                                                         std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data,
                                                         const Core::NativeDraw::ExecDraw& draw, u32 index_count = ~0u, u32 draw_class = 15);
    /// The plan of a fast clear eliminate record: the operation and the render target it works on, decoded from the record's register pairs.
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanExecOperation(const Core::NativeDraw::ExecDraw& draw);
    /// The differences between a plan made from the record and the same draw planned from the registers ("" when they agree; the user data under the masks).
    [[nodiscard]] static std::string DiffExecPlans(const DrawPlan& typed, const DrawPlan& reference, u16 vertex_mask, u16 pixel_mask, u32 draw_kind);
    [[nodiscard]] std::unique_ptr<DispatchPlan> PlanDispatch(const AmdGpu::ComputeProgram& cs, const AmdGpu::BorderColorBuffer& ta_bc_base);

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
        ring_draw = draw;
        ring_dispatch = dispatch;
    }
    void EndRingUse() noexcept;
    struct RingStats {
        u64 queued{}, written_buffers{}, offered{}, taken{}, bytes{};
        u64 no_sharp{}, written{}, formatted{}, too_large{}, immutable{}, gpu_modified{}, pending_write{}, ring_full{};
        u64 bound{};
        u64 verified{}, differing{};
    };
    RingStats TakeRingStats() noexcept;
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
    /// Render state memo hits and misses since the last call (raster thread).
    std::pair<u64, u64> TakeBeginMemoStats() noexcept {
        return {std::exchange(begin_memo_hits, 0), std::exchange(begin_memo_misses, 0)};
    }

    PipelineCache::PlanStats TakePlanStats() noexcept {
        return pipeline_cache.TakePlanStats();
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
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool IsMapped(VAddr addr, u64 size);
    /// N5c (raster thread): uploads the ranges the engine announced (outside a render pass: it is called in front of a draw or dispatch).
    void PreSynchronize(const std::vector<Core::NativeUploads::Range>& ranges);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();

    /// Raster thread: records the following operations into the asynchronous stream (second Vulkan queue, TLG_ASC_QUEUE=1) or the graphics stream.
    /// Leaving the asynchronous stream submits what it recorded. No effect without the second queue; the first windows stay on the graphics stream
    /// (the inputs shared with the graphics are uploaded there first).
    void SelectStream(bool side);
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
    const DrawPlan* ring_draw = nullptr;
    const DispatchPlan* ring_dispatch = nullptr;
    const RingBinding* FindRingBinding(const Shader::Info& stage, u32 index) const noexcept;
    /// TLG_CONST_RING_VERIFY=N: every Nth ring binding of the raster thread is compared with the guest memory it was copied from (a difference is the guest
    /// writing it after the copy, or a stale copy; the counts tell how often).
    u32 ring_verify_interval = 0;
    u64 ring_verify_counter = 0;
    std::atomic<u64> ring_verified{0}, ring_differing{0};
    std::atomic<u64> ring_queued{0}, ring_written_buffers{0}, ring_offered{0}, ring_taken{0}, ring_bytes{0};
    std::atomic<u64> ring_no_sharp{0}, ring_written{0}, ring_formatted{0}, ring_too_large{0}, ring_immutable{0}, ring_gpu_modified{0}, ring_pending_write{0},
        ring_full{0}, ring_bound{0};

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
    u64 begin_memo_hits = 0, begin_memo_misses = 0;
    void Resolve(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0, const AmdGpu::CbDbExtent& cb1);
    void DepthStencilCopy(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent, bool is_depth, bool is_stencil);
    void EliminateFastClear(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0);
    /// Clears the bound depth / stencil target to the clear values of the registers (native replacement of the SDK's clear quad).
    void ClearDepthTarget(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent);

    const FixedState& FixedFor(const DrawPlan* plan, FixedState& storage) const;
    /// The draw inputs of the draw being recorded: the plan's, or built from the register mirror without a plan.
    const DrawInputs& InputsFor(const DrawPlan* plan, DrawInputs& storage) const;
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
    /// What FilterDraw does with a draw, decided from the registers only. `executed` is the fixed-function state of a draw
    /// the native path executed from its record: such a draw is a normal draw by construction, only the shadow skip
    /// switch looks at it.
    static DrawClassification ClassifyDraw(const AmdGpu::DrawRegs& regs,
                                           const FixedState* executed = nullptr);
    /// The part of ClassifyDraw that finds the game's own transition operations; true and `out` set when the draw is one.
    static bool ClassifyOperation(const AmdGpu::DrawRegs& regs, DrawClassification& out);
    /// What is left for a draw that is not such an operation, from the plan only (the shadow skip switch).
    static DrawClassification ClassifyRemaining(const DrawInputs& inputs, const FixedState& fixed);

    void ProcessReadbacksSync();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding);

    /// Texture set memo (documents/tlg/pc-port-draw-pipeline-architecture.md, T3; the port's `TextureSetMemo`): the images a stage's T#s resolve to, remembered by
    /// the raw T# and the use flags of each resource. A hit redoes only the per draw effects (touch, update, bound flags, layout transition, descriptor info).
    /// Raster thread only. Env TLG_TEXTURE_SET_MEMO=1 turns it on.
    struct TextureSetEntry {
        AmdGpu::Image raw{};
        u64 flags{};
        VideoCore::ImageId found{}; ///< what FindImage returned
        VideoCore::ImageId id{};    ///< after the depth redirect (null entry: an empty descriptor)
        vk::ImageView view{};
        const void* backing{};
        VideoCore::SubresourceRange range{};
        bool null_entry{};
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
    bool BindTexturesFromSet(const Shader::Info& stage, TextureSet& set, const AmdGpu::Image* raws, const u64* flags, u32 count,
                             Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline, const FixedState* fixed = nullptr);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);

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
    std::unique_ptr<Scheduler::SideScope> side_scope;
    u32 side_windows{};
    bool side_requested{};
    u64 side_epoch{1};
    u64 side_last_end_ns{};
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    const std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS>* current_cb_extent{}; ///< The extent hints of the targets of the draw being recorded.
    const AmdGpu::CbDbExtent* current_db_extent{};
    const AmdGpu::BorderColorBuffer* current_bc_base{}; ///< The border colour table of the draw or dispatch being recorded.
    const AmdGpu::ComputeProgram* current_cs{}; ///< The compute program of the dispatch being recorded (its plan's, or the queue's without a plan).
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
    struct CachedTextureLookup {
        bool valid{};
        u64 generation{};
        VideoCore::ImageId image_id{};
        std::array<u64, 6> key{};
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<CachedTextureLookup, 512> texture_lookup_cache{};
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
    std::mutex plan_pool_mutex;
    std::vector<std::unique_ptr<DrawPlan>> draw_plan_pool;
    struct PendingFill {
        VAddr dst{};
        u32 bytes{};
        u32 dwords{};
        std::array<u32, 4> pattern{};
    };
    PendingFill pending_fill{};
    const DrawInputs* draw_inputs{}; ///< of the draw being recorded (Draw / DrawIndirect)
    const DrawPlan* binder_plan{};   ///< N3: the plan whose record binds the resources of the draw being recorded (null: from the flat user data)
    bool binder_verify{};
    const DrawPlan* NativeBinderFor(const DrawPlan* plan, u32 draw_class);
    /// kind: 1 buffer, 2 image, 3 sampler descriptor
    const u32* BinderWords(const Shader::Info& stage, u32 offset, u32 dwords, u32 kind, u32* binding_index = nullptr) const;
    /// N3b: the binder's plan has no flat user data for this stage
    bool StageFlatSkipped(const Shader::Info& stage) const;
    u64 dispatch_caller{}; ///< diagnostics: the game call that made the dispatch being bound
    u8 clip_plane_mask{}; ///< user clip planes of the draw being bound
    std::vector<std::unique_ptr<DispatchPlan>> dispatch_plan_pool;
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

    bool attachment_feedback_loop{};
    bool needs_barrier{};
};

} // namespace Vulkan
