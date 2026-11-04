// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The rasterizer as the native executor sees it (G4e, documents/tlg/native plan/executor-consumer-plan.md). The executor (native/native_executor.h) plans draws and dispatches, queues
// them for the raster thread and executes them there; everything that touches Vulkan is done through this interface. The Vulkan side (Vulkan::Rasterizer) implements it; the
// executor never includes a Vulkan header. A plan is an opaque object of the Vulkan side: the executor holds it in a `DrawPlanPtr` / `DispatchPlanPtr` and asks the host for
// the few things it needs to know about it.
//
// Threads: the functions are called from the thread named in their comment. "front" is the command processor thread, "raster" the raster thread (the same thread without the split).

#include <array>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/types.h"
#include "common/unique_function.h"
#include "native/native_device.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw.h"
#include "native/native_draw_tag.h"
#include "native/native_raster_queue.h"
#include "native/native_uploads.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/readback_stats.h"

namespace Vulkan {
struct FixedState;
struct DrawInputs;
} // namespace Vulkan

namespace Core {

/// How a draw gets its first vertex and instance offsets from the argument buffer of an indirect draw.
struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

using DrawPlanPtr = std::unique_ptr<Vulkan::DrawPlan, NativeRasterQueue::PlanDelete>;
using DispatchPlanPtr = std::unique_ptr<Vulkan::DispatchPlan, NativeRasterQueue::PlanDelete>;

class RasterHost {
public:
    virtual ~RasterHost() = default;

    /// What the interval statistics of the host hand back to the executor (the host publishes its own part, native/native_debug_server.h).
    struct IntervalStats {
        u64 plan_draws_all_stages_same{}, plan_draws{}, plan_stages_same{}, plan_stages{};
        u64 memo_hits{}, memo_misses{};
        u64 native_tagged{}, native_untagged{}, native_dynamic_skipped{}, native_bind_skipped{};
    };

    // ---- memory ----
    /// The range has a GPU mapping (so it must not be written off the raster thread).
    virtual bool IsMapped(VAddr addr, u64 size) = 0;
    /// A guest memory write done by the command processor itself (raster thread, in order with the draws).
    virtual void NoteCpGuestWrite(VAddr addr, u64 size) = 0;

    // ---- planning (front) ----
    [[nodiscard]] virtual DrawPlanPtr MakeDrawPlan(DrawIndirectParams params, const Vulkan::FixedState& native, u64 exec_key, u64 inputs_key, const NativeDraw::ExecDraw& record,
                                                   u32 index_count) = 0;
    [[nodiscard]] virtual DrawPlanPtr MakeExecDrawPlan(const Vulkan::FixedState& native, u64 exec_key, u64 inputs_key, std::span<const u32, 16> vertex_user_data,
                                                       std::span<const u32, 16> pixel_user_data, const NativeDraw::ExecDraw& draw, u32 index_count, u32 draw_class) = 0;
    /// The plan of a fast clear eliminate record.
    [[nodiscard]] virtual DrawPlanPtr MakeOperationPlan(const NativeDraw::ExecDraw& draw) = 0;
    [[nodiscard]] virtual DispatchPlanPtr MakeDispatchPlan(const NativeDispatch::DispatchInputs& cs) = 0;
    /// A hash of what the planning read from guest memory and put into the plan (the verification of a planning after a drain, native/native_drain_audit.h).
    [[nodiscard]] virtual u64 DrawPlanDigest(const Vulkan::DrawPlan& plan) = 0;
    [[nodiscard]] virtual u64 DispatchPlanDigest(const Vulkan::DispatchPlan& plan) = 0;
    /// Hands a plan back for reuse (any thread that finished with it).
    virtual void RecycleDrawPlan(DrawPlanPtr&& plan) = 0;
    virtual void RecycleDispatchPlan(DispatchPlanPtr&& plan) = 0;
    /// What the executor reads of and writes into a plan.
    [[nodiscard]] virtual u64 DrawPlanSeq(const Vulkan::DrawPlan& plan) = 0;
    [[nodiscard]] virtual bool DrawPlanIsDepthOnly(const Vulkan::DrawPlan& plan) = 0;
    /// The extent hints and the engine resources behind the bound targets, as the executor holds them when the plan is queued.
    virtual void SetDrawPlanTargets(Vulkan::DrawPlan& plan, const std::array<AmdGpu::CbDbExtent, 8>& cb_extent, const AmdGpu::CbDbExtent& db_extent,
                                    const std::array<u64, 9>& target_resources) = 0;
    [[nodiscard]] virtual const NativeDispatch::DispatchInputs& DispatchPlanInputs(const Vulkan::DispatchPlan& plan) = 0;
    /// Gives a dispatch plan the record it was planned for and the bindings the record carries.
    virtual void TagDispatchPlan(Vulkan::DispatchPlan& plan, u64 seq) = 0;

    // ---- queueing (front) ----
    [[nodiscard]] virtual u32 ConstRingLevel() const noexcept = 0;
    virtual void FrontQueueOp(Vulkan::DrawPlan* draw, Vulkan::DispatchPlan* dispatch, u64 position, u64 done, u64 seq, bool indexed, u32 index_offset) = 0;
    virtual void ResolveAhead(Vulkan::DrawPlan* draw_plan, Vulkan::DispatchPlan* dispatch_plan) = 0;

    // ---- execution (raster) ----
    virtual void SelectStream(bool second) = 0;
    virtual void PreSynchronize(const std::vector<NativeUploads::Range>& ranges) = 0;
    virtual void BeginRingUse(const Vulkan::DrawPlan* draw, const Vulkan::DispatchPlan* dispatch) noexcept = 0;
    virtual void EndRingUse() noexcept = 0;
    virtual void SetNativeTag(const VideoCore::NativeTag& tag) = 0;
    /// Anything other than a tagged draw ran: the reuse state is stale.
    virtual void NativeInvalidate() = 0;
    virtual void Draw(bool is_indexed, u32 index_offset, const Vulkan::DrawPlan* plan) = 0;
    virtual void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count, VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset,
                              const Vulkan::DrawPlan* plan) = 0;
    virtual void DispatchDirect(const Vulkan::DispatchPlan* plan) = 0;
    virtual void DispatchIndirect(VAddr address, u32 offset, u32 size, const Vulkan::DispatchPlan* plan) = 0;
    virtual void SetDispatchCaller(u64 caller) = 0;
    virtual void SetPendingFill(VAddr dst, u32 bytes, const u32* pattern, u32 dwords) = 0;
    virtual void SetExplicitImageReadback(bool on) = 0;
    virtual void MaybeFlushRecorded() = 0;
    virtual void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) = 0;
    virtual void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds, bool staging_readback = false) = 0;
    virtual bool TryClearImage(VAddr dst, u32 bytes, const u32* pattern, u32 dwords) = 0;
    virtual void ClearDepthStencilTarget(const Vulkan::DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent) = 0;
    /// B11: the engine clear whose packets the raster thread is running now (null: none); the record is copied.
    virtual void SetEngineClear(const NativeDevice::ClearRecord* record) = 0;
    /// Real occlusion queries (native/native_occlusion.h).
    virtual void OcclusionOpen(VAddr address, u32 pairs) = 0;
    virtual void OcclusionClose(VAddr address) = 0;

    // ---- markers (raster) ----
    [[nodiscard]] virtual bool MarkersEnabled() const noexcept = 0;
    virtual void MarkerBegin(std::string_view text) = 0;
    virtual void MarkerEnd() = 0;
    /// Runs `func` inside a debug marker named by `fmt` and `args` (when markers are on).
    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (MarkersEnabled()) {
            MarkerBegin(fmt::vformat(fmt, args));
            func();
            MarkerEnd();
        } else {
            func();
        }
    }

    // ---- fences and submits (raster) ----
    virtual bool CommandStreamBarrier(u64 producer_tick) = 0;
    [[nodiscard]] virtual u64 CurrentTick() const noexcept = 0;
    virtual void OnFence(Common::UniqueFunction<void>&& signal, VideoCore::FenceKind kind, bool must_sync) = 0;
    virtual void OnSubmit() = 0;
    virtual u64 Flush() = 0;

    // ---- statistics (raster) ----
    /// Publishes the host's own interval statistics to the debug server (planning, memos, census, buffers, rings) and returns the numbers the executor logs with its own.
    virtual IntervalStats PublishIntervalStats() = 0;
};

} // namespace Core
