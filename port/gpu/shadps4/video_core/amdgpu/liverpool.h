// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "native/native_sync.h"
#include "native/native_direct.h"
#include "native/native_draw.h"

#include "native/native_draw_tag.h"
#include "native/native_raster_queue.h"
#include "native/native_executor.h"
#include "native/native_executor_sync.h"
#include "native/native_raster_host.h"
#include "native/native_readback_policy.h"

#include <source_location>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/readback_stats.h"

namespace Vulkan {
class Rasterizer;
struct DrawPlan;
struct DispatchPlan;
struct FixedState;
} // namespace Vulkan

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

union PM4Header;

struct Liverpool : Core::NativeExecutor {
    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    u32 index_buffer_num_indices{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};
    std::array<u64, 9> last_target_resources{}; ///< E3: engine resource ids behind the bound color views (0..7) and the depth view (8)

public:
    explicit Liverpool();
    ~Liverpool() override;

    /// What the rasterizer reads while it executes draws and dispatches (F3.5: the extent hints and the target resources; no register image).
    [[nodiscard]] std::array<CbDbExtent, NUM_COLOR_BUFFERS>& RasterCbExtents() noexcept {
        return last_cb_extent;
    }
    [[nodiscard]] CbDbExtent& RasterDbExtent() noexcept {
        return last_db_extent;
    }
    [[nodiscard]] std::array<u64, 9>& RasterTargetResources() noexcept {
        return last_target_resources;
    }
    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Core::RasterHost* rasterizer_);

    inline ComputeProgram& GetCsRegs() {
        return cs_states[curr_qid];
    }

private:
    /// The compute register block of every queue (the dispatches the command processor runs itself read it).
    std::array<ComputeProgram, NumTotalQueues> cs_states{};

    // what the executor asks of the host (native/native_executor.h)
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, std::shared_ptr<Core::NativeDirect::Buffer> direct) override;
    Task ProcessDirect(std::shared_ptr<Core::NativeDirect::Buffer> direct, std::span<const u32> ccb) override;
    Task ProcessCompute(std::span<const u32> acb, u32 vqid, std::shared_ptr<Core::NativeDirect::Buffer> direct) override;
    Task ProcessDirectAsc(std::shared_ptr<Core::NativeDirect::Buffer> direct, u32 vqid) override;
    void SubmitEnd() override;

    Task DirectWait(const Core::NativeSync::LabelWait& wait);
    /// E6.2c: the wait for a free display buffer (native/native_display.h)
    Task DirectWaitDisplay(u32 buffer);
    void StartNativeTag(const u32* tag);
    void ExecuteDma(const Core::NativeSync::DmaCopy& dma);
    void FakeOcclusionDump(VAddr address); ///< TLG_REAL_OCCLUSION=0: the made up answer
    void ExecuteConstantEngine(std::span<const u32> ccb); ///< native/native_liverpool.cpp
    /// B2: queues the announced updates a submission carries to the raster thread (before its work)
    void QueueAnnounced(Core::NativeDirect::Buffer& direct);
    Task DirectWaitAsc(const Core::NativeSync::LabelWait& wait, u32 vqid);
    void StartAscTag(const u32* tag, const void* address, u32 vqid);
    void SignalComputeRelease(const Core::NativeSync::Fence& release, u32 vqid, u32 pipe_id);

    // ---- Command processor split (documents/tlg/cp-split-design.md) ----
    /// What the raster thread does with an operation (native/native_raster_queue.h); the member functions are in liverpool.cpp.
    struct RasterConsumer final : Core::NativeRasterQueue::Consumer {
        explicit RasterConsumer(Liverpool& liverpool) : owner{liverpool} {}
        void ThreadStarted() override;
        [[nodiscard]] bool CallbacksPending() const override;
        void RunCallbacks() override;
        void SelectStream(bool second) override;
        void RunCall(RasterOp& op) override;
        void RunClosure(RasterOp& op) override;
        [[nodiscard]] bool IsDepthOnly(const Vulkan::DrawPlan& plan) const override;
        void Release(RasterOp& op) override;
        void PublishIntervalStats(RasterSplit& queue, u64 now_ns) override;
        void BatchFinished() override;
        Liverpool& owner;
    };
    RasterConsumer raster_consumer{*this};
    /// Tag written by the game in front of the next draw (front thread only).
    VideoCore::NativeTag native_tag_pending{};
    u64 native_exec_calls{};
    u32 native_tag_skips{};
    void ExecRasterCall(const RasterCall& call, const Vulkan::DrawPlan* draw_plan,
                        const Vulkan::DispatchPlan* dispatch_plan);
    /// Queues a draw or dispatch. Without the split it runs right away.
    void RasterDispatch(const RasterCall& call);
    /// Draws the pending native draw tag from its record instead of from command packets.
    void ExecuteNativeDraw(const void* header = nullptr, u32 asc_vqid = ~0u, u32 index_count = ~0u);
    void SignalGraphicsEop(const Core::NativeSync::Fence& eop);
    bool native_follow_pending{}; ///< an executed draw waits for its draw packet (index count)
    /// The place of the wait before a planning: the full wait, or with TLG_DRAIN_LAZY=1 no wait (the planning's reads wait where needed, native_drain_audit.h).
    /// Returns 0, or 1 when the planning is verified by comparing the bytes it copied at its end, or 2 when it must be repeated after a full wait and compared (ReplanAfterDrain).
    u32 LookupDrain(u32 caller);
    Core::DrawPlanPtr ReplanAfterDrain(Core::DrawPlanPtr plan, u32 site, const std::function<Core::DrawPlanPtr()>& make);
    Core::DispatchPlanPtr ReplanAfterDrain(Core::DispatchPlanPtr plan, u32 site, const std::function<Core::DispatchPlanPtr()>& make);
    /// A command processor write to guest memory, done in order with the draws.
    void CpWriteData(void* address, const u32* data, u32 data_size, u32 queue);
    void ApplyStateTemplate(const Core::NativeDraw::StateTemplate& state);

    /// The registers of a record come in groups (targets, the two stage programs, their user data, the interpolants, the draw arguments, the rest). A group
    /// whose values are the ones the last record applied, and that nothing has written since, is not applied again (front thread only).
    static constexpr u32 NumRegisterGroups = 8;
    bool native_exec_scope{}; ///< the draw being dispatched is an executed record
    u64 native_inputs_key{};                        ///< identity of the render target inputs of the draw being dispatched (0: unknown)
    std::array<u32, 16> exec_vertex_user_data{};     ///< the typed user data of the last executed records, per stage
    std::array<u32, 16> exec_pixel_user_data{};
    Core::DrawPlanPtr native_exec_plan;              ///< a draw planned from its record: RasterDispatch pushes it instead of planning
    u64 native_exec_key{}; ///< pipeline key of the draw being dispatched, from the record (0: look the pipeline up from the registers)
    const Vulkan::FixedState* native_fixed{}; ///< fixed-function state of the draw being dispatched, from the engine's objects (null: from the registers)

    /// The label timeline of the recorded fences (native/native_executor_sync.h).
    using FenceMatch = Core::NativeExecutorSync::FenceMatch;
    Core::NativeExecutorSync::LabelTimeline label_timeline;
    FenceMatch MatchRecordedFence(const Core::NativeSync::LabelWait& wait_reg_mem);

    /// readback_fence_wait_shortcut: 0 = off, 1 = graphics queue waits, 2 = graphics and compute
    /// queue waits, 3 = like 2 plus a GPU barrier that orders the work after the wait on the
    /// GPU (the wait passes on a recorded fence instead of a finished one).
    [[nodiscard]] bool FenceShortcutEnabled(bool is_compute) const noexcept {
        const s32 override_mode = VideoCore::ReadbackPolicy::Get().fence_shortcut_override.load(std::memory_order_relaxed);
        const u32 mode = override_mode >= 0 ? static_cast<u32>(override_mode) : fence_wait_shortcut;
        return mode != 0 && (!is_compute || mode >= 2);
    }

    /// Called when a wait passed on a recorded fence: orders the following GPU work after it.
    void OnWaitPassedOnFence(const Core::NativeSync::LabelWait& wait_reg_mem, const FenceMatch& match,
                             u32 queue);

    /// Logs a command stream wait that has been blocked for more than 2 seconds.
    void ReportLongWait(const Core::NativeSync::LabelWait& wait_reg_mem, const char* queue, u64 wait_start,
                        bool& reported);
    /// Logs a MemSemaphore/Rewind wait that has been blocked for more than 2 seconds.
    void ReportLongSpin(const char* what, const void* address, u64 wait_start, bool& reported);

    /// A command processor memory write (WriteData) normally lands when the packet is processed.
    /// If a fence write to the same label is still held back, the GPU would run it after that
    /// fence, so it is queued behind it instead. Returns false when it can be written now.
    bool DeferWriteBehindFences(void* address, const u32* data, u32 num_bytes, u32 queue);

    /// Fence latency statistics per FenceKind (native/native_executor_sync.h).
    Core::NativeExecutorSync::FenceLatency fence_latency;
    void NoteFenceLatency(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns);

    /// Runs `signal` once the readbacks before this fence are in guest memory, recording the
    /// fence value (if known) in the label timeline meanwhile.
    void SignalFenceAfterReadbacks(Common::UniqueFunction<void>&& signal, bool must_sync,
                                   VAddr address, u64 value, u32 num_bytes, u32 queue,
                                   VideoCore::FenceKind kind);

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};
    VAddr occlusion_dump_address{}; ///< the result block address of the last occlusion query dump the command processor passed (native/native_occlusion.h)

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    /// The graphics queue's task runs until it blocks on a wait; a compute queue kicked meanwhile (sceGnmDingDong) would wait for the whole run
    /// (up to 4 ms in the light scene, documents/tlg/light-scene-cap-plan.md). True when another queue has work and `last_ns` is at least 50 us old
    /// (then set to now): the graphics task yields so the round robin runs it. Switch: env TLG_CP_ASC_YIELD=0 turns it off.
    [[nodiscard]] bool ShouldYieldToOtherQueue(u64& last_ns);

    Core::RasterHost* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const u32 fence_wait_shortcut;
};

} // namespace AmdGpu
