// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The native executor (G4d, documents/tlg/native plan/executor-consumer-plan.md): the command processor's skeleton. It owns the submit queues and their coroutine tasks, the
// thread that runs them round robin (`Process`), the wait for the GPU to go idle, the commands other threads send (`SendCommand`) and the raster thread's queue
// (native/native_raster_queue.h). Since G4e it also holds the coroutine tasks (ProcessDirect and the others), the draw and dispatch execution, the fences and the end of a submit
// (native_executor_direct.cpp, native_executor_raster.cpp); what they need of the rasterizer is the
// host's, reached through the rasterizer host interface (native/native_raster_host.h).

#include <array>
#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <semaphore>
#include <source_location>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "native/native_direct.h"
#include "native/native_draw.h"
#include "native/native_draw_tag.h"
#include "native/native_executor_sync.h"
#include "native/native_raster_host.h"
#include "native/native_raster_queue.h"
#include "native/native_readback_policy.h"
#include "native/native_sync.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/readback_stats.h"

namespace Core {

class NativeExecutor {
public:
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    /// A command buffer run as a coroutine: suspended at the start, resumed round robin by `Process`, yields while it waits.
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
    };

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };

    using RasterSplit = NativeRasterQueue::Queue;
    using RasterCall = NativeRasterQueue::Call;
    using RasterOp = NativeRasterQueue::Op;

    NativeExecutor();
    ~NativeExecutor();

    /// What the rasterizer reads while it executes draws and dispatches (F3.5: the extent hints and the target resources; no register image).
    [[nodiscard]] std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS>& RasterCbExtents() noexcept {
        return last_cb_extent;
    }
    [[nodiscard]] AmdGpu::CbDbExtent& RasterDbExtent() noexcept {
        return last_db_extent;
    }
    [[nodiscard]] std::array<u64, 9>& RasterTargetResources() noexcept {
        return last_target_resources;
    }

    /// The video out labels (the display buffers' flip labels): a wait on one sleeps on the video out's condition instead of polling.
    struct VoLabels {
        std::function<bool(const u64*)> is_label;
        std::function<void(const std::function<bool()>&)> wait;
    };
    void SetVoLabels(VoLabels labels) {
        vo_labels = std::move(labels);
    }

    /// The host that executes the planned draws and dispatches (null: none yet, or the rasterizer is gone).
    void BindRasterizer(RasterHost* rasterizer_);

    AmdGpu::ComputeProgram& GetCsRegs() {
        return cs_states[curr_qid];
    }

    /// `partial`: a part of a command buffer that the game is still recording (streaming of the draws); it does not count as work submitted by the game.
    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool partial = false);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0 && RasterIdle(); });
    }

    bool IsGpuIdle() const {
        return num_submits == 0 && RasterIdle();
    }

    /// The GPU is idle as far as the game's own submits are concerned (work streamed ahead of the game's submit does not count).
    bool IsGpuIdleForGame() const {
        return IsGpuIdle() || stream_open.load(std::memory_order_acquire) != 0;
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (split_active && std::this_thread::get_id() == front_id) {
            // The command processor thread read guest memory that faulted (GPU data): the
            // callback must see all earlier work executed, as it did before the split.
            DrainRaster();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            WakeRaster();
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
        WakeRaster();
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

    /// True while the raster thread runs the rasterizer (cp_split_mode): the pipeline lookup is
    /// then done by the command processor thread.
    [[nodiscard]] bool SplitActive() const noexcept {
        return split_active;
    }

    // ---- the raster queue (native/native_raster_queue.h) ----
    /// Queues `func` (a closure) for the raster thread, or runs it now without the split.
    /// `writes_memory`: the closure writes guest memory that later pipeline lookups may read
    /// (descriptor tables, shader code), so the command processor thread must not plan a draw
    /// before it has run.
    void RasterRun(Common::UniqueFunction<void>&& func, bool wait = false,
                   bool writes_memory = false,
                   std::source_location site = std::source_location::current());
    void RasterPush(RasterOp&& op, bool wait);
    /// Blocks until every queued closure that writes guest memory has run.
    void DrainRasterWrites(u32 caller = 3);
    /// Blocks until the raster thread has executed everything queued so far.
    void DrainRaster();
    [[nodiscard]] bool RasterIdle() const noexcept;
    /// Wakes the raster thread (it runs SendCommand callbacks) when the split is active.
    void WakeRaster();
    void StartRasterSplit(NativeRasterQueue::Consumer& consumer);
    void StopRasterSplit();

    // ---- state ----
    /// Streaming of the draws: a frame has parts of its command buffer handed over and the real submit is still to come; the GpuIdle signal waits for it.
    std::atomic<u32> stream_open{0};
    std::atomic<u32> partial_pending{0}; ///< streamed spans queued and not done
    std::mutex partial_mutex;
    std::unordered_set<void*> partial_handles;
    Common::SlotVector<AscQueueInfo> asc_queues{64};
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    std::thread::id front_id;
#if defined(__linux__) || defined(_WIN32)
    u32 gpu_tid{};
#endif
    s32 curr_qid{-1};

    /// The raster thread split of the command processor (cp_split_mode).
    bool split_active{};
    std::unique_ptr<RasterSplit> split;

    /// Runs the callbacks other threads sent (on the raster thread when the split is active).
    void ProcessCommands();

private:
    /// Starts / stops the thread that runs `Process` (the constructor and the destructor).
    void StartProcess();
    void StopProcess();

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    void Process(std::stop_token stoken);

    // ---- the coroutine tasks (native_executor_raster.cpp, native_executor_direct.cpp) ----
    /// A graphics / compute command buffer the direct path cannot run (skipped with an error), and the ones it runs.
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, std::shared_ptr<NativeDirect::Buffer> direct);
    Task ProcessDirect(std::shared_ptr<NativeDirect::Buffer> direct, std::span<const u32> ccb);
    Task ProcessCompute(std::span<const u32> acb, u32 vqid, std::shared_ptr<NativeDirect::Buffer> direct);
    Task ProcessDirectAsc(std::shared_ptr<NativeDirect::Buffer> direct, u32 vqid);
    /// The end of a game submit: the readbacks and the flush, queued for the raster thread.
    void SubmitEnd();

    u32 index_buffer_num_indices{};
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> last_cb_extent{};
    AmdGpu::CbDbExtent last_db_extent{};
    std::array<u64, 9> last_target_resources{}; ///< E3: engine resource ids behind the bound color views (0..7) and the depth view (8)

    /// The compute register block of every queue (the dispatches the command processor runs itself read it).
    std::array<AmdGpu::ComputeProgram, NumTotalQueues> cs_states{};

    Task DirectWait(const Core::NativeSync::LabelWait& wait);
    /// E6.2c: the wait for a free display buffer (native/native_display.h)
    Task DirectWaitDisplay(u32 buffer);
    void StartNativeTag(const u32* tag);
    void ExecuteDma(const Core::NativeSync::DmaCopy& dma);
    void FakeOcclusionDump(VAddr address); ///< TLG_REAL_OCCLUSION=0: the made up answer
    void ExecuteConstantEngine(std::span<const u32> ccb); ///< native_executor_direct.cpp
    /// B2: queues the announced updates a submission carries to the raster thread (before its work)
    void QueueAnnounced(Core::NativeDirect::Buffer& direct);
    Task DirectWaitAsc(const Core::NativeSync::LabelWait& wait, u32 vqid);
    void StartAscTag(const u32* tag, const void* address, u32 vqid);
    void SignalComputeRelease(const Core::NativeSync::Fence& release, u32 vqid, u32 pipe_id);

    // ---- Command processor split (documents/tlg/cp-split-design.md) ----
    /// What the raster thread does with an operation (native/native_raster_queue.h); the member functions are in native_executor_raster.cpp.
    struct RasterConsumer final : Core::NativeRasterQueue::Consumer {
        explicit RasterConsumer(NativeExecutor& executor) : owner{executor} {}
        void ThreadStarted() override;
        [[nodiscard]] bool CallbacksPending() const override;
        void RunCallbacks() override;
        void SelectStream(bool second) override;
        [[nodiscard]] bool LaneUsable() const override;
        void RunCall(RasterOp& op) override;
        void RunClosure(RasterOp& op) override;
        [[nodiscard]] bool IsDepthOnly(const Vulkan::DrawPlan& plan) const override;
        void Release(RasterOp& op) override;
        void PublishIntervalStats(RasterSplit& queue, u64 now_ns) override;
        void BatchFinished() override;
        NativeExecutor& owner;
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

    /// Early compute fences: per virtual compute queue, the compute work with a result the CPU may read (anything but copies and fills) that the command processor has issued, and the
    /// part of it a landed fence covers. A fence of copies may land at the parse when issued <= covered.
    std::array<std::atomic<u64>, NumTotalQueues> asc_noncopy_issued{};
    std::array<std::atomic<u64>, NumTotalQueues> asc_noncopy_covered{};

    RasterHost* rasterizer{};
    VoLabels vo_labels;
    const u32 fence_wait_shortcut;
};

} // namespace Core
