// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The native executor (G4d, documents/tlg/native plan/executor-consumer-plan.md): the command processor's skeleton. It owns the submit queues and their coroutine tasks, the
// thread that runs them round robin (`Process`), the wait for the GPU to go idle, the commands other threads send (`SendCommand`) and the raster thread's queue
// (native/native_raster_queue.h). What a task does with the rasterizer (ProcessDirect and the other coroutines, the draw and dispatch execution, the end of a submit) is the
// host's, reached through the pure virtual functions at the end of the class; Liverpool derives from this class until G4e removes it.

#include <array>
#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "native/native_direct.h"
#include "native/native_raster_queue.h"

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
    virtual ~NativeExecutor() = default;

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

protected:
    /// Starts / stops the thread that runs `Process` (the derived class calls them after it is constructed / before it is destroyed: the thread calls the virtual functions below).
    void StartProcess();
    void StopProcess();

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    void Process(std::stop_token stoken);

    // ---- what the host does (Liverpool until G4e) ----
    /// A graphics command buffer the direct path cannot run (skipped with an error), and the one it runs.
    virtual Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, std::shared_ptr<NativeDirect::Buffer> direct) = 0;
    virtual Task ProcessDirect(std::shared_ptr<NativeDirect::Buffer> direct, std::span<const u32> ccb) = 0;
    virtual Task ProcessCompute(std::span<const u32> acb, u32 vqid, std::shared_ptr<NativeDirect::Buffer> direct) = 0;
    virtual Task ProcessDirectAsc(std::shared_ptr<NativeDirect::Buffer> direct, u32 vqid) = 0;
    /// The end of a game submit: the readbacks and the flush, queued for the raster thread.
    virtual void SubmitEnd() = 0;
};

} // namespace Core
