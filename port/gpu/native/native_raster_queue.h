// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The raster thread and its operation queue (G4b, documents/tlg/native plan/executor-consumer-plan.md; cp_split_mode, documents/tlg/cp-split-design.md).
// The command processor (the front thread) queues draws, dispatches and closures; the raster thread executes them in order. The queue, its positions, the waits of the front
// thread (drain to a position, drain the queued memory writes) and the thread loop live here. What an operation does is the `Consumer`'s: the draw and dispatch calls, the
// callbacks sent by other threads and the statistics of the rasterizer belong to the executor (native/native_executor.h implements the interface; the draw and dispatch calls go through native/native_raster_host.h).

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <source_location>
#include <thread>
#include <vector>

#include "common/polyfill_thread.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw_tag.h"

namespace AmdGpu {
struct ComputeProgram;
}

namespace Core::NativePlan {
struct DrawPlan;
struct DispatchPlan;
} // namespace Core::NativePlan

namespace Vulkan {
using Core::NativePlan::DrawPlan;
using Core::NativePlan::DispatchPlan;
} // namespace Vulkan

namespace Core::NativeRasterQueue {

/// Deletes a plan the queue still holds when it is destroyed (defined beside the plan types, in vk_rasterizer_host.cpp: native/ does not include them).
struct PlanDelete {
    void operator()(Vulkan::DrawPlan* plan) const;
    void operator()(Vulkan::DispatchPlan* plan) const;
};

/// A draw or dispatch for the rasterizer, with everything it needs besides the registers.
struct Call {
    enum class Kind : u8 { Draw, DrawIndirect, DispatchDirect, DispatchIndirect };
    Kind kind{};
    bool asc{};       ///< Marker names come from a compute ring (vqid argument).
    bool no_marker{}; ///< Run without a ScopeMarker.
    u32 vqid{};
    const void* cmd_address{}; ///< Only printed in markers, never dereferenced.
    const AmdGpu::ComputeProgram* cs_override{}; ///< Dispatch: this program instead of the queue's (valid until RasterDispatch returns).
    const char* marker{};      ///< Format string literal.
    bool indexed{};
    u32 index_offset{};
    VAddr args_addr{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_addr{};
    u16 vertex_sgpr_offset{};
    u16 instance_sgpr_offset{};
    u32 dispatch_size{};
    VideoCore::NativeTag tag{}; ///< From the game's native draw hook, draws only.
    VAddr fill_destination{};   ///< A dispatch of the fill shader: the pattern fill it performs (the raster thread may clear an image instead).
    u32 fill_bytes{};
    u32 fill_dwords{};
    std::array<u32, 4> fill_pattern{};
};

struct Op {
    bool executed{};  ///< Lane: already run out of order from this batch (skipped when its turn comes).
    u64 pushed_ns{}; ///< Lane: when the operation entered the queue.
    u64 depends_on{}; ///< Lane: the queue position of the operation this one must run after (0 = none, ~0 = everything queued before it).
    bool hoist{};         ///< Early announce: runs at the start of the batch / at the next operation boundary, ahead of the graphics operations queued before it.
    bool lane_passable{}; ///< Lane: a queue 1 operation may run ahead of this closure (no barrier).
    bool has_call{};
    bool writes_memory{};
    u8 audit_class{0xff}; ///< Core::ResourceAudit: the queue whose command stream pushed this operation
    u32 write_line{}; ///< Source line of the RasterRun call that queued a closure.
    const char* write_file{nullptr}; ///< Its source file (a string with static storage).
    u64 position{};   ///< The queue position (Queue::pushed after the push)
    Call call{};
    Common::UniqueFunction<void> closure{};
    Core::NativeDispatch::DispatchInputs cs{}; ///< G1: the dispatch's program as plain numbers (made from the record's dwords by MakeDispatchInputs)
    std::unique_ptr<Vulkan::DrawPlan, PlanDelete> draw_plan{};
    std::unique_ptr<Vulkan::DispatchPlan, PlanDelete> dispatch_plan{};
};

class Queue;

/// What the raster thread does with an operation (and around it). Called on the raster thread only.
class Consumer {
public:
    virtual ~Consumer() = default;
    /// The raster thread started (before its first wait).
    virtual void ThreadStarted() = 0;
    /// Callbacks sent by other threads are waiting (fault readbacks, fences): they run between operations, like they ran between packets before.
    [[nodiscard]] virtual bool CallbacksPending() const = 0;
    virtual void RunCallbacks() = 0;
    /// Before an operation: the stream the operation's queue uses.
    virtual void SelectStream(bool second) = 0;
    /// Executes a draw or dispatch operation (`op.has_call`).
    virtual void RunCall(Op& op) = 0;
    /// Executes a closure operation: the consumer invalidates what it caches, then calls `op.closure`.
    virtual void RunClosure(Op& op) = 0;
    /// True when the draw's plan has no colour target (z-prepass, shadows): counted apart in the statistics.
    [[nodiscard]] virtual bool IsDepthOnly(const Vulkan::DrawPlan& plan) const = 0;
    /// After an operation: the plans go back to the consumer's pool.
    virtual void Release(Op& op) = 0;
    /// Every ~5 s after a batch: the consumer's statistics (it also resets the queue's interval counters).
    virtual void PublishIntervalStats(Queue& queue, u64 now_ns) = 0;
    /// After a batch was counted done: the consumer wakes whoever waits for the GPU to go idle.
    virtual void BatchFinished() = 0;
    /// The compute lane may be used: the asynchronous stream is running (queue 1 operations then record into a command buffer of their own).
    [[nodiscard]] virtual bool LaneUsable() const = 0;
};

class Queue {
public:
    size_t MaxQueued = 256; ///< TLG_RASTER_QUEUE

    /// Starts the raster thread (and the housekeeping thread). `verify`: cp_split_mode 2, every operation is waited for (synchronous).
    void Start(Consumer& consumer, bool verify);
    /// Stops the raster thread (it finishes what is queued first). Safe to call twice.
    void Stop();

    /// Queues an operation; `wait` blocks until the raster thread has executed it. Front thread only.
    void Push(Op&& op, bool wait);
    /// Queues `func` (a closure) for the raster thread. `writes_memory`: the closure writes guest memory that later pipeline lookups may read (descriptor tables, shader code),
    /// so the command processor thread must not plan a draw before it has run.
    void PushClosure(Common::UniqueFunction<void>&& func, bool wait, bool writes_memory, std::source_location site);
    /// Front thread, before the next PushClosure: the closure is no barrier for queue 1 operations (compute lane).
    void MarkNextPassable() noexcept {
        next_passable = true;
    }
    /// Front thread, before the next PushClosure: the closure runs as early as the raster thread can (TLG_EARLY_ANNOUNCE).
    void MarkNextHoist() noexcept {
        next_hoist = true;
    }
    /// Front thread, before the next PushClosure: the closure depends on the operation at `position` (compute lane).
    void SetNextDependency(u64 position) noexcept {
        next_dependency = position;
    }
    /// Front thread (compute lane): the queue is nearly full, so the next graphics push may block the whole front thread (and with it every compute queue).
    [[nodiscard]] bool NearlyFull() const noexcept {
        return lane_on && queued_hint.load(std::memory_order_relaxed) + 8 >= MaxQueued;
    }
    [[nodiscard]] u64 LanePushes() const noexcept {
        return lane_pushed.load(std::memory_order_relaxed);
    }
    /// The queue position of the newest operation pushed.
    [[nodiscard]] u64 LastPosition() const noexcept {
        return pushed.load(std::memory_order_acquire);
    }
    /// Blocks until the raster thread has executed everything queued so far.
    void Drain();
    /// Blocks until every queued closure that writes guest memory has run.
    void DrainWrites(u32 caller);
    [[nodiscard]] bool Idle() const noexcept {
        return pushed.load(std::memory_order_acquire) == done.load(std::memory_order_acquire);
    }
    /// Wakes the raster thread (it runs the consumer's callbacks).
    void Wake();

    // ---- state shared with the front thread and the consumer (all of it was the members of Liverpool::RasterSplit) ----
    std::jthread thread;
    std::jthread housekeeping; ///< Slow periodic work that must not stall the raster thread.
    std::mutex mutex;
    std::condition_variable_any work_cv; ///< Raster thread: work arrived.
    std::condition_variable_any done_cv; ///< Front: space in the queue or operations done.
    std::vector<Op> queue; // swapped with the raster thread's batch: no per-op allocation
    std::atomic<u64> pushed{};
    std::atomic<u64> done{};
    std::atomic<u32> pending_writes{}; ///< Queued closures with writes_memory.
    std::atomic<u64> write_done{};     ///< The queue position of the newest writes_memory closure the raster thread has finished (`done` moves in batches)
    /// What the front thread is doing (for the raster thread's idle statistics): 0 working,
    /// 1 waiting for the game to submit, 2 waiting for queued memory writes, 3 queue full.
    std::atomic<u8> front_state{0};
    std::array<std::atomic<u64>, 4> raster_idle_ns_by_front_state{};
    /// Idle time by the kind of the operation that ended it: draw, indirect draw, dispatch,
    /// indirect dispatch, closure (fence, write, flip, ...).
    std::array<std::atomic<u64>, 5> raster_idle_ns_by_next_op{};
    std::array<std::atomic<u64>, 5> raster_idle_count_by_next_op{};
    /// Statistics per RasterRun call site that queues memory-writing closures: how many, and how
    /// long the command processor waited for them before a pipeline lookup.
    struct WriteSite {
        std::atomic<u32> line{};
        std::atomic<u64> pushes{};
        std::atomic<u64> wait_ns{};
    };
    std::array<WriteSite, 24> write_sites;
    u32 last_write_line{}; ///< Command processor thread: line of the newest queued write closure.
    WriteSite* SiteFor(u32 line) {
        for (auto& site : write_sites) {
            u32 expected = 0;
            if (site.line.load(std::memory_order_relaxed) == line ||
                (site.line.compare_exchange_strong(expected, line) || expected == line)) {
                return &site;
            }
        }
        return &write_sites.back();
    }
    // Compute lane (TLG_ASC_LANE=1, documents/tlg/native plan/engine/compute-lane-design.md): operations of queue 1 run ahead of the graphics operations queued before them,
    // as far as the newest graphics closure (a barrier: memory writes, announce batches, uploads, ...). The raster thread checks between operations.
    bool lane_on{};
    bool lane_wide{};
    // Lane timing: how long queue 1 operations sat in the queue before they ran, and how long single raster operations took.
    std::atomic<u64> lane_age_ns[2]{};   ///< [0] hoisted out of the batch, [1] picked up while a batch ran
    std::atomic<u64> lane_age_n[2]{};
    std::atomic<u64> lane_age_max_ns[2]{};
    std::atomic<u64> lane_age_over_2ms[2]{};
    std::atomic<u64> raster_long_ops{};     ///< operations of 1 ms or more
    std::atomic<u64> raster_long_ops_ns{};
    std::atomic<u64> raster_longest_ns{};
    std::atomic<u32> queued_hint{0};   ///< Operations in `queue` (written under the mutex, read without).
    std::atomic<u64> lane_pushed{0};   ///< Queue 1 operations pushed while the lane was usable.
    bool next_passable{};   ///< front thread only
    bool next_hoist{};      ///< front thread only
    std::atomic<u32> hoist_waiting{0}; ///< Hoisted closures in `queue` the raster thread has not taken yet.
    std::atomic<u64> hoist_ops{}, hoist_picked{}, hoist_ns{}, hoist_distance{}; ///< run early / of which picked from the queue while a batch ran / their time / operations they skipped
    u64 next_dependency{};  ///< front thread only
    std::atomic<u32> lane_waiting{0}; ///< Queue 1 operations in `queue` that the raster thread has not taken yet.
    std::atomic<u64> lane_ops{};      ///< Queue 1 operations run through the lane logic.
    std::atomic<u64> lane_hoisted{};  ///< ... of which ahead of at least one operation pushed before them.
    std::atomic<u64> lane_distance{}; ///< Sum of the operations they skipped.
    std::atomic<u64> lane_pickups{};  ///< Taken from the newly pushed operations while a batch ran.
    std::atomic<u64> lane_bursts{};   ///< Groups of lane operations (one stream switch pair each).
    std::atomic<u64> lane_barrier_waits{}; ///< Boundaries where lane operations were waiting behind a graphics closure.
    std::mutex lane_block_mutex;
    std::map<std::tuple<const char*, u32, int>, std::pair<u64, u64>> long_op_sites; ///< (file, line, call kind) of raster operations of 1 ms or more: count, total ns (under lane_block_mutex)
    std::map<std::pair<const char*, u32>, u64> lane_block_sites; ///< The closure (queueing site) that held lane operations back, how often.
    // Statistics, logged every few seconds.
    u64 stat_ops{};
    u64 stat_calls{};
    u64 stat_write_drains{};
    std::atomic<u64> stat_front_blocked{}; ///< Pushes that found the queue full.
    std::atomic<u64> stat_raster_idle_ns{};
    u64 stat_draws{};
    // Raster thread wall time by kind of work (ns), logged every few seconds.
    u64 ns_draws{};
    u64 ns_depth_only_draws{}; ///< part of ns_draws spent in draws without a color target (z-prepass, shadows)
    u64 n_draws{};
    u64 n_depth_only_draws{};
    u64 ns_dispatches{};
    u64 ns_write_closures{};
    u64 ns_other_closures{};
    u64 ns_callbacks{};
    /// by the queue that pushed the operation (0 graphics, 1 compute queue 1, 2 queue 2, 3 queues 3 and up, 4 no queue task): wall time and count of dispatches and of closures (sizing of a lane thread, 2026-11-07)
    u64 ns_class_dispatch[5]{}, ns_class_closure[5]{}, n_class_dispatch[5]{}, n_class_closure[5]{}, ns_class_draw[5]{}, n_class_draw[5]{};
    u64 stat_last_log_ns{};
    bool verify{}; ///< cp_split_mode 2: every operation is waited for (synchronous)

private:
    void Run(std::stop_token stoken);
    Consumer* consumer{};
};

} // namespace Core::NativeRasterQueue
