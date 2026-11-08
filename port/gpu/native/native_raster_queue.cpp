// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <optional>
#include <utility>
#include "native/native_raster_queue.h"

#include <chrono>

#include "common/logging/log.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "native/native_debug_server.h"
#include "native/native_drain_audit.h"
#include "native/native_flight.h"
#include "native/native_prof.h"
#include "native/native_readback_policy.h"
#include "native/native_resource_audit.h"
#include "native/native_split_census.h"
#include "native/native_timeline.h"
#include "video_core/readback_stats.h"
#include "video_core/thread_usage.h"

namespace Core::NativeRasterQueue {

static u32 ClassBucket(u8 c) {
    return c == 0xff ? 4u : c >= 3 ? 3u : c;
}

void Queue::Start(Consumer& consumer_, bool verify_) {
    consumer = &consumer_;
    verify = verify_;
    if (const char* e = std::getenv("TLG_RASTER_QUEUE"); e != nullptr && std::atoi(e) >= 16) {
        MaxQueued = static_cast<size_t>(std::atoi(e));
    }
    lane_on = [] { const char* e = std::getenv("TLG_ASC_LANE"); return e != nullptr && (e[0] == '1' || e[0] == '2'); }();
    lane_wide = [] { const char* e = std::getenv("TLG_ASC_LANE"); return e != nullptr && e[0] == '2'; }(); // 2: closures marked passable are no barriers
    Core::DrainAudit::SetBackend(&done, &write_done, [this](u64 position) {
        front_state.store(2, std::memory_order_relaxed);
        {
            Core::FrameTimeline::StageScope drain{Core::FrameTimeline::StageFrontDrain};
            TLG_PROF_LEVEL("wait.front_drain_position (front: queued memory writes executed up to a position)", 1);
            FLIGHT_SCOPE("front: waiting for queued memory writes to execute");
            std::unique_lock lk{mutex};
            done_cv.wait(lk, [&] { return done.load(std::memory_order_acquire) >= position || write_done.load(std::memory_order_acquire) >= position; });
        }
        front_state.store(0, std::memory_order_relaxed);
    });
    thread = std::jthread{std::bind_front(&Queue::Run, this)};
    housekeeping = std::jthread{[](std::stop_token stoken) {
        Common::SetCurrentThreadName("shadPS4:Housekeeping");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        std::mutex mutex;
        std::condition_variable_any cv;
        std::unique_lock lk{mutex};
        while (!stoken.stop_requested()) {
            if (cv.wait_for(lk, stoken, std::chrono::seconds(5), [] { return false; })) {
                break;
            }
            const auto start = std::chrono::steady_clock::now();
            if (const auto usage = VideoCore::ThreadUsageReport(); !usage.empty()) {
                LOG_INFO(Render, "Thread CPU use (100% = one core): {}", usage);
            }
            const auto middle = std::chrono::steady_clock::now();
            VideoCore::ReapplyCoreIsolation(); // threads created since (guest threads)
            const auto end = std::chrono::steady_clock::now();
            Core::DebugServer::RecordPeriodic(
                "housekeeping_thread_cpu_report",
                std::chrono::duration<double, std::milli>(middle - start).count());
            Core::DebugServer::RecordPeriodic(
                "housekeeping_core_isolation",
                std::chrono::duration<double, std::milli>(end - middle).count());
        }
    }};
    LOG_INFO(Render, "Command processor split: raster thread started{}",
             verify ? " (synchronous)" : "");
}

void Queue::Stop() {
    if (!thread.joinable()) {
        return;
    }
    thread.request_stop();
    work_cv.notify_all();
    thread.join();
}

void Queue::Wake() {
    std::scoped_lock lk{mutex};
    work_cv.notify_all();
}

void Queue::Drain() {
    const u64 target = pushed.load(std::memory_order_acquire);
    if (done.load(std::memory_order_acquire) >= target) {
        return;
    }
    FLIGHT_SCOPE("front: waiting for the raster thread to reach a position");
    std::unique_lock lk{mutex};
    done_cv.wait(lk, [&] { return done.load(std::memory_order_acquire) >= target; });
}

void Queue::Push(Op&& op, bool wait) {
    op.audit_class = Core::ResourceAudit::front_class;
    op.pushed_ns = VideoCore::ReadbackStats::NowNs();
    const bool audit_write = op.writes_memory;
    const u32 audit_line = op.write_line;
    if (op.has_call) {
        Core::DrainAudit::LookupDone(); // N6: the planning of this operation is done
    } else if (!op.writes_memory) {
        Core::DrainAudit::ClearNextWrite();
    }
    if (op.writes_memory) {
        pending_writes.fetch_add(1, std::memory_order_acq_rel);
        last_write_line = op.write_line;
        SiteFor(op.write_line)->pushes.fetch_add(1, std::memory_order_relaxed);
    }
    ++stat_ops;
    u64 target;
    const bool lane_op = lane_on && op.audit_class == 1 && consumer->LaneUsable();
    // A lane operation is not held up by a full graphics queue until the queue holds twice as much.
    const size_t limit = lane_op ? MaxQueued * 2 : MaxQueued;
    {
        std::unique_lock lk{mutex};
        if (queue.size() >= limit) {
            stat_front_blocked.fetch_add(1, std::memory_order_relaxed);
        }
        if (queue.size() >= limit) {
            front_state.store(3, std::memory_order_relaxed);
        }
        {
            Core::FrameTimeline::StageScope queue_full{Core::FrameTimeline::StageFrontQueueFull};
            TLG_PROF_LEVEL("wait.front_queue_full (front: the raster queue has no room)", 1);
            FLIGHT_SCOPE("front: the raster queue has no room");
            done_cv.wait(lk, [&] { return queue.size() < limit; });
        }
        front_state.store(0, std::memory_order_relaxed);
        op.position = pushed.load(std::memory_order_relaxed) + 1;
        queue.push_back(std::move(op));
        queued_hint.store(static_cast<u32>(queue.size()), std::memory_order_relaxed);
        if (queue.back().hoist) {
            hoist_waiting.fetch_add(1, std::memory_order_acq_rel);
        }
        if (lane_op) {
            lane_waiting.fetch_add(1, std::memory_order_acq_rel);
            lane_pushed.fetch_add(1, std::memory_order_relaxed);
        }
        target = pushed.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (audit_write) {
            Core::DrainAudit::WriteQueued(target, audit_line);
        }
        work_cv.notify_all();
        if (wait || verify) {
            done_cv.wait(lk, [&] { return done.load(std::memory_order_acquire) >= target; });
        }
    }
}

void Queue::PushClosure(Common::UniqueFunction<void>&& func, bool wait, bool writes_memory, std::source_location site) {
    Op op;
    op.closure = std::move(func);
    op.writes_memory = writes_memory;
    op.write_line = site.line();
    op.write_file = site.file_name();
    for (const char* p = site.file_name(); *p != 0; ++p) { // the file name without the directories
        if (*p == '/' || *p == '\\') {
            op.write_file = p + 1;
        }
    }
    op.lane_passable = std::exchange(next_passable, false) && lane_wide;
    op.depends_on = std::exchange(next_dependency, 0);
    op.hoist = std::exchange(next_hoist, false);
    Push(std::move(op), wait);
}

void Queue::DrainWrites(u32 caller) {
    if (pending_writes.load(std::memory_order_acquire) == 0) {
        Core::DrainAudit::DrainNothingPending();
        return;
    }
    Core::DrainAudit::DrainBegin(done.load(std::memory_order_acquire), caller);
    ++stat_write_drains;
    front_state.store(2, std::memory_order_relaxed);
    const u64 wait_start = VideoCore::ReadbackStats::NowNs();
    {
        Core::FrameTimeline::StageScope drain{Core::FrameTimeline::StageFrontDrain};
        TLG_PROF_LEVEL("wait.front_drain_writes (front: all queued memory writing closures executed)", 1);
        FLIGHT_SCOPE("front: draining queued memory writes");
        std::unique_lock lk{mutex};
        done_cv.wait(lk, [&] { return pending_writes.load(std::memory_order_acquire) == 0; });
    }
    front_state.store(0, std::memory_order_relaxed);
    Core::DrainAudit::DrainWaited(VideoCore::ReadbackStats::NowNs() - wait_start);
    SiteFor(last_write_line)
        ->wait_ns.fetch_add(VideoCore::ReadbackStats::NowNs() - wait_start,
                            std::memory_order_relaxed);
}

void Queue::Run(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuRaster");
    if (const u32 priority = EmulatorSettings.GetCpThreadPriority(); priority >= 2) {
        Common::SetCurrentThreadPriority(static_cast<Common::ThreadPriority>(priority));
    }
    consumer->ThreadStarted();
    s32 applied_core = -1;
    VideoCore::ReadbackStats::MarkCpThread();
    auto& sp = *this;
    std::vector<Op> batch;
    while (true) {
        // cp_raster_core can be changed live (debug server /set?name=raster_core&value=N).
        if (const s32 wanted = VideoCore::ReadbackPolicy::Get().raster_core.load();
            wanted != applied_core) {
            if (applied_core >= 0) {
                VideoCore::ClearCoreIsolation();
            }
            if (wanted >= 0) {
                VideoCore::PinThreadAndIsolateCore(static_cast<u32>(wanted));
            }
            applied_core = wanted;
        }
        consumer->Idle();
        {
            const u64 idle_start = VideoCore::ReadbackStats::NowNs();
            const u8 front_state_at_idle = sp.front_state.load(std::memory_order_relaxed);
            TLG_PROF_LEVEL("wait.raster_idle (raster: no work queued)", 1);
            FLIGHT_SCOPE("raster: no work queued");
            std::unique_lock lk{sp.mutex};
            Common::CondvarWait(sp.work_cv, lk, stoken,
                                [&] { return !sp.queue.empty() || consumer->CallbacksPending(); });
            {
                const u64 idle = VideoCore::ReadbackStats::NowNs() - idle_start;
                if (idle >= 20'000 && Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
                    Core::FrameTimeline::Record(Core::FrameTimeline::StageSpan, idle | (u64(Core::FrameTimeline::StageRasterIdle) << 56), Core::FrameTimeline::Now(),
                                                sp.front_state.load(std::memory_order_relaxed));
                }
                sp.stat_raster_idle_ns.fetch_add(idle, std::memory_order_relaxed);
                sp.raster_idle_ns_by_front_state[front_state_at_idle & 3].fetch_add(
                    idle, std::memory_order_relaxed);
            }
            if (stoken.stop_requested() && sp.queue.empty() && !consumer->CallbacksPending()) {
                return;
            }
            batch.swap(sp.queue);
            sp.queued_hint.store(0, std::memory_order_relaxed);
            sp.hoist_waiting.store(0, std::memory_order_release);
            sp.lane_waiting.store(0, std::memory_order_release); // every queued lane operation is in the batch now
            if (!batch.empty()) {
                const auto& first = batch.front();
                const size_t kind = first.has_call ? static_cast<size_t>(first.call.kind) : 4;
                sp.raster_idle_ns_by_next_op[kind].fetch_add(
                    VideoCore::ReadbackStats::NowNs() - idle_start, std::memory_order_relaxed);
                sp.raster_idle_count_by_next_op[kind].fetch_add(1, std::memory_order_relaxed);
            }
            sp.done_cv.notify_all(); // The queue has room again.
        }
        FLIGHT_SCOPE("raster: running queued work");
        // Capture: consecutive operations of one kind are one span (a run of draws, of dispatches, of closures), written when the kind changes.
        u32 run_kind = 0;
        u64 run_start = 0, run_count = 0;
        const bool capturing = Core::FrameTimeline::capture_active.load(std::memory_order_relaxed);
        const auto end_run = [&] {
            if (run_kind != 0 && run_count != 0) {
                const u64 now = Core::FrameTimeline::Now();
                Core::FrameTimeline::Record(Core::FrameTimeline::StageSpan, (now - run_start) | (u64(Core::FrameTimeline::StageRasterOp) << 56), now,
                                            run_kind | (run_count << 8));
            }
            run_kind = 0;
            run_count = 0;
        };
        const auto run_op = [&](Op& op) {
            Core::ResourceAudit::raster_class = op.audit_class;
            // Diagnostic (side stream hunt, TLG_ASC_SIDE_ONLY): 1 = only the dispatches of queue 1 run on the side stream, 2 = only its closures (copies, fills, fences)
            static const int side_only = [] { const char* e = std::getenv("TLG_ASC_SIDE_ONLY"); return e != nullptr ? std::atoi(e) : 0; }();
            consumer->SelectStream(op.audit_class == 1 && (side_only == 0 || (side_only == 1) == op.has_call));
            if (capturing) {
                // run key: kind (bits 0-3) | the queue that pushed the operation (bits 4-7, 15 = none), so a run never mixes queues
                const u32 kind = (op.has_call ? (op.call.kind == Call::Kind::Draw || op.call.kind == Call::Kind::DrawIndirect ? 1u : 2u) : 3u) |
                                 (u32(op.audit_class & 0xf) << 4);
                if (kind != run_kind) {
                    end_run();
                    run_kind = kind;
                    run_start = Core::FrameTimeline::Now();
                }
                ++run_count;
            }
            // Callbacks sent by other threads (fault readbacks, fences) run between operations,
            // like they ran between packets before.
            if (consumer->CallbacksPending()) {
                TLG_PROF_NAMED_LEVEL(prof_callbacks, "raster.callbacks", 1);
                const u64 callback_start = VideoCore::ReadbackStats::NowNs();
                consumer->RunCallbacks();
                sp.ns_callbacks += VideoCore::ReadbackStats::NowNs() - callback_start;
            }
            Core::SplitCensus::Begin(); // plan S0: no-op unless TLG_SPLIT_CENSUS=1 (the mask is only read when it is on)
            const u64 op_start = VideoCore::ReadbackStats::NowNs();
            if (op.has_call) {
                if (const u32 delay = VideoCore::ReadbackPolicy::Get().raster_delay_ns.load(std::memory_order_relaxed);
                    delay != 0 && (op.call.kind == Call::Kind::Draw || op.call.kind == Call::Kind::DrawIndirect)) {
                    const u64 until = Core::FrameTimeline::Now() + delay;
                    while (Core::FrameTimeline::Now() < until) {
                    }
                }
                {
                    static Core::Prof::Site site_draw{"raster.op.draw", 1}, site_draw_indirect{"raster.op.draw_indirect", 1}, site_dispatch{"raster.op.dispatch", 1},
                        site_dispatch_indirect{"raster.op.dispatch_indirect", 1};
                    Core::Prof::Site& op_site = op.call.kind == Call::Kind::Draw           ? site_draw
                                                : op.call.kind == Call::Kind::DrawIndirect ? site_draw_indirect
                                                : op.call.kind == Call::Kind::DispatchDirect ? site_dispatch
                                                                                             : site_dispatch_indirect;
                    const Core::Prof::Scope prof_op{op_site};
                    consumer->RunCall(op);
                }
                const u64 op_ns = VideoCore::ReadbackStats::NowNs() - op_start;
                if (op.call.kind == Call::Kind::Draw ||
                    op.call.kind == Call::Kind::DrawIndirect) {
                    sp.ns_draws += op_ns;
                    ++sp.n_draws;
                    sp.ns_class_draw[ClassBucket(op.audit_class)] += op_ns;
                    ++sp.n_class_draw[ClassBucket(op.audit_class)];
                    const bool depth_only = op.draw_plan && consumer->IsDepthOnly(*op.draw_plan);
                    if (depth_only) {
                        sp.ns_depth_only_draws += op_ns;
                        ++sp.n_depth_only_draws;
                    }
                    if (Core::SplitCensus::On()) {
                        Core::SplitCensus::Record(depth_only ? 1 : 0, ClassBucket(op.audit_class), nullptr, 0, Core::SplitCensus::Current(), op_ns);
                    }
                } else {
                    sp.ns_dispatches += op_ns;
                    sp.ns_class_dispatch[ClassBucket(op.audit_class)] += op_ns;
                    ++sp.n_class_dispatch[ClassBucket(op.audit_class)];
                    if (Core::SplitCensus::On()) {
                        Core::SplitCensus::Record(2, ClassBucket(op.audit_class), nullptr, 0, Core::SplitCensus::Current(), op_ns);
                    }
                }
            } else if (op.closure) {
                consumer->RunClosure(op);
                (op.writes_memory ? sp.ns_write_closures : sp.ns_other_closures) +=
                    VideoCore::ReadbackStats::NowNs() - op_start;
                sp.ns_class_closure[ClassBucket(op.audit_class)] += VideoCore::ReadbackStats::NowNs() - op_start;
                ++sp.n_class_closure[ClassBucket(op.audit_class)];
                if (Core::SplitCensus::On()) {
                    if (op.writes_memory) {
                        Core::SplitCensus::Touch(Core::SplitCensus::GuestWrite);
                    }
                    Core::SplitCensus::Record(3, ClassBucket(op.audit_class), op.write_file, op.write_line, Core::SplitCensus::Current(),
                                              VideoCore::ReadbackStats::NowNs() - op_start);
                }
                if (op.writes_memory) {
                    sp.write_done.store(op.position, std::memory_order_release);
                    sp.pending_writes.fetch_sub(1, std::memory_order_acq_rel);
                    std::scoped_lock lk{sp.mutex};
                    sp.done_cv.notify_all();
                }
            }
            op.closure = {};
            consumer->Release(op);
        };
        // Compute lane (documents/tlg/native plan/engine/compute-lane-design.md): a queue 1 operation runs as soon as (a) every closure that is a barrier ahead of it ran (announce batches, uploads,
        // frame events, ...: closures marked passable are none) and (b) the one operation it depends on ran (a wait that was passed on a recorded fence names the fence's closure), and after the
        // queue 1 operations before it. `lane_ready[j]` is the batch index the main loop must have reached for lane operation j.
        const bool lane_active = sp.lane_on && consumer->LaneUsable();
        const auto note_age = [&](const Op& o, int path) {
            const u64 age = VideoCore::ReadbackStats::NowNs() - o.pushed_ns;
            sp.lane_age_ns[path].fetch_add(age, std::memory_order_relaxed);
            sp.lane_age_n[path].fetch_add(1, std::memory_order_relaxed);
            if (age > 2'000'000) {
                sp.lane_age_over_2ms[path].fetch_add(1, std::memory_order_relaxed);
            }
            u64 mx = sp.lane_age_max_ns[path].load(std::memory_order_relaxed);
            while (age > mx && !sp.lane_age_max_ns[path].compare_exchange_weak(mx, age)) {
            }
        };
        std::vector<u32> lane_idx, lane_ready, barrier_idx;
        if (lane_active) {
            u32 after_barrier = 0, prev_ready = 0;
            const auto index_of = [&](u64 position) -> std::optional<u32> {
                const auto it = std::lower_bound(batch.begin(), batch.end(), position, [](const Op& o, u64 p) { return o.position < p; });
                if (it == batch.end() || it->position != position) {
                    return std::nullopt;
                }
                return static_cast<u32>(it - batch.begin());
            };
            for (u32 k = 0; k < batch.size(); ++k) {
                if (batch[k].audit_class == 1) {
                    u32 ready = std::max(after_barrier, prev_ready);
                    if (const u64 dep = batch[k].depends_on; dep == ~u64{0}) {
                        ready = std::max(ready, k);
                    } else if (dep != 0) {
                        if (const auto j = index_of(dep); j && batch[*j].audit_class != 1) {
                            ready = std::max(ready, *j + 1); // not found: it ran already (an earlier batch, or a queue 1 operation of this one)
                        }
                    }
                    lane_idx.push_back(k);
                    lane_ready.push_back(ready);
                    prev_ready = ready;
                } else if (!batch[k].has_call && !batch[k].lane_passable && !batch[k].hoist) {
                    barrier_idx.push_back(k);
                    after_barrier = k + 1;
                }
            }
        }
        size_t lane_next = 0, barrier_next = 0;
        std::vector<Op> picked;
        // Early announce: hoisted closures run first
        const auto run_hoisted = [&](Op& o, u64 skipped) {
            const u64 t0 = VideoCore::ReadbackStats::NowNs();
            sp.hoist_ops.fetch_add(1, std::memory_order_relaxed);
            sp.hoist_distance.fetch_add(skipped, std::memory_order_relaxed);
            run_op(o);
            sp.hoist_ns.fetch_add(VideoCore::ReadbackStats::NowNs() - t0, std::memory_order_relaxed);
        };
        bool hoisted_from_batch = false;
        std::vector<Op> picked_hoist;
        for (u32 i = 0; i < batch.size(); ++i) {
            if (sp.hoist_waiting.load(std::memory_order_acquire) != 0) { // hoisted closures pushed while this batch runs
                picked_hoist.clear();
                {
                    std::scoped_lock lk{sp.mutex};
                    size_t keep = 0;
                    for (size_t k = 0; k < sp.queue.size(); ++k) {
                        Op& o = sp.queue[k];
                        if (o.hoist) {
                            picked_hoist.push_back(std::move(o));
                        } else {
                            if (keep != k) {
                                sp.queue[keep] = std::move(o);
                            }
                            ++keep;
                        }
                    }
                    sp.queue.resize(keep);
                    sp.queued_hint.store(static_cast<u32>(keep), std::memory_order_relaxed);
                    sp.hoist_waiting.fetch_sub(static_cast<u32>(picked_hoist.size()), std::memory_order_acq_rel);
                }
                if (!picked_hoist.empty()) {
                    sp.done_cv.notify_all();
                    sp.hoist_picked.fetch_add(picked_hoist.size(), std::memory_order_relaxed);
                    for (Op& o : picked_hoist) {
                        run_hoisted(o, batch.size() - i);
                    }
                }
            }
            if (lane_active) {
                while (barrier_next < barrier_idx.size() && barrier_idx[barrier_next] < i) {
                    ++barrier_next;
                }
                u64 burst = 0;
                while (lane_next < lane_idx.size() && lane_ready[lane_next] <= i) {
                    const u32 k = lane_idx[lane_next++];
                    if (k > i) {
                        sp.lane_hoisted.fetch_add(1, std::memory_order_relaxed);
                        sp.lane_distance.fetch_add(k - i, std::memory_order_relaxed);
                    }
                    batch[k].executed = true;
                    note_age(batch[k], 0);
                    sp.lane_ops.fetch_add(1, std::memory_order_relaxed);
                    ++burst;
                    run_op(batch[k]);
                }
                if (lane_next < lane_idx.size()) {
                    sp.lane_barrier_waits.fetch_add(1, std::memory_order_relaxed); // lane operations wait for the operation at lane_ready - 1
                    if (const u32 r = lane_ready[lane_next]; r > 0 && r - 1 < batch.size()) {
                        std::scoped_lock lk{sp.lane_block_mutex};
                        ++sp.lane_block_sites[{batch[r - 1].write_file, batch[r - 1].write_line}];
                    }
                }
                // Lane operations pushed while this batch runs: taken when every queue 1 operation of the batch ran, no barrier is left in it, none precedes them in the new pushes, and
                // the operation they depend on ran already.
                if (barrier_next >= barrier_idx.size() && lane_next >= lane_idx.size() && sp.lane_waiting.load(std::memory_order_acquire) != 0) {
                    picked.clear();
                    const u64 next_position = batch[i].position; // the first operation of the batch not run yet
                    {
                        std::scoped_lock lk{sp.mutex};
                        size_t keep = 0;
                        bool blocked = false;
                        for (size_t k = 0; k < sp.queue.size(); ++k) {
                            Op& o = sp.queue[k];
                            const bool dep_ran = o.depends_on == 0 || (o.depends_on != ~u64{0} && o.depends_on < next_position);
                            if (!blocked && o.audit_class == 1 && dep_ran) {
                                picked.push_back(std::move(o));
                            } else {
                                if (o.audit_class == 1 || (!o.has_call && !o.lane_passable && !o.hoist)) {
                                    blocked = true; // a barrier, or a queue 1 operation that has to wait (the order of queue 1 stays)
                                }
                                if (keep != k) {
                                    sp.queue[keep] = std::move(o);
                                }
                                ++keep;
                            }
                        }
                        sp.queue.resize(keep);
                        sp.queued_hint.store(static_cast<u32>(keep), std::memory_order_relaxed);
                        sp.lane_waiting.fetch_sub(static_cast<u32>(picked.size()), std::memory_order_acq_rel);
                    }
                    if (!picked.empty()) {
                        sp.done_cv.notify_all(); // the queue has room again
                        sp.lane_pickups.fetch_add(picked.size(), std::memory_order_relaxed);
                        sp.lane_hoisted.fetch_add(picked.size(), std::memory_order_relaxed);
                        sp.lane_ops.fetch_add(picked.size(), std::memory_order_relaxed);
                        burst += picked.size();
                        for (Op& o : picked) {
                            note_age(o, 1);
                            run_op(o);
                        }
                    }
                }
                if (burst != 0) {
                    sp.lane_bursts.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!hoisted_from_batch) { // after the first lane operations (they are ready now), before the first graphics operation
                hoisted_from_batch = true;
                for (u32 k = 0; k < batch.size(); ++k) {
                    if (batch[k].hoist && !batch[k].executed) {
                        batch[k].executed = true;
                        run_hoisted(batch[k], k);
                    }
                }
            }
            if (!batch[i].executed) {
                if (!lane_active) {
                    run_op(batch[i]);
                    continue;
                }
                const u64 t0 = VideoCore::ReadbackStats::NowNs();
                run_op(batch[i]);
                if (const u64 d = VideoCore::ReadbackStats::NowNs() - t0; d >= 1'000'000) {
                    {
                        std::scoped_lock lk{sp.lane_block_mutex};
                        auto& entry = sp.long_op_sites[{batch[i].write_file, batch[i].write_line, batch[i].has_call ? static_cast<int>(batch[i].call.kind) + 1 : 0}];
                        ++entry.first;
                        entry.second += d;
                    }
                    sp.raster_long_ops.fetch_add(1, std::memory_order_relaxed);
                    sp.raster_long_ops_ns.fetch_add(d, std::memory_order_relaxed);
                    u64 mx = sp.raster_longest_ns.load(std::memory_order_relaxed);
                    while (d > mx && !sp.raster_longest_ns.compare_exchange_weak(mx, d)) {
                    }
                }
            }
        }
        end_run();
        Core::ResourceAudit::raster_class = Core::ResourceAudit::NoClass;
        consumer->SelectStream(false);
        if (consumer->CallbacksPending()) {
            consumer->RunCallbacks();
        }
        batch.clear();
        if (const u64 now = VideoCore::ReadbackStats::NowNs();
            now - sp.stat_last_log_ns > 5'000'000'000ULL) {
            consumer->PublishIntervalStats(sp, now);
        }
        {
            // done is the queue position up to which everything has run: lane operations taken out of order leave the queue before the graphics operations pushed ahead of
            // them, so it is the position of the first operation still queued, minus one.
            std::scoped_lock lk{sp.mutex};
            sp.done.store(sp.queue.empty() ? sp.pushed.load(std::memory_order_acquire) : sp.queue.front().position - 1, std::memory_order_release);
            sp.done_cv.notify_all();
        }
        if (Idle()) {
            consumer->BatchFinished();
        }
    }
}

} // namespace Core::NativeRasterQueue
