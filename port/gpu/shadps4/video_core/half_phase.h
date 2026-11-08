// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/thread.h"
#include "common/types.h"
#include "common/unique_function.h"

namespace VideoCore {

/// Plan S step P3d: which half of the split bind code the calling thread is in. The diagnostics (`Core::NativeOcclusion::NoteCrossing`) use it to count record-owned work
/// (commands, layout transitions, backing changes) that happens inside a resolve half, and `ResolveCommands` uses it to hold such work back.
struct HalfPhase {
    enum : u32 { None = 0, Resolve = 1, Record = 2 };
    static inline thread_local u32 current{None};
    struct Scope {
        u32 previous;
        explicit Scope(u32 half) noexcept : previous{current} { current = half; }
        ~Scope() { current = previous; }
    };
};

/// The record side work a resolve half decides on (an image upload and its detile, the copies of a block placed, the copy of an arena synchronization). With TLG_SPLIT_BIND=1
/// a resolve half queues it here, in the order it was decided, and the record half runs the queue before it records anything of its own. Anything else that records
/// from inside a resolve half (`Run` is called by the first command, transit or backing swap it makes) drains the queue first, so the order of the commands is the one the
/// old code had. Everything the work reads (images, buffers) must still exist when it runs: whoever frees one (`TextureCache::FreeImage`) drains the queue first.
struct ResolveCommands {
    static inline thread_local std::vector<Common::UniqueFunction<void>> pending;
    static u32 SplitMode() noexcept {
        static const u32 mode = [] {
            const char* e = std::getenv("TLG_SPLIT_BIND");
            return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u;
        }();
        return mode;
    }
    static bool Deferring() noexcept {
        return HalfPhase::current == HalfPhase::Resolve && SplitMode() == 1;
    }
    static void Defer(Common::UniqueFunction<void>&& work) {
        pending.push_back(std::move(work));
    }
    static inline void Run(); // defined below `RecordQueue`: draws that wait are recorded before the work queued after them
};

/// Plan S step P3e.4b: the queue between the resolve half and the record half of draws. `TLG_SPLIT_QUEUE=N` (with TLG_SPLIT_BIND=1) lets up to N draws wait in it; a draw is
/// recorded when N newer ones are behind it, or at the next drain point. This first version runs the record half on the same thread, later (the lag makes every ordering
/// and lifetime mistake show up deterministically): nothing else changes between the two halves, so a wrong picture or a validation error means the resolve half did
/// something to the world that the record half of an earlier draw relies on. The drain points are the things that are not draws (closures, dispatches, copies, fills)
/// and a `Guard` in every API the record half owns (scheduler, runtime, the free of an image): called from outside a running job with draws waiting, it drains first.
struct RecordQueue {
    enum Reason : u32 { NonDraw, Closure, Callbacks, Idle, Commands, Transit, Backing, SchedulerApi, SchedulerDefer, FreeImage, Diagnostics, Other, ReasonCount };
    static constexpr const char* names[ReasonCount] = {"non-draw op", "closure", "callbacks", "idle", "commands", "transit", "backing swap", "scheduler api", "scheduler defer", "free image", "diagnostics", "other"};
    static inline thread_local std::deque<Common::UniqueFunction<void>> jobs; ///< same-thread mode
    static inline thread_local bool executing{false};                         ///< true while a queued job runs (always on the recorder thread)
    static inline thread_local bool disabled{false};
    static inline std::atomic<u64> posted{0}, ran{0}, max_depth{0};
    static inline std::atomic<u64> drains[ReasonCount]{}, drained_jobs[ReasonCount]{}, drain_wait_ns[ReasonCount]{}, post_wait_ns{0}, worker_idle_ns{0}, worker_busy_ns{0};

    /// `TLG_SPLIT_THREAD=1` (with TLG_SPLIT_BIND=1): the record half runs on its own thread (`shadPS4:GpuRecord`), the queue is shared and bounded by `Lag()`.
    static bool Threaded() noexcept {
        static const bool on = [] {
            const char* e = std::getenv("TLG_SPLIT_THREAD");
            return ResolveCommands::SplitMode() == 1 && e != nullptr && *e == '1';
        }();
        return on;
    }
    static u32 Lag() noexcept {
        static const u32 lag = [] {
            if (ResolveCommands::SplitMode() != 1) {
                return 0u;
            }
            const char* e = std::getenv("TLG_SPLIT_QUEUE");
            if (e != nullptr) {
                return static_cast<u32>(std::atoi(e));
            }
            return Threaded() ? 64u : 0u;
        }();
        return lag;
    }
    static bool Enabled() noexcept {
        return Lag() != 0 && !disabled;
    }

    // ---- thread mode
    static inline std::mutex mutex;
    static inline std::condition_variable work_cv, done_cv;
    static inline std::deque<Common::UniqueFunction<void>> shared;
    static inline std::atomic<u64> done{0};
    static inline std::atomic<bool> started{false};
    static inline std::thread::id resolver_id{};
    /// Draws wait (posted and not yet run). Meaningful on the resolver thread; elsewhere it only says "something is in flight".
    static bool Pending() noexcept {
        if (Threaded()) {
            return posted.load(std::memory_order_relaxed) != done.load(std::memory_order_acquire);
        }
        return !jobs.empty();
    }
    static void WorkerMain() {
        Common::SetCurrentThreadName("shadPS4:GpuRecord");
        executing = true;
        while (true) {
            Common::UniqueFunction<void> work;
            {
                const auto idle_start = std::chrono::steady_clock::now();
                std::unique_lock lock{mutex};
                work_cv.wait(lock, [] { return !shared.empty(); });
                worker_idle_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - idle_start).count(), std::memory_order_relaxed);
                work = std::move(shared.front());
                shared.pop_front();
            }
            const auto busy_start = std::chrono::steady_clock::now();
            work();
            work = {};
            worker_busy_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - busy_start).count(), std::memory_order_relaxed);
            ran.fetch_add(1, std::memory_order_relaxed);
            {
                std::scoped_lock lock{mutex};
                done.fetch_add(1, std::memory_order_release);
            }
            done_cv.notify_all();
        }
    }

    static void RunOne() {
        auto work = std::move(jobs.front());
        jobs.pop_front();
        executing = true;
        work();
        executing = false;
        ran.fetch_add(1, std::memory_order_relaxed);
    }
    static void Post(Common::UniqueFunction<void>&& work) {
        if (Threaded()) {
            if (!started.exchange(true)) {
                resolver_id = std::this_thread::get_id();
                std::thread{WorkerMain}.detach();
            }
            const auto wait_start = std::chrono::steady_clock::now();
            std::unique_lock lock{mutex};
            done_cv.wait(lock, [] { return posted.load(std::memory_order_relaxed) - done.load(std::memory_order_relaxed) < Lag(); });
            post_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
            shared.push_back(std::move(work));
            const u64 depth = posted.fetch_add(1, std::memory_order_relaxed) + 1 - done.load(std::memory_order_relaxed);
            if (depth > max_depth.load(std::memory_order_relaxed)) {
                max_depth.store(depth, std::memory_order_relaxed);
            }
            lock.unlock();
            work_cv.notify_one();
            return;
        }
        jobs.push_back(std::move(work));
        posted.fetch_add(1, std::memory_order_relaxed);
        if (jobs.size() > max_depth.load(std::memory_order_relaxed)) {
            max_depth.store(jobs.size(), std::memory_order_relaxed);
        }
        while (jobs.size() > Lag()) {
            RunOne();
        }
    }
    /// Runs (or waits for) everything that waits. Nothing from inside a running job (everything older is done already) and nothing from a thread that is not the resolver.
    static void Drain(Reason reason) {
        if (executing) {
            return;
        }
        if (Threaded()) {
            if (!started.load(std::memory_order_acquire) || std::this_thread::get_id() != resolver_id) {
                return;
            }
            const u64 target = posted.load(std::memory_order_relaxed);
            if (done.load(std::memory_order_acquire) == target) {
                return;
            }
            drains[reason].fetch_add(1, std::memory_order_relaxed);
            drained_jobs[reason].fetch_add(target - done.load(std::memory_order_relaxed), std::memory_order_relaxed);
            const auto wait_start = std::chrono::steady_clock::now();
            std::unique_lock lock{mutex};
            done_cv.wait(lock, [target] { return done.load(std::memory_order_relaxed) >= target; });
            drain_wait_ns[reason].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_start).count(), std::memory_order_relaxed);
            return;
        }
        if (jobs.empty()) {
            return;
        }
        drains[reason].fetch_add(1, std::memory_order_relaxed);
        drained_jobs[reason].fetch_add(jobs.size(), std::memory_order_relaxed);
        while (!jobs.empty()) {
            RunOne();
        }
    }
    static void Guard(Reason reason) {
        if (!executing && Pending()) {
            Drain(reason);
        }
    }
};

inline void ResolveCommands::Run() {
    if (pending.empty() || RecordQueue::executing) {
        return; // (inside a queued draw: what is in `pending` belongs to draws resolved after it; its own work is the job's `pre`)
    }
    RecordQueue::Guard(RecordQueue::Other);
    HalfPhase::Scope scope{HalfPhase::Record};
    auto list = std::move(pending);
    pending.clear();
    for (auto& work : list) {
        work();
    }
}

} // namespace VideoCore
