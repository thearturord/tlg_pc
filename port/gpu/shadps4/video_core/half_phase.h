// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <cstdlib>
#include <deque>
#include <vector>

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
    static inline thread_local std::deque<Common::UniqueFunction<void>> jobs;
    static inline thread_local bool executing{false};
    static inline thread_local bool disabled{false};
    static inline std::atomic<u64> posted{0}, ran{0}, max_depth{0};
    static inline std::atomic<u64> drains[ReasonCount]{}, drained_jobs[ReasonCount]{};
    static u32 Lag() noexcept {
        static const u32 lag = [] {
            if (ResolveCommands::SplitMode() != 1) {
                return 0u;
            }
            const char* e = std::getenv("TLG_SPLIT_QUEUE");
            return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u;
        }();
        return lag;
    }
    static bool Enabled() noexcept {
        return Lag() != 0 && !disabled;
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
        jobs.push_back(std::move(work));
        posted.fetch_add(1, std::memory_order_relaxed);
        if (jobs.size() > max_depth.load(std::memory_order_relaxed)) {
            max_depth.store(jobs.size(), std::memory_order_relaxed);
        }
        while (jobs.size() > Lag()) {
            RunOne();
        }
    }
    /// Runs everything that waits (nothing, from inside a running job: everything older is done already).
    static void Drain(Reason reason) {
        if (executing || jobs.empty()) {
            return;
        }
        drains[reason].fetch_add(1, std::memory_order_relaxed);
        drained_jobs[reason].fetch_add(jobs.size(), std::memory_order_relaxed);
        while (!jobs.empty()) {
            RunOne();
        }
    }
    static void Guard(Reason reason) {
        if (!jobs.empty() && !executing) {
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
