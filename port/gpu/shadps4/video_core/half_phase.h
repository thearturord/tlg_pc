// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdlib>
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
    static void Run() {
        if (pending.empty()) {
            return;
        }
        HalfPhase::Scope scope{HalfPhase::Record};
        auto list = std::move(pending);
        pending.clear();
        for (auto& work : list) {
            work();
        }
    }
};

} // namespace VideoCore
