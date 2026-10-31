// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_frame_events.h"

#include <array>
#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "common/assert.h"

namespace Core::NativeFrameEvents {

namespace {

struct Context {
    std::mutex lock;
    std::unordered_map<void*, Handler> persistent;
    std::queue<Handler> once;
    std::atomic<u64> signals{};          ///< the event happened
    std::atomic<u64> persistent_runs{};  ///< persistent subscribers run
    std::atomic<u64> once_runs{};        ///< one-time subscribers run
    std::atomic<u64> once_waiting_at_signal{}; ///< signals that found no one-time subscriber waiting
};

std::array<Context, MaxId + 1> g_contexts; ///< by interrupt number; an array, so the lookup needs no lock

Context& Of(u32 id) {
    ASSERT_MSG(id <= MaxId, "Invalid frame event number");
    return g_contexts[id];
}

void Signal(u32 id) {
    auto& ctx = Of(id);
    ctx.signals.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock{ctx.lock};
    for (auto& [uid, handler] : ctx.persistent) {
        handler(id);
        ctx.persistent_runs.fetch_add(1, std::memory_order_relaxed);
    }
    if (!ctx.once.empty()) {
        ctx.once.front()(id);
        ctx.once.pop();
        ctx.once_runs.fetch_add(1, std::memory_order_relaxed);
    } else {
        ctx.once_waiting_at_signal.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

void EndOfPipe() {
    Signal(EndOfPipeId);
}

void ComputeRelease(u32 pipe) {
    ASSERT_MSG(pipe <= 6, "Unexpected compute queue release");
    Signal(pipe);
}

void GpuIdle() {
    Signal(GpuIdleId);
}

void FlipReady() {
    Signal(FlipReadyId);
}

void Subscribe(u32 id, Handler handler, void* uid) {
    auto& ctx = Of(id);
    std::unique_lock lock{ctx.lock};
    ASSERT_MSG(ctx.persistent.find(uid) == ctx.persistent.cend(), "The handler is already registered!");
    ctx.persistent.emplace(uid, std::move(handler));
}

void Unsubscribe(u32 id, void* uid) {
    auto& ctx = Of(id);
    std::unique_lock lock{ctx.lock};
    ctx.persistent.erase(uid);
}

void SubscribeOnce(u32 id, Handler handler) {
    auto& ctx = Of(id);
    std::unique_lock lock{ctx.lock};
    ctx.once.push(std::move(handler));
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    const auto name = [](u32 id) -> std::string {
        if (id == EndOfPipeId) return "end_of_pipe";
        if (id == FlipReadyId) return "flip_ready";
        if (id == GpuIdleId) return "gpu_idle";
        return "compute_release_" + std::to_string(id);
    };
    for (u32 id = 0; id <= MaxId; ++id) {
        auto& ctx = g_contexts[id];
        const u64 signals = ctx.signals.load();
        std::unique_lock lock{ctx.lock};
        if (signals == 0 && ctx.persistent.empty() && ctx.once.empty()) {
            continue;
        }
        j[name(id)] = {{"id", id}, {"signals", signals}, {"persistent_subscribers", ctx.persistent.size()}, {"persistent_runs", ctx.persistent_runs.load()},
                       {"one_time_waiting", ctx.once.size()}, {"one_time_runs", ctx.once_runs.load()}, {"signals_with_no_one_time_subscriber", ctx.once_waiting_at_signal.load()}};
    }
    return j;
}

} // namespace Core::NativeFrameEvents
