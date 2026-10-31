// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_display.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>

#include <nlohmann/json.hpp>

namespace Core::NativeDisplay {

namespace {
std::mutex g_mutex;
std::condition_variable g_cv;
std::array<bool, MaxBuffers> g_locked{};
const volatile u64* g_labels = nullptr;
std::atomic<u64> g_locks{}, g_releases{}, g_waits_passed{}, g_passed_while_locked_state{}, g_label_disagrees{}, g_blocked_waits{};
} // namespace

void SetLabels(const volatile u64* labels) {
    g_labels = labels;
}

void Lock(u32 buffer) {
    if (buffer >= MaxBuffers) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    g_locked[buffer] = true;
    g_locks.fetch_add(1, std::memory_order_relaxed);
}

void Release(u32 buffer) {
    if (buffer >= MaxBuffers) {
        return;
    }
    {
        std::scoped_lock lock{g_mutex};
        g_locked[buffer] = false;
        g_releases.fetch_add(1, std::memory_order_relaxed);
    }
    g_cv.notify_all();
}

bool IsFree(u32 buffer) {
    if (buffer >= MaxBuffers) {
        return true;
    }
    std::scoped_lock lock{g_mutex};
    return !g_locked[buffer];
}

void WaitFree(u32 buffer) {
    if (buffer >= MaxBuffers) {
        return;
    }
    std::unique_lock lock{g_mutex};
    if (g_locked[buffer]) {
        g_blocked_waits.fetch_add(1, std::memory_order_relaxed);
    }
    g_cv.wait(lock, [&] { return !g_locked[buffer]; });
}

void NoteWaitPassed(u32 buffer) {
    g_waits_passed.fetch_add(1, std::memory_order_relaxed);
    if (buffer >= MaxBuffers || g_labels == nullptr) {
        return;
    }
    const bool locked = !IsFree(buffer);
    const bool label_free = g_labels[buffer] == 0;
    if (locked) {
        g_passed_while_locked_state.fetch_add(1, std::memory_order_relaxed);
    }
    if (label_free == locked) {
        g_label_disagrees.fetch_add(1, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["locks"] = g_locks.load();
    j["releases"] = g_releases.load();
    j["waits_passed"] = g_waits_passed.load();
    j["waits_that_had_to_block_the_thread"] = g_blocked_waits.load();
    j["passed_while_the_state_said_locked"] = g_passed_while_locked_state.load();
    j["label_disagrees_with_the_state"] = g_label_disagrees.load();
    std::scoped_lock lock{g_mutex};
    auto& locked = j["locked_now"] = nlohmann::json::array();
    for (u32 i = 0; i < MaxBuffers; ++i) {
        if (g_locked[i]) {
            locked.push_back(i);
        }
    }
    return j;
}

} // namespace Core::NativeDisplay
