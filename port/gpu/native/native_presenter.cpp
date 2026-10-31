// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_presenter.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <nlohmann/json.hpp>

namespace Core::NativePresenter {

namespace {
std::mutex g_mutex;
std::condition_variable g_cv;
std::queue<Request> g_queue;
u32 g_limit = 0;
Clock::duration g_period{};
Clock::time_point g_next{};
s32 g_shown = -1;
std::atomic<u64> g_pushed{}, g_taken{}, g_peak{}, g_shown_count{}, g_starved{}, g_limit_changes{}, g_blank_shown{}, g_queued_deeper_than_one_at_take{};
} // namespace

void Push(const Request& request) {
    {
        std::scoped_lock lock{g_mutex};
        g_queue.push(request);
        u64 depth = g_queue.size();
        u64 peak = g_peak.load(std::memory_order_relaxed);
        while (depth > peak && !g_peak.compare_exchange_weak(peak, depth, std::memory_order_relaxed)) {
        }
    }
    g_pushed.fetch_add(1, std::memory_order_relaxed);
    g_cv.notify_one();
}

bool TryTake(Request& out) {
    std::scoped_lock lock{g_mutex};
    if (g_queue.empty()) {
        return false;
    }
    if (g_queue.size() > 1) {
        g_queued_deeper_than_one_at_take.fetch_add(1, std::memory_order_relaxed);
    }
    out = g_queue.front();
    g_queue.pop();
    g_taken.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool WaitForRequest(Clock::time_point deadline) {
    std::unique_lock lock{g_mutex};
    return g_cv.wait_until(lock, deadline, [] { return !g_queue.empty(); });
}

std::size_t Pending() {
    std::scoped_lock lock{g_mutex};
    return g_queue.size();
}

void SetFrameLimit(u32 fps, Clock::time_point now) {
    std::scoped_lock lock{g_mutex};
    if (fps != g_limit) {
        g_limit_changes.fetch_add(1, std::memory_order_relaxed);
    }
    g_limit = fps;
    g_period = fps ? std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(1000000000ull / fps)) : Clock::duration{};
    g_next = now;
}

u32 FrameLimit() {
    std::scoped_lock lock{g_mutex};
    return g_limit;
}

bool ShowBetweenVblanks() {
    std::scoped_lock lock{g_mutex};
    return g_limit != 0;
}

bool SlotOpen(Clock::time_point now) {
    std::scoped_lock lock{g_mutex};
    return g_limit == 0 || now >= g_next;
}

Clock::time_point NextSlot() {
    std::scoped_lock lock{g_mutex};
    return g_next;
}

void NoteTaken(Clock::time_point now) {
    std::scoped_lock lock{g_mutex};
    if (g_limit != 0) {
        g_next = std::max(g_next + g_period, now - g_period);
    }
}

void NoteStarved() {
    g_starved.fetch_add(1, std::memory_order_relaxed);
}

s32 NoteShown(s32 index) {
    std::scoped_lock lock{g_mutex};
    const s32 before = g_shown;
    g_shown = index;
    g_shown_count.fetch_add(1, std::memory_order_relaxed);
    if (index == -1) {
        g_blank_shown.fetch_add(1, std::memory_order_relaxed);
    }
    return before;
}

void Reset() {
    std::scoped_lock lock{g_mutex};
    g_shown = -1;
}

nlohmann::json Report() {
    nlohmann::json j;
    {
        std::scoped_lock lock{g_mutex};
        j["frame_limit_fps"] = g_limit;
        j["queue_now"] = g_queue.size();
        j["on_screen_buffer"] = g_shown;
    }
    j["flips_queued"] = g_pushed.load();
    j["flips_taken_by_the_present_thread"] = g_taken.load();
    j["queue_peak"] = g_peak.load();
    j["taken_with_another_flip_still_queued"] = g_queued_deeper_than_one_at_take.load();
    j["frames_shown"] = g_shown_count.load();
    j["blank_frames_shown"] = g_blank_shown.load();
    j["slot_open_but_no_flip_waiting"] = g_starved.load();
    j["frame_limit_changes"] = g_limit_changes.load();
    return j;
}

} // namespace Core::NativePresenter
