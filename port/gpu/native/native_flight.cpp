// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_flight.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <fmt/format.h>

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId();
#endif

namespace Core::Flight {

namespace {

const char* const kind_names[] = {
#define X(id, text) text,
    TLG_FLIGHT_KINDS(X)
#undef X
};

std::mutex register_mutex;

u32 CurrentThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return 0;
#endif
}

u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// The processor's time stamp counter against the steady clock, taken when the first thread registers.
struct Calibration {
    u64 tick0;
    u64 ns0;
};
Calibration start{};

const bool env_read = [] {
    const char* e = std::getenv("TLG_FLIGHT");
    if (e != nullptr && *e == '0') {
        g_enabled.store(false);
    }
    return true;
}();

} // namespace

const char* KindName(u32 kind) {
    return kind < KindCount ? kind_names[kind] : "?";
}

Ring* RegisterThisThread() {
    if (t_ring != nullptr) {
        return t_ring;
    }
    std::scoped_lock lock{register_mutex};
    const u32 slot = g_ring_count.load(std::memory_order_relaxed);
    if (slot >= MaxThreads) {
        return nullptr; // a thread beyond the table writes no events
    }
    if (slot == 0) {
        start = {Tick(), SteadyNs()};
    }
    auto* ring = new Ring();
    ring->slot = slot;
    ring->thread_id = CurrentThreadId();
    std::snprintf(ring->name, sizeof(ring->name), "thread %u", ring->thread_id);
    g_rings[slot] = ring;
    g_ring_count.store(slot + 1, std::memory_order_release);
    t_ring = ring;
    return ring;
}

void NameThisThread(const char* name) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    Ring* ring = t_ring != nullptr ? t_ring : RegisterThisThread();
    if (ring != nullptr) {
        std::snprintf(ring->name, sizeof(ring->name), "%s", name);
    }
}

u64 EventsWritten() {
    u64 total = 0;
    const u32 count = g_ring_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        total += g_rings[i]->head.load(std::memory_order_relaxed);
    }
    return total;
}

std::string Report(u32 per_thread, bool bench) {
    std::string out = "{";
    if (bench) {
        const bool was = g_enabled.exchange(true);
        constexpr u32 events = 1'000'000;
        const u64 t0 = SteadyNs();
        for (u32 i = 0; i < events; ++i) {
            Add(Bench, i, i, i);
        }
        const u64 t1 = SteadyNs();
        g_enabled.store(was);
        out += fmt::format("\"bench_ns_per_event\": {:.2f}, \"bench_events\": {}, ", static_cast<double>(t1 - t0) / events, events);
    }
    const u64 tick_now = Tick();
    const u64 ns_now = SteadyNs();
    // ticks per ns from the span since the first thread registered (at least a few ms long, else 1)
    double ticks_per_ns = 1.0;
    if (ns_now > start.ns0 + 5'000'000 && tick_now > start.tick0) {
        ticks_per_ns = static_cast<double>(tick_now - start.tick0) / static_cast<double>(ns_now - start.ns0);
    }
    const auto ms_ago = [&](u64 tick) { return -static_cast<double>(static_cast<s64>(tick_now - tick)) / ticks_per_ns / 1e6; };
    out += fmt::format("\"enabled\": {}, \"threads\": {}, \"events_written\": {}, \"ring_events\": {}, \"ticks_per_ns\": {:.3f}, \"unit\": \"event time = ms before this report\", ",
                       g_enabled.load() ? "true" : "false", g_ring_count.load(), EventsWritten(), RingEvents, ticks_per_ns);

    struct Line {
        u64 tick;
        const Entry* entry;
    };
    std::vector<Line> lines;
    out += "\"thread_list\": [";
    const u32 count = g_ring_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        const Ring& ring = *g_rings[i];
        const u64 head = ring.head.load(std::memory_order_acquire);
        const char* scope = ring.scope.load(std::memory_order_relaxed);
        out += fmt::format("{}{{\"slot\": {}, \"name\": \"{}\", \"thread_id\": {}, \"doing\": \"{}\", \"events\": {}}}", i == 0 ? "" : ", ", i, ring.name, ring.thread_id,
                           scope != nullptr ? scope : "", head);
        const u64 first = head > per_thread ? head - per_thread : 0;
        for (u64 h = first; h < head && head - h <= RingEvents; ++h) {
            const Entry& entry = ring.events[h & (RingEvents - 1)];
            lines.push_back({entry.tick, &entry});
        }
    }
    out += "], \"events\": [";
    std::sort(lines.begin(), lines.end(), [](const Line& x, const Line& y) { return x.tick < y.tick; });
    bool first_line = true;
    for (const Line& line : lines) {
        const Entry& e = *line.entry;
        out += fmt::format("{}[{:.3f}, {}, \"{}\", {}, {}, {}, {}]", first_line ? "" : ", ", ms_ago(e.tick), e.slot, KindName(e.kind), e.seq, e.a, e.b, e.flags);
        first_line = false;
    }
    out += "], \"columns\": \"ms before report, thread slot, kind, seq, a, b, flags\"}";
    return out;
}

} // namespace Core::Flight
