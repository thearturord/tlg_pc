// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_invariant.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>

#include <fmt/format.h>

#include "native/native_debug_server.h"
#include "native/native_flight.h"
#include "native/native_marker.h"

namespace Core::Invariant {

namespace {

struct Info {
    const char* name;
    const char* text;
};
constexpr Info infos[] = {
#define X(id, name, text) {name, text},
    TLG_INVARIANTS(X)
#undef X
};

std::array<std::atomic<u64>, Count> counts{};
std::array<std::atomic<u64>, Count> last_seq{};

constexpr u32 Window = 64;
std::array<double, Window> intervals_ms{};
u32 interval_count = 0;
u64 last_present_ns = 0;
u64 last_rows = 0;
u64 frames_seen = 0;

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

void Hit(Id id, u64 seq, u64 a, u64 b, bool note) {
    const u64 n = counts[id].fetch_add(1, std::memory_order_relaxed) + 1;
    last_seq[id].store(seq, std::memory_order_relaxed);
    Flight::Add(Flight::Invariant, seq, id, n, 0);
    if (note && n <= 3) {
        RareEvents::Note(std::string{"inv_"} + infos[id].name, fmt::format("#{} seq {} a {:#x} b {:#x}: {}", n, seq, a, b, infos[id].text));
    }
}

void NoteFrame() {
    // called by the thread that presents; only that thread touches the statics below
    const u64 now = NowNs();
    const u64 rows = Marker::RowsWritten();
    ++frames_seen;
    if (last_present_ns != 0) {
        const double ms = static_cast<double>(now - last_present_ns) / 1e6;
        if (interval_count >= 16) {
            std::array<double, Window> sorted = intervals_ms;
            const u32 n = std::min(interval_count, Window);
            std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.begin() + n);
            const double median = sorted[n / 2];
            if (frames_seen > 600 && ms > 3.0 * median && ms > 40.0) {
                Hit(SlowFrame, 0, static_cast<u64>(ms * 1000.0), static_cast<u64>(median * 1000.0), false);
            }
        }
        intervals_ms[interval_count % Window] = ms;
        ++interval_count;
    }
    if (frames_seen > 600 && rows == last_rows && Marker::Enabled()) {
        Hit(FrameWithoutDraws, 0, frames_seen, 0, false);
    }
    last_rows = rows;
    last_present_ns = now;
}

nlohmann::json Report() {
    nlohmann::json out;
    for (u32 i = 0; i < Count; ++i) {
        out[infos[i].name] = {{"count", counts[i].load(std::memory_order_relaxed)}, {"last_seq", last_seq[i].load(std::memory_order_relaxed)}, {"means", infos[i].text}};
    }
    return out;
}

} // namespace Core::Invariant
