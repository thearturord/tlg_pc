// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_dispatch_split.h"

#include <chrono>
#include <cstdlib>

#include <nlohmann/json.hpp>

namespace Core::DispatchSplit {

thread_local bool t_active = false;
thread_local u64 t_stamp[PointCount] = {};

namespace {
const bool g_on = [] {
    const char* e = std::getenv("TLG_DISPATCH_SPLIT");
    return e != nullptr && e[0] == '1';
}();
const u32 g_every = [] {
    const char* e = std::getenv("TLG_DISPATCH_SPLIT_EVERY");
    const long v = e != nullptr ? std::atol(e) : 8;
    return static_cast<u32>(v >= 1 ? v : 8);
}();
thread_local u32 t_count = 0;
thread_local bool t_planned = false;
thread_local u32 t_access[3] = {}; // RecordBufferAccess of the sampled dispatch
std::atomic<u64> g_access[3]{};
thread_local u64 t_calib = 0; // ticks of one empty segment (two stamps back to back)

// [planned][step] ticks; step k is stamp[k] - stamp[k-1] (k >= 1); [planned][0] holds the calibration ticks
std::atomic<u64> g_ticks[2][PointCount]{};
std::atomic<u64> g_samples[2]{}, g_dropped{0}, g_begun{0};
std::atomic<u64> g_tsc0{0}, g_ns0{0};

u64 SteadyNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

constexpr const char* Names[PointCount] = {"(calibration: one empty segment)", "ring_use_begin", "record_queue_drain", "caller_fill_verify", "explicit_readback_flag",
                                           "marker_and_census_scope_open", "clear_checks_class_tags", "pop_pending_operations", "pending_fill", "barrier_census_op_and_group",
                                           "resolve_pipeline", "light_frame_op_and_hle", "binder_plan_and_light_begin", "BIND_RESOURCES", "after_bind_census",
                                           "flush_barriers", "end_rendering", "descriptor_push", "marker_row", "record_dispatch", "debug_count",
                                           "reset_resolve_commands", "reset_image_bindings", "reset_record_buffer_access", "census_and_marker_scope_close", "stats_and_readback_flag", "ring_use_end", "flush_recorded"};
} // namespace

bool On() noexcept {
    return g_on;
}

void Begin() noexcept {
    t_active = false;
    if (!g_on || (++t_count % g_every) != 0) {
        return;
    }
    if (g_tsc0.load(std::memory_order_relaxed) == 0) {
        u64 expected = 0;
        const u64 tsc = Tsc();
        if (g_tsc0.compare_exchange_strong(expected, tsc, std::memory_order_relaxed)) {
            g_ns0.store(SteadyNs(), std::memory_order_relaxed);
        }
    }
    for (auto& s : t_stamp) {
        s = 0;
    }
    t_planned = false;
    t_access[0] = t_access[1] = t_access[2] = 0;
    const u64 a = Tsc();
    const u64 b = Tsc();
    t_calib = b - a;
    t_stamp[PExecBegin] = Tsc();
    t_active = true;
    g_begun.fetch_add(1, std::memory_order_relaxed);
}

void NoteAccess(u32 what) noexcept {
    if (t_active && what < 3) {
        ++t_access[what];
    }
}

void NotePlanned(bool planned) noexcept {
    if (t_active) {
        t_planned = planned;
    }
}

void End() noexcept {
    if (!t_active) {
        return;
    }
    t_active = false;
    for (u32 k = 1; k < PointCount; ++k) {
        if (t_stamp[k] == 0 || t_stamp[k] < t_stamp[k - 1]) {
            g_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    for (u32 k = 0; k < 3; ++k) {
        g_access[k].fetch_add(t_access[k], std::memory_order_relaxed);
    }
    const u32 p = t_planned ? 1 : 0;
    g_samples[p].fetch_add(1, std::memory_order_relaxed);
    g_ticks[p][0].fetch_add(t_calib, std::memory_order_relaxed);
    for (u32 k = 1; k < PointCount; ++k) {
        g_ticks[p][k].fetch_add(t_stamp[k] - t_stamp[k - 1], std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["on"] = g_on;
    j["every"] = g_every;
    const u64 tsc0 = g_tsc0.load(), ns0 = g_ns0.load();
    const u64 tsc1 = Tsc(), ns1 = SteadyNs();
    const double ticks_per_ns = (ns1 > ns0 && tsc1 > tsc0) ? static_cast<double>(tsc1 - tsc0) / static_cast<double>(ns1 - ns0) : 0.0;
    j["ticks_per_ns"] = ticks_per_ns;
    j["begun"] = g_begun.load();
    j["dropped_left_early"] = g_dropped.load();
    j["record_buffer_access"] = {{"stream_reads_skipped", g_access[0].load()}, {"reads_recorded", g_access[1].load()}, {"writes_recorded", g_access[2].load()}};
    j["note"] = "cumulative; per step: total ticks over the samples (ns = ticks / ticks_per_ns); each step includes one time stamp read, the calibration row is one empty segment";
    for (u32 p = 0; p < 2; ++p) {
        nlohmann::json o = nlohmann::json::object();
        o["samples"] = g_samples[p].load();
        for (u32 k = 0; k < PointCount; ++k) {
            o["ticks"][Names[k]] = g_ticks[p][k].load();
        }
        j[p ? "planned" : "not_planned"] = std::move(o);
    }
    return j;
}

} // namespace Core::DispatchSplit
