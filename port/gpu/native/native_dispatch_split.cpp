// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_dispatch_split.h"

#include <algorithm>
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

thread_local bool t_planner_sampled = false;
thread_local u64 t_oracle_ticks = 0;
namespace {
const bool g_planner_on = [] {
    const char* e = std::getenv("TLG_PLANNER_CENSUS");
    return e != nullptr && e[0] == '1';
}();
thread_local u32 t_planner_count = 0;
// access: 0 samples, 1 space, 2 oracle, 3 lookup, 4 add, 5 rest, 6 placed, 7 new nodes, 8 writes, 9 read list size sum, 10 write list size sum, 11 calibration ticks
std::atomic<u64> g_pa[12]{};
// query: 0 samples, 1 space, 2 oracle, 3 lookup and overlaps, 4 new nodes, 5 answered true
std::atomic<u64> g_pq[6]{};
// flush: 0 flushes, 1 with a barrier, 2 clear ticks (with a barrier), 3 record ticks, 4 object keys, 5 address keys, 6 group keys, 7 clear ticks without a barrier
std::atomic<u64> g_pf[8]{};
} // namespace

bool PlannerOn() noexcept {
    return g_planner_on;
}

bool PlannerSample() noexcept {
    t_planner_sampled = g_planner_on && (++t_planner_count & 7) == 0;
    t_oracle_ticks = 0;
    if (t_planner_sampled && g_tsc0.load(std::memory_order_relaxed) == 0) {
        u64 expected = 0;
        const u64 tsc = Tsc();
        if (g_tsc0.compare_exchange_strong(expected, tsc, std::memory_order_relaxed)) {
            g_ns0.store(SteadyNs(), std::memory_order_relaxed);
        }
    }
    if (t_planner_sampled) {
        const u64 a = Tsc();
        g_pa[11].fetch_add(Tsc() - a, std::memory_order_relaxed);
    }
    return t_planner_sampled;
}

void NotePlannerAccess(u64 space, u64 oracle, u64 lookup, u64 add, u64 rest, bool placed, bool new_node, bool write, u32 read_size, u32 write_size) noexcept {
    t_planner_sampled = false;
    const u64 v[11] = {1, space - std::min(space, oracle), oracle, lookup, add, rest, placed ? 1u : 0u, new_node ? 1u : 0u, write ? 1u : 0u, read_size, write_size};
    for (u32 k = 0; k < 11; ++k) {
        g_pa[k].fetch_add(v[k], std::memory_order_relaxed);
    }
}

void NotePlannerQuery(u64 space, u64 oracle, u64 lookup, bool new_node, bool answer) noexcept {
    t_planner_sampled = false;
    const u64 v[6] = {1, space - std::min(space, oracle), oracle, lookup, new_node ? 1u : 0u, answer ? 1u : 0u};
    for (u32 k = 0; k < 6; ++k) {
        g_pq[k].fetch_add(v[k], std::memory_order_relaxed);
    }
}

void NotePlannerFlush(u64 clear, u64 record, u32 object_keys, u32 address_keys, u32 group_keys, bool actual) noexcept {
    g_pf[0].fetch_add(1, std::memory_order_relaxed);
    if (actual) {
        g_pf[1].fetch_add(1, std::memory_order_relaxed);
        g_pf[2].fetch_add(clear, std::memory_order_relaxed);
        g_pf[3].fetch_add(record, std::memory_order_relaxed);
        g_pf[4].fetch_add(object_keys, std::memory_order_relaxed);
        g_pf[5].fetch_add(address_keys, std::memory_order_relaxed);
        g_pf[6].fetch_add(group_keys, std::memory_order_relaxed);
    } else {
        g_pf[7].fetch_add(clear, std::memory_order_relaxed);
    }
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

thread_local bool t_draw_active = false;
thread_local u64 t_draw_stamp[DrawPointCount] = {};
namespace {
const bool g_draw_on = [] {
    const char* e = std::getenv("TLG_DRAW_SPLIT");
    return e != nullptr && e[0] == '1';
}();
thread_local u32 t_draw_count = 0;
thread_local u32 t_draw_class = 7;
thread_local bool t_draw_indexed = false, t_draw_planned = false;
thread_local u64 t_draw_calib = 0;
constexpr u32 DrawClasses = 8;
std::atomic<u64> g_dticks[DrawClasses][DrawPointCount]{}; // [class][step], [class][0] the calibration
std::atomic<u64> g_dsamples[DrawClasses]{}, g_dindexed[DrawClasses]{}, g_dplanned[DrawClasses]{}, g_ddropped{0};
constexpr const char* DrawNames[DrawPointCount] = {"(calibration: one empty segment)", "ring_use_begin", "set_native_tag", "marker_and_census_scope_open", "clear_checks",
                                                   "pop_pending_operations", "tag_class_stats", "filter_draw", "resolve_pipeline", "diagnostics_and_inputs",
                                                   "prepare_render_state", "binder_plan", "BIND_RESOURCES", "begin_rendering_state", "vertex_buffers", "index_buffer",
                                                   "pass_state_difference", "flush_barriers", "descriptor_push", "dynamic_state", "scheduler_begin_rendering", "draw_offsets",
                                                   "marker_row", "record_draw", "counts", "reset_resolve_commands", "reset_image_bindings", "reset_record_buffer_access",
                                                   "census_and_marker_scope_close", "ring_use_end", "flush_recorded"};
} // namespace

bool DrawOn() noexcept {
    return g_draw_on;
}

void BeginDraw() noexcept {
    t_draw_active = false;
    if (!g_draw_on || (++t_draw_count % g_every) != 0) {
        return;
    }
    if (g_tsc0.load(std::memory_order_relaxed) == 0) {
        u64 expected = 0;
        const u64 tsc = Tsc();
        if (g_tsc0.compare_exchange_strong(expected, tsc, std::memory_order_relaxed)) {
            g_ns0.store(SteadyNs(), std::memory_order_relaxed);
        }
    }
    for (auto& s : t_draw_stamp) {
        s = 0;
    }
    t_draw_class = 7;
    t_draw_indexed = t_draw_planned = false;
    const u64 a = Tsc();
    const u64 b = Tsc();
    t_draw_calib = b - a;
    t_draw_stamp[DExecBegin] = Tsc();
    t_draw_active = true;
}

void NoteDrawKind(u32 draw_class, bool indexed, bool planned) noexcept {
    if (t_draw_active) {
        t_draw_class = draw_class < 7 ? draw_class : 7;
        t_draw_indexed = indexed;
        t_draw_planned = planned;
    }
}

void EndDraw() noexcept {
    if (!t_draw_active) {
        return;
    }
    t_draw_active = false;
    for (u32 k = 1; k < DrawPointCount; ++k) {
        if (t_draw_stamp[k] == 0 || t_draw_stamp[k] < t_draw_stamp[k - 1]) {
            g_ddropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    const u32 c = t_draw_class;
    g_dsamples[c].fetch_add(1, std::memory_order_relaxed);
    g_dindexed[c].fetch_add(t_draw_indexed ? 1 : 0, std::memory_order_relaxed);
    g_dplanned[c].fetch_add(t_draw_planned ? 1 : 0, std::memory_order_relaxed);
    g_dticks[c][0].fetch_add(t_draw_calib, std::memory_order_relaxed);
    for (u32 k = 1; k < DrawPointCount; ++k) {
        g_dticks[c][k].fetch_add(t_draw_stamp[k] - t_draw_stamp[k - 1], std::memory_order_relaxed);
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
    if (g_planner_on) {
        static constexpr const char* A[12] = {"samples", "space_ticks", "oracle_ticks", "map_lookup_ticks", "interval_add_ticks", "rest_ticks", "placed", "new_nodes", "writes",
                                              "read_list_size_sum", "write_list_size_sum", "calibration_ticks"};
        static constexpr const char* Q[6] = {"samples", "space_ticks", "oracle_ticks", "lookup_and_overlap_ticks", "new_nodes", "answered_true"};
        static constexpr const char* F[8] = {"flushes", "with_barrier", "clear_ticks", "vulkan_record_ticks", "object_keys", "address_keys", "group_keys", "clear_ticks_without_barrier"};
        for (u32 k = 0; k < 12; ++k) {
            j["planner"]["access"][A[k]] = g_pa[k].load();
        }
        for (u32 k = 0; k < 6; ++k) {
            j["planner"]["query"][Q[k]] = g_pq[k].load();
        }
        for (u32 k = 0; k < 8; ++k) {
            j["planner"]["flush"][F[k]] = g_pf[k].load();
        }
    }
    for (u32 p = 0; p < 2; ++p) {
        nlohmann::json o = nlohmann::json::object();
        o["samples"] = g_samples[p].load();
        for (u32 k = 0; k < PointCount; ++k) {
            o["ticks"][Names[k]] = g_ticks[p][k].load();
        }
        j[p ? "planned" : "not_planned"] = std::move(o);
    }
    if (g_draw_on) {
        j["draws"]["dropped_left_early"] = g_ddropped.load();
        for (u32 c = 0; c < DrawClasses; ++c) {
            const u64 n = g_dsamples[c].load();
            if (n == 0) {
                continue;
            }
            nlohmann::json o = nlohmann::json::object();
            o["samples"] = n;
            o["indexed"] = g_dindexed[c].load();
            o["planned"] = g_dplanned[c].load();
            for (u32 k = 0; k < DrawPointCount; ++k) {
                o["ticks"][DrawNames[k]] = g_dticks[c][k].load();
            }
            j["draws"]["class_" + std::to_string(c)] = std::move(o);
        }
    }
    return j;
}

} // namespace Core::DispatchSplit
