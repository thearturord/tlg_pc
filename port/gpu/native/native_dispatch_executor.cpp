// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_dispatch_executor.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "native/native_debug_server.h"

namespace Core::DispatchExecutor {

bool g_on = [] {
    const char* e = std::getenv("TLG_DISPATCH_EXECUTOR");
    return e != nullptr && e[0] == '1';
}();

namespace {
// raster thread only (one writer); the report reads relaxed
thread_local bool t_open = false;
thread_local std::array<u32, SrcCount> t_src{};
thread_local std::array<u32, 2> t_calls{};
std::atomic<u64> dispatches{0}, eligible{0}, eligible_with_calls{0}, eligible_bound_entries{0}, eligible_barriers{0}, eligible_empty{0};
std::array<std::atomic<u64>, SrcCount> src_all{}, src_eligible{};
std::array<std::atomic<u64>, SrcCount> blocked_only_by{}; ///< dispatches whose only old source is this one
std::array<std::atomic<u64>, 2> calls_in_eligible{}, calls_all{};
thread_local u64 t_since_publish = 0;

nlohmann::json Report() {
    static constexpr const char* names[SrcCount] = {"hook_ring", "constant_ring", "gds", "flat_user_data", "clip_planes", "shared_memory", "own", "own_crossing", "own_placed",
                                                    "program_consts", "hook_ring_window_via_obtain", "null", "texture_table", "sampler_record", "OLD_resolved_copy",
                                                    "OLD_resolved_resident", "OLD_in_place", "OLD_obtain_stream", "OLD_obtain_other", "OLD_texture_stage", "OLD_sampler_user_data"};
    const auto r = [](const std::atomic<u64>& a) { return a.load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["mode"] = g_on ? 1 : 0;
    j["dispatches"] = r(dispatches);
    j["executor_class_every_binding_native"] = r(eligible);
    j["executor_class_with_no_binding"] = r(eligible_empty);
    j["executor_class_still_calling_address_lookups"] = r(eligible_with_calls);
    j["address_calls_in_executor_class"] = {{"ObtainBuffer", r(calls_in_eligible[0])}, {"FindImage", r(calls_in_eligible[1])}};
    j["address_calls_all_dispatches"] = {{"ObtainBuffer", r(calls_all[0])}, {"FindImage", r(calls_all[1])}};
    j["executor_class_side_effects"] = {{"bound_list_entries", r(eligible_bound_entries)}, {"dispatches_needing_a_barrier", r(eligible_barriers)}};
    for (u32 s = 0; s < SrcCount; ++s) {
        if (const u64 n = r(src_all[s]); n != 0) {
            j["bindings_by_source"][names[s]] = n;
        }
        if (const u64 n = r(src_eligible[s]); n != 0) {
            j["executor_class_bindings_by_source"][names[s]] = n;
        }
        if (const u64 n = r(blocked_only_by[s]); n != 0) {
            j["dispatches_blocked_only_by"][names[s]] = n;
        }
    }
    j["note"] = "note 152 shadow: the old loop binds every dispatch; this counts what an executor without it would take and what it would still call";
    return j;
}
} // namespace

void Begin(bool compute) noexcept {
    if (!g_on) {
        return;
    }
    t_open = compute;
    t_src.fill(0);
    t_calls.fill(0);
}

void Note(Source s) noexcept {
    if (t_open && s < SrcCount) {
        ++t_src[s];
    }
}

void NoteAddressCall(u32 kind) noexcept {
    if (t_open && kind < 2) {
        ++t_calls[kind];
    }
}

void End(u32 bound_entries, bool barrier) noexcept {
    if (!t_open) {
        return;
    }
    t_open = false;
    dispatches.fetch_add(1, std::memory_order_relaxed);
    u32 old_kinds = 0, last_old = 0, total = 0;
    for (u32 s = 0; s < SrcCount; ++s) {
        total += t_src[s];
        if (t_src[s] != 0) {
            src_all[s].fetch_add(t_src[s], std::memory_order_relaxed);
            if (s >= FirstOld) {
                ++old_kinds;
                last_old = s;
            }
        }
    }
    for (u32 k = 0; k < 2; ++k) {
        calls_all[k].fetch_add(t_calls[k], std::memory_order_relaxed);
    }
    if (old_kinds == 0) {
        eligible.fetch_add(1, std::memory_order_relaxed);
        if (total == 0) {
            eligible_empty.fetch_add(1, std::memory_order_relaxed);
        }
        for (u32 s = 0; s < SrcCount; ++s) {
            if (t_src[s] != 0) {
                src_eligible[s].fetch_add(t_src[s], std::memory_order_relaxed);
            }
        }
        if (t_calls[0] + t_calls[1] != 0) {
            eligible_with_calls.fetch_add(1, std::memory_order_relaxed);
            for (u32 k = 0; k < 2; ++k) {
                calls_in_eligible[k].fetch_add(t_calls[k], std::memory_order_relaxed);
            }
        }
        eligible_bound_entries.fetch_add(bound_entries, std::memory_order_relaxed);
        eligible_barriers.fetch_add(barrier ? 1 : 0, std::memory_order_relaxed);
    } else if (old_kinds == 1) {
        blocked_only_by[last_old].fetch_add(1, std::memory_order_relaxed);
    }
    if (++t_since_publish >= 4096) {
        t_since_publish = 0;
        Core::DebugServer::Publish("dispatch_executor", Report());
    }
}

} // namespace Core::DispatchExecutor
