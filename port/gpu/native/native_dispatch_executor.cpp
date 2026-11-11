// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_dispatch_executor.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "native/native_debug_server.h"

namespace Core::DispatchExecutor {

u32 g_mode = [] {
    const char* e = std::getenv("TLG_DISPATCH_EXECUTOR");
    if (e == nullptr || e[0] == 0) {
        return 2u; // default 2 (real) since 2026-11-11 (note 159); LEGACY(dispatch_executor): =0 the old body binds every buffer binding, =1 shadow (tags and compare)
    }
    return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
}();
bool g_on = g_mode == 1;

namespace {
// raster thread only (one writer); the report reads relaxed
thread_local bool t_open = false;
thread_local Source t_last = SrcCount;
thread_local std::array<u32, SrcCount> t_src{};
thread_local std::array<u32, 2> t_calls{};
std::atomic<u64> dispatches{0}, eligible{0}, eligible_with_calls{0}, eligible_bound_entries{0}, eligible_barriers{0}, eligible_empty{0};
std::array<std::atomic<u64>, SrcCount> src_all{}, src_eligible{};
std::array<std::atomic<u64>, SrcCount> blocked_only_by{}; ///< dispatches whose only old source is this one
std::array<std::atomic<u64>, 2> calls_in_eligible{}, calls_all{};
thread_local u64 t_since_publish = 0;
// note 159
std::array<std::atomic<u64>, ExKindCount> x_bound{};
std::array<std::atomic<u64>, RfCount> x_refused{};
std::array<std::atomic<u64>, CmpCount> x_compare{};
std::array<std::atomic<u64>, RfCount> x_refused_old_native{};
std::array<std::array<std::atomic<u64>, OwCount>, RfCount> x_refused_own_why{};
std::atomic<u64> x_real{0}, x_real_all_executor{0}, x_shadowed{0}, x_real_bindings{0}, x_real_refused{0};

constexpr const char* SrcNames[SrcCount] = {"hook_ring", "constant_ring", "gds", "flat_user_data", "clip_planes", "shared_memory", "own", "own_crossing", "own_placed",
                                            "program_consts", "hook_ring_window", "null", "texture_table", "sampler_record", "OLD_resolved_copy",
                                            "OLD_resolved_resident", "OLD_in_place", "OLD_obtain_stream", "OLD_obtain_other", "OLD_texture_stage", "OLD_sampler_user_data"};
constexpr const char* KindNames[ExKindCount] = {"refused", "hook_ring", "constant_ring", "gds", "flat_user_data", "flat_user_data_indexed", "shared_memory", "own_placed",
                                                "own", "own_crossing", "program_consts", "null", "hook_ring_window", "own_after_sync"};
std::atomic<u64> x_sync_would{0}, x_sync_left{0};
const u32 g_own_sync = [] {
    const char* e = std::getenv("TLG_OWN_SYNC");
    return e == nullptr || e[0] == 0 ? 2u : e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u; // default 2 since 2026-11-12 (note 162); LEGACY(own_sync): =0, =1 shadow
}();
constexpr const char* OwnWhyNames[OwCount] = {"no_id", "no_record", "outside_record", "dead", "immutable_written", "no_own_block", "past_exact_end", "cpu_bytes_waiting",
                                              "written_not_gpu_marked", "crossing_in_frame_ring", "crossing_no_holder", "crossing_cpu_bytes_waiting", "crossing_not_gpu_marked"};
constexpr const char* RefusalNames[RfCount] = {"none", "special", "resolved_r1b_r2", "no_record", "unmapped_or_clamped", "program_consts_full", "obtain_buffer"};

nlohmann::json Report() {
    const auto r = [](const std::atomic<u64>& a) { return a.load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["mode"] = g_mode;
    if (g_on) {
        j["dispatches"] = r(dispatches);
        j["executor_class_every_binding_native"] = r(eligible);
        j["executor_class_with_no_binding"] = r(eligible_empty);
        j["executor_class_still_calling_address_lookups"] = r(eligible_with_calls);
        j["address_calls_in_executor_class"] = {{"ObtainBuffer", r(calls_in_eligible[0])}, {"FindImage", r(calls_in_eligible[1])}};
        j["address_calls_all_dispatches"] = {{"ObtainBuffer", r(calls_all[0])}, {"FindImage", r(calls_all[1])}};
        j["executor_class_side_effects"] = {{"bound_list_entries", r(eligible_bound_entries)}, {"dispatches_needing_a_barrier", r(eligible_barriers)}};
        for (u32 s = 0; s < SrcCount; ++s) {
            if (const u64 n = r(src_all[s]); n != 0) {
                j["bindings_by_source"][SrcNames[s]] = n;
            }
            if (const u64 n = r(src_eligible[s]); n != 0) {
                j["executor_class_bindings_by_source"][SrcNames[s]] = n;
            }
            if (const u64 n = r(blocked_only_by[s]); n != 0) {
                j["dispatches_blocked_only_by"][SrcNames[s]] = n;
            }
        }
    }
    // note 159
    auto& x = j["executor"];
    x["dispatches_real"] = r(x_real);
    x["dispatches_real_every_buffer_by_executor"] = r(x_real_all_executor);
    x["dispatches_compared"] = r(x_shadowed);
    x["real_buffer_bindings"] = r(x_real_bindings);
    x["real_buffer_bindings_refused"] = r(x_real_refused);
    for (u32 k = 1; k < ExKindCount; ++k) {
        if (const u64 n = r(x_bound[k]); n != 0) {
            x["bound_by_source"][KindNames[k]] = n;
        }
    }
    for (u32 k = 1; k < RfCount; ++k) {
        if (const u64 n = r(x_refused[k]); n != 0) {
            x["refused_by_reason"][RefusalNames[k]] = n;
        }
        for (u32 w = 0; w < OwCount; ++w) {
            if (const u64 n = r(x_refused_own_why[k][w]); n != 0) {
                x["refused_why_own_memory_did_not_answer"][RefusalNames[k]][OwnWhyNames[w]] = n;
            }
        }
        if (const u64 n = r(x_refused_old_native[k]); n != 0) {
            x["compare_refused_where_the_old_body_was_native"][RefusalNames[k]] = n;
        }
    }
    x["compare_same"] = r(x_compare[CmpSame]);
    x["compare_own_sync_where_the_old_body_copied"] = r(x_compare[CmpOwnSyncOld]);
    x["own_sync_mode"] = g_own_sync;
    x["own_sync_shadow_would_sync"] = r(x_sync_would);
    x["compare_old_body_audited"] = r(x_compare[CmpAudited]);
    x["compare_refused_old_body_old_source"] = r(x_compare[CmpRefusedOldOld]);
    x["compare_refused_old_body_native"] = r(x_compare[CmpRefusedOldNative]);
    j["EXECUTOR_DIFFERENT_MUST_BE_0"] = r(x_compare[CmpDifferent]);
    j["EXECUTOR_OTHER_SOURCE_MUST_BE_0"] = r(x_compare[CmpOtherSource]);
    j["OWN_SYNC_LEFT_CPU_BYTES_MUST_BE_0"] = r(x_sync_left);
    j["note"] = "note 152 tags (mode 1) and note 159 executor: per buffer binding of a compute stage, native sources decided with the old loop's tests, refusals to the old body";
    return j;
}
} // namespace

Source SourceOf(u32 kind) noexcept {
    switch (kind) {
    case ExHookRing:
        return SrcHookRing;
    case ExConstantRing:
        return SrcConstantRing;
    case ExGds:
        return SrcGds;
    case ExFlat:
        return SrcFlatUserData;
    case ExShared:
        return SrcSharedMemory;
    case ExOwnPlaced:
        return SrcOwnPlaced;
    case ExOwn:
        return SrcOwn;
    case ExOwnCrossing:
        return SrcOwnCrossing;
    case ExProgramConsts:
        return SrcProgramConsts;
    case ExNull:
        return SrcNull;
    case ExHookRingWindow:
        return SrcHookRingWindow;
    case ExOwnSync:
        return SrcOwn;
    default:
        return SrcCount;
    }
}

void Begin(bool compute) noexcept {
    if (!g_on) {
        return;
    }
    t_open = compute;
    t_src.fill(0);
    t_calls.fill(0);
}

void Note(Source s) noexcept {
    t_last = s;
    if (t_open && s < SrcCount) {
        ++t_src[s];
    }
}

Source LastSource() noexcept {
    return t_last;
}

void ClearLast() noexcept {
    t_last = SrcCount;
}

void NoteAddressCall(u32 kind) noexcept {
    if (t_open && kind < 2) {
        ++t_calls[kind];
    }
}

void NoteBound(u32 kind) noexcept {
    if (kind < ExKindCount) {
        x_bound[kind].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteRefused(u32 refusal, u32 own_why) noexcept {
    if (refusal < RfCount) {
        x_refused[refusal].fetch_add(1, std::memory_order_relaxed);
        if (own_why < OwCount) {
            x_refused_own_why[refusal][own_why].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void NoteCompare(u32 what, u32 refusal) noexcept {
    if (what < CmpCount) {
        x_compare[what].fetch_add(1, std::memory_order_relaxed);
    }
    if (what == CmpRefusedOldNative && refusal < RfCount) {
        x_refused_old_native[refusal].fetch_add(1, std::memory_order_relaxed);
    }
}

u32 OwnSyncMode() noexcept {
    return g_own_sync;
}

void NoteOwnSync(bool would, bool left_bytes) noexcept {
    if (would) {
        x_sync_would.fetch_add(1, std::memory_order_relaxed);
    }
    if (left_bytes) {
        x_sync_left.fetch_add(1, std::memory_order_relaxed);
    }
}

void DispatchDone(bool real, u32 bindings, u32 refused) noexcept {
    if (real) {
        x_real.fetch_add(1, std::memory_order_relaxed);
        x_real_bindings.fetch_add(bindings, std::memory_order_relaxed);
        x_real_refused.fetch_add(refused, std::memory_order_relaxed);
        if (refused == 0) {
            x_real_all_executor.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        x_shadowed.fetch_add(1, std::memory_order_relaxed);
    }
}

void End(u32 bound_entries, bool barrier) noexcept {
    if (t_open) {
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
    }
    if (g_mode != 0 && ++t_since_publish >= 4096) {
        t_since_publish = 0;
        Core::DebugServer::Publish("dispatch_executor", Report());
    }
}

} // namespace Core::DispatchExecutor
