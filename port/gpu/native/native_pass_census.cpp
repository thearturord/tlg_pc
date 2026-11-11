// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_pass_census.h"
#include "native/native_plan.h"

namespace Core::PassCensus {

bool g_on = [] {
    const char* e = std::getenv("TLG_PASS_CENSUS");
    return e != nullptr && e[0] == '1';
}();

namespace {

using nlohmann::json;
using Core::NativePlan::DispatchPlan;
using Core::NativePlan::DrawPlan;
using Core::NativePlan::StagePlan;

u64 Mix(u64 h, u64 v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

u64 HashWords(u64 h, const u32* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h = Mix(h, p[i]);
    }
    return h;
}

/// The difference classes between two consecutive operations (bit numbers of the mask).
enum DrawDiff : u32 {
    DPipeline = 0, ///< pipeline pointer
    DTarget,       ///< bound targets (engine resource ids)
    DViewport,     ///< viewport, scissor
    DFixed,        ///< depth / cull / bias / topology state
    DUserData,     ///< user data registers and flat user data (descriptors, root constants)
    DResources,    ///< the engine ids of the record's bound resources
    DRing,         ///< constant ring offsets of the stages
    DGeometry,     ///< what is drawn: index count, instances, index source, vertex / index ring ranges
    DiffBits,
};
const char* const DrawDiffNames[DiffBits] = {"pipeline", "target", "viewport", "fixed_state", "user_data", "resource_ids", "constant_ring", "geometry"};

struct DrawKey {
    u64 part[DiffBits]{};
    u32 num_indices{}, num_instances{};
    u64 index_source{};
};

u64 HashStages(const std::array<StagePlan, Core::NativePlan::StageCount>& stages, bool ring_only, bool user_only) {
    u64 h = 1469598103934665603ull;
    for (const StagePlan& s : stages) {
        if (ring_only) {
            for (const auto& r : s.ring) {
                h = Mix(Mix(Mix(h, r.index), r.offset), r.size);
            }
        } else if (user_only) {
            h = HashWords(h, s.user_data.data(), s.user_data_count);
            h = HashWords(h, s.flat.data(), s.flat.size());
            h = Mix(h, s.pgm_base);
        }
    }
    return h;
}

DrawKey MakeDrawKey(const DrawPlan& p) {
    DrawKey k;
    const auto& f = *p.fixed;
    k.part[DPipeline] = reinterpret_cast<u64>(p.pipeline);
    u64 t = 7;
    for (const u64 id : p.target_resources) {
        t = Mix(t, id);
    }
    k.part[DTarget] = t;
    u64 v = Mix(f.viewport_count, f.viewport_empty);
    v = HashWords(v, reinterpret_cast<const u32*>(&f.transform), sizeof(f.transform) / 4);
    for (u32 i = 0; i < std::max(f.viewport_count, 1u) && i < f.viewports.size(); ++i) {
        v = HashWords(v, reinterpret_cast<const u32*>(&f.viewports[i]), sizeof(f.viewports[i]) / 4);
        v = HashWords(v, reinterpret_cast<const u32*>(&f.scissors[i]), sizeof(f.scissors[i]) / 4);
    }
    k.part[DViewport] = v;
    u64 s = 11;
    s = Mix(s, f.depth_test);
    s = Mix(s, f.depth_write);
    s = Mix(s, static_cast<u64>(f.depth_compare));
    s = Mix(s, f.depth_bias);
    float bias[3] = {f.depth_bias_constant, f.depth_bias_clamp, f.depth_bias_slope};
    s = HashWords(s, reinterpret_cast<const u32*>(bias), 3);
    s = Mix(s, static_cast<u32>(f.cull_mode));
    s = Mix(s, static_cast<u64>(f.front_face));
    s = Mix(s, static_cast<u64>(f.primitive_type));
    s = Mix(s, static_cast<u64>(f.polygon_mode));
    s = Mix(s, f.depth_clamp);
    s = Mix(s, f.depth_clip);
    s = Mix(s, f.stencil_test);
    s = Mix(s, f.rasterizer_discard);
    s = Mix(s, f.primitive_restart);
    k.part[DFixed] = s;
    k.part[DUserData] = HashStages(p.stages, false, true);
    u64 r = 13;
    for (u32 i = 0; i < p.binding_count && i < p.bindings.size(); ++i) {
        const auto& b = p.bindings[i];
        r = Mix(Mix(Mix(Mix(r, b.stage), b.kind), b.id), b.view_id);
    }
    k.part[DResources] = r;
    k.part[DRing] = HashStages(p.stages, true, false);
    k.num_indices = p.inputs.num_indices;
    k.num_instances = p.inputs.num_instances.NumInstances();
    k.index_source = Mix(Mix(p.inputs.index_base_address.Address(), p.inputs.index_offset), static_cast<u64>(p.inputs.index_buffer_type.index_type));
    u64 g = Mix(Mix(k.num_indices, k.num_instances), k.index_source);
    for (const auto& gr : p.geometry_ring) {
        g = Mix(Mix(Mix(g, gr.index), gr.offset), gr.size);
    }
    for (u32 i = 0; i < p.geometry_handle_count && i < p.geometry_handles.size(); ++i) {
        g = Mix(g, reinterpret_cast<u64>(p.geometry_handles[i]));
    }
    k.part[DGeometry] = g;
    return k;
}

struct DrawStats {
    u64 draws = 0, indirect = 0, instanced = 0, instances = 0, indices = 0, with_complete_bindings = 0, with_geometry_handles = 0;
    u64 index_buckets[5]{}; // <=32, <=256, <=2048, <=16384, more
    u64 diff_hist[1u << DiffBits]{};
    u64 pairs = 0;
    u64 runs_pipeline_state = 0; // runs of equal (pipeline, fixed state)
    u64 runs_full = 0;           // runs of equal (pipeline, target, viewport, fixed state)
    u64 runs_target_viewport = 0;
    // distinct keys inside the current window
    std::set<u64> win_pipeline, win_pipeline_state, win_full, win_target_viewport;
    std::unordered_map<u64, u64> win_full_draws;
    std::unordered_map<u64, u64> win_pipeline_draws;
    std::set<u64> win_pipeline_fixed_target;
    u64 windows = 0, max_pipeline_state = 0, max_full = 0, max_pipeline = 0, max_target_viewport = 0, sum_pipeline_state = 0, sum_full = 0;
    u64 max_pipeline_state_no_bias = 0;
};

struct DispatchKey {
    u64 pipeline{}, dims{}, user{}, ring{}, resources{};
};

struct KernelStats {
    u64 dispatches = 0, workgroups = 0, one_group = 0, runs = 0;
    u32 dim_x_min = ~0u, dim_x_max = 0;
    u64 address = 0;
    u64 same_dims_as_previous_of_kernel = 0, same_user_as_previous_of_kernel = 0;
    u64 last_dims = ~0ull, last_user = ~0ull;
};

struct DispatchStats {
    u64 dispatches = 0, indirect = 0, one_group = 0, workgroups = 0, pairs = 0;
    u64 diff_hist[1u << 5]{};
    std::unordered_map<u64, KernelStats> kernels;
    u64 runs = 0;                 // runs of the same pipeline in order
    u64 run_len_hist[6]{};        // 1, 2-3, 4-7, 8-15, 16-63, 64+
    u64 dispatches_in_runs_ge4 = 0;
    u64 cur_run = 0;
    u64 last_pipeline = 0;
    u64 windows = 0, sum_kernels = 0, max_kernels = 0;
    std::set<u64> win_kernels;
    u64 win_count = 0;
};

DrawStats ds;
DispatchStats cs;
DrawKey last_depth_key;
bool have_last_depth = false;
u64 last_depth_sequence = 0, op_sequence = 0;
u64 last_pipeline_state = 0, last_full = 0, last_target_viewport = 0;
DispatchKey last_dispatch;
bool have_last_dispatch = false;
u64 published_at_draws = 0;

constexpr u64 WindowDraws = 20000;

u64 RunBucketIndex(u64 len) {
    return len <= 1 ? 0 : len <= 3 ? 1 : len <= 7 ? 2 : len <= 15 ? 3 : len <= 63 ? 4 : 5;
}

void EndDispatchRun() {
    if (cs.cur_run != 0) {
        ++cs.runs;
        ++cs.run_len_hist[RunBucketIndex(cs.cur_run)];
        if (cs.cur_run >= 4) {
            cs.dispatches_in_runs_ge4 += cs.cur_run;
        }
        cs.cur_run = 0;
    }
}

json Report() {
    json out;
    {
        json d;
        d["draws"] = ds.draws;
        d["indirect_draws"] = ds.indirect;
        d["instanced_draws"] = ds.instanced;
        d["instances_total"] = ds.instances;
        d["indices_total"] = ds.indices;
        d["with_complete_record_bindings"] = ds.with_complete_bindings;
        d["with_geometry_handles_from_engine_buffers"] = ds.with_geometry_handles;
        d["index_count_buckets_le32_le256_le2048_le16384_more"] = {ds.index_buckets[0], ds.index_buckets[1], ds.index_buckets[2], ds.index_buckets[3], ds.index_buckets[4]};
        d["consecutive_pairs"] = ds.pairs;
        d["runs_of_equal_pipeline_and_fixed_state"] = ds.runs_pipeline_state;
        d["runs_of_equal_pipeline_fixed_state_target_viewport"] = ds.runs_full;
        d["runs_of_equal_target_and_viewport"] = ds.runs_target_viewport;
        d["windows"] = ds.windows;
        d["window_draws"] = WindowDraws;
        const u64 w = std::max<u64>(ds.windows, 1);
        d["distinct_per_window"] = {{"max_pipelines", ds.max_pipeline},
                                    {"max_pipeline_plus_fixed_state", ds.max_pipeline_state},
                                    {"max_target_plus_viewport", ds.max_target_viewport},
                                    {"max_pipeline_fixed_target_viewport", ds.max_full},
                                    {"avg_pipeline_plus_fixed_state", static_cast<double>(ds.sum_pipeline_state) / w},
                                    {"avg_pipeline_fixed_target_viewport", static_cast<double>(ds.sum_full) / w}};
        json diffs = json::array();
        for (u32 m = 0; m < (1u << DiffBits); ++m) {
            if (ds.diff_hist[m] == 0) {
                continue;
            }
            std::string names;
            for (u32 b = 0; b < DiffBits; ++b) {
                if (m & (1u << b)) {
                    names += (names.empty() ? "" : "+");
                    names += DrawDiffNames[b];
                }
            }
            diffs.push_back({{"mask", m}, {"changed", names.empty() ? "nothing" : names}, {"pairs", ds.diff_hist[m]}});
        }
        std::sort(diffs.begin(), diffs.end(), [](const json& a, const json& b) { return a["pairs"].get<u64>() > b["pairs"].get<u64>(); });
        if (diffs.size() > 24) {
            diffs.erase(diffs.begin() + 24, diffs.end());
        }
        d["what_changes_between_consecutive_depth_only_draws"] = std::move(diffs);
        // per class: share of pairs where a class changed
        json by_class = json::object();
        for (u32 b = 0; b < DiffBits; ++b) {
            u64 n = 0;
            for (u32 m = 0; m < (1u << DiffBits); ++m) {
                if (m & (1u << b)) {
                    n += ds.diff_hist[m];
                }
            }
            by_class[DrawDiffNames[b]] = n;
        }
        d["pairs_in_which_class_changed"] = std::move(by_class);
        // pairs that differ only in what a GPU driven list carries per draw (constants, resource ids, user data, geometry)
        constexpr u32 per_draw = (1u << DUserData) | (1u << DResources) | (1u << DRing) | (1u << DGeometry);
        u64 only_per_draw = 0, needs_state = 0;
        for (u32 m = 0; m < (1u << DiffBits); ++m) {
            ((m & ~per_draw) == 0 ? only_per_draw : needs_state) += ds.diff_hist[m];
        }
        d["pairs_differing_only_in_per_draw_data"] = only_per_draw;
        d["pairs_with_a_pipeline_target_viewport_or_fixed_state_change"] = needs_state;
        out["depth_only_draws"] = std::move(d);
    }
    {
        json c;
        c["dispatches"] = cs.dispatches;
        c["indirect_dispatches"] = cs.indirect;
        c["one_workgroup_dispatches"] = cs.one_group;
        c["workgroups_total"] = cs.workgroups;
        c["consecutive_pairs"] = cs.pairs;
        c["distinct_kernels_since_start"] = cs.kernels.size();
        c["windows"] = cs.windows;
        c["distinct_kernels_per_window_max"] = cs.max_kernels;
        c["distinct_kernels_per_window_avg"] = static_cast<double>(cs.sum_kernels) / std::max<u64>(cs.windows, 1);
        c["runs_of_the_same_kernel"] = cs.runs;
        c["run_length_histogram_1_2to3_4to7_8to15_16to63_64plus"] = {cs.run_len_hist[0], cs.run_len_hist[1], cs.run_len_hist[2], cs.run_len_hist[3], cs.run_len_hist[4], cs.run_len_hist[5]};
        c["dispatches_in_runs_of_4_or_more"] = cs.dispatches_in_runs_ge4;
        static const char* const names[5] = {"pipeline", "dims", "user_data", "constant_ring", "resource_ids"};
        json diffs = json::array();
        for (u32 m = 0; m < (1u << 5); ++m) {
            if (cs.diff_hist[m] == 0) {
                continue;
            }
            std::string text;
            for (u32 b = 0; b < 5; ++b) {
                if (m & (1u << b)) {
                    text += (text.empty() ? "" : "+");
                    text += names[b];
                }
            }
            diffs.push_back({{"mask", m}, {"changed", text.empty() ? "nothing" : text}, {"pairs", cs.diff_hist[m]}});
        }
        std::sort(diffs.begin(), diffs.end(), [](const json& a, const json& b) { return a["pairs"].get<u64>() > b["pairs"].get<u64>(); });
        if (diffs.size() > 16) {
            diffs.erase(diffs.begin() + 16, diffs.end());
        }
        c["what_changes_between_consecutive_dispatches"] = std::move(diffs);
        std::vector<std::pair<u64, const std::pair<const u64, KernelStats>*>> order;
        for (const auto& e : cs.kernels) {
            order.push_back({e.second.dispatches, &e});
        }
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        json top = json::array();
        for (size_t i = 0; i < order.size() && i < 24; ++i) {
            const auto& e = *order[i].second;
            char program[24];
            std::snprintf(program, sizeof(program), "%llx", static_cast<unsigned long long>(e.second.address));
            top.push_back({{"pipeline", e.first},
                           {"program_address", program},
                           {"dispatches", e.second.dispatches},
                           {"workgroups", e.second.workgroups},
                           {"one_group", e.second.one_group},
                           {"dim_x_min", e.second.dim_x_min == ~0u ? 0 : e.second.dim_x_min},
                           {"dim_x_max", e.second.dim_x_max},
                           {"runs", e.second.runs},
                           {"same_dims_as_the_previous_of_this_kernel", e.second.same_dims_as_previous_of_kernel},
                           {"same_user_data_as_the_previous_of_this_kernel", e.second.same_user_as_previous_of_kernel}});
        }
        c["top_kernels"] = std::move(top);
        out["dispatches"] = std::move(c);
    }
    out["note"] = "no behaviour change; window = 20000 depth-only draws or 4000 dispatches for the distinct counts";
    return out;
}

void CloseDrawWindow() {
    ++ds.windows;
    ds.max_pipeline = std::max<u64>(ds.max_pipeline, ds.win_pipeline.size());
    ds.max_pipeline_state = std::max<u64>(ds.max_pipeline_state, ds.win_pipeline_state.size());
    ds.max_full = std::max<u64>(ds.max_full, ds.win_full.size());
    ds.max_target_viewport = std::max<u64>(ds.max_target_viewport, ds.win_target_viewport.size());
    ds.sum_pipeline_state += ds.win_pipeline_state.size();
    ds.sum_full += ds.win_full.size();
    ds.win_pipeline.clear();
    ds.win_pipeline_state.clear();
    ds.win_full.clear();
    ds.win_target_viewport.clear();
}

void CloseDispatchWindow() {
    ++cs.windows;
    cs.sum_kernels += cs.win_kernels.size();
    cs.max_kernels = std::max<u64>(cs.max_kernels, cs.win_kernels.size());
    cs.win_kernels.clear();
    cs.win_count = 0;
}

void MaybePublish() {
    if (ds.draws + cs.dispatches >= published_at_draws + 5000) {
        published_at_draws = ds.draws + cs.dispatches;
        Core::DebugServer::Publish("pass_census", Report());
    }
}

} // namespace

void NoteDraw(const NativePlan::DrawPlan& plan, bool depth_only, bool indirect) {
    ++op_sequence;
    ++ds.draws;
    ds.indirect += indirect ? 1 : 0;
    if (depth_only) {
        const DrawKey k = MakeDrawKey(plan);
        ds.instanced += k.num_instances > 1 ? 1 : 0;
        ds.instances += k.num_instances;
        ds.indices += k.num_indices;
        ds.index_buckets[k.num_indices <= 32 ? 0 : k.num_indices <= 256 ? 1 : k.num_indices <= 2048 ? 2 : k.num_indices <= 16384 ? 3 : 4] += 1;
        ds.with_complete_bindings += plan.bindings_complete ? 1 : 0;
        ds.with_geometry_handles += plan.has_geometry_handles ? 1 : 0;
        const u64 pipeline_state = Mix(k.part[DPipeline], k.part[DFixed]);
        const u64 target_viewport = Mix(k.part[DTarget], k.part[DViewport]);
        const u64 full = Mix(pipeline_state, target_viewport);
        ds.win_pipeline.insert(k.part[DPipeline]);
        ds.win_pipeline_state.insert(pipeline_state);
        ds.win_target_viewport.insert(target_viewport);
        ds.win_full.insert(full);
        if (have_last_depth) {
            u32 mask = 0;
            for (u32 b = 0; b < DiffBits; ++b) {
                mask |= (k.part[b] != last_depth_key.part[b] ? 1u : 0u) << b;
            }
            ++ds.diff_hist[mask];
            ++ds.pairs;
            ds.runs_pipeline_state += pipeline_state != last_pipeline_state ? 1 : 0;
            ds.runs_full += full != last_full ? 1 : 0;
            ds.runs_target_viewport += target_viewport != last_target_viewport ? 1 : 0;
        } else {
            ++ds.runs_pipeline_state;
            ++ds.runs_full;
            ++ds.runs_target_viewport;
        }
        last_depth_key = k;
        last_pipeline_state = pipeline_state;
        last_full = full;
        last_target_viewport = target_viewport;
        have_last_depth = true;
        static u64 window_count = 0;
        if (++window_count >= WindowDraws) {
            window_count = 0;
            CloseDrawWindow();
        }
    }
    MaybePublish();
}

void NoteDispatch(const NativePlan::DispatchPlan& plan, bool indirect) {
    ++op_sequence;
    ++cs.dispatches;
    cs.indirect += indirect ? 1 : 0;
    DispatchKey k;
    k.pipeline = reinterpret_cast<u64>(plan.pipeline);
    const u64 groups = u64{plan.cs.dim_x} * plan.cs.dim_y * plan.cs.dim_z;
    k.dims = Mix(Mix(plan.cs.dim_x, plan.cs.dim_y), plan.cs.dim_z);
    u64 u = HashWords(17, plan.stage.user_data.data(), plan.stage.user_data_count);
    u = HashWords(u, plan.stage.flat.data(), plan.stage.flat.size());
    k.user = u;
    u64 r = 19;
    for (const auto& rb : plan.stage.ring) {
        r = Mix(Mix(Mix(r, rb.index), rb.offset), rb.size);
    }
    k.ring = r;
    u64 res = 23;
    for (u32 i = 0; i < plan.binding_count && i < plan.bindings.size(); ++i) {
        const auto& b = plan.bindings[i];
        res = Mix(Mix(Mix(Mix(res, b.stage), b.kind), b.id), b.view_id);
    }
    k.resources = res;
    cs.workgroups += groups;
    cs.one_group += groups == 1 ? 1 : 0;
    KernelStats& ks = cs.kernels[k.pipeline];
    ++ks.dispatches;
    ks.workgroups += groups;
    ks.one_group += groups == 1 ? 1 : 0;
    ks.dim_x_min = std::min(ks.dim_x_min, plan.cs.dim_x);
    ks.dim_x_max = std::max(ks.dim_x_max, plan.cs.dim_x);
    ks.address = plan.cs.address << 8;
    ks.same_dims_as_previous_of_kernel += ks.last_dims == k.dims ? 1 : 0;
    ks.same_user_as_previous_of_kernel += ks.last_user == k.user ? 1 : 0;
    ks.last_dims = k.dims;
    ks.last_user = k.user;
    cs.win_kernels.insert(k.pipeline);
    if (have_last_dispatch) {
        u32 mask = 0;
        mask |= (k.pipeline != last_dispatch.pipeline ? 1u : 0u) << 0;
        mask |= (k.dims != last_dispatch.dims ? 1u : 0u) << 1;
        mask |= (k.user != last_dispatch.user ? 1u : 0u) << 2;
        mask |= (k.ring != last_dispatch.ring ? 1u : 0u) << 3;
        mask |= (k.resources != last_dispatch.resources ? 1u : 0u) << 4;
        ++cs.diff_hist[mask];
        ++cs.pairs;
    }
    if (have_last_dispatch && k.pipeline == last_dispatch.pipeline) {
        ++cs.cur_run;
    } else {
        EndDispatchRun();
        cs.cur_run = 1;
        ++ks.runs;
    }
    last_dispatch = k;
    have_last_dispatch = true;
    if (++cs.win_count >= 4000) {
        CloseDispatchWindow();
    }
    MaybePublish();
}

} // namespace Core::PassCensus
