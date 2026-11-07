// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_split_census.h"

namespace Core::SplitCensus {

bool g_on = [] {
    const char* e = std::getenv("TLG_SPLIT_CENSUS");
    return e != nullptr && e[0] == '1';
}();
thread_local u32 t_mask = 0;

namespace {

struct MaskStat {
    u64 n{};
    u64 ns{};
};
struct Group {
    u64 n{};
    u64 ns{};
    std::map<u32, MaskStat> masks;
};

constexpr const char* BitNames[BitCount] = {"buf_obtain", "buf_sync",   "buf_dirty", "buf_memo_erase", "buf_placed", "buf_stream", "tex_find", "tex_refresh",
                                            "tex_sampler", "tex_register", "pipeline", "barrier",       "indexed",    "gpu_ranges", "object_notes", "image_index", "label",      "commands", "guest_write"};
constexpr const char* KindNames[4] = {"draw", "draw_depth_only", "dispatch", "closure"};
constexpr const char* ClassNames[5] = {"graphics_queue", "compute_queue_1", "queue_2", "queues_3_and_up", "no_queue_task"};

// Raster thread only.
std::map<std::tuple<u32, u32, const char*, u32>, Group> groups;
u64 total_ns = 0;
u64 total_n = 0;
u64 ns_by_cache_class[BitCount]{};      ///< time of operations that touched the class (an operation counts for every class it touched)
u64 ns_no_cache = 0;                    ///< operations that touched no cache class
u64 ns_one_cache = 0;                   ///< operations that touched exactly one cache class
u64 ns_no_cache_by_kind[4]{};
u64 ns_by_kind[4]{};
u64 ns_by_class_kind[5][4]{};
u64 ns_no_cache_by_class_kind[5][4]{};

std::string MaskText(u32 mask) {
    if (mask == 0) {
        return "none";
    }
    std::string text;
    for (u32 b = 0; b < BitCount; ++b) {
        if ((mask & (1u << b)) != 0) {
            if (!text.empty()) {
                text += '+';
            }
            text += BitNames[b];
        }
    }
    return text;
}

} // namespace

void Record(u32 kind, u32 queue_class, const char* file, u32 line, u32 mask, u64 ns) {
    if (queue_class > 4) {
        queue_class = 4;
    }
    // drawing and dispatching groups are by kind and queue only; closures by queueing site
    Group& group = groups[{kind, queue_class, kind == 3 ? file : nullptr, kind == 3 ? line : 0}];
    ++group.n;
    group.ns += ns;
    MaskStat& m = group.masks[mask];
    ++m.n;
    m.ns += ns;
    ++total_n;
    total_ns += ns;
    ns_by_kind[kind] += ns;
    ns_by_class_kind[queue_class][kind] += ns;
    const u32 cache = mask & CacheMask;
    if (cache == 0) {
        ns_no_cache += ns;
        ns_no_cache_by_kind[kind] += ns;
        ns_no_cache_by_class_kind[queue_class][kind] += ns;
    } else if ((cache & (cache - 1)) == 0) {
        ns_one_cache += ns;
    }
    for (u32 b = 0; b < BitCount; ++b) {
        if ((cache & (1u << b)) != 0) {
            ns_by_cache_class[b] += ns;
        }
    }
}

void PublishAndReset() {
    if (!g_on) {
        return;
    }
    using json = nlohmann::json;
    json out = json::object();
    out["note"] = "per ~5 s interval, raster thread; an operation counts for every cache class it touched; cache classes are the shared state a lane would need a lock for";
    out["operations"] = total_n;
    out["raster_ms_in_operations"] = total_ns / 1e6;
    out["ms_no_cache_touched"] = ns_no_cache / 1e6;
    out["ms_exactly_one_cache_class"] = ns_one_cache / 1e6;
    json by_cache = json::object();
    for (u32 b = 0; b < BitCount; ++b) {
        if ((CacheMask & (1u << b)) != 0) {
            by_cache[BitNames[b]] = ns_by_cache_class[b] / 1e6;
        }
    }
    out["ms_in_operations_touching_class"] = by_cache;
    json kinds = json::object();
    for (u32 k = 0; k < 4; ++k) {
        kinds[KindNames[k]] = {{"ms", ns_by_kind[k] / 1e6}, {"ms_no_cache", ns_no_cache_by_kind[k] / 1e6}};
    }
    out["by_kind"] = kinds;
    json classes = json::object();
    for (u32 c = 0; c < 5; ++c) {
        json row = json::object();
        for (u32 k = 0; k < 4; ++k) {
            if (ns_by_class_kind[c][k] != 0) {
                row[KindNames[k]] = {{"ms", ns_by_class_kind[c][k] / 1e6}, {"ms_no_cache", ns_no_cache_by_class_kind[c][k] / 1e6}};
            }
        }
        classes[ClassNames[c]] = row;
    }
    out["by_queue_and_kind"] = classes;
    // the groups by time, with their top masks
    std::vector<std::pair<u64, const std::pair<const std::tuple<u32, u32, const char*, u32>, Group>*>> order;
    for (const auto& entry : groups) {
        order.push_back({entry.second.ns, &entry});
    }
    std::ranges::sort(order, [](const auto& a, const auto& b) { return a.first > b.first; });
    json list = json::array();
    for (size_t i = 0; i < order.size() && i < 40; ++i) {
        const auto& [key, group] = *order[i].second;
        json row = {{"kind", KindNames[std::get<0>(key)]},
                    {"queue", ClassNames[std::get<1>(key)]},
                    {"n", group.n},
                    {"ms", group.ns / 1e6}};
        if (std::get<2>(key) != nullptr) {
            row["site"] = std::string{std::get<2>(key)} + ":" + std::to_string(std::get<3>(key));
        }
        std::vector<std::pair<u64, u32>> masks;
        for (const auto& [mask, stat] : group.masks) {
            masks.push_back({stat.ns, mask});
        }
        std::ranges::sort(masks, std::greater<>{});
        json top = json::array();
        for (size_t m = 0; m < masks.size() && m < 6; ++m) {
            const MaskStat& stat = group.masks.at(masks[m].second);
            top.push_back({{"touched", MaskText(masks[m].second)}, {"n", stat.n}, {"ms", stat.ns / 1e6}});
        }
        row["masks"] = top;
        list.push_back(std::move(row));
    }
    out["groups"] = list;
    Core::DebugServer::Publish("split_census", std::move(out));
    groups.clear();
    total_ns = total_n = ns_no_cache = ns_one_cache = 0;
    std::fill(std::begin(ns_by_cache_class), std::end(ns_by_cache_class), 0);
    std::fill(std::begin(ns_no_cache_by_kind), std::end(ns_no_cache_by_kind), 0);
    std::fill(std::begin(ns_by_kind), std::end(ns_by_kind), 0);
    for (u32 c = 0; c < 5; ++c) {
        std::fill(std::begin(ns_by_class_kind[c]), std::end(ns_by_class_kind[c]), 0);
        std::fill(std::begin(ns_no_cache_by_class_kind[c]), std::end(ns_no_cache_by_class_kind[c]), 0);
    }
}

} // namespace Core::SplitCensus
