// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <nlohmann/json.hpp>
#include "core/guest_hooks.h"
#include "core/native_resources.h"
#include "core/native_updates.h"

namespace Core::NativeUpdates {

std::atomic<bool> enabled{[] {
    const char* e = std::getenv("TLG_UPDATE_AUDIT");
    return e != nullptr && *e != 0 && *e != '0';
}()};

namespace {

constexpr size_t WindowCount = 4096;
constexpr u64 NowWindowNs = 200'000'000; ///< a write this soon after the Map of its own thread belongs to that Map

struct Window {
    u64 begin{}, end{};
    u64 id{};
    u64 time_ns{};
    u64 thread{};
    u32 usage{};
    u32 type{};
    bool in_ring{}; ///< the engine handed out a window of the frame ring, not the buffer's own memory
};

std::mutex mutex;
std::array<Window, WindowCount> windows{};
u64 window_count = 0;

struct MapCounters {
    std::array<std::atomic<u64>, 8> by_type{};   ///< D3D11 map type 1..5 (0 and 6, 7 other)
    std::array<std::atomic<u64>, 4> by_usage{};  ///< 0 default 1 immutable 2 dynamic 3 staging
    std::atomic<u64> in_ring{}, own_memory{};
    std::atomic<u64> total{};
} maps;

struct FaultCounters {
    std::atomic<u64> total{};
    std::atomic<u64> covered_now{}, covered_earlier{}, covered_other_thread{}, uncovered{};
    std::array<std::atomic<u64>, 16> uncovered_by_class{};
    std::array<std::atomic<u64>, 16> covered_by_class{};
} faults;

struct Site {
    u64 count{};
    u64 sample_address{};
    u32 usage_class{};
};
struct SiteKey {
    u64 rip, st0, st1;
    bool operator<(const SiteKey& o) const {
        return std::tie(rip, st0, st1) < std::tie(o.rip, o.st0, o.st1);
    }
};
std::map<SiteKey, Site> uncovered_sites; ///< by faulting instruction and the first two stack words (the caller of a leaf function such as memcpy)

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

u64 ThreadKey() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace

void Start() {
    enabled.store(true, std::memory_order_relaxed);
}

void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller) {
    (void)caller;
    maps.total.fetch_add(1, std::memory_order_relaxed);
    maps.by_type[map_type < 8 ? map_type : 7].fetch_add(1, std::memory_order_relaxed);
    maps.by_usage[usage < 4 ? usage : 3].fetch_add(1, std::memory_order_relaxed);
    const bool in_ring = address != memory;
    (in_ring ? maps.in_ring : maps.own_memory).fetch_add(1, std::memory_order_relaxed);
    if (address == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    windows[window_count++ % WindowCount] = {address, address + size, resource_id, NowNs(), ThreadKey(), usage, map_type, in_ring};
}

void NoteFault(u64 address, u64 size) {
    faults.total.fetch_add(1, std::memory_order_relaxed);
    const u32 usage_class = NativeResources::UsageClassOf(address) & 15;
    const u64 now = NowNs();
    const u64 thread = ThreadKey();
    enum class Kind { None, Now, Earlier, Other } kind = Kind::None;
    {
        std::scoped_lock lock{mutex};
        const u64 count = std::min<u64>(window_count, WindowCount);
        for (u64 i = 0; i < count && kind != Kind::Now; ++i) {
            const Window& w = windows[(window_count - 1 - i) % WindowCount];
            if (address < w.begin || address + size > w.end) {
                continue;
            }
            if (w.thread != thread) {
                kind = kind == Kind::None ? Kind::Other : kind;
            } else if (now - w.time_ns <= NowWindowNs) {
                kind = Kind::Now;
            } else if (kind != Kind::Now) {
                kind = Kind::Earlier;
            }
        }
        if (kind == Kind::None) {
            Site& site = uncovered_sites[{GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)}];
            if (site.count++ == 0) {
                site.sample_address = address;
                site.usage_class = usage_class;
            }
        }
    }
    switch (kind) {
    case Kind::Now:
        faults.covered_now.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Earlier:
        faults.covered_earlier.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Other:
        faults.covered_other_thread.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::None:
        faults.uncovered.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    (kind == Kind::None ? faults.uncovered_by_class : faults.covered_by_class)[usage_class].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = enabled.load();
    auto& m = j["maps_of_buffers"] = nlohmann::json::object();
    m["total"] = maps.total.load();
    m["into_the_buffers_own_memory"] = maps.own_memory.load();
    m["into_a_frame_ring_window"] = maps.in_ring.load();
    constexpr const char* type_names[8] = {"0", "read", "write", "read_write", "write_discard", "write_no_overwrite", "6", "other"};
    for (u32 i = 0; i < 8; ++i) {
        if (const u64 n = maps.by_type[i].load()) {
            m["by_map_type"][type_names[i]] = n;
        }
    }
    constexpr const char* usage_names[4] = {"default", "immutable", "dynamic", "staging"};
    for (u32 i = 0; i < 4; ++i) {
        if (const u64 n = maps.by_usage[i].load()) {
            m["by_buffer_usage"][usage_names[i]] = n;
        }
    }
    auto& f = j["cpu_write_faults"] = nlohmann::json::object();
    const u64 total = faults.total.load();
    f["total"] = total;
    f["inside_a_map_window_of_the_same_thread_within_200ms"] = faults.covered_now.load();
    f["inside_an_older_map_window_of_the_same_thread"] = faults.covered_earlier.load();
    f["inside_a_map_window_of_another_thread"] = faults.covered_other_thread.load();
    f["in_no_map_window"] = faults.uncovered.load();
    if (total != 0) {
        f["percent_announced_by_a_map_on_the_same_thread_within_200ms"] = 100.0 * static_cast<double>(faults.covered_now.load()) / static_cast<double>(total);
    }
    for (u32 i = 0; i < 16; ++i) {
        const u64 covered = faults.covered_by_class[i].load();
        const u64 uncovered = faults.uncovered_by_class[i].load();
        if (covered + uncovered != 0) {
            const std::string name = NativeResources::UsageClassNameOf(i);
            f["by_memory_class"][name] = {{"in_a_map_window", covered}, {"in_no_map_window", uncovered}};
        }
    }
    std::scoped_lock lock{mutex};
    std::vector<std::pair<SiteKey, Site>> sites(uncovered_sites.begin(), uncovered_sites.end());
    std::ranges::sort(sites, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    auto& list = f["unannounced_writes_by_faulting_instruction"] = nlohmann::json::array();
    for (size_t i = 0; i < sites.size() && i < 24; ++i) {
        list.push_back({{"rip", sites[i].first.rip}, {"stack0", sites[i].first.st0}, {"stack1", sites[i].first.st1}, {"faults", sites[i].second.count}, {"sample_address", sites[i].second.sample_address},
                        {"memory_class", NativeResources::UsageClassNameOf(sites[i].second.usage_class)}});
    }
    return j;
}

} // namespace Core::NativeUpdates
