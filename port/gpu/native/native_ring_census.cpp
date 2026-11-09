// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <tuple>
#include "native/native_content.h"
#include "native/native_ring_census.h"
#include "native/native_updates.h"
#include "native/native_untracked_ring.h"

namespace Core::RingCensus {

namespace {

constexpr u64 GranuleShift = 4; // 16 bytes
constexpr u8 BitAllocated = 1;
constexpr u8 BitOurs = 0x80;
constexpr u8 BitsAnnounced = 0x7e;
constexpr u32 WriterBuckets = 6; // writers 0, 1, 2, 3, 4, other

std::atomic<std::atomic<u8>*> g_flags{nullptr};
std::atomic<std::atomic<u8>*> g_origin{nullptr}; // per granule: the index of the allocation's call site in g_site_list (0: unknown or ours)

// call sites of the engine's allocations (index 1..62; 63 = all others), written under the allocator lock, read by the report
constexpr u32 MaxOrigins = 64;
std::array<std::atomic<u32>, MaxOrigins> g_site_list{};
std::atomic<u32> g_site_count{1};
std::array<std::atomic<u64>, MaxOrigins> c_origin_allocs{}, c_origin_bytes{}, c_origin_silent_used{}, c_origin_announced_used{};

u8 OriginOf(u32 site) {
    if (site == 0) {
        return 0;
    }
    const u32 n = g_site_count.load(std::memory_order_relaxed);
    for (u32 i = 1; i < n; ++i) {
        if (g_site_list[i].load(std::memory_order_relaxed) == site) {
            return static_cast<u8>(i);
        }
    }
    if (n < MaxOrigins - 1) {
        g_site_list[n].store(site, std::memory_order_relaxed);
        g_site_count.store(n + 1, std::memory_order_release);
        return static_cast<u8>(n);
    }
    return MaxOrigins - 1;
}
std::atomic<u64> g_base{0}, g_size{0};

// allocator side (under the frame allocator lock)
u64 g_last_end = 0;
std::atomic<u64> c_allocs{0}, c_alloc_bytes{0}, c_laps{0}, c_before_ring{0}, c_outside_ring{0}, c_ours_allocs{0}, c_ours_bytes{0};
// gaps between one allocation's end and the next one's start: the engine's allocator rounds small blocks up (at least 1,088 bytes, Ghidra 2026-10-19), so most gaps are that;
// a gap far above the rounding is room the hook did not hand out (another allocator path, or skipped)
std::atomic<u64> c_gap[4][2]{}; // [<256, <2 KiB, <64 KiB, larger][count, bytes]

std::atomic<u64> c_announced[WriterBuckets][2]{}; // ranges, bytes inside the ring

struct SiteStats {
    std::atomic<u64> uses{0}, bytes{0}, announced{0}, ours{0}, silent{0}, never{0}, uses_with_silent{0}, uses_with_never{0}, uses_all_known{0};
    std::atomic<u64> by_writer[WriterBuckets]{};
    std::atomic<u64> by_size[3][3]{}; // use size [<=4 KiB, <=64 KiB, larger] x [uses, bytes, silent bytes]
};
std::array<SiteStats, SiteCount> g_sites;

struct Sample {
    u32 site;
    u64 offset; // from the ring base
    u64 size;
    u64 silent;
    u64 never;
};
std::atomic<u64> c_map_path[8][2][8][4]{}; // [resource kind][subresource 0 / other][map type][path]
thread_local u32 t_sync_caller = 0;

// wide raster ring reads (above 64 KiB) by who asked: (bind site, program hash, stage) -> totals; silent bytes split by whether a Map that is still open (mapped, not unmapped yet) covers them
struct WideStats {
    u64 uses{}, bytes{}, silent{}, silent_open{}, never{}, largest{}, example_id{};
};
std::mutex g_wide_mutex;
std::map<std::tuple<std::string, u64, u32>, WideStats> g_wide;
std::atomic<u64> c_sync_caller[CallerCount][4]{}; // raster sync uses by caller: [uses, bytes, silent bytes, never bytes]

std::mutex g_sample_mutex;
std::vector<Sample> g_samples;
constexpr size_t MaxSamples = 40;

u32 Bucket(u32 writer) {
    return std::min<u32>(writer, WriterBuckets - 1);
}

/// Padding between blocks (the allocator's rounding) is set to 0: never handed out. Only whole granules are cleared (the edge ones belong to a block).
void ClearPadding(std::atomic<u8>* f, u64 lo, u64 hi) {
    const u64 first = (lo + (1ull << GranuleShift) - 1) >> GranuleShift, end = hi >> GranuleShift;
    for (u64 g = first; g < end; ++g) {
        f[g].store(0, std::memory_order_relaxed);
    }
}

/// The flags array, made the first time the ring is known (called under the allocator lock).
std::atomic<u8>* EnsureFlags() {
    if (auto* f = g_flags.load(std::memory_order_acquire)) {
        return f;
    }
    const auto& ring = VideoCore::UntrackedRing::Get();
    if (!ring.Active()) {
        return nullptr;
    }
    const u64 base = ring.Start(), size = ring.End() - ring.Start();
    auto* f = new std::atomic<u8>[size >> GranuleShift]; // kept for the life of the process
    auto* o = new std::atomic<u8>[size >> GranuleShift];
    for (u64 i = 0; i < (size >> GranuleShift); ++i) {
        f[i].store(0, std::memory_order_relaxed);
        o[i].store(0, std::memory_order_relaxed);
    }
    g_origin.store(o, std::memory_order_release);
    g_base.store(base, std::memory_order_relaxed);
    g_size.store(size, std::memory_order_relaxed);
    g_flags.store(f, std::memory_order_release);
    return f;
}

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteAlloc(u64 block, u64 size, bool ours, u32 site) {
    if (!On() || block == 0 || size == 0) {
        return;
    }
    auto* f = EnsureFlags();
    if (f == nullptr) {
        c_before_ring.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed), rsize = g_size.load(std::memory_order_relaxed);
    if (block < base || block + size > base + rsize) {
        c_outside_ring.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 rel = block - base;
    c_allocs.fetch_add(1, std::memory_order_relaxed);
    c_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    if (rel < g_last_end) {
        c_laps.fetch_add(1, std::memory_order_relaxed);
        ClearPadding(f, g_last_end, rsize); // the tail the lap skipped and the head before the first block
        ClearPadding(f, 0, rel);
    } else if (rel > g_last_end && g_last_end != 0) {
        const u64 gap = rel - g_last_end;
        ClearPadding(f, g_last_end, rel);
        const u32 k = gap < 256 ? 0 : gap < 2048 ? 1 : gap < 65536 ? 2 : 3;
        c_gap[k][0].fetch_add(1, std::memory_order_relaxed);
        c_gap[k][1].fetch_add(gap, std::memory_order_relaxed);
    }
    g_last_end = rel + size;
    if (ours) {
        c_ours_allocs.fetch_add(1, std::memory_order_relaxed);
        c_ours_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    const u8 value = ours ? static_cast<u8>(BitAllocated | BitOurs) : BitAllocated;
    const u8 origin = ours ? 0 : OriginOf(site);
    c_origin_allocs[origin].fetch_add(1, std::memory_order_relaxed);
    c_origin_bytes[origin].fetch_add(size, std::memory_order_relaxed);
    auto* o = g_origin.load(std::memory_order_relaxed);
    const u64 first = rel >> GranuleShift, last = (rel + size - 1) >> GranuleShift;
    for (u64 g = first; g <= last; ++g) {
        f[g].store(value, std::memory_order_relaxed);
        o[g].store(origin, std::memory_order_relaxed);
    }
}

void NoteAnnounce(u32 writer, u64 address, u64 size) {
    if (!On() || size == 0) {
        return;
    }
    auto* f = g_flags.load(std::memory_order_acquire);
    if (f == nullptr) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed), end = base + g_size.load(std::memory_order_relaxed);
    const u64 lo = std::max(address, base), hi = std::min(address + size, end);
    if (lo >= hi) {
        return;
    }
    const u32 b = Bucket(writer);
    c_announced[b][0].fetch_add(1, std::memory_order_relaxed);
    c_announced[b][1].fetch_add(hi - lo, std::memory_order_relaxed);
    const u8 bit = static_cast<u8>(2u << b);
    const u64 first = (lo - base) >> GranuleShift, last = (hi - 1 - base) >> GranuleShift;
    for (u64 g = first; g <= last; ++g) {
        f[g].fetch_or(bit, std::memory_order_relaxed);
    }
}

void NoteUse(Site site, u64 address, u64 size) {
    if (!On() || size == 0 || site >= SiteCount) {
        return;
    }
    auto* f = g_flags.load(std::memory_order_acquire);
    if (f == nullptr) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed), end = base + g_size.load(std::memory_order_relaxed);
    const u64 lo = std::max(address, base), hi = std::min(address + size, end);
    if (lo >= hi) {
        return;
    }
    auto& s = g_sites[site];
    u64 announced = 0, ours = 0, silent = 0, never = 0;
    u64 by_writer[WriterBuckets]{};
    auto* o = g_origin.load(std::memory_order_acquire);
    const u64 first = (lo - base) >> GranuleShift, last = (hi - 1 - base) >> GranuleShift;
    for (u64 g = first; g <= last; ++g) {
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)), g_hi = std::min(hi, base + ((g + 1) << GranuleShift));
        const u64 n = g_hi - g_lo;
        const u8 v = f[g].load(std::memory_order_relaxed);
        if ((v & BitsAnnounced) != 0) {
            announced += n;
            c_origin_announced_used[o[g].load(std::memory_order_relaxed)].fetch_add(n, std::memory_order_relaxed);
            for (u32 w = 0; w < WriterBuckets; ++w) {
                if ((v & (2u << w)) != 0) {
                    by_writer[w] += n;
                }
            }
        } else if ((v & BitOurs) != 0) {
            ours += n; // our redirect's own copy wrote it
        } else if (v == BitAllocated) {
            silent += n;
            c_origin_silent_used[o[g].load(std::memory_order_relaxed)].fetch_add(n, std::memory_order_relaxed);
        } else {
            never += n;
        }
    }
    s.uses.fetch_add(1, std::memory_order_relaxed);
    s.bytes.fetch_add(hi - lo, std::memory_order_relaxed);
    if (site == SiteRasterSync || site == SiteRasterPreSync) {
        const u32 k = std::min<u32>(t_sync_caller, CallerCount - 1);
        c_sync_caller[k][0].fetch_add(1, std::memory_order_relaxed);
        c_sync_caller[k][1].fetch_add(hi - lo, std::memory_order_relaxed);
        c_sync_caller[k][2].fetch_add(silent, std::memory_order_relaxed);
        c_sync_caller[k][3].fetch_add(never, std::memory_order_relaxed);
    }
    s.announced.fetch_add(announced, std::memory_order_relaxed);
    s.ours.fetch_add(ours, std::memory_order_relaxed);
    {
        const u64 n = hi - lo;
        const u32 k = n <= 4096 ? 0 : n <= 65536 ? 1 : 2;
        s.by_size[k][0].fetch_add(1, std::memory_order_relaxed);
        s.by_size[k][1].fetch_add(n, std::memory_order_relaxed);
        s.by_size[k][2].fetch_add(silent, std::memory_order_relaxed);
    }
    s.silent.fetch_add(silent, std::memory_order_relaxed);
    s.never.fetch_add(never, std::memory_order_relaxed);
    for (u32 w = 0; w < WriterBuckets; ++w) {
        if (by_writer[w] != 0) {
            s.by_writer[w].fetch_add(by_writer[w], std::memory_order_relaxed);
        }
    }
    if (silent != 0) {
        s.uses_with_silent.fetch_add(1, std::memory_order_relaxed);
    }
    if (never != 0) {
        s.uses_with_never.fetch_add(1, std::memory_order_relaxed);
    }
    if ((site == SiteRasterSync || site == SiteRasterPreSync) && hi - lo > 65536) {
        // which silent bytes lie in a Map window that is still open (the engine writes it now, for a later record)
        u64 silent_open = 0;
        if (silent != 0) {
            u64 run_lo = 0, run_hi = 0;
            const auto flush = [&] {
                if (run_hi > run_lo && Core::NativeUpdates::OpenMapOverlaps(run_lo, run_hi - run_lo)) {
                    silent_open += run_hi - run_lo;
                }
                run_lo = run_hi = 0;
            };
            for (u64 g = first; g <= last; ++g) {
                if (f[g].load(std::memory_order_relaxed) != BitAllocated) {
                    flush();
                    continue;
                }
                const u64 g_lo = std::max(lo, base + (g << GranuleShift)), g_hi = std::min(hi, base + ((g + 1) << GranuleShift));
                if (run_hi == g_lo && run_hi != 0) {
                    run_hi = g_hi;
                } else {
                    flush();
                    run_lo = g_lo;
                    run_hi = g_hi;
                }
            }
            flush();
        }
        Core::NativeContent::BindContext ctx{};
        const bool has = Core::NativeContent::CurrentBindContext(ctx);
        const std::string where = has && ctx.site != nullptr ? std::string(ctx.site) : std::string("none");
        std::scoped_lock lock{g_wide_mutex};
        auto& w = g_wide[{where, has ? ctx.program : 0, has ? ctx.stage : 0}];
        ++w.uses;
        w.bytes += hi - lo;
        w.silent += silent;
        w.silent_open += silent_open;
        w.never += never;
        w.largest = std::max<u64>(w.largest, hi - lo);
        if (w.example_id == 0 && has) {
            w.example_id = ctx.engine_id;
        }
    }
    if (silent == 0 && never == 0) {
        s.uses_all_known.fetch_add(1, std::memory_order_relaxed);
    } else if (silent != 0) {
        std::scoped_lock lock{g_sample_mutex};
        if (g_samples.size() < MaxSamples) {
            g_samples.push_back({static_cast<u32>(site), lo - base, hi - lo, silent, never});
        }
    }
}

void SetSyncCaller(SyncCaller caller) noexcept {
    t_sync_caller = caller;
}

void NoteMapPath(u32 kind, u64 subresource, u32 map_type, u32 path) {
    if (!On()) {
        return;
    }
    c_map_path[std::min<u32>(kind, 7)][subresource != 0 ? 1 : 0][std::min<u32>(map_type, 7)][path & 3].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j;
    j["on"] = On();
    j["note"] = "TLG_RING_CENSUS (design review 2026-11-09 F1): frame ring bytes the GPU reads, sorted by whether a hook announced them since the frame allocator handed them out; "
                "16-byte granules, an announcement marks every granule it touches";
    j["ring"] = {{"base", fmt::format("{:#x}", g_base.load())}, {"size", g_size.load()}};
    j["allocator"] = {{"allocs", c_allocs.load()}, {"bytes", c_alloc_bytes.load()}, {"laps", c_laps.load()}, {"ours_allocs", c_ours_allocs.load()}, {"ours_bytes", c_ours_bytes.load()},
                      {"before_ring_known", c_before_ring.load()}, {"outside_ring", c_outside_ring.load()}};
    static constexpr const char* gap_names[4] = {"under_256", "under_2k", "under_64k", "larger"};
    for (u32 k = 0; k < 4; ++k) {
        j["allocator"]["gaps"][gap_names[k]] = {{"count", c_gap[k][0].load()}, {"bytes", c_gap[k][1].load()}};
    }
    static constexpr const char* writer_names[WriterBuckets] = {"0", "1", "2", "3", "4", "other"};
    for (u32 w = 0; w < WriterBuckets; ++w) {
        j["announced_in_ring"][writer_names[w]] = {{"ranges", c_announced[w][0].load()}, {"bytes", c_announced[w][1].load()}};
    }
    {
        auto& list = j["engine_alloc_sites"] = nlohmann::json::array();
        const u32 n = g_site_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < MaxOrigins; ++i) {
            if (i != 0 && i != MaxOrigins - 1 && i >= n) {
                continue;
            }
            const std::string name = i == 0 ? std::string("unknown_or_ours") : i == MaxOrigins - 1 ? std::string("other_sites") : fmt::format("{:#x}", g_site_list[i].load());
            list.push_back({{"site", name}, {"allocs", c_origin_allocs[i].load()}, {"bytes", c_origin_bytes[i].load()},
                            {"used_announced_bytes", c_origin_announced_used[i].load()}, {"used_SILENT_bytes", c_origin_silent_used[i].load()}});
        }
    }
    {
        static constexpr const char* path_names[4] = {"buffer_announced", "texture_record_announced", "texture_refused_announced", "NOT_HANDLED_NOT_ANNOUNCED"};
        auto& m = j["maps"] = nlohmann::json::object();
        for (u32 k = 0; k < 8; ++k) {
            for (u32 s = 0; s < 2; ++s) {
                for (u32 t = 0; t < 8; ++t) {
                    for (u32 p = 0; p < 4; ++p) {
                        if (const u64 n = c_map_path[k][s][t][p].load()) {
                            m[fmt::format("kind{} {} type{}", k, s == 0 ? "sub0" : "sub>0", t)][path_names[p]] = n;
                        }
                    }
                }
            }
        }
    }
    {
        static constexpr const char* caller_names[CallerCount] = {"unknown", "presync", "immutable", "bind_read", "bind_written", "bind_texel", "download"};
        for (u32 k = 0; k < CallerCount; ++k) {
            if (const u64 n = c_sync_caller[k][0].load()) {
                j["raster_sync_by_caller"][caller_names[k]] = {{"uses", n}, {"bytes", c_sync_caller[k][1].load()}, {"SILENT_BYTES", c_sync_caller[k][2].load()}, {"never_bytes", c_sync_caller[k][3].load()}};
            }
        }
    }
    {
        std::scoped_lock lock{g_wide_mutex};
        auto& list = j["wide_raster_reads"] = nlohmann::json::array();
        for (const auto& [key, w] : g_wide) {
            list.push_back({{"bind_site", std::get<0>(key)}, {"program", fmt::format("{:#010x}", std::get<1>(key))}, {"stage", std::get<2>(key)}, {"uses", w.uses}, {"bytes", w.bytes},
                            {"SILENT_BYTES", w.silent}, {"silent_in_open_maps", w.silent_open}, {"never_bytes", w.never}, {"largest", w.largest}, {"example_engine_id", fmt::format("{:#x}", w.example_id)}});
        }
    }
    static constexpr const char* site_names[SiteCount] = {"front_constant", "front_geometry", "raster_sync", "raster_presync"};
    for (u32 i = 0; i < SiteCount; ++i) {
        const auto& s = g_sites[i];
        auto& e = j["uses"][site_names[i]];
        e["uses"] = s.uses.load();
        e["bytes"] = s.bytes.load();
        e["announced_bytes"] = s.announced.load();
        e["ours_bytes"] = s.ours.load();
        e["SILENT_BYTES"] = s.silent.load();
        e["NEVER_HANDED_OUT_BYTES"] = s.never.load();
        e["uses_all_known"] = s.uses_all_known.load();
        static constexpr const char* size_names[3] = {"to_4k", "to_64k", "larger"};
        for (u32 k = 0; k < 3; ++k) {
            e["by_use_size"][size_names[k]] = {{"uses", s.by_size[k][0].load()}, {"bytes", s.by_size[k][1].load()}, {"silent_bytes", s.by_size[k][2].load()}};
        }
        e["uses_with_silent"] = s.uses_with_silent.load();
        e["uses_with_never_handed_out"] = s.uses_with_never.load();
        for (u32 w = 0; w < WriterBuckets; ++w) {
            e["announced_bytes_by_writer"][writer_names[w]] = s.by_writer[w].load();
        }
    }
    {
        std::scoped_lock lock{g_sample_mutex};
        auto& list = j["samples"] = nlohmann::json::array();
        for (const auto& smp : g_samples) {
            list.push_back({{"site", site_names[smp.site]}, {"ring_offset", fmt::format("{:#x}", smp.offset)}, {"size", smp.size}, {"silent", smp.silent}, {"never", smp.never}});
        }
    }
    return j;
}

} // namespace Core::RingCensus
