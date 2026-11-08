// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_n0_census.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include "native/native_content.h"

namespace Core::N0 {

bool g_on = [] {
    const char* e = std::getenv("TLG_N0_CENSUS");
    return e != nullptr && e[0] == '1';
}();
thread_local SyncSite t_sync_site{SyncArenaRead};
thread_local u64 t_hash_ns{0};
thread_local u32 t_effect{0};

namespace {
constexpr const char* SyncNames[SyncSiteCount] = {"immutable_whole", "arena_read", "arena_written", "arena_texel", "download", "pre_sync"};
constexpr const char* CopyNames[CopySiteCount] = {"object_copy", "stream_copy", "stream_copy_pass_open", "for_image", "zero_copy_direct", "staging"};

struct SyncStat {
    std::atomic<u64> calls{0}, noop_calls{0}, requested{0}, uploaded{0}, ns{0}, noop_ns{0};
};
struct CopyStat {
    std::atomic<u64> calls{0}, bytes{0}, ns{0}, hashed{0}, same_as_last{0}, same_bytes{0}, small_calls{0};
};
std::array<SyncStat, SyncSiteCount> sync_stats;
std::array<CopyStat, CopySiteCount> copy_stats;
std::array<std::atomic<u64>, 24> applied_first_log2{}, applied_again_log2{};
std::atomic<u64> first_ns{0};
std::array<std::atomic<u64>, 4> cause_bytes{}, cause_pieces{};
struct ApplyStat {
    std::atomic<u64> calls{0}, bytes{0}, ns{0};
};
std::array<std::array<ApplyStat, 64>, 2> apply_stats; // [ring][effect mask]
std::atomic<u64> image_calls{0}, image_tiled{0}, image_ns{0};

struct Last {
    u64 size;
    u64 hash;
};
std::unordered_map<u64, Last> last_copy; // raster thread only (the copies counted here all run on it)

void Touch() noexcept {
    u64 expected = 0;
    first_ns.compare_exchange_strong(expected, NowNs(), std::memory_order_relaxed);
}
} // namespace

void NoteSync(SyncSite site, u64 requested, u64 uploaded, u64 ns) noexcept {
    Touch();
    auto& s = sync_stats[site];
    s.calls.fetch_add(1, std::memory_order_relaxed);
    s.requested.fetch_add(requested, std::memory_order_relaxed);
    s.uploaded.fetch_add(uploaded, std::memory_order_relaxed);
    s.ns.fetch_add(ns, std::memory_order_relaxed);
    if (uploaded == 0) {
        s.noop_calls.fetch_add(1, std::memory_order_relaxed);
        s.noop_ns.fetch_add(ns, std::memory_order_relaxed);
    }
}

void NoteCopy(CopySite site, u64 address, u64 size, u64 ns, const void* data) noexcept {
    Touch();
    auto& c = copy_stats[site];
    c.calls.fetch_add(1, std::memory_order_relaxed);
    c.bytes.fetch_add(size, std::memory_order_relaxed);
    c.ns.fetch_add(ns, std::memory_order_relaxed);
    if (size <= 256) {
        c.small_calls.fetch_add(1, std::memory_order_relaxed);
    }
    // the bytes are hashed at their source (guest memory), never at the destination: staging and video memory are write combined, a read from them is orders of magnitude slow
    u64 hash = 0;
    const u64 hash_start = NowNs();
    if (data != nullptr && size != 0 && size <= (u64{1} << 20) && NativeContent::HashGuarded64(address, size, hash)) {
        c.hashed.fetch_add(1, std::memory_order_relaxed);
        if (last_copy.size() > 1'000'000) {
            last_copy.clear();
        }
        const auto [it, inserted] = last_copy.try_emplace(address, Last{size, hash});
        if (!inserted) {
            if (it->second.size == size && it->second.hash == hash) {
                c.same_as_last.fetch_add(1, std::memory_order_relaxed);
                c.same_bytes.fetch_add(size, std::memory_order_relaxed);
            }
            it->second = {size, hash};
        }
    }
    t_hash_ns += NowNs() - hash_start;
}

void NoteCause(u32 cause, u64 bytes) noexcept {
    cause_bytes[cause & 3].fetch_add(bytes, std::memory_order_relaxed);
    cause_pieces[cause & 3].fetch_add(1, std::memory_order_relaxed);
}

void NoteFromImage(u64 ns, bool tiled) noexcept {
    image_calls.fetch_add(1, std::memory_order_relaxed);
    image_ns.fetch_add(ns, std::memory_order_relaxed);
    if (tiled) {
        image_tiled.fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteApply(bool ring, u64 size, u32 effects, u64 ns) noexcept {
    Touch();
    auto& s = apply_stats[ring ? 1 : 0][effects & 63];
    s.calls.fetch_add(1, std::memory_order_relaxed);
    s.bytes.fetch_add(size, std::memory_order_relaxed);
    s.ns.fetch_add(ns, std::memory_order_relaxed);
}

void NoteApplied(u64 size, bool again) noexcept {
    u32 log = 0;
    while ((size >> log) > 1 && log < 23) {
        ++log;
    }
    (again ? applied_again_log2 : applied_first_log2)[log].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["on"] = g_on;
    const u64 first = first_ns.load();
    j["seconds"] = first == 0 ? 0.0 : static_cast<double>(NowNs() - first) / 1e9;
    for (u32 i = 0; i < SyncSiteCount; ++i) {
        const auto& s = sync_stats[i];
        j["sync"][SyncNames[i]] = {{"calls", s.calls.load()}, {"noop_calls", s.noop_calls.load()}, {"requested_bytes", s.requested.load()}, {"uploaded_bytes", s.uploaded.load()},
                                   {"ms", s.ns.load() / 1e6}, {"noop_ms", s.noop_ns.load() / 1e6}};
    }
    for (u32 i = 0; i < CopySiteCount; ++i) {
        const auto& c = copy_stats[i];
        j["copy"][CopyNames[i]] = {{"calls", c.calls.load()}, {"small_calls_le_256", c.small_calls.load()}, {"bytes", c.bytes.load()}, {"ms", c.ns.load() / 1e6}, {"hashed", c.hashed.load()},
                                   {"same_as_previous_copy_at_address", c.same_as_last.load()}, {"same_bytes", c.same_bytes.load()}};
    }
    static constexpr const char* CauseNames[3] = {"frame_ring_part_uploaded_whole", "announced_bytes_of_engine_buffer", "announced_bytes_of_unowned_memory"};
    for (u32 i = 0; i < 3; ++i) {
        j["upload_cause"][CauseNames[i]] = {{"pieces", cause_pieces[i].load()}, {"bytes", cause_bytes[i].load()}};
    }
    for (u32 r = 0; r < 2; ++r) {
        for (u32 m = 0; m < 64; ++m) {
            const auto& s = apply_stats[r][m];
            if (const u64 n = s.calls.load()) {
                j["apply"][r ? "ring" : "other"][std::to_string(m)] = {{"calls", n}, {"bytes", s.bytes.load()}, {"ms", s.ns.load() / 1e6}};
            }
        }
    }
    j["from_image"] = {{"calls", image_calls.load()}, {"tiled_calls", image_tiled.load()}, {"ms", image_ns.load() / 1e6}};
    for (u32 i = 0; i < 24; ++i) {
        if (const u64 n = applied_first_log2[i].load()) {
            j["announced_first_by_size_log2"][std::to_string(i)] = n;
        }
        if (const u64 n = applied_again_log2[i].load()) {
            j["announced_again_by_size_log2"][std::to_string(i)] = n;
        }
    }
    return j;
}

} // namespace Core::N0
