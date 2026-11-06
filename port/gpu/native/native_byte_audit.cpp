// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_byte_audit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_content.h"
#include "native/native_timeline.h"
#include "native/native_tsharp_watch.h"

namespace Core::ByteAudit {

namespace {

constexpr u64 Chunk = 64;                ///< bytes per hash; a read is audited by the chunks it covers completely
constexpr u32 KeepSubmits = 8;           ///< a range stays in the hot set this many submissions after its last read (the game's rings turn over every 3)
constexpr u64 MaxRange = 32ull << 20;    ///< a bigger range is cut here (textures and big arenas are not what this audit is for)
constexpr u64 MaxSnapshotBytes = 512ull << 20;
constexpr size_t MaxStaleKept = 200;
constexpr u32 MaxArms = 48;              ///< write watches armed per debugger session
constexpr u64 ArmGapNs = 1'500'000'000;  ///< and not more often than this

std::atomic<bool> g_active{false};

struct Hot {
    u64 size;
    u32 last;
};
std::mutex g_hot_mutex;
std::unordered_map<u64, Hot> g_hot;
std::atomic<u32> g_newest_snapshot{0};

struct SnapRange {
    u64 first_chunk; ///< absolute chunk number (address / Chunk)
    std::vector<u32> hashes;
};
struct Snapshot {
    u32 index{};
    std::vector<SnapRange> ranges; ///< sorted by first_chunk, no overlaps
    u64 bytes{};
};
std::mutex g_snap_mutex;
std::deque<std::shared_ptr<const Snapshot>> g_snaps;

struct Stale {
    u64 ns;
    u32 submit;
    u32 newest_submit;
    u64 address;
    u64 size;
    u64 first_diff_address;
    u64 last_diff_address;
    u32 chunks_different;
    u32 chunks_audited;
    std::string kind;
    std::string bind;
};
std::mutex g_stale_mutex;
std::deque<Stale> g_stale;

struct Counters {
    std::atomic<u64> reads{}, reads_audited{}, reads_stale{};
    std::atomic<u64> chunks_audited{}, chunks_different{}, chunks_not_in_snapshot{};
    std::atomic<u64> reads_no_submit{}, reads_before_snapshot{}, reads_snapshot_gone{}, reads_too_small{};
    std::atomic<u64> bytes_read{}, bytes_audited{}, bytes_stale{};
    std::atomic<u64> snapshots{}, snapshot_bytes{}, snapshot_ns{}, snapshot_ranges{}, snapshot_ranges_dropped_unreadable{}, snapshot_ranges_cut{}, snapshot_skipped_full{};
    std::atomic<u64> read_audit_ns{};
    std::atomic<u64> arms{}, arms_skipped{};
};
Counters c;
std::atomic<u64> g_last_arm_ns{0};
std::vector<u64> g_armed_pages;
std::mutex g_arm_mutex;
std::unordered_map<std::string, std::pair<u64, u64>> g_stale_by_kind; ///< kind -> reads, chunks (under g_stale_mutex)
struct KindCoverage {
    u64 reads{}, bytes_read{}, bytes_audited{}, reads_no_submit{}, reads_no_snapshot{};
};
std::mutex g_cov_mutex;
std::unordered_map<std::string, KindCoverage> g_cov_by_kind; ///< 2026-11-06: which kinds of reads the audit can and cannot see
void NoteCoverage(const char* kind, u64 bytes_read, u64 bytes_audited, bool no_submit, bool no_snapshot) {
    std::scoped_lock lock{g_cov_mutex};
    KindCoverage& k = g_cov_by_kind[kind != nullptr ? kind : "?"];
    ++k.reads;
    k.bytes_read += bytes_read;
    k.bytes_audited += bytes_audited;
    k.reads_no_submit += no_submit ? 1 : 0;
    k.reads_no_snapshot += no_snapshot ? 1 : 0;
}

void MaybeArm(u64 address, const char* kind) {
    const u64 now = Core::FrameTimeline::Now();
    std::scoped_lock lock{g_arm_mutex};
    const u64 page = address >> 12;
    const bool recent = std::find(g_armed_pages.begin(), g_armed_pages.end(), page) != g_armed_pages.end();
    if (recent || c.arms.load() >= MaxArms || now - g_last_arm_ns.load() < ArmGapNs) {
        c.arms_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_last_arm_ns.store(now);
    g_armed_pages.push_back(page);
    if (g_armed_pages.size() > 16) {
        g_armed_pages.erase(g_armed_pages.begin());
    }
    c.arms.fetch_add(1, std::memory_order_relaxed);
    Core::NativeTsharpWatch::ArmRotating(address, kind);
}

} // namespace

bool Active() {
    return g_active.load(std::memory_order_relaxed);
}

void SetActive(bool on) {
    g_active.store(on);
}

void Reset() {
    {
        std::scoped_lock lock{g_hot_mutex};
        g_hot.clear();
    }
    {
        std::scoped_lock lock{g_snap_mutex};
        g_snaps.clear();
    }
    {
        std::scoped_lock lock{g_stale_mutex};
        g_stale.clear();
        g_stale_by_kind.clear();
    }
    {
        std::scoped_lock lock{g_arm_mutex};
        g_armed_pages.clear();
    }
    {
        std::scoped_lock lock{g_cov_mutex};
        g_cov_by_kind.clear();
    }
    g_newest_snapshot.store(0);
    g_last_arm_ns.store(0);
    c.reads = 0;
    c.reads_audited = 0;
    c.reads_stale = 0;
    c.chunks_audited = 0;
    c.chunks_different = 0;
    c.chunks_not_in_snapshot = 0;
    c.reads_no_submit = 0;
    c.reads_before_snapshot = 0;
    c.reads_snapshot_gone = 0;
    c.reads_too_small = 0;
    c.bytes_read = 0;
    c.bytes_audited = 0;
    c.bytes_stale = 0;
    c.snapshots = 0;
    c.snapshot_bytes = 0;
    c.snapshot_ns = 0;
    c.snapshot_ranges = 0;
    c.snapshot_ranges_dropped_unreadable = 0;
    c.snapshot_ranges_cut = 0;
    c.snapshot_skipped_full = 0;
    c.read_audit_ns = 0;
    c.arms = 0;
    c.arms_skipped = 0;
}

void SnapshotAtSubmit(u32 index) {
    if (!Active()) {
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::pair<u64, u64>> ranges; // address, size
    {
        std::scoped_lock lock{g_hot_mutex};
        ranges.reserve(g_hot.size());
        for (auto it = g_hot.begin(); it != g_hot.end();) {
            if (it->second.last + KeepSubmits < index) {
                it = g_hot.erase(it);
            } else {
                ranges.emplace_back(it->first, it->second.size);
                ++it;
            }
        }
    }
    std::sort(ranges.begin(), ranges.end());
    // merge to chunk aligned, non overlapping pieces
    struct Piece {
        u64 first, end; // chunk numbers, end exclusive
    };
    std::vector<Piece> pieces;
    for (const auto& [address, size] : ranges) {
        u64 first = (address + Chunk - 1) / Chunk, end = (address + size) / Chunk;
        if (end <= first) {
            continue;
        }
        if (end - first > MaxRange / Chunk) {
            end = first + MaxRange / Chunk;
            c.snapshot_ranges_cut.fetch_add(1, std::memory_order_relaxed);
        }
        if (!pieces.empty() && first <= pieces.back().end) {
            pieces.back().end = std::max(pieces.back().end, end);
        } else {
            pieces.push_back({first, end});
        }
    }
    auto snap = std::make_shared<Snapshot>();
    snap->index = index;
    for (const Piece& p : pieces) {
        const u64 bytes = (p.end - p.first) * Chunk;
        if (snap->bytes + bytes > MaxSnapshotBytes) {
            c.snapshot_skipped_full.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        SnapRange r;
        r.first_chunk = p.first;
        r.hashes.resize(p.end - p.first);
        if (!Core::NativeContent::HashChunksGuarded(p.first * Chunk, p.end - p.first, Chunk, r.hashes.data())) {
            c.snapshot_ranges_dropped_unreadable.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        snap->bytes += bytes;
        snap->ranges.push_back(std::move(r));
    }
    c.snapshots.fetch_add(1, std::memory_order_relaxed);
    c.snapshot_bytes.fetch_add(snap->bytes, std::memory_order_relaxed);
    c.snapshot_ranges.fetch_add(snap->ranges.size(), std::memory_order_relaxed);
    {
        std::scoped_lock lock{g_snap_mutex};
        g_snaps.push_back(std::move(snap));
        while (g_snaps.size() > KeepSubmits + 2) {
            g_snaps.pop_front();
        }
    }
    g_newest_snapshot.store(index, std::memory_order_release);
    c.snapshot_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count()), std::memory_order_relaxed);
}

void NoteRead(u64 address, u64 size, u32 submit, const char* kind, std::string (*describe)()) {
    if (!Active() || size == 0 || address == 0) {
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    c.reads.fetch_add(1, std::memory_order_relaxed);
    c.bytes_read.fetch_add(size, std::memory_order_relaxed);
    const u32 newest = g_newest_snapshot.load(std::memory_order_acquire);
    {
        // 2026-11-06: a ring or an arena is filled with new data every frame, and the new part was never read before, so a snapshot of only what was read earlier never covers it. The hot
        // range is widened to the block it lies in: 4 MB for the dynamic arenas, 64 KB for stream copies (the rest is kept as read). TLG_AUDIT_WIDEN=0 turns this off.
        static const bool widen = [] { const char* e = std::getenv("TLG_AUDIT_WIDEN"); return e == nullptr || e[0] != '0'; }();
        u64 hot_address = address, hot_size = size;
        if (widen && kind != nullptr) {
            const u64 block = std::strstr(kind, "arena_sync") != nullptr ? (4ull << 20) : std::strstr(kind, "stream_copy") != nullptr ? (64ull << 10) : 0;
            if (block != 0) {
                hot_address = address & ~(block - 1);
                hot_size = ((address + size + block - 1) & ~(block - 1)) - hot_address;
            }
        }
        std::scoped_lock lock{g_hot_mutex};
        Hot& h = g_hot[hot_address];
        h.size = std::max(h.size, hot_size);
        h.last = std::max<u32>(h.last, submit != 0 ? submit : newest + 1);
    }
    if (submit == 0) {
        c.reads_no_submit.fetch_add(1, std::memory_order_relaxed);
        NoteCoverage(kind, size, 0, true, false);
        return;
    }
    std::shared_ptr<const Snapshot> snap;
    {
        std::scoped_lock lock{g_snap_mutex};
        for (const auto& s : g_snaps) {
            if (s->index == submit) {
                snap = s;
                break;
            }
        }
    }
    if (!snap) {
        (submit > newest ? c.reads_before_snapshot : c.reads_snapshot_gone).fetch_add(1, std::memory_order_relaxed);
        NoteCoverage(kind, size, 0, false, true);
        return;
    }
    const u64 c0 = (address + Chunk - 1) / Chunk, c1 = (address + size) / Chunk;
    if (c1 <= c0) {
        c.reads_too_small.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    u64 audited = 0, different = 0, first_diff = ~0ull, last_diff = 0;
    std::vector<u32> now_hashes;
    // the snapshot ranges are sorted: find the first one that can overlap
    auto it = std::upper_bound(snap->ranges.begin(), snap->ranges.end(), c0, [](u64 chunk, const SnapRange& r) { return chunk < r.first_chunk; });
    if (it != snap->ranges.begin()) {
        --it;
    }
    for (; it != snap->ranges.end() && it->first_chunk < c1; ++it) {
        const u64 r0 = it->first_chunk, r1 = r0 + it->hashes.size();
        const u64 o0 = std::max(r0, c0), o1 = std::min(r1, c1);
        if (o1 <= o0) {
            continue;
        }
        now_hashes.resize(o1 - o0);
        if (!Core::NativeContent::HashChunksGuarded(o0 * Chunk, o1 - o0, Chunk, now_hashes.data())) {
            continue;
        }
        audited += o1 - o0;
        for (u64 k = 0; k < o1 - o0; ++k) {
            if (now_hashes[k] != it->hashes[o0 - r0 + k]) {
                ++different;
                first_diff = std::min(first_diff, o0 + k);
                last_diff = std::max(last_diff, o0 + k);
            }
        }
    }
    NoteCoverage(kind, size, audited * Chunk, false, false);
    c.chunks_audited.fetch_add(audited, std::memory_order_relaxed);
    c.chunks_not_in_snapshot.fetch_add((c1 - c0) - audited, std::memory_order_relaxed);
    if (audited != 0) {
        c.reads_audited.fetch_add(1, std::memory_order_relaxed);
        c.bytes_audited.fetch_add(audited * Chunk, std::memory_order_relaxed);
    }
    if (different != 0) {
        c.reads_stale.fetch_add(1, std::memory_order_relaxed);
        c.chunks_different.fetch_add(different, std::memory_order_relaxed);
        c.bytes_stale.fetch_add(different * Chunk, std::memory_order_relaxed);
        const u64 now = Core::FrameTimeline::Now();
        Core::FrameTimeline::Record(Core::FrameTimeline::ResourceStale, first_diff * Chunk, now, (different << 32) | submit);
        bool keep = false;
        {
            std::scoped_lock lock{g_stale_mutex};
            auto& by_kind = g_stale_by_kind[kind != nullptr ? kind : "?"];
            by_kind.first += 1;
            by_kind.second += different;
            keep = g_stale.size() < MaxStaleKept;
        }
        if (keep) {
            Stale s{now, submit, newest, address, size, first_diff * Chunk, last_diff * Chunk, static_cast<u32>(different), static_cast<u32>(audited), kind != nullptr ? kind : "?", describe != nullptr ? describe() : std::string()};
            std::scoped_lock lock{g_stale_mutex};
            g_stale.push_back(std::move(s));
        }
        MaybeArm(first_diff * Chunk, kind != nullptr ? kind : "stale read");
    }
    c.read_audit_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count()), std::memory_order_relaxed);
}

std::string Report() {
    nlohmann::json j;
    j["what"] = "byte audit: each read of game memory by the raster thread, compared chunk by chunk (64 bytes) with the memory hashed at the game's submit of that read's submission; a differing chunk = the GPU read other bytes than the game submitted";
    j["active"] = Active();
    auto& k = j["counters"];
    k["reads"] = c.reads.load();
    k["reads_audited"] = c.reads_audited.load();
    k["reads_stale"] = c.reads_stale.load();
    k["bytes_read"] = c.bytes_read.load();
    k["bytes_audited"] = c.bytes_audited.load();
    k["bytes_stale"] = c.bytes_stale.load();
    k["chunks_audited"] = c.chunks_audited.load();
    k["chunks_different"] = c.chunks_different.load();
    k["chunks_not_in_snapshot_(range_not_read_in_the_last_8_submissions)"] = c.chunks_not_in_snapshot.load();
    k["reads_without_submission"] = c.reads_no_submit.load();
    k["reads_before_their_submit_snapshot_(raster ran ahead of the submit)"] = c.reads_before_snapshot.load();
    k["reads_snapshot_already_dropped"] = c.reads_snapshot_gone.load();
    k["reads_smaller_than_a_chunk"] = c.reads_too_small.load();
    k["snapshots"] = c.snapshots.load();
    k["snapshot_ranges_total"] = c.snapshot_ranges.load();
    k["snapshot_bytes_total"] = c.snapshot_bytes.load();
    k["snapshot_ms_total"] = double(c.snapshot_ns.load()) / 1e6;
    k["snapshot_ranges_cut_at_32MB"] = c.snapshot_ranges_cut.load();
    k["snapshot_ranges_unreadable"] = c.snapshot_ranges_dropped_unreadable.load();
    k["snapshot_pieces_skipped_over_512MB"] = c.snapshot_skipped_full.load();
    k["read_audit_ms_total"] = double(c.read_audit_ns.load()) / 1e6;
    k["write_watches_armed"] = c.arms.load();
    k["write_watch_arms_skipped_(throttle)"] = c.arms_skipped.load();
    {
        std::scoped_lock lock{g_stale_mutex};
        auto& by_kind = j["stale_by_read_kind"] = nlohmann::json::object();
        for (const auto& [name, v] : g_stale_by_kind) {
            by_kind[name] = {{"reads", v.first}, {"chunks", v.second}};
        }
        auto& list = j["stale_reads"] = nlohmann::json::array();
        for (const Stale& s : g_stale) {
            list.push_back({{"ns", s.ns},
                            {"kind", s.kind},
                            {"submit", s.submit},
                            {"newest_snapshot_then", s.newest_submit},
                            {"address", fmt::format("{:#x}", s.address)},
                            {"size", s.size},
                            {"first_different_chunk", fmt::format("{:#x}", s.first_diff_address)},
                            {"last_different_chunk", fmt::format("{:#x}", s.last_diff_address)},
                            {"chunks_different", s.chunks_different},
                            {"chunks_audited", s.chunks_audited},
                            {"bind", s.bind}});
        }
    }
    {
        std::scoped_lock lock{g_cov_mutex};
        auto& cov = j["coverage_by_read_kind"] = nlohmann::json::object();
        for (const auto& [name, v] : g_cov_by_kind) {
            cov[name] = {{"reads", v.reads}, {"MB_read", double(v.bytes_read) / 1048576.0}, {"MB_audited", double(v.bytes_audited) / 1048576.0}, {"audited_percent", v.bytes_read ? 100.0 * double(v.bytes_audited) / double(v.bytes_read) : 0.0},
                         {"reads_without_submission", v.reads_no_submit}, {"reads_without_snapshot", v.reads_no_snapshot}};
        }
    }
    j["writer_hits"] = nlohmann::json::parse(Core::NativeTsharpWatch::HitsJson(), nullptr, false);
    return j.dump(1);
}

} // namespace Core::ByteAudit
