// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include <unordered_map>
#include <fmt/format.h>
#include <xxhash.h>

#include "common/logging/log.h"
#include "core/memory.h"
#include "core/stream_probe.h"
#include "core/stream_submit.h"
#include "video_core/readback_policy.h"
#include "video_core/native_draw_tag.h"

namespace Core::StreamProbe {

namespace {

struct Segment {
    u64 from{};
    u64 to{};
    std::vector<u32> copy;
};

struct BufferShape {
    u32 index{}, count{};
    u32 dcb_dwords{}, ccb_dwords{};
    u32 packets{}, indirect{}, nops{}, tags{}, draws{}, dispatches{}, type0{}, other{};
    u32 segments{}, mismatched_dwords{}, mismatch_tail{};
    s64 first_mismatch_from_end{-1};
    bool end_matches{};
};

struct FrameRecord {
    std::vector<BufferShape> buffers;
    u32 hooks{}, chunks{}, anomalies{};
    u64 recorded_dwords{};
    u32 gap_min{}, gap_p50{}, gap_p90{}, gap_max{};
    u64 build_ns{}, prepare_ns{};
    u32 builds{};
    u32 pure{};
};

std::mutex mutex;
std::vector<Segment> segments;
u64 cur_end{};
u64 last_cursor{};
std::vector<u32> gaps;
u32 hooks{}, chunks{}, anomalies{};
u64 recorded_dwords{};
u64 build_ns{}, prepare_ns{};
u32 builds{}, pure_builds{};
std::vector<BufferShape> shapes;
std::deque<FrameRecord> history;
u64 total_frames{}, total_mismatch_frames{}, total_mismatch_dwords{}, total_tail_mismatch{};
std::array<u64, 8> gap_histogram{};
u32 logged{};
std::array<u64, 8> rewrite_lag{};  ///< hooks between recording a segment and the change of its dwords, by log2
u64 rewrites_ours{}, rewrites_other{}, changes_after_window{}, changes_after_window_other{};
u64 rewrite_dwords_max_distance{};
constexpr size_t LagWindow = 64;

struct Range {
    u64 base{};
    u32 size{};
    u32 table_kind{};
    u32 slot{};
    u64 caller{};
    u64 hash{};
};
std::vector<Range> ranges;
std::unordered_map<u64, u32> range_index; ///< (base, size) -> index in `ranges`
u64 ranges_hashed{}, ranges_large_skipped{}, ranges_unmapped{}, range_bytes{};
u64 range_changes{};
u64 changes_seen_before{};
std::unordered_map<u64, u64> changes_by_caller;
std::unordered_map<u32, u64> changes_by_kind;
std::vector<Range> change_samples;

/// A dword our own conversions write over an earlier packet: a NOP header (type 3, opcode 0x10), the tag payload text ("ND.."), or zero filling.
bool LooksLikeOurRewrite(u32 value) {
    return (value >> 30) == 3 && ((value >> 8) & 0xff) == 0x10 || (value >> 16) == 0x4e44 || value == 0 || (value & 0xffff) == 0x1000;
}

u32 Bucket(u32 dwords) {
    u32 bucket = 0;
    for (u32 limit = 16; bucket < 7 && dwords >= limit; limit <<= 1) {
        ++bucket;
    }
    return bucket;
}

} // namespace

void AtHook(u64 ctx) {
    if (VideoCore::ReadbackPolicy::Get().stream_probe.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const u64 begin = *reinterpret_cast<const u64*>(ctx + 0x47c0);
    const u64 end = *reinterpret_cast<const u64*>(ctx + 0x47c8);
    const u64 cursor = *reinterpret_cast<const u64*>(ctx + 0x47d0);
    std::scoped_lock lock{mutex};
    ++hooks;
    if (end != cur_end) {
        ++chunks;
        cur_end = end;
        const bool begin_ok = begin != 0 && begin <= cursor && cursor - begin < (8u << 20);
        if (logged++ < 8) {
            LOG_INFO(Core, "stream probe: new buffer begin {:#x} cursor {:#x} end {:#x} (begin plausible: {}, {} dwords written)", begin, cursor, end,
                     begin_ok, begin_ok ? (cursor - begin) / 4 : 0);
        }
        last_cursor = begin_ok ? begin : cursor;
    }
    if (cursor < last_cursor || cursor - last_cursor > (4u << 20)) {
        ++anomalies;
        last_cursor = cursor;
        return;
    }
    if (cursor == last_cursor) {
        return;
    }
    // Which of the latest segments changed since they were recorded, and how many hooks later (the distance a later hook reaches back).
    for (size_t k = 1; k <= LagWindow && k <= segments.size(); ++k) {
        auto& old = segments[segments.size() - k];
        const u32* now = reinterpret_cast<const u32*>(old.from);
        bool changed = false;
        for (size_t d = 0; d < old.copy.size(); ++d) {
            if (now[d] != old.copy[d]) {
                changed = true;
                if (LooksLikeOurRewrite(now[d])) {
                    ++rewrites_ours;
                } else {
                    ++rewrites_other;
                    if (logged++ < 60) {
                        LOG_INFO(Core, "stream probe: a dword other than a known rewrite changed {} hooks after it was recorded: {:#x} -> {:#x} at {:#x}", k, old.copy[d],
                                 now[d], old.from + d * 4);
                    }
                }
                old.copy[d] = now[d];
            }
        }
        if (changed) {
            u32 bucket = 0;
            for (size_t limit = 1; bucket < 7 && k > limit; limit <<= 1) {
                ++bucket;
            }
            ++rewrite_lag[bucket];
        }
    }
    Segment segment;
    segment.from = last_cursor;
    segment.to = cursor;
    segment.copy.resize((cursor - last_cursor) / 4);
    std::memcpy(segment.copy.data(), reinterpret_cast<const void*>(last_cursor), segment.copy.size() * 4);
    const u32 gap = static_cast<u32>(segment.copy.size());
    gaps.push_back(gap);
    ++gap_histogram[Bucket(gap)];
    recorded_dwords += gap;
    segments.push_back(std::move(segment));
    last_cursor = cursor;
}

void AtRecord(u64 caller, const TableView* tables, u32 count) {
    if (VideoCore::ReadbackPolicy::Get().stream_probe.load(std::memory_order_relaxed) == 0) {
        return;
    }
    auto* memory = Core::Memory::Instance();
    std::scoped_lock lock{mutex};
    for (u32 t = 0; t < count; ++t) {
        const TableView& table = tables[t];
        if (table.entry == 0 || (table.kind != 19 && table.kind != 22 && table.kind != 25)) {
            continue;
        }
        const u32* words = reinterpret_cast<const u32*>(table.address);
        for (u32 slot = 0; slot * table.entry + 3 < table.dwords && slot < 128; ++slot) {
            const u64 mask = slot < 64 ? table.mask0 : table.mask1;
            if (((mask >> (slot & 63)) & 1) == 0) {
                continue;
            }
            const u32* v = words + slot * table.entry;
            const u64 base = static_cast<u64>(v[0]) | (static_cast<u64>(v[1] & 0xffff) << 32);
            const u32 stride = (v[1] >> 16) & 0x3fff;
            const u32 records = v[2];
            const u64 size = stride != 0 ? static_cast<u64>(stride) * records : records;
            if (base < 0x200000000ull || base >= 0x4000000000ull || (base & 3) != 0 || size == 0 || records == 0) {
                continue;
            }
            if (size > 65536) {
                ++ranges_large_skipped;
                continue;
            }
            const u64 key = base ^ (size * 0x9E3779B97F4A7C15ull);
            if (range_index.contains(key)) {
                continue;
            }
            if (!memory->IsValidMapping(base, size)) {
                ++ranges_unmapped;
                continue;
            }
            range_index.emplace(key, static_cast<u32>(ranges.size()));
            Range r;
            r.base = base;
            r.size = static_cast<u32>(size);
            r.table_kind = table.kind;
            r.slot = slot;
            r.caller = caller;
            r.hash = XXH3_64bits(reinterpret_cast<const void*>(base), size);
            ranges.push_back(r);
            ++ranges_hashed;
            range_bytes += size;
        }
    }
}

void OnSubmitBuffer(u32 index, u32 count, const u32* dcb, u32 dcb_dwords, u32 ccb_dwords) {
    BufferShape shape;
    shape.index = index;
    shape.count = count;
    shape.dcb_dwords = dcb_dwords;
    shape.ccb_dwords = ccb_dwords;
    for (u32 i = 0; i < dcb_dwords;) {
        const u32 header = dcb[i];
        u32 length = 1;
        switch (header >> 30) {
        case 3: {
            length = ((header >> 16) & 0x3fff) + 2;
            const u32 opcode = (header >> 8) & 0xff;
            ++shape.packets;
            if (opcode == 0x3f) {
                ++shape.indirect;
            } else if (opcode == 0x10) {
                ++shape.nops;
                if (i + 1 < dcb_dwords && dcb[i + 1] == VideoCore::NativeTagMagic) {
                    ++shape.tags;
                }
            } else if (opcode == 0x2d || opcode == 0x27 || opcode == 0x2c || opcode == 0x2e) {
                ++shape.draws;
            } else if (opcode == 0x15 || opcode == 0x16) {
                ++shape.dispatches;
            }
            break;
        }
        case 2:
            length = 1;
            break;
        case 0:
            length = ((header >> 16) & 0x3fff) + 2;
            ++shape.type0;
            break;
        default:
            ++shape.other;
            length = 2;
            break;
        }
        i += std::max(length, 1u);
    }
    const u64 low = reinterpret_cast<u64>(dcb);
    const u64 high = low + static_cast<u64>(dcb_dwords) * 4;
    std::scoped_lock lock{mutex};
    for (const auto& segment : segments) {
        if (segment.from < low || segment.to > high) {
            continue;
        }
        ++shape.segments;
        shape.end_matches |= segment.to + 4 >= high;
        const u32* now = reinterpret_cast<const u32*>(segment.from);
        for (size_t d = 0; d < segment.copy.size(); ++d) {
            if (now[d] != segment.copy[d]) {
                ++shape.mismatched_dwords;
                ++changes_after_window;
                changes_after_window_other += LooksLikeOurRewrite(now[d]) ? 0 : 1;
                const s64 from_end = static_cast<s64>((high - (segment.from + d * 4)) / 4);
                if (shape.first_mismatch_from_end < 0 || from_end > shape.first_mismatch_from_end) {
                    shape.first_mismatch_from_end = from_end;
                }
                if (from_end <= 64) {
                    ++shape.mismatch_tail;
                }
                if (logged++ < 40) {
                    LOG_INFO(Core, "stream probe: buffer {:#x} dword {} (from the end {}) was {:#x} at the hook, is {:#x} at the submit",
                             low, (segment.from + d * 4 - low) / 4, from_end, segment.copy[d], now[d]);
                }
            }
        }
    }
    shapes.push_back(shape);
}

void EndSubmit() {
    std::scoped_lock lock{mutex};
    for (const auto& r : ranges) {
        if (XXH3_64bits(reinterpret_cast<const void*>(r.base), r.size) != r.hash) {
            ++range_changes;
            ++changes_by_caller[r.caller];
            ++changes_by_kind[r.table_kind];
            if (change_samples.size() < 40) {
                change_samples.push_back(r);
            }
        }
    }
    if (range_changes != changes_seen_before) {
        changes_seen_before = range_changes;
        if (VideoCore::ReadbackPolicy::Get().stream_submit.load() != 0) {
            StreamSubmit::NoteViolation("the data a draw reads changed between the hook and the submit");
        }
    }
    ranges.clear();
    range_index.clear();
    FrameRecord frame;
    frame.buffers = std::move(shapes);
    shapes.clear();
    frame.hooks = hooks;
    frame.chunks = chunks;
    frame.anomalies = anomalies;
    frame.recorded_dwords = recorded_dwords;
    frame.build_ns = build_ns;
    frame.prepare_ns = prepare_ns;
    frame.builds = builds;
    frame.pure = pure_builds;
    if (!gaps.empty()) {
        std::ranges::sort(gaps);
        frame.gap_min = gaps.front();
        frame.gap_p50 = gaps[gaps.size() / 2];
        frame.gap_p90 = gaps[gaps.size() * 9 / 10];
        frame.gap_max = gaps.back();
    }
    bool mismatch = false;
    for (const auto& b : frame.buffers) {
        total_mismatch_dwords += b.mismatched_dwords;
        total_tail_mismatch += b.mismatch_tail;
        mismatch |= b.mismatched_dwords != 0;
    }
    total_mismatch_frames += mismatch ? 1 : 0;
    ++total_frames;
    history.push_back(std::move(frame));
    while (history.size() > 40) {
        history.pop_front();
    }
    segments.clear();
    gaps.clear();
    hooks = chunks = anomalies = 0;
    recorded_dwords = 0;
    build_ns = prepare_ns = 0;
    builds = pure_builds = 0;
    cur_end = 0;
    last_cursor = 0;
}

void NoteBuild(u64 ns, bool pure, u32) {
    std::scoped_lock lock{mutex};
    build_ns += ns;
    ++builds;
    pure_builds += pure ? 1 : 0;
}

void NotePrepare(u64 ns) {
    std::scoped_lock lock{mutex};
    prepare_ns += ns;
}

nlohmann::json Report() {
    std::scoped_lock lock{mutex};
    nlohmann::json j;
    j["note"] = "per submit call: buffers (the dcb / ccb pairs), packets, indirect buffer packets, tag nops, draws, how many dwords the game changed after the hook saw them "
                "(mismatched_dwords; mismatch_tail: within the last 64 dwords of the buffer, i.e. the flip request patch), the gaps in dwords between hooks";
    j["streaming"] = StreamSubmit::Report();
    j["frames_seen"] = total_frames;
    j["frames_with_changed_dwords"] = total_mismatch_frames;
    j["changed_dwords_total"] = total_mismatch_dwords;
    j["changed_dwords_in_the_last_64_of_a_buffer"] = total_tail_mismatch;
    j["gap_histogram_dwords_lt16_32_64_128_256_512_1024_more"] = gap_histogram;
    j["rewrites_by_later_hooks"] = {{"our_pattern", rewrites_ours},
                                    {"other_pattern", rewrites_other},
                                    {"lag_in_hooks_1_2_3to4_5to8_9to16_17to32_33to64_more", rewrite_lag}};
    {
        nlohmann::json by_caller = nlohmann::json::object();
        for (const auto& [caller, n] : changes_by_caller) {
            by_caller[fmt::format("{:#x}", caller)] = n;
        }
        nlohmann::json by_kind = nlohmann::json::object();
        for (const auto& [kind, n] : changes_by_kind) {
            by_kind[std::to_string(kind)] = n;
        }
        nlohmann::json samples = nlohmann::json::array();
        for (const auto& r : change_samples) {
            samples.push_back({{"base", fmt::format("{:#x}", r.base)}, {"size", r.size}, {"table_kind", r.table_kind}, {"slot", r.slot}, {"caller_ghidra", fmt::format("{:#x}", r.caller)}});
        }
        j["data_ranges"] = {{"hashed_ranges_total", ranges_hashed},
                            {"hashed_bytes_total", range_bytes},
                            {"skipped_larger_than_64k", ranges_large_skipped},
                            {"skipped_unmapped", ranges_unmapped},
                            {"changed_between_the_hook_and_the_submit", range_changes},
                            {"changed_by_caller", by_caller},
                            {"changed_by_table_kind", by_kind},
                            {"first_changed_ranges", samples}};
    }
    j["changes_older_than_64_hooks_at_submit"] = {{"total", changes_after_window}, {"other_pattern", changes_after_window_other}};
    auto& frames = j["last_frames"] = nlohmann::json::array();
    for (const auto& f : history) {
        nlohmann::json buffers = nlohmann::json::array();
        for (const auto& b : f.buffers) {
            buffers.push_back({{"index", b.index},
                               {"of", b.count},
                               {"dcb_dwords", b.dcb_dwords},
                               {"ccb_dwords", b.ccb_dwords},
                               {"packets", b.packets},
                               {"indirect", b.indirect},
                               {"nops", b.nops},
                               {"tags", b.tags},
                               {"draws", b.draws},
                               {"dispatches", b.dispatches},
                               {"type0", b.type0},
                               {"other", b.other},
                               {"segments", b.segments},
                               {"end_matches", b.end_matches},
                               {"changed_dwords", b.mismatched_dwords},
                               {"changed_in_tail", b.mismatch_tail},
                               {"first_changed_from_end", b.first_mismatch_from_end}});
        }
        frames.push_back({{"hooks", f.hooks},
                          {"chunks_seen_by_hook", f.chunks},
                          {"anomalies", f.anomalies},
                          {"recorded_dwords", f.recorded_dwords},
                          {"gap_dwords_min_p50_p90_max", {f.gap_min, f.gap_p50, f.gap_p90, f.gap_max}},
                          {"build_us", f.build_ns / 1000},
                          {"builds", f.builds},
                          {"pure_builds", f.pure},
                          {"submit_sweep_us", f.prepare_ns / 1000},
                          {"buffers", std::move(buffers)}});
    }
    return j;
}

} // namespace Core::StreamProbe
