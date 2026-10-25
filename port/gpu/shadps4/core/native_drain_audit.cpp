// SPDX-License-Identifier: GPL-2.0-or-later
// N6a: the drain audit, see native_drain_audit.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "core/native_drain_audit.h"

namespace Core::DrainAudit {

namespace {

struct PendingWrite {
    u64 position;
    u64 address;
    u64 bytes;
    u32 line;
};

constexpr const char* kind_names[KindCount] = {"shader code", "flat user data walker (approximate)", "vertex descriptor table", "constant ring bytes", "vertex ring bytes", "index ring bytes"};

struct State {
    std::mutex mutex;
    std::deque<PendingWrite> writes; // queued write closures, newest last (pruned by position)
    std::atomic<u64> writes_queued{};
    std::atomic<u64> writes_unknown_range{};
    std::map<u32, u64> writes_by_line;
    std::map<u32, u64> unknown_by_line;
    std::map<u32, u64> bytes_by_line;
    std::map<u32, u64> max_bytes_by_line;
    std::map<u64, u64> near_pairs; // reads within 256 KB of a pending write that did not overlap it: (kind << 32 | line) -> count
    std::atomic<u64> drains_nothing_pending{};
    std::atomic<u64> drains_with_pending{};
    std::atomic<u64> wait_ns{};
    std::array<std::atomic<u64>, 4> drains_by_caller{};
    std::array<std::atomic<u64>, KindCount> reads{};
    std::array<std::atomic<u64>, KindCount> reads_in_window{};
    std::atomic<u64> planned_without_wait{};
    std::atomic<u64> windows{};
    std::atomic<u64> snapshot_writes{};
    std::atomic<u64> snapshot_writes_max{};
    std::atomic<u64> window_wait_ns{};
    // outcome by the set of kinds that overlapped a pending write: mask -> {windows, wait ns}
    std::map<u32, std::pair<u64, u64>> by_mask;
    // overlaps by (kind, line of the write)
    std::map<u64, u64> overlap_pairs;
};

State& S() {
    static State state;
    return state;
}

struct Window {
    bool open{false};
    u32 mask{0};
    u64 wait_ns{0};
    std::vector<PendingWrite> pending;
};
thread_local Window window;
thread_local u64 next_address = 0, next_bytes = 0;
thread_local bool next_set = false;

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_DRAIN_AUDIT");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

void SetNextWrite(u64 address, u64 bytes) {
    if (!Enabled()) {
        return;
    }
    next_address = address;
    next_bytes = bytes;
    next_set = true;
}

void ClearNextWrite() {
    next_set = false;
}

void WriteQueued(u64 position, u32 line) {
    if (!Enabled()) {
        return;
    }
    PendingWrite write{position, 0, 0, line};
    if (next_set) {
        write.address = next_address;
        write.bytes = next_bytes;
    }
    next_set = false;
    auto& s = S();
    s.writes_queued.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lk{s.mutex};
    ++s.writes_by_line[line];
    s.bytes_by_line[line] += write.bytes;
    s.max_bytes_by_line[line] = std::max(s.max_bytes_by_line[line], write.bytes);
    if (write.address == 0 && write.bytes == 0) {
        s.writes_unknown_range.fetch_add(1, std::memory_order_relaxed);
        ++s.unknown_by_line[line];
    }
    s.writes.push_back(write);
    while (s.writes.size() > 4096) {
        s.writes.pop_front();
    }
}

void DrainBegin(u64 done, u32 caller) {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    s.drains_with_pending.fetch_add(1, std::memory_order_relaxed);
    s.drains_by_caller[std::min<u32>(caller, 3)].fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lk{s.mutex};
        for (const auto& write : s.writes) {
            if (write.position > done &&
                std::none_of(window.pending.begin(), window.pending.end(), [&](const PendingWrite& w) { return w.position == write.position; })) {
                window.pending.push_back(write);
            }
        }
    }
    if (!window.open) {
        window.open = true;
        window.mask = 0;
        window.wait_ns = 0;
    }
    s.snapshot_writes.fetch_add(window.pending.size(), std::memory_order_relaxed);
    u64 seen = s.snapshot_writes_max.load(std::memory_order_relaxed);
    while (window.pending.size() > seen && !s.snapshot_writes_max.compare_exchange_weak(seen, window.pending.size())) {
    }
}

void DrainNothingPending() {
    if (!Enabled()) {
        return;
    }
    S().drains_nothing_pending.fetch_add(1, std::memory_order_relaxed);
}

void DrainWaited(u64 ns) {
    if (!Enabled()) {
        return;
    }
    S().wait_ns.fetch_add(ns, std::memory_order_relaxed);
    if (window.open) {
        window.wait_ns += ns;
    }
}

void Read(Kind kind, u64 address, u64 bytes) {
    if (!Enabled() || bytes == 0) {
        return;
    }
    auto& s = S();
    s.reads[kind].fetch_add(1, std::memory_order_relaxed);
    if (!window.open) {
        return;
    }
    s.reads_in_window[kind].fetch_add(1, std::memory_order_relaxed);
    for (const auto& write : window.pending) {
        if (write.address == NoGuestMemory) {
            continue;
        }
        const bool unknown = write.address == 0 && write.bytes == 0;
        if (unknown || (address < write.address + write.bytes && write.address < address + bytes)) {
            window.mask |= 1u << kind;
            std::scoped_lock lk{s.mutex};
            ++s.overlap_pairs[(u64(kind) << 32) | write.line];
        } else if (address < write.address + write.bytes + (256u << 10) && write.address < address + bytes + (256u << 10)) {
            std::scoped_lock lk{s.mutex};
            ++s.near_pairs[(u64(kind) << 32) | write.line];
        }
    }
}

void WalkerRead(const u32* user_data, u32 count) {
    if (!Enabled() || count < 2) {
        return;
    }
    for (u32 i = 0; i + 1 < count; ++i) {
        const u64 value = u64(user_data[i]) | (u64(user_data[i + 1]) << 32);
        if ((value >> 48) != 0) {
            continue;
        }
        if (value >= 0x10000 && value < (1ull << 47)) {
            Read(ReadWalker, value, 1024);
        }
    }
}

void Planned() {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    if (!window.open) {
        s.planned_without_wait.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    s.windows.fetch_add(1, std::memory_order_relaxed);
    s.window_wait_ns.fetch_add(window.wait_ns, std::memory_order_relaxed);
    {
        std::scoped_lock lk{s.mutex};
        auto& entry = s.by_mask[window.mask];
        ++entry.first;
        entry.second += window.wait_ns;
    }
    window.open = false;
    window.pending.clear();
}

nlohmann::json Report() {
    nlohmann::json out;
    auto& s = S();
    out["enabled"] = Enabled();
    if (!Enabled()) {
        out["note"] = "start the game with TLG_DRAIN_AUDIT=1";
        return out;
    }
    out["writes_queued"] = s.writes_queued.load();
    out["writes_with_unknown_range"] = s.writes_unknown_range.load();
    out["drains_with_pending_writes"] = s.drains_with_pending.load();
    out["drains_with_nothing_pending"] = s.drains_nothing_pending.load();
    out["drains_with_pending_by_caller"] = {{"dispatch_lookup_931", s.drains_by_caller[0].load()}, {"exec_plan_1463", s.drains_by_caller[1].load()}, {"draw_plan_1555", s.drains_by_caller[2].load()}, {"other", s.drains_by_caller[3].load()}};
    out["wait_ms_total"] = s.wait_ns.load() / 1e6;
    out["plannings_that_waited"] = s.windows.load();
    out["plannings_that_did_not_wait"] = s.planned_without_wait.load();
    out["pending_writes_seen_at_a_wait"] = {{"average", s.drains_with_pending.load() ? double(s.snapshot_writes.load()) / s.drains_with_pending.load() : 0.0}, {"largest", s.snapshot_writes_max.load()}};
    out["wait_ms_in_plannings"] = s.window_wait_ns.load() / 1e6;
    nlohmann::json reads = nlohmann::json::object();
    for (u32 k = 0; k < KindCount; ++k) {
        reads[kind_names[k]] = {{"reads", s.reads[k].load()}, {"reads_while_a_wait_was_open", s.reads_in_window[k].load()}};
    }
    out["guest_reads_recorded"] = reads;
    std::scoped_lock lk{s.mutex};
    nlohmann::json masks = nlohmann::json::array();
    u64 unnecessary = 0, necessary = 0, walker_only = 0;
    double unnecessary_ms = 0, necessary_ms = 0, walker_only_ms = 0;
    for (const auto& [mask, entry] : s.by_mask) {
        std::string what;
        for (u32 k = 0; k < KindCount; ++k) {
            if (mask & (1u << k)) {
                what += (what.empty() ? "" : " + ") + std::string(kind_names[k]);
            }
        }
        if (what.empty()) {
            what = "no read touched a pending write";
        }
        masks.push_back({{"overlapping_reads", what}, {"plannings", entry.first}, {"wait_ms", entry.second / 1e6}});
        if (mask == 0) {
            unnecessary += entry.first;
            unnecessary_ms += entry.second / 1e6;
        } else if (mask == (1u << ReadWalker)) {
            walker_only += entry.first;
            walker_only_ms += entry.second / 1e6;
        } else {
            necessary += entry.first;
            necessary_ms += entry.second / 1e6;
        }
    }
    out["verdict"] = {{"wait_was_not_needed", {{"plannings", unnecessary}, {"wait_ms", unnecessary_ms}}},
                      {"wait_was_needed_only_because_of_the_walker_approximation", {{"plannings", walker_only}, {"wait_ms", walker_only_ms}}},
                      {"wait_was_needed", {{"plannings", necessary}, {"wait_ms", necessary_ms}}}};
    out["by_overlapping_reads"] = masks;
    nlohmann::json pairs = nlohmann::json::array();
    for (const auto& [key, count] : s.overlap_pairs) {
        pairs.push_back({{"read", kind_names[key >> 32]}, {"write_queued_at_liverpool.cpp_line", key & 0xFFFFFFFFu}, {"overlaps", count}});
    }
    out["overlaps_by_read_and_write_site"] = pairs;
    nlohmann::json near = nlohmann::json::array();
    for (const auto& [key, count] : s.near_pairs) {
        near.push_back({{"read", kind_names[key >> 32]}, {"write_queued_at_liverpool.cpp_line", key & 0xFFFFFFFFu}, {"reads_within_256KB_but_not_overlapping", count}});
    }
    out["near_misses_by_read_and_write_site"] = near;
    nlohmann::json by_line = nlohmann::json::object();
    for (const auto& [line, count] : s.writes_by_line) {
        by_line[std::to_string(line)] = {{"queued", count}, {"range_unknown", s.unknown_by_line.count(line) ? s.unknown_by_line[line] : 0}, {"bytes_total", s.bytes_by_line[line]}, {"bytes_largest", s.max_bytes_by_line[line]}};
    }
    out["writes_by_site"] = by_line;
    return out;
}

} // namespace Core::DrainAudit
