// SPDX-License-Identifier: GPL-2.0-or-later
// N6a / N6b: the drain audit and the lazy wait, see native_drain_audit.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
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

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct State {
    std::mutex mutex;
    std::deque<PendingWrite> writes; // queued write closures, newest last (pruned by position)
    std::atomic<u64> last_write_position{0};
    const std::atomic<u64>* done{nullptr};
    const std::atomic<u64>* write_done{nullptr};
    std::function<void(u64)> wait_for;
    std::atomic<u64> writes_queued{};
    std::atomic<u64> writes_unknown_range{};
    std::map<u32, u64> writes_by_line;
    std::map<u32, u64> unknown_by_line;
    std::map<u32, u64> bytes_by_line;
    std::map<u32, u64> max_bytes_by_line;
    // audit
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
    std::map<u32, std::pair<u64, u64>> by_mask; // mask of overlapping kinds -> {plannings, wait ns}
    std::map<u64, u64> overlap_pairs;           // (kind << 32 | line) -> overlaps
    std::map<u64, u64> near_pairs;              // reads within 256 KB of a pending write that did not overlap it
    // lazy
    std::atomic<u64> lazy_lookups{};
    std::atomic<u64> lazy_lookups_with_pending{};
    std::atomic<u64> lazy_waits{};
    std::atomic<u64> lazy_wait_ns{};
    std::array<std::atomic<u64>, KindCount> lazy_waits_by_kind{};
    std::array<std::atomic<u64>, KindCount> lazy_wait_ns_by_kind{};
    std::map<u32, std::pair<u64, u64>> lazy_by_line; // line of the write waited for -> {waits, ns}
    // verification
    std::atomic<u64> bytes_lookups{}, bytes_copies{}, bytes_unreadable{}, bytes_stale{}, bytes_stale_near_write{};
    std::array<std::atomic<u64>, 4> plan_lookups{}, plan_differs{};
    std::atomic<u64> logged{0};
};

State& S() {
    static State state;
    return state;
}

/// The newest queue position the raster thread has finished (a batch of operations, or a write closure).
u64 Completed(const State& s) {
    if (s.done == nullptr) {
        return 0;
    }
    const u64 a = s.done->load(std::memory_order_acquire);
    const u64 b = s.write_done ? s.write_done->load(std::memory_order_acquire) : 0;
    return std::max(a, b);
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

struct CopyNote {
    Kind kind;
    u64 address;
    u64 bytes;
    u64 hash;
};
thread_local bool verify_bytes = false;
thread_local std::vector<CopyNote> copies;

u64 Fnv(const u8* data, u64 bytes) {
    u64 h = 0xcbf29ce484222325ull;
    u64 i = 0;
    for (; i + 8 <= bytes; i += 8) {
        u64 word;
        std::memcpy(&word, data + i, 8);
        h = (h ^ word) * 0x100000001b3ull;
    }
    for (; i < bytes; ++i) {
        h = (h ^ data[i]) * 0x100000001b3ull;
    }
    return h;
}

constexpr u64 HashLimit = 64 * 1024;

/// Hash of guest memory [address, address + bytes) (at most the first 64 KB); false if it cannot be read.
bool HashGuest(u64 address, u64 bytes, u64& hash) {
#ifdef _WIN32
    thread_local std::vector<u8> buffer;
    const u64 n = std::min(bytes, HashLimit);
    buffer.resize(n);
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), buffer.data(), n, &got) || got != n) {
        return false;
    }
    hash = Fnv(buffer.data(), n);
    return true;
#else
    hash = Fnv(reinterpret_cast<const u8*>(address), std::min(bytes, HashLimit));
    return true;
#endif
}

bool Overlaps(const PendingWrite& write, u64 address, u64 bytes) {
    if (write.address == NoGuestMemory) {
        return false;
    }
    if (write.address == 0 && write.bytes == 0) {
        return true; // unknown range
    }
    return address < write.address + write.bytes && write.address < address + bytes;
}

bool EnvOn(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != 0 && *e != '0';
}

} // namespace

bool Enabled() {
    static const bool on = EnvOn("TLG_DRAIN_AUDIT");
    return on;
}

bool LazyEnabled() {
    // on by default (verified in the light scene, the heavy scene and a cutscene); TLG_DRAIN_LAZY=0 brings back the full wait before every planning
    static const bool on = [] {
        const char* e = std::getenv("TLG_DRAIN_LAZY");
        return e == nullptr || *e == 0 || *e != '0';
    }();
    return on;
}

u32 VerifyEvery() {
    static const u32 every = [] {
        const char* e = std::getenv("TLG_DRAIN_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 0u; // off by default: the full waits of the verified plannings made the frame rate uneven
    }();
    return every;
}

bool Active() {
    static const bool on = Enabled() || LazyEnabled();
    return on;
}

void SetBackend(const std::atomic<u64>* done, const std::atomic<u64>* write_done, std::function<void(u64 position)> wait_for) {
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    s.done = done;
    s.write_done = write_done;
    s.wait_for = std::move(wait_for);
    s.writes.clear();
    s.last_write_position.store(0, std::memory_order_release);
}

void SetNextWrite(u64 address, u64 bytes) {
    if (!Active()) {
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
    if (!Active()) {
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
    if (Enabled()) {
        ++s.writes_by_line[line];
        s.bytes_by_line[line] += write.bytes;
        s.max_bytes_by_line[line] = std::max(s.max_bytes_by_line[line], write.bytes);
        if (write.address == 0 && write.bytes == 0) {
            s.writes_unknown_range.fetch_add(1, std::memory_order_relaxed);
            ++s.unknown_by_line[line];
        }
    }
    s.writes.push_back(write);
    const u64 done = Completed(s);
    while (!s.writes.empty() && (s.writes.size() > 4096 || s.writes.front().position <= done)) {
        s.writes.pop_front();
    }
    s.last_write_position.store(position, std::memory_order_release);
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

namespace {
/// Lazy wait: if a pending write overlaps the read, wait until the newest of them has run.
void LazyWait(Kind kind, u64 address, u64 bytes) {
    auto& s = S();
    if (s.done == nullptr || s.last_write_position.load(std::memory_order_acquire) <= Completed(s)) {
        return;
    }
    u64 need = 0;
    u32 line = 0;
    {
        std::scoped_lock lk{s.mutex};
        const u64 done = Completed(s);
        for (auto it = s.writes.rbegin(); it != s.writes.rend() && it->position > done; ++it) {
            if (Overlaps(*it, address, bytes) && it->position > need) {
                need = it->position;
                line = it->line;
            }
        }
    }
    if (need == 0 || need <= Completed(s)) {
        return;
    }
    const u64 start = NowNs();
    s.wait_for(need);
    const u64 ns = NowNs() - start;
    s.lazy_waits.fetch_add(1, std::memory_order_relaxed);
    s.lazy_wait_ns.fetch_add(ns, std::memory_order_relaxed);
    s.lazy_waits_by_kind[kind].fetch_add(1, std::memory_order_relaxed);
    s.lazy_wait_ns_by_kind[kind].fetch_add(ns, std::memory_order_relaxed);
    std::scoped_lock lk{s.mutex};
    auto& entry = s.lazy_by_line[line];
    ++entry.first;
    entry.second += ns;
}
} // namespace

void Read(Kind kind, u64 address, u64 bytes) {
    if (!Active() || bytes == 0) {
        return;
    }
    if (LazyEnabled()) {
        LazyWait(kind, address, bytes);
    }
    if (!Enabled()) {
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
        if (Overlaps(write, address, bytes)) {
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
    if (!Active() || count < 2) {
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

void Copied(Kind kind, u64 address, const void* copy, u64 bytes) {
    if (!verify_bytes || bytes == 0 || copies.size() >= 512) {
        return;
    }
    copies.push_back({kind, address, bytes, Fnv(static_cast<const u8*>(copy), std::min(bytes, HashLimit))});
}

Verify LookupBegin(u32 site) {
    if (!LazyEnabled()) {
        return Verify::None;
    }
    auto& s = S();
    s.lazy_lookups.fetch_add(1, std::memory_order_relaxed);
    const bool pending = s.done != nullptr && s.last_write_position.load(std::memory_order_acquire) > Completed(s);
    if (pending) {
        s.lazy_lookups_with_pending.fetch_add(1, std::memory_order_relaxed);
    }
    const u32 every = VerifyEvery();
    if (every == 0 || !pending) {
        return Verify::None;
    }
    // the verified plannings are chosen among those that find a write pending (the others cannot show anything)
    static std::atomic<u64> tick{0};
    const u64 n = tick.fetch_add(1, std::memory_order_relaxed);
    if (n % every != 0) {
        return Verify::None;
    }
    (void)site;
    if (((n / every) & 1) == 0) {
        verify_bytes = true;
        copies.clear();
        return Verify::Bytes;
    }
    return Verify::Plan;
}

void NotePlanVerify(u32 site, bool same) {
    auto& s = S();
    s.plan_lookups[std::min<u32>(site, 3)].fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        s.plan_differs[std::min<u32>(site, 3)].fetch_add(1, std::memory_order_relaxed);
        if (s.logged.fetch_add(1) < 20) {
            LOG_ERROR(Render, "lazy drain: the plan made without waiting differs from the plan made after the full wait (site {})", site);
        }
    }
}

void LookupDone() {
    if (!Active()) {
        return;
    }
    auto& s = S();
    if (verify_bytes) {
        verify_bytes = false;
        std::vector<PendingWrite> unfinished; // the writes this check waits for: a range that changes and lies near one of them is a missed dependency
        if (s.done != nullptr) {
            const u64 last = s.last_write_position.load(std::memory_order_acquire);
            if (last > Completed(s)) {
                {
                    std::scoped_lock lk{s.mutex};
                    const u64 done = Completed(s);
                    for (auto it = s.writes.rbegin(); it != s.writes.rend() && it->position > done; ++it) {
                        unfinished.push_back(*it);
                    }
                }
                s.wait_for(last);
            }
        }
        s.bytes_lookups.fetch_add(1, std::memory_order_relaxed);
        for (const auto& note : copies) {
            u64 now = 0;
            s.bytes_copies.fetch_add(1, std::memory_order_relaxed);
            if (!HashGuest(note.address, note.bytes, now)) {
                s.bytes_unreadable.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (now != note.hash) {
                s.bytes_stale.fetch_add(1, std::memory_order_relaxed);
                const PendingWrite* nearest = nullptr;
                u64 nearest_gap = ~0ull;
                for (const auto& write : unfinished) {
                    if (write.address == NoGuestMemory) {
                        continue;
                    }
                    const u64 lo = write.address, hi = write.address + write.bytes;
                    const u64 gap = note.address >= hi ? note.address - hi : lo >= note.address + note.bytes ? lo - (note.address + note.bytes) : 0;
                    if (gap < nearest_gap) {
                        nearest_gap = gap;
                        nearest = &write;
                    }
                }
                const bool near_write = nearest != nullptr && nearest_gap <= (64u << 10);
                if (near_write) {
                    s.bytes_stale_near_write.fetch_add(1, std::memory_order_relaxed);
                }
                if (s.logged.fetch_add(1) < 20) {
                    if (nearest != nullptr) {
                        LOG_ERROR(Render, "lazy drain: {} copied from {:#x} ({} bytes) changed after the full wait; {} writes were unfinished, the nearest (liverpool.cpp:{}, {:#x}, {} bytes) is {} bytes away",
                                  kind_names[note.kind], note.address, note.bytes, unfinished.size(), nearest->line, nearest->address, nearest->bytes, nearest_gap);
                    } else {
                        LOG_ERROR(Render, "lazy drain: {} copied from {:#x} ({} bytes) changed after the full wait; no write was unfinished (changed by something else)", kind_names[note.kind], note.address, note.bytes);
                    }
                }
            }
        }
        copies.clear();
    }
    if (!Enabled()) {
        return;
    }
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
    out["audit"] = Enabled();
    out["lazy"] = LazyEnabled();
    if (!Active()) {
        out["note"] = "start the game with TLG_DRAIN_AUDIT=1 or TLG_DRAIN_LAZY=1";
        return out;
    }
    out["writes_queued"] = s.writes_queued.load();
    if (LazyEnabled()) {
        nlohmann::json lazy;
        lazy["plannings"] = s.lazy_lookups.load();
        lazy["plannings_that_found_a_write_pending"] = s.lazy_lookups_with_pending.load();
        lazy["waits_at_a_read"] = s.lazy_waits.load();
        lazy["wait_ms_at_reads"] = s.lazy_wait_ns.load() / 1e6;
        nlohmann::json by_kind = nlohmann::json::object();
        for (u32 k = 0; k < KindCount; ++k) {
            by_kind[kind_names[k]] = {{"waits", s.lazy_waits_by_kind[k].load()}, {"ms", s.lazy_wait_ns_by_kind[k].load() / 1e6}};
        }
        lazy["waits_by_kind_of_read"] = by_kind;
        nlohmann::json verify;
        verify["every"] = VerifyEvery();
        verify["bytes"] = {{"plannings_checked", s.bytes_lookups.load()},
                           {"ranges_compared", s.bytes_copies.load()},
                           {"unreadable", s.bytes_unreadable.load()},
                           {"changed_after_the_full_wait", s.bytes_stale.load()},
                           {"of_those_within_64KB_of_a_write_the_full_wait_waited_for", s.bytes_stale_near_write.load()}};
        static constexpr const char* site_names[4] = {"dispatch / draw fallback", "executed draw", "draw from the private registers", "other"};
        nlohmann::json plan = nlohmann::json::object();
        for (u32 i = 0; i < 4; ++i) {
            plan[site_names[i]] = {{"plannings_checked", s.plan_lookups[i].load()}, {"plans_that_differ", s.plan_differs[i].load()}};
        }
        verify["plan"] = plan;
        lazy["verification"] = verify;
        std::scoped_lock lk{s.mutex};
        nlohmann::json by_line = nlohmann::json::object();
        for (const auto& [line, entry] : s.lazy_by_line) {
            by_line[std::to_string(line)] = {{"waits", entry.first}, {"ms", entry.second / 1e6}};
        }
        lazy["waits_by_site_of_the_write"] = by_line;
        out["lazy_wait"] = lazy;
    }
    if (!Enabled()) {
        return out;
    }
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
    nlohmann::json near_misses = nlohmann::json::array();
    for (const auto& [key, count] : s.near_pairs) {
        near_misses.push_back({{"read", kind_names[key >> 32]}, {"write_queued_at_liverpool.cpp_line", key & 0xFFFFFFFFu}, {"reads_within_256KB_but_not_overlapping", count}});
    }
    out["near_misses_by_read_and_write_site"] = near_misses;
    nlohmann::json by_line = nlohmann::json::object();
    for (const auto& [line, count] : s.writes_by_line) {
        by_line[std::to_string(line)] = {{"queued", count}, {"range_unknown", s.unknown_by_line.count(line) ? s.unknown_by_line[line] : 0}, {"bytes_total", s.bytes_by_line[line]}, {"bytes_largest", s.max_bytes_by_line[line]}};
    }
    out["writes_by_site"] = by_line;
    return out;
}

} // namespace Core::DrainAudit
