// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_row_version.h"

#include <array>
#include <cstdio>
#include <cstdlib>

#include <nlohmann/json.hpp>

namespace Core::RowVersion {
namespace {
constexpr u32 SlotBits = 16;
std::array<std::atomic<u64>, 1u << SlotBits> slots{};
std::atomic<u64> unattributed{0};
// [inside the buffer][0 checks, 1 both clean, 2 both dirty, 3 VERSION_CLEAN_SETS_DIRTY (MUST_BE_0), 4 version dirty while the sets were clean (extra misses), 5 sets ns]
std::atomic<u64> checks[2][6]{};
std::atomic<u64> learned[2]{};
std::atomic<u64> bumps{0}, bumps_unattributed{0};
std::atomic<u64> real[5]{};
std::atomic<u64> walk[7]{};
std::atomic<u64> old_answer[5][2]{};
std::atomic<u64> static_rows[7][2]{};
std::atomic<u64> static_real[6]{};
std::atomic<u64> written_rows[2]{}; // calls, steps, ns, max steps, none, overlaps, unknown

inline u32 SlotOf(u64 id) noexcept {
    return static_cast<u32>((id * 0x9E3779B97F4A7C15ull) >> (64 - SlotBits));
}
} // namespace

u32 Mode() noexcept {
    static const u32 mode = [] {
        // default 2 (real) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        const char* e = std::getenv("TLG_ROW_VERSION");
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

void Bump(u64 id) noexcept {
    if (Mode() == 0) {
        return;
    }
    slots[SlotOf(id)].fetch_add(1, std::memory_order_acq_rel);
    bumps.fetch_add(1, std::memory_order_relaxed);
}

void BumpUnattributed() noexcept {
    if (Mode() == 0) {
        return;
    }
    unattributed.fetch_add(1, std::memory_order_acq_rel);
    bumps_unattributed.fetch_add(1, std::memory_order_relaxed);
}

u64 Of(u64 id) noexcept {
    return slots[SlotOf(id)].load(std::memory_order_acquire);
}

u64 Unattributed() noexcept {
    return unattributed.load(std::memory_order_acquire);
}

void NoteCheck(bool version_clean, bool sets_dirty, bool inside, u64 sets_ns, u64 id, u64 address, u64 size) noexcept {
    auto& c = checks[inside ? 1 : 0];
    c[0].fetch_add(1, std::memory_order_relaxed);
    c[5].fetch_add(sets_ns, std::memory_order_relaxed);
    if (version_clean && !sets_dirty) {
        c[1].fetch_add(1, std::memory_order_relaxed);
    } else if (!version_clean && sets_dirty) {
        c[2].fetch_add(1, std::memory_order_relaxed);
    } else if (version_clean) {
        c[3].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::fprintf(stderr, "F2 row version: id %#llx %#llx+%#llx (inside %d): the versions say clean, the sets say dirty\n", static_cast<unsigned long long>(id),
                         static_cast<unsigned long long>(address), static_cast<unsigned long long>(size), inside ? 1 : 0);
        }
    } else {
        c[4].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteReal(u32 what) noexcept {
    if (what < 5) {
        real[what].fetch_add(1, std::memory_order_relaxed);
    }
}

bool WalkCensusOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_ROW_WALK_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteWalk(u64 steps, u64 ns, u32 answer) noexcept {
    walk[0].fetch_add(1, std::memory_order_relaxed);
    walk[1].fetch_add(steps, std::memory_order_relaxed);
    walk[2].fetch_add(ns, std::memory_order_relaxed);
    u64 mx = walk[3].load(std::memory_order_relaxed);
    while (steps > mx && !walk[3].compare_exchange_weak(mx, steps)) {
    }
    walk[4 + (answer < 3 ? answer : 2)].fetch_add(1, std::memory_order_relaxed);
}

void NoteOldAnswer(u32 what, u64 bytes) noexcept {
    if (what < 5) {
        old_answer[what][0].fetch_add(1, std::memory_order_relaxed);
        old_answer[what][1].fetch_add(bytes, std::memory_order_relaxed);
    }
}

bool StaticCensusOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_STATIC_ROW_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteStatic(u32 what, u64 bytes) noexcept {
    if (what < 7) {
        static_rows[what][0].fetch_add(1, std::memory_order_relaxed);
        static_rows[what][1].fetch_add(bytes, std::memory_order_relaxed);
    }
}

bool Watched(u64 address, u64 size) noexcept {
    static const u64 watch = [] {
        const char* e = std::getenv("TLG_WATCH_ADDR");
        return e != nullptr ? static_cast<u64>(std::strtoull(e, nullptr, 0)) : u64{0};
    }();
    return watch != 0 && address <= watch && watch < address + size;
}

bool StaticRowsOn() noexcept {
    static const bool on = [] {
        // default on since 2026-11-10 after the owner's play; =0 off
        const char* e = std::getenv("TLG_STATIC_ROWS");
        return e == nullptr || e[0] == 0 || e[0] != '0';
    }();
    return on;
}

void NoteStaticReal(u32 what, u64 n) noexcept {
    if (what < 6) {
        static_real[what].fetch_add(n, std::memory_order_relaxed);
    }
}

bool WrittenRowsOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_WRITTEN_ROWS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteWritten(u32 what) noexcept {
    if (what < 2) {
        written_rows[what].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteLearn(bool inside) noexcept {
    learned[inside ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["mode"] = Mode();
    j["bumps"] = bumps.load();
    j["bumps_unattributed"] = bumps_unattributed.load();
    j["identity_walk"] = {{"calls", walk[0].load()}, {"events_stepped", walk[1].load()}, {"ms", walk[2].load() / 1e6}, {"max_events_one_walk", walk[3].load()},
                          {"none", walk[4].load()}, {"overlaps", walk[5].load()}, {"unknown", walk[6].load()}};
    {
        static constexpr const char* Names[5] = {"hook_ring", "stream_ring_in_frame_ring", "stream_ring_outside_frame_ring", "buffer_in_frame_ring", "buffer_outside_frame_ring"};
        for (u32 k = 0; k < 5; ++k) {
            j["old_path_answers_without_row"][Names[k]] = {{"reads", old_answer[k][0].load()}, {"bytes", old_answer[k][1].load()}};
        }
    }
    {
        static constexpr const char* Names[7] = {"no_previous_copy", "version_same_bytes_same_ROW_RIGHT", "VERSION_SAME_BYTES_DIFFERENT_MUST_BE_0", "version_moved_bytes_same_extra_miss",
                                                 "version_moved_bytes_different_right_miss", "not_inside_its_buffer", "not_hashed"};
        for (u32 k = 0; k < 7; ++k) {
            j["static_row_census"][Names[k]] = {{"reads", static_rows[k][0].load()}, {"bytes", static_rows[k][1].load()}};
        }
    }
    j["written_rows"] = {{"on", WrittenRowsOn()}, {"served", written_rows[0].load()}, {"learned", written_rows[1].load()}};
    j["static_rows"] = {{"on", StaticRowsOn()}, {"learned", static_real[0].load()}, {"served", static_real[1].load()}, {"served_checked_by_hash", static_real[2].load()},
                        {"STATIC_BYTES_CHANGED_MUST_BE_0", static_real[3].load()}, {"bytes_uploaded_at_learns", static_real[4].load()}, {"learns_refused", static_real[5].load()}};
    j["real"] = {{"served_by_version", real[0].load()}, {"also_asked_of_the_sets", real[1].load()}, {"SETS_DIRTY_VERSION_CLEAN_MUST_BE_0", real[2].load()},
                 {"not_learned_outside_its_buffer", real[3].load()}, {"version_changed", real[4].load()}};
    for (u32 i = 0; i < 2; ++i) {
        auto& c = checks[i];
        j[i ? "inside_buffer" : "not_inside_buffer"] = {{"checks", c[0].load()},
                                                        {"both_clean", c[1].load()},
                                                        {"both_dirty", c[2].load()},
                                                        {"VERSION_CLEAN_SETS_DIRTY_MUST_BE_0", c[3].load()},
                                                        {"version_dirty_sets_clean", c[4].load()},
                                                        {"sets_ms", c[5].load() / 1e6},
                                                        {"rows_learned", learned[i].load()}};
    }
    return j;
}

} // namespace Core::RowVersion
