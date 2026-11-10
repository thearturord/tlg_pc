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

void NoteLearn(bool inside) noexcept {
    learned[inside ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["mode"] = Mode();
    j["bumps"] = bumps.load();
    j["bumps_unattributed"] = bumps_unattributed.load();
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
