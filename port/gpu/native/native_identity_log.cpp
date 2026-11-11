// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "native/native_identity_log.h"
#include <cstdlib>

namespace Core::NativeIdentity {

namespace {
constexpr u64 RingSize = u64{1} << 16;

struct Entry {
    std::atomic<u64> sequence{0}; ///< the event number once the entry is complete, 0 while it is written or empty
    std::atomic<u64> begin{0}, end{0};
    std::atomic<u32> kind{0};
};

std::array<Entry, RingSize> ring;
std::atomic<u64> clock{0};
std::array<std::atomic<u64>, KindCount> counts{};
} // namespace

/// Notes 140, 141: the events are walked only by the recipe censuses (the rows that walked them are deleted); with neither on nothing is recorded.
bool Needed() noexcept {
    static const bool on = [] {
        const auto one = [](const char* n) { const char* e = std::getenv(n); return e != nullptr && e[0] == '1'; };
        return one("TLG_REUSE_CENSUS") || one("TLG_VERSION_CENSUS");
    }();
    return on;
}

void Note(u64 begin, u64 end, Kind kind) noexcept {
    if (!Needed()) {
        return;
    }
    const u64 number = clock.fetch_add(1, std::memory_order_acq_rel) + 1;
    Entry& entry = ring[number % RingSize];
    entry.sequence.store(0, std::memory_order_release);
    entry.begin.store(begin, std::memory_order_relaxed);
    entry.end.store(end > begin ? end : begin + 1, std::memory_order_relaxed);
    entry.kind.store(kind, std::memory_order_relaxed);
    entry.sequence.store(number, std::memory_order_release);
    counts[kind < KindCount ? kind : 0].fetch_add(1, std::memory_order_relaxed);
}

u64 Clock() noexcept {
    return clock.load(std::memory_order_acquire);
}

Change ChangedSince(u64 since, u64 begin, u64 end, Kind* kind_out) noexcept {
    const u64 now = clock.load(std::memory_order_acquire);
    if (now <= since) {
        return Change::None;
    }
    if (now - since >= RingSize - 16) {
        return Change::Unknown;
    }
    if (end <= begin) {
        end = begin + 1;
    }
    for (u64 number = since + 1; number <= now; ++number) {
        const Entry& entry = ring[number % RingSize];
        if (entry.sequence.load(std::memory_order_acquire) != number) {
            return Change::Unknown; // written right now by another thread, or overwritten
        }
        const u64 entry_begin = entry.begin.load(std::memory_order_relaxed);
        const u64 entry_end = entry.end.load(std::memory_order_relaxed);
        const u32 kind = entry.kind.load(std::memory_order_relaxed);
        if (entry.sequence.load(std::memory_order_acquire) != number) {
            return Change::Unknown;
        }
        if (entry_begin < end && begin < entry_end) {
            if (kind_out != nullptr) {
                *kind_out = static_cast<Kind>(kind);
            }
            return Change::Overlaps;
        }
    }
    return Change::None;
}

u64 Count(Kind kind) noexcept {
    return counts[kind < KindCount ? kind : 0].load(std::memory_order_relaxed);
}

} // namespace Core::NativeIdentity
