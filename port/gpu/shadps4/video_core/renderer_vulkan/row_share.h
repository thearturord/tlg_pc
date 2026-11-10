// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Rows ahead (design review sections 53 to 55, `TLG_ROWS_AHEAD`): the buffer rows the raster thread learned, published so the front thread can read them when it plans a
// draw. The raster thread is the only writer; any thread may read. One slot per key hash (a collision overwrites: a row not found is only a missed shortcut). Each slot has a
// sequence counter (odd while written), so a reader takes a whole row or none.

#include "native/native_legacy.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>

#include "common/types.h"

namespace Vulkan::RowShare {

/// The key of a row: engine id, guest range, texel and written flags (the raster thread's `RowKey`).
[[nodiscard]] inline u64 Key(u64 id, u64 address, u64 size, bool texel, bool written) noexcept {
    u64 h = 0xcbf29ce484222325ull;
    for (const u64 v : {id, address, size, static_cast<u64>(texel) | (written ? 2u : 0u)}) {
        h = (h ^ v) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    return h;
}

enum RowFlags : u64 { FlagTexel = 1, FlagWritten = 2, FlagInside = 4, FlagStatic = 8 };

struct Row {
    u64 id{}, address{}, size{}, flags{};
    const void* buffer{};
    u64 offset{}, epoch{}, version{};
    u64 clock{}; ///< the identity clock the row was last found valid at (the raster thread moves it at each use)
};

namespace detail {
constexpr u32 Bits = 18;
struct Slot {
    std::atomic<u64> seq{0};
    std::atomic<u64> key{0};
    std::array<std::atomic<u64>, 8> f{};
    std::atomic<u64> clock{0};
};
inline Slot* Table() noexcept {
    static const std::unique_ptr<Slot[]> table{new Slot[std::size_t{1} << Bits]};
    return table.get();
}
inline Slot& SlotOf(u64 key) noexcept {
    return Table()[(key * 0x9E3779B97F4A7C15ull) >> (64 - Bits)];
}
} // namespace detail

/// Raster thread: the row of `key` is (now) this.
inline void Publish(u64 key, const Row& r) noexcept {
    auto& s = detail::SlotOf(key);
    const u64 q = s.seq.load(std::memory_order_relaxed);
    s.seq.store(q + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    s.key.store(key, std::memory_order_relaxed);
    const u64 v[8] = {r.id, r.address, r.size, r.flags, reinterpret_cast<u64>(r.buffer), r.offset, r.epoch, r.version};
    for (u32 i = 0; i < 8; ++i) {
        s.f[i].store(v[i], std::memory_order_relaxed);
    }
    s.clock.store(r.clock, std::memory_order_relaxed);
    s.seq.store(q + 2, std::memory_order_release);
}

/// Raster thread: the row of `key` was found valid again at `clock` (a later check walks the identity events from there).
inline void TouchClock(u64 key, u64 clock) noexcept {
    auto& s = detail::SlotOf(key);
    if (s.key.load(std::memory_order_relaxed) == key) {
        s.clock.store(clock, std::memory_order_relaxed);
    }
}

/// Raster thread: the row of `key` is gone.
inline void Erase(u64 key) noexcept {
    auto& s = detail::SlotOf(key);
    if (s.key.load(std::memory_order_relaxed) != key) {
        return;
    }
    const u64 q = s.seq.load(std::memory_order_relaxed);
    s.seq.store(q + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    s.key.store(0, std::memory_order_relaxed);
    s.seq.store(q + 2, std::memory_order_release);
}

/// Any thread: the row of `key` as last published (false: none, or being written).
[[nodiscard]] inline bool Find(u64 key, Row& out) noexcept {
    auto& s = detail::SlotOf(key);
    const u64 q1 = s.seq.load(std::memory_order_acquire);
    if ((q1 & 1) != 0 || s.key.load(std::memory_order_relaxed) != key) {
        return false;
    }
    u64 v[8];
    for (u32 i = 0; i < 8; ++i) {
        v[i] = s.f[i].load(std::memory_order_relaxed);
    }
    const u64 clock = s.clock.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s.seq.load(std::memory_order_relaxed) != q1) {
        return false;
    }
    out = Row{v[0], v[1], v[2], v[3], reinterpret_cast<const void*>(v[4]), v[5], v[6], v[7], clock};
    return true;
}

/// 2 real (default since 2026-11-10 after the owner's play: a planned row valid at the bind is bound without the stage loop's lookup; one in 64 takes the full path and is
/// compared), 1 shadow, 0 off.
[[nodiscard]] inline u32 Mode() noexcept {
    static const u32 mode = [] {
        if (!Core::Legacy::EngineRows()) {
            return 0u; // LEGACY(engine_rows): rows ahead deleted by default (design review 89); TLG_LEGACY_ENGINE_ROWS=1 brings them back
        }
        const char* e = std::getenv("TLG_ROWS_AHEAD");
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

} // namespace Vulkan::RowShare
