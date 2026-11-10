// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "native/native_hook_ring.h"
#include "native/native_untracked_ring.h"

namespace Core::HookRing {

namespace {

constexpr u64 GranuleShift = 4; // 16 bytes

std::mutex g_init_mutex;
std::atomic<u8*> g_bytes{nullptr};               // the ring copy: two halves of the ring size
std::atomic<std::atomic<u8>*> g_filled{nullptr}; // per granule: 1 once a finished write was copied since the block was handed out
std::atomic<std::atomic<u8>*> g_half{nullptr};   // per granule: the lap half its block was handed out in
std::atomic<u64> g_base{0}, g_size{0};
u64 g_last_end = 0; // allocator side, under the allocator lock
u8 g_lap_half = 0;

u8* g_storage = nullptr; // mode 2
u64 g_storage_capacity = 0;
void (*g_flush)(void*, u64, u64) = nullptr;
void* g_flush_context = nullptr;

std::atomic<u64> c_fills{0}, c_fill_bytes{0}, c_allocs{0}, c_laps{0}, c_fill_split{0};
std::atomic<u64> c_compares{0}, c_compare_bytes{0}, c_equal{0}, c_diff_compares{0}, c_diff_bytes{0}, c_unfilled_compares{0}, c_unfilled_bytes{0};
std::atomic<u64> c_resolve_asked{0}, c_resolve_outside{0}, c_resolve_unfilled{0}, c_resolve_two_halves{0}, c_bound{0}, c_bound_bytes{0}, c_bound_checked{0};

struct Example {
    u64 offset, size, first_diff, diff_bytes, unfilled;
};
std::mutex g_example_mutex;
std::vector<Example> g_examples;
struct RasterStats {
    std::atomic<u64> compares{0}, bytes{0}, equal{0}, diff_compares{0}, diff_bytes{0}, unfilled_compares{0}, unfilled_bytes{0}, partial{0};
};
RasterStats g_raster[2];
std::atomic<u64> c_raster_wide{0}, c_raster_wide_bytes{0};
std::vector<Example> g_raster_examples;

u32 Mode() noexcept {
    // default 2 since 2026-11-10 (owner play good); =0 turns it off, =1 is the shadow
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_HOOK_RING");
        if (e == nullptr || *e == 0) {
            return 2u;
        }
        return e[0] == '0' ? 0u : e[0] == '1' ? 1u : 2u;
    }();
    return mode;
}

bool Ensure() {
    if (g_bytes.load(std::memory_order_acquire) != nullptr) {
        return true;
    }
    const auto& ring = VideoCore::UntrackedRing::Get();
    if (!ring.Active()) {
        return false;
    }
    std::scoped_lock lock{g_init_mutex};
    if (g_bytes.load(std::memory_order_acquire) != nullptr) {
        return true;
    }
    const u64 base = ring.Start(), size = ring.End() - ring.Start();
    u8* bytes = nullptr;
    if (Mode() == 2) {
        if (g_storage == nullptr || g_storage_capacity < 2 * size) {
            return false; // no GPU-visible storage of the right size: nothing is filled, every range takes the old path
        }
        bytes = g_storage;
    } else {
        bytes = new u8[2 * size](); // kept for the life of the process
    }
    auto* filled = new std::atomic<u8>[size >> GranuleShift];
    auto* half = new std::atomic<u8>[size >> GranuleShift];
    for (u64 i = 0; i < (size >> GranuleShift); ++i) {
        filled[i].store(0, std::memory_order_relaxed);
        half[i].store(0, std::memory_order_relaxed);
    }
    g_base.store(base, std::memory_order_relaxed);
    g_size.store(size, std::memory_order_relaxed);
    g_filled.store(filled, std::memory_order_release);
    g_half.store(half, std::memory_order_release);
    g_bytes.store(bytes, std::memory_order_release);
    return true;
}

bool Clip(u64 address, u64 size, u64& lo, u64& hi) {
    const u64 base = g_base.load(std::memory_order_relaxed), end = base + g_size.load(std::memory_order_relaxed);
    lo = std::max(address, base);
    hi = std::min(address + size, end);
    return lo < hi;
}

/// Where the byte at ring offset `rel` of half `h` lives in the copy.
u64 CopyOffset(u64 rel, u8 h) {
    return static_cast<u64>(h) * g_size.load(std::memory_order_relaxed) + rel;
}

} // namespace

bool On() noexcept {
    return Mode() != 0;
}

bool FillsWriter(u32 writer) noexcept {
    static const u32 mask = [] {
        if (const char* e = std::getenv("TLG_HOOK_RING_WRITERS"); e != nullptr && *e != 0) {
            return static_cast<u32>(std::strtoul(e, nullptr, 0));
        }
        return Mode() == 2 ? (1u << 3) : 0xffu;
    }();
    return Mode() != 0 && writer < 8 && (mask & (1u << writer)) != 0;
}

bool RealOn() noexcept {
    return Mode() == 2;
}

void SetStorage(u8* data, u64 capacity, void (*flush)(void*, u64, u64), void* flush_context) {
    std::scoped_lock lock{g_init_mutex};
    g_storage = data;
    g_storage_capacity = capacity;
    g_flush = flush;
    g_flush_context = flush_context;
}

void NoteAlloc(u64 block, u64 size) {
    if (!On() || block == 0 || size == 0 || !Ensure()) {
        return;
    }
    u64 lo, hi;
    if (!Clip(block, size, lo, hi)) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const u64 rel = lo - base;
    if (rel < g_last_end) {
        g_lap_half ^= 1; // the allocator wrapped: the new lap writes the other half
        c_laps.fetch_add(1, std::memory_order_relaxed);
    }
    g_last_end = hi - base;
    c_allocs.fetch_add(1, std::memory_order_relaxed);
    auto* filled = g_filled.load(std::memory_order_acquire);
    auto* half = g_half.load(std::memory_order_acquire);
    for (u64 g = rel >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        filled[g].store(0, std::memory_order_relaxed);
        half[g].store(g_lap_half, std::memory_order_relaxed);
    }
}

void Fill(u64 address, u64 size) {
    if (!On() || size == 0 || !Ensure()) {
        return;
    }
    u64 lo, hi;
    if (!Clip(address, size, lo, hi)) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    // copy run by run of granules of the same half (a write lies in one block, so this is one run unless the range spans two blocks of two laps)
    u64 run_lo = lo;
    u8 run_half = half[(lo - base) >> GranuleShift].load(std::memory_order_relaxed);
    const auto copy_run = [&](u64 a, u64 b, u8 h) {
        const u64 dst = CopyOffset(a - base, h);
        std::memcpy(bytes + dst, reinterpret_cast<const void*>(a), b - a); // the ring is plain host memory of this process
        if (Mode() == 2 && g_flush != nullptr) {
            g_flush(g_flush_context, dst, b - a);
        }
    };
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        const u8 h = half[g].load(std::memory_order_relaxed);
        if (h != run_half) {
            const u64 at = base + (g << GranuleShift);
            copy_run(run_lo, at, run_half);
            c_fill_split.fetch_add(1, std::memory_order_relaxed);
            run_lo = at;
            run_half = h;
        }
    }
    copy_run(run_lo, hi, run_half);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        filled[g].store(1, std::memory_order_release);
    }
    c_fills.fetch_add(1, std::memory_order_relaxed);
    c_fill_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
}

void Compare(u64 address, u64 size, const void* copied) {
    if (!On() || size == 0 || copied == nullptr) {
        return;
    }
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    if (bytes == nullptr) {
        return;
    }
    u64 lo, hi;
    if (!Clip(address, size, lo, hi) || lo != address || hi != address + size) {
        return; // only ranges wholly inside the ring (the front constant copies are)
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u8* theirs = static_cast<const u8*>(copied);
    u64 diff = 0, unfilled = 0, first_diff = ~0ull;
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)) - lo, g_hi = std::min(hi, base + ((g + 1) << GranuleShift)) - lo;
        if (filled[g].load(std::memory_order_acquire) == 0) {
            unfilled += g_hi - g_lo;
            continue;
        }
        const u8* mine = bytes + CopyOffset(lo - base, half[g].load(std::memory_order_relaxed));
        for (u64 i = g_lo; i < g_hi; ++i) {
            if (mine[i] != theirs[i]) {
                ++diff;
                first_diff = std::min(first_diff, i);
            }
        }
    }
    c_compares.fetch_add(1, std::memory_order_relaxed);
    c_compare_bytes.fetch_add(size, std::memory_order_relaxed);
    if (diff == 0 && unfilled == 0) {
        c_equal.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (diff != 0) {
        c_diff_compares.fetch_add(1, std::memory_order_relaxed);
        c_diff_bytes.fetch_add(diff, std::memory_order_relaxed);
    }
    if (unfilled != 0) {
        c_unfilled_compares.fetch_add(1, std::memory_order_relaxed);
        c_unfilled_bytes.fetch_add(unfilled, std::memory_order_relaxed);
    }
    std::scoped_lock lock{g_example_mutex};
    if (g_examples.size() < 32) {
        g_examples.push_back({lo - base, size, first_diff, diff, unfilled});
    }
}

bool Raster1bOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_HOOK_RING_1B");
        return e != nullptr && e[0] == '1';
    }();
    return on && On();
}

void NoteRasterWide(u64 size) {
    c_raster_wide.fetch_add(1, std::memory_order_relaxed);
    c_raster_wide_bytes.fetch_add(size, std::memory_order_relaxed);
}

void CompareRaster(u32 cls, u64 address, u64 size) {
    if (!Raster1bOn() || size == 0) {
        return;
    }
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    if (bytes == nullptr) {
        return;
    }
    auto& st = g_raster[cls & 1];
    u64 lo, hi;
    if (!Clip(address, size, lo, hi)) {
        return;
    }
    if (lo != address || hi != address + size) {
        st.partial.fetch_add(1, std::memory_order_relaxed); // the caller passes the ring part; counted, compared as clipped
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u8* theirs = reinterpret_cast<const u8*>(lo); // game memory now: what the old path uploads
    u64 diff = 0, unfilled = 0, first_diff = ~0ull;
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)) - lo, g_hi = std::min(hi, base + ((g + 1) << GranuleShift)) - lo;
        if (filled[g].load(std::memory_order_acquire) == 0) {
            unfilled += g_hi - g_lo;
            continue;
        }
        const u8* mine = bytes + CopyOffset(lo - base, half[g].load(std::memory_order_relaxed));
        for (u64 i = g_lo; i < g_hi; ++i) {
            if (mine[i] != theirs[i]) {
                ++diff;
                first_diff = std::min(first_diff, i);
            }
        }
    }
    st.compares.fetch_add(1, std::memory_order_relaxed);
    st.bytes.fetch_add(hi - lo, std::memory_order_relaxed);
    if (diff == 0 && unfilled == 0) {
        st.equal.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (diff != 0) {
        st.diff_compares.fetch_add(1, std::memory_order_relaxed);
        st.diff_bytes.fetch_add(diff, std::memory_order_relaxed);
    }
    if (unfilled != 0) {
        st.unfilled_compares.fetch_add(1, std::memory_order_relaxed);
        st.unfilled_bytes.fetch_add(unfilled, std::memory_order_relaxed);
    }
    std::scoped_lock lock{g_example_mutex};
    if (g_raster_examples.size() < 32) {
        g_raster_examples.push_back({(lo - base) | (static_cast<u64>(cls & 1) << 63), hi - lo, first_diff, diff, unfilled});
    }
}

bool Resolve(u64 address, u64 size, u64& offset) {
    if (!RealOn() || size == 0 || g_bytes.load(std::memory_order_acquire) == nullptr) {
        return false;
    }
    c_resolve_asked.fetch_add(1, std::memory_order_relaxed);
    u64 lo, hi;
    if (!Clip(address, size, lo, hi) || lo != address || hi != address + size) {
        c_resolve_outside.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u64 first = (lo - base) >> GranuleShift, last = (hi - 1 - base) >> GranuleShift;
    const u8 h = half[first].load(std::memory_order_relaxed);
    for (u64 g = first; g <= last; ++g) {
        if (filled[g].load(std::memory_order_acquire) == 0) {
            c_resolve_unfilled.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (half[g].load(std::memory_order_relaxed) != h) {
            c_resolve_two_halves.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    offset = CopyOffset(lo - base, h);
    return true;
}

const u8* Data(u64 offset) {
    const u8* bytes = g_bytes.load(std::memory_order_acquire);
    return bytes != nullptr ? bytes + offset : nullptr;
}

void NoteBound(u64 size, bool checked) {
    c_bound.fetch_add(1, std::memory_order_relaxed);
    c_bound_bytes.fetch_add(size, std::memory_order_relaxed);
    if (checked) {
        c_bound_checked.fetch_add(1, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["mode"] = Mode();
    j["note"] = "TLG_HOOK_RING (F1 class 1a): 1 shadow (ring copy filled at the writes, compared with every front constant copy), 2 real (the front binds constants from the copy, "
                "1 in 64 compared with game memory)";
    j["ring"] = {{"base", fmt::format("{:#x}", g_base.load())}, {"size", g_size.load()}, {"storage", g_storage != nullptr}, {"storage_capacity", g_storage_capacity}};
    j["fills"] = {{"count", c_fills.load()}, {"bytes", c_fill_bytes.load()}, {"split_over_two_halves", c_fill_split.load()}, {"allocs_cleared", c_allocs.load()}, {"laps", c_laps.load()}};
    j["compares"] = {{"count", c_compares.load()},
                     {"bytes", c_compare_bytes.load()},
                     {"equal", c_equal.load()},
                     {"with_diff", c_diff_compares.load()},
                     {"DIFF_BYTES_MUST_BE_0", c_diff_bytes.load()},
                     {"with_unfilled", c_unfilled_compares.load()},
                     {"UNFILLED_BYTES_MUST_BE_0", c_unfilled_bytes.load()}};
    j["real"] = {{"asked", c_resolve_asked.load()},
                 {"bound", c_bound.load()},
                 {"bound_bytes", c_bound_bytes.load()},
                 {"bound_and_checked", c_bound_checked.load()},
                 {"old_path_outside_ring", c_resolve_outside.load()},
                 {"old_path_unfilled", c_resolve_unfilled.load()},
                 {"old_path_two_halves", c_resolve_two_halves.load()}};
    static constexpr const char* cls_names[2] = {"read_up_to_64k", "texel"};
    j["raster_1b"]["on"] = Raster1bOn();
    for (u32 k = 0; k < 2; ++k) {
        const auto& st = g_raster[k];
        j["raster_1b"][cls_names[k]] = {{"compares", st.compares.load()},   {"bytes", st.bytes.load()},
                                        {"equal", st.equal.load()},         {"with_diff", st.diff_compares.load()},
                                        {"DIFF_BYTES_MUST_BE_0", st.diff_bytes.load()}, {"with_unfilled", st.unfilled_compares.load()},
                                        {"UNFILLED_BYTES_MUST_BE_0", st.unfilled_bytes.load()}, {"partial_ranges", st.partial.load()}};
    }
    j["raster_1b"]["wide_reads_old_path"] = {{"count", c_raster_wide.load()}, {"bytes", c_raster_wide_bytes.load()}};
    std::scoped_lock lock{g_example_mutex};
    {
        auto& rl = j["raster_1b"]["examples"] = nlohmann::json::array();
        for (const auto& e : g_raster_examples) {
            rl.push_back({{"class", cls_names[e.offset >> 63]}, {"ring_offset", fmt::format("{:#x}", e.offset & ~(u64{1} << 63))}, {"size", e.size},
                          {"first_diff", e.first_diff == ~0ull ? -1 : static_cast<s64>(e.first_diff)}, {"diff_bytes", e.diff_bytes}, {"unfilled", e.unfilled}});
        }
    }
    auto& list = j["examples"] = nlohmann::json::array();
    for (const auto& e : g_examples) {
        list.push_back({{"ring_offset", fmt::format("{:#x}", e.offset)}, {"size", e.size}, {"first_diff", e.first_diff == ~0ull ? -1 : static_cast<s64>(e.first_diff)},
                        {"diff_bytes", e.diff_bytes}, {"unfilled", e.unfilled}});
    }
    return j;
}

} // namespace Core::HookRing
