// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "native/native_draw.h"
#include "native/native_device.h"
#include "native/native_content.h"
#include <cstdio>
#include "native/native_hook_ring.h"
#include "native/native_hook_diff.h"
#include "native/native_ring_mirror.h"
#include "native/native_updates.h"
#include "native/native_untracked_ring.h"

namespace Core::HookRing {

u32 Raster1bMode() noexcept; // 1b mode: 0 off, 1 shadow, 2 real (defined below)

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
RasterStats g_raster[4]; // note 168: [class | (bound answer ? 0 : 2)]
std::atomic<u64> c_bound_refused{0};
std::atomic<u64> c_raster_wide{0}, c_raster_wide_bytes{0};
// 1b shadow, why a raster upload differs from the hook ring (one count per differing compare, the first rule that holds)
std::atomic<u32>* g_fill_seq = nullptr; // per granule: the record sequence number when it was last filled
std::atomic<u8>* g_fill_writer = nullptr; // per granule: the writer of that fill (8: our redirects)
std::atomic<u64> c_why_landed{0}, c_why_open_map{0}, c_why_gpu_written{0}, c_why_none{0}, c_landings{0}, c_landing_bytes{0};
std::atomic<u64> c_why_writer[9]{};  // the writer of the last fill of the first differing granule
std::atomic<u64> c_why_age[5]{};     // records since that fill: <100, <1k, <10k, <100k, more
std::atomic<u64> c_why_whole{0};
// Map window tags of writer 0 fills
struct FillTag {
    u32 how, caller, map_type;
    u64 resource;
};
thread_local FillTag t_tag{};
thread_local bool t_tag_set = false;
std::mutex g_tag_mutex;
std::vector<FillTag> g_tags{FillTag{}}; // index 0: no tag
constexpr u32 MaxTags = 4096;
std::atomic<u16>* g_fill_tag = nullptr; // per granule
std::array<std::atomic<u64>, MaxTags> c_tag_fills{}, c_tag_diffs{}, c_tag_diff_bytes{};
std::atomic<u64> c_how_fills[4]{}, c_how_diffs[4]{};

u16 TagIndex(const FillTag& t) {
    std::scoped_lock lock{g_tag_mutex};
    for (u32 i = 1; i < g_tags.size(); ++i) {
        const auto& e = g_tags[i];
        if (e.how == t.how && e.caller == t.caller && e.resource == t.resource && e.map_type == t.map_type) {
            return static_cast<u16>(i);
        }
    }
    if (g_tags.size() < MaxTags - 1) {
        g_tags.push_back(t);
        return static_cast<u16>(g_tags.size() - 1);
    }
    return static_cast<u16>(MaxTags - 1);
}     // every filled byte of the range differs in at least... : the first differing byte is at offset 0
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
    g_fill_tag = new std::atomic<u16>[size >> GranuleShift];
    for (u64 i = 0; i < (size >> GranuleShift); ++i) {
        g_fill_tag[i].store(0, std::memory_order_relaxed);
    }
    g_fill_seq = new std::atomic<u32>[size >> GranuleShift];
    g_fill_writer = new std::atomic<u8>[size >> GranuleShift];
    for (u64 i = 0; i < (size >> GranuleShift); ++i) {
        g_fill_seq[i].store(0, std::memory_order_relaxed);
        g_fill_writer[i].store(0xff, std::memory_order_relaxed);
    }
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
        return Mode() == 2 && Raster1bMode() == 0 ? (1u << 3) : 0xffu; // 1b (shadow or real) needs every writer
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

void Fill(u64 address, u64 size, u32 writer) {
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
    const u32 seq_now = Raster1bOn() ? static_cast<u32>(Core::NativeDraw::NextSeq()) : 0u;
    u16 tag = 0;
    if (Raster1bOn() && writer == 0 && t_tag_set) {
        tag = TagIndex(t_tag);
        c_tag_fills[tag].fetch_add(1, std::memory_order_relaxed);
        c_how_fills[t_tag.how & 3].fetch_add(1, std::memory_order_relaxed);
    }
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        if (Raster1bOn()) {
            g_fill_seq[g].store(seq_now, std::memory_order_relaxed);
            g_fill_writer[g].store(static_cast<u8>(std::min<u32>(writer, 8)), std::memory_order_relaxed);
            g_fill_tag[g].store(tag, std::memory_order_relaxed);
        }
        filled[g].store(1, std::memory_order_release); // (also clears the "a GPU result landed" bit)
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

u32 Raster1bMode() noexcept {
    static const u32 mode = [] {
        // default 2 (real) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        const char* e = std::getenv("TLG_HOOK_RING_1B");
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return On() ? mode : 0u;
}

bool Raster1bOn() noexcept {
    return Raster1bMode() != 0;
}

bool Raster1bReal() noexcept {
    return Raster1bMode() == 2 && RealOn();
}

std::atomic<u64> c_raster_bound[2]{}, c_raster_bound_bytes[2]{}, c_raster_bound_checked{0};

void NoteRasterBound(u32 cls, u64 size, bool checked) {
    c_raster_bound[cls & 1].fetch_add(1, std::memory_order_relaxed);
    c_raster_bound_bytes[cls & 1].fetch_add(size, std::memory_order_relaxed);
    if (checked) {
        c_raster_bound_checked.fetch_add(1, std::memory_order_relaxed);
    }
}

u32 WideRowsMode() noexcept {
    static const u32 mode = [] {
        // default 2 (real) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        const char* e = std::getenv("TLG_WIDE_ROWS");
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

std::atomic<u64> c_wide_rows[11]{};
void NoteWideRows(u32 what) noexcept {
    if (what < 11) {
        c_wide_rows[what].fetch_add(1, std::memory_order_relaxed);
    }
}

std::atomic<u64> c_past_end[7]{};

u32 WidePastEndMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_WIDE_PAST_END");
        if (e == nullptr || e[0] == 0) {
            return 2u; // note 167: default 2 since 2026-11-12 (owner play p6play); LEGACY(wide_past_end): =0 those windows through ObtainBuffer, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

void NotePastEnd(u32 what) noexcept {
    if (what < 7) {
        c_past_end[what].fetch_add(1, std::memory_order_relaxed);
    }
}

bool ResolveWidePastEnd(u64 address, u64 size, u64& offset) {
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    const u64 base = g_base.load(std::memory_order_relaxed), ring_size = g_size.load(std::memory_order_relaxed);
    if (bytes == nullptr || size == 0 || address < base || address >= base + ring_size || address + size <= base + ring_size) {
        return false;
    }
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u64 first = (address - base) >> GranuleShift;
    if (filled[first].load(std::memory_order_acquire) == 0) {
        return false;
    }
    const u8 h0 = half[first].load(std::memory_order_relaxed);
    // the buffer's own block (the filled prefix in one lap half) must end inside the ring: a block that reaches the ring's end goes on in memory the hook ring does not hold
    const u64 last = (ring_size - 1) >> GranuleShift;
    bool ended = false;
    for (u64 g = first; g <= last; ++g) {
        if (filled[g].load(std::memory_order_acquire) == 0 || half[g].load(std::memory_order_relaxed) != h0) {
            ended = true;
            break;
        }
    }
    if (!ended) {
        return false;
    }
    offset = CopyOffset(address - base, h0);
    return g_storage_capacity != 0 && offset + size <= g_storage_capacity; // (the descriptor's range must stay inside the hook ring's buffer)
}

bool ResolveWide(u64 address, u64 size, u64& offset, u64& prefix) {
    if (g_bytes.load(std::memory_order_acquire) == nullptr || size == 0) {
        return false;
    }
    u64 lo, hi;
    if (!Clip(address, size, lo, hi) || lo != address || hi != address + size) {
        return false;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u64 first = (lo - base) >> GranuleShift, last = (hi - 1 - base) >> GranuleShift;
    if (filled[first].load(std::memory_order_acquire) == 0) {
        NoteWideRows(2);
        return false;
    }
    const u8 h = half[first].load(std::memory_order_relaxed);
    u64 g = first;
    // the buffer's own block: the filled granules from the window's start in the start's lap half (a granule of the other half is a block of another lap: the block ended)
    for (; g <= last && filled[g].load(std::memory_order_acquire) != 0 && half[g].load(std::memory_order_relaxed) == h; ++g) {
    }
    prefix = std::min(hi, base + (g << GranuleShift)) - lo;
    offset = CopyOffset(lo - base, h);
    NoteWideRows(4);
    return true;
}

bool ResolveWideFast(u64 address, u64 size, u64& offset) {
    if (g_bytes.load(std::memory_order_acquire) == nullptr || size == 0) {
        return false;
    }
    u64 lo, hi;
    if (!Clip(address, size, lo, hi) || lo != address || hi != address + size) {
        return false;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const u64 first = (lo - base) >> GranuleShift;
    if (g_filled.load(std::memory_order_acquire)[first].load(std::memory_order_acquire) == 0) {
        NoteWideRows(2);
        return false;
    }
    offset = CopyOffset(lo - base, g_half.load(std::memory_order_acquire)[first].load(std::memory_order_relaxed));
    NoteWideRows(4);
    return true;
}

u64 ComparePrefix(u64 address, u64 size) {
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    u64 lo, hi;
    if (bytes == nullptr || size == 0 || !Clip(address, size, lo, hi)) {
        return 0;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u8* theirs = reinterpret_cast<const u8*>(lo);
    u64 diff = 0;
    const u8 h0 = half[(lo - base) >> GranuleShift].load(std::memory_order_relaxed);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        if (filled[g].load(std::memory_order_acquire) == 0 || half[g].load(std::memory_order_relaxed) != h0) {
            break; // the end of the prefix (the buffer's own block)
        }
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)) - lo, g_hi = std::min(hi, base + ((g + 1) << GranuleShift)) - lo;
        const u8* mine = bytes + CopyOffset(lo - base, half[g].load(std::memory_order_relaxed));
        for (u64 i = g_lo; i < g_hi; ++i) {
            diff += mine[i] != theirs[i] ? 1 : 0;
        }
    }
    return diff;
}

bool WideCensusOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_WIDE_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// 0 windows, 1 bytes, 2 filled prefix bytes, 3 diff bytes in the prefix, 4 diff bytes after it (filled granules), 5 unfilled bytes, 6 windows with a diff in the prefix,
// 7 windows with a diff after it, 8 windows wholly filled, 9 granules that differ, by writer below
std::atomic<u64> c_wide[10]{};
std::atomic<u64> c_wide_diff_writer[9]{};
std::atomic<u64> c_wide_landed[2]{}, c_wide_probed{0};
std::array<std::atomic<u64>, MaxTags> c_wide_prefix_tag_granules{}; // granules changed in the prefix, not landed, by fill tag
std::mutex g_wide_mutex;
std::vector<std::array<u64, 5>> g_wide_examples; // ring offset of the window, its first differing ring offset, diff bytes, unfilled bytes, prefix bytes

void CompareWide(u64 address, u64 size, bool probe) {
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    u64 lo, hi;
    if (bytes == nullptr || size == 0 || !Clip(address, size, lo, hi)) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    const auto* filled = g_filled.load(std::memory_order_acquire);
    const auto* half = g_half.load(std::memory_order_acquire);
    const u8* theirs = reinterpret_cast<const u8*>(lo);
    u64 prefix = 0, diff_prefix = 0, diff_after = 0, unfilled = 0, first_diff = ~0ull, first_prefix_diff = ~0ull, granules_differ = 0;
    bool in_prefix = true;
    const u8 prefix_half = half[(lo - base) >> GranuleShift].load(std::memory_order_relaxed);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)) - lo, g_hi = std::min(hi, base + ((g + 1) << GranuleShift)) - lo;
        if (filled[g].load(std::memory_order_acquire) == 0) {
            unfilled += g_hi - g_lo;
            in_prefix = false;
            continue;
        }
        if (half[g].load(std::memory_order_relaxed) != prefix_half) {
            in_prefix = false; // a block of another lap: the buffer's own block ended
        }
        const u8* mine = bytes + CopyOffset(lo - base, half[g].load(std::memory_order_relaxed));
        u64 d = 0;
        for (u64 i = g_lo; i < g_hi; ++i) {
            if (mine[i] != theirs[i]) {
                ++d;
                first_diff = std::min(first_diff, i);
            }
        }
        if (d != 0) {
            ++granules_differ;
            c_wide_diff_writer[std::min<u32>(g_fill_writer[g].load(std::memory_order_relaxed), 8)].fetch_add(1, std::memory_order_relaxed);
            if ((filled[g].load(std::memory_order_relaxed) & 2) != 0) {
                c_wide_landed[in_prefix ? 0 : 1].fetch_add(1, std::memory_order_relaxed); // a GPU result landed there since the fill
            } else if (in_prefix) {
                c_wide_prefix_tag_granules[g_fill_tag[g].load(std::memory_order_relaxed) % MaxTags].fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (in_prefix) {
            prefix += g_hi - g_lo;
            diff_prefix += d;
            if (d != 0 && first_prefix_diff == ~0ull) {
                for (u64 i = g_lo; i < g_hi; ++i) {
                    if (mine[i] != theirs[i]) {
                        first_prefix_diff = i;
                        break;
                    }
                }
            }
        } else {
            diff_after += d;
        }
    }
    c_wide[0].fetch_add(1, std::memory_order_relaxed);
    c_wide[1].fetch_add(hi - lo, std::memory_order_relaxed);
    c_wide[2].fetch_add(prefix, std::memory_order_relaxed);
    c_wide[3].fetch_add(diff_prefix, std::memory_order_relaxed);
    c_wide[4].fetch_add(diff_after, std::memory_order_relaxed);
    c_wide[5].fetch_add(unfilled, std::memory_order_relaxed);
    c_wide[6].fetch_add(diff_prefix != 0 ? 1 : 0, std::memory_order_relaxed);
    c_wide[7].fetch_add(diff_after != 0 ? 1 : 0, std::memory_order_relaxed);
    c_wide[8].fetch_add(unfilled == 0 ? 1 : 0, std::memory_order_relaxed);
    c_wide[9].fetch_add(granules_differ, std::memory_order_relaxed);
    if (first_prefix_diff != ~0ull) {
        // the details of the change in the buffer's own block (the first 24 windows that have one)
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
            const u64 at = lo + first_prefix_diff;
            const u64 g = (at - base) >> GranuleShift;
            const u16 tag = g_fill_tag[g].load(std::memory_order_relaxed);
            FillTag t{};
            {
                std::scoped_lock lock{g_tag_mutex};
                if (tag < g_tags.size()) {
                    t = g_tags[tag];
                }
            }
            Core::NativeContent::BindContext bc{};
            const bool have_bc = Core::NativeContent::CurrentBindContext(bc);
            std::fprintf(stderr,
                         "wide census: window %#llx+%#llx (bound id %#llx program %#llx) prefix %#llx: changed byte at %#llx (ring+%#llx), fill writer %u age %u seq, fill tag how %u caller %#x "
                         "resource %#llx map type %u, open Map over it %d, landed %d; Maps covering it:%s\n",
                         static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi - lo), static_cast<unsigned long long>(have_bc ? bc.engine_id : 0),
                         static_cast<unsigned long long>(have_bc ? bc.program : 0), static_cast<unsigned long long>(prefix), static_cast<unsigned long long>(at),
                         static_cast<unsigned long long>(at - base), g_fill_writer[g].load(std::memory_order_relaxed),
                         static_cast<u32>(Core::NativeDraw::NextSeq()) - g_fill_seq[g].load(std::memory_order_relaxed), t.how, t.caller, static_cast<unsigned long long>(t.resource), t.map_type,
                         Core::NativeUpdates::OpenMapOverlaps(at, 1) ? 1 : 0, (filled[g].load(std::memory_order_relaxed) & 2) != 0 ? 1 : 0,
                         Core::NativeDevice::MapsCovering(at).c_str());
        }
    }
    if (first_prefix_diff != ~0ull && probe) {
        Core::RingMirror::ProbeWrites(lo + first_prefix_diff); // (the buffer's own block only)
        c_wide_probed.fetch_add(1, std::memory_order_relaxed);
    }
    std::scoped_lock lock{g_wide_mutex};
    if (g_wide_examples.size() < 24) {
        g_wide_examples.push_back({lo - base, first_diff == ~0ull ? ~0ull : lo - base + first_diff, diff_prefix + diff_after, unfilled, prefix});
    }
}

void NoteRasterWide(u64 size) {
    c_raster_wide.fetch_add(1, std::memory_order_relaxed);
    c_raster_wide_bytes.fetch_add(size, std::memory_order_relaxed);
}

void SetFillTag(u32 how, u32 caller_ghidra, u64 resource_id, u32 map_type) {
    t_tag = {how, caller_ghidra, map_type, resource_id};
    t_tag_set = true;
}

void ClearFillTag() {
    t_tag_set = false;
}

void NoteLanding(u64 address, u64 size) {
    if (!Raster1bOn() || size == 0 || g_bytes.load(std::memory_order_acquire) == nullptr) {
        return;
    }
    if (WideRowsMode() != 0) {
        // F1 wide windows: the GPU's result is in game memory now (callers note a landing after the write); the hook ring takes it too
        u64 lo, hi;
        if (Clip(address, size, lo, hi)) {
            Fill(lo, hi - lo, 8);
            NoteWideRows(10);
            c_landings.fetch_add(1, std::memory_order_relaxed);
            c_landing_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
        }
        return;
    }
    u64 lo, hi;
    if (!Clip(address, size, lo, hi)) {
        return;
    }
    const u64 base = g_base.load(std::memory_order_relaxed);
    auto* filled = g_filled.load(std::memory_order_acquire);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        filled[g].fetch_or(2, std::memory_order_relaxed);
    }
    c_landings.fetch_add(1, std::memory_order_relaxed);
    c_landing_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
}

u64 CompareRaster(u32 cls, u64 address, u64 size, bool gpu_written, bool bound) {
    if (!Raster1bOn() || size == 0) {
        return 0;
    }
    u8* bytes = g_bytes.load(std::memory_order_acquire);
    if (bytes == nullptr) {
        return 0;
    }
    auto& st = g_raster[(cls & 1) | (bound ? 0u : 2u)];
    u64 lo, hi;
    if (!Clip(address, size, lo, hi)) {
        return 0;
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
        return diff;
    }
    if (diff != 0) {
        st.diff_compares.fetch_add(1, std::memory_order_relaxed);
        st.diff_bytes.fetch_add(diff, std::memory_order_relaxed);
        // why: the first rule that holds
        bool landed = false;
        for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift && !landed; ++g) {
            landed = (filled[g].load(std::memory_order_relaxed) & 2) != 0;
        }
        if (landed) {
            c_why_landed.fetch_add(1, std::memory_order_relaxed);
        } else if (Core::NativeUpdates::OpenMapOverlaps(lo, hi - lo)) {
            c_why_open_map.fetch_add(1, std::memory_order_relaxed);
        } else if (gpu_written) {
            c_why_gpu_written.fetch_add(1, std::memory_order_relaxed);
        } else {
            c_why_none.fetch_add(1, std::memory_order_relaxed);
        }
        const u64 g0 = (lo - base + first_diff) >> GranuleShift;
        c_why_writer[std::min<u32>(g_fill_writer[g0].load(std::memory_order_relaxed), 8)].fetch_add(1, std::memory_order_relaxed);
        const u32 age = static_cast<u32>(Core::NativeDraw::NextSeq()) - g_fill_seq[g0].load(std::memory_order_relaxed);
        c_why_age[age < 100 ? 0 : age < 1000 ? 1 : age < 10000 ? 2 : age < 100000 ? 3 : 4].fetch_add(1, std::memory_order_relaxed);
        if (first_diff == 0) {
            c_why_whole.fetch_add(1, std::memory_order_relaxed);
        }
        const u16 tag = g_fill_tag[g0].load(std::memory_order_relaxed);
        c_tag_diffs[tag].fetch_add(1, std::memory_order_relaxed);
        c_tag_diff_bytes[tag].fetch_add(diff, std::memory_order_relaxed);
        if (tag != 0) {
            std::scoped_lock lock{g_tag_mutex};
            c_how_diffs[g_tags[tag].how & 3].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (unfilled != 0) {
        st.unfilled_compares.fetch_add(1, std::memory_order_relaxed);
        st.unfilled_bytes.fetch_add(unfilled, std::memory_order_relaxed);
    }
    std::scoped_lock lock{g_example_mutex};
    if (g_raster_examples.size() < 32) {
        g_raster_examples.push_back({(lo - base) | (static_cast<u64>(cls & 1) << 63), hi - lo, first_diff, diff, unfilled});
    }
    return diff;
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

std::atomic<u64> c_open_guard[3]{};

u32 OpenMapGuardMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OPEN_MAP_GUARD");
        if (e == nullptr || e[0] == 0) {
            return 2u; // note 168: default 2 since 2026-11-12 (owner play p7play); LEGACY(open_map_guard): =0 answers over an open Map given, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

void NoteOpenMapGuard(u32 what) noexcept {
    if (what < 3) {
        c_open_guard[what].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteBoundRefused() noexcept {
    c_bound_refused.fetch_add(1, std::memory_order_relaxed);
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
                     {"unfilled_bytes_the_front_copy_was_bound", c_unfilled_bytes.load()}}; // note 168: an unfilled range is bound from the front's copy, as designed
    j["real"] = {{"asked", c_resolve_asked.load()},
                 {"bound", c_bound.load()},
                 {"bound_bytes", c_bound_bytes.load()},
                 {"bound_and_checked", c_bound_checked.load()},
                 {"old_path_outside_ring", c_resolve_outside.load()},
                 {"old_path_unfilled", c_resolve_unfilled.load()},
                 {"old_path_two_halves", c_resolve_two_halves.load()}};
    static constexpr const char* cls_names[2] = {"read_up_to_64k", "texel"};
    {
        auto& w = j["wide_windows_census"];
        w["on"] = WideCensusOn();
        static constexpr const char* Names[10] = {"windows", "bytes", "filled_prefix_bytes", "DIFF_BYTES_IN_PREFIX", "diff_bytes_after_prefix", "unfilled_bytes",
                                                  "windows_with_diff_in_prefix", "windows_with_diff_after_prefix", "windows_wholly_filled", "granules_differing"};
        for (u32 k = 0; k < 10; ++k) {
            w[Names[k]] = c_wide[k].load();
        }
        for (u32 k = 0; k < 9; ++k) {
            w["differing_granules_by_fill_writer"][std::to_string(k)] = c_wide_diff_writer[k].load();
        }
        w["differing_granules_where_a_gpu_result_landed"] = {{"in_prefix", c_wide_landed[0].load()}, {"after_prefix", c_wide_landed[1].load()}};
        w["prefix_diffs_probed"] = c_wide_probed.load();
        static constexpr const char* RowNames[11] = {"asked", "outside_ring_or_gpu_written", "no_prefix", "prefix_in_two_halves", "resolved", "bound", "compared",
                                                     "PREFIX_DIFFERS_MUST_BE_0", "shadow_checked", "SHADOW_PREFIX_DIFFERS_MUST_BE_0", "landings_copied_into_hook_ring"};
        j["wide_rows"]["mode"] = WideRowsMode();
        j["wide_rows"]["past_end_note_167"] = {{"mode", WidePastEndMode()}, {"asked", c_past_end[0].load()}, {"resolved", c_past_end[1].load()}, {"refused", c_past_end[2].load()},
                                                {"compared", c_past_end[3].load()}, {"compared_same", c_past_end[4].load()},
                                                {"PAST_END_BLOCK_DIFFERENT_MUST_BE_0", c_past_end[5].load()}, {"served", c_past_end[6].load()}};
        for (u32 k = 0; k < 11; ++k) {
            j["wide_rows"][RowNames[k]] = c_wide_rows[k].load();
        }
        {
            std::scoped_lock tag_lock{g_tag_mutex};
            for (u32 t = 0; t < g_tags.size() && t < MaxTags; ++t) {
                if (const u64 n = c_wide_prefix_tag_granules[t].load()) {
                    const auto& e = g_tags[t];
                    w["prefix_changed_not_landed_by_fill_tag"].push_back(fmt::format("{} granules: how {} caller {:#x} resource {:#x} map type {}", n, e.how, e.caller, e.resource, e.map_type));
                }
            }
        }
        std::scoped_lock lock{g_wide_mutex};
        for (const auto& e : g_wide_examples) {
            w["examples"].push_back(fmt::format("window ring+{:#x} first diff ring+{:#x} diff {} unfilled {} prefix {}", e[0], e[1], e[2], e[3], e[4]));
        }
    }
    j["raster_1b"]["on"] = Raster1bOn();
    j["raster_1b"]["real"] = Raster1bReal();
    j["raster_1b"]["bound_from_hook_ring"] = {{"reads_up_to_64k", c_raster_bound[0].load()}, {"reads_bytes", c_raster_bound_bytes[0].load()},
                                              {"texel", c_raster_bound[1].load()}, {"texel_bytes", c_raster_bound_bytes[1].load()},
                                              {"checked", c_raster_bound_checked.load()}};
    for (u32 k = 0; k < 2; ++k) {
        // note 168: the answers the GPU is given (sampled at the bind); a differing one is refused, so it counts here and is not bound
        const auto& st = g_raster[k];
        j["raster_1b"][cls_names[k]] = {{"compares", st.compares.load()},   {"bytes", st.bytes.load()},
                                        {"equal", st.equal.load()},         {"with_diff_refused", st.diff_compares.load()},
                                        {"diff_bytes_refused", st.diff_bytes.load()}, {"with_unfilled", st.unfilled_compares.load()},
                                        {"BOUND_UNFILLED_BYTES_MUST_BE_0", st.unfilled_bytes.load()}, {"partial_ranges", st.partial.load()}};
        // the old path's uploads compared as a shadow (unfilled granules expected: the hook ring refused those ranges)
        const auto& sh = g_raster[k | 2];
        j["raster_1b"][std::string("old_path_shadow_") + cls_names[k]] = {{"compares", sh.compares.load()}, {"bytes", sh.bytes.load()}, {"equal", sh.equal.load()},
                                                                          {"with_diff", sh.diff_compares.load()}, {"diff_bytes", sh.diff_bytes.load()},
                                                                          {"with_unfilled", sh.unfilled_compares.load()}, {"unfilled_bytes", sh.unfilled_bytes.load()}};
    }
    j["raster_1b"]["bound_answers_refused_by_their_compare"] = c_bound_refused.load();
    j["typed_buffers_note_175"] = Core::HookDiff::TypedReport();
    {
        const auto su = Core::NativeDraw::SkipUserDataCounts();
        j["skip_user_data_note_179"] = {{"mode", Core::NativeDraw::SkipUserDataMode()}, {"shaders_published_covered", su[0]}, {"stages_skipped_at_the_hook", su[1]},
                                        {"shadow_stages_would_skip", su[2]}, {"stages_rebuilt_at_the_front", su[3]}, {"SKIPPED_STAGE_NOT_COVERED_MUST_BE_0", su[4]},
                                        {"covered_id_other_gnm_refused", su[5]}};
    }
    j["raster_1b"]["open_map_guard_note_168"] = {{"mode", OpenMapGuardMode()}, {"small_reads_over_an_open_map", c_open_guard[0].load()},
                                                  {"wide_windows_over_an_open_map", c_open_guard[1].load()}, {"past_end_windows_over_an_open_map", c_open_guard[2].load()}};
    j["raster_1b"]["wide_reads_old_path"] = {{"count", c_raster_wide.load()}, {"bytes", c_raster_wide_bytes.load()}};
    j["raster_1b"]["why_differs"] = {{"gpu_result_landed_since_fill", c_why_landed.load()}, {"inside_an_open_map", c_why_open_map.load()},
                                     {"range_gpu_written", c_why_gpu_written.load()}, {"NONE_OF_THESE", c_why_none.load()}, {"first_byte_differs", c_why_whole.load()}};
    for (u32 w = 0; w < 9; ++w) {
        if (const u64 n = c_why_writer[w].load()) {
            j["raster_1b"]["why_differs"]["last_fill_writer"][w == 8 ? std::string("ours") : std::to_string(w)] = n;
        }
    }
    static constexpr const char* age_names[5] = {"under_100", "under_1k", "under_10k", "under_100k", "more"};
    for (u32 k = 0; k < 5; ++k) {
        j["raster_1b"]["why_differs"]["records_since_fill"][age_names[k]] = c_why_age[k].load();
    }
    j["raster_1b"]["landings"] = {{"count", c_landings.load()}, {"bytes", c_landing_bytes.load()}};
    {
        static constexpr const char* how_names[4] = {"unmap", "replaced_by_second_map", "after_hooked_draw_call", "frame_start_never_unmapped"};
        for (u32 k = 0; k < 4; ++k) {
            j["raster_1b"]["map_window_fills_by_how"][how_names[k]] = {{"fills", c_how_fills[k].load()}, {"DIFFS", c_how_diffs[k].load()}};
        }
        std::scoped_lock lock{g_tag_mutex};
        auto& list = j["raster_1b"]["diffs_by_map"] = nlohmann::json::array();
        for (u32 i = 0; i < MaxTags; ++i) {
            const u64 d = c_tag_diffs[i].load();
            if (d == 0) {
                continue;
            }
            const FillTag t = i < g_tags.size() ? g_tags[i] : FillTag{};
            list.push_back({{"how", i == 0 ? std::string("untagged") : std::string(how_names[t.how & 3])}, {"caller", fmt::format("{:#x}", t.caller)},
                            {"resource_id", fmt::format("{:#x}", t.resource)}, {"map_type", t.map_type}, {"diffs", d}, {"diff_bytes", c_tag_diff_bytes[i].load()},
                            {"fills_of_this_map", c_tag_fills[i].load()}});
        }
        j["raster_1b"]["map_tags"] = g_tags.size() - 1;
        // the Maps announced only at the next frame start (their Unmap was not seen): which call sites
        auto& never = j["raster_1b"]["never_unmapped_maps"] = nlohmann::json::array();
        for (u32 i = 1; i < g_tags.size(); ++i) {
            const FillTag& t = g_tags[i];
            if ((t.how & 3) == 3 && c_tag_fills[i].load() != 0) {
                never.push_back({{"caller", fmt::format("{:#x}", t.caller)}, {"resource_id", fmt::format("{:#x}", t.resource)}, {"map_type", t.map_type}, {"fills", c_tag_fills[i].load()}});
            }
        }
    }
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
