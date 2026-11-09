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
std::atomic<u8*> g_bytes{nullptr};          // the host copy of the ring
std::atomic<std::atomic<u8>*> g_filled{nullptr}; // per granule: 1 once a finished write was copied since the block was handed out
std::atomic<u64> g_base{0}, g_size{0};

std::atomic<u64> c_fills{0}, c_fill_bytes{0}, c_allocs{0};
std::atomic<u64> c_compares{0}, c_compare_bytes{0}, c_equal{0}, c_diff_compares{0}, c_diff_bytes{0}, c_unfilled_compares{0}, c_unfilled_bytes{0};

struct Example {
    u64 offset, size, first_diff, diff_bytes, unfilled;
};
std::mutex g_example_mutex;
std::vector<Example> g_examples;

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
    auto* filled = new std::atomic<u8>[size >> GranuleShift]; // kept for the life of the process
    for (u64 i = 0; i < (size >> GranuleShift); ++i) {
        filled[i].store(0, std::memory_order_relaxed);
    }
    auto* bytes = new u8[size]();
    g_base.store(base, std::memory_order_relaxed);
    g_size.store(size, std::memory_order_relaxed);
    g_filled.store(filled, std::memory_order_release);
    g_bytes.store(bytes, std::memory_order_release);
    return true;
}

bool Clip(u64 address, u64 size, u64& lo, u64& hi) {
    const u64 base = g_base.load(std::memory_order_relaxed), end = base + g_size.load(std::memory_order_relaxed);
    lo = std::max(address, base);
    hi = std::min(address + size, end);
    return lo < hi;
}

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_HOOK_RING");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteAlloc(u64 block, u64 size) {
    if (!On() || block == 0 || size == 0 || !Ensure()) {
        return;
    }
    u64 lo, hi;
    if (!Clip(block, size, lo, hi)) {
        return;
    }
    c_allocs.fetch_add(1, std::memory_order_relaxed);
    auto* filled = g_filled.load(std::memory_order_acquire);
    const u64 base = g_base.load(std::memory_order_relaxed);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        filled[g].store(0, std::memory_order_relaxed);
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
    std::memcpy(bytes + (lo - base), reinterpret_cast<const void*>(lo), hi - lo); // the ring is plain host memory of this process
    auto* filled = g_filled.load(std::memory_order_acquire);
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        filled[g].store(1, std::memory_order_relaxed);
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
    const u8* mine = bytes + (lo - base);
    const u8* theirs = static_cast<const u8*>(copied);
    u64 diff = 0, unfilled = 0, first_diff = ~0ull;
    for (u64 g = (lo - base) >> GranuleShift; g <= (hi - 1 - base) >> GranuleShift; ++g) {
        const u64 g_lo = std::max(lo, base + (g << GranuleShift)) - lo, g_hi = std::min(hi, base + ((g + 1) << GranuleShift)) - lo;
        if (filled[g].load(std::memory_order_relaxed) == 0) {
            unfilled += g_hi - g_lo;
            continue;
        }
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

nlohmann::json Report() {
    nlohmann::json j;
    j["on"] = On();
    j["note"] = "TLG_HOOK_RING (F1 class 1a shadow): a host ring filled at the writes, compared with every front constant copy; the old path still decides";
    j["ring"] = {{"base", fmt::format("{:#x}", g_base.load())}, {"size", g_size.load()}};
    j["fills"] = {{"count", c_fills.load()}, {"bytes", c_fill_bytes.load()}, {"allocs_cleared", c_allocs.load()}};
    j["compares"] = {{"count", c_compares.load()},
                     {"bytes", c_compare_bytes.load()},
                     {"equal", c_equal.load()},
                     {"with_diff", c_diff_compares.load()},
                     {"DIFF_BYTES_MUST_BE_0", c_diff_bytes.load()},
                     {"with_unfilled", c_unfilled_compares.load()},
                     {"UNFILLED_BYTES_MUST_BE_0", c_unfilled_bytes.load()}};
    std::scoped_lock lock{g_example_mutex};
    auto& list = j["examples"] = nlohmann::json::array();
    for (const auto& e : g_examples) {
        list.push_back({{"ring_offset", fmt::format("{:#x}", e.offset)}, {"size", e.size}, {"first_diff", e.first_diff == ~0ull ? -1 : static_cast<s64>(e.first_diff)},
                        {"diff_bytes", e.diff_bytes}, {"unfilled", e.unfilled}});
    }
    return j;
}

} // namespace Core::HookRing
