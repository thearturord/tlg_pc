// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_ring_mirror.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_draw.h"

namespace Core::RingMirror {

namespace {

std::mutex mutex;
std::vector<u8> shadow;
VAddr shadow_start = 0, shadow_end = 0;
std::atomic<u64> fills{0}, windows{0}, window_bytes{0}, announced_taken{0}, unannounced_bytes{0}, unannounced_windows{0}, unannounced_blocks{0}, mismatch_events_logged{0};
std::atomic<u64> largest_window{0};

constexpr u64 Block = 64;
/// blocks found different and not announced since (block index in the ring -> milliseconds since the start when first found)
std::unordered_map<u32, u32> suspects;
std::atomic<u64> suspects_cancelled{0}, suspects_found{0}, never_announced_total{0};
std::vector<u64> never_announced_samples; // ring offsets of the first blocks retired without an announcement
u32 NowMs() {
    static const auto t0 = std::chrono::steady_clock::now();
    return static_cast<u32>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
}

/// The writer probe (mode 3). Lock free on the fault side: a table of the pages made read only and a table of what wrote into them.
constexpr size_t ProbePages = 64, ProbeWriters = 64;
std::array<std::atomic<u64>, ProbePages> probe_pages{};
std::array<std::atomic<u64>, ProbePages> probe_targets{}; // the block that stayed unannounced, per protected page
struct ProbeWriter {
    std::atomic<u64> rip{0}, count{0}, exact{0}, first_address{0}, thread{0};
    std::array<std::atomic<u64>, 48> stack{};
};
std::array<ProbeWriter, ProbeWriters> probe_writers{};
std::atomic<u64> probe_protected{0}, probe_faults{0}, probe_other_faults{0};
std::atomic<bool> probe_installed{false};

#ifdef _WIN32
LONG CALLBACK ProbeHandler(PEXCEPTION_POINTERS info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || info->ExceptionRecord->NumberParameters < 2 || info->ExceptionRecord->ExceptionInformation[0] != 1) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const u64 address = info->ExceptionRecord->ExceptionInformation[1];
    const u64 page = address & ~u64{4095};
    for (size_t slot_index = 0; slot_index < probe_pages.size(); ++slot_index) {
        auto& slot = probe_pages[slot_index];
        u64 expected = page;
        if (slot.load(std::memory_order_acquire) == page && slot.compare_exchange_strong(expected, 0)) {
            const bool exact = (address & ~u64{63}) == probe_targets[slot_index].load(std::memory_order_relaxed);
            DWORD old;
            VirtualProtect(reinterpret_cast<void*>(page), 4096, PAGE_READWRITE, &old);
            probe_faults.fetch_add(1, std::memory_order_relaxed);
            const u64 rip = info->ContextRecord->Rip;
            const u64* stack = reinterpret_cast<const u64*>(info->ContextRecord->Rsp);
            for (auto& w : probe_writers) {
                u64 seen = w.rip.load(std::memory_order_acquire);
                if (seen == 0 && w.rip.compare_exchange_strong(seen, rip)) {
                    seen = rip;
                    w.first_address.store(address, std::memory_order_relaxed);
                    for (size_t i = 0; i < w.stack.size(); ++i) {
                        w.stack[i].store(stack[i], std::memory_order_relaxed);
                    }
                    w.thread.store(GetCurrentThreadId(), std::memory_order_relaxed);
                }
                if (seen == rip) {
                    w.count.fetch_add(1, std::memory_order_relaxed);
                    w.exact.fetch_add(exact ? 1 : 0, std::memory_order_relaxed);
                    break;
                }
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

/// Makes the page of a block that stayed different without an announcement read only (once); the handler above records its writer.
void ProbeProtect(VAddr address) {
#ifdef _WIN32
    if (!probe_installed.exchange(true)) {
        AddVectoredExceptionHandler(1, ProbeHandler);
    }
    const u64 page = address & ~u64{4095};
    for (const auto& slot : probe_pages) {
        if (slot.load(std::memory_order_acquire) == page) {
            return;
        }
    }
    for (size_t slot_index = 0; slot_index < probe_pages.size(); ++slot_index) {
        auto& slot = probe_pages[slot_index];
        u64 expected = 0;
        if (slot.load(std::memory_order_acquire) != 0) {
            continue;
        }
        probe_targets[slot_index].store(address & ~u64{63}, std::memory_order_relaxed);
        if (slot.compare_exchange_strong(expected, page)) {
            DWORD old;
            if (VirtualProtect(reinterpret_cast<void*>(page), 4096, PAGE_READONLY, &old)) {
                probe_protected.fetch_add(1, std::memory_order_relaxed);
            } else {
                slot.store(0);
            }
            return;
        }
    }
#else
    (void)address;
#endif
}

} // namespace

u32 Mode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_RING_MIRROR");
        return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u;
    }();
    return mode;
}

bool NeedFill(VAddr start, VAddr end) {
    std::scoped_lock lock{mutex};
    return shadow_start != start || shadow_end != end;
}

void Fill(VAddr start, VAddr end) {
    std::scoped_lock lock{mutex};
    shadow.resize(end - start);
    std::memcpy(shadow.data(), reinterpret_cast<const void*>(start), end - start);
    shadow_start = start;
    shadow_end = end;
    suspects.clear();
    fills.fetch_add(1, std::memory_order_relaxed);
}

void CheckWindow(VAddr lo, VAddr hi, const std::pair<u64, u64>* announced, size_t count) {
    std::scoped_lock lock{mutex};
    if (shadow_start == 0 || lo < shadow_start || hi > shadow_end || lo >= hi) {
        return;
    }
    windows.fetch_add(1, std::memory_order_relaxed);
    window_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
    largest_window.store(std::max<u64>(largest_window.load(std::memory_order_relaxed), hi - lo), std::memory_order_relaxed);
    {
        // a block that stayed different for 5 s without an announcement covering it is retired and counted (the rate is the figure)
        static u32 last_sweep_ms = 0;
        const u32 now = NowMs();
        if (now - last_sweep_ms > 1000) {
            last_sweep_ms = now;
            u64 retiring = 0;
            for (const auto& [block, found] : suspects) {
                retiring += now - found > 5000;
            }
            const bool steady = retiring <= 400; // (the burst while a level loads is not probed: a protected page that the loader writes through a system call would fail it)
            for (auto it = suspects.begin(); it != suspects.end();) {
                if (now - it->second > 5000) {
                    never_announced_total.fetch_add(1, std::memory_order_relaxed);
                    if (Mode() == 3 && steady && probe_protected.load(std::memory_order_relaxed) < 400) {
                        ProbeProtect(shadow_start + u64{it->first} * Block);
                    }
                    if (never_announced_samples.size() < 4000 && now > 40000) {
                        never_announced_samples.push_back(u64{it->first} * Block);
                    }
                    it = suspects.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    // the mirror takes the announced bytes of the window...
    for (size_t i = 0; i < count; ++i) {
        const VAddr a = std::max<VAddr>(announced[i].first, lo);
        const VAddr e = std::min<VAddr>(announced[i].first + announced[i].second, hi);
        if (a < e) {
            std::memcpy(shadow.data() + (a - shadow_start), reinterpret_cast<const void*>(a), e - a);
            announced_taken.fetch_add(e - a, std::memory_order_relaxed);
        }
    }
    // ...and what the window holds now must be what the mirror holds: a difference is a write nobody announced
    const u8* const guest = reinterpret_cast<const u8*>(lo);
    u8* const mirror = shadow.data() + (lo - shadow_start);
    const u64 size = hi - lo;
    u64 bad_blocks = 0, bad_bytes = 0, first_bad = 0;
    for (u64 at = 0; at < size; at += Block) {
        const u64 n = std::min<u64>(Block, size - at);
        if (std::memcmp(guest + at, mirror + at, n) != 0) {
            if (bad_blocks == 0) {
                first_bad = at;
            }
            ++bad_blocks;
            if (suspects.emplace(static_cast<u32>((lo + at - shadow_start) / Block), NowMs()).second) {
                suspects_found.fetch_add(1, std::memory_order_relaxed);
            }
            for (u64 i = 0; i < n; ++i) {
                bad_bytes += guest[at + i] != mirror[at + i];
            }
            std::memcpy(mirror + at, guest + at, n); // (resynchronized: one missed write is counted once)
        }
    }
    if (bad_blocks != 0) {
        unannounced_windows.fetch_add(1, std::memory_order_relaxed);
        unannounced_blocks.fetch_add(bad_blocks, std::memory_order_relaxed);
        unannounced_bytes.fetch_add(bad_bytes, std::memory_order_relaxed);
        if (mismatch_events_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
            Core::RareEvents::Note("ring_mirror_unannounced", fmt::format("window {:#x}+{:#x}: {} blocks, {} bytes differ, first at {:#x} (ring offset {:#x})", lo, size, bad_blocks, bad_bytes,
                                                                          lo + first_bad, lo + first_bad - shadow_start));
        }
    }
}

void NoteAnnounced(VAddr address, u64 size) {
    std::scoped_lock lock{mutex};
    if (suspects.empty() || shadow_start == 0 || address + size <= shadow_start || address >= shadow_end) {
        return;
    }
    const VAddr a = std::max<VAddr>(address, shadow_start), e = std::min<VAddr>(address + size, shadow_end);
    for (u64 block = (a - shadow_start) / Block; block <= (e - 1 - shadow_start) / Block; ++block) {
        if (suspects.erase(static_cast<u32>(block)) != 0) {
            suspects_cancelled.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

nlohmann::json Report() {
    u64 open = 0, old = 0;
    nlohmann::json runs = nlohmann::json::array();
    {
        std::scoped_lock lock{mutex};
        const u32 now = NowMs();
        open = suspects.size();
        for (const auto& [block, found] : suspects) {
            old += now - found > 3000;
        }
        // the old unexplained blocks as runs of adjacent blocks: (ring offset, bytes), the 16 longest
        std::vector<u32> blocks;
        for (const auto& [block, found] : suspects) {
            if (now - found > 3000) {
                blocks.push_back(block);
            }
        }
        std::sort(blocks.begin(), blocks.end());
        std::vector<std::pair<u64, u64>> list; // (length in blocks, first block)
        for (size_t i = 0; i < blocks.size();) {
            size_t j = i;
            while (j + 1 < blocks.size() && blocks[j + 1] == blocks[j] + 1) {
                ++j;
            }
            list.emplace_back(j - i + 1, blocks[i]);
            i = j + 1;
        }
        std::sort(list.begin(), list.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; i < list.size() && i < 16; ++i) {
            runs.push_back({{"offset", fmt::format("{:#x}", list[i].second * Block)}, {"bytes", list[i].first * Block}});
        }
        runs.push_back({{"runs_total", list.size()}});
    }
    return {{"mode", Mode()},
            {"fills", fills.load()},
            {"windows", windows.load()},
            {"window_bytes", window_bytes.load()},
            {"largest_window", largest_window.load()},
            {"announced_bytes_taken", announced_taken.load()},
            {"raw_differing_bytes", unannounced_bytes.load()},
            {"suspect_blocks_found", suspects_found.load()},
            {"suspect_blocks_announced_later", suspects_cancelled.load()},
            {"suspect_blocks_open", open},
            {"NEVER_ANNOUNCED_BLOCKS_OLDER_3S_MUST_BE_0", old},
            {"NEVER_ANNOUNCED_BLOCKS_RETIRED", never_announced_total.load()},
            {"never_announced_offsets", [] {
                 std::scoped_lock lock{mutex};
                 nlohmann::json j = nlohmann::json::array();
                 for (const u64 v : never_announced_samples) {
                     j.push_back(fmt::format("{:#x}", v));
                 }
                 return j;
             }()},
            {"never_announced_runs", runs},
            {"writer_probe", [] {
                 nlohmann::json j = {{"pages_protected", probe_protected.load()}, {"faults", probe_faults.load()}, {"writers", nlohmann::json::array()}};
                 for (const auto& w : probe_writers) {
                     if (w.rip.load() != 0) {
                         const u64 base = NativeDraw::ImageBase();
                         j["writers"].push_back({{"rip", fmt::format("{:#x}", w.rip.load() - base + 0x1000000)}, {"count", w.count.load()}, {"exact_block", w.exact.load()}, {"first_address", fmt::format("{:#x}", w.first_address.load())},
                                                 {"stack_in_game_image", [&] {
                                                     std::string s;
                                                     for (const auto& word : w.stack) {
                                                         const u64 v = word.load();
                                                         if (v >= base && v - base < 0x4000000) {
                                                             s += fmt::format("{:#x} ", v - base + 0x1000000);
                                                         }
                                                     }
                                                     return s;
                                                 }()},
                                                 {"thread", w.thread.load()}});
                     }
                 }
                 return j;
             }()},
            {"unannounced_windows", unannounced_windows.load()},
            {"unannounced_blocks", unannounced_blocks.load()}};
}

} // namespace Core::RingMirror
