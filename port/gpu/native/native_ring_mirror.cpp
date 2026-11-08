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
#include <bit>
#include <functional>
#include <memory>
#include <string>
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
std::vector<u64> never_announced_samples;
std::vector<u8> writer_of_block;                 // tag + 1 of the newest announcement over each 64 byte block of the ring (0: none yet)
std::unordered_map<u32, u64> retired_classes;    // (own tag + 1) | (previous block's) << 8 | (next block's) << 16 -> blocks
std::vector<std::string> retired_dumps;
std::vector<std::pair<u64, u64>> host_pending; // guest memory the port wrote itself (command processor labels), taken into the shadow at the next window that holds them
std::atomic<u64> host_write_calls{0}, host_write_bytes_taken{0};
std::vector<u32> ann_start_of_block, ann_end_of_block; // byte offsets in the ring of the newest announcement over each block
std::unordered_map<std::string, u64> retired_relations; // ring offsets of the first blocks retired without an announcement
u32 NowMs() {
    static const auto t0 = std::chrono::steady_clock::now();
    return static_cast<u32>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
}


/// Mode 4, the sampled write trace: one page in eight of the ring is read only (the eighth rotates every 4 s); the first write to such a page is logged (address, instruction, the
/// game code addresses on the stack) and the page given back. 300 ms later the log entry is judged: was the block announced around that time (`ann_time_of_block`)? Writes that were not
/// are grouped by instruction and caller.
struct TraceFault {
    u64 address, rip;
    u32 ms, thread;
    u64 frames[3];
};
constexpr size_t FaultCap = 1u << 16;
std::vector<TraceFault> trace_faults_log;
std::atomic<u64> trace_head{0};
u64 trace_eval = 0;
std::unique_ptr<std::atomic<u8>[]> trace_prot;
size_t trace_pages = 0;
std::atomic<VAddr> trace_start{0};
std::vector<u32> ann_time_of_block;
std::atomic<u64> trace_faults{0}, trace_covered{0}, trace_ambiguous{0}, trace_uncovered{0}, trace_lost{0};
std::unordered_map<std::string, std::pair<u64, std::vector<u64>>> trace_uncovered_by_site;
u32 trace_phase = 0, trace_rotated_ms = 0, trace_evaluated_ms = 0;
u32 steady_sweeps = 0; // consecutive sweeps with few retiring blocks (the level is not loading any more)

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
    {
        const VAddr ts = trace_start.load(std::memory_order_acquire);
        if (ts != 0 && address >= ts && address < ts + trace_pages * 4096) {
            const size_t idx = (address - ts) >> 12;
            if (trace_prot[idx].exchange(0) == 1) {
                DWORD old;
                VirtualProtect(reinterpret_cast<void*>(address & ~u64{4095}), 4096, PAGE_READWRITE, &old);
                const u64 slot = trace_head.fetch_add(1, std::memory_order_relaxed);
                TraceFault& f = trace_faults_log[slot % FaultCap];
                f.address = address;
                f.rip = info->ContextRecord->Rip;
                f.ms = NowMs();
                f.thread = GetCurrentThreadId();
                f.frames[0] = f.frames[1] = f.frames[2] = 0;
                const u64 base = NativeDraw::ImageBase();
                const u64* stack = reinterpret_cast<const u64*>(info->ContextRecord->Rsp);
                size_t n = 0;
                for (size_t i = 0; i < 48 && n < 3; ++i) {
                    if (stack[i] >= base && stack[i] - base < 0x4000000) {
                        f.frames[n++] = stack[i] - base + 0x1000000;
                    }
                }
                trace_faults.fetch_add(1, std::memory_order_relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
    }
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


/// Mode 4, called with the mutex held on the raster thread: rotates the sampled pages and judges the faults that are 300 ms old.
void TraceStep(u32 now) {
#ifdef _WIN32
    if (shadow_start == 0) {
        return;
    }
    if (trace_pages == 0 && steady_sweeps < 3) {
        return; // (protecting the ring while a level loads fails the load)
    }
    if (trace_pages == 0) {
        trace_pages = (shadow_end - shadow_start) >> 12;
        trace_prot = std::make_unique<std::atomic<u8>[]>(trace_pages);
        trace_faults_log.resize(FaultCap);
        ann_time_of_block.assign((shadow_end - shadow_start) / Block + 1, 0);
        if (!probe_installed.exchange(true)) {
            AddVectoredExceptionHandler(1, ProbeHandler);
        }
        trace_start.store(shadow_start, std::memory_order_release);
        trace_rotated_ms = now - 100000; // rotate at once
    }
    if (now - trace_rotated_ms > 4000) {
        trace_rotated_ms = now;
        for (size_t i = 0; i < trace_pages; ++i) {
            if (trace_prot[i].exchange(0) == 1) {
                DWORD old;
                VirtualProtect(reinterpret_cast<void*>(shadow_start + (i << 12)), 4096, PAGE_READWRITE, &old);
            }
        }
        trace_phase = (trace_phase + 1) & 7;
        for (size_t i = trace_phase; i < trace_pages; i += 8) {
            trace_prot[i].store(1);
            DWORD old;
            if (!VirtualProtect(reinterpret_cast<void*>(shadow_start + (i << 12)), 4096, PAGE_READONLY, &old)) {
                trace_prot[i].store(0);
            }
        }
    }
    if (now - trace_evaluated_ms > 200) {
        trace_evaluated_ms = now;
        const u64 head = trace_head.load(std::memory_order_acquire);
        if (head - trace_eval > FaultCap) {
            trace_lost.fetch_add(head - trace_eval - FaultCap, std::memory_order_relaxed);
            trace_eval = head - FaultCap;
        }
        const u64 image_base = NativeDraw::ImageBase();
        while (trace_eval < head) {
            const TraceFault& f = trace_faults_log[trace_eval % FaultCap];
            if (now - f.ms < 300) {
                break;
            }
            ++trace_eval;
            if (f.address < shadow_start || f.address >= shadow_end) {
                continue;
            }
            const u64 block = (f.address - shadow_start) / Block;
            const u32 t = block < ann_time_of_block.size() ? ann_time_of_block[block] : 0;
            if (t != 0 && t + 300 >= f.ms && t <= f.ms + 300) {
                trace_covered.fetch_add(1, std::memory_order_relaxed);
            } else if (t > f.ms + 300) {
                trace_ambiguous.fetch_add(1, std::memory_order_relaxed); // the block was announced again later (next lap): cannot tell
            } else {
                trace_uncovered.fetch_add(1, std::memory_order_relaxed);
                const bool game = f.rip >= image_base && f.rip - image_base < 0x4000000;
                const std::string key = fmt::format("rip {:#x}{} frames {:#x} {:#x} {:#x}", game ? f.rip - image_base + 0x1000000 : f.rip, game ? " (game)" : " (host)", f.frames[0], f.frames[1], f.frames[2]);
                auto& entry = trace_uncovered_by_site[key];
                ++entry.first;
                if (entry.second.size() < 3) {
                    entry.second.push_back(f.address - shadow_start);
                }
            }
        }
    }
#else
    (void)now;
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

bool AuditOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_MIRROR_AUDIT");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

bool ShadowOn() noexcept {
    return Mode() >= 2 || (Mode() == 1 && AuditOn());
}

namespace {
std::atomic<u64> real_pieces_in{0}, real_pieces_out{0}, real_merged_bytes{0};
std::atomic<u64> real_windows{0}, real_window_bytes{0}, real_uploaded_whole{0}, real_announced{0}, real_repaired_bytes{0}, real_repairs{0};
} // namespace

u64 MergeGap() noexcept {
    static const u64 gap = [] {
        const char* e = std::getenv("TLG_RING_MERGE");
        return e != nullptr ? static_cast<u64>(std::atoll(e)) : u64{2048};
    }();
    return gap;
}

void NoteReal(u64 window, u64 uploaded_whole, u64 announced, u64 pieces_in, u64 pieces_out, u64 merged_bytes) {
    real_pieces_in.fetch_add(pieces_in, std::memory_order_relaxed);
    real_pieces_out.fetch_add(pieces_out, std::memory_order_relaxed);
    real_merged_bytes.fetch_add(merged_bytes, std::memory_order_relaxed);
    real_windows.fetch_add(1, std::memory_order_relaxed);
    real_window_bytes.fetch_add(window, std::memory_order_relaxed);
    real_uploaded_whole.fetch_add(uploaded_whole, std::memory_order_relaxed);
    real_announced.fetch_add(announced, std::memory_order_relaxed);
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
    writer_of_block.assign((end - start) / Block + 1, 0);
    ann_start_of_block.assign((end - start) / Block + 1, 0);
    ann_end_of_block.assign((end - start) / Block + 1, 0);
    fills.fetch_add(1, std::memory_order_relaxed);
}

void CheckWindow(VAddr lo, VAddr hi, const std::pair<u64, u64>* announced, size_t count, std::vector<std::pair<u64, u64>>* repair) {
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
            const bool steady = retiring <= 400;
            steady_sweeps = steady && now > 20000 ? steady_sweeps + 1 : 0; // (the burst while a level loads is not probed: a protected page that the loader writes through a system call would fail it)
            for (auto it = suspects.begin(); it != suspects.end();) {
                if (now - it->second > 5000) {
                    never_announced_total.fetch_add(1, std::memory_order_relaxed);
                    if (now > 40000) {
                        const u32 b = it->first;
                        const u32 own = b < writer_of_block.size() ? writer_of_block[b] : 0, prev = b > 0 && b - 1 < writer_of_block.size() ? writer_of_block[b - 1] : 0, next = b + 1 < writer_of_block.size() ? writer_of_block[b + 1] : 0;
                        ++retired_classes[own | (prev << 8) | (next << 16)];
                        {
                            const u64 at = u64{b} * Block;
                            const bool has_prev = b > 0 && b - 1 < ann_end_of_block.size() && ann_end_of_block[b - 1] != 0;
                            const bool has_next = b + 1 < ann_start_of_block.size() && ann_end_of_block[b + 1] != 0;
                            const bool tail = has_prev && ann_end_of_block[b - 1] == at;
                            const bool head = has_next && ann_start_of_block[b + 1] == at + Block;
                            const u64 prev_len = has_prev ? ann_end_of_block[b - 1] - ann_start_of_block[b - 1] : 0;
                            const u64 own_len = ann_end_of_block[b] != 0 ? ann_end_of_block[b] - ann_start_of_block[b] : 0;
                            const std::string key = fmt::format("{}{} prev tag {} len<2^{} own tag {} len<2^{}", tail ? "TAIL " : "", head ? "HEAD " : "", has_prev ? int(prev) - 1 : -1, std::bit_width(prev_len), own ? int(own) - 1 : -1, std::bit_width(own_len));
                            ++retired_relations[key];
                        }
                        if (retired_dumps.size() < 12) {
                            const u8* g = reinterpret_cast<const u8*>(shadow_start + u64{b} * Block);
                            std::string hex;
                            bool zero = true;
                            for (u32 i = 0; i < Block; ++i) {
                                hex += fmt::format("{:02x}", g[i]);
                                zero &= g[i] == 0;
                            }
                            retired_dumps.push_back(fmt::format("ring {:#x} own {} prev {} next {} zero {}: {}", u64{b} * Block, own, prev, next, zero, hex));
                        }
                    }
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
    if (Mode() == 4) {
        TraceStep(NowMs());
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
    // the bytes the port wrote itself (labels): taken as they are
    for (size_t i = 0; i < host_pending.size();) {
        const auto [ha, hn] = host_pending[i];
        const VAddr a = std::max<VAddr>(ha, lo), e = std::min<VAddr>(ha + hn, hi);
        if (a < e) {
            std::memcpy(shadow.data() + (a - shadow_start), reinterpret_cast<const void*>(a), e - a);
            host_write_bytes_taken.fetch_add(e - a, std::memory_order_relaxed);
            if (a == ha && e == ha + hn) {
                host_pending[i] = host_pending.back();
                host_pending.pop_back();
                continue;
            }
        }
        ++i;
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
            if (repair != nullptr) {
                const u64 abs = lo + at;
                if (!repair->empty() && repair->back().first + repair->back().second == abs) {
                    repair->back().second += n;
                } else {
                    repair->emplace_back(abs, n);
                }
                real_repaired_bytes.fetch_add(n, std::memory_order_relaxed);
            }
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

void NoteWriter(VAddr address, u64 size, u32 tag) {
    std::scoped_lock lock{mutex};
    if (shadow_start == 0 || address + size <= shadow_start || address >= shadow_end) {
        return;
    }
    const VAddr a = std::max<VAddr>(address, shadow_start), e = std::min<VAddr>(address + size, shadow_end);
    for (u64 block = (a - shadow_start) / Block; block <= (e - 1 - shadow_start) / Block && block < writer_of_block.size(); ++block) {
        writer_of_block[block] = static_cast<u8>(tag + 1);
        if (block < ann_time_of_block.size()) {
            ann_time_of_block[block] = std::max<u32>(NowMs(), 1);
        }
        ann_start_of_block[block] = static_cast<u32>(a - shadow_start);
        ann_end_of_block[block] = static_cast<u32>(e - shadow_start);
    }
}

void NoteHostWrite(VAddr address, u64 size) {
    std::scoped_lock lock{mutex};
    if (shadow_start == 0 || address + size <= shadow_start || address >= shadow_end) {
        return;
    }
    host_write_calls.fetch_add(1, std::memory_order_relaxed);
    if (host_pending.size() > 200000) {
        host_pending.erase(host_pending.begin(), host_pending.begin() + 100000);
    }
    host_pending.emplace_back(address, size);
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
            {"real", {{"windows", real_windows.load()}, {"window_bytes", real_window_bytes.load()}, {"uploaded_whole_first_read", real_uploaded_whole.load()}, {"uploaded_from_announcements", real_announced.load()}, {"announced_pieces", real_pieces_in.load()}, {"copies_after_merge", real_pieces_out.load()}, {"gap_bytes_added_by_merge", real_merged_bytes.load()},
                      {"AUDIT_REPAIRED_BYTES_MUST_BE_0", real_repaired_bytes.load()}, {"audit", AuditOn()}}},
            {"raw_differing_bytes", unannounced_bytes.load()},
            {"suspect_blocks_found", suspects_found.load()},
            {"suspect_blocks_announced_later", suspects_cancelled.load()},
            {"suspect_blocks_open", open},
            {"NEVER_ANNOUNCED_BLOCKS_OLDER_3S_MUST_BE_0", old},
            {"NEVER_ANNOUNCED_BLOCKS_RETIRED", never_announced_total.load()},
            {"retired_classes", [] {
                 std::scoped_lock lock{mutex};
                 std::vector<std::pair<u64, u32>> v;
                 for (const auto& [k, n] : retired_classes) {
                     v.emplace_back(n, k);
                 }
                 std::sort(v.begin(), v.end(), std::greater<>());
                 nlohmann::json j = nlohmann::json::array();
                 for (size_t i = 0; i < v.size() && i < 14; ++i) {
                     const u32 k = v[i].second;
                     j.push_back(fmt::format("{} blocks: own tag {} prev {} next {} (tag+1, 0 none)", v[i].first, (k & 255), (k >> 8) & 255, (k >> 16) & 255));
                 }
                 return j;
             }()},
            {"retired_relations", [] {
                 std::scoped_lock lock{mutex};
                 std::vector<std::pair<u64, std::string>> v;
                 for (const auto& [k, n] : retired_relations) {
                     v.emplace_back(n, k);
                 }
                 std::sort(v.begin(), v.end(), std::greater<>());
                 nlohmann::json j = nlohmann::json::array();
                 for (size_t i = 0; i < v.size() && i < 16; ++i) {
                     j.push_back(fmt::format("{} blocks: {}", v[i].first, v[i].second));
                 }
                 return j;
             }()},
            {"retired_dumps", [] {
                 std::scoped_lock lock{mutex};
                 return nlohmann::json(retired_dumps);
             }()},
            {"never_announced_offsets", [] {
                 std::scoped_lock lock{mutex};
                 nlohmann::json j = nlohmann::json::array();
                 for (const u64 v : never_announced_samples) {
                     j.push_back(fmt::format("{:#x}", v));
                 }
                 return j;
             }()},
            {"never_announced_runs", runs},
            {"host_writes", {{"calls", host_write_calls.load()}, {"bytes_taken_into_shadow", host_write_bytes_taken.load()}}},
            {"write_trace", [] {
                 std::scoped_lock lock{mutex};
                 nlohmann::json j = {{"faults", trace_faults.load()}, {"announced_around_the_write", trace_covered.load()}, {"announced_again_later_cannot_tell", trace_ambiguous.load()}, {"NOT_ANNOUNCED", trace_uncovered.load()}, {"log_overflow_lost", trace_lost.load()}, {"sites", nlohmann::json::array()}};
                 std::vector<std::pair<u64, std::string>> v;
                 for (const auto& [k, e] : trace_uncovered_by_site) {
                     v.emplace_back(e.first, k + fmt::format(" ring offsets {:#x} {:#x} {:#x}", e.second.size() > 0 ? e.second[0] : 0, e.second.size() > 1 ? e.second[1] : 0, e.second.size() > 2 ? e.second[2] : 0));
                 }
                 std::sort(v.begin(), v.end(), std::greater<>());
                 for (size_t i = 0; i < v.size() && i < 20; ++i) {
                     j["sites"].push_back(fmt::format("{} writes: {}", v[i].first, v[i].second));
                 }
                 return j;
             }()},
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
