// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <mutex>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef TLG_PORT
#include "tlg_toggles.h" // sigjmp_buf, tlg_setjmp, runtime_fault_recover
#endif
#include "native/native_content.h"
#include "native/native_device.h"
#include "native/native_registry.h"
#include "native/native_updates.h"

namespace Core::NativeContent {

namespace {

constexpr u32 WriterCopy = 100;           ///< entries made by `NoteCopy`
constexpr u64 MaxBytesWriter6 = 64ull << 20;
u64 MaxBytesOther() {
    static const u64 limit = [] {
        const char* e = std::getenv("TLG_F2_STORE_MAXKB"); // announcements of writers other than 6 over this size are not kept (default 256 KB)
        const u64 kb = e != nullptr ? std::strtoull(e, nullptr, 10) : 0;
        return (kb != 0 ? kb : 256ull) << 10;
    }();
    return limit;
}

struct Entry {
    u64 size{};
    u64 seq{};
    u64 stamp{}; ///< the record number the write was announced at (the number the next record got): a draw of record S may see the entries stamped at most S
    u32 writer{};
    std::shared_ptr<const u8[]> block; ///< the bytes live in a block that a copy's entries share with the entry they were copied from
    u64 offset{};                      ///< where this entry's first byte is in the block
    [[nodiscard]] const u8* Bytes() const {
        return block.get() + offset;
    }
};

std::mutex g_mutex;
std::multimap<u64, Entry> g_entries; ///< by begin address
std::deque<std::pair<u64, u64>> g_fifo; ///< (begin, seq) in insertion order, for the cap
u64 g_stored_bytes = 0;
u64 g_max_size = 0;
std::atomic<u64> g_seq_atomic{0};
u64 g_mark_serial = 0;
std::unordered_set<u64> g_uploaded_images;

/// Writers 1, 2 and 4 announce before they write, so their bytes are not kept, but the range and the order are: a mark says "something wrote here after the entries older than `seq`".
struct Mark {
    u64 size;
    u64 seq;
    u32 writer;
};
std::mutex g_marks_mutex;
std::multimap<u64, Mark> g_marks;
std::deque<std::pair<u64, u64>> g_marks_fifo;
u64 g_marks_max_size = 0;
constexpr size_t MaxMarks = 4u << 20;

std::atomic<u64> c_announced{0}, c_announced_bytes{0}, c_skip_writer{0}, c_skip_big{0}, c_skip_unreadable{0}, c_evicted{0}, c_evicted_bytes{0}, c_replaced{0};
std::atomic<u64> g_ring_begin{0}, g_ring_end{0};
u64 RingStoreMaxBytes() {
    static const u64 limit = [] {
        const char* e = std::getenv("TLG_F2_RING_STORE_KB");
        const u64 kb = e != nullptr ? std::strtoull(e, nullptr, 10) : 0;
        return (kb != 0 ? kb : 256ull) << 10;
    }();
    return limit;
}
std::atomic<u64> c_ov_calls{0}, c_ov_images{0}, c_ov_bytes{0}, c_ov_changed_bytes{0}, c_ov_images_changed{0}, c_ov_uncovered_bytes{0}, c_ov_skipped_open_map{0}, c_ov_skipped_no_seq{0}, c_ov_not_ring{0};
std::atomic<u64> c_cmp_open{0}, c_cmp_open_differs{0}, c_cmp_closed{0}, c_cmp_closed_differs{0};
std::atomic<u64> c_copies{0}, c_copy_bytes_covered{0}, c_copy_bytes_uncovered{0}, c_copy_skipped_big{0}, c_deleted{0}, c_deleted_bytes{0};
std::atomic<u64> c_announced_by_writer[8]{};
std::atomic<u64> c_skipped_big_by_writer[8]{}, c_skipped_big_by_log2[40]{};

// per group and class
enum Class : u32 { ClassEqual, ClassEmptyEqual, ClassDiffers, ClassHole, ClassCount };
constexpr const char* ClassNames[ClassCount] = {"equal (every byte the store covers equals the memory; the rest is zero)", "empty_equal (nothing covered, memory is zero)",
                                                "differs (a covered byte differs from the memory)", "hole (bytes nothing announced hold data)"};
u64 g_class_uploads[3][ClassCount]{};
u64 g_class_bytes[3][ClassCount]{};
u64 g_class_differing_bytes[3]{};
u64 g_class_hole_bytes[3]{};
u64 g_gpu_covered_bytes = 0, g_gpu_total_bytes = 0, g_gpu_uploads = 0, g_gpu_fully_covered = 0, g_gpu_unreadable = 0;
u64 g_unreadable = 0;
u64 g_differs_by_writer[16]{};
u64 g_differs_explained[8]{};
std::map<std::string, u64> g_differs_by_owner; ///< "kind usage cpu_access" of the engine resource that owns the memory, for the differing uploads ///< differing uploads by the writer of a later mark over the image (0: no mark: unexplained)
u64 g_equal_bytes_by_writer[16]{};
u64 g_equal_nonzero_by_writer[16]{};  ///< bytes of CPU sourced uploads that equal the store, by the writer of the entry (15: a copy)
u64 g_uploads_by_main_writer[16]{}; ///< uploads by the writer that supplies most of their covered bytes
std::vector<nlohmann::json> g_examples_differs, g_examples_hole;

// ---- F2.3 ring windows (shadow) ---------------------------------------------------------------------------------------------------------------------------------------
struct RingWindow {
    u64 size;
    u64 serial;
};
std::mutex g_windows_mutex;
std::map<u64, RingWindow> g_windows;
u64 g_window_serial = 0;
std::atomic<u64> c_win_noted{0}, c_win_dropped{0};
enum RingClass : u32 { RingExact, RingExactTooSmall, RingInterior, RingSpans, RingNone, RingNotInRing, RingClassCount };
constexpr const char* RingClassNames[RingClassCount] = {"exact_start_and_fits", "start_but_window_too_small", "inside_a_window", "spans_windows", "no_window", "not_in_ring"};
std::atomic<u64> c_ring_lookups[5][RingClassCount]{};   ///< [binding type][class]
std::atomic<u64> c_ring_age_log2[RingClassCount][24]{}; ///< windows older than 2^n allocations, by class
std::vector<nlohmann::json> g_ring_examples;
std::map<u64, RingWindow> g_gpu_windows; ///< F2.3: the engine's own GPU memory allocations (AllocGarlicMemory / AllocOnionMemory), under g_windows_mutex
enum GpuClass : u32 { GpuExact, GpuInside, GpuNone, GpuClassCount };
constexpr const char* GpuClassNames[GpuClassCount] = {"allocation_starts_here_and_fits", "inside_an_allocation", "no_allocation"};
std::atomic<u64> c_gpu_lookups[5][GpuClassCount]{};
std::vector<nlohmann::json> g_gpu_examples;

std::atomic<u64> c_maps[8][8][4]{}; ///< [resource kind (0xff: unknown -> 7)][map type][path]
const char* GroupName(Group g) {
    return g == Group::FirstUpload ? "first_upload" : g == Group::RepeatUpload ? "repeat_upload" : "gpu_source";
}

// ---- guarded reads of guest memory ---------------------------------------------------------------------------------------------------------------------------------
using Thunk = void (*)(void*);
#if defined(TLG_PORT)
__declspec(noinline) bool RunGuarded(Thunk thunk, void* argument) {
    sigjmp_buf recovery;
    if (tlg_setjmp(recovery) != 0) {
        return false;
    }
    runtime_fault_recover = &recovery;
    thunk(argument);
    runtime_fault_recover = nullptr;
    return true;
}
#elif defined(_WIN32)
__declspec(noinline) bool RunGuarded(Thunk thunk, void* argument) {
    __try {
        thunk(argument);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}
#else
bool RunGuarded(Thunk thunk, void* argument) {
    thunk(argument);
    return true;
}
#endif

struct CopyJob {
    void* destination;
    const void* source;
    u64 size;
};
bool GuardedCopy(void* destination, const void* source, u64 size) {
    CopyJob job{destination, source, size};
    return RunGuarded([](void* p) { std::memcpy(static_cast<CopyJob*>(p)->destination, static_cast<CopyJob*>(p)->source, static_cast<CopyJob*>(p)->size); }, &job);
}

struct CompareJob {
    const void* a;
    const void* b;
    u64 size;
    u64 first_diff; ///< offset of the first differing byte, ~0 when equal
    u64 nonzero;    ///< CompareJob with b == nullptr: counts the non-zero bytes of a
};
bool GuardedCompare(const void* memory, const void* stored, u64 size, u64& first_diff) {
    CompareJob job{memory, stored, size, ~0ull, 0};
    const bool ok = RunGuarded(
        [](void* p) {
            auto& j = *static_cast<CompareJob*>(p);
            if (std::memcmp(j.a, j.b, j.size) == 0) {
                return;
            }
            const u8* x = static_cast<const u8*>(j.a);
            const u8* y = static_cast<const u8*>(j.b);
            u64 i = 0;
            while (i < j.size && x[i] == y[i]) {
                ++i;
            }
            j.first_diff = i;
        },
        &job);
    first_diff = job.first_diff;
    return ok;
}
bool GuardedNonZero(const void* memory, u64 size, u64& nonzero) {
    CompareJob job{memory, nullptr, size, ~0ull, 0};
    const bool ok = RunGuarded(
        [](void* p) {
            auto& j = *static_cast<CompareJob*>(p);
            const u8* x = static_cast<const u8*>(j.a);
            u64 n = 0;
            for (u64 i = 0; i < j.size; ++i) {
                n += x[i] != 0;
            }
            j.nonzero = n;
        },
        &job);
    nonzero = job.nonzero;
    return ok;
}

u64 CapBytes() {
    static const u64 cap = [] {
        const bool swap_only = !ShadowOn();
        const char* e = std::getenv(swap_only ? "TLG_F2_RING_STORE_MB" : "TLG_F2_STORE_MB");
        const u64 mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 0;
        return (mb != 0 ? mb : swap_only ? 256ull : 1024ull) << 20;
    }();
    return cap;
}

/// Inserts an entry under the lock: older entries it covers completely are dropped, the oldest entries go when the cap is reached.
void InsertLocked(u64 begin, Entry&& entry) {
    const u64 end = begin + entry.size;
    for (auto it = g_entries.lower_bound(begin); it != g_entries.end() && it->first < end;) {
        if (!RingSwapOn() && it->first + it->second.size <= end) { // the swap reads the store as of an older record: nothing is dropped because a newer entry covers it
            g_stored_bytes -= it->second.writer == WriterCopy ? 0 : it->second.size;
            c_replaced.fetch_add(1, std::memory_order_relaxed);
            it = g_entries.erase(it);
        } else {
            ++it;
        }
    }
    entry.seq = g_seq_atomic.fetch_add(1, std::memory_order_relaxed) + 1;
    g_stored_bytes += entry.writer == WriterCopy ? 0 : entry.size; // a copy's entries share the bytes of the entry they came from
    g_max_size = std::max(g_max_size, entry.size);
    g_fifo.emplace_back(begin, entry.seq);
    g_entries.emplace(begin, std::move(entry));
    const u64 cap = CapBytes();
    while (g_stored_bytes > cap && !g_fifo.empty()) {
        const auto [b, seq] = g_fifo.front();
        g_fifo.pop_front();
        const auto range = g_entries.equal_range(b);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second.seq == seq) {
                g_stored_bytes -= it->second.writer == WriterCopy ? 0 : it->second.size;
                c_evicted.fetch_add(1, std::memory_order_relaxed);
                c_evicted_bytes.fetch_add(it->second.size, std::memory_order_relaxed);
                g_entries.erase(it);
                break;
            }
        }
    }
}

struct Piece {
    u64 begin; ///< address
    u64 end;
    const Entry* entry;
    u64 entry_begin;
};

/// The newest entry per byte of [begin, end): pieces (disjoint, entry known) and the segments no entry covers. Under the lock.
void ResolveLocked(u64 begin, u64 end, std::vector<Piece>& pieces, std::vector<std::pair<u64, u64>>& uncovered, u64 max_stamp = ~0ull) {
    std::vector<Piece> candidates;
    const u64 from = begin > g_max_size ? begin - g_max_size : 0;
    for (auto it = g_entries.lower_bound(from); it != g_entries.end() && it->first < end; ++it) {
        if (it->first + it->second.size > begin && it->second.stamp <= max_stamp) {
            candidates.push_back({std::max(begin, it->first), std::min(end, it->first + it->second.size), &it->second, it->first});
        }
    }
    const bool by_stamp = max_stamp != ~0ull;
    std::sort(candidates.begin(), candidates.end(), [by_stamp](const Piece& a, const Piece& b) {
        if (by_stamp && a.entry->stamp != b.entry->stamp) {
            return a.entry->stamp > b.entry->stamp;
        }
        return a.entry->seq > b.entry->seq;
    });
    uncovered.assign(1, {begin, end});
    for (const Piece& c : candidates) {
        std::vector<std::pair<u64, u64>> next;
        for (const auto& [s, t] : uncovered) {
            const u64 x = std::max(s, c.begin);
            const u64 y = std::min(t, c.end);
            if (x >= y) {
                next.emplace_back(s, t);
                continue;
            }
            pieces.push_back({x, y, c.entry, c.entry_begin});
            if (s < x) {
                next.emplace_back(s, x);
            }
            if (y < t) {
                next.emplace_back(y, t);
            }
        }
        uncovered.swap(next);
        if (uncovered.empty()) {
            break;
        }
    }
}

} // namespace

bool ShadowOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_STORE");
        return e != nullptr && *e == '1';
    }();
    return on;
}

bool RingSwapOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_RING_STORE"); // default on since 2026-11-03 (owner play, 0 bytes differ); =0 turns it off
        return e == nullptr || *e != '0';
    }();
    return on;
}

bool On() noexcept {
    return ShadowOn() || RingSwapOn();
}

void NoteMark(u32 writer, u64 address, u64 size) {
    if (!ShadowOn() || address == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{g_marks_mutex};
    const u64 seq = g_seq_atomic.load(std::memory_order_relaxed);
    g_marks.emplace(address, Mark{size, seq, writer});
    g_marks_max_size = std::max(g_marks_max_size, size);
    g_marks_fifo.emplace_back(address, ++g_mark_serial);
    if (g_marks.size() > MaxMarks) {
        const auto [b, serial] = g_marks_fifo.front();
        g_marks_fifo.pop_front();
        (void)serial;
        g_marks.erase(g_marks.find(b)); // the oldest mark at that address
    }
}

bool InRing(u64 address, u64 size) noexcept {
    const u64 begin = g_ring_begin.load(std::memory_order_relaxed);
    return begin != 0 && address >= begin && address + size <= g_ring_end.load(std::memory_order_relaxed);
}

namespace {
std::atomic<u64> c_ring_index[static_cast<u32>(RingIndexEvent::Count)]{};
}
void NoteRingIndex(RingIndexEvent event) noexcept {
    c_ring_index[static_cast<u32>(event)].fetch_add(1, std::memory_order_relaxed);
}

bool RingWindowsOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_RING_WINDOWS");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void NoteRingWindow(u64 address, u64 size) {
    if (!RingWindowsOn() || address == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{g_windows_mutex};
    const u64 end = address + size;
    auto it = g_windows.lower_bound(address);
    if (it != g_windows.begin()) {
        const auto before = std::prev(it);
        if (before->first + before->second.size > address) {
            it = before;
        }
    }
    while (it != g_windows.end() && it->first < end) {
        if (it->first + it->second.size > address) {
            it = g_windows.erase(it);
            c_win_dropped.fetch_add(1, std::memory_order_relaxed);
        } else {
            ++it;
        }
    }
    g_windows.emplace(address, RingWindow{size, ++g_window_serial});
    c_win_noted.fetch_add(1, std::memory_order_relaxed);
}

bool InGpuAllocation(u64 address, u64 size) {
    std::scoped_lock lock{g_windows_mutex};
    auto it = g_gpu_windows.upper_bound(address);
    if (it == g_gpu_windows.begin()) {
        return false;
    }
    --it;
    return address + size <= it->first + it->second.size;
}

void NoteGpuAllocation(u64 address, u64 size) {
    if (address == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{g_windows_mutex};
    const u64 end = address + size;
    auto it = g_gpu_windows.lower_bound(address);
    if (it != g_gpu_windows.begin()) {
        const auto before = std::prev(it);
        if (before->first + before->second.size > address) {
            it = before;
        }
    }
    while (it != g_gpu_windows.end() && it->first < end) {
        it = it->first + it->second.size > address ? g_gpu_windows.erase(it) : std::next(it);
    }
    g_gpu_windows.emplace(address, RingWindow{size, ++g_window_serial});
}

void ClassifyRingLookup(u64 address, u64 size, u32 binding_type) {
    if (!RingWindowsOn() || size == 0) {
        return;
    }
    const u64 type = binding_type < 5 ? binding_type : 4;
    const u64 ring_begin = g_ring_begin.load(std::memory_order_relaxed);
    if (ring_begin == 0 || address < ring_begin || address >= g_ring_end.load(std::memory_order_relaxed)) {
        c_ring_lookups[type][RingNotInRing].fetch_add(1, std::memory_order_relaxed);
        GpuClass gpu = GpuNone;
        u64 alloc_begin = 0, alloc_size = 0;
        {
            std::scoped_lock lock{g_windows_mutex};
            auto it = g_gpu_windows.upper_bound(address);
            if (it != g_gpu_windows.begin()) {
                const auto before = std::prev(it);
                if (before->first + before->second.size > address) {
                    gpu = before->first == address && before->second.size >= size ? GpuExact : GpuInside;
                    alloc_begin = before->first, alloc_size = before->second.size;
                }
            }
            if (gpu != GpuExact && g_gpu_examples.size() < 16) {
                g_gpu_examples.push_back({{"class", GpuClassNames[gpu]}, {"binding_type", type}, {"image_address", fmt::format("{:#x}", address)}, {"image_size", size},
                                          {"allocation_address", fmt::format("{:#x}", alloc_begin)}, {"allocation_size", alloc_size}});
            }
        }
        c_gpu_lookups[type][gpu].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    RingClass cls = RingNone;
    u64 age = 0;
    u64 window_begin = 0, window_size = 0;
    {
        std::scoped_lock lock{g_windows_mutex};
        auto it = g_windows.upper_bound(address);
        if (it != g_windows.begin()) {
            const auto before = std::prev(it);
            if (before->first == address) {
                cls = before->second.size >= size ? RingExact : RingExactTooSmall;
                window_begin = before->first, window_size = before->second.size, age = g_window_serial - before->second.serial;
            } else if (before->first + before->second.size > address) {
                cls = RingInterior;
                window_begin = before->first, window_size = before->second.size, age = g_window_serial - before->second.serial;
            }
        }
        if (cls == RingNone && it != g_windows.end() && it->first < address + size) {
            cls = RingSpans;
            window_begin = it->first, window_size = it->second.size, age = g_window_serial - it->second.serial;
        }
    }
    c_ring_lookups[type][cls].fetch_add(1, std::memory_order_relaxed);
    c_ring_age_log2[cls][std::min<u32>(23u, static_cast<u32>(std::bit_width(age)))].fetch_add(1, std::memory_order_relaxed);
    if (cls != RingExact) {
        std::scoped_lock lock{g_windows_mutex};
        if (g_ring_examples.size() < 16) {
            g_ring_examples.push_back({{"class", RingClassNames[cls]}, {"binding_type", type}, {"image_address", fmt::format("{:#x}", address)}, {"image_size", size}, {"window_address", fmt::format("{:#x}", window_begin)},
                                       {"window_size", window_size}, {"window_age_allocations", age}});
        }
    }
}

void SetRing(u64 begin, u64 end) {
    g_ring_begin.store(begin, std::memory_order_relaxed);
    g_ring_end.store(end, std::memory_order_relaxed);
}

void Announce(u32 writer, u64 address, u64 size, u64 stamp) {
    if (!On() || address == 0 || size == 0) {
        return;
    }
    if (!ShadowOn()) {
        // the swap alone keeps the ring's announcements only
        const u64 begin = g_ring_begin.load(std::memory_order_relaxed);
        if (begin == 0 || address < begin || address + size > g_ring_end.load(std::memory_order_relaxed) || size > RingStoreMaxBytes()) {
            return;
        }
    }
    if (writer == 1 || writer == 2 || writer == 4) {
        c_skip_writer.fetch_add(1, std::memory_order_relaxed); // announced before the write: no content yet, and buffers only; kept as a mark
        NoteMark(writer, address, size);
        return;
    }
    if (size > (writer == 6 ? MaxBytesWriter6 : MaxBytesOther())) {
        c_skip_big.fetch_add(1, std::memory_order_relaxed);
        c_skipped_big_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
        c_skipped_big_by_log2[std::min<u32>(39u, static_cast<u32>(std::bit_width(size)))].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Entry entry;
    entry.size = size;
    entry.writer = writer;
    entry.stamp = stamp;
    std::shared_ptr<u8[]> block{new u8[size]};
    if (!GuardedCopy(block.get(), reinterpret_cast<const void*>(address), size)) {
        c_skip_unreadable.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    entry.block = std::move(block);
    c_announced.fetch_add(1, std::memory_order_relaxed);
    c_announced_bytes.fetch_add(size, std::memory_order_relaxed);
    c_announced_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{g_mutex};
    InsertLocked(address, std::move(entry));
}

void NoteCopy(u64 destination, u64 source, u64 size) {
    if (!ShadowOn() || size == 0) {
        return;
    }
    if (size > MaxBytesWriter6) {
        c_copy_skipped_big.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_copies.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{g_mutex};
    std::vector<Piece> pieces;
    std::vector<std::pair<u64, u64>> uncovered;
    ResolveLocked(source, source + size, pieces, uncovered);
    u64 covered = 0;
    std::vector<std::pair<u64, Entry>> made;
    for (const Piece& p : pieces) {
        Entry e;
        e.size = p.end - p.begin;
        e.writer = WriterCopy;
        e.block = p.entry->block;
        e.offset = p.entry->offset + (p.begin - p.entry_begin);
        covered += e.size;
        made.emplace_back(destination + (p.begin - source), std::move(e));
    }
    c_copy_bytes_covered.fetch_add(covered, std::memory_order_relaxed);
    c_copy_bytes_uncovered.fetch_add(size - covered, std::memory_order_relaxed);
    // where the source has no content the destination keeps what it had: older entries under an uncovered part stay
    for (auto& [begin, e] : made) {
        InsertLocked(begin, std::move(e));
    }
}

void CountMap(u32 kind, u32 type, u32 path) {
    if (ShadowOn()) {
        c_maps[kind < 7 ? kind : 7][type < 8 ? type : 7][path & 3].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteDeleted(u64, u64 size) {
    if (!On()) {
        return;
    }
    c_deleted.fetch_add(1, std::memory_order_relaxed);
    c_deleted_bytes.fetch_add(size, std::memory_order_relaxed);
}

void OverlayImage(u64 address, u64 size, u8* staged) {
    if (!RingSwapOn() || size == 0) {
        return;
    }
    const u64 begin = g_ring_begin.load(std::memory_order_relaxed);
    if (begin == 0 || address < begin || address + size > g_ring_end.load(std::memory_order_relaxed)) {
        c_ov_not_ring.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_ov_calls.fetch_add(1, std::memory_order_relaxed);
    const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
    if (seq == 0) {
        c_ov_skipped_no_seq.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (Core::NativeUpdates::OpenMapOverlaps(address, size)) {
        // a Map of this memory has not been announced yet (its writer has not finished or has not unmapped): the newest entry may be an older one, the memory is what is left
        c_ov_skipped_open_map.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::scoped_lock lock{g_mutex};
    std::vector<Piece> pieces;
    std::vector<std::pair<u64, u64>> uncovered;
    ResolveLocked(address, address + size, pieces, uncovered, seq);
    u64 covered = 0, changed = 0;
    for (const Piece& p : pieces) {
        u8* destination = staged + (p.begin - address);
        const u8* source = p.entry->Bytes() + (p.begin - p.entry_begin);
        const u64 n = p.end - p.begin;
        if (std::memcmp(destination, source, n) != 0) {
            for (u64 i = 0; i < n; ++i) {
                changed += destination[i] != source[i];
            }
            std::memcpy(destination, source, n);
        }
        covered += n;
    }
    c_ov_images.fetch_add(1, std::memory_order_relaxed);
    c_ov_bytes.fetch_add(covered, std::memory_order_relaxed);
    c_ov_uncovered_bytes.fetch_add(size - covered, std::memory_order_relaxed);
    if (changed != 0) {
        c_ov_changed_bytes.fetch_add(changed, std::memory_order_relaxed);
        c_ov_images_changed.fetch_add(1, std::memory_order_relaxed);
    }
}

void CompareImage(const ImageUpload& upload) {
    if (!ShadowOn() || upload.size == 0) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    Group group = upload.group_hint;
    if (group != Group::GpuSource) {
        group = g_uploaded_images.insert(upload.image_uid).second ? Group::FirstUpload : Group::RepeatUpload;
    }
    std::vector<Piece> pieces;
    std::vector<std::pair<u64, u64>> uncovered;
    ResolveLocked(upload.address, upload.address + upload.size, pieces, uncovered);
    u64 covered = 0;
    for (const Piece& p : pieces) {
        covered += p.end - p.begin;
    }
    if (group == Group::GpuSource) {
        // the upload read GPU written memory: the store cannot be compared with it, only asked how much of the image it knows
        ++g_gpu_uploads;
        g_gpu_total_bytes += upload.size;
        g_gpu_covered_bytes += covered;
        g_gpu_fully_covered += covered == upload.size;
        return;
    }
    u64 equal = 0, differing = 0, first_diff = ~0ull, newest_differing_seq = 0, oldest_differing_seq = ~0ull;
    u64 bytes_by_writer[16]{}, nonzero_by_writer[16]{};
    u32 differing_writer = 0;
    bool unreadable = false;
    std::vector<std::pair<u64, std::pair<u32, u32>>> diff_samples; ///< (offset, memory dword, store dword) of the first differing dwords, small images only
    for (const Piece& p : pieces) {
        u64 diff_at = ~0ull;
        if (!GuardedCompare(reinterpret_cast<const void*>(p.begin), p.entry->Bytes() + (p.begin - p.entry_begin), p.end - p.begin, diff_at)) {
            unreadable = true;
            continue;
        }
        bytes_by_writer[std::min<u32>(p.entry->writer == WriterCopy ? 15u : p.entry->writer, 15u)] += p.end - p.begin;
        if (diff_at == ~0ull) {
            u64 nz = 0;
            GuardedNonZero(reinterpret_cast<const void*>(p.begin), p.end - p.begin, nz);
            nonzero_by_writer[std::min<u32>(p.entry->writer == WriterCopy ? 15u : p.entry->writer, 15u)] += nz;
            equal += p.end - p.begin;
        } else {
            u64 d = 0; // count the differing bytes of this piece
            const bool want_samples = g_examples_differs.size() < 12 && upload.size <= (64u << 10) && diff_samples.size() < 8;
            const u8* m = reinterpret_cast<const u8*>(p.begin);
            const u8* s = p.entry->Bytes() + (p.begin - p.entry_begin);
            for (u64 i = diff_at; i < p.end - p.begin; ++i) {
                d += m[i] != s[i];
                if (want_samples && m[i] != s[i] && diff_samples.size() < 8 && (diff_samples.empty() || p.begin + i - upload.address >= diff_samples.back().first + 4)) {
                    u32 mv = 0, sv = 0;
                    std::memcpy(&mv, m + (i & ~3ull), 4);
                    std::memcpy(&sv, s + (i & ~3ull), 4);
                    diff_samples.emplace_back(p.begin + (i & ~3ull) - upload.address, std::pair<u32, u32>{mv, sv});
                }
            }
            differing += d;
            equal += (p.end - p.begin) - d;
            if (p.begin + diff_at - upload.address < first_diff) {
                first_diff = p.begin + diff_at - upload.address;
            }
            oldest_differing_seq = std::min(oldest_differing_seq, p.entry->seq);
            if (p.entry->seq >= newest_differing_seq) {
                newest_differing_seq = p.entry->seq;
                differing_writer = p.entry->writer;
            }
        }
    }
    u64 hole = 0;
    for (const auto& [s, t] : uncovered) {
        u64 nonzero = 0;
        if (!GuardedNonZero(reinterpret_cast<const void*>(s), t - s, nonzero)) {
            unreadable = true;
            continue;
        }
        hole += nonzero;
    }
    if (unreadable) {
        ++g_unreadable;
    }
    u32 explained_by = 0; // the writer (1, 2, 4) of a mark newer than the oldest differing entry that overlaps the image, 0 when none
    if (differing != 0) {
        std::scoped_lock marks_lock{g_marks_mutex};
        const u64 from = upload.address > g_marks_max_size ? upload.address - g_marks_max_size : 0;
        for (auto it = g_marks.lower_bound(from); it != g_marks.end() && it->first < upload.address + upload.size; ++it) {
            if (it->first + it->second.size > upload.address && it->second.seq >= oldest_differing_seq) {
                explained_by = it->second.writer;
            }
        }
        ++g_differs_explained[explained_by & 7];
        Core::NativeDevice::Object owner;
        if (Core::NativeDevice::ResourceAt(upload.address, owner)) {
            ++g_differs_by_owner[fmt::format("kind {} usage {} cpu_access {:#x} bind {:#x}", static_cast<u32>(owner.kind), owner.usage, owner.cpu_access, owner.bind)];
        } else {
            ++g_differs_by_owner["no engine resource owns the memory"];
        }
    }
    if (differing == 0) {
        u32 main = 0;
        for (u32 w = 0; w < 16; ++w) {
            g_equal_bytes_by_writer[w] += bytes_by_writer[w];
            g_equal_nonzero_by_writer[w] += nonzero_by_writer[w];
            if (bytes_by_writer[w] > bytes_by_writer[main]) {
                main = w;
            }
        }
        if (covered != 0) {
            ++g_uploads_by_main_writer[main];
        }
    }
    // a Map of this memory still open (not unmapped, not yet flushed at a frame start): its writer's announcement has not been made, so a differing upload there only says the store lags
    const bool map_open = Core::NativeUpdates::OpenMapOverlaps(upload.address, upload.size);
    (map_open ? c_cmp_open : c_cmp_closed).fetch_add(1, std::memory_order_relaxed);
    if (differing != 0) {
        (map_open ? c_cmp_open_differs : c_cmp_closed_differs).fetch_add(1, std::memory_order_relaxed);
    }
    const Class cls = differing != 0 ? ClassDiffers : hole != 0 ? ClassHole : covered == 0 ? ClassEmptyEqual : ClassEqual;
    const u32 g = static_cast<u32>(group);
    ++g_class_uploads[g][cls];
    g_class_bytes[g][cls] += upload.size;
    g_class_differing_bytes[g] += differing;
    g_class_hole_bytes[g] += hole;
    if (differing != 0) {
        ++g_differs_by_writer[std::min<u32>(differing_writer == WriterCopy ? 15u : differing_writer, 15u)];
    }
    if ((cls == ClassDiffers && g_examples_differs.size() < 12) || (cls == ClassHole && g_examples_hole.size() < 12)) {
        nlohmann::json e;
        e["group"] = GroupName(group);
        e["address"] = fmt::format("{:#x}", upload.address);
        e["size"] = upload.size;
        e["width"] = upload.width;
        e["height"] = upload.height;
        e["format"] = upload.format;
        e["caller"] = upload.caller;
        e["covered_bytes"] = covered;
        e["equal_bytes"] = equal;
        e["differing_bytes"] = differing;
        e["uncovered_nonzero_bytes"] = hole;
        if (differing != 0) {
            e["first_differing_offset"] = first_diff;
            e["newest_differing_writer"] = differing_writer;
            e["announcements_ago"] = g_seq_atomic.load() - newest_differing_seq;
            e["later_mark_writer"] = explained_by;
            for (const auto& [offset, values] : diff_samples) {
                e["diff_dwords"].push_back(fmt::format("+{:#x}: memory {:#010x} store {:#010x}", offset, values.first, values.second));
            }
            if (Core::NativeDevice::Object owner; Core::NativeDevice::ResourceAt(upload.address, owner)) {
                e["owner"] = fmt::format("kind {} id {} usage {} cpu_access {:#x} bind {:#x} {}x{} fmt {} at {:#x}+{:#x}", static_cast<u32>(owner.kind), owner.id, owner.usage, owner.cpu_access,
                                         owner.bind, owner.width, owner.height, owner.format, static_cast<u64>(owner.address), owner.bytes);
            }
        }
        (cls == ClassDiffers ? g_examples_differs : g_examples_hole).push_back(std::move(e));
    }
}

nlohmann::json Report() {
    nlohmann::json out;
    if (RingWindowsOn()) {
        nlohmann::json w;
        w["windows_noted"] = c_win_noted.load();
        w["older_windows_dropped"] = c_win_dropped.load();
        static constexpr const char* types[5] = {"texture", "storage", "render_target", "depth_target", "video_out"};
        for (u32 t = 0; t < 5; ++t) {
            for (u32 c = 0; c < RingClassCount; ++c) {
                if (const u64 n = c_ring_lookups[t][c].load(); n != 0) {
                    w["lookups"][types[t]][RingClassNames[c]] = n;
                }
            }
        }
        for (u32 c = 0; c < RingClassCount; ++c) {
            for (u32 b = 0; b < 24; ++b) {
                if (const u64 n = c_ring_age_log2[c][b].load(); n != 0) {
                    w["window_age_below_allocations"][RingClassNames[c]][std::to_string(1ull << b)] = n;
                }
            }
        }
        {
            std::scoped_lock lock{g_windows_mutex};
            w["windows_now"] = g_windows.size();
            w["examples_not_exact"] = g_ring_examples;
        }
        for (u32 t = 0; t < 5; ++t) {
            for (u32 c = 0; c < GpuClassCount; ++c) {
                if (const u64 n = c_gpu_lookups[t][c].load(); n != 0) {
                    w["outside_the_ring_against_gpu_allocations"][types[t]][GpuClassNames[c]] = n;
                }
            }
        }
        {
            std::scoped_lock lock{g_windows_mutex};
            w["gpu_allocations_now"] = g_gpu_windows.size();
            w["examples_outside_ring_not_exact"] = g_gpu_examples;
        }
        out["ring_windows"] = std::move(w);
    }
    out["ring_index"] = {{"reused_exact_match", c_ring_index[0].load()},
                         {"created_new_image", c_ring_index[1].load()},
                         {"stale_images_freed", c_ring_index[2].load()},
                         {"stale_images_kept_not_old_enough", c_ring_index[3].load()},
                         {"owner_less_lookups_refused_not_in_ring_or_gpu_allocation", c_ring_index[4].load()}};
    out["enabled"] = On();
    out["shadow"] = ShadowOn();
    out["compare_by_open_map"] = {{"open_map_uploads", c_cmp_open.load()}, {"open_map_differs", c_cmp_open_differs.load()}, {"no_open_map_uploads", c_cmp_closed.load()}, {"no_open_map_differs", c_cmp_closed_differs.load()}};
    out["ring_swap"] = {{"on", RingSwapOn()}, {"images_overlaid", c_ov_images.load()}, {"bytes_from_store", c_ov_bytes.load()}, {"bytes_not_covered_left_to_memory", c_ov_uncovered_bytes.load()},
                        {"images_where_store_differs_from_memory", c_ov_images_changed.load()}, {"bytes_where_store_differs_from_memory", c_ov_changed_bytes.load()},
                        {"calls", c_ov_calls.load()}, {"not_in_ring", c_ov_not_ring.load()}, {"skipped_open_map", c_ov_skipped_open_map.load()}, {"skipped_no_record", c_ov_skipped_no_seq.load()},
                        {"switch", "TLG_F2_RING_STORE=0 turns it off"}};
    if (!ShadowOn()) {
        std::scoped_lock lock{g_mutex};
        out["store"] = {{"entries", g_entries.size()}, {"bytes", g_stored_bytes}, {"announced", c_announced.load()}, {"evicted", c_evicted.load()}};
        if (!On()) {
            out["switch"] = "TLG_F2_STORE=1 or TLG_F2_RING_STORE=1";
        }
        return out;
    }
    out["announced"] = {{"entries", c_announced.load()}, {"bytes", c_announced_bytes.load()}, {"skipped_writer_1_2_4", c_skip_writer.load()}, {"skipped_too_big", c_skip_big.load()},
                        {"skipped_unreadable", c_skip_unreadable.load()}, {"older_entries_replaced", c_replaced.load()}, {"evicted", c_evicted.load()}, {"evicted_bytes", c_evicted_bytes.load()}};
    nlohmann::json by_writer;
    for (u32 w = 0; w < 8; ++w) {
        by_writer[std::to_string(w)] = c_announced_by_writer[w].load();
    }
    out["announced_by_writer"] = by_writer;
    nlohmann::json skipped;
    for (u32 w = 0; w < 8; ++w) {
        skipped["by_writer"][std::to_string(w)] = c_skipped_big_by_writer[w].load();
    }
    for (u32 b = 0; b < 40; ++b) {
        if (const u64 n = c_skipped_big_by_log2[b].load(); n != 0) {
            skipped["by_size_below"][std::to_string(1ull << b)] = n;
        }
    }
    out["skipped_too_big_detail"] = skipped;
    out["copies"] = {{"copies", c_copies.load()}, {"bytes_resolved_from_store", c_copy_bytes_covered.load()}, {"bytes_source_unknown", c_copy_bytes_uncovered.load()}, {"skipped_too_big", c_copy_skipped_big.load()}};
    nlohmann::json maps;
    for (u32 k = 0; k < 8; ++k) {
        for (u32 t = 0; t < 8; ++t) {
            for (u32 path = 0; path < 4; ++path) {
                if (const u64 n = c_maps[k][t][path].load(); n != 0) {
                    maps[fmt::format("resource_kind {} map_type {} path {}", k == 7 ? "unknown" : std::to_string(k), t, path)] = n;
                }
            }
        }
    }
    out["maps"] = maps;
    out["maps_note"] = "path 0 = buffer, announced at Unmap; 1 = texture, native upload record; 2 = texture, record refused; 3 = not handled by the hook (nothing announced)";
    out["deleted_owner_notes"] = {{"count", c_deleted.load()}, {"bytes", c_deleted_bytes.load()}};
    std::scoped_lock lock{g_mutex};
    out["store"] = {{"entries", g_entries.size()}, {"bytes", g_stored_bytes}, {"cap_bytes", CapBytes()}};
    nlohmann::json uploads;
    for (u32 g = 0; g < 2; ++g) {
        nlohmann::json one;
        for (u32 c = 0; c < ClassCount; ++c) {
            one[ClassNames[c]] = {{"uploads", g_class_uploads[g][c]}, {"image_bytes", g_class_bytes[g][c]}};
        }
        one["differing_bytes"] = g_class_differing_bytes[g];
        one["uncovered_nonzero_bytes"] = g_class_hole_bytes[g];
        uploads[GroupName(static_cast<Group>(g))] = std::move(one);
    }
    uploads["gpu_source"] = {{"uploads", g_gpu_uploads}, {"image_bytes", g_gpu_total_bytes}, {"bytes_the_store_knows", g_gpu_covered_bytes}, {"fully_known", g_gpu_fully_covered}};
    out["uploads"] = std::move(uploads);
    nlohmann::json writers;
    for (u32 w = 0; w < 16; ++w) {
        if (g_differs_by_writer[w] != 0) {
            writers[w == 15 ? "copy" : std::to_string(w)] = g_differs_by_writer[w];
        }
    }
    out["differing_uploads_by_newest_writer"] = writers;
    nlohmann::json supplied;
    for (u32 w = 0; w < 16; ++w) {
        if (g_equal_bytes_by_writer[w] != 0 || g_uploads_by_main_writer[w] != 0) {
            supplied[w == 15 ? "copy" : std::to_string(w)] = {{"equal_bytes", g_equal_bytes_by_writer[w]}, {"of_which_nonzero", g_equal_nonzero_by_writer[w]}, {"uploads_where_it_supplies_most", g_uploads_by_main_writer[w]}};
        }
    }
    out["equal_uploads_supplied_by_writer"] = supplied;
    nlohmann::json explained;
    for (u32 w = 0; w < 8; ++w) {
        if (g_differs_explained[w] != 0) {
            explained[w == 0 ? "no_later_mark_unexplained" : "later_mark_of_writer_" + std::to_string(w)] = g_differs_explained[w];
        }
    }
    out["differing_uploads_explained_by"] = explained;
    out["differing_uploads_by_owner"] = g_differs_by_owner;
    out["uploads_with_unreadable_memory"] = g_unreadable;
    out["examples_differs"] = g_examples_differs;
    out["examples_hole"] = g_examples_hole;
    out["note"] = "F2.2 shadow: bytes taken when the writer announced them (newest wins) against the memory the image upload read; differs = a covered byte changed after the announcement or came from a later write; hole = bytes no announcement covers hold data";
    return out;
}

} // namespace Core::NativeContent
