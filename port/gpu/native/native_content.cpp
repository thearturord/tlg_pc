// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <mutex>
#include <set>
#include <condition_variable>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <xxhash.h>
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef TLG_PORT
#include "tlg_toggles.h" // sigjmp_buf, tlg_setjmp, runtime_fault_recover
#endif
#include "native/native_byte_audit.h"
#include "native/native_content.h"
#include "native/native_debug_server.h"
#include "native/native_device.h"
#include "native/native_draw.h"
#include "native/native_registry.h"
#include "native/native_tsharp_watch.h"
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
/// F2 census: ranges the store dropped for its cap (tombstones: begin to end, bytes not kept). A hole in an upload that lies in tombstones was announced once and evicted; a hole nobody ever
/// announced has no tombstone.
std::multimap<u64, u64> g_tomb;
u64 g_tomb_max = 0;
constexpr size_t MaxTombs = 4u << 20;
u64 g_hole_nonzero_evicted = 0, g_hole_nonzero_never = 0, g_hole_nonzero_mixed = 0, g_hole_uploads_all_evicted = 0, g_hole_uploads_with_never = 0;
u64 g_gpu_covered_bytes = 0, g_gpu_total_bytes = 0, g_gpu_uploads = 0, g_gpu_fully_covered = 0, g_gpu_unreadable = 0;
u64 g_unreadable = 0;
u64 g_differs_by_writer[16]{};
u64 g_differs_explained[8]{};
std::map<std::string, u64> g_differs_by_owner; ///< "kind usage cpu_access" of the engine resource that owns the memory, for the differing uploads ///< differing uploads by the writer of a later mark over the image (0: no mark: unexplained)
u64 g_equal_bytes_by_writer[16]{};
u64 g_equal_nonzero_by_writer[16]{};  ///< bytes of CPU sourced uploads that equal the store, by the writer of the entry (15: a copy)
u64 g_uploads_by_main_writer[16]{}; ///< uploads by the writer that supplies most of their covered bytes
std::vector<nlohmann::json> g_examples_differs, g_examples_hole;

// ---- F2.5 write ledger (shadow) -------------------------------------------------------------------------------------------------------------------------------------
/// Every announced write, whatever its writer and size, is remembered as a range and the record number it was announced at (no bytes). Non-overlapping ranges, each holding the newest stamp that
/// wrote it. It answers "was this memory written after the draw's record?": if not, the memory an upload reads is what the draw saw, and the upload needs no copy of the bytes.
struct LedgerRange {
    u64 end;
    u64 stamp;
    u32 writer; ///< the writer number of the announcement that made the stamp (M0 census: who wrote the memory a late read finds changed)
    u32 submit; ///< `g_submit_count` when it was announced (M1)
    bool rewrite; ///< M3: an earlier announced write covered some of this memory (so an older version of it existed); false: the first write the ledger saw
    u32 prev;     ///< M3: the writer of the write this one replaced (15: none)
    u32 prev_submit; ///< M3: `g_submit_count` when that write was announced
    u32 site{};      ///< `TLG_WRITER_SITES`: the call site of the announcement that made the stamp (SiteId; 0 unknown)
};
std::mutex g_ledger_mutex;
std::map<u64, LedgerRange> g_ledger;
constexpr size_t MaxLedger = 1u << 20;
/// M2: the writers whose bytes the store keeps when it serves reads (`TLG_FRAME_STORE=2`): the ones that write every frame (0 map, 1 skinning, 2 lights, 3 command stream copy, 4 recording stores, 7 instance group upload). Buffer creation (5) and texture init (6) are the loader threads' big one-time writes.
constexpr u32 FsServedWriters = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 7);
/// M1: the number of graphics submissions so far (`NoteSubmit`). A write announced while it is N belongs to the frame the (N+1)th submission carries: the GPU of that submission sees it, so it is not a late write for a record of that submission.
std::atomic<u32> g_submit_count{0};
std::atomic<u64> c_ledger_notes{0}, c_ledger_overflow{0};
u64 g_ledger_max_len = 0;
std::vector<std::pair<u64, u64>> g_fs_pending; ///< M1: ranges announced since the last submission (only filled with the store on; under g_ledger_mutex)

constexpr u32 MaxSites = 160;
thread_local u32 t_site = 0;
std::mutex g_site_mutex;
std::vector<std::string> g_site_names{"?"};

void LedgerNote(u64 begin, u64 end, u64 stamp, u32 writer) {
    const u32 site = t_site;
    std::scoped_lock lock{g_ledger_mutex};
    c_ledger_notes.fetch_add(1, std::memory_order_relaxed);
    if (FrameStoreOn() && (!FrameStoreServes() || ((1u << std::min<u32>(writer, 31u)) & FsServedWriters) != 0)) {
        g_fs_pending.emplace_back(begin, end - begin);
    }
    if (g_ledger.size() >= MaxLedger) {
        c_ledger_overflow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    struct Seg {
        u64 b, e, s;
        u32 w;
        u32 sub;
        bool rw;
        u32 prev;
        u32 prev_sub;
        u32 site;
    };
    const u32 submit_now = g_submit_count.load(std::memory_order_relaxed);
    std::vector<Seg> add;
    auto it = g_ledger.upper_bound(begin);
    if (it != g_ledger.begin() && std::prev(it)->second.end > begin) {
        --it;
    }
    u64 cursor = begin;
    while (it != g_ledger.end() && it->first < end) {
        const u64 ob = it->first, oe = it->second.end, os = it->second.stamp;
        const u32 ow = it->second.writer, osub = it->second.submit;
        const bool orw = it->second.rewrite;
        if (ob < begin) {
            add.push_back({ob, begin, os, ow, osub, orw, it->second.prev, it->second.prev_submit, it->second.site});
        }
        if (oe > end) {
            add.push_back({end, oe, os, ow, osub, orw, it->second.prev, it->second.prev_submit, it->second.site});
        }
        const u64 x = std::max(ob, begin), y = std::min(oe, end);
        if (cursor < x) {
            add.push_back({cursor, x, stamp, writer, submit_now, false, 15u, 0u, site});
        }
        add.push_back({x, y, std::max(os, stamp), stamp >= os ? writer : ow, std::max(osub, submit_now), true, stamp >= os ? ow : it->second.prev, stamp >= os ? osub : it->second.prev_submit, stamp >= os ? site : it->second.site});
        cursor = y;
        it = g_ledger.erase(it);
    }
    if (cursor < end) {
        add.push_back({cursor, end, stamp, writer, submit_now, false, 15u, 0u, site});
    }
    for (const Seg& g : add) {
        g_ledger.emplace(g.b, LedgerRange{g.e, g.s, g.w, g.sub, g.rw, g.prev, g.prev_sub, g.site});
        g_ledger_max_len = std::max(g_ledger_max_len, g.e - g.b);
    }
}

struct LedgerAnswer {
    u64 newest{}; ///< the newest stamp of any write that touched the range (0: none)
    u64 covered{}; ///< bytes of the range some write touched
    u32 newest_writer{}; ///< the writer number of the write with the newest stamp
    u32 newest_submit{}; ///< the newest `g_submit_count` any write of the range was announced at (M1)
    u32 newest_prev{15}; ///< M3: the writer the newest write replaced
    u32 newest_prev_submit{0}; ///< M3: when that write was announced
    u32 newest_site{};         ///< the call site id of the newest write
    u64 rewritten{}; ///< M3: bytes of the range some earlier write had covered before the newest one (the memory had an older version)
    u64 late_lo{~u64{0}}, late_hi{}; ///< span of the bytes whose newest write was announced at or after `min_submit` of the query (0: no query submission)
};
LedgerAnswer LedgerQuery(u64 begin, u64 end, u32 min_submit = 0) {
    LedgerAnswer a;
    std::scoped_lock lock{g_ledger_mutex};
    // the ranges do not overlap and are sorted by start, so only the one before the first start above `begin` can reach into the query (the old scan from
    // `begin - longest range` walked the whole ledger once a single long range existed: M0 asks for every buffer copy, not only for image uploads)
    auto it = g_ledger.upper_bound(begin);
    if (it != g_ledger.begin() && std::prev(it)->second.end > begin) {
        --it;
    }
    for (; it != g_ledger.end() && it->first < end; ++it) {
        if (it->second.end > begin) {
            a.newest_submit = std::max(a.newest_submit, it->second.submit);
            if (it->second.stamp >= a.newest) {
                a.newest = it->second.stamp;
                a.newest_writer = it->second.writer;
                a.newest_prev = it->second.prev;
                a.newest_prev_submit = it->second.prev_submit;
                a.newest_site = it->second.site;
            }
            const u64 n = std::min(end, it->second.end) - std::max(begin, it->first);
            a.covered += n;
            if (min_submit != 0 && it->second.submit >= min_submit) {
                a.late_lo = std::min(a.late_lo, std::max(begin, it->first));
                a.late_hi = std::max(a.late_hi, std::min(end, it->second.end));
            }
            if (it->second.rewrite) {
                a.rewritten += n;
            }
        }
    }
    return a;
}

/// How an upload's memory stands against the ledger at the draw's record S.
enum LedgerClass : u32 { LedgerNoSeq, LedgerOpenMap, LedgerNeverWritten, LedgerPartlyWrittenUnchanged, LedgerWrittenUnchanged, LedgerWrittenAfter, LedgerClassCount };
constexpr const char* LedgerNames[LedgerClassCount] = {"no_draw_record", "a_map_of_it_is_open", "never_written_by_anything_announced", "partly_written_none_after_the_draw",
                                                       "written_none_after_the_draw", "written_after_the_draw"};
u64 g_ledger_uploads[LedgerClassCount][4]{}; ///< [ledger class][store class: equal, empty_equal, differs, hole]
u64 g_ledger_bytes[LedgerClassCount][4]{};
std::vector<nlohmann::json> g_ledger_examples;

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

struct HashJob {
    u64 address;
    u64 count;
    u64 chunk;
    u32* out;
};
struct CompareJob {
    const void* a;
    const void* b;
    u64 size;
    u64 first_diff; ///< offset of the first differing byte, ~0 when equal
    u64 nonzero;    ///< CompareJob with b == nullptr: counts the non-zero bytes of a
};
} // namespace

bool HashChunksGuarded(u64 address, u64 count, u64 chunk_bytes, u32* out) {
    HashJob job{address, count, chunk_bytes, out};
    return RunGuarded(
        [](void* p) {
            const auto& j = *static_cast<HashJob*>(p);
            const u8* base = reinterpret_cast<const u8*>(j.address);
            for (u64 i = 0; i < j.count; ++i) {
                j.out[i] = static_cast<u32>(XXH3_64bits(base + i * j.chunk, j.chunk));
            }
        },
        &job);
}

bool HashGuarded64(u64 address, u64 size, u64& out) {
    struct Job {
        u64 address, size, out;
    } job{address, size, 0};
    const bool ok = RunGuarded([](void* p) { auto& j = *static_cast<Job*>(p); j.out = XXH64(reinterpret_cast<const void*>(j.address), j.size, 0); }, &job);
    out = job.out;
    return ok;
}

namespace {

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
                if (g_tomb.size() < MaxTombs) {
                    g_tomb.emplace(b, b + it->second.size);
                    g_tomb_max = std::max(g_tomb_max, it->second.size);
                }
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

bool LedgerOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_LEDGER"); // default on since 2026-11-03; =0 turns it off
        return e == nullptr || *e != '0';
    }();
    return on;
}

bool On() noexcept {
    return ShadowOn() || RingSwapOn() || LedgerOn();
}

void NoteMark(u32 writer, u64 address, u64 size) {
    if (address == 0 || size == 0) {
        return;
    }
    if (LedgerOn() && (writer == 3 || writer == 6) && !InRing(address, size)) { // writers 1, 2 and 4 are noted by Announce with their own stamp
        LedgerNote(address, address + size, Core::NativeDraw::NextSeq(), writer);
    }
    if (!ShadowOn()) {
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
        const char* e = std::getenv("TLG_F2_RING_WINDOWS"); // default on since 2026-11-03 (the whole-window Map announcement needs the ring windows); =0 turns it off
        return e == nullptr || *e != '0';
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

namespace {
std::atomic<u64> c_tmap_total{0}, c_tmap_in_window{0}, c_tmap_equal{0}, c_tmap_announced_smaller{0}, c_tmap_announced_larger{0}, c_tmap_smaller_bytes{0};
std::mutex g_tmap_mutex;
std::map<std::string, u64> g_tmap_shapes; ///< announced / window remaining / row pitch / depth pitch -> count
std::vector<nlohmann::json> g_tmap_examples;
} // namespace

u64 MapAnnounceBytes(u64 pointer, u64 resource_bytes) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_MAP_WHOLE_WINDOW"); // default on since 2026-11-03 (shadow play: 17,430 swap reads, 0 uncovered, 0 differing); =0 turns it off
        return e == nullptr || *e != '0';
    }();
    if (!on || pointer == 0) {
        return resource_bytes;
    }
    std::scoped_lock lock{g_windows_mutex};
    auto it = g_windows.upper_bound(pointer);
    if (it == g_windows.begin()) {
        return resource_bytes;
    }
    const auto before = std::prev(it);
    const u64 remaining = before->first + before->second.size > pointer ? before->first + before->second.size - pointer : 0;
    return std::max(resource_bytes, remaining);
}

void NoteTextureMap(u64 resource_id, u64 pointer, u64 announced_bytes, u32 row_pitch, u32 depth_pitch, u32 map_type) {
    if (!RingWindowsOn() || pointer == 0) {
        return;
    }
    c_tmap_total.fetch_add(1, std::memory_order_relaxed);
    u64 remaining = 0, window_begin = 0, window_size = 0;
    {
        std::scoped_lock lock{g_windows_mutex};
        auto it = g_windows.upper_bound(pointer);
        if (it != g_windows.begin()) {
            const auto before = std::prev(it);
            if (before->first + before->second.size > pointer) {
                window_begin = before->first;
                window_size = before->second.size;
                remaining = before->first + before->second.size - pointer;
            }
        }
    }
    if (remaining == 0) {
        return;
    }
    c_tmap_in_window.fetch_add(1, std::memory_order_relaxed);
    if (announced_bytes == remaining) {
        c_tmap_equal.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (announced_bytes < remaining) {
        c_tmap_announced_smaller.fetch_add(1, std::memory_order_relaxed);
        c_tmap_smaller_bytes.fetch_add(remaining - announced_bytes, std::memory_order_relaxed);
    } else {
        c_tmap_announced_larger.fetch_add(1, std::memory_order_relaxed);
    }
    const std::string key = fmt::format("announced {} window_remaining {} row_pitch {} depth_pitch {} type {}", announced_bytes, remaining, row_pitch, depth_pitch, map_type);
    std::scoped_lock lock{g_tmap_mutex};
    if (g_tmap_shapes.size() < 48 || g_tmap_shapes.count(key) != 0) {
        ++g_tmap_shapes[key];
    }
    if (g_tmap_examples.size() < 8) {
        g_tmap_examples.push_back({{"resource_id", resource_id}, {"pointer", fmt::format("{:#x}", pointer)}, {"window_begin", fmt::format("{:#x}", window_begin)}, {"window_size", window_size},
                                   {"announced_bytes", announced_bytes}, {"window_remaining", remaining}, {"row_pitch", row_pitch}, {"depth_pitch", depth_pitch}});
    }
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

namespace {
std::mutex g_done_mutex;
std::condition_variable g_done_changed;
std::set<u32> g_done_pending; // done submissions above the contiguous floor
std::atomic<u32> g_done_floor{0};
std::atomic<u64> c_done_wait_timeouts{0};
} // namespace

u32 SubmitCount() {
    return g_submit_count.load(std::memory_order_acquire);
}

void NoteRasterSubmissionDone(u32 index) {
    {
        std::scoped_lock lock{g_done_mutex};
        g_done_pending.insert(index);
        u32 floor = g_done_floor.load(std::memory_order_relaxed);
        while (!g_done_pending.empty() && *g_done_pending.begin() <= floor + 1) {
            floor = std::max(floor, *g_done_pending.begin());
            g_done_pending.erase(g_done_pending.begin());
        }
        g_done_floor.store(floor, std::memory_order_release);
    }
    g_done_changed.notify_all();
}

u32 RasterDoneSubmissions() {
    return g_done_floor.load(std::memory_order_acquire);
}

void WaitRasterSubmissions(u32 target) {
    std::unique_lock lock{g_done_mutex};
    if (!g_done_changed.wait_for(lock, std::chrono::seconds(2), [target] { return g_done_floor.load(std::memory_order_acquire) >= target; })) {
        c_done_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
    }
}

u64 GateWaitTimeouts() {
    return c_done_wait_timeouts.load(std::memory_order_relaxed);
}

u64 GpuAllocationSize(u64 address) {
    std::scoped_lock lock{g_windows_mutex};
    const auto it = g_gpu_windows.find(address);
    return it != g_gpu_windows.end() ? it->second.size : 0;
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

u32 SiteId(const char* name) {
    if (name == nullptr) {
        return 0;
    }
    std::scoped_lock lock{g_site_mutex};
    for (u32 i = 0; i < g_site_names.size(); ++i) {
        if (g_site_names[i] == name) {
            return i;
        }
    }
    if (g_site_names.size() >= MaxSites - 1) {
        return 0;
    }
    g_site_names.emplace_back(name);
    return static_cast<u32>(g_site_names.size() - 1);
}

void SetSite(u32 site) {
    t_site = site;
}

namespace {
/// TLG_REWRITE_GAP=1 (measurement): per writer (1 skinning, 2 lights, 4 scene slots), how many graphics submissions lie between two writes that start at the same address
std::mutex g_gap_mutex;
std::unordered_map<u64, u32> g_gap_last[8];
std::mutex g_heap_free_mutex;
std::unordered_map<u64, u32> g_heap_free_at; // engine memory frees (Gomalloc): address -> submit count
std::map<u64, std::pair<u64, u32>> g_heap_free_blocks; ///< freed GPU allocations: start -> (size, submit count), to find the block that contains a read address
std::atomic<u64> c_gap[8][18]{};
} // namespace

void Announce(u32 writer, u64 address, u64 size, u64 stamp) {
    if (!On() || address == 0 || size == 0) {
        return;
    }
    {
        static const bool gap_on = [] { const char* e = std::getenv("TLG_REWRITE_GAP"); return e != nullptr && e[0] == '1'; }();
        if (gap_on && (writer == 1 || writer == 2 || writer == 4)) {
            const u32 now = g_submit_count.load(std::memory_order_relaxed);
            std::scoped_lock gap_lock{g_gap_mutex};
            auto& last = g_gap_last[writer];
            if (last.size() > 400000) {
                last.clear();
            }
            const auto [it, inserted] = last.try_emplace(address, now);
            if (inserted) {
                c_gap[writer][17].fetch_add(1, std::memory_order_relaxed); // first write seen
            } else {
                c_gap[writer][std::min<u32>(now - it->second, 16u)].fetch_add(1, std::memory_order_relaxed);
                it->second = now;
            }
        }
    }
    if (LedgerOn() && !InRing(address, size)) { // the frame ring is the ring swap's (stamped bytes); the ledger is for the memory outside it
        LedgerNote(address, address + size, stamp, writer);
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

namespace {
/// F2.4 research (ring swap differences): why does the store's copy of a ring image differ from the bytes memory holds now? Per differing byte: does a NEWER entry (announced after the draw's
/// record, stamp above it) hold the byte memory has (the raster thread drew late, memory was overwritten by a later frame: the store is right), or no entry does (an unannounced write, or
/// a writer whose bytes are not kept: the store may be wrong).
std::atomic<u64> c_dif_images{0}, c_dif_images_all_newer{0}, c_dif_images_unexplained{0}, c_dif_images_unexplained_mark{0};
std::atomic<u64> c_dif_bytes_newer{0}, c_dif_bytes_unexplained{0}, c_dif_images_with_uncovered{0}, c_dif_images_uncovered_only{0};
std::atomic<u64> c_dif_writer[16]{}; ///< differing bytes by the writer of the store's (older) entry
std::mutex g_dif_mutex;
std::vector<nlohmann::json> g_dif_examples;
/// The bytes of a swapped image that the store has no entry for (left to memory), by what is known about them: a newer entry (announced after the draw's record), a mark of a writer whose
/// bytes are not kept (by writer), or nothing at all (never announced: zero bytes or not).
enum UncClass : u32 { UncNewerEntry, UncMark, UncNothingZero, UncNothingNonZero, UncClassCount };
constexpr const char* UncNames[UncClassCount] = {"a_newer_entry_overlaps", "a_mark_overlaps", "nothing_known_memory_all_zero", "nothing_known_memory_not_zero"};
std::atomic<u64> c_unc_segments[UncClassCount]{}, c_unc_bytes[UncClassCount]{}, c_unc_mark_writer[8]{}, c_unc_images{0}, c_unc_images_whole{0};
std::vector<nlohmann::json> g_unc_examples;
} // namespace

namespace {
std::mutex g_shape_mutex;
std::map<std::string, u64> g_shapes;
} // namespace

void NoteRingImageShape(u64 address, u64 size, u32 width, u32 height, u32 depth, u32 format, u32 mips, u32 layers, u32 pitch, u32 tile_mode) {
    if (!RingSwapOn() || !InRing(address, size)) {
        return;
    }
    const std::string key = fmt::format("size {} : {}x{}x{} format {} mips {} layers {} pitch {} tile {}", size, width, height, depth, format, mips, layers, pitch, tile_mode);
    std::scoped_lock lock{g_shape_mutex};
    if (g_shapes.size() < 64 || g_shapes.count(key) != 0) {
        ++g_shapes[key];
    }
}

// ---- M1: submissions and the frame data store (shadow) ------------------------------------------------------------------------------------------------------------------
namespace {
struct SubmitMark {
    u32 index;
    u64 stamp; ///< `NativeDraw::NextSeq()` at the submission: every record below it was made for this submission or an earlier one
};
std::mutex g_submit_mutex;
std::deque<SubmitMark> g_submits; ///< the last 512 graphics submissions

struct FsChunk {
    u64 address;
    u64 size;
    std::unique_ptr<u8[]> bytes;
};
struct FsFrame {
    u32 index;
    std::vector<FsChunk> chunks; ///< sorted by address
    u64 bytes{0};
};
std::mutex g_fs_mutex;
std::deque<FsFrame> g_fs_frames; ///< the last FsKeep frames
constexpr size_t FsKeep = 8;
/// M3: the last version of a range that fell out of the window of FsKeep submissions and that no newer version in the window covers. Ranges the game rewrites rarely (a map written once and
/// read for minutes) would otherwise leave a hole as soon as the game rewrites them while a late reader still wants the old bytes. Non-overlapping, keyed by start; a newer retired version
/// trims the older entries it overlaps. Under g_fs_mutex.
struct FsBaseChunk {
    u64 end;
    u32 index;                   ///< the submission whose capture the bytes are
    std::shared_ptr<u8[]> owner; ///< the buffer of the captured chunk (pieces of it share it)
    const u8* data;              ///< the byte for the entry's start
};
std::map<u64, FsBaseChunk> g_fs_base;
u64 g_fs_base_bytes = 0;
constexpr u64 FsBaseCap = 768ull << 20; ///< a retired piece that would go over this is counted and dropped
constexpr u64 FsMaxChunk = 16ull << 20;  ///< a bigger announcement is counted and not copied
constexpr u64 FsMaxFrame = 128ull << 20; ///< per submission
std::atomic<u64> c_fs_submits{0}, c_fs_chunks{0}, c_fs_bytes{0}, c_fs_skipped_big{0}, c_fs_skipped_unreadable{0}, c_fs_skipped_frame_full{0}, c_fs_capture_ns{0}, c_fs_max_frame_bytes{0};
std::atomic<u64> c_fs_live_bytes{0};
std::atomic<u64> c_pre_calls{0}, c_pre_never_written{0}, c_pre_written_this_period{0}, c_pre_have_version{0}, c_pre_unreadable{0}, c_pre_captured{0}, c_pre_bytes{0}, c_pre_loader_bytes{0};
std::atomic<u64> c_fs_base_bytes{0}, c_fs_base_entries{0}, c_fs_base_added_bytes{0}, c_fs_base_trimmed_bytes{0}, c_fs_base_dropped_cap_bytes{0};

u32 SubmitIndexOf(u64 seq); // below
void FsRetireLocked(FsFrame&& frame); // below

/// M5 follow-up (`TLG_FS_RECHECK=1`): is a captured version ever wrong? A writer that announces before it writes (1 skinning, 2 lights, 4 recording stores) can have its range captured at the
/// submission while the write is still going on: the version is then the old bytes and nothing recaptures it. Two submissions later every chunk of the frame is compared with the memory: a
/// difference that no later announcement explains (the ledger's newest write for the range is not newer than the chunk) is such a case, or a write nobody announced.
bool FsRecheckOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_FS_RECHECK");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}
std::atomic<u64> c_recheck_chunks{0}, c_recheck_bytes{0}, c_recheck_changed_chunks{0}, c_recheck_explained{0}, c_recheck_unexplained_chunks{0}, c_recheck_unexplained_bytes{0};
std::atomic<u64> c_recheck_unexplained_by_writer[16]{}, c_recheck_unexplained_bytes_by_writer[16]{}, c_recheck_noted[16]{};
void FsRecheckLocked(const FsFrame& frame); // below

struct FsKindCounters {
    std::atomic<u64> reads{0}, reads_overlapping{0}, reads_with_differences{0}, bytes_covered{0}, bytes_equal{0}, bytes_differ{0}, unreadable{0};
};
FsKindCounters c_fs_kind[static_cast<size_t>(LateKind::Count)];
struct FsServeCounters {
    std::atomic<u64> reads{0}, replaced_reads{0}, replaced_bytes{0}, hole_reads{0}, hole_bytes{0}, loader_newest_reads{0};
    std::atomic<u64> hole_pieces_by_writer[16]{}, hole_bytes_by_writer[16]{};
    std::atomic<u64> from_base_bytes{0}, from_base_replaced_bytes{0}; ///< M3: bytes served from the retired layer / of those the ones that really changed
    std::atomic<u64> first_write_reads{0}, first_write_bytes{0};      ///< M3: bytes written after the record for the first time (no older version ever existed: nothing to keep)
    std::atomic<u64> first_write_bytes_by_writer[16]{};
    std::atomic<u64> born_after_bytes{0}, born_after_reads{0}; ///< M3: bytes whose replaced write was itself after the record: the memory did not hold this resource's data at the record
    std::atomic<u64> hole_bytes_by_prev[16]{}; ///< M3: bytes of the (rewrite) holes by the writer whose version was replaced (15: none known)
    std::atomic<u64> loader_skip_pieces{0}, loader_skip_bytes{0};     ///< M3: version pieces not served because a loader write (5, 6) came after the version and before the record
};
FsServeCounters c_fs_serve[static_cast<size_t>(LateKind::Count)];
} // namespace

int FrameStoreMode() noexcept {
    static const int mode = [] {
        const char* e = std::getenv("TLG_FRAME_STORE");
        return e == nullptr || *e == '\0' ? 0 : std::clamp(std::atoi(e), 0, 2);
    }();
    return mode;
}

bool FrameStoreOn() noexcept {
    return FrameStoreMode() >= 1;
}

bool FrameStoreServes() noexcept {
    return FrameStoreMode() >= 2;
}

u32 CurrentRecordSubmit() {
    const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
    return seq != 0 ? SubmitIndexOf(seq) : 0;
}

namespace {
/// The index (1 based) of the submission that carries the record `seq`: the first whose stamp is above it. 0: unknown (no submission yet, or older than the 512 kept).
u32 SubmitIndexOf(u64 seq) {
    std::scoped_lock lock{g_submit_mutex};
    if (g_submits.empty() || seq >= g_submits.back().stamp) {
        return 0;
    }
    const auto it = std::upper_bound(g_submits.begin(), g_submits.end(), seq, [](u64 value, const SubmitMark& m) { return value < m.stamp; });
    return it == g_submits.end() ? 0 : it->index;
}
} // namespace

namespace {
std::atomic<u64> g_raster_seq{0}; ///< the record the raster thread started last
} // namespace

void NoteRasterRecord(u64 seq) {
    g_raster_seq.store(seq, std::memory_order_relaxed);
}

u64 LastSubmitStamp() {
    std::scoped_lock lock{g_submit_mutex};
    return g_submits.empty() ? 0 : g_submits.back().stamp;
}

void SilentHarvest(u32 index); // W1.5 census, below

void NoteSubmit(const std::vector<std::pair<u64, u64>>& ranges) {
    const u32 index = g_submit_count.load(std::memory_order_relaxed) + 1;
    Core::ByteAudit::SnapshotAtSubmit(index);
    SilentHarvest(index);
    if (FrameStoreOn()) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::pair<u64, u64>> all;
        {
            std::scoped_lock lock{g_ledger_mutex};
            all.swap(g_fs_pending);
        }
        all.insert(all.end(), ranges.begin(), ranges.end());
        FsFrame frame;
        frame.index = index;
        frame.chunks.reserve(all.size());
        for (const auto& [address, size] : all) {
            if (address == 0 || size == 0) {
                continue;
            }
            if (size > FsMaxChunk) {
                c_fs_skipped_big.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (frame.bytes + size > FsMaxFrame) {
                c_fs_skipped_frame_full.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            auto bytes = std::make_unique<u8[]>(size);
            if (!GuardedCopy(bytes.get(), reinterpret_cast<const void*>(address), size)) {
                c_fs_skipped_unreadable.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            frame.bytes += size;
            frame.chunks.push_back({address, size, std::move(bytes)});
        }
        std::sort(frame.chunks.begin(), frame.chunks.end(), [](const FsChunk& a, const FsChunk& b) { return a.address < b.address; });
        c_fs_chunks.fetch_add(frame.chunks.size(), std::memory_order_relaxed);
        c_fs_bytes.fetch_add(frame.bytes, std::memory_order_relaxed);
        for (u64 seen = c_fs_max_frame_bytes.load(std::memory_order_relaxed); frame.bytes > seen && !c_fs_max_frame_bytes.compare_exchange_weak(seen, frame.bytes, std::memory_order_relaxed);) {
        }
        {
            std::scoped_lock lock{g_fs_mutex};
            c_fs_live_bytes.fetch_add(frame.bytes, std::memory_order_relaxed);
            if (FsRecheckOn() && g_fs_frames.size() >= 2) {
                FsRecheckLocked(g_fs_frames[g_fs_frames.size() - 2]);
            }
            g_fs_frames.push_back(std::move(frame));
            while (g_fs_frames.size() > FsKeep) {
                c_fs_live_bytes.fetch_sub(g_fs_frames.front().bytes, std::memory_order_relaxed);
                FsFrame old = std::move(g_fs_frames.front());
                g_fs_frames.pop_front();
                if (FrameStoreServes()) {
                    FsRetireLocked(std::move(old)); // M3
                }
            }
        }
        c_fs_capture_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    }
    c_fs_submits.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{g_submit_mutex};
        g_submits.push_back({index, Core::NativeDraw::NextSeq()});
        if (g_submits.size() > 512) {
            g_submits.pop_front();
        }
    }
    g_submit_count.store(index, std::memory_order_release); // after the capture: writes announced from here on belong to later submissions
}

namespace {
struct Cover {
    u64 b, e;
    const u8* src; ///< the store's bytes for b (valid while g_fs_mutex is held)
    u32 index;     ///< the submission whose capture holds the bytes
    bool base;     ///< from the retired layer
};
using Pieces = std::vector<std::pair<u64, u64>>;

/// The pieces of [address, address + size) some version at or before `submit` covers, the newest version per byte (`out`), and the covered pieces sorted (`done`). The window first, newest
/// capture first, then (`with_base`) the retired layer. g_fs_mutex held by the caller.
void FsCoverLocked(u64 address, u64 size, u32 submit, std::vector<Cover>& out, Pieces& done, bool with_base = true) {
    const u64 end = address + size;
    const u64 lo = address > FsMaxChunk ? address - FsMaxChunk : 0; // a chunk is at most FsMaxChunk long, so none that starts before `lo` reaches the range
    thread_local Pieces pieces;
    // the part of [b, e) no newer version already decided goes to `out` (bytes for b: src), and into `done`
    const auto take = [&](u64 b, u64 e, const u8* src, u32 index, bool base) {
        pieces.clear();
        u64 cursor = b;
        for (const auto& d : done) { // what a newer version already decided is taken out
            if (d.second <= cursor) {
                continue;
            }
            if (d.first >= e) {
                break;
            }
            if (d.first > cursor) {
                pieces.emplace_back(cursor, d.first);
            }
            cursor = std::max(cursor, d.second);
        }
        if (cursor < e) {
            pieces.emplace_back(cursor, e);
        }
        for (const auto& p : pieces) {
            out.push_back({p.first, p.second, src + (p.first - b), index, base});
            done.insert(std::lower_bound(done.begin(), done.end(), p), p);
        }
    };
    // the retired layer's entries that reach the range, newest capture first: a pre-write capture (NoteMapOpen) can be newer than a frame of the window
    thread_local std::vector<std::pair<u32, std::map<u64, FsBaseChunk>::const_iterator>> base_hits;
    base_hits.clear();
    if (with_base) {
        auto it = g_fs_base.upper_bound(address);
        if (it != g_fs_base.begin() && std::prev(it)->second.end > address) {
            --it;
        }
        for (; it != g_fs_base.end() && it->first < end; ++it) {
            if (it->second.index <= submit && std::max(address, it->first) < std::min(end, it->second.end)) {
                base_hits.emplace_back(it->second.index, it);
            }
        }
        std::sort(base_hits.begin(), base_hits.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    }
    size_t next_base = 0;
    const auto take_base = [&](u32 above) { // the entries whose capture is newer than `above`
        for (; next_base < base_hits.size() && base_hits[next_base].first > above; ++next_base) {
            const auto it = base_hits[next_base].second;
            const u64 b = std::max(address, it->first), e = std::min(end, it->second.end);
            take(b, e, it->second.data + (b - it->first), it->second.index, true);
        }
    };
    for (auto frame = g_fs_frames.rbegin(); frame != g_fs_frames.rend(); ++frame) {
        if (frame->index > submit) {
            continue; // a later submission's capture holds writes the record's submission did not see
        }
        take_base(frame->index);
        auto it = std::lower_bound(frame->chunks.begin(), frame->chunks.end(), lo, [](const FsChunk& c, u64 value) { return c.address < value; });
        for (; it != frame->chunks.end() && it->address < end; ++it) {
            const u64 b = std::max(address, it->address), e = std::min(end, it->address + it->size);
            if (b < e) {
                take(b, e, it->bytes.get() + (b - it->address), frame->index, false);
            }
        }
    }
    take_base(0);
}

/// M3: `[b, e)` of a captured buffer becomes the retired layer's version (capture `index`) of that memory; the older retired versions of it go or shrink. g_fs_mutex held.
void FsBaseAddLocked(u64 b, u64 e, u32 index, const std::shared_ptr<u8[]>& owner, const u8* data) {
    if (b >= e) {
        return;
    }
    auto it = g_fs_base.upper_bound(b);
    if (it != g_fs_base.begin() && std::prev(it)->second.end > b) {
        --it;
    }
    while (it != g_fs_base.end() && it->first < e) {
        const u64 ob = it->first;
        FsBaseChunk old = std::move(it->second);
        it = g_fs_base.erase(it);
        const u64 x = std::max(ob, b), y = std::min(old.end, e);
        g_fs_base_bytes -= y - x;
        c_fs_base_trimmed_bytes.fetch_add(y - x, std::memory_order_relaxed);
        if (ob < b) {
            g_fs_base.emplace(ob, FsBaseChunk{b, old.index, old.owner, old.data});
        }
        if (old.end > e) {
            g_fs_base.emplace(e, FsBaseChunk{old.end, old.index, old.owner, old.data + (e - ob)});
        }
    }
    if (g_fs_base_bytes + (e - b) > FsBaseCap) {
        c_fs_base_dropped_cap_bytes.fetch_add(e - b, std::memory_order_relaxed);
        return;
    }
    g_fs_base.emplace(b, FsBaseChunk{e, index, owner, data});
    g_fs_base_bytes += e - b;
    c_fs_base_added_bytes.fetch_add(e - b, std::memory_order_relaxed);
}

void FsRecheckLocked(const FsFrame& frame) {
    for (const FsChunk& chunk : frame.chunks) {
        c_recheck_chunks.fetch_add(1, std::memory_order_relaxed);
        c_recheck_bytes.fetch_add(chunk.size, std::memory_order_relaxed);
        u64 first_diff = ~0ull;
        if (!GuardedCompare(reinterpret_cast<const void*>(chunk.address), chunk.bytes.get(), chunk.size, first_diff) || first_diff == ~0ull) {
            continue;
        }
        c_recheck_changed_chunks.fetch_add(1, std::memory_order_relaxed);
        const LedgerAnswer a = LedgerQuery(chunk.address, chunk.address + chunk.size);
        // an announcement made after this chunk's period explains the change (announced with `submit` = index - 1 of its capture, so >= frame.index is later)
        if (a.covered != 0 && a.newest_submit >= frame.index) {
            c_recheck_explained.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const u32 w = std::min<u32>(a.newest_writer, 15u);
        c_recheck_unexplained_chunks.fetch_add(1, std::memory_order_relaxed);
        c_recheck_unexplained_bytes.fetch_add(chunk.size - first_diff, std::memory_order_relaxed);
        c_recheck_unexplained_by_writer[w].fetch_add(1, std::memory_order_relaxed);
        c_recheck_unexplained_bytes_by_writer[w].fetch_add(chunk.size - first_diff, std::memory_order_relaxed);
        if (c_recheck_noted[w].fetch_add(1, std::memory_order_relaxed) < 3) {
            Core::RareEvents::Note(fmt::format("m5_captured_version_changed_unannounced_writer{}", w),
                                   fmt::format("{:#x} + {:#x}: captured in submission {} (first differing byte {}), ledger: newest write in submission {} (stamp {}), writer {}", chunk.address, chunk.size, frame.index,
                                               first_diff, a.newest_submit, a.newest, a.newest_writer));
        }
    }
}

/// M3: a frame fell out of the window: what of its chunks no newer version in the window covers becomes the retired layer's version of that memory. g_fs_mutex held.
void FsRetireLocked(FsFrame&& frame) {
    thread_local std::vector<Cover> covers;
    thread_local Pieces done;
    // the oldest submission a read can still ask for: the one that carries the record the raster thread is on (0: not started)
    const u64 raster_seq = g_raster_seq.load(std::memory_order_relaxed);
    u32 oldest_reader = 1;
    if (raster_seq != 0) {
        // a record newer than the last submission mark belongs to the submission that comes next
        oldest_reader = raster_seq >= LastSubmitStamp() ? g_submit_count.load(std::memory_order_relaxed) + 1 : std::max<u32>(SubmitIndexOf(raster_seq), 1u);
    }
    for (FsChunk& chunk : frame.chunks) {
        // most chunks are rewritten every frame: one newer chunk of the window holds all their bytes, nothing to keep. But only a newer chunk that every pending read picks over this one
        // counts: a read of an older submission (the raster thread is behind) must still get this version, even when the memory was rewritten and captured since (owner play 2026-11-04:
        // light blocks rewritten exactly one window later were holes, writer 2, 18 in 8 minutes)
        const u64 chunk_end = chunk.address + chunk.size;
        bool superseded = false;
        for (auto newer = g_fs_frames.rbegin(); newer != g_fs_frames.rend() && !superseded; ++newer) {
            if (newer->index > oldest_reader) {
                continue;
            }
            auto it = std::upper_bound(newer->chunks.begin(), newer->chunks.end(), chunk.address, [](u64 value, const FsChunk& c) { return value < c.address; });
            if (it != newer->chunks.begin()) {
                const FsChunk& c = *std::prev(it);
                superseded = c.address + c.size >= chunk_end; // c.address <= chunk.address by the search
            }
        }
        if (superseded) {
            continue;
        }
        covers.clear();
        done.clear();
        FsCoverLocked(chunk.address, chunk.size, oldest_reader, covers, done, false); // only versions every pending read picks over this one hide it
        std::shared_ptr<u8[]> owner{std::move(chunk.bytes)};
        const auto add = [&](u64 b, u64 e) { FsBaseAddLocked(b, e, frame.index, owner, owner.get() + (b - chunk.address)); };
        u64 cursor = chunk.address;
        const u64 end = chunk.address + chunk.size;
        for (const auto& d : done) {
            add(cursor, d.first);
            cursor = std::max(cursor, d.second);
        }
        add(cursor, end);
    }
    c_fs_base_bytes.store(g_fs_base_bytes, std::memory_order_relaxed);
    c_fs_base_entries.store(g_fs_base.size(), std::memory_order_relaxed);
}

/// One read of the game's memory for a record of submission `submit`: what the store holds for [address, address + size) as of that submission (the newest version at or before it, per
/// byte) against what the memory holds now. Only the bytes some version covers are compared.
void FrameStoreCompare(size_t kind, u64 address, u64 size, u32 submit) {
    FsKindCounters& c = c_fs_kind[kind];
    c.reads.fetch_add(1, std::memory_order_relaxed);
    thread_local std::vector<Cover> covers;
    thread_local Pieces done;
    covers.clear();
    done.clear();
    u64 covered = 0, equal = 0, differ = 0;
    bool unreadable = false;
    {
        std::scoped_lock lock{g_fs_mutex};
        FsCoverLocked(address, size, submit, covers, done);
        for (const Cover& cv : covers) {
            u64 first_diff = ~0ull;
            if (!GuardedCompare(reinterpret_cast<const void*>(cv.b), cv.src, cv.e - cv.b, first_diff)) {
                unreadable = true;
                continue;
            }
            covered += cv.e - cv.b;
            if (first_diff == ~0ull) {
                equal += cv.e - cv.b;
            } else {
                // the piece counts as different from its first differing byte on (an upper bound)
                differ += cv.e - cv.b - first_diff;
                equal += first_diff;
            }
        }
    }
    if (!covers.empty()) {
        c.reads_overlapping.fetch_add(1, std::memory_order_relaxed);
    }
    if (differ != 0) {
        c.reads_with_differences.fetch_add(1, std::memory_order_relaxed);
    }
    if (unreadable) {
        c.unreadable.fetch_add(1, std::memory_order_relaxed);
    }
    c.bytes_covered.fetch_add(covered, std::memory_order_relaxed);
    c.bytes_equal.fetch_add(equal, std::memory_order_relaxed);
    c.bytes_differ.fetch_add(differ, std::memory_order_relaxed);
}


} // namespace

void NoteMapOpen(u64 address, u64 size) {
    if (!FrameStoreServes() || address == 0 || size == 0 || size > FsMaxChunk || InRing(address, size)) {
        return;
    }
    c_pre_calls.fetch_add(1, std::memory_order_relaxed);
    const u32 now = g_submit_count.load(std::memory_order_acquire);
    // the ledger's writes inside the range: (begin, end, submit count at the announcement, writer)
    struct Segment {
        u64 b, e;
        u32 submit, writer;
    };
    thread_local std::vector<Segment> segments;
    segments.clear();
    {
        std::scoped_lock lock{g_ledger_mutex};
        auto it = g_ledger.upper_bound(address);
        if (it != g_ledger.begin() && std::prev(it)->second.end > address) {
            --it;
        }
        for (; it != g_ledger.end() && it->first < address + size; ++it) {
            if (it->second.end > address) {
                segments.push_back({std::max(address, it->first), std::min(address + size, it->second.end), it->second.submit, it->second.writer});
            }
        }
    }
    if (segments.empty()) {
        c_pre_never_written.fetch_add(1, std::memory_order_relaxed); // the first write: no older version exists
        return;
    }
    thread_local std::vector<Cover> covers;
    thread_local Pieces done;
    covers.clear();
    done.clear();
    {
        std::scoped_lock lock{g_fs_mutex};
        FsCoverLocked(address, size, now, covers, done);
    }
    for (const Segment& seg : segments) {
        if (seg.submit >= now) {
            c_pre_written_this_period.fetch_add(1, std::memory_order_relaxed); // written in the period now running: the capture at the next submission takes it
            continue;
        }
        // the memory holds the writes up to period `seg.submit + 1`, which is capture `index`: a version of that capture or a newer one is already there, or this memory is that version
        const u32 index = seg.submit + 1;
        Pieces missing;
        u64 cursor = seg.b;
        Pieces current; // the pieces of versions at least that new, sorted
        for (const Cover& cv : covers) {
            if (cv.index >= index && cv.b < seg.e && cv.e > seg.b) {
                current.emplace_back(std::max(cv.b, seg.b), std::min(cv.e, seg.e));
            }
        }
        std::sort(current.begin(), current.end());
        for (const auto& d : current) {
            if (d.first > cursor) {
                missing.emplace_back(cursor, d.first);
            }
            cursor = std::max(cursor, d.second);
        }
        if (cursor < seg.e) {
            missing.emplace_back(cursor, seg.e);
        }
        if (missing.empty()) {
            c_pre_have_version.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        for (const auto& [b, e] : missing) {
            std::shared_ptr<u8[]> bytes{new u8[e - b]};
            if (!GuardedCopy(bytes.get(), reinterpret_cast<const void*>(b), e - b)) {
                c_pre_unreadable.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            {
                std::scoped_lock lock{g_fs_mutex};
                FsBaseAddLocked(b, e, index, bytes, bytes.get());
            }
            c_pre_captured.fetch_add(1, std::memory_order_relaxed);
            c_pre_bytes.fetch_add(e - b, std::memory_order_relaxed);
            if (seg.writer == 5 || seg.writer == 6) {
                c_pre_loader_bytes.fetch_add(e - b, std::memory_order_relaxed);
            }
        }
    }
}

u64 FrameStoreServe(LateKind kind, u64 address, void* dst, u64 size) {
    if (!FrameStoreServes() || size == 0 || dst == nullptr || !LedgerOn()) {
        return 0;
    }
    const size_t k = static_cast<size_t>(kind);
    const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
    const u32 submit = seq != 0 ? SubmitIndexOf(seq) : 0;
    if (submit == 0 || InRing(address, size)) {
        return 0;
    }
    FsServeCounters& c = c_fs_serve[k];
    c.reads.fetch_add(1, std::memory_order_relaxed);
    thread_local std::vector<Cover> covers;
    thread_local Pieces done;
    covers.clear();
    done.clear();
    u64 replaced = 0, from_base = 0, from_base_replaced = 0;
    {
        std::scoped_lock lock{g_fs_mutex};
        FsCoverLocked(address, size, submit, covers, done);
        auto* out = static_cast<u8*>(dst);
        for (const Cover& cv : covers) {
            // dst is only written, never read: it may be video memory the CPU maps write-combined (the zero copy upload), where a read is very slow. What changed is counted against the game's memory.
            u8* at = out + (cv.b - address);
            const u64 n = cv.e - cv.b;
            // M3: a loader write (5 buffer creation, 6 texture init: never captured) came after this version and before the record: the memory is newer than the version and right for the record
            if (cv.base || cv.index + 2 < submit) {
                const LedgerAnswer a = LedgerQuery(cv.b, cv.e);
                if ((a.newest_writer == 5 || a.newest_writer == 6) && a.covered != 0 && a.newest_submit >= cv.index && a.newest_submit < submit) {
                    c.loader_skip_pieces.fetch_add(1, std::memory_order_relaxed);
                    c.loader_skip_bytes.fetch_add(n, std::memory_order_relaxed);
                    continue;
                }
            }
            u64 first_diff = ~0ull;
            if (GuardedCompare(reinterpret_cast<const void*>(cv.b), cv.src, n, first_diff) && first_diff != ~0ull) {
                replaced += n - first_diff; // from the first differing byte on (an upper bound)
                if (cv.base) {
                    from_base_replaced += n - first_diff;
                }
            }
            if (cv.base) {
                from_base += n;
            }
            std::memcpy(at, cv.src, n);
        }
    }
    if (from_base != 0) {
        c.from_base_bytes.fetch_add(from_base, std::memory_order_relaxed);
        c.from_base_replaced_bytes.fetch_add(from_base_replaced, std::memory_order_relaxed);
    }
    if (replaced != 0) {
        c.replaced_reads.fetch_add(1, std::memory_order_relaxed);
        c.replaced_bytes.fetch_add(replaced, std::memory_order_relaxed);
    }
    // what no version covers is the memory as it is now: right unless the game wrote it at or after the record's submission (a hole: the old bytes are gone)
    u64 hole = 0, first_write = 0, born_after = 0;
    bool loader_newest = false;
    u64 cursor = address;
    const u64 end = address + size;
    auto check = [&](u64 b, u64 e) {
        if (b >= e) {
            return;
        }
        const LedgerAnswer a = LedgerQuery(b, e);
        if (a.covered != 0 && a.newest_submit >= submit) {
            const u32 w = std::min<u32>(a.newest_writer, 15u);
            // M3: a write that is the first the ledger saw of this memory replaced no older version: the record cannot have used what was there before (nothing, or memory from before the ledger)
            u64 rewrite = std::min(a.rewritten, e - b);
            if (rewrite != 0 && a.newest_prev_submit >= submit) {
                // the write this one replaced came after the record too: the record saw something older still (or the memory of a resource made after it)
                born_after += rewrite;
                rewrite = 0;
            }
            const u64 fresh = e - b - rewrite;
            if (fresh != 0) {
                first_write += fresh;
                c.first_write_bytes_by_writer[w].fetch_add(fresh, std::memory_order_relaxed);
            }
            if (rewrite != 0) {
                hole += rewrite;
                c.hole_pieces_by_writer[w].fetch_add(1, std::memory_order_relaxed);
                c.hole_bytes_by_writer[w].fetch_add(rewrite, std::memory_order_relaxed);
                c.hole_bytes_by_prev[std::min<u32>(a.newest_prev, 15u)].fetch_add(rewrite, std::memory_order_relaxed);
                if (w != 5 && w != 6) {
                    // M5 follow-up: the dark frame at depth 2 is searched in the holes of the per-frame writers (the lights, writer 2, in the stream copy); the first three of each kind are written out
                    Core::RareEvents::Note(fmt::format("m3_store_hole_writer{}_kind{}", w, k),
                                           fmt::format("{:#x} + {:#x}: record {} of submission {}, newest write in submission {} (stamp {}), it replaced a write of writer {} from submission {}, now submission {}", b, e - b, seq, submit,
                                                       a.newest_submit, a.newest, a.newest_prev, a.newest_prev_submit, g_submit_count.load(std::memory_order_relaxed)));
                }
            }
        }
    };
    for (const auto& d : done) {
        check(cursor, d.first);
        cursor = std::max(cursor, d.second);
    }
    check(cursor, end);
    if (hole != 0) {
        c.hole_reads.fetch_add(1, std::memory_order_relaxed);
        c.hole_bytes.fetch_add(hole, std::memory_order_relaxed);
    }
    if (born_after != 0) {
        c.born_after_reads.fetch_add(1, std::memory_order_relaxed);
        c.born_after_bytes.fetch_add(born_after, std::memory_order_relaxed);
    }
    if (first_write != 0) {
        c.first_write_reads.fetch_add(1, std::memory_order_relaxed);
        c.first_write_bytes.fetch_add(first_write, std::memory_order_relaxed);
    }
    if (!done.empty()) {
        const LedgerAnswer all = LedgerQuery(address, end);
        loader_newest = all.newest_writer == 5 || all.newest_writer == 6;
        if (loader_newest) {
            c.loader_newest_reads.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return replaced;
}

namespace {
nlohmann::json FrameStoreReport(size_t kind) {
    const FsKindCounters& c = c_fs_kind[kind];
    return {{"reads", c.reads.load()},
            {"reads_overlapping_a_version", c.reads_overlapping.load()},
            {"reads_with_bytes_that_differ_from_memory", c.reads_with_differences.load()},
            {"bytes_covered_by_versions", c.bytes_covered.load()},
            {"bytes_equal", c.bytes_equal.load()},
            {"bytes_differ_upper_bound", c.bytes_differ.load()},
            {"unreadable", c.unreadable.load()},
            {"served_reads", c_fs_serve[kind].reads.load()},
            {"served_reads_with_bytes_replaced", c_fs_serve[kind].replaced_reads.load()},
            {"served_bytes_replaced", c_fs_serve[kind].replaced_bytes.load()},
            {"served_reads_with_a_hole", c_fs_serve[kind].hole_reads.load()},
            {"served_hole_bytes", c_fs_serve[kind].hole_bytes.load()},
            {"served_bytes_from_the_retired_layer", c_fs_serve[kind].from_base_bytes.load()},
            {"served_bytes_from_the_retired_layer_that_really_changed", c_fs_serve[kind].from_base_replaced_bytes.load()},
            {"reads_with_first_writes_after_the_record", c_fs_serve[kind].first_write_reads.load()},
            {"reads_where_the_replaced_write_was_after_the_record_too", c_fs_serve[kind].born_after_reads.load()},
            {"bytes_where_the_replaced_write_was_after_the_record_too", c_fs_serve[kind].born_after_bytes.load()},
            {"first_write_bytes_after_the_record", c_fs_serve[kind].first_write_bytes.load()},
            {"first_write_bytes_by_writer", [&] {
                 nlohmann::json j = nlohmann::json::object();
                 for (u32 w = 0; w < 16; ++w) {
                     if (const u64 n = c_fs_serve[kind].first_write_bytes_by_writer[w].load()) {
                         j[std::to_string(w)] = n;
                     }
                 }
                 return j;
             }()},
            {"hole_bytes_by_the_writer_replaced", [&] {
                 nlohmann::json j = nlohmann::json::object();
                 for (u32 w = 0; w < 16; ++w) {
                     if (const u64 n = c_fs_serve[kind].hole_bytes_by_prev[w].load()) {
                         j[std::to_string(w)] = n;
                     }
                 }
                 return j;
             }()},
            {"version_pieces_skipped_for_a_newer_loader_write", c_fs_serve[kind].loader_skip_pieces.load()},
            {"version_bytes_skipped_for_a_newer_loader_write", c_fs_serve[kind].loader_skip_bytes.load()},
            {"served_reads_where_a_loader_write_is_the_newest", c_fs_serve[kind].loader_newest_reads.load()},
            {"hole_pieces_by_writer", [&] {
                 nlohmann::json j = nlohmann::json::object();
                 for (u32 w = 0; w < 16; ++w) {
                     if (const u64 n = c_fs_serve[kind].hole_pieces_by_writer[w].load()) {
                         j[std::to_string(w)] = {{"pieces", n}, {"bytes", c_fs_serve[kind].hole_bytes_by_writer[w].load()}};
                     }
                 }
                 return j;
             }()},
            {"note", "memory now against the newest version at or before the record's submission; differences at depth 0 are writes after the submission, at depth 2 the game running ahead"}};
}

nlohmann::json FrameStoreSummary() {
    return {{"on", FrameStoreOn()},
            {"mode", FrameStoreMode()},
            {"switch", "TLG_FRAME_STORE=1 shadow (all writers), =2 readers take the store (per-frame writers only); default off"},
            {"submissions", c_fs_submits.load()},
            {"chunks_captured", c_fs_chunks.load()},
            {"bytes_captured", c_fs_bytes.load()},
            {"largest_frame_bytes", c_fs_max_frame_bytes.load()},
            {"bytes_held_now", c_fs_live_bytes.load()},
            {"recheck", [&] {
                 nlohmann::json j = {{"on", FsRecheckOn()}, {"chunks_compared", c_recheck_chunks.load()}, {"bytes_compared", c_recheck_bytes.load()}, {"chunks_changed_since_capture", c_recheck_changed_chunks.load()},
                                     {"changed_and_explained_by_a_later_announcement", c_recheck_explained.load()}, {"changed_unexplained_chunks", c_recheck_unexplained_chunks.load()},
                                     {"changed_unexplained_bytes", c_recheck_unexplained_bytes.load()}};
                 nlohmann::json by = nlohmann::json::object();
                 for (u32 w = 0; w < 16; ++w) {
                     if (const u64 n = c_recheck_unexplained_by_writer[w].load()) {
                         by[std::to_string(w)] = {{"chunks", n}, {"bytes", c_recheck_unexplained_bytes_by_writer[w].load()}};
                     }
                 }
                 j["unexplained_by_newest_writer"] = by;
                 return j;
             }()},
            {"pre_write_map_opens_seen", c_pre_calls.load()},
            {"pre_write_never_written_before", c_pre_never_written.load()},
            {"pre_write_written_in_the_running_period", c_pre_written_this_period.load()},
            {"pre_write_version_already_held", c_pre_have_version.load()},
            {"pre_write_unreadable", c_pre_unreadable.load()},
            {"pre_write_pieces_captured", c_pre_captured.load()},
            {"pre_write_bytes_captured", c_pre_bytes.load()},
            {"pre_write_bytes_captured_of_loader_writes", c_pre_loader_bytes.load()},
            {"retired_layer_bytes_now", c_fs_base_bytes.load()},
            {"retired_layer_entries_now", c_fs_base_entries.load()},
            {"retired_layer_bytes_added", c_fs_base_added_bytes.load()},
            {"retired_layer_bytes_trimmed_by_newer", c_fs_base_trimmed_bytes.load()},
            {"retired_layer_bytes_dropped_for_the_cap", c_fs_base_dropped_cap_bytes.load()},
            {"skipped_over_16mb", c_fs_skipped_big.load()},
            {"skipped_frame_over_128mb", c_fs_skipped_frame_full.load()},
            {"skipped_unreadable", c_fs_skipped_unreadable.load()},
            {"capture_ms_on_the_submitting_thread", c_fs_capture_ns.load() / 1e6}};
}
} // namespace

namespace {
/// F2.5 (default on, `TLG_F2_LEDGER=0` off) and M0 (memory-ownership-plan.md phase 0): every read of the game's memory after the frame's record was made (an image upload, a buffer
/// copied at bind time, a constant or geometry ring copy on the command processor) asks the write ledger whether that memory was written after the record. Per kind of read.
enum CheckClass : u32 { CheckInRing, CheckNoRecord, CheckOpenMap, CheckNeverWritten, CheckPartlyWritten, CheckWrittenUnchanged, CheckWrittenAfter, CheckClassCount };
constexpr const char* CheckNames[CheckClassCount] = {"in_the_ring_not_checked_here", "no_draw_record_before_the_first_draw", "a_map_of_it_is_open_not_checked", "never_written_by_anything_announced",
                                                     "partly_written_none_after_the_draw", "written_none_after_the_draw", "WRITTEN_AFTER_THE_DRAW"};
constexpr size_t LateKinds = static_cast<size_t>(LateKind::Count);
constexpr const char* LateKindNames[LateKinds] = {"image_upload_raster_thread", "buffer_stream_copy_raster_thread", "buffer_object_copy_raster_thread", "buffer_arena_sync_raster_thread",
                                                  "constant_ring_command_processor", "geometry_ring_command_processor"};
std::atomic<u64> c_check[LateKinds][CheckClassCount]{};
std::atomic<u64> c_check_bytes[LateKinds][CheckClassCount]{};
std::atomic<u64> c_late_after_bytes_by_writer[LateKinds][16]{}; ///< bytes read late (the range was written after the record) by the writer of the newest write
std::atomic<u64> c_late_after_by_writer[LateKinds][16]{};
std::atomic<u64> c_late_margin[LateKinds][MaxSites][10]{}; ///< submissions between the read's record and the newest write that made it late (0..8, 9 more), by call site
std::atomic<u64> c_late_notdone_by_site[LateKinds][MaxSites]{}; ///< late reads whose record's submission was not yet done by the marker (the ones that can be real), by newest writer site
std::atomic<u64> c_late_overreach[LateKinds]{}; ///< not-done late reads whose newer bytes all lie past the end of the engine object the read starts in (a bind sized larger than its object: the draw does not use those bytes)
std::atomic<u64> c_late_marker[LateKinds][3]{}; ///< late reads by the raster thread's submission-done marker: [0] the record's submission was not done, [1] done (the read came after the marker), [2] no record
std::atomic<u64> c_late_free_class[LateKinds][MaxSites][3]{}; ///< rewritten late reads by the engine free of the address: [0] none seen, [1] freed after the record's submission, [2] freed before it
std::atomic<u64> c_late_rewrite_by_site[LateKinds][MaxSites]{}, c_late_rewrite_bytes_by_site[LateKinds][MaxSites]{}; ///< late reads whose range had an older version before the newest write (memory reuse), and the bytes rewritten
std::atomic<u64> c_late_after_by_site[LateKinds][MaxSites]{}, c_late_after_bytes_by_site[LateKinds][MaxSites]{}; ///< `TLG_WRITER_SITES`: the same by call site of the newest write
std::atomic<u64> c_same_frame_after[LateKinds]{}; ///< reads that M0 counted as late (written after the record) but that were written before the record's submission
std::atomic<u32> c_late_rare_noted[LateKinds]{}; ///< rare events written per kind (the first few only; the counters carry the rest)
std::atomic<u64> c_check_record_seen{0}, c_check_no_record_after_a_record{0};
std::atomic<u64> c_check_margin_log2[LateKinds][40]{}; ///< checked reads by log2(draw record - newest write stamp): how many records ahead of the draw the memory was last written
} // namespace

const std::vector<std::string>& CounterNames() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> n;
        static constexpr const char* per_kind[] = {"written_after_reads", "written_after_bytes", "store_reads_with_differences", "served_reads", "replaced_reads", "replaced_bytes",
                                                   "hole_reads", "hole_bytes", "first_write_bytes", "loader_skip_bytes"};
        for (size_t k = 0; k < LateKinds; ++k) {
            for (const char* name : per_kind) {
                n.push_back(std::string(LateKindNames[k]) + "." + name);
            }
        }
        for (u32 w = 0; w < 8; ++w) {
            n.push_back("writer" + std::to_string(w) + ".late_bytes");
            n.push_back("writer" + std::to_string(w) + ".hole_bytes");
        }
        n.push_back("recheck_unexplained_chunks");
        return n;
    }();
    return names;
}

void CounterSample(std::vector<u64>& values) {
    values.clear();
    constexpr auto relaxed = std::memory_order_relaxed;
    for (size_t k = 0; k < LateKinds; ++k) {
        const auto& s = c_fs_serve[k];
        values.push_back(c_check[k][CheckWrittenAfter].load(relaxed));
        values.push_back(c_check_bytes[k][CheckWrittenAfter].load(relaxed));
        values.push_back(c_fs_kind[k].reads_with_differences.load(relaxed));
        values.push_back(s.reads.load(relaxed));
        values.push_back(s.replaced_reads.load(relaxed));
        values.push_back(s.replaced_bytes.load(relaxed));
        values.push_back(s.hole_reads.load(relaxed));
        values.push_back(s.hole_bytes.load(relaxed));
        values.push_back(s.first_write_bytes.load(relaxed));
        values.push_back(s.loader_skip_bytes.load(relaxed));
    }
    for (u32 w = 0; w < 8; ++w) {
        u64 late = 0, hole = 0;
        for (size_t k = 0; k < LateKinds; ++k) {
            late += c_late_after_bytes_by_writer[k][w].load(relaxed);
            hole += c_fs_serve[k].hole_bytes_by_writer[w].load(relaxed);
        }
        values.push_back(late);
        values.push_back(hole);
    }
    values.push_back(c_recheck_unexplained_chunks.load(relaxed));
}

// ---- W1.5: silent writer census (TLG_SILENT_CENSUS=1, diagnostics only) ---------------------------------------------------------------------------------------------
// The kernel's write watch (runtime_memory.c) reports every page of GPU visible memory the game wrote; once per graphics submission the written pages are compared with
// the ledger of announced writes. A page with no announcement in this window or the one before is a silent write (a writer nobody hooked). The raster thread's reads
// are then checked against those pages: a read for submission S of a page silently written for a later submission is a late read the ledger cannot see.
// Page granular: a silent write elsewhere in the page counts too (samples show the ledger's view of the page to tell).
extern "C" uint64_t runtime_memory_census_harvest(void (*page)(uintptr_t address, void* context), void* context, uint64_t* gpu_side_resets);

namespace {
bool SilentCensusOn() noexcept {
    static const bool on = [] { const char* e = std::getenv("TLG_SILENT_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::mutex g_silent_mutex;
std::unordered_map<u64, u32> g_silent_at;   ///< page -> the submission its newest silent write belongs to
std::unordered_map<u64, u32> g_silent_hits; ///< page -> windows with a silent write
std::atomic<u32> g_silent_gen{0};           ///< bumped by every harvest that found silent pages
std::atomic<u64> c_sc_harvests{0}, c_sc_written{0}, c_sc_ring{0}, c_sc_announced_now{0}, c_sc_announced_before{0}, c_sc_silent{0}, c_sc_ns{0}, c_sc_max_ns{0};
std::atomic<u64> c_sc_late[LateKinds][2]{}, c_sc_late_bytes[LateKinds][2]{}, c_sc_reads_checked{0};
struct HarvestContext {
    u32 index;
    std::vector<u64> silent;
};
void SilentPage(uintptr_t page, void* context) {
    auto& ctx = *static_cast<HarvestContext*>(context);
    c_sc_written.fetch_add(1, std::memory_order_relaxed);
    if (InRing(page, 4096)) {
        c_sc_ring.fetch_add(1, std::memory_order_relaxed); // the frame ring: per-frame by design, the ring swap's
        return;
    }
    const LedgerAnswer a = LedgerQuery(page, page + 4096);
    // an announcement made while the count was index - 1 belongs to this window; writers that announce before they write may have announced in the window before
    if (a.covered != 0 && a.newest_submit + 1 >= ctx.index) {
        c_sc_announced_now.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (a.covered != 0 && a.newest_submit + 2 >= ctx.index) {
        c_sc_announced_before.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_sc_silent.fetch_add(1, std::memory_order_relaxed);
    ctx.silent.push_back(page);
}
} // namespace

void SilentHarvest(u32 index) {
    if (!SilentCensusOn()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    HarvestContext ctx{index, {}};
    runtime_memory_census_harvest(&SilentPage, &ctx, nullptr);
    if (!ctx.silent.empty()) {
        std::scoped_lock lock{g_silent_mutex};
        for (const u64 page : ctx.silent) {
            g_silent_at[page] = index;
            ++g_silent_hits[page];
        }
        g_silent_gen.fetch_add(1, std::memory_order_relaxed);
    }
    c_sc_harvests.fetch_add(1, std::memory_order_relaxed);
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    c_sc_ns.fetch_add(ns, std::memory_order_relaxed);
    if (ns > c_sc_max_ns.load(std::memory_order_relaxed)) {
        c_sc_max_ns.store(ns, std::memory_order_relaxed);
    }
}

namespace {
void SilentCheck(size_t k, u64 address, u64 size, u32 submit) {
    c_sc_reads_checked.fetch_add(1, std::memory_order_relaxed);
    const u64 first = address & ~u64{4095}, last = (address + size - 1) & ~u64{4095};
    u32 newest = 0;
    u64 newest_page = 0, hits = 0;
    {
        std::scoped_lock lock{g_silent_mutex};
        if (g_silent_at.empty()) {
            return;
        }
        u32 n = 0;
        for (u64 page = first; page <= last && n < 4096; page += 4096, ++n) {
            const auto it = g_silent_at.find(page);
            if (it != g_silent_at.end() && it->second > newest) {
                newest = it->second;
                newest_page = page;
            }
        }
        if (newest != 0) {
            hits = g_silent_hits[newest_page];
        }
    }
    if (newest <= submit) {
        return; // no silent write after the record's submission
    }
    const size_t marker = RasterDoneSubmissions() >= submit ? 1 : 0; // 1: stale record (the raster thread is past it)
    c_sc_late[k][marker].fetch_add(1, std::memory_order_relaxed);
    c_sc_late_bytes[k][marker].fetch_add(size, std::memory_order_relaxed);
    static std::atomic<u32> examples{0};
    if (marker == 0 && examples.fetch_add(1, std::memory_order_relaxed) < 40) {
        const LedgerAnswer a = LedgerQuery(newest_page, newest_page + 4096);
        Core::RareEvents::Note("w15_silent_late_read",
                               fmt::format("{} {:#x} + {:#x}: record submit {}, page {:#x} silently written for submission {} (silent in {} windows), raster done floor {}; ledger of the page: covered {:#x}, newest announced at {} by writer {} site {}",
                                           LateKindNames[k], address, size, submit, newest_page, newest, hits, RasterDoneSubmissions(), a.covered, a.newest_submit, a.newest_writer,
                                           a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?")));
    }
}

// Binds served from a copy made earlier (resident arena, reused object copy) read no memory, so no late read check sees them. If the game wrote the buffer silently after the
// copy, the GPU draws with old bytes (a missed update, not a late read). Margin = harvest windows between the copy and the newest silent write: 1 is ambiguous (the write
// may have landed before the copy in the same window), 2 or more is a missed update.
std::mutex g_synced_mutex;
std::unordered_map<u64, u32> g_synced[2]; ///< buffer key -> g_submit_count when the copy was made
std::atomic<u64> c_rs_uses[2]{}, c_rs_no_copy_record[2]{}, c_rs_ambiguous[2]{}, c_rs_missed[2]{}, c_rs_missed_bytes[2]{}, c_rs_synced[2]{};
std::atomic<u64> c_rs_missed_by_usage[2][4]{};
std::unordered_map<u64, u32> g_rs_missed_buffers[2]; ///< key -> missed uses (under g_synced_mutex), for the top list
std::unordered_map<u64, std::vector<u8>> g_synced_bytes[2]; ///< key -> the bytes at the copy (buffers up to 4 KB; under g_synced_mutex)
std::atomic<u64> c_rs_bytes_changed[2]{}, c_rs_bytes_same[2]{}, c_rs_bytes_unknown[2]{};

u32 SilentNewest(u64 address, u64 size, u64* page_out) {
    const u64 first = address & ~u64{4095}, last = (address + size - 1) & ~u64{4095};
    u32 newest = 0;
    std::scoped_lock lock{g_silent_mutex};
    if (g_silent_at.empty()) {
        return 0;
    }
    u32 n = 0;
    for (u64 page = first; page <= last && n < 1024; page += 4096, ++n) {
        const auto it = g_silent_at.find(page);
        if (it != g_silent_at.end() && it->second > newest) {
            newest = it->second;
            if (page_out != nullptr) {
                *page_out = page;
            }
        }
    }
    return newest;
}
} // namespace

void SilentSynced(u32 path, u64 key, u64 size) {
    if (!SilentCensusOn() || path > 1) {
        return;
    }
    c_rs_synced[path].fetch_add(1, std::memory_order_relaxed);
    std::vector<u8> bytes;
    if (size != 0 && size <= 4096) {
        bytes.resize(size);
        std::memcpy(bytes.data(), reinterpret_cast<const void*>(key), size);
    }
    std::scoped_lock lock{g_synced_mutex};
    if (g_synced[path].size() > 400000) {
        g_synced[path].clear();
        g_synced_bytes[path].clear();
    }
    g_synced[path][key] = g_submit_count.load(std::memory_order_acquire);
    if (bytes.empty()) {
        g_synced_bytes[path].erase(key);
    } else {
        g_synced_bytes[path][key] = std::move(bytes);
    }
}

void SilentResidentUse(u32 path, u64 key, u64 address, u64 size, u32 usage) {
    if (!SilentCensusOn() || path > 1 || size == 0) {
        return;
    }
    c_rs_uses[path].fetch_add(1, std::memory_order_relaxed);
    u32 copied = 0;
    {
        std::scoped_lock lock{g_synced_mutex};
        const auto it = g_synced[path].find(key);
        if (it == g_synced[path].end()) {
            c_rs_no_copy_record[path].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        copied = it->second;
    }
    u64 page = 0;
    const u32 newest = SilentNewest(address, size, &page);
    if (newest <= copied) {
        return;
    }
    if (newest - copied == 1) {
        c_rs_ambiguous[path].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_rs_missed[path].fetch_add(1, std::memory_order_relaxed);
    c_rs_missed_bytes[path].fetch_add(size, std::memory_order_relaxed);
    // byte check: do the bound bytes differ from the bytes at the copy (the page alone may have been written elsewhere)?
    u32 differing = ~0u;
    {
        std::scoped_lock lock{g_synced_mutex};
        const auto it = g_synced_bytes[path].find(key);
        if (it != g_synced_bytes[path].end() && address >= key && address + size <= key + it->second.size()) {
            differing = 0;
            const u8* now = reinterpret_cast<const u8*>(address);
            const u8* then = it->second.data() + (address - key);
            for (u64 i = 0; i < size; ++i) {
                differing += now[i] != then[i] ? 1u : 0u;
            }
        }
    }
    (differing == ~0u ? c_rs_bytes_unknown : differing == 0 ? c_rs_bytes_same : c_rs_bytes_changed)[path].fetch_add(1, std::memory_order_relaxed);
    if (differing != ~0u && differing != 0) {
        // name the writer: a hardware write watch on the first differing dword of this buffer (once per buffer, four in all)
        static std::mutex armed_mutex;
        static std::unordered_set<u64> armed;
        u64 first_diff = 0;
        {
            std::scoped_lock lock{g_synced_mutex};
            const auto it = g_synced_bytes[path].find(key);
            if (it != g_synced_bytes[path].end()) {
                const u8* now = reinterpret_cast<const u8*>(address);
                const u8* then = it->second.data() + (address - key);
                for (u64 i = 0; i < size; ++i) {
                    if (now[i] != then[i]) {
                        first_diff = address + i;
                        break;
                    }
                }
            }
        }
        bool arm = false;
        {
            std::scoped_lock lock{armed_mutex};
            arm = first_diff != 0 && armed.size() < 4 && armed.insert(key).second;
        }
        if (arm) {
            Core::NativeTsharpWatch::ArmAddress(first_diff, "w15 missed update");
        }
    }
    c_rs_missed_by_usage[path][std::min<u32>(usage, 3u)].fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{g_synced_mutex};
        if (g_rs_missed_buffers[path].size() < 100000) {
            ++g_rs_missed_buffers[path][key];
        }
    }
    static std::atomic<u32> examples{0};
    if (examples.fetch_add(1, std::memory_order_relaxed) < 40) {
        const LedgerAnswer a = LedgerQuery(page, page + 4096);
        Core::RareEvents::Note("w15_missed_update",
                               fmt::format("{} buffer {:#x} usage {}: bind {:#x} + {:#x} served from a copy made at submit count {}, page {:#x} silently written for submission {} (count now {}); bound bytes differing from the copy: {}; ledger of the page: covered {:#x}, newest announced at {} by writer {} site {}",
                                           path == 0 ? "resident arena" : "object copy reused", key, usage, address, size, copied, page, newest, g_submit_count.load(std::memory_order_relaxed),
                                           differing == ~0u ? std::string("unknown") : std::to_string(differing), a.covered,
                                           a.newest_submit, a.newest_writer, a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?")));
    }
}

// Texture cache: an image uploaded once and used again while it is not dirty reads no memory. If the game wrote its memory silently after the upload, the GPU samples old
// texels (a missed update). The upload records the harvest count and a hash of the guest bytes; a kept use is checked once per harvest that found silent pages.
extern "C" int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
namespace {
struct SilentImageRecord {
    u64 size;
    u32 uploaded;     ///< g_submit_count at the upload
    u64 hash;         ///< XXH3 of the guest bytes at the upload (0: not hashed)
    u32 checked_gen;  ///< the silent generation of the last check
    u32 hashed_newest; ///< the newest silent window the hash was compared for
    bool gpu_source;   ///< uploaded from GPU written memory (not checked)
};
std::atomic<u64> c_img_uploads_gpu{0}, c_img_size_differs{0}, c_img_gpu_source_kept{0};
std::unordered_map<u64, u32> g_img_no_record;      ///< kept image without an upload record -> uses (under g_img_mutex)
std::unordered_map<u64, u64> g_img_no_record_size; ///< its size
std::mutex g_img_mutex;
std::unordered_map<u64, SilentImageRecord> g_img_up;
std::atomic<u64> c_img_uploads{0}, c_img_kept{0}, c_img_kept_checked{0}, c_img_no_record{0}, c_img_ambiguous{0}, c_img_flagged{0}, c_img_changed{0}, c_img_same{0}, c_img_unknown{0};
std::unordered_map<u64, u32> g_img_changed; ///< image -> kept uses with changed bytes (under g_img_mutex)
u64 HashGuest(u64 address, u64 size) {
    if (size == 0 || size > (64ull << 20) || runtime_memory_is_mapped(address, size) == 0) {
        return 0;
    }
    const u64 h = XXH3_64bits(reinterpret_cast<const void*>(address), size);
    return h == 0 ? 1 : h;
}
} // namespace

void SilentImageUploaded(u64 address, u64 size, bool gpu_source) {
    if (!SilentCensusOn() || address == 0 || size == 0) {
        return;
    }
    (gpu_source ? c_img_uploads_gpu : c_img_uploads).fetch_add(1, std::memory_order_relaxed);
    const u64 hash = gpu_source ? 0 : HashGuest(address, size);
    std::scoped_lock lock{g_img_mutex};
    if (g_img_up.size() > 200000) {
        g_img_up.clear();
    }
    g_img_up[address] = {size, g_submit_count.load(std::memory_order_acquire), hash, g_silent_gen.load(std::memory_order_relaxed), 0, gpu_source};
}

void SilentImageKept(u64 address, u64 size) {
    if (!SilentCensusOn() || address == 0 || size == 0) {
        return;
    }
    c_img_kept.fetch_add(1, std::memory_order_relaxed);
    const u32 gen = g_silent_gen.load(std::memory_order_relaxed);
    u32 uploaded = 0, hashed_newest = 0;
    u64 hash = 0;
    {
        std::scoped_lock lock{g_img_mutex};
        const auto it = g_img_up.find(address);
        if (it == g_img_up.end() || it->second.size != size) {
            (it == g_img_up.end() ? c_img_no_record : c_img_size_differs).fetch_add(1, std::memory_order_relaxed);
            if (it == g_img_up.end() && g_img_no_record.size() < 50000) {
                ++g_img_no_record[address];
                g_img_no_record_size[address] = size;
            }
            return;
        }
        if (it->second.gpu_source) {
            c_img_gpu_source_kept.fetch_add(1, std::memory_order_relaxed); // content made by the GPU: memory is not the source
            return;
        }
        if (it->second.checked_gen == gen) {
            return; // no new silent pages since the last check of this image
        }
        it->second.checked_gen = gen;
        uploaded = it->second.uploaded;
        hash = it->second.hash;
        hashed_newest = it->second.hashed_newest;
    }
    c_img_kept_checked.fetch_add(1, std::memory_order_relaxed);
    u64 page = 0;
    u32 newest = 0;
    {
        const u64 first = address & ~u64{4095}, last = (address + size - 1) & ~u64{4095};
        std::scoped_lock lock{g_silent_mutex};
        for (u64 p = first; p <= last; p += 4096) {
            const auto it = g_silent_at.find(p);
            if (it != g_silent_at.end() && it->second > newest) {
                newest = it->second;
                page = p;
            }
        }
    }
    if (newest <= uploaded) {
        return;
    }
    // same window as the upload: the write may have come before it; the hash (taken after the upload read) decides either way
    (newest - uploaded == 1 ? c_img_ambiguous : c_img_flagged).fetch_add(1, std::memory_order_relaxed);
    if (newest == hashed_newest) {
        return; // compared already for this silent window
    }
    const u64 now = hash != 0 ? HashGuest(address, size) : 0;
    {
        std::scoped_lock lock{g_img_mutex};
        const auto it = g_img_up.find(address);
        if (it != g_img_up.end()) {
            it->second.hashed_newest = newest;
        }
        if (hash != 0 && now != 0 && now != hash && g_img_changed.size() < 100000) {
            ++g_img_changed[address];
        }
    }
    if (hash == 0 || now == 0) {
        c_img_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (now == hash) {
        c_img_same.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_img_changed.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<u32> examples{0};
    if (examples.fetch_add(1, std::memory_order_relaxed) < 40) {
        const LedgerAnswer a = LedgerQuery(page, page + 4096);
        Core::RareEvents::Note("w15_image_missed_update",
                               fmt::format("image {:#x} + {:#x} kept since its upload at submit count {}, bytes changed (hash), page {:#x} silently written for submission {} (count now {}); ledger of the page: covered {:#x}, newest announced at {} by writer {} site {}",
                                           address, size, uploaded, page, newest, g_submit_count.load(std::memory_order_relaxed), a.covered, a.newest_submit, a.newest_writer,
                                           a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?")));
    }
}

namespace {
nlohmann::json SilentReport() {
    nlohmann::json out;
    {
        nlohmann::json r;
        r["uploads"] = c_img_uploads.load();
        r["kept_uses"] = c_img_kept.load();
        r["kept_uses_checked"] = c_img_kept_checked.load();
        r["kept_without_an_upload_record"] = c_img_no_record.load();
        r["uploads_from_gpu_written_memory"] = c_img_uploads_gpu.load();
        r["kept_uses_of_gpu_source_images"] = c_img_gpu_source_kept.load();
        r["kept_with_a_record_of_another_size"] = c_img_size_differs.load();
        {
            std::vector<std::pair<u32, u64>> nr;
            std::scoped_lock lock{g_img_mutex};
            r["distinct_images_without_a_record"] = g_img_no_record.size();
            for (const auto& [a, n] : g_img_no_record) {
                nr.emplace_back(n, a);
            }
            std::sort(nr.begin(), nr.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            nlohmann::json t = nlohmann::json::array();
            for (size_t i = 0; i < nr.size() && i < 12; ++i) {
                t.push_back({{"image", fmt::format("{:#x}", nr[i].second)}, {"size", g_img_no_record_size[nr[i].second]}, {"uses", nr[i].first},
                             {"in_gpu_allocation", InGpuAllocation(nr[i].second, 1)}});
            }
            r["top_without_a_record"] = std::move(t);
        }
        r["silent_write_in_the_upload_window_ambiguous"] = c_img_ambiguous.load();
        r["flagged_silent_write_after_the_upload"] = c_img_flagged.load();
        r["hash_compares_changed_same_unknown"] = {c_img_changed.load(), c_img_same.load(), c_img_unknown.load()};
        std::vector<std::pair<u32, u64>> top;
        {
            std::scoped_lock lock{g_img_mutex};
            r["distinct_images_changed"] = g_img_changed.size();
            for (const auto& [a, n] : g_img_changed) {
                top.emplace_back(n, a);
            }
        }
        std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        nlohmann::json t = nlohmann::json::array();
        for (size_t i = 0; i < top.size() && i < 20; ++i) {
            u64 size = 0;
            {
                std::scoped_lock lock{g_img_mutex};
                const auto it = g_img_up.find(top[i].second);
                size = it != g_img_up.end() ? it->second.size : 0;
            }
            t.push_back({{"image", fmt::format("{:#x}", top[i].second)}, {"size", size}, {"changed_checks", top[i].first}});
        }
        r["top_images"] = std::move(t);
        out["texture_cache_kept_images"] = std::move(r);
    }
    for (u32 p = 0; p < 2; ++p) {
        nlohmann::json r;
        r["copies_made"] = c_rs_synced[p].load();
        r["binds_from_the_copy"] = c_rs_uses[p].load();
        r["binds_without_a_copy_record"] = c_rs_no_copy_record[p].load();
        r["silent_write_in_the_same_window_ambiguous"] = c_rs_ambiguous[p].load();
        r["missed_updates_silent_write_after_the_copy"] = c_rs_missed[p].load();
        r["missed_update_bytes"] = c_rs_missed_bytes[p].load();
        r["missed_bound_bytes_changed_same_unknown"] = {c_rs_bytes_changed[p].load(), c_rs_bytes_same[p].load(), c_rs_bytes_unknown[p].load()};
        r["missed_by_usage_0_1_2_3plus"] = {c_rs_missed_by_usage[p][0].load(), c_rs_missed_by_usage[p][1].load(), c_rs_missed_by_usage[p][2].load(), c_rs_missed_by_usage[p][3].load()};
        std::vector<std::pair<u32, u64>> top;
        {
            std::scoped_lock lock{g_synced_mutex};
            r["distinct_buffers_missed"] = g_rs_missed_buffers[p].size();
            for (const auto& [key, n] : g_rs_missed_buffers[p]) {
                top.emplace_back(n, key);
            }
        }
        std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        nlohmann::json t = nlohmann::json::array();
        for (size_t i = 0; i < top.size() && i < 20; ++i) {
            t.push_back({{"buffer", fmt::format("{:#x}", top[i].second)}, {"missed_binds", top[i].first}});
        }
        r["top_buffers"] = std::move(t);
        out[p == 0 ? "resident_arena_binds" : "object_copy_reuse_binds"] = std::move(r);
    }
    u64 gpu_side = 0;
    runtime_memory_census_harvest(nullptr, nullptr, &gpu_side);
    out["on"] = SilentCensusOn();
    out["harvests"] = c_sc_harvests.load();
    out["harvest_ms_total"] = c_sc_ns.load() / 1000000;
    out["harvest_ms_max"] = static_cast<double>(c_sc_max_ns.load()) / 1e6;
    out["pages_written"] = c_sc_written.load();
    out["pages_in_the_frame_ring"] = c_sc_ring.load();
    out["pages_announced_in_this_window"] = c_sc_announced_now.load();
    out["pages_announced_in_the_window_before"] = c_sc_announced_before.load();
    out["pages_silent"] = c_sc_silent.load();
    out["pages_dropped_as_gpu_side_writes"] = gpu_side;
    out["raster_reads_checked"] = c_sc_reads_checked.load();
    for (size_t k = 0; k < LateKinds; ++k) {
        if (c_sc_late[k][0].load() + c_sc_late[k][1].load() != 0) {
            out["silent_late_reads"][LateKindNames[k]] = {{"before_the_done_marker", c_sc_late[k][0].load()}, {"stale_record", c_sc_late[k][1].load()},
                                                          {"bytes_before_the_done_marker", c_sc_late_bytes[k][0].load()}};
        }
    }
    std::vector<std::pair<u32, u64>> top;
    {
        std::scoped_lock lock{g_silent_mutex};
        out["distinct_silent_pages"] = g_silent_hits.size();
        for (const auto& [page, n] : g_silent_hits) {
            top.emplace_back(n, page);
        }
    }
    std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    nlohmann::json pages = nlohmann::json::array();
    for (size_t i = 0; i < top.size() && i < 40; ++i) {
        const LedgerAnswer a = LedgerQuery(top[i].second, top[i].second + 4096);
        pages.push_back({{"page", fmt::format("{:#x}", top[i].second)}, {"silent_windows", top[i].first}, {"ledger_covered", a.covered}, {"ledger_newest_writer", a.newest_writer},
                         {"ledger_newest_site", a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?")}, {"ledger_newest_submit", a.newest_submit},
                         {"in_gpu_allocation", InGpuAllocation(top[i].second, 4096)}});
    }
    out["top_pages"] = std::move(pages);
    return out;
}
} // namespace

namespace {
thread_local BindContext t_bind;
thread_local bool t_bind_set = false;
} // namespace

void SetBindContext(const BindContext& context) {
    t_bind = context;
    t_bind_set = true;
}

void ClearBindContext() {
    t_bind_set = false;
}

bool HasBindContext() {
    return t_bind_set;
}

bool CurrentBindIsCopyDestination() {
    return t_bind_set && t_bind.site != nullptr && std::strcmp(t_bind.site, "copy destination") == 0;
}

namespace {
std::atomic<u64> c_copy_dst_syncs{0}, c_copy_dst_bytes{0};
} // namespace

void NoteCopyDestinationSync(u64 bytes) {
    c_copy_dst_syncs.fetch_add(1, std::memory_order_relaxed);
    c_copy_dst_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

namespace {
std::string DescribeBind() {
    if (!t_bind_set) {
        return "no bind context";
    }
    std::string owner = "no engine object for the id";
    if (t_bind.engine_id != 0) {
        Core::NativeDevice::Object o;
        if (Core::NativeDevice::Lookup(t_bind.engine_id, o)) {
            owner = fmt::format("engine buffer id {} alive at {:#x}+{:#x} created {}", o.id, static_cast<u64>(o.address), o.bytes, o.created_submit);
        } else {
            owner = fmt::format("engine id {} is not alive (deleted or unknown)", t_bind.engine_id);
        }
    }
    u64 rva = 0;
#ifdef _WIN32
    rva = t_bind.caller != 0 ? t_bind.caller - reinterpret_cast<u64>(GetModuleHandleW(nullptr)) : 0;
#endif
    return fmt::format("{} {:#x}+{:#x}, view id {}, program {:#x} stage {}, via handle {}, written {}, caller rva {:#x}: {}", t_bind.site != nullptr ? t_bind.site : "?", t_bind.address, t_bind.size, t_bind.view_id,
                       t_bind.program, t_bind.stage, t_bind.handle, t_bind.written, rva, owner);
}
} // namespace

void NoteLateRead(LateKind kind, u64 address, u64 size, u64 seq) {
    if (Core::ByteAudit::Active()) [[unlikely]] {
        // debugger mode: the bytes of this read against the memory hashed at the game's submit of its record (native_byte_audit.h)
        const u64 s = seq != 0 ? seq : Core::NativeRegistry::CurrentDrawSeq();
        Core::ByteAudit::NoteRead(address, size, s != 0 ? SubmitIndexOf(s) : 0u, LateKindNames[static_cast<size_t>(kind)], [] { return DescribeBind(); });
    }
    if (!LedgerOn() || size == 0) {
        return;
    }
    if (SilentCensusOn()) {
        const u64 s = seq != 0 ? seq : Core::NativeRegistry::CurrentDrawSeq();
        const u32 sub = s != 0 ? SubmitIndexOf(s) : 0;
        if (sub != 0 && !InRing(address, size)) {
            SilentCheck(static_cast<size_t>(kind), address, size, sub);
        }
    }
    const size_t k = static_cast<size_t>(kind);
    CheckClass cls;
    LedgerAnswer a;
    if (seq == 0) {
        seq = Core::NativeRegistry::CurrentDrawSeq();
    }
    const u32 submit = seq != 0 ? SubmitIndexOf(seq) : 0;
    if (InRing(address, size)) {
        cls = CheckInRing;
    } else if (seq == 0) {
        cls = CheckNoRecord;
        if (c_check_record_seen.load(std::memory_order_relaxed) != 0) {
            c_check_no_record_after_a_record.fetch_add(1, std::memory_order_relaxed); // a read on a thread that never draws (the presenter) or after the first draw of its thread
        }
    } else if (Core::NativeUpdates::OpenMapOverlaps(address, size)) {
        cls = CheckOpenMap;
    } else {
        c_check_record_seen.store(1, std::memory_order_relaxed);
        a = LedgerQuery(address, address + size, submit);
        if (a.covered != 0 && a.newest <= seq) {
            c_check_margin_log2[k][std::min<u32>(39u, static_cast<u32>(std::bit_width(seq - a.newest)))].fetch_add(1, std::memory_order_relaxed);
        }
        // M1: what the GPU of the record's submission sees is the memory as the game left it at the submission, so a write counts as late only when it was announced after that
        // submission (M0 compared with the record, which also counted writes made later in the same frame: those are in the draw's data)
        const bool after_record = a.newest > seq;
        const bool after_submit = submit != 0 ? a.newest_submit >= submit : after_record;
        if (after_record && !after_submit) {
            c_same_frame_after[k].fetch_add(1, std::memory_order_relaxed);
        }
        cls = a.covered == 0 ? CheckNeverWritten : after_submit ? CheckWrittenAfter : a.covered < size ? CheckPartlyWritten : CheckWrittenUnchanged;
    }
    if (FrameStoreOn() && k != static_cast<size_t>(LateKind::ImageUpload) && submit != 0) {
        FrameStoreCompare(k, address, size, submit);
    }
    c_check[k][cls].fetch_add(1, std::memory_order_relaxed);
    c_check_bytes[k][cls].fetch_add(size, std::memory_order_relaxed);
    if (cls == CheckWrittenAfter) {
        // the game wrote this memory after the draw that uses it was recorded: the bytes the read finds may not be the draw's (at frames in flight 0 the game is held back, so this stays near 0)
        const u32 writer = std::min<u32>(a.newest_writer, 15u);
        c_late_after_by_writer[k][writer].fetch_add(1, std::memory_order_relaxed);
        c_late_after_bytes_by_writer[k][writer].fetch_add(size, std::memory_order_relaxed);
        c_late_after_by_site[k][std::min<u32>(a.newest_site, MaxSites - 1)].fetch_add(1, std::memory_order_relaxed);
        {
            const size_t marker = submit == 0 ? 2 : RasterDoneSubmissions() >= submit ? 1 : 0; // SubmitIndexOf is the (1 based) submission that carries the record: done when the floor reached it (a read after that has a stale record: the raster thread is on a later task)
            bool overreach = false;
            if (marker == 0 && submit != 0 && a.late_lo != ~u64{0}) {
                const u64 owner_end = Core::NativeDevice::OwnerEndAt(address, submit);
                overreach = owner_end != 0 && a.late_lo >= owner_end;
                if (overreach) {
                    c_late_overreach[k].fetch_add(1, std::memory_order_relaxed);
                    static std::atomic<u32> overreach_examples{0};
                    if (overreach_examples.fetch_add(1, std::memory_order_relaxed) < 6) {
                        Core::RareEvents::Note("late_read_past_owner_end", fmt::format("{} {:#x} + {:#x}: the owner object ends at {:#x}, the newer bytes are {:#x}..{:#x}, record submit {}, bind: {}", LateKindNames[k], address, size,
                                                                                        owner_end, a.late_lo, a.late_hi, submit, DescribeBind()));
                    }
                }
            }
            c_late_marker[k][overreach ? 1 : marker].fetch_add(1, std::memory_order_relaxed);
            if (marker == 0 && !overreach) {
                c_late_notdone_by_site[k][std::min<u32>(a.newest_site, MaxSites - 1)].fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> before_marker_examples{0};
                if (before_marker_examples.fetch_add(1, std::memory_order_relaxed) < 40) {
                    Core::RareEvents::Note("late_read_before_done_marker",
                                           fmt::format("{} {:#x} + {:#x} (rewritten {:#x}): record {} write stamp {}, record submit {}, raster done floor {}, submit count now {}, newest write announced at {} (site {}, writer {}), previous write at {}, freed by the engine at {}, containing freed block {}; owners at the record's submission {}; owners at the write's submission {}; bind: {}",
                                                       LateKindNames[k], address, size, a.rewritten, seq, a.newest, submit, RasterDoneSubmissions(), g_submit_count.load(std::memory_order_relaxed), a.newest_submit,
                                                       a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?"), a.newest_writer, a.newest_prev_submit, HeapFreeAt(address),
                                                       [&] { u64 b = 0, n = 0; const u32 at = HeapFreeContaining(address, b, n); return at == 0 ? std::string("none") : fmt::format("{:#x}+{:#x} freed at {}", b, n, at); }(),
                                                       Core::NativeDevice::OwnersAt(address, size, submit), Core::NativeDevice::OwnersAt(address, size, static_cast<u32>(a.newest_submit)), DescribeBind()));
                }
            }
        }
        if (a.rewritten != 0) {
            c_late_rewrite_by_site[k][std::min<u32>(a.newest_site, MaxSites - 1)].fetch_add(1, std::memory_order_relaxed);
            {
                u32 freed_at = 0;
                bool freed = false;
                {
                    std::scoped_lock lock{g_heap_free_mutex};
                    const auto it = g_heap_free_at.find(address);
                    if (it != g_heap_free_at.end()) {
                        freed = true;
                        freed_at = it->second;
                    }
                }
                c_late_free_class[k][std::min<u32>(a.newest_site, MaxSites - 1)][!freed ? 0 : freed_at > submit ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
            }
            c_late_rewrite_bytes_by_site[k][std::min<u32>(a.newest_site, MaxSites - 1)].fetch_add(a.rewritten, std::memory_order_relaxed);
        }
        if (submit != 0) {
            c_late_margin[k][std::min<u32>(a.newest_site, MaxSites - 1)][std::min<u32>(a.newest_submit > submit ? a.newest_submit - submit : 0u, 9u)].fetch_add(1, std::memory_order_relaxed);
        }
        c_late_after_bytes_by_site[k][std::min<u32>(a.newest_site, MaxSites - 1)].fetch_add(size, std::memory_order_relaxed);
        {
            static const bool gap_on = [] { const char* e = std::getenv("TLG_REWRITE_GAP"); return e != nullptr && e[0] == '1'; }();
            static std::atomic<u32> examples[LateKinds]{};
            if (gap_on && (k == static_cast<size_t>(LateKind::BufferObjectCopy) || k == static_cast<size_t>(LateKind::BufferArenaSync)) && examples[k].fetch_add(1, std::memory_order_relaxed) % (k == static_cast<size_t>(LateKind::BufferArenaSync) ? 150 : 40000) < (k == static_cast<size_t>(LateKind::BufferArenaSync) ? 1 : 6)) {
                Core::RareEvents::Note(k == static_cast<size_t>(LateKind::BufferArenaSync) ? "late_read_example_arena_sync" : "late_read_example_object_copy",
                                       fmt::format("{:#x} + {:#x} (covered {:#x}, rewritten {:#x}): record submit {}, newest write announced at submit count {} (belongs to submission {}), the write it replaced was at {}, submit count now {}, writer {}, site {}, memory freed by the engine at submit {}", address, size, a.covered, a.rewritten,
                                                   submit, a.newest_submit, a.newest_submit + 1, a.newest_prev_submit, g_submit_count.load(std::memory_order_relaxed), a.newest_writer,
                                                   a.newest_site < g_site_names.size() ? g_site_names[a.newest_site] : std::string("?"),
                                                   [&] { std::scoped_lock lock{g_heap_free_mutex}; const auto it = g_heap_free_at.find(address); return it == g_heap_free_at.end() ? -1 : static_cast<int>(it->second); }()));
            }
        }
        if (c_late_rare_noted[k].fetch_add(1, std::memory_order_relaxed) < 3) {
            Core::RareEvents::Note(fmt::format("m0_late_read_written_after_its_record_{}", LateKindNames[k]), fmt::format("{:#x} + {:#x}: record {}, newest write stamp {}, writer {}", address, size, seq, a.newest, a.newest_writer));
        }
    }
}

void NoteHeapFree(u64 address) {
    const u64 size = GpuAllocationSize(address);
    std::scoped_lock lock{g_heap_free_mutex};
    if (g_heap_free_at.size() > 400000) {
        g_heap_free_at.clear();
        g_heap_free_blocks.clear();
    }
    const u32 at = g_submit_count.load(std::memory_order_relaxed);
    g_heap_free_at[address] = at;
    if (size != 0) {
        g_heap_free_blocks[address] = {size, at};
    }
}

u32 HeapFreeAt(u64 address) {
    std::scoped_lock lock{g_heap_free_mutex};
    const auto it = g_heap_free_at.find(address);
    return it == g_heap_free_at.end() ? 0u : it->second;
}

u32 HeapFreeContaining(u64 address, u64& block_start, u64& block_size) {
    std::scoped_lock lock{g_heap_free_mutex};
    auto it = g_heap_free_blocks.upper_bound(address);
    if (it == g_heap_free_blocks.begin()) {
        return 0;
    }
    --it;
    if (address >= it->first + it->second.first) {
        return 0;
    }
    block_start = it->first;
    block_size = it->second.first;
    return it->second.second;
}

void CheckUploadMemory(u64 address, u64 size) {
    NoteLateRead(LateKind::ImageUpload, address, size, 0);
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
    u64 image_newer = 0, image_unexplained = 0, image_first_diff = ~0ull, image_newest_stamp = 0, image_chosen_stamp = 0;
    u32 image_writer = 0, image_writer_unexplained = 0;
    u64 image_unexplained_seq = ~0ull; // the oldest entry (by seq) with an unexplained byte
    for (const Piece& p : pieces) {
        u8* destination = staged + (p.begin - address);
        const u8* source = p.entry->Bytes() + (p.begin - p.entry_begin);
        const u64 n = p.end - p.begin;
        if (std::memcmp(destination, source, n) != 0) {
            // the entries announced after this draw's record that overlap the piece (they hold what memory may have been overwritten with)
            std::vector<const Entry*> newer;
            std::vector<u64> newer_begin;
            const u64 from = p.begin > g_max_size ? p.begin - g_max_size : 0;
            for (auto it = g_entries.lower_bound(from); it != g_entries.end() && it->first < p.end; ++it) {
                if (it->first + it->second.size > p.begin && it->second.stamp > seq) {
                    newer.push_back(&it->second);
                    newer_begin.push_back(it->first);
                    image_newest_stamp = std::max(image_newest_stamp, it->second.stamp);
                }
            }
            for (u64 i = 0; i < n; ++i) {
                if (destination[i] == source[i]) {
                    continue;
                }
                ++changed;
                image_first_diff = std::min(image_first_diff, p.begin + i - address);
                bool explained = false;
                for (size_t k = 0; k < newer.size() && !explained; ++k) {
                    const u64 at = p.begin + i;
                    explained = at >= newer_begin[k] && at < newer_begin[k] + newer[k]->size && newer[k]->Bytes()[at - newer_begin[k]] == destination[i];
                }
                const u32 w = std::min<u32>(p.entry->writer == WriterCopy ? 15u : p.entry->writer, 15u);
                c_dif_writer[w].fetch_add(1, std::memory_order_relaxed);
                if (explained) {
                    ++image_newer;
                } else {
                    ++image_unexplained;
                    image_writer_unexplained = w;
                    image_unexplained_seq = std::min(image_unexplained_seq, p.entry->seq);
                }
                image_writer = w;
            }
            image_chosen_stamp = std::max(image_chosen_stamp, p.entry->stamp);
            std::memcpy(destination, source, n);
        }
        covered += n;
    }
    if (changed != 0) {
        c_dif_images.fetch_add(1, std::memory_order_relaxed);
        c_dif_bytes_newer.fetch_add(image_newer, std::memory_order_relaxed);
        c_dif_bytes_unexplained.fetch_add(image_unexplained, std::memory_order_relaxed);
        u32 mark_writer = 0;
        if (image_unexplained == 0) {
            c_dif_images_all_newer.fetch_add(1, std::memory_order_relaxed);
        } else {
            c_dif_images_unexplained.fetch_add(1, std::memory_order_relaxed);
            if (ShadowOn()) {
                // a writer whose bytes are not kept (1, 2, 4) announced a write over this memory after the oldest unexplained entry
                std::scoped_lock marks_lock{g_marks_mutex};
                const u64 mfrom = address > g_marks_max_size ? address - g_marks_max_size : 0;
                for (auto it = g_marks.lower_bound(mfrom); it != g_marks.end() && it->first < address + size; ++it) {
                    if (it->first + it->second.size > address && it->second.seq >= image_unexplained_seq) {
                        mark_writer = it->second.writer;
                    }
                }
                c_dif_images_unexplained_mark.fetch_add(mark_writer != 0, std::memory_order_relaxed);
            }
        }
        if (!uncovered.empty()) {
            c_dif_images_with_uncovered.fetch_add(1, std::memory_order_relaxed);
        }
        std::scoped_lock dif_lock{g_dif_mutex};
        if (g_dif_examples.size() < 24 && (image_unexplained != 0 || g_dif_examples.size() < 8)) {
            Core::NativeDevice::Object owner;
            const bool owned = Core::NativeDevice::ResourceAt(address, owner);
            g_dif_examples.push_back({{"address", fmt::format("{:#x}", address)}, {"size", size}, {"draw_record", seq}, {"image_bytes_differing", changed},
                                      {"bytes_matching_a_newer_entry", image_newer}, {"bytes_matching_nothing", image_unexplained}, {"first_diff_offset", image_first_diff},
                                      {"chosen_entry_stamp", image_chosen_stamp}, {"newest_overlapping_stamp", image_newest_stamp}, {"writer_of_last_differing_entry", image_writer},
                                      {"writer_of_last_unexplained_entry", image_writer_unexplained}, {"mark_writer_after", mark_writer}, {"uncovered_segments", uncovered.size()},
                                      {"engine_resource", owned ? fmt::format("kind {} usage {} cpu_access {:#x}", static_cast<u32>(owner.kind), owner.usage, owner.cpu_access) : "none"}});
        }
    }
    if (!uncovered.empty()) {
        c_unc_images.fetch_add(1, std::memory_order_relaxed);
        c_unc_images_whole.fetch_add(covered == 0, std::memory_order_relaxed);
        for (const auto& [s, t] : uncovered) {
            u32 cls = UncNothingZero;
            u32 writer = 0;
            bool zero = true;
            for (u64 i = s; i < t && zero; ++i) {
                zero = staged[i - address] == 0;
            }
            u64 newest_overlap_stamp = 0;
            const u64 from = s > g_max_size ? s - g_max_size : 0;
            for (auto it = g_entries.lower_bound(from); it != g_entries.end() && it->first < t; ++it) {
                if (it->first + it->second.size > s && it->second.stamp > seq) {
                    newest_overlap_stamp = std::max(newest_overlap_stamp, it->second.stamp);
                }
            }
            if (newest_overlap_stamp != 0) {
                cls = UncNewerEntry;
            } else {
                if (ShadowOn()) {
                    std::scoped_lock marks_lock{g_marks_mutex};
                    const u64 mfrom = s > g_marks_max_size ? s - g_marks_max_size : 0;
                    for (auto it = g_marks.lower_bound(mfrom); it != g_marks.end() && it->first < t; ++it) {
                        if (it->first + it->second.size > s) {
                            writer = it->second.writer;
                        }
                    }
                }
                cls = writer != 0 ? UncMark : zero ? UncNothingZero : UncNothingNonZero;
            }
            c_unc_segments[cls].fetch_add(1, std::memory_order_relaxed);
            c_unc_bytes[cls].fetch_add(t - s, std::memory_order_relaxed);
            if (cls == UncMark) {
                c_unc_mark_writer[writer & 7].fetch_add(t - s, std::memory_order_relaxed);
            }
            std::scoped_lock dif_lock{g_dif_mutex};
            if (g_unc_examples.size() < 24 && (cls == UncNothingNonZero || g_unc_examples.size() < 8)) {
                Core::NativeDevice::Object owner;
                const bool owned = Core::NativeDevice::ResourceAt(address, owner);
                g_unc_examples.push_back({{"class", UncNames[cls]}, {"image", fmt::format("{:#x}", address)}, {"image_size", size}, {"segment_offset", s - address}, {"segment_bytes", t - s},
                                          {"covered_bytes_of_image", covered}, {"draw_record", seq}, {"mark_writer", writer}, {"newer_entry_stamp", newest_overlap_stamp}, {"memory_all_zero", zero},
                                          {"engine_resource", owned ? fmt::format("kind {} usage {} cpu_access {:#x}", static_cast<u32>(owner.kind), owner.usage, owner.cpu_access) : "none"}});
            }
        }
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
    bool hole_never = false;
    for (const auto& [s, t] : uncovered) {
        u64 nonzero = 0;
        if (!GuardedNonZero(reinterpret_cast<const void*>(s), t - s, nonzero)) {
            unreadable = true;
            continue;
        }
        hole += nonzero;
        if (nonzero != 0) {
            // how much of this segment lies in dropped (evicted) ranges
            u64 evicted = 0, at = s;
            for (auto it = g_tomb.lower_bound(s > g_tomb_max ? s - g_tomb_max : 0); it != g_tomb.end() && it->first < t; ++it) {
                const u64 x = std::max(at, it->first), y = std::min(t, it->second);
                if (x < y) {
                    evicted += y - x;
                    at = y;
                }
            }
            if (evicted >= t - s) {
                g_hole_nonzero_evicted += nonzero;
            } else if (evicted == 0) {
                g_hole_nonzero_never += nonzero;
                hole_never = true;
            } else {
                g_hole_nonzero_mixed += nonzero;
                hole_never = true;
            }
        }
    }
    if (hole != 0) {
        (hole_never ? g_hole_uploads_with_never : g_hole_uploads_all_evicted) += 1;
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
    {
        // F2.5 shadow: what the write ledger says about this memory at the draw's record
        LedgerClass lc;
        const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
        LedgerAnswer a;
        if (seq == 0) {
            lc = LedgerNoSeq;
        } else if (map_open) {
            lc = LedgerOpenMap;
        } else {
            a = LedgerQuery(upload.address, upload.address + upload.size);
            lc = a.covered == 0 ? LedgerNeverWritten : a.newest > seq ? LedgerWrittenAfter : a.covered < upload.size ? LedgerPartlyWrittenUnchanged : LedgerWrittenUnchanged;
        }
        const u32 sc = cls == ClassEqual ? 0 : cls == ClassEmptyEqual ? 1 : cls == ClassDiffers ? 2 : 3;
        static u64 ordinal = 0; // under g_mutex
        ++ordinal;
        ++g_ledger_uploads[lc][sc];
        g_ledger_bytes[lc][sc] += upload.size;
        if ((lc == LedgerWrittenAfter || lc == LedgerNoSeq || (lc == LedgerNeverWritten && cls != ClassEmptyEqual) || cls == ClassDiffers) && g_ledger_examples.size() < 16) {
            Core::NativeDevice::Object owner;
            const bool owned = Core::NativeDevice::ResourceAt(upload.address, owner);
            g_ledger_examples.push_back({{"ledger", LedgerNames[lc]}, {"store_class", sc}, {"group", GroupName(group)}, {"address", fmt::format("{:#x}", upload.address)}, {"size", upload.size}, {"caller", upload.caller}, {"upload_number", ordinal},
                                         {"draw_record", seq}, {"newest_write_stamp", a.newest}, {"ledger_covered_bytes", a.covered}, {"covered_by_store", covered}, {"hole_nonzero", hole},
                                         {"engine_resource", owned ? fmt::format("kind {} usage {} cpu_access {:#x}", static_cast<u32>(owner.kind), owner.usage, owner.cpu_access) : "none"}});
        }
    }
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
    if (SilentCensusOn()) {
        out["w15_silent_census"] = SilentReport();
    }
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
    {
        nlohmann::json check = {{"on", LedgerOn()}, {"switch", "TLG_F2_LEDGER=0 turns it off"}, {"ledger_notes", c_ledger_notes.load()}, {"ledger_overflow_skipped", c_ledger_overflow.load()}, {"graphics_submissions_so_far", g_submit_count.load()}};
        for (u32 w : {1u, 2u, 4u}) {
            for (u32 g = 0; g < 18; ++g) {
                if (const u64 n = c_gap[w][g].load()) {
                    check["rewrite_gap_submissions_by_writer"][std::to_string(w)][g == 17 ? "first" : std::to_string(g)] = n;
                }
            }
        }
        {
            std::scoped_lock ledger_lock{g_ledger_mutex};
            check["ledger_ranges"] = g_ledger.size();
        }
        for (u32 c = 0; c < CheckClassCount; ++c) {
            check["cpu_image_uploads"][CheckNames[c]] = {{"uploads", c_check[0][c].load()}, {"image_bytes", c_check_bytes[0][c].load()}};
        }
        check["frame_store"] = FrameStoreSummary();
        check["no_record_uploads_after_a_record_was_seen"] = c_check_no_record_after_a_record.load();
        for (u32 b = 0; b < 40; ++b) {
            if (const u64 n = c_check_margin_log2[0][b].load(); n != 0) {
                check["records_between_the_newest_write_and_the_draw_log2"][b == 0 ? "0" : fmt::format("{}..{}", 1ull << (b - 1), (1ull << b) - 1)] = n;
            }
        }
        // M0: the same question for every kind of late read (reads of the game's memory after the frame's record)
        for (size_t k = 0; k < LateKinds; ++k) {
            auto& kj = check["m0_late_reads"][LateKindNames[k]];
            u64 reads = 0, bytes = 0;
            for (u32 c = 0; c < CheckClassCount; ++c) {
                const u64 n = c_check[k][c].load(), nb = c_check_bytes[k][c].load();
                reads += n;
                bytes += nb;
                kj[CheckNames[c]] = {{"reads", n}, {"bytes", nb}};
            }
            kj["total_reads"] = reads;
            kj["written_after_the_record_but_before_its_submission_not_late"] = c_same_frame_after[k].load();
            if (FrameStoreOn()) {
                kj["frame_store_shadow"] = FrameStoreReport(k);
            }
            kj["total_bytes"] = bytes;
            kj["late_reads_by_done_marker_not_done_done_norecord"] = {c_late_marker[k][0].load(), c_late_marker[k][1].load(), c_late_marker[k][2].load()};
            kj["late_reads_not_done_but_newer_bytes_past_the_owner_object_end"] = c_late_overreach[k].load(); // counted in "done" above, not in "not done"
            for (u32 w = 0; w < 16; ++w) {
                if (const u64 n = c_late_after_by_writer[k][w].load()) {
                    kj["written_after_by_writer"][std::to_string(w)] = {{"reads", n}, {"bytes", c_late_after_bytes_by_writer[k][w].load()}};
                }
            }
            {
                std::scoped_lock site_lock{g_site_mutex};
                for (u32 i = 0; i < MaxSites; ++i) {
                    if (const u64 n = c_late_after_by_site[k][i].load()) {
                        auto& site_json = kj["written_after_by_site"][i < g_site_names.size() ? g_site_names[i] : std::to_string(i)];
                        site_json = {{"reads", n}, {"bytes", c_late_after_bytes_by_site[k][i].load()}, {"reads_with_rewritten_memory", c_late_rewrite_by_site[k][i].load()}, {"rewritten_bytes", c_late_rewrite_bytes_by_site[k][i].load()},
                                     {"reads_before_the_done_marker_of_the_records_submission", c_late_notdone_by_site[k][i].load()},
                                     {"rewritten_reads_by_engine_free_none_after_record_before_record", {c_late_free_class[k][i][0].load(), c_late_free_class[k][i][1].load(), c_late_free_class[k][i][2].load()}}};
                        for (u32 m = 0; m < 10; ++m) {
                            if (const u64 x = c_late_margin[k][i][m].load()) {
                                site_json["submissions_between_record_and_write"][std::to_string(m)] = x;
                            }
                        }
                    }
                }
            }
            for (u32 b = 0; b < 40; ++b) {
                if (const u64 n = c_check_margin_log2[k][b].load(); n != 0) {
                    kj["records_between_the_newest_write_and_the_draw_log2"][b == 0 ? "0" : fmt::format("{}..{}", 1ull << (b - 1), (1ull << b) - 1)] = n;
                }
            }
        }
        out["image_memory_check"] = std::move(check);
    }
    out["ring_index"] = {{"reused_exact_match", c_ring_index[0].load()},
                         {"created_new_image", c_ring_index[1].load()},
                         {"stale_images_freed", c_ring_index[2].load()},
                         {"stale_images_kept_not_old_enough", c_ring_index[3].load()},
                         {"owner_less_lookups_refused_not_in_ring_or_gpu_allocation", c_ring_index[4].load()}};
    out["enabled"] = On();
    out["shadow"] = ShadowOn();
    out["copy_destination_syncs"] = {{"syncs_with_upload", c_copy_dst_syncs.load()}, {"bytes", c_copy_dst_bytes.load()},
                                     {"note", "arena sync of a GPU buffer copy's destination: the copy overwrites the whole range right after, so the bytes read are never used"}};
    out["compare_by_open_map"] = {{"open_map_uploads", c_cmp_open.load()}, {"open_map_differs", c_cmp_open_differs.load()}, {"no_open_map_uploads", c_cmp_closed.load()}, {"no_open_map_differs", c_cmp_closed_differs.load()}};
    out["ring_swap"] = {{"on", RingSwapOn()}, {"images_overlaid", c_ov_images.load()}, {"bytes_from_store", c_ov_bytes.load()}, {"bytes_not_covered_left_to_memory", c_ov_uncovered_bytes.load()},
                        {"images_where_store_differs_from_memory", c_ov_images_changed.load()}, {"bytes_where_store_differs_from_memory", c_ov_changed_bytes.load()},
                        {"calls", c_ov_calls.load()}, {"not_in_ring", c_ov_not_ring.load()}, {"skipped_open_map", c_ov_skipped_open_map.load()}, {"skipped_no_record", c_ov_skipped_no_seq.load()},
                        {"switch", "TLG_F2_RING_STORE=0 turns it off"}};
    {
        nlohmann::json writers = nlohmann::json::object();
        for (u32 w = 0; w < 16; ++w) {
            if (const u64 n = c_dif_writer[w].load(); n != 0) {
                writers[w == 15 ? "copy" : std::to_string(w)] = n;
            }
        }
        std::scoped_lock dif_lock{g_dif_mutex};
        nlohmann::json unc = {{"images_with_uncovered_bytes", c_unc_images.load()}, {"of_those_nothing_covered_at_all", c_unc_images_whole.load()}};
        for (u32 c = 0; c < UncClassCount; ++c) {
            unc["segments"][UncNames[c]] = c_unc_segments[c].load();
            unc["bytes"][UncNames[c]] = c_unc_bytes[c].load();
        }
        for (u32 w = 0; w < 8; ++w) {
            if (const u64 n = c_unc_mark_writer[w].load(); n != 0) {
                unc["bytes_by_mark_writer"][std::to_string(w)] = n;
            }
        }
        unc["examples"] = g_unc_examples;
        out["ring_swap_uncovered"] = std::move(unc);
        {
            std::scoped_lock tmap_lock{g_tmap_mutex};
            out["texture_maps"] = {{"maps_for_writing", c_tmap_total.load()}, {"inside_a_ring_window", c_tmap_in_window.load()}, {"announced_equals_window_remaining", c_tmap_equal.load()},
                                   {"announced_SMALLER_than_window_remaining", c_tmap_announced_smaller.load()}, {"bytes_not_announced", c_tmap_smaller_bytes.load()},
                                   {"announced_larger", c_tmap_announced_larger.load()}, {"shapes", g_tmap_shapes}, {"examples", g_tmap_examples}};
        }
        {
            std::scoped_lock shape_lock{g_shape_mutex};
            for (const auto& [key, n] : g_shapes) {
                out["ring_image_shapes"][key] = n;
            }
        }
        out["ring_swap_differences"] = {{"images", c_dif_images.load()}, {"images_every_byte_matches_a_newer_entry", c_dif_images_all_newer.load()},
                                        {"images_with_bytes_matching_nothing", c_dif_images_unexplained.load()}, {"of_those_a_kept_less_writer_marked_after", c_dif_images_unexplained_mark.load()},
                                        {"bytes_matching_a_newer_entry", c_dif_bytes_newer.load()}, {"bytes_matching_nothing", c_dif_bytes_unexplained.load()},
                                        {"images_that_also_had_uncovered_bytes", c_dif_images_with_uncovered.load()}, {"differing_bytes_by_writer_of_the_stores_entry", writers},
                                        {"examples", g_dif_examples}};
    }
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
        one["hole_nonzero_bytes_total_by_cause"] = {{"announced_then_evicted_for_the_cap", g_hole_nonzero_evicted}, {"never_announced", g_hole_nonzero_never}, {"partly_evicted_partly_never", g_hole_nonzero_mixed},
                                                    {"hole_uploads_every_segment_evicted", g_hole_uploads_all_evicted}, {"hole_uploads_with_a_never_announced_part", g_hole_uploads_with_never}};
        uploads[GroupName(static_cast<Group>(g))] = std::move(one);
    }
    {
        nlohmann::json ledger = {{"notes", c_ledger_notes.load()}, {"overflow_skipped", c_ledger_overflow.load()}};
        {
            std::scoped_lock ledger_lock{g_ledger_mutex};
            ledger["ranges"] = g_ledger.size();
        }
        static constexpr const char* StoreNames[4] = {"equal", "empty_equal", "differs", "hole"};
        for (u32 l = 0; l < LedgerClassCount; ++l) {
            for (u32 c = 0; c < 4; ++c) {
                if (g_ledger_uploads[l][c] != 0) {
                    ledger["cpu_uploads_by_ledger_class_and_store_class"][LedgerNames[l]][StoreNames[c]] = {{"uploads", g_ledger_uploads[l][c]}, {"image_bytes", g_ledger_bytes[l][c]}};
                }
            }
        }
        ledger["examples"] = g_ledger_examples;
        out["write_ledger"] = std::move(ledger);
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
