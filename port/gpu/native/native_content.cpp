// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
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
u64 g_seq = 0;
std::unordered_set<u64> g_uploaded_images;

std::atomic<u64> c_announced{0}, c_announced_bytes{0}, c_skip_writer{0}, c_skip_big{0}, c_skip_unreadable{0}, c_evicted{0}, c_evicted_bytes{0}, c_replaced{0};
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
u64 g_equal_bytes_by_writer[16]{};
u64 g_equal_nonzero_by_writer[16]{};  ///< bytes of CPU sourced uploads that equal the store, by the writer of the entry (15: a copy)
u64 g_uploads_by_main_writer[16]{}; ///< uploads by the writer that supplies most of their covered bytes
std::vector<nlohmann::json> g_examples_differs, g_examples_hole;

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
        const char* e = std::getenv("TLG_F2_STORE_MB");
        const u64 mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 0;
        return (mb != 0 ? mb : 1024ull) << 20;
    }();
    return cap;
}

/// Inserts an entry under the lock: older entries it covers completely are dropped, the oldest entries go when the cap is reached.
void InsertLocked(u64 begin, Entry&& entry) {
    const u64 end = begin + entry.size;
    for (auto it = g_entries.lower_bound(begin); it != g_entries.end() && it->first < end;) {
        if (it->first + it->second.size <= end) {
            g_stored_bytes -= it->second.writer == WriterCopy ? 0 : it->second.size;
            c_replaced.fetch_add(1, std::memory_order_relaxed);
            it = g_entries.erase(it);
        } else {
            ++it;
        }
    }
    entry.seq = ++g_seq;
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
void ResolveLocked(u64 begin, u64 end, std::vector<Piece>& pieces, std::vector<std::pair<u64, u64>>& uncovered) {
    std::vector<Piece> candidates;
    const u64 from = begin > g_max_size ? begin - g_max_size : 0;
    for (auto it = g_entries.lower_bound(from); it != g_entries.end() && it->first < end; ++it) {
        if (it->first + it->second.size > begin) {
            candidates.push_back({std::max(begin, it->first), std::min(end, it->first + it->second.size), &it->second, it->first});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Piece& a, const Piece& b) { return a.entry->seq > b.entry->seq; });
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

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F2_STORE");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void Announce(u32 writer, u64 address, u64 size) {
    if (!On() || address == 0 || size == 0) {
        return;
    }
    if (writer == 1 || writer == 2 || writer == 4) {
        c_skip_writer.fetch_add(1, std::memory_order_relaxed); // announced before the write: no content yet, and buffers only
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
    if (!On() || size == 0) {
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

void NoteDeleted(u64, u64 size) {
    if (!On()) {
        return;
    }
    c_deleted.fetch_add(1, std::memory_order_relaxed);
    c_deleted_bytes.fetch_add(size, std::memory_order_relaxed);
}

void CompareImage(const ImageUpload& upload) {
    if (!On() || upload.size == 0) {
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
    u64 equal = 0, differing = 0, first_diff = ~0ull, newest_differing_seq = 0;
    u64 bytes_by_writer[16]{}, nonzero_by_writer[16]{};
    u32 differing_writer = 0;
    bool unreadable = false;
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
            const u8* m = reinterpret_cast<const u8*>(p.begin);
            const u8* s = p.entry->Bytes() + (p.begin - p.entry_begin);
            for (u64 i = diff_at; i < p.end - p.begin; ++i) {
                d += m[i] != s[i];
            }
            differing += d;
            equal += (p.end - p.begin) - d;
            if (p.begin + diff_at - upload.address < first_diff) {
                first_diff = p.begin + diff_at - upload.address;
            }
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
            e["announcements_ago"] = g_seq - newest_differing_seq;
        }
        (cls == ClassDiffers ? g_examples_differs : g_examples_hole).push_back(std::move(e));
    }
}

nlohmann::json Report() {
    nlohmann::json out;
    out["enabled"] = On();
    if (!On()) {
        out["switch"] = "TLG_F2_STORE=1";
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
    out["uploads_with_unreadable_memory"] = g_unreadable;
    out["examples_differs"] = g_examples_differs;
    out["examples_hole"] = g_examples_hole;
    out["note"] = "F2.2 shadow: bytes taken when the writer announced them (newest wins) against the memory the image upload read; differs = a covered byte changed after the announcement or came from a later write; hole = bytes no announcement covers hold data";
    return out;
}

} // namespace Core::NativeContent
