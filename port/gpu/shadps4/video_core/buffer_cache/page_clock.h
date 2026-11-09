// SPDX-License-Identifier: GPL-2.0-or-later
// Producer-side announce state, step S1 (documents/tlg/native plan/producer-side-state-design.md): a page clock kept in shadow beside the dirty sets. `TLG_PAGE_CLOCK=1`.
//
// Every announced byte range that enters the dirty sets (StampedRangeSet::Add) stamps its 4 KiB pages with a growing sequence number. `SynchronizeMemory` then simulates what the clock's
// consumer would do for the same request (upload the pages whose stamp is newer than their last sync) and counts bytes and copies against what the old path uploaded; the pages the clock
// calls clean are checked against the memory itself (a hash taken when the clock synced them): a changed page that the clock calls clean is a missed stamp (MUST_BE_0).

#pragma once

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {

extern bool g_page_clock_on;
void PageClockStampSlow(VAddr base, u64 size) noexcept;
inline void PageClockStamp(VAddr base, u64 size) noexcept {
    if (g_page_clock_on) [[unlikely]] {
        PageClockStampSlow(base, size);
    }
}

/// A RangeSet whose Add also stamps the page clock (so no mark site of the dirty sets can be forgotten).
struct StampedRangeSet : RangeSet {
    void Add(VAddr base_address, size_t size) {
        PageClockStamp(base_address, size);
        RangeSet::Add(base_address, size);
    }
};


// ---- S3b.2 (shadow): placement events and the query comparison of the per-buffer clock.
struct Buffer;
struct ClockInsideSource {
    const Buffer* buffer; ///< a placed buffer the new one takes over (its contents come over as they are)
    VAddr begin, end;
};
/// A placed buffer was created: the pages it copied from memory (`rest`) are synced now, the pages it took over from older buffers keep those buffers' sync stamps.
void PageClockBufferCreated(const Buffer* dst, VAddr begin, VAddr end, const std::vector<std::pair<u64, u64>>& rest, const std::vector<ClockInsideSource>& inside) noexcept;
/// `old_dirty`: what the dirty sets say about [addr, addr + size) for this buffer; the clock's answer is computed and compared (counters only).
void PageClockQueryShadow(const Buffer* buffer, VAddr addr, u64 size, bool old_dirty) noexcept;

// ---- S2/S3a: the page filter for the GPU-written set. A page bit is set when `gpu_modified_ranges` gains bytes of the page and cleared only when the set no longer holds any byte of it, so
// "no bit set over a range" implies "the set holds nothing there" (MUST_BE_0 in the counters); a set bit over a range with nothing in it is a counted false positive.
// The table is lock-free for readers (two levels: 4096 chunk pointers, each chunk 1024 words = 65,536 pages = 256 MiB of guest memory); writers are the callers of the set, which already hold
// `gpu_ranges_mutex`. Addresses at or above 2^40 are not tracked and always answer "maybe".
extern bool g_gpu_filter_bits; ///< the bits are kept (TLG_PAGE_CLOCK=1 or TLG_GPU_FILTER=1)
extern bool g_gpu_filter_skip; ///< TLG_GPU_FILTER=1: SupersedeGpuWritten returns early where the filter says none

struct GpuPageTable {
    std::atomic<std::atomic<u64>*> chunk[4096]{};
};
extern GpuPageTable g_gpu_page_table;

void GpuPagesSet(VAddr base, u64 size) noexcept;
void GpuPagesClearFree(const RangeSet& set, VAddr base, u64 size) noexcept;

inline bool GpuPagesMaybe(VAddr base, u64 size) noexcept {
    if (size == 0) {
        return false;
    }
    const u64 first = base >> 12, last = (base + size - 1) >> 12;
    if ((last >> 28) != 0) {
        return true;
    }
    u64 page = first;
    while (page <= last) {
        const std::atomic<u64>* words = g_gpu_page_table.chunk[page >> 16].load(std::memory_order_acquire);
        const u64 stop = std::min<u64>(last + 1, ((page >> 16) + 1) << 16);
        if (words == nullptr) {
            page = stop;
            continue;
        }
        while (page < stop) {
            const u64 word_end = std::min<u64>(stop, (page | 63) + 1);
            const u64 lo = page & 63, hi = lo + (word_end - page); // bits [lo, hi) of this word
            const u64 mask = (hi >= 64 ? ~u64{0} : ((u64{1} << hi) - 1)) & ~((u64{1} << lo) - 1);
            if ((words[(page >> 6) & 1023].load(std::memory_order_relaxed) & mask) != 0) {
                return true;
            }
            page = word_end;
        }
    }
    return false;
}

/// The GPU-written set: a RangeSet that keeps the page filter in step.
struct GpuRangeSet : RangeSet {
    void Add(VAddr base_address, size_t size) {
        RangeSet::Add(base_address, size);
        if (g_gpu_filter_bits) [[unlikely]] {
            GpuPagesSet(base_address, size);
        }
    }
    void Subtract(VAddr base_address, size_t size) {
        RangeSet::Subtract(base_address, size);
        if (g_gpu_filter_bits) [[unlikely]] {
            GpuPagesClearFree(*this, base_address, size);
        }
    }
};

} // namespace VideoCore
