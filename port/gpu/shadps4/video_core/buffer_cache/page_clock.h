// SPDX-License-Identifier: GPL-2.0-or-later
// Producer-side announce state, step S1 (documents/tlg/native plan/producer-side-state-design.md): a page clock kept in shadow beside the dirty sets. `TLG_PAGE_CLOCK=1`.
//
// Every announced byte range that enters the dirty sets (StampedRangeSet::Add) stamps its 4 KiB pages with a growing sequence number. `SynchronizeMemory` then simulates what the clock's
// consumer would do for the same request (upload the pages whose stamp is newer than their last sync) and counts bytes and copies against what the old path uploaded; the pages the clock
// calls clean are checked against the memory itself (a hash taken when the clock synced them): a changed page that the clock calls clean is a missed stamp (MUST_BE_0).

#pragma once

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


// ---- S2: the page filter for the GPU-written set (shadow). A page bit is set when `gpu_modified_ranges` gains bytes of the page and cleared only when the set no longer holds any byte of it, so
// "no bit set over a range" must imply "the set holds nothing there" (MUST_BE_0 in the counters); a set bit over a range with nothing in it is a counted false positive.
void GpuPagesSet(VAddr base, u64 size) noexcept;
void GpuPagesClearFree(const RangeSet& set, VAddr base, u64 size) noexcept;
bool GpuPagesMaybe(VAddr base, u64 size) noexcept;

/// The GPU-written set: a RangeSet that keeps the page filter in step.
struct GpuRangeSet : RangeSet {
    void Add(VAddr base_address, size_t size) {
        RangeSet::Add(base_address, size);
        if (g_page_clock_on) [[unlikely]] {
            GpuPagesSet(base_address, size);
        }
    }
    void Subtract(VAddr base_address, size_t size) {
        RangeSet::Subtract(base_address, size);
        if (g_page_clock_on) [[unlikely]] {
            GpuPagesClearFree(*this, base_address, size);
        }
    }
};

} // namespace VideoCore
