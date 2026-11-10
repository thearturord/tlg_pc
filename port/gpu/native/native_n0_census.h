// SPDX-License-Identifier: GPL-2.0-or-later
// Plan N+T, step N0 (documents/tlg/native plan/native-path-and-threads-plan.md): the census of the emulation-shaped data path. Counters only, no behaviour change.
//
// Off by default (`TLG_N0_CENSUS=1`). It answers three questions on the raster thread:
//   1. SynchronizeMemory: per caller site, how many calls, how much of the time is spent in calls that upload nothing (pure guessing) and how much in calls that copy.
//   2. CopySparseMemory (the page walk of the guest memory): per caller site, calls, bytes, time, and how many copies carry exactly the bytes the previous copy at the same
//      address carried (an announced write that changed nothing).
//   3. The announcements: sizes of the applied ranges (a cheap size histogram; the writer split is in /native).
// The report is published as `n0_census` (cumulative, with the seconds since the first count; take two reads for rates).

#pragma once

#include <atomic>
#include <chrono>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

namespace Core::N0 {

enum SyncSite : u32 {
    SyncImmutable = 0, ///< an engine buffer synchronized whole (not resident yet)
    SyncArenaRead,     ///< a read-only request that goes through the arena / placed buffer
    SyncArenaWritten,  ///< a request the shader writes
    SyncArenaTexel,    ///< a texel buffer request
    SyncDownload,      ///< ObtainBufferForDownload
    SyncPre,           ///< announced uploads before the draws (off by default)
    SyncSiteCount,
};

enum CopySite : u32 {
    CopyObject = 0,   ///< the once-per-generation copy of a whole engine buffer into the stream ring
    CopyStream,       ///< a small read-only range copied into the stream ring
    CopyStreamFn,     ///< StreamCopy() while a pass is open
    CopyForImage,     ///< bytes read for an image upload
    CopyZeroDirect,   ///< SynchronizeMemory, straight into video memory
    CopyStaging,      ///< SynchronizeMemory, through a staging buffer
    CopySiteCount,
};

extern bool g_on;
extern thread_local SyncSite t_sync_site;
/// What one applied announcement did that a later draw could observe (N1 census). Each place that changes shared state for a range sets its bit.
enum Effect : u32 { EffMemo = 1, EffRecipe = 2, EffGpuRanges = 4, EffResident = 8, EffBufferParts = 16, EffImage = 32, EffUnattributed = 64 }; ///< 64: F1 1c, the apply marked bytes no engine buffer holds
extern thread_local u32 t_effect;
extern thread_local bool t_ring_candidate; ///< F1 1c: set by the batch around the apply of a range tagged as a ring candidate
extern thread_local bool t_force_effect; ///< F1 1c real: the sampled full apply records its effects even with the census off
extern thread_local u32 t_writer; ///< announce batch census: the writer tag of the range being applied (0-7, 8-15 the same writers without a site name; 16 unknown)
inline void SetEffect(u32 bit) noexcept {
    if (g_on || t_force_effect) {
        t_effect |= bit;
    }
}
extern thread_local u64 t_hash_ns; ///< time the census itself spent hashing on this thread (taken out of the sync times)

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

[[nodiscard]] inline u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void NoteSync(SyncSite site, u64 requested, u64 uploaded, u64 ns) noexcept;
void NoteCopy(CopySite site, u64 address, u64 size, u64 ns, const void* data) noexcept;
void NoteApplied(u64 size, bool again) noexcept;
/// Why bytes were uploaded: 0 the frame ring part is uploaded whole (nobody announced what changed), 1 the announced bytes of an engine buffer, 2 announced bytes of memory no engine buffer holds.
void NoteCause(u32 cause, u64 bytes) noexcept;
void NoteApply(bool ring, u64 size, u32 effects, u64 ns) noexcept;
/// F1 1c (who reads what an in-ring announcement leaves behind): a row check (`ObjectRangeDirty`) over a range inside the frame ring or outside it, its answer, and whether
/// the answer was "dirty" only because of the unattributed set (bytes no engine buffer held when they were announced).
void NoteRowCheck(bool in_ring, bool dirty, bool only_unattributed) noexcept;
/// F1 1c: a row the raster thread tried to learn, over a range inside the frame ring or outside it.
void NoteRowLearn(bool in_ring) noexcept;
/// F1 1c: the time of one step of `ApplyCpuWrite` for a ring candidate (0 memo erase, 1 GPU-written supersede, 2 resident note, 3 buffer marks, 4 image index).
void NoteCandidateStep(u32 step, u64 ns) noexcept;
/// Times one step of the apply when the census is on (ring candidates and, by writer, every other range: the announce batch census).
struct CandidateStep {
    u32 step;
    u64 start;
    explicit CandidateStep(u32 step_) noexcept : step(step_), start(0) {
        if (g_on) {
            start = NowNs();
        }
    }
    ~CandidateStep() {
        if (start != 0) {
            NoteCandidateStep(step, NowNs() - start);
        }
    }
};
/// F1 1 finish (ring writes out of the batch), census: what the steps a ring candidate still runs in the batch protect. Counters only.
enum RingOut : u32 {
    OutImageDirtied = 0,          ///< a ring candidate dirtied an image (CpuDirty)
    OutImageRefreshOnlyCand,      ///< an image was refreshed whose only dirtying since its last refresh came from ring candidates (MUST_BE_0 to drop the image step)
    OutImageRefreshOnlyCandBytes, ///< ... its bytes
    OutImageRefreshCandAndOther,  ///< refreshed, dirtied by a candidate and by something else
    OutImageKeptForGpu,           ///< a ring candidate met an image the GPU wrote (kept)
    OutClearErased,               ///< a ring candidate erased a waiting clear (MUST_BE_0 to drop the image step)
    OutMemoMissOnlyCand,          ///< an upload memo miss whose only cause was the epoch ticks of ring candidate applies
    OutMemoMissOnlyCandUploaded,  ///< ... after which the obtain uploaded bytes (MUST_BE_0 to drop the tick)
    OutMemoMissOnlyCandBytes,     ///< ... those bytes
    OutAgainUnchanged,            ///< the second application of a ring candidate: bytes unchanged (TLG_REAPPLY_AUDIT=1)
    OutAgainChangedAnnounced,     ///< changed, and a newer announcement covers it
    OutAgainChangedUnannounced,   ///< changed, nothing newer announced it (MUST_BE_0 to drop the second application)
    OutCount
};
void NoteRingOut(RingOut what, u64 n = 1) noexcept;
/// The sum of the buffer cache's memo epochs that ring candidate applies added (raster thread).
extern std::atomic<u64> g_cand_epoch;
void NoteFromImage(u64 ns, bool tiled) noexcept;
/// Announce batch census: the hash the batch takes of a range of `writer` (first list: the gate's next comparison; again list: the comparison itself).
void NoteBatchHash(u32 writer, bool again, u64 ns) noexcept;
[[nodiscard]] nlohmann::json Report();

/// Times one SynchronizeMemory call. `uploaded` is the thread's upload counter (it is reset by the caller before the call), read again at the end.
struct SyncScope {
    SyncSite site;
    u64 requested;
    const u64& uploaded;
    u64 before;
    u64 start;
    u64 hash_before;
    SyncScope(u64 requested_, const u64& uploaded_) noexcept : site(t_sync_site), requested(requested_), uploaded(uploaded_), before(uploaded_), start(g_on ? NowNs() : 0), hash_before(t_hash_ns) {}
    ~SyncScope() {
        if (start != 0) {
            NoteSync(site, requested, uploaded - before, NowNs() - start - (t_hash_ns - hash_before));
            t_sync_site = SyncArenaRead;
        }
    }
};

/// Times SynchronizeMemoryFromImage (the texel buffer path: a buffer whose memory is an image).
struct FromImageScope {
    u64 start;
    bool tiled{false};
    FromImageScope() noexcept : start(g_on ? NowNs() : 0) {}
    ~FromImageScope() {
        if (start != 0) {
            NoteFromImage(NowNs() - start, tiled);
        }
    }
};

/// One ApplyCpuWrite. `ring`: the range lies in the frame ring window.
struct ApplyScope {
    bool ring;
    u64 size, start;
    ApplyScope(bool ring_, u64 size_) noexcept : ring(ring_), size(size_), start(g_on ? NowNs() : 0) {
        t_effect = 0;
    }
    ~ApplyScope() {
        if (start != 0) {
            NoteApply(ring, size, t_effect, NowNs() - start);
        }
    }
};

/// Sets the caller site of the next SynchronizeMemory call on this thread.
inline void SetSyncSite(SyncSite site) noexcept {
    if (g_on) {
        t_sync_site = site;
    }
}

/// Times one CopySparseMemory call; `Done(data)` after the copy lets the census hash the bytes.
struct CopyScope {
    CopySite site;
    u64 address, size, start;
    const void* data{nullptr};
    CopyScope(CopySite site_, u64 address_, u64 size_) noexcept : site(site_), address(address_), size(size_), start(g_on ? NowNs() : 0) {}
    void Done(const void* data_) noexcept {
        data = data_;
    }
    ~CopyScope() {
        if (start != 0) {
            NoteCopy(site, address, size, NowNs() - start, data);
        }
    }
};

} // namespace Core::N0
