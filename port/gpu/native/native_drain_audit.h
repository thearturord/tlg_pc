// SPDX-License-Identifier: GPL-2.0-or-later
// N6 (documents/tlg/native-renderer-plan.md): the command processor's wait for queued memory writes.
//
// Before it plans a draw or a dispatch the command processor thread waits until the raster thread has run every queued closure that writes guest memory
// (`Liverpool::DrainRasterWrites`), because the planning reads guest memory (shader code, vertex descriptor tables, the bytes the constant ring copies). That wait was 21% of
// the command processor's time in the light scene. Every queued write has a destination that is known when it is queued.
//
// N6a, the audit (`TLG_DRAIN_AUDIT=1`, report at /drain): records the ranges of the queued writes and every guest read the planning makes, and for every planning that had
// to wait it says whether any read touched a pending write. Result in the light scene: none did. The wait still happens.
//
// N6b, the lazy wait (on by default, `TLG_DRAIN_LAZY=0` turns it off): the wait before the planning is gone. Every recorded read (`Read`) checks the writes that are still pending and waits, only then, until
// the newest overlapping one has run (the raster thread runs its queue in order). A read the planning makes without recording it is a read this cannot protect, which is
// why the verify mode exists (`TLG_DRAIN_VERIFY=N`, default 0 = off: one planning in N is checked, alternately two ways):
//   bytes: the planning runs lazily; at its end the old full wait is done, and every range the constant or geometry ring copied is compared with guest memory again;
//   plan:  the planning runs lazily, then the full wait is done, the planning runs again and the two plans (pipeline, stages, user data, flat user data) are compared.
// A difference is counted and logged; the reports are in /drain under "lazy".
//
// The reads it knows are the ones listed in `Kind`. The SRT walker reads memory through generated code; its reads are approximated by a window after every pointer-like
// pair in the stage's user data (the plan check covers what that approximation misses).

#pragma once

#include <atomic>
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::DrainAudit {

enum Kind : u32 {
    ReadCode = 0,      ///< the shader binary (hash lookup of a stage)
    ReadWalker = 1,    ///< the flat user data walker (approximate windows after every pointer in the user data)
    ReadFetch = 2,     ///< a vertex descriptor (V#) read through the fetch shader's table
    ReadRingConstant = 3, ///< state and bytes of a range the constant ring considers
    ReadRingVertex = 4,   ///< the same for vertex buffers (geometry ring)
    ReadRingIndex = 5,    ///< the same for the index buffer
    KindCount = 6,
};

/// Written by a closure, but not to guest memory (a GDS counter): overlaps nothing.
constexpr u64 NoGuestMemory = ~0ull;

/// The audit is on (TLG_DRAIN_AUDIT=1).
[[nodiscard]] bool Enabled();
/// The lazy wait is on (the default; TLG_DRAIN_LAZY=0 turns it off).
[[nodiscard]] bool LazyEnabled();
/// One planning in this many is verified (TLG_DRAIN_VERIFY, default 0 = never; 64 was tried and made the frame rate uneven).
[[nodiscard]] u32 VerifyEvery();
/// The ranges of queued writes are kept (audit or lazy wait).
[[nodiscard]] bool Active();

/// The raster queue: `done` counts the operations it has finished (in batches), `write_done` is the queue position of the newest write closure it has finished (one by one),
/// `wait_for(position)` blocks until either has reached `position`. Without a queue (null) nothing is tracked.
void SetBackend(const std::atomic<u64>* done, const std::atomic<u64>* write_done, std::function<void(u64 position)> wait_for);

/// The next queued write closure of this thread writes [address, address + bytes). Without a call the range is unknown, and an unknown range overlaps everything.
void SetNextWrite(u64 address, u64 bytes);
/// The next queued closure writes no guest memory that is tracked (forget a range set for another one).
void ClearNextWrite();
/// A write closure was queued at queue position `position`; the range is the one set by SetNextWrite (consumed here).
void WriteQueued(u64 position, u32 line);

// ---- audit
/// The command processor is about to wait for the writes still queued (`done` operations have finished).
void DrainBegin(u64 done, u32 caller);
void DrainNothingPending();
void DrainWaited(u64 ns);

// ---- the planning's guest reads
/// A guest read by the planning on this thread. Audit: compared with the pending writes. Lazy: waits if one of them overlaps.
void Read(Kind kind, u64 address, u64 bytes);
/// The user data of a stage went through the walker: the windows of its pointers.
void WalkerRead(const u32* user_data, u32 count);
/// The planning copied `bytes` bytes of guest memory [address, ...) into `copy` (a ring): remembered in a bytes verification.
void Copied(Kind kind, u64 address, const void* copy, u64 bytes);

// ---- lazy wait and its verification
enum class Verify : u32 { None, Bytes, Plan };
/// A planning begins (the place of the old wait). Lazy: counts it and decides whether it is verified; for `Plan` the caller plans, waits, plans again and reports with NotePlanVerify.
[[nodiscard]] Verify LookupBegin(u32 site);
/// The plans of a verified lookup were compared.
void NotePlanVerify(u32 site, bool same);
/// The planning of one operation is done and the operation is being queued: finishes a bytes verification, classifies an audited wait.
void LookupDone();

nlohmann::json Report();

} // namespace Core::DrainAudit
