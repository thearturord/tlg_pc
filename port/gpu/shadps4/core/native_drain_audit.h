// SPDX-License-Identifier: GPL-2.0-or-later
// N6a (documents/tlg/native-renderer-plan.md): the drain audit. Before it plans a draw or a dispatch the command processor thread waits until the raster thread has run every
// queued closure that writes guest memory (`Liverpool::DrainRasterWrites`), because the planning reads guest memory (shader code, vertex descriptor tables, the bytes the
// constant ring copies). That wait was 21% of the command processor's time in the light scene. Every queued write has a destination that is known when it is queued, so the
// wait could be limited to the writes that touch what the planning reads. This audit decides whether that is true: it records the ranges of the queued writes and every
// guest read the planning makes, and for every planning that had to wait it says whether any read touched a pending write (and which kind of read, by which kind of write).
// It changes nothing: the wait still happens. `TLG_DRAIN_AUDIT=1`, report at /drain.
//
// The reads it knows are the ones listed in `Kind`. A read the audit does not know is a read it cannot see: the result "no overlap" is only as good as that list, which is why
// the report says which kinds were recorded and how often. The SRT walker reads memory through generated code; its reads are approximated by a window after every pointer-like
// pair in the stage's user data (marked as approximate).

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::DrainAudit {

enum Kind : u32 {
    ReadCode = 0,      ///< the shader binary (hash lookup of a stage)
    ReadWalker = 1,    ///< the flat user data walker (approximate windows after every pointer in the user data)
    ReadFetch = 2,     ///< a vertex descriptor (V#) read through the fetch shader's table
    ReadRingConstant = 3, ///< bytes the constant ring copies for a buffer binding
    ReadRingVertex = 4,   ///< bytes the geometry ring copies for vertex buffers
    ReadRingIndex = 5,    ///< bytes the geometry ring copies for the index buffer
    KindCount = 6,
};

/// Written by a closure, but not to guest memory (a GDS counter): overlaps nothing.
constexpr u64 NoGuestMemory = ~0ull;

[[nodiscard]] bool Enabled();
/// The next queued write closure of this thread writes [address, address + bytes). Without a call the range is unknown, and an unknown range overlaps everything.
void SetNextWrite(u64 address, u64 bytes);
/// The next queued closure writes no guest memory that the audit tracks (forget a range set for another one).
void ClearNextWrite();
/// A write closure was queued at queue position `position`; the range is the one set by SetNextWrite (consumed here).
void WriteQueued(u64 position, u32 line);
/// The command processor is about to wait for the writes still queued (`done` operations have finished); `pending` writes are not finished.
void DrainBegin(u64 done, u32 caller);
/// Nothing was pending (counted for the share).
void DrainNothingPending();
/// The wait took `ns`.
void DrainWaited(u64 ns);
/// A guest read by the planning on this thread.
void Read(Kind kind, u64 address, u64 bytes);
/// The user data of a stage went through the walker: the windows of its pointers.
void WalkerRead(const u32* user_data, u32 count);
/// The planning of one operation is done (the operation is being queued): classify the planning that waited.
void Planned();

nlohmann::json Report();

} // namespace Core::DrainAudit
