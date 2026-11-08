// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Plan N+T, step N2 (documents/tlg/native plan/native-path-and-threads-plan.md): a mirror of the frame ring that only the announced bytes keep current.
//
// Today every read of a ring window uploads the whole window (once per submission): 438 MB/s uploaded against 130 MB/s announced (N0). The mirror would upload the ring once and then
// only the bytes the engine announced. That is only correct if EVERY write into the ring is announced, and a missed one is stale data in a compute shader (the GPU hang class of
// reference/gpu-driver-crashes.md). So this step starts as a shadow: a copy of the ring in host memory that is kept current exactly as the mirror would be (first fill, then the
// announced bytes of every window read), and every window read is compared with it. A byte that differs is a write nobody announced: `unannounced_bytes` must be 0.
//
// `TLG_RING_MIRROR`: 0 off (default), 2 shadow (the uploads stay as they are; only counts), 3 shadow plus the writer probe: the pages of blocks that stayed different for 3 s without an
// announcement are made read only for one write; the fault handler records the instruction that wrote (code address in the game's image numbering) and gives the page back.

#pragma once

#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

namespace Core::RingMirror {

[[nodiscard]] u32 Mode() noexcept;
/// Mode 1 is the real mirror (`TLG_RING_MIRROR=1`): a ring window is uploaded whole the first time it is read, afterwards only the announced bytes. `TLG_RING_MIRROR_AUDIT=1` keeps the host shadow
/// beside it and uploads every block the shadow finds different (test runs: the compare costs about what the saved copy does).
[[nodiscard]] bool AuditOn() noexcept;
/// The shadow bookkeeping (announcement and writer tags, host writes) runs in the shadow modes and in the audited real mode.
[[nodiscard]] bool ShadowOn() noexcept;
/// The real mirror served a window: `uploaded_whole` bytes were read for the first time, `announced` bytes came from announcements; the rest of the window needed no upload.
void NoteReal(u64 window, u64 uploaded_whole, u64 announced, u64 pieces_in, u64 pieces_out, u64 merged_bytes);
/// Announced pieces closer than this many bytes are uploaded as one copy (`TLG_RING_MERGE`, default 2048, 0 off): a copy has a fixed cost, the gap bytes are current guest memory.
[[nodiscard]] u64 MergeGap() noexcept;

/// True once for every definition of the ring [start, end): the caller then drops all marks over the ring (they predate the fill) and calls `Fill`.
[[nodiscard]] bool NeedFill(VAddr start, VAddr end);
void Fill(VAddr start, VAddr end);
/// The window [lo, hi) of the ring is read now. `announced`: the announced (marked, not yet taken) ranges inside it, start and size. The shadow takes them, then the window is compared with the shadow.
/// `repair` (audited real mode): receives the ranges that differ, merged; the caller uploads them.
void CheckWindow(VAddr lo, VAddr hi, const std::pair<u64, u64>* announced, size_t count, std::vector<std::pair<u64, u64>>* repair = nullptr);

/// An announcement was applied for [address, size): blocks that were found different and are inside it are explained (announced late: the window was read between the write and its announcement).
void NoteAnnounced(VAddr address, u64 size);
/// At announcement time (the writing thread): the writer tag (writer id, +8 without a site name) of the ring blocks the range covers, remembered per block to classify the blocks that are never announced.
void NoteWriter(VAddr address, u64 size, u32 tag);
/// The port itself wrote guest memory (the command processor's WriteData labels, fills): not a game write, nothing to announce, and no shader reads it. The shadow takes the bytes when a window that holds them is read.
void NoteHostWrite(VAddr address, u64 size);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::RingMirror
