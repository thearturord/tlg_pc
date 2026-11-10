// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// F1 class 1a (design review 2026-11-09 sections 2 and 9 to 14). `TLG_HOOK_RING`, default 2 (real) since 2026-11-10 after the owner's play; 0 = off.
///   1 = shadow: a host copy of the 64 MB frame ring, filled on the writing thread at the moment a write is known to be finished (every announcement inside
///       the ring, the light and slot ring copies, the recording and skinning ring blocks when the engine's writing call returns). Every front constant copy
///       is compared with it. Nothing the GPU reads changes.
///   2 = real: the same copy lives in GPU-visible memory (`SetStorage`, the rasterizer's buffer), and the front binds a constant range straight from it
///       (`Resolve`) instead of copying it into the constant ring; 1 in 64 of those bindings are compared with game memory. Ranges not wholly filled take
///       the old path.
/// The copy has two halves: a block is written into the half of the allocator's lap it was handed out in (lap parity), so the GPU can still read lap N
/// while the engine writes lap N + 1 (a lap is 4 to 7 frames in the three scenes measured).
namespace Core::HookRing {

bool On() noexcept;     ///< mode 1 or 2
bool RealOn() noexcept; ///< mode 2

/// Mode 2: the GPU-visible storage (at least twice the ring size) and how to make written bytes visible to the GPU.
void SetStorage(u8* data, u64 capacity, void (*flush)(void*, u64, u64), void* flush_context);

/// The frame allocator handed out [block, block + size): its granules are not filled any more (the next write fills them). Called under the allocator lock.
void NoteAlloc(u64 block, u64 size);
/// Whether announcements of this writer fill the copy. Mode 1 (shadow): every writer. Mode 2: writer 3 only (the constant buffer copies), because the front binds
/// constants only from writer 3 and from our own redirects (ring census, section 9); the Map windows of writer 0 are 65 to 81 % of the announced bytes and the
/// front never binds them. `TLG_HOOK_RING_WRITERS=<bit mask>` overrides (bit n = writer n).
bool FillsWriter(u32 writer) noexcept;
/// A write to [address, address + size) is finished: the bytes are copied into the ring copy now (only the part inside the frame ring).
void Fill(u64 address, u64 size, u32 writer = 8);
/// The front thread copied [address, address + size) of game memory into `copied`: compare with the ring copy.
void Compare(u64 address, u64 size, const void* copied);
/// Mode 2: the offset in the storage of [address, address + size) when every byte of it was filled in one lap half; false otherwise.
bool Resolve(u64 address, u64 size, u64& offset);
/// The bytes at `offset` of the ring copy (an offset `Resolve` returned).
const u8* Data(u64 offset);
/// Mode 2: the front bound a range from the storage (`checked`: it was also compared with game memory).
void NoteBound(u64 size, bool checked);

/// F1 1b shadow (`TLG_HOOK_RING_1B=1`, default off; needs every writer filling: `TLG_HOOK_RING_WRITERS=0xff`): the raster thread's ring uploads compared with
/// the hook ring. `cls`: 0 read-only binding up to 64 KiB, 1 texel binding. The wide read bindings (above 64 KiB, the 4 MB dynamic buffers of section 12) stay
/// on the old path by design and are only counted (`NoteRasterWide`). The source bytes are game memory now (what the old path uploads).
bool Raster1bOn() noexcept;
void CompareRaster(u32 cls, u64 address, u64 size, bool gpu_written);
/// A GPU result was written into game memory at [address, address + size) (readbacks, image downloads, fence and label writes): the 1b shadow files a later
/// difference over it under "a GPU result landed since the fill".
void NoteLanding(u64 address, u64 size);
void NoteRasterWide(u64 size);

nlohmann::json Report();

} // namespace Core::HookRing
