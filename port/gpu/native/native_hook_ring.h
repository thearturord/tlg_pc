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
bool Raster1bOn() noexcept;   ///< TLG_HOOK_RING_1B=1 (shadow) or 2 (real)
/// F1 1b real (`TLG_HOOK_RING_1B=2`, default off): the raster thread binds texel ranges and reads up to 64 KiB of the frame ring straight from the hook ring
/// (BufferCache::ObtainBuffer) instead of uploading them from game memory; 1 in 64 is compared with game memory. Every writer fills in this mode.
bool Raster1bReal() noexcept;
void NoteRasterBound(u32 cls, u64 size, bool checked);
void CompareRaster(u32 cls, u64 address, u64 size, bool gpu_written);
/// A GPU result was written into game memory at [address, address + size) (readbacks, image downloads, fence and label writes): the 1b shadow files a later
/// difference over it under "a GPU result landed since the fill".
void NoteLanding(u64 address, u64 size);
/// 1b shadow: the Map window announced next on this thread (how: 0 Unmap, 1 replaced by a second Map, 2 after a hooked draw call, 3 at the next frame start
/// because it was never unmapped); fills of writer 0 keep it per granule, and a later difference reports it. `ClearFillTag` after the announcement.
void SetFillTag(u32 how, u32 caller_ghidra, u64 resource_id, u32 map_type);
void ClearFillTag();
void NoteRasterWide(u64 size);
/// F1 wide windows census (`TLG_WIDE_CENSUS=1`, counters only; design review section 39): a 4 MB window the old path uploads whole, compared with the hook ring granule by granule.
/// Filled granules that differ (a write after the fill nothing announced), granules never filled (silent or not written yet), and the filled prefix from the window's start
/// (the Map's block). `probe`: the address of the first differing byte in the filled part is made read only once (writer probe) when nonzero.
[[nodiscard]] bool WideCensusOn() noexcept;
void CompareWide(u64 address, u64 size, bool probe);
/// F1 wide windows (`TLG_WIDE_ROWS`, default 2 since 2026-11-10, 0 off; design review sections 41 to 43): 1 shadow (every window the old path uploads is checked: the buffer's own block in the hook ring
/// against game memory; the old path decides), 2 real (a 4 MB read window inside the frame ring is bound from the hook ring when its block is filled in one lap half; one in 16
/// is compared first and a window whose block differs takes the old path). With either mode a GPU result that lands in the ring is copied into the hook ring.
[[nodiscard]] u32 WideRowsMode() noexcept;
/// The offset in the storage of the window [address, address + size) when its filled prefix (the block from its start) lies in one lap half; `prefix` its length.
bool ResolveWide(u64 address, u64 size, u64& offset, u64& prefix);
/// The same for the bind, in constant time: the window's first granule filled; its lap half gives the offset (the block's length is not walked).
bool ResolveWideFast(u64 address, u64 size, u64& offset);
/// The bytes of the window's filled prefix that differ from game memory now.
u64 ComparePrefix(u64 address, u64 size);
/// Counters of the stage: 0 asked, 1 outside the ring or GPU written, 2 no prefix, 3 prefix in two halves, 4 resolved, 5 bound, 6 compared, 7 PREFIX_DIFFERS (MUST_BE_0; not bound),
/// 8 shadow windows checked, 9 SHADOW_PREFIX_DIFFERS (MUST_BE_0), 10 landings copied into the hook ring.
void NoteWideRows(u32 what) noexcept;

nlohmann::json Report();

} // namespace Core::HookRing
