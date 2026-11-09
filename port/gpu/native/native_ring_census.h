// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Ring coverage census (design review 2026-11-09, F1 class 1 precondition; `TLG_RING_CENSUS=1`, default off, counters only).
/// Question: of the frame ring bytes the GPU reads, how many were announced by a hook since the frame allocator handed the block out?
/// Today the ring is copied from game memory at every use, so a write no hook announces still reaches the GPU; a ring filled only at the hooks would lose it.
/// Every 16 bytes of the ring carry one byte of flags: bit 0 = handed out by the frame allocator hook (and the announcements cleared), bits 1 to 6 = announced
/// by writer 0, 1, 2, 3, 4, other, bit 7 = the block was taken by one of our redirects (our own copy fills it). At every use the census sorts the used bytes into announced, handed out but not announced (silent), and never handed out.
/// Nothing here changes what the GPU reads.
namespace Core::RingCensus {

bool On() noexcept;

/// The frame allocator handed out [block, block + size) (called under the allocator lock). `ours`: one of our own redirects (skinning, light, recording, slot rings)
/// took the block and copies into it itself; otherwise the engine asked for it (through the allocator hook).
/// `site`: the engine's call site (Ghidra address of the return address), 0 when unknown or ours.
void NoteAlloc(u64 block, u64 size, bool ours, u32 site);
/// A hook announced [address, address + size) (any thread).
void NoteAnnounce(u32 writer, u64 address, u64 size);

enum Site : u32 {
    SiteFrontConstant = 0, ///< the front thread copies a constant buffer range into the constant ring
    SiteFrontGeometry = 1, ///< the front thread copies vertex or index bytes into the constant ring
    SiteRasterSync = 2,    ///< the raster thread uploads a ring range at a binding (the old "uploaded on every use" path)
    SiteCount = 3,
};
/// The GPU will read [address, address + size) of game memory (only the part inside the frame ring counts).
void NoteUse(Site site, u64 address, u64 size);

/// The Map hook's path for one Map: 0 buffer (announced at Unmap), 1 texture with a native upload record, 2 texture, record refused (both announced at Unmap),
/// 3 not handled (the engine's function runs, nothing is announced). `kind` is the resource kind (+0x124), `subresource` the mip/slice index.
void NoteMapPath(u32 kind, u64 subresource, u32 map_type, u32 path);

nlohmann::json Report();

} // namespace Core::RingCensus
