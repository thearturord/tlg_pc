// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Design review notes 90 to 92, section 79 step 2 (`TLG_HOOK_OWNED`: 2 real by default since the owner's play, note 94, with 1 in 64 owned writes still announced as an audit; 1 shadow; 0 off, the old path): the light constant and scene slot buffers (writer tags 2 and 12)
/// become *hook-owned*. Their writes leave the announce batch; the address path treats their bytes as always changed (pinned dirty, no cached answer over them).
///
/// Membership in two steps. The hook proposes it, in game order: the first light or slot write of a buffer with no engine texture in its memory is announced marked "join".
/// The raster thread confirms it at that write's apply, in batch order, when no image or waiting clear of the texture cache and no GPU-written byte is in the buffer's memory
/// (shadow `ho1`: images left by deleted textures sit in some of these buffers, which the hook cannot see); otherwise it excludes the buffer. Writes stay announced until the
/// confirmation. Afterwards a texture created over a member excludes it (hook, game order); the raster thread drops members at a texture create or buffer delete event and
/// checks the guards: a GPU write or a registered image over a member must not happen.
///
/// The shadow announces everything as before and only counts: the writes the real mode would not announce, what their applies still did (an image touched must be 0), the guard
/// events, and the address requests that meet a member (what the real mode would copy each time).
namespace Core::HookOwned {

[[nodiscard]] u32 Mode() noexcept;

/// The mark a write carries through the batch. Real mode: an owned write is not announced (never reaches the batch) except the audit one in 64.
enum Mark : u8 { MarkNone = 0, MarkJoin = 1, MarkOwned = 2, MarkAudit = 3 };

/// Real mode: buffer `id` is a confirmed member, so its bytes are pinned dirty (every upload walk takes them, no subtraction removes them). Lock-free, any thread.
[[nodiscard]] bool Pinned(u64 id) noexcept;
/// Real mode: [address, address + size) meets a member (raster thread; a bind or obtain memo must not keep an answer over it).
[[nodiscard]] bool MemberIn(u64 address, u64 size) noexcept;
/// Real mode, the audit apply: whether the member's dirty set held the written range (it must: the pin).
void NoteAuditPinned(bool pinned) noexcept;
/// Real mode: an owned write left out of the batch (hook side).
void NoteDropped() noexcept;

// ---- hook side (any game thread) ----
/// A write with writer tag `tag` (writer | 8 without a site name) naming engine buffer `id`: its mark.
[[nodiscard]] u8 OnWrite(u32 tag, u64 id, u64 address, u64 size) noexcept;
/// A texture was created over buffer `id`'s memory (called by the engine object table under its lock): the buffer is excluded for good.
void ExcludeBuffer(u64 id) noexcept;
/// Buffer `id` was deleted (called by the engine object table under its lock).
void BufferGone(u64 id) noexcept;

// ---- raster side (batch order) ----
/// The mark of the range being applied now (set by the batch around each apply).
extern thread_local u8 t_mark;
/// The batch applies [address, address + size) named `object` with mark `t_mark`. Returns true when the apply's effects are to be captured (`ApplyEnd`).
/// For a join apply, `join_begin`/`join_end` get the buffer's memory (else equal): the caller checks it and answers with `Confirm`.
[[nodiscard]] bool ApplyBegin(u64 object, u64 address, u64 size, u64& join_begin, u64& join_end) noexcept;
/// The raster thread's answer to a join: why it refuses (0 none: confirmed).
enum Refuse : u32 { RefuseNone = 0, RefuseImage = 1, RefuseGpuWritten = 2, RefuseUnknownBuffer = 3 };
/// Returns true when the buffer is now a member (the caller pins it in real mode).
[[nodiscard]] bool Confirm(u64 id, u64 begin, u64 end, u32 refuse) noexcept;
/// The captured effects of the apply (N0 effect bits).
void ApplyEnd(u32 effects) noexcept;
/// An engine object event of the batch (`what` as in `NoteEngineDelete` / `NoteEngineCreate`: 2 texture created, 4 buffer deleted).
void Event(u32 what, u64 id, u64 address, u64 bytes) noexcept;
/// A GPU-written request over [address, address + size).
void GpuWrite(u64 address, u64 size) noexcept;
/// An image registered over [address, address + size).
void ImageRegistered(u64 address, u64 size) noexcept;
/// A read request of the address path over [address, address + size).
void Request(u64 address, u64 size) noexcept;

[[nodiscard]] nlohmann::json Report();

} // namespace Core::HookOwned
