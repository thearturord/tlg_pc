// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// F1 class 1a, shadow stage (design review 2026-11-09 sections 2 and 9 to 12; `TLG_HOOK_RING=1`, default off).
/// A host copy of the 64 MB frame ring, filled on the writing thread at the moment a write is known to be finished: every announcement inside the ring
/// (writers 0 to 4), the light and slot ring copies, and the recording ring blocks when the engine's call that writes them returns. At every constant
/// copy of the front thread (FrontStageRing) the bytes the old path copied are compared with the copy. Nothing the GPU reads changes: the old path still
/// copies from game memory and decides every byte. Gate: DIFF_BYTES_MUST_BE_0 and UNFILLED_BYTES_MUST_BE_0 in three scenes.
namespace Core::HookRing {

bool On() noexcept;

/// The frame allocator handed out [block, block + size): its granules are not filled any more (the next write fills them).
void NoteAlloc(u64 block, u64 size);
/// A write to [address, address + size) is finished: the bytes are copied into the host ring now (only the part inside the frame ring).
void Fill(u64 address, u64 size);
/// The front thread copied [address, address + size) of game memory into `copied`: compare with the host ring.
void Compare(u64 address, u64 size, const void* copied);

nlohmann::json Report();

} // namespace Core::HookRing
