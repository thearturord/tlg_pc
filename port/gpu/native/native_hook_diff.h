// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

/// Notes 153 and 154 (plan R3.1 and R3.3): two shadows of the hook and the plan. Nothing they count changes what is drawn.
/// * `TLG_HOOK_DIFF=1` (R3.1): the hook builds every draw's fixed state and binding list from the context again. The shadow keeps the previous draw's (per guest thread) and
///   counts how often the fixed state's key words are the same (a real mode would reuse the previous state with no lock and no map), with the guard that the same key gives the
///   same state, and per binding slot how often the object is the same and whether the binding it gives is the same too (where it differs, the real mode needs a version).
/// * `TLG_PLAN_POINTERS=1` (R3.3): a plan copies the cached fixed state and the record's bindings. At the draw's execution the shadow checks whether the plan's copies still
///   equal their sources (the cached state, the record slot), so a pointer could replace the copy.
/// Published as `hook_diff`.
namespace Core::HookDiff {

[[nodiscard]] bool DiffOn() noexcept;
[[nodiscard]] bool PointersOn() noexcept;

/// R3.1, the fixed state of a draw record: whether a previous draw existed, its key words were the same, and the state built was the same.
void NoteFixed(bool have_prev, bool same_key, bool same_state) noexcept;
/// R3.1, the binding list of a draw record against the previous draw's.
void NoteBindings(u32 count, bool same_count, u32 same_object_slots, u32 same_object_same_binding, u32 other_object_same_binding, bool all_same) noexcept;
/// R3.3 at the draw's execution: the plan's fixed state equals the cached state it was copied from.
void NotePlanFixed(bool same) noexcept;
/// R3.3 at the draw's execution: 0 the record slot still holds the record and its bindings equal the plan's, 1 the slot was reused, 2 the same record but other bindings, 3 no record.
void NotePlanBindings(u32 state) noexcept;

} // namespace Core::HookDiff
