// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json.hpp>

#include "common/types.h"

/// L5 of documents/tlg/native plan/diagnostics-logging-plan.md: checks at the boundaries that failed before. A failed check is counted, written to the flight recorder and (the first three of
/// each) to user/log/rare_events.txt with the record number; it never stops or changes the game. The counters are in `/invariants` and in every counter snapshot.
namespace Core::Invariant {

#define TLG_INVARIANTS(X)                                                                                                                                                      \
    X(RecordOverwritten, "record_overwritten", "a record slot was reused while the command processor had not consumed the record that lived there (run ahead past the record ring)")  \
    X(WaitGaveUp, "wait_gave_up", "a wait for the raster thread (frame ring lap fence, free hold) saw no progress for 10 s and went on (TLG_RING_LAP_FENCE / TLG_PROGRESS_WAITS)")                 \
    X(GateGaveUp, "gate_gave_up", "the frame gate saw no progress for 10 s (TLG_PROGRESS_WAITS) or waited 2 s (old rule) and let the game go on")                                           \
    X(HoldReleasedAll, "hold_released_all", "the free hold released every held block without the raster thread being done (heap dry, last resort)")                                        \
    X(SlowFrame, "slow_frame", "a frame took more than three times the running median (flight event only, no file)")                                                         \
    X(FrameWithoutDraws, "frame_without_draws", "a frame was presented although no draw or dispatch ran since the previous one (menus and movies count too; flight event only)")

enum Id : u32 {
#define X(id, name, text) id,
    TLG_INVARIANTS(X)
#undef X
        Count
};

/// Any thread. `note`: also write the first three occurrences to rare_events.txt.
void Hit(Id id, u64 seq = 0, u64 a = 0, u64 b = 0, bool note = true);

/// Once per presented frame (the debug server's NotePresent): the frame time and draw table checks.
void NoteFrame();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::Invariant
