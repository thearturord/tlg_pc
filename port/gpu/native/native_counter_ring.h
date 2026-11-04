// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

/// Counter ring (`TLG_COUNTER_RING=1`, diagnostic): at every present the frame store, late read and draw counters are sampled and appended to user/log/counter_ring.csv (cumulative values, one row
/// per present, flushed every 60 rows). `Mark` writes a line to counter_ring_marks.txt (the dark frame monitor calls it with the present tick of the frame it found).
/// documents/tlg/tools/counter_ring.py compares the deltas around the marked rows with the deltas of all other rows: a counter that jumps at every dark frame and at no other is the lead.
namespace Core::CounterRing {

[[nodiscard]] bool Enabled();
/// A frame is presented (the present tick is its key).
void NotePresent(u64 tick);
/// The monitor found a frame of `kind` ("dark" or "bright") that was presented with `tick`.
void Mark(const char* kind, u64 tick);

} // namespace Core::CounterRing
