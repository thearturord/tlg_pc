// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json.hpp>

#include "common/types.h"

namespace Core::StreamProbe {

/// Diagnostics (study S0 of the streaming design): the shape of the command buffers a frame is submitted in, the points where a partly recorded buffer
/// could be handed over, and whether the game changes the buffer after the draw hook has seen it. Nothing here changes what the emulator does.

/// The draw hook (render thread): remembers the dwords written since the previous hook (a copy) as one segment of the buffer being recorded.
void AtHook(u64 ctx);

/// A descriptor table of a draw record (the tables our hook built from the engine's objects).
struct TableView {
    u32 kind{};  ///< 19 resource table, 22 constant buffer table, 25 RW table, 27 extended user data
    u32 dwords{};
    u64 address{};
    u32 entry{}; ///< dwords per entry
    u64 mask0{}, mask1{};
};

/// The draw hook, after the record was built: hashes the buffer ranges the draw's descriptor tables name (constants, structured buffers) so that the submit can tell
/// whether the game changed them after the draw was recorded (S1 of the streaming design: a streamed draw would have read the old contents).
void AtRecord(u64 caller, const TableView* tables, u32 count);

/// One command buffer of a submit (before the submit sweep strips packets): shape and comparison of the recorded segments with the buffer.
void OnSubmitBuffer(u32 index, u32 count, const u32* dcb, u32 dcb_dwords, u32 ccb_dwords);

/// End of one submit call: the frame record is closed.
void EndSubmit();

void NoteBuild(u64 ns, bool pure, u32 dwords);
void NotePrepare(u64 ns);

nlohmann::json Report();

} // namespace Core::StreamProbe
