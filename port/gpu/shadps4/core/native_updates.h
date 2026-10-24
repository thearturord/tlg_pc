// SPDX-License-Identifier: GPL-2.0-or-later
// N5a (documents/tlg/native-renderer-plan.md): the update audit. Today the backend learns that the game changed GPU-visible memory from page faults (the pages of
// guest memory the buffer cache watches are write protected). The native way is that the engine says so: every dynamic buffer is written between a
// JsDeviceContext::Map and the draws that use it, and Map is hooked. This audit checks that claim: it records every Map of a buffer (resource id, the pointer the engine
// hands out, the size, the thread) and classifies every CPU write fault by whether an announced Map window covers it. A fault with no window is a write the engine does
// not announce, and has to be found (by the faulting instruction) before page protection can be left out for those buffers. Off by default: `TLG_UPDATE_AUDIT=1` or the
// debug server endpoint `/updates?start=1`.

#pragma once

#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeUpdates {

extern std::atomic<bool> enabled;

void Start();
/// The engine mapped a buffer: `address` / `size` are what Map returned (the buffer's own memory, or a window of the frame ring), `memory` the buffer's own memory.
void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller);
/// A hooked engine function announces the memory it is about to write (`writer` names the function: 1 skinning matrices, 2 light constants).
void NoteWriterRange(u32 writer, u64 address, u64 size);
/// The game wrote tracked memory (called from the fault handler, on the writing thread).
void NoteFault(u64 address, u64 size);

nlohmann::json Report();

} // namespace Core::NativeUpdates
