// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

/// Byte audit of the memory the renderer reads (debugger mode, Shift+F5; documents/tlg/native plan/engine/frame-walk.md).
///
/// The write ledger (native_content) only knows writes somebody announced, and it does not check reads in the game's rings at all. This audit asks the memory itself:
///  * every read of game memory the raster thread makes (NoteLateRead) adds its range to a "hot" set (kept for 8 submissions);
///  * at each game submit (NoteSubmit) the hot set is hashed in 128 byte chunks: the memory as the game left it for that submission;
///  * when the raster thread then reads a range of that submission, the chunks it covers are hashed again and compared with the snapshot. A different chunk means the
///    bytes the GPU reads are not the bytes the game submitted: a stale or late read, whoever wrote them and whether or not the write was announced.
/// The first differing chunks arm a hardware write watch (native_tsharp_watch) on their first dword, so the next write to it names the writing instruction.
/// Everything is off (one relaxed load) until debugger mode starts it.
namespace Core::ByteAudit {

[[nodiscard]] bool Active();
void SetActive(bool on);

/// Raster thread: a read of [address, address + size) belonging to the submission `submit` (0 unknown). `describe` is called only for the first few stale reads (text for the report).
void NoteRead(u64 address, u64 size, u32 submit, const char* kind, std::string (*describe)());

/// Game submit thread, before the submission counter moves: hashes the hot set as submission `index`.
void SnapshotAtSubmit(u32 index);

/// JSON: counters, the stale reads seen (newest 200) and the write watch hits.
[[nodiscard]] std::string Report();
/// Zeroes everything (debugger mode starts).
void Reset();

} // namespace Core::ByteAudit
