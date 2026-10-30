// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json.hpp>
#include "common/types.h"

/// E0 census (documents/tlg/native plan/no-emulation-plan.md): how much PS4 GPU emulation still runs. Read only counters (relaxed atomics, always on),
/// reported by the debug server's `/emulation`. Answers: which raw PM4 packets still execute (by opcode, and which registers they set), whether the
/// game submits constant engine buffers, and how much image refresh, detiling and overlap resolution still runs from PS4 memory.
namespace Core::NativeCensus {

/// The entry kinds of Core::NativeDirect::EntryKind, by value (Tag 0, Sync 1, Flip 2, StateBuffer 3, Packet 4)
void NoteEntry(u32 queue, u32 kind, u32 opcode);
/// A SET_SH_REG (0x76), SET_CONTEXT_REG (0x69) or SET_UCONFIG_REG (0x79) packet that executes: its first register offset and the number of values
void NoteRegisterPacket(u32 opcode, u32 first_register, u32 count);
/// A raw packet that executes from the game's stream: counts event types of EVENT_WRITE and keeps the first samples of each opcode (E5)
void NoteRawPacket(u32 opcode, const u32* words, u32 dwords);
/// A packet of the game's stream that is not an entry (draw packets, markers, predication, ...), by opcode
void NoteOtherPacket(u32 opcode);
/// Work (a draw: kind 0, a dispatch: kind 1) the raster thread skipped because it sits inside a native engine clear, by program hash (B11 investigation)
void NoteSkippedInClear(u32 kind, u64 program_hash, u64 index_or_groups);
/// A packet of a constant engine buffer, by opcode and dwords
void NoteCePacket(u32 opcode, u32 dwords);
/// A submission with a constant engine buffer (CCB) of `dwords`
void NoteCeSubmission(u32 dwords);

enum ImageEvent : u32 { ImageRefresh, ImageDetile, ImageOverlap, ImageDepthOverlap, ImageExpand, NumImageEvents };
/// The texture cache did `what` (a refresh of an image from guest memory, a detile of `bytes`, an overlap resolution, an expansion)
void NoteImage(ImageEvent what, u64 bytes = 0);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeCensus
