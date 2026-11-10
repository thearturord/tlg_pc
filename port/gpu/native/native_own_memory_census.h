// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Design review 79, step 1a (`TLG_OWN_MEMORY_CENSUS=1`, counters only, default off): what F1 class 2 for real needs, an engine buffer owning its device memory.
/// The bind side classifies every shader buffer binding (ring or special, engine id inside its buffer, engine id crossing it, engine id the registry does not know,
/// no engine id); the write side counts, per engine buffer, the announce batch's applies by writer tag and the CPU write faults on its pages.
namespace Core::OwnMemoryCensus {

[[nodiscard]] bool On() noexcept;

enum BindClass : u32 { Ring, Special, IdInside, IdCrossing, IdUnknown, NoId, RingHook, RingCopyInEngineBuffer, RingCopyNoEngineBuffer, NoIdInEngineBuffer, NoIdNoEngineBuffer, NoIdNoWords, BindClassCount };

/// The engine created a buffer resource (from the resource registry; kind 1 only).
void NoteResource(u64 id, u64 address, u64 size, u32 usage, u32 bind, u32 misc);
void NoteResourceGone(u64 address);
void NoteBind(u32 bind_class) noexcept;
/// A ring binding: from the hook ring (`hook`), else a copy into the constant ring of [address, address + size), attributed to the engine buffer holding it.
void NoteBindRing(u64 address, u64 size, bool hook);
/// A shader buffer binding that names engine id `id` for [address, address + size); `written`: the shader writes it.
void NoteBindId(u64 id, u64 address, u64 size, bool written, bool compute);
/// The announce batch applied [address, address + size) for writer tag `writer` (0-15, 16 unknown).
void NoteApply(u64 address, u64 size, u32 writer);
/// A CPU write fault on tracked memory (a write no hook announced).
void NoteFault(u64 address, u64 size);
/// A binding with no engine id, by the address its descriptor names (`address` 0: the record had no words for it).
void NoteBindNoId(u64 address, u64 size, bool written);
/// A GPU copy of `size` bytes from `src` to `dst` (the engine's copy commands).
void NoteCopy(u64 dst, u64 src, u64 size);
/// A vertex or index buffer range a draw reads.
void NoteGeometryRead(u64 address, u64 size);
[[nodiscard]] nlohmann::json Report();

} // namespace Core::OwnMemoryCensus
