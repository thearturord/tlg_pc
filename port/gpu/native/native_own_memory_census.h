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
/// Design review 81: `TLG_DROP_DIRTY_APPLIES` (default 2 real since the owner's play 2026-11-10; 0 off, 1 shadow): a light constant or scene slot write to a buffer whose bytes are still marked dirty needs no batch entry.
[[nodiscard]] u32 DropDirtyMode() noexcept;
enum Drop : u32 {
    DropAsked,          ///< lean applies of tags 2 and 12 seen
    DropNoResident,     ///< the id names no registered buffer (no flag to keep)
    DropCovered,        ///< the buffer's dirty set already held the whole range
    DropFlagSet,        ///< the flag was set (the drop decision)
    DropFlagNotCovered, ///< FLAG_SET_BUT_NOT_DIRTY_MUST_BE_0: dropping would lose a mark
    DropCoveredNoFlag,  ///< the set held the range, the flag was clear (a drop missed: safe)
    DropLiveCopy,       ///< DROP_WITH_LIVE_COPY_MUST_BE_0: a stream ring copy made at the buffer's current generation
    DropMemoErased,     ///< the bind memo erase removed an entry (the erase is kept in the drop: section 81)
    DropFlagCleared,    ///< uploads that cleared a set flag
    DropDropped,        ///< real: dropped (the image step only ran)
    DropAudited,        ///< real: a would-drop write applied in full and checked (1 in 64)
    DropCount
};
void NoteDrop(u32 what) noexcept;
/// Design review 83: `TLG_OWN_IMMUTABLE` (default 2 real since the owner's play 2026-11-10, 1 in 64 audited; 0 off, 1 shadow): read bindings of immutable engine buffers bound from the buffer's own memory.
[[nodiscard]] u32 OwnImmutableMode() noexcept;
enum Own : u32 {
    OwnAsked,          ///< read bindings with an engine id the front planned
    OwnImmutable,      ///< ... naming an immutable buffer, inside it
    OwnNoOwn,          ///< ... not in a placed buffer of its own (yet)
    OwnNotSynced,      ///< ... with CPU bytes not synchronized
    OwnBound,          ///< real: bound from the own memory
    OwnCompared,       ///< shadow or the audit: compared with the old path's answer
    OwnDifferent,      ///< DIFFERENT_MUST_BE_0
    OwnOldPathWork,    ///< OLD_PATH_DID_WORK_MUST_BE_0: the old path uploaded bytes for it
    OwnOldStream,      ///< the old path answered from the stream ring (a copy, not compared)
    OwnSet,            ///< own pointers set
    OwnCleared,        ///< own pointers cleared (placed buffer dropped)
    OwnNotGpuMarked,   ///< written: the range is not marked GPU-written yet (the old path marks it)
    OwnResolved,       ///< own pointers found by the lazy resolution
    OwnByResolved,     ///< bindings valid only through a resolved pointer
    OwnCount
};
/// `cls` 0 immutable read, 1 mutable read, 2 mutable written (design review 85).
void NoteOwn(u32 what, u32 cls = 0) noexcept;
/// Design review 85: `TLG_OWN_MUTABLE` (default 2 real since the owner's play 2026-11-10, 1 in 64 audited; 0 off, 1 shadow): bindings of default and dynamic engine buffers (reads and GPU writes) bound
/// from the buffer's own placed buffer by id while no CPU bytes wait for it.
[[nodiscard]] u32 OwnMutableMode() noexcept;
/// Design review 87: `TLG_OWN_CLOSE` (default 0; 1 shadow, 2 real): a binding whose buffer has no own pointer resolves it once from the placed block that holds the
/// whole buffer. Shadow: bindings valid only through such a pointer are compared with the old path.
[[nodiscard]] u32 OwnCloseMode() noexcept;
[[nodiscard]] nlohmann::json Report();

} // namespace Core::OwnMemoryCensus
