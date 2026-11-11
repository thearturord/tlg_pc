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
/// `cls` 0 immutable read, 1 mutable read, 2 mutable written (design review 85); 3 large read, 4 large written (note 117); 5 range read, 6 range written (notes 117, 120).
/// For the range classes `OwnImmutable` counts the bindings its record does not hold. 7 index buffer, 8 vertex buffer (note 117): `OwnImmutable`
/// counts the reads a geometry id's buffer holds whole. 9 vertex split (note 121): the members of a merged vertex range that no one buffer holds.
void NoteOwn(u32 what, u32 cls = 0) noexcept;
/// Notes 117 to 119 (`TLG_OWN_LARGE`: 2 real by default since the owner's play, note 119; 1 shadow; 0 the old path): default and dynamic engine buffers above the resident cap (`TLG_RESIDENT_CAP`) get a record of their own
/// (`NativeResources::FindLargeById`) that only the own-memory path and its upkeep read, and are bound from their placed block like the mutable class.
[[nodiscard]] u32 OwnLargeMode() noexcept;
/// Notes 117 and 120 (`TLG_OWN_RANGE`: 0 off by default, 1 shadow, 2 real): a binding named by an engine id that its record does not hold (no record: a buffer on caller memory;
/// a range outside the record: a dynamic buffer whose memory a Map moved, or a range past the buffer's end) is bound from the placed block that holds the whole range (no slack)
/// while the range has no CPU bytes waiting (`IsRegionCpuModified`: every engine buffer part and the bytes between them) and, written, is GPU-marked.
[[nodiscard]] u32 OwnRangeMode() noexcept;
/// Notes 117 to 119 (`TLG_OWN_GEOMETRY`: 2 real by default since the owner's play, note 119; 1 shadow; 0 the old path): index and vertex reads of an engine buffer the draw names by id (the record's geometry ids) are bound
/// from that buffer's own memory, on the terms of its class (immutable: synchronized; default and dynamic: no CPU bytes waiting).
[[nodiscard]] u32 OwnGeometryMode() noexcept;

/// Note 119 (`TLG_F1_DIAG=1`, counters and the first examples in the log only; default off): why a shader buffer binding is not bound from own memory, by cause.
[[nodiscard]] bool DiagOn() noexcept;
enum Diag : u32 {
    DiagOutsideBelow,     ///< the binding has an id and a record, but starts below the buffer
    DiagOutsidePast,      ///< ... starts at or past the buffer's end
    DiagOutsideCrossing,  ///< ... starts inside and ends past it
    DiagOutsideDead,      ///< ... the record is dead
    DiagNoRecordCaller,   ///< the id has no record: a buffer on caller memory (misc bit 19)
    DiagNoRecordStaging,  ///< ... a staging buffer (usage 3)
    DiagNoRecordOther,    ///< ... another buffer the registry refused
    DiagNoRecordNotBuffer,///< ... the id names no buffer (a texture or view)
    DiagNoRecordUnknown,  ///< ... the device table does not know the id
    DiagNoOwnNone,        ///< inside its record, no own pointer: no placed block meets the buffer
    DiagNoOwnPartial,     ///< ... placed blocks meet it but none holds it whole
    DiagNoIdRing,         ///< no engine id: the range is in the frame ring
    DiagNoIdEngine,       ///< ... the range is inside one engine buffer (the record did not name it)
    DiagNoIdEngineSpan,   ///< ... the range meets engine buffers, not inside one
    DiagNoIdNone,         ///< ... no engine buffer at the range
    DiagNoIdOffsetBase,   ///< (reason) the descriptor is offset by the program base
    DiagNoIdNotSingle,    ///< (reason) not a single load
    DiagNoIdNoWords,      ///< (reason) no record words
    DiagNoIdZero,         ///< (reason) record words, id 0
    DiagNoIdOther,        ///< (reason) words and id, but no plan entry (address or size the front refused)
    DiagNoOwnOldStream,   ///< a binding with no own block: the old path answered from the stream ring (a copy)
    DiagNoOwnOldPlacedNow,///< ... the old path placed a block that holds the buffer whole (its first use, or after a drop)
    DiagNoOwnOldOther,    ///< ... another buffer
    DiagCount
};
void NoteDiag(Diag what, const char* detail = nullptr) noexcept;

/// Note 121 (`TLG_OWN_PLACE`: 0 off by default, 1 shadow): a read binding of a registered engine buffer with no own block that the old path copied into the stream ring
/// (a read of at most `STREAM_THRESHOLD`). The shadow counts what placing the buffer on its first bind would do: the binds own memory would serve (no CPU bytes waiting)
/// and the binds that would need its bytes uploaded first. Real (`=2`, 1 in 64 audited): such a read of a default or dynamic buffer is answered by placing the buffer whole
/// (`BufferCache::ObtainStatic`: the whole buffer taken from guest memory into its own block, the bytes the stream copy would have taken), so the next binds find own memory.
[[nodiscard]] u32 OwnPlaceMode() noexcept;
/// Note 121 (`TLG_OWN_VERTEX_SPLIT`: 0 off by default, 1 shadow, 2 real): a merged vertex range no one engine buffer holds is bound per member buffer from each one's own memory
/// when every member has current own memory (needs `TLG_OWN_GEOMETRY`).
[[nodiscard]] u32 OwnVertexSplitMode() noexcept;
enum Place : u32 {
    PlaceBinds,         ///< stream-ring binds of a buffer with no own block
    PlaceBuffers,       ///< ... distinct buffers
    PlaceClean,         ///< ... binds after which no write was marked since the buffer's previous bind (own memory would serve them as they are)
    PlaceDirty,         ///< ... binds with a write marked since the previous bind, or the first bind (an upload into the block first)
    PlaceDirtyChanges,  ///< (unused)
    PlaceZeroCopy,      ///< ... binds of buffers a placement would make CPU-writable (zero copy: the upload is a CPU write, no pass break)
    PlaceBytes,         ///< ... bytes of the distinct buffers
    PlaceCount
};
/// `version`: the buffer's write version now (`RowVersion::Of`, bumped by every mark of its bytes); a bind is clean when it did not move since the buffer's previous bind.
void NotePlace(u64 id, u64 bytes, u64 version, bool zero_copy) noexcept;
/// Design review 85: `TLG_OWN_MUTABLE` (default 2 real since the owner's play 2026-11-10, 1 in 64 audited; 0 off, 1 shadow): bindings of default and dynamic engine buffers (reads and GPU writes) bound
/// from the buffer's own placed buffer by id while no CPU bytes wait for it.
[[nodiscard]] u32 OwnMutableMode() noexcept;
/// Design review 87: `TLG_OWN_CLOSE` (default 2 real since the owner's play 2026-11-10; 0 off, 1 shadow): a binding whose buffer has no own pointer resolves it once from the placed block that holds the
/// whole buffer. Shadow: bindings valid only through such a pointer are compared with the old path.
[[nodiscard]] u32 OwnCloseMode() noexcept;
[[nodiscard]] nlohmann::json Report();

} // namespace Core::OwnMemoryCensus
