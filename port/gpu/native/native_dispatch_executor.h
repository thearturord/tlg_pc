// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

/// Plan R2.10, `TLG_DISPATCH_EXECUTOR` (2 real by default since 2026-11-11; 1 shadow; 0 off).
/// * Note 152 (the tags, mode 1): each binding of a dispatch is tagged with where its answer came from (a native source, or an old, address-keyed one); the dispatches whose
///   every binding has a native source are the executor's class; inside it, every call still made into the address-keyed machinery is counted.
/// * Note 159 (the executor): per buffer binding of a compute stage, in the old loop's order, the executor decides the source with the old loop's own tests for the native sources
///   only, and refuses (before it changes anything) where the old loop would reach an old source; the old body then binds that one binding. Mode 1 compares the executor's
///   decision with what the old body bound (`EXECUTOR_DIFFERENT_MUST_BE_0`, `EXECUTOR_OTHER_SOURCE_MUST_BE_0`); mode 2 binds with it, one dispatch in 64 (`TLG_AUDITS`)
///   compared as in mode 1. LEGACY(dispatch_executor): =0 the old body binds every binding.
/// Published as `dispatch_executor`.
namespace Core::DispatchExecutor {

extern u32 g_mode;
extern bool g_on; ///< the note 152 tags (mode 1)
[[nodiscard]] inline bool On() noexcept {
    return g_on;
}
[[nodiscard]] inline u32 Mode() noexcept {
    return g_mode;
}

/// Where one binding's answer came from.
enum Source : u32 {
    // native: no address lookup decided it
    SrcHookRing,       ///< a frame ring range from the hook ring (front-copied ring entry with the hook flag)
    SrcConstantRing,   ///< the front's constant ring copy
    SrcGds,            ///< the GDS buffer
    SrcFlatUserData,   ///< the flattened user data, copied per dispatch
    SrcClipPlanes,     ///< clip planes (draws only)
    SrcSharedMemory,   ///< the LDS stand-in, zeroed per dispatch
    SrcOwn,            ///< an engine buffer's own memory by its id
    SrcOwnCrossing,    ///< the placed block holding a crossing range (note 117)
    SrcOwnPlaced,      ///< a buffer placed whole and bound from its own memory (note 121)
    SrcProgramConsts,  ///< the kept copy of program constants (note 144)
    SrcHookRingWindow, ///< a frame ring window answered by the hook ring (note 156: asked before ObtainBuffer)
    SrcNull,           ///< a null descriptor
    SrcTextureTable,   ///< a texture stage bound from the view table
    SrcSamplerRecord,  ///< a sampler from the record's words
    // old: decided by the address-keyed path
    SrcResolvedCopy,     ///< R1b: the object's copy in the stream ring (note 137 B: to delete)
    SrcResolvedResident, ///< R2: the resident arena buffer (note 137 B: to delete)
    SrcInPlace,          ///< N7.5: the host import window bound in place
    SrcObtainStream,     ///< ObtainBuffer answered with a stream ring copy
    SrcObtainOther,      ///< ObtainBuffer answered with another buffer (arena, placed block)
    SrcTextureOld,       ///< a texture stage the view table did not bind
    SrcSamplerUserData,  ///< a sampler decoded from the stage's user data words
    SrcCount
};
constexpr u32 FirstOld = SrcResolvedCopy;

/// Note 159: the executor's answer for one buffer binding.
enum ExecKind : u32 {
    ExRefuse,
    ExHookRing,
    ExConstantRing,
    ExGds,
    ExFlat,
    ExFlatIndexed, ///< the flat user data of an indexed stage: read from the row, no binding
    ExShared,
    ExOwnPlaced,
    ExOwn,
    ExOwnCrossing,
    ExProgramConsts,
    ExNull,
    ExHookRingWindow,
    ExOwnSync, ///< note 162: own memory after the CPU bytes waiting in the range were uploaded into it
    ExKindCount
};
/// Note 159: why the executor left a binding to the old body.
enum Refusal : u32 {
    RfNone,
    RfSpecial,           ///< clip planes, or shared memory of an indexed stage
    RfResolved,          ///< an R1b / R2 answer of the command processor whose epoch holds
    RfNoRecord,          ///< a stage whose flat user data was skipped and no record entry (must not happen; the old body logs it)
    RfUnmapped,          ///< a huge range at an unmapped address, or a clamped size (the old body logs it)
    RfProgramConstsBind, ///< the program constants copy could not be made (full)
    RfObtain,            ///< ObtainBuffer
    RfCount
};

/// Note 161 (pass 3 item 4): why own memory by id did not answer a binding the executor refused.
enum OwnWhy : u32 {
    OwNoId,           ///< the record names no engine id for it
    OwNoRecord,       ///< no live buffer record for the id and no crossing block
    OwOutsideRecord,  ///< the range is outside the id's record (crossing tried)
    OwDead,           ///< the record is dead
    OwImmutableWrite, ///< an immutable buffer bound for writing
    OwNoOwnBlock,     ///< no own block, placement refused
    OwExactEnd,       ///< the binding runs past the block's exact end
    OwCpuBytes,       ///< CPU bytes wait in the buffer (not synchronized)
    OwNotGpuMarked,   ///< a written range not yet GPU-marked
    OwCrossInRing,    ///< crossing: the range is (or meets) the frame ring
    OwCrossNoHolder,  ///< crossing: no placed block holds the range
    OwCrossCpuBytes,  ///< crossing: CPU bytes wait in the range
    OwCrossNotMarked, ///< crossing: a written range not yet GPU-marked
    OwCount
};

/// The note 152 source of an executor answer (SrcCount for one the old body does not tag).
[[nodiscard]] Source SourceOf(u32 kind) noexcept;

/// The raster thread starts binding an operation (only dispatches are followed by the tags).
void Begin(bool compute) noexcept;
/// One binding's source (always kept as the last source; counted while a dispatch is open in mode 1).
void Note(Source s) noexcept;
/// The source of the last binding noted on this thread, SrcCount after `ClearLast`.
[[nodiscard]] Source LastSource() noexcept;
void ClearLast() noexcept;
/// An address-keyed call made while the current dispatch is bound (0 ObtainBuffer, 1 FindImage).
void NoteAddressCall(u32 kind) noexcept;
/// The current dispatch's binding is finished: `bound_entries` bound-list entries were added, `barrier` whether a barrier is needed.
void End(u32 bound_entries, bool barrier) noexcept;

/// Note 159 counters (raster thread).
void NoteBound(u32 kind) noexcept;
void NoteRefused(u32 refusal, u32 own_why) noexcept;
enum Compare : u32 { CmpSame, CmpAudited, CmpRefusedOldNative, CmpRefusedOldOld, CmpDifferent, CmpOtherSource, CmpOwnSyncOld, CmpCount };
/// Note 162 (`TLG_OWN_SYNC`: 2 real by default, 1 shadow, 0 off): a read binding of an engine buffer whose own block waits for CPU bytes is synchronized and bound from it
/// (instead of the R1b copy of the whole object in the stream ring). LEGACY(own_sync): =0.
[[nodiscard]] u32 OwnSyncMode() noexcept;
/// Note 162: the shadow would synchronize (mode 1); the real sync left CPU bytes in the range (must not happen).
void NoteOwnSync(bool would, bool left_bytes) noexcept;
void NoteCompare(u32 what, u32 refusal) noexcept;
/// A dispatch bound with the executor on: `real` (mode 2, not audited), its buffer bindings and how many the executor refused.
void DispatchDone(bool real, u32 bindings, u32 refused) noexcept;

} // namespace Core::DispatchExecutor
