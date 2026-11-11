// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

/// Note 152 (plan R2.10, `TLG_DISPATCH_EXECUTOR`: 0 off by default, 1 shadow; the real mode is not built): what a dispatch executor that runs without the old binding loop
/// needs. The old loop still binds every dispatch. The shadow tags each binding of a dispatch with where its answer came from (a native source, or an old, address-keyed
/// one), counts the dispatches whose every binding has a native source (the executor's class), and counts, inside that class, every call the old loop still makes into the
/// address-keyed machinery (`ObtainBuffer`, `FindImage`) and the side effects the executor must keep (bound-list entries, barrier queries). Published as `dispatch_executor`.
namespace Core::DispatchExecutor {

extern bool g_on;
[[nodiscard]] inline bool On() noexcept {
    return g_on;
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
    SrcHookRingWindow, ///< a frame ring window answered by the hook ring inside ObtainBuffer (native answer, address-keyed call)
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

/// The raster thread starts binding an operation (only dispatches are followed).
void Begin(bool compute) noexcept;
/// One binding of the current dispatch.
void Note(Source s) noexcept;
/// An address-keyed call made while the current dispatch is bound (0 ObtainBuffer, 1 FindImage).
void NoteAddressCall(u32 kind) noexcept;
/// The current dispatch's binding is finished: `bound_entries` bound-list entries were added, `barrier` whether a barrier is needed.
void End(u32 bound_entries, bool barrier) noexcept;

} // namespace Core::DispatchExecutor
