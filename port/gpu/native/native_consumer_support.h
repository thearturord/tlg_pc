// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Two decisions of the native path that need no emulator type: which engine function owns the work inside a clear's packet range, and the handle switch.

#include <cstdlib>

#include "native/native_draw.h"

namespace Vulkan {

/// B11: work inside an engine clear's packet range is the clear's own only when the engine function that issued it is one of the device context's clear functions
/// (ghidra 0x2483380 ClearRenderTargetView .. 0x2487000, the fill helper included). The range also holds draws and dispatches the engine had deferred and flushes
/// when a clear starts (a clear first runs the target's resource transition, which flushes pending work: the depth pass draws, the particle and feather simulation,
/// skinning; skipping those was the giant feathers over the view, 2026-10-30). Everything else inside the range is the game's real work and runs.
inline bool IsEngineClearOwnWork(u64 native_seq) {
    if (native_seq == 0) {
        return false;
    }
    const u64 caller = Core::NativeDraw::RecordCallerGhidra(native_seq);
    return caller >= 0x2483380 && caller < 0x2487000;
}

/// N3 native binder switch (`TLG_NATIVE_BINDER`): 0 off, 1 the depth passes (class 0), 2 every class with a complete binding list, 3 like 2 and every descriptor is compared with
/// the legacy one (the legacy one wins on a difference). Modes 1 and 2 also leave the flat user data out of the plan of a stage the record covers completely (N3b).
/// N3c handle switch (`TLG_NATIVE_HANDLES`): 0 off, 1 the immutable engine buffers of a draw are found by the engine's id (resolved when the plan is made),
/// 2 like 1 and every handle is compared with the lookup by address (the lookup wins on a difference). Needs the binder (the record's bindings) and the immutable residency.
inline u32 NativeHandleMode() {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_NATIVE_HANDLES");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 2u; // default 2 since 2026-10-27 (the owner's play test); 0 turns it off
    }();
    return mode;
}

} // namespace Vulkan
