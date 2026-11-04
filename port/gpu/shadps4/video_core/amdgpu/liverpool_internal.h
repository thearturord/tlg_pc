// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Private to liverpool.cpp and native/native_liverpool.cpp: MakeDispatchInputs and IsFixedStateRegister (the raster thread's queue is native/native_raster_queue.h since G4b). Include after video_core/amdgpu/liverpool.h and its usual companions.

#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

#include "common/polyfill_thread.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw_tag.h"
#include "video_core/amdgpu/liverpool.h"

namespace AmdGpu {

// ---- Command processor split (documents/tlg/cp-split-design.md) ----

/// G1: the producer of the dispatch inputs; the record's register block is read here and nowhere else in the consumers.
Core::NativeDispatch::DispatchInputs MakeDispatchInputs(const ComputeProgram& program);

/// Registers whose only reader is the translation to the fixed-function state (blend, stencil, depth range of the viewport,
/// screen scissor, pixel shader colour mask).
inline bool IsFixedStateRegister(u32 index) {
    if (index < Regs::ContextRegWordOffset) {
        return false;
    }
    const u32 reg = index - Regs::ContextRegWordOffset;
    return (reg >= 0x1e0 && reg <= 0x1e7) || (reg >= 0x105 && reg <= 0x108) || (reg >= 0x10b && reg <= 0x10d) ||
           reg == 0xb4 || reg == 0xb5 || reg == 0xc || reg == 0xd || reg == 0x8f || reg == 0x8e;
}

} // namespace AmdGpu
