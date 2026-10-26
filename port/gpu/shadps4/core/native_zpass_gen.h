// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3i: the cascade generator (documents/tlg/n6-pass-generator-design.md). The engine's depth pass loop (`RenderDev_DrawZPass`, ghidra 0x01c74a00) is replaced, for the view types the
// switch names, by native code that reads the same snapshot, writes the same per primitive state into the device context and issues the same draws through the engine's draw entry.
// Behind switches, off by default:
//   TLG_ZPASS_GEN=1            the generator runs (the hook of DrawZPass is wrapped; a pass the generator declines is run by the engine's loop through the original entry)
//   TLG_ZPASS_GEN_VIEWS=0,1,2,5,6,7  the view types it handles (default: the Z passes 0, 1, 2 and the shadow cascades 5, 6, 7)
//   TLG_ZPASS_GEN_VERIFY=N     one generated pass in N is run by the engine's loop instead, and compared draw by draw with what the generator would have written (default 32, 0: never);
//                              a difference switches the generator off for that view type for the rest of the run
// Report: /zpassgen.

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeZPassGen {

[[nodiscard]] bool Enabled();
/// `DrawZPass(render_device, desc)` was entered. Returns true when the pass is done (generated, or run by the loop through `original_loop`, the hook's original entry) and the
/// hooked function must be skipped, false when nothing was done (the view type is not handled: the engine's loop runs).
bool HandlePass(u64 render_device, u64 desc, u64 original_loop);
/// N6.6, the G-buffer loop (`RenderDev_DrawGBufferPrimitives`): `TLG_GBUFFER_GEN=1`, verify one pass in `TLG_GBUFFER_GEN_VERIFY` (default 32). Same contract as HandlePass; `param7` is the
/// byte the loop takes on the stack, `original_loop` the hook's original entry.
[[nodiscard]] bool GBufferEnabled();
bool HandleGBuffer(u64 render_device, u64 mode, u64 list, u64 args, u64 flags, u64 query_mode, u64 param7, u64 original_loop);
/// True while a verify pass runs the engine's loop.
[[nodiscard]] bool VerifyActive();
/// A draw of the engine's loop during a verify pass: `caller` is the ghidra address of the call site, kind 0 `DrawIndexedCommon`.
void NoteLoopDraw(u64 caller, u32 kind, u64 ctx, u64 count, u64 first, u64 a3);
/// N6.7, the forward loop (`RenderDev_DrawForwardPass(renderDev, a1, a2, list, mode)`): `TLG_FORWARD_GEN=1`, verify one pass in `TLG_FORWARD_GEN_VERIFY` (default 32). Same contract.
[[nodiscard]] bool ForwardEnabled();
bool HandleForward(u64 render_device, u64 a1, u64 a2, u64 list, u64 mode, u64 original_loop);

/// A7, the decal pass loop (`FUN_01c71680(renderDev, view, a3, a4, desc, flags, type, width, height)`): `TLG_DECAL_GEN=1`, verify one pass in `TLG_DECAL_GEN_VERIFY` (default 32). Same contract.
[[nodiscard]] bool DecalEnabled();
bool HandleDecal(u64 render_device, u64 view, u64 a3, u64 a4, u64 desc, u32 flags, s32 type, u32 width, u32 height, u64 original_loop);

nlohmann::json Report();

} // namespace Core::NativeZPassGen
