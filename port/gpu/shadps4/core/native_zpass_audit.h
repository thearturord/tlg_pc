// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3a (documents/tlg/native-renderer-plan.md): read-only audit of the engine's depth pass loop (`RenderDev_DrawZPass`, ghidra 0x01c74a00: the shadow cascades and the Z pre-pass).
// The pass reads a finished snapshot: a list of batches (`list + 0` array of batch pointers, `list + 8` count), each batch with an array of primitive records (`batch + 0x58`,
// 0x38 bytes each, count `batch + 100`), per primitive a first index (`+0x10`), a triangle count (`+0x14`) and a material (`+0x18`). The audit builds, from that snapshot alone, the
// list of draws the loop should issue for a pass, and compares it with the draws the loop really issues (the hooks of `DrawIndexedCommon` / `DrawIndexedIndirect` whose caller lies
// inside the function). It changes nothing: it is the proof that a native generator of the cascade draws could read `frameData` instead of running the loop.
// `TLG_ZPASS_AUDIT=N` audits every pass of one frame in N (1: all), report at /zpass.

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeZPass {

[[nodiscard]] bool Enabled();
/// `DrawZPass(render_device, desc)` was entered (the hook, before the original runs): the expected draws are built from `desc` now.
void PassEnter(u64 render_device, u64 desc);
/// A draw call of the engine: `caller` is the ghidra address of the call site, kind 0 `DrawIndexedCommon` (count, first index, instances), kind 1 `DrawIndexedIndirect`.
/// `vertex_shader` / `pixel_shader`: the shader objects the device context holds at the call (the variant the loop chose, audit b).
/// `ctx`: the device context at the call (audit c compares the state the per-primitive binding wrote into it with what the material's render object holds).
void NoteDraw(u64 caller, u32 kind, u64 count, u64 first, u64 instances, u64 vertex_shader, u64 pixel_shader, u64 ctx);
/// A new frame begins on the render thread: the open pass is closed.
void FrameBoundary();
/// N6.3h, the end state audit (`TLG_ZPASS_ENDSTATE=1`, needs `TLG_ZPASS_AUDIT`): the hook runs the loop itself (through the original entry) between PassEnter and PassExit, so the live
/// context can be compared with the context the generator would have produced right after the loop returns.
[[nodiscard]] bool WrapEnabled();
void PassExit(u64 render_device);
/// N6.3e, the shadow generator: the live record of the draw was just built (`seq`). The audit builds the record the same draw would get from a copy of the pass's device context
/// patched with what the snapshot says (material blocks, batch level, texture pointers, shader pair from the learned key table) and compares the two.
void RealRecord(u64 ctx, u64 seq, u32 kind, u64 a1, u64 a2, u64 a3);

nlohmann::json Report();

} // namespace Core::NativeZPass
