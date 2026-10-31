// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E7 Task 2 (port/gpu/native/E7-notes.md): the render target and depth target descriptions derived from the engine's own objects, compared with the console words the
/// engine keeps in its view objects. Shadow only: nothing decides from this module.
///
/// What is compared. The render target view object holds a 64 byte block at `+0x54` and the depth stencil view a block at `+0x58`; `Synthesize` copies their words into
/// the register image, and from there they become `AmdGpu::ColorBuffer` / `AmdGpu::DepthBuffer` / `DepthView` in `DrawInputs`. The same words are decoded here straight from
/// the blocks (the register layout is the struct layout) and compared with what the resource and view objects say: the Direct3D 11 description (format, dimension, mip and
/// slice range, sample count at resource `+0x194`), `NativeImageDesc::Derive` (format tables, tiling index, pitch) and the resource's own fields (memory `+0x30`, cmask `+0x58`,
/// clear words `+0x150` / `+0x154`).
///
/// Where it runs. `Observe` is called by the record builder (the game's thread, while the engine objects are alive) once per draw record. A target whose view pointer and block
/// contents did not change since the last time on this thread is only counted, not evaluated again.
///
/// Switch: `TLG_TARGET_DESC=1` (default off; with it off `Observe` is never called and nothing here runs). Counters: /stats native_backend.device.e2_targets.
namespace Core::NativeTargetDesc {

[[nodiscard]] bool Enabled() noexcept;

/// The render targets (slots 0 to 7) and the depth target bound in the device context `ctx`.
void Observe(u64 ctx);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeTargetDesc
