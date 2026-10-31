// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Vulkan {
struct DrawInputs;
}

/// E7 Task 2 (port/gpu/native/E7-notes.md): the render target and depth target descriptions derived from the engine's own objects, compared with the console words the
/// engine keeps in its view objects. Shadow only unless `TLG_TARGET_DESC_USE=1` (round 5, below).
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

/// E7 Task 2, round 5: the description feeds the draw (`TLG_TARGET_DESC_USE=1`, default off, read once).
///
/// The target part of `DrawInputs` (colour_buffers, depth_buffer, depth_view, depth_htile_data_base, depth_clear, stencil_clear) is built from the register decode as before, and then the fields the
/// engine's objects decide (E7-notes.md section 9) are replaced by the description's values; the fields marked not needed or not decidable keep the register decode and are counted as such.
/// The description is made where the engine's objects are alive (the record builders on the game's thread) and travels with the record as an immutable `TargetSet`; the command processor applies it at
/// the three places that build the target inputs: the fill of `exec_inputs` (`PlanDraw`), the depth clear record (kind 7) and the fast clear eliminate record (kind 4).
///
/// `TLG_TARGET_DESC_USE_VERIFY=1` (implies the build of the sets): the description is built and compared with the register decode field by field (counted per path and field), and the register result
/// is always used, so verify never changes what reaches the GPU.
struct TargetSet;

[[nodiscard]] bool UseOn() noexcept;
[[nodiscard]] bool VerifyOn() noexcept;

/// Game thread, draw records: the description of the targets bound in the device context `ctx` (null: not made; the same pointer while the targets do not change).
[[nodiscard]] const TargetSet* PrepareDraw(u64 ctx);
/// Game thread, the fast clear eliminate record: the description of the resource's colour target (mip 0, the whole resource).
[[nodiscard]] const TargetSet* PrepareEliminate(u64 resource);
/// Game thread, the depth clear record: `target` is the depth block the clear function was called with (the view's block at +0x58, or the resource's prebuilt block at +0xf0); the view or the
/// resource is found from it by address and checked against the engine's table.
[[nodiscard]] const TargetSet* PrepareDepthClear(u64 target);

/// Command processor: the set of the draw that is being planned (the fill point reads it).
struct Scope {
    explicit Scope(const TargetSet* set);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    const TargetSet* previous;
};

/// `PlanDraw`, right after the inputs were made from the registers.
void ApplyAtFill(Vulkan::DrawInputs& inputs);
/// The depth clear record: the inputs made from the registers, with the set of the record.
void ApplyDepthClear(Vulkan::DrawInputs& inputs, const TargetSet* set);
/// The fast clear eliminate record.
void ApplyEliminate(Vulkan::DrawInputs& inputs, const TargetSet* set);

} // namespace Core::NativeTargetDesc
