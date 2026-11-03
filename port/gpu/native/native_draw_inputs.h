// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"
#include "native/native_draw.h"

namespace Vulkan {
struct DrawInputs;
}

/// F5: the inputs of a draw (`Vulkan::DrawInputs`: target descriptions, draw arguments, step rates) made from a native record alone. Free functions of plain data: no rasterizer,
/// no command processor, no register image. The record keeps register pairs as its format; the table in the .cpp says which word of the inputs each register is.
namespace Core::NativeDrawInputs {

/// Writes the register pairs `begin` to `end` of the record into the inputs (registers the inputs have no member for are skipped).
void ApplyRegisters(Vulkan::DrawInputs& in, const NativeDraw::ExecDraw& draw, u32 begin, u32 end);

/// Overwrites the draw arguments the record's typed `args` set (a field it did not set keeps what the inputs hold).
void ApplyDrawArgs(Vulkan::DrawInputs& in, const NativeDraw::DrawArgs& args);

/// A draw: the record's pairs, the target description, the typed arguments, the inline index count (`~0u`: none) and the step rates of the vertex shader object.
[[nodiscard]] Vulkan::DrawInputs FromRecord(const NativeDraw::ExecDraw& draw, u32 index_count);

/// A depth / stencil clear (record kind 7): the depth target words the record holds and the target description of the clear.
[[nodiscard]] Vulkan::DrawInputs ForDepthClear(const NativeDraw::ExecDraw& draw);

/// A fast clear eliminate (record kind 4): resets `in` and fills it from the record.
void ForEliminate(Vulkan::DrawInputs& in, const NativeDraw::ExecDraw& draw);

} // namespace Core::NativeDrawInputs
