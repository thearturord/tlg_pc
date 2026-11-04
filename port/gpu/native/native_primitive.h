// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

/// G2 (documents/tlg/native plan/executor-consumer-plan.md): the primitive type of a draw as a plain enum for the fixed state, the pipeline key and the Vulkan topology mapping. The numbers
/// are the ones the draw record carries (the console's primitive type codes), so the producer converts with a cast; `shadps4/video_core/renderer_vulkan/vk_shader_boundary.cpp` asserts that
/// they are the same numbers as `AmdGpu::PrimitiveType`, the type the shader recompiler keeps for its own geometry inputs.
namespace Core::NativePrimitive {

enum class Kind : u32 {
    None = 0,
    PointList = 1,
    LineList = 2,
    LineStrip = 3,
    TriangleList = 4,
    TriangleFan = 5,
    TriangleStrip = 6,
    PatchPrimitive = 9,
    AdjLineList = 10,
    AdjLineStrip = 11,
    AdjTriangleList = 12,
    AdjTriangleStrip = 13,
    RectList = 17,
    LineLoop = 18,
    QuadList = 19,
    QuadStrip = 20,
    Polygon = 21,
};

} // namespace Core::NativePrimitive
