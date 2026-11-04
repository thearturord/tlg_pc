// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <string_view>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Vulkan {

void SetFixedStateTopology(FixedState& out, Core::NativePrimitive::Kind type, vk::CullModeFlags cull_mode) {
    out.primitive_type = type;
    const auto topology = type != Core::NativePrimitive::Kind::None
                              ? LiverpoolToVK::PrimitiveType(type)
                              : vk::PrimitiveTopology::ePatchList;
    out.list_topology = topology == vk::PrimitiveTopology::ePointList ||
                        topology == vk::PrimitiveTopology::eLineList ||
                        topology == vk::PrimitiveTopology::eTriangleList ||
                        topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
                        topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    // Quad and rect lists are emulated using tessellation.
    out.patch_list_topology = type == Core::NativePrimitive::Kind::PatchPrimitive ||
                              type == Core::NativePrimitive::Kind::QuadList ||
                              type == Core::NativePrimitive::Kind::RectList;
    out.cull_mode = LiverpoolToVK::IsPrimitiveCulled(type) ? cull_mode : vk::CullModeFlagBits::eNone;
}

const FixedState& DefaultFixedState() {
    // F3.6: the state of a draw whose registers hold their defaults, written out (it was built from the register defaults before; the values were dumped once and compared)
    static const FixedState state = [] {
        FixedState s;
        s.viewport_count = 1;
        s.viewport_empty = true; // no viewport described: the placeholder viewport
        s.clip_disabled = true;
        s.viewports[0] = vk::Viewport{.x = -1.0f, .y = -1.0f, .width = 1.0f, .height = 1.0f, .minDepth = 0.0f, .maxDepth = 1.0f};
        s.scissors[0] = vk::Rect2D{.offset = {0, 0}, .extent = {1, 1}};
        SetFixedStateTopology(s, Core::NativePrimitive::Kind::TriangleList, vk::CullModeFlagBits::eNone);
        s.front_face = vk::FrontFace::eClockwise;
        s.target_write_mask = {15, 14, 14, 11, 13, 10, 14, 13};
        s.logic_op_enabled = true;
        s.logic_op = vk::LogicOp::eClear;
        s.depth_clamp = true;
        s.depth_clip = true;
        s.provoking_first = true;
        return s;
    }();
    return state;
}

} // namespace Vulkan
