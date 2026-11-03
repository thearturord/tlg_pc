// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_draw_inputs.h"

#include <array>
#include <cstddef>
#include <cstring>

#include "native/native_shader_env.h"
#include "native/native_target_desc.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Core::NativeDrawInputs {

namespace {
/// Register index -> where a record's register word goes in the draw inputs: 1 + the word index inside `DrawInputs` (0: the draw inputs have no member for the register). F3.6: the
/// table is made from the register file's own layout (`Regs`) and `DrawInputs`; no draw state struct stands between them.
const std::array<u32, AmdGpu::Regs::NumRegs>& InputWordMap() {
    static const auto map = [] {
        using Vulkan::DrawInputs;
        std::array<u32, AmdGpu::Regs::NumRegs> table{};
        const auto add = [&table](size_t reg_byte_offset, size_t input_byte_offset, size_t size) {
            for (size_t w = 0; w < size / sizeof(u32); ++w) {
                table[reg_byte_offset / sizeof(u32) + w] = static_cast<u32>(input_byte_offset / sizeof(u32) + w + 1);
            }
        };
#define TLG_INPUT_MEMBER(name) add(offsetof(AmdGpu::Regs, name), offsetof(DrawInputs, name), sizeof(DrawInputs::name))
        TLG_INPUT_MEMBER(color_buffers);
        TLG_INPUT_MEMBER(depth_buffer);
        TLG_INPUT_MEMBER(depth_view);
        TLG_INPUT_MEMBER(depth_control);
        TLG_INPUT_MEMBER(depth_htile_data_base);
        TLG_INPUT_MEMBER(depth_render_control);
        TLG_INPUT_MEMBER(depth_clear);
        TLG_INPUT_MEMBER(stencil_clear);
        TLG_INPUT_MEMBER(ta_bc_base);
        TLG_INPUT_MEMBER(clip_user_data);
        TLG_INPUT_MEMBER(num_indices);
        TLG_INPUT_MEMBER(num_instances);
        TLG_INPUT_MEMBER(index_offset);
        TLG_INPUT_MEMBER(vgt_instance_step_rate_0);
        TLG_INPUT_MEMBER(vgt_instance_step_rate_1);
        TLG_INPUT_MEMBER(index_base_address);
        TLG_INPUT_MEMBER(index_buffer_type);
        TLG_INPUT_MEMBER(max_index_size);
#undef TLG_INPUT_MEMBER
        return table;
    }();
    return map;
}
} // namespace

void ApplyRegisters(Vulkan::DrawInputs& in, const NativeDraw::ExecDraw& draw, u32 begin, u32 end) {
    const auto& map = InputWordMap();
    for (u32 i = begin; i < end; ++i) {
        const u32 index = draw.regs[i].index;
        if (index >= map.size() || map[index] == 0) {
            continue;
        }
        std::memcpy(reinterpret_cast<u8*>(&in) + (map[index] - 1) * sizeof(u32), &draw.regs[i].value, sizeof(u32));
    }
}

void ApplyDrawArgs(Vulkan::DrawInputs& in, const NativeDraw::DrawArgs& a) {
    if ((a.mask & NativeDraw::DrawArgs::NumIndices) != 0) {
        in.num_indices = a.num_indices;
    }
    if ((a.mask & NativeDraw::DrawArgs::NumInstances) != 0) {
        in.num_instances.num_instances = a.num_instances;
    }
    if ((a.mask & NativeDraw::DrawArgs::IndexBase) != 0) {
        in.index_base_address.base_addr_hi = a.index_base_hi;
        in.index_base_address.base_addr_lo = a.index_base_lo;
    }
    if ((a.mask & NativeDraw::DrawArgs::IndexType) != 0) {
        in.index_buffer_type.raw = a.index_type;
    }
    if ((a.mask & NativeDraw::DrawArgs::MaxIndexSize) != 0) {
        in.max_index_size = a.max_index_size;
    }
}

Vulkan::DrawInputs FromRecord(const NativeDraw::ExecDraw& draw, u32 index_count) {
    Vulkan::DrawInputs in{};
    ApplyRegisters(in, draw, 0, draw.count);
    NativeTargetDesc::ApplyAtFill(in);
    ApplyDrawArgs(in, draw.args);
    if (index_count != ~0u) {
        in.num_indices = index_count;
    }
    NativeShaderEnv::DrawEnv env;
    if (NativeShaderEnv::Current(env) && env.has_vertex) {
        in.vgt_instance_step_rate_0 = env.vertex.rates.instance_step_rate_0;
        in.vgt_instance_step_rate_1 = env.vertex.rates.instance_step_rate_1;
    }
    return in;
}

Vulkan::DrawInputs ForDepthClear(const NativeDraw::ExecDraw& draw) {
    Vulkan::DrawInputs in{};
    ApplyRegisters(in, draw, 0, draw.count);
    NativeTargetDesc::ApplyDepthClear(in, draw.native_targets); // E7 (TLG_TARGET_DESC_USE)
    return in;
}

void ForEliminate(Vulkan::DrawInputs& in, const NativeDraw::ExecDraw& draw) {
    in = Vulkan::DrawInputs{};
    ApplyRegisters(in, draw, 0, draw.count);
    NativeTargetDesc::ApplyEliminate(in, draw.native_targets); // E7 (TLG_TARGET_DESC_USE)
}

} // namespace Core::NativeDrawInputs
