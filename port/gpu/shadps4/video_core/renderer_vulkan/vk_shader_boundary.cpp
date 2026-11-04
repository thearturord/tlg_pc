// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include "native/native_primitive.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_shader_boundary.h"

namespace Vulkan::ShaderBoundary {

using Shader::HwStage;
using Shader::Output;

// the plain primitive enum of the fixed state is the console's primitive type code, the same numbers the recompiler's geometry inputs use
#define TLG_PRIM_SAME(name) static_assert(static_cast<u32>(Core::NativePrimitive::Kind::name) == static_cast<u32>(AmdGpu::PrimitiveType::name))
TLG_PRIM_SAME(None);
TLG_PRIM_SAME(PointList);
TLG_PRIM_SAME(LineList);
TLG_PRIM_SAME(LineStrip);
TLG_PRIM_SAME(TriangleList);
TLG_PRIM_SAME(TriangleFan);
TLG_PRIM_SAME(TriangleStrip);
TLG_PRIM_SAME(PatchPrimitive);
TLG_PRIM_SAME(AdjLineList);
TLG_PRIM_SAME(AdjLineStrip);
TLG_PRIM_SAME(AdjTriangleList);
TLG_PRIM_SAME(AdjTriangleStrip);
TLG_PRIM_SAME(RectList);
TLG_PRIM_SAME(LineLoop);
TLG_PRIM_SAME(QuadList);
TLG_PRIM_SAME(QuadStrip);
TLG_PRIM_SAME(Polygon);
#undef TLG_PRIM_SAME

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

void ApplyShaderEnv(Shader::RuntimeInfo& info, const Core::NativeShaderEnv::DrawEnv& draw, HwStage stage, u32 aa_samples) {
    const Core::ShaderEnv::ShaderEnv& env = stage == HwStage::Vertex ? draw.vertex : draw.pixel;
    info.props.fp_denorm_mode32 = static_cast<AmdGpu::FpDenormMode>(env.program.fp_denorm_mode32);
    info.props.fp_denorm_mode16_64 = static_cast<AmdGpu::FpDenormMode>(env.program.fp_denorm_mode16_64);
    info.props.fp_round_mode32 = static_cast<AmdGpu::FpRoundMode>(env.program.fp_round_mode32);
    info.props.fp_round_mode16_64 = static_cast<AmdGpu::FpRoundMode>(env.program.fp_round_mode16_64);
    info.props.num_allocated_vgprs = env.program.vgprs;
    info.props.num_user_data = env.program.user_data_count;
    info.props.num_input_vgprs = env.program.input_vgpr_components;
    info.props.dx10_clamp = env.program.dx10_clamp ? 1u : 0u;
    if (stage == HwStage::Vertex) {
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, std::bit_cast<AmdGpu::VsOutputControl>(env.exports.vs_out_control));
        if (info.sw_stage == Shader::SwStage::Vertex) {
            info.sw.vs.step_rate_0 = env.rates.instance_step_rate_0;
            info.sw.vs.step_rate_1 = env.rates.instance_step_rate_1;
        }
        return;
    }
    const auto& f = env.fragment;
    info.hw.fs.en_flags = std::bit_cast<AmdGpu::PsInput>(f.input_enable);
    info.hw.fs.addr_flags = std::bit_cast<AmdGpu::PsInput>(f.input_address);
    info.hw.fs.num_inputs = f.num_interpolants;
    info.hw.fs.front_face_all_bits = f.front_face_all_bits ? 1u : 0u;
    info.hw.fs.depth_before_shader = f.depth_before_shader ? 1u : 0u;
    info.hw.fs.z_export_format = static_cast<AmdGpu::ShaderExportFormat>(f.z_export_format);
    info.hw.fs.mrtz_mask = f.depth_export_mask;
    info.hw.fs.num_samples = f.sample_coverage ? aa_samples : 1;
    for (u32 i = 0; i < f.num_interpolants && i < 32; ++i) {
        const auto& want = f.interpolants[i];
        info.hw.fs.inputs[i] = {
            .param_index = u16(want.input_offset),
            .is_default = u16(want.use_default ? 1 : 0),
            .is_flat = u16(want.flat_shade ? 1 : 0),
            .default_value = u16(want.default_value),
        };
    }
    for (u32 i = 0; i < Shader::MaxColorBuffers; ++i) {
        // the export format of each target is the shader's (SPI_SHADER_COL_FORMAT); only the targets the key keeps bound have one
        if (info.hw.fs.color_buffers[i].data_format != AmdGpu::DataFormat::FormatInvalid) {
            info.hw.fs.color_buffers[i].export_format = static_cast<AmdGpu::ShaderExportFormat>(f.color_export_format[i]);
        }
    }
}

void FillComputeInfo(Shader::RuntimeInfo& info, const Core::NativeDispatch::DispatchInputs& cs_pgm) {
    info.props.num_user_data = cs_pgm.settings.num_user_regs;
    info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
    info.props.fp_denorm_mode32 = static_cast<decltype(info.props.fp_denorm_mode32)>(cs_pgm.settings.fp_denorm_mode32);
    info.props.fp_denorm_mode16_64 = static_cast<decltype(info.props.fp_denorm_mode16_64)>(cs_pgm.settings.fp_denorm_mode64);
    info.props.fp_round_mode32 = static_cast<decltype(info.props.fp_round_mode32)>(cs_pgm.settings.fp_round_mode32);
    info.props.fp_round_mode16_64 = static_cast<decltype(info.props.fp_round_mode16_64)>(cs_pgm.settings.fp_round_mode64);
    info.props.dx10_clamp = cs_pgm.settings.dx10_clamp;
    info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full, cs_pgm.num_thread_z.full};
    info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1), cs_pgm.IsTgidEnabled(2)};
    info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
}

bool GuestClipDistances(const Core::ShaderEnv::ShaderEnv& vertex) {
    return std::bit_cast<AmdGpu::VsOutputControl>(vertex.exports.vs_out_control).clip_distance_enable;
}

void SetExportFormat(Shader::PsColorBuffer& color_buffer, u32 code) {
    color_buffer.export_format = static_cast<AmdGpu::ShaderExportFormat>(code);
}

void SetNoExport(Shader::PsColorBuffer& color_buffer) {
    color_buffer.export_format = AmdGpu::ShaderExportFormat::Zero; // no pixel shader: nothing is exported
}

} // namespace Vulkan::ShaderBoundary
