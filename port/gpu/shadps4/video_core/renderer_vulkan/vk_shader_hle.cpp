// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_raycast_cpu.h"
#include "native/native_skin_cpu.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);

        // Apply found buffer base.
        const auto vk_copies = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : vk_copies) {
            copy.srcOffset = copy.srcOffset - src_offset_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_offset_min + dst_buf_offset;
        }

        // Execute buffer copies.
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        runtime.CopyBuffer(src_buf, dst_buf, vk_copies);
        batch_start = batch_end;
    }

    return true;
}

/// Shadow run of the CPU port of the game's ray against triangle shader (native/native_raycast_cpu.h): reads the five buffers from guest memory, keeps the CPU's hit
/// list, and lets the GPU dispatch run as before.
static void ShadowRaycast(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program) {
    if (info.buffers.size() < 5) {
        return;
    }
    const auto sharp = [&](u32 index) { return info.buffers[index].GetSharp(info); };
    const auto p = sharp(0), i = sharp(1), v = sharp(2), r = sharp(3), o = sharp(4);
    const auto ptr = [](const AmdGpu::Buffer& s) { return reinterpret_cast<const u32*>(static_cast<u64>(s.base_address)); };
    Core::NativeRaycast::Buffers buffers{};
    buffers.params = ptr(p);
    buffers.params_dwords = p.GetSize() / 4;
    buffers.indices = ptr(i);
    buffers.indices_dwords = i.GetSize() / 4;
    buffers.vertices = ptr(v);
    buffers.vertices_dwords = v.GetSize() / 4;
    if (const u32* skinned = Core::NativeSkinCpu::Lookup(v.base_address, buffers.vertices_dwords)) {
        buffers.vertices = skinned; // the GPU skinning wrote these; guest memory holds zeros
        buffers.vertices_from_skin = true;
    }
    buffers.rays = ptr(r);
    buffers.rays_dwords = r.GetSize() / 4;
    buffers.out_base = o.base_address;
    buffers.out_capacity = o.num_records;
    buffers.groups_x = cs_program.dim_x;
    Core::NativeRaycast::ShadowDispatch(buffers);
}

/// Shadow run of the CPU port of the skinning shader (native/native_skin_cpu.h).
static void ShadowSkin(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program) {
    if (info.buffers.size() < 6) {
        return;
    }
    const auto sharp = [&](u32 index) { return info.buffers[index].GetSharp(info); };
    const auto b0 = sharp(0), b1 = sharp(1), b2 = sharp(2), b3 = sharp(3), o4 = sharp(4), o5 = sharp(5);
    const auto ptr = [](const AmdGpu::Buffer& s) { return reinterpret_cast<const u32*>(static_cast<u64>(s.base_address)); };
    Core::NativeSkinCpu::Buffers buffers{};
    buffers.b0 = ptr(b0); buffers.n0 = b0.GetSize() / 4;
    buffers.b1 = ptr(b1); buffers.n1 = b1.GetSize() / 4;
    buffers.b2 = ptr(b2); buffers.n2 = b2.GetSize() / 4;
    buffers.b3 = ptr(b3); buffers.n3 = b3.GetSize() / 4;
    buffers.out_a = o4.base_address; buffers.out_a_dwords = o4.GetSize() / 4;
    buffers.out_b = o5.base_address; buffers.out_b_dwords = o5.GetSize() / 4;
    buffers.groups_x = cs_program.dim_x;
    Core::NativeSkinCpu::ShadowDispatch(buffers);
}

static void ShadowSkinWide(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program) {
    if (info.buffers.size() < 8) {
        return;
    }
    Core::NativeSkinCpu::WideBuffers buffers{};
    for (u32 i = 0; i < 13 && i < info.buffers.size(); ++i) {
        const auto s = info.buffers[i].GetSharp(info);
        buffers.p[i] = reinterpret_cast<const u32*>(static_cast<u64>(s.base_address));
        buffers.n[i] = s.GetSize() / 4;
    }
    const auto o6 = info.buffers[6].GetSharp(info), o7 = info.buffers[7].GetSharp(info);
    buffers.out_a = o6.base_address; buffers.out_a_dwords = o6.GetSize() / 4;
    buffers.out_b = o7.base_address; buffers.out_b_dwords = o7.GetSize() / 4;
    buffers.groups_x = cs_program.dim_x;
    Core::NativeSkinCpu::ShadowDispatchWide(buffers);
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    if (Core::NativeRaycast::ShadowOn()) {
        if (info.pgm_hash == Core::NativeSkinCpu::ShaderHashWide) {
            ShadowSkinWide(info, cs_program);
        }
        if (info.pgm_hash == Core::NativeSkinCpu::ShaderHash) {
            ShadowSkin(info, cs_program);
        }
        if (info.pgm_hash == Core::NativeRaycast::ShaderHash) {
            ShadowRaycast(info, cs_program);
            return false;
        }
        for (const auto& buffer : info.buffers) {
            if (buffer.is_written) {
                const auto sharp = buffer.GetSharp(info);
                Core::NativeRaycast::NoteComputeWrite(static_cast<u32>(info.pgm_hash), sharp.base_address, sharp.GetSize(), cs_program.dim_x);
            }
        }
    }
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
