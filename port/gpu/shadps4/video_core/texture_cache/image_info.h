// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "native/native_desc.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/tile.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
struct DepthBuffer;
struct Image;
enum class ImageType : u64;
} // namespace AmdGpu

namespace Libraries::VideoOut {
struct BufferAttributeGroup;
}

namespace Shader {
struct ImageResource;
}

namespace VideoCore {

struct ImageProperties {
    u32 is_volume : 1;
    u32 is_tiled : 1;
    u32 is_pow2 : 1;
    u32 is_block : 1;
    u32 is_depth : 1;
    u32 has_stencil : 1;
    /// tlg: block-compressed 1D image: still 1D for the cache, a 2D image of height 1 in Vulkan.
    u32 is_host_2d : 1;
};

/// F4.1: the plain description (`native/native_desc.h`) read from a T# with the emulator's own helper functions: the reference the native description is compared with.
/// `ImageDescAccessorMismatch` checks the accessors of a description against the T# helpers (bit per accessor, `Core::NativeImageDesc::AccessorNames`) for `is_array` both ways.
Core::NativeDesc::ImageDesc ImageDescFromSharp(const AmdGpu::Image& sharp) noexcept;
u32 ImageDescAccessorMismatch(const AmdGpu::Image& sharp, const Core::NativeDesc::ImageDesc& desc) noexcept;
struct ImageInfo;
struct ImageViewInfo;
/// F4.2 shadow: bit per differing field (`Core::NativeImageDesc::InfoFieldNames` / `ViewFieldNames`) between two infos built for the same binding.
u32 ImageInfoDiffMask(const ImageInfo& a, const ImageInfo& b) noexcept;
u32 ImageViewInfoDiffMask(const ImageViewInfo& a, const ImageViewInfo& b) noexcept;

struct ImageInfo {
    ImageInfo() = default;
    ImageInfo(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address) noexcept;
    ImageInfo(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint) noexcept;
    ImageInfo(const AmdGpu::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
              AmdGpu::CbDbExtent hint, bool write_buffer = false) noexcept;
    ImageInfo(const AmdGpu::Image& sharp, const Shader::ImageResource& desc) noexcept;
    /// F4.2: the same info from the plain description (no T# read). `is_depth` is the shader's use of the image (the host promotes the format to a depth one).
    /// A description with the neo mode alternative tiling set is not supported here (the caller keeps the T# path for it).
    ImageInfo(const Core::NativeDesc::ImageDesc& desc, bool is_depth) noexcept;

    bool IsTiled() const {
        return tile_mode != AmdGpu::TileMode::DisplayLinearAligned;
    }

    Extent2D BlockDim() const {
        const auto dim = props.is_block ? 2 : 0;
        return Extent2D{pitch >> dim, size.height >> dim};
    }

    s32 MipOf(const ImageInfo& info) const;
    s32 SliceOf(const ImageInfo& info, s32 mip) const;

    bool IsCompatible(const ImageInfo& info) const;
    void UpdateSize() noexcept;

    struct {
        VAddr cmask_addr;
        VAddr fmask_addr;
        VAddr htile_addr;
    } meta_info{};

    ImageProperties props;
    vk::Format pixel_format;
    AmdGpu::ImageType type;
    SubresourceExtent resources;
    Extent3D size;
    u32 num_bits;
    u32 num_samples;
    u32 pitch;
    AmdGpu::TileMode tile_mode;
    AmdGpu::ArrayMode array_mode;
    std::array<MipInfo, MAX_MIPS> mips_layout;
    u32 micro_tiled_mips{};
    VAddr guest_address;
    u32 guest_size;
    u8 bank_swizzle{};
    bool alt_tile{};
    VAddr stencil_addr{};
    u32 stencil_size{};
};

} // namespace VideoCore
