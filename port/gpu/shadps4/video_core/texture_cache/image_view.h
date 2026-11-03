// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "native/native_desc.h"
#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
}

namespace Shader {
struct ImageResource;
}

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

struct ImageViewInfo {
    ImageViewInfo() = default;
    /// F4.2: the same view info from the plain description; the three flags are the shader's use of the image (`ImageResource::is_depth`, `is_written`, `is_array`).
    ImageViewInfo(const Core::NativeDesc::ImageDesc& image, bool is_depth, bool is_written, bool is_array) noexcept;
    ImageViewInfo(const AmdGpu::ColorBuffer& col_buffer) noexcept;
    ImageViewInfo(const AmdGpu::DepthBuffer& depth_buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl);

    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    SubresourceRange range;
    AmdGpu::CompMapping mapping{AmdGpu::IdentityMapping};
    u32 min_lod = 0;
    bool is_storage = false;

    bool operator==(const ImageViewInfo&) const = default;
};

struct Image;

struct ImageView {
    ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info, const Image& image);
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    ImageViewInfo info;
    vk::UniqueImageView image_view;
};

} // namespace VideoCore
