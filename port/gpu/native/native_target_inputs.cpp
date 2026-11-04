// SPDX-License-Identifier: GPL-2.0-or-later
#include <atomic>
#include <cstring>
#include <nlohmann/json.hpp>
#include "native/native_target_inputs.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Core::NativeTarget {

static_assert(sizeof(AmdGpu::ColorBuffer) == RawTargetWords::ColorWords * sizeof(u32));
static_assert(sizeof(AmdGpu::DepthBuffer) == RawTargetWords::DepthWords * sizeof(u32));
static_assert(sizeof(AmdGpu::DepthView) == sizeof(u32));
static_assert(sizeof(AmdGpu::DepthControl) == sizeof(u32));
static_assert(AmdGpu::NUM_COLOR_BUFFERS == RawTargetWords::ColorSlots);

namespace {
std::atomic<u64> g_checked{0};
std::atomic<u64> g_differences{0};
std::atomic<u64> g_color_decoded{0};
std::atomic<u64> g_depth_decoded{0};

void Note(bool equal) noexcept {
    g_checked.fetch_add(1, std::memory_order_relaxed);
    if (!equal) {
        g_differences.fetch_add(1, std::memory_order_relaxed);
    }
}

bool Same(const ColorTarget& t, const AmdGpu::ColorBuffer& r) {
    return t.operator bool() == r.operator bool() && t.Pitch() == r.Pitch() && t.Address() == r.Address() && t.CmaskAddress() == r.CmaskAddress() && t.FmaskAddress() == r.FmaskAddress() &&
           t.NumSamples() == r.NumSamples() && t.BaseSlice() == r.BaseSlice() && t.NumSlices() == r.NumSlices() && t.GetTileMode() == r.GetTileMode() && t.IsTiled() == r.IsTiled() &&
           t.info.format == r.info.format && t.info.number_type == r.info.number_type && static_cast<u32>(t.info.comp_swap) == static_cast<u32>(r.info.comp_swap) &&
           t.info.fast_clear == r.info.fast_clear && t.info.compression == r.info.compression && t.info.dcc_enable == r.info.dcc_enable && t.info.blend_bypass == r.info.blend_bypass && t.info.alt_tile_mode == r.info.alt_tile_mode && t.info.endian == static_cast<u32>(r.info.endian) &&
           t.clear_word0 == r.clear_word0 && t.clear_word1 == r.clear_word1 && t.slice.tile_max == r.slice.tile_max && t.pitch.fmask_tile_max == r.pitch.fmask_tile_max &&
           (r.info.format == 0 || (t.Height() == r.Height() && t.GetColorSliceSize() == r.GetColorSliceSize() && t.GetDataFmt() == r.GetDataFmt() && t.GetNumberFmt() == r.GetNumberFmt() &&
                                   t.GetNumberConversion() == r.GetNumberConversion()));
}

bool Same(const DepthTarget& t, const AmdGpu::DepthBuffer& r) {
    return t.DepthValid() == r.DepthValid() && t.StencilValid() == r.StencilValid() && t.DepthWriteValid() == r.DepthWriteValid() && t.StencilWriteValid() == r.StencilWriteValid() &&
           t.Pitch() == r.Pitch() && t.Height() == r.Height() && t.DepthAddress() == r.DepthAddress() && t.StencilAddress() == r.StencilAddress() && t.DepthWriteAddress() == r.DepthWriteAddress() &&
           t.StencilWriteAddress() == r.StencilWriteAddress() && t.NumSamples() == r.NumSamples() && t.NumBits() == r.NumBits() && t.GetTileMode() == r.GetTileMode() && t.IsTiled() == r.IsTiled() &&
           (r.z_info.format == AmdGpu::DepthBuffer::ZFormat::Invalid || t.GetDepthSliceSize() == r.GetDepthSliceSize()) && static_cast<u32>(t.z_info.format) == static_cast<u32>(r.z_info.format) &&
           static_cast<u32>(t.stencil_info.format) == static_cast<u32>(r.stencil_info.format) && t.z_info.tile_split == r.z_info.tile_split && t.z_info.tile_surface_enable == r.z_info.tile_surface_enable;
}
} // namespace

ColorTarget ColorFromWords(const u32* words) {
    AmdGpu::ColorBuffer r;
    std::memcpy(&r, words, sizeof(r));
    ColorTarget t;
    t.base_address = r.base_address;
    t.pitch.tile_max = r.pitch.tile_max;
    t.pitch.fmask_tile_max = r.pitch.fmask_tile_max;
    t.slice.tile_max = r.slice.tile_max;
    t.view.slice_start = r.view.slice_start;
    t.view.slice_max = r.view.slice_max;
    t.info.endian = static_cast<u32>(r.info.endian);
    t.info.format = r.info.format;
    t.info.linear_general = r.info.linear_general;
    t.info.number_type = r.info.number_type;
    t.info.comp_swap = static_cast<ColorTarget::SwapMode>(static_cast<u32>(r.info.comp_swap));
    t.info.fast_clear = r.info.fast_clear;
    t.info.compression = r.info.compression;
    t.info.blend_bypass = r.info.blend_bypass;
    t.info.dcc_enable = r.info.dcc_enable;
    t.info.alt_tile_mode = r.info.alt_tile_mode;
    t.attrib.tile_mode_index = r.attrib.tile_mode_index;
    t.attrib.num_samples_log2 = r.attrib.num_samples_log2;
    t.attrib.num_fragments_log2 = r.attrib.num_fragments_log2;
    t.cmask_base_address = r.cmask_base_address;
    t.fmask_base_address = r.fmask_base_address;
    t.clear_word0 = r.clear_word0;
    t.clear_word1 = r.clear_word1;
    Note(Same(t, r));
    return t;
}

DepthTarget DepthFromWords(const u32* words) {
    AmdGpu::DepthBuffer r;
    std::memcpy(&r, words, sizeof(r));
    DepthTarget t;
    t.z_info.format = static_cast<DepthTarget::ZFormat>(static_cast<u32>(r.z_info.format));
    t.z_info.num_samples = r.z_info.num_samples;
    t.z_info.tile_split = r.z_info.tile_split;
    t.z_info.tile_mode_index = r.z_info.tile_mode_index;
    t.z_info.tile_surface_enable = r.z_info.tile_surface_enable;
    t.stencil_info.format = static_cast<DepthTarget::StencilFormat>(static_cast<u32>(r.stencil_info.format));
    t.z_read_base = r.z_read_base;
    t.stencil_read_base = r.stencil_read_base;
    t.z_write_base = r.z_write_base;
    t.stencil_write_base = r.stencil_write_base;
    t.depth_size.pitch_tile_max = r.depth_size.pitch_tile_max;
    t.depth_size.height_tile_max = r.depth_size.height_tile_max;
    t.depth_slice.tile_max = r.depth_slice.tile_max;
    Note(Same(t, r));
    return t;
}

namespace {
DepthTargetView ViewFromWord(u32 word) {
    AmdGpu::DepthView r;
    std::memcpy(&r, &word, sizeof(r));
    DepthTargetView v;
    v.slice_start = r.slice_start;
    v.slice_max = r.slice_max;
    v.z_read_only = r.z_read_only;
    v.stencil_read_only = r.stencil_read_only;
    Note(v.NumSlices() == r.NumSlices());
    return v;
}

DepthTargetControl ControlFromWord(u32 word) {
    AmdGpu::DepthControl r;
    std::memcpy(&r, &word, sizeof(r));
    DepthTargetControl c;
    c.stencil_enable = r.stencil_enable;
    c.depth_enable = r.depth_enable;
    c.depth_write_enable = r.depth_write_enable;
    c.depth_bounds_enable = r.depth_bounds_enable;
    c.depth_func = static_cast<CompareFunc>(static_cast<u32>(r.depth_func));
    c.backface_enable = r.backface_enable;
    c.stencil_ref_func = static_cast<CompareFunc>(static_cast<u32>(r.stencil_ref_func));
    c.stencil_bf_func = static_cast<CompareFunc>(static_cast<u32>(r.stencil_bf_func));
    c.enable_color_writes_on_depth_fail = r.enable_color_writes_on_depth_fail;
    c.disable_color_writes_on_depth_pass = r.disable_color_writes_on_depth_pass;
    return c;
}
} // namespace

void DecodeColor(Vulkan::DrawInputs& in, u32 slot) {
    in.color_buffers[slot] = ColorFromWords(in.raw_targets.color[slot]);
    g_color_decoded.fetch_add(1, std::memory_order_relaxed);
}

void DecodeDepth(Vulkan::DrawInputs& in) {
    in.depth_buffer = DepthFromWords(in.raw_targets.depth);
    in.depth_view = ViewFromWord(in.raw_targets.depth_view);
    in.depth_control = ControlFromWord(in.raw_targets.depth_control);
    g_depth_decoded.fetch_add(1, std::memory_order_relaxed);
}

void DecodeAll(Vulkan::DrawInputs& in) {
    for (u32 slot = 0; slot < RawTargetWords::ColorSlots; ++slot) {
        DecodeColor(in, slot);
    }
    DecodeDepth(in);
}

nlohmann::json Report() {
    return {{"decodes_checked", g_checked.load()}, {"differences", g_differences.load()}, {"colour_decoded", g_color_decoded.load()}, {"depth_decoded", g_depth_decoded.load()}};
}

} // namespace Core::NativeTarget
