// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <nlohmann/json_fwd.hpp>
#include "common/assert.h"
#include "common/types.h"
#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/tiling.h"

namespace Vulkan {
struct DrawInputs;
}

/// G1b (documents/tlg/native plan/executor-consumer-plan.md): the render target and depth target descriptions a draw reads, as plain numbers.
/// They replace `AmdGpu::ColorBuffer`, `DepthBuffer`, `DepthView` and `DepthControl` in the consumers (rasterizer, pipeline cache, image code). The members keep the names and paths the
/// consumers used (`info.format`, `view.slice_max`, `z_info.format` ...) and the methods keep their meaning, so the code that reads them did not change. The data layout enums (tile mode,
/// data format, number format) are the layout of the memory behind the targets and stay (G3 moves them to the layout module).
///
/// The only code that knows the register layout is `Decode*` (native_target_inputs.cpp): the producers (`native_draw_inputs`, `native_target_desc`) keep the record's register words in
/// `RawTargetWords`, merge the engine objects' description into them as before, and decode them into these structs.
namespace Core::NativeTarget {

enum class CompareFunc : u32 {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
};

struct ColorTarget {
    enum class SwapMode : u32 {
        Standard = 0,
        Alternate = 1,
        StandardReverse = 2,
        AlternateReverse = 3,
    };

    u32 base_address{};
    struct {
        u32 tile_max{};
        u32 fmask_tile_max{};
    } pitch;
    struct {
        u32 tile_max{};
    } slice;
    struct {
        u32 slice_start{};
        u32 slice_max{};
    } view;
    struct {
        u32 endian{};
        u32 format{};
        u32 linear_general{};
        u32 number_type{};
        SwapMode comp_swap{};
        u32 fast_clear{};
        u32 compression{};
        u32 blend_bypass{};
        u32 dcc_enable{};
        u32 alt_tile_mode{};
    } info;
    struct {
        AmdGpu::TileMode tile_mode_index{};
        u32 num_samples_log2{};
        u32 num_fragments_log2{};
    } attrib;
    u32 cmask_base_address{};
    u32 fmask_base_address{};
    u32 clear_word0{};
    u32 clear_word1{};

    operator bool() const {
        return base_address && info.format;
    }

    u32 Pitch() const {
        return (pitch.tile_max + 1) << 3;
    }

    u32 Height() const {
        return (slice.tile_max + 1) * 64 / Pitch();
    }

    u64 Address() const {
        return u64(base_address) << 8 | (info.linear_general ? (view.slice_start & 0xff) : 0);
    }

    VAddr CmaskAddress() const {
        return VAddr(cmask_base_address) << 8;
    }

    VAddr FmaskAddress() const {
        return VAddr(fmask_base_address) << 8;
    }

    u32 NumSamples() const {
        return 1 << attrib.num_fragments_log2;
    }

    u32 BaseSlice() const {
        return info.linear_general ? 0 : view.slice_start;
    }

    u32 NumSlices() const {
        return view.slice_max + 1;
    }

    u32 GetColorSliceSize() const {
        const auto num_bytes_per_element = AmdGpu::NumBitsPerBlock(AmdGpu::DataFormat(info.format)) / 8u;
        const auto slice_size = num_bytes_per_element * (slice.tile_max + 1) * 64u * NumSamples();
        return slice_size;
    }

    AmdGpu::TileMode GetTileMode() const {
        return info.linear_general ? AmdGpu::TileMode::DisplayLinearGeneral : attrib.tile_mode_index;
    }

    bool IsTiled() const {
        return GetTileMode() != AmdGpu::TileMode::DisplayLinearAligned && GetTileMode() != AmdGpu::TileMode::DisplayLinearGeneral;
    }

    AmdGpu::DataFormat GetDataFmt() const {
        return AmdGpu::RemapDataFormat(AmdGpu::DataFormat(info.format));
    }

    AmdGpu::NumberFormat GetNumberFmt() const {
        return AmdGpu::RemapNumberFormat(GetFixedNumberFormat(), AmdGpu::DataFormat(info.format));
    }

    AmdGpu::NumberConversion GetNumberConversion() const {
        return AmdGpu::MapNumberConversion(GetFixedNumberFormat(), AmdGpu::DataFormat(info.format));
    }

    AmdGpu::CompMapping Swizzle() const {
        using AmdGpu::CompMapping;
        using AmdGpu::CompSwizzle;
        // clang-format off
        static constexpr std::array<std::array<CompMapping, 4>, 4> mrt_swizzles{{
            // Standard
            std::array<CompMapping, 4>{{
                {.r = CompSwizzle::Red, .g = CompSwizzle::Zero, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Red, .g = CompSwizzle::Green, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Red, .g = CompSwizzle::Green, .b = CompSwizzle::Blue, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Red, .g = CompSwizzle::Green, .b = CompSwizzle::Blue, .a = CompSwizzle::Alpha},
            }},
            // Alternate
            std::array<CompMapping, 4>{{
                {.r = CompSwizzle::Green, .g = CompSwizzle::Zero, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Red, .g = CompSwizzle::Alpha, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Red, .g = CompSwizzle::Green, .b = CompSwizzle::Alpha, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Blue, .g = CompSwizzle::Green, .b = CompSwizzle::Red, .a = CompSwizzle::Alpha},
            }},
            // StandardReverse
            std::array<CompMapping, 4>{{
                {.r = CompSwizzle::Blue, .g = CompSwizzle::Zero, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Green, .g = CompSwizzle::Red, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Blue, .g = CompSwizzle::Green, .b = CompSwizzle::Red, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Alpha, .g = CompSwizzle::Blue, .b = CompSwizzle::Green, .a = CompSwizzle::Red},
            }},
            // AlternateReverse
            std::array<CompMapping, 4>{{
                {.r = CompSwizzle::Alpha, .g = CompSwizzle::Zero, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Alpha, .g = CompSwizzle::Red, .b = CompSwizzle::Zero, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Alpha, .g = CompSwizzle::Green, .b = CompSwizzle::Red, .a = CompSwizzle::Zero},
                {.r = CompSwizzle::Alpha, .g = CompSwizzle::Red, .b = CompSwizzle::Green, .a = CompSwizzle::Blue},
            }},
        }};
        // clang-format on
        const auto swap_idx = static_cast<u32>(info.comp_swap);
        const auto components_idx = AmdGpu::NumComponents(AmdGpu::DataFormat(info.format)) - 1;
        const auto mrt_swizzle = mrt_swizzles[swap_idx][components_idx];
        return AmdGpu::RemapSwizzle(AmdGpu::DataFormat(info.format), mrt_swizzle);
    }

    AmdGpu::NumberFormat GetFixedNumberFormat() const {
        // There is a small difference between T# and CB number types, account for it.
        const auto number_fmt = AmdGpu::NumberFormat(info.number_type);
        return number_fmt == AmdGpu::NumberFormat::SnormNz ? AmdGpu::NumberFormat::Srgb : number_fmt;
    }
};

struct DepthTarget {
    enum class ZFormat : u32 {
        Invalid = 0,
        Z16 = 1,
        Z32Float = 3,
    };

    enum class StencilFormat : u32 {
        Invalid = 0,
        Stencil8 = 1,
    };

    struct {
        ZFormat format{};
        u32 num_samples{};
        u32 tile_split{};
        u32 tile_mode_index{};
        u32 tile_surface_enable{};
    } z_info;
    struct {
        StencilFormat format{};
    } stencil_info;
    u32 z_read_base{};
    u32 stencil_read_base{};
    u32 z_write_base{};
    u32 stencil_write_base{};
    struct {
        u32 pitch_tile_max{};
        u32 height_tile_max{};
    } depth_size;
    struct {
        u32 tile_max{};
    } depth_slice;

    bool DepthValid() const {
        return DepthAddress() != 0 && z_info.format != ZFormat::Invalid;
    }

    bool StencilValid() const {
        return StencilAddress() != 0 && stencil_info.format != StencilFormat::Invalid;
    }

    bool DepthWriteValid() const {
        return DepthWriteAddress() != 0 && z_info.format != ZFormat::Invalid;
    }

    bool StencilWriteValid() const {
        return StencilWriteAddress() != 0 && stencil_info.format != StencilFormat::Invalid;
    }

    u32 Pitch() const {
        return (depth_size.pitch_tile_max + 1) << 3;
    }

    u32 Height() const {
        return (depth_size.height_tile_max + 1) << 3;
    }

    u64 DepthAddress() const {
        return u64(z_read_base) << 8;
    }

    u64 StencilAddress() const {
        return u64(stencil_read_base) << 8;
    }

    u64 DepthWriteAddress() const {
        return u64(z_write_base) << 8;
    }

    u64 StencilWriteAddress() const {
        return u64(stencil_write_base) << 8;
    }

    u32 NumSamples() const {
        return 1u << z_info.num_samples; // spec doesn't say it is a log2
    }

    u32 NumBits() const {
        return z_info.format == ZFormat::Z32Float ? 32 : 16;
    }

    u32 GetDepthSliceSize() const {
        ASSERT(z_info.format != ZFormat::Invalid);
        const auto bpe = NumBits() >> 3; // in bytes
        return (depth_slice.tile_max + 1) * 64 * bpe * NumSamples();
    }

    AmdGpu::TileMode GetTileMode() const {
        return static_cast<AmdGpu::TileMode>(z_info.tile_mode_index);
    }

    bool IsTiled() const {
        return GetTileMode() != AmdGpu::TileMode::DisplayLinearAligned && GetTileMode() != AmdGpu::TileMode::DisplayLinearGeneral;
    }
};

struct DepthTargetView {
    u32 slice_start{};
    u32 slice_max{};
    u32 z_read_only{};
    u32 stencil_read_only{};

    u32 NumSlices() const {
        return slice_max + 1u;
    }
};

struct DepthTargetControl {
    u32 stencil_enable{};
    u32 depth_enable{};
    u32 depth_write_enable{};
    u32 depth_bounds_enable{};
    CompareFunc depth_func{};
    u32 backface_enable{};
    CompareFunc stencil_ref_func{};
    CompareFunc stencil_bf_func{};
    u32 enable_color_writes_on_depth_fail{};
    u32 disable_color_writes_on_depth_pass{};
};

/// The register words of the targets exactly as the record carries them (colour buffer 15 dwords, depth buffer 8, depth view 1, depth control 1): the working copy of the producers.
struct RawTargetWords {
    static constexpr u32 ColorWords = 15;
    static constexpr u32 DepthWords = 8;
    static constexpr u32 ColorSlots = 8;
    u32 color[ColorSlots][ColorWords]{};
    u32 depth[DepthWords]{};
    u32 depth_view{};
    u32 depth_control{};
};

/// The decoders: the one place that reads the register layout. `DecodeColor(in, slot)` etc. make `in.color_buffers[slot]` ... from `in.raw_targets`; `DecodeAll` all of them.
void DecodeColor(Vulkan::DrawInputs& in, u32 slot);
void DecodeDepth(Vulkan::DrawInputs& in);
void DecodeAll(Vulkan::DrawInputs& in);

[[nodiscard]] ColorTarget ColorFromWords(const u32* words);
[[nodiscard]] DepthTarget DepthFromWords(const u32* words);

/// Shadow of the decoders (`g1_target_inputs` in /native): each decode is read back through the accessors and compared with the register-layout struct it came from.
[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeTarget
