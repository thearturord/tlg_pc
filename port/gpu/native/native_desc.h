// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <bit>
#include "common/types.h"

/// F4.1 (documents/tlg/native plan/phase-notes/F4-notes.md): the description of an image a draw samples, in plain numbers and with no PS4 type. It holds what the host image and
/// view constructors read from the console descriptor (T#): memory, size, level and layer ranges, type, tile mode, format (a Vulkan format as a number), bits per block, component
/// select after the host's remap. The accessors are the descriptor's helper functions in plain code (layer count rules, view level / layer constraints, view type), so that
/// the sharp decode can go at F4.6.
///
/// Two producers: `NativeImageDesc::DeriveDesc` (from the engine's objects, native) and `ImageDescFromSharp` (from the T# words with the emulator's helpers, the reference).
/// F4.1 compares them field by field in the shadow (`TLG_DESC_CHECKS=1`); F4.2 builds the host `ImageInfo` / `ImageViewInfo` from this struct.
namespace Core::NativeDesc {

/// The type numbers of the descriptor (same values as the console's image type field)
enum ImageType : u32 { TypeInvalid = 0, Type1D = 8, Type2D = 9, Type3D = 10, TypeCube = 11, Type1DArray = 12, Type2DArray = 13, Type2DMsaa = 14, Type2DMsaaArray = 15 };

/// The component select values (same numbers as the console's)
enum Swizzle : u8 { SwizzleZero = 0, SwizzleOne = 1, SwizzleR = 4, SwizzleG = 5, SwizzleB = 6, SwizzleA = 7 };

struct ImageDesc {
    u64 address{};         ///< bytes
    u32 vk_format{};       ///< the host pixel format (a `vk::Format` number) before the shader's depth promotion
    u32 vk_format_storage{}; ///< the format of a written (storage) view: `vk_format` with the sRGB variant replaced by its unorm one
    u32 num_bits{};        ///< bits per block (element for plain formats, 4x4 block for block compressed ones)
    bool is_block{};
    u32 type{};            ///< `ImageType`
    u32 width{};           ///< texels
    u32 height{};
    u32 depth{};           ///< the descriptor's depth field: volume depth - 1, cube count - 1, or layer count - 1
    u32 pitch{};           ///< texels
    u32 base_level{}, last_level{};
    u32 base_array{}, last_array{};
    u32 tile_mode{};       ///< the tiling index
    bool pow2_pad{};
    bool alt_tile_mode{};  ///< neo mode alternative tiling: the engine never sets it, a description with it set is refused by the native consumer
    u32 min_lod{};
    u8 swizzle[4]{};       ///< r, g, b, a after the host's per-format remap

    [[nodiscard]] u32 BaseType() const noexcept {
        if (type == Type1DArray) {
            return Type1D;
        }
        if (type == Type2DArray || type == Type2DMsaa || type == Type2DMsaaArray || type == TypeCube) {
            return Type2D;
        }
        return type;
    }
    [[nodiscard]] bool IsVolume() const noexcept {
        return BaseType() == Type3D;
    }
    [[nodiscard]] bool IsMsaa() const noexcept {
        return type == Type2DMsaa || type == Type2DMsaaArray;
    }
    [[nodiscard]] u32 NumLayers() const noexcept {
        u32 slices = depth + 1; // the depth field is the layer count for arrays
        if (type == Type3D) {
            slices = 1; // the depth of a volume is its real depth
        } else if (type == TypeCube) {
            slices *= 6; // the number of whole cubes
        }
        return pow2_pad ? std::bit_ceil(slices) : slices;
    }
    [[nodiscard]] u32 NumLevels() const noexcept {
        return IsMsaa() ? 1 : last_level + 1;
    }
    [[nodiscard]] u32 NumSamples() const noexcept {
        return IsMsaa() ? 1u << last_level : 1;
    }
    [[nodiscard]] u32 ViewType(bool is_array) const noexcept {
        if (type == Type1DArray && !is_array) {
            return Type1D;
        }
        if (type == Type2DArray && !is_array) {
            return Type2D;
        }
        if (type == Type2DMsaaArray && !is_array) {
            return Type2DMsaa;
        }
        return type;
    }
    [[nodiscard]] u32 NumViewLevels(bool is_array) const noexcept {
        const u32 view = ViewType(is_array);
        if (view == Type2DMsaa || view == Type2DMsaaArray) {
            return 1;
        }
        const u32 max_level = std::min<u32>(last_level + 1, NumLevels());
        return max_level > base_level ? max_level - base_level : 1;
    }
    [[nodiscard]] u32 NumViewLayers(bool is_array) const noexcept {
        switch (ViewType(is_array)) {
        case Type1D:
        case Type2D:
        case Type2DMsaa:
        case Type3D:
            return 1;
        default: {
            const u32 max_array = std::min<u32>(last_array + 1, NumLayers());
            return max_array > base_array ? max_array - base_array : 1;
        }
        }
    }
    [[nodiscard]] bool IsTiled() const noexcept {
        return tile_mode != 8 && tile_mode != 31; // display linear aligned, display linear general
    }
};

} // namespace Core::NativeDesc
