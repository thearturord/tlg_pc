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
    bool alt_tile_mode{};  ///< neo mode alternative tiling: the engine never sets it (0 in every scene counted); only a descriptor read from guest memory can set it
    u8 bank_swizzle{};     ///< the bank swizzle bits of the address when `alt_tile_mode` is set (the host computes them from the tile mode, format and samples when it decodes a T#)
    u32 min_lod{};
    u8 swizzle[4]{};       ///< r, g, b, a after the host's per-format remap

    bool operator==(const ImageDesc&) const = default;

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

/// A hash of every field (the lookup cache of the rasterizer keys by it and compares with `==`)
[[nodiscard]] inline u64 HashImageDesc(const ImageDesc& d, u64 seed = 0) noexcept {
    u64 h = seed ^ 0x9E3779B97F4A7C15ull;
    const auto mix = [&h](u64 v) {
        h = (h ^ v) * 0xC2B2AE3D27D4EB4Full;
        h ^= h >> 29;
    };
    mix(d.address);
    mix(d.vk_format | static_cast<u64>(d.vk_format_storage) << 32);
    mix(d.num_bits | static_cast<u64>(d.is_block) << 32 | static_cast<u64>(d.pow2_pad) << 33 | static_cast<u64>(d.alt_tile_mode) << 34 | static_cast<u64>(d.bank_swizzle) << 40);
    mix(d.type | static_cast<u64>(d.width) << 32);
    mix(d.height | static_cast<u64>(d.depth) << 32);
    mix(d.pitch | static_cast<u64>(d.tile_mode) << 32);
    mix(static_cast<u64>(d.base_level) | static_cast<u64>(d.last_level) << 16 | static_cast<u64>(d.base_array) << 32 | static_cast<u64>(d.last_array) << 48);
    mix(d.min_lod | static_cast<u64>(d.swizzle[0]) << 32 | static_cast<u64>(d.swizzle[1]) << 40 | static_cast<u64>(d.swizzle[2]) << 48 | static_cast<u64>(d.swizzle[3]) << 56);
    return h;
}

/// F4.1 buffers: what the host reads of a buffer descriptor (V#) when it binds a storage / uniform buffer: where, how big and, for a formatted buffer, the format numbers.
struct BufferDesc {
    u64 address{};
    u64 size{};       ///< bytes (records x stride, or the byte count of a raw buffer)
    u32 stride{};
    u32 num_records{};
    u32 data_format{}; ///< the console's data / number format numbers (formatted buffer bindings only; the host maps them with the recompiler's tables)
    u32 num_format{};
};

/// The one decode of the four dwords of a V#: base address (40 bits), stride (14 bits), record count, number / data format. `size` is the byte count the host binds (the console's own
/// rule: the record count alone when the stride is 0, else records x stride, in 32 bits).
[[nodiscard]] inline BufferDesc BufferDescFromWords(const u32* w) noexcept {
    BufferDesc d;
    d.address = w[0] | static_cast<u64>(w[1] & 0xFFu) << 32;
    d.stride = (w[1] >> 16) & 0x3FFFu;
    d.num_records = w[2];
    d.num_format = (w[3] >> 12) & 7u;
    d.data_format = (w[3] >> 15) & 0xFu;
    d.size = d.stride == 0 ? d.num_records : static_cast<u32>(d.stride * d.num_records);
    return d;
}

/// The same from the recompiler's V# type (16 bytes of the same words)
template <class Sharp>
[[nodiscard]] inline BufferDesc BufferDescOf(const Sharp& sharp) noexcept {
    static_assert(sizeof(Sharp) == 16);
    return BufferDescFromWords(reinterpret_cast<const u32*>(&sharp));
}

/// F4.1 samplers: what the host sampler creation reads of the console sampler descriptor (S#), as Vulkan-ready numbers. All fields are plain 32 bit values with no padding, so the struct
/// is its own cache key (hash of its bytes). `valid` false: the descriptor holds a value the host refuses (the caller substitutes the all-zero sampler).
struct SamplerDesc {
    u32 mag_filter{};  ///< `vk::Filter`
    u32 min_filter{};
    u32 mip_mode{};    ///< `vk::SamplerMipmapMode`
    u32 address_u{};   ///< `vk::SamplerAddressMode`
    u32 address_v{};
    u32 address_w{};
    u32 compare_op{};  ///< `vk::CompareOp` (applied only for depth uses)
    u32 border{};      ///< `vk::BorderColor`
    u32 border_index{}; ///< index into the custom border colour table (border colour pointer), used when `border` is the custom one
    u32 aniso_filter{}; ///< 1: one of the two filters is an anisotropic one
    float max_aniso{1.0f};
    float lod_bias{};
    float min_lod{};
    float max_lod{};
    u32 valid{1};
    u32 pad{};         ///< keeps the struct free of padding bytes (hashed)
};
static_assert(sizeof(SamplerDesc) == 64);

} // namespace Core::NativeDesc
