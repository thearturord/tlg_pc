// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_image_desc.h"
#include "native/native_view_table.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_desc.h"
#include "native/native_device.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Core::NativeImageDesc {

namespace {

/// The console data format / number format and the component select (x | y << 3 | z << 6 | w << 9) the engine's descriptor builder gives each Direct3D format it uses.
/// Read off the live census (TLG_DESC_CHECKS, 4 million bindings: every engine format maps to exactly one entry); a format with no entry is reported and the record's T# decides.
struct FormatEntry {
    u32 dxgi, data_format, num_format, dst_sel;
};
constexpr FormatEntry FormatTable[] = {
    {10, 12, 7, 0xfac}, {16, 11, 7, 0x22c}, {24, 9, 0, 0xfac},  {26, 6, 7, 0x3ac}, {28, 10, 0, 0xfac}, {29, 10, 9, 0xfac}, {31, 10, 1, 0xfac}, {34, 5, 7, 0x22c},
    {37, 5, 1, 0x22c},  {39, 4, 4, 0x4},    {40, 4, 7, 0x204},  {41, 4, 7, 0x204}, {46, 4, 7, 0x204}, {49, 3, 0, 0x22c},  {54, 2, 7, 0x204},  {61, 1, 0, 0x204},
    {70, 35, 0, 0xfac}, {76, 37, 0, 0xfac}, {79, 38, 0, 0x204}, {84, 39, 1, 0x22c}, {87, 10, 0, 0xf2e}, {94, 40, 0, 0x3ac}, {95, 40, 0, 0x3ac}, {97, 41, 0, 0xfac},
};

const FormatEntry* FindFormat(u32 dxgi) {
    for (const auto& entry : FormatTable) {
        if (entry.dxgi == dxgi) {
            return &entry;
        }
    }
    return nullptr;
}

/// F4.1: what the host makes of each console format above, as plain numbers: the Vulkan format (and the one a written view uses: sRGB replaced by unorm), bits per block, block
/// compression, and the component select after the host's per-format remap (RemapSwizzle). Checked field by field against the host's own decode (`CompareDesc`).
struct PlainEntry {
    u32 dxgi;
    vk::Format vk, vk_storage;
    u32 bits;
    bool block;
    u8 swz[4];
};
constexpr u8 R = NativeDesc::SwizzleR, G = NativeDesc::SwizzleG, B = NativeDesc::SwizzleB, A = NativeDesc::SwizzleA, Z = NativeDesc::SwizzleZero, O = NativeDesc::SwizzleOne;
constexpr PlainEntry PlainTable[] = {
    {10, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, 64, false, {R, G, B, A}},
    {16, vk::Format::eR32G32Sfloat, vk::Format::eR32G32Sfloat, 64, false, {R, G, Z, O}},
    {24, vk::Format::eA2B10G10R10UnormPack32, vk::Format::eA2B10G10R10UnormPack32, 32, false, {R, G, B, A}},
    {26, vk::Format::eB10G11R11UfloatPack32, vk::Format::eB10G11R11UfloatPack32, 32, false, {R, G, B, O}},
    {28, vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm, 32, false, {R, G, B, A}},
    {29, vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm, 32, false, {R, G, B, A}},
    {31, vk::Format::eR8G8B8A8Snorm, vk::Format::eR8G8B8A8Snorm, 32, false, {R, G, B, A}},
    {34, vk::Format::eR16G16Sfloat, vk::Format::eR16G16Sfloat, 32, false, {R, G, Z, O}},
    {37, vk::Format::eR16G16Snorm, vk::Format::eR16G16Snorm, 32, false, {R, G, Z, O}},
    {39, vk::Format::eR32Uint, vk::Format::eR32Uint, 32, false, {R, Z, Z, Z}},
    {40, vk::Format::eR32Sfloat, vk::Format::eR32Sfloat, 32, false, {R, Z, Z, O}},
    {41, vk::Format::eR32Sfloat, vk::Format::eR32Sfloat, 32, false, {R, Z, Z, O}},
    {46, vk::Format::eR32Sfloat, vk::Format::eR32Sfloat, 32, false, {R, Z, Z, O}},
    {49, vk::Format::eR8G8Unorm, vk::Format::eR8G8Unorm, 16, false, {R, G, Z, O}},
    {54, vk::Format::eR16Sfloat, vk::Format::eR16Sfloat, 16, false, {R, Z, Z, O}},
    {61, vk::Format::eR8Unorm, vk::Format::eR8Unorm, 8, false, {R, Z, Z, O}},
    {70, vk::Format::eBc1RgbaUnormBlock, vk::Format::eBc1RgbaUnormBlock, 64, true, {R, G, B, A}},
    {76, vk::Format::eBc3UnormBlock, vk::Format::eBc3UnormBlock, 128, true, {R, G, B, A}},
    {79, vk::Format::eBc4UnormBlock, vk::Format::eBc4UnormBlock, 64, true, {R, Z, Z, O}},
    {84, vk::Format::eBc5SnormBlock, vk::Format::eBc5SnormBlock, 128, true, {R, G, Z, O}},
    {87, vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm, 32, false, {B, G, R, A}},
    {94, vk::Format::eBc6HUfloatBlock, vk::Format::eBc6HUfloatBlock, 128, true, {R, G, B, O}},
    {95, vk::Format::eBc6HUfloatBlock, vk::Format::eBc6HUfloatBlock, 128, true, {R, G, B, O}},
    {97, vk::Format::eBc7UnormBlock, vk::Format::eBc7UnormBlock, 128, true, {R, G, B, A}},
};

const PlainEntry* FindPlain(u32 dxgi) {
    for (const auto& entry : PlainTable) {
        if (entry.dxgi == dxgi) {
            return &entry;
        }
    }
    return nullptr;
}

// D3D11_SRV_DIMENSION / D3D11_UAV_DIMENSION values the views carry
constexpr u32 Dim1D = 2, Dim1DArray = 3, Dim2D = 4, Dim2DArray = 5, Dim3D = 8, DimCube = 9, DimCubeArray = 10;

// AmdGpu::ImageType
constexpr u32 TypeColor1D = 8, TypeColor2D = 9, TypeColor3D = 10, TypeCube = 11, TypeColor1DArray = 12, TypeColor2DArray = 13;

enum Field : u32 {
    FAddress,
    FDataFormat,
    FNumFormat,
    FWidth,
    FHeight,
    FDepth,
    FPitch,
    FDstSel,
    FBaseLevel,
    FLastLevel,
    FTiling,
    FPow2Pad,
    FType,
    FBaseArray,
    FLastArray,
    FMemType,
    FRest,
    FieldCount
};
constexpr const char* FieldNames[FieldCount] = {"base_address", "data_format", "num_format", "width", "height", "depth", "pitch", "dst_sel",
                                                "base_level", "last_level", "tiling_index", "pow2pad", "type", "base_array", "last_array", "memory_type", "other_words"};

std::atomic<u64> g_sampler_derived{}, g_sampler_unknown{}, g_sampler_differing{};
std::atomic<u32> g_sampler_unknown_logged{};
std::atomic<u64> g_held_equal{}, g_held_differ{}, g_held_none{}, g_snap_equal{}, g_snap_differ{};
std::array<std::atomic<u64>, 8> g_snap_word_differs{};
std::array<std::atomic<u64>, 8> g_held_word_differs{};
std::atomic<u64> g_no_view{}, g_view_unknown{}, g_view_without_desc{}, g_view_with_desc{}, g_compared{}, g_equal{}, g_unknown_object{}, g_unknown_format{};
std::array<std::atomic<u64>, FieldCount> g_differing{};
std::atomic<u32> g_logged{};
std::mutex g_mutex;
std::map<std::string, std::set<std::string>> g_sampler_map; ///< sampler object field value -> the S# values seen with it
std::array<std::array<std::string, 12>, FieldCount> g_examples; ///< the first distinct examples per field (derived -> actual, then the resource)
std::array<u32, FieldCount> g_example_count{};
std::map<std::string, std::set<std::pair<u32, u32>>> g_pitches; ///< "format F tile T" -> (width, pitch) pairs seen

} // namespace

/// Where a view object keeps its T#: shader resource view +0x58, unordered access view +0x54
constexpr u64 HeldOffset(NativeDevice::Kind kind) noexcept {
    return kind == NativeDevice::Kind::UnorderedAccessView ? 0x54 : 0x58;
}

Result Derive(u64 resource_id, u64 view_id, AmdGpu::Image& out) {
    using NativeDevice::Kind;
    if (Core::ViewTable::IsSubresourceKey(view_id)) {
        view_id = 0; // note 113: a texture subresource key, no view object (as before the key)
    }
    NativeDevice::Object res, view;
    if (resource_id == 0 || !NativeDevice::Lookup(resource_id, res)) {
        return Result::UnknownObject;
    }
    const bool view_known = view_id != 0 && NativeDevice::Lookup(view_id, view);
    const bool has_view = view_known && view.has_desc;
    (view_id == 0 ? g_no_view : !view_known ? g_view_unknown : has_view ? g_view_with_desc : g_view_without_desc).fetch_add(1, std::memory_order_relaxed);
    const u32 dxgi = has_view ? view.desc[0] : res.format;
    const FormatEntry* format = FindFormat(dxgi);
    if (format == nullptr) {
        return Result::UnknownFormat;
    }
    out = {};
    res.mips = std::max<u32>(res.mips, 1);
    out.base_address = res.address >> 8;
    out.data_format = format->data_format;
    out.num_format = format->num_format;
    out.dst_sel_x = format->dst_sel & 7;
    out.dst_sel_y = (format->dst_sel >> 3) & 7;
    out.dst_sel_z = (format->dst_sel >> 6) & 7;
    out.dst_sel_w = (format->dst_sel >> 9) & 7;
    out.width = res.width - 1;
    out.height = res.height - 1;
    const u32 layers = std::max<u32>(res.layers, 1);
    // tiling: the engine puts depth targets in the linear-aligned mode, color targets in 14, every other 2D / 1D texture in 13 and volumes in 19
    out.tiling_index = res.kind == Kind::Texture3D ? 19 : (res.bind & 0x40) != 0 ? 0 : (res.bind & 0x20) != 0 ? 14 : 13;
    out.pow2pad = res.mips > 1 ? 1 : 0;
    // the pitch the engine's surface has: the width (rounded up to a power of two when the texture has mips), aligned to 8 pixels, to 32 for block compressed formats and to 256
    // for depth surfaces and single byte render targets
    {
        const bool block_compressed = format->data_format >= 35 && format->data_format <= 41;
        const bool one_byte = format->data_format == 1;
        const u32 align = block_compressed ? 32 : (out.tiling_index == 0 || (out.tiling_index == 14 && one_byte)) ? 256 : 8;
        u32 pitch = out.pow2pad != 0 ? std::bit_ceil(res.width) : res.width;
        pitch = (pitch + align - 1) / align * align;
        out.pitch = pitch - 1;
    }
    out.perf_modulation = 7;
    // the memory type of a texture: unordered access 3 (and the L2 type 1), every other texture the second memory type bit
    if ((res.bind & 0x80) != 0) {
        out.mtype = 3;
        out.mtype_l2 = 1;
    } else {
        out.mtype2 = 1;
    }

    // the type and the ranges
    u32 base_level = 0, last_level = res.mips - 1, base_array = 0, last_array = 0;
    u32 type = res.kind == Kind::Texture3D ? TypeColor3D : res.kind == Kind::Texture1D ? (layers > 1 ? TypeColor1DArray : TypeColor1D) : layers == 6 ? TypeCube : layers > 1 ? TypeColor2DArray : TypeColor2D;
    if (has_view) {
        const u32 dim = view.desc[1];
        if (view.kind == Kind::ShaderResourceView) {
            const u32 most_detailed = std::min(view.desc[2], res.mips - 1);
            const u32 levels = view.desc[3] == 0xffffffffu ? res.mips - most_detailed : view.desc[3];
            base_level = most_detailed;
            last_level = most_detailed + levels - 1;
            if (dim == Dim2DArray || dim == Dim1DArray) {
                base_array = view.desc[4];
                last_array = base_array + (view.desc[5] == 0xffffffffu ? layers - base_array : view.desc[5]) - 1;
            }
        } else if (view.kind == Kind::UnorderedAccessView) {
            base_level = last_level = view.desc[2];
            if (dim == Dim2DArray || dim == Dim1DArray) {
                base_array = view.desc[3];
                last_array = base_array + (view.desc[4] == 0xffffffffu ? layers - base_array : view.desc[4]) - 1;
            }
        }
        if (res.kind != Kind::Texture3D) {
            type = dim == Dim1D ? TypeColor1D : dim == Dim1DArray ? TypeColor1DArray : dim == Dim2D ? TypeColor2D : dim == Dim2DArray ? TypeColor2DArray : (dim == DimCube || dim == DimCubeArray) ? TypeCube : type;
        }
    }
    if (type == TypeColor3D && has_view && view.kind == Kind::UnorderedAccessView) {
        base_array = view.desc[3]; // a volume's slices of a UAV (FirstWSlice, WSize)
        last_array = base_array + (view.desc[4] == 0xffffffffu ? res.depth - base_array : view.desc[4]) - 1;
    } else if (type == TypeCube && !(has_view && view.kind == Kind::ShaderResourceView && (view.desc[1] == Dim2DArray || view.desc[1] == Dim1DArray))) {
        last_array = layers - 1;
    }
    out.type = type;
    out.base_level = base_level;
    out.last_level = last_level;
    out.base_array = base_array;
    out.last_array = last_array;
    // The engine builds the descriptor it hands to the shaders when it creates the view (a copy of the one inside the texture object, with the view's own size and level
    // fields) and keeps it in the view object (shader resource view +0x58, unordered access view +0x54: the creation functions `FUN_02493620` / `FUN_024999e0`).
    // Something changes it after the creation (the height of a 1088 row texture reads 1080 later; the writer is not identified), so the live object is read, not the table's
    // creation time copy (`tsharp`, kept only for the snapshot counter). The Direct3D description has two numbers it cannot say: a texture the engine allocated with padded rows (1088 for 1080) and a
    // level count it limits (7 of 8 for a block compressed cube). Where the view's copy still names this texture (same memory and format) those fields come from it.
    if (view_known && view.object != 0) {
        AmdGpu::Image held;
        std::memcpy(&held, reinterpret_cast<const void*>(view.object + HeldOffset(view.kind)), 32);
        if (held.base_address == out.base_address && held.data_format == out.data_format && held.num_format == out.num_format) {
            out.width = held.width;
            out.height = held.height;
            out.last_level = held.last_level;
        }
    }
    out.depth = type == TypeColor3D ? res.depth - 1 : type == TypeCube ? layers / 6 - 1 : layers - 1;
    return Result::Derived;
}

bool DeriveDesc(u64 resource_id, u64 view_id, NativeDesc::ImageDesc& out) {
    if (Core::ViewTable::IsSubresourceKey(view_id)) {
        view_id = 0; // note 113: a texture subresource key, no view object (as before the key)
    }
    using NativeDevice::Kind;
    NativeDevice::Object res, view;
    if (resource_id == 0 || !NativeDevice::Lookup(resource_id, res)) {
        return false;
    }
    const bool view_known = view_id != 0 && NativeDevice::Lookup(view_id, view);
    const bool has_view = view_known && view.has_desc;
    const u32 dxgi = has_view ? view.desc[0] : res.format;
    const FormatEntry* format = FindFormat(dxgi);
    const PlainEntry* plain = FindPlain(dxgi);
    if (format == nullptr || plain == nullptr) {
        return false;
    }
    out = {};
    res.mips = std::max<u32>(res.mips, 1);
    out.address = res.address >> 8 << 8;
    out.vk_format = static_cast<u32>(plain->vk);
    out.vk_format_storage = static_cast<u32>(plain->vk_storage);
    out.num_bits = plain->bits;
    out.is_block = plain->block;
    std::memcpy(out.swizzle, plain->swz, 4);
    out.width = res.width;
    out.height = res.height;
    const u32 layers = std::max<u32>(res.layers, 1);
    out.tile_mode = res.kind == Kind::Texture3D ? 19 : (res.bind & 0x40) != 0 ? 0 : (res.bind & 0x20) != 0 ? 14 : 13;
    out.pow2_pad = res.mips > 1;
    {
        const bool block_compressed = format->data_format >= 35 && format->data_format <= 41;
        const bool one_byte = format->data_format == 1;
        const u32 align = block_compressed ? 32 : (out.tile_mode == 0 || (out.tile_mode == 14 && one_byte)) ? 256 : 8;
        const u32 pitch = out.pow2_pad ? std::bit_ceil(res.width) : res.width;
        out.pitch = (pitch + align - 1) / align * align;
    }
    u32 base_level = 0, last_level = res.mips - 1, base_array = 0, last_array = 0;
    u32 type = res.kind == Kind::Texture3D ? NativeDesc::Type3D
               : res.kind == Kind::Texture1D ? (layers > 1 ? NativeDesc::Type1DArray : NativeDesc::Type1D)
               : layers == 6                 ? NativeDesc::TypeCube
               : layers > 1                  ? NativeDesc::Type2DArray
                                             : NativeDesc::Type2D;
    if (has_view) {
        const u32 dim = view.desc[1];
        if (view.kind == Kind::ShaderResourceView) {
            const u32 most_detailed = std::min(view.desc[2], res.mips - 1);
            const u32 levels = view.desc[3] == 0xffffffffu ? res.mips - most_detailed : view.desc[3];
            base_level = most_detailed;
            last_level = most_detailed + levels - 1;
            if (dim == Dim2DArray || dim == Dim1DArray) {
                base_array = view.desc[4];
                last_array = base_array + (view.desc[5] == 0xffffffffu ? layers - base_array : view.desc[5]) - 1;
            }
        } else if (view.kind == Kind::UnorderedAccessView) {
            base_level = last_level = view.desc[2];
            if (dim == Dim2DArray || dim == Dim1DArray) {
                base_array = view.desc[3];
                last_array = base_array + (view.desc[4] == 0xffffffffu ? layers - base_array : view.desc[4]) - 1;
            }
        }
        if (res.kind != Kind::Texture3D) {
            type = dim == Dim1D        ? NativeDesc::Type1D
                   : dim == Dim1DArray ? NativeDesc::Type1DArray
                   : dim == Dim2D      ? NativeDesc::Type2D
                   : dim == Dim2DArray ? NativeDesc::Type2DArray
                   : (dim == DimCube || dim == DimCubeArray) ? NativeDesc::TypeCube
                                                              : type;
        }
    }
    if (type == NativeDesc::Type3D && has_view && view.kind == Kind::UnorderedAccessView) {
        base_array = view.desc[3];
        last_array = base_array + (view.desc[4] == 0xffffffffu ? res.depth - base_array : view.desc[4]) - 1;
    } else if (type == NativeDesc::TypeCube && !(has_view && view.kind == Kind::ShaderResourceView && (view.desc[1] == Dim2DArray || view.desc[1] == Dim1DArray))) {
        last_array = layers - 1;
    }
    out.type = type;
    out.base_level = base_level;
    out.last_level = last_level;
    out.base_array = base_array;
    out.last_array = last_array;
    // the descriptor the engine keeps inside the view object: the padded rows and the limited level count (see Derive)
    if (view_known && view.object != 0) {
        u64 q[2];
        std::memcpy(q, reinterpret_cast<const void*>(view.object + HeldOffset(view.kind)), sizeof(q));
        const u64 held_address = q[0] & ((1ull << 38) - 1), held_dfmt = (q[0] >> 52) & 0x3f, held_nfmt = (q[0] >> 58) & 0xf;
        if (held_address == res.address >> 8 && held_dfmt == format->data_format && held_nfmt == format->num_format) {
            out.width = static_cast<u32>(q[1] & 0x3fff) + 1;
            out.height = static_cast<u32>((q[1] >> 14) & 0x3fff) + 1;
            out.last_level = static_cast<u32>((q[1] >> 48) & 0xf);
        }
    }
    out.depth = type == NativeDesc::Type3D ? res.depth - 1 : type == NativeDesc::TypeCube ? layers / 6 - 1 : layers - 1;
    return true;
}

bool Decides() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_IMAGE_DESC");
        return e == nullptr || *e != '0';
    }();
    return on;
}


std::atomic<u64> g_use_served{}, g_use_unavailable{}, g_use_alt_tile{}, g_use_depth_refused{};

NativeDesc::SamplerDesc SamplerDescFromWords(const u32* words) noexcept {
    NativeDesc::SamplerDesc d;
    const u64 raw0 = words[0] | static_cast<u64>(words[1]) << 32, raw1 = words[2] | static_cast<u64>(words[3]) << 32;
    const auto bits = [](u64 raw, u32 pos, u32 count) { return static_cast<u32>((raw >> pos) & ((1ull << count) - 1)); };
    const auto address = [](u32 mode) -> u32 {
        switch (mode) {
        case 0: return static_cast<u32>(vk::SamplerAddressMode::eRepeat);
        case 1: return static_cast<u32>(vk::SamplerAddressMode::eMirroredRepeat);
        case 2: return static_cast<u32>(vk::SamplerAddressMode::eClampToEdge);
        case 3: case 5: case 7: return static_cast<u32>(vk::SamplerAddressMode::eMirrorClampToEdge); // the host's closest equivalents of the mirror once modes
        default: return static_cast<u32>(vk::SamplerAddressMode::eClampToBorder);                    // 4 half border, 6 border
        }
    };
    const u32 aniso = bits(raw0, 9, 3), filter_mode = bits(raw0, 29, 2), mip = bits(raw1, 26, 2), mag = bits(raw1, 20, 2), min = bits(raw1, 22, 2);
    d.valid = aniso <= 4 && filter_mode <= 2 && mip <= 2;
    d.address_u = address(bits(raw0, 0, 3));
    d.address_v = address(bits(raw0, 3, 3));
    d.address_w = address(bits(raw0, 6, 3));
    d.max_aniso = aniso <= 4 ? static_cast<float>(1u << aniso) : 1.0f;
    d.compare_op = bits(raw0, 12, 3); // the console numbering equals vk::CompareOp (never .. always)
    d.min_lod = static_cast<float>(bits(raw0, 32, 12)) * SamplerDescLodScale;
    d.max_lod = static_cast<float>(bits(raw0, 44, 12)) * SamplerDescLodScale;
    d.lod_bias = static_cast<float>(static_cast<s16>((bits(raw1, 0, 14) ^ 0x2000u) - 0x2000u)) * SamplerDescLodScale;
    d.mag_filter = static_cast<u32>((mag & 1) ? vk::Filter::eLinear : vk::Filter::eNearest); // 1 bilinear, 3 anisotropic linear
    d.min_filter = static_cast<u32>((min & 1) ? vk::Filter::eLinear : vk::Filter::eNearest);
    d.aniso_filter = (mag >= 2 || min >= 2) ? 1 : 0;
    d.mip_mode = static_cast<u32>(mip == 2 ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest);
    d.border_index = bits(raw1, 32, 12);
    switch (bits(raw1, 62, 2)) {
    case 0: d.border = static_cast<u32>(vk::BorderColor::eFloatTransparentBlack); break;
    case 1: d.border = static_cast<u32>(vk::BorderColor::eFloatOpaqueBlack); break;
    case 2: d.border = static_cast<u32>(vk::BorderColor::eFloatOpaqueWhite); break;
    default: d.border = static_cast<u32>(vk::BorderColor::eFloatCustomEXT); break;
    }
    return d;
}

bool FromDescOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_IMAGE_FROM_DESC");
        return e == nullptr || *e != '0';
    }();
    return on;
}

bool DescForBinding(u64 resource_id, u64 view_id, bool cube_to_2d_array, bool is_depth, NativeDesc::ImageDesc& out) {
    if (!DeriveDesc(resource_id, view_id, out)) {
        g_use_unavailable.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (out.alt_tile_mode) {
        g_use_alt_tile.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (cube_to_2d_array) {
        out.type = NativeDesc::Type2DArray;
        out.depth = (out.depth + 1) * 6 - 1;
    }
    if (is_depth) {
        const auto format = static_cast<vk::Format>(out.vk_format);
        if (format != vk::Format::eR32Sfloat && format != vk::Format::eR32Uint && format != vk::Format::eR16Sfloat && format != vk::Format::eR16Unorm) {
            g_use_depth_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    g_use_served.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool DeriveSampler(u64 object, const u32* record_words, u32* out) {
    const auto rd = [&](u32 off) {
        u32 v;
        std::memcpy(&v, reinterpret_cast<const void*>(object + off), 4);
        return v;
    };
    const auto fl = [&](u32 off) {
        float v;
        std::memcpy(&v, reinterpret_cast<const void*>(object + off), 4);
        return v;
    };
    const auto undecided = [&](const char* why) {
        g_sampler_unknown.fetch_add(1, std::memory_order_relaxed);
        if (g_sampler_unknown_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            Core::RareEvents::Note("sampler_desc_value_unknown", fmt::format("{}: filter {:#x} address {} {} {} aniso {} compare {} border {:#x} {:#x} {:#x} {:#x} bias {:#x}", why, rd(0x30), rd(0x34), rd(0x38), rd(0x3c), rd(0x44),
                                                                              rd(0x48), rd(0x4c), rd(0x50), rd(0x54), rd(0x58), rd(0x40)));
        }
        return false;
    };
    AmdGpu::Sampler s{};
    // filters: the engine's filter values (met so far: 0 point, 0x7 and 0x10 trilinear, 0x8 anisotropic linear)
    switch (rd(0x30)) {
    case 0x0:
        s.xy_mag_filter.Assign(AmdGpu::Filter::Point);
        s.xy_min_filter.Assign(AmdGpu::Filter::Point);
        s.z_filter.Assign(1);
        s.mip_filter.Assign(AmdGpu::MipFilter::Point);
        break;
    case 0x7:
    case 0x10:
        s.xy_mag_filter.Assign(AmdGpu::Filter::Bilinear);
        s.xy_min_filter.Assign(AmdGpu::Filter::Bilinear);
        s.z_filter.Assign(2);
        s.mip_filter.Assign(AmdGpu::MipFilter::Linear);
        break;
    case 0x8:
        s.xy_mag_filter.Assign(AmdGpu::Filter::AnisoLinear);
        s.xy_min_filter.Assign(AmdGpu::Filter::AnisoLinear);
        s.z_filter.Assign(2);
        s.mip_filter.Assign(AmdGpu::MipFilter::Linear);
        break;
    default:
        return undecided("filter");
    }
    // address modes: the engine uses the console's numbering (0 wrap, 2 clamp to the last texel met so far)
    for (u32 i = 0; i < 3; ++i) {
        const u32 mode = rd(0x34 + i * 4);
        if (mode != 0 && mode != 2) {
            return undecided("address mode");
        }
    }
    s.clamp_x.Assign(static_cast<AmdGpu::ClampMode>(rd(0x34)));
    s.clamp_y.Assign(static_cast<AmdGpu::ClampMode>(rd(0x38)));
    s.clamp_z.Assign(static_cast<AmdGpu::ClampMode>(rd(0x3c)));
    const u32 aniso = rd(0x44);
    if (aniso != 1 && aniso != 2 && aniso != 4 && aniso != 8 && aniso != 16) {
        return undecided("anisotropy");
    }
    s.max_aniso.Assign(static_cast<AmdGpu::AnisoRatio>(std::countr_zero(aniso)));
    const u32 compare = rd(0x48);
    if (compare < 1 || compare > 8) {
        return undecided("comparison");
    }
    s.depth_compare_func.Assign(static_cast<AmdGpu::DepthCompare>(compare - 1));
    // the border color is white in every sampler met so far (the console's type 0, no border pointer)
    for (u32 i = 0; i < 4; ++i) {
        if (fl(0x4c + i * 4) != 1.0f) {
            return undecided("border color");
        }
    }
    if (fl(0x40) != 0.0f) {
        return undecided("level bias");
    }
    const auto lod = [](float v) -> u64 { return v <= 0.0f ? 0 : v >= 15.99f ? 0xfff : static_cast<u64>(v * 256.0f); };
    s.min_lod.Assign(lod(fl(0x5c)));
    s.max_lod.Assign(lod(fl(0x60)));
    std::memcpy(out, &s, sizeof(s));
    g_sampler_derived.fetch_add(1, std::memory_order_relaxed);
    if (record_words != nullptr && std::memcmp(record_words, out, 4 * sizeof(u32)) != 0) {
        if (g_sampler_differing.fetch_add(1, std::memory_order_relaxed) < 8) {
            Core::RareEvents::Note("sampler_desc_differs_from_record", fmt::format("derived {:#x} {:#x} {:#x} {:#x} record {:#x} {:#x} {:#x} {:#x} (filter {:#x} aniso {} compare {})", out[0], out[1], out[2], out[3],
                                                                                   record_words[0], record_words[1], record_words[2], record_words[3], rd(0x30), aniso, compare));
        }
    }
    return true;
}

nlohmann::json Report() {
    nlohmann::json j;
    j["view_holds_the_record_descriptor"] = {{"equal", g_held_equal.load()}, {"differ", g_held_differ.load()}, {"binding_has_no_view", g_held_none.load()}};
    j["view_creation_snapshot_vs_record"] = {{"equal", g_snap_equal.load()}, {"differ", g_snap_differ.load()}};
    for (u32 w = 0; w < 8; ++w) {
        j["view_creation_snapshot_vs_record"]["dword_differs"].push_back(g_snap_word_differs[w].load());
    }
    for (u32 w = 0; w < 8; ++w) {
        j["view_holds_the_record_descriptor"]["dword_differs"].push_back(g_held_word_differs[w].load());
    }
    j["samplers"] = {{"derived", g_sampler_derived.load()}, {"value_not_in_the_table", g_sampler_unknown.load()}, {"derived_differs_from_record_words", g_sampler_differing.load()}};
    j["sampler_words_from_the_engine_object"] = Decides();
    j["compared"] = g_compared.load();
    j["bindings_without_a_view"] = g_no_view.load();
    j["view_id_not_in_the_table"] = g_view_unknown.load();
    j["view_with_description"] = g_view_with_desc.load();
    j["view_without_description"] = g_view_without_desc.load();
    j["equal_in_every_field"] = g_equal.load();
    j["object_not_in_the_table"] = g_unknown_object.load();
    j["format_not_in_the_table"] = g_unknown_format.load();
    {
        {
            // F4.6: the draw code's own bits-per-element table against the console header's (-1 entries count as 0), every code of the field
            u32 differing = 0;
            for (u32 code = 0; code < 42; ++code) {
                const s32 reference = static_cast<s32>(Ps4Layout::NumBitsPerElement(static_cast<Ps4Layout::DataFormat>(code)));
                if (NativeDesc::BitsPerElementOfDataFormat(code) != static_cast<u32>(std::max(reference, 0))) {
                    ++differing;
                }
            }
            j["f4_6_bits_per_element_table_differs"] = differing;
        }
        j["f4_3_use"] = {{"on", FromDescOn()}, {"switch", "TLG_IMAGE_FROM_DESC=0 turns it off"}, {"bindings_served_from_the_description", g_use_served.load()}, {"engine_description_not_available", g_use_unavailable.load()},
                         {"alt_tile_set", g_use_alt_tile.load()}, {"depth_use_does_not_fit", g_use_depth_refused.load()}};
    }
    auto& fields = j["differing_fields"] = nlohmann::json::object();
    for (u32 i = 0; i < FieldCount; ++i) {
        fields[FieldNames[i]] = g_differing[i].load();
    }
    std::scoped_lock lock{g_mutex};
    for (const auto& [key, values] : g_sampler_map) {
        std::string text;
        for (const auto& v : values) {
            text += v + " ; ";
        }
        j["sampler_map"][key] = text;
    }
    for (const auto& [key, pairs] : g_pitches) {
        std::string text;
        for (const auto& [w, p] : pairs) {
            text += fmt::format("{}:{} ", w, p);
        }
        j["pitch_by_width"][key] = text;
    }
    auto& examples = j["examples"] = nlohmann::json::object();
    for (u32 i = 0; i < FieldCount; ++i) {
        for (u32 k = 0; k < g_example_count[i]; ++k) {
            examples[FieldNames[i]].push_back(g_examples[i][k]);
        }
    }
    return j;
}

} // namespace Core::NativeImageDesc
