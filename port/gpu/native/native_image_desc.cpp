// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_image_desc.h"

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
#include "native/native_device.h"

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
std::atomic<u64> g_decided{}, g_decide_unknown{}, g_decide_no_id{}, g_decide_differing{};
std::atomic<u64> g_held_equal{}, g_held_differ{}, g_held_none{};
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

Result Derive(u64 resource_id, u64 view_id, AmdGpu::Image& out) {
    using NativeDevice::Kind;
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
    // fields) and keeps it in the view object. The Direct3D description has two numbers it cannot say: a texture the engine allocated with padded rows (1088 for 1080) and a
    // level count it limits (7 of 8 for a block compressed cube). Where the view's copy still names this texture (same memory and format) those fields come from it.
    if (view_known && view.object != 0) {
        AmdGpu::Image held;
        std::memcpy(&held, reinterpret_cast<const void*>(view.object + 0x58), 32);
        if (held.base_address == out.base_address && held.data_format == out.data_format && held.num_format == out.num_format) {
            out.width = held.width;
            out.height = held.height;
            out.last_level = held.last_level;
        }
    }
    out.depth = type == TypeColor3D ? res.depth - 1 : type == TypeCube ? layers / 6 - 1 : layers - 1;
    return Result::Derived;
}

bool Decides() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_IMAGE_DESC");
        return e == nullptr || *e != '0';
    }();
    return on;
}

bool Decide(u64 resource_id, u64 view_id, const u32* record_words, AmdGpu::Image& out) {
    if (resource_id == 0) {
        g_decide_no_id.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (Derive(resource_id, view_id, out) != Result::Derived) {
        g_decide_unknown.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_decided.fetch_add(1, std::memory_order_relaxed);
    if (record_words != nullptr && std::memcmp(record_words, &out, 8 * sizeof(u32)) != 0) {
        if (g_decide_differing.fetch_add(1, std::memory_order_relaxed) < 8) {
            u32 d[8];
            std::memcpy(d, &out, sizeof(d));
            Core::RareEvents::Note("image_desc_decided_differs_from_record",
                                   fmt::format("derived {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} record {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                                               record_words[0], record_words[1], record_words[2], record_words[3], record_words[4], record_words[5], record_words[6], record_words[7]));
        }
    }
    return true;
}

void Compare(u64 resource_id, u64 view_id, const AmdGpu::Image& actual) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_DESC_CHECKS");
        return e != nullptr && *e == '1';
    }();
    if (!on || resource_id == 0) {
        return;
    }
    {
        // the descriptor the engine keeps inside the view object (+0x58, eight dwords): is it the one the record holds?
        NativeDevice::Object v;
        if (view_id != 0 && NativeDevice::Lookup(view_id, v) && v.object != 0) {
            u32 held[8], now[8];
            std::memcpy(held, reinterpret_cast<const void*>(v.object + 0x58), sizeof(held));
            std::memcpy(now, &actual, sizeof(now));
            bool same = true;
            for (u32 w = 0; w < 8; ++w) {
                if (held[w] != now[w]) {
                    same = false;
                    g_held_word_differs[w].fetch_add(1, std::memory_order_relaxed);
                }
            }
            (same ? g_held_equal : g_held_differ).fetch_add(1, std::memory_order_relaxed);
        } else {
            g_held_none.fetch_add(1, std::memory_order_relaxed);
        }
    }
    AmdGpu::Image derived;
    switch (Derive(resource_id, view_id, derived)) {
    case Result::UnknownObject:
        g_unknown_object.fetch_add(1, std::memory_order_relaxed);
        return;
    case Result::UnknownFormat: {
        g_unknown_format.fetch_add(1, std::memory_order_relaxed);
        NativeDevice::Object res, view;
        if (NativeDevice::Lookup(resource_id, res) && g_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
            const bool has_view = view_id != 0 && NativeDevice::Lookup(view_id, view) && view.has_desc;
            Core::RareEvents::Note("image_desc_unknown_format", fmt::format("engine format {} (resource {}) -> T# {}/{}", has_view ? view.desc[0] : res.format, res.format, u32(actual.data_format), u32(actual.num_format)));
        }
        return;
    }
    case Result::Derived:
        break;
    }
    g_compared.fetch_add(1, std::memory_order_relaxed);
    {
        NativeDevice::Object r;
        if (NativeDevice::Lookup(resource_id, r)) {
            std::scoped_lock lock{g_mutex};
            auto& mem = g_pitches[fmt::format("memtype kind {} bind {:#x} usage {} cpu {:#x} misc {:#x}", u32(r.kind), r.bind, r.usage, r.cpu_access, r.misc)];
            if (mem.size() < 8) {
                mem.insert({u32(actual.mtype) * 100 + u32(actual.mtype_l2) * 10 + u32(actual.mtype2), 0});
            }
            auto& set = g_pitches[fmt::format("format {} tile {}", r.format, u32(actual.tiling_index))];
            if (set.size() < 40) {
                set.insert({r.width, u32(actual.pitch) + 1});
            }
        }
    }
    bool differ[FieldCount] = {};
    differ[FAddress] = derived.base_address != actual.base_address;
    differ[FDataFormat] = derived.data_format != actual.data_format;
    differ[FNumFormat] = derived.num_format != actual.num_format;
    differ[FWidth] = derived.width != actual.width;
    differ[FHeight] = derived.height != actual.height;
    differ[FDepth] = derived.depth != actual.depth;
    differ[FPitch] = derived.pitch != actual.pitch;
    differ[FDstSel] = derived.dst_sel_x != actual.dst_sel_x || derived.dst_sel_y != actual.dst_sel_y || derived.dst_sel_z != actual.dst_sel_z || derived.dst_sel_w != actual.dst_sel_w;
    differ[FBaseLevel] = derived.base_level != actual.base_level;
    differ[FLastLevel] = derived.last_level != actual.last_level;
    differ[FTiling] = derived.tiling_index != actual.tiling_index;
    differ[FPow2Pad] = derived.pow2pad != actual.pow2pad;
    differ[FType] = derived.type != actual.type;
    differ[FBaseArray] = derived.base_array != actual.base_array;
    differ[FLastArray] = derived.last_array != actual.last_array;
    differ[FMemType] = derived.mtype != actual.mtype || derived.mtype_l2 != actual.mtype_l2 || derived.mtype2 != actual.mtype2 || derived.perf_modulation != actual.perf_modulation;
    // everything the derivation leaves zero: the remaining bits of the descriptor
    AmdGpu::Image rest = actual;
    rest.base_address = 0, rest.data_format = 0, rest.num_format = 0, rest.width = 0, rest.height = 0, rest.depth = 0, rest.pitch = 0;
    rest.dst_sel_x = rest.dst_sel_y = rest.dst_sel_z = rest.dst_sel_w = 0;
    rest.perf_modulation = 0, rest.mtype = 0, rest.mtype_l2 = 0, rest.mtype2 = 0;
    rest.base_level = rest.last_level = 0, rest.tiling_index = 0, rest.pow2pad = 0, rest.type = 0, rest.base_array = rest.last_array = 0;
    AmdGpu::Image zero{};
    differ[FRest] = std::memcmp(&rest, &zero, sizeof(rest)) != 0;
    bool any = false;
    NativeDevice::Object res_info, view_info;
    NativeDevice::Lookup(resource_id, res_info);
    const u64 view_object = view_id != 0 && NativeDevice::Lookup(view_id, view_info) ? view_info.object : 0;
    const auto value = [&](u32 field, const AmdGpu::Image& image) -> u64 {
        switch (field) {
        case FHeight: return image.height + 1;
        case FPitch: return image.pitch + 1;
        case FLastArray: return image.last_array;
        case FLastLevel: return image.last_level;
        default: return 0;
        }
    };
    for (u32 i = 0; i < FieldCount; ++i) {
        if (differ[i]) {
            any = true;
            g_differing[i].fetch_add(1, std::memory_order_relaxed);
            std::scoped_lock lock{g_mutex};
            if (g_example_count[i] < 4 + (i == FPitch ? 8 : 0)) {
                std::string text;
                if (i == FMemType) {
                    text = fmt::format("derived mtype {} l2 {} mtype2 {} actual mtype {} l2 {} mtype2 {}", u32(derived.mtype), u32(derived.mtype_l2), u32(derived.mtype2), u32(actual.mtype), u32(actual.mtype_l2), u32(actual.mtype2));
                } else if (i == FRest) {
                    u64 words[4];
                    std::memcpy(words, &rest, sizeof(words));
                    text = fmt::format("{:#x} {:#x} {:#x} {:#x}", words[0], words[1], words[2], words[3]);
                } else {
                    text = fmt::format("derived {} actual {}", value(i, derived), value(i, actual));
                }
                text += fmt::format(" (resource {}x{} mips {} layers {} depth {} bind {:#x} bytes {:#x} format {} T# type {} fmt {}/{} tile {} resource object {:#x} view object {:#x})", res_info.width, res_info.height, res_info.mips, res_info.layers, res_info.depth, res_info.bind,
                                    res_info.bytes, res_info.format, u32(actual.type), u32(actual.data_format), u32(actual.num_format), u32(actual.tiling_index), res_info.object, view_object);
                if (i == FHeight || i == FLastLevel) {
                    // where the engine keeps the number: the T# words and the values in the objects as they were at creation
                    u8 vb[0x400] = {}, rb[0x400] = {};
                    const bool have_v = view_id != 0 && NativeDevice::RawBytes(view_id, vb, sizeof(vb));
                    const bool have_r = NativeDevice::RawBytes(resource_id, rb, sizeof(rb));
                    u32 words[8];
                    std::memcpy(words, &actual, sizeof(words));
                    text += fmt::format(" raw view {} resource {}; T# {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x};", have_v, have_r, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
                    const auto find = [&](const u8* b, bool have, const char* name) {
                        if (!have) {
                            return;
                        }
                        for (u32 off = 0; off + 4 <= 0x400; off += 4) {
                            u32 v;
                            std::memcpy(&v, b + off, 4);
                            if (i == FHeight && (v == u32(actual.height) + 1 || v == u32(actual.height)) && v > 8) {
                                text += fmt::format(" {}+{:#x}={}", name, off, v);
                            }
                            for (u32 w = 0; w < 8; ++w) {
                                if (words[w] != 0 && v == words[w] && (words[w] > 0xffff)) {
                                    text += fmt::format(" {}+{:#x}=T#word{}", name, off, w);
                                }
                            }
                        }
                    };
                    find(vb, have_v, "view");
                    find(rb, have_r, "res");
                }
                bool seen = false;
                for (u32 k = 0; k < g_example_count[i]; ++k) {
                    seen |= g_examples[i][k] == text;
                }
                if (!seen) {
                    g_examples[i][g_example_count[i]++] = text;
                }
            }
        }
    }
    if (!any) {
        g_equal.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (false) {
        std::string names;
        for (u32 i = 0; i < FieldCount; ++i) {
            names += differ[i] ? std::string(FieldNames[i]) + " " : std::string();
        }
        NativeDevice::Object res;
        NativeDevice::Lookup(resource_id, res);
        Core::RareEvents::Note("image_desc_differs_from_t_sharp",
                               fmt::format("{}: derived {}x{} fmt {}/{} type {} lv {}-{} arr {}-{} depth {} pitch {} pad {} tile {} sel {}{}{}{}, T# {}x{} fmt {}/{} type {} lv {}-{} arr {}-{} depth {} pitch {} pad {} tile {} sel {}{}{}{} (resource {}x{} mips {} layers {} bind {:#x})",
                                           names, u32(derived.width) + 1, u32(derived.height) + 1, u32(derived.data_format), u32(derived.num_format), u32(derived.type), u32(derived.base_level), u32(derived.last_level),
                                           u32(derived.base_array), u32(derived.last_array), u32(derived.depth), u32(derived.pitch), u32(derived.pow2pad), u32(derived.tiling_index), u32(derived.dst_sel_x),
                                           u32(derived.dst_sel_y), u32(derived.dst_sel_z), u32(derived.dst_sel_w), u32(actual.width) + 1, u32(actual.height) + 1, u32(actual.data_format), u32(actual.num_format),
                                           u32(actual.type), u32(actual.base_level), u32(actual.last_level), u32(actual.base_array), u32(actual.last_array), u32(actual.depth), u32(actual.pitch), u32(actual.pow2pad),
                                           u32(actual.tiling_index), u32(actual.dst_sel_x), u32(actual.dst_sel_y), u32(actual.dst_sel_z), u32(actual.dst_sel_w), res.width, res.height, res.mips, res.layers, res.bind));
    }
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

void StudySampler(u64 object, const u32* words) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_DESC_CHECKS");
        return e != nullptr && *e == '1';
    }();
    if (!on || object == 0) {
        return;
    }
    const auto rd = [&](u32 off) {
        u32 v;
        std::memcpy(&v, reinterpret_cast<const void*>(object + off), 4);
        return v;
    };
    AmdGpu::Sampler s{};
    std::memcpy(&s, words, sizeof(s));
    std::scoped_lock lock{g_mutex};
    const auto note = [&](const std::string& key, const std::string& value) {
        auto& values = g_sampler_map[key];
        if (values.size() < 6) {
            values.insert(value);
        }
    };
    note(fmt::format("filter {:#x}", rd(0x30)), fmt::format("mag {} min {} z {} mip {} mode {}", u32(s.xy_mag_filter.Value()), u32(s.xy_min_filter.Value()), u32(s.z_filter.Value()), u32(s.mip_filter.Value()),
                                                                u32(s.filter_mode.Value())));
    note(fmt::format("address u {}", rd(0x34)), fmt::format("x {}", u32(s.clamp_x.Value())));
    note(fmt::format("address v {}", rd(0x38)), fmt::format("y {}", u32(s.clamp_y.Value())));
    note(fmt::format("address w {}", rd(0x3c)), fmt::format("z {}", u32(s.clamp_z.Value())));
    note(fmt::format("aniso {}", rd(0x44)), fmt::format("max_aniso {} threshold {} bias {}", u32(s.max_aniso.Value()), u32(s.aniso_threshold.Value()), u32(s.aniso_bias.Value())));
    note(fmt::format("compare {}", rd(0x48)), fmt::format("depth_compare {}", u32(s.depth_compare_func.Value())));
    note(fmt::format("border {:#x} {:#x} {:#x} {:#x}", rd(0x4c), rd(0x50), rd(0x54), rd(0x58)), fmt::format("type {} ptr {}", u32(s.border_color_type.Value()), u32(s.border_color_ptr.Value())));
    note("lod_bias raw", fmt::format("{:#x} -> {}", rd(0x40), u32(s.lod_bias.Value())));
    float min_lod, max_lod;
    const u32 min_raw = rd(0x5c), max_raw = rd(0x60);
    std::memcpy(&min_lod, &min_raw, 4);
    std::memcpy(&max_lod, &max_raw, 4);
    const auto lod = [](float v) { return std::clamp<s32>(s32(v * 256.0f), 0, 0xfff); };
    note("min_lod formula", lod(min_lod) == s32(s.min_lod.Value()) ? "clamp(v*256) equal" : fmt::format("DIFFERENT {} vs {}", lod(min_lod), u32(s.min_lod.Value())));
    note("max_lod formula", lod(max_lod) == s32(s.max_lod.Value()) ? "clamp(v*256) equal" : fmt::format("DIFFERENT {} vs {} ({:#x})", lod(max_lod), u32(s.max_lod.Value()), max_raw));
    AmdGpu::Sampler rest = s;
    rest.clamp_x.Assign(AmdGpu::ClampMode{});
    rest.clamp_y.Assign(AmdGpu::ClampMode{});
    rest.clamp_z.Assign(AmdGpu::ClampMode{});
    rest.max_aniso.Assign(AmdGpu::AnisoRatio{});
    rest.depth_compare_func.Assign(AmdGpu::DepthCompare{});
    rest.aniso_threshold.Assign(0);
    rest.aniso_bias.Assign(0);
    rest.filter_mode.Assign(AmdGpu::FilterMode{});
    rest.min_lod.Assign(0);
    rest.max_lod.Assign(0);
    rest.lod_bias.Assign(0);
    rest.xy_mag_filter.Assign(AmdGpu::Filter{});
    rest.xy_min_filter.Assign(AmdGpu::Filter{});
    rest.z_filter.Assign(0);
    rest.mip_filter.Assign(AmdGpu::MipFilter{});
    rest.border_color_ptr.Assign(0);
    rest.border_color_type.Assign(AmdGpu::BorderColor{});
    note("other bits", fmt::format("{:#x} {:#x}", rest.raw0, rest.raw1));
}

nlohmann::json Report() {
    nlohmann::json j;
    j["view_holds_the_record_descriptor"] = {{"equal", g_held_equal.load()}, {"differ", g_held_differ.load()}, {"binding_has_no_view", g_held_none.load()}};
    for (u32 w = 0; w < 8; ++w) {
        j["view_holds_the_record_descriptor"]["dword_differs"].push_back(g_held_word_differs[w].load());
    }
    j["samplers"] = {{"derived", g_sampler_derived.load()}, {"value_not_in_the_table", g_sampler_unknown.load()}, {"derived_differs_from_record_words", g_sampler_differing.load()}};
    j["decides"] = Decides();
    j["decided_by_the_engine_objects"] = g_decided.load();
    j["record_words_decided_no_engine_id"] = g_decide_no_id.load();
    j["record_words_decided_object_unknown"] = g_decide_unknown.load();
    j["decided_differs_from_record_words"] = g_decide_differing.load();
    j["compared"] = g_compared.load();
    j["bindings_without_a_view"] = g_no_view.load();
    j["view_id_not_in_the_table"] = g_view_unknown.load();
    j["view_with_description"] = g_view_with_desc.load();
    j["view_without_description"] = g_view_without_desc.load();
    j["equal_in_every_field"] = g_equal.load();
    j["object_not_in_the_table"] = g_unknown_object.load();
    j["format_not_in_the_table"] = g_unknown_format.load();
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
