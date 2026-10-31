// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_target_desc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "native/native_debug_server.h"
#include "native/native_device.h"
#include "native/native_image_desc.h"
#include "video_core/amdgpu/draw_state.h"

namespace Core::NativeTargetDesc {

namespace {

static_assert(sizeof(AmdGpu::ColorBuffer) == 15 * sizeof(u32));
static_assert(sizeof(AmdGpu::DepthBuffer) == 8 * sizeof(u32));
static_assert(sizeof(AmdGpu::DepthView) == sizeof(u32));
static_assert(sizeof(AmdGpu::Image) == 8 * sizeof(u32));

// Offsets in the engine's device context (the ones native_draw.cpp uses: Ctx::RenderTargetCount, Ctx::RenderTargets, Ctx::DepthStencilView, and the object of target 7)
constexpr u64 CtxRenderTargetCount = 0x4484;
constexpr u64 CtxRenderTargets = 0x4488;
constexpr u64 CtxDepthStencilView = 0x44d0;
constexpr u64 CtxRenderTarget7 = 0x130e0;
constexpr u32 NumSlots = 9; // eight render targets and the depth target

template <typename T>
T Rd(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

enum Field : u32 {
    FFormat,
    FNumberFormat,
    FTileMode,
    FPitch,
    FExtentWidth,
    FExtentHeight,
    FFirstSlice,
    FNumSlices,
    FSamples,
    FAddress,
    FCmaskAddress,
    FClearWord0,
    FClearWord1,
    FZFormat,
    FStencilFormat,
    FTFormat,
    FTPitch,
    FTTileMode,
    FBlendBypass,
    FFastClear,
    FWriteBase,
    FStencilAddress,
    FHtileBase,
    FHtileEnable,
    FieldCount,
};

// The first fifteen compare the words with the Direct3D 11 description and `NativeImageDesc::Derive`; the others (t_*, blend_bypass, ...) with what the Ghidra reading of the engine's
// own builders says (FUN_02491a30 builds the colour info word from the resource's T# at resource +0x70; FUN_02492370 the depth block from the resource): see E7-notes.md section 6.
constexpr std::array<const char*, FieldCount> FieldNames{
    "data_format", "number_format", "tile_mode", "pitch", "extent_width", "extent_height", "first_slice", "num_slices",
    "samples", "address", "cmask_address", "clear_word0", "clear_word1", "z_format", "stencil_format",
    "t_data_format", "t_pitch", "t_tile_mode", "blend_bypass", "fast_clear_flag", "z_write_base", "stencil_address", "htile_base", "htile_enable",
};

std::array<std::atomic<u64>, FieldCount> g_field_diffs[2]{}; // [0] colour, [1] depth
std::atomic<u64> g_observed[2]{}, g_cache_hits{}, g_evaluated[2]{}, g_differing[2]{}, g_unbound[2]{}, g_no_object[2]{};
std::atomic<u64> g_derive[3]{}; // NativeImageDesc::Result: derived, unknown object, unknown format
std::atomic<u64> g_extent_none[2]{}, g_extent_equal[2]{}, g_extent_larger[2]{}, g_extent_smaller[2]{};
std::atomic<u64> g_clear_packed_present{}, g_clear_block_equals_packed{}, g_clear_block_differs_packed{};
std::atomic<u64> g_unknown_depth_format{};
std::atomic<u64> g_tuple_overflow{}, g_shape_overflow{};

/// What a target looked like, as seen by a draw (the tuples the local side asked for)
struct Key {
    u32 is_depth;
    u32 dxgi_view;     ///< the format of the view's description (+0x40)
    u32 dxgi_resource; ///< the format of the resource's description
    u32 samples;       ///< the sample count of the resource's description (+0x194)
    u32 raw_a;         ///< colour: CB_COLORi_INFO; depth: DB_Z_INFO
    u32 raw_b;         ///< colour: CB_COLORi_ATTRIB; depth: DB_STENCIL_INFO
    u32 t_desc;        ///< the resource's own T# (+0x70): data format | number format << 8 | dst_sel (x, y, z, w; 3 bits each) << 16
    u32 mip;
    u32 dimension;     ///< the view's dimension (+0x44)
    u32 aux;           ///< colour: bit 0 cmask words set, 1 fmask words set, 2 resource +0x58 set; depth: bit 0 write base = read base, 1 htile base set, 2 resource +0x48 set, 3.. DSV flags
    auto operator<=>(const Key&) const = default;
};

struct Counter {
    std::atomic<u64> n{0};
};

std::mutex g_mutex;
std::map<Key, Counter> g_tuples;
std::map<std::array<u32, 8>, Counter> g_shapes; ///< {depth, hint width, hint height, pitch, height from the words, resource width, resource height, mip}
std::vector<std::string> g_examples;

Counter* TupleCounter(const Key& key) {
    std::scoped_lock lock{g_mutex};
    if (const auto it = g_tuples.find(key); it != g_tuples.end()) {
        return &it->second;
    }
    if (g_tuples.size() >= 512) {
        g_tuple_overflow.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    return &g_tuples.try_emplace(key).first->second;
}

void NoteShape(const std::array<u32, 8>& key) {
    std::scoped_lock lock{g_mutex};
    auto it = g_shapes.find(key);
    if (it == g_shapes.end()) {
        if (g_shapes.size() >= 256) {
            g_shape_overflow.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        it = g_shapes.try_emplace(key).first;
    }
    it->second.n.fetch_add(1, std::memory_order_relaxed);
}

void NoteExample(const std::string& text) {
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) >= 16) {
        return;
    }
    LOG_WARNING(Core, "E7 target description differs from the console words: {}", text);
    Core::RareEvents::Note("target_desc_differs", text);
    std::scoped_lock lock{g_mutex};
    g_examples.push_back(text);
}

struct Diffs {
    explicit Diffs(u32 depth) : is_depth{depth} {}
    u32 is_depth;
    u32 count{0};
    std::string text;
    /// `object`: what the engine's objects say; `words`: what the console words decode to
    void Check(Field field, u64 object, u64 words) {
        if (object == words) {
            return;
        }
        ++count;
        g_field_diffs[is_depth][field].fetch_add(1, std::memory_order_relaxed);
        text += fmt::format("{}(object {:#x}, words {:#x}) ", FieldNames[field], object, words);
    }
};

struct Range {
    u32 mip{0}, first{0}, count{1};
    bool known{true};
};

/// The mip slice and the slice range of the view's description (D3D11 layout: RTV dimension +0x44, union from +0x48; DSV has Flags at +0x48 and the union from +0x4c)
Range ReadRange(u64 view, bool dsv, const NativeDevice::Object& res) {
    Range r;
    const u32 dim = Rd<u32>(view, 0x44);
    const u64 u = dsv ? 0x4c : 0x48;
    if (dsv) {
        switch (dim) {
        case 1: // 1D
        case 3: // 2D
            r.mip = Rd<u32>(view, u);
            break;
        case 2: // 1D array
        case 4: // 2D array
            r.mip = Rd<u32>(view, u);
            r.first = Rd<u32>(view, u + 4);
            r.count = Rd<u32>(view, u + 8);
            break;
        case 5: // 2D MS
            break;
        case 6: // 2D MS array
            r.first = Rd<u32>(view, u);
            r.count = Rd<u32>(view, u + 4);
            break;
        default:
            r.known = false;
        }
    } else {
        switch (dim) {
        case 2: // 1D
        case 4: // 2D
            r.mip = Rd<u32>(view, u);
            break;
        case 3: // 1D array
        case 5: // 2D array
        case 8: // 3D (first and count are W slices)
            r.mip = Rd<u32>(view, u);
            r.first = Rd<u32>(view, u + 4);
            r.count = Rd<u32>(view, u + 8);
            break;
        case 6: // 2D MS
            break;
        case 7: // 2D MS array
            r.first = Rd<u32>(view, u);
            r.count = Rd<u32>(view, u + 4);
            break;
        default:
            r.known = false;
        }
    }
    r.mip = std::min<u32>(r.mip, 31);
    if (r.count == 0xffffffffu) {
        const u32 limit = std::max<u32>(res.kind == NativeDevice::Kind::Texture3D ? (res.depth >> r.mip) : res.layers, 1);
        r.count = limit > r.first ? limit - r.first : 1;
    }
    return r;
}

/// The depth formats the console has (Z16, Z32 float, Stencil8) for a Direct3D format; false for a format the table does not know yet (it is counted and shows in the tuples)
bool ExpectedDepth(u32 dxgi, u32& z_format, bool& stencil) {
    switch (dxgi) {
    case 40: // D32_FLOAT
        z_format = static_cast<u32>(AmdGpu::DepthBuffer::ZFormat::Z32Float);
        stencil = false;
        return true;
    case 20: // D32_FLOAT_S8X24_UINT
        z_format = static_cast<u32>(AmdGpu::DepthBuffer::ZFormat::Z32Float);
        stencil = true;
        return true;
    case 55: // D16_UNORM
        z_format = static_cast<u32>(AmdGpu::DepthBuffer::ZFormat::Z16);
        stencil = false;
        return true;
    default:
        return false;
    }
}

struct Hasher {
    u64 h{0xcbf29ce484222325ull};
    void Add(u64 value) {
        h = (h ^ value) * 0x100000001b3ull;
    }
};

/// What decides whether the last evaluation of a target still holds
u64 TargetHash(u64 view, bool depth) {
    Hasher hasher;
    hasher.Add(view);
    const u64 resource = Rd<u64>(view, 0x30);
    hasher.Add(resource);
    const u32 desc_words = depth ? 6 : 5;
    for (u32 k = 0; k < desc_words; ++k) {
        hasher.Add(Rd<u32>(view, 0x40 + k * 4));
    }
    const u64 block = view + (depth ? 0x58 : 0x54);
    const u32 block_words = depth ? 13 : 16;
    for (u32 k = 0; k < block_words; ++k) {
        hasher.Add(Rd<u32>(block, k * 4));
    }
    if (resource != 0) {
        hasher.Add(Rd<u64>(resource, 0x30));
        hasher.Add(Rd<u64>(resource, 0x40));
        hasher.Add(Rd<u64>(resource, 0x48));
        hasher.Add(Rd<u64>(resource, 0x58));
        for (u32 k = 0; k < 4; ++k) {
            hasher.Add(Rd<u64>(resource, 0x70 + k * 8)); // the resource's own T#
        }
        hasher.Add(Rd<u32>(resource, 0xdc));
        hasher.Add(Rd<u32>(resource, 0xe0));
        hasher.Add(Rd<u32>(resource, 0x14c));
        hasher.Add(Rd<u32>(resource, 0x150));
        hasher.Add(Rd<u32>(resource, 0x154));
        hasher.Add(Rd<u32>(resource, 0x194));
    }
    return hasher.h | 1;
}

void NoteExtent(u32 depth, u32 hint_width, u32 hint_height, u32 width, u32 height) {
    if (hint_width == 0 && hint_height == 0) {
        g_extent_none[depth].fetch_add(1, std::memory_order_relaxed);
    } else if (hint_width == width && hint_height == height) {
        g_extent_equal[depth].fetch_add(1, std::memory_order_relaxed);
    } else if (hint_width >= width && hint_height >= height) {
        g_extent_larger[depth].fetch_add(1, std::memory_order_relaxed);
    } else {
        g_extent_smaller[depth].fetch_add(1, std::memory_order_relaxed);
    }
}

/// One target, evaluated: returns the counter of its tuple (null: nothing to count)
Counter* Evaluate(u64 view, bool depth, u32 slot) {
    const u32 kind = depth ? 1u : 0u;
    const u64 block = view + (depth ? 0x58 : 0x54);
    // an unbound target: the same rules `Synthesize` uses (colour: base address word 0; depth: DB_DEPTH_VIEW and the extent both 0)
    if (depth ? (Rd<u32>(block, 8 * 4) == 0 && Rd<u32>(block, 12 * 4) == 0) : Rd<u32>(block, 0) == 0) {
        g_unbound[kind].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const u64 resource = Rd<u64>(view, 0x30);
    const u64 resource_id = resource != 0 ? Rd<u64>(resource, 0x10) : 0;
    const u64 view_id = Rd<u64>(view, 0x10);
    NativeDevice::Object res;
    if (resource_id == 0 || !NativeDevice::Lookup(resource_id, res)) {
        g_no_object[kind].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const u32 dxgi_view = Rd<u32>(view, 0x40);
    const u32 dimension = Rd<u32>(view, 0x44);
    const Range range = ReadRange(view, depth, res);
    u32 samples = res.kind == NativeDevice::Kind::Texture2D ? Rd<u32>(resource, 0x194) : 1;
    samples = std::max<u32>(samples, 1);
    const u32 extent_word = Rd<u32>(block, (depth ? 12 : 15) * 4);
    const u32 hint_width = extent_word & 0xffff;
    const u32 hint_height = extent_word >> 16;
    const u32 mip_width = std::max<u32>(res.width >> range.mip, 1);
    const u32 mip_height = std::max<u32>(res.height >> range.mip, 1);

    AmdGpu::Image image{};
    const auto result = NativeImageDesc::Derive(resource_id, view_id, image);
    g_derive[static_cast<u32>(result)].fetch_add(1, std::memory_order_relaxed);
    const bool derived = result == NativeImageDesc::Result::Derived;

    // the resource's own console texture description (T#, +0x70): the colour info word and the depth block are built from it (Ghidra: FUN_02491a30, FUN_02492370)
    AmdGpu::Image resource_t{};
    std::memcpy(&resource_t, reinterpret_cast<const void*>(resource + 0x70), sizeof(resource_t));

    Diffs diffs{kind};
    Key key{};
    key.is_depth = kind;
    key.dxgi_view = dxgi_view;
    key.dxgi_resource = res.format;
    key.samples = samples;
    key.mip = range.mip;
    key.dimension = dimension;
    key.t_desc = static_cast<u32>(resource_t.data_format) | static_cast<u32>(resource_t.num_format) << 8 | static_cast<u32>(resource_t.dst_sel_x) << 16 |
                 static_cast<u32>(resource_t.dst_sel_y) << 19 | static_cast<u32>(resource_t.dst_sel_z) << 22 | static_cast<u32>(resource_t.dst_sel_w) << 25;
    std::array<u32, 8> shape{};
    shape[0] = kind;
    shape[1] = hint_width;
    shape[2] = hint_height;
    shape[5] = res.width;
    shape[6] = res.height;
    shape[7] = range.mip;

    if (!depth) {
        AmdGpu::ColorBuffer cb{};
        std::memcpy(&cb, reinterpret_cast<const void*>(block), 14 * sizeof(u32));
        key.raw_a = Rd<u32>(block, 4 * 4);
        key.raw_b = Rd<u32>(block, 5 * 4);
        const u64 resource_cmask = Rd<u64>(resource, 0x58);
        key.aux = (cb.cmask_base_address != 0 ? 1u : 0u) | (cb.fmask_base_address != 0 ? 2u : 0u) | (resource_cmask != 0 ? 4u : 0u);
        shape[3] = cb.Pitch();
        shape[4] = cb.Height();
        if (derived) {
            diffs.Check(FFormat, image.data_format, static_cast<u64>(cb.GetDataFmt()));
            diffs.Check(FNumberFormat, image.num_format, static_cast<u64>(cb.GetNumberFmt()));
            diffs.Check(FTileMode, image.tiling_index, static_cast<u64>(cb.GetTileMode()));
            if (range.mip == 0) {
                diffs.Check(FPitch, static_cast<u64>(image.pitch) + 1, cb.Pitch());
            }
        }
        // Ghidra Q6: info.format is the T# data format (FUN_0246a420: format = data format << 2); the number type and comp_swap go through small tables, which the tuples dump
        diffs.Check(FTFormat, resource_t.data_format, cb.info.format);
        if (range.mip == 0) {
            diffs.Check(FTPitch, static_cast<u64>(resource_t.pitch) + 1, cb.Pitch());
            diffs.Check(FTTileMode, resource_t.tiling_index, static_cast<u64>(cb.GetTileMode()));
        }
        // blend_bypass is set for the integer number types (4 uint, 5 sint)
        diffs.Check(FBlendBypass, (cb.info.number_type == 4 || cb.info.number_type == 5) ? 1 : 0, cb.info.blend_bypass);
        if (extent_word != 0) {
            diffs.Check(FExtentWidth, mip_width, hint_width);
            diffs.Check(FExtentHeight, mip_height, hint_height);
        }
        NoteExtent(kind, hint_width, hint_height, mip_width, mip_height);
        if (range.known) {
            diffs.Check(FFirstSlice, range.first, cb.BaseSlice());
            diffs.Check(FNumSlices, range.count, cb.NumSlices());
        }
        diffs.Check(FSamples, samples, cb.NumSamples());
        if (range.mip == 0) {
            diffs.Check(FAddress, res.address, cb.Address());
        }
        // Ghidra Q9: the cmask base is resource +0x58 (>> 8) and the fast clear bit is set if and only if it is not 0
        diffs.Check(FFastClear, resource_cmask != 0 ? 1 : 0, cb.info.fast_clear);
        if (range.mip == 0) {
            diffs.Check(FCmaskAddress, resource_cmask, cb.CmaskAddress());
        }
        // Ghidra Q7: block words 11 and 12 are the prebuilt block's (resource +0xdc, +0xe0) for mip 0 and 0 above it; they are NOT the colour of the engine's clear call, which lives at
        // resource +0x150 / +0x154 (valid when +0x14c is not 0). The second pair of numbers below counts how often the two agree.
        diffs.Check(FClearWord0, range.mip == 0 ? Rd<u32>(resource, 0xdc) : 0, cb.clear_word0);
        diffs.Check(FClearWord1, range.mip == 0 ? Rd<u32>(resource, 0xe0) : 0, cb.clear_word1);
        if (Rd<u32>(resource, 0x14c) != 0) {
            g_clear_packed_present.fetch_add(1, std::memory_order_relaxed);
            if (Rd<u32>(resource, 0x150) == cb.clear_word0 && Rd<u32>(resource, 0x154) == cb.clear_word1) {
                g_clear_block_equals_packed.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_clear_block_differs_packed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        AmdGpu::DepthBuffer db{};
        std::memcpy(&db, reinterpret_cast<const void*>(block), sizeof(db));
        AmdGpu::DepthView depth_view{};
        std::memcpy(&depth_view, reinterpret_cast<const void*>(block + 8 * 4), sizeof(depth_view));
        key.raw_a = Rd<u32>(block, 0);
        key.raw_b = Rd<u32>(block, 1 * 4);
        const u32 flags = Rd<u32>(view, 0x48);
        key.aux = (db.z_read_base == db.z_write_base ? 1u : 0u) | (Rd<u32>(block, 9 * 4) != 0 ? 2u : 0u) | (Rd<u64>(resource, 0x48) != 0 ? 4u : 0u) | ((flags & 3u) << 3);
        shape[3] = db.Pitch();
        shape[4] = db.Height();
        u32 z_format = 0;
        bool stencil = false;
        if (ExpectedDepth(dxgi_view, z_format, stencil)) {
            diffs.Check(FZFormat, z_format, static_cast<u64>(db.z_info.format));
            diffs.Check(FStencilFormat, stencil ? 1 : 0, db.stencil_info.format != AmdGpu::DepthBuffer::StencilFormat::Invalid ? 1 : 0);
        } else {
            g_unknown_depth_format.fetch_add(1, std::memory_order_relaxed);
        }
        if (derived) {
            diffs.Check(FTileMode, image.tiling_index, static_cast<u64>(db.GetTileMode()));
            if (range.mip == 0) {
                diffs.Check(FPitch, static_cast<u64>(image.pitch) + 1, db.Pitch());
            }
        }
        if (range.mip == 0) {
            diffs.Check(FTPitch, static_cast<u64>(resource_t.pitch) + 1, db.Pitch());
            diffs.Check(FTTileMode, resource_t.tiling_index, static_cast<u64>(db.GetTileMode()));
        }
        // Ghidra Q8: the read and the write base are both resource +0x30 (read-only is a view flag); the stencil bases are resource +0x40; the htile base is resource +0x48 (>> 8, block
        // word 9) and tile_surface_enable is set if and only if it is not 0
        diffs.Check(FWriteBase, db.DepthAddress(), db.DepthWriteAddress());
        if (db.stencil_info.format != AmdGpu::DepthBuffer::StencilFormat::Invalid) {
            diffs.Check(FStencilAddress, Rd<u64>(resource, 0x40), db.StencilAddress());
        }
        diffs.Check(FHtileBase, Rd<u64>(resource, 0x48) >> 8, Rd<u32>(block, 9 * 4));
        diffs.Check(FHtileEnable, Rd<u64>(resource, 0x48) != 0 ? 1 : 0, db.z_info.tile_surface_enable);
        if (extent_word != 0) {
            diffs.Check(FExtentWidth, mip_width, hint_width);
            diffs.Check(FExtentHeight, mip_height, hint_height);
        }
        NoteExtent(kind, hint_width, hint_height, mip_width, mip_height);
        if (range.known) {
            diffs.Check(FFirstSlice, range.first, depth_view.slice_start);
            diffs.Check(FNumSlices, range.count, depth_view.NumSlices());
        }
        diffs.Check(FSamples, samples, db.NumSamples());
        if (range.mip == 0) {
            diffs.Check(FAddress, res.address, db.DepthAddress());
        }
    }

    NoteShape(shape);
    g_evaluated[kind].fetch_add(1, std::memory_order_relaxed);
    if (diffs.count != 0) {
        g_differing[kind].fetch_add(1, std::memory_order_relaxed);
        NoteExample(fmt::format("{} slot {} view id {} resource id {} dxgi view {} resource {} mip {} dimension {}: {}", depth ? "depth" : "colour", slot, view_id, resource_id, dxgi_view,
                                res.format, range.mip, dimension, diffs.text));
    }
    return TupleCounter(key);
}

struct SlotCache {
    u64 hash{0};
    Counter* entry{nullptr};
};

} // namespace

bool Enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_TARGET_DESC");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void Observe(u64 ctx) {
    if (!Enabled()) {
        return;
    }
    static thread_local std::array<SlotCache, NumSlots> cache{};
    const u32 count = Rd<u32>(ctx, CtxRenderTargetCount);
    for (u32 slot = 0; slot < NumSlots; ++slot) {
        const bool depth = slot == 8;
        u64 view = 0;
        if (depth) {
            view = Rd<u64>(ctx, CtxDepthStencilView);
        } else if (slot == 7) {
            view = Rd<u64>(ctx, CtxRenderTarget7);
        } else if (slot < count) {
            view = Rd<u64>(ctx, CtxRenderTargets + slot * 8);
        }
        SlotCache& slot_cache = cache[slot];
        if (view == 0) {
            slot_cache = {};
            continue;
        }
        g_observed[depth ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
        const u64 hash = TargetHash(view, depth);
        if (slot_cache.hash == hash) {
            g_cache_hits.fetch_add(1, std::memory_order_relaxed);
            if (slot_cache.entry != nullptr) {
                slot_cache.entry->n.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        slot_cache.entry = Evaluate(view, depth, slot);
        slot_cache.hash = hash;
        if (slot_cache.entry != nullptr) {
            slot_cache.entry->n.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["enabled"] = Enabled();
    j["note"] = "E7 Task 2 shadow (TLG_TARGET_DESC=1): the targets' console words against the engine's objects. 'observed' counts every target of every draw record, 'evaluated' the ones "
                "whose view or block changed since the thread last saw the slot; differences are counted per evaluation, tuples per observation. 'object' is what the resource / view "
                "objects say, 'words' what the console words decode to.";
    const char* names[2] = {"colour", "depth"};
    for (u32 k = 0; k < 2; ++k) {
        auto& section = j[names[k]];
        section["observed"] = g_observed[k].load();
        section["evaluated"] = g_evaluated[k].load();
        section["differing"] = g_differing[k].load();
        section["unbound_block"] = g_unbound[k].load();
        section["no_engine_object"] = g_no_object[k].load();
        auto& fields = section["differing_fields"] = nlohmann::json::object();
        for (u32 f = 0; f < FieldCount; ++f) {
            if (const u64 n = g_field_diffs[k][f].load(); n != 0) {
                fields[FieldNames[f]] = n;
            }
        }
        section["extent_hint"] = {{"none", g_extent_none[k].load()}, {"equal_to_mip_extent", g_extent_equal[k].load()},
                                  {"larger_than_mip_extent", g_extent_larger[k].load()}, {"smaller_than_mip_extent", g_extent_smaller[k].load()}};
    }
    j["cache_hits"] = g_cache_hits.load();
    j["derive"] = {{"derived", g_derive[0].load()}, {"unknown_object", g_derive[1].load()}, {"unknown_format", g_derive[2].load()}};
    j["colour"]["clear_colour_in_resource"] = {{"resource_14c_set", g_clear_packed_present.load()}, {"block_words_equal_resource_150_154", g_clear_block_equals_packed.load()},
                                               {"block_words_differ_from_resource_150_154", g_clear_block_differs_packed.load()}};
    j["depth"]["format_not_in_the_table"] = g_unknown_depth_format.load();
    j["tuple_overflow"] = g_tuple_overflow.load();
    j["shape_overflow"] = g_shape_overflow.load();

    std::vector<std::pair<Key, u64>> tuples;
    std::vector<std::pair<std::array<u32, 8>, u64>> shapes;
    {
        std::scoped_lock lock{g_mutex};
        for (const auto& [key, counter] : g_tuples) {
            tuples.emplace_back(key, counter.n.load());
        }
        for (const auto& [key, counter] : g_shapes) {
            shapes.emplace_back(key, counter.n.load());
        }
        j["first_differences"] = g_examples;
    }
    std::sort(tuples.begin(), tuples.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    auto& tuple_list = j["tuples"] = nlohmann::json::array();
    for (const auto& [key, n] : tuples) {
        tuple_list.push_back({{"kind", key.is_depth != 0 ? "depth" : "colour"}, {"dxgi_view", key.dxgi_view}, {"dxgi_resource", key.dxgi_resource}, {"samples", key.samples},
                              {key.is_depth != 0 ? "z_info" : "info", fmt::format("{:#x}", key.raw_a)},
                              {key.is_depth != 0 ? "stencil_info" : "attrib", fmt::format("{:#x}", key.raw_b)},
                              {"t_data_format", key.t_desc & 0xff}, {"t_number_format", (key.t_desc >> 8) & 0xff}, {"t_dst_sel", fmt::format("{:#x}", key.t_desc >> 16)},
                              {"mip", key.mip}, {"dimension", key.dimension},
                              {"aux", fmt::format("{:#x}", key.aux)}, {"count", n}});
    }
    auto& shape_list = j["extent_shapes"] = nlohmann::json::array();
    for (const auto& [key, n] : shapes) {
        shape_list.push_back({{"kind", key[0] != 0 ? "depth" : "colour"}, {"hint_width", key[1]}, {"hint_height", key[2]}, {"pitch", key[3]}, {"height_from_words", key[4]},
                              {"resource_width", key[5]}, {"resource_height", key[6]}, {"mip", key[7]}, {"count", n}});
    }
    return j;
}

} // namespace Core::NativeTargetDesc
