// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <vector>

#include "common/logging/log.h"
#include "common/thread.h"
#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_log.h"
#include "native/native_image_desc.h"
#include "native/native_device.h"
#include "native/native_resources.h"
#include "native/native_updates.h"
#include "native/native_target_desc.h"
#include "native/native_image_caller.h"
#include "native/native_content.h"
#include "native/native_sync.h"
#include "native/native_frame_gate.h"
#include "native/native_tsharp_watch.h"

namespace Core::NativeDevice {

namespace {

using GuestFunction6 = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64);

template <typename T>
T Rd(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

std::shared_mutex table_mutex;
std::unordered_map<u64, Object> by_id;
std::unordered_map<u64, u64> id_by_object;
/// live resources by their memory: what the walks by address need, kept inline (the raster thread walks this for every announcement and upload; a hash
/// lookup per entry was the hottest code of the raster thread, 2026-10-30 profile)
struct AddrEntry {
    u64 id;
    VAddr end;
    Kind kind;
    u32 usage, cpu_access;
};
std::map<VAddr, AddrEntry> resource_by_address;

/// E2 study (`TLG_DESC_CHECKS=1`): the first bytes of every resource and view object as they were when its creation returned (objects are recycled later)
std::mutex raw_mutex;
std::unordered_map<u64, std::array<u8, 0x400>> raw_by_id;
bool RawOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_DESC_CHECKS");
        return e != nullptr && *e == '1';
    }();
    return on;
}
void KeepRaw(u64 id, u64 object) {
    if (!RawOn() || id == 0 || object == 0) {
        return;
    }
    std::array<u8, 0x400> bytes;
    std::memcpy(bytes.data(), reinterpret_cast<const void*>(object), bytes.size());
    std::scoped_lock lock{raw_mutex};
    raw_by_id[id] = bytes;
}

std::atomic<u64> resources_noted{0}, views_noted{0}, deletes_noted{0}, deletes_unknown{0}, bad_reads{0};
/// AddDelayDeleteObject (native-v3-writers): the game's own delay (frames until the object is freed) by age of the object's stamp, and the extra delay TLG_DELETE_DELAY adds
std::atomic<u64> delay_by_age[6]{}, delay_added{0}, delay_requeued{0};
std::atomic<u64> deferred_count{0}, deferred_released{0}, deferred_forced{0}, deferred_max_pending{0};
std::atomic<u64> slot_advances{0}, slot_waits{0}, slot_waited_submissions{0};
std::atomic<u64> skin_writes{0}, skin_waits{0}, skin_waited_submissions{0};
std::atomic<u64> gate_frees{0}, gate_checks{0}, gate_hits{0}, gate_waited_frames{0}, gate_missed{0}, gate_missed_unsized{0}, gate_window_open{0};
std::array<std::atomic<u64>, 9> live_by_kind{};

constexpr const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Buffer: return "buffer";
    case Kind::Texture1D: return "texture_1d";
    case Kind::Texture2D: return "texture_2d";
    case Kind::Texture3D: return "texture_3d";
    case Kind::ShaderResourceView: return "shader_resource_view";
    case Kind::UnorderedAccessView: return "unordered_access_view";
    case Kind::RenderTargetView: return "render_target_view";
    case Kind::DepthStencilView: return "depth_stencil_view";
    default: return "unknown";
    }
}

/// A resource object's fields (the same layout `NativeResources::ReadResource` reads): id +0x10, memory +0x30, kind (high dword of +0x120), the
/// Direct3D 11 description at +0x180
bool ReadResourceObject(u64 object, Object& out) {
    if (object == 0) {
        return false;
    }
    out = {};
    out.object = object;
    out.id = Rd<u64>(object, 0x10);
    out.address = Rd<u64>(object, 0x30);
    const u32 kind = static_cast<u32>(Rd<u64>(object, 0x120) >> 32);
    const auto d = [&](u32 index) { return Rd<u32>(object, 0x180 + index * 4); };
    switch (kind) {
    case 1: // ByteWidth, Usage, BindFlags, CPUAccessFlags, MiscFlags, StructureByteStride
        out.kind = Kind::Buffer;
        out.width = d(0), out.usage = d(1), out.bind = d(2), out.cpu_access = d(3), out.misc = d(4), out.stride = d(5);
        out.bytes = out.width;
        break;
    case 2: // Width, MipLevels, ArraySize, Format, Usage, BindFlags
        out.kind = Kind::Texture1D;
        out.width = d(0), out.height = 1, out.depth = 1, out.mips = d(1), out.layers = d(2), out.format = d(3), out.usage = d(4), out.bind = d(5), out.cpu_access = d(6);
        break;
    case 3: // Width, Height, MipLevels, ArraySize, Format, SampleCount, SampleQuality, Usage, BindFlags
        out.kind = Kind::Texture2D;
        out.width = d(0), out.height = d(1), out.depth = 1, out.mips = d(2), out.layers = d(3), out.format = d(4), out.usage = d(7), out.bind = d(8), out.cpu_access = d(9);
        break;
    case 4: // Width, Height, Depth, MipLevels, Format, Usage, BindFlags
        out.kind = Kind::Texture3D;
        out.width = d(0), out.height = d(1), out.depth = d(2), out.mips = d(3), out.layers = 1, out.format = d(4), out.usage = d(5), out.bind = d(6), out.cpu_access = d(7);
        break;
    default:
        return false;
    }
    if (out.kind != Kind::Buffer) {
        out.bytes = Rd<u64>(object, 0x38);
    }
    out.alive = true;
    return out.id != 0;
}

void Insert(const Object& object) {
    std::unique_lock lock{table_mutex};
    if (!by_id.contains(object.id) && object.kind >= Kind::Texture1D && object.kind <= Kind::Texture3D && object.address != 0 && object.bytes != 0) {
        Core::NativeUpdates::NoteEngineCreate(object.id, object.address, object.bytes); // B8.4: the memory's older images are unlisted, in order with the GPU work
    } else if (!by_id.contains(object.id) && object.kind == Kind::Buffer && object.address != 0) {
        Core::NativeUpdates::NoteEngineBufferCreate(object.id, object.address, std::max<u64>(object.bytes, 1), object.usage, object.cpu_access); // B9.4
    }
    if (const auto it = by_id.find(object.id); it != by_id.end()) {
        if (it->second.alive) {
            live_by_kind[static_cast<u32>(it->second.kind)].fetch_sub(1, std::memory_order_relaxed);
        }
        id_by_object.erase(it->second.object);
    }
    if (const auto old = by_id.find(object.id); old != by_id.end() && old->second.address != object.address) {
        if (const auto at = resource_by_address.find(old->second.address); at != resource_by_address.end() && at->second.id == object.id) {
            resource_by_address.erase(at);
        }
    }
    by_id[object.id] = object;
    id_by_object[object.object] = object.id;
    if (object.kind >= Kind::Buffer && object.kind <= Kind::Texture3D && object.address != 0) {
        const u64 bytes = object.kind == Kind::Buffer ? std::max<u64>(object.bytes, 1) : object.bytes;
        resource_by_address[object.address] = AddrEntry{object.id, object.address + bytes, object.kind, object.usage, object.cpu_access};
    }
    live_by_kind[static_cast<u32>(object.kind)].fetch_add(1, std::memory_order_relaxed);
}

template <Kind ViewKind>
u64 ViewCreated(const char* hook_name, u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5) {
    static std::atomic<u64> original_entry{0};
    u64 original = original_entry.load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry(hook_name);
        original_entry.store(original, std::memory_order_relaxed);
    }
    if (original == 0) {
        return 0; // run the original in place
    }
    const u64 result = reinterpret_cast<GuestFunction6>(original)(resource, desc, out, a3, a4, a5);
    if (out != 0 && static_cast<u32>(result) == 0) {
        const u64 view = Rd<u64>(out, 0);
        if (view != 0) {
            NoteView(ViewKind, view, desc);
        }
    }
    return 0x8000000000000000ull | (result & 0x7fffffffffffffffull);
}

} // namespace

bool Enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_NATIVE_DEVICE");
        return e == nullptr || *e != '0';
    }();
    return on;
}

void NoteResource(u64 object) {
    if (!Enabled()) {
        return;
    }
    Object entry;
    if (!ReadResourceObject(object, entry)) {
        bad_reads.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    resources_noted.fetch_add(1, std::memory_order_relaxed);
    KeepRaw(entry.id, object);
    Insert(entry);
}

void NoteView(Kind kind, u64 view_object, u64 desc) {
    if (!Enabled() || view_object == 0) {
        return;
    }
    Object view;
    view.object = view_object;
    view.id = Rd<u64>(view_object, 0x10);
    view.kind = kind;
    view.alive = true;
    Object resource;
    const u64 resource_object = Rd<u64>(view_object, 0x30);
    if (view.id == 0 || !ReadResourceObject(resource_object, resource)) {
        bad_reads.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    view.resource_id = resource.id;
    view.resource_kind = resource.kind;
    if (desc != 0) {
        view.has_desc = true;
        std::memcpy(view.desc, reinterpret_cast<const void*>(desc), sizeof(view.desc));
    }
    if ((kind == Kind::ShaderResourceView || kind == Kind::UnorderedAccessView) && resource.kind >= Kind::Texture1D && resource.kind <= Kind::Texture3D) {
        view.has_tsharp = true;
        std::memcpy(view.tsharp, reinterpret_cast<const void*>(view_object + (kind == Kind::ShaderResourceView ? 0x58 : 0x54)), sizeof(view.tsharp));
        if (kind == Kind::ShaderResourceView && NativeTsharpWatch::On()) {
            NativeTsharpWatch::Arm(view_object, view_object + 0x60, resource.width, resource.height); // the T# height dword (diagnostic, F4 question a)
        }
    }
    view.address = resource.address;
    view.bytes = resource.bytes;
    view.width = resource.width, view.height = resource.height, view.depth = resource.depth, view.mips = resource.mips, view.layers = resource.layers;
    view.format = resource.format, view.usage = resource.usage, view.bind = resource.bind, view.misc = resource.misc;
    views_noted.fetch_add(1, std::memory_order_relaxed);
    KeepRaw(view.id, view_object);
    Insert(view);
    {
        // the resource itself is known from here on as well (textures are read once their initialization returned; their view creation comes after)
        std::shared_lock lock{table_mutex};
        if (by_id.contains(resource.id)) {
            return;
        }
    }
    KeepRaw(resource.id, resource_object);
    Insert(resource);
}

void NoteDelete(u64 object) {
    if (!Enabled() || object == 0) {
        return;
    }
    deletes_noted.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock{table_mutex};
    const auto it = id_by_object.find(object);
    if (it == id_by_object.end()) {
        deletes_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (const auto entry = by_id.find(it->second); entry != by_id.end() && entry->second.alive) {
        entry->second.alive = false;
        live_by_kind[static_cast<u32>(entry->second.kind)].fetch_sub(1, std::memory_order_relaxed);
        const Object& gone = entry->second;
        if (const auto at = resource_by_address.find(gone.address); at != resource_by_address.end() && at->second.id == gone.id) {
            resource_by_address.erase(at); // a deleted resource leaves the map by memory (the walks below would count it)
        }
        const bool texture = gone.kind >= Kind::Texture1D && gone.kind <= Kind::Texture3D;
        if (texture && gone.address != 0) { // (a texture whose size read 0 at creation still owns images: its lists go too)
            // B8.2: another live texture at the same memory (an alias) keeps the images (they are not freed; the deleted texture's lists still go)
            const auto at = resource_by_address.find(gone.address);
            const bool alias = at != resource_by_address.end() && at->second.id != gone.id && at->second.kind >= Kind::Texture1D && at->second.kind <= Kind::Texture3D;
            if (alias) {
                Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8TextureAliasKept);
            }
            Core::NativeUpdates::NoteEngineDelete(gone.id, gone.address, gone.bytes, alias ? 3u : 1u);
        } else if (gone.kind == Kind::ShaderResourceView || gone.kind == Kind::UnorderedAccessView) {
            Core::NativeUpdates::NoteEngineDelete(gone.id, gone.address, gone.bytes, 0u);
        } else if (gone.kind == Kind::Buffer && gone.address != 0) {
            Core::NativeUpdates::NoteEngineDelete(gone.id, gone.address, gone.bytes, 4u); // B9.2
        }
    }
    id_by_object.erase(it);
}

bool ResourceAt(VAddr address, Object& out) {
    std::shared_lock lock{table_mutex};
    auto it = resource_by_address.upper_bound(address);
    if (it == resource_by_address.begin()) {
        return false;
    }
    --it;
    if (address >= std::max(it->second.end, it->first + 1)) {
        return false; // the nearest resource below ends before the address
    }
    const auto entry = by_id.find(it->second.id);
    if (entry == by_id.end()) {
        return false;
    }
    out = entry->second;
    return true;
}

bool TexturesIn(VAddr address, u64 size, std::vector<u64>& out) {
    if (size == 0) {
        return false;
    }
    const VAddr end = address + size;
    std::array<std::pair<VAddr, VAddr>, 16> spans;
    u32 num_spans = 0;
    bool overflow = false;
    {
        std::shared_lock lock{table_mutex};
        auto it = resource_by_address.lower_bound(end);
        u32 below = 0; // entries that start before the range: a few are looked at (a texture that starts earlier and reaches in)
        while (it != resource_by_address.begin() && below < 16) {
            --it;
            const AddrEntry& e = it->second;
            if (it->first < address) {
                ++below;
            }
            if (e.kind < Kind::Texture1D || e.kind > Kind::Texture3D || e.end <= address) {
                continue;
            }
            out.push_back(e.id);
            if (num_spans < spans.size()) {
                spans[num_spans++] = {std::max(it->first, address), std::min(e.end, end)};
            } else {
                overflow = true;
            }
        }
    }
    if (overflow || num_spans == 0) {
        return false;
    }
    std::sort(spans.begin(), spans.begin() + num_spans);
    VAddr covered = address;
    for (u32 i = 0; i < num_spans && covered < end; ++i) {
        if (spans[i].first > covered) {
            return false;
        }
        covered = std::max(covered, spans[i].second);
    }
    return covered >= end;
}

void BuffersIn(VAddr address, u64 size, std::vector<BufferPart>& out) {
    if (size == 0) {
        return;
    }
    const VAddr end = address + size;
    std::shared_lock lock{table_mutex};
    auto it = resource_by_address.lower_bound(end);
    u32 below = 0; // entries that start before the range: a few are looked at (a buffer that starts earlier and reaches in)
    while (it != resource_by_address.begin() && below < 16) {
        --it;
        const AddrEntry& e = it->second;
        if (it->first < address) {
            ++below;
        }
        if (e.kind != Kind::Buffer || e.end <= address) {
            continue;
        }
        out.push_back({e.id, std::max(it->first, address), std::min<VAddr>(e.end, end), e.usage, e.cpu_access, it->first, e.end});
    }
}

bool B6NativeOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_B6_NATIVE");
        return e == nullptr || *e != '0'; // B7: default on (owner play 2026-10-29)
    }();
    return on;
}

u32 CpuReadReason(VAddr address, u64 size) {
    Object o;
    if (!ResourceAt(address, o) || address + size > o.address + std::max<u64>(o.bytes, 1)) {
        return 2; // not (wholly) an engine resource: kept
    }
    return o.usage == 3 || (o.cpu_access & 0x20000) != 0 ? 1u : 0u;
}

bool CpuReads(VAddr address, u64 size) {
    return CpuReadReason(address, size) != 0;
}

std::string OwnerName(VAddr address) {
    Object o;
    if (!ResourceAt(address, o)) {
        return "none";
    }
    return fmt::format("{} usage {} bind {:#x}", KindName(o.kind), o.usage, o.bind);
}

bool RawBytes(u64 id, u8* out, size_t size) {
    std::scoped_lock lock{raw_mutex};
    const auto it = raw_by_id.find(id);
    if (it == raw_by_id.end()) {
        return false;
    }
    std::memcpy(out, it->second.data(), std::min<size_t>(size, it->second.size()));
    return true;
}

bool Lookup(u64 id, Object& out) {
    std::shared_lock lock{table_mutex};
    const auto it = by_id.find(id);
    if (it == by_id.end() || !it->second.alive) {
        return false;
    }
    out = it->second;
    return true;
}

void ViewsOf(u64 resource_object, std::vector<u64>& out, std::vector<u64>& out_uav) {
    std::shared_lock lock{table_mutex};
    const auto rid = id_by_object.find(resource_object);
    if (rid == id_by_object.end()) {
        return;
    }
    for (const auto& [id, o] : by_id) {
        if (o.alive && o.resource_id == rid->second && o.object != 0) {
            if (o.kind == Kind::ShaderResourceView) {
                out.push_back(o.object);
            } else if (o.kind == Kind::UnorderedAccessView) {
                out_uav.push_back(o.object);
            }
        }
    }
}

namespace {
struct LastMap {
    u64 address, size;
    u32 type;
};
std::mutex map_mutex;
std::unordered_map<u64, LastMap> last_map;
} // namespace

void NoteMapped(u64 id, u64 address, u64 size, u32 type) {
    if (id == 0 || address == 0) {
        return;
    }
    std::scoped_lock lock{map_mutex};
    last_map[id] = {address, size, type};
}

namespace {
/// E2 census state: the keys are short descriptions ("size same", "format 28 -> 5/0"), the values how often they were seen
std::mutex sharp_mutex;
std::map<std::string, u64> sharp_counts;
std::atomic<u32> sharp_logged{0};
std::atomic<u64> sharp_checked{0}, sharp_unknown{0};

bool DescChecksOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_DESC_CHECKS");
        return e != nullptr && *e == '1';
    }();
    return on;
}
} // namespace

void CheckImageSharp(u64 id, u64 view_id, const SharpFacts& f) {
    if (!Enabled() || id == 0 || !DescChecksOn()) {
        return;
    }
    Object res, view;
    if (!Lookup(id, res)) {
        sharp_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool has_view = view_id != 0 && Lookup(view_id, view) && view.has_desc;
    sharp_checked.fetch_add(1, std::memory_order_relaxed);
    const u32 t_width = f.width + 1, t_height = f.height + 1;
    const u32 t_levels = f.last_level + 1;
    // AmdGpu::ImageType: 10 3D, 11 cube, 12 / 13 / 15 array types. The depth field is the layer count of an array, the number of cubes of a cube, the depth of a 3D image
    const u32 t_layers = f.type == 10 ? 1 : f.type == 11 ? (f.depth + 1) * 6 : f.depth + 1;
    std::string keys[7];
    keys[0] = (t_width == res.width && t_height == res.height) ? "size: same as the resource" : "size: DIFFERENT";
    keys[1] = f.last_level < res.mips ? (f.base_level == 0 && t_levels == res.mips ? "mips: all of the resource" : "mips: a part of the resource") : "mips: LAST LEVEL beyond the resource";
    if (f.type == 10) {
        keys[2] = (f.depth + 1 == res.depth) ? "depth: same" : "depth: DIFFERENT";
    } else {
        keys[2] = t_layers == std::max<u32>(res.layers, 1) ? "layers: same as the resource" : "layers: DIFFERENT";
    }
    keys[6] = f.last_array >= f.base_array && f.last_array < t_layers ? "array range: inside the layers" : "array range: OUTSIDE the layers";
    keys[3] = fmt::format("resource kind {} type {} (resource layers {})", KindName(res.kind), f.type, res.layers > 1 ? "many" : "one");
    // the format: the view description's when the view has one (a typeless resource has no format of its own), else the resource's
    const u32 engine_format = has_view ? view.desc[0] : res.format;
    keys[4] = fmt::format("format {} -> {}/{}", engine_format, f.data_format, f.num_format);
    // properties the console's descriptor holds that no D3D11 description has: they must be functions of the format (and kind) if the engine's data is to give them
    std::string more[4];
    more[0] = fmt::format("swizzle {} -> {:#x}", engine_format, f.dst_sel);
    more[1] = fmt::format("tiling {} bind {:#x} fmt {} -> {}", KindName(res.kind), res.bind, engine_format, f.tiling_index);
    more[2] = fmt::format("pow2pad {} -> {}", KindName(res.kind), f.pow2pad);
    more[3] = f.address == res.address ? "address: same as the resource" : "address: DIFFERENT from the resource";
    keys[5] = !has_view ? "view format: NO VIEW DESCRIPTION" : view.desc[0] == res.format ? "view format: same as the resource" : "view format: differs from the resource";
    std::string view_keys[3];
    if (has_view) {
        // D3D11_SHADER_RESOURCE_VIEW_DESC (UAV / RTV / DSV descriptions differ after the dimension): dimension, then the range
        const u32 dim = view.desc[1];
        view_keys[0] = fmt::format("view dimension {} -> T# type {}", dim, f.type);
        const bool srv = view.kind == Kind::ShaderResourceView;
        if (srv && (dim == 4 || dim == 5 || dim == 8 || dim == 9 || dim == 2 || dim == 3)) {
            const u32 most_detailed = view.desc[2];
            const u32 levels = view.desc[3] == 0xffffffffu ? res.mips - std::min(most_detailed, res.mips) : view.desc[3];
            view_keys[1] = (most_detailed == f.base_level && most_detailed + levels - 1 == f.last_level) ? "srv mip range: same as the T#" : "srv mip range: DIFFERENT";
            if (dim == 5 || dim == 3) {
                const u32 first = view.desc[4];
                const u32 count = view.desc[5] == 0xffffffffu ? std::max<u32>(res.layers, 1) - std::min(first, std::max<u32>(res.layers, 1)) : view.desc[5];
                view_keys[2] = (first == f.base_array && first + count - 1 == f.last_array) ? "srv array range: same as the T#" : "srv array range: DIFFERENT";
            }
        } else {
            view_keys[1] = fmt::format("{} view: range not decoded", KindName(view.kind));
        }
    }
    {
        std::scoped_lock lock{sharp_mutex};
        for (u32 i = 0; i < 7; ++i) {
            ++sharp_counts[keys[i]];
        }
        for (const auto& key : more) {
            ++sharp_counts[key];
        }
        for (const auto& key : view_keys) {
            if (!key.empty()) {
                ++sharp_counts[key];
            }
        }
    }
    static std::atomic<u32> logged_by[3]{};
    const u32 bad = keys[0][6] == 'D' ? 0u : keys[1][6] == 'L' ? 1u : keys[6][13] == 'O' ? 2u : 3u;
    if (bad < 3 && logged_by[bad].fetch_add(1, std::memory_order_relaxed) < 6) {
        TLG_LOG(GPU, Warn, "E2 census: {} id {}: T# {}x{} levels {}+{} layers {}+{} depth {} type {}, resource {}x{} mips {} layers {} depth {}", KindName(res.kind), id, t_width, t_height, f.base_level, t_levels,
                    f.base_array, t_layers, f.depth + 1, f.type, res.width, res.height, res.mips, res.layers, res.depth);
    }
}

void CheckBufferSharp(u64 id, u64 view_id, u64 base_address, u32 stride, u32 num_records, u32 data_format, u32 num_format, bool is_constant_buffer_hint) {
    if (!Enabled() || id == 0 || !DescChecksOn()) {
        return;
    }
    Object buf, view;
    if (!Lookup(id, buf) || buf.kind != Kind::Buffer) {
        sharp_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool view_known = view_id != 0 && Lookup(view_id, view);
    const bool has_desc = view_known && view.has_desc;
    const u64 size = stride != 0 ? u64(stride) * num_records : num_records;
    std::string keys[6];
    const u64 offset = base_address - buf.address;
    const bool inside = base_address >= buf.address && base_address + size <= buf.address + buf.bytes;
    const char* where = inside ? (offset == 0 && size == buf.bytes ? "whole buffer" : offset == 0 ? "from the start, shorter" : "a sub-range") : "OUTSIDE the buffer";
    keys[0] = fmt::format("buffer v# {} | bind {:#x} | view {}", where, buf.bind, !view_known ? "none" : has_desc ? "with a description" : "default description");
    {
        // the object's memory pointer as it is now (a Map with discard moves it)
        const u64 live = Rd<u64>(buf.object, 0x30);
        const u64 live_bytes = buf.bytes;
        const bool live_inside = base_address >= live && base_address + size <= live + live_bytes;
        keys[4] = fmt::format("buffer live memory pointer {} | bind {:#x} | {}", live == buf.address ? "unchanged" : "MOVED", buf.bind, live_inside ? "v# inside it" : "v# OUTSIDE it");
    }
    {
        LastMap map{};
        bool have = false;
        {
            std::scoped_lock lock{map_mutex};
            if (const auto it = last_map.find(id); it != last_map.end()) {
                map = it->second;
                have = true;
            }
        }
        const char* at = base_address == buf.address ? "the object's memory" : !have ? "NO MAP KNOWN" : (base_address >= map.address && base_address < map.address + std::max<u64>(map.size, 1)) ? "inside the last Map" : "ELSEWHERE";
        keys[5] = fmt::format("buffer v# at {} | bind {:#x} | last map type {}", at, buf.bind, have ? map.type : 0);
    }
    keys[1] = fmt::format("buffer stride v# {} | object stride {} | bind {:#x}", stride == 0 ? "0" : stride == buf.stride ? "same" : "other", buf.stride, buf.bind);
    if (has_desc) {
        // D3D11_BUFFER_SRV / UAV: format, dimension, FirstElement (or ElementOffset), NumElements
        const u32 first = view.desc[2], count = view.desc[3];
        const u32 element = stride != 0 ? stride : 1;
        keys[2] = fmt::format("buffer view range {} | {}", (first == offset / element && count == num_records) ? "equals the v#" : "DIFFERENT from the v#", view.kind == Kind::ShaderResourceView ? "srv" : "uav");
        keys[3] = fmt::format("buffer view format {} -> {}/{} stride {}", view.desc[0], data_format, num_format, stride);
    }
    {
        std::scoped_lock lock{sharp_mutex};
        for (const auto& key : keys) {
            if (!key.empty()) {
                ++sharp_counts[key];
            }
        }
    }
}

namespace {
/// B11 census
struct ClearFrame {
    u64 ret{}, ctx{}, cursor{};
    u32 kind{}; ///< 0 color, 1 depth / stencil
    u32 path{}; ///< color: 0 the resource has no surface pointer at +0x58 (a fill of its memory), 1 it has; depth: the same at +0x48
};
thread_local ClearFrame clear_stack[8];
thread_local u32 clear_depth = 0;
enum ClearCount : u32 { ClearColor, ClearDepth, ClearColorFill, ClearDepthNoSurface, ClearUnreadable, ClearPackets, ClearDwords, ClearDepthFlags1, ClearDepthFlags2, ClearDepthFlags3, ClearCounts };
std::array<std::atomic<u64>, ClearCounts> clear_counts{};
std::mutex clear_mutex;
std::map<u32, u64> clear_opcodes; ///< (kind * 2 + path) << 8 | opcode -> packets
std::map<u32, u64> clear_shapes;  ///< (kind * 2 + path) << 16 | packets in the clear -> clears
std::atomic<u32> clear_logged{0};
std::map<u64, ClearRecord> pending_clears; ///< by begin (clear_mutex)
thread_local ClearRecord clear_record[8];
enum ClearMatch : u32 { MatchRecorded, MatchAttached, MatchDropped, MatchEntered, MatchEntriesTag, MatchEntriesSync, MatchEntriesPacket, MatchEntriesOther, MatchCounts };
std::array<std::atomic<u64>, MatchCounts> clear_match{};
std::array<std::array<std::atomic<u64>, 2>, ClearUses> clear_uses{};
std::array<std::atomic<u64>, ClearNatives> clear_native{};

u64 PS4_SYSV_ABI ClearExit() {
    const ClearFrame f = clear_stack[--clear_depth];
    const u64 end = Rd<u64>(f.ctx, 0x47d0);
    const u32 group = f.kind * 2 + f.path;
    if (end < f.cursor || end - f.cursor > (u64{64} << 10)) {
        clear_counts[ClearUnreadable].fetch_add(1, std::memory_order_relaxed); // the command buffer moved to a new chunk
        return f.ret;
    }
    u32 packets = 0;
    std::scoped_lock lock{clear_mutex};
    for (u64 at = f.cursor; at + 4 <= end;) {
        const u32 header = Rd<u32>(at, 0);
        const u32 type = header >> 30;
        if (type == 3) {
            const u32 opcode = (header >> 8) & 0xff;
            const u32 count = ((header >> 16) & 0x3fff) + 2;
            ++clear_opcodes[group << 8 | opcode];
            ++packets;
            at += u64{count} * 4;
        } else if (type == 0) {
            const u32 count = ((header >> 16) & 0x3fff) + 2;
            ++clear_opcodes[group << 8 | 0xfe]; // a type 0 register write
            ++packets;
            at += u64{count} * 4;
        } else {
            at += 4; // type 2 filler
        }
    }
    clear_counts[ClearPackets].fetch_add(packets, std::memory_order_relaxed);
    clear_counts[ClearDwords].fetch_add((end - f.cursor) / 4, std::memory_order_relaxed);
    ++clear_shapes[group << 16 | std::min<u32>(packets, 0xffff)];
    if (end > f.cursor) {
        // B11 step 2: the clear waits for the submission of its command buffer
        ClearRecord record = clear_record[clear_depth];
        record.begin = f.cursor;
        record.end = end;
        pending_clears[f.cursor] = record;
        clear_match[MatchRecorded].fetch_add(1, std::memory_order_relaxed);
        if (pending_clears.size() > 4096) {
            pending_clears.erase(pending_clears.begin()); // never submitted (or its memory reused): dropped
            clear_match[MatchDropped].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return f.ret;
}

/// B11 step 4: the subresources an RTV (`dsv` false) or DSV names, from its description (Format +0x40, ViewDimension +0x44, then the D3D11 union;
/// the DSV has Flags at +0x48 before the union). Dimensions: RTV 2/3 1D(array), 4/5 2D(array), 6/7 2D MS(array), 8 3D; DSV 1/2 1D(array),
/// 3/4 2D(array), 5/6 2D MS(array) (JsRenderTargetView.cpp / JsDepthStencilView.cpp, CreateRenderTargetViewDesc 0x2491370, CreateDepthStencilViewDesc 0x2474fb0)
void ReadViewRange(u64 view, bool dsv, ClearRecord& r) {
    const u32 dim = Rd<u32>(view, 0x44);
    const u64 u = dsv ? 0x4c : 0x48;
    r.mip = 0, r.first_layer = 0, r.layer_count = 1;
    const bool plain = dsv ? (dim == 1 || dim == 3) : (dim == 2 || dim == 4);
    const bool array = dsv ? (dim == 2 || dim == 4) : (dim == 3 || dim == 5);
    const bool ms = dsv ? dim == 5 : dim == 6;
    const bool ms_array = dsv ? dim == 6 : dim == 7;
    if (plain) {
        r.mip = Rd<u32>(view, u);
    } else if (array) {
        r.mip = Rd<u32>(view, u);
        r.first_layer = Rd<u32>(view, u + 4);
        r.layer_count = Rd<u32>(view, u + 8);
    } else if (ms_array) {
        r.first_layer = Rd<u32>(view, u);
        r.layer_count = Rd<u32>(view, u + 4);
    } else if (!ms) {
        r.layer_count = 0; // 3D (slices) or not a texture view
    }
}

u64 ClearExitStub() {
    static const u64 stub = Core::GuestHooks::ExitStub(reinterpret_cast<u64>(&ClearExit));
    return stub;
}
} // namespace

u64 PS4_SYSV_ABI ClearRtvEnter(Core::GuestHooks::MidContext* c) {
    const u64 stub = ClearExitStub();
    if (clear_depth >= 8 || stub == 0 || c->rdi == 0 || c->rsi == 0) {
        return 0;
    }
    ClearFrame& f = clear_stack[clear_depth++];
    f.ctx = c->rdi;
    f.kind = 0;
    f.cursor = Rd<u64>(c->rdi, 0x47d0);
    const u64 resource = Rd<u64>(c->rsi, 0x30);
    f.path = resource != 0 && Rd<u64>(resource, 0x58) != 0 ? 1 : 0;
    clear_counts[ClearColor].fetch_add(1, std::memory_order_relaxed);
    clear_counts[ClearColorFill].fetch_add(f.path == 0 ? 1 : 0, std::memory_order_relaxed);
    {
        ClearRecord& r = clear_record[clear_depth - 1];
        r = {};
        r.kind = 0;
        r.path = f.path;
        r.view_id = Rd<u64>(c->rsi, 0x10);
        r.resource_id = resource != 0 ? Rd<u64>(resource, 0x10) : 0;
        r.address = resource != 0 ? Rd<u64>(resource, 0x30) : 0;
        ReadViewRange(c->rsi, false, r);
        if (c->rdx != 0) {
            std::memcpy(r.value, reinterpret_cast<const void*>(c->rdx), sizeof(r.value));
        }
    }
    if (clear_logged.fetch_add(1, std::memory_order_relaxed) < 16 && c->rdx != 0) {
        TLG_LOG(GPU, Warn, "B11: ClearRenderTargetView rtv id {} resource id {} memory {:#x} surface {:#x} color {} {} {} {}", Rd<u64>(c->rsi, 0x10),
                    resource != 0 ? Rd<u64>(resource, 0x10) : 0, resource != 0 ? Rd<u64>(resource, 0x30) : 0, resource != 0 ? Rd<u64>(resource, 0x58) : 0,
                    Rd<float>(c->rdx, 0), Rd<float>(c->rdx, 4), Rd<float>(c->rdx, 8), Rd<float>(c->rdx, 12));
    }
    u64* ret_slot = reinterpret_cast<u64*>(c->rsp + 16); // the hook sits after push rbp; mov rbp, rsp; push r15 (the B0 count hook displaced those 6 bytes)
    f.ret = *ret_slot;
    *ret_slot = stub; // the function returns into the exit stub
    return 0;
}

u64 PS4_SYSV_ABI ClearDsvEnter(Core::GuestHooks::MidContext* c) {
    const u64 stub = ClearExitStub();
    if (clear_depth >= 8 || stub == 0 || c->rdi == 0 || c->rsi == 0) {
        return 0;
    }
    ClearFrame& f = clear_stack[clear_depth++];
    f.ctx = c->rdi;
    f.kind = 1;
    f.cursor = Rd<u64>(c->rdi, 0x47d0);
    const u64 resource = Rd<u64>(c->rsi, 0x30);
    f.path = resource != 0 && Rd<u64>(resource, 0x48) != 0 ? 1 : 0;
    const u32 flags = static_cast<u32>(c->rdx);
    clear_counts[ClearDepth].fetch_add(1, std::memory_order_relaxed);
    {
        ClearRecord& r = clear_record[clear_depth - 1];
        r = {};
        r.kind = 1;
        r.path = f.path;
        r.flags = flags;
        r.stencil = static_cast<u32>(c->rcx & 0xff);
        r.view_id = Rd<u64>(c->rsi, 0x10);
        r.resource_id = resource != 0 ? Rd<u64>(resource, 0x10) : 0;
        r.address = resource != 0 ? Rd<u64>(resource, 0x30) : 0;
        std::memcpy(&r.value[0], c->ymm[0], sizeof(float));
        ReadViewRange(c->rsi, true, r);
    }
    clear_counts[ClearDepthNoSurface].fetch_add(f.path == 0 ? 1 : 0, std::memory_order_relaxed);
    if (flags >= 1 && flags <= 3) {
        clear_counts[ClearDepthFlags1 + flags - 1].fetch_add(1, std::memory_order_relaxed);
    }
    float depth;
    std::memcpy(&depth, c->ymm[0], sizeof(depth));
    if (clear_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        TLG_LOG(GPU, Warn, "B11: ClearDepthStencilView dsv id {} resource id {} memory {:#x} surface {:#x} flags {} depth {} stencil {}", Rd<u64>(c->rsi, 0x10),
                    resource != 0 ? Rd<u64>(resource, 0x10) : 0, resource != 0 ? Rd<u64>(resource, 0x30) : 0, resource != 0 ? Rd<u64>(resource, 0x48) : 0, flags, depth,
                    static_cast<u32>(c->rcx & 0xff));
    }
    u64* ret_slot = reinterpret_cast<u64*>(c->rsp + 16); // the hook sits after push rbp; mov rbp, rsp; push r15 (the B0 count hook displaced those 6 bytes)
    f.ret = *ret_slot;
    *ret_slot = stub;
    return 0;
}

bool CurrentDepthClear(ClearRecord& out) {
    if (clear_depth == 0 || clear_depth > 8 || clear_record[clear_depth - 1].kind != 1) {
        return false;
    }
    out = clear_record[clear_depth - 1];
    return true;
}

void TakeClears(u64 begin, u64 end, std::vector<ClearRecord>& out) {
    std::scoped_lock lock{clear_mutex};
    for (auto it = pending_clears.lower_bound(begin); it != pending_clears.end() && it->first < end;) {
        out.push_back(it->second);
        it = pending_clears.erase(it);
        clear_match[MatchAttached].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteClearNative(ClearNative what, u64 n) {
    clear_native[what].fetch_add(n, std::memory_order_relaxed);
}

void NoteClearUse(ClearUse use, bool inside) {
    clear_uses[use][inside ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

void NoteClearEntered(const ClearRecord& clear, const u32 (&entries)[4]) {
    (void)clear;
    clear_match[MatchEntered].fetch_add(1, std::memory_order_relaxed);
    clear_match[MatchEntriesTag].fetch_add(entries[0], std::memory_order_relaxed);
    clear_match[MatchEntriesSync].fetch_add(entries[1], std::memory_order_relaxed);
    clear_match[MatchEntriesPacket].fetch_add(entries[2], std::memory_order_relaxed);
    clear_match[MatchEntriesOther].fetch_add(entries[3], std::memory_order_relaxed);
}

nlohmann::json ClearReport() {
    const auto c = [](ClearCount what) { return clear_counts[what].load(std::memory_order_relaxed); };
    nlohmann::json opcodes = nlohmann::json::object();
    nlohmann::json shapes = nlohmann::json::object();
    static constexpr const char* groups[] = {"color_fill", "color_surface", "depth_no_surface", "depth_surface"};
    {
        std::scoped_lock lock{clear_mutex};
        for (const auto& [key, n] : clear_opcodes) {
            opcodes[fmt::format("{} {:#04x}", groups[(key >> 8) & 3], key & 0xff)] = n;
        }
        for (const auto& [key, n] : clear_shapes) {
            shapes[fmt::format("{} {} packets", groups[(key >> 16) & 3], key & 0xffff)] = n;
        }
    }
    return {{"color_clears", c(ClearColor)}, {"of_those_fills", c(ClearColorFill)}, {"depth_clears", c(ClearDepth)}, {"of_those_without_surface", c(ClearDepthNoSurface)},
            {"depth_flags", {c(ClearDepthFlags1), c(ClearDepthFlags2), c(ClearDepthFlags3)}}, {"unreadable_new_chunk", c(ClearUnreadable)},
            {"packets", c(ClearPackets)}, {"dwords", c(ClearDwords)}, {"packets_by_opcode", opcodes}, {"clears_by_packet_count", shapes},
            {"matching", {{"recorded", clear_match[MatchRecorded].load()}, {"attached_to_their_submission", clear_match[MatchAttached].load()},
                          {"dropped_never_submitted", clear_match[MatchDropped].load()}, {"met_by_the_command_processor", clear_match[MatchEntered].load()},
                          {"entries_inside", {{"tags", clear_match[MatchEntriesTag].load()}, {"syncs", clear_match[MatchEntriesSync].load()},
                                              {"packets", clear_match[MatchEntriesPacket].load()}, {"other", clear_match[MatchEntriesOther].load()}}}}},
            {"native", {{"clears_done_as_vulkan_clears", clear_native[NativeDone].load()}, {"images_cleared", clear_native[NativeImages].load()},
                        {"not_found_in_the_object_table", clear_native[NativeNotEligible].load()},
                        {"work_skipped_inside", clear_native[NativeSkippedWork].load()}, {"fast_clears_of_a_part_cleared_at_once", clear_native[NativeFastClearKept].load()},
                        {"of_one_mip_or_layer_or_a_mipped_texture", clear_native[NativeSubresource].load()},
                        {"fast_clears_pending", clear_native[NativeFastClearPending].load()}, {"fast_clears_applied_at_a_render_pass", clear_native[NativeFastClearAtRenderPass].load()},
                        {"fast_clears_applied_at_the_eliminate", clear_native[NativeFastClearAtEliminate].load()},
                        {"before_the_first_image", clear_native[NativeClearBeforeImage].load()}, {"applied_at_the_first_image", clear_native[NativeClearAtFirstImage].load()},
                        {"first_image_did_not_fit", clear_native[NativeClearFirstImageMismatch].load()}}},
            {"back_end_inside_vs_outside_a_clear", {
                {"meta_clear_dispatch", {clear_uses[UseMetaClearDispatch][1].load(), clear_uses[UseMetaClearDispatch][0].load()}},
                {"fast_clear_eliminate_clearing", {clear_uses[UseFastClearEliminate][1].load(), clear_uses[UseFastClearEliminate][0].load()}},
                {"fast_clear_eliminate_nothing", {clear_uses[UseFastClearEliminateNothing][1].load(), clear_uses[UseFastClearEliminateNothing][0].load()}},
                {"depth_clear_target", {clear_uses[UseDepthClearTarget][1].load(), clear_uses[UseDepthClearTarget][0].load()}},
                {"render_pass_clear", {clear_uses[UseRenderPassClear][1].load(), clear_uses[UseRenderPassClear][0].load()}},
                {"draws", {clear_uses[UseDraw][1].load(), clear_uses[UseDraw][0].load()}},
                {"dispatches", {clear_uses[UseDispatch][1].load(), clear_uses[UseDispatch][0].load()}}}},
            {"note", "B11 census, cumulative: the packets each engine clear writes into its command buffer; inside / outside pairs: [inside an engine clear's packets, outside]"}};
}

nlohmann::json W2Report();
nlohmann::json FreeHoldReport();
nlohmann::json Report() {
    nlohmann::json out;
    out["enabled"] = Enabled();
    out["resources_noted"] = resources_noted.load();
    out["views_noted"] = views_noted.load();
    out["deletes_noted"] = deletes_noted.load();
    out["deletes_of_objects_not_in_the_table"] = deletes_unknown.load();
    {
        nlohmann::json ages = nlohmann::json::array();
        for (const auto& count : delay_by_age) {
            ages.push_back(count.load());
        }
        out["delete_by_stamp_age_0_1_2_3_over_3_negative"] = std::move(ages);
        out["delete_delay_added"] = delay_added.load();
        out["slot_gate"] = {{"group_advances", slot_advances.load()}, {"advances_that_waited", slot_waits.load()}, {"submissions_waited_for", slot_waited_submissions.load()}};
        out["skin_gate"] = {{"matrix_slot_writes", skin_writes.load()}, {"writes_that_waited", skin_waits.load()}, {"submissions_waited_for", skin_waited_submissions.load()}};
        out["loader_gate"] = {{"frees_noted", gate_frees.load()}, {"allocations_checked", gate_checks.load()}, {"allocations_that_waited", gate_hits.load()}, {"submissions_waited_for", gate_waited_frames.load()}, {"submit_count", Core::NativeContent::SubmitCount()}, {"raster_done_floor", Core::NativeContent::RasterDoneSubmissions()}, {"wait_timeouts", Core::NativeContent::GateWaitTimeouts()}, {"alloc_on_recently_freed_address_without_gate_entry", gate_missed.load()}, {"of_those_free_had_no_known_size", gate_missed_unsized.load()}, {"alloc_in_the_window_of_the_free_not_waitable", gate_window_open.load()}};
        out["w2_window"] = W2Report();
        out["free_hold"] = FreeHoldReport();
        out["deferred_frees"] = {{"held", deferred_count.load()}, {"released", deferred_released.load()}, {"forced", deferred_forced.load()}, {"max_pending", deferred_max_pending.load()}};
    }
    out["unreadable_objects"] = bad_reads.load();
    nlohmann::json live;
    for (u32 k = 1; k < live_by_kind.size(); ++k) {
        live[KindName(static_cast<Kind>(k))] = live_by_kind[k].load();
    }
    out["live"] = std::move(live);
    {
        nlohmann::json census;
        census["checked"] = sharp_checked.load();
        census["id_not_in_the_table"] = sharp_unknown.load();
        std::scoped_lock lock{sharp_mutex};
        // formats: one line per engine format with the pairs seen (more than one pair means the T# format is not a function of the engine's)
        std::map<std::string, std::string> formats;
        for (const auto& [key, count] : sharp_counts) {
            const size_t arrow = key.find(" -> ");
            if (arrow != std::string::npos && (key.rfind("format ", 0) == 0 || key.rfind("swizzle ", 0) == 0 || key.rfind("tiling ", 0) == 0 || key.rfind("pow2pad ", 0) == 0)) {
                formats[key.substr(0, arrow)] += key.substr(arrow + 4) + " x" + std::to_string(count) + "  ";
            } else {
                census[key] = count;
            }
        }
        for (const auto& [engine, pairs] : formats) {
            census["format_map"][engine] = pairs;
        }
        out["e2_census"] = std::move(census);
    }
    out["e2_derive"] = Core::NativeImageDesc::Report();
    out["e2_targets"] = Core::NativeTargetDesc::Report();
    out["f2_store"] = Core::NativeContent::Report();
    out["sync_typed"] = Core::NativeSync::Typed::Report();
    out["note"] = "B1 native device table (cumulative); bindings: the texture cache's / buffer cache's answer against the engine object the record names";
    return out;
}

u64 PS4_SYSV_ABI ViewCreatedSrv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64) {
    return ViewCreated<Kind::ShaderResourceView>("NativeViewSrv", resource, desc, out, a3, a4, a5);
}
u64 PS4_SYSV_ABI ViewCreatedUav(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64) {
    return ViewCreated<Kind::UnorderedAccessView>("NativeViewUav", resource, desc, out, a3, a4, a5);
}
u64 PS4_SYSV_ABI ViewCreatedRtv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64) {
    return ViewCreated<Kind::RenderTargetView>("NativeViewRtv", resource, desc, out, a3, a4, a5);
}
u64 PS4_SYSV_ABI ViewCreatedDsv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64) {
    return ViewCreated<Kind::DepthStencilView>("NativeViewDsv", resource, desc, out, a3, a4, a5);
}

u64 PS4_SYSV_ABI DelayDelete(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64) {
    NoteDelete(object);
    // native-v3-writers: the game frees an object `3 - age` frames after its last stamp (age = frame - [object + 0x2c]), 0 frames when it was stamped this frame or more than 3 frames ago; the freed memory
    // is then reused by the next resource while the raster thread (up to 2 frames behind the game) can still read the old one (late reads with rewritten memory, 2026-11-04). TLG_DELETE_DELAY=N adds N frames.
    static const u32 extra = [] { const char* e = std::getenv("TLG_DELETE_DELAY"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }();
    static const u64 original = Core::GuestHooks::OriginalEntry("NativeDelayDelete");
    const u64 holder = Rd<u64>(Core::NativeDraw::ImageBase() + 0x6b03558 - 0x1000000, 0);
    if (holder == 0 || object == 0) {
        return 0;
    }
    const s64 frame = Rd<s64>(holder, 0x10240);
    const s64 stamp = static_cast<s64>(Rd<u32>(object, 0x2c));
    const s64 age = frame - stamp;
    delay_by_age[age < 0 ? 5 : age > 3 ? 4 : age].fetch_add(1, std::memory_order_relaxed);
    Core::NativeDraw::LifeDeleted(object, age < 0 || stamp == 0xffffffffll ? -1 : age);
    if (extra == 0 || original == 0 || Rd<u64>(object, 8) != 0) {
        return 0; // the original runs
    }
    reinterpret_cast<GuestFunction6>(original)(object, a1, a2, a3, a4, a5);
    auto* delay = reinterpret_cast<std::atomic<u32>*>(object + 0x28);
    delay->fetch_add(extra, std::memory_order_relaxed);
    delay_added.fetch_add(1, std::memory_order_relaxed);
    return 0x8000000000000000ull;
}

namespace {

struct DeferredFree {
    u64 heap, address;
    s64 due_frame;
    u64 ordinal;
};
std::mutex deferred_mutex;
std::deque<DeferredFree> deferred_frees;
constexpr size_t DeferredMax = 20000; // more pending than this: the oldest are released at once (the heap must not run dry)

s64 EngineFrame() {
    const u64 holder = Rd<u64>(Core::NativeDraw::ImageBase() + 0x6b03558 - 0x1000000, 0);
    return holder != 0 ? Rd<s64>(holder, 0x10240) : 0;
}

u32 DeferFrames() {
    static const u32 frames = [] { const char* e = std::getenv("TLG_DEFER_FREE"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }();
    return frames;
}

// Free hold (memory-ownership-plan 0b, TLG_FREE_HOLD=1): the game frees heap blocks by frame counts and never waits for the GPU; our raster thread reads guest memory later than the
// game's counts assume. Each Gomalloc free is held until the raster thread has finished the submission that could still read the block (the one announced after the free: draws
// recorded before the free in the open window belong to it), then the real free runs. No game thread waits; the block returns to the heap a few submissions later.
struct HeldFree {
    u64 heap, address, size;
    u64 need; // RasterDoneSubmissions() must reach this before the block is freed
    s64 due_frame; // TLG_FREE_HOLD_FRAMES: the engine frame from which on the block may go (0: no frame rule)
    bool armed;    // the frame rule passed and need was raised to the submissions recorded by then
};
/// TLG_FREE_HOLD_FRAMES=N: also keep a freed block N engine frames (the game's own rule: an object lives 3 frames after its last use, but never-stamped objects are freed at once
/// while draws the game built ahead still use them), then until the raster thread has finished everything recorded up to then
u32 FreeHoldFrames() {
    static const u32 n = [] { const char* e = std::getenv("TLG_FREE_HOLD_FRAMES"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }();
    return n;
}
std::deque<HeldFree> held_frees; // under deferred_mutex, in free order (need never decreases)
u64 held_bytes = 0;              // under deferred_mutex (known sizes only)
std::atomic<u64> hold_count{0}, hold_at_once{0}, hold_released{0}, hold_forced{0}, hold_max_pending{0}, hold_max_bytes{0}, hold_alloc_failed{0}, hold_alloc_failed_twice{0};

bool FreeHoldOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_FREE_HOLD"); return e != nullptr && e[0] == '1'; }();
    return on;
}

/// Frees the held blocks the raster thread is done with (all of them when `all`). Any guest thread.
void ReleaseHeld(bool all) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocFree");
    if (original == 0) {
        return;
    }
    const u64 done = Core::NativeContent::RasterDoneSubmissions();
    std::vector<HeldFree> due;
    {
        std::scoped_lock lock{deferred_mutex};
        const s64 frame = FreeHoldFrames() != 0 ? EngineFrame() : 0;
        while (!held_frees.empty()) {
            HeldFree& front = held_frees.front();
            if (!all && front.due_frame != 0 && !front.armed) {
                if (frame < front.due_frame) {
                    break;
                }
                front.need = std::max<u64>(front.need, u64{Core::NativeContent::SubmitCount()} + 1);
                front.armed = true;
            }
            if (!all && front.need > done) {
                break;
            }
            if (held_frees.front().need > done || (front.due_frame != 0 && !front.armed)) {
                hold_forced.fetch_add(1, std::memory_order_relaxed);
            }
            held_bytes -= held_frees.front().size;
            due.push_back(held_frees.front());
            held_frees.pop_front();
        }
    }
    for (const auto& entry : due) {
        reinterpret_cast<GuestFunction6>(original)(entry.heap, entry.address, 0, 0, 0, 0);
        hold_released.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

nlohmann::json FreeHoldReport() {
    u64 pending = 0, bytes = 0, oldest_need = 0;
    {
        std::scoped_lock lock{deferred_mutex};
        pending = held_frees.size();
        bytes = held_bytes;
        oldest_need = held_frees.empty() ? 0 : held_frees.front().need;
    }
    return {{"on", FreeHoldOn()},
            {"frees_held", hold_count.load()},
            {"frees_done_at_once_raster_already_past", hold_at_once.load()},
            {"held_frees_released", hold_released.load()},
            {"released_before_the_raster_thread_was_done", hold_forced.load()},
            {"pending_now", pending},
            {"pending_bytes_now", bytes},
            {"max_pending", hold_max_pending.load()},
            {"max_pending_bytes", hold_max_bytes.load()},
            {"oldest_pending_needs_submission", oldest_need},
            {"raster_done_floor", Core::NativeContent::RasterDoneSubmissions()},
            {"allocation_failed_with_frees_held", hold_alloc_failed.load()},
            {"allocation_failed_again_after_release", hold_alloc_failed_twice.load()}};
}

namespace {

std::mutex gate_mutex_w2;
struct FreedRange {
    u64 end;
    u64 submitted; // graphics submissions announced when the memory was freed
    u64 caller;    // W2 census: Ghidra address of the Gomalloc free's caller
    u32 thread;    // W2 census: interned name of the freeing thread
    bool tick;     // W2: the free ran inside JsSwapChain_BeginFrame (the game's delayed delete tick, render thread, before this frame's recording)
};
thread_local bool w2_in_begin_frame = false;
std::atomic<u64> w2_window_in_tick{0}, w2_window_not_tick{0}, w2_tick_waited{0};
bool W2TickOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_W2_TICK"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::vector<std::string> w2_thread_names; // under gate_mutex
u32 W2Thread() {
    const std::string name = Common::GetCurrentThreadName();
    std::scoped_lock lock{gate_mutex_w2};
    for (u32 i = 0; i < w2_thread_names.size(); ++i) {
        if (w2_thread_names[i] == name) {
            return i;
        }
    }
    w2_thread_names.push_back(name);
    return static_cast<u32>(w2_thread_names.size() - 1);
}
struct W2Key {
    u32 free_thread, alloc_thread;
    u64 free_caller, alloc_caller;
    u32 size_class; // log2 of the allocated block
    bool operator<(const W2Key& o) const { return std::tie(free_thread, alloc_thread, free_caller, alloc_caller, size_class) < std::tie(o.free_thread, o.alloc_thread, o.free_caller, o.alloc_caller, o.size_class); }
};
std::map<W2Key, u64> w2_window_cases; // allocations in the open window of the free (the case that cannot be waited for)
std::atomic<u64> w2_same_thread{0}, w2_other_thread{0};
u64 W2Caller(u64 ret) {
    const VAddr base = Core::NativeDraw::ImageBase();
    return base != 0 ? ret - base + 0x1000000 : ret;
}
std::mutex gate_mutex;
std::map<u64, FreedRange> gate_freed;
std::unordered_set<u64> gate_unsized; // freed addresses whose size was unknown
std::unordered_map<u64, u64> gate_block_size; // live engine heap blocks: address -> size (every Gomalloc allocation, while the gate is on)

bool LoaderGateOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_LOADER_GATE"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void GateNoteFree(u64 address, u64 ret) {
    u64 size = 0;
    {
        std::scoped_lock lock{gate_mutex};
        const auto it = gate_block_size.find(address);
        if (it != gate_block_size.end()) {
            size = it->second;
            gate_block_size.erase(it);
        }
    }
    if (size == 0) {
        std::scoped_lock lock{gate_mutex};
        if (gate_unsized.size() > 100000) {
            gate_unsized.clear();
        }
        gate_unsized.insert(address);
        return;
    }
    const u64 submitted = Core::NativeContent::SubmitCount();
    const u64 reached = Core::NativeContent::RasterDoneSubmissions();
    std::scoped_lock lock{gate_mutex};
    if (gate_freed.size() > 100000) {
        for (auto it = gate_freed.begin(); it != gate_freed.end();) {
            it = it->second.submitted <= reached ? gate_freed.erase(it) : std::next(it);
        }
    }
    gate_freed[address] = {address + size, submitted, W2Caller(ret), W2Thread(), w2_in_begin_frame};
    gate_frees.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

nlohmann::json W2Report() {
    // W2 census: the allocations that landed in the open window of the free (cannot be waited for), by freeing thread, allocating thread and callers (Ghidra addresses)
    nlohmann::json cases = nlohmann::json::array();
    std::scoped_lock lock{gate_mutex_w2};
    std::vector<std::pair<u64, W2Key>> sorted;
    for (const auto& [key, count] : w2_window_cases) {
        sorted.emplace_back(count, key);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < sorted.size() && i < 40; ++i) {
        const W2Key& k = sorted[i].second;
        cases.push_back({{"count", sorted[i].first}, {"free_thread", k.free_thread < w2_thread_names.size() ? w2_thread_names[k.free_thread] : "?"}, {"alloc_thread", k.alloc_thread < w2_thread_names.size() ? w2_thread_names[k.alloc_thread] : "?"},
                         {"free_caller", fmt::format("{:#x}", k.free_caller)}, {"alloc_caller", fmt::format("{:#x}", k.alloc_caller)}, {"block_log2", k.size_class}});
    }
    return {{"window_allocs_after_a_free_in_the_delete_tick", w2_window_in_tick.load()}, {"window_allocs_after_a_free_outside_the_tick", w2_window_not_tick.load()}, {"tick_switch_on", W2TickOn()}, {"tick_cases_made_waitable", w2_tick_waited.load()}, {"same_thread", w2_same_thread.load()}, {"other_thread", w2_other_thread.load()}, {"cases_top40", std::move(cases)}};
}

void LoaderGateAtAllocation(u64 address, u64 size, u64 ret) {
    if (!LoaderGateOn() || address == 0 || size == 0) {
        return;
    }
    gate_checks.fetch_add(1, std::memory_order_relaxed);
    const u64 end = address + size;
    u64 need = 0;
    FreedRange newest_free{};
    {
        std::scoped_lock lock{gate_mutex};
        auto it = gate_freed.lower_bound(address);
        if (it != gate_freed.begin()) {
            const auto before = std::prev(it);
            if (before->second.end > address) {
                it = before;
            }
        }
        while (it != gate_freed.end() && it->first < end) {
            if (it->second.end > address) {
                if (it->second.submitted >= need) {
                    newest_free = it->second;
                }
                need = std::max(need, it->second.submitted);
                it = gate_freed.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (need == 0) {
        const u32 freed_at = Core::NativeContent::HeapFreeAt(address);
        if (freed_at != 0 && freed_at > Core::NativeContent::RasterDoneSubmissions()) {
            gate_missed.fetch_add(1, std::memory_order_relaxed);
            std::scoped_lock lock{gate_mutex};
            if (gate_unsized.contains(address)) {
                gate_missed_unsized.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (need != 0) {
        // the free was announced at submission count `need`: draws recorded before it in that window belong to submission need + 1 (a record's submission is the first announced after it), so that one has to be
        // done too. When it is not announced yet (count still `need`) nothing can be waited for: the loader writes before the draws leave (counted)
        const u64 count = Core::NativeContent::SubmitCount();
        if (need < count) {
            need += 1;
        } else {
            if (newest_free.tick) {
                w2_window_in_tick.fetch_add(1, std::memory_order_relaxed);
                if (W2TickOn()) {
                    // The free ran in the delete tick: the render thread was between two frames, so every draw that read the old contents was recorded before the tick and belongs to a submission
                    // announced by then (`need`): that one can be waited for (no "+1").
                    w2_tick_waited.fetch_add(1, std::memory_order_relaxed);
                    goto window_done;
                }
            } else {
                w2_window_not_tick.fetch_add(1, std::memory_order_relaxed);
            }
            gate_window_open.fetch_add(1, std::memory_order_relaxed);
            const u32 alloc_thread = W2Thread();
            (alloc_thread == newest_free.thread ? w2_same_thread : w2_other_thread).fetch_add(1, std::memory_order_relaxed);
            u32 size_class = 0;
            while ((u64{1} << (size_class + 1)) <= size) {
                ++size_class;
            }
            std::scoped_lock lock{gate_mutex_w2};
            if (w2_window_cases.size() < 4000) {
                ++w2_window_cases[{newest_free.thread, alloc_thread, newest_free.caller, W2Caller(ret), size_class}];
            }
        }
    window_done:;
    }
    if (need != 0 && Core::NativeContent::RasterDoneSubmissions() < need) {
        gate_hits.fetch_add(1, std::memory_order_relaxed);
        gate_waited_frames.fetch_add(need - Core::NativeContent::RasterDoneSubmissions(), std::memory_order_relaxed);
        Core::NativeContent::WaitRasterSubmissions(static_cast<u32>(need));
    }
}

void SceneGroupAdvanced(u32 new_group) {
    static const int mode = [] { const char* e = std::getenv("TLG_SLOT_GATE"); return e != nullptr ? std::atoi(e) : 0; }();
    const bool on = mode >= 1;
    static u32 left_at[3]{};
    static u32 previous = ~0u;
    if (!on || new_group >= 3) {
        return;
    }
    slot_advances.fetch_add(1, std::memory_order_relaxed);
    const u32 count = Core::NativeContent::SubmitCount();
    if (previous < 3) {
        left_at[previous] = count; // draws recorded so far with the group we leave belong to submissions up to count + 1
    }
    previous = new_group;
    // the group we come back to was left at left_at[new_group]: its draws sit in submissions up to left_at + 1 (the one being built cannot be waited for)
    // mode 2 (experiment): wait for everything submitted so far, to tell a weak formula from a late read counter that is wrong
    // mode 1: the draws of the group we come back to may sit in a submission announced after the group was left (the frame is submitted at its end): the count at which the FOLLOWING group was left
    // is past all of them
    const u32 following = left_at[(new_group + 1) % 3];
    const u32 need = mode >= 2 ? count : std::min<u32>(following, count);
    if ((mode >= 2 || following != 0) && need > Core::NativeContent::RasterDoneSubmissions()) {
        slot_waits.fetch_add(1, std::memory_order_relaxed);
        slot_waited_submissions.fetch_add(need - Core::NativeContent::RasterDoneSubmissions(), std::memory_order_relaxed);
        Core::NativeContent::WaitRasterSubmissions(need);
    }
}

void SkinSlotAboutToWrite(u32 index) {
    static const int mode = [] { const char* e = std::getenv("TLG_SKIN_GATE"); return e != nullptr ? std::atoi(e) : 0; }();
    static u32 written_at[4]{}; // SubmitCount() when the slot was last written (0 = never)
    if (mode < 1 || index >= 4) {
        return;
    }
    skin_writes.fetch_add(1, std::memory_order_relaxed);
    const u32 count = Core::NativeContent::SubmitCount();
    // The ring has 4 slots and two passes per frame advance it, so slot `index` is written again two frames after its last write. The draws and dispatches that read it sit in submissions
    // announced by the time the following frame began, which is when the slot two passes after `index` was written: wait until the raster thread has finished those.
    const u32 following = written_at[(index + 2) & 3];
    const u32 need = std::min<u32>(following, count);
    if (following != 0 && need > Core::NativeContent::RasterDoneSubmissions()) {
        skin_waits.fetch_add(1, std::memory_order_relaxed);
        skin_waited_submissions.fetch_add(need - Core::NativeContent::RasterDoneSubmissions(), std::memory_order_relaxed);
        Core::NativeContent::WaitRasterSubmissions(need);
    }
    written_at[index] = count;
}

/// JsSwapChain_BeginFrame (0x2495480): runs the game's delayed delete tick; the Gomalloc frees inside it carry the tick flag (W2).
u64 PS4_SYSV_ABI BeginFrameTick(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64) {
    static const u64 original = Core::GuestHooks::OriginalEntry("B0_SwapChainBeginFrame");
    if (original == 0) {
        return 0;
    }
    w2_in_begin_frame = true;
    const u64 result = reinterpret_cast<GuestFunction6>(original)(a0, a1, a2, a3, a4, a5);
    w2_in_begin_frame = false;
    if (FreeHoldOn()) {
        ReleaseHeld(false);
    }
    return 0x8000000000000000ull | (result & 0x7fffffffffffffffull);
}

u64 PS4_SYSV_ABI GomallocAlloc(u64 heap, u64 size, u64 alignment, u64 type, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocAlloc");
    const bool hold = FreeHoldOn();
    if ((!LoaderGateOn() && !hold) || original == 0) {
        return 0; // the original runs
    }
    if (hold) {
        ReleaseHeld(false);
    }
    u64 result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
    if (result == 0 && hold) {
        // the heap ran dry while blocks are held: wait for the raster thread to pass the newest held free (2 s limit), release, retry; then release everything and retry once more
        u64 newest = 0;
        {
            std::scoped_lock lock{deferred_mutex};
            newest = held_frees.empty() ? 0 : held_frees.back().need;
        }
        newest = std::min<u64>(newest, Core::NativeContent::SubmitCount()); // never wait for a submission not announced yet (this thread may be the one that announces it)
        if (newest != 0) {
            hold_alloc_failed.fetch_add(1, std::memory_order_relaxed);
            Core::NativeContent::WaitRasterSubmissions(static_cast<u32>(newest));
            ReleaseHeld(false);
            result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
            if (result == 0) {
                hold_alloc_failed_twice.fetch_add(1, std::memory_order_relaxed);
                ReleaseHeld(true);
                result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
            }
        }
    }
    if (hold && result != 0) {
        const u64 block = size == 0 ? 0x40 : static_cast<u32>(static_cast<u32>(size) + 0x3f) & 0xffffffc0u;
        std::scoped_lock lock{gate_mutex};
        gate_block_size[result] = block;
    }
    if (result != 0 && LoaderGateOn()) {
        const u64 block = size == 0 ? 0x40 : static_cast<u32>(static_cast<u32>(size) + 0x3f) & 0xffffffc0u; // the allocator rounds up to 64 bytes
        {
            std::scoped_lock lock{gate_mutex};
            gate_block_size[result] = block;
        }
        LoaderGateAtAllocation(result, block, ret);
    }
    return 0x8000000000000000ull | (result & 0x7fffffffffffffffull);
}

u64 PS4_SYSV_ABI GomallocFree(u64 heap, u64 address, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocFree");
    const u32 frames = DeferFrames();
    if (address != 0) {
        Core::NativeContent::NoteHeapFree(address);
        if (LoaderGateOn()) {
            GateNoteFree(address, ret);
        }
    }
    if (FreeHoldOn() && original != 0 && heap != 0 && address != 0) {
        u64 size = 0;
        if (!LoaderGateOn()) { // with the gate on, GateNoteFree took the size already
            std::scoped_lock lock{gate_mutex};
            const auto it = gate_block_size.find(address);
            if (it != gate_block_size.end()) {
                size = it->second;
                gate_block_size.erase(it);
            }
        }
        const u64 need = Core::NativeContent::SubmitCount() + 1;
        if (FreeHoldFrames() == 0 && Core::NativeContent::RasterDoneSubmissions() >= need) {
            hold_at_once.fetch_add(1, std::memory_order_relaxed);
            return 0; // the original runs
        }
        std::scoped_lock lock{deferred_mutex};
        held_frees.push_back({heap, address, size, need, FreeHoldFrames() != 0 ? EngineFrame() + FreeHoldFrames() : 0, false});
        held_bytes += size;
        hold_count.fetch_add(1, std::memory_order_relaxed);
        hold_max_pending.store(std::max<u64>(hold_max_pending.load(), held_frees.size()), std::memory_order_relaxed);
        hold_max_bytes.store(std::max<u64>(hold_max_bytes.load(), held_bytes), std::memory_order_relaxed);
        return 0x8000000000000000ull; // the free runs later (ReleaseHeld)
    }
    if (frames == 0 || original == 0 || heap == 0 || address == 0) {
        return 0; // the original runs
    }
    {
        // TLG_DEFER_HEAP=garlic|onion|other: only the frees of that heap are held (default all); the device keeps the garlic heap at +0x100b8 and the onion heap at +0x100e8
        static const std::string only = [] { const char* e = std::getenv("TLG_DEFER_HEAP"); return std::string(e != nullptr ? e : ""); }();
        if (!only.empty()) {
            const u64 holder = Rd<u64>(Core::NativeDraw::ImageBase() + 0x6b03558 - 0x1000000, 0);
            const char* kind = holder == 0 ? "other" : heap == Rd<u64>(holder, 0x100b8) ? "garlic" : heap == Rd<u64>(holder, 0x100e8) ? "onion" : "other";
            if (only != kind) {
                return 0;
            }
        }
    }
    std::vector<DeferredFree> forced;
    {
        std::scoped_lock lock{deferred_mutex};
        deferred_frees.push_back({heap, address, EngineFrame() + frames, deferred_count.fetch_add(1, std::memory_order_relaxed)});
        deferred_max_pending.store(std::max<u64>(deferred_max_pending.load(), deferred_frees.size()), std::memory_order_relaxed);
        while (deferred_frees.size() > DeferredMax) {
            forced.push_back(deferred_frees.front());
            deferred_frees.pop_front();
        }
    }
    for (const auto& entry : forced) {
        reinterpret_cast<GuestFunction6>(original)(entry.heap, entry.address, 0, 0, 0, 0);
        deferred_forced.fetch_add(1, std::memory_order_relaxed);
    }
    (void)a2, (void)a3, (void)a4, (void)a5;
    return 0x8000000000000000ull; // the free is skipped (done later)
}

void DrainDeferredFrees() {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocFree");
    if (FreeHoldOn()) {
        ReleaseHeld(false);
    }
    if (DeferFrames() == 0 || original == 0) {
        return;
    }
    const s64 frame = EngineFrame();
    std::vector<DeferredFree> due;
    {
        std::scoped_lock lock{deferred_mutex};
        while (!deferred_frees.empty() && deferred_frees.front().due_frame <= frame) {
            due.push_back(deferred_frees.front());
            deferred_frees.pop_front();
        }
    }
    for (const auto& entry : due) {
        reinterpret_cast<GuestFunction6>(original)(entry.heap, entry.address, 0, 0, 0, 0);
        deferred_released.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace Core::NativeDevice
