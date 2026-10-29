// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "common/logging/log.h"
#include "core/guest_hooks.h"
#include "core/native_device.h"
#include "core/native_resources.h"
#include "core/native_updates.h"

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
std::map<VAddr, u64> resource_by_address; ///< live resources by their memory (address -> id)

std::atomic<u64> resources_noted{0}, views_noted{0}, deletes_noted{0}, deletes_unknown{0}, bad_reads{0};
std::array<std::atomic<u64>, 9> live_by_kind{};
/// B1 shadow check counters: images [0 same memory and size, 1 same memory other size, 2 other memory, 3 id unknown], buffers [0 inside the object, 1 outside, 2 id unknown]
std::array<std::atomic<u64>, 4> image_checks{};
std::array<std::atomic<u64>, 3> buffer_checks{};
std::atomic<u32> mismatches_logged{0}, outside_logged{0};
std::array<std::array<std::atomic<u64>, 4>, 2> outside_by{};

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
        out.width = d(0), out.usage = d(1), out.bind = d(2), out.cpu_access = d(3), out.misc = d(4);
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
        Core::NativeUpdates::NoteEngineBufferCreate(object.id, object.address, std::max<u64>(object.bytes, 1)); // B9.4
    }
    if (const auto it = by_id.find(object.id); it != by_id.end()) {
        if (it->second.alive) {
            live_by_kind[static_cast<u32>(it->second.kind)].fetch_sub(1, std::memory_order_relaxed);
        }
        id_by_object.erase(it->second.object);
    }
    by_id[object.id] = object;
    id_by_object[object.object] = object.id;
    if (object.kind >= Kind::Buffer && object.kind <= Kind::Texture3D && object.address != 0) {
        resource_by_address[object.address] = object.id;
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
            NoteView(ViewKind, view);
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
    Insert(entry);
}

void NoteView(Kind kind, u64 view_object) {
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
    view.address = resource.address;
    view.bytes = resource.bytes;
    view.width = resource.width, view.height = resource.height, view.depth = resource.depth, view.mips = resource.mips, view.layers = resource.layers;
    view.format = resource.format, view.usage = resource.usage, view.bind = resource.bind, view.misc = resource.misc;
    views_noted.fetch_add(1, std::memory_order_relaxed);
    Insert(view);
    {
        // the resource itself is known from here on as well (textures are read once their initialization returned; their view creation comes after)
        std::shared_lock lock{table_mutex};
        if (by_id.contains(resource.id)) {
            return;
        }
    }
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
        const bool texture = gone.kind >= Kind::Texture1D && gone.kind <= Kind::Texture3D;
        if (texture && gone.address != 0) { // (a texture whose size read 0 at creation still owns images: its lists go too)
            // B8.2: another live texture at the same memory (an alias) keeps the images (they are not freed; the deleted texture's lists still go)
            const auto at = resource_by_address.find(gone.address);
            const auto owner = at != resource_by_address.end() && at->second != gone.id ? by_id.find(at->second) : by_id.end();
            const bool alias = owner != by_id.end() && owner->second.alive && owner->second.kind >= Kind::Texture1D && owner->second.kind <= Kind::Texture3D;
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
    while (it != resource_by_address.begin()) {
        --it;
        const auto entry = by_id.find(it->second);
        if (entry != by_id.end() && entry->second.alive && address < entry->second.address + std::max<u64>(entry->second.bytes, 1)) {
            out = entry->second;
            return true;
        }
        if (entry != by_id.end() && entry->second.alive) {
            return false; // the nearest live resource below ends before the address
        }
    }
    return false;
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
            if (it->first < address) {
                ++below;
            }
            const auto entry = by_id.find(it->second);
            if (entry == by_id.end() || !entry->second.alive || entry->second.kind < Kind::Texture1D || entry->second.kind > Kind::Texture3D ||
                entry->second.address != it->first || it->first + entry->second.bytes <= address) {
                continue;
            }
            out.push_back(entry->second.id);
            if (num_spans < spans.size()) {
                spans[num_spans++] = {std::max(it->first, address), std::min(it->first + entry->second.bytes, end)};
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
        if (it->first < address) {
            ++below;
        }
        const auto entry = by_id.find(it->second);
        if (entry == by_id.end() || !entry->second.alive || entry->second.kind != Kind::Buffer || entry->second.address != it->first ||
            it->first + std::max<u64>(entry->second.bytes, 1) <= address) {
            continue;
        }
        out.push_back({entry->second.id, std::max(it->first, address), std::min<VAddr>(it->first + std::max<u64>(entry->second.bytes, 1), end), entry->second.usage,
                       entry->second.cpu_access});
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

bool Lookup(u64 id, Object& out) {
    std::shared_lock lock{table_mutex};
    const auto it = by_id.find(id);
    if (it == by_id.end() || !it->second.alive) {
        return false;
    }
    out = it->second;
    return true;
}

void CheckImage(u64 id, VAddr address, u32 width, u32 height) {
    if (!Enabled() || id == 0) {
        return;
    }
    Object object;
    if (!Lookup(id, object)) {
        image_checks[3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 result = object.address != address ? 2u : (object.width == width && object.height == height) ? 0u : 1u;
    image_checks[result].fetch_add(1, std::memory_order_relaxed);
    if (result == 2 && mismatches_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
        LOG_WARNING(Core, "B1 native device: binding of {} id {} names memory {:#x} {}x{}, the texture cache bound {:#x} {}x{}", KindName(object.kind), id, object.address,
                    object.width, object.height, address, width, height);
    }
}

void CheckBuffer(u64 id, VAddr address, u64 size) {
    if (!Enabled() || id == 0) {
        return;
    }
    Object object;
    if (!Lookup(id, object)) {
        buffer_checks[2].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool inside = address >= object.address && address + size <= object.address + object.bytes;
    buffer_checks[inside ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
    if (!inside) {
        // where the bytes are instead: 0 the frame allocator (a Map discard), 1 elsewhere; by usage (0 default, 1 immutable, 2 dynamic, 3 staging)
        u64 ring_base = 0, ring_size = 0;
        const bool ring = Core::NativeResources::FrameRing(&ring_base, &ring_size) && address >= ring_base && address < ring_base + ring_size;
        outside_by[ring ? 0 : 1][std::min<u32>(object.usage, 3)].fetch_add(1, std::memory_order_relaxed);
        if (!ring && outside_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
            LOG_WARNING(Core, "B1 native device: buffer binding of {} id {} (usage {}, bind {:#x}, memory {:#x}+{:#x}) reads {:#x}+{:#x}", KindName(object.kind), id, object.usage, object.bind,
                        object.address, object.bytes, address, size);
        }
    }
}

namespace {
/// [color, depth][0 engine texture with the same size, 1 engine texture of another size (a mip / slice / part), 2 inside an engine texture, 3 an engine buffer, 4 none]
std::array<std::array<std::atomic<u64>, 5>, 2> target_checks{};
std::atomic<u32> targets_logged{0};
} // namespace

void CheckTarget(bool depth, VAddr address, u32 width, u32 height) {
    if (!Enabled()) {
        return;
    }
    Object o;
    u32 result = 4;
    if (ResourceAt(address, o)) {
        result = o.kind == Kind::Buffer ? 3u : o.address != address ? 2u : (o.width == width && o.height == height) ? 0u : 1u;
    }
    target_checks[depth ? 1 : 0][result].fetch_add(1, std::memory_order_relaxed);
    if (result >= 3 && targets_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Core, "B4 census: {} target {:#x} {}x{} has no engine texture ({})", depth ? "depth" : "color", address, width, height, OwnerName(address));
    }
}

nlohmann::json Report() {
    nlohmann::json out;
    out["enabled"] = Enabled();
    out["resources_noted"] = resources_noted.load();
    out["views_noted"] = views_noted.load();
    out["deletes_noted"] = deletes_noted.load();
    out["deletes_of_objects_not_in_the_table"] = deletes_unknown.load();
    out["unreadable_objects"] = bad_reads.load();
    nlohmann::json live;
    for (u32 k = 1; k < live_by_kind.size(); ++k) {
        live[KindName(static_cast<Kind>(k))] = live_by_kind[k].load();
    }
    out["live"] = std::move(live);
    out["image_bindings"] = {{"same_memory_and_size", image_checks[0].load()}, {"same_memory_other_size", image_checks[1].load()},
                             {"OTHER_MEMORY", image_checks[2].load()}, {"id_not_in_the_table", image_checks[3].load()}};
    out["buffer_bindings"] = {{"inside_the_object", buffer_checks[0].load()}, {"OUTSIDE_THE_OBJECT", buffer_checks[1].load()}, {"id_not_in_the_table", buffer_checks[2].load()},
                              {"outside_in_the_frame_allocator_by_usage", {outside_by[0][0].load(), outside_by[0][1].load(), outside_by[0][2].load(), outside_by[0][3].load()}},
                              {"outside_elsewhere_by_usage", {outside_by[1][0].load(), outside_by[1][1].load(), outside_by[1][2].load(), outside_by[1][3].load()}}};
    static constexpr const char* target_names[5] = {"engine_texture_same_size", "engine_texture_other_size", "inside_an_engine_texture", "ENGINE_BUFFER", "NONE"};
    for (u32 d = 0; d < 2; ++d) {
        for (u32 r = 0; r < 5; ++r) {
            out[d ? "depth_targets" : "color_targets"][target_names[r]] = target_checks[d][r].load();
        }
    }
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

u64 PS4_SYSV_ABI DelayDelete(u64 object, u64, u64, u64, u64, u64, u64) {
    NoteDelete(object);
    return 0; // the original runs
}

} // namespace Core::NativeDevice
