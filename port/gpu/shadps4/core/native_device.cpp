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
/// live resources by their memory: what the walks by address need, kept inline (the raster thread walks this for every announcement and upload; a hash
/// lookup per entry was the hottest code of the raster thread, 2026-10-30 profile)
struct AddrEntry {
    u64 id;
    VAddr end;
    Kind kind;
    u32 usage, cpu_access;
};
std::map<VAddr, AddrEntry> resource_by_address;

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

bool Lookup(u64 id, Object& out) {
    std::shared_lock lock{table_mutex};
    const auto it = by_id.find(id);
    if (it == by_id.end() || !it->second.alive) {
        return false;
    }
    out = it->second;
    return true;
}

static bool ShadowChecksOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_SHADOW_CHECKS");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void CheckImage(u64 id, VAddr address, u32 width, u32 height) {
    if (!Enabled() || id == 0 || !ShadowChecksOn()) { // B1 shadow census: TLG_SHADOW_CHECKS=1 (off by default since 2026-10-30)
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
    if (!Enabled() || id == 0 || !ShadowChecksOn()) {
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
    if (!Enabled() || !ShadowChecksOn()) {
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
        LOG_WARNING(Core, "B11: ClearRenderTargetView rtv id {} resource id {} memory {:#x} surface {:#x} color {} {} {} {}", Rd<u64>(c->rsi, 0x10),
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
        LOG_WARNING(Core, "B11: ClearDepthStencilView dsv id {} resource id {} memory {:#x} surface {:#x} flags {} depth {} stencil {}", Rd<u64>(c->rsi, 0x10),
                    resource != 0 ? Rd<u64>(resource, 0x10) : 0, resource != 0 ? Rd<u64>(resource, 0x30) : 0, resource != 0 ? Rd<u64>(resource, 0x48) : 0, flags, depth,
                    static_cast<u32>(c->rcx & 0xff));
    }
    u64* ret_slot = reinterpret_cast<u64*>(c->rsp + 16); // the hook sits after push rbp; mov rbp, rsp; push r15 (the B0 count hook displaced those 6 bytes)
    f.ret = *ret_slot;
    *ret_slot = stub;
    return 0;
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
