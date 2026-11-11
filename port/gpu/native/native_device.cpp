// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <absl/container/btree_map.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <thread>
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

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_hook_owned.h"
#include "native/native_texture_version.h"
#include "common/thread.h"
#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_log.h"
#include "native/native_invariant.h"
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
absl::btree_map<VAddr, AddrEntry> resource_by_address; // a B-tree: the walks by memory (BuffersIn, TexturesIn, ResourceAt) were a cache miss per level of a node tree

std::atomic<u64> resources_noted{0}, views_noted{0}, deletes_noted{0}, deletes_unknown{0}, bad_reads{0};
/// note 114: textures at their creation
enum Tac : u32 {
    TacCreated,         ///< textures read after their creation returned
    TacBadRead,         ///< ... unreadable then
    TacAlreadyKnown,    ///< ... already in the table (the id seen before)
    TacInserted,        ///< real mode: noted in the table at creation
    TacLaterViewed,     ///< shadow: not in the table at creation, got a view later (the real mode notes them earlier)
    TacDeletedUnviewed, ///< shadow: deleted without ever getting a view (the real mode gives them an engine delete event)
    TacBufferOverlap,   ///< created over a live buffer's memory (the real mode excludes such a buffer from the hook-owned class, note 90)
    TacHookMember,      ///< ... over a hook-owned member
    TacReadDiffers,     ///< the creation read differs from the read at the first view (CREATION_READ_DIFFERS_MUST_BE_0; textures that are not dynamic)
    TacReadDiffersDyn,  ///< ... of a dynamic texture (its memory may move with a Map before its first view)
    TacReinit,          ///< note 116: the object was already in the table, alive, under another id (the engine initialized a live texture object again)
    TacCount_
};
std::array<std::atomic<u64>, TacCount_> tac{};
std::mutex tac_mutex;
std::unordered_map<u64, Object> tac_pending;       ///< shadow: id -> the creation read, until a view or the delete
std::unordered_map<u64, u64> tac_pending_by_object; ///< shadow: object -> id
/// AddDelayDeleteObject (native-v3-writers): the game's own delay (frames until the object is freed) by age of the object's stamp, and the extra delay TLG_DELETE_DELAY adds
std::atomic<u64> delay_by_age[6]{}, delay_added{0}, delay_requeued{0};
std::atomic<u64> deferred_count{0}, deferred_released{0}, deferred_forced{0}, deferred_max_pending{0};
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
    by_id[object.id].created_submit = Core::NativeContent::SubmitCount();
    id_by_object[object.object] = object.id;
    if (object.kind >= Kind::Buffer && object.kind <= Kind::Texture3D && object.address != 0) {
        const u64 bytes = object.kind == Kind::Buffer ? std::max<u64>(object.bytes, 1) : object.bytes;
        if (bytes != 0 && Core::TextureVersion::Mode() != 0) {
            // note 100: a texture is in the class when it is not dynamic and its memory holds no other live buffer or texture; a resource created over a class texture excludes it
            // (looked at before this resource's own entry replaces one at the same address)
            const VAddr end = object.address + bytes;
            bool overlap = false;
            auto it = resource_by_address.lower_bound(end);
            u32 below = 0;
            while (it != resource_by_address.begin() && below < 16) {
                --it;
                if (it->first < object.address) {
                    ++below;
                }
                if (it->second.id == object.id || it->second.end <= object.address || it->second.kind < Kind::Buffer || it->second.kind > Kind::Texture3D) {
                    continue;
                }
                overlap = true;
                if (it->second.kind != Kind::Buffer) {
                    Core::TextureVersion::Exclude(it->second.id);
                }
            }
            if (object.kind != Kind::Buffer) {
                Core::TextureVersion::Created(object.id, !overlap && object.usage != 2);
            }
        }
        resource_by_address[object.address] = AddrEntry{object.id, object.address + bytes, object.kind, object.usage, object.cpu_access};
        if (object.kind != Kind::Buffer && bytes != 0 && Core::HookOwned::Mode() != 0) {
            // note 90: a texture in a buffer's memory excludes the buffer from the hook-owned class for good (here, in game order, before any later write of it)
            const VAddr end = object.address + bytes;
            auto it = resource_by_address.lower_bound(end);
            u32 below = 0;
            while (it != resource_by_address.begin() && below < 16) {
                --it;
                if (it->first < object.address) {
                    ++below;
                }
                if (it->second.kind == Kind::Buffer && it->second.end > object.address) {
                    Core::HookOwned::ExcludeBuffer(it->second.id);
                }
            }
        }
    }
    live_by_kind[static_cast<u32>(object.kind)].fetch_add(1, std::memory_order_relaxed);
}

/// note 114 guard: the fields a creation read and a view-time read of the same texture must agree on
bool SameRead(const Object& a, const Object& b) {
    return a.id == b.id && a.object == b.object && a.kind == b.kind && a.address == b.address && a.bytes == b.bytes && a.width == b.width && a.height == b.height &&
           a.depth == b.depth && a.mips == b.mips && a.layers == b.layers && a.format == b.format && a.usage == b.usage && a.bind == b.bind && a.cpu_access == b.cpu_access;
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
    Insert(view);
    if (TextureAtCreationMode() != 0) {
        std::scoped_lock lock{tac_mutex};
        if (const auto it = tac_pending.find(resource.id); it != tac_pending.end()) {
            tac[TacLaterViewed].fetch_add(1, std::memory_order_relaxed);
            if (!SameRead(it->second, resource)) {
                tac[resource.usage == 2 ? TacReadDiffersDyn : TacReadDiffers].fetch_add(1, std::memory_order_relaxed);
            }
            tac_pending_by_object.erase(it->second.object);
            tac_pending.erase(it);
        }
    }
    {
        // the resource itself is known from here on as well (textures are read once their initialization returned; their view creation comes after; note 114: or at their creation)
        std::shared_lock lock{table_mutex};
        if (by_id.contains(resource.id)) {
            return;
        }
    }
    Insert(resource);
}

u32 TextureAtCreationMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_TEXTURE_AT_CREATION");
        if (!Enabled()) {
            return 0u;
        }
        if (e == nullptr || e[0] == 0) {
            return 0u; // default 0: parked 2026-11-11 (note 116: owner play clean, but one view table difference in the default check, cause not found); =1 shadow, =2 real
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

void NoteTextureCreated(u64 object) {
    const u32 mode = TextureAtCreationMode();
    if (mode == 0 || object == 0) {
        return;
    }
    Object entry;
    if (!ReadResourceObject(object, entry) || entry.kind < Kind::Texture1D || entry.kind > Kind::Texture3D) {
        tac[TacBadRead].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    tac[TacCreated].fetch_add(1, std::memory_order_relaxed);
    {
        std::shared_lock lock{table_mutex};
        if (const auto o = id_by_object.find(object); o != id_by_object.end() && o->second != entry.id) {
            if (const auto old = by_id.find(o->second); old != by_id.end() && old->second.alive) {
                tac[TacReinit].fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                    LOG_WARNING(Render_Vulkan, "texture at creation: object {:#x} initialized again: was id {} {:#x}+{:#x} {}x{} mips {} format {} usage {}, now id {} {:#x}+{:#x} {}x{} mips {} format {} usage {}",
                                object, old->second.id, old->second.address, old->second.bytes, old->second.width, old->second.height, old->second.mips, old->second.format,
                                old->second.usage, entry.id, entry.address, entry.bytes, entry.width, entry.height, entry.mips, entry.format, entry.usage);
                }
            }
        }
        if (by_id.contains(entry.id)) {
            tac[TacAlreadyKnown].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (entry.address != 0 && entry.bytes != 0) {
            // what the hook-owned exclusion of Insert would find (note 90)
            const VAddr end = entry.address + entry.bytes;
            auto it = resource_by_address.lower_bound(end);
            u32 below = 0;
            bool overlap = false;
            while (it != resource_by_address.begin() && below < 16) {
                --it;
                if (it->first < entry.address) {
                    ++below;
                }
                overlap |= it->second.kind == Kind::Buffer && it->second.end > entry.address;
            }
            if (overlap) {
                tac[TacBufferOverlap].fetch_add(1, std::memory_order_relaxed);
                if (Core::HookOwned::Mode() != 0 && Core::HookOwned::MemberIn(entry.address, entry.bytes)) {
                    tac[TacHookMember].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    {
        // both modes: the creation read waits for the first view (the guard) or the delete
        std::scoped_lock lock{tac_mutex};
        if (tac_pending.size() < 400000) {
            tac_pending[entry.id] = entry;
            tac_pending_by_object[entry.object] = entry.id;
        }
    }
    if (mode == 2) {
        resources_noted.fetch_add(1, std::memory_order_relaxed);
        tac[TacInserted].fetch_add(1, std::memory_order_relaxed);
        Insert(entry);
    }
}

void NoteDelete(u64 object) {
    if (!Enabled() || object == 0) {
        return;
    }
    if (TextureAtCreationMode() != 0) {
        std::scoped_lock lock{tac_mutex};
        if (const auto it = tac_pending_by_object.find(object); it != tac_pending_by_object.end()) {
            tac[TacDeletedUnviewed].fetch_add(1, std::memory_order_relaxed);
            tac_pending.erase(it->second);
            tac_pending_by_object.erase(it);
        }
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
        entry->second.deleted_submit = std::max<u32>(1u, Core::NativeContent::SubmitCount());
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
            Core::HookOwned::BufferGone(gone.id); // note 90
            Core::NativeUpdates::NoteEngineDelete(gone.id, gone.address, gone.bytes, 4u); // B9.2
        }
    }
    id_by_object.erase(it);
}

u64 OwnerEndAt(VAddr address, u32 submit) {
    u64 end = 0;
    u32 newest = 0;
    std::shared_lock lock{table_mutex};
    for (const auto& [id, o] : by_id) {
        if (o.kind != Kind::Buffer || o.address == 0 || o.created_submit > submit || (o.deleted_submit != 0 && o.deleted_submit <= submit)) {
            continue;
        }
        const u64 bytes = std::max<u64>(o.bytes, 1);
        if (address >= o.address && address < o.address + bytes && o.created_submit >= newest) {
            newest = o.created_submit;
            end = o.address + bytes;
        }
    }
    return end;
}

std::string OwnersAt(VAddr address, u64 size, u32 submit) {
    struct Hit {
        u64 id;
        u32 created;
        std::string text;
    };
    std::vector<Hit> hits;
    {
        std::shared_lock lock{table_mutex};
        for (const auto& [id, o] : by_id) {
            if (o.kind < Kind::Buffer || o.kind > Kind::Texture3D || o.address == 0 || o.created_submit > submit || (o.deleted_submit != 0 && o.deleted_submit <= submit)) {
                continue;
            }
            const u64 bytes = o.kind == Kind::Buffer ? std::max<u64>(o.bytes, 1) : o.bytes;
            if (address >= o.address + bytes || o.address >= address + size) {
                continue;
            }
            hits.push_back({id, o.created_submit, fmt::format("[id {} kind {} {:#x}+{:#x} usage {} bind {:#x} created {} {}]", id, static_cast<u32>(o.kind), static_cast<u64>(o.address), bytes, o.usage, o.bind,
                                                              o.created_submit, o.deleted_submit != 0 ? fmt::format("deleted {}", o.deleted_submit) : std::string("alive"))});
        }
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.created > b.created; });
    std::string out;
    for (size_t i = 0; i < hits.size() && i < 3; ++i) {
        out += hits[i].text;
    }
    return out.empty() ? std::string("none") : out;
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

bool BufferPartOf(u64 id, VAddr address, u64 size, BufferPart& out) {
    if (size == 0) {
        return false;
    }
    std::shared_lock lock{table_mutex};
    const auto it = by_id.find(id);
    if (it == by_id.end() || !it->second.alive || it->second.kind != Kind::Buffer || it->second.address == 0) {
        return false;
    }
    const Object& o = it->second;
    const VAddr begin = std::max<VAddr>(o.address, address);
    const VAddr end = std::min<VAddr>(o.address + std::max<u64>(o.bytes, 1), address + size);
    if (begin >= end) {
        return false;
    }
    out = {o.id, begin, end, o.usage, o.cpu_access, o.address, o.address + std::max<u64>(o.bytes, 1)};
    return true;
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

std::string MapsCovering(u64 address, u32 max) {
    std::string out;
    u32 n = 0;
    std::scoped_lock lock{map_mutex};
    for (const auto& [id, m] : last_map) {
        if (m.address <= address && address < m.address + std::max<u64>(m.size, 1)) {
            out += fmt::format(" [id {:#x} {:#x}+{:#x} type {}]", id, m.address, m.size, m.type);
            if (++n >= max) {
                break;
            }
        }
    }
    return out.empty() ? std::string{" none"} : out;
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
        out["free_hold"] = FreeHoldReport();
        out["deferred_frees"] = {{"held", deferred_count.load()}, {"released", deferred_released.load()}, {"forced", deferred_forced.load()}, {"max_pending", deferred_max_pending.load()}};
    }
    out["unreadable_objects"] = bad_reads.load();
    {
        const auto t = [](Tac k) { return tac[k].load(std::memory_order_relaxed); };
        u64 pending = 0;
        {
            std::scoped_lock lock{tac_mutex};
            pending = tac_pending.size();
        }
        out["texture_at_creation"] = {{"mode", TextureAtCreationMode()}, {"textures_read_at_creation", t(TacCreated)}, {"unreadable", t(TacBadRead)},
                                      {"already_known", t(TacAlreadyKnown)}, {"real_noted_at_creation", t(TacInserted)},
                                      {"later_viewed", t(TacLaterViewed)}, {"deleted_never_viewed", t(TacDeletedUnviewed)},
                                      {"live_never_viewed", pending}, {"created_over_a_buffer", t(TacBufferOverlap)},
                                      {"created_over_a_hook_owned_member", t(TacHookMember)}, {"CREATION_READ_DIFFERS_MUST_BE_0", t(TacReadDiffers)},
                                      {"creation_read_differs_dynamic", t(TacReadDiffersDyn)},
                                      {"live_object_initialized_again", t(TacReinit)}};
    }
    nlohmann::json live;
    for (u32 k = 1; k < live_by_kind.size(); ++k) {
        live[KindName(static_cast<Kind>(k))] = live_by_kind[k].load();
    }
    out["live"] = std::move(live);
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
    static const u32 n = [] { const char* e = std::getenv("TLG_FREE_HOLD_FRAMES"); return e != nullptr && e[0] != 0 ? static_cast<u32>(std::atoi(e)) : 3u; }(); // default 3 since 2026-11-11 (review notes A3; the play set's value), =0 no frame rule
    return n;
}
std::atomic<u64> hold_bucket_count[6], hold_bucket_bytes[6];
std::atomic<u64> early_fence_counts[6]; // fences, then label writes: landed at the parse, kept (compute work with a readable result not covered), kept (switch off or no free hold)
std::deque<HeldFree> held_frees; // under deferred_mutex, in free order (need never decreases)
u64 held_bytes = 0;              // under deferred_mutex (known sizes only)
std::atomic<u64> hold_freed_bytes_total{0}, hold_released_bytes_total{0}, hold_armed_early{0};
/// TLG_HOLD_ARM_ALL: see ReleaseHeld. On by default since 2026-11-07 (=0 turns it off): one block per call left 200 to 450 MB waiting and the game's Garlic heap ran dry after scene changes.
bool HoldArmAllOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_HOLD_ARM_ALL"); return e == nullptr || e[0] != '0'; }();
    return on;
}
std::atomic<u64> hold_unsafe_blocks{0}, hold_unsafe_bytes{0}, hold_open_waits{0}, hold_open_wait_ns{0}, hold_open_same_thread{0}, hold_open_gave_up{0};
/// TLG_HOLD_WAIT_OPEN (default on, =0 off; needs TLG_PROGRESS_WAITS=1): a dry heap whose held blocks were freed in the submission window still being built waits for the game to announce that submission
/// (when another thread does) and for the raster thread to finish it, instead of releasing those blocks unsafely.
bool HoldWaitOpenOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_HOLD_WAIT_OPEN"); return e == nullptr || e[0] != '0'; }(); // default on since 2026-11-06, =0 turns it off
    return on;
}
std::atomic<u64> hold_count{0}, hold_at_once{0}, hold_released{0}, hold_forced{0}, hold_max_pending{0}, hold_max_bytes{0}, hold_alloc_failed{0}, hold_alloc_failed_twice{0}, hold_mid_released{0};

/// TLG_FREE_HOLD_MID=1: when the heap runs dry, first free the held blocks the raster thread is done with even if their frame rule has not passed (before releasing everything)
bool FreeHoldMidOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_FREE_HOLD_MID"); return e != nullptr && e[0] == '1'; }();
    return on;
}

bool FreeHoldOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_FREE_HOLD"); return e == nullptr || e[0] != '0'; }(); // LEGACY(free_hold): default on since 2026-11-11 (review notes A3), =0 old path
    return on;
}

/// Frees the held blocks the raster thread is done with (all of them when `all`). Any guest thread.
void ReleaseHeld(bool all, bool ignore_frames = false) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocFree");
    if (original == 0) {
        return;
    }
    const u64 done = Core::NativeContent::RasterDoneSubmissions();
    std::vector<HeldFree> due;
    u64 early_blocks = 0, early_bytes = 0; // blocks handed back before the game's own frame rule said so (the heap was dry)
    s64 early_max_frames = 0;
    s64 early_frame = 0;
    {
        std::scoped_lock lock{deferred_mutex};
        const s64 frame = FreeHoldFrames() != 0 ? EngineFrame() : 0;
        if (HoldArmAllOn() && !all && !ignore_frames) {
            // TLG_HOLD_ARM_ALL=1: every block whose frame rule has passed is armed now, with the same need (the submission announced next). The loop below arms ONE block per call and then waits for a
            // submission to finish before it looks at the next one, so a burst of frees drained at one block per finished submission (hold trace 2026-11-06: 450 MB ready but held for 1000+ frames).
            const u64 armed_need = u64{Core::NativeContent::SubmitCount()} + 1;
            for (HeldFree& entry : held_frees) {
                if (entry.due_frame == 0 || entry.armed) {
                    continue;
                }
                if (frame < entry.due_frame) {
                    break;
                }
                entry.need = std::max<u64>(entry.need, armed_need);
                entry.armed = true;
                hold_armed_early.fetch_add(1, std::memory_order_relaxed);
            }
        }
        while (!held_frees.empty()) {
            HeldFree& front = held_frees.front();
            if (!all && !ignore_frames && front.due_frame != 0 && !front.armed) {
                if (frame < front.due_frame) {
                    break;
                }
                front.need = std::max<u64>(front.need, u64{Core::NativeContent::SubmitCount()} + 1);
                front.armed = true;
            }
            if (!all && front.need > done) {
                break;
            }
            if (front.need > done) {
                hold_unsafe_blocks.fetch_add(1, std::memory_order_relaxed);
                hold_unsafe_bytes.fetch_add(front.size, std::memory_order_relaxed);
            }
            if (held_frees.front().need > done || (front.due_frame != 0 && !front.armed && !ignore_frames)) {
                hold_forced.fetch_add(1, std::memory_order_relaxed);
            } else if (ignore_frames && front.due_frame != 0 && !front.armed) {
                hold_mid_released.fetch_add(1, std::memory_order_relaxed);
                ++early_blocks;
                early_bytes += front.size;
                early_frame = frame;
                early_max_frames = std::max<s64>(early_max_frames, front.due_frame - frame);
            }
            held_bytes -= held_frees.front().size;
            due.push_back(held_frees.front());
            held_frees.pop_front();
        }
    }
    if (early_blocks != 0) {
        // 2026-11-06 slab / white blob hunt: the heap ran dry and the game's frame rule was dropped. Compare the time of this line with the time of a glitch (auto freeze line).
        TLG_LOG(GPU, Warn, "free hold: heap dry, {} blocks / {} MB handed back before their frame rule (up to {} frames early, engine frame {}), raster done {}, submits {}", early_blocks,
                early_bytes >> 20, early_max_frames, early_frame, done, Core::NativeContent::SubmitCount());
    }
    for (const auto& entry : due) {
        reinterpret_cast<GuestFunction6>(original)(entry.heap, entry.address, 0, 0, 0, 0);
        hold_released.fetch_add(1, std::memory_order_relaxed);
        hold_released_bytes_total.fetch_add(entry.size, std::memory_order_relaxed);
    }
}

/// Pool statistics the game reads (hook `GomallocStats` = FUN_0246d6e0, read only): out[0] is the free bytes of the pool, `PrepareMaterial` goes down its memory-pressure path (texture
/// mip reduction, compute copies, a spin on a compute-ring fence) when that figure is <= 0x11a00000. Recorded per pool: the figure as the game gets it, the bytes our free hold keeps
/// back in that pool, and how often the figure is below the threshold with and without the held bytes added (frame-order design, free-hold-fix-design.md step 0).
constexpr u64 PoolPressureThreshold = 0x11a00000;
/// TLG_POOL_FIGURE_FIX=0 turns the correction off (default on; it only acts while the free hold holds bytes in that pool)
bool PoolFigureFixOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_POOL_FIGURE_FIX"); return e == nullptr || e[0] != '0'; }();
    return on;
}
struct PoolStatsSlot {
    std::atomic<u64> corrected_calls{0}, corrected_bytes_last{0};
    std::atomic<u64> pool{0}, calls{0}, free_min{~0ull}, free_max{0}, free_last{0}, held_last{0}, held_max{0}, held_unknown_last{0}, below_real{0}, below_corrected{0};
};
PoolStatsSlot pool_stats_slots[4];

PoolStatsSlot* PoolStatsSlotFor(u64 pool) {
    for (auto& slot : pool_stats_slots) {
        u64 expected = 0;
        if (slot.pool.load(std::memory_order_relaxed) == pool || slot.pool.compare_exchange_strong(expected, pool) || expected == pool) {
            return &slot;
        }
    }
    return nullptr;
}

} // namespace

u64 PS4_SYSV_ABI GomallocStats(u64 pool, u64 out, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocStats");
    if (original == 0 || pool == 0 || out == 0) {
        return 0; // the original runs
    }
    reinterpret_cast<GuestFunction6>(original)(pool, out, a2, a3, a4, a5);
    PoolStatsSlot* slot = PoolStatsSlotFor(pool);
    if (slot != nullptr) {
        const u64 real = Rd<u64>(out, 0);
        u64 held = 0, unknown = 0;
        if (FreeHoldOn()) {
            std::scoped_lock lock{deferred_mutex};
            for (const auto& entry : held_frees) {
                if (entry.heap == pool) {
                    held += entry.size;
                    unknown += entry.size == 0 ? 1 : 0;
                }
            }
        }
        if (held != 0 && PoolFigureFixOn()) {
            // the game must see the free figure of lockstep: a block it freed is free for it, only our raster thread still reads it
            *reinterpret_cast<u64*>(out) = real + held;
            slot->corrected_calls.fetch_add(1, std::memory_order_relaxed);
            slot->corrected_bytes_last.store(held, std::memory_order_relaxed);
        }
        slot->calls.fetch_add(1, std::memory_order_relaxed);
        slot->free_last.store(real, std::memory_order_relaxed);
        slot->held_last.store(held, std::memory_order_relaxed);
        slot->held_unknown_last.store(unknown, std::memory_order_relaxed);
        if (real < slot->free_min.load(std::memory_order_relaxed)) {
            slot->free_min.store(real, std::memory_order_relaxed);
        }
        if (real > slot->free_max.load(std::memory_order_relaxed)) {
            slot->free_max.store(real, std::memory_order_relaxed);
        }
        if (held > slot->held_max.load(std::memory_order_relaxed)) {
            slot->held_max.store(held, std::memory_order_relaxed);
        }
        if (real <= PoolPressureThreshold) {
            slot->below_real.fetch_add(1, std::memory_order_relaxed);
        }
        if (real + held <= PoolPressureThreshold) {
            slot->below_corrected.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return 0x8000000000000000ull;
}

/// Early compute fences (frame-order design V2): whether a fence of copy-only compute work may land at the parse. Needs TLG_FREE_HOLD=1 (the hold keeps what the game frees after the
/// fence until the raster thread has run the queued copy) and no uncovered compute work with a CPU readable result on the queue. TLG_ASC_EARLY_FENCE=0 turns it off.
bool EarlyFenceAllowed(bool covered, bool label_write) {
    static const bool env_on = [] { const char* e = std::getenv("TLG_ASC_EARLY_FENCE"); return e == nullptr || e[0] != '0'; }();
    const u32 base = label_write ? 3 : 0;
    if (!env_on || !FreeHoldOn()) {
        early_fence_counts[base + 2].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!covered) {
        early_fence_counts[base + 1].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    early_fence_counts[base].fetch_add(1, std::memory_order_relaxed);
    return true;
}

/// Read only probes of the two helpers `PrepareMaterial` calls (FUN_01f7e520 = texture mip reduction under memory pressure, FUN_01f7e850 = block move on the compute queue): how often each ran
/// and how often it did work (returned 1). Told apart: the pressure path (needs the pool figure) and the move path (needs a block in the real free list).
std::atomic<u64> material_probe[6]; // pressure calls, pressure with arg 0, pressure did work, move calls, move did work, moves in frames where pressure did nothing

u64 PS4_SYSV_ABI MaterialPressureProbe(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("MaterialPressure");
    if (original == 0) {
        return 0;
    }
    const u32 result = static_cast<u32>(reinterpret_cast<GuestFunction6>(original)(a0, a1, a2, a3, a4, a5));
    material_probe[0].fetch_add(1, std::memory_order_relaxed);
    material_probe[1].fetch_add(static_cast<u32>(a0) == 0 ? 1 : 0, std::memory_order_relaxed);
    material_probe[2].fetch_add(result != 0 ? 1 : 0, std::memory_order_relaxed);
    return 0x8000000000000000ull | result;
}

u64 PS4_SYSV_ABI MaterialMoveProbe(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("MaterialMove");
    if (original == 0) {
        return 0;
    }
    const u32 result = static_cast<u32>(reinterpret_cast<GuestFunction6>(original)(a0, a1, a2, a3, a4, a5));
    material_probe[3].fetch_add(1, std::memory_order_relaxed);
    material_probe[4].fetch_add(result != 0 ? 1 : 0, std::memory_order_relaxed);
    return 0x8000000000000000ull | result;
}

nlohmann::json FreeHoldReport() {
    nlohmann::json pools = nlohmann::json::array();
    for (auto& slot : pool_stats_slots) {
        if (slot.pool.load() == 0) {
            continue;
        }
        pools.push_back({{"pool", slot.pool.load()},
                         {"stats_calls", slot.calls.load()},
                         {"figure_fix_on", PoolFigureFixOn()},
                         {"calls_figure_raised_by_held_bytes", slot.corrected_calls.load()},
                         {"free_last", slot.free_last.load()},
                         {"free_min", slot.free_min.load()},
                         {"free_max", slot.free_max.load()},
                         {"held_last", slot.held_last.load()},
                         {"held_max", slot.held_max.load()},
                         {"held_blocks_unknown_size_last", slot.held_unknown_last.load()},
                         {"calls_at_or_below_threshold_as_the_game_got_it", slot.below_real.load()},
                         {"calls_at_or_below_threshold_with_held_added", slot.below_corrected.load()},
                         {"threshold", PoolPressureThreshold}});
    }
    u64 pending = 0, bytes = 0, oldest_need = 0;
    {
        std::scoped_lock lock{deferred_mutex};
        pending = held_frees.size();
        bytes = held_bytes;
        oldest_need = held_frees.empty() ? 0 : held_frees.front().need;
    }
    return {{"on", FreeHoldOn()},
            {"pool_stats", pools},
            {"compute_fences", {{"landed_at_the_parse", early_fence_counts[0].load()}, {"kept_for_the_raster_thread_uncovered_compute_work", early_fence_counts[1].load()},
                                {"kept_switch_off_or_no_free_hold", early_fence_counts[2].load()},
                                {"label_writes_landed_at_the_parse", early_fence_counts[3].load()}, {"label_writes_kept_uncovered_compute_work", early_fence_counts[4].load()},
                                {"label_writes_kept_switch_off_or_no_free_hold", early_fence_counts[5].load()}}},
            {"held_frees_by_size", {{"lt_64KB", {hold_bucket_count[0].load(), hold_bucket_bytes[0].load()}}, {"lt_1MB", {hold_bucket_count[1].load(), hold_bucket_bytes[1].load()}},
                                    {"lt_16MB", {hold_bucket_count[2].load(), hold_bucket_bytes[2].load()}}, {"lt_64MB", {hold_bucket_count[3].load(), hold_bucket_bytes[3].load()}},
                                    {"ge_64MB", {hold_bucket_count[4].load(), hold_bucket_bytes[4].load()}}, {"unknown_size", {hold_bucket_count[5].load(), hold_bucket_bytes[5].load()}},
                                    {"note", "[count, bytes] of every free the hold kept since the start"}}},
            {"material_probe", {{"pressure_helper_calls", material_probe[0].load()}, {"pressure_helper_calls_with_arg_0", material_probe[1].load()}, {"pressure_helper_did_work", material_probe[2].load()},
                                {"move_helper_calls", material_probe[3].load()}, {"move_helper_did_work", material_probe[4].load()}}},
            {"frees_held", hold_count.load()},
            {"frees_done_at_once_raster_already_past", hold_at_once.load()},
            {"held_frees_released", hold_released.load()},
            {"released_before_the_raster_thread_was_done", hold_forced.load()},
            {"released_with_the_raster_thread_not_done_blocks", hold_unsafe_blocks.load()},
            {"released_with_the_raster_thread_not_done_bytes", hold_unsafe_bytes.load()},
            {"open_window_wait_switch_on", HoldWaitOpenOn()},
            {"arm_all_switch_on", HoldArmAllOn()},
            {"blocks_armed_early_by_arm_all", hold_armed_early.load()},
            {"open_window_waits", hold_open_waits.load()},
            {"open_window_wait_ms", hold_open_wait_ns.load() / 1.0e6},
            {"open_window_same_thread_no_wait", hold_open_same_thread.load()},
            {"open_window_gave_up", hold_open_gave_up.load()},
            {"pending_now", pending},
            {"pending_bytes_now", bytes},
            {"max_pending", hold_max_pending.load()},
            {"max_pending_bytes", hold_max_bytes.load()},
            {"oldest_pending_needs_submission", oldest_need},
            {"raster_done_floor", Core::NativeContent::RasterDoneSubmissions()},
            {"allocation_failed_with_frees_held", hold_alloc_failed.load()},
            {"allocation_failed_again_after_release", hold_alloc_failed_twice.load()},
            {"mid_step_on", FreeHoldMidOn()},
            {"released_by_mid_step_frame_rule_skipped", hold_mid_released.load()}};
}

namespace {
std::mutex gate_mutex;
std::unordered_map<u64, u64> gate_block_size; // live engine heap blocks: address -> size (every Gomalloc allocation, for the free hold)
} // namespace


/// `TLG_HOLD_TRACE=1`: every 250 ms one line in user/log/free_hold.csv about the held frees and why they are still held: bytes still inside the game's 3 frame rule, bytes waiting for the raster
/// thread, bytes ready to go that nobody asked for yet (the release runs at the frame tick and at an allocation), and what was freed / released since the start.
void HoldTraceSample() {
    static const bool on = [] { const char* e = std::getenv("TLG_HOLD_TRACE"); return e != nullptr && e[0] == '1'; }();
    if (!on) {
        return;
    }
    static std::FILE* file = [] {
        std::FILE* f = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "free_hold.csv").string().c_str(), "w");
        if (f != nullptr) {
            std::fputs("t_s,engine_frame,pending_blocks,pending_mb,wait_frame_rule_mb,wait_arm_mb,wait_raster_mb,ready_mb,largest_pending_mb,oldest_age_frames,freed_mb_total,released_mb_total,raster_done,submit_count,pool_free_mb\n", f);
        }
        return f;
    }();
    static u64 last_ns = 0;
    const u64 now_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    if (file == nullptr || now_ns - last_ns < 250'000'000ull) {
        return;
    }
    static const u64 start_ns = now_ns;
    last_ns = now_ns;
    const s64 frame = EngineFrame();
    const u64 done = Core::NativeContent::RasterDoneSubmissions();
    u64 blocks = 0, bytes = 0, w_frame = 0, w_arm = 0, w_raster = 0, ready = 0, largest = 0;
    s64 oldest = 0;
    {
        std::scoped_lock lock{deferred_mutex};
        for (const HeldFree& h : held_frees) {
            ++blocks;
            bytes += h.size;
            largest = std::max(largest, h.size);
            const bool frame_ok = h.due_frame == 0 || frame >= h.due_frame;
            const bool raster_ok = h.need <= done;
            if (!frame_ok) {
                w_frame += h.size;
            } else if (h.due_frame != 0 && !h.armed) {
                w_arm += h.size; // frame rule passed, not armed yet: waits for ReleaseHeld to arm it (one per call without TLG_HOLD_ARM_ALL)
            } else if (!raster_ok) {
                w_raster += h.size;
            } else {
                ready += h.size;
            }
            if (h.due_frame != 0) {
                oldest = std::max<s64>(oldest, frame - (h.due_frame - static_cast<s64>(FreeHoldFrames())));
            }
        }
    }
    u64 pool_free = 0;
    for (auto& slot : pool_stats_slots) {
        if (slot.pool.load(std::memory_order_relaxed) != 0 && slot.free_max.load(std::memory_order_relaxed) > (1ull << 30)) {
            pool_free = slot.free_last.load(std::memory_order_relaxed);
        }
    }
    constexpr double MB = 1.0 / (1024.0 * 1024.0);
    std::fprintf(file, "%.2f,%lld,%llu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%lld,%.1f,%.1f,%llu,%u,%.1f\n", (now_ns - start_ns) / 1.0e9, static_cast<long long>(frame), static_cast<unsigned long long>(blocks),
                 bytes * MB, w_frame * MB, w_arm * MB, w_raster * MB, ready * MB, largest * MB, static_cast<long long>(oldest), hold_freed_bytes_total.load() * MB, hold_released_bytes_total.load() * MB,
                 static_cast<unsigned long long>(done), Core::NativeContent::SubmitCount(), pool_free * MB);
    std::fflush(file);
}

/// JsSwapChain_BeginFrame (0x2495480): runs the game's delayed delete tick, then releases the held frees the raster thread is done with.
u64 PS4_SYSV_ABI BeginFrameTick(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64) {
    static const u64 original = Core::GuestHooks::OriginalEntry("B0_SwapChainBeginFrame");
    if (original == 0) {
        return 0;
    }
    const u64 result = reinterpret_cast<GuestFunction6>(original)(a0, a1, a2, a3, a4, a5);
    if (FreeHoldOn()) {
        ReleaseHeld(false);
        HoldTraceSample();
    }
    return 0x8000000000000000ull | (result & 0x7fffffffffffffffull);
}

u64 PS4_SYSV_ABI GomallocAlloc(u64 heap, u64 size, u64 alignment, u64 type, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocAlloc");
    const bool hold = FreeHoldOn();
    if (!hold || original == 0) {
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
            if (Core::NativeContent::ProgressWaitsOn()) {
                // TLG_PROGRESS_WAITS=1: wait while the raster thread makes progress (no 2 s limit)
                if (!Core::NativeContent::WaitRasterSubmissionsProgress(static_cast<u32>(newest), true)) {
                    Core::Invariant::Hit(Core::Invariant::WaitGaveUp, newest, Core::NativeContent::RasterDoneSubmissions(), 1);
                }
            } else {
                Core::NativeContent::WaitRasterSubmissions(static_cast<u32>(newest));
            }
            ReleaseHeld(false);
            result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
            if (result == 0 && (FreeHoldMidOn() || Core::NativeContent::ProgressWaitsOn())) { // (the game thread is here: the engine frame cannot advance, so the frame rule is dropped, the raster rule stays)
                ReleaseHeld(false, true);
                result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
            }
            bool same_thread_blocked = false;
            const bool wait_open = HoldWaitOpenOn() && Core::NativeContent::ProgressWaitsOn();
            // other threads keep freeing while this one waits: repeat until the held blocks are gone or nothing more can be waited for
            for (int round = 0; wait_open && result == 0 && round < 6; ++round) {
                u64 open_need = 0;
                {
                    std::scoped_lock lock{deferred_mutex};
                    open_need = held_frees.empty() ? 0 : held_frees.back().need;
                }
                if (open_need > Core::NativeContent::RasterDoneSubmissions()) { // (announced since the wait above, or still open)
                    if (open_need > Core::NativeContent::SubmitCount() && Core::NativeContent::IsSubmitThread()) {
                        hold_open_same_thread.fetch_add(1, std::memory_order_relaxed); // this thread announces the submission: nothing to wait for
                        same_thread_blocked = true;
                        break;
                    } else {
                        const auto start = std::chrono::steady_clock::now();
                        u32 last = Core::NativeContent::SubmitCount();
                        auto changed_at = start;
                        bool announced = false;
                        for (;;) {
                            if (Core::NativeContent::SubmitCount() >= open_need) {
                                announced = true;
                                break;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            const auto now = std::chrono::steady_clock::now();
                            if (const u32 count = Core::NativeContent::SubmitCount(); count != last) {
                                last = count;
                                changed_at = now;
                            } else if (now - changed_at > std::chrono::seconds(10)) {
                                break;
                            }
                        }
                        if (announced && Core::NativeContent::WaitRasterSubmissionsProgress(static_cast<u32>(open_need), true)) {
                            ReleaseHeld(false, true);
                            result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
                        } else {
                            hold_open_gave_up.fetch_add(1, std::memory_order_relaxed);
                            Core::Invariant::Hit(Core::Invariant::WaitGaveUp, open_need, Core::NativeContent::RasterDoneSubmissions(), 2);
                            same_thread_blocked = true; // nothing more can be waited for: the old last resort applies
                            break;
                        }
                        hold_open_waits.fetch_add(1, std::memory_order_relaxed);
                        hold_open_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
                    }
                } else {
                    ReleaseHeld(false, true); // whatever became safe meanwhile
                    result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
                    break;
                }
            }
            if (result == 0) {
                hold_alloc_failed_twice.fetch_add(1, std::memory_order_relaxed);
                if (wait_open && !same_thread_blocked) {
                    ReleaseHeld(false, true); // everything that can go safely has gone: the heap is simply full (blocks freed meanwhile by other threads are not released unsafely)
                } else {
                    Core::Invariant::Hit(Core::Invariant::HoldReleasedAll, newest, Core::NativeContent::RasterDoneSubmissions(), 0);
                    ReleaseHeld(true);
                }
                result = reinterpret_cast<GuestFunction6>(original)(heap, size, alignment, type, a4, a5);
            }
        }
    }
    if (hold && result != 0) {
        const u64 block = size == 0 ? 0x40 : static_cast<u32>(static_cast<u32>(size) + 0x3f) & 0xffffffc0u;
        std::scoped_lock lock{gate_mutex};
        gate_block_size[result] = block;
    }
    return 0x8000000000000000ull | (result & 0x7fffffffffffffffull);
}

u64 PS4_SYSV_ABI GomallocFree(u64 heap, u64 address, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static const u64 original = Core::GuestHooks::OriginalEntry("GomallocFree");
    const u32 frames = DeferFrames();
    if (address != 0) {
        Core::NativeContent::NoteHeapFree(address);
    }
    if (FreeHoldOn() && original != 0 && heap != 0 && address != 0) {
        u64 size = 0;
        {
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
        hold_freed_bytes_total.fetch_add(size, std::memory_order_relaxed);
        {
            // size census of the held frees (what makes up the held bytes): < 64 KB, < 1 MB, < 16 MB, < 64 MB, >= 64 MB, unknown
            const u32 bucket = size == 0 ? 5 : size < (64u << 10) ? 0 : size < (1u << 20) ? 1 : size < (16u << 20) ? 2 : size < (64u << 20) ? 3 : 4;
            hold_bucket_count[bucket].fetch_add(1, std::memory_order_relaxed);
            hold_bucket_bytes[bucket].fetch_add(size, std::memory_order_relaxed);
        }
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
