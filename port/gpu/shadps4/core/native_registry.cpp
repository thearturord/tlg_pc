// SPDX-License-Identifier: GPL-2.0-or-later
// See native_registry.h. Layouts: documents/tlg/render-engine/native-resource-model.md.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <nlohmann/json.hpp>

#include "core/native_registry.h"
#include "core/native_resources.h"

namespace Core::NativeRegistry {

std::atomic<bool> enabled{[] {
    const char* value = std::getenv("TLG_REGISTRY");
    return value != nullptr && value[0] == '1';
}()};

namespace {

constexpr u64 RawBytes = 0x1c0;
constexpr const char* KindNames[static_cast<size_t>(Kind::NumKinds)] = {
    "unknown",      "buffer",        "texture1d",      "texture2d",     "texture3d", "srv",           "rtv",
    "dsv",          "uav",           "blend_state",    "raster_state",  "depth_state", "sampler_state", "input_layout",
    "vertex_shader", "pixel_shader", "compute_shader"};

struct Entry {
    u64 id{};
    u64 object{};
    Kind kind{};     ///< what the slot said
    Kind actual{};   ///< what the object says about itself (resources: the kind at +0x120), else the same as `kind`
    u32 tag{};       ///< u32 at object + 0x18
    u64 address{};   ///< GNM memory of a resource (object + 0x30), 0 otherwise
    u64 size{};      ///< buffers: ByteWidth
    u32 usage{};     ///< resources: D3D11 usage
    u32 format{}, width{}, height{}, mips{}, layers{};
    bool alive{true};
    std::array<u8, RawBytes> raw{};
};

std::shared_mutex registry_mutex;
std::unordered_map<u64, Entry> by_id;
std::unordered_map<u64, u64> id_by_object;
std::map<u64, u64> buffer_intervals;          ///< begin -> id (live buffers)
std::unordered_map<u64, u64> texture_by_start; ///< start address -> id (live textures)
std::array<u64, static_cast<size_t>(Kind::NumKinds)> seen_by_kind{};
u64 dead_objects{}, reused_addresses{}, bad_reads{};
std::map<std::pair<u8, u32>, u64> tags_by_hint; ///< (slot kind, tag) -> objects
struct Dump {
    Kind hint;
    Kind actual;
    u32 tag;
    u64 id;
    std::string hex;
};
std::vector<Dump> dumps;
std::map<std::pair<u8, u32>, bool> dumped;

thread_local u32 draw_class = 15;
thread_local u64 current_draw_seq = 0;
constexpr u32 DrawRing = 8192;
std::unique_ptr<DrawResources[]> draw_ring;
std::mutex draw_ring_mutex;
std::array<std::atomic<u64>, 16> explained_buffers{}, unexplained_registered_buffers{}, unexplained_unknown_buffers{}, no_draw_record_buffers{};
std::array<std::atomic<u64>, 16> explained_images{}, unexplained_images{}, no_draw_record_images{};
struct ClassCounters {
    std::atomic<u64> buffers{}, buffers_inside{}, images{}, images_found{};
};
std::array<ClassCounters, 16> by_class;
struct AuditCounters {
    std::atomic<u64> buffers{}, buffers_inside{}, buffers_partial{}, buffers_missing{}, buffers_missing_written{};
    std::atomic<u64> images{}, images_found{}, images_missing{}, images_in_buffer{}, images_in_texture_block{};
};
AuditCounters audit;
std::mutex sample_mutex;
struct Sample {
    u64 address, size;
    bool written;
    u32 draw_class{};
    u64 reg_address{}, reg_bytes{};
    u32 reg_usage{~0u};
    u32 draw_buffers{};
};
std::vector<Sample> unexplained_samples;
std::array<u32, 16> unexplained_per_class{};
std::vector<Sample> buffer_miss_samples, buffer_partial_samples, image_miss_samples;
struct MissCount {
    u64 count{};
    u64 size{};
    bool written{};
};
std::unordered_map<u64, MissCount> buffer_misses; ///< guarded by sample_mutex, capped
std::unordered_map<u64, MissCount> image_misses;  ///< same for images
std::array<std::atomic<u64>, 17> miss_by_usage_class{};  ///< buffers not registered, by the heap block class from the census (0 unknown)
std::array<std::atomic<u64>, 17> miss_written_by_usage_class{};

struct Cached {
    u64 object{};
    u64 id{};
};
thread_local std::array<Cached, 4096> seen_cache;

template <typename T>
T Read(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

bool Plausible(u64 pointer) {
    return pointer > 0x10000 && pointer < 0x7fffffffffffull && (pointer & 7) == 0;
}

Kind ResourceKind(u32 kind_field) {
    switch (kind_field) {
    case 1:
        return Kind::Buffer;
    case 2:
        return Kind::Texture1D;
    case 3:
        return Kind::Texture2D;
    case 4:
        return Kind::Texture3D;
    default:
        return Kind::Unknown;
    }
}

void Retire(Entry& e) {
    if (!e.alive) {
        return;
    }
    e.alive = false;
    ++dead_objects;
    if (e.actual == Kind::Buffer) {
        if (const auto it = buffer_intervals.find(e.address); it != buffer_intervals.end() && it->second == e.id) {
            buffer_intervals.erase(it);
        }
    } else if (e.actual == Kind::Texture1D || e.actual == Kind::Texture2D || e.actual == Kind::Texture3D) {
        if (const auto it = texture_by_start.find(e.address); it != texture_by_start.end() && it->second == e.id) {
            texture_by_start.erase(it);
        }
    }
    id_by_object.erase(e.object);
}

/// Called with the unique lock held, the object not known (or known under another id).
void Add(u64 object, Kind hint, u64 id) {
    if (const auto it = id_by_object.find(object); it != id_by_object.end()) {
        if (const auto old = by_id.find(it->second); old != by_id.end()) {
            ++reused_addresses;
            Retire(old->second);
        }
    }
    Entry e;
    e.id = id;
    e.object = object;
    e.kind = hint;
    e.tag = Read<u32>(object, 0x18);
    std::memcpy(e.raw.data(), reinterpret_cast<const void*>(object), RawBytes);
    e.actual = hint;
    const bool resource_slot = hint == Kind::Buffer || hint == Kind::Texture1D || hint == Kind::Texture2D || hint == Kind::Texture3D;
    const Kind own = ResourceKind(Read<u32>(object, 0x124));
    if (resource_slot && own != Kind::Unknown) {
        // JsResource: address +0x30, description from +0x180 (see native_resources.cpp ReadResource)
        e.actual = own;
        e.address = Read<u64>(object, 0x30);
        const auto dword = [&](u32 index) { return Read<u32>(object, 0x180 + index * 4); };
        switch (own) {
        case Kind::Buffer:
            e.size = dword(0);
            e.usage = dword(1);
            break;
        case Kind::Texture1D:
            e.width = dword(0);
            e.mips = dword(1);
            e.layers = dword(2);
            e.format = dword(3);
            e.usage = dword(4);
            break;
        case Kind::Texture2D:
            e.width = dword(0);
            e.height = dword(1);
            e.mips = dword(2);
            e.layers = dword(3);
            e.format = dword(4);
            e.usage = dword(7);
            break;
        case Kind::Texture3D:
            e.width = dword(0);
            e.height = dword(1);
            e.layers = dword(2);
            e.mips = dword(3);
            e.format = dword(4);
            e.usage = dword(5);
            break;
        default:
            break;
        }
        if (e.address != 0) {
            if (own == Kind::Buffer && e.size != 0) {
                buffer_intervals[e.address] = id;
            } else if (own != Kind::Buffer) {
                texture_by_start[e.address] = id;
            }
        }
    }
    ++seen_by_kind[static_cast<size_t>(e.actual)];
    ++tags_by_hint[{static_cast<u8>(hint), e.tag}];
    if (!dumped[{static_cast<u8>(hint), e.tag}] && dumps.size() < 80) {
        dumped[{static_cast<u8>(hint), e.tag}] = true;
        std::string hex;
        hex.reserve(RawBytes * 2);
        for (u64 i = 0; i < RawBytes; ++i) {
            hex += fmt::format("{:02x}", e.raw[i]);
        }
        dumps.push_back({hint, e.actual, e.tag, id, std::move(hex)});
    }
    id_by_object[object] = id;
    by_id[id] = std::move(e);
}

} // namespace

namespace {
std::mutex pending_mutex;
std::vector<u64> pending_textures;
std::atomic<u32> audit_calls{0};

void DrainTextures() {
    std::vector<u64> list;
    {
        std::scoped_lock lock{pending_mutex};
        list.swap(pending_textures);
    }
    for (const u64 object : list) {
        if (!Plausible(object)) {
            continue;
        }
        const Kind own = ResourceKind(Read<u32>(object, 0x124));
        if (own != Kind::Unknown && own != Kind::Buffer) {
            Observe(object, own);
        }
    }
}
} // namespace

void NoteBufferCreated(u64 object) {
    if (enabled.load(std::memory_order_relaxed) && Plausible(object)) {
        Observe(object, Kind::Buffer);
    }
}

void NoteTextureInitializing(u64 object) {
    if (enabled.load(std::memory_order_relaxed)) {
        std::scoped_lock lock{pending_mutex};
        if (pending_textures.size() < 65536) {
            pending_textures.push_back(object);
        }
    }
}

void NoteBufferDestroyed(u64 object) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::unique_lock lock{registry_mutex};
    if (const auto it = id_by_object.find(object); it != id_by_object.end()) {
        if (const auto e = by_id.find(it->second); e != by_id.end()) {
            Retire(e->second);
        }
    }
}

void Observe(u64 object, Kind hint) {
    if (!Plausible(object)) {
        return;
    }
    const u64 id = Read<u64>(object, 0x10);
    auto& slot = seen_cache[(object >> 4) & 4095];
    if (slot.object == object && slot.id == id) {
        return;
    }
    {
        std::shared_lock lock{registry_mutex};
        if (const auto it = by_id.find(id); it != by_id.end() && it->second.object == object && it->second.alive) {
            slot = {object, id};
            return;
        }
    }
    {
        std::unique_lock lock{registry_mutex};
        if (const auto it = by_id.find(id); it == by_id.end() || it->second.object != object || !it->second.alive) {
            Add(object, hint, id);
        }
    }
    slot = {object, id};
    // views point at the resource they look at (+0x30, the shader resource view: see native_draw.cpp ResourceDescriptor)
    if (hint == Kind::ShaderResourceView || hint == Kind::RenderTargetView || hint == Kind::DepthStencilView || hint == Kind::UnorderedAccessView) {
        const u64 resource = Read<u64>(object, 0x30);
        if (Plausible(resource)) {
            const Kind own = ResourceKind(Read<u32>(resource, 0x124));
            if (own != Kind::Unknown) {
                Observe(resource, own);
            }
        }
    }
}

void DrawResources::AddBuffer(u64 begin, u64 end) {
    for (u32 i = 0; i < buffer_count; ++i) {
        if (buffers[i].begin == begin && buffers[i].end == end) {
            return;
        }
    }
    if (buffer_count < MaxBuffers) {
        buffers[buffer_count++] = {begin, end};
    } else {
        ++dropped;
    }
}

void DrawResources::AddImage(u64 start) {
    for (u32 i = 0; i < image_count; ++i) {
        if (images[i] == start) {
            return;
        }
    }
    if (image_count < MaxImages) {
        images[image_count++] = start;
    } else {
        ++dropped;
    }
}

void CollectResource(DrawResources& draw, u64 object) {
    if (!Plausible(object)) {
        return;
    }
    const Kind own = ResourceKind(Read<u32>(object, 0x124));
    const u64 address = Read<u64>(object, 0x30);
    if (own == Kind::Buffer) {
        draw.AddBuffer(address, address + Read<u32>(object, 0x180));
    } else if (own != Kind::Unknown) {
        draw.AddImage(address);
    }
}

void StoreDraw(const DrawResources& draw) {
    if (draw.seq == 0) {
        return;
    }
    if (!draw_ring) {
        std::scoped_lock lock{draw_ring_mutex};
        if (!draw_ring) {
            draw_ring = std::make_unique<DrawResources[]>(DrawRing);
        }
    }
    draw_ring[draw.seq % DrawRing] = draw;
}

std::mutex slot_mutex;
std::map<std::pair<u32, u32>, u64> context_slots;
void NoteContextSlot(u32 offset, u32 resource_kind) {
    std::scoped_lock lock{slot_mutex};
    ++context_slots[{offset, resource_kind}];
}

void SetCurrentDraw(u64 seq) {
    current_draw_seq = seq;
}

void SetDrawClass(u32 draw_class_) {
    draw_class = draw_class_ & 15;
}

void AuditBuffer(u64 address, u64 size, bool written) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    by_class[draw_class].buffers.fetch_add(1, std::memory_order_relaxed);
    if ((audit_calls.fetch_add(1, std::memory_order_relaxed) & 0xfff) == 0) {
        DrainTextures();
    }
    audit.buffers.fetch_add(1, std::memory_order_relaxed);
    bool inside = false, partial = false;
    u64 reg_address = 0, reg_bytes = 0;
    u32 reg_usage = ~0u;
    {
        std::shared_lock lock{registry_mutex};
        auto it = buffer_intervals.upper_bound(address);
        if (it != buffer_intervals.begin()) {
            --it;
            if (const auto e = by_id.find(it->second); e != by_id.end() && address >= e->second.address && address < e->second.address + e->second.size) {
                reg_address = e->second.address;
                reg_bytes = e->second.size;
                reg_usage = e->second.usage;
                if (address + size <= e->second.address + e->second.size) {
                    inside = true;
                } else {
                    partial = true;
                }
            }
        }
    }
    {
        // Is it one of the buffers this draw's own objects describe?
        const DrawResources* draw = (current_draw_seq != 0 && draw_ring) ? &draw_ring[current_draw_seq % DrawRing] : nullptr;
        if (draw == nullptr || draw->seq != current_draw_seq) {
            no_draw_record_buffers[draw_class].fetch_add(1, std::memory_order_relaxed);
        } else {
            bool explained = false;
            for (u32 i = 0; i < draw->buffer_count && !explained; ++i) {
                explained = address >= draw->buffers[i].begin && address + size <= draw->buffers[i].end;
            }
            if (explained) {
                explained_buffers[draw_class].fetch_add(1, std::memory_order_relaxed);
            } else {
                (inside ? unexplained_registered_buffers : unexplained_unknown_buffers)[draw_class].fetch_add(1, std::memory_order_relaxed);
                if (draw->dropped == 0) {
                    std::scoped_lock lock{sample_mutex};
                    if (unexplained_per_class[draw_class]++ < 6) {
                        unexplained_samples.push_back({address, size, written, draw_class, reg_address, reg_bytes, reg_usage, draw->buffer_count});
                    }
                }
            }
        }
    }
    if (inside) {
        audit.buffers_inside.fetch_add(1, std::memory_order_relaxed);
        by_class[draw_class].buffers_inside.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    (partial ? audit.buffers_partial : audit.buffers_missing).fetch_add(1, std::memory_order_relaxed);
    if (!partial) {
        const u32 usage_class = NativeResources::UsageClassOf(address);
        miss_by_usage_class[usage_class < 17 ? usage_class : 0].fetch_add(1, std::memory_order_relaxed);
        if (written) {
            miss_written_by_usage_class[usage_class < 17 ? usage_class : 0].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!partial && written) {
        audit.buffers_missing_written.fetch_add(1, std::memory_order_relaxed);
    }
    std::scoped_lock lock{sample_mutex};
    if (!partial && (buffer_misses.size() < 20000 || buffer_misses.contains(address))) {
        auto& m = buffer_misses[address];
        ++m.count;
        m.size = std::max(m.size, size);
        m.written |= written;
    }
    auto& samples = partial ? buffer_partial_samples : buffer_miss_samples;
    if (samples.size() < 48) {
        for (const auto& s : samples) {
            if (s.address == address && s.size == size) {
                return;
            }
        }
        samples.push_back({address, size, written});
    }
}

void AuditImage(u64 address) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    audit.images.fetch_add(1, std::memory_order_relaxed);
    by_class[draw_class].images.fetch_add(1, std::memory_order_relaxed);
    {
        const DrawResources* draw = (current_draw_seq != 0 && draw_ring) ? &draw_ring[current_draw_seq % DrawRing] : nullptr;
        if (draw == nullptr || draw->seq != current_draw_seq) {
            no_draw_record_images[draw_class].fetch_add(1, std::memory_order_relaxed);
        } else {
            bool explained = false;
            for (u32 i = 0; i < draw->image_count && !explained; ++i) {
                explained = draw->images[i] == address;
            }
            (explained ? explained_images : unexplained_images)[draw_class].fetch_add(1, std::memory_order_relaxed);
        }
    }
    bool found = false, in_buffer = false;
    {
        std::shared_lock lock{registry_mutex};
        found = texture_by_start.contains(address);
        if (!found) {
            auto it = buffer_intervals.upper_bound(address);
            if (it != buffer_intervals.begin()) {
                --it;
                if (const auto e = by_id.find(it->second); e != by_id.end() && address >= e->second.address && address < e->second.address + e->second.size) {
                    in_buffer = true;
                }
            }
        }
    }
    if (in_buffer) {
        audit.images_in_buffer.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (found) {
        audit.images_found.fetch_add(1, std::memory_order_relaxed);
        by_class[draw_class].images_found.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (u64 start = 0, size = 0; NativeResources::BlockContaining(address, &start, &size)) {
        std::shared_lock lock{registry_mutex};
        if (texture_by_start.contains(start)) {
            audit.images_in_texture_block.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    audit.images_missing.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{sample_mutex};
    if (image_misses.size() < 20000 || image_misses.contains(address)) {
        ++image_misses[address].count;
    }
    if (image_miss_samples.size() < 48) {
        for (const auto& s : image_miss_samples) {
            if (s.address == address) {
                return;
            }
        }
        image_miss_samples.push_back({address, 0, false});
    }
}

void Start() {
    enabled.store(true, std::memory_order_release);
}

nlohmann::json Report() {
    DrainTextures();
    nlohmann::json j;
    j["enabled"] = enabled.load();
    {
        std::shared_lock lock{registry_mutex};
        auto& kinds = j["objects_by_kind"] = nlohmann::json::object();
        std::array<u64, static_cast<size_t>(Kind::NumKinds)> live{};
        std::array<u64, 4> usage_buffers{};
        for (const auto& [id, e] : by_id) {
            if (e.alive) {
                ++live[static_cast<size_t>(e.actual)];
                if (e.actual == Kind::Buffer && e.usage < 4) {
                    ++usage_buffers[e.usage];
                }
            }
        }
        for (size_t k = 0; k < live.size(); ++k) {
            kinds[KindNames[k]] = {{"live", live[k]}, {"seen", seen_by_kind[k]}};
        }
        j["buffers_by_usage"] = {{"default", usage_buffers[0]}, {"immutable", usage_buffers[1]}, {"dynamic", usage_buffers[2]}, {"staging", usage_buffers[3]}};
        {
            auto& list = j["texture3d_entries"] = nlohmann::json::array();
            for (const auto& [id, e] : by_id) {
                if (e.actual == Kind::Texture3D && list.size() < 40) {
                    list.push_back({{"id", id}, {"alive", e.alive}, {"address", fmt::format("{:#x}", e.address)}, {"w", e.width}, {"h", e.height}, {"d", e.layers}, {"mips", e.mips}, {"format", e.format}, {"usage", e.usage}});
                }
            }
        }
        {
            // How a buffer's own descriptor (V# at +0x60) relates to its D3D11 state (memory at +0x30, ByteWidth): the rule the native description needs.
            u64 ring_base = 0, ring_size = 0;
            const bool ring = NativeResources::FrameRing(&ring_base, &ring_size);
            struct Rel {
                u64 equal{}, inside{}, in_frame_ring{}, other{}, zero{};
            };
            std::array<Rel, 4> rel{};
            std::array<std::string, 6> examples;
            for (const auto& [id, e] : by_id) {
                if (!e.alive || e.actual != Kind::Buffer || e.usage > 3 || e.address == 0) {
                    continue;
                }
                u32 v[4]{};
#ifdef _WIN32
                SIZE_T got = 0;
                if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(e.object + 0x60), v, sizeof(v), &got) || got != sizeof(v)) {
                    continue;
                }
#endif
                const u64 base = u64{v[0]} | (u64{v[1] & 0xffff} << 32);
                const u64 stride = (v[1] >> 16) & 0x3fff;
                const u64 bytes = stride != 0 ? stride * v[2] : v[2];
                Rel& r = rel[e.usage];
                if (base == 0 && bytes == 0) {
                    ++r.zero;
                } else if (base == e.address && bytes == e.size) {
                    ++r.equal;
                } else if (base >= e.address && base + bytes <= e.address + e.size) {
                    ++r.inside;
                } else if (ring && base >= ring_base && base < ring_base + ring_size) {
                    ++r.in_frame_ring;
                } else {
                    ++r.other;
                    if (examples[e.usage].empty()) {
                        examples[e.usage] = fmt::format("id {} mem {:#x}+{} vsharp {:#x}+{} dwords {:08x} {:08x} {:08x} {:08x}", id, e.address, e.size, base, bytes, v[0], v[1], v[2], v[3]);
                    }
                }
            }
            auto& out = j["buffer_vsharp_vs_d3d11_state"] = nlohmann::json::object();
            constexpr const char* usage_names[4] = {"default", "immutable", "dynamic", "staging"};
            for (u32 u = 0; u < 4; ++u) {
                out[usage_names[u]] = {{"equal_to_memory_and_bytewidth", rel[u].equal}, {"inside_the_buffer", rel[u].inside}, {"in_frame_ring", rel[u].in_frame_ring},
                                       {"zero", rel[u].zero}, {"other", rel[u].other}, {"example_other", examples[u]}};
            }
        }
        {
            std::scoped_lock lock{slot_mutex};
            auto& slots = j["context_slots_holding_resources"] = nlohmann::json::object();
            for (const auto& [key, count] : context_slots) {
                slots[fmt::format("{}{:#x}:{}", (key.first & 0x80000000u) != 0 ? "NOT_COLLECTED " : "", key.first & 0x7fffffffu, KindNames[key.second < 5 ? (key.second == 1 ? 1 : key.second) : 0])] = count;
            }
        }
        j["dead_objects"] = dead_objects;
        j["address_reuse"] = reused_addresses;
        auto& tags = j["tags_by_slot_kind"] = nlohmann::json::object();
        for (const auto& [key, count] : tags_by_hint) {
            tags[std::string(KindNames[key.first]) + ":tag" + std::to_string(key.second)] = count;
        }
        auto& raw = j["raw_objects"] = nlohmann::json::array();
        for (const auto& d : dumps) {
            raw.push_back({{"slot", KindNames[static_cast<size_t>(d.hint)]}, {"actual", KindNames[static_cast<size_t>(d.actual)]}, {"tag", d.tag}, {"id", d.id}, {"hex", d.hex}});
        }
    }
    const auto samples = [](const std::vector<Sample>& list, size_t limit = 8) {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& s : list) {
            if (out.size() >= limit) {
                break;
            }
            out.push_back({{"address", fmt::format("{:#x}", s.address)}, {"size", s.size}, {"written", s.written}});
        }
        return out;
    };
    {
        auto& v2 = j["draw_description_audit"] = nlohmann::json::object();
        constexpr const char* names[16] = {"z_prepass_shadow", "g_buffer", "forward", "compute_other", "", "", "", "", "", "", "", "", "", "", "", "untagged"};
        for (u32 c = 0; c < 16; ++c) {
            const u64 total = explained_buffers[c] + unexplained_registered_buffers[c] + unexplained_unknown_buffers[c] + no_draw_record_buffers[c] + explained_images[c] +
                              unexplained_images[c] + no_draw_record_images[c];
            if (total != 0) {
                v2[names[c][0] ? names[c] : std::to_string(c)] = {{"buffers_explained_by_the_draws_objects", explained_buffers[c].load()},
                                                                  {"buffers_other_registered_object", unexplained_registered_buffers[c].load()},
                                                                  {"buffers_not_registered", unexplained_unknown_buffers[c].load()},
                                                                  {"buffers_no_draw_record", no_draw_record_buffers[c].load()},
                                                                  {"images_explained", explained_images[c].load()},
                                                                  {"images_not_explained", unexplained_images[c].load()},
                                                                  {"images_no_draw_record", no_draw_record_images[c].load()}};
            }
        }
        auto& ex = v2["samples_unexplained_buffers"] = nlohmann::json::array();
        std::scoped_lock lock{sample_mutex};
        for (const auto& s : unexplained_samples) {
            ex.push_back({{"address", fmt::format("{:#x}", s.address)}, {"size", s.size}, {"written", s.written}, {"class", s.draw_class}, {"block", NativeResources::DescribeAddress(s.address)},
                       {"registered_buffer", fmt::format("{:#x}+{}", s.reg_address, s.reg_bytes)}, {"registered_usage", s.reg_usage}, {"buffers_in_draw_description", s.draw_buffers}});
        }
    }
    {
        auto& by = j["audit_by_pass_class"] = nlohmann::json::object();
        constexpr const char* names[16] = {"z_prepass_shadow", "g_buffer", "forward", "compute_other", "", "", "", "", "", "", "", "", "", "", "", "untagged"};
        for (u32 c = 0; c < 16; ++c) {
            if (by_class[c].buffers.load() + by_class[c].images.load() != 0) {
                by[names[c][0] ? names[c] : std::to_string(c)] = {{"buffer_requests", by_class[c].buffers.load()}, {"buffers_inside_registered", by_class[c].buffers_inside.load()},
                                                                  {"image_requests", by_class[c].images.load()}, {"images_at_registered_start", by_class[c].images_found.load()}};
            }
        }
    }
    {
        std::scoped_lock lock{sample_mutex};
        j["audit_buffers"] = {{"requests", audit.buffers.load()},
                              {"inside_a_registered_buffer", audit.buffers_inside.load()},
                              {"partly_outside", audit.buffers_partial.load()},
                              {"not_registered", audit.buffers_missing.load()},
                              {"not_registered_written", audit.buffers_missing_written.load()},
                              {"samples_not_registered", samples(buffer_miss_samples, 8)},
                              {"samples_partly_outside", samples(buffer_partial_samples, 8)}};
        {
            std::vector<std::pair<u64, MissCount>> top(buffer_misses.begin(), buffer_misses.end());
            std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
            auto& list = j["audit_buffers"]["top_not_registered"] = nlohmann::json::array();
            for (size_t i = 0; i < top.size() && i < 16; ++i) {
                list.push_back({{"address", fmt::format("{:#x}", top[i].first)}, {"size", top[i].second.size}, {"requests", top[i].second.count}, {"written", top[i].second.written},
                                {"block", NativeResources::DescribeAddress(top[i].first)}});
            }
            j["audit_buffers"]["distinct_not_registered"] = buffer_misses.size();
            // Probe: does a registered buffer's descriptor (V# at object + 0x60: base, stride, records) cover the address right now?
            auto& probe = j["audit_buffers"]["vsharp_probe"] = nlohmann::json::array();
            std::shared_lock registry_lock{registry_mutex};
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                const u64 address = top[i].first;
                nlohmann::json hit = {{"address", fmt::format("{:#x}", address)}, {"covering", nlohmann::json::array()}};
                for (const auto& [id, e] : by_id) {
                    if (!e.alive || e.actual != Kind::Buffer) {
                        continue;
                    }
                    u32 v[4]{};
#ifdef _WIN32
                    SIZE_T got = 0;
                    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(e.object + 0x60), v, sizeof(v), &got) || got != sizeof(v)) {
                        continue;
                    }
#endif
                    const u64 base = u64{v[0]} | (u64{v[1] & 0xffff} << 32);
                    const u64 stride = (v[1] >> 16) & 0x3fff;
                    const u64 bytes = stride != 0 ? stride * v[2] : v[2];
                    if (base > 0x1000000000ull && base < 0x2000000000ull && bytes != 0 && bytes <= (64ull << 20) && address >= base && address < base + bytes && hit["covering"].size() < 3) {
                        hit["covering"].push_back({{"id", id}, {"object", fmt::format("{:#x}", e.object)}, {"memory", fmt::format("{:#x}", e.address)}, {"byte_width", e.size}, {"usage", e.usage},
                                                  {"vsharp_base", fmt::format("{:#x}", base)}, {"vsharp_bytes", bytes}});
                    }
                }
                probe.push_back(std::move(hit));
            }
        }
        auto& by_class = j["audit_buffers"]["not_registered_by_heap_class"] = nlohmann::json::object();
        for (u32 c = 0; c < 17; ++c) {
            if (const u64 n = miss_by_usage_class[c].load(); n != 0) {
                by_class[NativeResources::UsageClassNameOf(c)] = {{"requests", n}, {"written", miss_written_by_usage_class[c].load()}};
            }
        }
        j["audit_images"] = {{"requests", audit.images.load()},
                             {"start_address_registered", audit.images_found.load()},
                             {"inside_a_registered_buffer", audit.images_in_buffer.load()},
                             {"inside_a_registered_texture_allocation", audit.images_in_texture_block.load()},
                             {"not_registered", audit.images_missing.load()},
                             {"samples_not_registered", samples(image_miss_samples, 8)}};
        {
            std::vector<std::pair<u64, MissCount>> top(image_misses.begin(), image_misses.end());
            std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
            auto& list = j["audit_images"]["top_not_registered"] = nlohmann::json::array();
            for (size_t i = 0; i < top.size() && i < 12; ++i) {
                list.push_back({{"address", fmt::format("{:#x}", top[i].first)}, {"requests", top[i].second.count}, {"block", NativeResources::DescribeAddress(top[i].first)}});
            }
            j["audit_images"]["distinct_not_registered"] = image_misses.size();
        }
    }
    return j;
}

} // namespace Core::NativeRegistry
