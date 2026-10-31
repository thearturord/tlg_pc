// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "native/native_audit.h"
#include "native/native_device.h"
#include "native/native_readbacks.h"
#include "native/native_registry.h"
#include "native/native_updates.h"
#include "native/native_resources.h"
#include "native/native_resource_audit.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "native/native_readback_policy.h"
#include "native/native_uploads.h"
#include "native/native_untracked_ring.h"
#include <nlohmann/json.hpp>
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#include "native/native_write_watch.h"

#include <vk_mem_alloc.h>

// Native renderer: member functions of an emulator class that belong to the native path. They live here (not in the emulator's file)
// so that the emulator files only call into native/; the class stays declared in the emulator's header until the owning plan phase
// (documents/tlg/native plan/no-emulation-plan.md, E-1 part 2) gives the native side its own interface.

namespace VideoCore {

void BufferCache::NoteEngineBufferDelete(u64 id, VAddr address, u64 bytes, bool native) {
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BuffersDeleted);
    if (Core::NativeUpdates::B9PlaceOn()) {
        // B9.4: the block that holds it loses the member; a block with no member left is freed (its contents are gone with them)
        VAddr block = 0;
        bool last = false;
        {
            std::scoped_lock lock{placed_mutex};
            auto it = placed.upper_bound(address);
            if (it != placed.begin()) {
                --it;
                auto& ids = it->second.ids;
                if (const auto at = std::ranges::find(ids, id); at != ids.end()) {
                    ids.erase(at);
                    block = it->first;
                    last = ids.empty();
                }
            }
        }
        if (block != 0 && last) {
            // E4: the block's last engine buffer is gone, but the GPU wrote bytes of it outside that buffer (memory no engine buffer owns, taken in when the block grew):
            // nobody deleted those and there is no arena that keeps them, so the block stays as their pseudo resource
            u64 foreign = 0;
            {
                std::shared_lock lock{gpu_ranges_mutex};
                const auto& entry = placed.at(block);
                gpu_modified_ranges.ForEachInRange(entry.begin, entry.end - entry.begin, [&](u64 x, u64 e) {
                    const u64 lo = std::max<u64>(x, entry.begin), hi = std::min<u64>(e, entry.end);
                    if (lo < address) {
                        foreign += std::min<u64>(hi, address) - lo;
                    }
                    if (hi > address + bytes) {
                        foreign += hi - std::max<u64>(lo, address + bytes);
                    }
                });
            }
            if (foreign != 0) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4FreedWithForeignGpu);
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4FreedForeignGpuBytes, foreign);
                std::scoped_lock lock{placed_mutex};
                placed.at(block).ids.push_back((u64{1} << 63) | block);
            } else {
                DropPlaced(block);
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedFreed);
            }
        } else if (block != 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9MemberLeft);
            if (auto* object = Core::NativeResources::FindImmutableById(id)) {
                object->resident_epoch.store(~u64{0}, std::memory_order_release);
            }
        }
        place_refused.erase(id);
    }
    {
        std::scoped_lock lock{object_dirty_mutex};
        if (const auto it = object_dirty.find(id); it != object_dirty.end()) {
            if (!it->second.m_ranges_set.empty()) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9DeletedPending);
            }
            object_dirty.erase(it);
        }
    }
    if (bytes == 0) {
        return;
    }
    bool gpu_written;
    {
        std::shared_lock lock{gpu_ranges_mutex};
        gpu_written = gpu_modified_ranges.Intersects(address, bytes);
    }
    if (!gpu_written) {
        return;
    }
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9DeletedGpuWritten);
    // another live buffer over the same memory (an alias, a buffer holding this one) keeps what the GPU wrote there
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    parts.clear();
    Core::NativeDevice::BuffersIn(address, bytes, parts);
    if (!parts.empty()) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9DeletedAlias);
        return;
    }
    if (native) {
        // the GPU's bytes of a buffer that is gone are nobody's: the next owner's uploads are not kept back for them
        {
            std::unique_lock lock{gpu_ranges_mutex};
            BindMemoBump(); // GPU written bytes change hands: memo answers are checked again
            gpu_modified_ranges.Subtract(address, bytes);
        }
        InvalidateStreamReuse(address, bytes);
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9DeletedGpuDropped);
    }
}

bool BufferCache::ObjectCopiesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_OBJECT_COPIES");
        return e == nullptr || *e != '0';
    }();
    return on;
}

void BufferCache::VerifyObjectCopy(const Core::NativeResources::ImmutableBuffer& object) {
    static const u32 every = [] {
        const char* e = std::getenv("TLG_OBJECT_COPY_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::max(std::atoi(e), 0)) : 0u;
    }();
    static u64 counter = 0;
    if (every == 0 || ++counter % every != 0) {
        return;
    }
    object_copy_checks.fetch_add(1, std::memory_order_relaxed);
    const u64 whole = object.end - object.begin;
    if (std::memcmp(stream_buffer.mapped_data.data() + object.copy_offset, reinterpret_cast<const void*>(object.begin), whole) != 0) {
        if (object_copy_stale.fetch_add(1, std::memory_order_relaxed) < 12) {
            LOG_ERROR(Render, "R1b: the copy of engine buffer {:#x}+{:#x} (usage {}) differs from guest memory at generation {}", object.begin, whole, object.usage,
                      object.copy_generation);
        }
    }
}

bool BufferCache::StreamWhilePassOpen(VAddr device_addr, u32 size) {
    // a page the GPU wrote somewhere is no reason to cut the pass: only bytes the GPU wrote (the precise ranges) must come from the arena
    static const bool on = [] {
        const char* e = std::getenv("TLG_STREAM_IN_PASS");
        return e == nullptr || *e != '0';
    }();
    static const u32 max_bytes = [] {
        const char* e = std::getenv("TLG_STREAM_IN_PASS_MAX"); // bytes, default 16 KiB (the stream threshold)
        return e != nullptr && *e != 0 ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : static_cast<u32>(STREAM_THRESHOLD);
    }();
    return on && size <= max_bytes && scheduler.IsRendering() && !IsRangeGpuWritten(device_addr, size);
}

BufferCache::PlacedBuffer* BufferCache::PlacedAt(VAddr address, u64 size, bool written) {
    auto it = placed.upper_bound(address);
    if (it == placed.begin()) {
        return nullptr;
    }
    --it;
    return address >= it->second.begin && address + size <= it->second.end + (written ? 0 : PlacedSlack) && address < it->second.end ? &it->second : nullptr;
}

bool BufferCache::PlacedMeets(VAddr address, u64 size) const {
    auto it = placed.upper_bound(address + size - 1);
    while (it != placed.begin()) {
        --it;
        if (it->second.end > address) {
            return true;
        }
        if (it->second.end <= address && it->first < address) {
            // placed buffers do not overlap each other: the nearest one below that ends before the range is the last candidate
            return false;
        }
    }
    return false;
}

const Buffer* BufferCache::Place(std::vector<u64> ids, VAddr begin, VAddr end) {
    const VAddr base = begin & ~VAddr{255};
    // zero copy: engine buffers the CPU writes (not the ones it reads back) live in video memory the CPU can write (resizable BAR), so their announced bytes
    // can be written there directly (SynchronizeMemory). A device without such memory keeps device local buffers (the GPU must not read system memory: N7.5).
    bool zero_copy = ZeroCopyOn();
    if (zero_copy) {
        for (const u64 id : ids) {
            Core::NativeDevice::Object object;
            if (Core::NativeDevice::Lookup(id, object) && (object.usage == 3 || (object.cpu_access & 0x20000) != 0)) {
                zero_copy = false;
            }
        }
    }
    auto buffer = std::make_unique<Buffer>(instance, base, end - base + PlacedSlack, zero_copy ? MemoryType::DeviceMapped : MemoryType::DeviceLocal,
                                           fmt::format("engine buffer {}{} {:#x}+{:#x}", ids.front(), ids.size() > 1 ? "+" : "", begin, end - begin));
    if (zero_copy && (!buffer->is_device_local || buffer->mapped_data.empty())) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            LOG_WARNING(Render_Vulkan, "zero copy: no device local memory the CPU can write (resizable BAR off?): engine buffers stay device local");
        }
        zero_copy_unavailable.store(true, std::memory_order_relaxed);
        buffer = std::make_unique<Buffer>(instance, base, end - base + PlacedSlack, MemoryType::DeviceLocal,
                                          fmt::format("engine buffer {}{} {:#x}+{:#x}", ids.front(), ids.size() > 1 ? "+" : "", begin, end - begin));
    }
    Buffer* dst = buffer.get();
    scheduler.EndRendering();
    // placed buffers inside the block: their contents come over as they are
    boost::container::small_vector<VAddr, 4> inside;
    for (auto it = placed.lower_bound(begin); it != placed.end() && it->first < end; ++it) {
        inside.push_back(it->first);
        const vk::BufferCopy copy{it->second.buffer->Offset(it->second.begin), dst->Offset(it->second.begin), it->second.end - it->second.begin};
        runtime.CopyBuffer(it->second.buffer.get(), dst, std::span<const vk::BufferCopy>{&copy, 1});
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GroupReused);
    }
    // the rest: the CPU's bytes from memory, the bytes the GPU wrote (and nothing newer since) from the arena
    boost::container::small_vector<std::pair<u64, u64>, 8> rest;
    {
        VAddr at = begin;
        for (const VAddr b : inside) {
            const auto& entry = placed.at(b);
            if (entry.begin > at) {
                rest.emplace_back(at, entry.begin - at);
            }
            at = entry.end;
        }
        if (at < end) {
            rest.emplace_back(at, end - at);
        }
    }
    boost::container::small_vector<std::pair<u64, u64>, 8> gpu_pieces, cpu_pieces;
    {
        std::shared_lock lock{gpu_ranges_mutex};
        for (const auto& [a, n] : rest) {
            gpu_modified_ranges.ForEachInRange(a, n, [&](u64 x, u64 e) { gpu_pieces.emplace_back(x, e - x); });
            cpu_pieces.emplace_back(a, n);
        }
    }
    if (!gpu_pieces.empty()) {
        // E4: bytes the GPU wrote that no placed buffer holds (should stay near 0: a block with GPU bytes of memory no engine buffer owns is not freed)
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedGpuCopied);
        u64 lost = 0;
        for (const auto& [a, e] : gpu_pieces) {
            lost += e;
        }
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4GpuBytesLost, lost);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
            LOG_WARNING(Render_Vulkan, "E4: placing {:#x}+{:#x} ({} ids, first {:#x}): {} GPU written bytes no placed buffer holds, first {:#x} ({})", begin, end - begin, ids.size(), ids.front(), lost,
                        gpu_pieces[0].first, Core::NativeDevice::OwnerName(gpu_pieces[0].first));
        }
    }
    u64 total = 0;
    for (const auto& [a, n] : cpu_pieces) {
        total += n;
    }
    if (total != 0) {
        const auto staging = staging_pool.Request(total, MemoryType::HostUncached);
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        u64 at = 0;
        for (const auto& [a, n] : cpu_pieces) {
            memory->CopySparseMemory(a, staging.mapped + at, n);
            copies.push_back({staging.offset + at, dst->Offset(a), n});
            at += n;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, dst, copies);
    }
    {
        // the block holds memory as it is now: the announced bytes outside the placed buffers it took over are uploaded (those inside stay announced:
        // the members' next uploads put them in)
        std::scoped_lock lock{object_dirty_mutex};
        for (const u64 id : ids) {
            if (const auto it = object_dirty.find(id); it != object_dirty.end()) {
                for (const auto& [a, n] : rest) {
                    it->second.Subtract(a, n);
                }
            }
        }
        for (const auto& [a, n] : rest) {
            b9_unattributed.Subtract(a, n);
            all_object_dirty.Subtract(a, n);
        }
    }
    // the placed buffers inside are replaced (freed after the GPU is done, the copies above included)
    for (const VAddr b : inside) {
        DropPlaced(b);
    }
    {
        std::scoped_lock lock{placed_mutex};
        placed.emplace(begin, PlacedBuffer{ids, {}, begin, end, std::move(buffer)});
    }
    ForgetShadows(begin, end);
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9Placed);
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedBytes, end - begin);
    if (ids.size() > 1 || !inside.empty()) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9Groups);
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GroupMembers, ids.size());
    }
    range_epoch.fetch_add(1, std::memory_order_acq_rel); // the upload memo's answers (this command buffer only) that name the arena for this memory are stale
    for (const u64 id : ids) {
        NoteResolutionUser(dst, id);
        if (auto* object = Core::NativeResources::FindImmutableById(id)) {
            // the resident object's binding resolution (R2) names the block's buffer from now on; the other objects' resolutions stay valid
            object->resident_buffer.store(dst, std::memory_order_relaxed);
            object->resident_base.store(dst->cpu_addr, std::memory_order_relaxed);
            object->resident_epoch.store(PlacementEpoch(), std::memory_order_release);
        }
    }
    return dst;
}

void BufferCache::NoteResolutionUser(const Buffer* buffer, u64 id) {
    if (id == 0) {
        return;
    }
    std::scoped_lock lock{placed_mutex};
    for (auto it = placed.upper_bound(buffer->cpu_addr + 255); it != placed.begin();) {
        --it;
        if (it->second.buffer.get() == buffer) {
            if (std::ranges::find(it->second.users, id) == it->second.users.end()) {
                it->second.users.push_back(id);
            }
            return;
        }
        if (it->first < buffer->cpu_addr) {
            return;
        }
    }
}

void BufferCache::DropPlaced(VAddr begin) {
    std::unique_ptr<Buffer> buffer;
    std::vector<u64> ids;
    {
        std::scoped_lock lock{placed_mutex};
        const auto it = placed.find(begin);
        if (it == placed.end()) {
            return;
        }
        ids = std::move(it->second.ids);
        ids.insert(ids.end(), it->second.users.begin(), it->second.users.end()); // every resolution that names it, members or not
        buffer = std::move(it->second.buffer);
        placed.erase(it);
    }
    range_epoch.fetch_add(1, std::memory_order_acq_rel);
    for (const u64 id : ids) {
        if (auto* object = Core::NativeResources::FindImmutableById(id)) {
            object->resident_epoch.store(~u64{0}, std::memory_order_release); // its binding resolution is made again
        }
    }
    // freed once the GPU is done with the work recorded so far (its descriptor slots are dropped by the buffer's destructor)
    scheduler.DeferOperation([b = std::move(buffer)]() mutable { b.reset(); });
}

void BufferCache::Demote(VAddr begin, bool by_request) {
    // E4: there is no arena to move to. The block stays; the next request that reaches over it grows the block (Place copies the placed buffers inside)
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4DemoteKept);
}

void BufferCache::NoteEngineResourceCreate(u64 id, VAddr address, u64 bytes, bool texture) {
    if (!Core::NativeUpdates::B9PlaceOn() || bytes == 0) {
        return;
    }
    boost::container::small_vector<VAddr, 4> under;
    {
        std::scoped_lock lock{placed_mutex};
        // placed buffers do not overlap each other: only the one that starts below the address can reach into the range from below
        auto it = placed.upper_bound(address);
        if (it != placed.begin()) {
            --it;
        }
        for (; it != placed.end() && it->first < address + bytes; ++it) {
            if (it->second.end > address) {
                under.push_back(it->first);
            }
        }
    }
    if (texture && under.size() == 1 && Core::NativeUpdates::B9AliasesOn()) {
        // a texture inside one block: the block stays (the texture's image reads memory, or the block for bytes the GPU wrote)
        const auto& entry = placed.at(under[0]);
        if (address >= entry.begin && address + bytes <= entry.end) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TextureInBlockKept);
            return;
        }
    }
    if (!texture && under.size() == 1) {
        // a new buffer inside one placed block that overlaps no live member of it (a gap, or a deleted member's memory): it joins the block
        // (TLG_B9_ALIASES=1: also over live members)
        auto& entry = placed.at(under[0]);
        if (address >= entry.begin && address + bytes <= entry.end) {
            static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
            static thread_local std::vector<u64> textures;
            parts.clear();
            textures.clear();
            Core::NativeDevice::BuffersIn(address, bytes, parts);
            (void)Core::NativeDevice::TexturesIn(address, bytes, textures);
            const bool alone = textures.empty() && std::ranges::all_of(parts, [&](const auto& part) { return part.id == id; });
            if (!alone && Core::NativeUpdates::B9AliasesOn()) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9AdoptedOverMember);
            }
            if (alone || Core::NativeUpdates::B9AliasesOn()) {
                entry.ids.push_back(id);
                entry.users.push_back(id);
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9Adopted);
                {
                    // the block holds older bytes there (a deleted member's, a gap's): the new buffer's first upload takes all of its memory, as a placement would
                    std::scoped_lock lock{object_dirty_mutex};
                    object_dirty[id].Add(address, bytes);
                    all_object_dirty.Add(address, bytes);
                }
                b9_epoch.fetch_add(1, std::memory_order_acq_rel);
                if (auto* object = Core::NativeResources::FindImmutableById(id)) {
                    object->resident_buffer.store(entry.buffer.get(), std::memory_order_relaxed);
                    object->resident_base.store(entry.buffer->cpu_addr, std::memory_order_relaxed);
                    object->resident_epoch.store(PlacementEpoch(), std::memory_order_release);
                }
                return;
            }
        }
    }
    for (const VAddr begin : under) {
        Demote(begin, false);
    }
}

u64 BufferCache::BufferHolding(VAddr addr, size_t size) const {
    Core::NativeDevice::Object object;
    if (Core::NativeDevice::ResourceAt(addr, object) && object.kind == Core::NativeDevice::Kind::Buffer && addr + size <= object.address + std::max<u64>(object.bytes, 1)) {
        return object.id;
    }
    return 0;
}

bool BufferCache::ZeroCopyOn() const noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_ZERO_COPY");
        return e != nullptr && *e == '1';
    }();
    return on && !zero_copy_unavailable.load(std::memory_order_relaxed);
}

void BufferCache::ZeroCopyVerifyNote(const Buffer* buffer, std::span<const vk::BufferCopy> copies) {
    // TLG_ZERO_COPY_VERIFY=N: every Nth direct write keeps the bytes it wrote; once the submission recording now has finished, the buffer must still hold
    // them unless the GPU wrote it since (a GPU copy recorded before the write but executed after it would show up here)
    static const u32 every = [] {
        const char* e = std::getenv("TLG_ZERO_COPY_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::max(std::atoi(e), 0)) : 0u;
    }();
    if (every == 0) {
        return;
    }
    static u64 n = 0;
    if (++n % every != 0 || zero_copy_checks.size() >= 64) {
        return;
    }
    for (const auto& copy : copies) {
        if (copy.size > 65536) {
            continue;
        }
        ZeroCopyCheck check{buffer, copy.dstOffset - buffer->cpu_addr, scheduler.CurrentTick(), buffer->gpu_write_generation, {}};
        check.bytes.assign(buffer->mapped_data.data() + check.offset, buffer->mapped_data.data() + check.offset + copy.size);
        zero_copy_checks.push_back(std::move(check));
        break;
    }
}

void BufferCache::ZeroCopyVerifyPoll() {
    for (auto it = zero_copy_checks.begin(); it != zero_copy_checks.end();) {
        if (!scheduler.IsFree(it->tick)) {
            ++it;
            continue;
        }
        bool alive = false;
        {
            std::scoped_lock lock{placed_mutex};
            for (const auto& [b, entry] : placed) {
                if (entry.buffer.get() == it->buffer) {
                    alive = true;
                    break;
                }
            }
        }
        if (alive && it->buffer->gpu_write_generation == it->generation) {
            zero_copy_counts[5].fetch_add(1, std::memory_order_relaxed);
            if (std::memcmp(it->buffer->mapped_data.data() + it->offset, it->bytes.data(), it->bytes.size()) != 0) {
                zero_copy_counts[6].fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render_Vulkan, "zero copy verify: buffer {:#x}+{:#x} ({} bytes) changed after its direct write with no GPU write planned", it->buffer->cpu_addr,
                                it->offset, it->bytes.size());
                }
            }
        }
        it = zero_copy_checks.erase(it);
    }
}

nlohmann::json BufferCache::ZeroCopyReport() const {
    return {{"on", ZeroCopyOn()},
            {"direct_writes", zero_copy_counts[0].load()},
            {"direct_write_ranges", zero_copy_counts[1].load()},
            {"direct_write_bytes", zero_copy_counts[2].load()},
            {"uploads_copied_gpu_still_using_the_buffer", zero_copy_counts[3].load()},
            {"bytes_copied_gpu_still_using_the_buffer", zero_copy_counts[4].load()},
            {"verify_checked", zero_copy_counts[5].load()},
            {"uploads_into_placed_device_local", zero_copy_counts[8].load()},
            {"uploads_into_placed_cpu_writable", zero_copy_counts[9].load()},
            {"VERIFY_CHANGED", zero_copy_counts[6].load()}};
}

} // namespace VideoCore
