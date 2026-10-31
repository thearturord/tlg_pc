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

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>()},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      stream_reuse_enabled{EmulatorSettings.IsReadbackStreamReuseEnabled()} {
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
    Core::NativeReadbacks::SetLandingGate(
        [this](VAddr address, u64 size, u32 timeout_ms) { return WaitForExplicitDownloads(address, size, timeout_ms); });
    // B2 cutover: an applied announcement is a CPU write for the caches
    Core::NativeUpdates::SetApply([](void* context, u64 address, u64 size, bool newer) { static_cast<BufferCache*>(context)->ApplyCpuWrite(address, size, newer); }, this);
    if (Core::NativeUpdates::B2NativeMode() == 2 && !Core::NativeUpdates::HarvestAuditOn() && !(std::getenv("TLG_B2_REFEREE") != nullptr && *std::getenv("TLG_B2_REFEREE") == '1')) {
        Core::WriteWatch::SetQuiet(true); // B7: the write watch is not the referee any more
    }
}

BufferCache::~BufferCache() {
    Core::NativeReadbacks::SetLandingGate({});
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (Core::NativeUpdates::B9TrackerOff()) {
        // a CPU change found without an announcement (a write fault, a decoder writing memory): the engine buffers it meets take it too
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9OtherChanges);
        MarkBuffersChanged(device_addr, size, false, nullptr);
    }
    // E4: no GPU bytes are flushed to guest memory here (the fault readback is gone: no page is ever made unreadable)
    memory_tracker->MarkRegionAsCpuModified(device_addr, size);
}

namespace {
thread_local VAddr t_audit_begin = 0, t_audit_end = 0; ///< B2 audit: the range of the current harvest request
} // namespace

void BufferCache::ApplyCpuWrite(VAddr address, u64 size, bool newer) {
    BindMemoErase(address, size);
    if (newer) {
        // B2: the engine wrote these bytes after anything the GPU wrote there before (applied in order on the raster thread)
        SupersedeGpuWritten(address, size);
    }
    Core::NativeResources::NoteCpuWrite(address, size);
    if (Core::NativeUpdates::B9TrackerOff()) {
        // B9.3 step 2: the page tracker only marks what no engine buffer holds
        static thread_local std::vector<std::pair<VAddr, VAddr>> gaps;
        gaps.clear();
        MarkBuffersChanged(address, size, true, &gaps);
        u64 marked = 0;
        for (const auto& [a, e] : gaps) {
            memory_tracker->MarkRegionAsCpuModified(a, e - a);
            marked += e - a;
        }
        if (marked < size) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerMarksSkipped);
        }
    } else {
        memory_tracker->MarkRegionAsCpuModified(address, size);
        if (Core::NativeUpdates::B9BufferMode() != 0) {
            MarkBuffersChanged(address, size, true, nullptr);
        }
    }
    const u32 b8 = Core::NativeUpdates::B8ImageMode();
    if (b8 != 0) {
        texture_cache.NoteAnnouncedWrite(address, size, newer);
    }
    if (b8 != 2 && Core::WriteWatch::LiftImages()) {
        texture_cache.InvalidatePagesWritten(address, size);
    }
}

void BufferCache::MarkBuffersChanged(VAddr address, u64 size, bool announced, std::vector<std::pair<VAddr, VAddr>>* gaps) {
    // B9.1: each engine buffer the bytes belong to keeps its part until its next upload (an announcement can run over several buffers)
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    parts.clear();
    Core::NativeDevice::BuffersIn(address, size, parts);
    std::scoped_lock lock{object_dirty_mutex};
    all_object_dirty.Add(address, size); // audit: every changed byte, by address
    if (announced && b9_owed.Intersects(address, size)) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9VerifyLate);
        b9_owed.Subtract(address, size);
    }
    if (announced) {
        Core::NativeUpdates::NoteB9(parts.empty() ? Core::NativeUpdates::B9AnnouncedNoBuffer
                                    : parts.size() == 1 && parts[0].begin == address && parts[0].end == address + size ? Core::NativeUpdates::B9AnnouncedOnBuffer
                                                                                                                      : Core::NativeUpdates::B9AnnouncedClipped);
    }
    for (const auto& part : parts) {
        object_dirty[part.id].Add(part.begin, part.end - part.begin);
    }
    // the bytes no buffer holds now: the first upload of a buffer over them takes them
    b9_unattributed.Add(address, size);
    for (const auto& part : parts) {
        b9_unattributed.Subtract(part.begin, part.end - part.begin);
    }
    if (!parts.empty()) {
        b9_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    if (gaps != nullptr) {
        std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
        VAddr at = address;
        for (const auto& part : parts) {
            if (part.begin > at) {
                gaps->emplace_back(at, part.begin);
            }
            at = std::max(at, part.end);
        }
        if (at < address + size) {
            gaps->emplace_back(at, address + size);
        }
    }
}

void BufferCache::HarvestWriteWatch(VAddr device_addr, u64 size, bool image) {
    if (!Core::WriteWatch::Enabled()) {
        return;
    }
    // B7: with the engine's announcements deciding (mode 2) the write watch would only count misses: it is not asked any more (TLG_B2_AUDIT=1 keeps it as the referee)
    static const bool referee = Core::NativeUpdates::B2NativeMode() != 2 || Core::NativeUpdates::HarvestAuditOn() || [] {
        const char* e = std::getenv("TLG_B2_REFEREE");
        return e != nullptr && *e == '1';
    }();
    if (!referee) {
        return;
    }
    t_audit_begin = device_addr;
    t_audit_end = device_addr + size;
    // what the write fault did for these pages (Rasterizer::InvalidateMemory, buffer side; readbacks are off whenever the
    // protection is lifted, so the invalidation is the CPU mark); a page an image watches is still protected and faulted already
    Core::WriteWatch::Harvest(
        device_addr, size,
        [](void* context, VAddr page, u64 bytes) {
            auto* self = static_cast<BufferCache*>(context);
            if (const u32 mode = Core::NativeUpdates::B2NativeMode(); mode != 0) {
                // B2 cutover: the announcements marked what the engine wrote; a page here none covered is a miss (mode 1 replays it, mode 2 does not)
                VAddr run = 0;
                u64 run_bytes = 0;
                const auto flush = [&] {
                    if (run_bytes != 0) {
                        Core::NativeUpdates::NoteMiss(run, run_bytes);
                        if (mode == 1) {
                            self->ApplyCpuWrite(run, run_bytes);
                        }
                    }
                    run_bytes = 0;
                };
                for (VAddr p = page; p < page + bytes; p += 4096) {
                    if (Core::NativeUpdates::ConsumeMark(p)) {
                        flush();
                    } else {
                        if (run_bytes == 0) {
                            run = p;
                        }
                        run_bytes += 4096;
                    }
                }
                flush();
                return;
            }
            if (Core::NativeUpdates::HarvestAuditOn()) {
                // only the part the GPU asked for (the harvest covers whole blocks)
                const VAddr a = std::max<VAddr>(page, t_audit_begin), b = std::min<VAddr>(page + bytes, t_audit_end);
                if (a < b) {
                    const auto [lo, hi] = UntrackedRing::Get().Clip(a, b - a);
                    Core::NativeUpdates::NoteHarvested(a, b - a, lo < hi);
                }
            }
            Core::NativeResources::NoteCpuWrite(page, bytes);
            self->BindMemoBump();
            self->memory_tracker->MarkRegionAsCpuModified(page, bytes);
            if (Core::WriteWatch::LiftImages()) {
                // the texture side of the fault (Rasterizer::InvalidateMemory): images on these pages are no longer protected
                self->texture_cache.InvalidatePagesWritten(page, bytes);
            }
        },
        [](void* context, VAddr page, u64 bytes) {
            return static_cast<BufferCache*>(context)->memory_tracker->IsRegionCpuModified(page, bytes);
        },
        [](void* context, VAddr page, u64 bytes) {
            return static_cast<BufferCache*>(context)->memory_tracker->IsRegionGpuModified(page, bytes);
        },
        this, image);
}

void BufferCache::InvalidateStreamReuse(VAddr device_addr, u64 size) {
    for (auto& set : stream_reuse_sets) {
        if (set.map.empty() || device_addr >= set.max || device_addr + size <= set.min) {
            continue;
        }
        set.map.clear();
        set.min = ~0ULL;
        set.max = 0;
    }
}

void BufferCache::RecordDmaDownload(const Buffer* arena, VAddr dst, u32 size, bool tracked) {
    // The game's own readbacks are small lists and counters; a bigger copy is not one of them.
    if (size == 0 || size > MaxDmaDownload) {
        return;
    }
    const auto staging = staging_pool.Request(size, MemoryType::HostCached, 0, true);
    const vk::BufferCopy region = {
        .srcOffset = arena->Offset(dst),
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, staging.buffer, std::span{&region, 1});
    auto& policy = ReadbackPolicy::Get();
    policy.dma_async_copies.fetch_add(1, std::memory_order_relaxed);
    policy.dma_async_bytes.fetch_add(size, std::memory_order_relaxed);
    async_recorded.store(1, std::memory_order_release);
    // The priority thread runs its operations in the order they were deferred, so a fence deferred
    // after this one is signaled only once the bytes are in guest memory.
    u64 id = 0;
    if (tracked) {
        std::scoped_lock lock{explicit_mutex};
        id = explicit_next_id++;
        explicit_inflight.push_back({dst, size, id});
    }
    scheduler.DeferPriorityOperation(
        [this, staging, dst, size, tracked, id] {
            if (tracked) {
                Core::NativeReadbacks::LandingBegin(dst, size);
            }
            staging.buffer->Invalidate(staging.offset, size);
            memory->TryWriteBacking(std::bit_cast<u8*>(dst), staging.mapped, size);
            staging_pool.FreeDeferred(staging);
            if (tracked) {
                Core::NativeReadbacks::LandingEnd(dst, size);
                {
                    std::scoped_lock lock{explicit_mutex};
                    std::erase_if(explicit_inflight, [id](const ExplicitDownload& download) { return download.id == id; });
                }
                explicit_landed.notify_all();
            }
        },
        scheduler.CurrentTick());
}

u64 BufferCache::PreSynchronize(VAddr device_addr, u64 size) {
    if (size == 0 || !memory->IsValidMapping(device_addr, 1)) {
        return 0;
    }
    size = std::min<u64>(memory->ClampRangeSize(device_addr, size), u64{1} << 26);
    if (size == 0) {
        return 0;
    }
    const auto* arena = BufferFor(device_addr, size, false); // B9.4: a range over placed buffers' edges is left to their own requests
    if (arena == nullptr) {
        return 0;
    }
    last_upload_bytes = 0;
    Core::NativeUploads::SetPreSyncActive(true);
    SynchronizeMemory(arena, device_addr, static_cast<u32>(size), false, false);
    Core::NativeUploads::SetPreSyncActive(false);
    return last_upload_bytes;
}

static bool BindMemoOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_BIND_MEMO");
        return e == nullptr || *e != '0';
    }();
    return on;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                        Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object) {
    if (is_written) {
        BindMemoErase(device_addr, size); // the GPU writes it: answers over this memory are stale
    }
    if (is_written || is_texel_buffer || size > BindMemoMax || !BindMemoOn()) {
        return ObtainBufferUncached(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
    }
    auto& memo = bind_memos[Vulkan::Scheduler::SideSelected() ? 1 : 0];
    const u64 tick = scheduler.CurrentTick();
    if (memo.tick != tick) {
        // a new command buffer: copies in the stream buffer belong to the old one (the game may rewrite their memory between submissions); answers that name
        // an arena or a placed buffer stay valid until a write or a move erases them (the frame has about 13 command buffers)
        memo.tick = tick;
        for (auto it = memo.entries.begin(); it != memo.entries.end();) {
            it->second.erase(std::remove_if(it->second.begin(), it->second.end(), [&](const BindMemoEntry& e) { return e.buffer == &stream_buffer; }), it->second.end());
            it = it->second.empty() ? memo.entries.erase(it) : std::next(it);
        }
    }
    const u64 epoch = bind_memo_epoch.load(std::memory_order_acquire);
    const u64 ranges = range_epoch.load(std::memory_order_acquire);
    const u64 arenas = PlacementEpoch();
    if (const auto it = memo.entries.find(device_addr); it != memo.entries.end()) {
        for (const auto& e : it->second) {
            if (e.size == size && e.epoch == epoch && e.range_epoch == ranges && e.placement_epoch == arenas &&
                (e.buffer != &stream_buffer || e.wraps == stream_buffer.WrapCount())) {
                bind_memo_counts[0].fetch_add(1, std::memory_order_relaxed);
                return {e.buffer, e.offset};
            }
        }
    }
    bind_memo_counts[1].fetch_add(1, std::memory_order_relaxed);
    const auto result = ObtainBufferUncached(device_addr, size, is_written, is_texel_buffer, immutable_hint, verify_hint, known_no_object);
    if (memo.tick == tick) {
        auto& list = memo.entries[device_addr];
        list.erase(std::remove_if(list.begin(), list.end(), [&](const BindMemoEntry& e) { return e.size == size; }), list.end());
        // the epochs as they were before the request: anything the request itself changed makes the next one ask again
        list.push_back({size, result.first, result.second, ranges, arenas, result.first == &stream_buffer ? stream_buffer.WrapCount() : 0, epoch});
    }
    return result;
}

void BufferCache::BindMemoErase(VAddr address, u64 size) {
    for (auto& memo : bind_memos) {
        if (memo.entries.empty()) {
            continue;
        }
        auto it = memo.entries.lower_bound(address >= BindMemoMax ? address - BindMemoMax : 0);
        while (it != memo.entries.end() && it->first < address + size) {
            it->second.erase(std::remove_if(it->second.begin(), it->second.end(), [&](const BindMemoEntry& e) { return it->first + e.size > address; }), it->second.end());
            if (it->second.empty()) {
                it = memo.entries.erase(it);
                bind_memo_counts[2].fetch_add(1, std::memory_order_relaxed);
            } else {
                ++it;
            }
        }
    }
}

nlohmann::json BufferCache::BindMemoReport() const {
    return {{"on", BindMemoOn()}, {"hits", bind_memo_counts[0].load()}, {"misses", bind_memo_counts[1].load()}, {"addresses_erased_by_writes", bind_memo_counts[2].load()}};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferUncached(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
                                                                Core::NativeResources::ImmutableBuffer* immutable_hint, bool verify_hint, bool known_no_object) {
    Core::ResourceAudit::Note(Core::ResourceAudit::BufferUse, device_addr, size, is_written);
    HarvestWriteWatch(device_addr, size);
    if (static const bool census = [] { const char* e = std::getenv("TLG_LIFT_CENSUS"); return e != nullptr && *e == '1'; }(); census && Core::NativeRegistry::enabled.load(std::memory_order_relaxed)) {
        Core::NativeRegistry::NoteRequestBuffer(device_addr, size, is_written);
    }
    // An immutable engine buffer (see NativeResources::ImmutableBuffer): after one synchronization of the whole buffer it is
    // read straight from the arena.
    // known_no_object (R2): the command processor's lookup by address found no engine object and none was registered since
    if (!is_written && !is_texel_buffer && !known_no_object && Core::NativeResources::ImmutableResidencyEnabled()) {
        Core::NativeResources::ImmutableBuffer* immutable = nullptr;
        if (immutable_hint != nullptr) {
            // N3c: the handle of the engine id replaces the lookup by address; it must contain the request
            if (device_addr >= immutable_hint->begin && device_addr + size <= immutable_hint->end && immutable_hint->alive.load(std::memory_order_acquire)) {
                immutable = immutable_hint;
                Core::NativeResources::NoteHandle(0);
                if (verify_hint && Core::NativeResources::FindImmutable(device_addr, size) != immutable) {
                    Core::NativeResources::NoteHandle(4);
                    immutable = Core::NativeResources::FindImmutable(device_addr, size);
                }
            } else {
                Core::NativeResources::NoteHandle(3);
                immutable = Core::NativeResources::FindImmutable(device_addr, size);
            }
        } else {
            immutable = Core::NativeResources::FindImmutable(device_addr, size);
        }
        if (immutable != nullptr && immutable->usage != 1 && ObjectCopiesOn()) {
            // R1b (native back end): a default or dynamic engine buffer has its own copy in the stream ring, made once per generation from its exact bytes
            // (no page of a neighbour is uploaded with it, so a page the GPU wrote elsewhere does not matter); bytes of the object the GPU wrote keep it in the arena
            u64 offset = 0;
            if (const Buffer* copy = ObjectCopy(*immutable, device_addr, offset)) {
                VerifyObjectCopy(*immutable);
                Core::NativeResources::NoteResident(0);
                return {copy, offset};
            }
            const u32 whole = static_cast<u32>(immutable->end - immutable->begin);
            if (!IsRangeGpuWritten(immutable->begin, whole)) {
                const u32 generation = immutable->generation.load(std::memory_order_acquire);
                const u32 gpu_writes = immutable->gpu_writes.load(std::memory_order_acquire);
                const auto [data, at] = stream_buffer.Map(whole, instance.UniformMinAlignment());
                memory->CopySparseMemory(immutable->begin, data, whole);
                stream_buffer.Commit();
                immutable->copy_offset = at;
                immutable->copy_wraps = stream_buffer.WrapCount();
                immutable->copy_generation = generation;
                immutable->copy_gpu_writes = gpu_writes;
                Core::NativeResources::NoteResident(1);
                Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::StreamCopy, whole);
                return {&stream_buffer, at + (device_addr - immutable->begin)};
            }
        }
        if (immutable != nullptr && immutable->usage != 1 && IsRegionGpuModified(device_addr, size)) {
            // a resident (mutable usage) buffer the GPU modified: the normal path decides
            Core::NativeResources::NoteResident(4);
            immutable = nullptr;
        }
        if (immutable != nullptr) {
            // a resident buffer is also asked of the tracker: the fault handler marks a page after it bumped the generation (the mark can land after a synchronization that
            // saw nothing), so the generation alone is not enough; the tracker's dirty bit is the safety net of the engine's own knowledge
            if (immutable->synced.load(std::memory_order_acquire) && !Core::NativeResources::ImmutableCheck(*immutable) &&
                (immutable->usage == 1 || !IsRegionCpuModified(device_addr, size))) {
                const auto* arena = BufferFor(device_addr, size); // B9.4: the object's own Vulkan buffer when it is placed
                if (immutable->resident_epoch.load(std::memory_order_relaxed) != PlacementEpoch() && immutable->begin >= arena->cpu_addr &&
                    immutable->end <= arena->cpu_addr + arena->size_bytes) {
                    // R1: the arenas moved since the synchronization; the object is resident in this one now
                    NoteResolutionUser(arena, immutable->id);
                    immutable->resident_buffer.store(arena, std::memory_order_relaxed);
                    immutable->resident_base.store(arena->cpu_addr, std::memory_order_relaxed);
                    immutable->resident_epoch.store(PlacementEpoch(), std::memory_order_release);
                }
                Core::NativeResources::NoteImmutableHit();
                if (immutable->usage != 1) {
                    Core::NativeResources::NoteResident(0);
                    const u32 interval = Core::NativeResources::ResidentVerifyInterval();
                    static std::atomic<u64> resident_hits{0};
                    if (interval != 0 && resident_hits.fetch_add(1, std::memory_order_relaxed) % interval == 0) {
                        // verify: one synchronization of the request; anything it has to upload means the answer without it would have been stale
                        last_upload_bytes = 0;
                        SynchronizeMemory(arena, device_addr, size, false, false);
                        Core::NativeResources::NoteResident(2);
                        if (last_upload_bytes != 0) {
                            Core::NativeResources::NoteResident(3);
                            immutable->synced.store(false, std::memory_order_release);
                            static std::atomic<u32> logged{0};
                            if (logged.fetch_add(1) < 12) {
                                // who announced the last write that covers it (needs TLG_UPDATE_AUDIT=1; ~0: no announcement window covers it), the engine id and the bytes now in guest memory
                                const u32 who = Core::NativeUpdates::WhoWrote(device_addr, size);
                                u32 words[16] = {};
                                std::memcpy(words, reinterpret_cast<const void*>(device_addr), std::min<u32>(size, 64));
                                LOG_ERROR(Render, "resident buffer {:#x}+{} (engine buffer {:#x}..{:#x}, usage {}, engine id {}) was stale: {} bytes had to be uploaded; last announcer {} (0 Map 1 skin 2 light 3 stream copy 4 recording / scene slots 5 creation 6 texture 7 instance group, bit 8 render thread, age ms {}); words {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
                                          device_addr, size, immutable->begin, immutable->end, immutable->usage, immutable->id, last_upload_bytes, who == ~0u ? 99u : (who & 0xff), who == ~0u ? 0u : (who >> 8) & 1, who == ~0u ? 0u : who >> 16,
                                          words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
                            }
                        }
                    }
                }
                return {arena, arena->Offset(device_addr)};
            }
            if (StreamWhilePassOpen(device_addr, size)) {
                // O1 / C6 step 1: a small read-only range of a buffer that needs a synchronization while a render pass is open goes through the stream buffer
                // (no copy command, so the pass is not cut); the buffer stays unsynchronized and the next use outside a pass synchronizes it
                Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::StreamCopy, size);
                return StreamCopy(device_addr, size);
            }
            const u32 generation = immutable->generation.load(std::memory_order_acquire);
            const u32 whole = static_cast<u32>(immutable->end - immutable->begin);
            const auto* arena = BufferFor(immutable->begin, immutable->end - immutable->begin);
            last_upload_bytes = 0;
            const bool immutable_pass_open = Core::NativeUploads::OnDemandAudit() && scheduler.IsRendering();
            SynchronizeMemory(arena, immutable->begin, whole, false, false);
            if (last_upload_bytes != 0 && Core::NativeUploads::OnDemandAudit()) {
                Core::NativeUploads::NoteOnDemand(immutable_pass_open, whole, last_upload_bytes, immutable->begin, true);
            }
            // R1: the object keeps where it is resident (read by the command processor's binding resolution once `synced` is set)
            NoteResolutionUser(arena, immutable->id);
            immutable->resident_buffer.store(arena, std::memory_order_relaxed);
            immutable->resident_base.store(arena->cpu_addr, std::memory_order_relaxed);
            immutable->resident_epoch.store(PlacementEpoch(), std::memory_order_relaxed);
            Core::NativeResources::ImmutableSynced(*immutable, generation);
            Core::NativeResources::NoteImmutableSync();
            if (immutable->usage != 1) {
                Core::NativeResources::NoteResident(1);
            }
            return {arena, arena->Offset(device_addr)};
        }
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && ((size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) || (!is_texel_buffer && StreamWhilePassOpen(device_addr, size)))) {
        // Stream copies are at most STREAM_THRESHOLD bytes, so the size fits in the low 16 bits.
        const u64 reuse_key = (device_addr << 16) | size;
        auto& reuse = stream_reuse_sets[Vulkan::Scheduler::SideSelected() ? 1 : 0];
        if (StreamReuseOn()) {
            const u64 tick = scheduler.CurrentTick();
            const u64 wraps = stream_buffer.WrapCount();
            if (tick != reuse.tick || wraps != reuse.wraps) {
                // A new command buffer: the game may have changed the data since. A wrap of the
                // stream buffer may overwrite earlier copies.
                reuse.tick = tick;
                reuse.wraps = wraps;
                reuse.map.clear();
                reuse.min = ~0ULL;
                reuse.max = 0;
            } else if (const auto it = reuse.map.find(reuse_key); it != reuse.map.end()) {
                Core::NativeResources::NoteBufferRequest(device_addr, size, false,
                                                         Core::NativeResources::BufferPath::StreamReuse, 0);
                return {&stream_buffer, it->second};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        Core::NativeResources::NoteBufferRequest(device_addr, size, false,
                                                 Core::NativeResources::BufferPath::StreamCopy, size);
        if (StreamReuseOn() && stream_buffer.WrapCount() == reuse.wraps) {
            reuse.map.emplace(reuse_key, offset);
            reuse.min = std::min<VAddr>(reuse.min, device_addr);
            reuse.max = std::max<VAddr>(reuse.max, device_addr + size);
        }
        return {&stream_buffer, offset};
    }
    const bool memoizable = !is_texel_buffer && VideoCore::ReadbackPolicy::Get().native_draw_mode.load(
                                                    std::memory_order_relaxed) >= 2;
    ObtainMemo* memo = nullptr;
    u64 epoch_before = 0;
    if (memoizable) {
        if (!obtain_memo) {
            obtain_memo = std::make_unique<ObtainMemo[]>(ObtainMemoSlots);
        }
        const u64 hash = (device_addr * 0x9E3779B97F4A7C15ull) ^ (u64(size) * 0xC2B2AE3D27D4EB4Full) ^ (is_written ? 0x55 : 0);
        memo = &obtain_memo[(hash >> 20) & (ObtainMemoSlots - 1)];
        epoch_before = memory_tracker->Epoch() + range_epoch.load(std::memory_order_acquire) + b9_epoch.load(std::memory_order_acquire);
        if (memo->addr == device_addr && memo->size == size && memo->flags == (is_written ? 1u : 0u) &&
            memo->epoch == epoch_before && memo->tick == scheduler.CurrentTick()) {
            Core::NativeResources::NoteBufferRequest(device_addr, size, is_written,
                                                     Core::NativeResources::BufferPath::Memo, 0);
            return {memo->arena, memo->offset};
        }

    }
    const auto* arena = BufferFor(device_addr, size, true, is_written); // B9.4: a placed engine buffer's own Vulkan buffer, or the arena
    last_upload_bytes = 0;
    const bool pass_open_before = Core::NativeUploads::OnDemandAudit() && scheduler.IsRendering(); // the upload itself ends the pass
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (last_upload_bytes != 0 && Core::NativeUploads::OnDemandAudit()) {
        Core::NativeUploads::NoteOnDemand(pass_open_before, size, last_upload_bytes, device_addr, false);
    }
    Core::NativeResources::NoteBufferRequest(device_addr, size, is_written,
                                             Core::NativeResources::BufferPath::Arena, last_upload_bytes);
    if (is_written) {
        std::unique_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.Add(device_addr, size);
        Core::NativeResources::NoteGpuWritten(device_addr, size); // R1b: copies of engine buffers with these bytes are stale
        if (Core::NativeAudit::Enabled()) {
            Core::NativeAudit::NoteGpuWrite(device_addr, size);
        }
    } else if (Core::NativeAudit::Enabled() && !is_texel_buffer && !IsRegionGpuModified(device_addr, size)) {
        Core::NativeAudit::CheckClean(device_addr, size);
    }
    const u64 offset = arena->Offset(device_addr);
    if (memo) {
        // the epoch read before the work: an upload this call did makes the entry stale, as it should
        *memo = {device_addr, size, is_written ? 1u : 0u, arena, offset, epoch_before, scheduler.CurrentTick()};
    }
    return {arena, offset};
}

u32 BufferCache::WaitForExplicitDownloads(VAddr device_addr, u64 size, u32 timeout_ms) {
    std::unique_lock lock{explicit_mutex};
    const auto overlaps = [&] {
        return std::ranges::any_of(explicit_inflight, [&](const ExplicitDownload& download) {
            return download.addr < device_addr + size && device_addr < download.addr + download.size;
        });
    };
    if (!overlaps()) {
        return 0;
    }
    return explicit_landed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return !overlaps(); }) ? 1 : 2;
}

std::pair<const Buffer*, u64> BufferCache::StreamCopy(VAddr device_addr, u32 size) {
    const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
    memory->CopySparseMemory(device_addr, data, size);
    stream_buffer.Commit();
    return {&stream_buffer, offset};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForDownload(VAddr device_addr, u32 size) {
    const auto* arena = BufferFor(device_addr, size);
    last_upload_bytes = 0;
    SynchronizeMemory(arena, device_addr, size, false, false);
    Core::NativeResources::NoteBufferRequest(device_addr, size, false, Core::NativeResources::BufferPath::Arena, last_upload_bytes);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        last_image_source_gpu = true;
        return ObtainBuffer(device_addr, size, false);
    }
    last_image_source_gpu = false;
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

void BufferCache::ForgetShadows(VAddr begin, VAddr end) {
    // B9.1 verify: the CPU copies of the pages are rebuilt (a placement or a move fills a Vulkan buffer outside the uploads the copies follow)
    if (b9_shadow.empty()) {
        return;
    }
    for (VAddr page = begin & ~VAddr{4095}; page < end; page += 4096) {
        b9_shadow.erase(page);
    }
}

const Buffer* BufferCache::BufferFor(VAddr address, u64 size, bool demote_spanning, bool written) {
    if (const PlacedBuffer* entry = PlacedAt(address, size, written)) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedServed);
        return entry->buffer.get();
    }
    if (!demote_spanning && PlacedMeets(address, size)) {
        return nullptr; // a range over placed buffers' edges is left to their own requests
    }
    // the block the request needs: the engine buffers it meets and the placed buffers it meets, grown until no buffer reaches over its edges
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    static thread_local std::vector<u64> textures;
    VAddr lo = address, hi = address + size;
    for (u32 round = 0; round < 4; ++round) {
        parts.clear();
        Core::NativeDevice::BuffersIn(lo, hi - lo, parts);
        VAddr new_lo = lo, new_hi = hi;
        for (const auto& part : parts) {
            new_lo = std::min(new_lo, part.object_begin);
            new_hi = std::max(new_hi, part.object_end);
        }
        for (auto it = placed.upper_bound(new_lo) == placed.begin() ? placed.begin() : std::prev(placed.upper_bound(new_lo)); it != placed.end() && it->first < new_hi; ++it) {
            if (it->second.end > new_lo) {
                new_lo = std::min(new_lo, it->second.begin);
                new_hi = std::max(new_hi, it->second.end);
            }
        }
        if (new_lo == lo && new_hi == hi) {
            break;
        }
        lo = new_lo;
        hi = new_hi;
    }
    const bool any_buffer = !parts.empty();
    // members may overlap each other and textures may lie in the block: one Vulkan buffer is one memory range
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.object_begin < b.object_begin; });
    std::vector<u64> ids;
    VAddr last_end = 0;
    bool overlapping = false;
    for (const auto& part : parts) {
        overlapping |= part.object_begin < last_end;
        last_end = std::max(last_end, part.object_end);
        ids.push_back(part.id);
    }
    textures.clear();
    (void)Core::NativeDevice::TexturesIn(lo, hi - lo, textures);
    if (!any_buffer) {
        // E4: memory no engine buffer owns gets a pseudo resource of its own: the frame allocator's ring as a whole (the engine's allocator hook found it),
        // any other range as the request's block (grown over the placed buffers it meets)
        const auto& ring = VideoCore::UntrackedRing::Get();
        if (ring.Active() && address >= ring.Start() && address + size <= ring.End() && !PlacedMeets(ring.Start(), ring.End() - ring.Start())) {
            lo = ring.Start();
            hi = ring.End();
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4PseudoRing);
        }
        ids.push_back((u64{1} << 63) | lo);
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4PseudoPlaced);
    } else if (ids.size() > 1 || PlacedMeets(lo, hi - lo) || lo != parts.front().object_begin || hi != parts.back().object_end) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan, "B9.4: request {:#x}+{:#x} places a block {:#x}+{:#x} of {} engine buffers", address, size, lo, hi - lo, ids.size());
        }
    }
    if (hi - lo > (u64{256} << 20)) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan, "E4: request {:#x}+{:#x} places a block of {} MB", address, size, (hi - lo) >> 20);
        }
    }
    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9PlacedServed);
    if (overlapping) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BlockAliased);
    }
    if (!textures.empty()) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BlockWithTextures);
    }
    return Place(std::move(ids), lo, hi);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    if (Core::NativeUpdates::B9TrackerOff()) {
        // B9.3 step 2: the tracker is not kept for engine buffers: a range over several of them is answered by theirs, the bytes between them by the tracker
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        Core::NativeDevice::BuffersIn(addr, size, parts);
        if (parts.size() > 1 || (parts.size() == 1 && (parts[0].begin != addr || parts[0].end != addr + size))) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuQueries);
            std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
            VAddr at = addr;
            {
                std::scoped_lock lock{object_dirty_mutex};
                if (b9_unattributed.Intersects(addr, size)) {
                    return true;
                }
                for (const auto& part : parts) {
                    if (const auto it = object_dirty.find(part.id); it != object_dirty.end() && it->second.Intersects(part.begin, part.end - part.begin)) {
                        return true;
                    }
                }
            }
            for (const auto& part : parts) {
                if (part.begin > at && memory_tracker->IsRegionCpuModified(at, part.begin - at)) {
                    return true;
                }
                at = std::max<VAddr>(at, part.end);
            }
            return at < addr + size && memory_tracker->IsRegionCpuModified(at, addr + size - at);
        }
    }
    // B9.3: a range inside one engine buffer: changed when the buffer holds announced bytes there that were not uploaded
    if (const u32 mode = Core::NativeUpdates::B9QueryMode(); mode != 0 && Core::NativeUpdates::B9BufferMode() != 0) {
        if (const u64 id = BufferHolding(addr, size); id != 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuQueries);
            bool object_says;
            {
                std::scoped_lock lock{object_dirty_mutex};
                const auto it = object_dirty.find(id);
                object_says = (it != object_dirty.end() && it->second.Intersects(addr, size)) || b9_unattributed.Intersects(addr, size);
            }
            if (mode == 2) {
                return object_says;
            }
            const bool tracker_says = memory_tracker->IsRegionCpuModified(addr, size);
            if (tracker_says == object_says) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuAgree);
            } else if (object_says) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9CpuObjectOnly);
            } else {
                const VAddr lo = addr & ~VAddr{4095}, hi = (addr + size + 4095) & ~VAddr{4095};
                bool neighbor;
                {
                    std::scoped_lock lock{object_dirty_mutex};
                    neighbor = all_object_dirty.Intersects(lo, hi - lo);
                }
                Core::NativeUpdates::NoteB9(neighbor ? Core::NativeUpdates::B9CpuTrackerNeighbor : Core::NativeUpdates::B9CpuTrackerOnly);
            }
            return tracker_says;
        }
    }
    HarvestWriteWatch(addr, size);
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (Core::NativeUpdates::B9TrackerOff()) {
        // B9.3 step 2: a range that meets engine buffers is answered by the GPU written bytes (the tracker has no GPU marks for them any more)
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        Core::NativeDevice::BuffersIn(addr, size, parts);
        if (!parts.empty() && BufferHolding(addr, size) == 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GpuQueries);
            std::shared_lock lock{gpu_ranges_mutex};
            return gpu_modified_ranges.Intersects(addr, size) || memory_tracker->IsRegionGpuModified(addr, size);
        }
    }
    // B9.3: a range inside one engine buffer: written when the GPU wrote bytes of it that were not superseded or downloaded since
    if (const u32 mode = Core::NativeUpdates::B9QueryMode(); mode != 0 && Core::NativeUpdates::B9BufferMode() != 0) {
        if (BufferHolding(addr, size) != 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GpuQueries);
            bool bytes_say;
            {
                std::shared_lock lock{gpu_ranges_mutex};
                bytes_say = gpu_modified_ranges.Intersects(addr, size);
            }
            if (mode == 2) {
                return bytes_say;
            }
            const bool tracker_says = memory_tracker->IsRegionGpuModified(addr, size);
            if (tracker_says == bytes_say) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GpuAgree);
            } else if (bytes_say) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9GpuBytesOnly);
            } else {
                const VAddr lo = addr & ~VAddr{4095}, hi = (addr + size + 4095) & ~VAddr{4095};
                bool page;
                {
                    std::shared_lock lock{gpu_ranges_mutex};
                    page = gpu_modified_ranges.Intersects(lo, hi - lo);
                }
                Core::NativeUpdates::NoteB9(page ? Core::NativeUpdates::B9GpuTrackerPage : Core::NativeUpdates::B9GpuTrackerOnly);
            }
            return tracker_says;
        }
    }
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SupersedeGpuWritten(VAddr addr, size_t size) {
    std::unique_lock lock{gpu_ranges_mutex};
    BindMemoBump(); // GPU written bytes change hands: memo answers are checked again
    gpu_modified_ranges.Subtract(addr, size);
}

bool BufferCache::IsRangeGpuWritten(VAddr addr, size_t size) {
    std::shared_lock lock{gpu_ranges_mutex};
    return gpu_modified_ranges.Intersects(addr, size);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    HarvestWriteWatch(device_addr, size);
    // B9.4: a placed buffer holds [begin, end) of its block; a read into its slack must not take (and lose) the bytes of the memory after it
    VAddr held_lo = device_addr, held_hi = device_addr + size;
    {
        std::scoped_lock lock{placed_mutex};
        for (auto it = placed.upper_bound(arena->cpu_addr + 255); it != placed.begin();) {
            --it;
            if (it->second.buffer.get() == arena) {
                held_lo = std::max<VAddr>(held_lo, it->second.begin);
                held_hi = std::min<VAddr>(held_hi, it->second.end);
                break;
            }
            if (it->first < arena->cpu_addr) {
                break;
            }
        }
        if (held_lo >= held_hi) {
            return false;
        }
    }
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const auto add_piece = [&](u64 addr, u64 copy_size) {
        {
            // B9.4: a placed engine buffer takes only the memory it holds (a page range of the tracker, or a read past its end into the slack, can reach beyond)
            const u64 lo = std::max<u64>(addr, held_lo), hi = std::min<u64>(addr + copy_size, held_hi);
            if (lo >= hi) {
                return;
            }
            addr = lo;
            copy_size = hi - lo;
        }
        copies.emplace_back(total_size_bytes, addr, copy_size);
        total_size_bytes += copy_size;
        if (Core::WriteWatch::VerifyOn()) {
            // write watch verify: the page's CPU mark is cleared before its new upload hash exists; no hash meanwhile
            Core::WriteWatch::NoteGpuSideWrite(addr, copy_size);
        }
    };
    // B5 / B6: bytes the GPU wrote (and nothing the CPU wrote since) are the GPU's: a page upload of PS4 memory leaves them out
    static const bool keep_gpu_bytes = Core::NativeUpdates::B2NativeMode() != 0;
    const auto add_copy = [&](u64 addr, u64 copy_size) {
        if (!keep_gpu_bytes) {
            add_piece(addr, copy_size);
            return;
        }
        std::shared_lock lock{gpu_ranges_mutex};
        if (!gpu_modified_ranges.Intersects(addr, copy_size)) {
            lock.unlock();
            add_piece(addr, copy_size);
            return;
        }
        upload_bytes_kept_for_gpu.fetch_add(1, std::memory_order_relaxed);
        boost::container::small_vector<std::pair<u64, u64>, 8> pieces;
        gpu_modified_ranges.ForEachNotInRange(addr, copy_size, [&](u64 a, u64 n) { pieces.emplace_back(a, n); });
        lock.unlock();
        for (const auto& [a, n] : pieces) {
            add_piece(a, n);
        }
    };
    // B9.1: an upload over engine buffers: their announced bytes against the page tracker's ranges (B9.4 step 2: a request may cover several buffers of a block;
    // the bytes between them, which no buffer holds, keep the tracker)
    const u32 b9 = Core::NativeUpdates::B9BufferMode();
    u64 b9_object = 0; // the first engine buffer the request meets (0: none)
    bool b9_cpu_read = false; // a buffer the CPU reads (staging, CPU read access): its memory also takes downloads, which the verify's copy does not see
    boost::container::small_vector<std::pair<u64, u64>, 8> b9_pieces;
    boost::container::small_vector<std::pair<u64, u64>, 4> b9_gaps; // the request's bytes no engine buffer holds
    if (b9 != 0) {
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        Core::NativeDevice::BuffersIn(held_lo, held_hi - held_lo, parts);
        if (parts.empty()) {
            // the walk by memory found nothing: the buffer holding the request, found the long way (the request's bytes must not be left to the tracker)
            Core::NativeDevice::Object object;
            if (Core::NativeDevice::ResourceAt(device_addr, object) && object.kind == Core::NativeDevice::Kind::Buffer) {
                const VAddr object_end = object.address + std::max<u64>(object.bytes, 1);
                parts.push_back({object.id, held_lo, std::min<VAddr>(object_end, held_hi), object.usage, object.cpu_access, object.address, object_end});
            }
        }
        if (!parts.empty()) {
            std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
            b9_object = parts.front().id;
            VAddr at = held_lo;
            for (const auto& part : parts) {
                b9_cpu_read |= part.usage == 3 || (part.cpu_access & 0x20000) != 0;
                if (part.begin > at) {
                    b9_gaps.emplace_back(at, part.begin - at);
                }
                at = std::max<VAddr>(at, part.end);
            }
            if (at < held_hi) {
                b9_gaps.emplace_back(at, held_hi - at);
            }
            std::scoped_lock lock{object_dirty_mutex};
            for (const auto& part : parts) {
                if (const auto it = object_dirty.find(part.id); it != object_dirty.end()) {
                    it->second.ForEachInRange(part.begin, part.end - part.begin, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); }); // (start, end) pairs
                    it->second.Subtract(part.begin, part.end - part.begin);
                }
                b9_unattributed.ForEachInRange(part.begin, part.end - part.begin, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); });
                b9_unattributed.Subtract(part.begin, part.end - part.begin);
            }
            for (const auto& [a, n] : b9_pieces) {
                all_object_dirty.Subtract(a, n);
            }
        }
        Core::NativeUpdates::NoteB9(b9_object != 0 ? Core::NativeUpdates::B9SyncByObject : Core::NativeUpdates::B9SyncNoObject);
    }
    // the parts of [addr, addr + n) between the engine buffers of the request (they keep the tracker's copies in native mode)
    const auto for_each_gap = [&](u64 addr, u64 n, auto&& fn) {
        for (const auto& [g, gn] : b9_gaps) {
            const u64 lo = std::max<u64>(addr, g), hi = std::min<u64>(addr + n, g + gn);
            if (lo < hi) {
                fn(lo, hi - lo);
            }
        }
    };
    const auto b9_forget = [&](u64 addr, u64 n) {
        // audit: bytes an upload of no engine buffer took (texture memory, unowned memory) leave the announced set
        if (b9 != 0 && b9_object == 0) {
            std::scoped_lock lock{object_dirty_mutex};
            all_object_dirty.Subtract(addr, n);
            b9_unattributed.Subtract(addr, n);
        }
    };
    const bool b9_native = b9 == 2 && b9_object != 0;
    // B9.3 step 2: an upload inside one engine buffer neither asks nor clears the page tracker
    const bool b9_no_tracker = b9_native && Core::NativeUpdates::B9TrackerOff();
    if (b9_no_tracker) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerUploadsSkipped);
    }
    boost::container::small_vector<bool, 8> b9_piece_seen(b9_pieces.size(), false);
    // E4 census (TLG_E4_CENSUS=1): over memory no engine buffer holds the page tracker still picks the uploads; this compares its ranges with the announced bytes
    static const bool e4_census = [] {
        const char* e = std::getenv("TLG_E4_CENSUS");
        return e != nullptr && *e == '1';
    }();
    boost::container::small_vector<std::pair<u64, u64>, 8> e4_announced, e4_tracked;
    const bool e4_counting = e4_census && b9_object == 0 && !(UntrackedRing::Get().Active() && [&] {
        const auto [lo, hi] = UntrackedRing::Get().Clip(device_addr, size);
        return lo < hi;
    }());
    if (e4_counting) {
        std::scoped_lock lock{object_dirty_mutex};
        all_object_dirty.ForEachInRange(held_lo, held_hi - held_lo, [&](u64 a, u64 e) { e4_announced.emplace_back(a, e - a); });
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusRequests);
    }
    const auto e4_note_tracker = [&](u64 x, u64 xn) {
        if (!e4_counting) {
            return;
        }
        e4_tracked.emplace_back(x, xn);
        std::scoped_lock lock{object_dirty_mutex};
        for (u64 page = x & ~u64{4095}; page < x + xn; page += 4096) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusTrackerPages);
            static std::unordered_set<u64> seen;
            const bool first = seen.insert(page).second;
            if (!all_object_dirty.Intersects(page, 4096)) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusPagesNoAnnouncement);
                if (first) {
                    Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusFirstPages);
                    continue;
                }
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusRepeatNoAnnouncement);
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render_Vulkan, "E4 census: tracker page {:#x} (request {:#x}+{:#x}{}) has no announced byte ({})", page, device_addr, size, is_written ? ", written" : "",
                                Core::NativeDevice::OwnerName(page));
                }
            }
        }
    };
    // E4: bytes a page tracker range still uploads, named (first 24)
    const auto e4_tracker_upload = [&](u64 x, u64 xn, const char* how) {
        Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4TrackerUploads, xn);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
            LOG_WARNING(Render_Vulkan, "E4: tracker upload ({}) {:#x}+{:#x} ({}) request {:#x}+{:#x}{} object {}", how, x, xn, Core::NativeDevice::OwnerName(x), device_addr, size,
                        is_written ? ", written" : "", b9_object);
        }
    };
    // a page tracker range (the copies of the tracker): compared with the announced bytes; native mode leaves the copy to the announcements
    const auto tracker_copy = [&](u64 addr, u64 copy_size) {
        e4_note_tracker(addr, copy_size);
        if (b9_object != 0) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BytesByTracker, copy_size);
            bool any = false;
            for (size_t i = 0; i < b9_pieces.size(); ++i) {
                const auto& [a, n] = b9_pieces[i];
                if (a < addr + copy_size && addr < a + n) {
                    any = true;
                    b9_piece_seen[i] = true;
                }
            }
            if (!any) {
                bool elsewhere;
                {
                    std::scoped_lock lock{object_dirty_mutex};
                    elsewhere = all_object_dirty.Intersects(addr, copy_size);
                }
                if (elsewhere) {
                    Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerElsewhere);
                    any = true;
                }
            }
            if (!any) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerOnlyRanges);
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9TrackerOnlyBytes, copy_size);
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render_Vulkan, "B9.1: page tracker range {:#x}+{:#x} of engine buffer {} ({}) has no announced byte (request {:#x}+{:#x}{})", addr,
                                copy_size, b9_object, Core::NativeDevice::OwnerName(addr), device_addr, size, is_written ? ", written" : "");
                }
            }
            if (b9_native) {
                for_each_gap(addr, copy_size, [&](u64 a, u64 n) { e4_tracker_upload(a, n, "native gap"); add_copy(a, n); });
                return;
            }
        }
        b9_forget(addr, copy_size);
        e4_tracker_upload(addr, copy_size, "copy");
        add_copy(addr, copy_size);
    };
    // E4 (`TLG_E4_ANNOUNCED=0` turns it off): memory no engine buffer holds takes the announced bytes (the placement already copied the rest). The tracker is still asked so its state
    // (CPU/GPU marks) stays as it was, but the ranges it returns are not uploaded.
    static const bool e4_from_announced = [] {
        const char* e = std::getenv("TLG_E4_ANNOUNCED");
        return e == nullptr || *e != '0'; // default on (owner play 2026-10-31); =0 is the way back
    }();
    // the announced bytes inside [x, x + xn) of memory no engine buffer holds (a request over unowned memory, or a gap between the buffers of one): uploaded, then no longer announced
    const auto take_announced = [&](u64 x, u64 xn) {
        boost::container::small_vector<std::pair<u64, u64>, 8> from_announced;
        {
            std::scoped_lock lock{object_dirty_mutex};
            all_object_dirty.ForEachInRange(x, xn, [&](u64 lo, u64 e) { from_announced.emplace_back(lo, e - lo); });
            for (const auto& [lo, n] : from_announced) {
                all_object_dirty.Subtract(lo, n);
                b9_unattributed.Subtract(lo, n);
            }
        }
        for (const auto& [lo, n] : from_announced) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4AnnouncedUploads, n);
            add_copy(lo, n);
        }
    };
    const auto tracker_ranges = [&](VAddr a, u64 n) {
        if (e4_from_announced && b9 != 0) {
            // the tracker is still asked so its marks stay as they were, but its ranges are not uploaded
            if (b9_object == 0) {
                memory_tracker->ForEachUploadRange(a, n, is_written, [&](u64 x, u64 xn) { e4_note_tracker(x, xn); });
                take_announced(a, n);
            } else {
                for_each_gap(a, n, [&](u64 g, u64 gn) {
                    memory_tracker->ForEachUploadRange(g, gn, is_written, [&](u64 x, u64 xn) { e4_note_tracker(x, xn); });
                    take_announced(g, gn);
                });
            }
            return;
        }
        if (!b9_no_tracker) {
            memory_tracker->ForEachUploadRange(a, n, is_written, tracker_copy);
            return;
        }
        // the tracker is not kept for engine buffers: it is asked only for the bytes between them
        for_each_gap(a, n, [&](u64 g, u64 gn) { memory_tracker->ForEachUploadRange(g, gn, is_written, [&](u64 x, u64 xn) { e4_tracker_upload(x, xn, "gap"); add_copy(x, xn); }); });
    };
    const auto& ring = UntrackedRing::Get();
    VAddr ring_lo = 0, ring_hi = 0; // the part the untracked ring uploads whole (native mode adds no announced piece there)
    if (ring.Active() && !is_written) {
        const auto [lo, hi] = ring.Clip(device_addr, size);
        if (lo < hi && !(b9_no_tracker ? IsRegionGpuModified(lo, hi - lo) : memory_tracker->IsRegionGpuModified(lo, hi - lo))) {
            ring_lo = lo;
            ring_hi = hi;
            // Untracked upload ring: nobody told us what the CPU wrote, so the requested part is uploaded.
            if (device_addr < lo) {
                tracker_ranges(device_addr, lo - device_addr);
            }
            // Read again in the same command buffer: the data cannot have changed since the first upload.
            auto& done = ring_uploads[(lo >> 6) & 255];
            const u64 tick = Vulkan::Scheduler::SideSelected() ? (side_upload_epoch | Vulkan::Scheduler::SideTag) : scheduler.CurrentTick();
            if (done.tick != tick || done.lo > lo || done.hi < hi) {
                add_copy(lo, hi - lo);
                Core::NativeUploads::NoteRingUpload(hi - lo);
                done = {lo, hi, tick};
            } else {
                Core::NativeUploads::NoteRingSkipped(hi - lo);
            }
            if (hi < device_addr + size) {
                tracker_ranges(hi, device_addr + size - hi);
            }
        } else {
            tracker_ranges(device_addr, size);
        }
    } else {
        tracker_ranges(device_addr, size);
    }
    if (e4_counting) {
        RangeSet left;
        for (const auto& [a, n] : e4_announced) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusAnnouncedBytes, n);
            left.Add(a, n);
        }
        for (const auto& [a, n] : e4_tracked) {
            left.Subtract(a, n);
        }
        left.ForEachInRange(held_lo, held_hi - held_lo, [&](u64 a, u64 e) { Core::NativeUpdates::NoteB9(Core::NativeUpdates::E4CensusAnnouncedNotUploaded, e - a); });
    }
    if (b9_object != 0) {
        for (size_t i = 0; i < b9_pieces.size(); ++i) {
            auto [a, n] = b9_pieces[i];
            if (!b9_piece_seen[i]) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9ObjectOnlyRanges);
            }
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9BytesByObject, n);
            if (!b9_native) {
                continue;
            }
            // native: the announced bytes are the upload (outside the part the untracked ring uploads whole)
            const u64 end = a + n;
            if (ring_lo < ring_hi && a < ring_hi && ring_lo < end) {
                if (a < ring_lo) {
                    add_copy(a, ring_lo - a);
                }
                if (ring_hi < end) {
                    add_copy(ring_hi, end - ring_hi);
                }
                continue;
            }
            add_copy(a, n);
        }
    }
    if (b9_native && !b9_cpu_read && Core::NativeUpdates::B9VerifyEvery() != 0) {
        // B9.1 verify: the sampled pages of the request; their CPU copy takes what this upload copies, then the request's bytes must equal memory
        const u32 every = Core::NativeUpdates::B9VerifyEvery();
        const auto sampled = [&](u64 page) { return (page >> 12) % every == 0; };
        // guest bytes are read as the upload reads them (memory that is not mapped for the CPU reads as zeros)
        std::array<u8, 4096> now{};
        const size_t num_copies = copies.size();
        for (size_t i = 0; i < num_copies; ++i) {
            const u64 a = copies[i].dstOffset, e = a + copies[i].size;
            for (u64 page = a & ~u64{4095}; page < e; page += 4096) {
                if (!sampled(page)) {
                    continue;
                }
                if (const auto it = b9_shadow.find(page); it != b9_shadow.end()) {
                    const u64 lo = std::max(a, page), hi = std::min(e, page + 4096);
                    memory->CopySparseMemory(lo, it->second->data() + (lo - page), hi - lo);
                }
            }
        }
        const u64 req_end = held_hi; // the bytes the destination holds (a placed buffer's slack is not memory's)
        for (u64 page = held_lo & ~u64{4095}; page < req_end; page += 4096) {
            if (!sampled(page)) {
                continue;
            }
            const u64 lo = std::max<u64>(held_lo, page), hi = std::min<u64>(req_end, page + 4096);
            bool gpu_written;
            {
                std::shared_lock lock{gpu_ranges_mutex};
                gpu_written = gpu_modified_ranges.Intersects(lo, hi - lo);
            }
            auto it = b9_shadow.find(page);
            if (gpu_written) {
                if (it != b9_shadow.end()) {
                    b9_shadow.erase(it);
                }
                continue;
            }
            if (it == b9_shadow.end()) {
                if (b9_shadow.size() >= 8192) {
                    continue;
                }
                // a new sampled page: memory as it is now. The check only reads: a change later that no upload took is a write nobody announced (it must not change what the
                // GPU sees: uploading here put memory's bytes over a buffer's own, garbling the picture of a new game)
                auto copy = std::make_unique<std::array<u8, 4096>>();
                memory->CopySparseMemory(lo, copy->data() + (lo - page), hi - lo);
                b9_shadow.emplace(page, std::move(copy));
                continue;
            }
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9VerifyChecked);
            memory->CopySparseMemory(lo, now.data() + (lo - page), hi - lo);
            if (std::memcmp(it->second->data() + (lo - page), now.data() + (lo - page), hi - lo) != 0) {
                Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9VerifyMiss);
                {
                    std::scoped_lock lock{object_dirty_mutex};
                    b9_owed.Add(lo, hi - lo);
                }
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    u64 first = lo;
                    while (first < hi && it->second->at(first - page) == now[first - page]) {
                        ++first;
                    }
                    LOG_WARNING(Render_Vulkan, "B9.1 verify: engine buffer {} request {:#x}+{:#x}: byte {:#x} differs from memory after the upload ({})", b9_object, device_addr,
                                size, first, Core::NativeDevice::OwnerName(first));
                }
                std::memcpy(it->second->data() + (lo - page), now.data() + (lo - page), hi - lo);
            }
        }
    }
    last_upload_bytes += total_size_bytes;
    Core::NativeUploads::NoteUploaded(total_size_bytes);
    if (Core::NativeUploads::Enabled()) {
        for (const auto& copy : copies) {
            Core::NativeUploads::NoteUploadedRange(copy.dstOffset, copy.size);
        }
    }
    if (!copies.empty()) {
        // where uploads go: the arena (memory no placed buffer holds), a placed buffer in plain device memory, a placed buffer the CPU can write
        zero_copy_counts[arena->mem_type == MemoryType::DeviceMapped ? 9 : 8].fetch_add(1, std::memory_order_relaxed);
    }
    if (!copies.empty() && arena->mem_type == MemoryType::DeviceMapped && arena->is_device_local && !arena->mapped_data.empty() && arena->CpuAddr() != 0) {
        // zero copy: the GPU is done with this engine buffer (every submission that used it finished): its bytes are written straight into its video memory,
        // with no staging and no copy in the stream (the submission that follows makes the host writes visible). Otherwise the copy below runs as before.
        const bool idle = (arena->last_gpu_tick[0] == 0 || scheduler.IsFree(arena->last_gpu_tick[0])) &&
                          (arena->last_gpu_tick[1] == 0 || scheduler.IsFree(arena->last_gpu_tick[1]));
        if (idle) {
            u8* const base = const_cast<u8*>(arena->mapped_data.data());
            for (const auto& copy : copies) {
                memory->CopySparseMemory(copy.dstOffset, base + (copy.dstOffset - arena->cpu_addr), copy.size);
            }
            if (!arena->is_coherent) {
                const_cast<Buffer*>(arena)->Flush(0, arena->SizeBytes());
            }
            zero_copy_counts[0].fetch_add(1, std::memory_order_relaxed);
            zero_copy_counts[1].fetch_add(copies.size(), std::memory_order_relaxed);
            zero_copy_counts[2].fetch_add(total_size_bytes, std::memory_order_relaxed);
            ZeroCopyVerifyNote(arena, copies);
            copies.clear();
        } else {
            zero_copy_counts[3].fetch_add(1, std::memory_order_relaxed);
            zero_copy_counts[4].fetch_add(total_size_bytes, std::memory_order_relaxed);
        }
    }
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        if (Core::NativeAudit::Enabled()) {
            for (const auto& copy : copies) {
                Core::NativeAudit::NoteUpload(copy.dstOffset + arena->cpu_addr, staging.mapped + (copy.srcOffset - staging.offset), copy.size);
            }
        }
        if (Core::WriteWatch::VerifyOn()) {
            for (const auto& copy : copies) {
                Core::WriteWatch::NoteUpload(copy.dstOffset + arena->cpu_addr, staging.mapped + (copy.srcOffset - staging.offset), copy.size);
            }
        }
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

} // namespace VideoCore
