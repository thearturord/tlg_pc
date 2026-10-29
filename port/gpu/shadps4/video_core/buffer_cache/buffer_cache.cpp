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
#include "core/native_audit.h"
#include "core/native_device.h"
#include "core/native_readbacks.h"
#include "core/native_registry.h"
#include "core/native_updates.h"
#include "core/native_resources.h"
#include "core/resource_audit.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/readback_policy.h"
#include "core/native_uploads.h"
#include "video_core/untracked_ring.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#include "core/native_write_watch.h"

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

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      stream_reuse_enabled{EmulatorSettings.IsReadbackStreamReuseEnabled()},
      async_guest_faults{EmulatorSettings.IsReadbackAsyncGuestFaultsEnabled()},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
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

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

namespace {
thread_local VAddr t_audit_begin = 0, t_audit_end = 0; ///< B2 audit: the range of the current harvest request
} // namespace

void BufferCache::ApplyCpuWrite(VAddr address, u64 size, bool newer) {
    if (newer) {
        // B2: the engine wrote these bytes after anything the GPU wrote there before (applied in order on the raster thread)
        SupersedeGpuWritten(address, size);
    }
    Core::NativeResources::NoteCpuWrite(address, size);
    memory_tracker->MarkRegionAsCpuModified(address, size);
    if (Core::NativeUpdates::B9BufferMode() != 0) {
        // B9.1: each engine buffer the bytes belong to keeps its part until its next upload (an announcement can run over several buffers)
        std::scoped_lock lock{object_dirty_mutex};
        all_object_dirty.Add(address, size); // audit: every announced byte, by address
        if (b9_owed.Intersects(address, size)) {
            Core::NativeUpdates::NoteB9(Core::NativeUpdates::B9VerifyLate);
            b9_owed.Subtract(address, size);
        }
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        Core::NativeDevice::BuffersIn(address, size, parts);
        Core::NativeUpdates::NoteB9(parts.empty() ? Core::NativeUpdates::B9AnnouncedNoBuffer
                                    : parts.size() == 1 && parts[0].begin == address && parts[0].end == address + size ? Core::NativeUpdates::B9AnnouncedOnBuffer
                                                                                                                      : Core::NativeUpdates::B9AnnouncedClipped);
        for (const auto& part : parts) {
            object_dirty[part.id].Add(part.begin, part.end - part.begin);
        }
        // the bytes no buffer holds now: the first upload of a buffer over them takes them
        b9_unattributed.Add(address, size);
        for (const auto& part : parts) {
            b9_unattributed.Subtract(part.begin, part.end - part.begin);
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

static u64 SteadyNowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

std::pair<VAddr, VAddr> BufferCache::ReadbackWindow(const Buffer* arena, VAddr device_addr,
                                                    u64 size) const {
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_start = arena->cpu_addr;
    const VAddr arena_end = arena_start + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena_start);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    return {window_start, window_end};
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    if (async_guest_faults && !assume_locks && !Vulkan::Scheduler::OnPriorityThread()) {
        ReadMemoryForGuest(device_addr, size, is_write);
        return;
    }
    const auto flush_request = [this, device_addr, size, is_write] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        const auto [window_start, window_end] = ReadbackWindow(arena, device_addr, size);
        const u64 window_size = window_end - window_start;
        ApplyPendingUnmarks();
        auto result = DownloadMemory(arena, window_start, window_size);
        const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
        const u64 page_size = page_end - page_start;
        if (result.bytes == 0 && memory_tracker->IsRegionGpuModified(page_start, page_size) &&
            IsDownloadInFlight(page_start, page_size)) {
            // The page has nothing left to download: an asynchronous readback of it was already
            // submitted. Wait for that one to land instead of finishing all GPU work.
            WaitForDownloads(page_start, page_size);
            ApplyPendingUnmarks();
        }
        FinishFaultReadback(device_addr, size, is_write, window_start, window_size, result.bytes);
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::FinishFaultReadback(VAddr device_addr, u64 size, bool is_write,
                                      VAddr window_start, u64 window_size, u64& bytes) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    const u64 page_size = page_end - page_start;
    // The faulting page must end up accessible, or the access faults again forever. Download the
    // page, then drop a GPU mark with nothing left behind it.
    if (memory_tracker->IsRegionGpuModified(page_start, page_size)) {
        const auto* arena = GetArena(page_start >> block_shift, (page_end - 1) >> block_shift);
        bytes += DownloadMemory(arena, page_start, page_size).bytes;
        // Downloads of this page have landed by now, so a mark with no GPU write behind it is
        // stale.
        const bool still_marked = memory_tracker->IsRegionGpuModified(page_start, page_size);
        UnmarkSettledPages(page_start, page_size);
        if (still_marked && !memory_tracker->IsRegionGpuModified(page_start, page_size)) {
            static u32 num_logged = 0;
            if (num_logged < 20) {
                ++num_logged;
                LOG_INFO(Render_Vulkan,
                         "Readback: cleared a stale GPU mark on {:#x}-{:#x} (fault at {:#x})",
                         page_start, page_end, device_addr);
            }
        }
    }
    const bool window_has_gpu_bytes = [&] {
        std::shared_lock lock{gpu_ranges_mutex};
        return gpu_modified_ranges.Intersects(window_start, window_size);
    }();
    if (is_write && !window_has_gpu_bytes &&
        memory_tracker->IsRegionGpuModified(window_start, window_size)) {
        // Everything in the window is in guest memory, drop any stale GPU mark before marking
        // the CPU write: a page can't be both, it would need write-only protection.
        memory_tracker->UnmarkRegionAsGpuModified(window_start, window_size, false);
    }
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::ReadMemoryForGuest(VAddr device_addr, u64 size, bool is_write) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    const u64 page_size = page_end - page_start;
    auto landed = std::make_shared<std::atomic<bool>>(false);
    bool recorded = false;
    bool in_flight = false;
    VAddr window_start{};
    u64 window_size{};
    u64 bytes = 0;

    // 1. Command processor: record the downloads of the window and of the faulting page, and
    // submit them. The write backs run on the priority thread once the GPU is done, then this
    // thread is released. Guest memory and protections are not touched yet.
    liverpool->SendCommand<true>([&] {
        ApplyPendingUnmarks();
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        const auto [win_start, win_end] = ReadbackWindow(arena, device_addr, size);
        window_start = win_start;
        window_size = win_end - win_start;
        std::vector<Common::UniqueFunction<void>> callbacks;
        const auto record = [&](VAddr range_start, u64 range_size) {
            DownloadCopies copies;
            const u64 total = CollectDownloadCopies(arena, range_start, range_size, copies);
            if (total == 0) {
                return;
            }
            bytes += total;
            callbacks.push_back(RecordDownloadCopies(arena, range_start, range_size,
                                                     std::move(copies), total, true));
        };
        record(window_start, window_size);
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr range_start = std::max<VAddr>(page_start, arena->cpu_addr);
        const VAddr range_end = std::min<VAddr>(page_end, arena_end);
        if (range_end > range_start) {
            record(range_start, range_end - range_start);
        }
        if (!callbacks.empty()) {
            recorded = true;
            scheduler.DeferPriorityOperation(
                [callbacks = std::move(callbacks), landed]() mutable {
                    for (auto& callback : callbacks) {
                        callback();
                    }
                    landed->store(true, std::memory_order_release);
                    landed->notify_all();
                },
                scheduler.CurrentTick());
            scheduler.Flush();
        } else if (memory_tracker->IsRegionGpuModified(page_start, page_size) &&
                   IsDownloadInFlight(page_start, page_size)) {
            // Nothing left to download: an asynchronous readback of the page is on its way.
            in_flight = true;
            if (scheduler.PriorityOperationsNeedSubmit()) {
                scheduler.Flush();
            }
        }
    });

    // 2. This (guest) thread waits for the data. The command processor goes on meanwhile.
    if (recorded) {
        landed->wait(false, std::memory_order_acquire);
    }
    if (in_flight) {
        while (IsDownloadInFlight(page_start, page_size)) {
            std::this_thread::yield();
        }
    }

    // 3. Command processor: apply the protection changes of the landed downloads. Anything the
    // GPU wrote to the page after step 1 is downloaded there, waiting as before (rare).
    liverpool->SendCommand<true>([&] {
        ApplyPendingUnmarks();
        FinishFaultReadback(device_addr, size, is_write, window_start, window_size, bytes);
    });
}

u64 BufferCache::CollectDownloadCopies(const Buffer* arena, VAddr device_addr, u64 size,
                                       DownloadCopies& copies) {
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 range_size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        std::unique_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.ForEachInRange(address, range_size, add_download);
        gpu_modified_ranges.Subtract(address, range_size);
        range_epoch.fetch_add(1, std::memory_order_acq_rel);
    });
    return total_size_bytes;
}

Common::UniqueFunction<void> BufferCache::RecordDownloadCopies(const Buffer* arena,
                                                               VAddr device_addr, u64 size,
                                                               DownloadCopies&& copies,
                                                               u64 total_size_bytes,
                                                               bool deferred_unmark) {
    const auto download = staging_pool.Request(total_size_bytes, MemoryType::HostCached, 0,
                                               deferred_unmark);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    u64 inflight_id = 0;
    if (deferred_unmark) {
        std::scoped_lock lk{pending_unmarks_mutex};
        inflight_id = ++next_inflight_id;
        inflight_downloads.push_back({device_addr, size, inflight_id});
    }
    return [this, copies = std::move(copies), download, arena_base = arena->cpu_addr, device_addr,
            size, deferred_unmark, inflight_id] {
        download.buffer->Invalidate(download.offset, download.size);
        for (const auto& copy : copies) {
            auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
            memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                    copy.size);
        }
        if (!deferred_unmark) {
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
            return;
        }
        // Running outside the GPU thread. The GPU may have written the range again after the
        // copy was recorded, so the unprotect is applied on the GPU thread, where writes are
        // tracked (see ApplyPendingUnmarks). Until then the range stays protected.
        staging_pool.FreeDeferred(download);
        std::scoped_lock lk{pending_unmarks_mutex};
        std::erase_if(inflight_downloads,
                      [inflight_id](const auto& item) { return item.id == inflight_id; });
        pending_unmarks.emplace_back(device_addr, size);
    };
}

BufferCache::DownloadResult BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr,
                                                        u64 size) {
    DownloadCopies copies;
    const u64 total_size_bytes = CollectDownloadCopies(arena, device_addr, size, copies);
    if (total_size_bytes == 0) {
        return {};
    }
    auto write_data = RecordDownloadCopies(arena, device_addr, size, std::move(copies),
                                           total_size_bytes, false);
    scheduler.Finish();
    write_data();
    return {.bytes = total_size_bytes};
}

void BufferCache::ApplyPendingUnmarks() {
    std::vector<std::pair<VAddr, u64>> unmarks;
    {
        std::scoped_lock lk{pending_unmarks_mutex};
        if (pending_unmarks.empty()) {
            return;
        }
        unmarks.swap(pending_unmarks);
    }
    for (const auto& [device_addr, size] : unmarks) {
        // Parts the GPU wrote again stay protected so the CPU reads the new data. The rest is
        // unprotected: skipping the whole range left pages protected with nothing left to
        // download, which made the CPU fault on them forever.
        UnmarkSettledPages(device_addr, size);
    }
}

bool BufferCache::IsDownloadInFlight(VAddr device_addr, u64 size) {
    std::scoped_lock lk{pending_unmarks_mutex};
    return std::ranges::any_of(inflight_downloads, [&](const InflightDownload& download) {
        return download.addr < device_addr + size && device_addr < download.addr + download.size;
    });
}

void BufferCache::WaitForDownloads(VAddr device_addr, u64 size) {
    // Asynchronous downloads are submitted when recorded. Submit anyway if an operation waits on
    // the command buffer being recorded, since the downloads land in order after it.
    if (scheduler.PriorityOperationsNeedSubmit()) {
        scheduler.Flush();
    }
    const u64 start = SteadyNowNs();
    bool reported = false;
    while (IsDownloadInFlight(device_addr, size)) {
        if (!reported && SteadyNowNs() - start > 2'000'000'000ULL) {
            reported = true;
            LOG_WARNING(Render_Vulkan,
                        "Hang check: waiting >2s for an asynchronous readback of {:#x}:{:#x}",
                        device_addr, size);
        }
        std::this_thread::yield();
    }
}

void BufferCache::UnmarkSettledPages(VAddr device_addr, u64 size) {
    const VAddr start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    RangeSet settled;
    settled.Add(start, end - start);
    // Only data the GPU wrote and that wasn't downloaded yet keeps a page protected. A page with
    // a newer download still in flight is unprotected too: its guest memory holds everything that
    // landed so far, and the game only sees the fence of the newer write once it has landed
    // (keeping it protected made the CPU wait for the hot ranges that are prefetched at nearly
    // every fence).
    {
        std::shared_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.ForEachInRange(start, end - start,
                                           [&](VAddr range_start, VAddr range_end) {
                                               settled.Subtract(range_start,
                                                                range_end - range_start);
                                           });
    }
    // Protection is per page: a page partly outside the settled ranges stays protected.
    settled.ForEach([&](VAddr range_start, VAddr range_end) {
        const VAddr page_start = Common::AlignUp(range_start, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignDown(range_end, BYTES_PER_PAGE);
        if (page_end > page_start) {
            memory_tracker->UnmarkRegionAsGpuModified(page_start, page_end - page_start, false);
        }
    });
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
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    last_upload_bytes = 0;
    Core::NativeUploads::SetPreSyncActive(true);
    SynchronizeMemory(arena, device_addr, static_cast<u32>(size), false, false);
    Core::NativeUploads::SetPreSyncActive(false);
    return last_upload_bytes;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
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
                (immutable->usage == 1 || !memory_tracker->IsRegionCpuModified(device_addr, size))) {
                const auto* arena = GetArena(device_addr >> block_shift, (device_addr + size - 1) >> block_shift);
                if (immutable->gpu_arena_epoch.load(std::memory_order_relaxed) != ArenaEpoch() && immutable->begin >= arena->cpu_addr &&
                    immutable->end <= arena->cpu_addr + arena->size_bytes) {
                    // R1: the arenas moved since the synchronization; the object is resident in this one now
                    immutable->gpu_arena.store(arena, std::memory_order_relaxed);
                    immutable->gpu_arena_base.store(arena->cpu_addr, std::memory_order_relaxed);
                    immutable->gpu_arena_epoch.store(ArenaEpoch(), std::memory_order_release);
                }
                Core::NativeResources::NoteImmutableHit();
                if (immutable->usage != 1) {
                    Core::NativeResources::NoteResident(0);
                    const u32 interval = Core::NativeResources::ResidentVerifyInterval();
                    static std::atomic<u64> resident_hits{0};
                    if (interval != 0 && resident_hits.fetch_add(1, std::memory_order_relaxed) % interval == 0) {
                        // verify: one synchronization of the request; anything it has to upload means the answer without it would have been stale
                        last_upload_bytes = 0;
                        EnsureResident(arena, device_addr >> block_shift, (device_addr + size - 1) >> block_shift);
                        SynchronizeMemory(arena, device_addr, size, false, false);
                        Core::NativeResources::NoteResident(2);
                        if (last_upload_bytes != 0) {
                            Core::NativeResources::NoteResident(3);
                            immutable->synced.store(false, std::memory_order_release);
                            static std::atomic<u32> logged{0};
                            if (logged.fetch_add(1) < 12) {
                                LOG_ERROR(Render, "resident buffer {:#x}+{} (engine buffer {:#x}..{:#x}, usage {}) was stale: {} bytes had to be uploaded", device_addr, size, immutable->begin, immutable->end, immutable->usage, last_upload_bytes);
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
            const auto* arena = GetArena(immutable->begin >> block_shift, (immutable->end - 1) >> block_shift);
            EnsureResident(arena, immutable->begin >> block_shift, (immutable->end - 1) >> block_shift);
            last_upload_bytes = 0;
            const bool immutable_pass_open = Core::NativeUploads::OnDemandAudit() && scheduler.IsRendering();
            SynchronizeMemory(arena, immutable->begin, whole, false, false);
            if (last_upload_bytes != 0 && Core::NativeUploads::OnDemandAudit()) {
                Core::NativeUploads::NoteOnDemand(immutable_pass_open, whole, last_upload_bytes, immutable->begin, true);
            }
            // R1: the object keeps where it is resident (read by the command processor's binding resolution once `synced` is set)
            immutable->gpu_arena.store(arena, std::memory_order_relaxed);
            immutable->gpu_arena_base.store(arena->cpu_addr, std::memory_order_relaxed);
            immutable->gpu_arena_epoch.store(ArenaEpoch(), std::memory_order_relaxed);
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
        epoch_before = memory_tracker->Epoch() + range_epoch.load(std::memory_order_acquire);
        if (memo->addr == device_addr && memo->size == size && memo->flags == (is_written ? 1u : 0u) &&
            memo->epoch == epoch_before && memo->tick == scheduler.CurrentTick()) {
            Core::NativeResources::NoteBufferRequest(device_addr, size, is_written,
                                                     Core::NativeResources::BufferPath::Memo, 0);
            return {memo->arena, memo->offset};
        }

    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
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

std::pair<const Buffer*, u64> BufferCache::StreamCopy(VAddr device_addr, u32 size) {
    const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
    memory->CopySparseMemory(device_addr, data, size);
    stream_buffer.Commit();
    return {&stream_buffer, offset};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForDownload(VAddr device_addr, u32 size) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
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

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    HarvestWriteWatch(addr, size);
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SupersedeGpuWritten(VAddr addr, size_t size) {
    std::unique_lock lock{gpu_ranges_mutex};
    gpu_modified_ranges.Subtract(addr, size);
}

bool BufferCache::IsRangeGpuWritten(VAddr addr, size_t size) {
    std::shared_lock lock{gpu_ranges_mutex};
    return gpu_modified_ranges.Intersects(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    range_epoch.fetch_add(1, std::memory_order_acq_rel); // answers that name the old arenas are stale
    arena_epoch.fetch_add(1, std::memory_order_acq_rel);
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    for (const auto& [lo, hi] : resident_range_cache) {
        if (first_block >= lo && last_block < hi) {
            return;
        }
    }
    const bool cacheable = last_block - first_block < 4;
    if (cacheable) {
        bool all_known = true;
        for (u64 block = first_block; block <= last_block; ++block) {
            if (resident_block_cache[block & 4095] != block + 1) {
                all_known = false;
                break;
            }
        }
        if (all_known) {
            return;
        }
    }
    const auto remember = [&] {
        if (cacheable) {
            for (u64 block = first_block; block <= last_block; ++block) {
                resident_block_cache[block & 4095] = block + 1;
            }
        } else {
            resident_range_cache[resident_range_next++ & 7] = {first_block, last_block + 1};
        }
    };

    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        remember();
        return;
    }

    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info = {
        .pNext = &alloc_flags,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    remember();
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    HarvestWriteWatch(device_addr, size);
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const auto add_piece = [&](u64 addr, u64 copy_size) {
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
    // B9.1: an upload inside one engine buffer: its announced bytes against the page tracker's ranges
    const u32 b9 = Core::NativeUpdates::B9BufferMode();
    u64 b9_object = 0;
    bool b9_cpu_read = false; // a buffer the CPU reads (staging, CPU read access): its memory also takes downloads, which the verify's copy does not see
    boost::container::small_vector<std::pair<u64, u64>, 8> b9_pieces;
    if (b9 != 0) {
        Core::NativeDevice::Object object;
        if (Core::NativeDevice::ResourceAt(device_addr, object) && object.kind == Core::NativeDevice::Kind::Buffer &&
            device_addr + size <= object.address + std::max<u64>(object.bytes, 1)) {
            b9_object = object.id;
            b9_cpu_read = object.usage == 3 || (object.cpu_access & 0x20000) != 0;
            std::scoped_lock lock{object_dirty_mutex};
            if (const auto it = object_dirty.find(b9_object); it != object_dirty.end()) {
                it->second.ForEachInRange(device_addr, size, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); }); // (start, end) pairs
                it->second.Subtract(device_addr, size);
            }
            b9_unattributed.ForEachInRange(device_addr, size, [&](u64 a, u64 e) { b9_pieces.emplace_back(a, e - a); });
            b9_unattributed.Subtract(device_addr, size);
            for (const auto& [a, n] : b9_pieces) {
                all_object_dirty.Subtract(a, n);
            }
        }
        Core::NativeUpdates::NoteB9(b9_object != 0 ? Core::NativeUpdates::B9SyncByObject : Core::NativeUpdates::B9SyncNoObject);
    }
    const auto b9_forget = [&](u64 addr, u64 n) {
        // audit: bytes an upload of no engine buffer took (texture memory, unowned memory) leave the announced set
        if (b9 != 0 && b9_object == 0) {
            std::scoped_lock lock{object_dirty_mutex};
            all_object_dirty.Subtract(addr, n);
            b9_unattributed.Subtract(addr, n);
        }
    };
    const bool b9_native = b9 == 2 && b9_object != 0;
    boost::container::small_vector<bool, 8> b9_piece_seen(b9_pieces.size(), false);
    // a page tracker range (the copies of the tracker): compared with the announced bytes; native mode leaves the copy to the announcements
    const auto tracker_copy = [&](u64 addr, u64 copy_size) {
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
                return;
            }
        }
        b9_forget(addr, copy_size);
        add_copy(addr, copy_size);
    };
    const auto& ring = UntrackedRing::Get();
    VAddr ring_lo = 0, ring_hi = 0; // the part the untracked ring uploads whole (native mode adds no announced piece there)
    if (ring.Active() && !is_written) {
        const auto [lo, hi] = ring.Clip(device_addr, size);
        if (lo < hi && !memory_tracker->IsRegionGpuModified(lo, hi - lo)) {
            ring_lo = lo;
            ring_hi = hi;
            // Untracked upload ring: nobody told us what the CPU wrote, so the requested part is uploaded.
            if (device_addr < lo) {
                memory_tracker->ForEachUploadRange(device_addr, lo - device_addr, is_written, tracker_copy);
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
                memory_tracker->ForEachUploadRange(hi, device_addr + size - hi, is_written, tracker_copy);
            }
        } else {
            memory_tracker->ForEachUploadRange(device_addr, size, is_written, tracker_copy);
        }
    } else {
        memory_tracker->ForEachUploadRange(device_addr, size, is_written, tracker_copy);
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
        const u64 req_end = device_addr + size;
        for (u64 page = device_addr & ~u64{4095}; page < req_end; page += 4096) {
            if (!sampled(page)) {
                continue;
            }
            const u64 lo = std::max<u64>(device_addr, page), hi = std::min<u64>(req_end, page + 4096);
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
                // a new sampled page: the request's bytes are uploaded so the GPU copy and the CPU copy agree from here on
                auto copy = std::make_unique<std::array<u8, 4096>>();
                memory->CopySparseMemory(lo, copy->data() + (lo - page), hi - lo);
                b9_shadow.emplace(page, std::move(copy));
                add_piece(lo, hi - lo);
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
                add_piece(lo, hi - lo);
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

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        // With a second command stream the stream that did not carry the binds has to wait for them too.
        if (last_bind_tick != 0 && instance.HasAsyncQueue()) {
            info.AddWait(memory_semaphore.Handle(), last_bind_tick);
        }
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    last_bind_tick = signal_tick;
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    std::unique_lock queue_lock{Vulkan::Scheduler::submit_mutex, std::defer_lock};
    if (!Vulkan::Scheduler::submit_lock_held) {
        queue_lock.lock();
    }
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
