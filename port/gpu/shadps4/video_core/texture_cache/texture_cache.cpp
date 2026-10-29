// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <fstream>
#include <limits>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/native_resources.h"
#include "core/native_device.h"
#include "core/native_updates.h"
#include "core/native_write_watch.h"
#include "core/resource_audit.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/page_manager.h"
#include "video_core/readback_policy.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u32 MAX_IMAGES = std::numeric_limits<u16>::max();
static constexpr u32 MAX_IMAGE_VIEWS = std::numeric_limits<u16>::max();
static constexpr u32 MAX_SAMPLERS = std::numeric_limits<u16>::max();

namespace {
constexpr u8 B8Fault = 1, B8Announced = 2, B8Owed = 4, B8Deleted = 8; ///< Image::b8_marks
enum ImageSite : u32 { SiteFind, SiteOverlap, SiteExpand, SiteStencil, NumImageSites };
std::array<std::atomic<u64>, NumImageSites> images_created{};
std::atomic<u64> images_freed{};

void NoteImageCreated(ImageSite site, const ImageInfo& info) {
    static constexpr const char* names[NumImageSites] = {"find", "overlap", "expand", "stencil"};
    const u64 count = images_created[site].fetch_add(1, std::memory_order_relaxed);
    if (count < 24 || count % 5000 == 0) {
        LOG_INFO(Render_Vulkan, "new image ({} #{}): address {:#x} size {:#x} {}x{}x{} levels {} layers {} format {} tile {} samples {} depth {}", names[site], count,
                 info.guest_address, info.guest_size, info.size.width, info.size.height, info.size.depth, info.resources.levels, info.resources.layers,
                 vk::to_string(info.pixel_format), static_cast<u32>(info.tile_mode), info.num_samples, info.props.is_depth);
    }
}
} // namespace

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, slot_images{MAX_IMAGES},
      slot_image_views{MAX_IMAGE_VIEWS}, slot_samplers{MAX_SAMPLERS},
      blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    const s64 min_pressure_floor =
        std::clamp<s64>(device_local_memory / 4, 256_MB, DEFAULT_PRESSURE_GC_MEMORY);
    const s64 min_critical_floor =
        std::clamp<s64>(device_local_memory / 2, 512_MB, DEFAULT_CRITICAL_GC_MEMORY);
    pressure_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      min_pressure_floor));
    critical_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      min_critical_floor));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

u32 TextureCache::ImageDownloadSize(const Image& image) {
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    return download_size;
}

Vulkan::StagingBufferRef TextureCache::RecordImageCopy(Image& image, u32 download_size,
                                                       bool deferred) {
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, deferred);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    return download;
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const u32 download_size = ImageDownloadSize(image);
    const auto download = RecordImageCopy(image, download_size, !sync);
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                download.Invalidate();
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download.mapped, download_size);
                runtime.GetStagingPool().FreeDeferred(download);
            });
    }
}

void TextureCache::RecordImageCopyAsync(Image& image, u32 download_size) {
    const auto download = RecordImageCopy(image, download_size, true);
    async_recorded.store(1, std::memory_order_release);
    // The priority thread runs its operations in the order they were deferred, so the fence of
    // this frame is signaled only once the pixels are in guest memory.
    scheduler.DeferPriorityOperation(
        [this, addr = image.info.guest_address, download, download_size] {
            download.Invalidate();
            Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(addr), download.mapped,
                                                      download_size);
            runtime.GetStagingPool().FreeDeferred(download);
        },
        scheduler.CurrentTick());
}

void TextureCache::RecordImageDownloads(ReadbackBatch& batch, u32 fence_kind) {
    const auto& policy = ReadbackPolicy::Get();
    if (!(policy.image_download_kinds_mask.load(std::memory_order_relaxed) & fence_kind)) {
        return; // the images stay listed for a fence of an allowed kind
    }
    const bool async = policy.images_async.load(std::memory_order_relaxed);
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        Image& image = slot_images[image_id];
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            continue;
        }
        const u32 download_size = ImageDownloadSize(image);
        if (async) {
            RecordImageCopyAsync(image, download_size);
            continue;
        }
        batch.Reserve(download_size);
        const auto download = RecordImageCopy(image, download_size, true);
        batch.Add(download_size, true,
                  [this, device_addr = image.info.guest_address, download, download_size] {
                      download.Invalidate();
                      Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                                download.mapped, download_size);
                      runtime.GetStagingPool().FreeDeferred(download);
                  });
    }
    download_images.clear();
}

void TextureCache::NoteImageEvent(const ImageInfo& info) noexcept {
    const u64 generation = image_generation.fetch_add(1, std::memory_order_release) + 1;
    ImageEvent& event = image_events[generation % ImageEventRing];
    event.generation.store(0, std::memory_order_release);
    event.address.store(info.guest_address, std::memory_order_relaxed);
    event.size.store(info.guest_size, std::memory_order_relaxed);
    event.generation.store(generation, std::memory_order_release);
}

bool TextureCache::ImageUnchangedSince(ImageId image_id, u64 uid, u64 since, VAddr address, u64 size) const noexcept {
    if (uid == 0 || !image_id || !slot_images.IsAllocated(image_id)) {
        return false;
    }
    const Image& image = slot_images[image_id];
    if (image.image_uid != uid || False(image.flags & ImageFlagBits::Registered)) {
        return false;
    }
    const u64 now = image_generation.load(std::memory_order_acquire);
    if (now < since || now - since > ImageEventRing) {
        return false;
    }
    const u64 begin = address & ~u64{4095};
    const u64 end = (address + std::max<u64>(size, 1) + 4095) & ~u64{4095};
    for (u64 generation = since + 1; generation <= now; ++generation) {
        const ImageEvent& event = image_events[generation % ImageEventRing];
        if (event.generation.load(std::memory_order_acquire) != generation) {
            return false; // not written yet, or overwritten
        }
        const u64 event_address = event.address.load(std::memory_order_relaxed);
        const u64 event_size = event.size.load(std::memory_order_relaxed);
        if (event.generation.load(std::memory_order_acquire) != generation) {
            return false;
        }
        const u64 event_begin = event_address & ~u64{4095};
        const u64 event_end = (event_address + std::max<u64>(event_size, 1) + 4095) & ~u64{4095};
        if (event_begin < end && begin < event_end) {
            return false;
        }
    }
    return true;
}

void TextureCache::TouchFoundImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    image.tick_accessed_last = gc_tick;
    TouchImage(image);
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(pages_start, pages_end - pages_start,
                         [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });

    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};

        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            image.b8_marks |= B8Fault;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    }
}

namespace {
constexpr VAddr WwPage = 4096;
/// the image's bytes on the page holding `at`
std::pair<VAddr, VAddr> ImageBytesOnPage(const Image& image, VAddr at) {
    const VAddr page = at & ~(WwPage - 1);
    return {std::max(page, image.info.guest_address), std::min(page + WwPage, image.info.guest_address + image.info.guest_size)};
}
u64 HashBytes(std::pair<VAddr, VAddr> range) {
    return range.first < range.second ? (XXH3_64bits(std::bit_cast<const u8*>(range.first), range.second - range.first) | 1) : 0;
}
} // namespace

void TextureCache::NoteRefreshedForWriteWatch(Image& image) {
    if (!Core::WriteWatch::LiftImages()) {
        return;
    }
    const VAddr begin = image.info.guest_address;
    const VAddr end = begin + image.info.guest_size;
    image.ww_head_hash = (begin & (WwPage - 1)) != 0 || end - begin < WwPage ? HashBytes(ImageBytesOnPage(image, begin)) : 0;
    image.ww_tail_hash = (end & (WwPage - 1)) != 0 ? HashBytes(ImageBytesOnPage(image, end - 1)) : 0;
}

void TextureCache::InvalidatePagesWritten(VAddr addr, size_t size) {
    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(addr, size, [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });
    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};
        const VAddr begin = image.info.guest_address;
        const VAddr end = begin + image.info.guest_size;
        if (!image.Overlaps(addr, size) || True(image.flags & ImageFlagBits::CpuDirty)) {
            continue;
        }
        // a written page that lies wholly inside the image: the write was the image's
        const VAddr first_full = (begin + WwPage - 1) & ~(WwPage - 1);
        const VAddr last_full_end = end & ~(WwPage - 1);
        const VAddr a = std::max(addr, first_full);
        const VAddr b = std::min<VAddr>(addr + size, last_full_end);
        bool dirty = a < b;
        if (!dirty) {
            // only the image's partial first or last page was written: its own bytes there decide
            const VAddr head_page = begin & ~(WwPage - 1);
            const VAddr tail_page = (end - 1) & ~(WwPage - 1);
            if (head_page >= addr && head_page < addr + size && image.ww_head_hash != 0) {
                dirty |= HashBytes(ImageBytesOnPage(image, begin)) != image.ww_head_hash;
            }
            if (tail_page >= addr && tail_page < addr + size && image.ww_tail_hash != 0) {
                dirty |= HashBytes(ImageBytesOnPage(image, end - 1)) != image.ww_tail_hash;
            }
        }
        if (dirty) {
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        }
    }
}

void TextureCache::NoteAnnouncedWrite(VAddr addr, size_t size, bool newer) {
    const bool native = Core::NativeUpdates::B8ImageMode() == 2;
    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(addr, size, [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });
    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};
        if (!image.Overlaps(addr, size)) {
            continue;
        }
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8AnnouncedOnImage);
        if ((image.b8_marks & B8Owed) != 0) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8AnnouncedLater);
            image.b8_marks &= ~B8Owed;
        }
        if (!newer && True(image.flags & ImageFlagBits::GpuModified)) {
            // the GPU wrote the image after the first application: its bytes are newer than the CPU's
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8KeptForGpu);
            continue;
        }
        image.b8_marks |= B8Announced;
        if (native) {
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        }
    }
}

void TextureCache::NoteEngineTextureDelete(VAddr address, u64 bytes, bool free) {
    Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8TexturesDeleted);
    SmallVector<ImageId, 4> image_ids;
    ForEachImageInRegion(address, bytes, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (image.info.guest_size > bytes) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImagesBigger);
            return;
        }
        image_ids.push_back(image_id);
    });
    for (const auto image_id : image_ids) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImages);
        if (free) {
            FreeImage(image_id);
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImagesFreed);
        } else {
            slot_images[image_id].b8_marks |= B8Deleted;
        }
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size, const char* by) {
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        if (image.info.guest_size > (8u << 20)) {
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1) < 12) {
                LOG_INFO(Render_Vulkan, "large image {:#x} + {:#x} ({}x{}) made GPU dirty by {} ({} bytes written)", address,
                         image.info.guest_size, image.info.size.width, image.info.size.height, by, max_size);
            }
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    SmallVector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        NoteImageCreated(SiteOverlap, new_info);
        const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        if (new_info.num_samples == 1 &&
            (new_info.resources.layers > cache_image.info.resources.layers ||
             new_info.resources.levels > cache_image.info.resources.levels)) {
            RefreshImage(new_image);
        }
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    static constexpr u64 NUM_FRAMES_BEFORE_REMOVAL = 32;

    auto& cache_image = slot_images[cache_image_id];
    // Age is counted in guest submissions (gc_tick advances once per submit), not scheduler ticks:
    // those advance at every flush, and with async readback fences that is many times per frame.
    const bool safe_to_delete = gc_tick - cache_image.tick_accessed_last > NUM_FRAMES_BEFORE_REMOVAL;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), gc_tick,
                  gc_tick - cache_image.tick_accessed_last);

        // Seen in The Last Guardian: a 3D texture at the same address grows in depth. Replace
        // the old image when it hasn't been used recently instead of aborting.
        ASSERT_MSG(safe_to_delete,
                   "Encountered unresolvable image overlap with equal memory address.");
        FreeImage(cache_image_id);
        return {merged_image_id, -1, -1};
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    NoteImageCreated(SiteExpand, info);
    const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    runtime.CopyImage(&src_image, &new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    Core::ResourceAudit::Note(Core::ResourceAudit::ImageUse, info.guest_address, info.guest_size, desc.type == BindingType::Storage || desc.type == BindingType::RenderTarget);
    ASSERT(info.guest_address != 0);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        NoteImageCreated(SiteFind, info);
        image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = gc_tick;
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    SmallVector<ImageId, 4> image_ids;
    ForEachImageWithAddress(address, [&](ImageId image_id, Image& image) {
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (((readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) || explicit_download) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

void TextureCache::RegisterMeta(VAddr address, MetaDataInfo info) {
    const auto it = surface_metas.find(address);
    if (it == surface_metas.end()) {
        surface_metas.emplace(address, info);
        return;
    }

    auto& registered = it->second;
    if (registered.type != info.type || registered.owner_uid != info.owner_uid) {
        registered = info;
        return;
    }

    // Preserve the current clear state when the same render target is rebound.
    registered.owner_id = info.owner_id;
    registered.num_samples = info.num_samples;
}

void TextureCache::UnregisterMeta(VAddr address, u64 owner_uid) {
    const auto it = surface_metas.find(address);
    if (it != surface_metas.end() && it->second.owner_uid == owner_uid) {
        surface_metas.erase(address);
    }
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    const VAddr cmask_addr = desc.info.meta_info.cmask_addr;
    if (image.info.meta_info.cmask_addr != cmask_addr) {
        UnregisterMeta(image.info.meta_info.cmask_addr, image.image_uid);
        image.info.meta_info.cmask_addr = cmask_addr;
    }
    if (cmask_addr) {
        RegisterMeta(cmask_addr, MetaDataInfo{
                                     .type = MetaType::CMask,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                 });
    }

    const VAddr fmask_addr = desc.info.meta_info.fmask_addr;
    if (image.info.meta_info.fmask_addr != fmask_addr) {
        UnregisterMeta(image.info.meta_info.fmask_addr, image.image_uid);
        image.info.meta_info.fmask_addr = fmask_addr;
    }
    if (fmask_addr) {
        RegisterMeta(fmask_addr, MetaDataInfo{
                                     .type = MetaType::FMask,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                 });
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    const VAddr htile_addr = desc.info.meta_info.htile_addr;
    if (image.info.meta_info.htile_addr != htile_addr) {
        UnregisterMeta(image.info.meta_info.htile_addr, image.image_uid);
        image.info.meta_info.htile_addr = htile_addr;
    }
    if (htile_addr) {
        RegisterMeta(htile_addr, MetaDataInfo{
                                     .type = MetaType::HTile,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                     .clear_mask = image.info.meta_info.htile_clear_mask,
                                 });
    }

    AssociateStencil(image_id, desc.info);

    return image.FindView(desc.view_info, false);
}

void TextureCache::AssociateStencil(ImageId depth_id, const ImageInfo& depth_info) {
    if (depth_info.stencil_addr == 0) {
        return;
    }
    ImageId stencil_id{};
    ForEachImageInRegion(
        depth_info.stencil_addr, depth_info.stencil_size, [&](ImageId image_id, Image& image) {
            if (image.info.guest_address != depth_info.stencil_addr) {
                return;
            }
            if (image.info.pixel_format == vk::Format::eUndefined ||
                Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                stencil_id = image_id;
            }
        });
    if (!stencil_id) {
        ImageInfo info{};
        info.guest_address = depth_info.stencil_addr;
        info.guest_size = depth_info.stencil_size;
        info.size = depth_info.size;
        NoteImageCreated(SiteStencil, info);
        stencil_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(stencil_id);
    }
    Image& stencil_image = slot_images[stencil_id];
    TouchImage(stencil_image);
    stencil_image.AssociateDepth(depth_id, slot_images[depth_id].image_uid);
}

void TextureCache::RefreshImage(Image& image) {
    // L3 write watch: an image's first and last page, when it shares them with other data, are not protected; their CPU writes are learned
    // here, before the dirty flags are read (the pages the image covers whole stay protected and fault as before)
    const auto harvest_shared_pages = [&] {
        const VAddr begin = image.info.guest_address;
        const VAddr end = begin + image.info.guest_size;
        Core::WriteWatch::NoteImageRange(begin, image.info.guest_size); // P3: the command processor harvests these pages ahead next time
        if ((begin & 4095) != 0 || image.info.guest_size < 4096) {
            buffer_cache.HarvestWriteWatch(begin & ~VAddr{4095}, 4096, true);
        }
        if ((end & 4095) != 0 && ((end - 1) & ~VAddr{4095}) != (begin & ~VAddr{4095})) {
            buffer_cache.HarvestWriteWatch((end - 1) & ~VAddr{4095}, 4096, true);
        }
    };
    harvest_shared_pages();
    if ((image.b8_marks & B8Deleted) != 0) {
        // B8.2 audit: an image the engine deleted the texture of is used again
        image.b8_marks &= ~B8Deleted;
        Core::NativeDevice::Object owner;
        const bool owned = Core::NativeDevice::ResourceAt(image.info.guest_address, owner) && owner.kind != Core::NativeDevice::Kind::Buffer;
        Core::NativeUpdates::NoteB8(owned ? Core::NativeUpdates::B8UsedAfterDeleteReused : Core::NativeUpdates::B8UsedAfterDeleteNoOwner);
        if (!owned) {
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                LOG_WARNING(Render_Vulkan, "B8.2: image {:#x}+{:#x} ({}x{}) used after the engine deleted its texture, no engine texture owns the memory now",
                            image.info.guest_address, image.info.guest_size, image.info.size.width, image.info.size.height);
            }
        }
    }
    const u32 b8 = Core::NativeUpdates::B8ImageMode();
    if (image.b8_hash != 0 && False(image.flags & ImageFlagBits::Dirty) && False(image.flags & ImageFlagBits::GpuModified) && Core::NativeUpdates::B8VerifySample()) {
        // B8.1 verify: an image no announcement dirtied must still hold the bytes of its last refresh
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8VerifyChecked);
        if ((XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size) | 1) != image.b8_hash) {
            Core::NativeUpdates::NoteB8Unannounced(image.info.guest_address, image.info.guest_size, false);
            image.b8_marks |= B8Owed;
            image.flags |= ImageFlagBits::CpuDirty;
            buffer_cache.InvalidateMemory(image.info.guest_address, image.info.guest_size, false);
        }
    }
    if (b8 != 0 && True(image.flags & ImageFlagBits::CpuDirty) && (image.b8_marks & (B8Fault | B8Announced)) != 0) {
        // B8.1: what told us the image changed
        if ((image.b8_marks & B8Announced) != 0) {
            Core::NativeUpdates::NoteB8((image.b8_marks & B8Fault) != 0 ? Core::NativeUpdates::B8RefreshFaultAndAnnounced : Core::NativeUpdates::B8RefreshAnnounced);
        } else {
            Core::NativeUpdates::NoteB8Unannounced(image.info.guest_address, image.info.guest_size, true);
            image.b8_marks |= B8Owed;
        }
        image.b8_marks &= B8Owed;
    }
    if (False(image.flags & ImageFlagBits::Dirty) && image.ww_hash != 0 && False(image.flags & ImageFlagBits::GpuModified) &&
        Core::WriteWatch::SampleImageCheck()) {
        // verify: an image that is not dirty must still hold the bytes of its last refresh
        bool changed = (XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size) | 1) != image.ww_hash;
        if (changed) {
            // a write after the harvest above has its record by now: harvest again; still not dirty is a miss
            harvest_shared_pages();
            changed = False(image.flags & ImageFlagBits::Dirty);
        }
        Core::WriteWatch::NoteImageCheck(changed, image.info.guest_address, image.info.guest_size);
        if (changed) {
            image.flags |= ImageFlagBits::CpuDirty;
        }
    }
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    SmallVector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
    Core::NativeResources::NoteImageUpload(image.info.guest_address, image.info.guest_size,
                                           BufferCache::last_image_source_gpu);
    NoteRefreshedForWriteWatch(image);
    image.b8_hash = Core::NativeUpdates::B8VerifyOn() && !BufferCache::last_image_source_gpu
                        ? XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size) | 1
                        : 0;
    image.ww_hash = Core::WriteWatch::VerifyOn() && !BufferCache::last_image_source_gpu
                        ? XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size) | 1
                        : 0;
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sharp,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    // Compare and plain uses of one S# need separate samplers.
    const u64 hash = HashCombine(XXH3_64bits(&sharp, sizeof(sharp)), is_depth);
    const auto [it, new_sampler] = samplers.try_emplace(hash);
    if (new_sampler) {
        it->second = slot_samplers.Insert(instance, sharp, border_color_base, is_depth);
        Sampler& sampler = slot_samplers[it->second];
        sampler.hash = hash;
        sampler_lru_cache.Insert(sampler, gc_tick);
    }

    Sampler& sampler = slot_samplers[it->second];
    sampler_lru_cache.Touch(sampler, gc_tick);
    return sampler.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    NoteImageEvent(image.info);
    Core::NativeResources::NoteImageCensus(image.info.guest_address, image.info.size.width, image.info.size.height, (image.usage_flags & vk::ImageUsageFlagBits::eColorAttachment) ? true : false,
                                           image.info.props.is_depth != 0); // B1 census
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image_lru_cache.Insert(image, gc_tick);
    const auto& info = image.info;
    ASSERT_MSG((info.guest_address & 0xff) == 0, "Trying to register an unaligned image");
    ForEachPage(info.guest_address, info.guest_size, [this, image_id, info](u64 page) {
        page_table[page].entries.emplace_back(BucketEntry{
            .key = u32(info.guest_address >> 8),
            .size = info.guest_size,
            .id = image_id,
        });
    });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    NoteImageEvent(image.info);
    image_lru_cache.Free(image);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        ASSERT_MSG(page_it, "Unregistering unregistered page={:#x}", page << Traits::PAGE_BITS);
        auto& entries = page_it->entries;
        const auto vector_it = std::ranges::find(entries, image_id, &BucketEntry::id);
        ASSERT_MSG(vector_it != entries.end(), "Unregistering unregistered image in page={:#x}",
                   page << Traits::PAGE_BITS);
        entries.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    std::scoped_lock lk{image.mutex};
    if (image.IsUntracked()) {
        tracker.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track);
    } else {
        if (image_begin < image.track_addr) {
            tracker.UpdatePageWatchers(image_begin, image.track_addr - image_begin, PageOp::Track);
        }
        if (image.track_addr_end < image_end) {
            tracker.UpdatePageWatchers(image.track_addr_end, image_end - image.track_addr_end,
                                       PageOp::Track);
        }
    }
    image.track_addr = image_begin;
    image.track_addr_end = image_end;
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (image.IsUntracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    if (size != 0) {
        tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
    image.track_addr = 0;
    image.track_addr_end = 0;
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (image.IsUntracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Untrack);

    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image.IsUntracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Image& image) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        const auto image_id = slot_images.GetSlotId(image);
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

/// The image table has a fixed number of slots, and the memory based collection above never runs for images that take no memory: the game makes small images
/// (the 4x4 float targets of its exposure pass) at a new address of its frame ring all the time. When many slots are in use the least recently used images are freed
/// whatever the memory use; render targets (tiled images the GPU wrote) are kept.
void TextureCache::GarbageCollectImageSlots() {
    const size_t alive = slot_images.Size();
    if (alive < MAX_IMAGES / 4) {
        return;
    }
    const bool critical = alive >= MAX_IMAGES * 3 / 4;
    const u64 ticks_to_destroy = std::min<u64>(critical ? 2 : 8, gc_tick);
    size_t num_deletions = critical ? 2048 : 256;
    image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, [&](Image& image) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        if (image.info.IsTiled() && image.SafeToDownload()) {
            return false;
        }
        FreeImage(slot_images.GetSlotId(image));
        return false;
    });
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Sampler& sampler) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        sampler_lru_cache.Free(sampler);
        samplers.erase(sampler.hash);
        const auto sampler_id = slot_samplers.GetSlotId(sampler);
        slot_samplers.Erase(sampler_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    GarbageCollectImageSlots();
    GarbageCollectImages();
    GarbageCollectSamplers();
    ++gc_tick;
    NoteSlotUsage();
}

void TextureCache::NoteSlotUsage() {
    static constexpr u64 RowEveryTicks = 300;
    if (gc_tick % RowEveryTicks != 0) {
        return;
    }
    static std::ofstream file;
    static const auto start = std::chrono::steady_clock::now();
    if (!file.is_open()) {
        file.open(Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "slot_usage.csv", std::ios::trunc);
        file << "seconds,gc_tick,images,image_views,samplers,images_limit,total_used_memory_mb,created_find,created_overlap,created_expand,created_stencil,freed\n";
    }
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    file << seconds << ',' << gc_tick << ',' << slot_images.Size() << ',' << slot_image_views.Size() << ',' << slot_samplers.Size() << ',' << MAX_IMAGES << ','
         << (total_used_memory >> 20) << ',' << images_created[SiteFind].load() << ',' << images_created[SiteOverlap].load() << ','
         << images_created[SiteExpand].load() << ',' << images_created[SiteStencil].load() << ',' << images_freed.load() << std::endl;
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(image.IsUntracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        UnregisterMeta(meta_info.cmask_addr, image.image_uid);
    }
    if (meta_info.fmask_addr) {
        UnregisterMeta(meta_info.fmask_addr, image.image_uid);
    }
    if (meta_info.htile_addr) {
        UnregisterMeta(meta_info.htile_addr, image.image_uid);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.Erase(image_view_id);
            }
        }
        slot_images.Erase(image_id);
        images_freed.fetch_add(1, std::memory_order_relaxed);
    });
}

} // namespace VideoCore
