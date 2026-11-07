// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <fstream>
#include <limits>
#include <xxhash.h>
#include "native/native_census.h"
#include "native/native_ui_trace.h"
#include "native/native_image_caller.h"
#include "native/native_content.h"
#include "native/native_debug_server.h"
#include "native/native_flight.h"
#include "native/native_prof.h"

#include "native/native_image_marks.h"
#include "native/native_untracked_ring.h"

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "native/native_resources.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_resource_audit.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "native/native_readback_policy.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include <fmt/format.h>
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u32 MAX_IMAGES = std::numeric_limits<u16>::max();
static constexpr u32 MAX_IMAGE_VIEWS = std::numeric_limits<u16>::max();
static constexpr u32 MAX_SAMPLERS = std::numeric_limits<u16>::max();

namespace {
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
                           Vulkan::Runtime& runtime_, Core::NativeExecutor* executor_,
                           BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, executor{executor_},
      buffer_cache{buffer_cache_}, slot_images{MAX_IMAGES},
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

void TextureCache::NoteImageEvent(const ImageInfo& info, bool registered) noexcept {
    const u64 generation = image_generation.fetch_add(1, std::memory_order_release) + 1;
    ImageEvent& event = image_events[generation % ImageEventRing];
    event.generation.store(0, std::memory_order_release);
    event.address.store(info.guest_address, std::memory_order_relaxed);
    event.size.store(info.guest_size, std::memory_order_relaxed);
    event.format.store((registered ? 1ull : 0ull) | (static_cast<u64>(info.pixel_format) << 8) | (static_cast<u64>(info.type) << 40), std::memory_order_relaxed);
    event.shape.store(u64{info.size.width} | (u64{info.size.height} << 16) | (u64{info.resources.levels} << 32) | (u64{info.resources.layers} << 40) | (u64{info.num_samples} << 48), std::memory_order_relaxed);
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

std::string TextureCache::DescribeImageEvents(u64 first, u64 last) const {
    std::string out;
    if (last <= first || last - first > ImageEventRing) {
        return out;
    }
    for (u64 generation = first + 1; generation <= last; ++generation) {
        const ImageEvent& event = image_events[generation % ImageEventRing];
        if (event.generation.load(std::memory_order_acquire) != generation) {
            continue;
        }
        const u64 format = event.format.load(std::memory_order_relaxed), shape = event.shape.load(std::memory_order_relaxed);
        out += fmt::format("  image event {} {} addr {:#x} size {:#x} format {} type {} {}x{} levels {} layers {} samples {}\n", generation, (format & 1) ? "REGISTER  " : "UNREGISTER", event.address.load(std::memory_order_relaxed),
                           event.size.load(std::memory_order_relaxed), (format >> 8) & 0xffffffffull, format >> 40, shape & 0xffff, (shape >> 16) & 0xffff, (shape >> 32) & 0xff, (shape >> 40) & 0xff, shape >> 48);
    }
    return out;
}

void TextureCache::TouchFoundImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    image.tick_accessed_last = gc_tick;
    TouchImage(image);
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size, const char* by) {
    Core::UiTrace::NoteGpuWrite(by, address, max_size);
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
    Core::NativeCensus::NoteImage(Core::NativeCensus::ImageDepthOverlap);
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
            RefreshImage(new_image, Core::ImageCaller::ResolveDepthOverlapNewImage);
        }
        new_image.flags &= ~ImageFlagBits::Dirty;
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
    Core::NativeCensus::NoteImage(Core::NativeCensus::ImageOverlap);
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
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 0);
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 1);
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 2);
            return {ExpandImage(image_info, cache_image_id, Core::ImageCaller::OverlapBlockView), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 3);
            return {ExpandImage(image_info, cache_image_id, Core::ImageCaller::OverlapVolume), -1, -1};
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
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, is_compatible ? 4 : 5);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 6);
            return {ExpandImage(image_info, cache_image_id, Core::ImageCaller::OverlapGrow), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 7);
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
        Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 8);
        return {merged_image_id, -1, -1};
    }

    // F2.3: the right / left overlaps (a requested mip or layer of a cached image at another address, a cached image inside the requested one) are gone. A named lookup only lists images at
    // its own address and an owner-less one inside the ring is answered by the ring index, so no caller reaches here with another address (3 play censuses 2026-11-03, 0 of them).
    Core::NativeUpdates::NoteE3Overlap(e3_overlap_class, 15);
    Core::RareEvents::Note("f23_overlap_at_other_address", fmt::format("requested {:#x} size {:#x}, cached {:#x} size {:#x}", image_info.guest_address, image_info.guest_size,
                                                                       cache_image.info.guest_address, cache_image.info.guest_size));
    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id, Core::ImageCaller caller) {
    Core::NativeCensus::NoteImage(Core::NativeCensus::ImageExpand);
    NoteImageCreated(SiteExpand, info);
    const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image, Core::ImageCaller::ExpandImageCopy);
    runtime.CopyImage(&src_image, &new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    TLG_PROF_LEVEL("texture.find_image", 2);
    const auto& info = desc.info;
    Core::ResourceAudit::Note(Core::ResourceAudit::ImageUse, info.guest_address, info.guest_size, desc.type == BindingType::Storage || desc.type == BindingType::RenderTarget);
    ASSERT(info.guest_address != 0);

    // B8.3: the engine texture that starts at this memory, and its image that matches the description exactly (the perfect match below, found by id)
    const u32 b8_find = Core::NativeUpdates::B8FindMode();
    const u64 uid_floor = Image::NewestUid(); // images up to this uid existed before this lookup
    u64 owner_id = 0;
    u64 owner_bytes = 0;
    ImageId by_id{};
    if (desc.engine_id == 0) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::E3FindUnnamed);
        {
            Core::NativeDevice::Object at;
            const bool found = Core::NativeDevice::ResourceAt(info.guest_address, at);
            const u32 owner_state = !found ? 0u : at.kind == Core::NativeDevice::Kind::Buffer ? 3u : at.address == info.guest_address ? 1u : 2u;
            Core::NativeUpdates::NoteE3Unnamed(static_cast<u32>(desc.type), desc.engine_site, owner_state);
            if (owner_state == 0) {
                Core::NativeContent::ClassifyRingLookup(info.guest_address, info.guest_size, static_cast<u32>(desc.type));
            }
            if (owner_state == 0 || owner_state == 3) {
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                    LOG_WARNING(Render_Vulkan, "F2.3 census: FindImage with no engine texture: type {} address {:#x} size {:#x} {}x{}x{} format {} tile {} layers {} levels {} (owner state {})", static_cast<u32>(desc.type),
                                info.guest_address, info.guest_size, info.size.width, info.size.height, info.size.depth, vk::to_string(info.pixel_format), static_cast<u32>(info.tile_mode),
                                info.resources.layers, info.resources.levels, owner_state);
                }
            }
        }
    } else {
        // E3 census: the resource the record names against the texture the address lookup finds
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::E3FindNamed);
        Core::NativeDevice::Object named;
        if (!Core::NativeDevice::Lookup(desc.engine_id, named)) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::E3NamedUnknownObject);
        } else {
            const u64 named_resource = named.resource_id != 0 ? named.resource_id : named.id;
            Core::NativeDevice::Object at;
            if (!Core::NativeDevice::ResourceAt(info.guest_address, at) || at.kind == Core::NativeDevice::Kind::Buffer) {
                Core::NativeUpdates::NoteB8(Core::NativeUpdates::E3NamedNoOwner);
            } else if (at.id != named_resource) {
                Core::NativeUpdates::NoteB8(Core::NativeUpdates::E3NamedOtherOwner);
                static std::array<std::atomic<u32>, 8> logged{};
                if (logged[desc.engine_site & 7].fetch_add(1, std::memory_order_relaxed) < 3) {
                    LOG_WARNING(Render_Vulkan, "E3 named differs: site {} named resource {} (kind {}, memory {:#x}) address {:#x} size {}x{} format {}; the address lookup finds resource {} (kind {}, memory {:#x})",
                                desc.engine_site, named_resource, u32(named.kind), named.address, info.guest_address, info.size.width, info.size.height, vk::to_string(info.pixel_format), at.id,
                                u32(at.kind), at.address);
                }
            } else {
                Core::NativeUpdates::NoteB8(at.address == info.guest_address ? Core::NativeUpdates::E3NamedSameOwner : Core::NativeUpdates::E3NamedInterior);
            }
        }
    }
    // the perfect match below, for an image of the engine texture's list
    const auto exact_match = [&](const Image& cache_image) {
        return True(cache_image.flags & ImageFlagBits::Registered) && cache_image.info.guest_address == info.guest_address &&
               cache_image.info.guest_size == info.guest_size && cache_image.info.size == info.size &&
               IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) &&
               !(cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1}) && !(exact_fmt && info.pixel_format != cache_image.info.pixel_format) &&
               !(cache_image.info.resources < info.resources);
    };
    const auto counted_exact_match = [&](const Image& cache_image, Core::ImageCaller caller) {
        return exact_match(cache_image);
    };
    if (b8_find != 0) {
        Core::NativeDevice::Object owner;
        // E3.2: a named description (a record binding, a bound target) finds its engine texture by the id; only the unnamed ones ask the address
        bool owner_found = false;
        if (desc.engine_id != 0 && Core::NativeUpdates::E3IdDecides()) {
            Core::NativeDevice::Object named;
            if (Core::NativeDevice::Lookup(desc.engine_id, named)) {
                owner_found = named.resource_id == 0 ? (owner = named, true) : Core::NativeDevice::Lookup(named.resource_id, owner);
            }
            Core::NativeUpdates::NoteB8(owner_found ? Core::NativeUpdates::E3OwnerById : Core::NativeUpdates::E3OwnerByIdMissed);
        }
        if (!owner_found) {
            owner_found = Core::NativeDevice::ResourceAt(info.guest_address, owner);
        }
        if (owner_found && owner.address == info.guest_address && owner.kind != Core::NativeDevice::Kind::Buffer) {
            owner_id = owner.id;
            owner_bytes = owner.bytes;
            if (const auto it = engine_index.images.find(owner_id); it != engine_index.images.end()) {
                for (const auto& [cache_id, uid] : it->second) {
                    if (!slot_images.IsAllocated(cache_id) || slot_images[cache_id].image_uid != uid) {
                        continue;
                    }
                    if (!counted_exact_match(slot_images[cache_id], Core::ImageCaller::FindImageExactById)) {
                        continue;
                    }
                    by_id = cache_id;
                }
            }
            Core::NativeUpdates::NoteB8(by_id ? Core::NativeUpdates::B8FindById : Core::NativeUpdates::B8FindNoImageYet);
        } else {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindNoOwner);
        }
        if (by_id && b8_find == 2) {
            Image& image = slot_images[by_id];
            image.tick_accessed_last = gc_tick;
            TouchImage(image);
            return by_id;
        }
    }
    // B8.4: an engine texture's other answers, by the request (everything overlap resolution looks at)
    u64 answer_key = 0;
    const EngineAnswer* answer = nullptr;
    if (owner_id != 0 && !by_id) {
        u32 props_bits;
        std::memcpy(&props_bits, &info.props, sizeof(props_bits));
        const u64 words[] = {u64(info.guest_size), u64(info.size.width) | u64(info.size.height) << 32, u64(info.size.depth) | u64(info.pitch) << 32,
                             u64(info.pixel_format) | u64(info.type) << 32, u64(info.resources.levels) | u64(info.resources.layers) << 32,
                             u64(info.tile_mode) | u64(info.num_samples) << 32, u64(info.num_bits) | u64(props_bits) << 32,
                             u64(desc.type) | u64(exact_fmt) << 8 | u64(info.alt_tile) << 16 | u64(info.bank_swizzle) << 24, info.meta_info.htile_addr};
        answer_key = 0x9E3779B97F4A7C15ull;
        for (const u64 w : words) {
            answer_key = (answer_key ^ w) * 0x100000001B3ull;
            answer_key ^= answer_key >> 31;
        }
        if (const auto it = engine_index.answers.find(owner_id); it != engine_index.answers.end()) {
            for (const auto& a : it->second) {
                if (a.key == answer_key) {
                    answer = &a;
                }
            }
        }
        if (answer != nullptr) {
            if (!ImageUnchangedSince(answer->image_id, answer->uid, answer->generation, info.guest_address, info.guest_size)) {
                Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindAnswerStale);
                answer = nullptr;
            } else {
                // valid up to now: the next check only reads the events after this one (it used to read every event since the answer was made)
                const_cast<EngineAnswer*>(answer)->generation = image_generation.load(std::memory_order_acquire);
            }
        }
        if (answer != nullptr) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindByAnswer);
            if (b8_find == 2) {
                Image& image = slot_images[answer->image_id];
                image.tick_accessed_last = gc_tick;
                TouchImage(image);
                if (answer->mip > 0) {
                    desc.view_info.range.base.level = answer->mip;
                }
                if (answer->slice > 0) {
                    desc.view_info.range.base.layer = answer->slice;
                }
                return answer->image_id;
            }
        }
    }
    // B8.3: the image the lookup by address returns becomes the engine texture's when it lies at its memory
    const auto own = [&](ImageId found) -> bool {
        if (owner_id == 0) {
            return false;
        }
        if (by_id && found != by_id) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindByIdOther);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                LOG_WARNING(Render_Vulkan, "B8.3 audit: engine texture {} at {:#x}: image {} by id, {} by address ({}x{} format {})", owner_id, info.guest_address,
                            by_id.index, found.index, info.size.width, info.size.height, vk::to_string(info.pixel_format));
            }
        }
        const Image& image = slot_images[found];
        if (image.info.guest_address != info.guest_address || (owner_bytes != 0 && image.info.guest_size > owner_bytes)) {
            return false; // not the texture's own image: remembered as an answer below
        }
        auto& list = engine_index.images[owner_id];
        if (std::ranges::find(list, std::make_pair(found, image.image_uid)) == list.end()) {
            list.erase(std::remove_if(list.begin(), list.end(), [&](const auto& e) { return !slot_images.IsAllocated(e.first) || slot_images[e.first].image_uid != e.second; }), list.end());
            list.emplace_back(found, image.image_uid);
            slot_images[found].b8_listed_by = owner_id;
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindOwned);
            if (const auto pending = engine_index.pending_clears.find(owner_id); pending != engine_index.pending_clears.end()) {
                // B11 step 6: the texture's clear that came before its first image
                Image& target = slot_images[found];
                const EngineClear clear = pending->second;
                engine_index.pending_clears.erase(pending);
                const u32 layers = clear.layer_count == 0 ? target.info.resources.layers : clear.layer_count;
                if (target.info.props.is_depth == clear.depth && clear.mip < target.info.resources.levels && clear.first_layer + layers <= target.info.resources.layers) {
                    const SubresourceRange range{.base = {static_cast<u16>(clear.mip), static_cast<u16>(clear.first_layer)}, .extent = {1, static_cast<u16>(layers)}};
                    const bool whole = target.info.resources.levels == 1 && clear.first_layer == 0 && layers == target.info.resources.layers;
                    if (!whole && True(target.flags & (ImageFlagBits::CpuDirty | ImageFlagBits::MaybeCpuDirty))) {
                        RefreshImage(target, Core::ImageCaller::FindImageClearRest); // the rest of the image from memory first
                    }
                    if (clear.fast && whole) {
                        target.pending_fast_clear = EngineColorValue(target.info.pixel_format, clear.value);
                    } else {
                        if (clear.depth) {
                            runtime.ClearDepthStencilImage(&target, range, (clear.flags & 1) != 0, clear.value[0], (clear.flags & 2) != 0, clear.stencil);
                            target.usage.depth_target = true;
                        } else {
                            runtime.ClearImage(&target, range, EngineColorValue(target.info.pixel_format, clear.value));
                        }
                        if (whole) {
                            target.flags &= ~(ImageFlagBits::CpuDirty | ImageFlagBits::MaybeCpuDirty);
                        }
                        target.flags |= ImageFlagBits::GpuModified;
                    }
                    Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeClearAtFirstImage);
                } else {
                    Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeClearFirstImageMismatch);
                }
            }
        }
        return true;
    };

    SmallVector<ImageId, 8> image_ids;
    const bool ring_lookup = owner_id == 0 && RingIndexed(info.guest_address, info.guest_size);
    if (owner_id != 0) {
        // E3.3: an engine texture's candidates are its own images, nothing else (E3.1 census: a named lookup never overlapped another engine texture's image)
        if (const auto it = engine_index.images.find(owner_id); it != engine_index.images.end()) {
            for (const auto& [cache_id, uid] : it->second) {
                // registered only: a freed image stays allocated until its deferred deletion (the page table never listed it)
                if (slot_images.IsAllocated(cache_id) && slot_images[cache_id].image_uid == uid && True(slot_images[cache_id].flags & ImageFlagBits::Registered) &&
                    slot_images[cache_id].info.guest_address == info.guest_address) {
                    image_ids.push_back(cache_id);
                }
            }
            std::sort(image_ids.begin(), image_ids.end(), [&](ImageId a, ImageId b) { return slot_images[a].image_uid < slot_images[b].image_uid; });
        }
        Core::NativeUpdates::NoteB8(image_ids.empty() ? Core::NativeUpdates::B8FirstNoOverlap : Core::NativeUpdates::B8FirstOverlap);
    } else if (ring_lookup) {
        // F2.3: the images of the frame ring over the window, from the ring index (no page table scan)
        for (auto it = ring_images.lower_bound(info.guest_address > ring_max_size ? info.guest_address - ring_max_size : 0); it != ring_images.end() && it->first < info.guest_address + info.guest_size; ++it) {
            const Image& candidate = slot_images[it->second];
            if (it->first + candidate.info.guest_size > info.guest_address) {
                image_ids.push_back(it->second);
            }
        }
    } else {
        // F2.3: no engine texture owns the memory and it is neither in the frame ring nor in an engine GPU allocation (the 3 play censuses of 2026-11-03 never reached this): refused, counted and
        // written down; the image is made new, no page table scan for candidates
        Core::NativeContent::NoteRingIndex(Core::NativeContent::RingIndexEvent::RefusedNoOwner);
        Core::RareEvents::Note("f23_owner_less_lookup_outside_ring",
                               fmt::format("type {} named {} address {:#x} size {:#x} {}x{}x{} format {}", static_cast<u32>(desc.type), desc.engine_id != 0, info.guest_address, info.guest_size,
                                           info.size.width, info.size.height, info.size.depth, vk::to_string(info.pixel_format)));
    }

    ImageId image_id{};

    // Check for a perfect match first
    const auto perfect = [&](const Image& cache_image) {
        return cache_image.info.guest_address == info.guest_address && cache_image.info.guest_size == info.guest_size && cache_image.info.size == info.size &&
               IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) && !(cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1}) &&
               !(exact_fmt && info.pixel_format != cache_image.info.pixel_format);
    };
    for (const auto& cache_id : image_ids) {
        if (perfect(slot_images[cache_id])) {
            image_id = cache_id;
        }
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (ring_lookup) {
        // F2.3 ring rule: the window is the image's; an exact image is reused, every other image over the window is stale (freed once old enough, as the pool reuse case of ResolveOverlap)
        if (image_id) {
            Core::NativeContent::NoteRingIndex(Core::NativeContent::RingIndexEvent::ReusedExact);
        }
        for (const auto& cache_id : image_ids) {
            if (cache_id == image_id) {
                continue;
            }
            if (gc_tick - slot_images[cache_id].tick_accessed_last > 32) {
                Core::NativeContent::NoteRingIndex(Core::NativeContent::RingIndexEvent::StaleFreed);
                FreeImage(cache_id);
            } else {
                Core::NativeContent::NoteRingIndex(Core::NativeContent::RingIndexEvent::StaleKept);
            }
        }
    } else if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            {
                // E3 census: whose image the candidate is (the requested texture's own, another engine texture's, or none's)
                const u64 listed = slot_images[cache_id].b8_listed_by & 0x3fffffffffffffffull;
                e3_overlap_class = owner_id == 0 ? 3 : listed == 0 ? 2 : listed == owner_id ? 0 : 1;
            }
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
        if (ring_lookup) {
            Core::NativeContent::NoteRingIndex(Core::NativeContent::RingIndexEvent::CreatedNew);
        }
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

    const bool owned = view_mip <= 0 && view_slice <= 0 && own(image_id);
    if (answer_key != 0) {
        const Image& found = slot_images[image_id];
        if (answer != nullptr && (answer->image_id != image_id || answer->uid != found.image_uid || answer->mip != view_mip || answer->slice != view_slice)) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8FindAnswerOther);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                LOG_WARNING(Render_Vulkan, "B8.4 audit: engine texture {} at {:#x}: answer image {} mip {} slice {}, by address image {} mip {} slice {}", owner_id,
                            info.guest_address, answer->image_id.index, answer->mip, answer->slice, image_id.index, view_mip, view_slice);
            }
        }
        if (!owned || !counted_exact_match(found, Core::ImageCaller::FindImageViewReuse) || view_mip > 0 || view_slice > 0) {
            // remembered (every answer the texture's own list does not give): the next lookup of this request is answered by the engine texture's id
            auto& list = engine_index.answers[owner_id];
            EngineAnswer fresh{answer_key, image_id, found.image_uid, ImageGeneration(), view_mip, view_slice};
            bool replaced = false;
            for (auto& a : list) {
                if (a.key == answer_key) {
                    a = fresh;
                    replaced = true;
                }
            }
            if (!replaced) {
                list.push_back(fresh);
            }
            slot_images[image_id].b8_listed_by = owner_id | (1ull << 63);
        }
    }
    if (b8_find != 0) {
        // B8.4: an older image given out for the first time may have missed announcements (the lists did not hold it): refreshed from memory
        Image& found = slot_images[image_id];
        if ((found.b8_marks & B8Listed) == 0) {
            found.b8_marks |= B8Listed;
            if (found.image_uid <= uid_floor) {
                const bool gpu = True(found.flags & ImageFlagBits::GpuModified);
                Core::NativeUpdates::NoteB8(gpu ? Core::NativeUpdates::B8AdoptedGpu : Core::NativeUpdates::B8Adopted);
                if (b8_find == 2 && !gpu) {
                    found.flags |= ImageFlagBits::CpuDirty;
                }
            }
        }
    }
    if (b8_find != 0 && owner_id == 0) {
        // B8.4: an image inside engine textures' memory that none of them starts at is listed with them (key 0: never an answer), so their lists hold every image of
        // their memory
        static thread_local std::vector<u64> ids;
        ids.clear();
        const Image& found = slot_images[image_id];
        if (Core::NativeDevice::TexturesIn(found.info.guest_address, found.info.guest_size, ids) || !ids.empty()) {
            for (const u64 id : ids) {
                auto& list = engine_index.answers[id];
                if (std::ranges::none_of(list, [&](const EngineAnswer& a) { return a.image_id == image_id && a.uid == found.image_uid; })) {
                    list.erase(std::remove_if(list.begin(), list.end(),
                                              [&](const EngineAnswer& a) { return a.key == 0 && (!slot_images.IsAllocated(a.image_id) || slot_images[a.image_id].image_uid != a.uid); }),
                               list.end());
                    list.push_back(EngineAnswer{0, image_id, found.image_uid, 0, -1, -1});
                    slot_images[image_id].b8_listed_by = id | (1ull << 62);
                }
            }
        }
    }
    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    SmallVector<ImageId, 4> image_ids;
    // B8.4: memory an engine texture starts at finds its images through the texture's lists; B8.5: engine buffer memory through the buffers' lists
    bool routed = false;
    SmallVector<ImageId, 8> engine_ids;
    Core::NativeDevice::Object owner;
    if (Core::NativeDevice::ResourceAt(address, owner) && owner.address == address && owner.kind != Core::NativeDevice::Kind::Buffer) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8RouteCopy);
        static thread_local std::vector<u64> ids;
        ids.assign(1, owner.id);
        engine_index.ImagesIn(ids, address, 1, engine_ids);
        routed = true;
    } else if (engine_index.BufferImagesIn(address, 1, engine_ids)) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8BufCopy);
        routed = true;
    } else {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8RouteCopyFallback);
    }
    if (routed) {
        for (const ImageId image_id : engine_ids) {
            const Image& image = slot_images[image_id];
            if (image.info.guest_address != address || (ensure_valid && !image.SafeToDownload())) {
                continue;
            }
            image_ids.push_back(image_id);
        }
    } else {
        ForEachImageWithAddress(address, [&](ImageId image_id, Image& image) {
            if (ensure_valid && !image.SafeToDownload()) {
                return;
            }
            image_ids.push_back(image_id);
        });
    }
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
    TLG_PROF_LEVEL("texture.find_view", 3);
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (((readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) || explicit_download) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id, Core::ImageCaller::FindTexture);
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
    TLG_PROF_LEVEL("texture.find_view", 3);
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id, Core::ImageCaller::FindRenderTarget);

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
    TLG_PROF_LEVEL("texture.find_view", 3);
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id, Core::ImageCaller::FindDepthTarget);

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

void TextureCache::RefreshImage(Image& image, Core::ImageCaller caller) {
    TLG_PROF_LEVEL("texture.refresh_image", 3);
    Core::NativeCensus::NoteImage(Core::NativeCensus::ImageRefresh, image.info.guest_size);
    // E3.3: the shared page edges of an image are no longer harvested here. The engine's announced writes tell when an image's content changed (E3.1 census: the harvest alone
    // found a change in 0 of 8.7 million refreshes); the sampled hash check below still counts a change nothing announced.
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
    // F2.4: the B8.1 verify (a hash of the guest bytes after each refresh, off by default) and the accounting of what dirtied the image are gone: an image is refreshed by the announcements and
    // the engine's own writes (E3.1: the address route found a change in 0 of 8.7 million refreshes), and the write ledger (F2.5) says whether the memory moved after the draw that reads it.
    if (False(image.flags & ImageFlagBits::Dirty)) {
        if (False(image.flags & ImageFlagBits::GpuModified) && image.info.num_samples <= 1 && !Core::NativeContent::InRing(image.info.guest_address, image.info.guest_size)) {
            Core::NativeContent::SilentImageKept(image.info.guest_address, image.info.guest_size); // W1.5 census (TLG_SILENT_CENSUS=1)
        }
        return;
    }
    if (image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

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

    const u64 ui_hash_before = Core::UiTrace::UploadHash(image.info.guest_address, image.info.guest_size);
    TLG_PROF_NAMED_LEVEL(prof_obtain, "texture.refresh.obtain_bytes", 2);
    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    prof_obtain.End();
    Core::NativeCensus::NoteImage(Core::NativeCensus::ImageDetile, image.info.guest_size);
    TLG_PROF_NAMED_LEVEL(prof_detile, "texture.refresh.detile", 2);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    prof_detile.End();
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    {
        TLG_PROF_NAMED_LEVEL(prof_upload, "texture.refresh.upload", 2);
        runtime.UploadImage(&image, buffer, image_copies);
    }
    if (ui_hash_before != 0) {
        Core::UiTrace::NoteImageUpload(image.info.guest_address, image.info.guest_size, ui_hash_before, Core::UiTrace::UploadHash(image.info.guest_address, image.info.guest_size));
    }
    if (!Core::NativeContent::InRing(image.info.guest_address, image.info.guest_size)) {
        Core::NativeContent::SilentImageUploaded(image.info.guest_address, image.info.guest_size, BufferCache::last_image_source_gpu); // W1.5 census
    }
    Core::Flight::Add(Core::Flight::ImageRefresh, 0, image.info.guest_address, image.info.guest_size, static_cast<u8>(caller)); // L1
    Core::NativeResources::NoteImageUpload(image.info.guest_address, image.info.guest_size,
                                           BufferCache::last_image_source_gpu);
    Core::NativeContent::NoteRingImageShape(image.info.guest_address, image.info.guest_size, image.info.size.width, image.info.size.height, image.info.size.depth,
                                            static_cast<u32>(image.info.pixel_format), num_mips, num_layers, image.info.pitch, static_cast<u32>(image.info.tile_mode));
    if (Core::NativeContent::ShadowOn()) {
        // F2.2 shadow (TLG_F2_STORE=1): the announced content store against the memory this upload read
        Core::NativeContent::CompareImage({.image_uid = image.image_uid, .address = image.info.guest_address, .size = image.info.guest_size, .width = image.info.size.width,
                                           .height = image.info.size.height, .format = static_cast<u32>(image.info.pixel_format),
                                           .group_hint = BufferCache::last_image_source_gpu ? Core::NativeContent::Group::GpuSource : Core::NativeContent::Group::FirstUpload,
                                           .caller = static_cast<u32>(caller)});
    }
}

vk::Sampler TextureCache::GetSampler(const Core::NativeDesc::SamplerDesc& sharp,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    TLG_PROF_LEVEL("texture.get_sampler", 3);
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

/// F2.3: memory the ring image index covers: the frame ring, and the engine's own GPU allocations that no engine resource (texture or buffer) starts at or contains.
bool TextureCache::RingIndexed(VAddr address, u64 size) const {
    if (Core::NativeContent::InRing(address, size)) {
        return true;
    }
    if (!Core::NativeContent::InGpuAllocation(address, size)) {
        return false;
    }
    Core::NativeDevice::Object owner;
    return !Core::NativeDevice::ResourceAt(address, owner);
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    NoteImageEvent(image.info, true);
    Core::NativeResources::NoteImageCensus(image.info.guest_address, image.info.size.width, image.info.size.height, (image.usage_flags & vk::ImageUsageFlagBits::eColorAttachment) ? true : false,
                                           image.info.props.is_depth != 0); // B1 census
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image_lru_cache.Insert(image, gc_tick);
    const auto& info = image.info;
    ASSERT_MSG((info.guest_address & 0xff) == 0, "Trying to register an unaligned image");
    if (Core::NativeUpdates::B8BufferImagesMode() != 0) {
        // B8.5: an image in engine buffer memory is listed by each buffer it meets
        static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
        parts.clear();
        Core::NativeDevice::BuffersIn(info.guest_address, info.guest_size, parts);
        for (const auto& part : parts) {
            auto& list = engine_index.buffer_images[part.id];
            if (std::ranges::find(list, std::make_pair(image_id, image.image_uid)) == list.end()) {
                list.emplace_back(image_id, image.image_uid);
                Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8BufListed);
            }
        }
    }
    if (RingIndexed(info.guest_address, info.guest_size)) {
        ring_images.emplace(info.guest_address, image_id);
        ring_max_size = std::max<u64>(ring_max_size, info.guest_size);
    }
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
    NoteImageEvent(image.info, false);
    image_lru_cache.Free(image);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    if (!ring_images.empty()) {
        const auto range = ring_images.equal_range(image.info.guest_address);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == image_id) {
                ring_images.erase(it);
                break;
            }
        }
    }
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
    image.track_addr = image_begin;
    image.track_addr_end = image_end;
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (image.IsUntracked()) {
        return;
    }
    image.track_addr = 0;
    image.track_addr_end = 0;
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

// ---- Core::NativeImageIndex::Host (native/native_image_index.h) ----
bool TextureCache::Live(ImageId id, u64 uid) const {
    return slot_images.IsAllocated(id) && slot_images[id].image_uid == uid;
}

bool TextureCache::Registered(ImageId id) const {
    return slot_images.IsAllocated(id) && True(slot_images[id].flags & ImageFlagBits::Registered);
}

bool TextureCache::Overlaps(ImageId id, VAddr address, u64 size) const {
    return slot_images[id].Overlaps(address, size);
}

u64 TextureCache::Uid(ImageId id) const {
    return slot_images[id].image_uid;
}

VAddr TextureCache::GuestAddress(ImageId id) const {
    return slot_images[id].info.guest_address;
}

u64 TextureCache::GuestSize(ImageId id) const {
    return slot_images[id].info.guest_size;
}

void TextureCache::ForEachImageIn(VAddr address, u64 size, const std::function<void(ImageId)>& func) {
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image&) { func(image_id); });
}

bool TextureCache::ClearListed(ImageId id) {
    auto& image = slot_images[id];
    const bool was_listed = (image.b8_marks & B8Listed) != 0;
    image.b8_marks &= ~B8Listed;
    return was_listed;
}

Core::NativeImageIndex::Host::Announced TextureCache::ApplyAnnounced(ImageId id, VAddr address, u64 size, bool newer) {
    Image& image = slot_images[id];
    std::scoped_lock lk{image.mutex};
    if (!image.Overlaps(address, size)) {
        return Announced::NotOverlapping;
    }
    if (!newer && True(image.flags & ImageFlagBits::GpuModified)) {
        return Announced::KeptForGpu; // the GPU wrote the image after the first application: its bytes are newer than the CPU's
    }
    image.flags |= ImageFlagBits::CpuDirty;
    UntrackImage(id);
    return Announced::Dirtied;
}

void TextureCache::MarkDeleted(ImageId id) {
    slot_images[id].b8_marks |= B8Deleted;
}

vk::ClearValue TextureCache::EngineColorValue(vk::Format format, const float (&value)[4]) {
    vk::ClearValue out{};
    const std::string name = vk::to_string(format);
    if (name.find("Uint") != std::string::npos || name.find("Sint") != std::string::npos) {
        for (u32 i = 0; i < 4; ++i) {
            out.color.int32[i] = static_cast<s32>(value[i]);
        }
    } else {
        for (u32 i = 0; i < 4; ++i) {
            out.color.float32[i] = value[i];
        }
    }
    return out;
}

} // namespace VideoCore
