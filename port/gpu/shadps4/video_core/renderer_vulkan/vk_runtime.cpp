// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/native_pass.h"
#include "core/native_registry.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/readback_policy.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("Unexpected image type {}", u32(type));
    }
}

static std::pair<u32, u32> SanitizeCopyLayers(const VideoCore::ImageInfo& src_info,
                                              const VideoCore::ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D source layers {} to 1.", src_layers);
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D destination layers {} to 1.", dst_layers);
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy source layers {} and destination layers {} to minimum.",
                        src_layers, dst_layers);
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D source layers {} to 3D destination depth {}", src_layers,
                        depth);
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D destination layers {} to 3D source depth {}", dst_layers,
                        depth);
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

static u32 BufferImageCopySize(const vk::BufferImageCopy& copy, const vk::Format pixel_format) {
    const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
    const u32 height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;

    const auto block = vk::blockExtent(pixel_format);
    const u32 block_size = vk::blockSize(pixel_format);
    const u32 row_pitch = (row_length / block[0]) * block_size;
    const u32 slice_pitch = (height / block[1]) * row_pitch;

    const u32 width_in_blocks = (copy.imageExtent.width + block[0] - 1) / block[0];
    const u32 height_in_blocks = (copy.imageExtent.height + block[1] - 1) / block[1];
    const u32 num_slices = copy.imageExtent.depth * copy.imageSubresource.layerCount;

    return (num_slices - 1) * slice_pitch + (height_in_blocks - 1) * row_pitch +
           width_in_blocks * block_size;
}

Runtime::Runtime(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, staging_pool{instance_, scheduler_} {
    blit_helper = std::make_unique<VideoCore::BlitHelper>(instance, scheduler);

    memory_barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    memory_barrier.dstAccessMask =
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    if (const char* e = std::getenv("TLG_HAZARD_AUDIT"); e != nullptr && *e != 0 && *e != '0') {
        hazard_audit_on = true;
    }
    // C8 (2026-10-28): the planner always decides; the tracker runs next to it and is only compared
    live_plan_on = true;
    live_plan_exact = true;
    live_plan_decides = true;
}

namespace {
struct HazardScope {
    u32& slot;
    u32 saved;
    HazardScope(u32& s, bool on, u32 value) : slot{s}, saved{s} {
        if (on) {
            slot = value;
        }
    }
    ~HazardScope() {
        slot = saved;
    }
};

u32 HazardLayoutIndex(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::eUndefined:
        return 0;
    case vk::ImageLayout::eGeneral:
        return 1;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return 2;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return 3;
    case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
        return 4;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return 5;
    case vk::ImageLayout::eTransferSrcOptimal:
        return 6;
    case vk::ImageLayout::eTransferDstOptimal:
        return 7;
    default:
        return 8;
    }
}
} // namespace

u32 Runtime::HazardSource() const {
    if (hazard_override != ~0u) {
        return hazard_override;
    }
    const u32 draw_class = Core::NativeRegistry::CurrentDrawClass();
    return draw_class <= 4 ? draw_class : 5;
}

namespace {
/// the planner's list rule: the same as the tracker's bounded lists (above 64 intervals one covering range replaces them, which can only add a barrier)
template <typename List>
void LivePlanAdd(List& list, u64 start, u64 end) {
    constexpr size_t MaxIntervals = 64;
    if (list.Contains(start, end)) {
        return;
    }
    if (list.Size() >= MaxIntervals) {
        u64 lo = start, hi = end;
        for (const auto& interval : list) {
            lo = std::min(lo, interval.start);
            hi = std::max(hi, interval.end);
        }
        list.Clear();
        list.Add({.start = lo, .end = hi});
        return;
    }
    list.Add({.start = start, .end = end});
}
} // namespace

vk::ImageLayout Runtime::CurrentLayout(const VideoCore::Image& image) const {
    if (live_plan_decides && image.backing) {
        if (const auto it = image_plan.find(ImagePlanKey(image, image.backing->num_samples)); it != image_plan.end()) {
            return it->second.state.layout;
        }
    }
    return image.backing ? image.backing->state.layout : vk::ImageLayout::eUndefined;
}

void Runtime::PlanImageBarriers(VideoCore::Image* image, VideoCore::Image::Barriers& barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                                vk::PipelineStageFlags2 dst_stage, std::optional<VideoCore::SubresourceRange> subres_range) {
    if (!image->backing) {
        return;
    }
    const auto [it, created] = image_plan.try_emplace(ImagePlanKey(*image, image->backing->num_samples));
    live_stats.images_planned += created ? 1 : 0;
    auto& last_state = it->second.state;
    auto& subresource_states = it->second.subresources;
    const auto& info = image->info;
    constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
    const bool needs_partial_transition =
        subres_range && (subres_range->base != VideoCore::SubresourceBase{} || subres_range->extent != info.resources);
    const bool partially_transited = !subresource_states.empty();
    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(info.resources.levels * info.resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
        }
        const u32 start_mip = needs_partial_transition ? subres_range->base.level : 0u;
        const u32 end_mip = needs_partial_transition ? subres_range->base.level + subres_range->extent.levels : info.resources.levels;
        const u32 start_layer = needs_partial_transition ? subres_range->base.layer : 0u;
        const u32 end_layer = needs_partial_transition ? subres_range->base.layer + subres_range->extent.layers : info.resources.layers;
        for (u32 mip = start_mip; mip < end_mip; ++mip) {
            for (u32 layer = start_layer; layer < end_layer; ++layer) {
                const auto subres_idx = mip * info.resources.layers + layer;
                if (subres_idx >= subresource_states.size()) {
                    continue;
                }
                auto& state = subresource_states[subres_idx];
                const bool is_write = static_cast<bool>(state.access_mask & write_flags);
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = image->GetImage(),
                        .subresourceRange{
                            .aspectMask = image->aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        },
                    });
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = dst_stage;
                }
            }
        }
        if (!needs_partial_transition) {
            subresource_states.clear();
        }
    } else {
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) {
            return;
        }
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->GetImage(),
            .subresourceRange{
                .aspectMask = image->aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }
    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    last_state.pl_stage = dst_stage;
}

u64 Runtime::LivePlanSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin) {
    const u64 base = handle->CpuAddr();
    auto& seen = live_seen[reinterpret_cast<u64>(handle)];
    const u64 size = handle->SizeBytes();
    if (seen.base != base || seen.size != size) {
        // the buffer cache moved or re-created the buffer: both guest ranges count as read and written by an unknown access (the tracker keeps lists in the buffer's own offset space)
        if (seen.size != 0) {
            ++live_stats.rebases;
            auto& space = live_plan.buffers[0];
            if (seen.base != 0) {
                LivePlanAdd(space.write, seen.base, seen.base + seen.size);
                LivePlanAdd(space.read, seen.base, seen.base + seen.size);
            }
            if (base != 0) {
                LivePlanAdd(space.write, base, base + size);
                LivePlanAdd(space.read, base, base + size);
            }
        }
        seen = {base, size};
    }
    begin = base != 0 ? base + offset : offset;
    return base != 0 ? 0 : reinterpret_cast<u64>(handle);
}

void Runtime::ReplayPlan() {
    constexpr size_t MaxIntervals = 64;
    const auto add_range = [&](Planner::List& list, const Interval& range) {
        if (!planner.bounded) {
            list.Add(range);
            return;
        }
        if (list.Contains(range.start, range.end)) {
            return;
        }
        if (list.Size() >= MaxIntervals) {
            u64 lo = range.start, hi = range.end;
            for (const auto& interval : list) {
                lo = std::min(lo, interval.start);
                hi = std::max(hi, interval.end);
            }
            list.Clear();
            list.Add({.start = lo, .end = hi});
            return;
        }
        list.Add(range);
    };
    for (const PlanEvent& e : plan_log) {
        ++hazard.plan_events;
        switch (e.kind) {
        case PlanEvent::Query: {
            auto& state = planner.buffers[e.handle];
            bool answer = state.write.Overlaps(e.offset, e.offset + e.size);
            if (e.check_read && !answer) {
                answer = state.read.Overlaps(e.offset, e.offset + e.size);
            }
            ++hazard.plan_queries;
            hazard.plan_query_mismatches += answer != e.answer ? 1 : 0;
            {
                // the same plan keyed by guest address (one list for all guest memory) instead of by buffer: what the command processor thread could plan before any buffer exists
                auto& space = planner_by_address.buffers[e.address != 0 ? 0 : e.handle];
                const u64 begin = e.address != 0 ? e.address : e.offset;
                bool by_address = space.write.Overlaps(begin, begin + e.size);
                if (e.check_read && !by_address) {
                    by_address = space.read.Overlaps(begin, begin + e.size);
                }
                hazard.addr_true_tracker_false += by_address && !e.answer ? 1 : 0;
                hazard.addr_false_tracker_true += !by_address && e.answer ? 1 : 0;
                if (!by_address && e.answer) {
                    static u32 logged = 0;
                    if (logged++ < 24) {
                        LOG_ERROR(Render_Vulkan, "C5 planner by address missed a barrier: handle {:#x} offset {:#x} size {:#x} address {:#x} check_read {}", e.handle, e.offset, e.size, e.address, e.check_read);
                    }
                }
            }
            break;
        }
        case PlanEvent::Access: {
            auto& state = planner.buffers[e.handle];
            const Interval range{.start = e.offset, .end = e.offset + e.size};
            if (e.write) {
                add_range(state.write, range);
            }
            if (e.read) {
                add_range(state.read, range);
            }
            planner.memory_pending |= e.staged;
            {
                auto& space = planner_by_address.buffers[e.address != 0 ? 0 : e.handle];
                const u64 begin = e.address != 0 ? e.address : e.offset;
                const Interval range2{.start = begin, .end = begin + e.size};
                if (e.write) {
                    add_range(space.write, range2);
                }
                if (e.read) {
                    add_range(space.read, range2);
                }
                planner_by_address.memory_pending |= e.staged;
            }
            break;
        }
        case PlanEvent::Flush: {
            const bool effective = planner.memory_pending || e.image_pending;
            ++hazard.plan_flushes;
            hazard.plan_flush_mismatches += effective != e.answer ? 1 : 0;
            if (effective) {
                planner.buffers.clear();
                planner.memory_pending = false;
                planner_by_address.buffers.clear();
                planner_by_address.memory_pending = false;
            }
            break;
        }
        }
    }
    plan_log.clear();
}

void Runtime::EndAuditUnit() {
    if (hazard_audit_on) {
        ReplayPlan();
    }
    if (!hazard_audit_on || unit_bits.empty()) {
        return;
    }
    ++hazard.units;
    const auto compare = [&](std::unordered_map<u64, std::vector<u8>>& seen, u64 shape, u64& repeated, u64& identical, u64& bits_differing, u64& bits) {
        auto [it, is_new] = seen.try_emplace(shape);
        if (!is_new) {
            ++repeated;
            bits += unit_bits.size();
            u64 differing = 0;
            for (size_t i = 0; i < unit_bits.size() && i < it->second.size(); ++i) {
                differing += unit_bits[i] != it->second[i] ? 1 : 0;
            }
            bits_differing += differing;
            identical += differing == 0 ? 1 : 0;
        }
        it->second = unit_bits;
    };
    compare(unit_loose_seen, unit_loose, hazard.loose_repeated, hazard.loose_identical, hazard.loose_bits_differing, hazard.loose_bits);
    compare(unit_strict_seen, unit_strict, hazard.strict_repeated, hazard.strict_identical, hazard.strict_bits_differing, hazard.strict_bits);
    unit_bits.clear();
    unit_loose = unit_strict = 0xcbf29ce484222325ull;
}

Runtime::HazardAudit Runtime::TakeHazardAudit() {
    return std::exchange(hazard, HazardAudit{});
}

void Runtime::TickFrame() {
    EndAuditUnit();
    staging_pool.TickFrame();
}

void Runtime::CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                         std::span<const vk::BufferCopy> copies) {
    const HazardScope hazard_scope{hazard_override, hazard_audit_on, 6};
    Core::NativePass::SetEndSite(__builtin_return_address(0));
    scheduler.EndRendering();

    bool needs_flush{};
    for (const auto& copy : copies) {
        needs_flush |= IsBufferAccessed(src, copy.srcOffset, copy.size);
        needs_flush |= IsBufferAccessed(dst, copy.dstOffset, copy.size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(src->Handle(), dst->Handle(), copies);

    for (const auto& copy : copies) {
        AccessBuffer(src, copy.srcOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
        AccessBuffer(dst, copy.dstOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value) {
    const HazardScope hazard_scope{hazard_override, hazard_audit_on, 7};
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, size, true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.fillBuffer(dst->Handle(), offset, size, value);

    AccessBuffer(dst, offset, size, vk::PipelineStageFlagBits2::eClear,
                 vk::AccessFlagBits2::eTransferWrite);
}

void Runtime::InlineData(VideoCore::Buffer* dst, u64 offset, u32 value) {
    const HazardScope hazard_scope{hazard_override, hazard_audit_on, 7};
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, sizeof(value), true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.updateBuffer(dst->Handle(), offset, sizeof(value), &value);

    AccessBuffer(dst, offset, sizeof(value), vk::PipelineStageFlagBits2::eCopy,
                 vk::AccessFlagBits2::eTransferWrite);
}

bool Runtime::Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                      vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                      std::optional<VideoCore::SubresourceRange> subres_range) {
    const size_t prev_num_barriers = static_cast<size_t>(image_barriers.size());
    const u32 layout_before = hazard_audit_on && image->backing ? HazardLayoutIndex(image->backing->state.layout) : 0;
    image->GetBarriers(image_barriers, dst_layout, dst_access, dst_stage, subres_range);
    if (live_plan_on) {
        // L4, images: the planner's barriers next to the image's own; mode 3 takes the planner's
        VideoCore::Image::Barriers planned;
        PlanImageBarriers(image, planned, dst_layout, dst_access, dst_stage, subres_range);
        ++live_stats.image_transits;
        live_stats.image_barriers += planned.size();
        bool same = planned.size() == image_barriers.size() - prev_num_barriers;
        for (size_t i = 0; same && i < planned.size(); ++i) {
            const auto& a = planned[i];
            const auto& b = image_barriers[prev_num_barriers + i];
            same = a.srcStageMask == b.srcStageMask && a.srcAccessMask == b.srcAccessMask && a.dstStageMask == b.dstStageMask &&
                   a.dstAccessMask == b.dstAccessMask && a.oldLayout == b.oldLayout && a.newLayout == b.newLayout && a.image == b.image &&
                   a.subresourceRange == b.subresourceRange;
        }
        if (!same) {
            ++live_stats.image_differences;
            static u32 logged = 0;
            if (logged++ < 24) {
                LOG_ERROR(Render_Vulkan, "L4 image planner differs: image {:#x} uid {} samples {}: planner {} barriers, image state {}", image->info.guest_address,
                          image->image_uid, image->backing ? image->backing->num_samples : 0u, planned.size(), image_barriers.size() - prev_num_barriers);
            }
        }
        if (live_plan_decides) {
            image_barriers.resize(prev_num_barriers);
            for (const auto& barrier : planned) {
                image_barriers.push_back(barrier);
            }
        }
    }
    const bool result = image_barriers.size() != prev_num_barriers;
    if (hazard_audit_on) {
        ++hazard.image_calls;
        if (result) {
            ++hazard.image_true;
            ++hazard.image_true_by_layouts[layout_before][HazardLayoutIndex(dst_layout)][HazardSource()];
        }
    }
    return result;
}

void Runtime::UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                          std::span<const vk::BufferImageCopy> upload_copies) {
    const HazardScope hazard_scope{hazard_override, hazard_audit_on, 8};
    SetBackingSamples(dst, dst->info.num_samples, false);
    scheduler.EndRendering();

    bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite);
    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        needs_flush |= IsBufferAccessed(src, copy.bufferOffset, copy_size);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBufferToImage(src->Handle(), dst->GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             upload_copies);

    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        AccessBuffer(src, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
    }

    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferImageCopy> download_copies) {
    const HazardScope hazard_scope{hazard_override, hazard_audit_on, 8};
    SetBackingSamples(src, src->info.num_samples);
    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        needs_flush |= IsBufferAccessed(dst, copy.bufferOffset, copy_size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->Handle(),
                             download_copies);

    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        AccessBuffer(dst, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::CopyImage(VideoCore::Image* src, VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src->info.pixel_format != dst->info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src->info.pixel_format), vk::to_string(dst->info.pixel_format));
    }

    const u32 base_width = src->info.size.width;
    const u32 base_height = src->info.size.height;
    const u32 base_depth =
        dst->info.type == AmdGpu::ImageType::Color3D ? dst->info.size.depth : src->info.size.depth;

    // Match sample count before copying
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    SmallVector<vk::ImageCopy, 8> regions;

    const vk::ImageAspectFlags src_aspect = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    const vk::ImageAspectFlags dst_aspect = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const bool src_is_2d = ConvertImageType(src->info.type) == vk::ImageType::e2D;
    const bool src_is_3d = ConvertImageType(src->info.type) == vk::ImageType::e3D;

    const bool dst_is_2d = ConvertImageType(dst->info.type) == vk::ImageType::e2D;
    const bool dst_is_3d = ConvertImageType(dst->info.type) == vk::ImageType::e3D;

    const bool is_2d_to_3d = src_is_2d && dst_is_3d;
    const bool is_3d_to_2d = src_is_3d && dst_is_2d;
    const bool is_same_type = !is_2d_to_3d && !is_3d_to_2d;

    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, mip_d);

        vk::ImageCopy region{};
        region.srcSubresource.aspectMask = src_aspect;
        region.srcSubresource.mipLevel = mip;
        region.srcSubresource.baseArrayLayer = 0;
        region.dstSubresource.aspectMask = dst_aspect;
        region.dstSubresource.mipLevel = mip;
        region.dstSubresource.baseArrayLayer = 0;

        if (is_same_type) {
            // 2D->2D OR 3D->3D
            if (src_is_3d) {
                // 3D images must use layerCount=1
                region.srcSubresource.layerCount = 1;
                region.dstSubresource.layerCount = 1;
                region.extent = vk::Extent3D(mip_w, mip_h, mip_d);
            } else {
                // Array images
                const u32 copy_layers = std::min(src_layers, dst_layers);
                region.srcSubresource.layerCount = copy_layers;
                region.dstSubresource.layerCount = copy_layers;
                region.extent = vk::Extent3D(mip_w, mip_h, 1);
            }
        } else if (is_2d_to_3d) {
            // 2D array -> 3D volume
            region.srcSubresource.layerCount = src_layers;
            region.dstSubresource.layerCount = 1;
            region.extent = vk::Extent3D(mip_w, mip_h, src_layers);
        } else if (is_3d_to_2d) {
            // 3D volume -> 2D array
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.layerCount = dst_layers;
            region.extent = vk::Extent3D(mip_w, mip_h, dst_layers);
        }

        regions.push_back(region);
    }

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, regions);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                  const VideoCore::Buffer* buffer, u64 offset) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);
    ASSERT(num_mips == 1);

    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    vk::BufferImageCopy buffer_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
            .mipLevel = 0u,
            .baseArrayLayer = 0,
            .layerCount = num_layers,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };
    const auto copy_size = BufferImageCopySize(buffer_copy, src->info.pixel_format);

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    needs_flush |= IsBufferAccessed(buffer, offset, copy_size);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             buffer->Handle(), buffer_copy);

    const vk::MemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &post_copy_barrier,
    });

    buffer_copy.imageSubresource.aspectMask = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    cmdbuf.copyBufferToImage(buffer->Handle(), dst->GetImage(),
                             vk::ImageLayout::eTransferDstOptimal, buffer_copy);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice) {
    const auto dst_dim = dst->info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(dst->info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(dst->info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max<u32>(dst->info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src->info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src->info.size.width >> src_dim));
    ASSERT(mip_block_h == (src->info.size.height >> src_dim));
    ASSERT(mip_block_p == (src->info.pitch >> src_dim));

    const auto [src_layers, dst_layers] =
        SanitizeCopyLayers(src->info, dst->info, src->info.size.depth);

    const vk::ImageCopy image_copy{
        .srcSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src_layers,
        },
        .dstSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = mip,
            .baseArrayLayer = slice,
            .layerCount = dst_layers,
        },
        .extent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };

    SetBackingSamples(dst, dst->info.num_samples);
    SetBackingSamples(src, src->info.num_samples);

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, image_copy);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.num_samples == 1 && dst->info.num_samples == 1) {
        if ((instance.IsMaintenance8Supported() ||
             src->info.props.is_depth == dst->info.props.is_depth) &&
            src->aspect_mask == dst->aspect_mask) {
            CopyImage(src, dst);
        } else {
            // Perform depth from/to color copy using the intermediate copy buffer.
            static constexpr size_t COPY_BUFFER_SIZE = 128_MB;
            const auto copy_ref =
                staging_pool.Request(COPY_BUFFER_SIZE, VideoCore::MemoryType::DeviceLocal);
            CopyImageWithBuffer(src, dst, copy_ref.buffer, copy_ref.offset);
        }
    } else if (src->info.num_samples == 1 && dst->info.num_samples > 1 &&
               dst->info.props.is_depth) {
        // Perform a rendering pass to transfer the channels of source as samples in dest.
        bool needs_flush =
            Transit(src, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
        needs_flush |= Transit(dst, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                               vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                   vk::PipelineStageFlagBits2::eLateFragmentTests,
                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
        if (needs_flush) {
            FlushBarriers();
        }

        blit_helper->ReinterpretColorAsMsDepth(
            dst->info.size.width, dst->info.size.height, dst->info.num_samples,
            src->info.pixel_format, dst->info.pixel_format, src->GetImage(), dst->GetImage());
    } else {
        LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
    }
}

void Runtime::CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                               const VideoCore::SubresourceRange& sub_range) {
    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto aspect_mask = src->aspect_mask & dst->aspect_mask;

    const vk::ImageCopy region = {
        .srcSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .dstOffset = {0, 0, 0},
        .extent = {dst->info.size.width, dst->info.size.height, 1},
    };

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                           const VideoCore::SubresourceRange& src_range,
                           const VideoCore::SubresourceRange& dst_range) {
    SetBackingSamples(dst, 1, false);
    scheduler.EndRendering();

    const bool needs_resolve = src->backing->num_samples != 1;
    const auto dst_stage =
        needs_resolve ? vk::PipelineStageFlagBits2::eResolve : vk::PipelineStageFlagBits2::eCopy;

    bool needs_flush = Transit(src, vk::ImageLayout::eTransferSrcOptimal, dst_stage,
                               vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal, dst_stage,
                           vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, 1);
    if (!needs_resolve) {
        const vk::ImageCopy region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->info.size.width, src->info.size.height, 1},
        };
        cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                         vk::ImageLayout::eTransferDstOptimal, region);
    } else {
        const vk::ImageResolve region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->info.size.width, src->info.size.height, 1},
        };
        cmdbuf.resolveImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                            vk::ImageLayout::eTransferDstOptimal, region);
    }

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                         const vk::ClearValue& clear_value) {
    scheduler.EndRendering();

    const bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eTransferWrite, range);
    if (needs_flush) {
        FlushBarriers();
    }

    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.clearColorImage(dst->GetImage(), vk::ImageLayout::eTransferDstOptimal, clear_value.color,
                           vk_range);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ClearDepthStencilImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range, bool clear_depth,
                                     float depth, bool clear_stencil, u32 stencil) {
    vk::ImageAspectFlags aspect{};
    if (clear_depth && (dst->aspect_mask & vk::ImageAspectFlagBits::eDepth)) {
        aspect |= vk::ImageAspectFlagBits::eDepth;
    }
    if (clear_stencil && (dst->aspect_mask & vk::ImageAspectFlagBits::eStencil)) {
        aspect |= vk::ImageAspectFlagBits::eStencil;
    }
    if (!aspect) {
        return;
    }
    scheduler.EndRendering();

    const bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eTransferWrite, range);
    if (needs_flush) {
        FlushBarriers();
    }

    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = aspect,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    const vk::ClearDepthStencilValue value{.depth = depth, .stencil = stencil};
    scheduler.CommandBuffer().clearDepthStencilImage(dst->GetImage(), vk::ImageLayout::eTransferDstOptimal, value,
                                                     vk_range);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing) {
    auto& backing = image->backing;
    auto& backing_images = image->backing_images;
    const auto& info = image->info;
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!image->info.props.is_depth, "Swapping samples is only valid for color images");
    VideoCore::Image::BackingImage* new_backing;
    auto it = std::ranges::find(backing_images, num_samples,
                                &VideoCore::Image::BackingImage::num_samples);
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, image->supported_samples);

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->image = VideoCore::UniqueImage{instance.GetDevice(), instance.GetAllocator()};
        new_backing->image.Create(new_image_ci);

        Vulkan::SetObjectName(instance.GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (copy_backing) {
        scheduler.EndRendering();
        ASSERT(image->info.resources.levels == 1 && image->info.resources.layers == 1);

        // Transition current backing to shader read layout
        Transit(image, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);

        // Transition dest backing to color attachment layout, not caring of previous contents
        constexpr auto dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        constexpr auto dst_access =
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        constexpr auto dst_layout = vk::ImageLayout::eColorAttachmentOptimal;
        image_barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = new_backing->image,
            .subresourceRange{
                .aspectMask = image->aspect_mask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = image->info.resources.layers,
            },
        });
        FlushBarriers();

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(
            info.size.width, info.size.height, new_backing->num_samples, info.pixel_format,
            backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
        if (live_plan_on) {
            // the image planner learns the new backing's state the same way (no transition of its own made it)
            image_plan[ImagePlanKey(*image, new_backing->num_samples)] = {{dst_stage, dst_access, dst_layout}, {}};
        }
    }

    backing = new_backing;
}

bool Runtime::IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                               bool check_read_access) {
    MakeCurrent(handle);
    bool has_access = resource->write_ranges.Overlaps(offset, offset + size);
    const bool plan_logged_has_write = has_access;
    (void)plan_logged_has_write;
    bool by_read = false;
    if (check_read_access && !has_access) {
        has_access |= resource->read_ranges.Overlaps(offset, offset + size);
        by_read = has_access;
    }
    if (hazard_audit_on) {
        plan_log.push_back({PlanEvent::Query, check_read_access, false, false, false, false, has_access, reinterpret_cast<u64>(handle), offset, size, handle->CpuAddr() != 0 ? handle->CpuAddr() + offset : 0});
        ++hazard.buffer_queries;
        {
            auto& ask = hazard.ask_keys[{HazardSource(), hazard_program, reinterpret_cast<u64>(handle)}];
            ++ask.first;
            ask.second += has_access ? 1 : 0;
            unit_bits.push_back(has_access ? 1 : 0);
            const u64 loose = (u64{HazardSource()} << 56) ^ hazard_program;
            unit_loose = (unit_loose ^ loose) * 0x100000001b3ull;
            unit_strict = (unit_strict ^ loose ^ (reinterpret_cast<u64>(handle) * 0x9E3779B97F4A7C15ull)) * 0x100000001b3ull;
        }
        if (has_access) {
            ++hazard.buffer_true;
            u32 writer = HazardAudit::Sources - 1; // unknown unless a record is found
            u64 writer_program = 0;
            for (auto it = resource->audit.rbegin(); it != resource->audit.rend(); ++it) {
                if (it->write == !by_read && it->start < offset + size && offset < it->end) {
                    writer = it->source;
                    writer_program = it->program;
                    break;
                }
            }
            ++hazard.buffer_true_by_writer_reader[writer][HazardSource()][check_read_access ? 1 : 0];
            if (writer == 3 && HazardSource() == 3) {
                ++hazard.compute_pairs[{writer_program, hazard_program}];
            }
        }
    }
    if (live_plan_on) {
        u64 begin = 0;
        auto& space = live_plan.buffers[LivePlanSpace(handle, offset, begin)];
        bool planned = space.write.Overlaps(begin, begin + size);
        if (check_read_access && !planned) {
            planned = space.read.Overlaps(begin, begin + size);
        }
        ++live_stats.queries;
        live_stats.answered_true += planned ? 1 : 0;
        live_stats.planner_extra += planned && !has_access ? 1 : 0;
        if (!planned && has_access) {
            // TLG_BARRIER_PLANNER=2 keeps an exact list (no collapse above 64 intervals) next to the planner: if it also says "no hazard" the tracker's answer came from its own
            // collapse (a covering range over the gaps of one big buffer), not from an access the planner lacks
            bool exact_true = true;
            if (live_plan_exact) {
                auto& exact = live_exact.buffers[LivePlanSpace(handle, offset, begin)];
                exact_true = exact.write.Overlaps(begin, begin + size) || (check_read_access && exact.read.Overlaps(begin, begin + size));
            }
            if (exact_true) {
                ++live_stats.planner_missed;
                static u32 logged = 0;
                if (logged++ < 24) {
                    LOG_ERROR(Render_Vulkan, "L4 planner missed a barrier: handle {:#x} offset {:#x} size {:#x} base {:#x} check_read {}", reinterpret_cast<u64>(handle), offset, size,
                              u64{handle->CpuAddr()}, check_read_access);
                }
            } else {
                ++live_stats.tracker_collapse_only;
            }
        }
        // mode 1 and 2: the tracker corrects a miss (union); mode 3: the planner alone decides, the tracker is only compared (a miss is counted and logged, no barrier is made for it)
        has_access = live_plan_decides ? planned : has_access || planned;
    }
    return has_access;
}

void Runtime::AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                           vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access) {
    MakeCurrent(handle);

    const Interval range = {
        .start = offset,
        .end = offset + size,
    };

    constexpr static vk::AccessFlags2 READ_MASK =
        vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eVertexAttributeRead |
        vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eShaderRead |
        vk::AccessFlagBits2::eColorAttachmentRead |
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eTransferRead |
        vk::AccessFlagBits2::eMemoryRead;

    constexpr static vk::AccessFlags2 WRITE_MASK =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eTransferWrite |
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eTransformFeedbackWriteEXT;

    // The lists are only consulted to decide whether a later access needs a barrier. A list that grows
    // without a barrier in between (thousands of tiny constant buffer ranges) made every insert shift the
    // vector; with the native draw path on, a list that holds many ranges is replaced by the one range that
    // covers them all, which can only add a barrier, never drop one.
    const bool bounded = VideoCore::ReadbackPolicy::Get().native_draw_mode.load(
                             std::memory_order_relaxed) >= 2;
    const auto add_range = [bounded](auto& list, const Interval& new_range) {
        if (!bounded) {
            list.Add(new_range);
            return;
        }
        if (list.Contains(new_range.start, new_range.end)) {
            return;
        }
        constexpr size_t MaxIntervals = 64;
        if (list.Size() >= MaxIntervals) {
            u64 lo = new_range.start;
            u64 hi = new_range.end;
            for (const auto& interval : list) {
                lo = std::min(lo, interval.start);
                hi = std::max(hi, interval.end);
            }
            list.Clear();
            list.Add({.start = lo, .end = hi});
            return;
        }
        list.Add(new_range);
    };
    if (src_access & WRITE_MASK) {
        add_range(resource->write_ranges, range);
    }
    if (src_access & READ_MASK) {
        add_range(resource->read_ranges, range);
    }
    if (hazard_audit_on) {
        auto& audit = resource->audit;
        if (audit.size() >= 512) {
            audit.erase(audit.begin(), audit.begin() + 256);
        }
        const u8 source = static_cast<u8>(HazardSource());
        if (src_access & WRITE_MASK) {
            audit.push_back({offset, offset + size, source, true, hazard_program});
        }
        if (src_access & READ_MASK) {
            audit.push_back({offset, offset + size, source, false, hazard_program});
        }
    }

    if (hazard_audit_on) {
        plan_log.push_back({PlanEvent::Access, false, (src_access & WRITE_MASK) != vk::AccessFlags2{}, (src_access & READ_MASK) != vk::AccessFlags2{}, src_stage != vk::PipelineStageFlags2{}, false, false,
                            reinterpret_cast<u64>(handle), offset, size, handle->CpuAddr() != 0 ? handle->CpuAddr() + offset : 0});
    }
    if (live_plan_on) {
        u64 begin = 0;
        auto& space = live_plan.buffers[LivePlanSpace(handle, offset, begin)];
        if (src_access & WRITE_MASK) {
            LivePlanAdd(space.write, begin, begin + size);
        }
        if (src_access & READ_MASK) {
            LivePlanAdd(space.read, begin, begin + size);
        }
        live_plan.memory_pending |= src_stage != vk::PipelineStageFlags2{};
        if (live_plan_exact) {
            auto& exact = live_exact.buffers[LivePlanSpace(handle, offset, begin)];
            if (src_access & WRITE_MASK) {
                exact.write.Add({.start = begin, .end = begin + size});
            }
            if (src_access & READ_MASK) {
                exact.read.Add({.start = begin, .end = begin + size});
            }
        }
    }
    memory_barrier.srcStageMask |= src_stage;
    memory_barrier.srcAccessMask |= src_access & WRITE_MASK;
}

void Runtime::FlushBarriers() {
    vk::DependencyInfo dep_info{};

    if (memory_barrier.srcStageMask) {
        dep_info.pMemoryBarriers = &memory_barrier;
        dep_info.memoryBarrierCount = 1U;
    }
    if (!image_barriers.empty()) {
        dep_info.pImageMemoryBarriers = image_barriers.data();
        dep_info.imageMemoryBarrierCount = static_cast<u32>(image_barriers.size());
    }

    if (hazard_audit_on) {
        plan_log.push_back({PlanEvent::Flush, false, false, false, false, !image_barriers.empty(), dep_info.memoryBarrierCount != 0 || dep_info.imageMemoryBarrierCount != 0, 0, 0, 0, 0});
    }
    if (live_plan_on) {
        const bool actual = dep_info.memoryBarrierCount != 0 || dep_info.imageMemoryBarrierCount != 0;
        const bool planned = live_plan.memory_pending || !image_barriers.empty();
        ++live_stats.flushes;
        live_stats.flush_differences += planned != actual ? 1 : 0;
        if (actual) {
            live_plan.buffers.clear();
            live_plan.memory_pending = false;
            live_exact.buffers.clear();
        }
    }
    if (!dep_info.memoryBarrierCount && !dep_info.imageMemoryBarrierCount) {
        return;
    }

    Core::NativePass::SetEndCause(Core::NativePass::EndCause::Barrier);
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(dep_info);

    memory_barrier.srcStageMask = vk::PipelineStageFlagBits2::eNone;
    memory_barrier.srcAccessMask = vk::AccessFlagBits2::eNone;

    image_barriers.clear();
    for (auto& resource : resources) {
        resource.read_ranges.Clear();
        resource.write_ranges.Clear();
    }
    resources.clear();
    resource = nullptr;
}

void Runtime::MakeCurrent(const VideoCore::Buffer* handle) {
    if (resource && resource->handle == handle) {
        return;
    }
    auto it = std::ranges::find(resources, handle, &BufferBarriers::handle);
    if (it != resources.end()) {
        resource = std::addressof(*it);
        return;
    }
    resource = &resources.emplace_back(handle);
}

} // namespace Vulkan
