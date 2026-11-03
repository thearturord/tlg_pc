// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <absl/container/node_hash_map.h>
#include <map>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class BlitHelper;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;

class Runtime {
public:
    explicit Runtime(const Instance& instance, Scheduler& scheduler);
    ~Runtime() = default;

    const Instance& GetInstance() const {
        return instance;
    }

    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }

    void TickFrame();

    void CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const vk::BufferCopy> copies);

    void FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value);

    void InlineData(VideoCore::Buffer* dst, u64 offset, u32 value);

    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> subres_range = {});

    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> upload_copies);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> download_copies);

    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                             const VideoCore::Buffer* buffer, u64 offset);
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);

    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);

    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& sub_range);

    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& clear_value);
    void ClearDepthStencilImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range, bool clear_depth,
                                float depth, bool clear_stencil, u32 stencil);

    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    void AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                      vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access);

    bool IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                          bool check_read_access = false);
    /// R2: false when the planner holds no write since the last barrier (a read then needs no barrier, whatever it reads)
    [[nodiscard]] bool PlanHasWrites() const noexcept {
        return live_plan_writes != 0;
    }
    /// R3 step 2: the planner decides the image layouts (after a transit to a layout, CurrentLayout of an image with a backing is that layout)
    [[nodiscard]] bool PlannerDecidesLayouts() const noexcept {
        return live_plan_on && live_plan_decides;
    }

    void FlushBarriers();

    /// L4: `TLG_BARRIER_PLANNER=1` makes the planner (a state machine over the access list keyed by guest address, see `Planner`) answer the buffer hazard queries; the tracker still runs
    /// next to it and corrects it (a barrier the planner lacks and the tracker has is counted as a miss and the barrier is made). 0 misses over a long run is the exit of L4.
    struct LivePlanStats {
        u64 queries{}, answered_true{}, planner_missed{}, planner_extra{}, tracker_collapse_only{}, flushes{}, flush_differences{}, rebases{};
        u64 barriers_recorded{}, barriers_memory_only{}, barriers_ending_a_pass{}, barriers_from_compute_writes{}; ///< GPU track: pipeline barriers actually recorded
        u64 image_transits{}, image_barriers{}, image_differences{}, images_planned{};
        /// B10: buffer hazards by object (a placed engine buffer in its own offset space, the arena by guest address)
        u64 object_queries{}, object_same{}, object_no_address_yes{}, object_yes_address_no{}, object_space_queries{};
    };
    [[nodiscard]] u32 B10Mode() const noexcept {
        return b10_mode;
    }
    [[nodiscard]] bool LivePlannerOn() const noexcept {
        return live_plan_on;
    }
    /// cumulative since start (read from another thread: counters only, a torn read is harmless)
    [[nodiscard]] LivePlanStats ReadLivePlanStats() const {
        return live_stats;
    }
    /// The layout an image is in for a descriptor or an attachment: the image planner's when it decides (mode 3), else the image's own state
    [[nodiscard]] vk::ImageLayout CurrentLayout(const VideoCore::Image& image) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    struct Planner {
        using List = IntervalList<Interval>;
        struct State {
            List read, write;
        };
        absl::node_hash_map<u64, State> buffers; // R5 step 1: node map (stable references, faster lookups than std)
        bool memory_pending{false};
    };
    bool live_plan_on{}, live_plan_decides{};
    u64 live_plan_writes{}; ///< R2: write ranges added to the live plan since its last clear
    const VideoCore::Buffer* plan_last_handle{}; ///< LivePlanSpace: the buffer of the last call and the range it had (no map lookup when it is the same)
    u64 plan_last_base{}, plan_last_size{};
    Planner live_plan; ///< L4: updated by every access, consulted by every query, cleared by every effective flush
    /// B10 (`TLG_B10_OBJECTS`): 0 off, 1 audit (the object planner runs next to the address planner, which answers), 2 native (the object planner answers alone)
    u32 b10_mode{};
    Planner live_object; ///< B10: the same state machine keyed by object: a placed engine buffer (its own Vulkan buffer) in its own offset space
    u64 LiveObjectSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin);
    struct SeenBuffer {
        u64 base{}, size{};
    };
    absl::node_hash_map<u64, SeenBuffer> live_seen; ///< buffer object -> the guest range it covered when last seen (a change is a rebase)
    LivePlanStats live_stats{};
    /// the list a range lives in (0 = all guest memory, else the buffer's own offset space) and its start there
    u64 LivePlanSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin);
    /// L4, images: the layout / access / stage of every image the planner has seen, per backing (sample count) and per subresource once a
    /// partial transition split it, keyed by the image's unique id; the same state machine as Image::GetBarriers, kept apart from the image
    struct ImagePlan {
        VideoCore::Image::State state;
        std::vector<VideoCore::Image::State> subresources;
    };
    absl::node_hash_map<u64, ImagePlan> image_plan;
    static u64 ImagePlanKey(const VideoCore::Image& image, u32 num_samples) {
        return (image.image_uid << 6) | (num_samples & 63);
    }
    void PlanImageBarriers(VideoCore::Image* image, VideoCore::Image::Barriers& out, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                           vk::PipelineStageFlags2 dst_stage, std::optional<VideoCore::SubresourceRange> subres_range);
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
};

} // namespace Vulkan
