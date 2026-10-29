// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

    void FlushBarriers();

    /// C5, the hazard audit (`TLG_HAZARD_AUDIT=1`, read only): which access made which later access need a barrier. Sources: 0 z / shadow draws, 1 g-buffer, 2 forward, 3 compute,
    /// 4 indirect draws, 5 untagged draws, 6 buffer copies (the buffer cache's uploads), 7 fills and inline data, 8 image copies and uploads, 9 unknown (ranges merged by the bound).
    /// Layouts: 0 undefined, 1 general, 2 color attachment, 3 depth stencil attachment, 4 depth stencil read only, 5 shader read only, 6 transfer src, 7 transfer dst, 8 other.
    struct HazardAudit {
        static constexpr u32 Sources = 10, Layouts = 9;
        u64 buffer_queries{}, buffer_true{};
        u64 buffer_true_by_writer_reader[Sources][Sources][2]{}; ///< [the access that left the range][the access that asked][the asker writes]
        u64 image_calls{}, image_true{};
        u64 image_true_by_layouts[Layouts][Layouts][Sources]{}; ///< [layout before][layout asked][the source asking]
        /// C5, compute half: the true buffer hazards between two compute dispatches, by (program address of the dispatch that left the range, program address of the one that asked)
        std::map<std::pair<u64, u64>, u64> compute_pairs;
        /// C5 recipe holdout: per asking access (source, compute program address or 0) the buffer queries and how many of them were true
        std::map<std::tuple<u32, u64, u64>, std::pair<u64, u64>> ask_keys; ///< key: (source, program, the buffer asked about)
        /// C5 frame recipe test, per submission ("unit"): the sequence of the tracker's answers as a bit vector, compared with the last unit that had the same shape (the sequence of
        /// askers, loose = (source, program), strict = also the buffer): units seen, units whose shape was seen before, of those the bit identical ones, the differing bits, the bits compared
        u64 plan_queries{}, plan_query_mismatches{}, plan_flushes{}, plan_flush_mismatches{}, plan_events{}, addr_true_tracker_false{}, addr_false_tracker_true{};
        u64 units{}, loose_repeated{}, loose_identical{}, loose_bits_differing{}, loose_bits{}, strict_repeated{}, strict_identical{}, strict_bits_differing{}, strict_bits{};
    };
    /// The compute program whose dispatch is being bound (the hazard audit attributes its accesses to it).
    void SetHazardProgram(u64 address) noexcept {
        hazard_program = address;
    }
    [[nodiscard]] bool HazardAuditOn() const noexcept {
        return hazard_audit_on;
    }
    HazardAudit TakeHazardAudit();

    /// L4: `TLG_BARRIER_PLANNER=1` makes the planner (a state machine over the access list keyed by guest address, see `Planner`) answer the buffer hazard queries; the tracker still runs
    /// next to it and corrects it (a barrier the planner lacks and the tracker has is counted as a miss and the barrier is made). 0 misses over a long run is the exit of L4.
    struct LivePlanStats {
        u64 queries{}, answered_true{}, planner_missed{}, planner_extra{}, tracker_collapse_only{}, flushes{}, flush_differences{}, rebases{};
        u64 image_transits{}, image_barriers{}, image_differences{}, images_planned{};
    };
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
    void MakeCurrent(const VideoCore::Buffer* handle);

private:
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    struct HazardRecord {
        u64 start, end;
        u8 source;
        bool write;
        u64 program;
    };
    u64 hazard_program{0};
    struct BufferBarriers {
        const VideoCore::Buffer* handle;
        using AccessList = IntervalList<Interval>;
        AccessList read_ranges;
        AccessList write_ranges;
        std::vector<HazardRecord> audit; ///< hazard audit only
    };
    BufferBarriers* resource{};
    bool hazard_audit_on{};
    std::vector<u8> unit_bits;
    u64 unit_loose{0xcbf29ce484222325ull}, unit_strict{0xcbf29ce484222325ull};
    std::unordered_map<u64, std::vector<u8>> unit_loose_seen, unit_strict_seen;
    void EndAuditUnit();
    /// C5 planner (audit only): the buffer tracker as a pure state machine over a log of events (query, access, flush). The log is the "frame recipe as data"; replaying it through the
    /// planner must give the live tracker's answer to every query and the live flush's effect: 0 mismatches means the barrier plan is a deterministic function of the access list.
    struct PlanEvent {
        enum Kind : u8 { Query, Access, Flush } kind;
        bool check_read{}, write{}, read{}, staged{}, image_pending{}, answer{}; ///< query: check_read, answer; access: write, read, staged (src stage non zero); flush: image_pending, answer = effective
        u64 handle{}, offset{}, size{};
        u64 address{}; ///< the guest address of the range (the buffer's CPU address + offset), 0 for a buffer that is not guest memory
    };
    struct Planner {
        using List = IntervalList<Interval>;
        struct State {
            List read, write;
        };
        std::unordered_map<u64, State> buffers;
        bool memory_pending{false};
        bool bounded{true};
    };
    std::vector<PlanEvent> plan_log;
    Planner planner, planner_by_address;
    void ReplayPlan();
    bool live_plan_on{}, live_plan_exact{}, live_plan_decides{};
    /// P1: the old buffer tracker runs next to the planner only as a comparison (`TLG_BARRIER_TRACKER=1`, or the hazard audit); off, the planner alone answers
    bool tracker_on{};
    u64 live_plan_writes{}; ///< R2: write ranges added to the live plan since its last clear
    const VideoCore::Buffer* plan_last_handle{}; ///< LivePlanSpace: the buffer of the last call and the range it had (no map lookup when it is the same)
    u64 plan_last_base{}, plan_last_size{};
    Planner live_exact; ///< TLG_BARRIER_PLANNER=2: the same lists without the collapse, the oracle for a miss
    Planner live_plan; ///< L4: updated by every access, consulted by every query, cleared by every effective flush
    struct SeenBuffer {
        u64 base{}, size{};
    };
    std::unordered_map<u64, SeenBuffer> live_seen; ///< buffer object -> the guest range it covered when last seen (a change is a rebase)
    LivePlanStats live_stats{};
    /// the list a range lives in (0 = all guest memory, else the buffer's own offset space) and its start there
    u64 LivePlanSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin);
    /// L4, images: the layout / access / stage of every image the planner has seen, per backing (sample count) and per subresource once a
    /// partial transition split it, keyed by the image's unique id; the same state machine as Image::GetBarriers, kept apart from the image
    struct ImagePlan {
        VideoCore::Image::State state;
        std::vector<VideoCore::Image::State> subresources;
    };
    std::unordered_map<u64, ImagePlan> image_plan;
    static u64 ImagePlanKey(const VideoCore::Image& image, u32 num_samples) {
        return (image.image_uid << 6) | (num_samples & 63);
    }
    void PlanImageBarriers(VideoCore::Image* image, VideoCore::Image::Barriers& out, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                           vk::PipelineStageFlags2 dst_stage, std::optional<VideoCore::SubresourceRange> subres_range);
    u32 hazard_override{~0u}; ///< set inside the runtime's own copies, fills and image transfers
    HazardAudit hazard{};
    u32 HazardSource() const;
    std::vector<BufferBarriers> resources;
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
};

} // namespace Vulkan
