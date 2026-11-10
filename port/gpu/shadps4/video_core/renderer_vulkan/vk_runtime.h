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

/// Phase 0b barrier-necessity census (`TLG_BARRIER_CENSUS=1`, default off, counters only): for each buffer hazard that makes a barrier, who produced the pending bytes, who asks for them, and how much of the
/// asking range overlaps. Report() is a JSON text for /stats. The calls cost nothing when it is off.
/// G2 (`TLG_SIM_INDEPENDENT`, default 2 since 2026-11-09, =0 off): operations of one "independent group" (the SimParticle dispatches: their slices of the particle buffers never overlap, Phase 0b) need no barrier between
/// each other. 1 shadow: the tracker answers as before and the group-aware answer is counted next to it; 2 real: the group-aware answer decides. Another operation still sees the group's
/// pending writes and reads, so every barrier to or from anything else stays. Counters: see IndepReport().
namespace IndependentGroup {
[[nodiscard]] u32 Mode();
std::string Report();
void NoteDispatch(bool had_barrier, bool real_buffer_hazard, bool aware_buffer_hazard);
} // namespace IndependentGroup

namespace BarrierCensus {
[[nodiscard]] bool On();
void SetOp(u64 tag);                  ///< a draw or dispatch starts (the tag names its program; graphics use a fixed tag)
void Dispatch(u64 tag, bool barrier, u64 pgm_hash = 0); ///< a dispatch finished binding; `barrier` = the tracker will flush before it
/// G2 step 1: one SimParticle dispatch. `status`: 0 ok, 1 no control buffer found, 2 control buffer too short, 3 count 0, 4 slice outside the pool, 5 no pool buffer found.
/// The slice is the guest byte range [begin, end) of the pool the dispatch reads and writes.
void SimSlice(u64 tag, u32 status, bool from_ring, u64 pool_key, u64 begin, u64 end, u32 count, u32 base);
void SimBinding(u64 tag, u32 position, u64 size, bool write);
/// G2: the render slot (word 6 of each particle record, read from guest memory) of the particles of one dispatch: how many equal the particle index, are zero, repeat inside the last dispatches
void SimSlots(const u32* slots, const u32* index, u32 n, bool memory_readable);
std::string Report();
} // namespace BarrierCensus

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

    /// G2: the group of the operation being bound (0 none, 1 SimParticle). Set by the rasterizer around one dispatch, cleared after.
    void SetIndependentGroup(u32 group) noexcept {
        independent_group = group;
        op_real_hazard = op_aware_hazard = false;
    }
    /// G2: the buffer hazards the operation since SetIndependentGroup met: by the old answer and by the group-aware one
    void TakeHazardFlags(bool& real, bool& aware) noexcept {
        real = op_real_hazard;
        aware = op_aware_hazard;
    }

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
    u32 independent_group{};
    bool op_real_hazard{}, op_aware_hazard{};
    Planner aware_normal, aware_group; ///< G2: the group-aware planner: accesses of operations outside any group, and accesses of the group (kept apart)
    Planner live_object; ///< B10: the same state machine keyed by object: a placed engine buffer (its own Vulkan buffer) in its own offset space
    u64 LiveObjectSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin);
    /// Design review 72 (`TLG_EPOCH_PLANNER`, default 0 off; 1 shadow, 2 real): one planner keyed by object whose entries survive a flush. A flush moves the epoch on instead of clearing
    /// the maps (no node freed and made again per key per flush window, no list allocated again); an entry of an older epoch counts as empty and is reset at its next access.
    /// It keeps the group-aware lists in the same entry and answers what the deciding planner answers (the group-aware one in G2 mode 2, the object planner otherwise).
    /// Shadow: every query compared with the old answer, which decides. Real: the old planners run only in one flush window in 64 (audit), where the answer is the OR of both.
    struct EpochEntry {
        u64 epoch{0};
        Planner::List read, write, group_read, group_write;
    };
    absl::node_hash_map<u64, EpochEntry> epoch_entries;
    u64 plan_epoch{1};
    bool epoch_audit{true}; ///< real mode: the old planners run in this flush window too
    u64 epoch_flushes{0};
public:
    struct EpochStats {
        u64 queries{}, compared{}, different{}, epoch_true_old_false{}, epoch_false_old_true{}, accesses{}, entries_reset{}, entries_new{}, cap_clears{}, audit_windows{}, windows{};
    };
    [[nodiscard]] static u32 EpochPlannerMode() noexcept;
    [[nodiscard]] EpochStats ReadEpochStats() const noexcept {
        return epoch_stats;
    }
    [[nodiscard]] size_t EpochEntries() const noexcept {
        return epoch_entries.size();
    }
private:
    EpochStats epoch_stats{};
    bool EpochQuery(u64 key, u64 begin, u64 size, bool check_read) const;
    void EpochAccess(u64 key, u64 begin, u64 size, bool write, bool read);
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
