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
    u32 hazard_override{~0u}; ///< set inside the runtime's own copies, fills and image transfers
    HazardAudit hazard{};
    u32 HazardSource() const;
    std::vector<BufferBarriers> resources;
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
};

} // namespace Vulkan
