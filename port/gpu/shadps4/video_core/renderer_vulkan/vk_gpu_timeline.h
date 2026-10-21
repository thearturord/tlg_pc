// SPDX-License-Identifier: GPL-2.0-or-later
// tlg-native: GPU side timestamps for the frame timeline capture (documents/tlg/frame-timeline-capture-design.md, source D).
//
// While a capture runs, the raster thread records a timestamp command into the command list at every start and end of a render pass and around every compute
// dispatch (an all-commands timestamp: written when everything before it has finished, so the time between two stamps is charged to what ends at the second).
// The encoder thread adds a stamp at the start and at the end of each command buffer, remembers which stamps belong to which submission, and after the
// submission has completed reads them back and writes events into the capture ring on the CPU clock (calibrated with VK_EXT_calibrated_timestamps).

#pragma once

#include <deque>
#include <mutex>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Semaphore;

/// What a stamp ends (kind), recorded next to its query index.
enum GpuStampKind : u8 {
    GpuStampPassBegin = 0x01,
    GpuStampPassEndDepthOnly = 0x11,
    GpuStampPassEndColor = 0x12,
    GpuStampPassEndGBuffer = 0x13,
    GpuStampDispatchBegin = 0x20,
    GpuStampDispatchEnd = 0x21,
};

struct GpuStamp {
    u32 index;
    u8 kind;
    u64 detail;
};

class GpuTimeline {
public:
    explicit GpuTimeline(const Instance& instance);
    ~GpuTimeline();

    static constexpr u32 NumQueries = 16384;

    /// Next query of the ring, reset for use (host reset).
    u32 Allocate();
    [[nodiscard]] vk::QueryPool Pool() const noexcept {
        return *pool;
    }

    /// A submission whose stamps are read once `tick` has completed on the GPU.
    struct Pending {
        u64 tick;
        std::vector<GpuStamp> stamps;
        u32 begin_index;
        u32 end_index;
    };
    void Submitted(Pending&& pending);
    /// Reads and reports the submissions that have completed (encoder thread).
    void Collect(Semaphore& work_semaphore);

private:
    struct Calibration {
        u64 gpu_ticks{};
        u64 cpu_ns{};
        bool valid{};
    };
    void Calibrate();
    [[nodiscard]] u64 ToCpuNs(u64 gpu_ticks) const;

    const Instance& instance;
    vk::UniqueQueryPool pool;
    double period_ns{1.0};
    std::atomic<u32> next{0};
    std::mutex mutex;
    std::deque<Pending> pending;
    Calibration calibration;
    u64 last_calibration_ns{};
    bool calibrated_supported{};
};

} // namespace Vulkan
