// SPDX-License-Identifier: GPL-2.0-or-later
// GPU time per submission (documents/tlg/native plan/native-data-path-notes.md, "GPU time"): every queue submission is bracketed by two tiny pre-recorded command buffers that write a
// timestamp (all commands done before it / all commands of the submission done), so the time between the two is the GPU's busy time for that submission and the time between one
// submission's end and the next one's start is the GPU waiting (for the CPU to submit, for a semaphore). Results are read back later without waiting and summed per second into
// /stats "gpu_time". Switch: TLG_GPU_TIME=0 turns it off (on by default: two timestamps per submission).

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

class GpuTime {
public:
    struct Bracket {
        vk::CommandBuffer begin{};
        vk::CommandBuffer end{};
    };

    /// The profiler for this device, or null when switched off.
    static GpuTime* Get(const Instance& instance);

    /// The two command buffers to put in front of and behind the submission of `stream` (empty when the ring is full).
    [[nodiscard]] Bracket Open(u32 stream);
    /// Reads back finished submissions and publishes the sums about once a second.
    void Poll();

private:
    explicit GpuTime(const Instance& instance);

    static constexpr u32 Streams = 2;
    static constexpr u32 SlotsPerStream = 1024;

    struct Window {
        u64 submissions = 0;
        double busy_ms = 0, gap_ms = 0, long_gap_ms = 0, max_span_ms = 0;
        u64 long_gaps = 0;
        u64 total_submissions = 0;
    };

    vk::Device device;
    vk::UniqueQueryPool pool;
    vk::UniqueCommandPool command_pool;
    std::vector<vk::CommandBuffer> begin_buffers, end_buffers;
    double period_ns = 1.0;
    u64 valid_mask = ~0ull;

    std::mutex mutex;
    std::array<u32, Streams> next_slot{};
    std::array<std::deque<u32>, Streams> pending;
    std::array<double, Streams> last_end_ms{};
    std::array<bool, Streams> have_last{};
    std::array<Window, Streams> window{};
    std::array<u64, Streams> dropped{};
    std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

} // namespace Vulkan
