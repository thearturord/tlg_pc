// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "video_core/renderer_vulkan/vk_gpu_time.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

GpuTime* GpuTime::Get(const Instance& instance) {
    static GpuTime* const result = [&]() -> GpuTime* {
        const char* e = std::getenv("TLG_GPU_TIME");
        if (e != nullptr && e[0] == '0') {
            return nullptr;
        }
        return new GpuTime(instance);
    }();
    return result;
}

GpuTime::GpuTime(const Instance& instance) : device{instance.GetDevice()} {
    constexpr u32 Slots = Streams * SlotsPerStream;
    pool = Check<"create GPU time query pool">(device.createQueryPoolUnique({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Slots * 2,
    }));
    const auto props = instance.GetPhysicalDevice().getProperties();
    period_ns = props.limits.timestampPeriod;
    const auto families = instance.GetPhysicalDevice().getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    if (family < families.size() && families[family].timestampValidBits != 0 && families[family].timestampValidBits < 64) {
        valid_mask = (1ull << families[family].timestampValidBits) - 1;
    }
    command_pool = Check<"create GPU time command pool">(device.createCommandPoolUnique({
        .queueFamilyIndex = family,
    }));
    begin_buffers = Check<"allocate GPU time begin buffers">(device.allocateCommandBuffers({
        .commandPool = *command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = Slots,
    }));
    end_buffers = Check<"allocate GPU time end buffers">(device.allocateCommandBuffers({
        .commandPool = *command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = Slots,
    }));
    for (u32 i = 0; i < Slots; ++i) {
        const vk::CommandBufferBeginInfo begin{};
        Check(begin_buffers[i].begin(begin));
        begin_buffers[i].resetQueryPool(*pool, i * 2, 2);
        begin_buffers[i].writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool, i * 2);
        Check(begin_buffers[i].end());
        Check(end_buffers[i].begin(begin));
        end_buffers[i].writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool, i * 2 + 1);
        Check(end_buffers[i].end());
    }
    std::printf("GPU time: on (timestamp period %.2f ns, %u slots)\n", period_ns, Slots);
}

GpuTime::Bracket GpuTime::Open(u32 stream) {
    stream = stream < Streams ? stream : Streams - 1;
    std::lock_guard lock{mutex};
    if (pending[stream].size() + 1 >= SlotsPerStream) {
        ++dropped[stream]; // the GPU is that far behind: this submission is not measured
        return {};
    }
    const u32 slot = stream * SlotsPerStream + (next_slot[stream]++ % SlotsPerStream);
    pending[stream].push_back(slot);
    return {begin_buffers[slot], end_buffers[slot]};
}

void GpuTime::Poll() {
    std::lock_guard lock{mutex};
    for (u32 stream = 0; stream < Streams; ++stream) {
        auto& queue = pending[stream];
        while (!queue.empty()) {
            const u32 slot = queue.front();
            std::array<u64, 4> data{}; // {begin, available, end, available}
            const auto result = device.getQueryPoolResults(*pool, slot * 2, 2, sizeof(data), data.data(), 2 * sizeof(u64),
                                                           vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
            if (result != vk::Result::eSuccess || data[1] == 0 || data[3] == 0) {
                break; // not finished yet; later ones cannot be earlier on the same queue
            }
            queue.pop_front();
            const double begin_ms = double(data[0] & valid_mask) * period_ns / 1e6;
            const double end_ms = double(data[2] & valid_mask) * period_ns / 1e6;
            auto& w = window[stream];
            const double span = end_ms - begin_ms;
            if (span >= 0 && span < 1000) {
                w.busy_ms += span;
                w.max_span_ms = std::max(w.max_span_ms, span);
                ++w.submissions;
                ++w.total_submissions;
            }
            if (have_last[stream]) {
                const double gap = begin_ms - last_end_ms[stream];
                if (gap >= 0 && gap < 100) {
                    w.gap_ms += gap;
                } else if (gap >= 100) {
                    w.long_gap_ms += gap; // loading, menus, a pause: not counted as starvation
                    ++w.long_gaps;
                }
            }
            last_end_ms[stream] = end_ms;
            have_last[stream] = true;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    if (seconds < 1.0) {
        return;
    }
    nlohmann::json out = {{"window_s", seconds}, {"timestamp_period_ns", period_ns}};
    for (u32 stream = 0; stream < Streams; ++stream) {
        auto& w = window[stream];
        out["stream" + std::to_string(stream)] = {
            {"submissions_per_s", w.submissions / seconds},
            {"busy_ms_per_s", w.busy_ms / seconds},
            {"gap_ms_per_s", w.gap_ms / seconds},
            {"busy_pct", 100.0 * w.busy_ms / (seconds * 1000.0)},
            {"mean_span_ms", w.submissions != 0 ? w.busy_ms / w.submissions : 0.0},
            {"max_span_ms", w.max_span_ms},
            {"long_gaps_ge_100ms", w.long_gaps},
            {"dropped_not_measured", dropped[stream]},
            {"total_submissions", w.total_submissions},
        };
        const u64 total = w.total_submissions;
        w = Window{};
        w.total_submissions = total;
    }
    window_start = now;
    Core::DebugServer::Publish("gpu_time", out);
}

} // namespace Vulkan
