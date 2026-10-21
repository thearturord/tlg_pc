// SPDX-License-Identifier: GPL-2.0-or-later
// See vk_gpu_timeline.h.

#include <atomic>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/frame_timeline.h"
#include "video_core/renderer_vulkan/vk_gpu_timeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long*);
#endif

namespace Vulkan {

GpuTimeline::GpuTimeline(const Instance& instance_) : instance{instance_} {
    const vk::QueryPoolCreateInfo info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumQueries,
    };
    pool = Check<"create the frame timeline query pool">(instance.GetDevice().createQueryPoolUnique(info));
    period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    calibrated_supported = instance.IsCalibratedTimestampsSupported();
    LOG_INFO(Render_Vulkan, "frame timeline: GPU timestamps ready (period {} ns, calibrated clocks {})", period_ns, calibrated_supported);
}

GpuTimeline::~GpuTimeline() = default;

u32 GpuTimeline::Allocate() {
    const u32 index = next.fetch_add(1, std::memory_order_relaxed) % NumQueries;
    instance.GetDevice().resetQueryPool(*pool, index, 1);
    return index;
}

void GpuTimeline::Submitted(Pending&& entry) {
    std::scoped_lock lock{mutex};
    pending.push_back(std::move(entry));
}

void GpuTimeline::Calibrate() {
    const u64 now = Core::FrameTimeline::Now();
    if (calibration.valid && now - last_calibration_ns < 500'000'000ull) {
        return;
    }
    last_calibration_ns = now;
#ifdef _WIN32
    if (!calibrated_supported) {
        return;
    }
    static const long long frequency = [] {
        long long f = 0;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    const std::array<vk::CalibratedTimestampInfoEXT, 2> infos = {
        vk::CalibratedTimestampInfoEXT{.timeDomain = vk::TimeDomainEXT::eDevice},
        vk::CalibratedTimestampInfoEXT{.timeDomain = vk::TimeDomainEXT::eQueryPerformanceCounter},
    };
    std::array<u64, 2> stamps{};
    u64 deviation = 0;
    const auto result = instance.GetDevice().getCalibratedTimestampsEXT(static_cast<u32>(infos.size()), infos.data(), stamps.data(), &deviation);
    if (result != vk::Result::eSuccess) {
        return;
    }
    calibration.gpu_ticks = stamps[0];
    // QPC counts to nanoseconds the way the standard library's steady clock does.
    calibration.cpu_ns = static_cast<u64>(static_cast<double>(stamps[1]) * 1e9 / static_cast<double>(frequency));
    calibration.valid = true;
    Core::FrameTimeline::Record(Core::FrameTimeline::GpuClockSync, calibration.gpu_ticks, calibration.cpu_ns,
                                static_cast<u64>(period_ns * 1000.0));
#endif
}

u64 GpuTimeline::ToCpuNs(u64 gpu_ticks) const {
    const double delta_ticks = static_cast<double>(static_cast<s64>(gpu_ticks - calibration.gpu_ticks));
    return static_cast<u64>(static_cast<double>(calibration.cpu_ns) + delta_ticks * period_ns);
}

void GpuTimeline::Collect(Semaphore& work_semaphore) {
    std::vector<Pending> ready;
    {
        std::scoped_lock lock{mutex};
        if (pending.empty()) {
            return;
        }
        work_semaphore.Refresh();
        while (!pending.empty() && work_semaphore.IsFree(pending.front().tick)) {
            ready.push_back(std::move(pending.front()));
            pending.pop_front();
        }
    }
    if (ready.empty()) {
        return;
    }
    Calibrate();
    const auto device = instance.GetDevice();
    const auto read = [&](u32 index, u64& out) {
        const auto result = device.getQueryPoolResults(*pool, index, 1, sizeof(u64), &out, sizeof(u64), vk::QueryResultFlagBits::e64);
        return result == vk::Result::eSuccess;
    };
    for (const auto& entry : ready) {
        u64 begin = 0, end = 0;
        if (!read(entry.begin_index, begin) || !read(entry.end_index, end) || !calibration.valid) {
            continue;
        }
        Core::FrameTimeline::Record(Core::FrameTimeline::GpuCommandBuffer, static_cast<u64>(static_cast<double>(end - begin) * period_ns), ToCpuNs(end), entry.tick);
        u64 previous = begin;
        for (const auto& stamp : entry.stamps) {
            u64 value = 0;
            if (!read(stamp.index, value)) {
                continue;
            }
            // The first stamp of a command buffer can be older than the command buffer's own start stamp (the driver writes that one when the queue gets to it,
            // not when the work before it is done): it then starts the chain, with no gap before it. Any later stamp older than the one before it (a recycled
            // query) is reported as kind 0xF0 with no duration, so the report can count them.
            if (&stamp == &entry.stamps.front() && value < previous) {
                previous = value;
            }
            const bool ordered = value >= previous;
            const u64 delta = ordered ? static_cast<u64>(static_cast<double>(value - previous) * period_ns) : 0;
            Core::FrameTimeline::Record(Core::FrameTimeline::GpuPass, delta | (static_cast<u64>(ordered ? stamp.kind : 0xF0) << 56), ToCpuNs(value), stamp.detail);
            if (ordered) {
                previous = value;
            }
        }
    }
}

} // namespace Vulkan
