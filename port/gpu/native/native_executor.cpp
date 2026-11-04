// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_executor.h"

#include <algorithm>
#include <cstring>
#include <functional>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "native/native_content.h"
#include "native/native_device.h"
#include "native/native_direct.h"
#include "native/native_draw.h"
#include "native/native_drain_audit.h"
#include "native/native_flight.h"
#include "native/native_frame_events.h"
#include "native/native_prof.h"
#include "native/native_resource_audit.h"
#include "native/native_stream_probe.h"
#include "native/native_timeline.h"
#include "native/native_updates.h"
#include "video_core/readback_stats.h"
#include "video_core/readback_trace.h"
#include "video_core/renderdoc.h"
#include "video_core/thread_usage.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
static u32 GetCurrentThreadIdForProfile() {
    return static_cast<u32>(GetCurrentThreadId());
}
#endif

namespace Core {

void NativeExecutor::StartRasterSplit(NativeRasterQueue::Consumer& consumer) {
    split = std::make_unique<RasterSplit>();
    split_active = true;
    split->Start(consumer, EmulatorSettings.GetCpSplitMode() >= 2);
}

void NativeExecutor::StartProcess() {
    process_thread = std::jthread{std::bind_front(&NativeExecutor::Process, this)};
}

void NativeExecutor::StopProcess() {
    process_thread.request_stop();
    process_thread.join();
    StopRasterSplit();
}

void NativeExecutor::StopRasterSplit() {
    if (!split) {
        return;
    }
    split->Stop();
}

bool NativeExecutor::RasterIdle() const noexcept {
    return !split_active || split->Idle();
}

void NativeExecutor::WakeRaster() {
    if (split_active) {
        split->Wake();
    }
}


void NativeExecutor::DrainRaster() {
    if (!split_active) {
        return;
    }
    split->Drain();
}

void NativeExecutor::RasterPush(RasterOp&& op, bool wait) {
    split->Push(std::move(op), wait);
}

void NativeExecutor::RasterRun(Common::UniqueFunction<void>&& func, bool wait, bool writes_memory,
                          std::source_location site) {
    if (!split_active) {
        func();
        return;
    }
    split->PushClosure(std::move(func), wait, writes_memory, site);
}

void NativeExecutor::DrainRasterWrites(u32 caller) {
    split->DrainWrites(caller);
}


void NativeExecutor::ProcessCommands() {
    if (split_active && std::this_thread::get_id() != gpu_id) {
        return; // The raster thread runs the callbacks.
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        VideoCore::ReadbackStats::CpPhase phase{"guest command (SendCommand)"};
        VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::GuestCommand};
        callback();
    }
}

void NativeExecutor::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    if (const u32 priority = EmulatorSettings.GetCpThreadPriority(); priority >= 2) {
        Common::SetCurrentThreadPriority(static_cast<Common::ThreadPriority>(priority));
    }
    gpu_id = std::this_thread::get_id();
    front_id = gpu_id;
    if (const s32 core = EmulatorSettings.GetCpFrontCore(); core >= 0) {
        VideoCore::PinThreadAndIsolateCore(static_cast<u32>(core));
    }
    VideoCore::ReadbackStats::MarkCpThread();
    VideoCore::ReadbackStats::RegisterSampledThread("cp");
#ifdef __linux__
    gpu_tid = gettid();
#elif defined(_WIN32)
    gpu_tid = GetCurrentThreadIdForProfile();
#endif

    while (!stoken.stop_requested()) {
        VideoCore::ReadbackTrace::NoteMark("front_wait");
        {
            VideoCore::ReadbackStats::CpTimer idle_timer{VideoCore::ReadbackStats::CpTime::Idle};
            std::unique_lock lk{submit_mutex};
            if (split_active) {
                split->front_state.store(1, std::memory_order_relaxed);
            }
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return (!split_active && num_commands) || num_submits || submit_done;
            });
            if (split_active) {
                split->front_state.store(0, std::memory_order_relaxed);
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();
        VideoCore::ReadbackTrace::NoteMark("submit_begin");

        curr_qid = -1;

        while (num_submits || (!split_active && num_commands)) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            const u64 resume_start = Core::FrameTimeline::capture_active.load(std::memory_order_relaxed) ? Core::FrameTimeline::Now() : 0;
            Core::ResourceAudit::front_class = static_cast<u8>(curr_qid);
            task.resume();
            Core::ResourceAudit::front_class = Core::ResourceAudit::NoClass;
            if (resume_start != 0) {
                const u64 resume_end = Core::FrameTimeline::Now();
                Core::FrameTimeline::Record(Core::FrameTimeline::CpResume, static_cast<u64>(curr_qid) | (task.done() ? 0x100u : 0u), resume_end, resume_end - resume_start);
            }

            if (task.done()) {
                bool was_partial = false;
                {
                    std::scoped_lock partial_lock{partial_mutex};
                    was_partial = partial_handles.erase(task.address()) != 0;
                }
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                --num_submits;
                if (was_partial) {
                    partial_pending.fetch_sub(1, std::memory_order_acq_rel);
                }
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
                if (!was_partial && stream_open.load(std::memory_order_acquire) != 0 && num_submits <= partial_pending.load(std::memory_order_acquire)) {
                    // everything the game submitted is done; the spans streamed for the frame it is still recording do not count: the game gets its idle signal now
                    RasterRun([] { Core::NativeFrameEvents::GpuIdle(); });
                }
            }
        }

        const bool submit_was_done = submit_done;
        if (submit_done) {
            VideoCore::EndCapture();
            SubmitEnd();
            submit_done = false;
        }

        Core::FrameTimeline::Mark(Core::FrameTimeline::CpIdleQueued, submit_was_done ? 1 : 0);
        if (stream_open.load(std::memory_order_acquire) != 0) {
            continue; // the frame is still being recorded: the real submit is yet to come, the GPU is not idle
        }
        RasterRun([] {
            Core::FrameTimeline::Mark(Core::FrameTimeline::RasterIdleSignal);
            Core::NativeFrameEvents::GpuIdle();
        });
    }
}

NativeExecutor::CmdBuffer NativeExecutor::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void NativeExecutor::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool partial) {
    Core::Flight::Add(Core::Flight::SubmitBegin, 0, dcb.size(), 0); // L1
    FLIGHT_SCOPE("command processor: graphics submission");
    auto announced = Core::NativeUpdates::TakeBatch(); // B2: applied on the raster thread before this submission's work
    Core::NativeContent::NoteSubmit(announced.first);  // M1: the frame data store (shadow) and the submission count
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    const u64 build_start = Core::FrameTimeline::Now();
    auto direct = Core::NativeDirect::Build(0, dcb.data(), static_cast<u32>(dcb.size()), !ccb.empty(), guest_markers_enabled);
    if (direct) {
        // B11: the engine's clears written into this command buffer travel with it
        Core::NativeDevice::TakeClears(direct->begin, direct->begin + u64{direct->dwords} * 4, direct->clears);
    }
    Core::StreamProbe::NoteBuild(Core::FrameTimeline::Now() - build_start, direct && direct->pure, static_cast<u32>(dcb.size()));
    const bool run_direct = direct && Core::NativeDirect::ExecuteEnabled();
    if (!announced.Empty()) {
        if (direct) {
            direct->announced_first = std::move(announced.first);
            direct->announced_again = std::move(announced.again);
            direct->announced_deleted = std::move(announced.deleted);
        } else {
            RasterRun([batch = std::move(announced)] { Core::NativeUpdates::ApplyBatch(batch); });
        }
    }
    auto task = run_direct ? ProcessDirect(direct, ccb) : ProcessGraphics(dcb, ccb, std::move(direct));
    Core::NativeDraw::Live().direct_executed.fetch_add(run_direct ? 1 : 0, std::memory_order_relaxed);
    if (partial) {
        std::scoped_lock lock{partial_mutex};
        partial_handles.insert(task.handle.address());
        partial_pending.fetch_add(1, std::memory_order_acq_rel);
    }
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
    Core::Flight::Add(Core::Flight::SubmitEnd, 0, 0); // L1
}

void NativeExecutor::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    Core::Flight::Add(Core::Flight::SubmitBegin, 0, acb.size(), gnm_vqid); // L1
    FLIGHT_SCOPE("command processor: compute submission");
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto announced = Core::NativeUpdates::TakeBatch();
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    auto direct = Core::NativeDirect::Build(gnm_vqid, acb.data(), static_cast<u32>(acb.size()), false, guest_markers_enabled);
    if (!announced.Empty()) {
        if (direct) {
            direct->announced_first = std::move(announced.first);
            direct->announced_again = std::move(announced.again);
            direct->announced_deleted = std::move(announced.deleted);
        } else {
            RasterRun([batch = std::move(announced)] { Core::NativeUpdates::ApplyBatch(batch); });
        }
    }
    const bool run_direct = direct && Core::NativeDirect::ExecuteEnabled();
    const auto& task = run_direct ? ProcessDirectAsc(direct, vqid) : ProcessCompute(acb, vqid, std::move(direct));
    Core::NativeDraw::Live().direct_executed_asc.fetch_add(run_direct ? 1 : 0, std::memory_order_relaxed);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}


} // namespace Core
