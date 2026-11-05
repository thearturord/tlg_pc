// SPDX-License-Identifier: GPL-2.0-or-later

// The executor's raster side (G4e; the code of Liverpool::liverpool.cpp, unchanged except for the names): queueing draws and dispatches for the raster thread, running
// them there through the rasterizer host (native/native_raster_host.h), the fences and the waits of the command stream, the command processor's own writes and copies.

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "native/native_content.h"
#include "native/native_wait_registers.h"
#include "native/native_flight.h"
#include "native/native_stream_probe.h"
#include "native/native_timeline.h"
#include "native/native_prof.h"

#include "native/native_registry.h"
#include "native/native_resource_audit.h"
#include "native/native_executor.h"
#include "native/native_executor_tasks.h"
#include "native/native_debug_server.h"
#include "common/arch.h"
#ifdef ARCH_X86_64
#include <immintrin.h>
#endif
#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/memory.h"
#include "native/native_census.h"
#include "native/native_direct.h"
#include "native/native_frame_gate.h"
#include "native/native_draw.h"
#include "native/native_uploads.h"
#include "native/native_drain_audit.h"
#include "native/native_readbacks.h"
#include "native/native_raycast_cpu.h"
#include "native/native_entry_dump.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_resources.h"
#include "core/platform.h"
#include "native/native_frame_events.h"
#include "native/native_display.h"
#include "native/native_readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/readback_trace.h"
#include "native/native_debug_server.h"
#include "video_core/thread_usage.h"
#include "video_core/renderdoc.h"


namespace Core {

std::array<u8, 48_KB> NativeExecutor::ConstantEngine::constants_heap;

using Core::NativeExecutorSync::NoFenceWrite;
using Core::NativeExecutorSync::WriteFenceMemory;

/// The range has no GPU mapping (so it can be written off the GPU thread).
static std::function<bool(VAddr, u32)> NotMapped(Core::RasterHost* rasterizer) {
    return [rasterizer](VAddr address, u32 num_bytes) { return rasterizer && !rasterizer->IsMapped(address, num_bytes); };
}

static bool TryWriteOffThread(Core::RasterHost* rasterizer, void* address, const void* data, u32 num_bytes) {
    return Core::NativeExecutorSync::TryWriteOffThread(address, data, num_bytes, NotMapped(rasterizer));
}

static bool TryWriteFenceOffThread(Core::RasterHost* rasterizer, void* address, u64 data,
                                   u32 num_bytes) {
    return TryWriteOffThread(rasterizer, address, &data, num_bytes);
}

/// Runs a fence signal that may execute outside the GPU thread (async readback fences). Fence
/// writes there go through the memory backing so they can never fault into the page tracker,
/// which may be waiting on that thread. If the backing can't be written, the whole signal is
/// redone on the GPU thread instead.
template <typename Write, typename Signal>
static void RunFenceSignal(NativeExecutor& executor, std::thread::id gpu_id, Write&& try_writes,
                           Signal&& full_signal) {
    if (std::this_thread::get_id() == gpu_id) {
        full_signal();
        return;
    }
    if (try_writes()) {
        return;
    }
    executor.SendCommand(std::forward<Signal>(full_signal));
}


void NativeExecutor::BindRasterizer(Core::RasterHost* rasterizer_) {
    if (!rasterizer_ && split_active) {
        // The rasterizer goes away: finish what is queued and stop the raster thread first.
        DrainRaster();
        split_active = false;
        StopRasterSplit();
        Core::DrainAudit::SetBackend(nullptr, nullptr, {});
        split.reset();
    }
    rasterizer = rasterizer_;
    if (rasterizer && EmulatorSettings.GetCpSplitMode() != 0 && !split) {
        StartRasterSplit(raster_consumer);
    }
}

u32 NativeExecutor::LookupDrain(u32 caller) {
    if (!Core::DrainAudit::LazyEnabled()) {
        DrainRasterWrites(caller);
        return 0;
    }
    return static_cast<u32>(Core::DrainAudit::LookupBegin(caller));
}

Core::DrawPlanPtr NativeExecutor::ReplanAfterDrain(Core::DrawPlanPtr plan, u32 site, const std::function<Core::DrawPlanPtr()>& make) {
    const u64 before = plan ? rasterizer->DrawPlanDigest(*plan) : 0;
    DrainRasterWrites(site);
    auto again = make();
    const u64 after = again ? rasterizer->DrawPlanDigest(*again) : 0;
    Core::DrainAudit::NotePlanVerify(site, before == after);
    rasterizer->RecycleDrawPlan(std::move(plan));
    return again;
}

static Core::NativeDispatch::DispatchInputs MakeDispatchInputs(const AmdGpu::ComputeProgram& p) {
    Core::NativeDispatch::DispatchInputs in;
    static_assert(sizeof(AmdGpu::ComputeProgram) == sizeof(in.words), "the dispatch record words are the register block of the compute program");
    static_assert(sizeof(p.user_data) == sizeof(in.user_data));
    std::memcpy(in.words.data(), &p, sizeof(p));
    in.dispatch_initiator = p.dispatch_initiator;
    in.dim_x = p.dim_x;
    in.dim_y = p.dim_y;
    in.dim_z = p.dim_z;
    in.num_thread_x = {p.num_thread_x.full, p.num_thread_x.partial};
    in.num_thread_y = {p.num_thread_y.full, p.num_thread_y.partial};
    in.num_thread_z = {p.num_thread_z.full, p.num_thread_z.partial};
    in.address = p.address;
    in.settings.num_vgprs = p.settings.num_vgprs;
    in.settings.num_sgprs = p.settings.num_sgprs;
    in.settings.fp_round_mode32 = static_cast<u32>(p.settings.fp_round_mode32);
    in.settings.fp_round_mode64 = static_cast<u32>(p.settings.fp_round_mode64);
    in.settings.fp_denorm_mode32 = static_cast<u32>(p.settings.fp_denorm_mode32);
    in.settings.fp_denorm_mode64 = static_cast<u32>(p.settings.fp_denorm_mode64);
    in.settings.dx10_clamp = p.settings.dx10_clamp;
    in.settings.ieee_mode = p.settings.ieee_mode;
    in.settings.scratch_en = p.settings.scratch_en;
    in.settings.num_user_regs = p.settings.num_user_regs;
    in.settings.tgid_enable = p.settings.tgid_enable;
    in.settings.lds_dwords = p.settings.lds_dwords;
    std::memcpy(in.user_data.data(), p.user_data.data(), sizeof(in.user_data));
    // shadow: read back through the accessors the consumers use and compare with the register block
    const bool equal = in.Address<u64>() == p.Address<u64>() && in.SharedMemSize() == p.SharedMemSize() && in.NumWorkgroups() == p.NumWorkgroups() &&
                       in.IsTgidEnabled(0) == p.IsTgidEnabled(0) && in.IsTgidEnabled(1) == p.IsTgidEnabled(1) && in.IsTgidEnabled(2) == p.IsTgidEnabled(2) &&
                       in.num_thread_x.full == p.num_thread_x.full && in.num_thread_y.full == p.num_thread_y.full && in.num_thread_z.full == p.num_thread_z.full &&
                       std::memcmp(in.user_data.data(), p.user_data.data(), sizeof(in.user_data)) == 0 && std::memcmp(in.words.data(), &p, sizeof(p)) == 0;
    Core::NativeDispatch::NoteChecked(equal);
    return in;
}

Core::DispatchPlanPtr NativeExecutor::ReplanAfterDrain(Core::DispatchPlanPtr plan, u32 site, const std::function<Core::DispatchPlanPtr()>& make) {
    const u64 before = plan ? rasterizer->DispatchPlanDigest(*plan) : 0;
    DrainRasterWrites(site);
    auto again = make();
    const u64 after = again ? rasterizer->DispatchPlanDigest(*again) : 0;
    Core::DrainAudit::NotePlanVerify(site, before == after);
    rasterizer->RecycleDispatchPlan(std::move(plan));
    return again;
}

void NativeExecutor::RasterDispatch(const RasterCall& call_in) {
    Core::FrameTimeline::StageScope front_op{Core::FrameTimeline::StageFrontOp, call_in.kind >= RasterCall::Kind::DispatchDirect ? 2u : 1u, 0};
    RasterCall call = call_in;
    if (native_tag_pending.valid) {
        // The tag belongs to the draw or dispatch the game called. The game's own tracking may issue internal draws
        // (full-screen quads) in between: a call that does not fit the tag (kind, count) leaves it pending.
        const u32 tag_kind = (native_tag_pending.meta >> 8) & 0xff;
        // a draw always has its plan here (F3.5: a draw without one is refused before this call); a dispatch matches a dispatch tag
        const bool match = native_exec_plan != nullptr || tag_kind == 3;
        if (match) {
            call.tag = native_tag_pending;
            native_tag_pending.valid = false;
            native_tag_skips = 0;
        } else if (++native_tag_skips > 6) {
            static u32 dropped_logs = 0;
            if (dropped_logs++ < 24) {
                LOG_INFO(Render, "native tag dropped unmatched: seq {} kind {} record count {} call kind {} indexed {} caller {:#x}", native_tag_pending.seq, tag_kind,
                         Core::NativeDraw::RecordDrawCount(native_tag_pending.seq), static_cast<u32>(call.kind), call.indexed, native_tag_pending.k1);
            }
            native_tag_pending.valid = false;
            native_tag_skips = 0;
        }
    }
    if (Core::NativeUploads::Enabled()) {
        // N5c: the uploads the engine announced before this record are queued in front of it (one operation per record that finds some)
        const u64 upload_seq = call.tag.valid ? call.tag.seq : (native_exec_plan ? rasterizer->DrawPlanSeq(*native_exec_plan) : 0);
        std::vector<Core::NativeUploads::Range> due;
        if (upload_seq != 0 && Core::NativeUploads::TakeDue(upload_seq, due)) {
            RasterRun([this, ranges = std::move(due)]() { rasterizer->PreSynchronize(ranges); });
        }
    }
    if (!split_active) {
        Core::RareEvents::Note("f3_5_raster_call_without_split", "a native draw or dispatch with the command processor split off: not run");
        return;
    }
    auto& sp = *split;
    // The lookup below reads descriptor tables and shader code from guest memory: closures that
    // write guest memory (constant engine dumps, WriteData, DMA) must have run first. (A draw planned from its record has had this at its own planning.)
    u32 verify = 0;
    {
        TLG_PROF_LEVEL("front.lookup_drain", 2);
        if (!Core::DrainAudit::LazyEnabled()) {
            DrainRasterWrites(0);
        } else if (native_exec_plan == nullptr) {
            verify = LookupDrain(0);
        }
    }
    ++sp.stat_calls;
    if (call.kind == RasterCall::Kind::Draw || call.kind == RasterCall::Kind::DrawIndirect) {
        ++sp.stat_draws;
    }
    RasterOp op;
    Core::DrawPlanPtr draw_plan; // handed to the op below
    Core::DispatchPlanPtr dispatch_plan;
    op.has_call = true;
    op.call = call;
    if (call.kind == RasterCall::Kind::DispatchDirect ||
        call.kind == RasterCall::Kind::DispatchIndirect) {
        op.cs = MakeDispatchInputs(call.cs_override ? *call.cs_override : GetCsRegs());
        // The pipeline lookup runs here, on the command processor thread (stage 2 of the split).
        TLG_PROF_NAMED_LEVEL(prof_plan_dispatch, "front.plan_dispatch", 2);
        dispatch_plan = rasterizer->MakeDispatchPlan(op.cs);
        if (verify == 2) {
            dispatch_plan = ReplanAfterDrain(std::move(dispatch_plan), 0, [&] { return rasterizer->MakeDispatchPlan(op.cs); });
        }
        if (dispatch_plan && call.tag.valid) {
            rasterizer->TagDispatchPlan(*dispatch_plan, call.tag.seq);
        }
    } else {
        draw_plan = std::move(native_exec_plan);
        if (draw_plan) {
            rasterizer->SetDrawPlanTargets(*draw_plan, last_cb_extent, last_db_extent, last_target_resources);
        }
    }
    if (rasterizer->ConstRingLevel() != 0) {
        // The queue position this operation will have, and the position the raster thread has finished (documents/tlg/pc-port-draw-pipeline-architecture.md, T1).
        rasterizer->FrontQueueOp(draw_plan.get(), dispatch_plan.get(), sp.pushed.load(std::memory_order_acquire) + 1,
                                 sp.done.load(std::memory_order_acquire), call.tag.valid ? call.tag.seq : (draw_plan ? rasterizer->DrawPlanSeq(*draw_plan) : 0),
                                 call.indexed, call.index_offset);
    }
    {
        TLG_PROF_LEVEL("front.resolve_ahead", 2);
        rasterizer->ResolveAhead(draw_plan.get(), dispatch_plan.get()); // R2
    }
    op.draw_plan = std::move(draw_plan);
    op.dispatch_plan = std::move(dispatch_plan);
    RasterPush(std::move(op), false);
}

void NativeExecutor::SignalGraphicsEop(const Core::NativeSync::Fence& event_eop) {
    if (!event_eop.valid) {
        return;
    }
    // The packet is copied: the signal may run after the command buffer is reused.
    Core::FrameTimeline::Mark(Core::FrameTimeline::EopParsed, event_eop.address);
    static constexpr auto irq = [] {
        Core::FrameGate::NoteEndOfPipe(); // frames in flight: the frame is finished
        Core::NativeFrameEvents::EndOfPipe();
    };
    const bool early = event_eop.interrupt && Core::FrameGate::EarlyInterrupt(); // M5b: the interrupt is raised by the raster thread at the frame end, not after the GPU work
    const u64 ordinal = early ? Core::FrameGate::NextEarlyOrdinal() : 0;
    auto signal = [this, eop = event_eop, early, ordinal] {
        const auto late_irq = [early, ordinal] {
            if (early) {
                Core::FrameGate::EarlyGpuDone(ordinal); // the GPU is done: the interrupt is raised now if it was not yet
            } else {
                irq();
            }
        };
        RunFenceSignal(
            *this, gpu_id,
            [&] {
                bool backed = true;
                eop.Signal(
                    [&](void* address, u64 data, u32 num_bytes) {
                        backed &= TryWriteFenceOffThread(rasterizer, address, data, num_bytes);
                    },
                    [] {});
                if (backed) {
                    // Writes landed, now raise the interrupt (if any) after them.
                    eop.Signal(NoFenceWrite, late_irq);
                }
                return backed;
            },
            [eop, late_irq] { eop.Signal(WriteFenceMemory, late_irq); });
    };
    if (event_eop.interrupt) {
        // frames in flight, game wait mode 2: the raster thread has executed everything the frame recorded (in order, before the end-of-pipe signal below)
        if (early) {
            RasterRun([this, ordinal] {
                Core::FrameGate::NoteRasterReached();
                Core::FrameGate::EarlyFrameRecorded(ordinal); // M5b: the frame's data is ours; the interrupt comes now if the occlusion answers of earlier frames are written, else later; the label below is written when the GPU is done
                // diagnostic TLG_GPU_SERIAL=1: the game still runs ahead, but the raster thread lets the GPU finish this frame before it records the next (no GPU side overlap of
                // frames: tells GPU side hazards from stale CPU data when a frames in flight bug is hunted)
                static const bool gpu_serial = [] { const char* e = std::getenv("TLG_GPU_SERIAL"); return e != nullptr && e[0] == '1'; }();
                if (gpu_serial) {
                    rasterizer->Finish();
                }
            });
        } else {
            RasterRun([] { Core::FrameGate::NoteRasterReached(); });
        }
    }
    SignalFenceAfterReadbacks(std::move(signal), false, static_cast<VAddr>(event_eop.address), event_eop.value, event_eop.Bytes(), GfxQueueId, VideoCore::FenceGfxEop);
}

void NativeExecutor::SignalComputeRelease(const Core::NativeSync::Fence& release_mem, u32 vqid, u32 pipe_id) {
    if (!release_mem.valid) {
        return;
    }
    // The packet is copied: the signal may run after the command buffer is reused.
    auto signal = [this, release = release_mem, pipe_id] {
        const auto irq = [pipe_id] { Core::NativeFrameEvents::ComputeRelease(pipe_id); };
        const auto gds_to_mem = [this](VAddr dst, u16 gds_index, u16 num_dwords) {
            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
        };
        RunFenceSignal(
            *this, gpu_id,
            [&] {
                bool backed = true;
                release.Signal(
                    [&](void* address, u64 data, u32 num_bytes) { backed &= TryWriteFenceOffThread(rasterizer, address, data, num_bytes); },
                    [] {}, [](VAddr, u16, u16) {});
                if (backed) {
                    release.Signal(NoFenceWrite, irq, [](VAddr, u16, u16) {});
                }
                return backed;
            },
            [release, irq, gds_to_mem] { release.Signal(WriteFenceMemory, irq, gds_to_mem); });
    };
    // Copying from GDS records GPU work, which must happen on the raster thread.
    SignalFenceAfterReadbacks(std::move(signal), release_mem.source == Core::NativeSync::Source::GdsCopy, static_cast<VAddr>(release_mem.address), release_mem.value,
                              release_mem.Bytes(), vqid + 1, VideoCore::FenceComputeRelease);
}

/// The async queue's wait on a memory label (the walk's WaitRegMem case for a tag).
NativeExecutor::Task NativeExecutor::DirectWaitAsc(const Core::NativeSync::LabelWait& wait_packet, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    const Core::NativeSync::LabelWait* wait_reg_mem = &wait_packet;
    ASSERT(!wait_reg_mem->pfp);
    if (!wait_reg_mem->Test()) {
        using VideoCore::ReadbackStats::WaitClass;
        const u64 wait_start = VideoCore::ReadbackStats::NowNs();
        bool reported = false;
        bool blocked = false;
        bool held_back = false;
        WaitClass wait_class{};
        while (true) {
            const FenceMatch match = MatchRecordedFence(*wait_reg_mem);
            if (match.satisfied && FenceShortcutEnabled(true)) {
                OnWaitPassedOnFence(*wait_reg_mem, match, vqid + 1);
                wait_class = blocked ? WaitClass::Producer : WaitClass::FencePass;
                break;
            }
            held_back |= match.satisfied;
            if (wait_reg_mem->Test()) {
                wait_class = held_back ? WaitClass::FenceBlocked : WaitClass::Memory;
                break;
            }
            blocked = true;
            ReportLongWait(*wait_reg_mem, "compute", wait_start, reported);
            VideoCore::ReadbackStats::CpWaitYield(false);
            YIELD_ASC(vqid);
        }
        if (blocked) {
            Core::FrameTimeline::Mark(Core::FrameTimeline::CpWait, (VideoCore::ReadbackStats::NowNs() - wait_start) | (u64(0) << 56));
        }
        if (VideoCore::ReadbackStats::IsEnabled()) {
            VideoCore::ReadbackStats::OnCommandWait(true, wait_reg_mem->Address<VAddr>(), wait_class,
                                                    blocked ? VideoCore::ReadbackStats::NowNs() - wait_start : 0);
        }
    }
    FIBER_EXIT;
}

void NativeExecutor::ExecuteDma(const Core::NativeSync::DmaCopy& dma) {
    if (!rasterizer || !Core::NativeExecutorSync::PrepareDma(dma)) {
        return;
    }
    struct Host final : Core::NativeExecutorSync::DmaHost {
        Core::RasterHost& rasterizer;
        explicit Host(Core::RasterHost& r) : rasterizer{r} {}
        void FillBuffer(u64 destination, u32 bytes, u32 fill, bool destination_is_gds) override {
            rasterizer.FillBuffer(destination, bytes, fill, destination_is_gds);
        }
        void CopyBuffer(u64 destination, u64 source, u32 bytes, bool destination_is_gds, bool source_is_gds) override {
            rasterizer.CopyBuffer(destination, source, bytes, destination_is_gds, source_is_gds);
        }
    };
    RasterRun(
        [this, dma] {
            Host host{*rasterizer};
            Core::NativeExecutorSync::RunDma(dma, host);
        },
        false, true);
}

void NativeExecutor::CpWriteData(void* address, const u32* data, u32 data_size, u32 queue) {
    const auto note = [this](void* address, u32 data_size) {
        if (rasterizer) {
            rasterizer->NoteCpGuestWrite(reinterpret_cast<VAddr>(address), data_size);
        }
    };
    if (!split_active) {
        note(address, data_size);
        if (!DeferWriteBehindFences(address, data, data_size, queue)) {
            std::memcpy(address, data, data_size);
        }
        return;
    }
    // A label write of a compute queue behind a fence that landed at the parse (SignalFenceAfterReadbacks) lands now too, not after the raster backlog.
    if (queue != GfxQueueId && queue - 1 < NumTotalQueues && (data_size == sizeof(u32) || data_size == sizeof(u64)) && !label_timeline.HeldBack(reinterpret_cast<VAddr>(address), data_size)) {
        const bool covered = asc_noncopy_issued[queue - 1].load(std::memory_order_acquire) <= asc_noncopy_covered[queue - 1].load(std::memory_order_acquire);
        if (Core::NativeDevice::EarlyFenceAllowed(covered, true) && TryWriteOffThread(rasterizer, address, data, data_size)) {
            if (Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
                Core::FrameTimeline::Record(Core::FrameTimeline::LabelWrite, data[0] | (u64(queue) << 32) | (u64(2) << 48), Core::FrameTimeline::Now(), reinterpret_cast<u64>(address));
            }
            return;
        }
    }
    Core::DrainAudit::SetNextWrite(reinterpret_cast<u64>(address), data_size);
    RasterRun([note, address, data_size] { note(address, data_size); }, false, true);
    // Whether a fence write to the label is held back is decided here, in stream order.
    if (!DeferWriteBehindFences(address, data, data_size, queue)) {
        Core::DrainAudit::SetNextWrite(reinterpret_cast<u64>(address), data_size);
        RasterRun(
            [address, words = std::vector<u32>(data, data + data_size / sizeof(u32)), queue] {
                Core::NativeExecutorSync::ApplyCpWrite(address, words, queue);
            },
            false, true);
    }
}

namespace {
/// Diagnostics (study): the compute dispatches the raster thread executes, by the game call that made them and the shader. Raster thread only.
struct DispatchStat {
    u64 count{}, ns{}, groups{};
    u32 dim_x{}, dim_y{}, dim_z{};
    u64 caller{}, program{};
    bool indirect{}, asc{};
};
std::unordered_map<u64, DispatchStat> dispatch_stats;

DispatchStat& DispatchStatFor(u64 caller, u64 program, bool indirect, bool asc) {
    auto& s = dispatch_stats[(caller * 0x9E3779B97F4A7C15ull) ^ (program * 0xC2B2AE3D27D4EB4Full) ^ (indirect ? 1u : 0u) ^ (asc ? 2u : 0u)];
    s.caller = caller;
    s.program = program;
    s.indirect = indirect;
    s.asc = asc;
    return s;
}
} // namespace

void NativeExecutor::ExecRasterCall(const RasterCall& c, const Vulkan::DrawPlan* draw_plan,
                               const Vulkan::DispatchPlan* dispatch_plan) {
    const auto run = [&](auto&& body) {
        if (c.no_marker) {
            body();
            return;
        }
        const void* cmd_address = c.cmd_address;
        u32 vqid = c.vqid;
        if (c.asc) {
            rasterizer->ScopeMarker(c.marker, fmt::make_format_args(vqid, cmd_address), body);
        } else {
            rasterizer->ScopeMarker(c.marker, fmt::make_format_args(cmd_address), body);
        }
    };
    static const Core::NativeDispatch::DispatchInputs no_program{}; // a dispatch always has a plan
    const Core::NativeDispatch::DispatchInputs& cs_program = dispatch_plan ? rasterizer->DispatchPlanInputs(*dispatch_plan) : no_program;
    {
        TLG_PROF_NAMED_LEVEL(prof_ring, "raster.ring_use", 3);
        rasterizer->BeginRingUse(draw_plan, dispatch_plan);
    }
    switch (c.kind) {
    case RasterCall::Kind::Draw:
        rasterizer->SetNativeTag(c.tag);
        run([&] { rasterizer->Draw(c.indexed, c.index_offset, draw_plan); });
        break;
    case RasterCall::Kind::DrawIndirect:
        rasterizer->NativeInvalidate();
        run([&] {
            rasterizer->DrawIndirect(c.indexed, c.args_addr, c.offset, c.stride, c.max_count,
                                     c.count_addr, c.vertex_sgpr_offset, c.instance_sgpr_offset,
                                     draw_plan);
        });
        break;
    case RasterCall::Kind::DispatchDirect:
        rasterizer->NativeInvalidate();
        rasterizer->SetDispatchCaller(c.tag.valid ? Core::NativeDraw::RecordCaller(c.tag.seq) : 0);
        if (c.fill_dwords != 0) {
            rasterizer->SetPendingFill(c.fill_destination, c.fill_bytes, c.fill_pattern.data(), c.fill_dwords);
        }
        if (c.tag.valid && c.tag.seq != 0 && (c.tag.meta & VideoCore::NativeTagExecFlag) == 0) {
            static u32 compute_debug = 0;
            if (compute_debug++ < 6) {
                const auto& program = cs_program;
                LOG_INFO(Render, "native compute verify: marker {} initiator {:#x} dims {} {} {} address {:#x} split {}",
                         c.marker, program.dispatch_initiator, program.dim_x, program.dim_y, program.dim_z,
                         static_cast<u64>(program.address), split_active);
            }
            Core::NativeDraw::VerifyComputeRecord(c.tag.seq, cs_program.words.data());
        }
        {
            const u64 start = VideoCore::ReadbackStats::NowNs();
            rasterizer->SetExplicitImageReadback(c.tag.valid && c.tag.seq != 0 && Core::NativeDraw::RecordReadsBack(c.tag.seq));
            run([&] { rasterizer->DispatchDirect(dispatch_plan); });
            rasterizer->SetExplicitImageReadback(false);
            auto& s = DispatchStatFor(c.tag.valid ? Core::NativeDraw::RecordCaller(c.tag.seq) : 0, cs_program.address, false, c.asc);
            ++s.count;
            s.ns += VideoCore::ReadbackStats::NowNs() - start;
            s.groups += u64(cs_program.dim_x) * cs_program.dim_y * cs_program.dim_z;
            s.dim_x = cs_program.dim_x;
            s.dim_y = cs_program.dim_y;
            s.dim_z = cs_program.dim_z;
        }
        break;
    case RasterCall::Kind::DispatchIndirect:
        rasterizer->NativeInvalidate();
        if (c.tag.valid && c.tag.seq != 0 && (c.tag.meta & VideoCore::NativeTagExecFlag) == 0) {
            Core::NativeDraw::VerifyComputeRecord(c.tag.seq, cs_program.words.data());
        }
        {
            const u64 start = VideoCore::ReadbackStats::NowNs();
            run([&] {
                rasterizer->DispatchIndirect(c.args_addr, c.offset, c.dispatch_size, dispatch_plan);
            });
            auto& s = DispatchStatFor(c.tag.valid ? Core::NativeDraw::RecordCaller(c.tag.seq) : 0, cs_program.address, true, c.asc);
            ++s.count;
            s.ns += VideoCore::ReadbackStats::NowNs() - start;
        }
        break;
    }
    {
        TLG_PROF_NAMED_LEVEL(prof_ring, "raster.ring_use", 3);
        rasterizer->EndRingUse();
    }
    rasterizer->MaybeFlushRecorded();
}

void NativeExecutor::RasterConsumer::ThreadStarted() {
    owner.gpu_id = std::this_thread::get_id();
}

bool NativeExecutor::RasterConsumer::CallbacksPending() const {
    return owner.num_commands != 0;
}

void NativeExecutor::RasterConsumer::RunCallbacks() {
    owner.ProcessCommands();
}

void NativeExecutor::RasterConsumer::SelectStream(bool second) {
    if (owner.rasterizer) {
        owner.rasterizer->SelectStream(second);
    }
}

bool NativeExecutor::RasterConsumer::LaneUsable() const {
    return owner.rasterizer != nullptr && owner.rasterizer->SideStreamReady();
}

void NativeExecutor::RasterConsumer::RunCall(RasterOp& op) {
    owner.ExecRasterCall(op.call, op.draw_plan.get(), op.dispatch_plan.get());
}

void NativeExecutor::RasterConsumer::RunClosure(RasterOp& op) {
    owner.rasterizer->NativeInvalidate();
    {
        TLG_PROF_NAMED_LEVEL(prof_closure, "raster.op.closure", 1);
        TLG_PROF_TAG("closure_site", op.write_line, op.write_file);
        op.closure();
    }
}

bool NativeExecutor::RasterConsumer::IsDepthOnly(const Vulkan::DrawPlan& plan) const {
    return owner.rasterizer->DrawPlanIsDepthOnly(plan);
}

void NativeExecutor::RasterConsumer::Release(RasterOp& op) {
    owner.rasterizer->RecycleDrawPlan(std::move(op.draw_plan));
    owner.rasterizer->RecycleDispatchPlan(std::move(op.dispatch_plan));
}

void NativeExecutor::RasterConsumer::BatchFinished() {
    std::scoped_lock lk{owner.submit_mutex};
    owner.submit_cv.notify_all();
}

void NativeExecutor::RasterConsumer::PublishIntervalStats(RasterSplit& sp, u64 now) {
    auto* rasterizer = owner.rasterizer;
    auto& fence_latency = owner.fence_latency;
    {
            const u64 block_start = now;
            const auto host_stats = rasterizer->PublishIntervalStats();
            if (sp.stat_last_log_ns != 0) {
                Core::DebugServer::Publish(
                    "cp_split",
                    {{"operations", sp.stat_ops},
                     {"draws_and_dispatches", sp.stat_calls},
                     {"write_drains", sp.stat_write_drains},
                     {"draws", sp.stat_draws},
                     {"draws_no_register_change", sp.stat_draws_no_reg_change},
                     {"plan_draws_all_stages_same", host_stats.plan_draws_all_stages_same},
                     {"plan_draws", host_stats.plan_draws},
                     {"render_state_memo_hits", host_stats.memo_hits},
                     {"render_state_memo_misses", host_stats.memo_misses},
                     {"note", "counts per ~5 s interval"}});
                LOG_INFO(Render,
                         "CP split: {} operations, {} draws/dispatches, {} waits for guest memory "
                         "writes before a pipeline lookup, front blocked on a full queue {} times, "
                         "raster thread idle {} ms, "
                         "draws {} of which {} had no register change since the previous op, "
                         "stage inputs same as the previous draw: draws {}/{}, stages {}/{} "
                         "(since the last line)",
                         sp.stat_ops, sp.stat_calls, sp.stat_write_drains,
                         sp.stat_front_blocked.exchange(0),
                         sp.stat_raster_idle_ns.exchange(0) / 1'000'000, sp.stat_draws,
                         sp.stat_draws_no_reg_change, host_stats.plan_draws_all_stages_same,
                         host_stats.plan_draws, host_stats.plan_stages_same, host_stats.plan_stages);
            }
            sp.stat_front_blocked = 0;
            sp.stat_raster_idle_ns = 0;
            sp.stat_ops = sp.stat_calls = sp.stat_write_drains = 0;
            sp.stat_draws = sp.stat_draws_no_reg_change = 0;
            LOG_INFO(Render,
                     "Raster thread wall time (ms per interval): draws {}, dispatches {}, memory "
                     "write closures {}, other closures (fences, submit end, flips) {}, "
                     "SendCommand callbacks {}; draws {} of which depth-only {} ms, {} draws of which "
                     "depth-only {}",
                     sp.ns_draws / 1'000'000, sp.ns_dispatches / 1'000'000,
                     sp.ns_write_closures / 1'000'000, sp.ns_other_closures / 1'000'000,
                     sp.ns_callbacks / 1'000'000, sp.ns_draws / 1'000'000,
                     sp.ns_depth_only_draws / 1'000'000, sp.n_draws, sp.n_depth_only_draws);
            {
                static constexpr const char* names[] = {"front_working", "front_waiting_for_game_submit",
                                                        "front_draining_writes", "queue_full"};
                nlohmann::json idle = nlohmann::json::object();
                for (size_t i = 0; i < 4; ++i) {
                    idle[names[i]] = sp.raster_idle_ns_by_front_state[i].exchange(0) / 1'000'000;
                }
                Core::DebugServer::Publish("raster_idle_ms_by_what_the_front_was_doing",
                                           std::move(idle));
                static constexpr const char* op_names[] = {"draw", "draw_indirect", "dispatch",
                                                           "dispatch_indirect", "closure"};
                nlohmann::json by_op = nlohmann::json::object();
                for (size_t i = 0; i < 5; ++i) {
                    by_op[op_names[i]] = {
                        {"idle_ms", sp.raster_idle_ns_by_next_op[i].exchange(0) / 1'000'000},
                        {"idle_episodes", sp.raster_idle_count_by_next_op[i].exchange(0)}};
                }
                Core::DebugServer::Publish("raster_idle_by_kind_of_op_that_ended_it",
                                           std::move(by_op));
            }
            Core::DebugServer::Publish("fence_latency", owner.fence_latency.Report());
            {
                nlohmann::json sites = nlohmann::json::array();
                std::vector<std::pair<u64, std::pair<const char*, u32>>> top;
                {
                    std::scoped_lock lk{sp.lane_block_mutex};
                    for (const auto& [site, count] : sp.lane_block_sites) {
                        top.push_back({count, site});
                    }
                }
                std::ranges::sort(top, std::greater<>{});
                for (size_t i = 0; i < top.size() && i < 12; ++i) {
                    sites.push_back({{"file", top[i].second.first ? top[i].second.first : "-"}, {"line", top[i].second.second}, {"times", top[i].first}});
                }
                Core::DebugServer::Publish("raster_lane_blockers", sites);
            }
            Core::DebugServer::Publish("raster_lane", {{"enabled", sp.lane_on},
                                                      {"usable", LaneUsable()},
                                                      {"queue_limit", sp.MaxQueued},
                                                      {"operations", sp.lane_ops.load()},
                                                      {"hoisted", sp.lane_hoisted.load()},
                                                      {"skipped_operations_total", sp.lane_distance.load()},
                                                      {"picked_up_while_a_batch_ran", sp.lane_pickups.load()},
                                                      {"bursts", sp.lane_bursts.load()},
                                                      {"boundaries_waiting_behind_a_graphics_closure", sp.lane_barrier_waits.load()}});
            {
                nlohmann::json sites = nlohmann::json::object();
                for (auto& site : sp.write_sites) {
                    const u32 line = site.line.load(std::memory_order_relaxed);
                    if (line == 0) {
                        continue;
                    }
                    sites["liverpool.cpp:" + std::to_string(line)] = {
                        {"writes_queued", site.pushes.exchange(0)},
                        {"front_wait_ms", site.wait_ns.exchange(0) / 1'000'000}};
                }
                Core::DebugServer::Publish("write_drains_by_call_site", std::move(sites));
            }
            {
                std::vector<const DispatchStat*> list;
                for (const auto& [key, stat] : dispatch_stats) {
                    list.push_back(&stat);
                }
                std::ranges::sort(list, [](const DispatchStat* a, const DispatchStat* b) { return a->ns > b->ns; });
                nlohmann::json out = nlohmann::json::array();
                u64 total_ns = 0, total_count = 0;
                for (const auto* s : list) {
                    total_ns += s->ns;
                    total_count += s->count;
                }
                for (size_t i = 0; i < list.size() && i < 200; ++i) {
                    const auto& s = *list[i];
                    out.push_back({{"ghidra_caller", s.caller ? s.caller - Core::NativeDraw::ImageBase() + 0x1000000 : 0},
                                   {"program", fmt::format("{:#x}", s.program)},
                                   {"dims_last", {s.dim_x, s.dim_y, s.dim_z}},
                                   {"count", s.count},
                                   {"groups", s.groups},
                                   {"ms", s.ns / 1'000'000},
                                   {"avg_us", s.count ? s.ns / 1000 / s.count : 0},
                                   {"indirect", s.indirect},
                                   {"async", s.asc}});
                }
                Core::DebugServer::Publish("dispatch_callers", {{"dispatches", total_count}, {"total_ms", total_ns / 1'000'000}, {"by_caller_and_shader", std::move(out)},
                                                                {"note", "per ~5 s interval, raster thread, wall time of the dispatch call including the marker"}});
                dispatch_stats.clear();
            }
            {
                const auto& native = host_stats;
                Core::DebugServer::Publish(
                    "native_draw_raster",
                    {{"tagged_draws", native.native_tagged},
                     {"untagged_draws", native.native_untagged},
                     {"dynamic_state_skipped", native.native_dynamic_skipped},
                     {"bind_pipeline_skipped", native.native_bind_skipped},
                     {"note", "counts per ~5 s interval"}});
                if (native.native_tagged != 0) {
                    LOG_INFO(Render,
                             "Native draw path: {} tagged draws, {} untagged, {} dynamic state "
                             "updates skipped, {} pipeline binds skipped",
                             native.native_tagged, native.native_untagged, native.native_dynamic_skipped,
                             native.native_bind_skipped);
                }
            }
            Core::DebugServer::Publish(
                "raster_thread_ms",
                {{"draws", sp.ns_draws / 1'000'000},
                 {"depth_only_draws_ms", sp.ns_depth_only_draws / 1'000'000},
                 {"draw_count", sp.n_draws},
                 {"depth_only_draw_count", sp.n_depth_only_draws},
                 {"dispatches", sp.ns_dispatches / 1'000'000},
                 {"memory_write_closures", sp.ns_write_closures / 1'000'000},
                 {"other_closures", sp.ns_other_closures / 1'000'000},
                 {"send_command_callbacks", sp.ns_callbacks / 1'000'000},
                 {"note", "wall time per ~5 s interval"}});
            sp.ns_draws = sp.ns_dispatches = sp.ns_write_closures = sp.ns_other_closures = 0;
            sp.ns_depth_only_draws = sp.n_draws = sp.n_depth_only_draws = 0;
            sp.ns_callbacks = 0;
            // The thread CPU report and the core isolation re-apply run on the housekeeping thread:
            // each took about 70 ms here, which showed as a dip to 50 fps every 5 seconds.
            sp.stat_last_log_ns = now;
            Core::DebugServer::RecordPeriodic(
                "raster_thread_5s_stats_block",
                static_cast<double>(VideoCore::ReadbackStats::NowNs() - block_start) / 1e6);
    }
}


NativeExecutor::NativeExecutor()
    : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      fence_wait_shortcut{[] {
          // dark frame experiment (`TLG_FENCE_SHORTCUT=0..3`): the level of the fence wait shortcut for this run, over the settings (1: the compute queues' waits poll memory instead of
          // passing on a fence the other queue has only recorded; 0: no queue passes early)
          if (const char* e = std::getenv("TLG_FENCE_SHORTCUT"); e != nullptr && *e >= '0' && *e <= '3') {
              return static_cast<u32>(*e - '0');
          }
          return static_cast<u32>(EmulatorSettings.GetReadbackFenceWaitShortcut());
      }()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    StartProcess();
}

NativeExecutor::~NativeExecutor() {
    StopProcess();
}

NativeExecutor::FenceMatch NativeExecutor::MatchRecordedFence(const Core::NativeSync::LabelWait& wait_reg_mem) {
    return label_timeline.Match(wait_reg_mem);
}

void NativeExecutor::OnWaitPassedOnFence(const Core::NativeSync::LabelWait& wait_reg_mem, const FenceMatch& match,
                                    u32 queue) {
    if (fence_wait_shortcut >= 3 && rasterizer) {
        // Recorded in order with the draws queued before and after this wait.
        // The fence's tick is set by an earlier operation of the raster queue, so it is known
        // when this one runs.
        RasterRun([this, address = match.label_address, seq = match.seq] {
            if (rasterizer->CommandStreamBarrier(label_timeline.Tick(address, seq))) {
                VideoCore::ReadbackStats::OnStreamBarrier();
            }
        });
    }
    if (queue != GfxQueueId) {
        label_timeline.NoteComputePass(wait_reg_mem, match, queue);
    }
}

void NativeExecutor::ReportLongWait(const Core::NativeSync::LabelWait& wait_reg_mem, const char* queue,
                               u64 wait_start, bool& reported) {
    Core::NativeExecutorSync::ReportLongWait(label_timeline, wait_reg_mem, queue, wait_start, reported);
}

void NativeExecutor::ReportLongSpin(const char* what, const void* address, u64 wait_start,
                               bool& reported) {
    Core::NativeExecutorSync::ReportLongSpin(what, address, wait_start, reported);
}

void NativeExecutor::NoteFenceLatency(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns) {
    fence_latency.Note(kind, queue_ns, gpu_ns);
}

void NativeExecutor::SignalFenceAfterReadbacks(Common::UniqueFunction<void>&& signal, bool must_sync,
                                          VAddr address, u64 value, u32 num_bytes, u32 queue,
                                          VideoCore::FenceKind kind) {
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::Fence};
    if (!rasterizer) {
        signal();
        return;
    }
    const u64 parsed_ns = VideoCore::ReadbackStats::NowNs();
    // Wraps the signal to time the fence: parsed -> started on the raster thread -> signal runs.
    const auto timed = [this, kind, parsed_ns, queue](Common::UniqueFunction<void>&& inner) {
        const u64 started_ns = VideoCore::ReadbackStats::NowNs();
        return Common::UniqueFunction<void>{
            [this, kind, parsed_ns, started_ns, queue, inner = std::move(inner)]() mutable {
                const u64 now = VideoCore::ReadbackStats::NowNs();
                NoteFenceLatency(kind, started_ns - parsed_ns, now - started_ns);
                if (kind == VideoCore::FenceGfxEop) {
                    Core::FrameTimeline::Record(Core::FrameTimeline::EopSignaled, started_ns - parsed_ns, Core::FrameTimeline::Now(), now - started_ns);
                }
                inner();
                Core::Flight::Add(Core::Flight::FenceSignaled, 0, kind, u64(queue)); // L1
                if (Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
                    // arg = ns from the parse to the start on the raster thread | queue << 48 | kind << 56, arg2 = ns from there to the signal; ts = after the write
                    Core::FrameTimeline::Record(Core::FrameTimeline::FenceSignaled, (started_ns - parsed_ns) | (u64(queue & 0xff) << 48) | (u64(kind) << 56),
                                                Core::FrameTimeline::Now(), now - started_ns);
                }
            }};
    };
    if (address == 0 || num_bytes == 0) {
        RasterRun(
            [this, signal = std::move(signal), kind, must_sync, timed]() mutable {
                rasterizer->OnFence(timed(std::move(signal)), kind, must_sync);
            },
            must_sync);
        return;
    }
    // The label timeline is updated here, when the packet is parsed, so command stream waits
    // see the fence without waiting for the raster thread. The raster thread records the work
    // before the fence and notes its tick (SetFenceTick).
    const u64 seq = label_timeline.Record(address, value, num_bytes, queue);
    // Compute fences that only wait for copies (frame-order design V2, free-hold-fix-design.md section 10): on the console the compute queue runs beside the graphics queue, so a fence of
    // GPU to GPU copies lands in microseconds; here it queued behind the raster backlog (4 ms) and the game's PrepareMaterial spun on it. Landed at the parse instead when no compute
    // work with a result the CPU may read is still uncovered and the free hold keeps every block the game frees after the fence (the copy is queued before it, in raster order).
    if (queue != GfxQueueId && queue - 1 < NumTotalQueues && !must_sync && (kind == VideoCore::FenceComputeRelease || kind == VideoCore::FenceHeldWrite)) {
        const u32 vq = queue - 1;
        const u64 issued = asc_noncopy_issued[vq].load(std::memory_order_acquire);
        const bool covered = issued <= asc_noncopy_covered[vq].load(std::memory_order_acquire);
        if (Core::NativeDevice::EarlyFenceAllowed(covered)) {
            timed(std::move(signal))();
            label_timeline.Retire(address, seq);
            return;
        }
        // kept: the signal runs after the raster thread, and covers the compute work issued so far
        RasterRun(
            [this, address, seq, signal = std::move(signal), kind, must_sync, timed, vq, issued]() mutable {
                label_timeline.SetTick(address, seq, rasterizer ? rasterizer->CurrentTick() : 0);
                rasterizer->OnFence(
                    timed([this, address, seq, vq, issued, signal = std::move(signal)]() mutable {
                        signal();
                        label_timeline.Retire(address, seq);
                        u64 seen = asc_noncopy_covered[vq].load(std::memory_order_relaxed);
                        while (seen < issued && !asc_noncopy_covered[vq].compare_exchange_weak(seen, issued, std::memory_order_release)) {
                        }
                    }),
                    kind, must_sync);
            },
            must_sync);
        return;
    }
    // CEILING TEST ONLY (TLG_CEILING_ASC_FENCE=1): the label of compute queue 1 is written as soon as the raster thread reaches it, without waiting for the
    // GPU. The results the game reads back are wrong (measuring how short the frame gets, documents/tlg/light-scene-cap-plan.md section 11).
    static const bool ceiling = [] {
        const char* value = std::getenv("TLG_CEILING_ASC_FENCE");
        return value != nullptr && value[0] == '1';
    }();
    if (ceiling && queue == 1 && (kind == VideoCore::FenceComputeRelease || kind == VideoCore::FenceHeldWrite)) {
        RasterRun(
            [this, address, seq, signal = std::move(signal), timed]() mutable {
                label_timeline.SetTick(address, seq, rasterizer ? rasterizer->CurrentTick() : 0);
                timed(std::move(signal))();
                label_timeline.Retire(address, seq);
            },
            must_sync);
        return;
    }
    RasterRun(
        [this, address, seq, signal = std::move(signal), kind, must_sync, timed]() mutable {
            label_timeline.SetTick(address, seq, rasterizer ? rasterizer->CurrentTick() : 0);
            rasterizer->OnFence(
                timed([this, address, seq, signal = std::move(signal)]() mutable {
                    signal();
                    label_timeline.Retire(address, seq);
                }),
                kind, must_sync);
        },
        must_sync);
}

bool NativeExecutor::DeferWriteBehindFences(void* address, const u32* data, u32 num_bytes, u32 queue) {
    const VAddr begin = reinterpret_cast<VAddr>(address);
    if (!label_timeline.HeldBack(begin, num_bytes)) {
        return false;
    }
    label_timeline.NoteDeferredWrite(begin, num_bytes, queue);
    std::vector<u32> words(data, data + num_bytes / sizeof(u32));
    u64 value{};
    std::memcpy(&value, words.data(), std::min<u32>(num_bytes, sizeof(u64)));
    auto signal = [this, address, queue, words = std::move(words)] {
        const u32 size = static_cast<u32>(words.size() * sizeof(u32));
        if (Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
            Core::FrameTimeline::Record(Core::FrameTimeline::LabelWrite, words[0] | (u64(queue) << 32) | (u64(1) << 48), Core::FrameTimeline::Now(), reinterpret_cast<u64>(address));
        }
        RunFenceSignal(
            *this, gpu_id,
            [&] { return TryWriteOffThread(rasterizer, address, words.data(), size); },
            [address, words] {
                Core::NativeExecutorSync::WriteBackingOrMemory(address, words.data(), static_cast<u32>(words.size() * sizeof(u32)));
            });
    };
    const bool is_label = num_bytes == sizeof(u32) || num_bytes == sizeof(u64);
    SignalFenceAfterReadbacks(std::move(signal), false, is_label ? begin : 0, value,
                              is_label ? num_bytes : 0, queue, VideoCore::FenceHeldWrite);
    return true;
}

bool NativeExecutor::ShouldYieldToOtherQueue(u64& last_ns) {
    static const bool enabled = [] {
        const char* value = std::getenv("TLG_CP_ASC_YIELD");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled || num_submits.load(std::memory_order_relaxed) <= mapped_queues[GfxQueueId].submits.size()) {
        return false;
    }
    const u64 now = Core::FrameTimeline::Now();
    if (now - last_ns < 50'000) {
        return false;
    }
    last_ns = now;
    return true;
}

/// A wait of the graphics queue on a memory label (the walk's WaitRegMem case for a tag): tests the label, passes on the recorded
/// fence timeline, and yields to the other queues until the label has the value.
NativeExecutor::Task NativeExecutor::DirectWaitDisplay(u32 buffer) {
    FIBER_ENTER(dcb_task_name);
    if (!Core::NativeDisplay::IsFree(buffer)) {
        if (num_submits == mapped_queues[GfxQueueId].submits.size()) {
            // nothing else to do: sleep until the present thread frees the buffer
            VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::FlipSleep};
            Core::FrameTimeline::Span flip_span{Core::FrameTimeline::FlipWait};
            Core::NativeDisplay::WaitFree(buffer);
        } else {
            while (!Core::NativeDisplay::IsFree(buffer)) {
                VideoCore::ReadbackStats::CpWaitYield(true);
                YIELD_GFX();
            }
        }
    }
    Core::NativeDisplay::NoteWaitPassed(buffer);
    FIBER_EXIT;
}

NativeExecutor::Task NativeExecutor::DirectWait(const Core::NativeSync::LabelWait& wait_packet) {
    FIBER_ENTER(dcb_task_name);
    const Core::NativeSync::LabelWait* wait_reg_mem = &wait_packet;
    const u64* wait_addr = wait_reg_mem->Address<u64*>();
    if (vo_labels.is_label(wait_addr) && num_submits == mapped_queues[GfxQueueId].submits.size()) {
        VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::FlipSleep};
        Core::FrameTimeline::Span flip_span{Core::FrameTimeline::FlipWait};
        vo_labels.wait([&] { return wait_reg_mem->Test(); });
    } else if (!wait_reg_mem->Test()) {
        using VideoCore::ReadbackStats::WaitClass;
        const u64 wait_start = VideoCore::ReadbackStats::NowNs();
        bool reported = false;
        bool blocked = false;
        bool held_back = false;
        WaitClass wait_class{};
        while (true) {
            const FenceMatch match = MatchRecordedFence(*wait_reg_mem);
            if (match.satisfied && FenceShortcutEnabled(false)) {
                OnWaitPassedOnFence(*wait_reg_mem, match, GfxQueueId);
                wait_class = blocked ? WaitClass::Producer : WaitClass::FencePass;
                break;
            }
            held_back |= match.satisfied;
            if (wait_reg_mem->Test()) {
                wait_class = held_back ? WaitClass::FenceBlocked : WaitClass::Memory;
                break;
            }
            blocked = true;
            ReportLongWait(*wait_reg_mem, "gfx", wait_start, reported);
            VideoCore::ReadbackStats::CpWaitYield(vo_labels.is_label(wait_addr));
            YIELD_GFX();
        }
        if (blocked) {
            Core::FrameTimeline::Mark(Core::FrameTimeline::CpWait, (VideoCore::ReadbackStats::NowNs() - wait_start) | (u64(1) << 56));
        }
        if (VideoCore::ReadbackStats::IsEnabled()) {
            VideoCore::ReadbackStats::OnCommandWait(false, wait_reg_mem->Address<VAddr>(), wait_class,
                                                    blocked ? VideoCore::ReadbackStats::NowNs() - wait_start : 0);
        }
    }
    FIBER_EXIT;
}

void NativeExecutor::FakeOcclusionDump(VAddr address) {
    // TLG_REAL_OCCLUSION=0: a made up answer, every query reads "visible" (the counters of each pair differ by a fixed step)
    static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
    static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
    u64* results = reinterpret_cast<u64*>(address);
    const u64 counter_value = pixel_counter | OcclusionCounterValidMask;
    const s32 pairs = num_counter_pairs;
    RasterRun([this, results, counter_value, pairs] {
        if (rasterizer) {
            rasterizer->NoteCpGuestWrite(reinterpret_cast<VAddr>(results), pairs * 16);
        }
        u64* out = results;
        for (s32 i = 0; i < pairs; ++i, out += 2) {
            *out = counter_value;
        }
    });
    pixel_counter += OcclusionCounterStep;
}

// C8 (2026-10-28): the packet interpreter is removed. Every command buffer of the game runs on the native direct path (ProcessDirect); one that
// cannot (no hooks, a packet the direct builder does not know) is skipped with an error instead of being interpreted from its packets.
NativeExecutor::Task NativeExecutor::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, std::shared_ptr<Core::NativeDirect::Buffer> direct) {
    FIBER_ENTER(dcb_task_name);
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1) < 20) {
        LOG_ERROR(Render, "C8: a graphics command buffer the direct path cannot run is skipped ({} dwords, built {}, pure {})", dcb.size(), direct != nullptr,
                  direct != nullptr && direct->pure);
    }
    (void)ccb;
    FIBER_EXIT;
    co_return;
}

NativeExecutor::Task NativeExecutor::ProcessCompute(std::span<const u32> acb, u32 vqid, std::shared_ptr<Core::NativeDirect::Buffer> direct) {
    FIBER_ENTER(acb_task_name[vqid]);
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1) < 20) {
        LOG_ERROR(Render, "C8: a compute command buffer the direct path cannot run is skipped (queue {}, {} dwords, built {})", vqid, acb.size(), direct != nullptr);
    }
    FIBER_EXIT;
    co_return;
}

void NativeExecutor::SubmitEnd() {
    if (rasterizer) {
        RasterRun([this] {
            VideoCore::ReadbackStats::CpPhase phase{"end of submit (readbacks, flush)"};
            VideoCore::ReadbackStats::CpTimer timer{
                VideoCore::ReadbackStats::CpTime::SubmitEnd};
            rasterizer->OnSubmit();
            rasterizer->Flush();
        });
    }
}


} // namespace Core
