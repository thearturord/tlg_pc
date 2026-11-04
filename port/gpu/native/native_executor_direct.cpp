// SPDX-License-Identifier: GPL-2.0-or-later

// The executor's direct path (G4e; the code of liverpool_engine.cpp): the command buffer of the game run entry by entry (ProcessDirect), the draw records executed from their
// engine objects, the constant engine buffer and the announced updates.

#include <bit>
#include <cstring>

#include "native/native_flight.h"
#include "native/native_log.h"
#include "native/native_raycast_cpu.h"
#include "native/native_config.h"
#include "native/native_stream_probe.h"
#include "native/native_timeline.h"
#include "native/native_prof.h"
#include "native/native_registry.h"
#include "native/native_resource_audit.h"
#include "native/native_executor.h"
#include "native/native_executor_tasks.h"
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
#include "native/native_occlusion.h"
#include "native/native_direct.h"
#include "native/native_draw.h"
#include "native/native_draw_inputs.h"
#include "native/native_uploads.h"
#include "native/native_drain_audit.h"
#include "native/native_readbacks.h"
#include "native/native_entry_dump.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_resources.h"
#include "core/platform.h"
#include "native/native_frame_events.h"
#include "native/native_flip.h"
#include "native/native_shader_env.h"
#include "native/native_target_desc.h"
#include "native/native_display.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "native/native_readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/readback_trace.h"
#include "native/native_debug_server.h"
#include "video_core/thread_usage.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Core {

void NativeExecutor::ApplyStateTemplate(const Core::NativeDraw::StateTemplate& state) {
    // G4a: the executor keeps no register file. The one state that is read back is the compute program of the graphics queue (the dispatches the command processor runs itself).
    using Core::NativeDraw::StateWrite;
    for (const StateWrite& write : state.writes) {
        if (write.kind == StateWrite::Sh && write.reg >= 0x200 && write.reg <= (0x200 + sizeof(AmdGpu::ComputeProgram) / 4)) {
            reinterpret_cast<u32*>(&cs_states[GfxQueueId])[write.reg - 0x200] = write.value;
        }
    }
}

void NativeExecutor::StartAscTag(const u32* tag, const void* address, u32 vqid) {
    native_tag_pending.valid = true;
    native_tag_pending.meta = tag[2];
    std::memcpy(&native_tag_pending.k1, &tag[3], sizeof(u64));
    std::memcpy(&native_tag_pending.k2, &tag[5], sizeof(u64));
    std::memcpy(&native_tag_pending.k3, &tag[7], sizeof(u64));
    std::memcpy(&native_tag_pending.seq, &tag[9], sizeof(u64));
    if ((native_tag_pending.meta & VideoCore::NativeTagExecFlag) != 0) {
        ExecuteNativeDraw(address, vqid);
    }
}

/// A pure window of an async compute queue: the entries copied at submit, executed in order (see ProcessDirect).
NativeExecutor::Task NativeExecutor::ProcessDirectAsc(std::shared_ptr<Core::NativeDirect::Buffer> direct, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];
    QueueAnnounced(*direct);
    Core::EntryDump::Span entry_dump{vqid + 1, reinterpret_cast<const u32*>(direct->begin), direct->dwords};
    using Core::NativeDirect::EntryKind;
    for (const auto& entry : direct->entries) {
        ProcessCommands();
        entry_dump.Packet(direct->PacketWords(entry), entry.address, entry.dwords != 0);
        Core::NativeCensus::NoteEntry(1, static_cast<u32>(entry.kind), entry.opcode);
        switch (entry.kind) {
        case EntryKind::Tag:
            StartAscTag(entry.words.data(), reinterpret_cast<const void*>(entry.address), vqid);
            break;
        case EntryKind::Flip:
        case EntryKind::StateBuffer:
            break;
        case EntryKind::Sync: {
            Core::NativeSync::SyncRecord record;
            const u32 opcode = Core::NativeDraw::ReadSyncTag(entry.words.data(), record);
            if (opcode == 0x49) {
                SignalComputeRelease(record.fence, vqid, queue.pipe_id);
            } else if (opcode == 0x37) {
                const auto& write = record.write;
                CpWriteData(reinterpret_cast<void*>(write.address), write.data, write.bytes, vqid + 1);
            } else if (opcode == 0x3c) {
                auto task = DirectWaitAsc(record.wait, vqid);
                RESUME_ASC(task, vqid);
                while (!task.handle.done()) {
                    YIELD_ASC(vqid);
                    RESUME_ASC(task, vqid);
                }
                task.handle.destroy();
            }
            break;
        }
        }
    }
    *queue.read_addr += direct->dwords;
    *queue.read_addr %= queue.ring_size_dw;
    FIBER_EXIT;
}

void NativeExecutor::StartNativeTag(const u32* tag) {
    native_tag_pending.valid = true;
    native_tag_pending.meta = tag[2];
    std::memcpy(&native_tag_pending.k1, &tag[3], sizeof(u64));
    std::memcpy(&native_tag_pending.k2, &tag[5], sizeof(u64));
    std::memcpy(&native_tag_pending.k3, &tag[7], sizeof(u64));
    std::memcpy(&native_tag_pending.seq, &tag[9], sizeof(u64));
    native_follow_pending = (native_tag_pending.meta & VideoCore::NativeTagFollowFlag) != 0;
    if (((native_tag_pending.meta >> 8) & 0xff) == 12) {
        // hardware default state: the register writes of a constant packet sequence, applied directly
        native_tag_pending.valid = false;
        native_follow_pending = false;
        if (native_tag_pending.k1 != 0) {
            ApplyStateTemplate(*reinterpret_cast<const Core::NativeDraw::StateTemplate*>(native_tag_pending.k1));
        }
    } else if (native_follow_pending) {
        // executed when the draw packet comes
    } else if ((native_tag_pending.meta & VideoCore::NativeTagExecFlag) != 0) {
        ExecuteNativeDraw();
    }
}

void NativeExecutor::ExecuteNativeDraw(const void* header, u32 asc_vqid, u32 index_count) {
    TLG_PROF_LEVEL("front.execute_native_draw", 1);
    native_exec_calls++;
    Core::NativeDraw::ExecDraw draw;
    if (!rasterizer || !Core::NativeDraw::TakeExecDraw(native_tag_pending.seq, draw, true)) {
        native_tag_pending.valid = false;
        Core::NativeReadbacks::ClearPending(native_tag_pending.seq); // M4
        TLG_LOG(CP, Error, "native draw: the record {} is gone, a draw is lost ({})", native_tag_pending.seq, Core::NativeDraw::DescribeMissingRecord(native_tag_pending.seq));
        return;
    }
    if (draw.kind == 10) {
        // memory copy: destination, source and the number of bytes travel in the record
        native_tag_pending.valid = false;
        if (rasterizer) {
            Core::DrainAudit::SetNextWrite(draw.args_addr, draw.a1);
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.args_addr), source = static_cast<VAddr>(draw.extra_addr), bytes = draw.a1, seq = draw.seq,
                 staging = draw.dst_usage == 3 && Core::NativeReadbacks::ExplicitEnabled()] {
                    if (Core::NativeRaycast::ShadowOn()) {
                        Core::NativeRaycast::NoteMemoryCopy(source, destination, bytes);
                    }
                    rasterizer->CopyBuffer(destination, source, bytes, false, false, staging);
                    Core::NativeReadbacks::ClearPending(seq); // M4: the download, if any, is tracked by the landing gate from here
                },
                false, true);
        }
        return;
    }
    if (draw.kind == 9) {
        // append / consume counter read-back: four bytes of GDS to the buffer's memory
        native_tag_pending.valid = false;
        if (rasterizer) {
            Core::DrainAudit::SetNextWrite(draw.args_addr, sizeof(u32));
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.args_addr), source = draw.start_index, seq = draw.seq,
                 staging = draw.dst_usage == 3 && Core::NativeReadbacks::ExplicitEnabled()] {
                    if (Core::NativeRaycast::ShadowOn()) {
                        Core::NativeRaycast::NoteCounterCopy(source, destination);
                    }
                    rasterizer->CopyBuffer(destination, source, sizeof(u32), false, true, staging);
                    Core::NativeReadbacks::ClearPending(seq); // M4
                },
                false, true);
        }
        return;
    }
    if (draw.kind == 6 || draw.kind == 8) {
        // buffer fill (kind 8: GDS, an append / consume counter): destination, value and the number of dwords travel in the record
        native_tag_pending.valid = false;
        if (rasterizer) {
            if (draw.kind == 8) {
                Core::DrainAudit::SetNextWrite(Core::DrainAudit::NoGuestMemory, 0);
            } else {
                Core::DrainAudit::SetNextWrite(draw.args_addr, draw.a1 * 4u);
            }
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.args_addr), value = draw.start_index, bytes = draw.a1 * 4u,
                 gds = draw.kind == 8] {
                    if (!gds && Core::NativeResources::ImageCopyEnabled() && rasterizer->TryClearImage(destination, bytes, &value, 1)) {
                        return;
                    }
                    rasterizer->FillBuffer(destination, bytes, value, gds);
                },
                // A fill that is not the GDS one is a CMASK / HTILE clear of the game's render targets (FillCopy has no other caller): the planning of
                // the next draw never reads those surfaces, so the command processor need not wait for it.
                false, draw.kind == 8 || VideoCore::ReadbackPolicy::Get().skip_fill_drain.load(std::memory_order_relaxed) == 0);
        }
        return;
    }
    if (draw.kind == 7) {
        // depth / stencil clear: the inputs of the clear come from the record (F3.5: no register image)
        native_tag_pending.valid = false;
        auto clear_inputs = Core::NativeDrawInputs::ForDepthClear(draw);
        Core::NativeDraw::Live().state_planned.fetch_add(1, std::memory_order_relaxed);
        RasterRun(
            [this, extent = draw.db_extent, inputs = std::move(clear_inputs)] {
                AmdGpu::CbDbExtent db_extent;
                db_extent.raw = extent;
                rasterizer->ClearDepthStencilTarget(inputs, db_extent);
            },
            false, false);
        return;
    }
    if (draw.kind == 3 && draw.copy_bytes != 0 && asc_vqid == ~0u && Core::NativeResources::ImageCopyEnabled()) {
        // the device's buffer copy shader: a memory copy (two images of the same layout are copied as images)
        native_tag_pending.valid = false;
        if (rasterizer) {
            Core::DrainAudit::SetNextWrite(draw.copy_destination, draw.copy_bytes);
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.copy_destination), source = static_cast<VAddr>(draw.copy_source),
                 bytes = draw.copy_bytes] { rasterizer->CopyBuffer(destination, source, bytes, false, false); },
                false, true);
        }
        return;
    }
    if (draw.kind == 3 || draw.kind == 11) {
        // dispatch (kind 11: indirect, the group counts are read from the argument buffer): the record holds dwords of the compute program (user data, dimensions); applied for this dispatch only
        // The dispatch runs on the program the record describes alone (a zeroed program with the record's dwords; C8: the queue's state is not touched)
        AmdGpu::ComputeProgram alone{};
        auto* alone_dwords = reinterpret_cast<u32*>(&alone);
        for (u32 i = 0; i < draw.count; ++i) {
            const u32 dword = draw.regs[i].index - Core::NativeDraw::ComputeRegisterBase;
            if (dword < sizeof(AmdGpu::ComputeProgram) / sizeof(u32)) {
                alone_dwords[dword] = draw.regs[i].value;
            }
        }
        RasterCall dispatch{};
        dispatch.tag = native_tag_pending;
        dispatch.kind = RasterCall::Kind::DispatchDirect;
        if (draw.fill_dwords != 0 && asc_vqid == ~0u && Core::NativeResources::ImageCopyEnabled()) {
            dispatch.fill_destination = static_cast<VAddr>(draw.fill_destination);
            dispatch.fill_bytes = draw.fill_bytes;
            dispatch.fill_dwords = draw.fill_dwords;
            dispatch.fill_pattern = draw.fill_pattern;
        }
        if (draw.kind == 11) {
            dispatch.kind = RasterCall::Kind::DispatchIndirect;
            dispatch.args_addr = draw.args_addr;
            dispatch.offset = draw.start_index;
            dispatch.dispatch_size = sizeof(AmdGpu::PM4CmdDispatchIndirect::GroupDimensions);
        }
        if (asc_vqid != ~0u) {
            dispatch.asc = true;
            dispatch.vqid = asc_vqid;
            dispatch.cmd_address = header;
            dispatch.marker = "asc[{}]:{}:NativeDispatch";
        } else {
            dispatch.marker = "gfx:{}:NativeDispatch";
        }
        dispatch.cs_override = &alone;
        if ((alone.dispatch_initiator & 1) != 0) {
            RasterDispatch(dispatch);
        }
        return;
    }
    // a fast clear eliminate takes the extent hint of its own target for this operation only
    const auto saved_cb_extent = last_cb_extent;
    const auto saved_target_resources = last_target_resources;
    if (draw.cb_extents_only) {
        native_tag_pending.valid = false;
    }
    if (draw.has_extents) {
        // the extent hints the game's target bind sends in a trailing NOP packet
        for (u32 slot = 0; slot < AmdGpu::NUM_COLOR_BUFFERS && slot < draw.cb_extent.size(); ++slot) {
            if ((draw.cb_extent_mask >> slot) & 1) {
                last_cb_extent[slot].raw = draw.cb_extent[slot];
            }
        }
        if (!draw.cb_extents_only) {
            last_db_extent.raw = draw.db_extent;
        }
        last_target_resources = draw.target_resources;
    }
    bool planned_from_record = false;
    // F3.5: a draw is planned from its record and the engine objects, never from a register image. A draw that cannot be (no typed user data, no native fixed state, another shader
    // stage bound) is a counted refusal: it is not drawn.
    const bool plannable = draw.kind <= 2 && !draw.cb_extents_only && draw.typed && draw.fixed != nullptr && !draw.other_stage && split_active;
    if (draw.typed) {
        for (u32 mask = draw.vs_user_data_mask; mask != 0; mask &= mask - 1) {
            const u32 i = static_cast<u32>(std::countr_zero(mask));
            exec_vertex_user_data[i] = draw.vs_user_data[i];
        }
        for (u32 mask = draw.ps_user_data_mask; mask != 0; mask &= mask - 1) {
            const u32 i = static_cast<u32>(std::countr_zero(mask));
            exec_pixel_user_data[i] = draw.ps_user_data[i];
        }
    }
    native_inputs_key = 0;
    {
        const auto& policy = VideoCore::ReadbackPolicy::Get();
        auto& live = Core::NativeDraw::Live();
        const bool exec_on = true;
        const bool switches = exec_on;
        const auto left = [&live, &draw](u32 reason) {
            Core::Flight::Add(Core::Flight::PlanLeft, draw.seq, reason); // L1
            live.exec_fallback.fetch_add(1, std::memory_order_relaxed);
            live.exec_reason[reason].fetch_add(1, std::memory_order_relaxed);
        };
        const auto take = [&](Core::DrawPlanPtr plan) {
            native_exec_plan = std::move(plan);
            planned_from_record = true;
            live.exec_typed.fetch_add(1, std::memory_order_relaxed);
            if (draw.cb_extents_only) {
                live.exec_fast_clear.fetch_add(1, std::memory_order_relaxed);
            } else if (index_count != ~0u) {
                live.exec_inline.fetch_add(1, std::memory_order_relaxed);
            }
        };
        if (index_count != ~0u && !draw.cb_extents_only) {
            live.exec_inline_seen.fetch_add(1, std::memory_order_relaxed);
        }
        if (draw.cb_extents_only) {
            // a fast clear eliminate: the operation and its render target come from the record
            if (split_active && draw.kind == 4) {
                take(rasterizer->MakeOperationPlan(draw));
            } else if (exec_on) {
                left(5);
            }
        } else if (exec_on && draw.kind <= 2) {
            if (!switches) {
                left(0);
            } else if (!draw.typed || draw.stage_key == 0 || draw.fixed == nullptr) {
                left(!draw.typed ? 1 : draw.stage_key == 0 ? 6 : 7);
            } else if (draw.other_stage) {
                // F3.2: the native pipeline is vertex plus pixel; a draw with another stage bound is refused below
                left(8);
                Core::RareEvents::Note("f3_2_other_shader_stage_bound", fmt::format("draw record {}: a hull, domain or geometry shader is bound; stage key {:#x}", draw.seq, draw.stage_key));
            } else if (!split_active) {
                left(2);
            } else {
                // an inline draw has its count with the draw packet (index_count), anything else carries it in the record
                native_inputs_key = draw.inputs_key;
                u32 verify;
                {
                    TLG_PROF_LEVEL("front.lookup_drain", 2);
                    verify = LookupDrain(1);
                }
                const auto make_plan = [&] {
                    return rasterizer->MakeExecDrawPlan(*draw.fixed, draw.stage_key, native_inputs_key, exec_vertex_user_data,
                                                    exec_pixel_user_data, draw, index_count,
                                                    native_tag_pending.valid ? (native_tag_pending.meta & 0xff) : 15u);
                };
                Core::DrawPlanPtr plan;
                {
                    TLG_PROF_LEVEL("front.plan_exec_draw", 2);
                    plan = make_plan();
                    if (verify == 2) {
                        plan = ReplanAfterDrain(std::move(plan), 1, make_plan);
                    }
                }
                if (plan) {
                    take(std::move(plan));
                } else {
                    left(4);
                }
            }
        }
    }
    if (VideoCore::ReadbackPolicy::Get().cs_audit.load(std::memory_order_relaxed) != 0) {
        Core::NativeDraw::Live().overlay_records.fetch_add(1, std::memory_order_relaxed);
    }
    if (!planned_from_record) {
        Core::NativeDraw::MaterializeRegs(draw); // A5.3: the register groups of the record (the plan's inputs are made from them)
    }
    RasterCall call{};
    if (draw.kind == 2) {
        call.kind = RasterCall::Kind::DrawIndirect;
        call.marker = "gfx:{}:NativeDrawIndexIndirect";
        call.indexed = true;
        call.args_addr = draw.args_addr;
        call.offset = draw.offset;
        call.stride = sizeof(AmdGpu::DrawIndexedIndirectArgs);
        call.max_count = 1;
        call.vertex_sgpr_offset = static_cast<u16>(draw.vertex_sgpr);
        call.instance_sgpr_offset = static_cast<u16>(draw.instance_sgpr);
    } else {
        call.kind = RasterCall::Kind::Draw;
        call.marker = draw.cb_extents_only ? "gfx:{}:NativeFastClearEliminate" : "gfx:{}:NativeDraw";
        call.indexed = draw.kind == 0;
        call.index_offset = draw.start_index;
    }
    if (const u32 delay = VideoCore::ReadbackPolicy::Get().cp_delay_ns.load(std::memory_order_relaxed); delay != 0) {
        const u64 until = Core::FrameTimeline::Now() + delay;
        while (Core::FrameTimeline::Now() < until) {
        }
    }
    native_exec_scope = true;
    native_fixed = draw.fixed;
    native_exec_key = draw.stage_key != 0 && draw.grouped != 0 && !draw.cb_extents_only && !draw.other_stage // F3.2: a draw with another stage bound never takes a cached vertex plus pixel pipeline
                          ? draw.stage_key
                          : 0;
    bool refused = false;
    if (!planned_from_record && (draw.kind <= 2 || draw.cb_extents_only) && !plannable) {
        // F3.5: nothing to plan this draw from; counted above (exec_reason), noted here once per kind of reason
        refused = true;
        Core::NativeDraw::Live().plan_refused.fetch_add(1, std::memory_order_relaxed);
        Core::RareEvents::Note("f3_5_draw_refused", fmt::format("draw record {}: typed {}, native fixed {}, other stage {}, split {}, stage key {:#x}", draw.seq, draw.typed, draw.fixed != nullptr,
                                                               draw.other_stage, split_active, draw.stage_key));
    } else if (!planned_from_record && plannable) {
        Core::NativeDraw::Live().plan_register_free.fetch_add(1, std::memory_order_relaxed);
        Core::DrawIndirectParams params{};
        if (call.kind == RasterCall::Kind::DrawIndirect) {
            params = Core::DrawIndirectParams{.vertex_sgpr_offset = call.vertex_sgpr_offset, .instance_sgpr_offset = call.instance_sgpr_offset, .is_indirect = true};
        }
        u32 verify;
        {
            TLG_PROF_LEVEL("front.lookup_drain", 2);
            verify = LookupDrain(2);
        }
        const auto make_candidate = [&] {
            // E1c: the shader objects of the draw, for the comparison of the translator's input with the one derived from them
            Core::NativeShaderEnv::Scope shader_env{draw.vs_gnm, draw.ps_gnm, draw.fetch_object, draw.other_stage, draw.typed ? exec_vertex_user_data.data() : nullptr,
                                                    draw.typed ? exec_pixel_user_data.data() : nullptr};
            Core::NativeTargetDesc::Scope target_desc{draw.native_targets}; // E7 (TLG_TARGET_DESC_USE): PlanDraw applies the record's target description to the inputs it makes
            auto made = rasterizer->MakeDrawPlan(params, *native_fixed, native_exec_key, native_inputs_key, draw, index_count);
            if (made && native_exec_key != 0 && native_inputs_key != 0 && draw.kind <= 2 && draw.typed && !draw.other_stage && draw.bindings_complete) {
                // F5: a learning draw (first sight of its pipeline): the pipeline and the inputs exist now, so the draw is planned from its record like every later one (the
                // flat user data of the stages the record covers is skipped); when that is not possible the draw keeps the plan it has
                if (auto from_record = rasterizer->MakeExecDrawPlan(*draw.fixed, native_exec_key, native_inputs_key, exec_vertex_user_data, exec_pixel_user_data, draw, index_count,
                                                                native_tag_pending.valid ? (native_tag_pending.meta & 0xff) : 15u)) {
                    rasterizer->RecycleDrawPlan(std::move(made));
                    return from_record;
                }
            }
            return made;
        };
        Core::DrawPlanPtr candidate;
        {
            TLG_PROF_LEVEL("front.plan_draw_registers", 2);
            candidate = make_candidate();
            if (verify == 2) {
                candidate = ReplanAfterDrain(std::move(candidate), 2, make_candidate);
            }
        }
        if (candidate == nullptr) {
            // no pipeline for the draw (a stage that does not compile, a target that cannot be bound): refused, not drawn from a register image
            refused = true;
            Core::NativeDraw::Live().plan_refused.fetch_add(1, std::memory_order_relaxed);
            Core::RareEvents::Note("f3_5_draw_refused_no_plan", fmt::format("draw record {}: no plan (stage key {:#x})", draw.seq, draw.stage_key));
        }
        Core::NativeDraw::Live().state_planned.fetch_add(1, std::memory_order_relaxed);
        native_exec_plan = std::move(candidate);
    }
    if (!refused) {
        RasterDispatch(call);
    }
    if (native_exec_plan) {
        rasterizer->RecycleDrawPlan(std::move(native_exec_plan));
    }
    native_inputs_key = 0;
    native_exec_key = 0;
    native_fixed = nullptr;
    native_exec_scope = false;
    if (draw.cb_extents_only) {
        // the fast clear eliminate registers (color control in eliminate mode) must not reach the next draw
        last_cb_extent = saved_cb_extent;
        last_target_resources = saved_target_resources;
    }
}

void NativeExecutor::QueueAnnounced(Core::NativeDirect::Buffer& direct) {
    if (direct.announced_first.empty() && direct.announced_again.empty() && direct.announced_deleted.empty()) {
        return;
    }
    Core::NativeUpdates::Batch batch;
    batch.first = std::move(direct.announced_first);
    batch.again = std::move(direct.announced_again);
    batch.deleted = std::move(direct.announced_deleted);
    RasterRun([b = std::move(batch)] { Core::NativeUpdates::ApplyBatch(b); });
}

/// A pure graphics command buffer: the entries copied at submit (Core::NativeDirect::Build), executed in order. No packet is decoded;
/// every entry runs the function the walk runs for its packet.
/// The constant engine buffer (CCB) of a submission, run in full before the submission's entries. The game submits one at start (E0 census: 10 WRITE_CONST_RAM, 2 DUMP_CONST_RAM,
/// the counter packets and a NOP, 452 dwords): the constant RAM is a 48 KB heap, a dump copies a part of it to guest memory in order with the draws. The counters that order
/// the constant engine against the draw engine need nothing here (it has finished when the entries start), so they are ignored. An opcode this does not know is a rare event.
void NativeExecutor::ExecuteConstantEngine(std::span<const u32> ccb) {
    while (!ccb.empty()) {
        const auto* header = reinterpret_cast<const AmdGpu::PM4Header*>(ccb.data());
        if (header->type != 3) {
            Core::RareEvents::Note("constant_engine_not_type3", "a constant engine packet that is not type 3");
            return;
        }
        const AmdGpu::PM4ItOpcode opcode = header->type3.opcode;
        const u32 dwords = header->type3.NumWords() + 1; // NumWords counts the body: the header dword is one more
        if (dwords > ccb.size()) {
            Core::RareEvents::Note("constant_engine_bad_length", "a constant engine packet that runs past its buffer");
            return;
        }
        Core::NativeCensus::NoteCePacket(static_cast<u32>(opcode), dwords);
        switch (opcode) {
        case AmdGpu::PM4ItOpcode::Nop:
        case AmdGpu::PM4ItOpcode::IncrementCeCounter:
        case AmdGpu::PM4ItOpcode::WaitOnDeCounterDiff:
            break;
        case AmdGpu::PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const AmdGpu::PM4WriteConstRam*>(header);
            std::memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data, write_const->Size());
            break;
        }
        case AmdGpu::PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const AmdGpu::PM4DumpConstRam*>(header);
            void* destination = dump_const->Address<void*>();
            const u8* source = cblock.constants_heap.data() + dump_const->Offset();
            const u32 bytes = dump_const->Size();
            if (split_active) {
                // a command processor write to guest memory: done in order with the draws
                Core::DrainAudit::SetNextWrite(reinterpret_cast<u64>(destination), bytes);
                RasterRun([destination, copy = std::vector<u8>(source, source + bytes)] { std::memcpy(destination, copy.data(), copy.size()); }, false, true);
            } else {
                std::memcpy(destination, source, bytes);
            }
            break;
        }
        case AmdGpu::PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const AmdGpu::PM4CmdIndirectBuffer*>(header);
            ExecuteConstantEngine({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            break;
        }
        default:
            Core::RareEvents::Note("constant_engine_opcode_" + std::to_string(static_cast<u32>(opcode)), "an opcode the constant engine buffer did not hold in the default scene");
            break;
        }
        ccb = ccb.subspan(dwords);
    }
}

NativeExecutor::Task NativeExecutor::ProcessDirect(std::shared_ptr<Core::NativeDirect::Buffer> direct, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);
    QueueAnnounced(*direct);
    if (!ccb.empty()) {
        Core::NativeCensus::NoteCeSubmission(static_cast<u32>(ccb.size()));
        ExecuteConstantEngine(ccb);
    }
    Core::EntryDump::Span entry_dump{0, reinterpret_cast<const u32*>(direct->begin), direct->dwords};
    using Core::NativeDirect::EntryKind;
    u64 last_yield_ns = Core::FrameTimeline::Now();
    // B11: the engine clears of this command buffer: the entries inside each clear's packets are marked on the raster thread
    size_t clear_at = 0;
    bool in_clear = false;
    u32 clear_entries[4]{};
    const auto leave_clear = [&] {
        if (in_clear) {
            Core::NativeDevice::NoteClearEntered(direct->clears[clear_at], clear_entries);
            RasterRun([this] { rasterizer->SetEngineClear(nullptr); });
            in_clear = false;
        }
    };
    for (const auto& entry : direct->entries) {
        ProcessCommands();
        if (ShouldYieldToOtherQueue(last_yield_ns)) {
            YIELD_GFX();
        }
        while (clear_at < direct->clears.size() && entry.address >= direct->clears[clear_at].end) {
            leave_clear();
            ++clear_at;
        }
        if (clear_at < direct->clears.size() && entry.address >= direct->clears[clear_at].begin && !in_clear) {
            in_clear = true;
            std::fill(std::begin(clear_entries), std::end(clear_entries), 0u);
            RasterRun([this, record = direct->clears[clear_at]] { rasterizer->SetEngineClear(&record); });
        }
        if (in_clear) {
            ++clear_entries[entry.kind == EntryKind::Tag ? 0 : entry.kind == EntryKind::Sync ? 1 : 3];
        }
        entry_dump.Packet(entry.words.data(), entry.address, entry.dwords != 0);
        Core::NativeCensus::NoteEntry(0, static_cast<u32>(entry.kind), entry.opcode);
        switch (entry.kind) {
        case EntryKind::Tag:
            StartNativeTag(entry.words.data());
            break;
        case EntryKind::Flip:
            if (entry.words[1] == Core::NativeDisplay::WaitTagMagic) {
                // E6.2c: wait until the display buffer is free
                auto task = DirectWaitDisplay(entry.words[2]);
                RESUME_GFX(task);
                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                task.handle.destroy();
                break;
            }
            if (entry.words[1] == Core::NativeFlip::TagMagic) {
                // E6.2b: the typed flip, the four things the rewritten tail did, in the same order
                Core::NativeFlip::Request request;
                if (!Core::NativeFlip::Take(entry.words[2] | static_cast<u64>(entry.words[3]) << 32, request)) {
                    Core::RareEvents::Note("flip_request_gone", "the typed flip request of a tail was not in the ring when the command processor took it");
                    RasterRun([] { Core::NativeFrameEvents::FlipReady(); });
                    break;
                }
                const u32 one = 1;
                Core::NativeDisplay::Lock(request.buffer_index);                                             // the display buffer is locked
                CpWriteData(reinterpret_cast<u64*>(request.display_label), &one, sizeof(one), GfxQueueId); // ... and its label says so
                RasterRun([] { Core::NativeFrameEvents::FlipReady(); });                                    // the flip is ready
                if (request.has_label) {
                    CpWriteData(reinterpret_cast<u64*>(request.label_address), &request.label_value, sizeof(u32), GfxQueueId);
                }
                if (request.has_fence) {
                    SignalGraphicsEop(request.fence);
                }
                break;
            }
            Core::RareEvents::Note("flip_entry_without_request", "a flip entry that is neither a typed flip request nor a display wait");
            break;
        case EntryKind::StateBuffer: {
            u64 pointer;
            std::memcpy(&pointer, &entry.seq, sizeof(pointer));
            ApplyStateTemplate(*reinterpret_cast<const Core::NativeDraw::StateTemplate*>(pointer));
            break;
        }
        case EntryKind::Sync: {
            Core::NativeSync::SyncRecord record;
            const auto& restored = record.words;
            const u32 opcode = Core::NativeDraw::ReadSyncTag(entry.words.data(), record);
            if (opcode == 0x47) {
                SignalGraphicsEop(record.fence);
            } else if (opcode == 0x37) {
                const auto& write = record.write;
                CpWriteData(reinterpret_cast<u64*>(write.address), write.data, write.bytes, GfxQueueId);
            } else if (opcode == 0x50) {
                // a CP DMA copy (E5: it travels as a sync tag, not as a raw packet)
                ExecuteDma(record.dma);
            } else if (opcode == 0x46) {
                // the occlusion query's Z pass dump (E5: it travels as a sync tag, not as a raw packet). With real queries it only names the result block
                // (the begin dumps to the block, the end to the block + 8); the answer is written when the graphics card has counted
                if (Core::NativeOcclusion::Enabled()) {
                    occlusion_dump_address = static_cast<VAddr>(Core::NativeSync::DecodeEventAddress(restored.data()));
                } else {
                    FakeOcclusionDump(static_cast<VAddr>(Core::NativeSync::DecodeEventAddress(restored.data())));
                }
            } else if (opcode == 0x69) {
                // DB_COUNT_CONTROL: 0x11000102 begins the counting of an occlusion query, 0x11000100 ends it
                if (Core::NativeOcclusion::Enabled() && occlusion_dump_address != 0 && rasterizer) {
                    const VAddr block = occlusion_dump_address;
                    if ((restored[2] & 2u) != 0) {
                        const u32 pairs = num_counter_pairs;
                        RasterRun([this, block, pairs] { rasterizer->OcclusionOpen(block, pairs); });
                    } else {
                        RasterRun([this, block] { rasterizer->OcclusionClose(block - 8); });
                    }
                }
            } else if (opcode == 0x3c) {
                auto task = DirectWait(record.wait);
                RESUME_GFX(task);
                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                task.handle.destroy();
            }
            break;
        }
        }
    }
    leave_clear();
    FIBER_EXIT;
}


} // namespace Core
