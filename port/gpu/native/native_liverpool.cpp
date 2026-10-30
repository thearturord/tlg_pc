// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include "native/native_stream_probe.h"
#include "native/native_timeline.h"
#include "native/native_write_watch.h"
#include "native/native_registry.h"
#include "native/native_resource_audit.h"
#include "common/arch.h"
#ifdef ARCH_X86_64
#include <immintrin.h>
#endif
#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "native/native_audit.h"
#include "native/native_census.h"
#include "native/native_occlusion.h"
#include "native/native_direct.h"
#include "native/native_draw.h"
#include "native/native_uploads.h"
#include "native/native_drain_audit.h"
#include "native/native_readbacks.h"
#include "native/native_entry_dump.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_resources.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/liverpool_internal.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "native/native_readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/readback_trace.h"
#include "native/native_debug_server.h"
#include "video_core/thread_usage.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

// Native renderer: member functions of an emulator class that belong to the native path. They live here (not in the emulator's file)
// so that the emulator files only call into native/; the class stays declared in the emulator's header until the owning plan phase
// (documents/tlg/native plan/no-emulation-plan.md, E-1 part 2) gives the native side its own interface.

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])



void Liverpool::ApplyStateTemplate(const Core::NativeDraw::StateTemplate& state) {
    using Core::NativeDraw::StateWrite;
    for (const StateWrite& write : state.writes) {
        switch (write.kind) {
        case StateWrite::ClearState:
            regs.SetDefaults();
            MirrorSetDefaults();
            break;
        case StateWrite::Context:
            regs.reg_array[Regs::ContextRegWordOffset + write.reg] = write.value;
            MirrorRegs(Regs::ContextRegWordOffset + write.reg, 1);
            break;
        case StateWrite::Sh:
            if (write.reg >= 0x200 && write.reg <= (0x200 + sizeof(ComputeProgram) / 4)) {
                reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state)[write.reg - 0x200] = write.value;
            } else {
                regs.reg_array[Regs::ShRegWordOffset + write.reg] = write.value;
                MirrorRegs(Regs::ShRegWordOffset + write.reg, 1);
            }
            break;
        case StateWrite::Uconfig:
            regs.reg_array[Regs::UconfigRegWordOffset + write.reg] = write.value;
            MirrorRegs(Regs::UconfigRegWordOffset + write.reg, 1);
            break;
        case StateWrite::NumInstances:
            regs.num_instances.num_instances = write.value;
            MirrorObject(&regs.num_instances, sizeof(regs.num_instances));
            break;
        case StateWrite::IndexType:
            regs.index_buffer_type.raw = write.value;
            MirrorObject(&regs.index_buffer_type, sizeof(regs.index_buffer_type));
            break;
        }
    }
}

/// Applies a record to the private register block, by register group (a group whose identity equals the last one applied is skipped).
void Liverpool::ExecApplyRecord(const Core::NativeDraw::ExecDraw& draw, bool skip_fixed, u32 index_count) {
    auto& live = Core::NativeDraw::Live();
    const auto set = [&](u32 index, u32 value) {
        if (u32* const word = exec_state.Word(index)) {
            ExecSetWord(index, word, value);
        }
    };
    const auto apply = [&](u32 begin, u32 end) {
        for (u32 i = begin; i < end; ++i) {
            if (skip_fixed && IsFixedStateRegister(draw.regs[i].index)) {
                continue;
            }
            set(draw.regs[i].index, draw.regs[i].value);
        }
    };
    if (draw.grouped != 0 && VideoCore::ReadbackPolicy::Get().group_sig.load(std::memory_order_relaxed) != 0) {
        for (u32 g = 0; g < NumRegisterGroups; ++g) {
            const u32 begin = draw.group_begin[g];
            const u32 end = draw.group_begin[g + 1];
            if (begin == end) {
                continue;
            }
            const u64 sig = draw.group_sig[g];
            if (sig != 0 && exec_group_sig_valid[g] && exec_group_applied_sig[g] == sig && (skip_fixed || exec_group_sig_full[g])) {
                if ((++exec_group_tick & 31) != 0) {
                    continue;
                }
                bool violation = false;
                for (u32 i = begin; i < end && !violation; ++i) {
                    const u32 index = draw.regs[i].index;
                    if (skip_fixed && IsFixedStateRegister(index)) {
                        continue;
                    }
                    const u32* const word = exec_state.Word(index);
                    violation = word != nullptr && *word != draw.regs[i].value;
                }
                if (!violation) {
                    continue;
                }
                live.state_group_violations.fetch_add(1, std::memory_order_relaxed);
            }
            apply(begin, end);
            exec_group_sig_valid[g] = sig != 0;
            exec_group_applied_sig[g] = sig;
            exec_group_sig_full[g] = !skip_fixed;
        }
        apply(draw.grouped, draw.count);
        if (draw.typed) {
            for (u32 mask = draw.vs_user_data_mask; mask != 0; mask &= mask - 1) {
                const u32 i = static_cast<u32>(std::countr_zero(mask));
                ExecSetWord(Core::NativeDraw::VsUserDataRegister + i, &exec_state.vs_program.user_data[i], draw.vs_user_data[i]);
            }
            for (u32 mask = draw.ps_user_data_mask; mask != 0; mask &= mask - 1) {
                const u32 i = static_cast<u32>(std::countr_zero(mask));
                ExecSetWord(Core::NativeDraw::PsUserDataRegister + i, &exec_state.ps_program.user_data[i], draw.ps_user_data[i]);
            }
        }
    } else {
        apply(0, draw.count);
    }
    set(static_cast<u32>(&regs.draw_initiator - regs.reg_array.data()), 0);
    if (index_count != ~0u) {
        set(static_cast<u32>(&regs.num_indices - regs.reg_array.data()), index_count);
    }
}

void Liverpool::StartAscTag(const u32* tag, const void* address, u32 vqid) {
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
Liverpool::Task Liverpool::ProcessDirectAsc(std::shared_ptr<Core::NativeDirect::Buffer> direct, u32 vqid) {
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
        case EntryKind::Packet:
            break;
        case EntryKind::Sync: {
            std::array<u32, 8> restored;
            const u32 opcode = Core::NativeDraw::RestoreSyncTag(entry.words.data(), restored);
            if (opcode == 0x49) {
                SignalComputeRelease(*reinterpret_cast<const PM4CmdReleaseMem*>(restored.data()), vqid, queue.pipe_id);
            } else if (opcode == 0x37) {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(restored.data());
                CpWriteData(write_data->Address<void*>(), write_data->data, (write_data->header.count.Value() - 2) * 4, vqid + 1);
            } else if (opcode == 0x3c) {
                auto task = DirectWaitAsc(*reinterpret_cast<const PM4CmdWaitRegMem*>(restored.data()), vqid);
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

void Liverpool::StartNativeTag(const u32* tag) {
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

void Liverpool::ExecuteNativeDraw(const void* header, u32 asc_vqid, u32 index_count) {
    native_exec_calls++;
    Core::NativeDraw::ExecDraw draw;
    if (!rasterizer || !Core::NativeDraw::TakeExecDraw(native_tag_pending.seq, draw, VideoCore::ReadbackPolicy::Get().group_sig.load(std::memory_order_relaxed) != 0)) {
        native_tag_pending.valid = false;
        LOG_ERROR(Render, "native draw: the record {} is gone, a draw is lost", native_tag_pending.seq);
        return;
    }
    SeedExecState();
    // C8 (2026-10-28): the shared register image is no longer written by records; a record plans from the private register block only
    if (draw.kind == 10) {
        // memory copy: destination, source and the number of bytes travel in the record
        native_tag_pending.valid = false;
        if (rasterizer) {
            Core::DrainAudit::SetNextWrite(draw.args_addr, draw.a1);
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.args_addr), source = static_cast<VAddr>(draw.extra_addr), bytes = draw.a1,
                 staging = draw.dst_usage == 3 && Core::NativeReadbacks::ExplicitEnabled()] {
                    rasterizer->CopyBuffer(destination, source, bytes, false, false, staging);
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
                [this, destination = static_cast<VAddr>(draw.args_addr), source = draw.start_index,
                 staging = draw.dst_usage == 3 && Core::NativeReadbacks::ExplicitEnabled()] {
                    rasterizer->CopyBuffer(destination, source, sizeof(u32), false, true, staging);
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
        // depth / stencil clear: the registers of the depth target go into the private register block, the clear is made from the inputs taken from it
        native_tag_pending.valid = false;
        for (u32 i = 0; i < draw.count; ++i) {
            if (u32* const word = exec_state.Word(draw.regs[i].index)) {
                ExecSetWord(draw.regs[i].index, word, draw.regs[i].value);
            }
        }
        auto clear_inputs = Vulkan::MakeDrawInputs(exec_state);
        Core::NativeDraw::Live().state_planned.fetch_add(1, std::memory_order_relaxed);
        RasterRun(
            [this, extent = draw.db_extent, inputs = std::move(clear_inputs)] {
                CbDbExtent db_extent;
                db_extent.raw = extent;
                rasterizer->ClearDepthStencilTarget(inputs, db_extent);
            },
            false, false);
        ExecRestore();
        // what the SDK's clear leaves behind: all colour targets writable again, no clear bits in the depth render control
        regs.reg_array[Regs::ContextRegWordOffset + 0x8e] = 0xf;
        MirrorRegs(Regs::ContextRegWordOffset + 0x8e, 1);
        regs.reg_array[Regs::ContextRegWordOffset + 0x0] = 0;
        MirrorRegs(Regs::ContextRegWordOffset + 0x0, 1);
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
        ComputeProgram alone{};
        auto* alone_dwords = reinterpret_cast<u32*>(&alone);
        for (u32 i = 0; i < draw.count; ++i) {
            const u32 dword = draw.regs[i].index - Core::NativeDraw::ComputeRegisterBase;
            if (dword < sizeof(ComputeProgram) / sizeof(u32)) {
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
            dispatch.dispatch_size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
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
    if (draw.cb_extents_only) {
        native_tag_pending.valid = false;
    }
    if (draw.has_extents) {
        // the extent hints the game's target bind sends in a trailing NOP packet
        for (u32 slot = 0; slot < NUM_COLOR_BUFFERS && slot < draw.cb_extent.size(); ++slot) {
            if ((draw.cb_extent_mask >> slot) & 1) {
                last_cb_extent[slot].raw = draw.cb_extent[slot];
            }
        }
        if (!draw.cb_extents_only) {
            last_db_extent.raw = draw.db_extent;
        }
    }
    // The record goes straight into the draw state (and the raster thread's deltas), not through the register file: the
    // next draw writes only what differs, and a draw of the packet path first gets back what its own packets wrote.
    // Registers no member of the draw state covers are not needed by anything and are skipped. The registers of the
    // fixed-function state the draw carries in Vulkan terms are only needed to compare the two descriptions (the first
    // draws, then every 256th).
    native_fixed_verify = true;
    if (draw.fixed) {
        const u64 n = native_fixed_draws++;
        // the first draws (TLG_FIXED_VERIFY_FIRST, default 20,000; 200,000 until 2026-10-28, 0 violations since) and then one in 256 take the
        // register path to compare the fixed state
        static const u64 first = [] {
            const char* e = std::getenv("TLG_FIXED_VERIFY_FIRST");
            return e != nullptr && *e != 0 ? std::strtoull(e, nullptr, 10) : 0ull; // off by default since 2026-10-30
        }();
        native_fixed_verify = n < first || (first != 0 && (n & 255) == 0);
    }
    bool planned_from_record = false;
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
        const bool exec_on = policy.exec_inputs.load(std::memory_order_relaxed) != 0;
        const bool switches = exec_on && policy.stage_key.load(std::memory_order_relaxed) != 0 && policy.group_sig.load(std::memory_order_relaxed) != 0;
        const auto left = [&live](u32 reason) {
            live.exec_fallback.fetch_add(1, std::memory_order_relaxed);
            live.exec_reason[reason].fetch_add(1, std::memory_order_relaxed);
        };
        const auto take = [&](std::unique_ptr<Vulkan::DrawPlan> plan) {
            // a fast clear has no other source for a comparison when the shared register image is not used
            if ((++exec_plan_tick & 31) == 0 && !draw.cb_extents_only) {
                native_audit_plan = plan.release();
                native_audit_kind = draw.kind;
                native_audit_vertex_mask = draw.vs_user_data_mask;
                native_audit_pixel_mask = draw.ps_user_data_mask;
            } else {
                native_exec_plan = plan.release();
                planned_from_record = true;
                live.exec_typed.fetch_add(1, std::memory_order_relaxed);
                if (draw.cb_extents_only) {
                    live.exec_fast_clear.fetch_add(1, std::memory_order_relaxed);
                } else if (index_count != ~0u) {
                    live.exec_inline.fetch_add(1, std::memory_order_relaxed);
                }
            }
        };
        if (index_count != ~0u && !draw.cb_extents_only) {
            live.exec_inline_seen.fetch_add(1, std::memory_order_relaxed);
        }
        if (draw.cb_extents_only) {
            // a fast clear eliminate: the operation and its render target come from the record
            if (split_active && draw.kind == 4) {
                take(rasterizer->PlanExecOperation(draw));
            } else if (exec_on) {
                left(5);
            }
        } else if (exec_on && draw.kind <= 2) {
            if (!switches) {
                left(0);
            } else if (!draw.typed || draw.stage_key == 0 || draw.fixed == nullptr) {
                left(!draw.typed ? 1 : draw.stage_key == 0 ? 6 : 7);
            } else if (!split_active) {
                left(2);
            } else {
                // an inline draw has its count with the draw packet (index_count), anything else carries it in the record
                native_inputs_key = draw.inputs_key;
                const u32 verify = LookupDrain(1);
                if (native_fixed_verify) {
                    left(3);
                } else {
                    const auto make_plan = [&] {
                        return rasterizer->PlanExecDraw(*draw.fixed, draw.fixed_stale, draw.stage_key, native_inputs_key, exec_vertex_user_data,
                                                        exec_pixel_user_data, draw, index_count,
                                                        native_tag_pending.valid ? (native_tag_pending.meta & 0xff) : 15u);
                    };
                    auto plan = make_plan();
                    if (verify == 2) {
                        plan = ReplanAfterDrain(std::move(plan), 1, make_plan);
                    }
                    if (plan) {
                        take(std::move(plan));
                    } else {
                        left(4);
                    }
                }
            }
        }
    }
    if (VideoCore::ReadbackPolicy::Get().cs_audit.load(std::memory_order_relaxed) != 0) {
        Core::NativeDraw::Live().overlay_records.fetch_add(1, std::memory_order_relaxed);
    }
    if (!planned_from_record && draw.typed && (++group_sig_tick & 63) == 0) {
        // the typed user data must be what the register pairs of the record say
        Core::NativeDraw::ExecDraw pairs;
        if (Core::NativeDraw::TakeExecDraw(native_tag_pending.seq, pairs, false)) {
            using namespace Core::NativeDraw;
            auto& live = Core::NativeDraw::Live();
            for (u32 i = 0; i < pairs.count; ++i) {
                const u32 index = pairs.regs[i].index;
                const u32 value = pairs.regs[i].value;
                const bool vs = index - VsUserDataRegister < 16;
                const bool ps = index - PsUserDataRegister < 16;
                if (!vs && !ps) {
                    continue;
                }
                const u32 slot = vs ? index - VsUserDataRegister : index - PsUserDataRegister;
                const bool present = ((vs ? draw.vs_user_data_mask : draw.ps_user_data_mask) >> slot & 1) != 0;
                if (!present || (vs ? draw.vs_user_data[slot] : draw.ps_user_data[slot]) != value) {
                    live.group_sig_violations.fetch_add(1, std::memory_order_relaxed);
                    live.group_sig_violations_by_group[vs ? GroupVsUserData : GroupPsUserData].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    if (!planned_from_record) {
        Core::NativeDraw::MaterializeRegs(draw); // A5.3: the register groups of a draw whose plan comes from the registers
        if (!draw.cb_extents_only) {
            ExecApplyRecord(draw, !native_fixed_verify, index_count); // a fast clear eliminate's registers are for that operation only (its plan comes from the record)
        }
    }
    RasterCall call{};
    if (draw.kind == 2) {
        call.kind = RasterCall::Kind::DrawIndirect;
        call.marker = "gfx:{}:NativeDrawIndexIndirect";
        call.indexed = true;
        call.args_addr = draw.args_addr;
        call.offset = draw.offset;
        call.stride = sizeof(DrawIndexedIndirectArgs);
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
    native_fixed_stale = draw.fixed_stale;
    native_exec_key = draw.stage_key != 0 && draw.grouped != 0 && !draw.cb_extents_only &&
                              VideoCore::ReadbackPolicy::Get().stage_key.load(std::memory_order_relaxed) != 0 &&
                              VideoCore::ReadbackPolicy::Get().group_sig.load(std::memory_order_relaxed) != 0
                          ? draw.stage_key
                          : 0;
    if (!planned_from_record && split_active && !draw.cb_extents_only) {
        // plan from the private register block (the shared image is not read)
        Vulkan::DrawIndirectParams params{};
        if (call.kind == RasterCall::Kind::DrawIndirect) {
            params = Vulkan::DrawIndirectParams{.vertex_sgpr_offset = call.vertex_sgpr_offset, .instance_sgpr_offset = call.instance_sgpr_offset, .is_indirect = true};
        }
        const u32 verify = LookupDrain(2);
        Core::NativeAudit::SetPlanCaller(native_tag_pending.valid ? Core::NativeDraw::RecordCaller(native_tag_pending.seq) : 0);
        const auto make_candidate = [&] {
            return rasterizer->PlanDraw(exec_state, params, native_fixed, native_fixed_stale, native_fixed_verify, native_exec_key, native_inputs_key);
        };
        auto candidate = make_candidate();
        if (verify == 2) {
            candidate = ReplanAfterDrain(std::move(candidate), 2, make_candidate);
        }
        Core::NativeDraw::Live().state_planned.fetch_add(1, std::memory_order_relaxed);
        native_exec_plan = candidate.release();
    }
    RasterDispatch(call);
    if (native_exec_plan) {
        rasterizer->RecyclePlan(std::unique_ptr<Vulkan::DrawPlan>{std::exchange(native_exec_plan, nullptr)});
    }
    if (native_audit_plan) {
        rasterizer->RecyclePlan(std::unique_ptr<Vulkan::DrawPlan>{std::exchange(native_audit_plan, nullptr)});
    }
    native_inputs_key = 0;
    native_exec_key = 0;
    native_fixed = nullptr;
    native_exec_scope = false;
    if (draw.cb_extents_only) {
        // the fast clear eliminate registers (color control in eliminate mode) must not reach the next draw
        last_cb_extent = saved_cb_extent;
        ExecRestore();
    }
}

void Liverpool::QueueAnnounced(Core::NativeDirect::Buffer& direct) {
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
// E5 (documents/tlg/native plan/no-emulation-plan.md): the raw PM4 packets the game's own stream still carries. The default scene (E0 census, 2026-10-30) has exactly
// three kinds: the occlusion query pair (EVENT_WRITE PIXEL_PIPE_STAT_CONTROL / PIXEL_PIPE_STAT_DUMP with index ZPASS_DONE, and a SET_CONTEXT_REG of context register
// 1, DB_COUNT_CONTROL) and CP DMA (17 per run). Anything else is recorded as a rare event (the packet still executes as before) so the cutscene and the heavy scene
// show what else would need a typed record before the generic packet executors can go.
static void NoteUnexpectedRawPacket(u32 opcode, const u32* words, u32 dwords) {
    bool expected = false;
    switch (opcode) {
    // 0x50 (CP DMA with the control dword 0x84004000) is a tag since E5 as well; every raw packet that reaches here was not converted
    // 0x46 (the occlusion query's EVENT_WRITE pair) and 0x69 (DB_COUNT_CONTROL) are tags since E5; one that is still raw here was not converted
    default:
        break;
    }
    if (!expected) {
        char detail[160];
        std::snprintf(detail, sizeof detail, "raw PM4 packet opcode %#x, %u dwords: %08x %08x %08x %08x", opcode, dwords, words[0], dwords > 1 ? words[1] : 0u,
                      dwords > 2 ? words[2] : 0u, dwords > 3 ? words[3] : 0u);
        Core::RareEvents::Note("raw_pm4_packet_" + std::to_string(opcode), detail);
    }
}

Liverpool::Task Liverpool::ProcessDirect(std::shared_ptr<Core::NativeDirect::Buffer> direct, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);
    cblock.Reset();
    // the constant engine buffer of the submission runs as its own task next to the entries, as in the walk
    Task ce_task{};
    if (!ccb.empty()) {
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    QueueAnnounced(*direct);
    if (!ccb.empty()) {
        Core::NativeCensus::NoteCeSubmission(static_cast<u32>(ccb.size()));
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
            ++clear_entries[entry.kind == EntryKind::Tag ? 0 : entry.kind == EntryKind::Sync ? 1 : entry.kind == EntryKind::Packet ? 2 : 3];
        }
        entry_dump.Packet(entry.words.data(), entry.address, entry.dwords != 0);
        Core::NativeCensus::NoteEntry(0, static_cast<u32>(entry.kind), entry.opcode);
        switch (entry.kind) {
        case EntryKind::Tag:
            StartNativeTag(entry.words.data());
            break;
        case EntryKind::Flip:
            RasterRun([] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip); });
            break;
        case EntryKind::Packet: {
            const auto* packet = reinterpret_cast<const PM4Header*>(direct->PacketWords(entry));
            Core::NativeCensus::NoteRawPacket(entry.opcode, reinterpret_cast<const u32*>(packet), entry.dwords);
            NoteUnexpectedRawPacket(entry.opcode, reinterpret_cast<const u32*>(packet), entry.dwords);
            if (entry.opcode == 0x76 || entry.opcode == 0x69) {
                const u32* words = reinterpret_cast<const u32*>(packet);
                Core::NativeCensus::NoteRegisterPacket(entry.opcode, words[1], words[0] >> 16 & 0x3fff);
            }
            if (entry.opcode == 0x76) {
                ExecuteSetShReg(packet);
            } else if (entry.opcode == 0x69) {
                ExecuteSetContextReg(packet);
            } else if (entry.opcode == 0x46) {
                ExecuteEventWrite(packet);
            } else if (entry.opcode == 0x85) {
                ++cblock.de_count;
            } else if (entry.opcode == 0x86) {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
            } else if (entry.opcode == 0x15) {
                ExecuteDispatchDirect(packet, direct->begin);
            } else if (entry.opcode == 0x37) {
                ExecuteWriteData(packet);
            } else {
                ExecuteDma(*reinterpret_cast<const PM4DmaData*>(packet));
            }
            break;
        }
        case EntryKind::StateBuffer: {
            u64 pointer;
            std::memcpy(&pointer, &entry.seq, sizeof(pointer));
            ApplyStateTemplate(*reinterpret_cast<const Core::NativeDraw::StateTemplate*>(pointer));
            break;
        }
        case EntryKind::Sync: {
            std::array<u32, 8> restored;
            const u32 opcode = Core::NativeDraw::RestoreSyncTag(entry.words.data(), restored);
            if (opcode == 0x47) {
                SignalGraphicsEop(*reinterpret_cast<const PM4CmdEventWriteEop*>(restored.data()));
            } else if (opcode == 0x37) {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(restored.data());
                CpWriteData(write_data->Address<u64*>(), write_data->data, (write_data->header.count.Value() - 2) * 4, GfxQueueId);
            } else if (opcode == 0x50) {
                // a CP DMA copy (E5: it travels as a sync tag, not as a raw packet)
                ExecuteDma(*reinterpret_cast<const PM4DmaData*>(restored.data()));
            } else if (opcode == 0x46) {
                // the occlusion query's Z pass dump (E5: it travels as a sync tag, not as a raw packet). With real queries it only names the result block
                // (the begin dumps to the block, the end to the block + 8); the answer is written when the graphics card has counted
                if (Core::NativeOcclusion::Enabled()) {
                    occlusion_dump_address = static_cast<VAddr>(restored[2] & 0xfffffff8u) | static_cast<VAddr>(restored[3] & 0xffu) << 32;
                } else {
                    ExecuteEventWrite(reinterpret_cast<const PM4Header*>(restored.data()));
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
                auto task = DirectWait(*reinterpret_cast<const PM4CmdWaitRegMem*>(restored.data()));
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
    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }
    FIBER_EXIT;
}

} // namespace AmdGpu
