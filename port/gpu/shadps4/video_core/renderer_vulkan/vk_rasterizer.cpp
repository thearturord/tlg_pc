// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <atomic>
#include <thread>
#include "common/arch.h"
#include "common/debug.h"
#include "core/debug_server.h"
#include "core/guest_hooks.h"
#include "core/native_draw.h"
#include "core/native_audit.h"
#include "core/native_readbacks.h"
#include "core/native_resources.h"
#include "core/resource_audit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

static Shader::PushData MakeUserData(const FixedState* fixed) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    if (fixed) {
        push_data.xoffset = fixed->transform.x_offset;
        push_data.xscale = fixed->transform.x_scale;
        push_data.yoffset = fixed->transform.y_offset;
        push_data.yscale = fixed->transform.y_scale;
    }
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      readback_batching{EmulatorSettings.IsReadbackBatchingEnabled()},
      async_fences{EmulatorSettings.IsReadbackAsyncFencesEnabled()},
      rt_reuse_enabled{EmulatorSettings.IsRtLookupReuseEnabled()},
      encoder_debug{EmulatorSettings.GetCpEncoderDebug()},
      full_eop_fences{EmulatorSettings.GetCpEncoderMode() >= 2 || (encoder_debug & 4) != 0},
      flush_commands{EmulatorSettings.GetCpFlushCommands()} {
    VideoCore::ReadbackPolicy::InitFromSettings();
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    scheduler.SetEncoderMode(EmulatorSettings.GetCpEncoderMode());
    memory->SetRasterizer(this);
    if (const char* env = std::getenv("TLG_CONST_RING")) {
        const_ring_level = static_cast<u32>(std::clamp(std::atoi(env), 0, 4));
    }
    if (const_ring_level >= 2) {
        constant_ring = std::make_unique<ConstantRing>(instance, scheduler);
    }
    if (const char* env = std::getenv("TLG_CONST_RING_VERIFY")) {
        ring_verify_interval = static_cast<u32>(std::max(std::atoi(env), 0));
    }
    LOG_INFO(Render, "constant ring: level {}, verify every {}", const_ring_level, ring_verify_interval);
    if (const char* env = std::getenv("TLG_TEXTURE_SET_MEMO"); env && env[0] == '1') {
        texture_set_on = true;
        texture_sets = std::make_unique<TextureSet[]>(TextureSetSlots);
    }
    LOG_INFO(Render, "texture set memo: {}", texture_set_on ? "on" : "off");

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::MaybeFlushRecorded() {
    const u32 limit =
        VideoCore::ReadbackPolicy::Get().flush_commands.load(std::memory_order_relaxed);
    if (limit != 0 && scheduler.PendingEncoderCommands() >= limit) {
        scheduler.Flush();
    }
}

u64 Rasterizer::CurrentTick() const noexcept {
    return scheduler.CurrentTick();
}

bool Rasterizer::CommandStreamBarrier(u64 producer_tick) {
    if (scheduler.IsFree(producer_tick)) {
        // The producer already finished on the GPU, nothing left to order against.
        return false;
    }
    if (last_barrier_work == recorded_work) {
        // No work was recorded since the last barrier: the producer comes before it and every
        // later command comes after it, so it already orders this wait too.
        return false;
    }
    last_barrier_work = recorded_work;
    runtime.FlushBarriers();
    scheduler.EndRendering();
    const vk::MemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    return true;
}

bool Rasterizer::ClassifyOperation(const AmdGpu::DrawRegs& regs, DrawClassification& out) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        out = {DrawClass::EliminateFastClear};
        return true;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        out = {DrawClass::FmaskDecompress};
        return true;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        out = {DrawClass::Resolve};
        return true;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        out = {DrawClass::PrimitiveNone};
        return true;
    }
    // The SDK's clear: a full-screen triangle (embedded vertex shader, three vertices, no color target) with the depth /
    // stencil clear bits of DB_RENDER_CONTROL set. Its pixels carry nothing; the render pass clear does the work.
    if ((regs.depth_render_control.depth_clear_enable || regs.depth_render_control.stencil_clear_enable) &&
        (regs.vs_program.address >> 32) != 0 && regs.num_indices == 3 &&
        regs.color_target_mask.raw == 0 && regs.depth_buffer.DepthValid()) {
        out = {DrawClass::DepthClear};
        return true;
    }
    if ((regs.depth_render_control.decompress_enable || regs.depth_render_control.resummarize_enable ||
         regs.depth_render_control.depth_compress_disable || regs.depth_render_control.stencil_compress_disable) &&
        !regs.depth_render_control.depth_clear_enable && !regs.depth_render_control.stencil_clear_enable &&
        !regs.depth_render_control.depth_copy && !regs.depth_render_control.stencil_copy &&
        (regs.vs_program.address >> 32) != 0 && regs.num_indices == 3 &&
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable) {
        out = {DrawClass::DepthTransition};
        return true;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        out = {DrawClass::DepthStencilCopy, depth_copy, stencil_copy};
        return true;
    }
    return false;
}

Rasterizer::DrawClassification Rasterizer::ClassifyDraw(const AmdGpu::DrawRegs& regs,
                                                        const FixedState* executed) {
    if (executed == nullptr) {
        DrawClassification operation;
        if (ClassifyOperation(regs, operation)) {
            return operation;
        }
    }
    const bool cb_disabled = executed ? !executed->color_enabled
                                      : regs.color_control.mode ==
                                            AmdGpu::ColorControl::OperationMode::Disable;
    const bool no_targets = executed ? executed->NoColorTargets() : regs.color_target_mask.raw == 0;
    const u32 skip = VideoCore::ReadbackPolicy::Get().skip_shadows.load(std::memory_order_relaxed);
    if (skip != 0 && (cb_disabled || no_targets) && regs.depth_buffer.DepthValid()) {
        const u32 pitch = regs.depth_buffer.Pitch();
        const u32 height = regs.depth_buffer.Height();
        if (skip == 1 ? pitch == height : pitch == skip || height == skip) {
            return {DrawClass::SkipShadow};
        }
    }
    return {DrawClass::Normal};
}

Rasterizer::DrawClassification Rasterizer::ClassifyRemaining(const DrawInputs& inputs, const FixedState& fixed) {
    const u32 skip = VideoCore::ReadbackPolicy::Get().skip_shadows.load(std::memory_order_relaxed);
    if (skip != 0 && (!fixed.color_enabled || fixed.NoColorTargets()) && inputs.depth_buffer.DepthValid()) {
        const u32 pitch = inputs.depth_buffer.Pitch();
        const u32 height = inputs.depth_buffer.Height();
        if (skip == 1 ? pitch == height : pitch == skip || height == skip) {
            return {DrawClass::SkipShadow};
        }
    }
    return {DrawClass::Normal};
}

bool Rasterizer::FilterDraw(const DrawPlan* plan) {
    // With a plan everything comes from the plan (the command processor thread made it from its own state); without one (the
    // unsplit path) from the register mirror.
    DrawInputs inputs_storage;
    const DrawInputs& inputs = plan ? plan->inputs : (inputs_storage = MakeDrawInputs(liverpool->RasterRegs()), inputs_storage);
    const auto& cb_extents = plan ? plan->cb_extent : liverpool->RasterCbExtents();
    const AmdGpu::CbDbExtent& db_extent = plan ? plan->db_extent : liverpool->RasterDbExtent();
    DrawClassification classified;
    if (plan != nullptr) {
        classified = plan->op_class != 0xff ? DrawClassification{static_cast<DrawClass>(plan->op_class), plan->depth_copy, plan->stencil_copy}
                                            : ClassifyRemaining(plan->inputs, plan->fixed);
    } else {
        classified = ClassifyDraw(liverpool->RasterRegs(), nullptr);
    }
    {
        // how often the game's own transition draws (fast clear eliminate, fmask, resolve, depth copy) occur
        static std::array<u64, 12> class_counts{};
        static u64 last_log_ns = 0;
        ++class_counts[static_cast<size_t>(classified.kind)];
        const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        if (now - last_log_ns > 5'000'000'000ull) {
            if (last_log_ns != 0) {
                LOG_INFO(Render_Vulkan,
                         "draw classes per 5 s: eliminate fast clear {}, fmask decompress {}, resolve {}, primitive none {}, "
                         "depth/stencil copy {}, skipped shadow {}, depth clear {}, depth transition {}, normal {}",
                         class_counts[static_cast<size_t>(DrawClass::EliminateFastClear)],
                         class_counts[static_cast<size_t>(DrawClass::FmaskDecompress)],
                         class_counts[static_cast<size_t>(DrawClass::Resolve)],
                         class_counts[static_cast<size_t>(DrawClass::PrimitiveNone)],
                         class_counts[static_cast<size_t>(DrawClass::DepthStencilCopy)],
                         class_counts[static_cast<size_t>(DrawClass::SkipShadow)],
                         class_counts[static_cast<size_t>(DrawClass::DepthClear)],
                         class_counts[static_cast<size_t>(DrawClass::DepthTransition)],
                         class_counts[static_cast<size_t>(DrawClass::Normal)]);
            }
            class_counts = {};
            last_log_ns = now;
        }
    }
    switch (classified.kind) {
    case DrawClass::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear(inputs, cb_extents[0]);
        return false;
    case DrawClass::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        EliminateFastClear(inputs, cb_extents[0]);
        return false;
    case DrawClass::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve(inputs, cb_extents[0], cb_extents[1]);
        return false;
    case DrawClass::PrimitiveNone:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    case DrawClass::DepthStencilCopy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(inputs, db_extent, classified.depth_copy, classified.stencil_copy);
        return false;
    case DrawClass::SkipShadow:
        VideoCore::ReadbackPolicy::Get().skipped_draws.fetch_add(1, std::memory_order_relaxed);
        return false;
    case DrawClass::DepthClear:
        ClearDepthTarget(inputs, db_extent);
        return false;
    case DrawClass::DepthTransition:
        return false;
    case DrawClass::Normal:
        break;
    }
    {
        // Which depth targets do the depth-only draws go to? (debug server /policy)
        const bool no_color =
            plan ? (!plan->fixed.color_enabled || plan->fixed.NoColorTargets())
                 : (liverpool->RasterRegs().color_control.mode == AmdGpu::ColorControl::OperationMode::Disable ||
                    liverpool->RasterRegs().color_target_mask.raw == 0);
        if (no_color && inputs.depth_buffer.DepthValid()) {
            VideoCore::ReadbackPolicy::Get().NoteDepthOnly(inputs.depth_buffer.Pitch(), inputs.depth_buffer.Height());
        }
    }
    return true;
}

std::unique_ptr<DrawPlan> Rasterizer::PlanDraw(const AmdGpu::DrawRegs& regs,
                                               DrawIndirectParams params,
                                               const FixedState* native, u32 native_stale,
                                               bool verify, u64 exec_key, u64 inputs_key) {
    // SkipShadow draws are still planned: the switch can change between the front and the raster
    // thread, and the raster thread must find a plan for every draw it does not drop.
    // An executed draw is a normal draw by construction (only the shadow skip switch looks at it, when the draw is
    // recorded); the first draws and every 256th are classified from the registers as well.
    bool executed = native && !verify;
    DrawClassification operation{};
    bool is_operation = false;
    if (!executed) {
        const auto classified = ClassifyDraw(regs);
        const auto kind = classified.kind;
        if (kind != DrawClass::Normal && kind != DrawClass::SkipShadow) {
            if (native) {
                LOG_INFO(Render_Vulkan, "executed draw is classified {} by the registers", static_cast<u32>(kind));
            }
            // one of the game's own transition operations: the plan carries what the raster thread needs to perform it
            is_operation = true;
            operation = classified;
        }
        executed = native && kind == DrawClass::Normal;
    }
    std::unique_ptr<DrawPlan> plan;
    {
        std::scoped_lock lk{plan_pool_mutex};
        if (!draw_plan_pool.empty()) {
            plan = std::move(draw_plan_pool.back());
            draw_plan_pool.pop_back();
        }
    }
    if (!plan) {
        plan = std::make_unique<DrawPlan>();
    }
    plan->executed = executed;
    plan->op_class = is_operation ? static_cast<u8>(operation.kind) : u8{0xff};
    plan->depth_copy = operation.depth_copy;
    plan->stencil_copy = operation.stencil_copy;
    if (is_operation) {
        plan->pipeline = nullptr;
        plan->executed = false;
        plan->inputs = MakeDrawInputs(regs);
        return plan;
    }
    if (native) {
        // The state built from the engine's objects replaces the one built from the registers. The first draws (and
        // then every 256th, `verify`) are compared with the registers' state; a difference is counted and the registers win.
        auto& live = Core::NativeDraw::Live();
        live.fixed_native.fetch_add(1, std::memory_order_relaxed);
        if ((VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 1073741824) != 0) {
            native_stale |= StaleClear;
        }
        // No group of the fixed-function state comes from the register mirror any more. The surface group (culling, front face, polygon
        // mode) comes from the engine's rasterizer state object, also where the engine did not write the register for this draw: the mirror's
        // value differed from the object's in nearly every such draw and changed with the frame (a period of three), which flickered tattoos,
        // leaves and whole meshes (native-isolation.md). The clip group and the clear flags never differed from the objects' values in 27 million
        // draws. Bits of native_sdk_strip restore the old behaviour for a comparison: 268435456 surface, 536870912 clip, 1073741824 clear flags.
        const u32 strip = VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed);
        const bool surface_from_mirror = (strip & 268435456) != 0;
        if (!surface_from_mirror) {
            native_stale &= ~StaleSurface;
        }
        if ((strip & 536870912) == 0) {
            native_stale &= ~StaleClip;
        }
        if ((strip & 1073741824) == 0) {
            native_stale &= ~StaleClear;
        }
        FixedState merged = *native;
        TakeFromRegisters(merged, regs, native_stale);
        if (Core::NativeAudit::StaleEnabled()) {
            const std::string stale_diff = DiffFixedState(*native, merged);
            Core::NativeAudit::NoteStale(native_stale, !stale_diff.empty(), stale_diff, Core::NativeAudit::PlanCaller(),
                                         Core::DebugServer::PresentCount());
        }
        const FixedState* use = &merged;
        if (verify) {
            plan->fixed = MakeFixedState(regs);
            if (!surface_from_mirror) {
                // the mirror's surface group is not the reference any more (see above): compare everything else
                plan->fixed.cull_mode = merged.cull_mode;
                plan->fixed.front_face = merged.front_face;
                plan->fixed.polygon_mode = merged.polygon_mode;
                plan->fixed.provoking_first = merged.provoking_first;
            }
            live.fixed_checked.fetch_add(1, std::memory_order_relaxed);
            const std::string diff = DiffFixedState(*use, plan->fixed);
            if (!diff.empty() && live.fixed_diff.fetch_add(1, std::memory_order_relaxed) < 60) {
                LOG_INFO(Render_Vulkan, "native fixed-function state differs from the registers in: {} (caller {:#x}, draw number {}, objects depth_clip {}, registers {} clipper {:#x})", diff, Core::NativeAudit::PlanCaller(),
                         live.fixed_checked.load(std::memory_order_relaxed), use->depth_clip, plan->fixed.depth_clip, std::bit_cast<u32>(regs.clipper_control));
            }
        } else {
            plan->fixed = merged;
        }
    } else {
        plan->fixed = MakeFixedState(regs);
    }
    plan->inputs = MakeDrawInputs(regs);
    if (inputs_key != 0 && plan->executed) {
        exec_inputs.insert_or_assign(inputs_key, plan->inputs);
    }
    pipeline_cache.PlanGraphics(regs, plan->fixed, params, *plan, plan->executed ? exec_key : 0);
    return plan;
}

namespace {
/// Writes register pairs of a record (entries `begin` to `end`) into the draw inputs, where the draw state would hold them.
void ApplyExecRegisters(DrawInputs& in, const Core::NativeDraw::ExecDraw& draw, u32 begin, u32 end) {
    static AmdGpu::DrawRegs layout{};
    struct Member {
        const void* in_regs;
        void* in_inputs;
        size_t size;
    };
    const std::array<Member, 18> members{{
        {&layout.color_buffers[0], &in.color_buffers[0], sizeof(in.color_buffers)},
        {&layout.depth_buffer, &in.depth_buffer, sizeof(in.depth_buffer)},
        {&layout.depth_view, &in.depth_view, sizeof(in.depth_view)},
        {&layout.depth_control, &in.depth_control, sizeof(in.depth_control)},
        {&layout.depth_htile_data_base, &in.depth_htile_data_base, sizeof(in.depth_htile_data_base)},
        {&layout.depth_render_control, &in.depth_render_control, sizeof(in.depth_render_control)},
        {&layout.depth_clear, &in.depth_clear, sizeof(in.depth_clear)},
        {&layout.stencil_clear, &in.stencil_clear, sizeof(in.stencil_clear)},
        {&layout.ta_bc_base, &in.ta_bc_base, sizeof(in.ta_bc_base)},
        {&layout.clip_user_data, &in.clip_user_data, sizeof(in.clip_user_data)},
        {&layout.num_indices, &in.num_indices, sizeof(in.num_indices)},
        {&layout.num_instances, &in.num_instances, sizeof(in.num_instances)},
        {&layout.index_offset, &in.index_offset, sizeof(in.index_offset)},
        {&layout.vgt_instance_step_rate_0, &in.vgt_instance_step_rate_0, sizeof(in.vgt_instance_step_rate_0)},
        {&layout.vgt_instance_step_rate_1, &in.vgt_instance_step_rate_1, sizeof(in.vgt_instance_step_rate_1)},
        {&layout.index_base_address, &in.index_base_address, sizeof(in.index_base_address)},
        {&layout.index_buffer_type, &in.index_buffer_type, sizeof(in.index_buffer_type)},
        {&layout.max_index_size, &in.max_index_size, sizeof(in.max_index_size)},
    }};
    for (u32 i = begin; i < end; ++i) {
        const auto* word = reinterpret_cast<const u8*>(layout.Word(draw.regs[i].index));
        if (!word) {
            continue;
        }
        for (const auto& member : members) {
            const auto* base = static_cast<const u8*>(member.in_regs);
            if (word >= base && word < base + member.size) {
                std::memcpy(static_cast<u8*>(member.in_inputs) + (word - base), &draw.regs[i].value, sizeof(u32));
            }
        }
    }
}
} // namespace

std::unique_ptr<DrawPlan> Rasterizer::PlanExecDraw(const FixedState& native, u32 native_stale, u64 exec_key, u64 inputs_key,
                                                   std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data,
                                                   const Core::NativeDraw::ExecDraw& draw, u32 index_count) {
    const auto strip = VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed);
    if ((strip & 1073741824) != 0) {
        native_stale |= StaleClear;
    }
    if ((strip & 268435456) == 0) {
        native_stale &= ~StaleSurface;
    }
    if ((strip & 536870912) == 0) {
        native_stale &= ~StaleClip;
    }
    if ((strip & 1073741824) == 0) {
        native_stale &= ~StaleClear;
    }
    const auto cached = exec_inputs.find(inputs_key);
    if (native_stale != 0 || cached == exec_inputs.end()) {
        return nullptr;
    }
    std::unique_ptr<DrawPlan> plan;
    {
        std::scoped_lock lk{plan_pool_mutex};
        if (!draw_plan_pool.empty()) {
            plan = std::move(draw_plan_pool.back());
            draw_plan_pool.pop_back();
        }
    }
    if (!plan) {
        plan = std::make_unique<DrawPlan>();
    }
    plan->executed = true;
    plan->op_class = 0xff;
    plan->depth_copy = plan->stencil_copy = false;
    plan->fixed = native;
    plan->inputs = cached->second;
    ApplyExecRegisters(plan->inputs, draw, draw.group_begin[AmdGpu::Liverpool::GroupArguments], draw.group_begin[AmdGpu::Liverpool::GroupArguments + 1]);
    if (index_count != ~0u) {
        plan->inputs.num_indices = index_count; // an inline draw: the count comes with the draw packet
    }
    if (!pipeline_cache.PlanGraphicsTyped(exec_key, plan->fixed, vertex_user_data, pixel_user_data, *plan)) {
        RecyclePlan(std::move(plan));
        return nullptr;
    }
    Core::NativeDraw::Live().fixed_native.fetch_add(1, std::memory_order_relaxed);
    return plan;
}

std::unique_ptr<DrawPlan> Rasterizer::PlanExecOperation(const Core::NativeDraw::ExecDraw& draw) {
    std::unique_ptr<DrawPlan> plan;
    {
        std::scoped_lock lk{plan_pool_mutex};
        if (!draw_plan_pool.empty()) {
            plan = std::move(draw_plan_pool.back());
            draw_plan_pool.pop_back();
        }
    }
    if (!plan) {
        plan = std::make_unique<DrawPlan>();
    }
    plan->pipeline = nullptr;
    plan->executed = false;
    plan->op_class = static_cast<u8>(DrawClass::EliminateFastClear);
    plan->depth_copy = plan->stencil_copy = false;
    plan->inputs = DrawInputs{};
    ApplyExecRegisters(plan->inputs, draw, 0, draw.count);
    return plan;
}

std::string Rasterizer::DiffExecPlans(const DrawPlan& typed, const DrawPlan& reference, u16 vertex_mask, u16 pixel_mask, u32 draw_kind) {
    std::string out;
    if (typed.op_class != 0xff || reference.op_class != 0xff) {
        if (typed.op_class != reference.op_class) {
            out += fmt::format("op_class({} vs {}) ", typed.op_class, reference.op_class);
        }
        if (std::memcmp(&typed.inputs.color_buffers[0], &reference.inputs.color_buffers[0], sizeof(AmdGpu::ColorBuffer)) != 0) {
            out += "color_buffer0 ";
        }
        return out;
    }
    if (typed.pipeline != reference.pipeline) {
        out += "pipeline ";
    }
    // what the draw does not use may differ: colour targets the pixel shader does not write, the depth target when none is bound, the index buffer of a draw without one
    DrawInputs used = typed.inputs;
    const u32 mrt_mask = reference.pipeline ? reference.pipeline->GetGraphicsKey().mrt_mask : 0;
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        if (((mrt_mask >> i) & 1) == 0 || reference.fixed.target_write_mask[i] == 0) {
            used.color_buffers[i] = reference.inputs.color_buffers[i];
        }
    }
    if (!reference.fixed.depth_target && !reference.fixed.stencil_target) {
        used.depth_buffer = reference.inputs.depth_buffer;
        used.depth_view = reference.inputs.depth_view;
        used.depth_control = reference.inputs.depth_control;
        used.depth_htile_data_base = reference.inputs.depth_htile_data_base;
        used.depth_render_control = reference.inputs.depth_render_control;
        used.depth_clear = reference.inputs.depth_clear;
        used.stencil_clear = reference.inputs.stencil_clear;
    }
    if (draw_kind == 2) {
        used.num_indices = reference.inputs.num_indices; // an indirect draw reads the count from its argument buffer
    }
    if (draw_kind == 1) {
        used.index_base_address = reference.inputs.index_base_address;
        used.index_buffer_type = reference.inputs.index_buffer_type;
        used.max_index_size = reference.inputs.max_index_size;
        used.index_offset = reference.inputs.index_offset;
    }
    out += DiffDrawInputs(used, reference.inputs);
    out += DiffFixedState(typed.fixed, reference.fixed);
    for (u32 s = 0; s < MaxShaderStages; ++s) {
        const auto& a = typed.stages[s];
        const auto& b = reference.stages[s];
        if (a.info != b.info || a.pgm_base != b.pgm_base || a.flat.size() != b.flat.size() || a.user_data_count != b.user_data_count) {
            out += fmt::format("stage{} ", s);
            continue;
        }
        const u16 mask = s == static_cast<u32>(Shader::SwStage::Fragment) ? pixel_mask : vertex_mask;
        bool differs = false;
        for (u32 i = 0; i < a.user_data_count; ++i) {
            differs |= ((mask >> i) & 1) != 0 && a.user_data[i] != b.user_data[i];
        }
        for (size_t i = 0; i < a.flat.size(); ++i) {
            differs |= (i >= 16 || ((mask >> i) & 1) != 0) && a.flat[i] != b.flat[i];
        }
        if (differs) {
            out += fmt::format("stage{}_data ", s);
        }
    }
    return out;
}

const DrawInputs& Rasterizer::InputsFor(const DrawPlan* plan, DrawInputs& storage) const {
    if (plan == nullptr) {
        storage = MakeDrawInputs(liverpool->RasterRegs());
        return storage;
    }
    return plan->inputs;
}

const FixedState& Rasterizer::FixedFor(const DrawPlan* plan, FixedState& storage) const {
    if (plan) {
        return plan->fixed;
    }
    storage = MakeFixedState(liverpool->RasterRegs());
    return storage;
}

namespace {
void ResetStage(StagePlan& stage) {
    stage.info = nullptr;
    stage.ring.clear();
    stage.user_data_count = 0;
    stage.flat.clear();
    stage.pgm_base = 0;
}
} // namespace

void Rasterizer::RecyclePlan(std::unique_ptr<DrawPlan>&& plan) {
    if (!plan) {
        return;
    }
    plan->pipeline = nullptr;
    plan->executed = false;
    plan->ring_end = 0;
    for (auto& stage : plan->stages) {
        ResetStage(stage);
    }
    std::scoped_lock lk{plan_pool_mutex};
    if (draw_plan_pool.size() < 512) {
        draw_plan_pool.push_back(std::move(plan));
    }
}

void Rasterizer::RecyclePlan(std::unique_ptr<DispatchPlan>&& plan) {
    if (!plan) {
        return;
    }
    plan->pipeline = nullptr;
    plan->ring_end = 0;
    ResetStage(plan->stage);
    std::scoped_lock lk{plan_pool_mutex};
    if (dispatch_plan_pool.size() < 512) {
        dispatch_plan_pool.push_back(std::move(plan));
    }
}

std::unique_ptr<DispatchPlan> Rasterizer::PlanDispatch(const AmdGpu::ComputeProgram& cs, const AmdGpu::BorderColorBuffer& ta_bc_base) {
    std::unique_ptr<DispatchPlan> plan;
    {
        std::scoped_lock lk{plan_pool_mutex};
        if (!dispatch_plan_pool.empty()) {
            plan = std::move(dispatch_plan_pool.back());
            dispatch_plan_pool.pop_back();
        }
    }
    if (!plan) {
        plan = std::make_unique<DispatchPlan>();
    }
    pipeline_cache.PlanCompute(cs, *plan);
    plan->cs = cs;
    plan->ta_bc_base = ta_bc_base;
    return plan;
}

const GraphicsPipeline* Rasterizer::ResolveDrawPipeline(const DrawPlan* plan,
                                                        DrawIndirectParams params) {
    if (plan) {
        PipelineCache::ApplyPlan(*plan);
        return plan->pipeline;
    }
    if (liverpool->SplitActive()) {
        // The command processor filtered this draw but the raster thread did not: the register
        // mirror is out of sync. Skip it rather than touch the command processor's state.
        LOG_ERROR(Render_Vulkan, "CP split: draw without a pipeline plan skipped");
        return nullptr;
    }
    return pipeline_cache.GetGraphicsPipeline(params);
}

const ComputePipeline* Rasterizer::ResolveDispatchPipeline(const DispatchPlan* plan) {
    if (plan) {
        PipelineCache::ApplyPlan(*plan);
        return plan->pipeline;
    }
    if (liverpool->SplitActive()) {
        LOG_ERROR(Render_Vulkan, "CP split: dispatch without a pipeline plan skipped");
        return nullptr;
    }
    return pipeline_cache.GetComputePipeline();
}

template <typename... Ts>
bool Rasterizer::TargetInputsMatch(CachedTarget& cache, const Ts&... parts) {
    std::array<u8, 256> inputs;
    u32 size = 0;
    static_assert((sizeof(Ts) + ...) <= inputs.size());
    ((std::memcpy(inputs.data() + size, &parts, sizeof(parts)), size += sizeof(parts)), ...);
    const bool match = cache.valid && cache.size == size &&
                       cache.generation == texture_cache.ImageGeneration() &&
                       std::memcmp(cache.inputs.data(), inputs.data(), size) == 0;
    if (!match) {
        // The caller does a full lookup and marks the cache valid afterwards.
        cache.valid = false;
        cache.size = size;
        std::memcpy(cache.inputs.data(), inputs.data(), size);
    }
    return match;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline, const FixedState& fixed) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = *draw_inputs;
    if (fixed.degamma) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding = !fixed.color_enabled;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = fixed.target_write_mask[cb];
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            rt_cache[cb].valid = false;
            continue;
        }
        const auto& hint = (*current_cb_extent)[cb];
        if (RtReuseOn() && TargetInputsMatch(rt_cache[cb], col_buf, hint)) {
            // Same registers and no image registered or removed since: FindImage would find the
            // same image again (image_id and desc still hold its result).
            bound_images.emplace_back(image_id);
            texture_cache.TouchFoundImage(image_id);
        } else {
            std::construct_at(&desc, col_buf, hint);
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            if (RtReuseOn()) {
                rt_cache[cb].valid = true;
                rt_cache[cb].generation = texture_cache.ImageGeneration();
            }
        }
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = *current_db_extent;
        auto& [image_id, desc] = db_desc;
        auto& cache = rt_cache[AmdGpu::NUM_COLOR_BUFFERS];
        if (RtReuseOn() && TargetInputsMatch(cache, regs.depth_buffer, regs.depth_view,
                                                  regs.depth_control, htile_address, hint)) {
            bound_images.emplace_back(image_id);
            texture_cache.TouchFoundImage(image_id);
        } else {
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            if (RtReuseOn()) {
                cache.valid = true;
                cache.generation = texture_cache.ImageGeneration();
            }
        }
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
        rt_cache[AmdGpu::NUM_COLOR_BUFFERS].valid = false;
    }
}

static std::pair<u32, u32> GetDrawOffsets(const DrawInputs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0) {
    const auto& col_buf = inputs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, cb0);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

void Rasterizer::ClearDepthStencilTarget(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent) {
    scheduler.PopPendingOperations();
    ClearDepthTarget(inputs, db_extent);
}

void Rasterizer::ClearDepthTarget(const DrawInputs& regs, const AmdGpu::CbDbExtent& db_extent) {
    const auto htile_address = regs.depth_htile_data_base.GetAddress();
    VideoCore::TextureCache::ImageDesc desc(regs.depth_buffer, regs.depth_view, regs.depth_control, htile_address,
                                            db_extent);
    const auto image_id = texture_cache.FindImage(desc);
    auto& image = texture_cache.GetImage(image_id);
    const bool clear_depth = regs.depth_render_control.depth_clear_enable && regs.depth_buffer.DepthValid();
    const bool clear_stencil = regs.depth_render_control.stencil_clear_enable && regs.depth_buffer.StencilValid();
    texture_cache.TouchMeta(htile_address, desc.view_info.range.base.layer, false);
    ScopeMarkerBegin(fmt::format("ClearDepthTarget:{:#x}", regs.depth_buffer.DepthAddress()));
    runtime.ClearDepthStencilImage(&image, desc.view_info.range, clear_depth, regs.depth_clear, clear_stencil,
                                   regs.stencil_clear);
    image.usage.depth_target = true;
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const DrawPlan* plan) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    // Native draw path: the game said (through the tag in the command stream) which state groups this draw
    // shares with the previous one. Nothing is reused unless the previous executed operation was a tagged draw
    // in the same command buffer.
    const VideoCore::NativeTag tag = std::exchange(native_tag, {});
    const bool use_tag =
        tag.valid &&
        VideoCore::ReadbackPolicy::Get().native_draw_mode.load(std::memory_order_relaxed) >= 2;
    const NativeLast last = native_last;
    native_last.valid = false;
    if (tag.valid) {
        ++native_stats.tagged;
    } else {
        ++native_stats.untagged;
    }

    if (!FilterDraw(plan)) {
        return;
    }

    const GraphicsPipeline* pipeline = ResolveDrawPipeline(plan);
    if (!pipeline) {
        return;
    }
    FixedState fixed_storage;
    const FixedState& fixed = FixedFor(plan, fixed_storage);
    DrawInputs inputs_storage;
    draw_inputs = &InputsFor(plan, inputs_storage);
    current_bc_base = &draw_inputs->ta_bc_base;
    current_cb_extent = plan ? &plan->cb_extent : &liverpool->RasterCbExtents();
    current_db_extent = plan ? &plan->db_extent : &liverpool->RasterDbExtent();
    const auto& regs = *draw_inputs;
    const bool same_cmdbuf = use_tag && last.valid && last.tick == scheduler.CurrentTick();
    const bool same_pipeline = same_cmdbuf && last.pipeline == pipeline;
    const bool same_dynamic_state = same_pipeline && last.k2 == tag.k2;

    PrepareRenderState(pipeline, fixed);
    if (!BindResources(pipeline, &fixed)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    if (same_dynamic_state) {
        ++native_stats.dynamic_skipped;
    } else {
        UpdateDynamicState(pipeline, is_indexed, fixed);
    }
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    if (same_pipeline) {
        ++native_stats.bind_skipped;
    } else {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
    }

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    ++recorded_work;

    ResetBindings(false);
    if (use_tag) {
        native_last = {
            .valid = true, .k2 = tag.k2, .pipeline = pipeline, .tick = scheduler.CurrentTick()};
    }
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset, const DrawPlan* plan) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw(plan)) {
        return;
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
        .is_indirect = true,
    };
    const GraphicsPipeline* pipeline = ResolveDrawPipeline(plan, params);
    if (!pipeline) {
        return;
    }
    FixedState fixed_storage;
    const FixedState& fixed = FixedFor(plan, fixed_storage);
    DrawInputs inputs_storage;
    draw_inputs = &InputsFor(plan, inputs_storage);
    current_bc_base = &draw_inputs->ta_bc_base;
    current_cb_extent = plan ? &plan->cb_extent : &liverpool->RasterCbExtents();
    current_db_extent = plan ? &plan->db_extent : &liverpool->RasterDbExtent();

    PrepareRenderState(pipeline, fixed);
    if (!BindResources(pipeline, &fixed)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(0, true);
    }

    const auto size = stride * max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(arg_address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4,
                                   vk::AccessFlagBits2::eIndirectCommandRead);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed, fixed);
    scheduler.BeginRendering(state);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
        ++recorded_work;
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_offset,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
        ++recorded_work;
    }

    ResetBindings(false);
}

void Rasterizer::SetExplicitImageReadback(bool on) {
    texture_cache.SetExplicitDownload(on);
}

void Rasterizer::DispatchDirect(const DispatchPlan* plan) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (const auto fill = std::exchange(pending_fill, {});
        fill.dwords != 0 && TryClearImage(fill.dst, fill.bytes, fill.pattern.data(), fill.dwords)) {
        return;
    }

    current_cs = plan ? &plan->cs : &liverpool->GetCsRegs();
    current_bc_base = plan ? &plan->ta_bc_base : &liverpool->RasterRegs().ta_bc_base;
    const auto& cs_program = *current_cs;
    const ComputePipeline* pipeline = ResolveDispatchPipeline(plan);
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (ExecuteShaderHLE(cs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.StampDispatch(true);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    scheduler.StampDispatch(false);
    DebugState.IncDispatch();
    ++recorded_work;

    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size, const DispatchPlan* plan) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    current_cs = plan ? &plan->cs : &liverpool->GetCsRegs();
    current_bc_base = plan ? &plan->ta_bc_base : &liverpool->RasterRegs().ta_bc_base;
    const ComputePipeline* pipeline = ResolveDispatchPipeline(plan);
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();
    ++recorded_work;

    ResetBindings(true);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (encoder_debug & 2) {
        scheduler.SyncEncoder();
    }
    buffer_cache.TickFrame();
    ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::ProcessDownloadImages() {
    if (async_fences) {
        // End of a submission, no guest fence attached: still avoid waiting on the GPU.
        OnFence({}, VideoCore::FenceSubmitEnd);
        return;
    }
    ProcessReadbacksSync();
}

void Rasterizer::ProcessReadbacksSync() {
    if (!readback_batching && !async_fences) {
        texture_cache.ProcessDownloadImages();
        return;
    }
    // Called right before the guest is told the GPU finished (EOP/EOS fences, end of submit).
    // Pending image readbacks and recently read back buffers that the GPU wrote again are
    // downloaded here with a single GPU wait, so the CPU finds the data ready instead of
    // faulting and waiting once per buffer or image.
    constexpr u64 MaxBatchBytes = 256_MB;
    VideoCore::ReadbackBatch batch{scheduler, MaxBatchBytes};
    texture_cache.RecordImageDownloads(batch);
    buffer_cache.ApplyPendingUnmarks();
}

void Rasterizer::OnFence(Common::UniqueFunction<void>&& signal, VideoCore::FenceKind kind,
                         bool must_sync) {
    if (encoder_debug & 1) {
        scheduler.SyncEncoder();
    }
    if (!async_fences || must_sync) {
        // Synchronous path: readbacks complete (and earlier async fences drain, via Finish)
        // before the fence is signaled from the GPU thread.
        ProcessReadbacksSync();
        if (async_fences && scheduler.HasPriorityOperations()) {
            scheduler.Finish();
        }
        if (signal) {
            signal();
        }
        return;
    }

    // Asynchronous path: record the copies, submit them and return without waiting. The
    // priority thread waits for the GPU, writes the data to guest memory and only then signals
    // the fence, so the guest still never sees the fence before the data. Fences are signaled
    // in order because the priority thread runs operations in the order they were deferred.
    constexpr u64 MaxBatchBytes = 256_MB;
    VideoCore::ReadbackBatch batch{scheduler, MaxBatchBytes};
    texture_cache.RecordImageDownloads(batch, kind);
    buffer_cache.ApplyPendingUnmarks();
    const bool has_copies = !batch.Empty();
    // Async image copies are written to the guest by the priority thread once the GPU runs them;
    // the fence does not wait for them but their command buffer must reach the GPU soon.
    const bool async_images = texture_cache.TakeAsyncRecorded() | buffer_cache.TakeAsyncRecorded();
    if (!has_copies && !scheduler.HasPriorityOperations()) {
        // Nothing to read back and nothing in flight: keep the old immediate behavior.
        if (async_images) {
            scheduler.Flush();
        }
        if (signal) {
            signal();
        }
        return;
    }
    // Without copies there is nothing new to wait for, only earlier fences to stay behind.
    // With the encoder thread the fence also waits for the work recorded before it in this
    // command buffer: its submission can lag, and signaling on the previous command buffer let
    // the game see the fence long before that work ran (flickering feathers). Also forced by
    // cp_encoder_debug bit 4. cp_full_fence_kinds limits it to some fence kinds.
    const bool full_eop =
        full_eop_fences && (VideoCore::ReadbackPolicy::Get().full_fence_kinds.load() & kind) != 0;
    const u64 tick = has_copies || full_eop ? scheduler.CurrentTick() : scheduler.CurrentTick() - 1;
    // Submit when this or an earlier operation waits on the current command buffer: the guest
    // may block on this fence without submitting anything else. Otherwise every tick waited on
    // is submitted already and submitting would only cost time.
    const bool needs_submit =
        has_copies || full_eop || async_images || scheduler.PriorityOperationsNeedSubmit();
    scheduler.DeferPriorityOperation(
        [callbacks = batch.TakeCallbacks(), signal = std::move(signal)]() mutable {
            for (auto& callback : callbacks) {
                callback();
            }
            if (signal) {
                signal();
            }
        },
        tick);
    if (needs_submit) {
        scheduler.Flush();
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline, const FixedState* fixed) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(fixed);
    clip_plane_mask = fixed ? fixed->user_clip_plane_mask : 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const auto& regs = *draw_inputs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        if (range.GetSize() >= (1ull << 30) && !memory->IsValidMapping(range.base_address)) {
            LOG_ERROR(Render, "vertex buffer range at unmapped address {:#x} of {} bytes: ignored", range.base_address, range.GetSize());
            range.buffer = nullptr;
            continue;
        }
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        bound_buffers.emplace_back(range.buffer, range.offset, size,
                                   vk::AccessFlagBits2::eVertexAttributeRead);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::BindIndexBuffer(u32 index_offset, bool is_indirect) {
    const auto& regs = *draw_inputs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr base_address = regs.index_base_address.Address<VAddr>();
    const VAddr index_address = base_address + u64(index_offset) * index_size;

    // Bind index buffer.
    const u32 num_indices =
        is_indirect ? regs.max_index_size : std::min(regs.num_indices, regs.max_index_size);
    const u64 requested_size = u64(num_indices) * index_size;
    if (base_address != 0 && requested_size >= (1ull << 30) && !memory->IsValidMapping(index_address)) {
        LOG_ERROR(Render, "index buffer at unmapped address {:#x}: {} indices of {} bytes (indirect {}, num_indices {}, max_index_size {}): ignored",
                  index_address, num_indices, index_size, is_indirect, regs.num_indices, regs.max_index_size);
    }
    const u64 mapped_size = base_address != 0 && (requested_size < (1ull << 30) || memory->IsValidMapping(index_address))
                                ? memory->ClampRangeSize(index_address, requested_size)
                                : 0;
    u32 index_buffer_size = static_cast<u32>(Common::AlignDown(mapped_size, index_size));
    vk::Buffer index_buffer{};
    u64 buffer_offset{};
    if (index_buffer_size != 0) {
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
        bound_buffers.emplace_back(buffer, offset, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = buffer->Handle();
        buffer_offset = offset;
    } else if (!instance.IsMaintenance6Supported()) {
        const auto* null_buffer = buffer_cache.GetNullIndexBuffer();
        index_buffer_size = static_cast<u32>(null_buffer->SizeBytes());
        needs_barrier |= runtime.IsBufferAccessed(null_buffer, 0, index_buffer_size);
        bound_buffers.emplace_back(null_buffer, 0, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = null_buffer->Handle();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsMaintenance5Supported()) {
        cmdbuf.bindIndexBuffer2KHR(index_buffer, buffer_offset, index_buffer_size, index_type);
    } else {
        cmdbuf.bindIndexBuffer(index_buffer, buffer_offset, index_type);
    }
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, src_access] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = *current_cs;
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = *current_cs;
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        const RingBinding* ring =
            const_ring_level >= 4 && (ring_draw || ring_dispatch) ? FindRingBinding(stage, buffer_index) : nullptr;
        if (ring) {
            // Copied by the command processor thread into the constant ring (read-only, small, not written by queued work).
            if (!desc.IsSpecial()) {
                push_data.AddOffset(binding.buffer, 0);
            }
            buffer_infos.emplace_back(constant_ring->Handle(), ring->offset, ring->size);
            ring_bound.fetch_add(1, std::memory_order_relaxed);
            if (ring_verify_interval != 0 && ring->address != 0 && ++ring_verify_counter % ring_verify_interval == 0) {
                std::vector<u8> now(ring->size);
                memory->CopySparseMemory(ring->address, now.data(), ring->size);
                ring_verified.fetch_add(1, std::memory_order_relaxed);
                if (std::memcmp(now.data(), constant_ring->Data(ring->offset), ring->size) != 0) {
                    ring_differing.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } else if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                Core::ResourceAudit::Note(Core::ResourceAudit::BufferUse, 0xfffff0000000ull, gds_buf->SizeBytes(), desc.is_written);
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (clip_plane_mask == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = draw_inputs->clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = *current_cs;
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else if (vsharp.GetSize() >= (1ull << 30) && !memory->IsValidMapping(vsharp.base_address)) {
                LOG_ERROR(Render, "buffer sharp at unmapped address {:#x} of {} bytes for stage {:#x}: bound as null", vsharp.base_address,
                          vsharp.GetSize(), stage.pgm_hash);
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, desc.is_written, desc.is_formatted);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, src_access);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    if (size >= (8u << 20)) {
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1) < 6) {
                            LOG_INFO(Render_Vulkan, "stage {:#x} (user data {}) writes {} bytes at {:#x} through a storage buffer, caller {:#x}",
                                     stage.pgm_hash, stage.user_data.size(), size, vsharp.base_address,
                                     dispatch_caller ? dispatch_caller - Core::NativeDraw::ImageBase() + 0x1000000 : 0);
                        }
                    }
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size, "a written storage buffer view");
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

namespace {
u64 ImageUseFlags(const Shader::ImageResource& image_desc) {
    return u64(image_desc.is_depth) | u64(image_desc.is_atomic) << 8 | u64(image_desc.is_array) << 16 | u64(image_desc.is_written) << 24 |
           u64(image_desc.is_r128) << 32 | u64(image_desc.constant_mip_index) << 40 | u64(static_cast<u8>(image_desc.mip_fallback_mode)) << 48 |
           u64(static_cast<u8>(image_desc.post_op)) << 56;
}
} // namespace

bool Rasterizer::BindTexturesFromSet(const Shader::Info& stage, TextureSet& set, const AmdGpu::Image* raws, const u64* flags, u32 count,
                                     Shader::Backend::Bindings& binding) {
    if (set.count != count) {
        ++texture_set_stats.no_entry;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (set.entries[i].flags != flags[i] || std::memcmp(&set.entries[i].raw, &raws[i], sizeof(AmdGpu::Image)) != 0) {
            ++texture_set_stats.no_entry;
            return false;
        }
    }
    if (set.generation != texture_cache.ImageGeneration()) {
        ++texture_set_stats.generation;
        return false;
    }
    // Everything the slow path does to an image before it looks at the view: touch and refresh. Idempotent, so a miss after it costs nothing.
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (entry.null_entry) {
            continue;
        }
        texture_cache.TouchFoundImage(entry.found);
        texture_cache.UpdateImage(entry.id);
    }
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (entry.null_entry) {
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        const bool redirected = entry.id != entry.found;
        const auto associated = texture_cache.GetAssociatedDepth(texture_cache.GetImage(entry.found));
        if (image.backing != entry.backing || image.binding.needs_rebind || image.binding.force_general || image.binding.is_target ||
            (image.backing->num_samples > 1) != (image.info.num_samples > 1) || (redirected ? associated != entry.id : bool(associated))) {
            ++texture_set_stats.image_check;
            return false;
        }
    }
    ++texture_set_stats.hits;
    const u32 first_image_idx = image_infos.size();
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (entry.null_entry) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        const auto new_layout = image.info.props.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, entry.range);
        image.usage.texture |= 1u;
        image_infos.emplace_back(VK_NULL_HANDLE, entry.view, image.backing->state.layout);
    }
    for (u32 i = 0; i < count; ++i) {
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[first_image_idx + i];
    }
    return true;
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    // Texture set memo: the set of resolved images for these T#s, when the stage qualifies (no written image, no mip fallback binding, at most 16 images).
    TextureSet* set_slot = nullptr;
    std::array<AmdGpu::Image, TextureSet::MaxImages> set_raws;
    std::array<u64, TextureSet::MaxImages> set_flags{};
    std::array<VideoCore::ImageId, TextureSet::MaxImages> set_found{};
    bool set_record = false;
    u64 set_generation = 0;
    const u32 stage_images = static_cast<u32>(stage.images.size());
    if (texture_set_on && stage_images != 0 && stage_images <= TextureSet::MaxImages) {
        bool eligible = true;
        for (const auto& image_desc : stage.images) {
            eligible &= !image_desc.is_written && image_desc.mip_fallback_mode == Shader::MipStorageFallbackMode::None;
        }
        if (eligible) {
            u64 key = 0x9E3779B97F4A7C15ull;
            const auto flat = stage.FlatUserData();
            for (u32 i = 0; i < stage_images; ++i) {
                AmdGpu::Image raw{};
                if (!stage.images[i].Fetch(flat.data(), &raw)) {
                    raw = {};
                }
                set_raws[i] = raw;
                set_flags[i] = ImageUseFlags(stage.images[i]);
                std::array<u64, 5> words{};
                std::memcpy(words.data(), &raw, sizeof(AmdGpu::Image));
                words[4] = set_flags[i];
                for (const u64 word : words) {
                    key = (key ^ word) * 0xC2B2AE3D27D4EB4Full;
                    key ^= key >> 29;
                }
            }
            set_slot = &texture_sets[key % TextureSetSlots];
            if (BindTexturesFromSet(stage, *set_slot, set_raws.data(), set_flags.data(), stage_images, binding)) {
                BindSamplers(stage, binding);
                return;
            }
            set_record = true;
            set_slot->count = 0; // the entries are rewritten below: no hit on a half written set
            set_slot->generation = ~0ull;
            set_generation = texture_cache.ImageGeneration();
        } else {
            ++texture_set_stats.not_eligible;
        }
    }

    u32 num_images{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            set_record = false;
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            // Same T# and use as an earlier lookup, no image registered or removed since: the description and the
            // image FindImage returned are the same again (native_draw_mode >= 2).
            std::array<u64, 6> key{};
            static_assert(sizeof(AmdGpu::Image) <= 32);
            std::memcpy(key.data(), &tsharp, sizeof(AmdGpu::Image));
            key[4] = u64(image_desc.is_depth) | u64(image_desc.is_atomic) << 8 |
                     u64(image_desc.is_array) << 16 | u64(image_desc.is_written) << 24 |
                     u64(image_desc.is_r128) << 32 | u64(image_desc.constant_mip_index) << 40 |
                     u64(static_cast<u8>(image_desc.mip_fallback_mode)) << 48 |
                     u64(static_cast<u8>(image_desc.post_op)) << 56;
            key[5] = u64(i);
            u64 hash = 0;
            for (const u64 word : key) {
                hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
                hash ^= hash >> 29;
            }
            auto& cached = texture_lookup_cache[hash & 511];
            const bool native_reuse =
                VideoCore::ReadbackPolicy::Get().native_draw_mode.load(std::memory_order_relaxed) >=
                2;
            if (native_reuse && cached.valid && cached.key == key &&
                cached.generation == texture_cache.ImageGeneration()) {
                std::construct_at(&desc, cached.desc);
                image_id = cached.image_id;
                texture_cache.TouchFoundImage(image_id);
            } else {
                std::construct_at(&desc, tsharp, image_desc);

                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    desc.view_info.range.base.level += image_desc.constant_mip_index;
                    desc.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    desc.view_info.range.base.level += i;
                    desc.view_info.range.extent.levels = 1;
                }

                image_id = texture_cache.FindImage(desc);
                if (native_reuse) {
                    cached.valid = true;
                    cached.generation = texture_cache.ImageGeneration();
                    cached.image_id = image_id;
                    cached.key = key;
                    cached.desc = desc;
                }
            }
            if (set_record) {
                set_found[num_images - 1] = image_id;
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
                set_record = false;
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (set_record && i < TextureSet::MaxImages) {
                if (binding.force_general || binding.is_target || is_storage) {
                    set_record = false;
                } else {
                    auto& entry = set_slot->entries[i];
                    entry.id = image_id;
                    entry.found = set_found[i];
                    entry.view = *image_view.image_view;
                    entry.backing = image.backing;
                    entry.range = desc.view_info.range;
                    entry.null_entry = false;
                }
            }
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    if (set_record && num_images == stage_images && set_generation == texture_cache.ImageGeneration()) {
        // Complete the entries: null images and the raw T# and flags of every image (the others were written in the second pass).
        for (u32 i = 0; i < num_images; ++i) {
            auto& entry = set_slot->entries[i];
            if (!image_bindings[i].image_id) {
                entry = {};
                entry.null_entry = true;
            }
            entry.raw = set_raws[i];
            entry.flags = set_flags[i];
        }
        set_slot->count = num_images;
        set_slot->generation = set_generation;
        ++texture_set_stats.recorded;
    }
    BindSamplers(stage, binding);
}

void Rasterizer::BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                current_bc_base->Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        current_bc_base->Address());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, *current_bc_base, sampler.is_depth);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

Rasterizer::BeginSignature Rasterizer::MakeBeginSignature(const GraphicsPipeline* pipeline) const {
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = *draw_inputs;
    BeginSignature sig;
    const u32 num_color = std::bit_width(key.mrt_mask);
    for (u32 cb = 0; cb < num_color && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        sig.ids[cb] = cb_descs[cb].image_id;
        if (cb_descs[cb].image_id) {
            sig.views[cb] = cb_descs[cb].desc.view_info;
        }
        sig.color_samples[cb] = key.color_samples[cb];
    }
    sig.ids[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.image_id;
    if (db_desc.image_id) {
        sig.views[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.desc.view_info;
    }
    sig.mrt_mask = key.mrt_mask;
    sig.num_samples = key.num_samples;
    sig.generation = texture_cache.ImageGeneration();
    std::memcpy(&sig.depth_control, &regs.depth_control, sizeof(u32));
    sig.depth_valid = regs.depth_buffer.DepthValid();
    sig.stencil_valid = regs.depth_buffer.StencilValid();
    return sig;
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    static const bool memo_enabled = [] {
        // Off by default until it is checked in play (documents/tlg/pc-port-draw-pipeline-architecture.md): TLG_RENDER_STATE_MEMO=1 turns it on.
        const char* env = std::getenv("TLG_RENDER_STATE_MEMO");
        return env && env[0] == '1';
    }();
    if (!memo_enabled) {
        return BeginRenderingFull(pipeline);
    }
    const auto& regs = *draw_inputs;
    BeginSignature signature = MakeBeginSignature(pipeline);
    bool usable = begin_memo.valid && scheduler.IsRenderingWith(begin_memo.state) &&
                  !regs.depth_render_control.depth_clear_enable &&
                  !regs.depth_render_control.stencil_clear_enable && signature == begin_memo.signature;
    for (const auto id : signature.ids) {
        if (usable && id) {
            const auto& image = texture_cache.GetImage(id);
            usable = !image.binding.is_bound && !image.binding.needs_rebind;
        }
    }
    if (usable) {
        ++begin_memo_hits;
        attachment_feedback_loop = false;
        return begin_memo.state;
    }
    ++begin_memo_misses;
    begin_memo.valid = false;
    RenderState state = BeginRenderingFull(pipeline);
    if (!attachment_feedback_loop) {
        // A state with a clear is not remembered: the next draw's differs (the clear is done).
        bool clears = state.depth_stencil_attachment.depth_clear || state.depth_stencil_attachment.stencil_clear;
        for (u32 i = 0; i < state.num_color_attachments; ++i) {
            clears |= state.color_attachments[i].is_clear != 0;
        }
        if (!clears) {
            begin_memo = {true, MakeBeginSignature(pipeline), state};
        }
    }
    return state;
}

RenderState Rasterizer::BeginRenderingFull(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = *draw_inputs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            rt_cache[cb].valid = false;
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.image_id; image_id) {
        auto& desc = db_desc.desc;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

namespace {
/// What a buffer resource needs from a stage to read its V#: the flattened user data and the program base, here those of a plan (the stage's Info is rewritten
/// by the next plan while the raster thread may still be using it).
struct PlanStageView {
    std::span<const u32> flat;
    VAddr base;
    std::span<const u32> FlatUserData() const noexcept {
        return flat;
    }
    VAddr ProgramBase() const noexcept {
        return base;
    }
};
} // namespace

void Rasterizer::NotePendingGpuWrite(VAddr address, u64 size, u64 position) {
    pending_gpu_writes.push_back({address, address + size, position});
    pending_min = std::min(pending_min, address);
    pending_max = std::max(pending_max, address + size);
}

bool Rasterizer::PendingGpuWriteOverlaps(VAddr address, u64 size, u64 done) {
    if (pending_gpu_writes.empty() || address >= pending_max || address + size <= pending_min) {
        return false;
    }
    // Now and then drop the writes the raster thread has done: the bounds shrink again.
    if ((++pending_checks & 63) == 0) {
        std::erase_if(pending_gpu_writes, [&](const PendingWrite& write) { return write.position <= done; });
        pending_min = ~VAddr{0};
        pending_max = 0;
        for (const auto& write : pending_gpu_writes) {
            pending_min = std::min(pending_min, write.begin);
            pending_max = std::max(pending_max, write.end);
        }
    }
    const VAddr end = address + size;
    for (const auto& write : pending_gpu_writes) {
        if (address < write.end && write.begin < end && write.position > done) {
            return true;
        }
    }
    return false;
}

void Rasterizer::FrontStageNotes(StagePlan& stage, u64 position) {
    stage.ring.clear();
    if (!stage.info) {
        return;
    }
    const PlanStageView view{std::span<const u32>{stage.flat.data(), stage.flat.size()}, stage.pgm_base};
    for (const auto& desc : stage.info->buffers) {
        if (desc.IsSpecial() || !desc.is_written) {
            continue;
        }
        const auto vsharp = desc.GetSharp(view);
        if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
            continue;
        }
        NotePendingGpuWrite(vsharp.base_address, vsharp.GetSize(), position);
        ring_written_buffers.fetch_add(1, std::memory_order_relaxed);
    }
}

void Rasterizer::FrontStageRing(StagePlan& stage, u64 done) {
    if (!stage.info) {
        return;
    }
    const PlanStageView view{std::span<const u32>{stage.flat.data(), stage.flat.size()}, stage.pgm_base};
    const u64 alignment = std::max<u64>(instance.StorageMinAlignment(), instance.UniformMinAlignment());
    const auto& buffers = stage.info->buffers;
    for (u32 index = 0; index < buffers.size(); ++index) {
        const auto& desc = buffers[index];
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf && !stage.flat.empty()) {
                ring_offered.fetch_add(1, std::memory_order_relaxed);
                const u64 size = stage.flat.size() * sizeof(u32);
                if (const auto offset = constant_ring->Allocate(size, alignment)) {
                    std::memcpy(constant_ring->Data(*offset), stage.flat.data(), size);
                    constant_ring->Flush(*offset, size);
                    stage.ring.push_back({index, static_cast<u32>(size), *offset, 0});
                    ring_taken.fetch_add(1, std::memory_order_relaxed);
                    ring_bytes.fetch_add(size, std::memory_order_relaxed);
                } else {
                    ring_full.fetch_add(1, std::memory_order_relaxed);
                }
            }
            continue;
        }
        ring_offered.fetch_add(1, std::memory_order_relaxed);
        if (desc.is_written) {
            ring_written.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (desc.is_formatted) {
            ring_formatted.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const auto vsharp = desc.GetSharp(view);
        const VAddr address = vsharp.base_address;
        const u64 size = vsharp.GetSize();
        if (address == 0 || size == 0) {
            ring_no_sharp.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (size > VideoCore::BufferCache::StreamThreshold()) {
            ring_too_large.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        // An immutable engine buffer is read straight from its arena (no copy at all): the ring would make it dearer.
        if (Core::NativeResources::ImmutableResidencyEnabled() && Core::NativeResources::FindImmutable(address, size)) {
            ring_immutable.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (buffer_cache.IsRegionGpuModified(address, size)) {
            ring_gpu_modified.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (PendingGpuWriteOverlaps(address, size, done)) {
            ring_pending_write.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const auto offset = constant_ring->Allocate(size, alignment);
        if (!offset) {
            ring_full.fetch_add(1, std::memory_order_relaxed);
            continue; // never wait: the binding stays on the raster thread
        }
        memory->CopySparseMemory(address, constant_ring->Data(*offset), size);
        constant_ring->Flush(*offset, size);
        stage.ring.push_back({index, static_cast<u32>(size), *offset, address});
        ring_taken.fetch_add(1, std::memory_order_relaxed);
        ring_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

void Rasterizer::FrontQueueOp(DrawPlan* draw, DispatchPlan* dispatch, u64 position, u64 done) {
    if (const_ring_level == 0 || (!draw && !dispatch)) {
        return;
    }
    ring_queued.fetch_add(1, std::memory_order_relaxed);
    // First everything the plan writes (a read-only binding of the same plan may alias a written one), then the copies.
    if (draw) {
        for (auto& stage : draw->stages) {
            FrontStageNotes(stage, position);
        }
    } else {
        FrontStageNotes(dispatch->stage, position);
    }
    if (const_ring_level < 3 || !constant_ring) {
        return;
    }
    if (draw) {
        for (auto& stage : draw->stages) {
            FrontStageRing(stage, done);
        }
        draw->ring_end = constant_ring->Position();
    } else {
        FrontStageRing(dispatch->stage, done);
        dispatch->ring_end = constant_ring->Position();
    }
}

const RingBinding* Rasterizer::FindRingBinding(const Shader::Info& stage, u32 index) const noexcept {
    const StagePlan* plan_stage = nullptr;
    if (ring_draw) {
        for (const auto& candidate : ring_draw->stages) {
            if (candidate.info == &stage) {
                plan_stage = &candidate;
                break;
            }
        }
    } else if (ring_dispatch && ring_dispatch->stage.info == &stage) {
        plan_stage = &ring_dispatch->stage;
    }
    if (!plan_stage) {
        return nullptr;
    }
    for (const auto& ring : plan_stage->ring) {
        if (ring.index == index) {
            return &ring;
        }
    }
    return nullptr;
}

void Rasterizer::EndRingUse() noexcept {
    if (constant_ring && (ring_draw || ring_dispatch)) {
        const u64 end = ring_draw ? ring_draw->ring_end : ring_dispatch->ring_end;
        if (end != 0) {
            constant_ring->Stamp(end);
        }
    }
    ring_draw = nullptr;
    ring_dispatch = nullptr;
}

Rasterizer::RingStats Rasterizer::TakeRingStats() noexcept {
    RingStats s;
    s.queued = ring_queued.exchange(0);
    s.written_buffers = ring_written_buffers.exchange(0);
    s.offered = ring_offered.exchange(0);
    s.taken = ring_taken.exchange(0);
    s.bytes = ring_bytes.exchange(0);
    s.no_sharp = ring_no_sharp.exchange(0);
    s.written = ring_written.exchange(0);
    s.formatted = ring_formatted.exchange(0);
    s.too_large = ring_too_large.exchange(0);
    s.immutable = ring_immutable.exchange(0);
    s.gpu_modified = ring_gpu_modified.exchange(0);
    s.pending_write = ring_pending_write.exchange(0);
    s.ring_full = ring_full.exchange(0);
    s.bound = ring_bound.exchange(0);
    s.verified = ring_verified.exchange(0);
    s.differing = ring_differing.exchange(0);
    return s;
}

void Rasterizer::Resolve(const DrawInputs& inputs, const AmdGpu::CbDbExtent& mrt0_hint, const AmdGpu::CbDbExtent& mrt1_hint) {
    VideoCore::TextureCache::ImageDesc mrt0_desc{inputs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{inputs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 inputs.color_buffers[0].Address(), inputs.color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(const DrawInputs& regs, const AmdGpu::CbDbExtent& db_extent, bool is_depth, bool is_stencil) {

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    const auto write_id = texture_cache.FindImage(write_desc);
    auto& write_image = texture_cache.GetImage(write_id);
    // The copy destination is never bound as a target, so its stencil plane is linked here.
    texture_cache.AssociateStencil(write_id, write_desc.info);

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = regs.depth_view.slice_start;
    sub_range.extent.layers = regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    Core::ResourceAudit::Note(is_gds ? Core::ResourceAudit::GdsFill : Core::ResourceAudit::MemoryCopyDst, address, num_bytes, true);
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            buffer_cache.NoteCpGuestWrite(address, num_bytes);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
    ++recorded_work;
}

namespace {
float HalfToFloat(u16 h) {
    const u32 sign = (h >> 15) & 1;
    const u32 exponent = (h >> 10) & 0x1f;
    const u32 mantissa = h & 0x3ff;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 0x400), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

float UnsignedFloat(u32 bits, u32 mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits;
    const u32 mantissa = bits & ((1u << mantissa_bits) - 1);
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits));
    }
    if (exponent == 31) {
        return mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    }
    return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits), static_cast<int>(exponent) - 15);
}

/// The clear colour of a texel (its bytes in memory order) in `format`; false for formats not handled.
bool DecodeTexel(vk::Format format, const u32* texel, vk::ClearColorValue& out) {
    const auto u8_at = [&](u32 i) { return static_cast<float>((texel[i / 4] >> ((i & 3) * 8)) & 0xff) / 255.0f; };
    const auto u16_at = [&](u32 i) { return static_cast<u16>(texel[i / 2] >> ((i & 1) * 16)); };
    switch (format) {
    case vk::Format::eR16G16B16A16Sfloat:
        out.float32 = {{HalfToFloat(u16_at(0)), HalfToFloat(u16_at(1)), HalfToFloat(u16_at(2)), HalfToFloat(u16_at(3))}};
        return true;
    case vk::Format::eR16G16B16A16Unorm:
        out.float32 = {{u16_at(0) / 65535.0f, u16_at(1) / 65535.0f, u16_at(2) / 65535.0f, u16_at(3) / 65535.0f}};
        return true;
    case vk::Format::eR16G16Sfloat:
        out.float32 = {{HalfToFloat(u16_at(0)), HalfToFloat(u16_at(1)), 0.0f, 1.0f}};
        return true;
    case vk::Format::eR16Sfloat:
        out.float32 = {{HalfToFloat(u16_at(0)), 0.0f, 0.0f, 1.0f}};
        return true;
    case vk::Format::eR8G8B8A8Unorm:
        out.float32 = {{u8_at(0), u8_at(1), u8_at(2), u8_at(3)}};
        return true;
    case vk::Format::eB8G8R8A8Unorm:
        out.float32 = {{u8_at(2), u8_at(1), u8_at(0), u8_at(3)}};
        return true;
    case vk::Format::eR8Unorm:
        out.float32 = {{u8_at(0), 0.0f, 0.0f, 1.0f}};
        return true;
    case vk::Format::eR32Sfloat:
        out.float32 = {{std::bit_cast<float>(texel[0]), 0.0f, 0.0f, 1.0f}};
        return true;
    case vk::Format::eR32G32Sfloat:
        out.float32 = {{std::bit_cast<float>(texel[0]), std::bit_cast<float>(texel[1]), 0.0f, 1.0f}};
        return true;
    case vk::Format::eR32G32B32A32Sfloat:
        out.float32 = {{std::bit_cast<float>(texel[0]), std::bit_cast<float>(texel[1]), std::bit_cast<float>(texel[2]),
                        std::bit_cast<float>(texel[3])}};
        return true;
    case vk::Format::eB10G11R11UfloatPack32:
        out.float32 = {{UnsignedFloat(texel[0] & 0x7ff, 6), UnsignedFloat((texel[0] >> 11) & 0x7ff, 6),
                        UnsignedFloat((texel[0] >> 22) & 0x3ff, 5), 1.0f}};
        return true;
    default:
        return false;
    }
}
} // namespace

bool Rasterizer::TryClearImage(VAddr dst, u32 bytes, const u32* pattern, u32 dwords) {
    const auto id = texture_cache.FindImageFromRange(dst, bytes, false);
    if (!id) {
        Core::NativeResources::NoteImageClear(false, 0);
        return false;
    }
    auto& image = texture_cache.GetImage(id);
    const auto& info = image.info;
    if (info.guest_address != dst || info.guest_size != bytes || info.num_samples != 1 || info.props.is_depth ||
        info.props.is_block || info.num_bits == 0 || (info.num_bits & 31) != 0) {
        Core::NativeResources::NoteImageClear(false, 1);
        return false;
    }
    // every texel must get the same bytes: the pattern repeats an integer number of times in a texel, or its texels are equal
    const u32 texel_dwords = info.num_bits / 32;
    bool uniform = true;
    std::array<u32, 4> texel{};
    if (texel_dwords > 4) {
        uniform = false;
    } else if (texel_dwords % dwords == 0) {
        for (u32 k = 0; k < texel_dwords; ++k) {
            texel[k] = pattern[k % dwords];
        }
    } else if (dwords % texel_dwords == 0) {
        for (u32 k = 0; k < texel_dwords; ++k) {
            texel[k] = pattern[k];
        }
        for (u32 k = texel_dwords; k < dwords; ++k) {
            uniform &= pattern[k] == texel[k % texel_dwords];
        }
    } else {
        uniform = false;
    }
    vk::ClearValue value{};
    if (!uniform || !DecodeTexel(info.pixel_format, texel.data(), value.color)) {
        Core::NativeResources::NoteImageClear(false, 2);
        return false;
    }
    {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 6) {
            LOG_INFO(Render_Vulkan, "memory fill of {:#x} + {} bytes is a clear of the {}x{} {} image: {:#x} {:#x} {:#x} {:#x} -> {} {} {} {}", dst,
                     bytes, info.size.width, info.size.height, vk::to_string(info.pixel_format), texel[0], texel[1], texel[2], texel[3],
                     value.color.float32[0], value.color.float32[1], value.color.float32[2], value.color.float32[3]);
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, bytes, "an image clear");
    runtime.ClearImage(&image, VideoCore::SubresourceRange{.base = {0, 0}, .extent = {static_cast<u16>(info.resources.levels), static_cast<u16>(info.resources.layers)}}, value);
    ++recorded_work;
    Core::NativeResources::NoteImageClear(true, 0);
    return true;
}

bool Rasterizer::TryCopyImages(VAddr dst, VAddr src, u32 num_bytes) {
    const auto src_id = texture_cache.FindImageFromRange(src, num_bytes);
    if (!src_id) {
        const auto any = texture_cache.FindImageFromRange(src, num_bytes, false);
        const auto dst_any = texture_cache.FindImageFromRange(dst, num_bytes, false);
        if (num_bytes >= (8u << 20) && (any || dst_any)) {
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1) < 16) {
                LOG_INFO(Render_Vulkan,
                         "image copy refused: source {:#x} has {} ({}), destination {:#x} has {}, {} bytes, source gpu modified {}",
                         src, any ? "an image that is not safe to download" : "no image",
                         any ? static_cast<u32>(texture_cache.GetImage(any).flags) : 0u, dst, dst_any ? "an image" : "no image",
                         num_bytes, buffer_cache.IsRegionGpuModified(src, num_bytes));
            }
        }
        Core::NativeResources::NoteImageCopy(false, 0);
        return false;
    }
    const auto dst_id = texture_cache.FindImageFromRange(dst, num_bytes, false);
    if (!dst_id || dst_id == src_id) {
        Core::NativeResources::NoteImageCopy(false, 1);
        return false;
    }
    auto& source = texture_cache.GetImage(src_id);
    auto& destination = texture_cache.GetImage(dst_id);
    const auto& a = source.info;
    const auto& b = destination.info;
    const bool same_layout =
        a.guest_address == src && b.guest_address == dst && a.guest_size == num_bytes && b.guest_size == num_bytes &&
        a.pixel_format == b.pixel_format && a.type == b.type && a.size.width == b.size.width &&
        a.size.height == b.size.height && a.size.depth == b.size.depth && a.resources.levels == b.resources.levels &&
        a.resources.layers == b.resources.layers && a.num_samples == 1 && b.num_samples == 1 && a.tile_mode == b.tile_mode &&
        a.array_mode == b.array_mode && a.num_bits == b.num_bits && a.pitch == b.pitch;
    if (!same_layout) {
        Core::NativeResources::NoteImageCopy(false, 2);
        return false;
    }
    // images at the destination's address in other formats are stale afterwards
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes, "an image copy");
    texture_cache.UpdateImage(src_id);
    runtime.CopyImage(&source, &destination);
    destination.flags |= VideoCore::ImageFlagBits::GpuModified;
    destination.flags &= ~VideoCore::ImageFlagBits::Dirty;
    ++recorded_work;
    Core::NativeResources::NoteImageCopy(true, 0);
    return true;
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds, bool staging_readback) {
    Core::ResourceAudit::Note(dst_gds ? Core::ResourceAudit::GdsFill : Core::ResourceAudit::MemoryCopyDst, dst, num_bytes, true);
    Core::ResourceAudit::Note(src_gds ? Core::ResourceAudit::GdsFill : Core::ResourceAudit::MemoryCopySrc, src, num_bytes, false);
    if (!dst_gds && !src_gds && Core::NativeResources::ImageCopyEnabled() && TryCopyImages(dst, src, num_bytes)) {
        return;
    }
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory. The destination is
            // invalidated as one range first: written page by page the copy would take a write fault per page.
            InvalidateMemory(dst, num_bytes, false);
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            buffer_cache.NoteCpGuestWrite(dst, num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes, "a memory copy");
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    // An explicit readback (a copy into a STAGING resource of the engine, bit 33554432): the bytes are downloaded in stream order below and
    // the destination is not marked GPU modified, so no page of it is read protected.
    const bool explicit_readback = staging_readback && !dst_gds && num_bytes <= VideoCore::BufferCache::MaxDmaDownload;
    if (staging_readback) {
        Core::NativeReadbacks::NoteExplicit(explicit_readback, num_bytes, dst_gds ? 0 : 1);
    }
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        if (explicit_readback) {
            return buffer_cache.ObtainBufferForDownload(dst, num_bytes);
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
    if (!dst_gds && VideoCore::ReadbackPolicy::Get().dma_async.load(std::memory_order_relaxed)) {
        buffer_cache.RecordDmaDownload(dst_buffer, dst, num_bytes, explicit_readback);
    }
    ++recorded_work;
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    Core::DebugServer::NoteWriteFault(addr);
    Core::NativeResources::NoteCpuWrite(addr, size);
#ifdef ARCH_X86_64
    Core::GuestHooks::NoteWriteFault(addr, size);
#endif
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    Core::NativeReadbacks::NoteReadFault(addr, size);
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed,
                                    const FixedState& fixed) const {
    UpdateViewportScissorState(fixed);
    UpdateDepthStencilState(fixed);
    UpdatePrimitiveState(is_indexed, fixed);
    UpdateRasterizationState(fixed);
    UpdateColorBlendingState(pipeline, fixed);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState(const FixedState& fixed) const {
    Viewports viewports;
    Scissors scissors;
    for (u32 i = 0; i < fixed.viewport_count; i++) {
        vk::Viewport viewport = fixed.viewports[i];
        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }
        if (fixed.clip_disabled && !fixed.viewport_empty) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        }
        viewports.push_back(viewport);
        scissors.push_back(fixed.scissors[i]);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState(const FixedState& fixed) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled = fixed.depth_test && fixed.depth_target;
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(fixed.depth_write && !fixed.depth_clear);
        dynamic_state.SetDepthCompareOp(fixed.depth_compare);
    }

    dynamic_state.SetDepthBoundsTestEnabled(fixed.depth_bounds_test);
    if (fixed.depth_bounds_test) {
        dynamic_state.SetDepthBounds(fixed.depth_bounds_min, fixed.depth_bounds_max);
    }

    dynamic_state.SetDepthBiasEnabled(fixed.depth_bias);
    if (fixed.depth_bias) {
        dynamic_state.SetDepthBias(fixed.depth_bias_constant, fixed.depth_bias_clamp,
                                   fixed.depth_bias_slope);
    }

    const auto stencil_test_enabled = fixed.stencil_test && fixed.stencil_target;
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        dynamic_state.SetStencilOps(fixed.stencil_front_ops, fixed.stencil_back_ops);
        dynamic_state.SetStencilReferences(fixed.stencil_reference[0], fixed.stencil_reference[1]);
        dynamic_state.SetStencilWriteMasks(fixed.stencil_clear ? 0U : fixed.stencil_write_mask[0],
                                           fixed.stencil_clear ? 0U : fixed.stencil_write_mask[1]);
        dynamic_state.SetStencilCompareMasks(fixed.stencil_compare_mask[0],
                                             fixed.stencil_compare_mask[1]);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed, const FixedState& fixed) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto prim_restart =
        fixed.primitive_restart && (instance.IsListRestartSupported() || !fixed.list_topology) &&
        (instance.IsPatchListRestartSupported() || !fixed.patch_list_topology);
    ASSERT_MSG(!is_indexed || !prim_restart || fixed.primitive_restart_index == 0xFFFF ||
                   fixed.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(fixed.rasterizer_discard);
    dynamic_state.SetCullMode(fixed.cull_mode);
    dynamic_state.SetFrontFace(fixed.front_face);
}

void Rasterizer::UpdateRasterizationState(const FixedState& fixed) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(fixed.line_width);
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline,
                                          const FixedState& fixed) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(fixed.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
