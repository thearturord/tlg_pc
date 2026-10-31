// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <atomic>
#include <thread>
#include "common/arch.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/debug.h"
#include "native/native_debug_server.h"
#include "video_core/renderer_vulkan/vk_identity_audit.h"
#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_light_recorder.h"
#include "native/native_untracked_ring.h"
#include "native/native_target_desc.h"
#include "native/native_audit.h"
#include "native/native_readbacks.h"
#include "native/native_device.h"
#include "native/native_resources.h"
#include "native/native_pass.h"
#include "native/native_updates.h"
#include "native/native_drain_audit.h"
#include "native/native_uploads.h"
#include "native/native_registry.h"
#include "native/native_resource_audit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "native/native_readback_policy.h"
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
#include "native/native_rasterizer_helpers.h"

// Native renderer: member functions of an emulator class that belong to the native path. They live here (not in the emulator's file)
// so that the emulator files only call into native/; the class stays declared in the emulator's header until the owning plan phase
// (documents/tlg/native plan/no-emulation-plan.md, E-1 part 2) gives the native side its own interface.

namespace Vulkan {

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

/// B9 (`TLG_DISPATCH_BINDER`): the descriptors of a dispatch come from its record's typed bindings. 0 off, 2 the record serves every descriptor it holds, 3 like 2 but every
/// descriptor is compared with the guest's (the guest's wins on a difference: the audit of the record against stale table reads). Default 0 (off): mode 2 caused a GPU device lost on 2026-10-28; 3 is a controlled test only.
static u32 DispatchBinderMode() {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_DISPATCH_BINDER");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 2u; // default 2 since 2026-10-28 (owner play: no glitch from it); 0 turns it off
    }();
    return mode;
}

static u32 NativeBinderMode() {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_NATIVE_BINDER");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 2u; // default 2 since 2026-10-27 (the owner's play test); 0 turns it off
    }();
    return mode;
}

namespace {
u64 Mix(u64 h, u64 v) {
    return (h ^ v) * 0x100000001b3ull;
}

u64 DigestStage(u64 h, const StagePlan& stage) {
    h = Mix(h, reinterpret_cast<u64>(stage.info));
    h = Mix(h, stage.pgm_base);
    h = Mix(h, stage.user_data_count);
    for (u32 i = 0; i < stage.user_data_count; ++i) {
        h = Mix(h, stage.user_data[i]);
    }
    h = Mix(h, stage.flat_skipped ? 1 : 0);
    for (const u32 word : stage.flat) {
        h = Mix(h, word);
    }
    return h;
}
} // namespace

namespace {
void ResetStage(StagePlan& stage) {
    stage.info = nullptr;
    stage.ring.clear();
    stage.user_data_count = 0;
    stage.flat.clear();
    stage.flat_skipped = false;
    stage.pgm_base = 0;
    stage.resolved.clear();
    stage.image_words.clear();
}
} // namespace

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

/// N3d: the V# of a buffer of a planned stage. A stage planned without flat user data takes it from the record's bindings of the plan (the binder's source); a
/// descriptor built from immediates needs no flat data at all.
template <class Desc>
AmdGpu::Buffer PlanBufferSharp(const DrawPlan* owner, const StagePlan& stage, const Desc& desc, const PlanStageView& view) {
    if (owner == nullptr || !stage.flat_skipped || desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
        return desc.GetSharp(view);
    }
    const u32 s = stage.info->sw_stage == Shader::SwStage::Vertex ? 0 : stage.info->sw_stage == Shader::SwStage::Fragment ? 1 : 2;
    const u32 offset = desc.sharp_fetch.offsets[0];
    if (s <= 1) {
        if (const u32* words = FindRecordWords(*stage.info, owner->bindings.data(), owner->binding_count, s, 1, offset, 4)) {
            return desc.FromWords(words, view);
        }
    }
    Core::NativeRegistry::NoteBinder(6); // must not happen: the cover test of the plan promised a binding
    return AmdGpu::Buffer::Null();
}
} // namespace

namespace {
/// True when [address, address + size) is committed, readable memory of this process (the vertex descriptor tables of native draws are in memory the runtime's mapping
/// table does not list, but the raster thread reads them by pointer all the same).
bool ReadableGuestMemory(VAddr address, size_t size) {
#ifdef _WIN32
    // A system call per vertex attribute took 60% of the command processor thread: the regions found readable are remembered (a few per thread, dropped now and then)
    struct Region {
        VAddr begin{}, end{};
    };
    thread_local std::array<Region, 4> known{};
    thread_local u32 next = 0, uses = 0;
    if ((++uses & 4095) == 0) {
        known = {};
    }
    for (const auto& region : known) {
        if (address >= region.begin && address + size <= region.end) {
            return true;
        }
    }
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
        return false;
    }
    constexpr DWORD NoRead = PAGE_NOACCESS | PAGE_GUARD;
    const VAddr begin = reinterpret_cast<VAddr>(info.BaseAddress);
    if (info.State != MEM_COMMIT || (info.Protect & NoRead) != 0 || address + size > begin + info.RegionSize) {
        return false;
    }
    known[next++ & 3] = {begin, begin + info.RegionSize};
    return true;
#else
    return true;
#endif
}
} // namespace


void Rasterizer::SetEngineClear(const Core::NativeDevice::ClearRecord* record) {
    engine_clear_on = record != nullptr;
    engine_clear_skip = false;
    if (record == nullptr) {
        return;
    }
    engine_clear = *record;
    engine_clear_skip = NativeClear(engine_clear);
}

bool Rasterizer::NativeClear(const Core::NativeDevice::ClearRecord& record) {
    // B11: the engine's clear as a Vulkan clear of the texture's images at its place in the stream. A color fast clear (the target has a meta surface) writes only
    // the CMASK on the console: the memory changes when the fast clear eliminate runs, and a texture read before that sees the old pixels (the game depends on it:
    // Trico's feathers over the view when it ran as a full clear, owner play 2026-10-29). It is kept on the image as a pending clear, applied as the clear of the
    // next render pass on the image or at the eliminate (audit against the old CMASK tracking: 8,832 same decisions, 0 value differences).
    const bool fast_clear = record.kind == 0 && record.path == 1;
    Core::NativeDevice::Object resource;
    if (!Core::NativeDevice::Lookup(record.resource_id, resource)) {
        Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeNotEligible);
        return false;
    }
    // a view of a 3D texture (slices): the whole texture
    const bool slices = resource.depth > 1 || record.layer_count == 0;
    decltype(std::declval<VideoCore::TextureCache&>().EngineImagesOfType()) ids;
    texture_cache.EngineImagesOf(record.resource_id, ids);
    // the view's mip and layers, in every image of the texture that holds them (an image of the texture's memory with the texture's size; it may hold fewer mips
    // or layers); an image of another size at that memory is another view of it and is left alone
    boost::container::small_vector<std::pair<VideoCore::ImageId, VideoCore::SubresourceRange>, 4> targets;
    for (const auto image_id : ids) {
        const auto& image = texture_cache.GetImage(image_id);
        if (image.info.props.is_depth != (record.kind == 1) || image.info.size.width != resource.width || image.info.size.height != resource.height) {
            continue;
        }
        const u32 levels = image.info.resources.levels, layers = image.info.resources.layers;
        if (slices) {
            targets.emplace_back(image_id, VideoCore::SubresourceRange{.base = {static_cast<u16>(std::min(record.mip, levels - 1)), 0},
                                                                       .extent = {1, static_cast<u16>(layers)}});
            continue;
        }
        if (record.mip >= levels || record.first_layer + record.layer_count > layers) {
            continue;
        }
        targets.emplace_back(image_id, VideoCore::SubresourceRange{.base = {static_cast<u16>(record.mip), static_cast<u16>(record.first_layer)},
                                                                   .extent = {1, static_cast<u16>(record.layer_count)}});
    }
    if (targets.empty()) {
        // a clear before the texture's first image waits on the texture and is applied when the image is made
        VideoCore::TextureCache::EngineClear clear{.depth = record.kind == 1, .fast = fast_clear, .flags = record.flags, .stencil = record.stencil,
                                                   .mip = record.mip, .first_layer = slices ? 0 : record.first_layer, .layer_count = slices ? 0 : record.layer_count};
        std::memcpy(clear.value, record.value, sizeof(clear.value));
        texture_cache.SetEnginePendingClear(record.resource_id, clear);
        Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeClearBeforeImage);
        Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeDone);
        ++recorded_work;
        return true;
    }
    if (record.mip != 0 || record.first_layer != 0 || resource.mips > 1 || resource.layers > 1) {
        Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeSubresource);
    }
    for (const auto& [image_id, range] : targets) {
        auto& image = texture_cache.GetImage(image_id);
        const bool whole = image.info.resources.levels == 1 && range.base.layer == 0 && range.extent.layers == image.info.resources.layers;
        if (!whole && True(image.flags & (VideoCore::ImageFlagBits::CpuDirty | VideoCore::ImageFlagBits::MaybeCpuDirty))) {
            // the rest of the image takes its CPU changes first (the upload covers the whole image)
            texture_cache.UpdateImage(image_id);
        }
        if (record.kind == 0) {
            const vk::ClearValue value = VideoCore::TextureCache::EngineColorValue(image.info.pixel_format, record.value);
            if (fast_clear && whole) {
                image.pending_fast_clear = value;
                Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeFastClearPending);
                continue;
            }
            if (fast_clear) {
                Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeFastClearKept); // a fast clear of a part of an image: cleared at once
            }
            runtime.ClearImage(&image, range, value);
        } else {
            runtime.ClearDepthStencilImage(&image, range, (record.flags & 1) != 0, record.value[0], (record.flags & 2) != 0, record.stencil);
            image.usage.depth_target = true;
        }
        // the image is what the clear made it: nothing older from memory is uploaded over it
        if (whole) {
            image.flags &= ~(VideoCore::ImageFlagBits::CpuDirty | VideoCore::ImageFlagBits::MaybeCpuDirty);
        }
        image.flags |= VideoCore::ImageFlagBits::GpuModified;
        Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeImages);
    }
    Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeDone);
    ++recorded_work;
    return true;
}

void Rasterizer::EngineObjectDeleted(u64 id, VAddr address, u64 bytes, u32 what) {
    if (const auto it = native_image_keys.find(id); it != native_image_keys.end()) {
        // B3: the deleted view's entries leave the table (the id is never used again)
        for (const u64 key : it->second) {
            native_images.erase(key);
        }
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8ViewIdsDropped);
        native_image_keys.erase(it);
    }
    if (what == 4) {
        texture_cache.NoteEngineBufferDelete(id); // B8.5
        buffer_cache.NoteEngineBufferDelete(id, address, bytes, Core::NativeUpdates::B9LifetimeMode() == 2);
        return;
    }
    if (what == 1 || what == 3) {
        // a texture with a live alias at its memory keeps its images (its lists go)
        texture_cache.NoteEngineTextureDelete(id, address, bytes, Core::NativeUpdates::B8LifetimeMode() == 2 && what == 1);
    }
}

std::unique_ptr<DrawPlan> Rasterizer::PlanExecDraw(const FixedState& native, u32 native_stale, u64 exec_key, u64 inputs_key,
                                                   std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data,
                                                   const Core::NativeDraw::ExecDraw& draw, u32 index_count, u32 draw_class) {
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
        Core::NativeDraw::Live().plan_null[native_stale != 0 ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
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
    plan->native_seq = draw.seq;
    plan->binding_count = draw.binding_count;
    plan->bindings_complete = draw.bindings_complete;
    for (u32 i = 0; i < draw.binding_count; ++i) {
        plan->bindings[i] = draw.bindings[i];
    }
    plan->op_class = 0xff;
    plan->depth_copy = plan->stencil_copy = false;
    plan->fixed = native;
    plan->inputs = cached->second;
    // E1b: the draw's arguments are the typed values the record builder set (the argument registers of the record are not decoded any more); a field the record did
    // not set keeps what the cached inputs hold, as before.
    {
        const auto& a = draw.args;
        auto& in = plan->inputs;
        if ((a.mask & Core::NativeDraw::DrawArgs::NumIndices) != 0) {
            in.num_indices = a.num_indices;
        }
        if ((a.mask & Core::NativeDraw::DrawArgs::NumInstances) != 0) {
            in.num_instances.num_instances = a.num_instances;
        }
        if ((a.mask & Core::NativeDraw::DrawArgs::IndexBase) != 0) {
            in.index_base_address.base_addr_hi = a.index_base_hi;
            in.index_base_address.base_addr_lo = a.index_base_lo;
        }
        if ((a.mask & Core::NativeDraw::DrawArgs::IndexType) != 0) {
            in.index_buffer_type.raw = a.index_type;
        }
        if ((a.mask & Core::NativeDraw::DrawArgs::MaxIndexSize) != 0) {
            in.max_index_size = a.max_index_size;
        }
        Core::NativeDraw::Live().draw_args_typed.fetch_add(1, std::memory_order_relaxed);
    }
    if (index_count != ~0u) {
        plan->inputs.num_indices = index_count; // an inline draw: the count comes with the draw packet
    }
    // the verify mode reads the flat user data of the plan; the constant ring takes the V#s of a stage without it from the record's bindings (N3d)
    const u32 binder_mode = NativeBinderMode();
    const bool skip_flat = (binder_mode == 1 || binder_mode == 2) && draw.bindings_complete && (binder_mode == 2 || draw_class == 0);
    if ((binder_mode == 1 || binder_mode == 2) && !draw.bindings_complete) {
        Core::NativeRegistry::NoteBinderReason(draw_class, 2, draw.bindings_overflow ? 10 : 1); // the whole draw keeps the flat user data: its binding list is not complete
    }
    if (!pipeline_cache.PlanGraphicsTyped(exec_key, plan->fixed, vertex_user_data, pixel_user_data, *plan, skip_flat ? &draw : nullptr, draw_class)) {
        Core::NativeDraw::Live().plan_null[2].fetch_add(1, std::memory_order_relaxed);
        RecyclePlan(std::move(plan));
        return nullptr;
    }
    plan->has_handles = false;
    plan->has_geometry_handles = false;
    plan->geometry_handle_count = 0;
    if (NativeHandleMode() != 0 && binder_mode != 0 && draw.bindings_complete && Core::NativeResources::ImmutableResidencyEnabled()) {
        plan->handle_epoch = Core::NativeResources::ImmutableEpoch();
        plan->geometry_handle_count = draw.geometry_id_count;
        for (u32 i = 0; i < draw.geometry_id_count; ++i) {
            plan->geometry_handles[i] = Core::NativeResources::FindImmutableById(draw.geometry_ids[i]);
            plan->has_geometry_handles |= plan->geometry_handles[i] != nullptr;
        }
        for (u32 i = 0; i < plan->binding_count; ++i) {
            const auto& b = plan->bindings[i];
            plan->handles[i] = b.kind == 1 && b.id != 0 ? Core::NativeResources::FindImmutableById(b.id) : nullptr;
            plan->has_handles |= plan->handles[i] != nullptr;
            if (b.kind == 1 && plan->handles[i] == nullptr && NativeHandleMode() == 2) {
                Core::NativeResources::NoteHandle(2);
            }
        }
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
    Core::NativeTargetDesc::ApplyEliminate(plan->inputs, draw.native_targets); // E7 (TLG_TARGET_DESC_USE)
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
        if (a.info != b.info || a.pgm_base != b.pgm_base || (!a.flat_skipped && a.flat.size() != b.flat.size()) || a.user_data_count != b.user_data_count) {
            out += fmt::format("stage{} ", s);
            continue;
        }
        const u16 mask = s == static_cast<u32>(Shader::SwStage::Fragment) ? pixel_mask : vertex_mask;
        bool differs = false;
        for (u32 i = 0; i < a.user_data_count; ++i) {
            differs |= ((mask >> i) & 1) != 0 && a.user_data[i] != b.user_data[i];
        }
        for (size_t i = 0; i < a.flat.size(); ++i) { // a skipped stage has none: its descriptors are the record's
            differs |= (i >= 16 || ((mask >> i) & 1) != 0) && a.flat[i] != b.flat[i];
        }
        if (differs) {
            out += fmt::format("stage{}_data ", s);
        }
    }
    return out;
}

u64 Rasterizer::PlanDigest(const DrawPlan& plan) {
    u64 h = 0xcbf29ce484222325ull;
    h = Mix(h, reinterpret_cast<u64>(plan.pipeline));
    h = Mix(h, plan.op_class);
    for (const auto& stage : plan.stages) {
        h = DigestStage(h, stage);
    }
    return h;
}

u64 Rasterizer::PlanDigest(const DispatchPlan& plan) {
    return DigestStage(Mix(0xcbf29ce484222325ull, reinterpret_cast<u64>(plan.pipeline)), plan.stage);
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
    plan->native_seq = 0;
    ResetStage(plan->stage);
    std::scoped_lock lk{plan_pool_mutex};
    if (dispatch_plan_pool.size() < 512) {
        dispatch_plan_pool.push_back(std::move(plan));
    }
}

/// N3 native binder switch (`TLG_NATIVE_BINDER`): 0 off, 1 the depth passes (class 0), 2 every class with a complete binding list, 3 like 2 and every descriptor is compared with
/// the legacy one (the legacy one wins on a difference).
/// B9: a binder plan made of a dispatch's record (the existing binder machinery reads (stage 5, kind, flat offset) from it). Null: the dispatch binds from the guest's tables.
const DrawPlan* Rasterizer::DispatchBinderFor(const DispatchPlan* plan) {
    const u32 mode = DispatchBinderMode();
    if (mode == 0 || plan == nullptr || !plan->bindings_complete || plan->binding_count == 0) {
        if (plan != nullptr) {
            NoteFlatRead(9, mode == 0 ? 1 : !plan->bindings_complete ? 2 : 3, 0, 0, Core::NativeDraw::RecordCallerGhidra(plan->native_seq), 0, plan->binding_count); // a dispatch with no usable record: why and where from
        } else {
            NoteFlatRead(9, 4, 0, 0, 0, 0, 0);
        }
        return nullptr;
    }
    if (!dispatch_binder_plan) {
        dispatch_binder_plan = std::make_unique<DrawPlan>();
    }
    DrawPlan& out = *dispatch_binder_plan;
    out.native_seq = plan->native_seq;
    out.binding_count = plan->binding_count;
    out.bindings_complete = true;
    // the record serves every kind of descriptor; an entry it lacks is a null descriptor (C8: no guest table read)
    out.any_flat_skipped = true;
    out.stages[static_cast<u32>(Shader::SwStage::Compute)].flat_skipped = true;
    out.has_handles = false;
    out.has_geometry_handles = false;
    out.geometry_handle_count = 0;
    std::memcpy(out.bindings.data(), plan->bindings.data(), sizeof(Core::NativeDraw::RecordBinding) * plan->binding_count);
    // L1(c): the engine buffer each kind 1 binding names, by the engine's id (the same handles the graphics plans carry, N3c); ObtainBuffer checks that the request lies inside it
    if (NativeHandleMode() != 0 && Core::NativeResources::ImmutableResidencyEnabled()) {
        out.handle_epoch = Core::NativeResources::ImmutableEpoch();
        for (u32 i = 0; i < out.binding_count; ++i) {
            const auto& b = out.bindings[i];
            out.handles[i] = b.kind == 1 && b.id != 0 ? Core::NativeResources::FindImmutableById(b.id) : nullptr;
            out.has_handles |= out.handles[i] != nullptr;
        }
    }
    binder_verify = mode == 3;
    g_dump_context = {out.bindings.data(), out.binding_count, plan->native_seq};
    return &out;
}

const DrawPlan* Rasterizer::NativeBinderFor(const DrawPlan* plan, u32 draw_class) {
    const u32 mode = NativeBinderMode();
    binder_verify = mode == 3;
    if (plan != nullptr && plan->any_flat_skipped) {
        return plan; // a stage of the plan has no flat user data: the record serves every descriptor, whatever the class
    }
    if (mode == 0 || plan == nullptr || !plan->bindings_complete || plan->binding_count == 0) {
        return nullptr;
    }
    if (mode == 1 && draw_class != 0) {
        return nullptr;
    }
    return plan;
}

Core::NativeResources::ImmutableBuffer* Rasterizer::GeometryHandle(VAddr address, u64 size) const {
    if (geometry_plan == nullptr || !geometry_plan->has_geometry_handles) {
        return nullptr;
    }
    if (geometry_plan->handle_epoch != Core::NativeResources::ImmutableEpoch()) {
        Core::NativeResources::NoteHandle(6);
        return nullptr;
    }
    for (u32 i = 0; i < geometry_plan->geometry_handle_count; ++i) {
        auto* handle = geometry_plan->geometry_handles[i];
        if (handle != nullptr && address >= handle->begin && address + size <= handle->end && handle->alive.load(std::memory_order_acquire)) {
            if (NativeHandleMode() == 2) {
                // compare mode: the lookup by address wins, and a difference is counted apart from the bindings'
                auto* lookup = Core::NativeResources::FindImmutable(address, size);
                if (lookup != handle) {
                    // Two live engine buffers can hold the same range (the log of the first run showed overlapping registrations: an engine buffer and another one that starts inside it).
                    // The data path is the same arena memory either way, the whole-buffer synchronization of either covers the request: that is an alias, not a difference.
                    // A difference is a lookup that finds no immutable buffer at all.
                    Core::NativeResources::NoteHandle(lookup == nullptr ? 7 : 8);
                    if (lookup == nullptr) {
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1) < 12) {
                            LOG_ERROR(Render_Vulkan, "N6.2 geometry handle differs: request {:#x}+{:#x}; handle {:#x}..{:#x}; the lookup finds none", address, size, handle->begin, handle->end);
                        }
                    }
                    return lookup;
                }
            }
            Core::NativeResources::NoteHandle(5);
            return handle;
        }
    }
    return nullptr;
}

/// The descriptor words of the binder's plan for the resource at flat user data `offset` of `stage` (null: the record has no binding there).
const u32* Rasterizer::BinderWords(const Shader::Info& stage, u32 offset, u32 dwords, u32 kind, u32* binding_index) const {
    const u32 s = stage.sw_stage == Shader::SwStage::Vertex ? 0 : stage.sw_stage == Shader::SwStage::Fragment ? 1 : stage.sw_stage == Shader::SwStage::Compute ? 5 : 2;
    if (s == 2) {
        return nullptr;
    }
    if (s == 5 && DispatchBinderMode() == 2 && (kind == 0 || kind > 3)) {
        return nullptr;
    }
    u32 first = 0;
    const u32* words = FindRecordWords(stage, binder_plan->bindings.data(), binder_plan->binding_count, s, kind, offset, dwords, &first);
    if (words != nullptr && s == 5) {
        // a dispatch's record can hold two entries that answer the same flat offset (the tables of the graphics context and the async context share slot numbers): when a second
        // entry gives other words the record does not say which one the shader reads, and the guest's tables serve it
        u32 second = 0;
        const u32* other = first + 1 < binder_plan->binding_count
                               ? FindRecordWords(stage, binder_plan->bindings.data() + first + 1, binder_plan->binding_count - first - 1, s, kind, offset, dwords, &second, false)
                               : nullptr;
        if (other != nullptr && std::memcmp(words, other, dwords * sizeof(u32)) != 0) {
            g_identity_audit.dispatch_diff[7].fetch_add(1, std::memory_order_relaxed); // counted as "ambiguous"
            return nullptr;
        }
    }
    if (binding_index != nullptr) {
        *binding_index = first;
    }
    return words;
}

void Rasterizer::ResolveAhead(DrawPlan* draw_plan, DispatchPlan* dispatch_plan) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RESOLVE_AHEAD");
        return e == nullptr || *e != '0';
    }();
    if (!on || !Core::NativeResources::ImmutableResidencyEnabled() || NativeBinderMode() == 3 || inplace_on) {
        return;
    }
    static const bool images_on = [] {
        const char* e = std::getenv("TLG_RESOLVE_IMAGES");
        return e == nullptr || *e != '0';
    }();
    // R2b: the engine's default and dynamic buffers too (resident buffers): the arena holds their contents while their generation is the synchronized one
    static const bool mutable_on = [] {
        const char* e = std::getenv("TLG_RESOLVE_MUTABLE");
        return e == nullptr || *e != '0';
    }();
    const auto resolve = [&](StagePlan& out, const Shader::Info& info, u32 s, const Core::NativeDraw::RecordBinding* list, u32 count,
                             Core::NativeResources::ImmutableBuffer* const* handles) {
        out.resolved.clear();
        out.resolved_epoch = Core::NativeResources::ImmutableEpoch();
        out.image_words.clear();
        if (count == 0) {
            return;
        }
        if (images_on) {
            // R3 step 2: the record entry of each image of the stage (the raster thread's BinderWords rule: the first match; a second match with other words, or
            // a compute stage the dispatch binder does not serve, leaves it to the raster thread)
            out.image_words.resize(info.images.size(), StagePlan::ImageUnknown);
            for (u32 i = 0; i < info.images.size(); ++i) {
                const auto& image = info.images[i];
                if (image.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad || (s == 5 && DispatchBinderMode() != 2)) {
                    continue;
                }
                const u32 dwords = image.is_r128 ? 4 : 8;
                u32 first = 0, second = 0;
                const u32* words = FindRecordWords(info, list, count, s, 2, image.sharp_fetch.offsets[0], dwords, &first, false);
                if (words == nullptr) {
                    out.image_words[i] = StagePlan::ImageNoEntry;
                    continue;
                }
                const u32* other = first + 1 < count ? FindRecordWords(info, list + first + 1, count - first - 1, s, 2, image.sharp_fetch.offsets[0], dwords, &second, false) : nullptr;
                if (other != nullptr && std::memcmp(words, other, dwords * sizeof(u32)) != 0) {
                    continue;
                }
                out.image_words[i] = (first << 8) | static_cast<u32>(words - list[first].words);
                image_words_planned.fetch_add(1, std::memory_order_relaxed);
            }
        }
        out.resolved.resize(info.buffers.size());
        for (u32 i = 0; i < info.buffers.size(); ++i) {
            const auto& desc = info.buffers[i];
            if (desc.IsSpecial() || desc.is_written || desc.is_formatted || desc.post_op == Shader::SharpFetchPostOp::OffsetByProgramBase ||
                desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
                continue;
            }
            // the same entry the raster thread's lookup takes: the first match, and none when a later entry answers the same load with other words
            u32 first = 0, second = 0;
            const u32* words = FindRecordWords(info, list, count, s, 1, desc.sharp_fetch.offsets[0], 4, &first, false);
            const auto reject = [&](u32 why) { resolve_stats.rejected[why].fetch_add(1, std::memory_order_relaxed); };
            if (words == nullptr) {
                reject(0);
                continue;
            }
            if (first + 1 < count) {
                const u32* other = FindRecordWords(info, list + first + 1, count - first - 1, s, 1, desc.sharp_fetch.offsets[0], 4, &second, false);
                if (other != nullptr && std::memcmp(words, other, 16) != 0) {
                    reject(0);
                    continue;
                }
            }
            const AmdGpu::Buffer v = desc.FromWords(words, info);
            const u64 bytes = v.GetSize();
            if (v.base_address == 0 || bytes == 0 || bytes >= (u64{1} << 28)) {
                reject(1);
                continue;
            }
            // the object of the engine id; in the handles' compare mode (2) the lookup by address wins, as it does in ObtainBuffer
            auto* object = NativeHandleMode() == 2 || (handles == nullptr && list[first].id == 0) ? Core::NativeResources::FindImmutable(v.base_address, static_cast<u32>(bytes))
                           : handles != nullptr                                               ? handles[first]
                                                                                              : Core::NativeResources::FindImmutableById(list[first].id);
            if (object == nullptr || (object->usage != 1 && !mutable_on) || v.base_address < object->begin || v.base_address + bytes > object->end) {
                reject(object == nullptr ? 2 : object->usage != 1 && !mutable_on ? 3 : 4);
                if (object == nullptr && (NativeHandleMode() == 2 || (handles == nullptr && list[first].id == 0))) {
                    out.resolved[i].no_object = true; // what ObtainBuffer's own lookup by address would find
                }
                continue;
            }
            const u32 generation = object->generation.load(std::memory_order_acquire);
            if (!object->alive.load(std::memory_order_acquire)) {
                reject(5);
                continue;
            }
            const u64 size = memory->ClampRangeSize(v.base_address, bytes);
            if (size != bytes) {
                reject(8);
                continue;
            }
            // not synchronized yet is fine: the raster thread binds it only once it is (at this generation)
            out.resolved[i] = {object, v.base_address, static_cast<u32>(size), generation};
            resolve_stats.planned.fetch_add(1, std::memory_order_relaxed);
        }
    };
    if (draw_plan != nullptr && draw_plan->pipeline != nullptr && (draw_plan->any_flat_skipped || (draw_plan->bindings_complete && draw_plan->binding_count != 0))) {
        const bool handles_ok = draw_plan->has_handles && draw_plan->handle_epoch == Core::NativeResources::ImmutableEpoch();
        for (const auto* info : draw_plan->pipeline->GetStages()) {
            if (info == nullptr || (info->sw_stage != Shader::SwStage::Vertex && info->sw_stage != Shader::SwStage::Fragment)) {
                continue;
            }
            resolve(draw_plan->stages[static_cast<u32>(info->sw_stage)], *info, info->sw_stage == Shader::SwStage::Vertex ? 0u : 1u, draw_plan->bindings.data(),
                    draw_plan->binding_count, handles_ok ? draw_plan->handles.data() : nullptr);
        }
    }
    if (dispatch_plan != nullptr && dispatch_plan->pipeline != nullptr && dispatch_plan->bindings_complete && dispatch_plan->binding_count != 0 && DispatchBinderMode() != 0) {
        resolve(dispatch_plan->stage, dispatch_plan->pipeline->GetStage(Shader::SwStage::Compute), 5u, dispatch_plan->bindings.data(), dispatch_plan->binding_count, nullptr);
    }
}

bool Rasterizer::PlannedImageWords(const Shader::Info& stage, u32 index, u32 dwords, const u32*& words, u32& binding_index) const {
    const StagePlan* rs = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                          : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                      : nullptr;
    if (rs == nullptr || binder_plan == nullptr || index >= rs->image_words.size() || rs->image_words[index] == StagePlan::ImageUnknown) {
        return false;
    }
    const u32 entry = rs->image_words[index];
    if (entry == StagePlan::ImageNoEntry) {
        words = nullptr;
        binding_index = ~0u;
    } else {
        binding_index = entry >> 8;
        if (binding_index >= binder_plan->binding_count || (entry & 0xff) + dwords > 8) {
            return false;
        }
        words = binder_plan->bindings[binding_index].words + (entry & 0xff);
    }
    static const u32 every = [] {
        const char* e = std::getenv("TLG_R3_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::max(std::atoi(e), 0)) : 0u;
    }();
    static thread_local u64 counter = 0;
    if (every != 0 && ++counter % every == 0) {
        // verify: the raster thread's own lookup gives the same words
        u32 check_index = ~0u;
        const u32* check = BinderWords(stage, stage.images[index].sharp_fetch.offsets[0], dwords, 2, &check_index);
        const_cast<Rasterizer*>(this)->image_words_verified.fetch_add(1, std::memory_order_relaxed);
        if ((check == nullptr) != (words == nullptr) || (check != nullptr && std::memcmp(check, words, dwords * sizeof(u32)) != 0)) {
            const_cast<Rasterizer*>(this)->image_words_differing.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return true;
}

std::array<u64, 18> Rasterizer::TakeResolveStats() {
    std::array<u64, 18> out{resolve_stats.planned.exchange(0), resolve_stats.hits.exchange(0), resolve_stats.stale.exchange(0)};
    for (u32 i = 0; i < 9; ++i) {
        out[3 + i] = resolve_stats.rejected[i].exchange(0);
    }
    for (u32 i = 0; i < 6; ++i) {
        out[12 + i] = resolve_stats.stale_why[i].exchange(0);
    }
    return out;
}

Runtime::HazardAudit Rasterizer::TakeHazardAudit() {
    return runtime.TakeHazardAudit();
}

void Rasterizer::DescriptorNoteBuffer(u32 stage, u32 index, const vk::DescriptorBufferInfo& info) noexcept {
    if (stage >= desc_prev_buffers.size() || index >= desc_prev_buffers[0].size()) {
        return;
    }
    auto& prev = desc_prev_buffers[stage][index];
    auto count = [&](DescKind k) { descriptor_census[k].fetch_add(1, std::memory_order_relaxed); };
    count(DBuffers);
    if (!info.buffer) {
        count(DBufferNull);
    }
    if (prev.buffer == info.buffer && prev.offset == info.offset && prev.range == info.range) {
        count(DBufferSame);
    } else if (prev.buffer == info.buffer && prev.range == info.range) {
        count(DBufferOffsetOnly);
        desc_draw_same = false;
        desc_draw_buffers_changed = true;
    } else if (prev.buffer == info.buffer) {
        count(DBufferOtherRange);
        desc_draw_same = false;
        desc_draw_same_but_offsets = false;
        desc_draw_buffers_changed = true;
    } else {
        count(DBufferOtherBuffer);
        desc_draw_same = false;
        desc_draw_same_but_offsets = false;
        desc_draw_buffers_changed = true;
    }
    prev = {info.buffer, info.offset, info.range};
}

void Rasterizer::DescriptorNoteImage(u32 stage, u32 index, vk::ImageView view) noexcept {
    if (stage >= desc_prev_views.size() || index >= desc_prev_views[0].size()) {
        return;
    }
    auto& prev = desc_prev_views[stage][index];
    descriptor_census[DImages].fetch_add(1, std::memory_order_relaxed);
    if (!view) {
        descriptor_census[DImageNull].fetch_add(1, std::memory_order_relaxed);
    }
    if (prev == view) {
        descriptor_census[DImageSame].fetch_add(1, std::memory_order_relaxed);
    } else {
        descriptor_census[DImageOther].fetch_add(1, std::memory_order_relaxed);
        desc_draw_same = false;
        desc_draw_same_but_offsets = false;
        desc_draw_images_changed = true;
    }
    prev = view;
}

void Rasterizer::DescriptorEndDraw() noexcept {
    descriptor_census[DDraws].fetch_add(1, std::memory_order_relaxed);
    if (desc_draw_same) {
        descriptor_census[DDrawsAllSame].fetch_add(1, std::memory_order_relaxed);
    }
    if (desc_draw_same_but_offsets) {
        descriptor_census[DDrawsSameButOffsets].fetch_add(1, std::memory_order_relaxed);
    }
    if (desc_draw_buffers_changed && !desc_draw_images_changed) {
        descriptor_census[DDrawsBuffersOnlyChanged].fetch_add(1, std::memory_order_relaxed);
    }
    if (desc_draw_images_changed && !desc_draw_buffers_changed) {
        descriptor_census[DDrawsImagesOnlyChanged].fetch_add(1, std::memory_order_relaxed);
    }
    desc_draw_same = desc_draw_same_but_offsets = true;
    desc_draw_buffers_changed = desc_draw_images_changed = false;
}

std::array<u64, 16> Rasterizer::TakeDescriptorCensus() noexcept {
    std::array<u64, 16> out{};
    for (u32 i = 0; i < DCount; ++i) {
        out[i] = descriptor_census[i].exchange(0, std::memory_order_relaxed);
    }
    return out;
}

std::array<u64, 64> Rasterizer::TakeConsumerCensus() noexcept {
    std::array<u64, 64> out{};
    for (u32 i = 0; i < 64; ++i) {
        out[i] = consumer_census[i].exchange(0, std::memory_order_relaxed);
    }
    return out;
}

void Rasterizer::CheckInPlaceVerifications() {
    // the submissions the GPU has finished: the bytes it read of an in place buffer may have been rewritten by the CPU since the bind; compare with the copy kept at the bind
    while (!inplace_checks.empty() && scheduler.IsFree(inplace_checks.front().tick)) {
        const InPlaceCheck& check = inplace_checks.front();
        std::vector<u8> now(check.bytes.size());
        memory->CopySparseMemory(check.address, now.data(), now.size());
        inplace_verified.fetch_add(1, std::memory_order_relaxed);
        if (std::memcmp(now.data(), check.bytes.data(), now.size()) != 0) {
            inplace_differing.fetch_add(1, std::memory_order_relaxed);
        }
        inplace_checks.pop_front();
    }
}

Rasterizer::InPlaceStats Rasterizer::TakeInPlaceStats() noexcept {
    InPlaceStats s;
    s.offered = inplace_offered.exchange(0);
    s.taken = inplace_taken.exchange(0);
    s.bytes = inplace_bytes.exchange(0);
    s.written = inplace_written.exchange(0);
    s.too_large = inplace_too_large.exchange(0);
    s.not_in_backing = inplace_not_in_backing.exchange(0);
    s.gpu_written = inplace_gpu_written.exchange(0);
    s.image_or_meta = inplace_image_or_meta.exchange(0);
    s.verified = inplace_verified.exchange(0);
    s.differing = inplace_differing.exchange(0);
    s.verify_dropped = inplace_verify_dropped.exchange(0);
    if (host_import) {
        s.imported_bytes = host_import->ImportedBytes();
        s.imported_chunks = host_import->ImportedChunks();
        s.failed_chunks = host_import->FailedChunks();
    }
    return s;
}

void Rasterizer::NoteIndexedRow(u32 slot) {
    if (indexed_row_size < indexed_row.size()) {
        indexed_row[indexed_row_size++] = slot;
    } else {
        LOG_ERROR(Render_Vulkan, "indexed data: the row of a draw is longer than {} entries", indexed_row.size());
    }
}

void Rasterizer::NoteIndexedFlat(const Shader::Info& stage) {
    const u32 count = stage.srt_info.flattened_bufsize_dw;
    if (indexed_row_size + count > indexed_row.size()) {
        LOG_ERROR(Render_Vulkan, "indexed data: the row of a draw is longer than {} entries (flat user data of {} dwords)", indexed_row.size(), count);
        return;
    }
    const u32 have = std::min<u32>(count, static_cast<u32>(stage.flattened_ud_buf.size()));
    if (have != 0) {
        std::memcpy(indexed_row.data() + indexed_row_size, stage.flattened_ud_buf.data(), have * sizeof(u32));
    }
    if (have < count) {
        std::memset(indexed_row.data() + indexed_row_size + have, 0, (count - have) * sizeof(u32));
    }
    indexed_row_size += count;
    IndexedData::Get()->stats.flat_dwords.fetch_add(count, std::memory_order_relaxed);
}

void Rasterizer::NoteIndexedImage(const vk::DescriptorImageInfo& info, u32 image_key) {
    auto* indexed = IndexedData::Get();
    const u32 slot = indexed->ImageSlot(image_key, info.imageView, info.imageLayout);
    NoteIndexedRow(slot);
    const u32 every = indexed->VerifyEvery();
    if (every != 0 && indexed->stats.draws.load(std::memory_order_relaxed) % every == 0) {
        indexed->stats.verified.fetch_add(1, std::memory_order_relaxed);
        vk::ImageView view_of;
        vk::ImageLayout layout_of;
        const bool known = indexed->ImageOf(image_key, slot, view_of, layout_of);
        const bool same = info.imageView ? (known && view_of == info.imageView && layout_of == info.imageLayout) : slot == 0;
        if (!same) {
            indexed->stats.verify_differences.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render_Vulkan, "indexed data: image slot {} (key {}) does not stand for the view {:#x} layout {} of the legacy binder", slot, image_key,
                      reinterpret_cast<u64>(static_cast<VkImageView>(info.imageView)), static_cast<u32>(info.imageLayout));
        }
    }
}

void Rasterizer::NoteIndexedRun(const vk::DescriptorImageInfo* infos, u32 count, u32 image_key) {
    auto* indexed = IndexedData::Get();
    bool all_null = true;
    for (u32 i = 0; i < count; ++i) {
        all_null &= !infos[i].imageView;
    }
    const u32 base = all_null ? indexed->NullRun(image_key) : indexed->ImageRun(image_key, infos, count);
    NoteIndexedRow(base);
    const u32 every = indexed->VerifyEvery();
    if (every != 0 && indexed->stats.draws.load(std::memory_order_relaxed) % every == 0) {
        indexed->stats.verified.fetch_add(1, std::memory_order_relaxed);
        bool same = true;
        for (u32 i = 0; i < count; ++i) {
            vk::ImageView view_of;
            vk::ImageLayout layout_of;
            const bool known = indexed->RunOf(image_key, base, i, view_of, layout_of);
            same &= infos[i].imageView ? (known && view_of == infos[i].imageView && layout_of == infos[i].imageLayout) : (base == indexed->NullRun(image_key) || (known && !view_of));
        }
        if (!same) {
            indexed->stats.verify_differences.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render_Vulkan, "indexed data: the run at slot {} (key {}, {} levels) does not stand for the views of the legacy binder", base, image_key, count);
        }
    }
}

void Rasterizer::NoteIndexedSampler(vk::Sampler sampler) {
    auto* indexed = IndexedData::Get();
    const u32 slot = indexed->SamplerSlot(sampler);
    NoteIndexedRow(slot);
    const u32 every = indexed->VerifyEvery();
    if (every != 0 && indexed->stats.draws.load(std::memory_order_relaxed) % every == 0) {
        indexed->stats.verified.fetch_add(1, std::memory_order_relaxed);
        vk::Sampler sampler_of;
        const bool same = indexed->SamplerOf(slot, sampler_of) && sampler_of == sampler;
        if (!same) {
            indexed->stats.verify_differences.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render_Vulkan, "indexed data: sampler slot {} does not stand for the sampler {:#x} of the legacy binder", slot,
                      reinterpret_cast<u64>(static_cast<VkSampler>(sampler)));
        }
    }
}

void Rasterizer::FrontStageNotes(StagePlan& stage, u64 position, const DrawPlan* owner) {
    stage.ring.clear();
    if (!stage.info) {
        return;
    }
    const PlanStageView view{std::span<const u32>{stage.flat.data(), stage.flat.size()}, stage.pgm_base};
    for (const auto& desc : stage.info->buffers) {
        if (desc.IsSpecial() || !desc.is_written) {
            continue;
        }
        const auto vsharp = PlanBufferSharp(owner, stage, desc, view);
        if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
            continue;
        }
        NotePendingGpuWrite(vsharp.base_address, vsharp.GetSize(), position);
        ring_written_buffers.fetch_add(1, std::memory_order_relaxed);
    }
}

void Rasterizer::FrontStageRing(StagePlan& stage, u64 done, const DrawPlan* owner, u64 seq) {
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
        if (desc.is_formatted && !Core::NativeUploads::RingFormatted()) {
            ring_formatted.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const auto vsharp = PlanBufferSharp(owner, stage, desc, view);
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
        // N6a: the decisions below read the buffer cache's state of this range (GPU written, pending writes) and then its bytes; a queued write to the range changes both
        Core::DrainAudit::Read(Core::DrainAudit::ReadRingConstant, address, size);
        if (Core::NativeUploads::RingPrecise()) {
            // N5e: the bytes decide, not the 4 KB page that holds them
            if (buffer_cache.IsRangeGpuWritten(address, size)) {
                ring_gpu_modified.fetch_add(1, std::memory_order_relaxed);
                Core::NativeUploads::NoteRingPrecise(2);
                continue;
            }
            if (buffer_cache.IsRegionGpuModified(address, size)) {
                Core::NativeUploads::NoteRingPrecise(0);
            }
        } else if (buffer_cache.IsRegionGpuModified(address, size)) {
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
        bool from_snapshot = false;
        if (seq != 0 && Core::NativeUploads::SnapshotsEnabled()) {
            // N5d: the writers' copy of the range as it was when this record was issued; a range the newest write only partly covers is not known at all
            const auto result = Core::NativeUploads::SnapshotRead(address, size, seq, constant_ring->Data(*offset));
            if (result == Core::NativeUploads::SnapshotResult::Partial) {
                Core::NativeUploads::NoteSnapshotRing(1);
                continue; // the binding stays on the raster thread (the ring bytes already allocated are simply unused)
            }
            from_snapshot = result == Core::NativeUploads::SnapshotResult::Found;
            if (from_snapshot) {
                // guard (on until the snapshots are proven): where guest memory holds other bytes than the snapshot, the guest bytes are used as before and the case is counted
                static const bool guard = [] {
                    const char* e = std::getenv("TLG_SNAPSHOT_GUARD");
                    return e == nullptr || *e != '0';
                }();
                if (guard) {
                    thread_local std::vector<u8> now;
                    now.resize(size);
                    memory->CopySparseMemory(address, now.data(), size);
                    if (std::memcmp(now.data(), constant_ring->Data(*offset), size) != 0) {
                        {
                            const u8* snap = static_cast<const u8*>(constant_ring->Data(*offset));
                            u64 first = ~0ull, count = 0;
                            for (u64 i = 0; i < size; ++i) {
                                if (snap[i] != now[i]) {
                                    first = first == ~0ull ? i : first;
                                    ++count;
                                }
                            }
                            u32 snap_word = 0, guest_word = 0;
                            const u64 word_at = first & ~u64{3};
                            if (word_at + 4 <= size) {
                                std::memcpy(&snap_word, snap + word_at, 4);
                                std::memcpy(&guest_word, now.data() + word_at, 4);
                            }
                            const u32 who = Core::NativeUpdates::WhoWrote(address, size);
                            Core::NativeUploads::NoteSnapshotDiff(address, size, first, count, snap_word, guest_word, who == ~0u ? 8 : (who & 0xff));
                        }
                        std::memcpy(constant_ring->Data(*offset), now.data(), size);
                        Core::NativeUploads::NoteSnapshotRing(4);
                        from_snapshot = false;
                    }
                }
                if (from_snapshot) {
                    Core::NativeUploads::NoteSnapshotRing(0);
                }
            }
            if (result == Core::NativeUploads::SnapshotResult::Found && !from_snapshot) {
                memory->CopySparseMemory(address, constant_ring->Data(*offset), size);
            }
        }
        if (!from_snapshot) {
            memory->CopySparseMemory(address, constant_ring->Data(*offset), size);
        }
        Core::DrainAudit::Copied(Core::DrainAudit::ReadRingConstant, address, constant_ring->Data(*offset), size);
        constant_ring->Flush(*offset, size);
        stage.ring.push_back({index, static_cast<u32>(size), *offset, address, from_snapshot, desc.is_formatted});
        ring_taken.fetch_add(1, std::memory_order_relaxed);
        ring_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

void Rasterizer::FrontGeometryRing(DrawPlan& plan, u64 done, bool indexed, u32 index_offset) {
    plan.geometry_ring.clear();
    if (!Core::NativeUploads::RingGeometry() || plan.pipeline == nullptr || plan.op_class != 0xff) {
        return;
    }
    const u64 cap = std::min<u64>(Core::NativeUploads::RingGeometryCap(), VideoCore::BufferCache::StreamThreshold() * 64ull);
    const u64 alignment = std::max<u64>(instance.StorageMinAlignment(), instance.UniformMinAlignment());
    const auto take = [&](VAddr address, u64 size, u32 kind) {
        if (address == 0 || size == 0) {
            return;
        }
        const u32 base = kind == GeometryVertex ? 16 : 24;
        if (size > cap) {
            Core::NativeUploads::NoteGeometry(5);
            Core::NativeUploads::NoteGeometry(base + 4);
            return;
        }
        if (Core::NativeResources::ImmutableResidencyEnabled() && Core::NativeResources::FindImmutable(address, size)) {
            Core::NativeUploads::NoteGeometry(7);
            Core::NativeUploads::NoteGeometry(base + 0);
            return;
        }
        Core::DrainAudit::Read(kind == GeometryVertex ? Core::DrainAudit::ReadRingVertex : Core::DrainAudit::ReadRingIndex, address, size); // N6a: state and bytes of the range
        const bool gpu_written = Core::NativeUploads::RingPrecise() ? buffer_cache.IsRangeGpuWritten(address, size) : buffer_cache.IsRegionGpuModified(address, size);
        if (gpu_written) {
            Core::NativeUploads::NoteGeometry(6);
            Core::NativeUploads::NoteGeometry(base + 1);
            return;
        }
        if (PendingGpuWriteOverlaps(address, size, done)) {
            Core::NativeUploads::NoteGeometry(7);
            Core::NativeUploads::NoteGeometry(base + 2);
            return;
        }
        const auto offset = constant_ring->Allocate(size, alignment);
        if (!offset) {
            Core::NativeUploads::NoteGeometry(7);
            Core::NativeUploads::NoteGeometry(base + 3);
            return;
        }
        memory->CopySparseMemory(address, constant_ring->Data(*offset), size);
        Core::DrainAudit::Copied(kind == GeometryVertex ? Core::DrainAudit::ReadRingVertex : Core::DrainAudit::ReadRingIndex, address, constant_ring->Data(*offset), size);
        constant_ring->Flush(*offset, size);
        plan.geometry_ring.push_back({kind, static_cast<u32>(size), *offset, address});
        Core::NativeUploads::NoteGeometry(kind == GeometryVertex ? 0 : 1);
    };
    // the vertex buffers: the V#s of the fetch shader's attributes, read from the vertex stage's user data as the plan holds them
    const auto& fetch_shader = plan.pipeline->GetFetchShader();
    const StagePlan* vertex = nullptr;
    for (const auto& stage : plan.stages) {
        if (stage.info != nullptr && stage.info->sw_stage == Shader::SwStage::Vertex) {
            vertex = &stage;
        }
    }
    Core::NativeUploads::NoteGeometry(10);
    if (vertex == nullptr || fetch_shader.Empty() || fetch_shader.attributes.empty()) {
        Core::NativeUploads::NoteGeometry(11);
    }
    if (vertex != nullptr && !fetch_shader.Empty() && !fetch_shader.attributes.empty()) {
        struct Range {
            VAddr begin, end;
        };
        boost::container::small_vector<Range, 16> ranges;
        for (const auto& attrib : fetch_shader.attributes) {
            const u32* table = vertex->user_data.data();
            if (attrib.sgpr_base != Shader::IR::NumScalarRegs) {
                if (attrib.sgpr_base >= vertex->user_data_count) {
                    Core::NativeUploads::NoteGeometry(12);
                    static std::atomic<u32> logged{0};
                    if (logged.fetch_add(1) < 6) {
                        LOG_ERROR(Core, "geometry ring: attribute reads its table pointer from user data register {} but the vertex stage has {} (attributes {})", attrib.sgpr_base,
                                 vertex->user_data_count, fetch_shader.attributes.size());
                    }
                    return;
                }
                u64 pointer = 0;
                std::memcpy(&pointer, &vertex->user_data[attrib.sgpr_base], sizeof(pointer));
                table = reinterpret_cast<const u32*>(pointer & 0xFFFFFFFFFFFFULL);
                const VAddr table_entry = reinterpret_cast<VAddr>(table + attrib.dword_offset);
                // the descriptor tables of native draws can live in the table arena window, which is no mapping of the runtime
                if (!ReadableGuestMemory(table_entry, sizeof(AmdGpu::Buffer))) {
                    Core::NativeUploads::NoteGeometry(12);
                    static std::atomic<u32> logged_pointer{0};
                    if (logged_pointer.fetch_add(1) < 6) {
                        LOG_ERROR(Core, "geometry ring: the vertex table pointer {:#x} (register {}, dword offset {}) is no mapping", pointer, attrib.sgpr_base, attrib.dword_offset);
                    }
                    return;
                }
            }
            AmdGpu::Buffer buffer;
            if (attrib.sgpr_base != Shader::IR::NumScalarRegs) {
                Core::DrainAudit::Read(Core::DrainAudit::ReadFetch, reinterpret_cast<u64>(table + attrib.dword_offset), sizeof(buffer));
            }
            std::memcpy(&buffer, table + attrib.dword_offset, sizeof(buffer));
            if (attrib.sgpr_base != Shader::IR::NumScalarRegs) {
                Core::DrainAudit::Copied(Core::DrainAudit::ReadFetch, reinterpret_cast<u64>(table + attrib.dword_offset), &buffer, sizeof(buffer));
            }
            buffer.base_address += attrib.inst_offset;
            if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                ranges.push_back({buffer.base_address, buffer.base_address + buffer.GetSize()});
            }
        }
        std::ranges::sort(ranges, [](const Range& a, const Range& b) { return a.begin < b.begin; });
        boost::container::small_vector<Range, 16> merged;
        for (const auto& range : ranges) {
            if (!merged.empty() && merged.back().end >= range.begin) {
                merged.back().end = std::max(merged.back().end, range.end);
            } else {
                merged.push_back(range);
            }
        }
        for (const auto& range : merged) {
            if (range.end - range.begin >= (1ull << 30) && !memory->IsValidMapping(range.begin, 1)) {
                continue;
            }
            Core::NativeUploads::NoteGeometry(13);
            take(range.begin, memory->ClampRangeSize(range.begin, range.end - range.begin), GeometryVertex);
        }
    }
    if (indexed) {
        const auto& regs = plan.inputs;
        const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
        const VAddr base_address = regs.index_base_address.Address<VAddr>();
        const VAddr index_address = base_address + u64(index_offset) * index_size;
        const u64 requested = u64(std::min(regs.num_indices, regs.max_index_size)) * index_size;
        if (base_address != 0 && requested != 0 && (requested < (1ull << 30) || memory->IsValidMapping(index_address, 1))) {
            const u64 mapped = memory->ClampRangeSize(index_address, requested);
            take(index_address, Common::AlignDown(mapped, index_size), GeometryIndex);
        }
    }
}

const RingBinding* Rasterizer::FindGeometryRing(VAddr address, u64 size, u32 kind) const noexcept {
    if (ring_draw == nullptr || const_ring_level < 4) {
        return nullptr;
    }
    for (const auto& ring : ring_draw->geometry_ring) {
        if (ring.index == kind && ring.address == address && ring.size == size) {
            return &ring;
        }
    }
    return nullptr;
}

void Rasterizer::FrontQueueOp(DrawPlan* draw, DispatchPlan* dispatch, u64 position, u64 done, u64 seq, bool indexed, u32 index_offset) {
    if (const_ring_level == 0 || (!draw && !dispatch)) {
        return;
    }
    ring_queued.fetch_add(1, std::memory_order_relaxed);
    // First everything the plan writes (a read-only binding of the same plan may alias a written one), then the copies.
    if (draw) {
        for (auto& stage : draw->stages) {
            FrontStageNotes(stage, position, draw);
        }
    } else {
        FrontStageNotes(dispatch->stage, position, nullptr);
    }
    if (const_ring_level < 3 || !constant_ring) {
        return;
    }
    if (draw) {
        for (auto& stage : draw->stages) {
            FrontStageRing(stage, done, draw, seq);
        }
        FrontGeometryRing(*draw, done, indexed, index_offset);
        draw->ring_end = constant_ring->Position();
    } else {
        FrontStageRing(dispatch->stage, done, nullptr, seq);
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

void Rasterizer::OcclusionOpen(VAddr address, u32 pairs) {
    if (!occlusion) {
        occlusion = std::make_unique<Core::NativeOcclusion::Queries>(instance, scheduler, [this](VAddr addr, u64 size) { NoteCpGuestWrite(addr, size); });
    }
    occlusion->Open(address, pairs);
}

void Rasterizer::OcclusionClose(VAddr address) {
    if (occlusion) {
        occlusion->Close(address);
    }
}

} // namespace Vulkan
