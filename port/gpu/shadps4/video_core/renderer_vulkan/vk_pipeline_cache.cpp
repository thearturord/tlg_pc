// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>
#include <optional>
#include <ranges>
#include <set>
#include <tuple>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "native/native_drain_audit.h"
#include "native/native_draw.h"
#include "native/native_registry.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "native/native_executor.h"
#include "native/native_shader_env.h"
#include "native/native_debug_server.h"
#include "video_core/cache_storage.h"
#include "native/native_readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_boundary.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "native/native_split_census.h"

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

namespace {

// N6a (TLG_DRAIN_AUDIT): the guest reads of the planning, see core/native_drain_audit.h
void NoteCode(const Shader::ShaderParams& params) {
    Core::DrainAudit::Read(Core::DrainAudit::ReadCode, reinterpret_cast<u64>(params.code.data()), params.code.size_bytes());
}

void NoteFlat(const Shader::Info& info) {
    if (info.srt_info.walker_func && Core::DrainAudit::Enabled()) {
        Core::DrainAudit::WalkerRead(info.user_data.data(), static_cast<u32>(info.user_data.size()));
    }
}

void NoteFetch(const Shader::Gcn::VertexAttribute& attrib, const Shader::Info& info) {
    if (attrib.sgpr_base == Shader::IR::NumScalarRegs || !Core::DrainAudit::Enabled()) {
        return;
    }
    const auto ud = info.UserData();
    if (attrib.sgpr_base + 1u >= ud.size()) {
        return;
    }
    u64 pointer = 0;
    std::memcpy(&pointer, &ud[attrib.sgpr_base], sizeof(pointer));
    Core::DrainAudit::Read(Core::DrainAudit::ReadFetch, (pointer & 0xFFFFFFFFFFFFULL) + u64(attrib.dword_offset) * 4, 16); // a V# is four dwords
}

} // Anonymous namespace

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

/// F3.3: the sample count the pixel shader's coverage input sees: that of the targets the draw has bound (the largest of the colour targets and the depth target; 1 without targets).
static u32 TargetSampleCount(const DrawInputs& in) {
    u32 samples = 1;
    for (const auto& cb : in.color_buffers) {
        if (cb) {
            samples = std::max(samples, cb.NumSamples());
        }
    }
    if (in.depth_buffer.DepthValid() || in.depth_buffer.StencilValid()) {
        samples = std::max(samples, in.depth_buffer.NumSamples());
    }
    return samples;
}

/// F3.2b: the parameters that locate a stage's program (code, hash, user data words) of a native draw from the shader environment (the program address of the shader object) and
/// the user data words the record's draws left; none for another stage than vertex and pixel or a draw without typed user data (the register image then decides).
static std::optional<Shader::ShaderParams> ParamsFromEnv(const Core::NativeShaderEnv::DrawEnv& env, u32 hw_stage) {
    const bool vertex = hw_stage == static_cast<u32>(HwStage::Vertex);
    if (!vertex && hw_stage != static_cast<u32>(HwStage::Fragment)) {
        return std::nullopt;
    }
    const u32* words = vertex ? env.vertex_user_words : env.pixel_user_words;
    const u64 address = vertex ? env.vertex.program.address : env.pixel.program.address;
    if (words == nullptr || address == 0 || !(vertex ? env.has_vertex : env.has_pixel)) {
        return std::nullopt;
    }
    const auto* code = reinterpret_cast<const u32*>(address);
    const auto& bininfo = AmdGpu::SearchBinaryInfo(code);
    return Shader::ShaderParams{
        .user_data = std::span<const u32, Shader::ShaderParams::NumShaderUserData>{words, Shader::ShaderParams::NumShaderUserData},
        .code = std::span{code, bininfo.length / sizeof(u32)},
        .hash = bininfo.shader_hash,
    };
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    info.Initialize(stage, l_stage);
    auto& live = Core::NativeDraw::Live();
    if (stage == HwStage::Compute) {
        ShaderBoundary::FillComputeInfo(info, *lookup_cs);
        return info;
    }
    // F3.5: a graphics stage is the vertex or the pixel shader of a native draw; its translator input is built from the shader environment, the fixed state, the key's targets and the
    // target sample count. No register image exists any more; a draw without a shader environment gets no pipeline (the stage selection refuses it first).
    Core::NativeShaderEnv::DrawEnv env;
    if ((stage != HwStage::Vertex && stage != HwStage::Fragment) || lookup_inputs == nullptr || !Core::NativeShaderEnv::Current(env) ||
        (stage == HwStage::Vertex ? !env.has_vertex : !env.has_pixel) || env.other_stage) {
        Core::NativeShaderEnv::NoteUnavailable();
        return info;
    }
    const auto& fixed = *lookup_fixed;
    if (stage == HwStage::Vertex) {
        info.hw.vs.user_clip_plane_mask = fixed.user_clip_plane_mask;
        info.hw.vs.emulate_depth_negative_one_to_one = !instance.IsDepthClipControlSupported() && fixed.negative_one_to_one;
        info.hw.vs.clip_disable = fixed.clip_disabled;
        info.sw.vs.vertex_sgpr_offset = draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.is_indirect = draw_indirect_params.is_indirect;
        info.sw.vs.tess_emulated_primitive = fixed.primitive_type == Core::NativePrimitive::Kind::RectList || fixed.primitive_type == Core::NativePrimitive::Kind::QuadList;
    } else {
        info.hw.fs.dual_source_blending = fixed.DualSourceBlending();
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // lowered user clip planes and guest exported clip distances both arm the emulation; no geometry or local stage exists in this pipeline
        const bool lowers_user_clip_planes = fixed.user_clip_plane_mask != 0;
        info.hw.fs.clip_distance_emulation = ((env.has_vertex && ShaderBoundary::GuestClipDistances(env.vertex)) || lowers_user_clip_planes) && profile.needs_clip_distance_emulation;
    }
    ShaderBoundary::ApplyShaderEnv(info, env, stage, TargetSampleCount(*lookup_inputs));
    live.key_targets[6].fetch_add(1, std::memory_order_relaxed);
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             Core::NativeExecutor* executor_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, executor{executor_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes},
      stage_lookup_reuse{EmulatorSettings.GetStageLookupReuse()} {
    lookup_fixed = &lookup_fixed_storage;
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    if (IndexedData::RequestedByEnvironment()) {
        if (instance.IsIndexedDataSupported() && IndexedData::DeviceFits(instance)) {
            indexed_data = std::make_unique<IndexedData>(instance, scheduler);
            profile.indexed_resources = true;
        } else {
            LOG_ERROR(Render_Vulkan, "TLG_INDEXED_DATA is set but the device lacks descriptor indexing features: the legacy binder is used");
        }
    }
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() = default;

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    Core::SplitCensus::Touch(Core::SplitCensus::Pipeline);
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::PipelineLookup};
    if (!lookup_pinned || lookup_inputs == nullptr) {
        return nullptr; // F3.5: only a plan (inputs, fixed state, shader environment) looks a graphics pipeline up
    }
    draw_indirect_params = params;
    ExecEntry* exec_entry = nullptr;
    bool exec_audit = false;
    auto& live = Core::NativeDraw::Live();
    if (lookup_exec_key != 0) {
        if (const auto found = exec_pipelines.find(lookup_exec_key); found != exec_pipelines.end() && found->second.epoch == exec_epoch) {
            exec_entry = &found.value();
            exec_audit = (++exec_tick & 31) == 0;
        }
        if (exec_entry && !exec_audit) {
            infos = exec_entry->infos;
            modules = exec_entry->modules;
            fetch_shader = FetchOf(*exec_entry);
            bool resolved_all = true;
            Core::NativeShaderEnv::DrawEnv entry_env;
            const bool entry_native = Core::NativeShaderEnv::Current(entry_env);
            for (u32 s = 0; s < MaxShaderStages && resolved_all; ++s) {
                if (!infos[s]) {
                    continue;
                }
                const std::optional<Shader::ShaderParams> stage_params_env = entry_native ? ParamsFromEnv(entry_env, exec_entry->stage_hw[s]) : std::optional<Shader::ShaderParams>{};
                live.key_targets[stage_params_env ? 8 : 9].fetch_add(1, std::memory_order_relaxed);
                if (!stage_params_env) {
                    resolved_all = false;
                    break;
                }
                const auto& stage_params = *stage_params_env;
                NoteCode(stage_params);
                auto* info = const_cast<Shader::Info*>(infos[s]);
                info->pgm_base = stage_params.Base();
                info->user_data = stage_params.user_data;
                info->RefreshFlatBuf();
                NoteFlat(*info);
            }
            if (resolved_all) {
                live.stage_key_hits.fetch_add(1, std::memory_order_relaxed);
                return exec_entry->pipeline;
            }
            exec_entry = nullptr;
        }
    } else {
        live.stage_key_uncached.fetch_add(1, std::memory_order_relaxed);
    }
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    const auto [it, is_new] = [&] {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::PipelineMap};
        return graphics_pipelines.try_emplace(graphics_key);
    }();
    if (is_new) {
        VideoCore::ReadbackStats::NoteMutation(
            VideoCore::ReadbackStats::Mutation::PipelineCompiled);
        VideoCore::ReadbackStats::CpTimer compile_timer{
            VideoCore::ReadbackStats::CpTime::PipelineCompile};
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
    }
    if (lookup_exec_key != 0) {
        const GraphicsPipeline* resolved = it->second.get();
        if (exec_entry) {
            live.stage_key_audits.fetch_add(1, std::memory_order_relaxed);
            if (exec_entry->pipeline != resolved || exec_entry->infos != infos || exec_entry->modules != modules) {
                live.stage_key_violations.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            live.stage_key_misses.fetch_add(1, std::memory_order_relaxed);
        }
        ExecEntry& entry = exec_pipelines[lookup_exec_key];
        entry.pipeline = resolved;
        entry.infos = infos;
        entry.modules = modules;
        entry.stage_hw = stage_hw;
        entry.pgm_base = stage_base;
        entry.fetch_program = fetch_shader != nullptr ? fetch_program : nullptr;
        entry.fetch_perm = fetch_perm;
        entry.epoch = exec_epoch;
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    Core::SplitCensus::Touch(Core::SplitCensus::Pipeline);
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::PipelineLookup};
    if (!lookup_pinned || lookup_cs == nullptr) {
        return nullptr; // G1: a dispatch is looked up from its plan's inputs only (it always has a plan)
    }
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = [&] {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::PipelineMap};
        return compute_pipelines.try_emplace(compute_key);
    }();
    if (is_new) {
        VideoCore::ReadbackStats::NoteMutation(
            VideoCore::ReadbackStats::Mutation::PipelineCompiled);
        VideoCore::ReadbackStats::CpTimer compile_timer{
            VideoCore::ReadbackStats::CpTime::PipelineCompile};
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

Shader::Info* PipelineCache::RasterInfoFor(const Shader::Info* shared) {
    auto it = raster_infos.find(shared);
    if (it == raster_infos.end()) {
        it = raster_infos.emplace(shared, std::make_unique<Shader::Info>(*shared)).first;
    }
    return it.value().get();
}

void PipelineCache::CapturePlan(StagePlan& out, Shader::Info* clone, const Shader::Info& shared, bool skip_flat) {
    out.info = clone;
    out.user_data_count = static_cast<u32>(shared.user_data.size());
    ASSERT(out.user_data_count <= out.user_data.size());
    std::copy_n(shared.user_data.begin(), out.user_data_count, out.user_data.begin());
    out.flat_skipped = skip_flat;
    if (skip_flat) {
        out.flat.clear();
    } else {
        out.flat.assign(shared.flattened_ud_buf.begin(), shared.flattened_ud_buf.end());
    }
    out.pgm_base = shared.pgm_base;
}

namespace {
/// N3b: true when every descriptor the stage's shader reads from the flat user data is a single load that the draw's record has a binding for (or is built from
/// immediates), so the flat user data is not needed: no flat buffer for the shader, no FMask, no sampler post operation that reads another descriptor.
/// N6.1 diagnostics: the first distinct loads of a stage that the record does not cover are logged with the record's list for that stage.
void LogUncovered(const Shader::Info& info, u32 record_stage, const Core::NativeDraw::ExecDraw& draw, u32 kind, u32 offset, u32 dwords) {
    static std::mutex mutex;
    static std::set<std::tuple<u64, u32, u32>> seen;
    static u32 logged = 0;
    std::scoped_lock lk{mutex};
    if (logged >= 80 || !seen.insert({info.pgm_hash, kind, offset}).second) {
        return;
    }
    ++logged;
    std::string list;
    for (u32 i = 0; i < draw.binding_count; ++i) {
        const auto& b = draw.bindings[i];
        if (b.stage == record_stage) {
            list += b.offset == 0xffff ? fmt::format(" {}:t{}[{}]", b.kind, b.table_sgpr, b.slot) : fmt::format(" {}:{}+{}", b.kind, b.offset, b.dwords);
        }
    }
    LOG_ERROR(Render_Vulkan, "N6.1 uncovered: shader {:#x} stage {} kind {} (1 buffer 2 image 3 sampler) offset {} dwords {}; the record lists (kind:offset+dwords){}", info.pgm_hash, record_stage, kind, offset, dwords, list);
}

/// 0 covered, else why not (Core::NativeRegistry::NoteBinderReason).
u32 StageCoverage(const Shader::Info& info, u32 record_stage, const Core::NativeDraw::ExecDraw& draw) {
    if (!draw.bindings_complete) {
        return 1;
    }
    if (info.has_flatbuf) {
        return 2;
    }
    if (!info.fmasks.empty()) {
        return 3;
    }
    const auto covered = [&](u32 kind, u32 offset, u32 dwords) {
        return FindRecordWords(info, draw.bindings.data(), draw.binding_count, record_stage, kind, offset, dwords) != nullptr;
    };
    for (const auto& desc : info.buffers) {
        if (desc.IsSpecial()) {
            continue;
        }
        if (desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::MultiLoad && desc.sharp_fetch.load_mask == 0) {
            continue; // built from immediates (and the program base)
        }
        if (desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
            return 4;
        }
        if (!covered(1, desc.sharp_fetch.offsets[0], 4)) {
            LogUncovered(info, record_stage, draw, 1, desc.sharp_fetch.offsets[0], 4);
            return 5;
        }
    }
    for (const auto& desc : info.images) {
        if (desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad) {
            return 6;
        }
        if (!covered(2, desc.sharp_fetch.offsets[0], desc.is_r128 ? 4 : 8)) {
            LogUncovered(info, record_stage, draw, 2, desc.sharp_fetch.offsets[0], desc.is_r128 ? 4 : 8);
            return 7;
        }
    }
    for (const auto& desc : info.samplers) {
        if (desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Sampler>::Summary::MultiLoad && desc.sharp_fetch.load_mask == 0 &&
            desc.post_op != Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
            continue; // F5: built from the shader's immediates, no descriptor word is read (as the buffers above)
        }
        if (desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Sampler>::Summary::SingleLoad) {
            return 8;
        }
        if (!covered(3, desc.sharp_fetch.offsets[0], 4)) {
            return 9;
        }
        if (desc.post_op == Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
            // L1: the post operation reads dword 3 of the T# the sampler is used with: the record's image entry gives it (a T# of 8 dwords, or a 4 dword one)
            const u32 off = desc.post_op_tsharp_dw3_off;
            if (off < 3 || off == Shader::UNKNOWN_LOCATION || (!covered(2, off - 3, 8) && !covered(2, off - 3, 4))) {
                return 11;
            }
        }
    }
    return 0;
}
} // namespace

/// The record's key and the parts of the fixed-function state that go into the pipeline key.
static u64 ExecKeyOf(u64 key, const FixedState& fixed) {
    u64 h = key;
    const auto mix = [&h](u64 value) { h = (h ^ value) * 0x100000001b3ull + (h >> 31); };
    mix(fixed.depth_clamp);
    mix(fixed.depth_clip);
    mix(fixed.negative_one_to_one);
    mix(fixed.provoking_first);
    mix(static_cast<u64>(fixed.primitive_type));
    mix(static_cast<u64>(fixed.polygon_mode));
    mix(static_cast<u64>(fixed.logic_op));
    mix(fixed.logic_op_enabled);
    mix(fixed.alpha_not_exported);
    mix(fixed.color_enabled);
    for (const u8 mask : fixed.target_write_mask) {
        mix(mask);
    }
    for (const auto& blend : fixed.blend) {
        u64 bytes[(sizeof(BlendTarget) + 7) / 8]{};
        std::memcpy(bytes, &blend, sizeof(BlendTarget));
        for (const u64 word : bytes) {
            mix(word);
        }
    }
    return h | 1;
}

bool PipelineCache::PlanGraphicsTyped(u64 exec_key, const FixedState& fixed, std::span<const u32, 16> vertex_user_data,
                                      std::span<const u32, 16> pixel_user_data, DrawPlan& plan, const Core::NativeDraw::ExecDraw* skip_flat_for, u32 draw_class) {
    const auto found = exec_pipelines.find(ExecKeyOf(exec_key, fixed));
    if (found == exec_pipelines.end() || found->second.epoch != exec_epoch) {
        return false;
    }
    const ExecEntry& entry = found->second;
    for (u32 s = 0; s < MaxShaderStages; ++s) {
        if (entry.infos[s] && entry.stage_hw[s] != static_cast<u32>(HwStage::Vertex) && entry.stage_hw[s] != static_cast<u32>(HwStage::Fragment)) {
            return false;
        }
    }
    infos = entry.infos;
    modules = entry.modules;
    fetch_shader = FetchOf(entry);
    std::array<bool, MaxShaderStages> skip_flat{};
    plan.any_flat_skipped = false;
    for (u32 s = 0; s < MaxShaderStages; ++s) {
        if (!infos[s]) {
            continue;
        }
        auto* info = const_cast<Shader::Info*>(infos[s]);
        info->pgm_base = entry.pgm_base[s];
        info->user_data = entry.stage_hw[s] == static_cast<u32>(HwStage::Vertex) ? vertex_user_data : pixel_user_data;
        if (skip_flat_for != nullptr) {
            const u32 record_stage = entry.stage_hw[s] == static_cast<u32>(HwStage::Vertex) ? 0 : 1;
            const u32 coverage = StageCoverage(*info, record_stage, *skip_flat_for);
            skip_flat[s] = coverage == 0;
            if (coverage != 0) {
                Core::NativeRegistry::NoteBinderReason(draw_class, record_stage, coverage);
            }
            plan.any_flat_skipped |= skip_flat[s];
            Core::NativeRegistry::NoteBinder(skip_flat[s] ? 9 : 10);
        }
        if (!skip_flat[s]) {
            info->RefreshFlatBuf();
            NoteFlat(*info);
        }
    }
    plan.pipeline = entry.pipeline;
    for (u32 i = 0; i < MaxShaderStages; ++i) {
        if (infos[i]) {
            CapturePlan(plan.stages[i], RasterInfoFor(infos[i]), *infos[i], skip_flat[i]);
        }
    }
    auto* mutable_pipeline = const_cast<GraphicsPipeline*>(entry.pipeline);
    if (!mutable_pipeline->StagesAdopted()) {
        std::array<const Shader::Info*, Shader::MaxStageTypes> clones{};
        for (u32 i = 0; i < MaxShaderStages; ++i) {
            clones[i] = plan.stages[i].info;
        }
        mutable_pipeline->AdoptStageInfos(clones);
    }
    return true;
}

void PipelineCache::PlanGraphics(const FixedState& fixed, DrawIndirectParams params, DrawPlan& plan, u64 exec_key) {
    lookup_inputs = &plan.inputs;
    lookup_fixed = &fixed;
    lookup_pinned = true;
    lookup_exec_key = exec_key != 0 ? ExecKeyOf(exec_key, fixed) : 0;
    const GraphicsPipeline* pipeline = GetGraphicsPipeline(params);
    lookup_exec_key = 0;
    lookup_pinned = false;
    lookup_inputs = nullptr;
    plan.pipeline = pipeline;
    if (!pipeline) {
        return;
    }
    for (u32 i = 0; i < MaxShaderStages; ++i) {
        if (infos[i]) {
            CapturePlan(plan.stages[i], RasterInfoFor(infos[i]), *infos[i]);
        }
    }
    auto* mutable_pipeline = const_cast<GraphicsPipeline*>(pipeline);
    if (!mutable_pipeline->StagesAdopted()) {
        std::array<const Shader::Info*, Shader::MaxStageTypes> clones{};
        for (u32 i = 0; i < MaxShaderStages; ++i) {
            clones[i] = plan.stages[i].info;
        }
        mutable_pipeline->AdoptStageInfos(clones);
    }
}

void PipelineCache::PlanCompute(const Core::NativeDispatch::DispatchInputs& cs, DispatchPlan& plan) {
    lookup_pinned = true;
    lookup_cs = &cs;
    const ComputePipeline* pipeline = GetComputePipeline();
    lookup_pinned = false;
    plan.pipeline = pipeline;
    if (!pipeline) {
        return;
    }
    CapturePlan(plan.stage, RasterInfoFor(infos[0]), *infos[0]);
    auto* mutable_pipeline = const_cast<ComputePipeline*>(pipeline);
    if (!mutable_pipeline->StagesAdopted()) {
        std::array<const Shader::Info*, Shader::MaxStageTypes> clones{};
        clones[u32(Shader::SwStage::Compute)] = plan.stage.info;
        mutable_pipeline->AdoptStageInfos(clones);
    }
}

void PipelineCache::ApplyPlan(const DrawPlan& plan) {
    for (const auto& stage : plan.stages) {
        if (!stage.info) {
            continue;
        }
        stage.info->user_data = std::span<const u32>{stage.user_data.data(), stage.user_data_count};
        stage.info->flattened_ud_buf.assign(stage.flat.begin(), stage.flat.end()); // empty for a skipped stage: nothing may read it
        stage.info->pgm_base = stage.pgm_base;
    }
}

void PipelineCache::ApplyPlan(const DispatchPlan& plan) {
    const auto& stage = plan.stage;
    stage.info->user_data = std::span<const u32>{stage.user_data.data(), stage.user_data_count};
    stage.info->flattened_ud_buf.assign(stage.flat.begin(), stage.flat.end());
    stage.info->pgm_base = stage.pgm_base;
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& fixed = *lookup_fixed;
    auto& key = graphics_key;

    auto& live = Core::NativeDraw::Live();
    if (lookup_inputs == nullptr) {
        return false; // F3.5: the key's targets are the plan's inputs (the native target description)
    }
    live.key_targets[0].fetch_add(1, std::memory_order_relaxed);
    const Core::NativeTarget::DepthTarget& depth_buffer = lookup_inputs->depth_buffer;
    const Core::NativeTarget::ColorTarget* color_buffers = lookup_inputs->color_buffers;
    // the export format of each target is the pixel shader's (SPI_SHADER_COL_FORMAT): from its environment
    Core::NativeShaderEnv::DrawEnv shader_env;
    const bool have_env = Core::NativeShaderEnv::Current(shader_env);
    const bool export_from_env = have_env && shader_env.has_pixel;

    const bool db_enabled = depth_buffer.DepthValid() || depth_buffer.StencilValid();

    key.z_format = depth_buffer.DepthValid() ? depth_buffer.z_info.format
                                             : Core::NativeTarget::DepthTarget::ZFormat::Invalid;
    key.stencil_format = depth_buffer.StencilValid()
                             ? depth_buffer.stencil_info.format
                             : Core::NativeTarget::DepthTarget::StencilFormat::Invalid;
    key.depth_clamp_enable = fixed.depth_clamp;
    key.depth_clip_enable = fixed.depth_clip;
    key.negative_one_to_one = fixed.negative_one_to_one;
    key.provoking_first = fixed.provoking_first;
    key.prim_type = fixed.primitive_type;
    key.polygon_mode = fixed.polygon_mode;
    key.patch_control_points = 0; // F3.2: no hull shader in a native draw's pipeline
    key.logic_op = fixed.logic_op;
    key.logic_op_enable = fixed.logic_op_enabled;
    key.depth_samples = db_enabled ? depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.alpha_not_exported = fixed.alpha_not_exported;

    const bool skip_cb_binding = !fixed.color_enabled;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = color_buffers[cb];
        if (!col_buf || !fixed.target_write_mask[cb]) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        if (export_from_env) {
            ShaderBoundary::SetExportFormat(color_buffer, shader_env.pixel.fragment.color_export_format[cb]);
            live.key_targets[2].fetch_add(1, std::memory_order_relaxed);
        } else {
            ShaderBoundary::SetNoExport(color_buffer); // no pixel shader: nothing is exported
        }
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = fixed.blend[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_op == vk::BlendOp::eMin || bc.color_op == vk::BlendOp::eMax) &&
            bc.src_color == vk::BlendFactor::eSrcColor &&
            bc.dst_color == vk::BlendFactor::eDstColor;
    }

    // Compile and bind shader stages
    const bool stages_ok = [&] {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::PipelineStages};
        return RefreshGraphicsStages();
    }();
    if (!stages_ok) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = color_buffers[cb];
        const u32 target_mask = fixed.target_write_mask[cb];
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (fixed.blend[cb].enable && !col_buf.info.blend_bypass) {
            key.blend[cb] = fixed.blend[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    auto& key = graphics_key;
    fetch_shader = nullptr;
    fetch_program = nullptr;

    // F3.5: the stage set of a native draw is the vertex shader plus the pixel shader when the record names one (the shader objects the record names); a hull, domain or geometry shader
    // in the device context refuses the draw, and so does a draw without a shader environment
    Core::NativeShaderEnv::DrawEnv stage_env;
    if (lookup_inputs == nullptr || !Core::NativeShaderEnv::Current(stage_env) || !stage_env.has_vertex || stage_env.other_stage) {
        return false;
    }
    auto& live = Core::NativeDraw::Live();
    live.key_targets[4].fetch_add(1, std::memory_order_relaxed);

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        // F3.2b: the program and the user data words from the shader environment and the record
        const auto env_params = ParamsFromEnv(stage_env, stage_in_idx);
        live.key_targets[env_params ? 8 : 9].fetch_add(1, std::memory_order_relaxed);
        if (!env_params) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }
        const auto& params = *env_params;
        NoteCode(params);
        stage_hw[stage_out_idx] = stage_in_idx;
        stage_base[stage_out_idx] = params.Base();
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    if (stage_env.has_pixel) { // a draw without a pixel shader (depth only) has no fragment stage
        bind_stage(HwStage::Fragment, SwStage::Fragment);
    }

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    bind_stage(HwStage::Vertex, SwStage::Vertex);

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            NoteFetch(attrib, *vs_info);
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = *lookup_cs;
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    NoteCode(cs_params);
    std::tie(infos[0], modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");
    // tlg C9: graphics shaders find their guest buffers through the indexed data table
    info.indexed = profile.indexed_resources;

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");


    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    const Shader::RuntimeInfo& runtime_info = [&]() -> const Shader::RuntimeInfo& {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::StageRuntimeInfo};
        return BuildRuntimeInfo(hw_stage, sw_stage);
    }();
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(hw_stage, sw_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        auto compile_info = runtime_info;
        const auto module = CompileModule(program->info, compile_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, compile_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        if (auto& fetch = program->modules[0].spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
            fetch_program = program.get();
            fetch_perm = 0;
        }
        program->last_perm = 0;
        return std::make_tuple(&program->info, module, perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::StageFlatBuf};
        info.RefreshFlatBuf();
        NoteFlat(info);
    }

    // Most lookups pick the same variant as the previous one: check it first, without building
    // a new specialization (stage_lookup_reuse).
    size_t fast_perm = ~size_t{0};
    const u32 stage_reuse = [&] {
        const s32 override_mode = VideoCore::ReadbackPolicy::Get().stage_reuse_override.load(std::memory_order_relaxed);
        return override_mode >= 0 ? static_cast<u32>(override_mode) : stage_lookup_reuse;
    }();
    if (stage_reuse != 0 && program->last_perm < program->modules.size()) {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::StageSpec};
        if (program->modules[program->last_perm].spec.Matches(info, runtime_info, binding)) {
            fast_perm = program->last_perm;
            VideoCore::ReadbackStats::OnReuse(VideoCore::ReadbackStats::Reuse::StageFastPath);
        }
    }
    if (fast_perm != ~size_t{0} && stage_reuse != 2) {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::StageFind};
        info.AddBindings(binding);
        const auto& found = program->modules[fast_perm];
        if (auto& fetch = program->modules[fast_perm].spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
            fetch_program = program.get();
            fetch_perm = fast_perm;
        }
        return std::make_tuple(&info, found.module, HashCombine(params.hash, fast_perm));
    }

    auto spec = [&] {
        VideoCore::ReadbackStats::CpTimer t{VideoCore::ReadbackStats::CpTime::StageSpec};
        return Shader::StageSpecialization(info, runtime_info, profile, binding);
    }();

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    VideoCore::ReadbackStats::CpTimer find_timer{VideoCore::ReadbackStats::CpTime::StageFind};
    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (VideoCore::ReadbackStats::IsEnabled()) {
        using VideoCore::ReadbackStats::Reuse;
        VideoCore::ReadbackStats::OnReuse(Reuse::StageLookups);
        const auto compared =
            it == program->modules.end()
                ? program->modules.size()
                : static_cast<size_t>(std::distance(program->modules.begin(), it)) + 1;
        VideoCore::ReadbackStats::OnReuse(Reuse::StageVariantCompares, compared);
        const auto found = it == program->modules.end()
                               ? program->modules.size()
                               : static_cast<size_t>(std::distance(program->modules.begin(), it));
        if (found == program->last_perm) {
            VideoCore::ReadbackStats::OnReuse(Reuse::StageSameAsLast);
        }
    }
    if (fast_perm != ~size_t{0}) {
        // Verify mode: the fast check must agree with the full lookup.
        const size_t full_perm =
            it == program->modules.end()
                ? program->modules.size()
                : static_cast<size_t>(std::distance(program->modules.begin(), it));
        if (full_perm != fast_perm) {
            VideoCore::ReadbackStats::OnReuse(
                VideoCore::ReadbackStats::Reuse::StageFastPathMismatch);
            if (logged_stage_mismatches.insert(params.hash).second) {
                LOG_WARNING(Render_Vulkan,
                            "stage_lookup_reuse mismatch: {} shader {:#x} fast variant {} full "
                            "lookup {}{}",
                            hw_stage, params.hash, fast_perm, full_perm,
                            it == program->modules.end() ? " (new variant)" : "");
            }
        }
    }
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        auto compile_info = runtime_info;
        module = CompileModule(new_info, compile_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
        fetch_program = program.get();
        fetch_perm = perm_idx;
    }
    program->last_perm = perm_idx;
    return std::make_tuple(&program->info, module, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        exec_pipelines.clear();
        ++exec_epoch;
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
