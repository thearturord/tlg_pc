// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "native/native_raster_host.h"
#include "native/native_binding.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw.h"
#include "native/native_plan.h"
#include "native/native_resources.h"
#include "video_core/amdgpu/regs.h"
#include <functional>
#include <memory>
#include <variant>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_indexed_data.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace Core {
class NativeExecutor;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    size_t last_perm{~size_t{0}}; ///< Variant picked by the last lookup.

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

using DrawIndirectParams = Core::DrawIndirectParams;

using Core::NativePlan::RingBinding;
using Core::NativePlan::GeometryVertex;
using Core::NativePlan::GeometryIndex;
using Core::NativePlan::ResolvedBuffer;
using Core::NativePlan::StagePlan;
using Core::NativePlan::DrawPlan;
using Core::NativePlan::DispatchPlan;
static_assert(Core::NativePlan::UserDataRegs == Shader::NUM_USER_DATA_REGS && Core::NativePlan::StageCount == MaxShaderStages);

/// N6.1: the descriptor words the record holds for the shader load at flat user data `offset` of `stage_s` (0 vertex, 1 pixel): a descriptor put into the user data directly
/// (`offset` inside the binding), or an entry of a descriptor table behind a user data register, found through the walker's table map of the shader. Null: not in the record.
inline const u32* FindRecordWords(const Shader::Info& info, const Core::NativeDraw::RecordBinding* list, u32 count, u32 stage_s, u32 kind, u32 offset, u32 dwords, u32* index_out = nullptr, bool use_hints = true) {
    u32 root = 0, src = 0;
    const bool table = info.srt_info.TableSource(offset, root, src);
    const auto match = [&](u32 i) -> const u32* {
        const auto& b = list[i];
        if (b.stage != stage_s || b.kind != kind) {
            return nullptr;
        }
        if (b.entry_dwords == 0xfe && b.offset != 0xffff) {
            // an async context entry: a tagged engine offset. Only an entry in a user data register (bit 15) answers a direct load of that register; a dword index into the staging table
            // (no bit 15) is the same number in another namespace and must not answer a register load (L2: the constant buffer at table index 8 was taken for the shader resource in register 8)
            if ((b.offset & 0x8000) == 0) {
                // a dword index into the staging table: answers a table load through the register that points to the table (table_sgpr, which starts at table_base in the staging table)
                const u32 index = b.offset & 0x3fff;
                const u32 absolute = src + b.table_base;
                return table && b.table_sgpr == root && absolute >= index && absolute + dwords <= index + b.dwords ? b.words + (absolute - index) : nullptr;
            }
            const u32 reg = b.offset & 0x3fff;
            return !table && offset >= reg && offset + dwords <= reg + b.dwords ? b.words + (offset - reg) : nullptr;
        }
        if (b.offset != 0xffff) {
            return offset >= b.offset && offset + dwords <= u32{b.offset} + b.dwords ? b.words + (offset - b.offset) : nullptr;
        }
        if (table && b.table_sgpr == root && b.entry_dwords != 0 && b.slot == src / b.entry_dwords) {
            const u32 within = src % b.entry_dwords;
            return within + dwords <= b.dwords ? b.words + within : nullptr;
        }
        return nullptr;
    };
    // The same shader asks for the same load in every draw and the record builder lists its descriptors in the same order: the index that served the last such lookup is tried first.
    static thread_local std::array<u16, 1024> hints{};
    u16& hint = hints[(((reinterpret_cast<u64>(&info) >> 4) * 0x9E3779B1u) ^ (offset * 131u) ^ (kind * 7u) ^ (stage_s * 3u)) & 1023];
    if (use_hints && hint < count) {
        if (const u32* words = match(hint)) {
            if (index_out != nullptr) {
                *index_out = hint;
            }
            return words;
        }
    }
    for (u32 i = 0; i < count; ++i) {
        if (const u32* words = match(i)) {
            if (use_hints) {
                hint = static_cast<u16>(i);
            }
            if (index_out != nullptr) {
                *index_out = i;
            }
            return words;
        }
    }
    return nullptr;
}

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           Core::NativeExecutor* executor, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();
    /// tlg C9: the indexed data path (null: off)
    [[nodiscard]] IndexedData* GetIndexedData() const noexcept {
        return indexed_data.get();
    }

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    /// cp_split_mode: looks the pipeline up for the command processor's registers `regs` and
    /// captures the per-draw state of its stages. Runs on the command processor thread, the only
    /// thread that touches this cache then.
    void PlanGraphics(const FixedState& fixed, DrawIndirectParams params, DrawPlan& plan, u64 exec_key = 0);
    /// cp_exec_inputs: the same for a draw whose pipeline was resolved before under `exec_key` (the record's key): the stages take the typed user data, nothing is read from
    /// registers. False when there is no entry (the caller plans the old way).
    /// skip_flat_for (N3b): the draw's record; a stage whose every descriptor it covers gets no flat user data.
    bool PlanGraphicsTyped(u64 exec_key, const FixedState& fixed, std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data, DrawPlan& plan,
                           const Core::NativeDraw::ExecDraw* skip_flat_for = nullptr, u32 draw_class = 15);
    void PlanCompute(const Core::NativeDispatch::DispatchInputs& cs, DispatchPlan& plan);
    /// Raster thread: writes the captured state into the stage Infos the pipeline uses.
    static void ApplyPlan(const DrawPlan& plan);
    static void ApplyPlan(const DispatchPlan& plan);


    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    Shader::Info* RasterInfoFor(const Shader::Info* shared);
    static void CapturePlan(StagePlan& out, Shader::Info* clone, const Shader::Info& shared, bool skip_flat = false);
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    Core::NativeExecutor* executor;
    DescriptorHeap desc_heap;
    std::unique_ptr<IndexedData> indexed_data; ///< tlg C9: null unless TLG_INDEXED_DATA=1 and the device can do it
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    /// F3.5: the plan's inputs while `PlanGraphics` looks the pipeline up; the key's render targets are read from them
    const DrawInputs* lookup_inputs{};
    const FixedState* lookup_fixed{};
    FixedState lookup_fixed_storage{};
    const Core::NativeDispatch::DispatchInputs* lookup_cs{};
    bool lookup_pinned{}; ///< The lookup reads lookup_inputs/lookup_fixed/lookup_cs as set by the caller.
    /// cp_split_mode: the raster thread's copy of each program's Info.
    tsl::robin_map<const Shader::Info*, std::unique_ptr<Shader::Info>> raster_infos;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    /// cp_stage_key: what a draw with a known record key resolves to (see GetGraphicsPipeline).
    struct ExecEntry {
        const GraphicsPipeline* pipeline{};
        std::array<const Shader::Info*, MaxShaderStages> infos{};
        std::array<vk::ShaderModule, MaxShaderStages> modules{};
        std::array<u32, MaxShaderStages> stage_hw{};
        std::array<VAddr, MaxShaderStages> pgm_base{};
        /// the fetch shader as (program, variant): a pointer into the variant list would move when a variant is added
        Program* fetch_program{};
        size_t fetch_perm{};
        u64 epoch{};
    };
    /// the fetch shader of an entry, resolved at use
    [[nodiscard]] static Shader::Gcn::FetchShaderData* FetchOf(const ExecEntry& entry) {
        return entry.fetch_program != nullptr ? &entry.fetch_program->modules[entry.fetch_perm].spec.fetch_shader_data : nullptr;
    }
    tsl::robin_map<u64, ExecEntry> exec_pipelines;
public:
    /// Note 182: the counters of the plan-local capture (`TLG_PURE_PLAN`).
    static nlohmann::json PurePlanReport();
private:
    static void CapturePlanPure(StagePlan& out, Shader::Info* clone, const Shader::Info& shared, bool skip_flat, std::span<const u32, 16> user_data, u64 pgm_base);
    std::array<u32, MaxShaderStages> stage_hw{};
    std::array<VAddr, MaxShaderStages> stage_base{};
    u64 lookup_exec_key{};
    u64 exec_epoch{1}; ///< bumped when a shader is replaced (debug); adding a variant no longer invalidates the entries (2026-10-28: they hold
                       ///< the fetch shader as program and variant index, every other pointer is stable)
    Program* fetch_program{}; ///< the program and variant of `fetch_shader` (set together with it)
    size_t fetch_perm{};
    u32 exec_tick{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{};      // new pipelines added to the cache since the game start
    const u32 stage_lookup_reuse; // 0 = off, 1 = check the last variant first, 2 = verify
    tsl::robin_set<u64> logged_stage_mismatches;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
