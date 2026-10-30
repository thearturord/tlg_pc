// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "native/native_binding.h"
#include "native/native_draw.h"
#include "native/native_resources.h"
#include "video_core/amdgpu/draw_state.h"
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

namespace AmdGpu {
class Liverpool;
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

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

/// Per-draw state of one shader stage, captured by the command processor thread when it looks the
/// pipeline up (cp_split_mode). `info` is the raster thread's own copy of the stage's Info: the
/// shared one is rewritten by the next lookup while this draw is still being recorded.
/// cp_split_prepare_images: the pure part of one image resource of a stage, decoded on the command
/// processor thread from the plan's own copy of the user data.
struct PreparedImage {
    AmdGpu::Image tsharp{};
    u32 num_bindings{1};
    boost::container::small_vector<VideoCore::ImageInfo, 1> infos;
    boost::container::small_vector<VideoCore::ImageViewInfo, 1> views;
};

/// A read-only buffer of a stage that the command processor thread copied into the constant ring (documents/tlg/pc-port-draw-pipeline-architecture.md, T1).
struct RingBinding {
    u32 index{};      ///< index in the stage's buffer list
    u32 size{};
    u64 offset{};     ///< offset in the ring buffer
    VAddr address{};  ///< the guest address it was copied from (0 for the flattened user data)
    bool snapshot{};  ///< N5d: the bytes are the writer's snapshot, not what guest memory held when the command processor looked
    bool formatted{}; ///< N5f: a formatted (texel) buffer: the raster thread checks at bind time that no image or metadata surface covers the range
};

constexpr u32 GeometryVertex = 0xFFFFFFFDu; ///< RingBinding::index of a vertex range in DrawPlan::geometry_ring
constexpr u32 GeometryIndex = 0xFFFFFFFEu;  ///< and of the index range

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

/// R2 (native back end): a shader buffer binding the command processor resolved from the record's engine object: the raster thread binds it from the object's
/// arena when the object is synchronized at the stamped generation (no buffer cache request, no record lookup). The stamp is taken when the command processor
/// plans the operation, after the game wrote the frame's data: an object the raster thread synchronizes at its first use in the frame serves the later uses.
struct ResolvedBuffer {
    Core::NativeResources::ImmutableBuffer* object{}; ///< null: not resolved (the raster thread takes the full path)
    VAddr address{};
    u32 size{};
    u32 generation{};
    bool no_object{}; ///< the lookup by address found no engine object (the raster thread need not look again while the immutable epoch holds)
};

struct StagePlan {
    boost::container::small_vector<RingBinding, 8> ring; ///< T1: filled by the command processor thread, read by the raster thread
    boost::container::small_vector<ResolvedBuffer, 8> resolved; ///< R2: by the stage's buffer index (empty: none resolved)
    u64 resolved_epoch{}; ///< R2: the immutable epoch the objects were looked up at
    /// R3 step 2: by the stage's image index, where the record holds the image's descriptor words: (binding index << 8) | dword offset in it,
    /// ImageNoEntry (the record has none) or ImageUnknown (not resolved: the raster thread looks itself)
    boost::container::small_vector<u32, 8> image_words;
    static constexpr u32 ImageUnknown = ~0u, ImageNoEntry = ~0u - 1;
    Shader::Info* info{};
    boost::container::small_vector<PreparedImage, 8> images;
    bool images_prepared{};
    std::array<u32, Shader::NUM_USER_DATA_REGS> user_data{};
    u32 user_data_count{};
    boost::container::small_vector<u32, 64> flat;
    bool flat_skipped{}; ///< N3b: every descriptor of the stage comes from the record's binding list; the flat user data was not built (flat is empty)
    VAddr pgm_base{};
};

struct DrawPlan {
    const GraphicsPipeline* pipeline{}; ///< Null if the pipeline lookup failed.
    FixedState fixed{};
    DrawInputs inputs{};
    bool executed{}; ///< the draw was executed from a record: no register based classification
    u8 op_class{0xff}; ///< not 0xff: one of the game's own transition operations (Rasterizer::DrawClass) that the raster thread performs from this plan, no pipeline
    bool depth_copy{}, stencil_copy{};
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent{}; ///< the extent hints of the targets when the draw was recorded
    AmdGpu::CbDbExtent db_extent{};
    std::array<StagePlan, MaxShaderStages> stages{};
    u64 ring_end{}; ///< T1: the constant ring position after this plan's copies
    u64 native_seq{}; ///< the record the plan was made from (0: planned from the registers); diagnostics
    u32 ww_epoch{};   ///< O1: the write watch epoch the command processor read before it harvested the plan's ranges (0: not harvested ahead)
    u32 ww_base_epoch{}; ///< P3: the base epoch read before the plan's images' shared pages were harvested (0: not harvested ahead)
    /// NativeDraw v2: the descriptors of the record in the user data of its stages; the native binder reads them by (stage, flat offset)
    std::array<Core::NativeDraw::RecordBinding, Core::NativeDraw::MaxRecordBindings> bindings{};
    u32 binding_count{};
    bool bindings_complete{};
    bool any_flat_skipped{}; ///< N3b: a stage of the plan has no flat user data, the binder must serve the draw
    /// N3c: the immutable engine buffer of each kind 1 binding, resolved by the engine's id when the plan was made (null: none); valid while `handle_epoch` is the
    /// current `Core::NativeResources::ImmutableEpoch()`
    std::array<Core::NativeResources::ImmutableBuffer*, Core::NativeDraw::MaxRecordBindings> handles{};
    u64 handle_epoch{};
    bool has_handles{};
    /// N6.2: the immutable engine buffers named by the record's vertex and index buffer ids (valid while `handle_epoch` is the immutable epoch)
    std::array<Core::NativeResources::ImmutableBuffer*, Core::NativeDraw::MaxGeometryIds> geometry_handles{};
    u32 geometry_handle_count{};
    bool has_geometry_handles{};
    /// N5g: vertex ranges (`index` = GeometryVertex) and the index range (`GeometryIndex`) the command processor copied into the constant ring
    boost::container::small_vector<RingBinding, 4> geometry_ring;
};

struct DispatchPlan {
    const ComputePipeline* pipeline{}; ///< Null if the pipeline lookup failed.
    AmdGpu::ComputeProgram cs{};       ///< The compute program of this dispatch: the rasterizer reads it from here.
    AmdGpu::BorderColorBuffer ta_bc_base{}; ///< The custom border colour table the dispatch's samplers use.
    StagePlan stage{};
    u64 ring_end{}; ///< T1: the constant ring position after this plan's copies
    u64 native_seq{}; ///< B2: the record this dispatch was planned from (the registry audit names its resources by it)
    u32 ww_epoch{};   ///< O1: see DrawPlan::ww_epoch
    u32 ww_base_epoch{}; ///< P3: see DrawPlan::ww_base_epoch
    /// B9: the typed bindings of that record (the descriptors in the dispatch's user data, with the engine ids); complete when none was dropped
    std::array<Core::NativeDraw::RecordBinding, Core::NativeDraw::MaxRecordBindings> bindings{};
    u32 binding_count{};
    bool bindings_complete{};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
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
    void PlanGraphics(const AmdGpu::DrawRegs& regs, const FixedState& fixed,
                      DrawIndirectParams params, DrawPlan& plan, u64 exec_key = 0);
    /// cp_exec_inputs: the same for a draw whose pipeline was resolved before under `exec_key` (the record's key): the stages take the typed user data, nothing is read from
    /// registers. False when there is no entry (the caller plans the old way).
    /// skip_flat_for (N3b): the draw's record; a stage whose every descriptor it covers gets no flat user data.
    bool PlanGraphicsTyped(u64 exec_key, const FixedState& fixed, std::span<const u32, 16> vertex_user_data, std::span<const u32, 16> pixel_user_data, DrawPlan& plan,
                           const Core::NativeDraw::ExecDraw* skip_flat_for = nullptr, u32 draw_class = 15);
    void PlanCompute(const AmdGpu::ComputeProgram& cs, DispatchPlan& plan);
    /// Raster thread: writes the captured state into the stage Infos the pipeline uses.
    static void ApplyPlan(const DrawPlan& plan);
    static void ApplyPlan(const DispatchPlan& plan);

    /// Plan statistics since the last call: stage inputs compared with the previous draw.
    struct PlanStats {
        u64 draws{};
        u64 draws_all_stages_same{};
        u64 stages{};
        u64 stages_same{};
    };
    PlanStats TakePlanStats() noexcept {
        return std::exchange(plan_stats, PlanStats{});
    }

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
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    std::unique_ptr<IndexedData> indexed_data; ///< tlg C9: null unless TLG_INDEXED_DATA=1 and the device can do it
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    /// Registers the current lookup reads: the raster mirror normally, the command processor's
    /// own registers (and the compute state of the dispatching queue) when planning.
    const AmdGpu::DrawRegs* lookup_regs{};
    const FixedState* lookup_fixed{};
    FixedState lookup_fixed_storage{};
    const AmdGpu::ComputeProgram* lookup_cs{};
    PlanStats plan_stats{};
    std::array<u64, MaxShaderStages> last_stage_hash{};
    bool lookup_regs_pinned{}; ///< The lookup reads lookup_regs/lookup_cs as set by the caller.
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
        u32 stage_enable{};
        u64 epoch{};
    };
    /// the fetch shader of an entry, resolved at use
    [[nodiscard]] static Shader::Gcn::FetchShaderData* FetchOf(const ExecEntry& entry) {
        return entry.fetch_program != nullptr ? &entry.fetch_program->modules[entry.fetch_perm].spec.fetch_shader_data : nullptr;
    }
    tsl::robin_map<u64, ExecEntry> exec_pipelines;
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
