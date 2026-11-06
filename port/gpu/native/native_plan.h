// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// G5c: the per-draw and per-dispatch plans the command processor thread makes and the raster thread executes (moved out of vk_pipeline_cache.h). Plain data: the pipelines and the
// shader `Info` are only pointers (declared, not defined, here); the fixed state, the draw inputs and the extent hints are the by-design headers of the allowlist.
//
// `PlanPool` keeps the finished plans for reuse; `PlanDigest` is the hash of what a plan holds (diagnostics and the plan cache check).

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "native/native_binding.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw.h"
#include "native/native_resources.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Shader {
struct Info;
}

namespace Vulkan {
class GraphicsPipeline;
class ComputePipeline;
} // namespace Vulkan

namespace Core::NativePlan {

using Vulkan::DrawInputs;
using Vulkan::FixedState;
using Vulkan::ComputePipeline;
using Vulkan::GraphicsPipeline;

/// Mirrors of `Shader::NUM_USER_DATA_REGS` and `Vulkan::MaxShaderStages` (the host checks them with static_asserts in vk_pipeline_cache.h).
constexpr u32 UserDataRegs = 16;
constexpr u32 StageCount = 6;

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

/// Per-draw state of one shader stage, captured by the command processor thread when it looks the
/// pipeline up (cp_split_mode). `info` is the raster thread's own copy of the stage's Info: the
/// shared one is rewritten by the next lookup while this draw is still being recorded.
struct StagePlan {
    boost::container::small_vector<RingBinding, 8> ring; ///< T1: filled by the command processor thread, read by the raster thread
    boost::container::small_vector<ResolvedBuffer, 8> resolved; ///< R2: by the stage's buffer index (empty: none resolved)
    u64 resolved_epoch{}; ///< R2: the immutable epoch the objects were looked up at
    /// R3 step 2: by the stage's image index, where the record holds the image's descriptor words: (binding index << 8) | dword offset in it,
    /// ImageNoEntry (the record has none) or ImageUnknown (not resolved: the raster thread looks itself)
    boost::container::small_vector<u32, 8> image_words;
    static constexpr u32 ImageUnknown = ~0u, ImageNoEntry = ~0u - 1;
    Shader::Info* info{};
    std::array<u32, UserDataRegs> user_data{};
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
    std::array<u64, 9> target_resources{}; ///< E3: engine resource ids behind the bound color views and the depth view when the draw was recorded
    std::array<StagePlan, StageCount> stages{};
    u64 ring_end{}; ///< T1: the constant ring position after this plan's copies
    u64 native_seq{}; ///< the record the plan was made from (0: planned from the registers); diagnostics
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
    Core::NativeDispatch::DispatchInputs cs{};  ///< The compute program of this dispatch: the rasterizer reads it from here.
    AmdGpu::BorderColorBuffer ta_bc_base{}; ///< The custom border colour table the dispatch's samplers use.
    StagePlan stage{};
    u64 ring_end{}; ///< T1: the constant ring position after this plan's copies
    u64 native_seq{}; ///< B2: the record this dispatch was planned from (the registry audit names its resources by it)
    /// B9: the typed bindings of that record (the descriptors in the dispatch's user data, with the engine ids); complete when none was dropped
    std::array<Core::NativeDraw::RecordBinding, Core::NativeDraw::MaxRecordBindings> bindings{};
    u32 binding_count{};
    bool bindings_complete{};
};


/// Hash of what a plan holds (pipeline, class, per stage: info, program base, user data, flat user data).
[[nodiscard]] u64 PlanDigest(const DrawPlan& plan);
[[nodiscard]] u64 PlanDigest(const DispatchPlan& plan);
/// Back to the empty state (kept capacity).
void ResetStage(StagePlan& stage);

/// Counters of the constant ring front (command processor thread) and of its use (raster thread); `RingUse::Take` reads and clears them.
struct RingStats {
    u64 queued{}, written_buffers{}, offered{}, taken{}, bytes{};
    u64 no_sharp{}, written{}, formatted{}, too_large{}, immutable{}, gpu_modified{}, pending_write{}, ring_full{};
    u64 bound{};
    u64 verified{}, differing{};
};

/// Raster thread: the plan the next draw or dispatch runs from, for its ring bindings (set before recording, cleared after), and the ring counters.
struct RingUse {
    const DrawPlan* draw = nullptr;
    const DispatchPlan* dispatch = nullptr;
    std::atomic<u64> queued{0}, written_buffers{0}, offered{0}, taken{0}, bytes{0};
    std::atomic<u64> no_sharp{0}, written{0}, formatted{0}, too_large{0}, immutable{0}, gpu_modified{0}, pending_write{0}, full{0}, bound{0};

    void Begin(const DrawPlan* d, const DispatchPlan* c) noexcept {
        draw = d;
        dispatch = c;
    }
    /// The ring binding of the buffer `index` of the stage whose Info is `info` in the current plan (null: none).
    [[nodiscard]] const RingBinding* Find(const Shader::Info* info, u32 index) const noexcept;
    /// The ring copy of a vertex or index range of the current draw (null: none).
    [[nodiscard]] const RingBinding* FindGeometry(VAddr address, u64 size, u32 kind) const noexcept;
    /// The ring position after the current plan's copies (0: none); clears the current plan.
    u64 End() noexcept;
    [[nodiscard]] RingStats Take() noexcept;
};

/// Finished plans, reused by the next planning (thread safe: the raster thread gives back, the command processor thread takes).
class PlanPool {
public:
    [[nodiscard]] std::unique_ptr<DrawPlan> TakeDraw();
    [[nodiscard]] std::unique_ptr<DispatchPlan> TakeDispatch();
    void Recycle(std::unique_ptr<DrawPlan>&& plan);
    void Recycle(std::unique_ptr<DispatchPlan>&& plan);

private:
    std::mutex mutex;
    std::vector<std::unique_ptr<DrawPlan>> draws;
    std::vector<std::unique_ptr<DispatchPlan>> dispatches;
};

} // namespace Core::NativePlan
