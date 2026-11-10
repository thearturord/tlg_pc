// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <thread>
#include "common/arch.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/debug.h"
#include "video_core/renderer_vulkan/row_share.h"
#include "native/native_debug_server.h"
#include "native/native_identity_log.h"
#include "native/native_image_caller.h"
#include "native/native_content.h"
#include "native/native_n0_census.h"
#include "native/native_row_version.h"
#include "native/native_dispatch_split.h"
#include "native/native_own_memory_census.h"
#include "native/native_legacy.h"
#include "native/native_view_table.h"
#include "native/native_texture_version.h"
#include "native/native_prof.h"
#include "native/native_marker.h"
#include "native/native_ui_trace.h"
#include "native/native_config.h"
#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_draw_inputs.h"
#include "native/native_target_desc.h"
#include "native/native_light_recorder.h"
#include "native/native_untracked_ring.h"
#include "native/native_readbacks.h"
#include "native/native_device.h"
#include "native/native_image_desc.h"
#include "native/native_resources.h"
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
#include "native/native_executor.h"
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
#include "video_core/half_phase.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer_engine.h"
#include "native/native_census.h"
#include "native/native_split_census.h"
#include "native/native_hook_ring.h"

namespace {
thread_local u32 t_cover_class = 15; // R-slice census: the draw class of the draw being timed
thread_local u64 t_w1_draw_begin = 0, t_w1_bind_begin = 0, t_w1_bind_end = 0; // W1.2 census: the clock at the start of Draw(), of BindResources and at its end (ns)

/// W1.0: what one draw bound, as the raster thread sees it; filled while the draw is built, consumed at the end of Draw()
struct W1Capture {
    u64 pipeline{};
    u64 depth{};                 ///< depth target address (the pass)
    u64 state{};                 ///< viewport, scissor, depth target, draw class hash
    u64 push{};                  ///< push constants hash
    u64 vertex{};                ///< hash of the vertex buffers (handle, offset)
    u64 index{};                 ///< hash of the index buffer (handle, offset)
    u64 index_handle{};          ///< the index buffer's Vulkan handle alone
    std::vector<u64> other_handles; ///< the Vulkan handle alone of every other shader buffer
    u32 index_count{}, instances{};
    u32 images{};
    std::vector<u64> ring;       ///< constant ring windows (offset, range)
    std::vector<u64> other;      ///< every other shader buffer (handle, offset, range)
    u32 bad{};                   ///< W1.1: what this draw needed besides lookups (W1Bad bits)
    u64 shape{};                 ///< W1.2a: everything the draw bound except the ring offsets and the index range (what a replay of an earlier draw's bind could reuse)
    bool valid{false};
};
thread_local W1Capture t_w1;
/// W1.1: why a shadow draw is not a pure lookup draw (a bit per reason; a draw is eligible when none is set)
enum W1Bad : u32 {
    W1BadImage = 1u << 0,      ///< an image is bound (W2 takes it)
    W1BadWritten = 1u << 1,    ///< a shader buffer the draw writes
    W1BadAddress = 1u << 2,    ///< a shader buffer found by address (no engine handle)
    W1BadStream = 1u << 3,     ///< a shader buffer answered from the stream copy ring
    W1BadWork = 1u << 4,       ///< a shader buffer request uploaded bytes
    W1BadFlat = 1u << 5,       ///< flat user data / clip planes copy (the front would produce them into the ring)
    W1BadGeoAddress = 1u << 6, ///< vertex or index buffer found by address
    W1BadGeoStream = 1u << 7,  ///< vertex or index buffer answered from the stream copy ring
    W1BadGeoWork = 1u << 8,    ///< vertex or index request uploaded bytes
    W1BadInPlace = 1u << 9,    ///< bound in place from the imported guest memory window (a lookup by address)
    W1BadImageWork = 1u << 10, ///< an image binding searched (FindImage), was a storage image or needed a layout transition
};
struct W1Row {
    const void* buffer{};
    u64 base{}; ///< guest address of byte 0 of the buffer
    u64 epoch{};
};
thread_local std::unordered_map<u64, W1Row> t_w1_rows; ///< W1.1 shadow table: engine id -> where the old path put it
/// W3.1: what the stamps of the id table say before the old path answers a buffer request
struct W31Pre {
    bool have_row{false}, epoch_ok{false}, dirty{false}, predicted{false};
    const void* row_buffer{};
    u64 row_base{};
    u64 commands{};
};
thread_local W31Pre t_w31;
/// the shadow comparators of W1 to W3.1 (rows by id, stamps) run only with `TLG_W1_CENSUS=1`: they cost several hundred ns per binding and would hide what the census clocks measure
inline bool W1ShadowOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_W1_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
/// W3.2 (`TLG_NATIVE_ROWS=1`): the answer the old path gave for a read-only request of one engine buffer, kept per (id, address, size). Valid while the placement epoch is the same, no identity
/// event (placement, drop, GPU written state, image registered) touched the range since the clock stored, and the engine buffer holds no announced bytes in the range.
struct RealRow {
    u64 id{}, address{}, size{};
    bool texel{};
    const void* buffer{};
    u64 offset{}, epoch{}, clock{};
    bool gpu{}; ///< the range was GPU written when the answer was made (the old path chooses the stream ring for small ranges that are not)
    u64 version{}, unattr{}; ///< F2 row version: the versions the answer was made under
    bool inside{};           ///< the range lies inside the buffer's own bytes (the unattributed version is not asked)
    bool is_static{};        ///< F1 class 2 static row: learned from the one upload of ObtainStatic
    u64 hash{};              ///< ... the guest bytes' hash at that upload
    bool written{};          ///< F2 item 3: the row of a written binding
};
thread_local std::unordered_map<u64, RealRow> t_rows_real;
struct StaticSeen {
    u64 version{}, hash{};
};
thread_local std::unordered_map<u64, StaticSeen> t_static_seen; // static-row census: the previous stream copy of a request
thread_local bool t_row_gpu = false, t_row_dirty_epoch = false; // (diagnostics of the last served row)
thread_local bool t_row_static = false; // the last served row is a static row
thread_local u64 t_row_hash = 0;
inline u64 RowKey(u64 id, u64 address, u64 size, bool texel, bool written = false) noexcept {
    return Vulkan::RowShare::Key(id, address, size, texel, written); // (the same key the front reads the shared row table by)
}
inline void ShareRow(u64 key, const RealRow& row) noexcept {
    if (Vulkan::RowShare::Mode() != 0) {
        Vulkan::RowShare::Publish(key, Vulkan::RowShare::Row{row.id, row.address, row.size,
                                                            (row.texel ? Vulkan::RowShare::FlagTexel : 0u) | (row.written ? Vulkan::RowShare::FlagWritten : 0u) |
                                                                (row.inside ? Vulkan::RowShare::FlagInside : 0u) | (row.is_static ? Vulkan::RowShare::FlagStatic : 0u),
                                                            row.buffer, row.offset, row.epoch, row.version, row.clock});
    }
}

/// W3.0 (dispatch census, `TLG_COVER_CENSUS=1`): what one dispatch bound and where its time went; summed per compute program in `g_w3`
struct W3Cur {
    u64 bind_ns{}, sync_ns{}, upload{};
    u32 kinds[8]{}; ///< bindings by kind: 1 ring, 2 special, 3 null, 4 engine handle, 5 address read, 6 address written, 7 resolved by the front
    u32 images{}, written{}, stream{}, with_upload{};
    u64 wr_ns{}, wr_n{}, rd_ns{}, rd_n{}, rd_row_ns{}, rd_row_n{}; ///< obtain ns and count of written requests, of read requests the old path answered, of read requests a row answered
    u64 buf[3]{}, buf_n{}; ///< the bindings that reached the obtain: ns before the obtain, in it, after it (to the end of the iteration), and how many
    u64 phase[6]{};     ///< ns of the compute bind by part: 0 entry (queue drains), 1 the compute image op checks, 2 setup (diagnostics, user data), 3 buffers, 4 textures and samplers, 5 FinishBindResources
    u32 why[8]{};       ///< shader buffers by why they have no engine handle (the W1 reasons: 0 has a handle, 1 no binder plan, 2 not a single load, 3 no record entry, 4 plan has no handles, 5 id not in the registry, 6 epoch stale)
    u32 written_id[2]{}; ///< written buffers: [0] without an engine id in the record, [1] with one
    u32 fast{};          ///< design review 66: buffer bindings bound from the front's planned row (rows ahead)
};
thread_local W3Cur t_w3;
struct W3Stat {
    u64 v[48]{}; ///< 0 dispatches, 1 total ns, 2 ns before the bind, 3 ns in the bind, 4 ns after the bind, 5 sync ns, 6 uploaded bytes, 7..13 bindings by kind 1..7, 14 images, 15 written buffers, 16 stream-ring answers, 17 requests that uploaded, 18 barriers, 19 groups
};
std::mutex g_w3_mutex;
std::unordered_map<u64, W3Stat> g_w3;
struct W1Pass {
    u64 depth{~0ull};
    u64 draws{0};
    std::unordered_set<u64> pipelines, geometry, other, ring, state, index_handles, other_handles, shapes;
    u64 prev_shape{};
    bool have_prev{false};
    u64 prev_pipeline{}, prev_geometry{}, prev_other{}, prev_push{}, prev_state{}, prev_vertex{}, prev_index{};
    u64 pairs{0}, same_pipeline{0}, same_geometry{0}, same_other{0}, same_push{0}, same_state{0}, same_vertex{0}, same_index{0};
};
thread_local W1Pass t_w1_pass;
inline u64 W1Mix(u64 h, u64 v) noexcept {
    h = (h ^ v) * 0x100000001b3ull;
    return h ^ (h >> 29);
}
} // namespace

namespace VideoCore {
extern thread_local u64 t_cover_sync_ns; // buffer_cache.cpp: A2 census, time in SynchronizeMemory
extern thread_local u64 t_cover_sync_copies;
}

namespace Vulkan {

namespace {
/// The names of the pass classes the draws are tagged with (Prof tag "pass_class").
const char* PassClassName(u32 pass_class) {
    switch (pass_class) {
    case 0:
        return "z prepass / shadow";
    case 1:
        return "g-buffer";
    case 2:
        return "forward";
    case 3:
        return "compute / other";
    case 4:
        return "indirect";
    default:
        return "untagged";
    }
}
} // namespace

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

namespace {
Rasterizer* supersede_target = nullptr;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       Core::NativeExecutor* executor_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      buffer_cache{instance, scheduler, runtime, executor_, texture_cache},
      texture_cache{instance, scheduler, runtime, executor_, buffer_cache},
      executor{executor_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, executor, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      readback_batching{EmulatorSettings.IsReadbackBatchingEnabled()},
      async_fences{EmulatorSettings.IsReadbackAsyncFencesEnabled()},
      rt_reuse_enabled{EmulatorSettings.IsRtLookupReuseEnabled()},
      encoder_debug{EmulatorSettings.GetCpEncoderDebug()},
      full_eop_fences{EmulatorSettings.GetCpEncoderMode() >= 2 || (encoder_debug & 4) != 0},
      flush_commands{EmulatorSettings.GetCpFlushCommands()} {
    Core::LightRecorder::SetEventDescriber([](void* context, u64 first, u64 last) { return static_cast<VideoCore::TextureCache*>(context)->DescribeImageEvents(first, last); }, &texture_cache);
    VideoCore::ReadbackPolicy::InitFromSettings();
    if (!EmulatorSettings.IsNullGPU()) {
        executor->BindRasterizer(this);
    }
    scheduler.SetEncoderMode(EmulatorSettings.GetCpEncoderMode());
    memory->SetRasterizer(this);
    // default 4 since 2026-11-07 (owner play good, draws only: see TLG_RING_KIND); TLG_CONST_RING=0 turns the ring off
    const_ring_level = 4;
    if (const char* env = std::getenv("TLG_CONST_RING")) {
        const_ring_level = static_cast<u32>(std::clamp(std::atoi(env), 0, 4));
    }
    if (const_ring_level >= 2) {
        constant_ring = std::make_unique<ConstantRing>(instance, scheduler);
    }
    if (Core::HookRing::RealOn() && constant_ring) {
        // F1 1a: two halves of the game's 64 MB frame ring, in the stream memory type (cached system memory by default: F5 found GPU memory written through the BAR dearer)
        constexpr u64 hook_capacity = u64{128} << 20;
        hook_ring_buffer = std::make_unique<VideoCore::Buffer>(instance, 0, hook_capacity, VideoCore::MemoryType::Stream, "hook ring");
        if (!hook_ring_buffer->mapped_data.empty()) {
            Core::HookRing::SetStorage(hook_ring_buffer->mapped_data.data(), hook_capacity,
                                       [](void* context, u64 offset, u64 size) { static_cast<VideoCore::Buffer*>(context)->Flush(offset, size); }, hook_ring_buffer.get());
            buffer_cache.SetHookRing(hook_ring_buffer.get()); // F1 1b real (TLG_HOOK_RING_1B=2): ObtainBuffer binds whole ring ranges from it
            LOG_INFO(Render, "hook ring (F1 1a): {} MB of GPU-visible memory, constants bound from it", hook_capacity >> 20);
        } else {
            LOG_ERROR(Render, "hook ring (F1 1a): the buffer is not mapped, the old path stays");
            hook_ring_buffer.reset();
        }
    }
    LOG_INFO(Render, "constant ring: level {}", const_ring_level);
    if (const char* env = std::getenv("TLG_INPLACE"); env != nullptr && *env != 0 && *env != '0') {
        host_import = std::make_unique<HostImport>(instance);
        inplace_on = host_import->Available();
        if (const char* cap = std::getenv("TLG_INPLACE_CAP"); cap != nullptr && *cap != 0) {
            inplace_cap = static_cast<u32>(std::clamp(std::atoi(cap), 16, 1 << 20));
        }
        LOG_INFO(Render, "in place buffers: {}, cap {} bytes", inplace_on ? "on" : "unavailable", inplace_cap);
    }
    supersede_target = this;
    Core::NativeUpdates::SetSupersede([](u64 address, u64 size) {
        if (supersede_target != nullptr) {
            supersede_target->SupersedeGpuWritten(address, size);
        }
    });
    // on by default since 2026-11-07 (=0 turns it off): 75% of the eligible stages hit it, the slow path costs 0.9 us a stage
    if (const char* env = std::getenv("TLG_TEXTURE_SET_MEMO"); env == nullptr || env[0] != '0') {
        texture_set_on = true;
        texture_sets = std::make_unique<TextureSet[]>(TextureSetSlots);
    }
    LOG_INFO(Render, "texture set memo: {}", texture_set_on ? "on" : "off");
    // B8.2: the engine's deletions, in order with the GPU work (raster thread)
    Core::NativeUpdates::SetDelete(
        [](void* context, u64 id, u64 address, u64 bytes, u32 what) { static_cast<Rasterizer*>(context)->EngineObjectDeleted(id, address, bytes, what); }, this);
    Core::NativeUpdates::SetCreate(
        [](void* context, u64 id, u64 address, u64 bytes, bool texture, u32 usage, u32 cpu_access) {
            auto* self = static_cast<Rasterizer*>(context);
            if (texture) {
                self->texture_cache.EngineIndex().NoteTextureCreate(address, bytes);
            } else {
                self->texture_cache.EngineIndex().NoteBufferCreate(id, address, bytes); // B8.5
            }
            self->buffer_cache.NoteEngineResourceCreate(id, address, bytes, texture, usage, cpu_access); // B9.4: a placed buffer under the new resource moves back into the arena (or takes it in)
        },
        this);

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
    });
}

Rasterizer::~Rasterizer() = default;

namespace {
/// Census kinds of the consumer (C0). Shader bindings: index = (compute ? 32 : 0) + kind; geometry (graphics only): 24 + kind.
enum CensusKind : u32 {
    CBufferTotal = 0, CBufferRing, CBufferSpecial, CBufferNull, CBufferHandle, CBufferAddrRead, CBufferAddrWritten, CBufferAddrReadLe256, CBufferAddrReadLe4k,
    CBufferAddrReadLe64k, CBufferAddrReadBig, CBufferBarrierQuery, CBufferBarrierTrue, CImageTotal, CImageNull, CImageSetHit, CImageLookupHit, CImageFind,
    CImageStorage, CImageTransit, CImageTransitTrue, CBindResources, CBufferInPlace, CBufferResolved,
    CGeometryVertexRing = 24, CGeometryVertexHandle, CGeometryVertexAddress, CGeometryIndexRing, CGeometryIndexHandle, CGeometryIndexAddress,
};
} // namespace

void Rasterizer::MaybeFlushRecorded() {
    if (VideoCore::RecordQueue::Enabled() && VideoCore::RecordQueue::Pending()) {
        VideoCore::RecordQueue::Post([this] { MaybeFlushRecordedBody(); }); // after the draws that wait
        return;
    }
    MaybeFlushRecordedBody();
}

void Rasterizer::MaybeFlushRecordedBody() {
    TLG_PROF_LEVEL("raster.flush_recorded", 3);
    const u32 limit =
        VideoCore::ReadbackPolicy::Get().flush_commands.load(std::memory_order_relaxed);
    if (limit != 0 && scheduler.PendingEncoderCommands() >= limit) {
        if (Core::NativeOcclusion::query_open.load(std::memory_order_acquire) != 0) {
            return; // a submit now would end the command buffer inside an occlusion query (begun and ended in different command buffers: invalid, device lost)
        }
        scheduler.FlushUnplanned(); // P3b: the only submission the recorder makes on its own; the logical count does not move
    }
}

/// The measuring of draw reuse (repeat census, key census, recipe verify) costs a hash and a map lookup per draw: off unless asked (2026-11-07, closed as a measurement).
static bool ReuseCensusOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_REUSE_CENSUS");
        const char* v = std::getenv("TLG_VERSION_CENSUS");
        return (e != nullptr && *e == '1') || (v != nullptr && *v == '1');
    }();
    return on;
}

/// Plan V, step V0: the recipes checked against the identity events that overlap their own ranges (buffer_cache.h `RecipePredictScoped`); implies the reuse census (the shape hash).
static bool VersionCensusOn() {
    static const bool on = [] {
        const char* v = std::getenv("TLG_VERSION_CENSUS");
        return v != nullptr && *v == '1';
    }();
    return on;
}

/// Plan S step P3e.3: the measuring code below reads what the record half produced (descriptor infos, push data, ticks). On a recorder thread it cannot run in the resolve half:
/// the switches are listed once, so a mode that moves the record half to another thread refuses them (UiTrace and LightRecorder are armed at run time and are listed by name).
static bool BindCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_BIND_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void Rasterizer::NoteRecordedStateDiagnostics() {
    static const bool once = [] {
        if (SplitBindMode() == 0) {
            return true;
        }
        std::string on;
        if (ReuseCensusOn()) {
            on += " TLG_REUSE_CENSUS/TLG_VERSION_CENSUS (reuse census, key census, recipe verify)";
        }
        if (BindCensusOn()) {
            on += " TLG_BIND_CENSUS";
        }
        if (!on.empty()) {
            LOG_WARNING(Render_Vulkan, "split bind: these switches read recorded state in the resolve half and cannot run on a recorder thread:{}", on);
        }
        return true;
    }();
    (void)once;
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

static std::atomic<u64> g_eliminate_total{0}, g_eliminate_with_clear{0};

bool Rasterizer::FilterDraw(const DrawPlan* plan) {
    TLG_PROF_LEVEL("raster.filter_draw", 3);
    // F3.5: everything comes from the plan (the command processor thread made it from the record); a draw without a plan is not drawn
    if (plan == nullptr) {
        return false;
    }
    const DrawInputs& inputs = plan->inputs;
    const auto& cb_extents = plan->cb_extent;
    const AmdGpu::CbDbExtent& db_extent = plan->db_extent;
    current_target_resources = &plan->target_resources; // E3: the game's own transition operations name their targets too
    const DrawClassification classified = plan->op_class != 0xff ? DrawClassification{static_cast<DrawClass>(plan->op_class), plan->depth_copy, plan->stencil_copy}
                                                                  : ClassifyRemaining(plan->inputs, plan->fixed);
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
            if (last_log_ns != 0 && SplitBindMode() == 1) {
                const u64 barrier_ops = class_counts[static_cast<size_t>(DrawClass::EliminateFastClear)] + class_counts[static_cast<size_t>(DrawClass::FmaskDecompress)] +
                                        class_counts[static_cast<size_t>(DrawClass::Resolve)] + class_counts[static_cast<size_t>(DrawClass::DepthStencilCopy)] +
                                        class_counts[static_cast<size_t>(DrawClass::DepthClear)];
                LOG_WARNING(Render_Vulkan, "split bind: per 5 s {} engine operations that resolve and record together (eliminate {} (the ones with a clear to record, so far: {} of {}), fmask {}, resolve {}, depth copy {}, depth clear {}), {} normal draws", barrier_ops,
                            class_counts[static_cast<size_t>(DrawClass::EliminateFastClear)], g_eliminate_with_clear.load(), g_eliminate_total.load(), class_counts[static_cast<size_t>(DrawClass::FmaskDecompress)], class_counts[static_cast<size_t>(DrawClass::Resolve)],
                            class_counts[static_cast<size_t>(DrawClass::DepthStencilCopy)], class_counts[static_cast<size_t>(DrawClass::DepthClear)], class_counts[static_cast<size_t>(DrawClass::Normal)]);
            }
            if (last_log_ns != 0 && VideoCore::RecordQueue::Enabled()) {
                std::string why;
                for (u32 i = 0; i < VideoCore::RecordQueue::ReasonCount; ++i) {
                    if (const u64 n = VideoCore::RecordQueue::drains[i].load(); n != 0) {
                        why += fmt::format(" {} {} ({} draws)", VideoCore::RecordQueue::names[i], n, VideoCore::RecordQueue::drained_jobs[i].load());
                    }
                }
                LOG_WARNING(Render_Vulkan, "split bind queue (lag {}): {} jobs posted, {} run, deepest {}; drains so far:{}; ring laps (none waiting / draws waiting): record {}/{}, resolve {}/{}, staging {}/{}",
                            VideoCore::RecordQueue::Lag(), VideoCore::RecordQueue::posted.load(), VideoCore::RecordQueue::ran.load(), VideoCore::RecordQueue::max_depth.load(), why,
                            VideoCore::StreamBuffer::wraps[0][0].load(), VideoCore::StreamBuffer::wraps[0][1].load(), VideoCore::StreamBuffer::wraps[1][0].load(), VideoCore::StreamBuffer::wraps[1][1].load(),
                            VideoCore::StreamBuffer::wraps[2][0].load(), VideoCore::StreamBuffer::wraps[2][1].load());
            }
            if (last_log_ns != 0 && VideoCore::RecordQueue::Threaded()) {
                std::string waits;
                for (u32 i = 0; i < VideoCore::RecordQueue::ReasonCount; ++i) {
                    if (const u64 ns = VideoCore::RecordQueue::drain_wait_ns[i].load(); ns != 0) {
                        waits += fmt::format(" {} {:.0f} ms", VideoCore::RecordQueue::names[i], static_cast<double>(ns) / 1e6);
                    }
                }
                LOG_WARNING(Render_Vulkan, "split bind thread: recorder busy {:.0f} ms, idle {:.0f} ms; resolver waited in Post {:.0f} ms, in drains:{}", static_cast<double>(VideoCore::RecordQueue::worker_busy_ns.load()) / 1e6,
                            static_cast<double>(VideoCore::RecordQueue::worker_idle_ns.load()) / 1e6, static_cast<double>(VideoCore::RecordQueue::post_wait_ns.load()) / 1e6, waits);
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
        const bool no_color = !plan->fixed.color_enabled || plan->fixed.NoColorTargets();
        if (no_color && inputs.depth_buffer.DepthValid()) {
            VideoCore::ReadbackPolicy::Get().NoteDepthOnly(inputs.depth_buffer.Pitch(), inputs.depth_buffer.Height());
        }
    }
    return true;
}

std::unique_ptr<DrawPlan> Rasterizer::PlanDraw(DrawIndirectParams params, const FixedState& native, u64 exec_key, u64 inputs_key, const Core::NativeDraw::ExecDraw& record, u32 index_count) {
    // F3.5: a draw is planned from its record, the fixed-function state built from the engine's objects, and the shader and target descriptions; no register image exists. SkipShadow draws are
    // still planned: the switch can change between the front and the raster thread.
    std::unique_ptr<DrawPlan> plan = plan_pool.TakeDraw();
    const DrawClassification no_operation{};
    plan->executed = true;
    plan->native_seq = 0;
    plan->bindings_complete = false;
    plan->has_geometry_handles = false;
    plan->binding_count = 0;
    plan->op_class = 0xff;
    plan->depth_copy = no_operation.depth_copy;
    plan->stencil_copy = no_operation.stencil_copy;
    Core::NativeDraw::Live().fixed_native.fetch_add(1, std::memory_order_relaxed);
    plan->fixed = native;
    plan->inputs = Core::NativeDrawInputs::FromRecord(record, index_count);
    Core::NativeDraw::Live().key_targets[10].fetch_add(1, std::memory_order_relaxed);
    {
        // F3.3: the words of the inputs that only the register image has (clip planes, border colour base, index offset) are checked: not zero would mean the native planning needs a value for them
        const auto not_zero = [](const auto& value) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
            for (size_t i = 0; i < sizeof(value); ++i) {
                if (bytes[i] != 0) {
                    return true;
                }
            }
            return false;
        };
        auto& live = Core::NativeDraw::Live();
        live.reg_only_checked.fetch_add(1, std::memory_order_relaxed);
        if (not_zero(plan->inputs.ta_bc_base) || not_zero(plan->inputs.clip_user_data) || plan->inputs.index_offset != 0) {
            live.reg_only_nonzero.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (inputs_key != 0 && plan->executed) {
        exec_inputs.insert_or_assign(inputs_key, plan->inputs);
    }
    pipeline_cache.PlanGraphics(plan->fixed, params, *plan, exec_key);
    return plan;
}

/// L1(b) (`TLG_TEXTURE_IDS`, default on; 0 turns it off): a cached image lookup is kept across generation changes when the texture cache proves that nothing touched its range.
/// B3 (`TLG_B3_IMAGES=0` turns it off): image bindings by the engine object's id
static bool B3ImagesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_B3_IMAGES");
        return e == nullptr || *e != '0';
    }();
    return on;
}

static bool TextureIdsOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_TEXTURE_IDS");
        return e == nullptr || *e != '0';
    }();
    return on;
}
std::unique_ptr<DispatchPlan> Rasterizer::PlanDispatch(const Core::NativeDispatch::DispatchInputs& cs) {
    std::unique_ptr<DispatchPlan> plan = plan_pool.TakeDispatch();
    pipeline_cache.PlanCompute(cs, *plan);
    plan->binding_count = 0;
    plan->bindings_complete = false;
    plan->cs = cs;
    plan->ta_bc_base = {}; // F3.5: no custom border colour table for a dispatch (the register mirror held zero in 780,000 dispatches over three scenes)
    return plan;
}

static const Core::NativeDispatch::DispatchInputs no_program{}; // a dispatch always has a plan
static const AmdGpu::BorderColorBuffer no_border_colors{}; // a dispatch always has a plan; the default stands for none

const GraphicsPipeline* Rasterizer::ResolveDrawPipeline(const DrawPlan* plan,
                                                        DrawIndirectParams params) {
    TLG_PROF_LEVEL("raster.pipeline_lookup", 2);
    if (plan) {
        PipelineCache::ApplyPlan(*plan);
        return plan->pipeline;
    }
    LOG_ERROR(Render_Vulkan, "draw without a pipeline plan skipped"); // F3.5: no plan, no register image to plan from
    return nullptr;
}

const ComputePipeline* Rasterizer::ResolveDispatchPipeline(const DispatchPlan* plan) {
    TLG_PROF_LEVEL("raster.pipeline_lookup", 2);
    if (plan) {
        PipelineCache::ApplyPlan(*plan);
        return plan->pipeline;
    }
    if (executor->SplitActive()) {
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
    TLG_PROF_LEVEL("raster.prepare_render_state", 2);
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = *draw_inputs;
    if (fixed.degamma) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding = !fixed.color_enabled;
    // notes 103 and 104 (`TLG_TARGET_TABLE`): 1 shadow (the old path finds every target and the table is compared), 2 real (a target whose view has a valid entry takes the entry's
    // image and description: no register compare, no generation, no decode, no FindImage; one draw in 64 takes the old path and is compared)
    const u32 tg_mode = Core::ViewTable::TargetMode();
    static thread_local u32 tg_tick = 0;
    const bool tg_audit = tg_mode == 2 && (++tg_tick & 63u) == 0;
    const bool tg_take = tg_mode == 2 && !tg_audit;
    bool tg_any = false, tg_all_in = true, tg_all_served = true, tg_old_find = false, tg_all_taken = true;
    if (tg_mode != 0) [[unlikely]] {
        target_use.fill(0);
    }
    const auto tg_note = [&](u32 slot, u64 use, VideoCore::ImageId id, const VideoCore::TextureCache::ImageDesc& d, bool found) {
        bool in_class = false, served = false;
        TargetTableShadow(slot, use, id, d, in_class, served);
        tg_any = true;
        tg_all_in &= in_class;
        tg_all_served &= served;
        tg_old_find |= in_class && found;
        tg_all_taken = false;
    };
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
        const u64 tg_use = tg_mode != 0 ? TargetUse(static_cast<u64>(cb), &hint, sizeof(hint)) : 0; // (the color slot and its extent hint)
        if (tg_take && TargetTableTake(static_cast<u32>(cb), tg_use, image_id, desc)) {
            rt_cache[cb].valid = false; // (its stored inputs no longer describe cb_descs)
            bound_images.emplace_back(image_id);
            texture_cache.TouchFoundImage(image_id);
            tg_any = true;
        } else {
            bool found = false;
            if (RtReuseOn() && TargetInputsMatch(rt_cache[cb], col_buf, hint)) {
                // Same registers and no image registered or removed since: FindImage would find the
                // same image again (image_id and desc still hold its result).
                bound_images.emplace_back(image_id);
                texture_cache.TouchFoundImage(image_id);
            } else {
                std::construct_at(&desc, col_buf, hint);
                desc.engine_id = (*current_target_resources)[cb]; // E3: the resource the draw's bound render target view looks at
                desc.engine_site = 1;
                image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
                found = true;
                if (RtReuseOn()) {
                    rt_cache[cb].valid = true;
                    rt_cache[cb].generation = texture_cache.ImageGeneration();
                }
            }
            if (tg_mode != 0 && image_id) [[unlikely]] {
                tg_note(static_cast<u32>(cb), tg_use, image_id, desc, found);
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
        u64 tg_use = 0;
        if (tg_mode != 0) [[unlikely]] {
            // the depth target's use: what its description takes besides the view (the view and control words, the htile address, the extent hint)
            std::array<u8, 128> bytes{};
            u32 n = 0;
            const auto put = [&](const auto& part) {
                static_assert(sizeof(part) <= 64);
                std::memcpy(bytes.data() + n, &part, sizeof(part));
                n += sizeof(part);
            };
            put(regs.depth_view);
            put(regs.depth_control);
            put(htile_address);
            put(hint);
            tg_use = TargetUse(8u, bytes.data(), n);
        }
        if (tg_take && TargetTableTake(8, tg_use, image_id, desc)) {
            cache.valid = false;
            bound_images.emplace_back(image_id);
            texture_cache.TouchFoundImage(image_id);
            tg_any = true;
        } else {
            bool found = false;
            if (RtReuseOn() && TargetInputsMatch(cache, regs.depth_buffer, regs.depth_view,
                                                      regs.depth_control, htile_address, hint)) {
                bound_images.emplace_back(image_id);
                texture_cache.TouchFoundImage(image_id);
            } else {
                std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                                  htile_address, hint);
                desc.engine_id = (*current_target_resources)[8]; // E3: the resource the bound depth view looks at
                desc.engine_site = 2;
                image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
                found = true;
                if (RtReuseOn()) {
                    cache.valid = true;
                    cache.generation = texture_cache.ImageGeneration();
                }
            }
            if (tg_mode != 0 && image_id) [[unlikely]] {
                tg_note(8, tg_use, image_id, desc, found);
            }
        }
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
        rt_cache[AmdGpu::NUM_COLOR_BUFFERS].valid = false;
    }
    if (tg_mode != 0 && tg_any) [[unlikely]] {
        using namespace Core::ViewTable;
        Note(TgDraws);
        if (tg_audit) {
            Note(TgAudits);
        }
        if (tg_mode == 2 && tg_all_taken) {
            Note(TgDrawsTaken);
        } else if (tg_all_in) {
            Note(TgDrawsAllInClass);
            if (tg_all_served) {
                Note(TgDrawsWouldServe);
            }
            if (tg_old_find) {
                Note(TgDrawsOldFind);
            }
        }
    }
}

u64 Rasterizer::TargetUse(u64 slot, const void* bytes, size_t size) noexcept {
    u64 h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const u8*>(bytes);
    for (size_t i = 0; i < size; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return (slot & 0xff) | ((h << 8) & ~(1ull << 63));
}

namespace {
/// Note 104: the fields of a target's description the render pass and the texture cache read (the image, its meta surfaces, its stencil, the view).
bool TargetDescSame(const VideoCore::TextureCache::ImageDesc& a, const VideoCore::TextureCache::ImageDesc& b) {
    const auto& x = a.info;
    const auto& y = b.info;
    return x.guest_address == y.guest_address && x.guest_size == y.guest_size && x.pixel_format == y.pixel_format && x.size == y.size && x.num_samples == y.num_samples &&
           x.type == y.type && x.resources == y.resources && x.pitch == y.pitch && x.tile_mode == y.tile_mode && x.meta_info.cmask_addr == y.meta_info.cmask_addr &&
           x.meta_info.fmask_addr == y.meta_info.fmask_addr && x.meta_info.htile_addr == y.meta_info.htile_addr && x.stencil_addr == y.stencil_addr &&
           x.stencil_size == y.stencil_size && a.view_info == b.view_info && a.type == b.type;
}
} // namespace

bool Rasterizer::TargetTableTake(u32 slot, u64 use, VideoCore::ImageId& image_id, VideoCore::TextureCache::ImageDesc& desc) {
    using namespace Core::ViewTable;
    const u64 view_id = (*current_target_resources)[9 + slot];
    if (view_id == 0) {
        return false;
    }
    const u64 flags = (1ull << 63) | use;
    const Entry* e = Find(view_id, flags);
    if (e == nullptr) {
        return false;
    }
    const VideoCore::ImageId id{e->image_index};
    if (e->image_uid == 0 || texture_cache.ImageUid(id) != e->image_uid || False(texture_cache.GetImage(id).flags & VideoCore::ImageFlagBits::Registered)) {
        return false;
    }
    const auto it = target_descs.find(view_id * 0x9E3779B97F4A7C15ull ^ flags);
    if (it == target_descs.end()) {
        return false;
    }
    image_id = id;
    desc = it->second;
    target_use[slot] = flags;
    Note(TgTaken);
    return true;
}

u64 Rasterizer::TargetTableShadow(u32 slot, u64 use, VideoCore::ImageId image_id, const VideoCore::TextureCache::ImageDesc& desc, bool& in_class, bool& served) {
    using namespace Core::ViewTable;
    const u64 view_id = (*current_target_resources)[9 + slot];
    if (view_id == 0) {
        Note(TgNoView);
        return 0;
    }
    const u64 flags = (1ull << 63) | use;
    Note(TgTargets);
    in_class = true;
    target_use[slot] = flags;
    Entry* e = Find(view_id, flags);
    const u64 uid = texture_cache.ImageUid(image_id);
    const u64 desc_key = view_id * 0x9E3779B97F4A7C15ull ^ flags;
    const auto keep = [&](Entry& x) {
        x.image_index = image_id.index;
        x.image_uid = uid;
        target_descs.insert_or_assign(desc_key, desc);
    };
    if (e == nullptr) {
        Entry& made = Put(view_id, flags);
        made.in_class = true;
        made.resource_id = (*current_target_resources)[slot];
        if (made.resource_id != 0) {
            LinkResource(made.resource_id, view_id);
        }
        keep(made);
        Note(TgNoEntry);
        return flags;
    }
    const VideoCore::ImageId entry_image{e->image_index};
    const bool valid = e->image_uid != 0 && texture_cache.ImageUid(entry_image) == e->image_uid &&
                       True(texture_cache.GetImage(entry_image).flags & VideoCore::ImageFlagBits::Registered);
    if (!valid) {
        Note(TgStale);
        keep(*e);
        e->view_handle = 0;
        return flags;
    }
    if (entry_image == image_id) {
        Note(TgSame);
        served = true;
        const auto it = target_descs.find(desc_key);
        if (it == target_descs.end()) {
            target_descs.insert_or_assign(desc_key, desc);
        } else if (!TargetDescSame(it->second, desc)) {
            Note(TgDescDifferent);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                LOG_ERROR(Render_Vulkan, "target table: view {:#x} slot {}: same image, but the old path's description differs from the stored one (format {} / {}, layers {} / {}, base {}:{} / {}:{})",
                          view_id, slot, vk::to_string(it->second.view_info.format), vk::to_string(desc.view_info.format), it->second.view_info.range.extent.layers,
                          desc.view_info.range.extent.layers, it->second.view_info.range.base.level, it->second.view_info.range.base.layer, desc.view_info.range.base.level,
                          desc.view_info.range.base.layer);
            }
            it->second = desc;
        }
        return flags;
    }
    Note(TgDifferent);
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
        const auto& a = texture_cache.GetImage(entry_image);
        const auto& b = texture_cache.GetImage(image_id);
        LOG_ERROR(Render_Vulkan, "target table: view {:#x} slot {} answered image {} {:#x} {}x{}; the old path found image {} {:#x} {}x{}", view_id, slot, entry_image.index,
                  a.info.guest_address, a.info.size.width, a.info.size.height, image_id.index, b.info.guest_address, b.info.size.width, b.info.size.height);
    }
    keep(*e);
    e->view_handle = 0;
    return flags;
}

void Rasterizer::InvalidateWrittenBuffer(u64 id, VAddr address, u64 size, bool inside) {
    using namespace Core::ViewTable;
    if (WrittenMode() == 0) [[likely]] {
        texture_cache.InvalidateMemoryFromGPU(address, size, "a written storage buffer view");
        return;
    }
    // note 107 shadow: the walk decides; beside it, the buffer's id answers whether any image or waiting clear can be in its memory (note 108, real: below)
    Note(WbCalls);
    bool named = false;
    if (id != 0 && Core::NativeUpdates::B8ImageMode() != 0) {
        if (inside) {
            named = true;
        } else {
            Core::NativeDevice::BufferPart part{};
            named = Core::NativeDevice::BufferPartOf(id, address, size, part) && part.begin == address && part.end == address + size;
        }
    }
    const bool id_none = named && texture_cache.EngineIndex().NoImageInBuffer(id, address, size);
    if (WrittenMode() == 2 && id_none) {
        // note 108, real: no image can be based in the buffer's memory, so the walk is skipped; one in 64 walks anyway and must mark nothing
        static thread_local u32 audit_tick = 0;
        if ((++audit_tick & 63u) != 0) {
            Note(WbSkipped);
            return;
        }
        Note(WbAudited);
        if (const u32 found = texture_cache.InvalidateMemoryFromGPU(address, size, "a written storage buffer view"); found != 0) {
            Note(WbAuditFound);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                LOG_ERROR(Render_Vulkan, "written buffer audit: engine buffer {:#x} {:#x}+{:#x}: skipped as having no image, the walk marked {}", id, address, size, found);
            }
        }
        return;
    }
    const u32 marked = texture_cache.InvalidateMemoryFromGPU(address, size, "a written storage buffer view");
    if (!named) {
        Note(WbNotNamed);
    } else {
        Note(WbNamed);
    }
    if (id_none) {
        Note(WbIdNone);
    }
    if (marked != 0) {
        Note(WbWalkFound);
        if (id_none) {
            Note(WbIdNoneWalkFound);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                LOG_ERROR(Render_Vulkan, "written buffer: engine buffer {:#x} {:#x}+{:#x}: its image list says no image, the walk marked {}", id, address, size, marked);
            }
        }
    }
}

void Rasterizer::TargetViewShadow(u32 slot, vk::ImageView view) {
    using namespace Core::ViewTable;
    const u64 flags = target_use[slot];
    const u64 view_id = (*current_target_resources)[9 + slot];
    if (flags == 0 || view_id == 0) {
        return;
    }
    Entry* e = Find(view_id, flags);
    if (e == nullptr) {
        return;
    }
    const u64 bits = reinterpret_cast<u64>(static_cast<VkImageView>(view));
    if (e->view_handle == 0) {
        e->view_handle = bits;
        Note(TgViewNoEntry);
    } else if (e->view_handle == bits) {
        Note(TgViewSame);
    } else {
        Note(TgViewDifferent);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
            LOG_ERROR(Render_Vulkan, "target table: view {:#x} slot {}: the render pass chose another image view", view_id, slot);
        }
        e->view_handle = bits;
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
    FastClearOp op;
    g_eliminate_total.fetch_add(1, std::memory_order_relaxed);
    if (ResolveFastClearEliminate(inputs, cb0, op)) {
        g_eliminate_with_clear.fetch_add(1, std::memory_order_relaxed);
        RecordFastClearEliminate(op);
    }
}

bool Rasterizer::ResolveFastClearEliminate(const DrawInputs& inputs, const AmdGpu::CbDbExtent& cb0, FastClearOp& op) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    const auto& col_buf = inputs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return false;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, cb0);
    desc.engine_id = (*current_target_resources)[0]; // E3
    desc.engine_site = 3;
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    (void)image_view;
    auto& image = texture_cache.GetImage(image_id);
    if (!image.pending_fast_clear) {
        Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseFastClearEliminateNothing, engine_clear_on);
        return false;
    }
    // B11: the engine fast clear pending on the image is written now (the eliminate is where the console writes it into the memory)
    Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseFastClearEliminate, engine_clear_on);
    // The colour is the one the hardware uses: the clear words in the target's registers, not the colour the engine passed to its clear call (for a fast clear
    // the engine's colour can be 1,1,1,1 where the registers hold 0,0,0,0: the B11 audit of 2026-10-30 found 470 such clears in 16,656, and the white fill of those
    // is the giant feathers over the view). The pending clear only says that a fast clear is waiting.
    op.value = LiverpoolToVK::ColorBufferClearValue(col_buf);
    image.pending_fast_clear.reset();
    Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeFastClearAtEliminate);
    op.image = &image;
    op.range = desc.view_info.range;
    op.address = col_buf.Address();
    return true;
}

void Rasterizer::RecordFastClearEliminate(const FastClearOp& op) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}", op.address));
    runtime.ClearImage(op.image, op.range, op.value);
    ScopeMarkerEnd();
}

void Rasterizer::ClearDepthStencilTarget(const DrawInputs& inputs, const AmdGpu::CbDbExtent& db_extent) {
    scheduler.PopPendingOperations();
    ClearDepthTarget(inputs, db_extent);
}

void Rasterizer::ClearDepthTarget(const DrawInputs& regs, const AmdGpu::CbDbExtent& db_extent) {
    Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseDepthClearTarget, engine_clear_on);
    if (SkipForEngineClear()) {
        return;
    }
    const auto htile_address = regs.depth_htile_data_base.GetAddress();
    VideoCore::TextureCache::ImageDesc desc(regs.depth_buffer, regs.depth_view, regs.depth_control, htile_address,
                                            db_extent);
    desc.engine_id = (*current_target_resources)[8]; // E3
    desc.engine_site = 4;
    const auto image_id = texture_cache.FindImage(desc);
    auto& image = texture_cache.GetImage(image_id);
    const bool clear_depth = regs.depth_render_control.depth_clear_enable && regs.depth_buffer.DepthValid();
    const bool clear_stencil = regs.depth_render_control.stencil_clear_enable && regs.depth_buffer.StencilValid();
    ScopeMarkerBegin(fmt::format("ClearDepthTarget:{:#x}", regs.depth_buffer.DepthAddress()));
    runtime.ClearDepthStencilImage(&image, desc.view_info.range, clear_depth, regs.depth_clear, clear_stencil,
                                   regs.stencil_clear);
    image.usage.depth_target = true;
    ScopeMarkerEnd();
}

// B11 (2026-10-30, giant feathers): the work inside an engine clear's packet range is NOT skipped any more except the clear's own fill of the meta memory (FillBuffer below).
// The engine's clear function first runs the resource transition of its target (FUN_02479db0), and that emits the game's own operations (fast clear eliminates, decompress
// draws and dispatches of the previous state); the census of the skipped work found 4,105 such draws and 4,000 dispatches in 20 s that the native clear dropped.
/// L3: one row of the glitch marker's draw table (what ran, from which record, with which programs and targets).
static void NoteMarkerRow(u8 kind, u8 draw_class, const Pipeline* pipeline, u64 seq, const DrawInputs* inputs, u32 count, u32 instances) {
    if (!Core::Marker::Enabled()) {
        return;
    }
    Core::Marker::Row row{};
    row.seq = seq;
    row.caller = seq != 0 ? static_cast<u32>(Core::NativeDraw::RecordCallerGhidra(seq)) : 0;
    row.frame = static_cast<u32>(Core::DebugServer::PresentCount());
    row.kind = kind;
    row.draw_class = draw_class;
    row.flags = seq != 0 ? 1 : 0;
    row.count = count;
    row.instances = instances;
    const auto stages = pipeline->GetStages();
    if (kind >= 2) {
        row.vs_hash = !stages.empty() && stages[0] != nullptr ? stages[0]->pgm_hash : 0;
    } else {
        constexpr u32 FragmentStage = static_cast<u32>(Shader::SwStage::Fragment);
        constexpr u32 VertexStage = static_cast<u32>(Shader::SwStage::Vertex);
        row.ps_hash = stages.size() > FragmentStage && stages[FragmentStage] != nullptr ? stages[FragmentStage]->pgm_hash : 0;
        row.vs_hash = stages.size() > VertexStage && stages[VertexStage] != nullptr ? stages[VertexStage]->pgm_hash : 0;
    }
    if (inputs != nullptr) {
        for (const auto& color : inputs->color_buffers) {
            if (color) {
                if (row.colors == 0) {
                    row.color0 = color.Address();
                    row.width = static_cast<u16>(color.Pitch());
                    row.height = static_cast<u16>(color.Height());
                }
                ++row.colors;
            }
        }
        row.depth = inputs->depth_buffer.DepthAddress();
        if (row.colors == 0 && row.depth != 0) {
            row.width = static_cast<u16>(inputs->depth_buffer.Pitch());
            row.height = static_cast<u16>(inputs->depth_buffer.Height());
        }
    }
    Core::Marker::NoteRow(row);
}

namespace {
/// UI trace image read backs (debugging): the GPU content of the interface text pictures, read back to host memory a few frames apart. Raster thread only.
struct UiShotPending {
    u64 address{};
    Vulkan::StagingBufferRef ref;
    u64 tick{};
    u32 frame{};
    std::string info;
};
std::vector<UiShotPending> ui_shots_pending;
/// The result of a GPU memory copy of picture size, read back right after it (UI trace)
struct UiCopyPending {
    u64 number{};
    int which{}; ///< 0 destination after the copy, 1 source image before it
    Vulkan::StagingBufferRef ref;
    u64 tick{};
};
std::vector<UiCopyPending> ui_copies_pending;
} // namespace

struct Rasterizer::VertexBind {
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    struct Query {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u64 size;
    };
    boost::container::small_vector<Query, 4> queries; ///< barrier queries and bound buffers, asked by the record half
};

struct Rasterizer::IndexBind {
    vk::Buffer buffer{};
    u64 offset{};
    u32 size{};
    vk::IndexType type{};
    const VideoCore::Buffer* query_buffer{nullptr};
    u64 query_offset{};
    u64 query_size{};
};

/// Plan S step P3e.2: everything the resolve half decides for a graphics draw besides its stages, as plain data: the render pass, the vertex buffers, the index buffer.
/// The record half (`RecordDrawJob`) turns it into commands; nothing in it is a pointer to something the resolver may change before then, except the image and buffer objects
/// that the engine keeps alive until the work is recorded.
struct Rasterizer::DrawJob {
    const DrawPlan* plan{nullptr};
    bool indexed{false};
    u32 index_offset{0};
    RenderResolved render;
    VertexBind vertex;
    IndexBind index;
    VideoCore::BufferCache::UploadSink vertex_uploads;
    VideoCore::BufferCache::UploadSink index_uploads;
    u32 vertex_offset{0}; ///< P3e.3: read from the shader info in the resolve half (the next draw's plan rewrites it)
    u32 instance_offset{0};
    // P3e.4: the rest of what the record half of a draw needs
    const GraphicsPipeline* pipeline{nullptr};
    boost::container::small_vector<StageResolved, 6> stages;
    Shader::PushData push_data{};
    bool indexed_active{false};
    bool uses_dma{false};
    bool feedback_loop{false};
    u64 resolve_ns{0};
    u64 op_seq{0};
    FixedState fixed;
    u32 num_indices{0};
    u32 num_instances{0};
    bool use_tag{false};
    u64 tag_k2{0};
    u32 draw_class{15};
    u64 call_seq{0};
    std::vector<Common::UniqueFunction<void>> pre; ///< P3e.4b: the record work the resolve half queued for this draw (image uploads, copies), run before everything else of it
};

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const DrawPlan* plan) {
    struct CoverDrawScope {
        Rasterizer* self;
        u64 begin{0};
        explicit CoverDrawScope(Rasterizer* r) noexcept : self(r) {
            if (self->cover_on) [[unlikely]] {
                t_cover_class = 15;
                begin = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                t_w1_draw_begin = begin;
                t_w1_bind_begin = t_w1_bind_end = 0;
            }
        }
        ~CoverDrawScope() {
            if (begin != 0) [[unlikely]] {
                const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                self->cover_class_total_ns[t_cover_class & 15u].fetch_add(now - begin, std::memory_order_relaxed);
                self->W1EndDraw();
            }
        }
    } cover_draw_scope{this};
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DEntry);
    BarrierCensus::SetOp(0x4000000000000000ull);
    Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseDraw, engine_clear_on);
    if (SkipForEngineClear() && plan != nullptr) {
        // a draw inside a clear's range is never the clear's own (the clear function draws nothing): counted as "run" (kind 2)
        Core::NativeCensus::NoteSkippedInClear(2, plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : (0x100000000ull | plan->op_class), plan->op_class);
    }
    RENDERER_TRACE;
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DClearChecks);

    {
        TLG_PROF_LEVEL("raster.pop_pending", 3);
        scheduler.PopPendingOperations();
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DPopPending);

    // Native draw path: the game said (through the tag in the command stream) which state groups this draw
    // shares with the previous one. Nothing is reused unless the previous executed operation was a tagged draw
    // in the same command buffer.
    const VideoCore::NativeTag tag = std::exchange(native_tag, {});
    Core::NativeRegistry::SetDrawClass(tag.valid ? (tag.meta & 0xff) : 15);
    Core::NativeRegistry::SetCurrentDraw(plan != nullptr && plan->native_seq != 0 ? plan->native_seq : (tag.valid ? tag.seq : 0), plan != nullptr && plan->native_seq != 0);
    if (Core::Prof::Running()) {
        const u32 pass_class = tag.valid ? (tag.meta & 0xff) : 15u;
        TLG_PROF_TAG_HEX("caller", plan != nullptr && plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0);
        TLG_PROF_TAG("pass_class", pass_class, PassClassName(pass_class));
    }
    const bool use_tag =
        tag.valid;
    // Plan S step P3e.2/4: with the split on (and the render state memo off) the render pass, the vertex buffers and the index buffer are resolved together with the stages
    // and the whole record half is one function (`RecordDraw`); then the "same as the previous draw" notes live on the record side and are keyed by the call number
    const bool hoisted = SplitBindMode() == 1 && !RenderStateMemoEnabled();
    const u64 call_seq = ++draw_call_seq;
    const NativeLast last = native_last;
    if (!hoisted) {
        native_last.valid = false;
    }
    if (tag.valid) {
        ++native_stats.tagged;
    } else {
        ++native_stats.untagged;
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DTagClassStats);

    if (!FilterDraw(plan)) {
        return;
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DFilter);

    const GraphicsPipeline* pipeline = ResolveDrawPipeline(plan);
    if (!pipeline) {
        return;
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DPipeline);
    // UI draw trace (native_ui_trace.h): the guest reads of the interface text draws are kept; EndDraw runs when this function returns
    struct UiTraceScope {
        bool on;
        ~UiTraceScope() {
            if (on) {
                Core::UiTrace::EndDraw();
            }
        }
    };
    if (Core::UiTrace::Armed()) {
        bool side_prev_free = false;
        const u64 side_tick = scheduler.SideTickDiag(&side_prev_free);
        Core::UiTrace::NoteHarvest(ui_copies_pending.size(), ui_copies_pending.empty() ? 0 : ui_copies_pending.front().tick, !ui_copies_pending.empty() && scheduler.IsFree(ui_copies_pending.front().tick), scheduler.CurrentTick(), side_tick, side_prev_free);
        for (auto it = ui_copies_pending.begin(); it != ui_copies_pending.end();) {
            if (scheduler.IsFree(it->tick)) {
                it->ref.Invalidate();
                u32 nonzero = 0;
                u64 hash = 1469598103934665603ull;
                for (u64 off = 0; off < it->ref.size; off += 64) {
                    bool any = false;
                    for (u64 k = 0; k < 64 && off + k < it->ref.size; ++k) {
                        const u8 b = it->ref.mapped[off + k];
                        any |= b != 0;
                        hash = (hash ^ b) * 1099511628211ull;
                    }
                    nonzero += any ? 1 : 0;
                }
                Core::UiTrace::NoteCopyResult(it->number, nonzero, hash, it->which);
                it = ui_copies_pending.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = ui_shots_pending.begin(); it != ui_shots_pending.end();) {
            if (scheduler.IsFree(it->tick)) {
                it->ref.Invalidate();
                Core::UiTrace::SaveShot(it->address, it->frame, it->ref.mapped, it->ref.size, it->info);
                it = ui_shots_pending.erase(it);
            } else {
                ++it;
            }
        }
        constexpr u32 FragmentStage = static_cast<u32>(Shader::SwStage::Fragment);
        constexpr u32 VertexStage = static_cast<u32>(Shader::SwStage::Vertex);
        const auto trace_stages = pipeline->GetStages();
        const u64 trace_ps = trace_stages.size() > FragmentStage && trace_stages[FragmentStage] != nullptr ? trace_stages[FragmentStage]->pgm_hash : 0;
        const u64 trace_vs = trace_stages.size() > VertexStage && trace_stages[VertexStage] != nullptr ? trace_stages[VertexStage]->pgm_hash : 0;
        Core::UiTrace::BeginDraw(plan->native_seq, trace_ps, trace_vs, plan->inputs.color_buffers[0].Address(), plan->inputs.num_indices, plan->inputs.num_instances.NumInstances(),
                                 plan->native_seq != 0 ? static_cast<u32>(Core::NativeDraw::RecordCallerGhidra(plan->native_seq)) : 0, static_cast<u32>(Core::DebugServer::PresentCount()));
    }
    const UiTraceScope ui_trace_scope{Core::UiTrace::Active()};
    if (Core::LightRecorder::On()) {
        const u32 draw_class = tag.valid ? (tag.meta & 0xff) : 15u;
        Core::LightRecorder::FrameOp(Core::DebugServer::PresentCount(), draw_class < 5 ? draw_class + 1 : 6, reinterpret_cast<u64>(pipeline), texture_cache.ImageGeneration());
    }
    TLG_PROF_NAMED_LEVEL(prof_inputs, "raster.plan_inputs", 3);
    const FixedState& fixed = plan->fixed;
    if (Core::UiTrace::Active()) {
        for (u32 i = 0; i < fixed.viewport_count; ++i) {
            const auto& v = fixed.viewports[i];
            const auto& sc = fixed.scissors[i];
            Core::UiTrace::NoteState(fmt::format("viewport {} x={} y={} w={} h={} depth {}..{}; scissor x={} y={} w={} h={}; clip_disabled={} viewport_empty={}", i, v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth,
                                                 sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height, fixed.clip_disabled, fixed.viewport_empty));
        }
    }
    draw_inputs = &plan->inputs;
    if (cover_on) [[unlikely]] {
        auto& w = t_w1;
        w.valid = false;
        w.depth = draw_inputs->depth_buffer.DepthAddress();
        u64 st = W1Mix(0xcbf29ce484222325ull, w.depth);
        for (u32 i = 0; i < fixed.viewport_count; ++i) {
            const auto& v = fixed.viewports[i];
            const auto& sc = fixed.scissors[i];
            st = W1Mix(st, std::bit_cast<u32>(v.x));
            st = W1Mix(st, std::bit_cast<u32>(v.y));
            st = W1Mix(st, std::bit_cast<u32>(v.width));
            st = W1Mix(st, std::bit_cast<u32>(v.height));
            st = W1Mix(st, static_cast<u64>(sc.offset.x) << 32 | static_cast<u32>(sc.offset.y));
            st = W1Mix(st, static_cast<u64>(sc.extent.width) << 32 | sc.extent.height);
        }
        w.state = st;
        w.vertex = 0;
        w.index = 0;
        w.index_count = draw_inputs->num_indices;
        w.instances = draw_inputs->num_instances.NumInstances();
    }
    prof_inputs.End();
    current_bc_base = &draw_inputs->ta_bc_base;
    current_cb_extent = plan ? &plan->cb_extent : &executor->RasterCbExtents();
    current_db_extent = plan ? &plan->db_extent : &executor->RasterDbExtent();
    current_target_resources = plan ? &plan->target_resources : &executor->RasterTargetResources();
    const auto& regs = *draw_inputs;
    const bool same_cmdbuf = !hoisted && use_tag && last.valid && last.tick == scheduler.CurrentTick();
    const bool same_pipeline = same_cmdbuf && last.pipeline == pipeline;
    const bool same_dynamic_state = same_pipeline && last.k2 == tag.k2;
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DDiagInputs);

    PrepareRenderState(pipeline, fixed);
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DPrepareState);
    binder_plan = NativeBinderFor(plan, tag.valid ? (tag.meta & 0xff) : 15u);
    Core::DispatchSplit::NoteDrawKind(tag.valid ? (tag.meta & 0xff) : 15u, is_indexed, binder_plan == plan);
    draw_class_now = tag.valid ? (tag.meta & 0xff) : 15u;
    resolved_draw = binder_plan == plan ? plan : nullptr;
    std::unique_ptr<DrawJob> job_owner;
    if (hoisted) {
        job_owner = TakeDrawJob();
        job_owner->plan = plan;
        job_owner->indexed = is_indexed;
        job_owner->index_offset = index_offset;
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DBinderPlan);
    const bool bound = BindResources(pipeline, &fixed, job_owner.get());
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DBind);
    resolved_draw = nullptr;
    binder_plan = nullptr;
    draw_class_now = 15;
    if (!bound) {
        return;
    }
    if (hoisted) {
        DrawJob& job = *job_owner;
        job.pipeline = pipeline;
        job.fixed = fixed;
        job.num_indices = regs.num_indices;
        job.num_instances = regs.num_instances.NumInstances();
        job.use_tag = use_tag;
        job.tag_k2 = tag.k2;
        job.draw_class = tag.valid ? (tag.meta & 0xff) : 15u;
        job.call_seq = call_seq;
        // what the record half needs from outside the job is read from these (markers, traces): a draw that is traced is recorded now
        NoteMarkerRow(0, job.draw_class, pipeline, plan != nullptr ? plan->native_seq : 0, draw_inputs, job.num_indices, job.num_instances); // the glitch marker's row: plain data of the resolve half, in draw order
        const bool diagnostics_now = host_markers_enabled || Core::UiTrace::Active() || Core::LightRecorder::On();
        if (diagnostics_now && VideoCore::RecordQueue::Enabled()) {
            static std::atomic<u64> refused{0};
            if (refused.fetch_add(1) == 0) {
                LOG_WARNING(Render_Vulkan, "split bind queue: a draw is recorded at once because of diagnostics (host markers {}, UiTrace {}, LightRecorder {})", host_markers_enabled, Core::UiTrace::Active(),
                            Core::LightRecorder::On());
            }
        }
        if (VideoCore::RecordQueue::Enabled() && !diagnostics_now) {
            job.pre = std::move(VideoCore::ResolveCommands::pending);
            VideoCore::ResolveCommands::pending.clear();
            VideoCore::RecordQueue::Post([this, owned = std::move(job_owner)]() mutable { RunDrawJob(std::move(owned)); });
        } else {
            VideoCore::RecordQueue::Drain(VideoCore::RecordQueue::Diagnostics);
            RunDrawJob(std::move(job_owner));
        }
        ResetImageBindings(); // the resolve half's flags: the old ResetBindings cleared them between the barrier notes and the access notes, nothing in between reads them
        return;
    }
    RenderState state;
    {
        geometry_plan = plan;
        state = BeginRendering(pipeline);
        Core::DispatchSplit::DStamp(Core::DispatchSplit::DBeginRenderingState);

        BindVertexBuffers(pipeline);
        Core::DispatchSplit::DStamp(Core::DispatchSplit::DVertexBuffers);
        if (is_indexed) {
            BindIndexBuffer(index_offset);
        }
        Core::DispatchSplit::DStamp(Core::DispatchSplit::DIndexBuffer);
        geometry_plan = nullptr;
    }

    const bool pass_barrier = needs_barrier;
    {
        const u32 diff = scheduler.PassStateDifference(state);
        if (diff != 0) {
            g_pass_changes.count[diff].fetch_add(1, std::memory_order_relaxed);
            g_pass_changes.draws_held[diff].fetch_add(0, std::memory_order_relaxed);
        }
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DPassDifference);
    if (needs_barrier) {
        TLG_PROF_LEVEL("raster.flush_barriers", 2);
        t_flush_in_same_pass = scheduler.WouldContinuePass(state);
        runtime.FlushBarriers();
        t_flush_in_same_pass = false;
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DFlushBarriers);

    pipeline->BindResources(set_writes, push_data);
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DDescriptorPush);
    if (same_dynamic_state) {
        ++native_stats.dynamic_skipped;
    } else {
        UpdateDynamicState(pipeline, is_indexed, fixed);
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DDynamicState);
    {
        TLG_PROF_LEVEL("sched.begin_rendering", 3);
        scheduler.BeginRendering(state);
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DSchedBeginRendering);

    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, pipeline->GetStage(Shader::SwStage::Vertex), pipeline->GetFetchShader());
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DDrawOffsets);

    NoteMarkerRow(0, tag.valid ? (tag.meta & 0xff) : 15u, pipeline, plan != nullptr ? plan->native_seq : 0, draw_inputs, regs.num_indices, regs.num_instances.NumInstances());
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DMarkerRow);

    const auto cmdbuf = scheduler.CommandBuffer();
    {
        TLG_PROF_LEVEL("raster.record_draw", 2);
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
    }
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DRecord);
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    ++recorded_work;
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DCounts);

    ResetBindings(false);
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DResetBufferAccess);

    if (use_tag) {
        native_last = {
            .valid = true, .k2 = tag.k2, .pipeline = pipeline, .tick = scheduler.CurrentTick()};
    }
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset, const DrawPlan* plan) {
    // B11: the work inside an engine clear is run (see Draw)
    BarrierCensus::SetOp(0x4000000000000000ull);
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    // audit context of this draw (class 4: indirect draws); without it the binds of this draw were compared with the previous normal draw
    Core::NativeRegistry::SetDrawClass(4);
    Core::NativeRegistry::SetCurrentDraw(plan != nullptr && plan->native_seq != 0 ? plan->native_seq : 0, plan != nullptr && plan->native_seq != 0);
    if (Core::Prof::Running()) {
        TLG_PROF_TAG_HEX("caller", plan != nullptr && plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0);
        TLG_PROF_TAG("pass_class", 4, PassClassName(4));
    }

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
    const FixedState& fixed = plan->fixed;
    draw_inputs = &plan->inputs;
    current_bc_base = &draw_inputs->ta_bc_base;
    current_cb_extent = plan ? &plan->cb_extent : &executor->RasterCbExtents();
    current_db_extent = plan ? &plan->db_extent : &executor->RasterDbExtent();
    current_target_resources = plan ? &plan->target_resources : &executor->RasterTargetResources();

    PrepareRenderState(pipeline, fixed);
    binder_plan = NativeBinderFor(plan, 4);
    resolved_draw = binder_plan == plan ? plan : nullptr;
    const bool bound = BindResources(pipeline, &fixed);
    resolved_draw = nullptr;
    binder_plan = nullptr;
    if (!bound) {
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

    const bool pass_barrier = needs_barrier;
    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed, fixed);
    scheduler.BeginRendering(state);

    NoteMarkerRow(1, 4, pipeline, plan != nullptr ? plan->native_seq : 0, draw_inputs, max_count, 0);
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

void Rasterizer::PreSynchronize(const std::vector<Core::NativeUploads::Range>& ranges) {
    scheduler.PopPendingOperations();
    u64 bytes = 0, uploaded = 0;
    for (const auto& range : ranges) {
        if (!IsMapped(range.begin, range.size)) {
            continue;
        }
        bytes += range.size;
        uploaded += buffer_cache.PreSynchronize(range.begin, range.size);
    }
    Core::NativeUploads::NoteApplied(ranges.size(), bytes, uploaded);
}

void Rasterizer::SetExplicitImageReadback(bool on) {
    texture_cache.SetExplicitDownload(on);
}

void Rasterizer::DispatchDirect(const DispatchPlan* plan) {
    struct W3Scope {
        Rasterizer* self;
        u64 begin{0};
        u64 program{0};
        u64 groups{0};
        bool barrier{false};
        explicit W3Scope(Rasterizer* r) noexcept : self(r) {
            if (self->cover_on) [[unlikely]] {
                begin = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                t_w3 = W3Cur{};
                t_w1_bind_begin = t_w1_bind_end = 0;
            }
        }
        ~W3Scope() {
            if (begin == 0 || t_w1_bind_begin == 0) [[likely]] {
                return; // census off, or the dispatch never reached the bind (cleared, HLE, no pipeline)
            }
            const u64 end = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
            std::scoped_lock lock{g_w3_mutex};
            auto& st = g_w3[program];
            auto& v = st.v;
            v[0] += 1;
            v[1] += end - begin;
            v[2] += t_w1_bind_begin - begin;
            v[3] += t_w1_bind_end - t_w1_bind_begin;
            v[4] += end - t_w1_bind_end;
            v[5] += t_w3.sync_ns;
            v[6] += t_w3.upload;
            for (u32 k = 1; k <= 7; ++k) {
                v[6 + k] += t_w3.kinds[k];
            }
            v[14] += t_w3.images;
            v[15] += t_w3.written;
            v[16] += t_w3.stream;
            v[17] += t_w3.with_upload;
            v[18] += barrier ? 1 : 0;
            v[19] += groups;
            for (u32 k = 0; k < 8; ++k) {
                v[20 + k] += t_w3.why[k];
            }
            v[28] += t_w3.written_id[0];
            v[29] += t_w3.written_id[1];
            // design review 66: the bindings rows ahead bound, and the dispatches whose every shader buffer was answered without the per-binding old path
            // (ring, special, null, the front's resolution, a planned row)
            v[30] += t_w3.fast;
            const u32 full_path = t_w3.kinds[4] + t_w3.kinds[5] + t_w3.kinds[6] - std::min<u32>(t_w3.fast, t_w3.kinds[4] + t_w3.kinds[5] + t_w3.kinds[6]);
            v[31] += full_path == 0 ? 1 : 0;
            for (u32 k = 0; k < 6; ++k) {
                v[32 + k] += t_w3.phase[k];
            }
            for (u32 k = 0; k < 3; ++k) {
                v[38 + k] += t_w3.buf[k];
            }
            v[41] += t_w3.buf_n;
            v[42] += t_w3.wr_ns;
            v[43] += t_w3.wr_n;
            v[44] += t_w3.rd_ns;
            v[45] += t_w3.rd_n;
            v[46] += t_w3.rd_row_ns;
            v[47] += t_w3.rd_row_n;
        }
    } w3_scope{this};
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PEntry);
    Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseDispatch, engine_clear_on);
    if (SkipForEngineClear() && plan != nullptr) {
        const bool own = IsEngineClearOwnWork(plan->native_seq);
        Core::NativeCensus::NoteSkippedInClear(own ? 1 : 3, plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0, plan->cs.dim_x);
        if (own) {
            return; // the clear's own fill: the native clear did it
        }
    }
    RENDERER_TRACE;
    Core::NativeRegistry::SetDrawClass(3);
    Core::NativeRegistry::SetCurrentDraw(plan != nullptr ? plan->native_seq : 0, plan != nullptr && plan->native_seq != 0);
    if (Core::Prof::Running()) {
        TLG_PROF_TAG_HEX("caller", plan != nullptr && plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0);
        TLG_PROF_TAG("pass_class", 3, PassClassName(3));
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PClassTags);

    {
        TLG_PROF_LEVEL("raster.pop_pending", 3);
        scheduler.PopPendingOperations();
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PPopPending);

    {
        TLG_PROF_LEVEL("raster.dispatch_fill", 3);
        if (const auto fill = std::exchange(pending_fill, {});
            fill.dwords != 0 && TryClearImage(fill.dst, fill.bytes, fill.pattern.data(), fill.dwords)) {
            return;
        }
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PFill);

    current_cs = plan ? &plan->cs : &no_program;
    current_bc_base = plan ? &plan->ta_bc_base : &no_border_colors;
    const auto& cs_program = *current_cs;
    w3_scope.program = cs_program.address;
    w3_scope.groups = static_cast<u64>(cs_program.dim_x) * cs_program.dim_y * cs_program.dim_z;
    BarrierCensus::SetOp(cs_program.address);
    // G2 (TLG_SIM_INDEPENDENT): the SimParticle dispatches are one independent group (Phase 0b)
    bool group_in_use = false;
    struct GroupScope {
        Runtime& rt;
        ~GroupScope() {
            rt.SetIndependentGroup(0);
        }
    } group_scope{runtime};
    if (IndependentGroup::Mode() != 0 && plan != nullptr) {
        static std::unordered_map<u64, bool> is_group;
        auto it = is_group.find(cs_program.address);
        if (it == is_group.end()) {
            it = is_group.emplace(cs_program.address, Core::NativeDraw::IsShaderEntry(cs_program.address, "CS_SimParticle", "SimParticle")).first;
        }
        group_in_use = it->second;
        runtime.SetIndependentGroup(it->second ? 1u : 0u);
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::POpGroup);
    const ComputePipeline* pipeline = ResolveDispatchPipeline(plan);
    if (!pipeline) {
        return;
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PPipeline);
    if (Core::LightRecorder::On()) {
        Core::LightRecorder::FrameOp(Core::DebugServer::PresentCount(), 0, cs_program.address, texture_cache.ImageGeneration());
    }
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    {
        TLG_PROF_LEVEL("raster.shader_hle", 3);
        if (ExecuteShaderHLE(cs, cs_program, *this)) {
            return;
        }
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PLightHle);

    binder_plan = DispatchBinderFor(plan);
    Core::DispatchSplit::NotePlanned(binder_plan != nullptr);
    resolved_dispatch = binder_plan != nullptr ? plan : nullptr;
    const bool light_recorded = Core::LightRecorder::On() && plan != nullptr && plan->native_seq != 0 && Core::LightRecorder::IsLightCaller(Core::NativeDraw::RecordCallerGhidra(plan->native_seq));
    {
        // Dark frame experiment (`TLG_LIGHT_FULL_BARRIER=1`): a full memory barrier (every earlier command's writes visible to every later command) before each dispatch of the four
        // lighting passes (LightVolume 0x1c6456d, its rasterize 0x1c67f59 and chromatic aberration 0x1c68036, ambient lighting 0x1c44f64). If the dark frames go away with it, the
        // sun / fog passes read the shadow maps (or another input) before the work that writes them was ordered before them.
        static const bool full_barrier = [] {
            const char* e = std::getenv("TLG_LIGHT_FULL_BARRIER");
            return e != nullptr && *e == '1';
        }();
        if (full_barrier && plan != nullptr && plan->native_seq != 0) {
            const u64 caller = Core::NativeDraw::RecordCallerGhidra(plan->native_seq);
            if (caller == 0x1c6456d || caller == 0x1c67f59 || caller == 0x1c68036 || caller == 0x1c44f64) {
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
                static std::atomic<u64> issued{0};
                if (issued.fetch_add(1, std::memory_order_relaxed) == 0) {
                    LOG_WARNING(Render_Vulkan, "TLG_LIGHT_FULL_BARRIER: full barriers before the lighting dispatches");
                }
            }
        }
    }
    if (light_recorded) {
        Core::LightRecorder::Begin(Core::DebugServer::PresentCount(), Core::NativeDraw::RecordCallerGhidra(plan->native_seq), cs_program.address);
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PBinderPlan);
    const bool dispatch_bound = BindResources(pipeline);
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PBind);
    w3_scope.barrier = needs_barrier;
    if (light_recorded) {
        Core::LightRecorder::End();
    }
    resolved_dispatch = nullptr;
    binder_plan = nullptr;
    if (!dispatch_bound) {
        return;
    }

    BarrierCensus::Dispatch(cs_program.address, needs_barrier, cs.pgm_hash);
    if (IndependentGroup::Mode() != 0 && group_in_use) {
        bool real_hazard = false, aware_hazard = false;
        runtime.TakeHazardFlags(real_hazard, aware_hazard);
        IndependentGroup::NoteDispatch(needs_barrier, real_hazard, aware_hazard);
    }
    if (BarrierCensus::On() && plan != nullptr && Core::NativeDraw::IsShaderEntry(cs_program.address, "CS_SimParticle", "SimParticle")) {
        // G2 step 1 (counters only): the slice [base, base + count) x 240 bytes of the particle pool this dispatch touches. Shader read 2026-11-09: buffer 0 holds count at dword 104 and
        // base at dword 106, buffer 1 is the pool with a 60 dword record; thread i < count touches record base + i.
        struct Located {
            u64 address{};
            u32 size{};
            const u8* data{};
            bool found{};
        };
        auto locate = [&](u32 index) {
            Located out;
            for (const auto& rb : plan->stage.ring) {
                if (rb.index == index) {
                    out.address = rb.address;
                    out.size = rb.size;
                    out.data = rb.hook ? Core::HookRing::Data(rb.offset) : constant_ring->Data(rb.offset);
                    out.found = true;
                    return out;
                }
            }
            if (index < plan->stage.resolved.size() && plan->stage.resolved[index].address != 0) {
                out.address = plan->stage.resolved[index].address;
                out.size = plan->stage.resolved[index].size;
                out.found = true;
            }
            return out;
        };
        const Located ctrl = locate(0);
        {
            u32 position = 0;
            for (const auto& [bbuf, boffset, bsize, baccess] : bound_buffers) {
                BarrierCensus::SimBinding(cs_program.address, position++, bsize, static_cast<bool>(baccess & vk::AccessFlagBits2::eShaderWrite));
            }
        }
        // the pool is the written buffer: among the buffers this dispatch bound, the largest one with shader write access (key = the Vulkan buffer object, offsets inside it)
        Located pool;
        u64 pool_key = 0, pool_offset = 0;
        for (const auto& [bbuf, boffset, bsize, baccess] : bound_buffers) {
            if ((baccess & vk::AccessFlagBits2::eShaderWrite) && bsize > pool.size) {
                pool.size = static_cast<u32>(bsize);
                pool.found = true;
                pool_key = reinterpret_cast<u64>(bbuf);
                pool_offset = boffset;
            }
        }
        u32 status = 0, count = 0, base = 0;
        u64 begin = 0, end = 0;
        bool from_ring = false;
        std::array<u32, 107> words{};
        if (!ctrl.found) {
            status = 1;
        } else if (ctrl.size < 107 * 4) {
            status = 2;
        } else if (!pool.found) {
            status = 5;
        } else {
            if (ctrl.data != nullptr) {
                std::memcpy(words.data(), ctrl.data, sizeof(words));
                from_ring = true;
            } else if (memory->IsValidMapping(ctrl.address, sizeof(words))) {
                memory->CopySparseMemory(ctrl.address, reinterpret_cast<u8*>(words.data()), sizeof(words));
            } else {
                status = 1;
            }
            if (status == 0) {
                count = words[104];
                base = words[106];
                const u64 n = std::min<u64>(count, static_cast<u64>(cs_program.dim_x) * 64);
                if (n == 0) {
                    status = 3;
                } else if ((static_cast<u64>(base) + n) * 240 > pool.size) {
                    status = 4;
                } else {
                    begin = pool_offset + static_cast<u64>(base) * 240;
                    end = pool_offset + (static_cast<u64>(base) + n) * 240;
                }
            }
        }
        BarrierCensus::SimSlice(cs_program.address, status, from_ring, pool_key, begin, end, count, base);
        if (status == 0 && !bound_buffers.empty()) {
            // render slot probe: word 6 of record base + i in the record buffer (the first buffer this dispatch bound), as guest memory holds it now
            const auto& [rbuf, roffset, rsize, raccess] = bound_buffers.front();
            const u32 n_read = static_cast<u32>(std::min<u64>(std::min<u64>(count, static_cast<u64>(cs_program.dim_x) * 64), 512));
            std::array<u32, 512> slots{}, index{};
            bool readable = rbuf->CpuAddr() != 0 && (static_cast<u64>(base) + n_read) * 240 <= rsize;
            for (u32 i = 0; readable && i < n_read; ++i) {
                const VAddr a = rbuf->CpuAddr() + roffset + static_cast<u64>(base + i) * 240 + 24;
                if (!memory->IsValidMapping(a, 4)) {
                    readable = false;
                    break;
                }
                memory->CopySparseMemory(a, reinterpret_cast<u8*>(&slots[i]), 4);
                index[i] = base + i;
            }
            BarrierCensus::SimSlots(slots.data(), index.data(), n_read, readable);
        }
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PAfterBindCensus);
    if (needs_barrier) {
        TLG_PROF_LEVEL("raster.flush_barriers", 2);
        runtime.FlushBarriers();
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PFlushBarriers);

    {
        TLG_PROF_LEVEL("sched.end_rendering", 3);
        scheduler.EndRendering();
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PEndRendering);
    pipeline->BindResources(set_writes, push_data);
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PDescriptorPush);

    NoteMarkerRow(2, 3, pipeline, plan != nullptr ? plan->native_seq : 0, nullptr, cs_program.dim_x, cs_program.dim_y);
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PMarkerRow);
    const auto cmdbuf = scheduler.CommandBuffer();
    {
        TLG_PROF_LEVEL("raster.record_dispatch", 2);
        scheduler.StampDispatch(true, cs_program.address);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
        scheduler.StampDispatch(false, cs_program.address);
    }
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PRecord);
    DebugState.IncDispatch();
    ++recorded_work;
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PCount);

    ResetBindings(true);
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PResetBindings);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size, const DispatchPlan* plan) {
    // B11: the work inside an engine clear is run (see Draw)
    RENDERER_TRACE;
    Core::NativeRegistry::SetDrawClass(3);
    Core::NativeRegistry::SetCurrentDraw(plan != nullptr ? plan->native_seq : 0, plan != nullptr && plan->native_seq != 0);
    if (Core::Prof::Running()) {
        TLG_PROF_TAG_HEX("caller", plan != nullptr && plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0);
        TLG_PROF_TAG("pass_class", 4, PassClassName(4));
    }

    {
        TLG_PROF_LEVEL("raster.pop_pending", 3);
        scheduler.PopPendingOperations();
    }

    current_cs = plan ? &plan->cs : &no_program;
    current_bc_base = plan ? &plan->ta_bc_base : &no_border_colors;
    BarrierCensus::SetOp(current_cs->address);
    const ComputePipeline* pipeline = ResolveDispatchPipeline(plan);
    if (!pipeline) {
        return;
    }

    binder_plan = DispatchBinderFor(plan);
    resolved_dispatch = binder_plan != nullptr ? plan : nullptr;
    // dark frame: the indirect dispatches (the deferred sun pass CS_Deferred is one, two a frame) are recorded with all their inputs, caller tag 0xd15 + the record's call site
    const bool indirect_recorded = Core::LightRecorder::On();
    if (indirect_recorded) {
        Core::LightRecorder::Begin(Core::DebugServer::PresentCount(), 0xd1500000000ull | (plan != nullptr && plan->native_seq != 0 ? Core::NativeDraw::RecordCallerGhidra(plan->native_seq) : 0), current_cs->address);
    }
    const bool dispatch_bound = BindResources(pipeline);
    if (indirect_recorded) {
        Core::LightRecorder::End();
    }
    resolved_dispatch = nullptr;
    binder_plan = nullptr;
    if (!dispatch_bound) {
        return;
    }

    if (Core::LightRecorder::On()) {
        // dark frame: the indirect arguments (the deferred sun pass takes its tile count from them) as the buffer cache sees them before this request
        const u64 present = Core::DebugServer::PresentCount();
        Core::LightRecorder::FrameOp(present, 0, 0x1d1d1d1d, texture_cache.ImageGeneration());
        Core::LightRecorder::IndirectArgs(present, buffer_cache.IsRegionCpuModified(address + offset, size), buffer_cache.IsRegionGpuModified(address + offset, size),
                                          buffer_cache.IsRangeGpuWritten(address + offset, size));
    }
    {
        static const bool full_barrier = [] {
            const char* e = std::getenv("TLG_LIGHT_FULL_BARRIER");
            return e != nullptr && *e == '1';
        }();
        if (full_barrier) {
            // TLG_LIGHT_FULL_BARRIER: every indirect dispatch too (the deferred sun pass is one): the writes of its arguments and inputs are visible to it
            runtime.FlushBarriers();
            scheduler.EndRendering();
            const vk::MemoryBarrier2 barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            };
            scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        }
    }
    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    BarrierCensus::Dispatch(current_cs->address, needs_barrier);
    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    NoteMarkerRow(3, 3, pipeline, plan != nullptr ? plan->native_seq : 0, nullptr, 0, 0);
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();
    ++recorded_work;

    ResetBindings(true);
}

void Rasterizer::SelectStream(bool side) {
    // TLG_ASC_WARMUP: the windows of queue 1 that stay on the graphics stream before the side stream starts (default 900: an entry at 240 windows, during the load of the scene, loses the device even with the waits)
    static const u32 WarmupWindows = [] { const char* e = std::getenv("TLG_ASC_WARMUP"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 900u; }();
    const bool new_window = side && !side_requested;
    side_requested = side;
    if (new_window) {
        ++side_windows;
    }
    if (side == (side_scope != nullptr)) {
        return;
    }
    if (side && (!scheduler.HasSideStream() || side_windows <= WarmupWindows)) {
        return;
    }
    if (side) {
        side_ready.store(true, std::memory_order_release);
    }
    // Barriers recorded so far belong to the stream being left.
    runtime.FlushBarriers();
    // Diagnostic (side stream hunt, TLG_ASC_SERIALIZE=1): both streams are made idle at every switch, so no cross queue race can exist. A crash that stays is not a race.
    // 1 = at both switches, 2 = only when entering the side stream (graphics results must be done before queue 1 reads them), 3 = only when leaving it.
    static const int serialize = [] { const char* e = std::getenv("TLG_ASC_SERIALIZE"); return e != nullptr ? std::atoi(e) : 0; }();
    if ((serialize == 1 || (serialize == 2 && side) || (serialize == 3 && !side)) && Core::NativeOcclusion::query_open.load(std::memory_order_acquire) == 0) {
        scheduler.Finish();
    }
    // The two streams are two VkQueues: entering the side stream submits the graphics work recorded so far and the side submission waits for it, leaving it makes the next
    // graphics submission wait for the side work (TLG_ASC_SYNC=0 turns this off, only for a comparison).
    static const bool stream_sync = [] { const char* e = std::getenv("TLG_ASC_SYNC"); return e == nullptr || e[0] != '0'; }();
    if (side) {
        if (stream_sync && !scheduler.OrderSideAfterGraphics(Core::NativeOcclusion::query_open.load(std::memory_order_acquire) == 0)) {
            return; // an occlusion query is open: this operation runs on the graphics stream, the side stream is entered at a later one
        }
        const u64 now = Core::FrameTimeline::Now();
        if (now - side_last_end_ns > 3'000'000) {
            buffer_cache.SetSideUploadEpoch(++side_epoch); // the work of the next frame: the CPU may have rewritten the inputs
        }
        side_scope = std::make_unique<Scheduler::SideScope>(scheduler, true);
        static bool first_entry_logged = false;
        if (!first_entry_logged) {
            first_entry_logged = true;
            LOG_ERROR(Render_Vulkan, "side stream: first entry at window {}, frame {}", side_windows, Core::FrameTimeline::Now() / 1'000'000);
        }
    } else {
        side_scope.reset();
        if (stream_sync) {
            scheduler.OrderGraphicsAfterSide();
        }
        side_last_end_ns = Core::FrameTimeline::Now();
    }
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
    if (occlusion) {
        occlusion->Poll();
    }
    if (encoder_debug & 2) {
        scheduler.SyncEncoder();
    }
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

bool Rasterizer::StageFlatSkipped(const Shader::Info& stage) const {
    const u32 index = static_cast<u32>(stage.sw_stage);
    return binder_plan != nullptr && binder_plan->any_flat_skipped && index < MaxShaderStages && binder_plan->stages[index].flat_skipped;
}

/// Draw binding cache, step 0 (TLG_BIND_CENSUS=1, raster-optimization-plan.md phase 2): would a cache of what a draw's buffer bindings produced hit, and would it be right?
/// The Nth use of a pipeline in a frame is taken as "the same draw" as the Nth use in the previous frame; the descriptors it got (buffer handle, offset, range) are hashed and compared
/// with what that draw got the last time. Same: a cache keyed by the draw's inputs could have answered. Different: the answer moved (a copy in the stream ring, a new upload). Counts
/// are split by whether any buffer of the draw is a stream ring copy (those move every command buffer by design). Logged every 20000 binds.
void Rasterizer::BindCensus(const Pipeline* pipeline) {
    if (!BindCensusOn()) {
        return;
    }
    static thread_local u64 frame = ~0ull;
    static thread_local std::unordered_map<u64, u32> ordinals;
    static thread_local std::unordered_map<u64, std::pair<u64, u64>> last; // key -> (hash, frame)
    static thread_local u64 counts[2][4]{};                                  // [has stream copy][new, same as last frame, different, no buffers]
    static thread_local u64 total = 0;
    const u64 now = Core::DebugServer::PresentCount();
    if (now != frame) {
        frame = now;
        ordinals.clear();
    }
    const u32 ordinal = ordinals[reinterpret_cast<u64>(pipeline)]++;
    const u64 key = reinterpret_cast<u64>(pipeline) * 0x9E3779B97F4A7C15ull ^ (u64{ordinal} * 0xC2B2AE3D27D4EB4Full);
    u64 h = 0xcbf29ce484222325ull;
    bool has_stream = false;
    for (const auto& info : buffer_infos) {
        const u64 raw = reinterpret_cast<u64>(static_cast<VkBuffer>(info.buffer));
        has_stream |= buffer_cache.IsStreamHandle(info.buffer);
        h = (h ^ raw) * 0x100000001b3ull;
        h = (h ^ info.offset) * 0x100000001b3ull;
        h = (h ^ info.range) * 0x100000001b3ull;
    }
    const u32 split = has_stream ? 1 : 0;
    if (buffer_infos.empty()) {
        ++counts[split][3];
    } else if (const auto [it, inserted] = last.try_emplace(key, std::pair<u64, u64>{h, now}); inserted) {
        ++counts[split][0];
    } else {
        ++counts[split][it->second.first == h ? 1 : 2];
        it->second = {h, now};
    }
    if (++total % 20000 == 0) {
        LOG_WARNING(Render_Vulkan, "bind census: plain draws new {} same {} different {} none {} | with stream copies new {} same {} different {} none {}", counts[0][0], counts[0][1], counts[0][2],
                 counts[0][3], counts[1][0], counts[1][1], counts[1][2], counts[1][3]);
        if (last.size() > 200000) {
            last.clear();
        }
    }
}

namespace {
thread_local u64 t_fseg[7] = {}; // design review 66 (TLG_COVER_CENSUS, one fast binding in 8): stamps through the rows ahead fast path and the tail of the loop
thread_local bool t_fseg_on = false;
thread_local const ResolvedBuffer* t_ahead_sample = nullptr; // rows ahead real: the planned row of the binding sampled through the full path
thread_local u64 t_bseg[6] = {}; // TLG_COVER_CENSUS: stamps of the handle path of one binding (1 after the descriptor, 2 before the row, 3 after the row, 4 after the answer, 5 before the binding is written)
thread_local bool t_bseg_served = false;
inline u64 BsegNow() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
thread_local u32 t_cover_mask = 0;
thread_local u32 t_cover_kind = 0; // R1': the kind of the last binding the stage loop counted
thread_local u64 t_cover_begin = 0;
} // namespace

enum W1Count : u32 {
    W1Passes = 0, W1Draws, W1SumPipelines, W1MaxPipelines, W1PipesLe8, W1PipesLe16, W1PipesLe24, W1PipesMore, W1SumGeometry, W1GeomLe10, W1GeomLe33, W1GeomLe66, W1GeomMore, W1SumOther,
    W1SumRing, W1SumState, W1Pairs, W1SamePipeline, W1SameGeometry, W1SameOther, W1SamePush, W1SameState, W1SameVertex, W1SameIndex, W1RingPerDraw, W1OtherPerDraw, W1ImagesPerDraw,
    W1Indices, W1Instances, W1PassesOne, W1DrawsInPassesOver50, W1SumIndexHandles, W1SumOtherHandles,
};

/// W1.1 counters in `w1s_counts`: 0 shadow draws, 1 eligible (no bad bit), 2 eligible when the front produces the flat constants, 3 eligible when stream copies are allowed too,
/// 4..13 draws with bad bit n, 14..23 draws whose only bad bit is n, 24 sum of integers in the draw record, 25 largest record, 26..29 shader buffer rows (same, DIFFERENT, learned,
/// epoch moved), 30..33 geometry rows (same, DIFFERENT, learned, epoch moved), 34 handle shader buffers answered from the stream ring, 35 from a resident buffer, 36 resident
/// with uploaded bytes, 37 geometry from the stream ring, 38 geometry resident, 39 geometry with uploaded bytes, 40 geometry without an engine handle.
void Rasterizer::W1ShadowRow(u64 id, VAddr address, const void* buffer, u64 offset, u32 base) noexcept {
    // slots base..base+3: same, DIFFERENT, learned, epoch moved (graphics shader buffers 26, geometry 30, compute read 80, compute written 84); id 0: no engine id, nothing to key
    if (id == 0) {
        return;
    }
    auto& s = w1s_counts;
    const u64 epoch = buffer_cache.PlacementEpoch();
    const auto it = t_w1_rows.find(id);
    if (it == t_w1_rows.end()) {
        t_w1_rows.emplace(id, W1Row{buffer, address - offset, epoch});
        s[base + 2].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    W1Row& r = it->second;
    if (r.epoch != epoch) {
        r = W1Row{buffer, address - offset, epoch};
        s[base + 3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool same = r.buffer == buffer && address >= r.base && address - r.base == offset;
    s[base + (same ? 0u : 1u)].fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 8) {
            LOG_ERROR(Render_Vulkan, "W1 shadow row differs: id {:#x} slot {} address {:#x}: row {:p} base {:#x}, answer {:p} offset {:#x}", id, base, address, r.buffer, r.base, buffer, offset);
        }
        r = W1Row{buffer, address - offset, epoch}; // follow the old path so one difference is counted once
    }
}

void Rasterizer::W1ShadowBinding(const Core::NativeResources::ImmutableBuffer* hint, u64 record_id, VAddr address, const void* buffer, u64 offset, bool written, u32 why, bool formatted) noexcept {
    auto& c = t_w1;
    auto& s = w1s_counts;
    // W1.2 question: why has this binding no engine handle? 41..48 by reason (0 has a handle, 1 no binder plan, 2 not a single load, 3 no record entry, 4 plan has no handles, 5 id not in the
    // registry, 6 handle epoch stale), 49..56 the same for formatted (texel) buffers
    s[41 + (why & 7)].fetch_add(1, std::memory_order_relaxed);
    if (formatted) {
        s[49 + (why & 7)].fetch_add(1, std::memory_order_relaxed);
    }
    const bool stream = buffer_cache.IsStreamBuffer(static_cast<const VideoCore::Buffer*>(buffer));
    const bool work = VideoCore::BufferCache::UploadBytes() != 0;
    if (written) {
        c.bad |= W1BadWritten;
    }
    if (hint == nullptr) {
        c.bad |= W1BadAddress;
    }
    if (stream) {
        c.bad |= W1BadStream;
    }
    if (work) {
        c.bad |= W1BadWork;
    }
    if (hint != nullptr) {
        s[stream ? 34 : 35].fetch_add(1, std::memory_order_relaxed);
        if (!stream && work) {
            s[36].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!stream && !written) {
        W1ShadowRow(hint != nullptr ? hint->id : record_id, address, buffer, offset, 26);
    }
}

void Rasterizer::W1ShadowGeometry(const Core::NativeResources::ImmutableBuffer* hint, VAddr address, const void* buffer, u64 offset, bool /*vertex*/) noexcept {
    auto& c = t_w1;
    auto& s = w1s_counts;
    const bool stream = buffer_cache.IsStreamBuffer(static_cast<const VideoCore::Buffer*>(buffer));
    const bool work = VideoCore::BufferCache::UploadBytes() != 0;
    if (hint == nullptr) {
        c.bad |= W1BadGeoAddress;
        s[40].fetch_add(1, std::memory_order_relaxed);
        // 57 no geometry plan, 58 plan has no handles, 59 handle epoch stale, 60 no handle holds the request
        const u32 reason = geometry_plan == nullptr ? 57u : !geometry_plan->has_geometry_handles ? 58u : geometry_plan->handle_epoch != Core::NativeResources::ImmutableEpoch() ? 59u : 60u;
        s[reason].fetch_add(1, std::memory_order_relaxed);
    }
    if (stream) {
        c.bad |= W1BadGeoStream;
        s[37].fetch_add(1, std::memory_order_relaxed);
    } else {
        s[38].fetch_add(1, std::memory_order_relaxed);
        if (work) {
            s[39].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (work) {
        c.bad |= W1BadGeoWork;
    }
    if (!stream) {
        W1ShadowRow(hint != nullptr ? hint->id : 0, address, buffer, offset, 30);
    }
}

/// W3.2 counters in `w1s_counts`: 128 requests served from a row, 129 of them also run by the old path (1 in 64), 130 verified answers that differ (MUST_BE_0), 131 verified requests where the old
/// path did work although the row was valid (MUST_BE_0), 132 rows learned, 133 not learned (texel buffer over an image or image metadata), 134 no row, 135 row stale (epoch, identity event, dirty bytes)
bool Rasterizer::RowsServe(u64 id, VAddr address, u64 size, bool texel, const VideoCore::Buffer*& buffer, u64& offset, bool written) noexcept {
    if (id == 0) {
        return false;
    }
    auto& s = w1s_counts;
    const auto it = t_rows_real.find(RowKey(id, address, size, texel, written));
    if (it == t_rows_real.end() || it->second.id != id || it->second.address != address || it->second.size != size || it->second.texel != texel || it->second.written != written) {
        s[134].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    RealRow& r = it->second;
    // (ChangedSince walks the identity events after the stored clock: the clock moves up to the one read before each successful check, so the walk covers only what is new)
    const u64 clock_now = Core::NativeIdentity::Clock();
    const auto sets_dirty = [&] {
        if (VideoCore::g_producer_state) [[unlikely]] {
            return buffer_cache.ObjectRangeDirtyClock(id, static_cast<const VideoCore::Buffer*>(r.buffer), address, size); // (only reached with the epoch unchanged: the buffer is alive)
        }
        const bool d = buffer_cache.ObjectRangeDirty(id, address, size);
        if (VideoCore::g_page_clock_on) [[unlikely]] {
            VideoCore::PageClockQueryShadow(static_cast<const VideoCore::Buffer*>(r.buffer), address, size, d); // (only reached with the epoch unchanged: the buffer is alive)
        }
        return d;
    };
    if (Core::RowVersion::WalkCensusOn() && r.epoch == buffer_cache.PlacementEpoch()) [[unlikely]] {
        const u64 t0 = Core::N0::NowNs();
        const auto answer = Core::NativeIdentity::ChangedSince(r.clock, address, address + size);
        Core::RowVersion::NoteWalk(clock_now > r.clock ? clock_now - r.clock : 0, Core::N0::NowNs() - t0, static_cast<u32>(answer));
    }
    if (r.epoch != buffer_cache.PlacementEpoch() || Core::NativeIdentity::ChangedSince(r.clock, address, address + size) != Core::NativeIdentity::Change::None) {
        s[135].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // (the GPU-written state of the range is covered by the identity events: every change of the set logs one)
    if (Core::RowVersion::Mode() == 2) {
        // F2 row version, real: one integer compare decides (rows exist only inside their buffer); one check in 64 also asks the sets, which win
        bool clean = r.inside && r.version == Core::RowVersion::Of(id);
        if (!clean) {
            Core::RowVersion::NoteReal(4);
        } else {
            static thread_local u32 sample = 0;
            if ((++sample & 63) == 0) {
                Core::RowVersion::NoteReal(1);
                if (sets_dirty()) {
                    Core::RowVersion::NoteReal(2);
                    static std::atomic<u32> logged{0};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                        LOG_ERROR(Render_Vulkan, "F2 row version real: id {:#x} {:#x}+{:#x}: the sets say dirty, the version said clean (not served)", id, address, size);
                    }
                    clean = false;
                }
            }
        }
        if (!clean) {
            s[135].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        Core::RowVersion::NoteReal(0);
    } else if (Core::RowVersion::Mode() == 1) [[unlikely]] {
        // F2 row version, shadow: the sets decide; the versions' answer is compared with theirs
        const bool version_clean = r.version == Core::RowVersion::Of(id) && (r.inside || r.unattr == Core::RowVersion::Unattributed());
        const u64 t0 = Core::N0::NowNs();
        const bool d = sets_dirty();
        Core::RowVersion::NoteCheck(version_clean, d, r.inside, Core::N0::NowNs() - t0, id, address, size);
        if (d) {
            s[135].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    } else if (sets_dirty()) {
        s[135].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    r.clock = clock_now;
    buffer = static_cast<const VideoCore::Buffer*>(r.buffer);
    offset = r.offset;
    t_row_gpu = r.gpu;
    t_row_static = r.is_static;
    t_row_hash = r.hash;
    return true;
}

void Rasterizer::RowsLearn(u64 id, VAddr address, u64 size, bool texel, const VideoCore::Buffer* buffer, u64 offset, u64 epoch_pre, u64 clock_pre, bool gpu_pre, u64 version_pre,
                           u64 unattr_pre, bool written) noexcept {
    if (id == 0 || buffer == nullptr || buffer_cache.IsStreamBuffer(buffer)) {
        return;
    }
    if (Core::N0::On()) [[unlikely]] {
        Core::N0::NoteRowLearn(Core::NativeContent::InRing(address, size)); // F1 1c census
    }
    if (Core::NativeUpdates::RingLeanApplyOn() && Core::NativeContent::InRing(address, size)) {
        Core::NativeUpdates::NoteRingLean(3); // F1 1c real: a ring range changes every lap without the marks a row check reads; it is never learned
        return;
    }
    if (texel && (texture_cache.AnyImageOverlaps(address, size) || texture_cache.IsMeta(address))) {
        w1s_counts[133].fetch_add(1, std::memory_order_relaxed); // a texel buffer over an image: its bytes change with the image, no stamp covers that
        return;
    }
    const bool gpu = buffer_cache.IsRangeGpuWritten(address, size);
    if (written && (!gpu || !gpu_pre)) {
        return; // a written row is learned once the range is GPU written before and after the old path's answer (its first write changes that state)
    }
    if (gpu != gpu_pre) {
        w1s_counts[133].fetch_add(1, std::memory_order_relaxed); // the GPU written state changed while the old path answered (a download completed): the answer says nothing about either state
        return;
    }
    if (texel && size <= 16384 && !gpu) {
        // (diagnostics) a small texel buffer that is not GPU written is a stream-ring copy for the old path; an answer from a buffer is unexpected
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 10) {
            LOG_ERROR(Render_Vulkan, "W3.2 learned a small non-GPU-written texel row: id {:#x} address {:#x}+{:#x}: buffer {:p}+{:#x} stream {} gpu pre {} now {}", id, address, size, static_cast<const void*>(buffer), offset,
                      buffer_cache.IsStreamBuffer(buffer), gpu_pre, buffer_cache.IsRangeGpuWritten(address, size));
        }
    }
    if (t_rows_real.size() > 400000) {
        t_rows_real.clear();
    }
    bool inside = false;
    if (Core::RowVersion::Mode() != 0) [[unlikely]] {
        Core::NativeDevice::BufferPart part{};
        inside = Core::NativeDevice::BufferPartOf(id, address, size, part) && part.begin == address && part.end == address + size;
        Core::RowVersion::NoteLearn(inside);
        if (!inside && Core::RowVersion::Mode() == 2) {
            Core::RowVersion::NoteReal(3); // (the shadow found every such row called dirty by the sets: none was ever served)
            return;
        }
    }
    {
        const u64 key = RowKey(id, address, size, texel, written);
        const RealRow& row = t_rows_real[key] = RealRow{id, address, size, texel, buffer, offset, epoch_pre, clock_pre, gpu, version_pre, unattr_pre, inside, false, 0, written};
        ShareRow(key, row);
    }
    if (written) {
        Core::RowVersion::NoteWritten(1);
    }
    w1s_counts[132].fetch_add(1, std::memory_order_relaxed);
}

/// W3.1 counters in `w1s_counts`, 88 + 16 * (compute ? 1 : 0) + 8 * (written ? 1 : 0) + n: 0 predicted valid, no work, same answer (good), 1 predicted valid but the old path did work (MUST_BE_0),
/// 2 predicted valid but the answer moved to another buffer or offset (needs the per-id version at placement), 3 not predicted, no work (the missed share), 4 not predicted, work (expected),
/// 5 no row yet, 6 not predicted because the id holds dirty bytes, 7 not predicted because the placement epoch moved
void Rasterizer::W31Before(u64 id, VAddr address, u64 size) noexcept {
    auto& p = t_w31;
    p = W31Pre{};
    const auto* list = scheduler.CommandBuffer().List();
    p.commands = list != nullptr ? list->Size() : 0;
    if (id == 0) {
        return;
    }
    const auto it = t_w1_rows.find(id);
    if (it == t_w1_rows.end()) {
        return;
    }
    p.have_row = true;
    p.row_buffer = it->second.buffer;
    p.row_base = it->second.base;
    p.epoch_ok = it->second.epoch == buffer_cache.PlacementEpoch();
    p.dirty = buffer_cache.ObjectRangeDirty(id, address, size);
    p.predicted = p.epoch_ok && !p.dirty;
}

void Rasterizer::W31After(const void* buffer, u64 offset, VAddr address, bool compute, bool written) noexcept {
    const auto& p = t_w31;
    auto& s = w1s_counts;
    const u32 base = 88u + (compute ? 16u : 0u) + (written ? 8u : 0u);
    const auto* list = scheduler.CommandBuffer().List();
    const u64 commands_after = list != nullptr ? list->Size() : 0;
    const bool stream = buffer_cache.IsStreamBuffer(static_cast<const VideoCore::Buffer*>(buffer));
    const bool work = VideoCore::BufferCache::UploadBytes() != 0 || commands_after != p.commands || stream;
    const bool same = p.have_row && p.row_buffer == buffer && address >= p.row_base && address - p.row_base == offset;
    if (!p.have_row) {
        s[base + 5].fetch_add(1, std::memory_order_relaxed);
    }
    if (p.have_row && !p.epoch_ok) {
        s[base + 7].fetch_add(1, std::memory_order_relaxed);
    }
    if (p.have_row && p.dirty) {
        s[base + 6].fetch_add(1, std::memory_order_relaxed);
    }
    if (p.predicted) {
        if (work) {
            s[base + 1].fetch_add(1, std::memory_order_relaxed);
            if (stream && VideoCore::BufferCache::UploadBytes() == 0 && commands_after == p.commands) {
                s[120 + (compute ? 2u : 0u) + (written ? 1u : 0u)].fetch_add(1, std::memory_order_relaxed); // the old path chose a stream-ring copy (a path choice, no upload, no command)
            }
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1) < 10) {
                LOG_ERROR(Render_Vulkan, "W3.1 predicted valid but the old path did work: address {:#x} uploaded {} bytes, {} commands, stream {}, compute {} written {}", address,
                          VideoCore::BufferCache::UploadBytes(), commands_after - p.commands, stream, compute, written);
            }
        } else if (!same) {
            s[base + 2].fetch_add(1, std::memory_order_relaxed);
        } else {
            s[base + 0].fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        s[base + (work ? 4u : 3u)].fetch_add(1, std::memory_order_relaxed);
    }
}

std::vector<std::pair<u64, std::array<u64, 48>>> Rasterizer::W3Rows() const {
    std::vector<std::pair<u64, std::array<u64, 48>>> out;
    std::scoped_lock lock{g_w3_mutex};
    for (const auto& [program, st] : g_w3) {
        std::array<u64, 48> a{};
        std::copy(std::begin(st.v), std::end(st.v), a.begin());
        out.emplace_back(program, a);
    }
    return out;
}

void Rasterizer::W1NoteBinding(const Pipeline* pipeline) noexcept {
    // called from FinishBindResources for every draw: the shader bindings of this draw
    auto& c = t_w1;
    c.pipeline = reinterpret_cast<u64>(pipeline);
    c.ring.clear();
    c.other.clear();
    c.other_handles.clear();
    const VkBuffer ring_handle = constant_ring ? static_cast<VkBuffer>(constant_ring->Handle()) : VK_NULL_HANDLE;
    for (const auto& b : buffer_infos) {
        if (static_cast<VkBuffer>(b.buffer) == ring_handle) {
            c.ring.push_back(W1Mix(W1Mix(0x52494e47ull, b.offset), b.range));
        } else {
            c.other.push_back(W1Mix(W1Mix(reinterpret_cast<u64>(static_cast<VkBuffer>(b.buffer)), b.offset), b.range));
            c.other_handles.push_back(reinterpret_cast<u64>(static_cast<VkBuffer>(b.buffer)));
        }
    }
    c.images = static_cast<u32>(image_infos.size());
    if (c.images != 0) {
        c.bad |= W1BadImage;
    }
    c.push =W1Mix(W1Mix(W1Mix(std::bit_cast<u32>(push_data.xoffset), std::bit_cast<u32>(push_data.yoffset)), std::bit_cast<u32>(push_data.xscale)), std::bit_cast<u32>(push_data.yscale));
    {
        u64 h = W1Mix(0x5348415045ull, c.pipeline);
        for (const auto& b : buffer_infos) {
            if (static_cast<VkBuffer>(b.buffer) == ring_handle) {
                h = W1Mix(W1Mix(h, 0x52494e47ull), b.range); // a ring window: the size matters, the offset is a parameter
            } else {
                h = W1Mix(W1Mix(W1Mix(h, reinterpret_cast<u64>(static_cast<VkBuffer>(b.buffer))), b.offset), b.range);
            }
        }
        for (const auto& i : image_infos) {
            h = W1Mix(W1Mix(W1Mix(h, reinterpret_cast<u64>(static_cast<VkImageView>(i.imageView))), reinterpret_cast<u64>(static_cast<VkSampler>(i.sampler))), static_cast<u64>(i.imageLayout));
        }
        h = W1Mix(h, c.push);
        for (const u8 o : push_data.buf_offsets) {
            h = W1Mix(h, o);
        }
        h = W1Mix(W1Mix(h, push_data.motion_param), push_data.scene_size);
        c.shape = h;
    }
    c.valid = true;
}

void Rasterizer::W1EndDraw() noexcept {
    auto& c = t_w1;
    auto& p = t_w1_pass;
    auto& n = w1_counts;
    const auto flush_pass = [&]() {
        if (p.draws == 0) {
            return;
        }
        n[W1Passes].fetch_add(1, std::memory_order_relaxed);
        n[W1Draws].fetch_add(p.draws, std::memory_order_relaxed);
        const u64 np = p.pipelines.size();
        n[W1SumPipelines].fetch_add(np, std::memory_order_relaxed);
        u64 old = n[W1MaxPipelines].load(std::memory_order_relaxed);
        while (np > old && !n[W1MaxPipelines].compare_exchange_weak(old, np, std::memory_order_relaxed)) {
        }
        n[np <= 8 ? W1PipesLe8 : np <= 16 ? W1PipesLe16 : np <= 24 ? W1PipesLe24 : W1PipesMore].fetch_add(1, std::memory_order_relaxed);
        const u64 ng = p.geometry.size();
        n[W1SumGeometry].fetch_add(ng, std::memory_order_relaxed);
        const double ratio = static_cast<double>(ng) / static_cast<double>(p.draws);
        n[ratio <= 0.10 ? W1GeomLe10 : ratio <= 0.33 ? W1GeomLe33 : ratio <= 0.66 ? W1GeomLe66 : W1GeomMore].fetch_add(1, std::memory_order_relaxed);
        n[W1SumOther].fetch_add(p.other.size(), std::memory_order_relaxed);
        n[W1SumRing].fetch_add(p.ring.size(), std::memory_order_relaxed);
        n[W1SumState].fetch_add(p.state.size(), std::memory_order_relaxed);
        n[W1Pairs].fetch_add(p.pairs, std::memory_order_relaxed);
        n[W1SamePipeline].fetch_add(p.same_pipeline, std::memory_order_relaxed);
        n[W1SameGeometry].fetch_add(p.same_geometry, std::memory_order_relaxed);
        n[W1SameOther].fetch_add(p.same_other, std::memory_order_relaxed);
        n[W1SamePush].fetch_add(p.same_push, std::memory_order_relaxed);
        n[W1SameState].fetch_add(p.same_state, std::memory_order_relaxed);
        n[W1SameVertex].fetch_add(p.same_vertex, std::memory_order_relaxed);
        n[W1SameIndex].fetch_add(p.same_index, std::memory_order_relaxed);
        if (p.draws > 50) {
            n[W1DrawsInPassesOver50].fetch_add(p.draws, std::memory_order_relaxed);
        }
        if (p.draws == 1) {
            n[W1PassesOne].fetch_add(1, std::memory_order_relaxed);
        }
        n[W1SumIndexHandles].fetch_add(p.index_handles.size(), std::memory_order_relaxed);
        n[W1SumOtherHandles].fetch_add(p.other_handles.size(), std::memory_order_relaxed);
        w1s_counts[76].fetch_add(p.shapes.size(), std::memory_order_relaxed);
        p.shapes.clear();
        p.index_handles.clear();
        p.other_handles.clear();
        p.pipelines.clear();
        p.geometry.clear();
        p.other.clear();
        p.ring.clear();
        p.state.clear();
        p.draws = p.pairs = p.same_pipeline = p.same_geometry = p.same_other = p.same_push = p.same_state = p.same_vertex = p.same_index = 0;
        p.have_prev = false;
    };
    if (t_cover_class != 0 || !c.valid) {
        flush_pass(); // the class changed: the pass ended
        p.depth = ~0ull;
        c.valid = false;
        c.bad = 0;
        return;
    }
    {
        // W1.1: this shadow draw's verdict and the size of the integer record it would need
        auto& s = w1s_counts;
        const u32 bad = c.bad;
        s[0].fetch_add(1, std::memory_order_relaxed);
        if (bad == 0) {
            s[1].fetch_add(1, std::memory_order_relaxed);
        }
        if ((bad & ~W1BadFlat) == 0) {
            s[2].fetch_add(1, std::memory_order_relaxed);
        }
        if ((bad & ~(W1BadFlat | W1BadStream | W1BadGeoStream)) == 0) {
            s[3].fetch_add(1, std::memory_order_relaxed);
        }
        for (u32 bit = 0; bit < 10; ++bit) {
            if ((bad >> bit & 1u) != 0) {
                s[4 + bit].fetch_add(1, std::memory_order_relaxed);
            }
            if (bad == (1u << bit)) {
                s[14 + bit].fetch_add(1, std::memory_order_relaxed);
            }
        }
        if ((bad & W1BadImageWork) != 0) {
            s[69].fetch_add(1, std::memory_order_relaxed);
        }
        if (bad == W1BadImageWork || bad == (W1BadImageWork | W1BadImage)) {
            s[70].fetch_add(1, std::memory_order_relaxed);
        }
        if ((bad & ~W1BadImage) == 0) {
            s[71].fetch_add(1, std::memory_order_relaxed); // images allowed, no image work: pure lookups
        }
        if ((bad & ~(W1BadImage | W1BadImageWork)) == 0) {
            s[72].fetch_add(1, std::memory_order_relaxed); // images allowed, image work allowed too
        }
        if (t_w1_bind_begin != 0 && t_w1_draw_begin != 0) {
            // where the time of a shadow draw goes: before BindResources, in it, after it (to the end of Draw())
            const u64 end = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
            const u32 base = (bad & ~(W1BadFlat | W1BadStream | W1BadGeoStream | W1BadImage | W1BadImageWork)) == 0 ? 61u : 65u;
            s[base + 0].fetch_add(t_w1_bind_begin - t_w1_draw_begin, std::memory_order_relaxed);
            s[base + 1].fetch_add(t_w1_bind_end - t_w1_bind_begin, std::memory_order_relaxed);
            s[base + 2].fetch_add(end - t_w1_bind_end, std::memory_order_relaxed);
            s[base + 3].fetch_add(1, std::memory_order_relaxed);
        }
        // record: pipeline id, index (buffer id, offset, first index, count), push constants (4), per ring window an offset, per other buffer an id and an offset
        const u64 ints = 1 + 4 + 4 + c.ring.size() + 2 * c.other.size();
        s[24].fetch_add(ints, std::memory_order_relaxed);
        u64 old = s[25].load(std::memory_order_relaxed);
        while (ints > old && !s[25].compare_exchange_weak(old, ints, std::memory_order_relaxed)) {
        }
        c.bad = 0;
    }
    if (p.depth != c.depth) {
        flush_pass();
        p.depth = c.depth;
    }
    const u64 geometry = W1Mix(c.vertex, c.index);
    u64 other_set = 0xcbf29ce484222325ull;
    for (const u64 v : c.other) {
        other_set = W1Mix(other_set, v);
    }
    ++p.draws;
    {
        // W1.2a: could a replay of an earlier draw's bind of this pass serve this draw? shape = all but the ring offsets and the index range, plus the index buffer and the vertex buffers
        const u64 shape = W1Mix(W1Mix(c.shape, c.index_handle), c.vertex);
        auto& s = w1s_counts;
        if (p.have_prev && p.prev_shape == shape) {
            s[74].fetch_add(1, std::memory_order_relaxed);
        } else if (p.shapes.count(shape) != 0) {
            s[75].fetch_add(1, std::memory_order_relaxed);
        } else {
            s[73].fetch_add(1, std::memory_order_relaxed);
        }
        p.shapes.insert(shape);
        p.prev_shape = shape;
    }
    p.pipelines.insert(c.pipeline);
    p.geometry.insert(geometry);
    p.other.insert(other_set);
    p.index_handles.insert(c.index_handle);
    for (const u64 v : c.other_handles) {
        p.other_handles.insert(v);
    }
    for (const u64 v : c.ring) {
        p.ring.insert(v);
    }
    p.state.insert(c.state);
    n[W1RingPerDraw].fetch_add(c.ring.size(), std::memory_order_relaxed);
    n[W1OtherPerDraw].fetch_add(c.other.size(), std::memory_order_relaxed);
    n[W1ImagesPerDraw].fetch_add(c.images, std::memory_order_relaxed);
    n[W1Indices].fetch_add(c.index_count, std::memory_order_relaxed);
    n[W1Instances].fetch_add(c.instances, std::memory_order_relaxed);
    if (p.have_prev) {
        ++p.pairs;
        p.same_pipeline += p.prev_pipeline == c.pipeline;
        p.same_geometry += p.prev_geometry == geometry;
        p.same_other += p.prev_other == other_set;
        p.same_push += p.prev_push == c.push;
        p.same_state += p.prev_state == c.state;
        p.same_vertex += p.prev_vertex == c.vertex;
        p.same_index += p.prev_index == c.index;
    }
    p.have_prev = true;
    p.prev_pipeline = c.pipeline;
    p.prev_geometry = geometry;
    p.prev_other = other_set;
    p.prev_push = c.push;
    p.prev_state = c.state;
    p.prev_vertex = c.vertex;
    p.prev_index = c.index;
    c.valid = false;
}

void Rasterizer::NoteCover(u32 index) noexcept {
    switch (index & 31) {
    case CBufferRing:
        t_cover_kind = 1;
        ++t_w3.kinds[1];
        break;
    case CBufferSpecial:
        t_cover_kind = 2;
        ++t_w3.kinds[2];
        break;
    case CBufferNull:
        t_cover_kind = 3;
        ++t_w3.kinds[3];
        break;
    case CBufferResolved:
        t_cover_kind = 7;
        ++t_w3.kinds[7];
        break;
    case CBindResources:
        t_cover_mask = 0;
        t_w3 = W3Cur{};
        ::VideoCore::t_cover_sync_ns = 0;
        t_cover_begin = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        break;
    case CBufferHandle:
        t_cover_mask |= 1;
        t_cover_kind = 4;
        ++t_w3.kinds[4];
        break;
    case CBufferAddrRead:
        t_cover_mask |= 2;
        t_cover_kind = 5;
        ++t_w3.kinds[5];
        break;
    case CBufferAddrWritten:
        t_cover_mask |= 2;
        t_cover_kind = 6;
        ++t_w3.kinds[6];
        break;
    case CImageLookupHit:
        t_cover_mask |= 4;
        break;
    case CImageFind:
        t_cover_mask |= 8;
        t_w1.bad |= W1BadImageWork;
        break;
    case CImageStorage:
        t_cover_mask |= 16;
        t_w1.bad |= W1BadImageWork;
        break;
    case CImageTransitTrue:
        t_cover_mask |= 32;
        t_w1.bad |= W1BadImageWork;
        break;
    case CGeometryVertexAddress:
    case CGeometryIndexAddress:
        t_cover_mask |= 64;
        break;
    default:
        break;
    }
}

void Rasterizer::CoverFinish(bool compute) noexcept {
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    const u32 slot = (compute ? 128u : 0u) + (t_cover_mask & 127u);
    cover_count[slot].fetch_add(1, std::memory_order_relaxed);
    cover_ns[slot].fetch_add(t_cover_begin != 0 ? now - t_cover_begin : 0, std::memory_order_relaxed);
    cover_sync_ns[slot].fetch_add(::VideoCore::t_cover_sync_ns, std::memory_order_relaxed);
    if (!compute) {
        const u32 c = draw_class_now & 15u;
        t_cover_class = c;
        cover_class_draws[c].fetch_add(1, std::memory_order_relaxed);
        cover_class_bind_ns[c].fetch_add(t_cover_begin != 0 ? now - t_cover_begin : 0, std::memory_order_relaxed);
    }
    if (!compute) {
        t_w1_bind_begin = t_cover_begin;
        t_w1_bind_end = now;
    } else {
        t_w3.bind_ns = t_cover_begin != 0 ? now - t_cover_begin : 0;
        t_w3.sync_ns = ::VideoCore::t_cover_sync_ns;
        t_w3.images = static_cast<u32>(image_infos.size());
        t_w1_bind_begin = t_cover_begin;
        t_w1_bind_end = now;
    }
    ::VideoCore::t_cover_sync_ns = 0;
    t_cover_begin = 0;
}

void Rasterizer::FinishBindResources(const Pipeline* pipeline, bool uses_dma, u64 resolve_ns) {
    if (cover_on) [[unlikely]] {
        CoverFinish(pipeline->IsCompute());
        static const bool w1_on = [] { const char* e = std::getenv("TLG_W1_CENSUS"); return e != nullptr && e[0] == '1'; }();
        if (w1_on && !pipeline->IsCompute()) {
            W1NoteBinding(pipeline);
        }
    }
    BindCensus(pipeline);
    if (ReuseCensusOn()) {
        // reuse census (counts only): the shape of this draw, with the ring normalized
        static thread_local std::unordered_set<u64> seen_previous, seen_this;
        static thread_local u64 frame = ~u64{0};
        const u64 now = Core::DebugServer::PresentCount();
        if (now != frame) {
            seen_previous.swap(seen_this);
            seen_this.clear();
            frame = now;
        }
        u64 h = 0xcbf29ce484222325ull;
        const auto mix = [&h](u64 v) { h = (h ^ v) * 0x100000001b3ull; h ^= h >> 29; };
        mix(reinterpret_cast<u64>(pipeline));
        u32 rings = 0, others = 0;
        const VkBuffer ring_handle = constant_ring ? static_cast<VkBuffer>(constant_ring->Handle()) : VK_NULL_HANDLE;
        for (const auto& b : buffer_infos) {
            if (static_cast<VkBuffer>(b.buffer) == ring_handle) {
                mix(0x52494e47ull);
                mix(b.range);
                ++rings;
            } else {
                mix(reinterpret_cast<u64>(static_cast<VkBuffer>(b.buffer)));
                mix(b.offset);
                mix(b.range);
                ++others;
            }
        }
        for (const auto& i : image_infos) {
            mix(reinterpret_cast<u64>(static_cast<VkImageView>(i.imageView)));
            mix(reinterpret_cast<u64>(static_cast<VkSampler>(i.sampler)));
            mix(static_cast<u64>(i.imageLayout));
        }
        mix(std::bit_cast<u32>(push_data.xoffset));
        mix(std::bit_cast<u32>(push_data.yoffset));
        mix(std::bit_cast<u32>(push_data.xscale));
        mix(std::bit_cast<u32>(push_data.yscale));
        const u32 k = pipeline->IsCompute() ? 1u : 0u;
        auto& c = g_repeat_census;
        c.total[k].fetch_add(1, std::memory_order_relaxed);
        c.ring_bindings[k].fetch_add(rings, std::memory_order_relaxed);
        c.other_buffers[k].fetch_add(others, std::memory_order_relaxed);
        c.images[k].fetch_add(image_infos.size(), std::memory_order_relaxed);
        bool repeated = true;
        if (seen_previous.count(h) != 0) {
            c.in_previous_frame[k].fetch_add(1, std::memory_order_relaxed);
        } else if (seen_this.count(h) != 0) {
            c.in_this_frame[k].fetch_add(1, std::memory_order_relaxed);
        } else {
            c.new_shape[k].fetch_add(1, std::memory_order_relaxed);
            repeated = false;
        }
        if (repeated) {
            c.repeat_ring_bindings[k].fetch_add(rings, std::memory_order_relaxed);
            c.repeat_other_buffers[k].fetch_add(others, std::memory_order_relaxed);
            c.repeat_images[k].fetch_add(image_infos.size(), std::memory_order_relaxed);
        }
        seen_this.insert(h);
        if (recipe_active) {
            bool uses_stream = false;
            for (const auto& b : buffer_infos) {
                uses_stream |= buffer_cache.IsStreamHandle(b.buffer);
            }
            const bool same = recipe_prediction != 0 && recipe_stored_shape == h;
            if (recipe_prediction == 0) {
                buffer_cache.RecipeCount(2);
            } else if (recipe_prediction == 1) {
                buffer_cache.RecipeCount(same ? 3 : 4);
            } else {
                buffer_cache.RecipeCount(same ? 5 : 6);
            }
            if (!recipe_reusable) {
                buffer_cache.RecipeCount(7);
            }
            u64 h2 = 0xcbf29ce484222325ull, kinds = 0;
            if (VersionCensusOn()) {
                // the same shape with the stream copies' offsets left out (a copy of an object's bytes sits at a different place in the stream buffer after every content change)
                const auto mix2 = [&h2](u64 v) { h2 = (h2 ^ v) * 0x100000001b3ull; h2 ^= h2 >> 29; };
                mix2(reinterpret_cast<u64>(pipeline));
                u32 kind_index = 0;
                for (const auto& b : buffer_infos) {
                    const VkBuffer handle = static_cast<VkBuffer>(b.buffer);
                    const u64 kind = handle == ring_handle ? 1 : buffer_cache.IsStreamHandle(b.buffer) ? 2 : 3;
                    kinds |= kind << (2 * (kind_index++ & 31));
                    if (handle == ring_handle) {
                        mix2(0x52494e47ull);
                        mix2(b.range);
                    } else if (buffer_cache.IsStreamHandle(b.buffer)) {
                        mix2(0x53545245ull);
                        mix2(b.range);
                    } else {
                        mix2(reinterpret_cast<u64>(handle));
                        mix2(b.offset);
                        mix2(b.range);
                    }
                }
                for (const auto& i : image_infos) {
                    mix2(reinterpret_cast<u64>(static_cast<VkImageView>(i.imageView)));
                    mix2(reinterpret_cast<u64>(static_cast<VkSampler>(i.sampler)));
                    mix2(static_cast<u64>(i.imageLayout));
                }
                mix2(std::bit_cast<u32>(push_data.xoffset));
                mix2(std::bit_cast<u32>(push_data.yoffset));
                mix2(std::bit_cast<u32>(push_data.xscale));
                mix2(std::bit_cast<u32>(push_data.yscale));
                const auto& vp = version_prediction;
                using BC = VideoCore::BufferCache;
                if (vp.state == 0) {
                    buffer_cache.ScopedCount(BC::SvNsNoRecipe, resolve_ns);
                } else if (vp.state == 1) {
                    if (h == vp.shape) {
                        buffer_cache.ScopedCount(BC::SvValidSame);
                    } else if (h2 == vp.shape_nostream) {
                        buffer_cache.ScopedCount(BC::SvValidStreamOffsetOnly);
                    } else {
                        buffer_cache.ScopedCount(BC::SvValidOtherDifference);
                        buffer_cache.ScopedCount(kinds != vp.kind_signature ? BC::SvValidPathSwitch : BC::SvValidDifferentSameKinds);
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1) < 8) {
                            LOG_ERROR(Render, "version census: a recipe was valid by identity but the shape differs (key {:#x}, images {}, buffers {}, uses stream {})", recipe_key, image_infos.size(),
                                      buffer_infos.size(), vp.uses_stream);
                        }
                    }
                    buffer_cache.ScopedCount(vp.content_clean ? BC::SvValidContentClean : BC::SvValidContentDirty);
                    if (vp.uses_stream) {
                        buffer_cache.ScopedCount(BC::SvValidUsesStream);
                        if (vp.other_command_buffer) {
                            buffer_cache.ScopedCount(BC::SvValidStreamOtherCommandBuffer);
                        }
                    }
                    buffer_cache.ScopedCount(BC::SvNsValid, resolve_ns);
                } else {
                    buffer_cache.ScopedCount(h == vp.shape ? BC::SvInvalidButSame : BC::SvInvalidAndDifferent);
                    buffer_cache.ScopedCount(BC::SvNsInvalid, resolve_ns);
                }
                buffer_cache.ScopedCount(BC::SvNsAll, resolve_ns);
            }
            buffer_cache.RecipeStore(recipe_key, h, recipe_ranges, uses_stream, recipe_reusable, scheduler.CurrentTick(), recipe_epochs_at_start);
            if (VersionCensusOn()) {
                buffer_cache.RecipeStoreScoped(recipe_key, h2, kinds, recipe_image_ranges, recipe_identity_clock);
            }
            recipe_active = false;
        }
        // key census: the record's identity against the shape
        {
            auto& kc = g_key_census;
            kc.total[k].fetch_add(1, std::memory_order_relaxed);
            const DrawPlan* record_plan = binder_plan != nullptr ? binder_plan : nullptr;
            if (record_plan == nullptr || !record_plan->bindings_complete) {
                kc.no_record[k].fetch_add(1, std::memory_order_relaxed);
            } else {
                kc.keyed[k].fetch_add(1, std::memory_order_relaxed);
                struct Seen {
                    u64 shape, buffers_hash, images_hash;
                };
                static thread_local std::unordered_map<u64, Seen> by_key;
                if (by_key.size() > 400000) {
                    by_key.clear();
                }
                u64 key = reinterpret_cast<u64>(pipeline) * 0x9E3779B97F4A7C15ull;
                for (u32 bi = 0; bi < record_plan->binding_count; ++bi) {
                    const auto& rb = record_plan->bindings[bi];
                    key = (key ^ (u64{rb.stage} | u64{rb.kind} << 8 | u64{rb.offset} << 16 | u64{rb.slot} << 32)) * 0x100000001b3ull;
                    key = (key ^ rb.id) * 0x100000001b3ull;
                    key = (key ^ rb.view_id) * 0x100000001b3ull;
                    key ^= key >> 31;
                }
                u64 hb = 0xcbf29ce484222325ull, hi = 0xcbf29ce484222325ull;
                for (const auto& b : buffer_infos) {
                    if (static_cast<VkBuffer>(b.buffer) == ring_handle) {
                        hb = (hb ^ (0x52494e47ull + b.range)) * 0x100000001b3ull;
                    } else {
                        hb = (hb ^ reinterpret_cast<u64>(static_cast<VkBuffer>(b.buffer))) * 0x100000001b3ull;
                        hb = (hb ^ b.offset) * 0x100000001b3ull;
                        hb = (hb ^ b.range) * 0x100000001b3ull;
                    }
                }
                for (const auto& i : image_infos) {
                    hi = (hi ^ reinterpret_cast<u64>(static_cast<VkImageView>(i.imageView))) * 0x100000001b3ull;
                    hi = (hi ^ reinterpret_cast<u64>(static_cast<VkSampler>(i.sampler))) * 0x100000001b3ull;
                    hi = (hi ^ static_cast<u64>(i.imageLayout)) * 0x100000001b3ull;
                }
                const auto it = by_key.find(key);
                if (it == by_key.end()) {
                    by_key.emplace(key, Seen{h, hb, hi});
                } else {
                    kc.key_seen[k].fetch_add(1, std::memory_order_relaxed);
                    if (it->second.shape == h) {
                        kc.same_shape[k].fetch_add(1, std::memory_order_relaxed);
                    } else {
                        kc.other_shape[k].fetch_add(1, std::memory_order_relaxed);
                        const bool b_differs = it->second.buffers_hash != hb, i_differs = it->second.images_hash != hi;
                        (b_differs && i_differs ? kc.differ_both : b_differs ? kc.differ_buffers_only : kc.differ_images_only)[k].fetch_add(1, std::memory_order_relaxed);
                    }
                    it->second = Seen{h, hb, hi};
                }
            }
        }
    }
    if (uses_dma) {
        b9_dma_binds.fetch_add(1, std::memory_order_relaxed); // E4: a shader that reads memory by a raw PS4 address (TLG_DIRECT_MEMORY_ACCESS, off): there is no address mirror to serve it
    }
    b9_all_binds.fetch_add(1, std::memory_order_relaxed);

    if (record_indexed_active) {
        // tlg C9: the plain bindings are the ones that got a write; the guest buffers are named by the draw's row, copied into the stream buffer the global set's table binding covers
        set_writes.resize(set_write_index);
        auto* indexed = IndexedData::Get();
        indexed->stats.legacy_writes.fetch_add(set_write_index, std::memory_order_relaxed);
        if (pipeline->IsCompute()) {
            indexed->stats.compute_dispatches.fetch_add(1, std::memory_order_relaxed);
        }
        if (indexed_row_size != 0) {
            auto& table = buffer_cache.GetStreamBuffer();
            indexed->SetTable(table.Handle(), table.SizeBytes());
            const u64 offset = table.Copy(indexed_row.data(), indexed_row_size * sizeof(u32), sizeof(u32));
            push_data.row_base = static_cast<u32>(offset / sizeof(u32));
            indexed->stats.rows_entries.fetch_add(indexed_row_size, std::memory_order_relaxed);
        }
        indexed->stats.draws.fetch_add(1, std::memory_order_relaxed);
        // P4: the descriptor writes of new slots and the release of finished ones run once per submission (Scheduler::SubmitExecution), not per draw:
        // the set is update-after-bind, a slot only has to be written before the command buffer that names it is submitted
        if (!IndexedData::FlushPerSubmit()) {
            indexed->FlushWrites();
            indexed->Maintain();
        }
    }

}

bool Rasterizer::BindResources(const Pipeline* pipeline, const FixedState* fixed, DrawJob* draw_job) {
    TLG_PROF_LEVEL("raster.bind_resources", 2);
    const bool w3_time = cover_on && pipeline->IsCompute();
    const auto w3_now = [] { return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); };
    u64 w3_p[6]{};
    u64 w3_t = w3_time ? w3_now() : 0;
    const auto w3_lap = [&](u32 k) {
        if (w3_time) {
            const u64 n = w3_now();
            w3_p[k] += n - w3_t;
            w3_t = n;
        }
    };
    VideoCore::OpSequence::resolving.fetch_add(1, std::memory_order_acq_rel); // P3c: a new operation (a draw or a dispatch)
    if (draw_job == nullptr || !VideoCore::RecordQueue::Enabled()) {
        VideoCore::ResolveCommands::Run();
    }
    w3_lap(0);
    {
        ComputeImageOp image_op;
        if (MatchComputeImageCopy(pipeline, image_op) || IsComputeMetaClear(pipeline) || MatchComputeImageClear(pipeline, image_op)) {
            if (image_op.kind != ComputeImageOp::None) {
                static std::atomic<u64> image_ops{0};
                const u64 n = image_ops.fetch_add(1, std::memory_order_relaxed) + 1;
                if (SplitBindMode() == 1 && (n & (n - 1)) == 0) {
                    LOG_WARNING(Render_Vulkan, "split bind: {} compute image copies/clears (record work decided in the resolve half, run in the record half)", n);
                }
                RunComputeImageOp(image_op);
            }
            return false;
        }
    }
    w3_lap(1);
    NoteRecordedStateDiagnostics();

    indexed_active = pipeline->IsIndexed();
    if (draw_job == nullptr) {
        // (a draw job's record half starts these empty itself: they belong to the record side)
        set_write_index = 0;
        set_writes.clear();
        buffer_infos.clear();
        image_infos.clear();
        record_indexed_active = indexed_active;
        indexed_row_size = 0;
    }
    Census((fixed != nullptr ? 0u : 32u) + CBindResources);

    // draw recipe verify: the identity of the record and the prediction, before anything is resolved
    recipe_active = false;
    recipe_ranges.clear();
    recipe_image_ranges.clear();
    recipe_reusable = true;
    version_prediction = {};
    if (ReuseCensusOn() && !pipeline->IsCompute() && binder_plan != nullptr) {
        buffer_cache.RecipeCount(0);
        if (!binder_plan->bindings_complete) {
            buffer_cache.RecipeCount(1);
        } else {
            u64 key = reinterpret_cast<u64>(pipeline) * 0x9E3779B97F4A7C15ull;
            for (u32 bi = 0; bi < binder_plan->binding_count; ++bi) {
                const auto& rb = binder_plan->bindings[bi];
                key = (key ^ (u64{rb.stage} | u64{rb.kind} << 8 | u64{rb.offset} << 16 | u64{rb.slot} << 32)) * 0x100000001b3ull;
                key = (key ^ rb.id) * 0x100000001b3ull;
                key = (key ^ rb.view_id) * 0x100000001b3ull;
                key ^= key >> 31;
            }
            recipe_key = key;
            recipe_epochs_at_start = buffer_cache.RecipeEpochsNow(Core::NativeResources::ImmutableEpoch(), texture_cache.ImageGeneration());
            recipe_prediction = buffer_cache.RecipePredict(key, scheduler.CurrentTick(), recipe_epochs_at_start, recipe_stored_shape);
            recipe_active = true;
            if (VersionCensusOn()) {
                recipe_identity_clock = Core::NativeIdentity::Clock();
                version_prediction = buffer_cache.RecipePredictScoped(key, scheduler.CurrentTick());
            }
        }
    }

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    const Shader::PushData first_push = MakeUserData(fixed);
    if (draw_job == nullptr) {
        push_data = first_push;
    }
    clip_plane_mask = fixed ? fixed->user_clip_plane_mask : 0;
    const bool timed_resolve = recipe_active && VersionCensusOn();
    const auto resolve_begin = timed_resolve ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (SplitBindMode() == 1) {
        // Plan S step P3e.1: the whole draw is resolved first (every stage: buffers, images, samplers), then recorded. What the resolve loop leaves is plain data per stage
        // (`StageResolved`): the unit a resolver thread would hand to the recorder. With a `DrawJob` (P3e.4) the records are left to `RecordDraw`.
        boost::container::small_vector<StageResolved, 6> local_jobs;
        auto& jobs = draw_job != nullptr ? draw_job->stages : local_jobs;
        {
            Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
            TLG_PROF_LEVEL("raster.resolve_stages", 2);
            for (const auto* stage : pipeline->GetStages()) {
                if (!stage) {
                    continue;
                }
                StageResolved& job = jobs.emplace_back();
                job.stage = stage;
                job.buffers.resize(stage->buffers.size());
                const u64 alignment = instance.StorageMinAlignment();
                const u32 census_base = stage->sw_stage == Shader::SwStage::Compute ? 32u : 0u;
                for (u32 i = 0; i < job.buffers.size(); ++i) {
                    ResolveBufferBinding(*stage, i, alignment, census_base, job.buffers[i]);
                }
                ResolveTextures(*stage, job.textures);
                ResolveSamplers(*stage, job.samplers);
                uses_dma |= stage->uses_dma;
            }
            if (draw_job != nullptr) {
                ResolveDrawJob(static_cast<const GraphicsPipeline*>(pipeline), *draw_job);
            }
        }
        if (draw_job != nullptr) {
            draw_job->push_data = first_push;
            draw_job->indexed_active = indexed_active;
            draw_job->uses_dma = uses_dma;
            draw_job->feedback_loop = draw_job->render.feedback_loop;
            draw_job->resolve_ns = timed_resolve ? static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - resolve_begin).count()) : 0;
            draw_job->op_seq = VideoCore::OpSequence::resolving.load(std::memory_order_acquire);
            return true;
        }
        {
            Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
            TLG_PROF_LEVEL("raster.record_stages", 2);
            for (StageResolved& job : jobs) {
                const Shader::Info& stage = *job.stage;
                set_writes.resize(set_writes.size() + stage.buffers.size() + stage.images.size() + stage.samplers.size());
                const u64 alignment = instance.StorageMinAlignment();
                const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
                RecordBufferBindings(stage, binding, push_data, job.buffers.data(), static_cast<u32>(job.buffers.size()), alignment, census_base);
                RecordTextures(stage, binding, job.textures);
                RecordSamplers(stage, binding, job.samplers.data(), static_cast<u32>(job.samplers.size()));
            }
        }
    } else
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        w3_lap(2);
        BindBuffers(*stage, binding, push_data);
        w3_lap(3);
        const u32 images_before = static_cast<u32>(image_infos.size());
        BindTextures(*stage, binding);
        w3_lap(4);
        uses_dma |= stage->uses_dma;
    }
    const u64 resolve_ns = timed_resolve ? static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - resolve_begin).count()) : 0;

    if (w3_time) {
        for (u32 k = 0; k < 5; ++k) {
            t_w3.phase[k] += w3_p[k];
        }
        w3_t = w3_now();
    }
    FinishBindResources(pipeline, uses_dma, resolve_ns);
    if (w3_time) {
        t_w3.phase[5] += w3_now() - w3_t;
    }
    return true;
}

void Rasterizer::BindVertexBuffersLegacy(const GraphicsPipeline* pipeline) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_vertex_buffers", 2);
    const auto& regs = *draw_inputs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<Core::NativeDesc::BufferDesc> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        verify_vertex.clear();
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;
        bool ring{}; ///< N5g: the bytes are in the constant ring (offset is a ring offset, buffer is null)

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.address != 0 && buffer.size > 0) {
            ranges.emplace_back(buffer.address, buffer.address + buffer.size);
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
        if (Core::OwnMemoryCensus::On()) [[unlikely]] {
            Core::OwnMemoryCensus::NoteGeometryRead(range.base_address, range.GetSize()); // design review 79, step 1a
        }
        if (range.GetSize() >= (1ull << 30) && !memory->IsValidMapping(range.base_address)) {
            LOG_ERROR(Render, "vertex buffer range at unmapped address {:#x} of {} bytes: ignored", range.base_address, range.GetSize());
            range.buffer = nullptr;
            continue;
        }
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        if (const RingBinding* ring = FindGeometryRing(range.base_address, size, GeometryVertex)) {
            if (buffer_cache.IsRangeGpuWritten(range.base_address, size)) {
                Core::NativeUploads::NoteGeometry(4); // the GPU wrote bytes of it after the command processor copied it: the arena has them
            } else {
                range.ring = true;
                range.offset = ring->offset;
                Census(CGeometryVertexRing);
                Core::NativeUploads::NoteGeometry(2);
                continue;
            }
        }
        auto* vertex_handle = GeometryHandle(range.base_address, size);
        Census(vertex_handle != nullptr ? CGeometryVertexHandle : CGeometryVertexAddress);
        Core::NativeContent::SetBindContext({"vertex", 0, 0, 0, 0, vertex_handle != nullptr, false, range.base_address, size});
        if (cover_on) [[unlikely]] {
            VideoCore::BufferCache::UploadBytesReset();
        }
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false, false, vertex_handle, NativeHandleMode() == 2);
        Core::NativeContent::ClearBindContext();
        if (cover_on && W1ShadowOn()) [[unlikely]] {
            W1ShadowGeometry(vertex_handle, range.base_address, range.buffer, range.offset, true);
        }
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
        if (buffer.address != 0 && buffer.size > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.address >= range.base_address &&
                           buffer.address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->ring ? constant_ring->Handle() : host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.size);
        host_strides.push_back(buffer.stride);
    }

    if (verify_slots_on) {
        verify_vertex.clear();
        for (size_t i = 0; i < host_buffers.size(); ++i) {
            verify_vertex.push_back({host_buffers[i], host_offsets[i], buffer_cache.IsStreamHandle(host_buffers[i])});
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (cover_on) [[unlikely]] {
        u64 hv = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < host_buffers.size(); ++i) {
            hv = W1Mix(W1Mix(hv, reinterpret_cast<u64>(static_cast<VkBuffer>(host_buffers[i]))), host_offsets[i]);
        }
        t_w1.vertex = hv;
    }
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::ResolveVertexBuffers(const GraphicsPipeline* pipeline, VertexBind& vb) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2: the resolve half of the vertex buffer binding (what is bound where); no command is recorded here
    const auto& regs = *draw_inputs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes = vb.attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings = vb.bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<Core::NativeDesc::BufferDesc> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers, regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    if (bindings.empty()) {
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;
        bool ring{};

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.address != 0 && buffer.size > 0) {
            ranges.emplace_back(buffer.address, buffer.address + buffer.size);
        }
    }
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) { return lhv.base_address < rhv.base_address; });
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
    for (auto& range : ranges_merged) {
        if (range.GetSize() >= (1ull << 30) && !memory->IsValidMapping(range.base_address)) {
            LOG_ERROR(Render, "vertex buffer range at unmapped address {:#x} of {} bytes: ignored", range.base_address, range.GetSize());
            range.buffer = nullptr;
            continue;
        }
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        if (const RingBinding* ring = FindGeometryRing(range.base_address, size, GeometryVertex)) {
            if (buffer_cache.IsRangeGpuWritten(range.base_address, size)) {
                Core::NativeUploads::NoteGeometry(4);
            } else {
                range.ring = true;
                range.offset = ring->offset;
                Census(CGeometryVertexRing);
                Core::NativeUploads::NoteGeometry(2);
                continue;
            }
        }
        auto* vertex_handle = GeometryHandle(range.base_address, size);
        Census(vertex_handle != nullptr ? CGeometryVertexHandle : CGeometryVertexAddress);
        Core::NativeContent::SetBindContext({"vertex", 0, 0, 0, 0, vertex_handle != nullptr, false, range.base_address, size});
        std::tie(range.buffer, range.offset) = buffer_cache.ObtainBuffer(range.base_address, size, false, false, vertex_handle, NativeHandleMode() == 2);
        Core::NativeContent::ClearBindContext();
        vb.queries.push_back({range.buffer, range.offset, size});
    }
    for (const auto& buffer : guest_buffers) {
        if (buffer.address != 0 && buffer.size > 0) {
            const auto host_buffer_info = std::ranges::find_if(ranges_merged, [&](const BufferRange& range) { return buffer.address >= range.base_address && buffer.address < range.end_address; });
            ASSERT(host_buffer_info != ranges_merged.cend());
            vb.host_buffers.emplace_back(host_buffer_info->ring ? constant_ring->Handle() : host_buffer_info->buffer->Handle());
            vb.host_offsets.push_back(host_buffer_info->offset + buffer.address - host_buffer_info->base_address);
        } else {
            vb.host_buffers.emplace_back(VK_NULL_HANDLE);
            vb.host_offsets.push_back(0);
        }
        vb.host_sizes.push_back(buffer.size);
        vb.host_strides.push_back(buffer.stride);
    }
}

void Rasterizer::RecordVertexBuffers(VertexBind& vb, VideoCore::BufferCache::UploadSink& uploads) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    // Plan S step P2: the record half. The order of the old function is kept: the vertex input state, the synchronization copies, the barrier queries, the bind.
    if (instance.IsVertexInputDynamicState()) {
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(vb.bindings, vb.attributes);
    }
    buffer_cache.RunDeferredUploads(uploads);
    if (vb.bindings.empty()) {
        return;
    }
    for (const auto& q : vb.queries) {
        needs_barrier |= runtime.IsBufferAccessed(q.buffer, q.offset, q.size);
        bound_buffers.emplace_back(q.buffer, q.offset, q.size, vk::AccessFlagBits2::eVertexAttributeRead);
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = vb.host_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        if (cover_on) [[unlikely]] {
            u64 hv = 0xcbf29ce484222325ull;
            for (size_t i = 0; i < vb.host_buffers.size(); ++i) {
                hv = W1Mix(W1Mix(hv, reinterpret_cast<u64>(static_cast<VkBuffer>(vb.host_buffers[i]))), vb.host_offsets[i]);
            }
            t_w1.vertex = hv;
        }
        cmdbuf.bindVertexBuffers(0, num_buffers, vb.host_buffers.data(), vb.host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, vb.host_buffers.data(), vb.host_offsets.data(), vb.host_sizes.data(), vb.host_strides.data());
    }
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const u32 mode = SplitBindMode();
    if (mode == 0) {
        BindVertexBuffersLegacy(pipeline);
        return;
    }
    static std::atomic<u64> checked{0}, differ{0};
    if (mode == 2) {
        verify_vertex.clear();
        verify_slots_on = true;
        BindVertexBuffersLegacy(pipeline);
        verify_slots_on = false;
        VertexBind vb;
        ResolveVertexBuffers(pipeline, vb);
        const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        bool same = vb.host_buffers.size() == verify_vertex.size();
        for (size_t i = 0; same && i < verify_vertex.size(); ++i) {
            same = vb.host_buffers[i] == verify_vertex[i].buffer && (verify_vertex[i].stream || vb.host_offsets[i] == verify_vertex[i].offset);
        }
        if (!same && differ.fetch_add(1) < 20) {
            LOG_WARNING(Render_Vulkan, "split bind verify: vertex buffers differ ({} legacy, {} split)", verify_vertex.size(), vb.host_buffers.size());
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind verify: {} vertex binds, {} differ", n, differ.load());
        }
        return;
    }
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_vertex_buffers", 2);
    VertexBind vb;
    VideoCore::BufferCache::UploadSink uploads;
    VideoCore::BufferCache::SetUploadSink(&uploads);
    ResolveVertexBuffers(pipeline, vb);
    VideoCore::BufferCache::SetUploadSink(nullptr);
    RecordVertexBuffers(vb, uploads);
}

void Rasterizer::BindIndexBufferLegacy(u32 index_offset, bool is_indirect) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_index_buffer", 2);
    const auto& regs = *draw_inputs;
    namespace DS = Core::DispatchSplit;
    const bool ic = DS::t_draw_active; // design review 76: the sampled draw's index bind, in six steps
    u64 it[6] = {ic ? DS::Tsc() : 0, 0, 0, 0, 0, 0};
    u32 ipath = 3;
    bool istream = false;
    const auto ilap = [&](u32 k) {
        if (ic) {
            it[k] = DS::Tsc();
        }
    };

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
    ilap(1);
    const RingBinding* index_ring = index_buffer_size != 0 && !is_indirect ? FindGeometryRing(index_address, index_buffer_size, GeometryIndex) : nullptr;
    if (index_buffer_size != 0 && !is_indirect && ring_use.draw != nullptr && Core::NativeUploads::RingGeometry()) {
        bool has_entry = false;
        for (const auto& ring : ring_use.draw->geometry_ring) {
            has_entry |= ring.index == GeometryIndex;
        }
        Core::NativeUploads::NoteGeometry(index_ring != nullptr ? 14 : has_entry ? 15 : 29);
    }
    if (index_ring != nullptr && buffer_cache.IsRangeGpuWritten(index_address, index_buffer_size)) {
        Core::NativeUploads::NoteGeometry(4);
        index_ring = nullptr;
    }
    ilap(2);
    if (Core::OwnMemoryCensus::On() && index_buffer_size != 0) [[unlikely]] {
        Core::OwnMemoryCensus::NoteGeometryRead(index_address, index_buffer_size); // design review 79, step 1a
    }
    if (index_ring != nullptr) {
        Census(CGeometryIndexRing);
        index_buffer = constant_ring->Handle();
        buffer_offset = index_ring->offset;
        Core::NativeUploads::NoteGeometry(3);
        ipath = 0;
        ilap(3);
        ilap(4);
        ilap(5);
    } else if (index_buffer_size != 0 && [&] {
                   // design review 76 (TLG_GEOMETRY_ROWS): the index buffer by row (engine id of the plan's geometry handle)
                   const u32 grm = Core::RowVersion::GeometryRowsMode();
                   if (grm == 0) {
                       return false;
                   }
                   Core::RowVersion::NoteGeometry(0);
                   const auto* gh = GeometryHandleOfPlan(index_address, index_buffer_size);
                   if (gh == nullptr) {
                       return false;
                   }
                   Core::RowVersion::NoteGeometry(1);
                   const VideoCore::Buffer* row_buffer = nullptr;
                   u64 row_offset = 0;
                   const bool row_valid = RowsServe(gh->id, index_address, index_buffer_size, false, row_buffer, row_offset, false);
                   static thread_local u32 geo_sample = 0;
                   if (row_valid) {
                       Core::RowVersion::NoteGeometry(2);
                       if (grm == 2 && (++geo_sample & 63u) != 0) {
                           Core::RowVersion::NoteGeometry(3);
                           const_cast<VideoCore::Buffer*>(row_buffer)->NoteResolveUse(); // P3c: asked for by the operation being resolved
                           needs_barrier |= BufferBarrierQuery(row_buffer, row_offset, index_buffer_size, false); // (the barrier guard: a read with no planned write needs none)
                           bound_buffers.emplace_back(row_buffer, row_offset, index_buffer_size, vk::AccessFlagBits2::eIndexRead);
                           index_buffer = row_buffer->Handle();
                           buffer_offset = row_offset;
                           ipath = 1;
                           ilap(3);
                           ilap(4);
                           ilap(5);
                           return true;
                       }
                   }
                   // the old path answers (shadow, a sampled real row, or no valid row) and is compared or learned
                   const u64 epoch_pre = buffer_cache.PlacementEpoch();
                   const u64 clock_pre = Core::NativeIdentity::Clock();
                   const u64 version_pre = Core::RowVersion::Mode() != 0 ? Core::RowVersion::Of(gh->id) : 0;
                   const u64 unattr_pre = Core::RowVersion::Mode() != 0 ? Core::RowVersion::Unattributed() : 0;
                   const bool gpu_pre = buffer_cache.IsRangeGpuWritten(index_address, index_buffer_size);
                   auto* index_handle = GeometryHandle(index_address, index_buffer_size);
                   ilap(3);
                   ipath = 1;
                   VideoCore::BufferCache::UploadBytesReset();
                   const auto* list = scheduler.CommandBuffer().List();
                   const u64 commands_before = list != nullptr ? list->Size() : 0;
                   Core::NativeContent::SetBindContext({"index", 0, 0, 0, 0, index_handle != nullptr, false, index_address, index_buffer_size});
                   const auto [buffer, offset] = buffer_cache.ObtainBuffer(index_address, index_buffer_size, false, false, index_handle, NativeHandleMode() == 2);
                   Core::NativeContent::ClearBindContext();
                   ilap(4);
                   if (row_valid) {
                       Core::RowVersion::NoteGeometry(4);
                       const auto* list_after = scheduler.CommandBuffer().List();
                       const u64 commands_after = list_after != nullptr ? list_after->Size() : 0;
                       if (buffer != row_buffer || offset != row_offset) {
                           Core::RowVersion::NoteGeometry(5);
                           static std::atomic<u32> logged{0};
                           if (logged.fetch_add(1) < 10) {
                               LOG_ERROR(Render_Vulkan, "geometry row differs: id {:#x} index {:#x}+{:#x}: row {:p}+{:#x}, old path {:p}+{:#x} (stream {}, hook ring {})", gh->id, index_address,
                                         index_buffer_size, static_cast<const void*>(row_buffer), row_offset, static_cast<const void*>(buffer), offset, buffer_cache.IsStreamBuffer(buffer),
                                         buffer_cache.IsHookRing(buffer));
                           }
                       } else if (VideoCore::BufferCache::UploadBytes() != 0 || commands_after != commands_before) {
                           Core::RowVersion::NoteGeometry(6);
                           static std::atomic<u32> logged_work{0};
                           if (logged_work.fetch_add(1) < 10) {
                               LOG_ERROR(Render_Vulkan, "geometry row valid but the old path did work: id {:#x} index {:#x}+{:#x}: uploaded {} bytes, {} commands", gh->id, index_address,
                                         index_buffer_size, VideoCore::BufferCache::UploadBytes(), commands_after - commands_before);
                           }
                       }
                   } else if (buffer_cache.IsStreamBuffer(buffer)) {
                       Core::RowVersion::NoteGeometry(8);
                   } else if (buffer_cache.IsHookRing(buffer)) {
                       Core::RowVersion::NoteGeometry(9);
                   } else {
                       Core::RowVersion::NoteGeometry(7);
                       RowsLearn(gh->id, index_address, index_buffer_size, false, buffer, offset, epoch_pre, clock_pre, gpu_pre, version_pre, unattr_pre);
                   }
                   istream = ic && buffer_cache.IsStreamBuffer(buffer);
                   needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size); // (the old path's query, as before)
                   ilap(5);
                   bound_buffers.emplace_back(buffer, offset, index_buffer_size, vk::AccessFlagBits2::eIndexRead);
                   index_buffer = buffer->Handle();
                   buffer_offset = offset;
                   return true;
               }()) {
        // (bound by the geometry row step above)
    } else if (index_buffer_size != 0) {
        auto* index_handle = GeometryHandle(index_address, index_buffer_size);
        ilap(3);
        ipath = index_handle != nullptr ? 1 : 2;
        Census(index_handle != nullptr ? CGeometryIndexHandle : CGeometryIndexAddress);
        Core::NativeContent::SetBindContext({"index", 0, 0, 0, 0, index_handle != nullptr, false, index_address, index_buffer_size});
        if (cover_on) [[unlikely]] {
            VideoCore::BufferCache::UploadBytesReset();
        }
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(index_address, index_buffer_size, false, false, index_handle, NativeHandleMode() == 2);
        Core::NativeContent::ClearBindContext();
        if (cover_on && W1ShadowOn()) [[unlikely]] {
            W1ShadowGeometry(index_handle, index_address, buffer, offset, false);
        }
        ilap(4);
        istream = ic && buffer_cache.IsStreamBuffer(buffer);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
        ilap(5);
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

    if (verify_slots_on) {
        verify_index = {index_buffer, buffer_offset, index_buffer_size, buffer_cache.IsStreamHandle(index_buffer)};
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsMaintenance5Supported()) {
        cmdbuf.bindIndexBuffer2KHR(index_buffer, buffer_offset, index_buffer_size, index_type);
    } else {
        cmdbuf.bindIndexBuffer(index_buffer, buffer_offset, index_type);
    }
    if (cover_on) [[unlikely]] {
        t_w1.index = W1Mix(reinterpret_cast<u64>(static_cast<VkBuffer>(index_buffer)), buffer_offset);
        t_w1.index_handle = reinterpret_cast<u64>(static_cast<VkBuffer>(index_buffer));
    }
    if (ic && it[5] != 0) {
        const u64 end = DS::Tsc();
        const u64 ticks[6] = {it[1] - it[0], it[2] - it[1], it[3] - it[2], it[4] - it[3], it[5] - it[4], end - it[5]};
        DS::NoteIndexBind(ticks, ipath, istream);
    }
}

void Rasterizer::ResolveIndexBuffer(u32 index_offset, bool is_indirect, IndexBind& ib) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2: the resolve half of the index buffer binding; no command is recorded here
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
    const RingBinding* index_ring = index_buffer_size != 0 && !is_indirect ? FindGeometryRing(index_address, index_buffer_size, GeometryIndex) : nullptr;
    if (index_buffer_size != 0 && !is_indirect && ring_use.draw != nullptr && Core::NativeUploads::RingGeometry()) {
        bool has_entry = false;
        for (const auto& ring : ring_use.draw->geometry_ring) {
            has_entry |= ring.index == GeometryIndex;
        }
        Core::NativeUploads::NoteGeometry(index_ring != nullptr ? 14 : has_entry ? 15 : 29);
    }
    if (index_ring != nullptr && buffer_cache.IsRangeGpuWritten(index_address, index_buffer_size)) {
        Core::NativeUploads::NoteGeometry(4);
        index_ring = nullptr;
    }
    if (index_ring != nullptr) {
        Census(CGeometryIndexRing);
        index_buffer = constant_ring->Handle();
        buffer_offset = index_ring->offset;
        Core::NativeUploads::NoteGeometry(3);
    } else if (index_buffer_size != 0) {
        auto* index_handle = GeometryHandle(index_address, index_buffer_size);
        Census(index_handle != nullptr ? CGeometryIndexHandle : CGeometryIndexAddress);
        Core::NativeContent::SetBindContext({"index", 0, 0, 0, 0, index_handle != nullptr, false, index_address, index_buffer_size});
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(index_address, index_buffer_size, false, false, index_handle, NativeHandleMode() == 2);
        Core::NativeContent::ClearBindContext();
        ib.query_buffer = buffer;
        ib.query_offset = offset;
        ib.query_size = index_buffer_size;
        index_buffer = buffer->Handle();
        buffer_offset = offset;
    } else if (!instance.IsMaintenance6Supported()) {
        const auto* null_buffer = buffer_cache.GetNullIndexBuffer();
        index_buffer_size = static_cast<u32>(null_buffer->SizeBytes());
        ib.query_buffer = null_buffer;
        ib.query_offset = 0;
        ib.query_size = index_buffer_size;
        index_buffer = null_buffer->Handle();
    }

    ib.buffer = index_buffer;
    ib.offset = buffer_offset;
    ib.size = index_buffer_size;
    ib.type = index_type;
}

void Rasterizer::RecordIndexBuffer(IndexBind& ib, VideoCore::BufferCache::UploadSink& uploads) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    buffer_cache.RunDeferredUploads(uploads);
    if (ib.query_buffer != nullptr) {
        needs_barrier |= runtime.IsBufferAccessed(ib.query_buffer, ib.query_offset, ib.query_size);
        bound_buffers.emplace_back(ib.query_buffer, ib.query_offset, ib.query_size, vk::AccessFlagBits2::eIndexRead);
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsMaintenance5Supported()) {
        cmdbuf.bindIndexBuffer2KHR(ib.buffer, ib.offset, ib.size, ib.type);
    } else {
        cmdbuf.bindIndexBuffer(ib.buffer, ib.offset, ib.type);
    }
    if (cover_on) [[unlikely]] {
        t_w1.index = W1Mix(reinterpret_cast<u64>(static_cast<VkBuffer>(ib.buffer)), ib.offset);
        t_w1.index_handle = reinterpret_cast<u64>(static_cast<VkBuffer>(ib.buffer));
    }
}

static std::mutex g_job_pool_mutex;

std::vector<std::unique_ptr<Rasterizer::DrawJob>>& Rasterizer::DrawJobPool() {
    static std::vector<std::unique_ptr<DrawJob>> pool; // (a job is taken by the resolver and given back by the recorder: one pool, one lock)
    return pool;
}

std::unique_ptr<Rasterizer::DrawJob> Rasterizer::TakeDrawJob() {
    auto& pool = DrawJobPool();
    std::unique_ptr<DrawJob> job;
    {
        std::scoped_lock lock{g_job_pool_mutex};
        if (!pool.empty()) {
            job = std::move(pool.back());
            pool.pop_back();
        }
    }
    if (!job) {
        return std::make_unique<DrawJob>();
    }
    DrawJob* raw = job.get();
    std::destroy_at(raw);
    std::construct_at(raw);
    return job;
}

void Rasterizer::RunDrawJob(std::unique_ptr<DrawJob>&& job) {
    auto& pool = DrawJobPool();
    if (!job->pre.empty()) {
        VideoCore::HalfPhase::Scope scope{VideoCore::HalfPhase::Record};
        for (auto& work : job->pre) {
            work();
        }
        job->pre.clear();
    }
    VideoCore::OpSequence::recording = job->op_seq;
    RecordDraw(*job);
    VideoCore::OpSequence::recording = 0;
    std::scoped_lock lock{g_job_pool_mutex};
    if (pool.size() < 256) {
        pool.push_back(std::move(job));
    }
}

void Rasterizer::RecordDraw(DrawJob& d) {
    BarrierCensus::SetOp(0x4000000000000000ull);
    TLG_PROF_LEVEL("raster.record_draw_all", 2);
    const GraphicsPipeline* pipeline = d.pipeline;
    const FixedState& fixed = d.fixed;
    // the per-draw state of the record side starts empty; what the resolve half decided comes from the job
    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    indexed_row_size = 0;
    record_indexed_active = d.indexed_active;
    push_data = d.push_data;
    attachment_feedback_loop = d.feedback_loop;
    {
        Shader::Backend::Bindings binding{};
        Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
        TLG_PROF_LEVEL("raster.record_stages", 2);
        for (StageResolved& job : d.stages) {
            const Shader::Info& stage = *job.stage;
            set_writes.resize(set_writes.size() + stage.buffers.size() + stage.images.size() + stage.samplers.size());
            const u64 alignment = instance.StorageMinAlignment();
            const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
            RecordBufferBindings(stage, binding, push_data, job.buffers.data(), static_cast<u32>(job.buffers.size()), alignment, census_base);
            RecordTextures(stage, binding, job.textures);
            RecordSamplers(stage, binding, job.samplers.data(), static_cast<u32>(job.samplers.size()));
        }
    }
    FinishBindResources(pipeline, d.uses_dma, d.resolve_ns);
    const RenderState state = RecordDrawJob(d);

    // the previous draw's notes: valid for the very next call of Draw, in the same command buffer
    const NativeLast last = native_last;
    const bool same_cmdbuf = d.use_tag && last.valid && last.seq + 1 == d.call_seq && last.tick == scheduler.CurrentTick();
    const bool same_pipeline = same_cmdbuf && last.pipeline == pipeline;
    const bool same_dynamic_state = same_pipeline && last.k2 == d.tag_k2;

    {
        const u32 diff = scheduler.PassStateDifference(state);
        if (diff != 0) {
            g_pass_changes.count[diff].fetch_add(1, std::memory_order_relaxed);
            g_pass_changes.draws_held[diff].fetch_add(0, std::memory_order_relaxed);
        }
    }
    if (needs_barrier) {
        TLG_PROF_LEVEL("raster.flush_barriers", 2);
        t_flush_in_same_pass = scheduler.WouldContinuePass(state);
        runtime.FlushBarriers();
        t_flush_in_same_pass = false;
    }

    pipeline->BindResources(set_writes, push_data);
    if (same_dynamic_state) {
        ++native_stats.dynamic_skipped;
    } else {
        UpdateDynamicState(pipeline, d.indexed, fixed);
    }
    {
        TLG_PROF_LEVEL("sched.begin_rendering", 3);
        scheduler.BeginRendering(state);
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    {
        TLG_PROF_LEVEL("raster.record_draw", 2);
        if (same_pipeline) {
            ++native_stats.bind_skipped;
        } else {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
        }

        if (d.indexed) {
            cmdbuf.drawIndexed(d.num_indices, d.num_instances, 0, s32(d.vertex_offset), d.instance_offset);
        } else {
            cmdbuf.draw(d.num_indices, d.num_instances, d.vertex_offset, d.instance_offset);
        }
    }
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    ++recorded_work;

    VideoCore::ResolveCommands::Run();
    RecordBufferAccess(false, d.op_seq);
    native_last = d.use_tag ? NativeLast{.valid = true, .k2 = d.tag_k2, .pipeline = pipeline, .tick = scheduler.CurrentTick(), .seq = d.call_seq} : NativeLast{};
}

bool Rasterizer::RenderStateMemoEnabled() {
    // Off by default until it is checked in play (documents/tlg/pc-port-draw-pipeline-architecture.md): TLG_RENDER_STATE_MEMO=1 turns it on.
    static const bool enabled = [] {
        const char* env = std::getenv("TLG_RENDER_STATE_MEMO");
        return env && env[0] == '1';
    }();
    return enabled;
}

void Rasterizer::ResolveDrawJob(const GraphicsPipeline* pipeline, DrawJob& job) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.resolve_geometry", 2);
    geometry_plan = job.plan;
    std::tie(job.vertex_offset, job.instance_offset) = GetDrawOffsets(*draw_inputs, pipeline->GetStage(Shader::SwStage::Vertex), pipeline->GetFetchShader());
    ResolveRenderTargets(pipeline, job.render);
    VideoCore::BufferCache::SetUploadSink(&job.vertex_uploads);
    ResolveVertexBuffers(pipeline, job.vertex);
    VideoCore::BufferCache::SetUploadSink(nullptr);
    if (job.indexed) {
        VideoCore::BufferCache::SetUploadSink(&job.index_uploads);
        ResolveIndexBuffer(job.index_offset, false, job.index);
        VideoCore::BufferCache::SetUploadSink(nullptr);
    }
    geometry_plan = nullptr;
}

RenderState Rasterizer::RecordDrawJob(DrawJob& job) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.record_geometry", 2);
    RenderState state = RecordRenderTargets(job.render);
    RecordVertexBuffers(job.vertex, job.vertex_uploads);
    if (job.indexed) {
        RecordIndexBuffer(job.index, job.index_uploads);
    }
    return state;
}

void Rasterizer::BindIndexBuffer(u32 index_offset, bool is_indirect) {
    const u32 mode = SplitBindMode();
    if (mode == 0) {
        BindIndexBufferLegacy(index_offset, is_indirect);
        return;
    }
    static std::atomic<u64> checked{0}, differ{0};
    if (mode == 2) {
        verify_slots_on = true;
        BindIndexBufferLegacy(index_offset, is_indirect);
        verify_slots_on = false;
        IndexBind ib;
        ResolveIndexBuffer(index_offset, is_indirect, ib);
        const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        const bool same = ib.buffer == verify_index.buffer && ib.size == verify_index.size && (verify_index.stream || ib.offset == verify_index.offset);
        if (!same && differ.fetch_add(1) < 20) {
            LOG_WARNING(Render_Vulkan, "split bind verify: index buffer differs");
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind verify: {} index binds, {} differ", n, differ.load());
        }
        return;
    }
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_index_buffer", 2);
    IndexBind ib;
    VideoCore::BufferCache::UploadSink uploads;
    VideoCore::BufferCache::SetUploadSink(&uploads);
    ResolveIndexBuffer(index_offset, is_indirect, ib);
    VideoCore::BufferCache::SetUploadSink(nullptr);
    RecordIndexBuffer(ib, uploads);
}

void Rasterizer::ResetBindings(bool is_compute) {
    TLG_PROF_LEVEL("raster.reset_bindings", 2);
    VideoCore::ResolveCommands::Run();
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PResetResolveCmds);
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DResetResolveCmds);
    // Plan S step P2c: two halves, each with its own owner. Same order as before.
    ResetImageBindings();
    Core::DispatchSplit::Stamp(Core::DispatchSplit::PResetImages);
    Core::DispatchSplit::DStamp(Core::DispatchSplit::DResetImages);
    RecordBufferAccess(is_compute);
}

void Rasterizer::ResetImageBindings() {
    // resolve side: the binding flags of the images this draw bound (state of the image objects the resolve pass owns)
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    bound_images.clear();
}

void Rasterizer::RecordBufferAccess(bool is_compute, u64 op_sequence) {
    // record side: the barrier plan learns what this draw reads and writes; the draw's own barrier flag ends here
    for (const auto [buffer, offset, size, src_access] : bound_buffers) {
        if (buffer_cache.IsStreamBuffer(buffer) && !(src_access & vk::AccessFlagBits2::eShaderWrite)) {
            Core::DispatchSplit::NoteAccess(0);
            continue; // R5: only the host writes the stream ring, so a read of it is no hazard for anything later
        }
        Core::DispatchSplit::NoteAccess((src_access & vk::AccessFlagBits2::eShaderWrite) ? 2 : 1);
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access);
    }
    bound_buffers.clear();
    needs_barrier = false;
    // P3c: everything the operation resolved is recorded now
    VideoCore::OpSequence::retired.store(op_sequence != 0 ? op_sequence : VideoCore::OpSequence::resolving.load(std::memory_order_acquire), std::memory_order_release);
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

    return false;
}

bool Rasterizer::MatchComputeImageCopy(const Pipeline* pipeline, ComputeImageOp& op) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
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
    const auto buf0 = Core::NativeDesc::BufferDescOf(desc0.GetSharp(info));
    const auto buf1 = Core::NativeDesc::BufferDescOf(desc1.GetSharp(info));
    if (buf0.size != buf1.size || cs_pgm.dim_x != (buf0.size / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.address, buf0.size);
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.address, buf1.size, false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.size ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    op.kind = ComputeImageOp::Copy;
    op.source = &src_image;
    op.target = &dst_image;
    return true;
}

bool Rasterizer::MatchComputeImageClear(const Pipeline* pipeline, ComputeImageOp& op) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
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
    const auto sharp1 = desc1.GetSharp(info);
    const auto buf0 = Core::NativeDesc::BufferDescOf(desc0.GetSharp(info));
    const auto buf1 = Core::NativeDesc::BufferDescOf(sharp1);
    const u32 buf1_bpp = Ps4Layout::NumBitsPerBlock(sharp1.GetDataFmt());
    if (buf0.size != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.size) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.address, buf1.size, false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.size || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.address);
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
    op.kind = ComputeImageOp::Clear;
    op.target = &image1;
    op.range = range;
    op.clear = clear;
    return true;
}

void Rasterizer::RunComputeImageOp(const ComputeImageOp& op) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    if (op.kind == ComputeImageOp::Copy) {
        runtime.CopyColorAndDepth(op.source, op.target);
    } else if (op.kind == ComputeImageOp::Clear) {
        runtime.ClearImage(op.target, op.range, op.clear);
    }
}


u32 Rasterizer::SplitBindMode() {
    static const u32 mode = [] { const char* e = std::getenv("TLG_SPLIT_BIND"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }();
    return mode;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    // D3 sizing (2026-11-07): the same time split by pipeline kind
    if (stage.sw_stage == Shader::SwStage::Compute) {
        TLG_PROF_LEVEL("raster.bind_buffers_compute", 2);
        BindBuffersBody(stage, binding, push_data);
    } else {
        TLG_PROF_LEVEL("raster.bind_buffers_graphics", 2);
        BindBuffersBody(stage, binding, push_data);
    }
}

void Rasterizer::BindBuffersBodyLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                                 Shader::PushData& push_data) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_buffers", 2);
    const u64 alignment = instance.StorageMinAlignment();
    const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        u64 cover_t0 = 0, cover_sync0 = 0, cover_copies0 = 0;
        u64 w3_a = 0, w3_b = 0; // (W3 census: the clock before and after the obtain of this binding)
        bool w3_row = false;
        u32 cover_cause = 0;
        if (cover_on) [[unlikely]] {
            t_cover_kind = 0;
            t_bseg[1] = t_bseg[2] = t_bseg[3] = t_bseg[4] = t_bseg[5] = 0;
            cover_t0 = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
            cover_sync0 = ::VideoCore::t_cover_sync_ns;
            cover_copies0 = ::VideoCore::t_cover_sync_copies;
        }
        Census(census_base + CBufferTotal);
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        // TLG_RING_KIND (bisect): 1 draws only, 2 dispatches only, 0 both
        static const u32 ring_kind = [] { const char* e = std::getenv("TLG_RING_KIND"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }(); // default 0 (draws and dispatches) since 2026-11-07, owner play good; 1 draws only
        // kind 3: draws and the dispatches whose stage is not indexed
        const bool dispatch_ok = ring_use.dispatch && (ring_kind == 0 || ring_kind == 2 || (ring_kind == 3 && !(indexed_active && stage.indexed)));
        const RingBinding* ring = const_ring_level >= 4 && ((ring_use.draw && ring_kind != 2) || dispatch_ok) ? FindRingBinding(stage, buffer_index) : nullptr;
        if (ring != nullptr && desc.IsSpecial() && indexed_active && stage.indexed) {
            // an indexed stage has no binding for its flat user data buffer (the shader reads the descriptor words from the row): a ring copy of it would write a binding the layout does not have
            ring = nullptr;
        }
        if (ring != nullptr && ring->formatted) {
            // N5f: a texel buffer may be the memory of an image (a surface a shader reads as a buffer, HTile, CMask): only the raster thread knows
            if (texture_cache.AnyImageOverlaps(ring->address, ring->size) || texture_cache.IsMeta(ring->address)) {
                Core::NativeUploads::NoteRingPrecise(5);
                ring = nullptr;
            } else if (buffer_cache.IsRangeGpuWritten(ring->address, ring->size)) {
                Core::NativeUploads::NoteRingPrecise(6);
                ring = nullptr;
            } else {
                Core::NativeUploads::NoteRingPrecise(4);
            }
        }
        if (Core::OwnMemoryCensus::On()) [[unlikely]] {
            // design review 79, step 1a: what names this binding (the front's engine id for it, from the record)
            namespace OM = Core::OwnMemoryCensus;
            if (ring) {
                OM::NoteBindRing(ring->address, ring->size, ring->hook && hook_ring_buffer);
            } else if (desc.IsSpecial()) {
                OM::NoteBind(OM::Special);
            } else {
                const StagePlan* op = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr;
                const ResolvedBuffer* r = op != nullptr && buffer_index < op->resolved.size() && op->resolved[buffer_index].ahead_id != 0 ? &op->resolved[buffer_index] : nullptr;
                if (r == nullptr) {
                    OM::NoteBind(OM::NoId);
                    u64 at = 0, bytes = 0;
                    const u32* words = binder_plan != nullptr && desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad
                                           ? BinderWords(stage, desc.sharp_fetch.offsets[0], 4, 1)
                                           : nullptr;
                    if (words != nullptr) {
                        const Core::NativeDesc::BufferDesc v = Core::NativeDesc::BufferDescOf(desc.FromWords(words, stage));
                        at = v.address;
                        bytes = v.size;
                    } else if (op == nullptr || !op->flat_skipped) {
                        const Core::NativeDesc::BufferDesc v = Core::NativeDesc::BufferDescOf(desc.GetSharp(stage)); // (the stage's user data and flat user data exist)
                        at = v.address;
                        bytes = v.size;
                    }
                    OM::NoteBindNoId(at, bytes, desc.is_written);
                } else {
                    OM::NoteBindId(r->ahead_id, r->ahead_address, r->ahead_size, desc.is_written, census_base != 0);
                }
            }
        }
        if (ring) {
            Census(census_base + CBufferRing);
            // Copied by the command processor thread into the constant ring (read-only, small, not written by queued work).
            if (ring->hook && hook_ring_buffer) {
                // F1 1a: bound straight from the hook ring at the guest address's ring offset (aligned down, the rest through the push offset as for every other buffer)
                const u64 aligned = Common::AlignDown(ring->offset, alignment);
                const u64 adjust = ring->offset - aligned;
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(hook_ring_buffer->Handle(), aligned, ring->size + adjust);
            } else {
                if (!desc.IsSpecial()) {
                    push_data.AddOffset(binding.buffer, 0);
                }
                buffer_infos.emplace_back(constant_ring->Handle(), ring->offset, ring->size);
            }
            ring_use.bound.fetch_add(1, std::memory_order_relaxed);
            static const bool ring_verify = [] { const char* e = std::getenv("TLG_RING_VERIFY"); return e != nullptr && e[0] == '1'; }();
            if (ring_verify && !desc.IsSpecial()) {
                // bisect: the descriptor the raster thread's own path would choose against the one the command processor copied for
                AmdGpu::Buffer own_sharp;
                bool own_record = false;
                if (binder_plan != nullptr && desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
                    u32 record_binding = ~0u;
                    if (const u32* words = BinderWords(stage, desc.sharp_fetch.offsets[0], 4, 1, &record_binding)) {
                        own_sharp = desc.FromWords(words, stage);
                        own_record = true;
                    }
                }
                if (!own_record) {
                    own_sharp = desc.GetSharp(stage);
                }
                const Core::NativeDesc::BufferDesc own = Core::NativeDesc::BufferDescOf(own_sharp);
                static std::atomic<u64> sharp_checked[2]{}, sharp_differ[2]{};
                static std::atomic<u32> sharp_logged{0};
                const u32 kind = ring_use.dispatch ? 1u : 0u;
                sharp_checked[kind].fetch_add(1, std::memory_order_relaxed);
                if (own.address != ring->address || own.size != ring->size) {
                    sharp_differ[kind].fetch_add(1, std::memory_order_relaxed);
                    if (sharp_logged.fetch_add(1) < 40) {
                        LOG_WARNING(Render_Vulkan, "ring verify sharp: {} stage {:#x} buffer {} ring {:#x}+{:#x} own {:#x}+{:#x} (from record {}); differ {}/{} of {}/{} (draw/dispatch)", kind ? "dispatch" : "draw",
                                    stage.pgm_hash, buffer_index, ring->address, ring->size, own.address, own.size, own_record, sharp_differ[0].load(), sharp_differ[1].load(),
                                    sharp_checked[0].load(), sharp_checked[1].load());
                    }
                }
            }
            if (ring_verify && !ring->hook && ring->address != 0 && memory->IsValidMapping(ring->address, ring->size)) { // (a hook ring binding is checked by TLG_HOOK_RING itself)
                // bisect: the ring copy (made by the command processor) against guest memory now (what the raster thread's own copy would read)
                std::vector<u8> now(ring->size);
                memory->CopySparseMemory(ring->address, now.data(), ring->size);
                static std::atomic<u64> checked[2]{}, differ[2]{};
                static std::atomic<u32> logged{0};
                const u32 kind = ring_use.dispatch ? 1u : 0u;
                const u64 n_checked = checked[kind].fetch_add(1, std::memory_order_relaxed) + 1;
                if ((n_checked & (n_checked - 1)) == 0) {
                    LOG_WARNING(Render_Vulkan, "ring verify: {} checked {} differ {}", kind ? "dispatch" : "draw", n_checked, differ[kind].load());
                }
                if (std::memcmp(now.data(), constant_ring->Data(ring->offset), ring->size) != 0) {
                    differ[kind].fetch_add(1, std::memory_order_relaxed);
                    if (logged.fetch_add(1) < 40) {
                        size_t first = 0;
                        while (first < ring->size && now[first] == constant_ring->Data(ring->offset)[first]) {
                            ++first;
                        }
                        LOG_WARNING(Render_Vulkan, "ring verify: {} stage {:#x} buffer {} addr {:#x}+{:#x} differs at +{:#x} (ring {:#x}, memory {:#x}); checked {}/{} differ {}/{} (draw/dispatch)",
                                    kind ? "dispatch" : "draw", stage.pgm_hash, buffer_index, ring->address, ring->size, first, constant_ring->Data(ring->offset)[first], now[first],
                                    checked[0].load(), checked[1].load(), differ[0].load(), differ[1].load());
                    }
                }
            }
            if (Core::NativeUploads::RingPrecise() && ring->address != 0 && buffer_cache.IsRangeGpuWritten(ring->address, ring->size)) {
                Core::NativeUploads::NoteRingPrecise(1); // the GPU wrote bytes of this range after the command processor copied it
            }
        } else if (desc.IsSpecial()) {
            Census(census_base + CBufferSpecial);
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                Core::ResourceAudit::Note(Core::ResourceAudit::BufferUse, 0xfffff0000000ull, gds_buf->SizeBytes(), desc.is_written);
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                if (indexed_active && stage.indexed) {
                    NoteIndexedFlat(stage, stage.flattened_ud_buf.data(), static_cast<u32>(stage.flattened_ud_buf.size()));
                    ++binding.buffer;
                    continue;
                }
                if (cover_on) [[unlikely]] {
                    t_w1.bad |= W1BadFlat;
                }
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
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                if (indexed_active && stage.indexed) {
                    Core::RareEvents::Note("c9_shared_memory", fmt::format("compute shader {:#x}: the shared memory is a storage buffer with a slot of its own per dispatch", stage.pgm_hash));
                }
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
            t_fseg_on = false;
            if (cover_on) [[unlikely]] {
                t_fseg[0] = BsegNow();
            }
            own_compare = nullptr;
            if (const u32 own_mode = Core::OwnMemoryCensus::OwnImmutableMode(), mutable_mode = Core::OwnMemoryCensus::OwnMutableMode();
                (own_mode != 0 && !desc.is_written) || mutable_mode != 0) {
                // design review 83 (TLG_OWN_IMMUTABLE): a read binding of an immutable engine buffer, named by its id, bound from the buffer's own memory. Its bytes never change
                // after the synchronization (a CPU write clears `synced`; GPU copies into it write that same buffer and the barrier query below orders them)
                namespace OM = Core::OwnMemoryCensus;
                const StagePlan* op = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr;
                const ResolvedBuffer* r = op != nullptr && buffer_index < op->resolved.size() && op->resolved[buffer_index].ahead_id != 0 ? &op->resolved[buffer_index] : nullptr;
                if (r != nullptr) {
                    auto* h = Core::NativeResources::FindImmutableById(r->ahead_id);
                    const bool imm = h != nullptr && h->usage == 1;
                    // class: 0 immutable read, 1 mutable read, 2 mutable written (design review 85); an immutable written binding is left alone
                    const u32 cls = imm ? 0u : desc.is_written ? 2u : 1u;
                    const u32 mode = imm ? (desc.is_written ? 0u : own_mode) : mutable_mode;
                    if (mode != 0) {
                        OM::NoteOwn(OM::OwnAsked, cls);
                    }
                    if (mode != 0 && h != nullptr && r->ahead_address >= h->begin && r->ahead_address + r->ahead_size <= h->end && h->alive.load(std::memory_order_acquire)) {
                        OM::NoteOwn(OM::OwnImmutable, cls);
                        const u32 close_mode = OM::OwnCloseMode();
                        const auto* own = static_cast<const VideoCore::Buffer*>(h->own.load(std::memory_order_acquire));
                        if (own == nullptr && close_mode != 0) {
                            own = buffer_cache.OwnResolve(*h); // design review 87: once per buffer (until the placements change)
                        }
                        const bool by_resolved = own != nullptr && h->own_new.load(std::memory_order_acquire);
                        u32 eff = mode;
                        if (by_resolved) {
                            eff = close_mode == 0 ? 0u : close_mode == 1 ? 1u : mode; // shadow: compared; real: as its class
                        }
                        if (own == nullptr || eff == 0) {
                            OM::NoteOwn(OM::OwnNoOwn, cls);
                        } else if (by_resolved ? h->cpu_dirty.load(std::memory_order_acquire)
                                               : imm ? !h->synced.load(std::memory_order_acquire) : h->cpu_dirty.load(std::memory_order_acquire)) {
                            OM::NoteOwn(OM::OwnNotSynced, cls);
                        } else if (desc.is_written && !buffer_cache.GpuWrittenContains(r->ahead_address, r->ahead_size)) {
                            OM::NoteOwn(OM::OwnNotGpuMarked, cls); // (the old path marks it: the address readers must see the GPU's bytes from the first write on)
                        } else {
                            if (by_resolved) {
                                OM::NoteOwn(OM::OwnByResolved, cls);
                            }
                            static thread_local u32 own_tick = 0;
                            if (eff == 2 && (++own_tick & 63u) != 0) {
                                OM::NoteOwn(OM::OwnBound, cls);
                                const VAddr address = r->ahead_address;
                                const u64 size = r->ahead_size;
                                const u64 offset = own->Offset(address);
                                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                                const u64 adjust = offset - offset_aligned;
                                const_cast<VideoCore::Buffer*>(own)->NoteResolveUse();
                                if (desc.is_written) {
                                    // what a GPU write must tell the address-keyed readers, as the written rows path does (design review 37)
                                    Core::NativeRegistry::SetBufferWriter(stage.pgm_hash, static_cast<u32>(stage.sw_stage));
                                    buffer_cache.BindMemoErase(address, size);
                                    Core::NativeResources::NoteGpuWritten(address, size);
                                    Core::RowVersion::NoteWritten(0);
                                    recipe_reusable = false;
                                }
                                recipe_ranges.emplace_back(address, static_cast<u32>(std::min<u64>(size, 0xffffffffull)));
                                if (size > 262144) {
                                    recipe_reusable = false;
                                }
                                push_data.AddOffset(binding.buffer, adjust);
                                buffer_infos.emplace_back(own->Handle(), offset_aligned, size + adjust);
                                bound_buffers.emplace_back(own, offset, size, src_access);
                                if (desc.is_written) {
                                    InvalidateWrittenBuffer(r->ahead_id, address, size, true);
                                }
                                if (BufferBarrierQuery(own, offset, size, desc.is_written)) {
                                    needs_barrier = true;
                                }
                                goto buffer_bound;
                            }
                            // the shadow, or the audit: the old path answers; compared at the end of the binding
                            own_compare = own;
                            own_compare_offset = own->Offset(r->ahead_address);
                            own_compare_size = r->ahead_size;
                            own_compare_slot = binding.buffer;
                            own_compare_cls = cls;
                            VideoCore::BufferCache::UploadBytesReset();
                        }
                    }
                }
            }
            if (Vulkan::RowShare::Mode() == 2) {
                // rows ahead real (design review 56): the row the front planned for this binding, still valid now, is bound without the descriptor decode, the bind context and the lookup
                t_ahead_sample = nullptr;
                auto& rc = rows_ahead_real[census_base != 0 ? 1 : 0];
                const StagePlan* ap = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr;
                const ResolvedBuffer* a = ap != nullptr && buffer_index < ap->resolved.size() && ap->resolved[buffer_index].plan_found ? &ap->resolved[buffer_index] : nullptr;
                // LEGACY(engine_rows): the rows for engine buffer bindings are deleted by default (design review 89); TLG_LEGACY_ENGINE_ROWS=1 brings them back
                static const bool rows_env_on = [] { const char* e = std::getenv("TLG_NATIVE_ROWS"); return Core::Legacy::EngineRows() && (e == nullptr || e[0] != '0'); }();
                if (a == nullptr || !rows_env_on || (desc.is_written && !Core::RowVersion::WrittenRowsOn())) {
                    rc[7].fetch_add(1, std::memory_order_relaxed);
                } else if ((a->plan_flags & Vulkan::RowShare::FlagStatic) != 0) {
                    rc[6].fetch_add(1, std::memory_order_relaxed);
                } else if ((a->plan_flags & Vulkan::RowShare::FlagInside) == 0 || a->plan_epoch != buffer_cache.PlacementEpoch() || a->plan_version != Core::RowVersion::Of(a->ahead_id)) {
                    rc[5].fetch_add(1, std::memory_order_relaxed);
                } else {
                    static thread_local u32 ahead_tick = 0;
                    const u64 clock_now = Core::NativeIdentity::Clock();
                    const u64 walk_t0 = cover_on ? BsegNow() : 0;
                    const bool walk_changed = Core::NativeIdentity::ChangedSince(a->plan_clock, a->ahead_address, a->ahead_address + a->ahead_size) != Core::NativeIdentity::Change::None;
                    if (walk_t0 != 0) [[unlikely]] {
                        auto& w = rows_ahead_walk[census_base != 0 ? 1 : 0];
                        w[0].fetch_add(1, std::memory_order_relaxed);
                        w[1].fetch_add(clock_now > a->plan_clock ? clock_now - a->plan_clock : 0, std::memory_order_relaxed);
                        w[2].fetch_add(BsegNow() - walk_t0, std::memory_order_relaxed);
                    }
                    if (walk_changed) {
                        rc[4].fetch_add(1, std::memory_order_relaxed);
                    } else if ((++ahead_tick & 63u) == 0) {
                        // (sampled only among the rows the whole check accepts, the identity walk included: those are the ones the fast path binds)
                        rc[1].fetch_add(1, std::memory_order_relaxed);
                        t_ahead_sample = a; // the full path answers; compared below
                    } else {
                        {
                            rc[0].fetch_add(1, std::memory_order_relaxed);
                            if (cover_on) [[unlikely]] {
                                ++t_w3.fast;
                                static thread_local u32 fseg_tick = 0;
                                t_fseg_on = (++fseg_tick & 7u) == 0;
                                t_fseg[1] = BsegNow();
                            }
                            const VAddr address = a->ahead_address;
                            const u64 size = a->ahead_size;
                            const auto* buffer = static_cast<const VideoCore::Buffer*>(a->plan_buffer);
                            const u64 offset = a->plan_offset;
                            Vulkan::RowShare::TouchClock(Vulkan::RowShare::Key(a->ahead_id, address, size, desc.is_formatted, desc.is_written), clock_now);
                            Census(census_base + CBufferHandle);
                            w1s_counts[128].fetch_add(1, std::memory_order_relaxed);
                            Core::NativeRegistry::SetBufferWriter(stage.pgm_hash, static_cast<u32>(stage.sw_stage));
                            VideoCore::BufferCache::UploadBytesReset();
                            const_cast<VideoCore::Buffer*>(buffer)->NoteResolveUse();
                            if (desc.is_written) {
                                // F2 item 3: what the old path's written answer does besides answering (as the row-served path above)
                                buffer_cache.BindMemoErase(address, size);
                                Core::NativeResources::NoteGpuWritten(address, size);
                                Core::RowVersion::NoteWritten(0);
                            }
                            recipe_ranges.emplace_back(address, static_cast<u32>(std::min<u64>(size, 0xffffffffull)));
                            if (desc.is_written || size > 262144) {
                                recipe_reusable = false;
                            }
                            if (t_fseg_on) [[unlikely]] {
                                t_fseg[2] = BsegNow();
                            }
                            const u64 offset_aligned = Common::AlignDown(offset, alignment);
                            const u64 adjust = offset - offset_aligned;
                            push_data.AddOffset(binding.buffer, adjust);
                            buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                            bound_buffers.emplace_back(buffer, offset, size, src_access);
                            if (desc.is_written) {
                                InvalidateWrittenBuffer(a->ahead_id, address, size, true); // LEGACY(engine_rows): rows ahead bind only inside the buffer
                            }
                            if (t_fseg_on) [[unlikely]] {
                                t_fseg[3] = BsegNow();
                            }
                            Census(census_base + CBufferBarrierQuery);
                            if (BufferBarrierQuery(buffer, offset, size, desc.is_written)) {
                                Census(census_base + CBufferBarrierTrue);
                                needs_barrier = true;
                            }
                            if (t_fseg_on) [[unlikely]] {
                                t_fseg[4] = BsegNow();
                            }
                            goto buffer_bound;
                        }
                    }
                }
            }
            bool cp_no_object = false;
            if (const StagePlan* ns = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr) {
                cp_no_object = buffer_index < ns->resolved.size() && ns->resolved[buffer_index].no_object && ns->resolved_epoch == Core::NativeResources::ImmutableEpoch();
                cover_cause = cp_no_object ? 1u : 0u;
            }
            if (const StagePlan* rs = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr;
                rs != nullptr && buffer_index < rs->resolved.size() && rs->resolved[buffer_index].object != nullptr) {
                // R2: resolved by the command processor from the record's engine object; bound when its stamps hold
                const ResolvedBuffer& r = rs->resolved[buffer_index];
                if (rs->resolved_epoch == Core::NativeResources::ImmutableEpoch()) {
                    const auto* object = r.object;
                    u64 copy_offset = 0;
                    if (const auto* copy = object->usage != 1 && object->alive.load(std::memory_order_acquire) && object->generation.load(std::memory_order_acquire) == r.generation
                                               ? buffer_cache.ObjectCopy(*object, r.address, copy_offset)
                                               : nullptr) {
                        // R1b: the object's own copy of this generation in the stream ring
                        const u64 offset_aligned = Common::AlignDown(copy_offset, alignment);
                        const u64 adjust = copy_offset - offset_aligned;
                        push_data.AddOffset(binding.buffer, adjust);
                        buffer_infos.emplace_back(copy->Handle(), offset_aligned, r.size + adjust);
                        cover_cause = 9;
                        Census(census_base + CBufferResolved);
                        resolve_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        recipe_ranges.emplace_back(r.address, r.size);
                        goto buffer_bound;
                    }
                    const bool ready = object->alive.load(std::memory_order_acquire) && object->synced.load(std::memory_order_acquire); // the arena fields are stored before `synced`
                    const auto* arena = ready ? static_cast<const VideoCore::Buffer*>(object->resident_buffer.load(std::memory_order_relaxed)) : nullptr;
                    if (ready && object->generation.load(std::memory_order_acquire) == r.generation && arena != nullptr &&
                        object->resident_epoch.load(std::memory_order_acquire) == buffer_cache.PlacementEpoch() &&
                        (object->usage == 1 || !buffer_cache.IsRegionCpuModified(r.address, r.size))) { // as ObtainBuffer: the tracker's dirty bit is the safety net of a resident buffer
                        const u64 offset = r.address - object->resident_base.load(std::memory_order_relaxed);
                        const u64 offset_aligned = Common::AlignDown(offset, alignment);
                        const u64 adjust = offset - offset_aligned;
                        push_data.AddOffset(binding.buffer, adjust);
                        buffer_infos.emplace_back(arena->Handle(), offset_aligned, r.size + adjust);
                        bound_buffers.emplace_back(arena, offset, r.size, src_access);
                        if (runtime.PlanHasWrites() && runtime.IsBufferAccessed(arena, offset, r.size, false)) {
                            needs_barrier = true;
                        }
                        cover_cause = 10;
                        Census(census_base + CBufferResolved);
                        resolve_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        recipe_ranges.emplace_back(r.address, r.size);
                        goto buffer_bound;
                    }
                    // why: 0 epoch, 1 gone, 2 not synchronized, 3 generation moved, 4 arena, 5 tracker dirty
                    const u32 why = !object->alive.load(std::memory_order_relaxed) ? 1 : !object->synced.load(std::memory_order_relaxed) ? 2
                                    : object->generation.load(std::memory_order_relaxed) != r.generation                       ? 3
                                    : arena == nullptr || object->resident_epoch.load(std::memory_order_relaxed) != buffer_cache.PlacementEpoch() ? 4 : 5;
                    resolve_stats.stale_why[why].fetch_add(1, std::memory_order_relaxed);
                    cover_cause = 3 + why;
                } else {
                    resolve_stats.stale_why[0].fetch_add(1, std::memory_order_relaxed);
                    cover_cause = 2;
                }
                resolve_stats.stale.fetch_add(1, std::memory_order_relaxed);
            }
            AmdGpu::Buffer chosen_sharp;
            bool from_record = false;
            u32 record_binding = ~0u;
            if (binder_plan != nullptr && desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
                if (const u32* words = BinderWords(stage, desc.sharp_fetch.offsets[0], 4, 1, &record_binding)) {
                    chosen_sharp = desc.FromWords(words, stage);
                    from_record = true;
                }
            }
            if (from_record) {
                Core::NativeRegistry::NoteBinder(0);
            } else if (StageFlatSkipped(stage) && desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::MultiLoad) {
                Core::NativeRegistry::NoteBinder(6); // must not happen: the plan skipped the flat user data for a stage the record covers
                if (stage.sw_stage == Shader::SwStage::Compute) {
                    DumpDispatchRecord(stage, 1, desc.sharp_fetch.offsets[0], "no record entry: buffer");
                }
                chosen_sharp = AmdGpu::Buffer::Null();
            } else {
                chosen_sharp = desc.GetSharp(stage);
                if (desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::MultiLoad && desc.sharp_fetch.load_mask == 0) {
                    Core::NativeRegistry::NoteBinder(12); // F5: built from the shader's immediates, GetSharp reads no flat dword (not a flat user data read)
                } else {
                    if (binder_plan != nullptr) {
                        Core::NativeRegistry::NoteBinder(2);
                    }
                }
            }
            // F4: from here on the buffer is a plain description (the descriptor type stays at the recompiler's interface above)
            const Core::NativeDesc::BufferDesc vsharp = Core::NativeDesc::BufferDescOf(chosen_sharp);
            if (vsharp.address == 0 || vsharp.size == 0) {
                Census(census_base + CBufferNull);
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else if (vsharp.size >= (1ull << 30) && !memory->IsValidMapping(vsharp.address)) {
                LOG_ERROR(Render, "buffer sharp at unmapped address {:#x} of {} bytes for stage {:#x}: bound as null", vsharp.address,
                          vsharp.size, stage.pgm_hash);
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.address, vsharp.size);
                if (size != vsharp.size) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.size, size, stage.pgm_hash);
                }
                // N7.5: a read-only buffer of guest memory bound where it is (imported view of the runtime's backing), no copy, no dirty tracking
                bool in_place = false;
                if (inplace_on) {
                    inplace_offered.fetch_add(1, std::memory_order_relaxed);
                    if (desc.is_written) {
                        inplace_written.fetch_add(1, std::memory_order_relaxed);
                    } else if (size > inplace_cap) {
                        inplace_too_large.fetch_add(1, std::memory_order_relaxed);
                    } else if (buffer_cache.IsRangeGpuWritten(vsharp.address, size)) {
                        inplace_gpu_written.fetch_add(1, std::memory_order_relaxed);
                    } else if (desc.is_formatted && (texture_cache.AnyImageOverlaps(vsharp.address, size) || texture_cache.IsMeta(vsharp.address))) {
                        inplace_image_or_meta.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        const auto window = host_import->Find(vsharp.address, size);
                        const u64 aligned = Common::AlignDown(window.offset, alignment);
                        if (window.buffer && window.offset - aligned + size <= window.chunk_remaining + (window.offset - aligned)) {
                            in_place = true;
                            Census(census_base + CBufferInPlace);
                            if (cover_on) [[unlikely]] {
                                t_w1.bad |= W1BadInPlace;
                            }
                            const u64 adjust = window.offset - aligned;
                            push_data.AddOffset(binding.buffer, adjust);
                            buffer_infos.emplace_back(window.buffer, aligned, size + adjust);
                            inplace_taken.fetch_add(1, std::memory_order_relaxed);
                            inplace_bytes.fetch_add(size, std::memory_order_relaxed);
                        } else {
                            inplace_not_in_backing.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                if (!in_place) {
                if (cover_on) [[unlikely]] {
                    t_bseg[1] = BsegNow();
                }
                // N3c: the immutable engine buffer named by the record's binding (by the engine's id), when the plan resolved one and nothing was registered since
                Core::NativeResources::ImmutableBuffer* immutable_hint = nullptr;
                if (record_binding != ~0u && binder_plan->has_handles && (binder_plan->handles[record_binding] != nullptr)) {
                    if (binder_plan->handle_epoch == Core::NativeResources::ImmutableEpoch()) {
                        immutable_hint = binder_plan->handles[record_binding];
                    } else {
                        Core::NativeResources::NoteHandle(1);
                        // (the life census that read the plan's handle here is gone: the handle was made before a create / delete, and the object it names may already have left the
                        // graveyard, so its fields cannot be read; a crash in bloom, 2026-11-09, `alive.load` of a freed object)
                    }
                }
                if (immutable_hint != nullptr) {
                    Census(census_base + CBufferHandle);
                    if (cover_on && desc.is_formatted && !desc.is_written) [[unlikely]] {
                        const u32 b = census_base != 0 ? 4u : 0u;
                        cover_texel[b + 0].fetch_add(1, std::memory_order_relaxed);
                        cover_texel[b + 1].fetch_add(texture_cache.AnyImageOverlaps(vsharp.address, size) ? 1 : 0, std::memory_order_relaxed);
                        cover_texel[b + 2].fetch_add(texture_cache.IsMeta(vsharp.address) ? 1 : 0, std::memory_order_relaxed);
                        cover_texel[b + 3].fetch_add(buffer_cache.IsRangeGpuWritten(vsharp.address, size) ? 1 : 0, std::memory_order_relaxed);
                    }
                } else if (desc.is_written) {
                    Census(census_base + CBufferAddrWritten);
                } else {
                    Census(census_base + CBufferAddrRead);
                    Census(census_base + (size <= 256 ? CBufferAddrReadLe256 : size <= 4096 ? CBufferAddrReadLe4k : size <= 65536 ? CBufferAddrReadLe64k : CBufferAddrReadBig));
                }
                Core::NativeRegistry::SetBufferWriter(stage.pgm_hash, static_cast<u32>(stage.sw_stage));
                Core::NativeContent::SetBindContext({"shader binding", record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0,
                                                     record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].view_id : 0, stage.pgm_hash,
                                                     static_cast<u32>(stage.sw_stage), immutable_hint != nullptr, desc.is_written, vsharp.address, size});
                const u64 w31_id = immutable_hint != nullptr ? immutable_hint->id : record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0;
                if (cover_on) [[unlikely]] {
                    w3_a = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                }
                if (cover_on && W1ShadowOn()) [[unlikely]] {
                    W31Before(w31_id, vsharp.address, size);
                }
                // W3.2 (TLG_NATIVE_ROWS=1): a read-only request of an engine buffer with a valid row is answered from the row; 1 in 64 also asks the old path and compares
                // LEGACY(engine_rows): the rows for engine buffer bindings (serve and learn) are deleted by default (design review 89): own memory by id answers them, the
                // rest take ObtainBuffer, whose answer the rows cached. TLG_LEGACY_ENGINE_ROWS=1 brings them back (then TLG_NATIVE_ROWS=0 turns them off as before)
                static const bool rows_on = [] {
                    const char* e = std::getenv("TLG_NATIVE_ROWS");
                    return Core::Legacy::EngineRows() && (e == nullptr || e[0] != '0');
                }();
                const VideoCore::Buffer* row_buffer = nullptr;
                u64 row_offset = 0;
                const bool rows_here = rows_on && (!desc.is_written || Core::RowVersion::WrittenRowsOn()); // F2 item 3: written bindings by row too (TLG_WRITTEN_ROWS=1)
                const bool row_served = rows_here && RowsServe(w31_id, vsharp.address, size, desc.is_formatted, row_buffer, row_offset, desc.is_written);
                if (cover_on) [[unlikely]] {
                    t_bseg[2] = w3_a;
                    t_bseg[3] = BsegNow();
                    t_bseg_served = row_served;
                }
                static const bool ahead_census = [] { const char* e = std::getenv("TLG_ROW_AHEAD_CENSUS"); return e != nullptr && e[0] == '1'; }();
                if (ahead_census && w31_id != 0) [[unlikely]] {
                    // rows ahead census (design review 54): would the front's key have found this row, and was its version still the one the front saw
                    auto& c = row_ahead_census[stage.sw_stage == Shader::SwStage::Compute ? 1 : 0];
                    c[0].fetch_add(1, std::memory_order_relaxed);
                    const StagePlan* ap = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                          : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                      : nullptr;
                    const ResolvedBuffer* a = ap != nullptr && buffer_index < ap->resolved.size() && ap->resolved[buffer_index].ahead_id != 0 ? &ap->resolved[buffer_index] : nullptr;
                    if (a == nullptr) {
                        c[1].fetch_add(1, std::memory_order_relaxed);
                    } else if (a->ahead_id != w31_id || a->ahead_address != vsharp.address || a->ahead_size != size) {
                        c[2].fetch_add(1, std::memory_order_relaxed);
                    } else {
                        c[!row_served ? 5 : a->ahead_version == Core::RowVersion::Of(w31_id) ? 3 : 4].fetch_add(1, std::memory_order_relaxed);
                        if (desc.is_written) {
                            c[6].fetch_add(1, std::memory_order_relaxed);
                        }
                        if (desc.is_formatted) {
                            c[7].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                // (what a learn needs, taken before the old path answers; only asked of the requests the old path answers)
                u64 rows_epoch_pre = 0, rows_clock_pre = 0, rows_version_pre = 0, rows_unattr_pre = 0;
                bool rows_gpu_pre = false;
                if (rows_here && (!row_served || false)) {
                    rows_epoch_pre = buffer_cache.PlacementEpoch();
                    rows_clock_pre = Core::NativeIdentity::Clock();
                    if (Core::RowVersion::Mode() != 0) [[unlikely]] {
                        rows_version_pre = Core::RowVersion::Of(w31_id);
                        rows_unattr_pre = Core::RowVersion::Unattributed();
                    }
                    rows_gpu_pre = buffer_cache.IsRangeGpuWritten(vsharp.address, size);
                }
                // D4 sizing (2026-11-07): the same call, timed by what kind of binding it serves
                const auto obtain_old = [&] {
                    VideoCore::BufferCache::UploadBytesReset();
                    if (immutable_hint != nullptr) {
                        TLG_PROF_LEVEL("bind.obtain_by_handle", 3);
                        if (stage.sw_stage == Shader::SwStage::Compute) {
                            if (desc.is_written) { TLG_PROF_COUNT("bind.handle.compute.written"); } else if (desc.is_formatted) { TLG_PROF_COUNT("bind.handle.compute.texel"); } else { TLG_PROF_COUNT("bind.handle.compute.plain"); }
                        } else {
                            if (desc.is_written) { TLG_PROF_COUNT("bind.handle.graphics.written"); } else if (desc.is_formatted) { TLG_PROF_COUNT("bind.handle.graphics.texel"); } else { TLG_PROF_COUNT("bind.handle.graphics.plain"); }
                        }
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_handle", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_handle", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    } else if (desc.is_written) {
                        TLG_PROF_LEVEL("bind.obtain_by_address_written", 3);
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_addr_written", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_addr_written", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    } else {
                        TLG_PROF_LEVEL("bind.obtain_by_address_read", 3);
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_addr_read", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_addr_read", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    }
                };
                std::pair<const VideoCore::Buffer*, u64> obtained;
                if (row_served && t_row_static) {
                    // F1 class 2 static row: one read in 64 hashes the guest bytes against the learn's hash; a change nothing announced is not served
                    Core::RowVersion::NoteStaticReal(1);
                    static thread_local u32 static_tick = 0;
                    bool keep = true;
                    if ((++static_tick & 63u) == 0) {
                        Core::RowVersion::NoteStaticReal(2);
                        u64 now_hash = 0;
                        if (!Core::NativeContent::HashGuarded64(vsharp.address, size, now_hash) || now_hash != t_row_hash) {
                            Core::RowVersion::NoteStaticReal(3);
                            static std::atomic<u32> logged{0};
                            if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                                LOG_ERROR(Render_Vulkan, "static row: id {:#x} {:#x}+{:#x}: guest bytes changed since the learn, the version did not move (not served)", w31_id, vsharp.address, size);
                            }
                            t_rows_real.erase(RowKey(w31_id, vsharp.address, size, desc.is_formatted));
                            if (Vulkan::RowShare::Mode() != 0) {
                                Vulkan::RowShare::Erase(RowKey(w31_id, vsharp.address, size, desc.is_formatted));
                            }
                            keep = false;
                        }
                    }
                    if (keep) {
                        VideoCore::BufferCache::UploadBytesReset();
                        const_cast<VideoCore::Buffer*>(row_buffer)->NoteResolveUse();
                        obtained = {row_buffer, row_offset};
                    } else {
                        obtained = obtain_old();
                    }
                } else if (row_served) {
                    w1s_counts[128].fetch_add(1, std::memory_order_relaxed);
                    static std::atomic<u64> verify_tick{0};
                    if ((verify_tick.fetch_add(1, std::memory_order_relaxed) & 63u) == 0) {
                        const auto* list = scheduler.CommandBuffer().List();
                        const u64 commands_before = list != nullptr ? list->Size() : 0;
                        obtained = obtain_old(); // the old answer decides
                        const auto* list_after = scheduler.CommandBuffer().List();
                        const u64 commands_after = list_after != nullptr ? list_after->Size() : 0;
                        w1s_counts[129].fetch_add(1, std::memory_order_relaxed);
                        if (obtained.first != row_buffer || obtained.second != row_offset) {
                            w1s_counts[130].fetch_add(1, std::memory_order_relaxed);
                            // 136 no engine handle, 137 handle of usage 1 (immutable), 138 handle of another usage; 139 the old path answered from the stream ring
                            w1s_counts[immutable_hint == nullptr ? 136 : immutable_hint->usage == 1 ? 137 : 138].fetch_add(1, std::memory_order_relaxed);
                            if (buffer_cache.IsStreamBuffer(obtained.first)) {
                                w1s_counts[139].fetch_add(1, std::memory_order_relaxed);
                            }
                            static std::atomic<u32> logged{0};
                            if (logged.fetch_add(1) < 10) {
                                LOG_ERROR(Render_Vulkan, "W3.2 row differs: id {:#x} address {:#x}+{:#x} (hint {} usage {} texel {} compute {} row gpu {} gpu now {} object dirty now {}): row {:p}+{:#x}, old path {:p}+{:#x}", w31_id, vsharp.address, size,
                                          immutable_hint != nullptr, immutable_hint != nullptr ? immutable_hint->usage : 0u, desc.is_formatted, stage.sw_stage == Shader::SwStage::Compute, t_row_gpu,
                                          buffer_cache.IsRangeGpuWritten(vsharp.address, size), buffer_cache.ObjectRangeDirty(w31_id, vsharp.address, size), static_cast<const void*>(row_buffer),
                                          row_offset, static_cast<const void*>(obtained.first), obtained.second);
                            }
                        } else if (VideoCore::BufferCache::UploadBytes() != 0 || commands_after != commands_before) {
                            w1s_counts[131].fetch_add(1, std::memory_order_relaxed);
                            static std::atomic<u32> logged_work{0};
                            if (logged_work.fetch_add(1) < 10) {
                                LOG_ERROR(Render_Vulkan, "W3.2 row valid but the old path did work: id {:#x} address {:#x}+{:#x} uploaded {} bytes, {} commands", w31_id, vsharp.address, size,
                                          VideoCore::BufferCache::UploadBytes(), commands_after - commands_before);
                            }
                        }
                    } else {
                        VideoCore::BufferCache::UploadBytesReset();
                        const_cast<VideoCore::Buffer*>(row_buffer)->NoteResolveUse(); // P3c: asked for by the operation being resolved
                        if (desc.is_written) {
                            // F2 item 3: what the old path's written answer does besides answering: answers over the memory are stale, the resident copies of its bytes too
                            buffer_cache.BindMemoErase(vsharp.address, size);
                            Core::NativeResources::NoteGpuWritten(vsharp.address, size);
                            Core::RowVersion::NoteWritten(0);
                        }
                        obtained = {row_buffer, row_offset};
                    }
                } else {
                    obtained = obtain_old();
                }
                if (rows_on && !desc.is_written && !row_served && Core::RowVersion::StaticRowsOn() && w31_id != 0 && buffer_cache.IsStreamBuffer(obtained.first) &&
                    !Core::NativeContent::InRing(vsharp.address, size) && !buffer_cache.IsRangeGpuWritten(vsharp.address, size) &&
                    !(desc.is_formatted && (texture_cache.AnyImageOverlaps(vsharp.address, size) || texture_cache.IsMeta(vsharp.address)))) {
                    // F1 class 2 static rows: the stream ring answered this time (the bytes are right); the buffer's own memory gets them once and the row is learned from it
                    Core::NativeDevice::BufferPart part{};
                    u64 learn_hash = 0;
                    if (Core::NativeDevice::BufferPartOf(w31_id, vsharp.address, size, part) && part.begin == vsharp.address && part.end == vsharp.address + size &&
                        Core::NativeContent::HashGuarded64(vsharp.address, size, learn_hash)) {
                        u64 uploaded = 0;
                        const auto [static_buffer, static_offset] = buffer_cache.ObtainStatic(w31_id, vsharp.address, size, uploaded);
                        Core::RowVersion::NoteStaticReal(4, uploaded);
                        if (static_buffer != nullptr && !buffer_cache.IsStreamBuffer(static_buffer) && !buffer_cache.IsHookRing(static_buffer)) {
                            if (t_rows_real.size() > 400000) {
                                t_rows_real.clear();
                            }
                            {
                                const u64 key = RowKey(w31_id, vsharp.address, size, desc.is_formatted);
                                const RealRow& row = t_rows_real[key] =
                                    RealRow{w31_id, vsharp.address, size, desc.is_formatted, static_buffer, static_offset, rows_epoch_pre, rows_clock_pre, false, rows_version_pre, rows_unattr_pre,
                                            true, true, learn_hash};
                                ShareRow(key, row);
                            }
                            Core::RowVersion::NoteStaticReal(0);
                        } else {
                            Core::RowVersion::NoteStaticReal(5);
                        }
                    } else {
                        Core::RowVersion::NoteStaticReal(5);
                    }
                }
                if (rows_here && desc.is_written && !row_served) {
                    if (!buffer_cache.IsHookRing(obtained.first) && !buffer_cache.IsStreamBuffer(obtained.first)) {
                        RowsLearn(w31_id, vsharp.address, size, desc.is_formatted, obtained.first, obtained.second, rows_epoch_pre, rows_clock_pre, rows_gpu_pre, rows_version_pre, rows_unattr_pre,
                                  true);
                    }
                }
                if (rows_on && !desc.is_written && !row_served) {
                    if (Core::RowVersion::StaticCensusOn() && w31_id != 0 && buffer_cache.IsStreamBuffer(obtained.first) &&
                        !Core::NativeContent::InRing(vsharp.address, size)) [[unlikely]] {
                        Core::NativeDevice::BufferPart part{};
                        if (!(Core::NativeDevice::BufferPartOf(w31_id, vsharp.address, size, part) && part.begin <= vsharp.address && part.end >= vsharp.address + size)) {
                            Core::RowVersion::NoteStatic(5, size);
                        } else if (u64 hash = 0; !Core::NativeContent::HashGuarded64(vsharp.address, size, hash)) {
                            Core::RowVersion::NoteStatic(6, size);
                        } else {
                            const u64 key = RowKey(w31_id, vsharp.address, size, desc.is_formatted);
                            if (t_static_seen.size() > 400000) {
                                t_static_seen.clear();
                            }
                            auto [it, fresh] = t_static_seen.try_emplace(key);
                            if (fresh) {
                                Core::RowVersion::NoteStatic(0, size);
                            } else {
                                const bool same_version = it->second.version == rows_version_pre;
                                const bool same_bytes = it->second.hash == hash;
                                Core::RowVersion::NoteStatic(same_version ? (same_bytes ? 1u : 2u) : (same_bytes ? 3u : 4u), size);
                                if (same_version && !same_bytes) {
                                    // (each id once, up to 64: which buffers change without an announcement)
                                    static thread_local std::unordered_map<u64, u64> silent_ids;
                                    if (++silent_ids[w31_id] == 1 && silent_ids.size() <= 64) {
                                        LOG_ERROR(Render_Vulkan, "static-row census: id {:#x} {:#x}+{:#x}: version unchanged, bytes changed since the previous copy (new id {}); owner {}, open Map over it {}, "
                                                  "buffer part {:#x}..{:#x}, immutable hint {} usage {}, compute {}, texel {}",
                                                  w31_id, vsharp.address, size, silent_ids.size(), Core::NativeDevice::OwnerName(vsharp.address),
                                                  Core::NativeUpdates::OpenMapOverlaps(vsharp.address, size), part.begin, part.end, immutable_hint != nullptr,
                                                  immutable_hint != nullptr ? immutable_hint->usage : 0u, stage.sw_stage == Shader::SwStage::Compute, desc.is_formatted);
                                    }
                                    if (Core::RowVersion::Watched(vsharp.address, size) && silent_ids[w31_id] % 25 == 1) {
                                        Core::RingMirror::ProbeWrites(vsharp.address); // diagnostics: the next write into this page records its writer
                                    }
                                    if (silent_ids[w31_id] % 200 == 0 && silent_ids[w31_id] <= 1000) {
                                        LOG_ERROR(Render_Vulkan, "static-row census: id {:#x} changed silently {} times; open Map over it now {}", w31_id, silent_ids[w31_id],
                                                  Core::NativeUpdates::OpenMapOverlaps(vsharp.address, size));
                                    }
                                }
                            }
                            it->second = {rows_version_pre, hash};
                        }
                    }
                    if (Core::RowVersion::WalkCensusOn() && w31_id != 0) [[unlikely]] {
                        const bool in_ring = Core::NativeContent::InRing(vsharp.address, size);
                        Core::RowVersion::NoteOldAnswer(buffer_cache.IsHookRing(obtained.first) ? 0u : buffer_cache.IsStreamBuffer(obtained.first) ? (in_ring ? 1u : 2u) : in_ring ? 3u : 4u, size);
                    }
                    if (!buffer_cache.IsHookRing(obtained.first)) { // a hook ring answer moves with the ring's laps: never learned
                        RowsLearn(w31_id, vsharp.address, size, desc.is_formatted, obtained.first, obtained.second, rows_epoch_pre, rows_clock_pre, rows_gpu_pre, rows_version_pre, rows_unattr_pre);
                    }
                }
                if (t_ahead_sample != nullptr) [[unlikely]] {
                    // rows ahead real: the sampled binding's full answer against the planned row
                    const ResolvedBuffer* a = t_ahead_sample;
                    t_ahead_sample = nullptr;
                    auto& rc = rows_ahead_real[stage.sw_stage == Shader::SwStage::Compute ? 1 : 0];
                    const bool same = a->ahead_id == w31_id && a->ahead_address == vsharp.address && a->ahead_size == size && obtained.first == a->plan_buffer &&
                                      obtained.second == a->plan_offset;
                    rc[same ? 3 : 2].fetch_add(1, std::memory_order_relaxed);
                    if (!same) {
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                            LOG_ERROR(Render_Vulkan, "rows ahead real: sampled binding id {:#x} {:#x}+{:#x}: planned row {:p}+{:#x} (id {:#x} {:#x}+{:#x}), full path {:p}+{:#x} (row served {})", w31_id,
                                      vsharp.address, size, a->plan_buffer, a->plan_offset, a->ahead_id, a->ahead_address, a->ahead_size, static_cast<const void*>(obtained.first),
                                      obtained.second, row_served);
                        }
                    }
                }
                if (Vulkan::RowShare::Mode() == 1 && w31_id != 0) [[unlikely]] {
                    // rows ahead shadow (design review 55; shadow mode only: beside the real mode its identity walk from old row clocks cost about 3 ms per frame): the row the front planned against the answer the raster thread gave
                    auto& c = rows_ahead_shadow[stage.sw_stage == Shader::SwStage::Compute ? 1 : 0];
                    c[0].fetch_add(1, std::memory_order_relaxed);
                    const StagePlan* ap = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                          : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                      : nullptr;
                    const ResolvedBuffer* a = ap != nullptr && buffer_index < ap->resolved.size() && ap->resolved[buffer_index].ahead_id == w31_id &&
                                                      ap->resolved[buffer_index].ahead_address == vsharp.address && ap->resolved[buffer_index].ahead_size == size
                                                  ? &ap->resolved[buffer_index]
                                                  : nullptr;
                    if (a == nullptr || !a->plan_found) {
                        if (row_served) {
                            c[8].fetch_add(1, std::memory_order_relaxed);
                        }
                    } else {
                        c[1].fetch_add(1, std::memory_order_relaxed);
                        const bool valid = a->plan_epoch == buffer_cache.PlacementEpoch() && a->plan_version == Core::RowVersion::Of(w31_id) &&
                                           (a->plan_flags & Vulkan::RowShare::FlagInside) != 0 &&
                                           Core::NativeIdentity::ChangedSince(a->plan_clock, a->ahead_address, a->ahead_address + a->ahead_size) == Core::NativeIdentity::Change::None;
                        if (!valid) {
                            c[7].fetch_add(1, std::memory_order_relaxed);
                        } else {
                            c[2].fetch_add(1, std::memory_order_relaxed);
                            const bool same = obtained.first == a->plan_buffer && obtained.second == a->plan_offset;
                            c[row_served ? (same ? 3 : 4) : (same ? 5 : 6)].fetch_add(1, std::memory_order_relaxed);
                            if (row_served && !same) {
                                static std::atomic<u32> logged{0};
                                if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                                    LOG_ERROR(Render_Vulkan, "rows ahead shadow: id {:#x} {:#x}+{:#x}: planned row {:p}+{:#x}, the raster's row {:p}+{:#x} (written {} texel {} static {})", w31_id,
                                              vsharp.address, size, a->plan_buffer, a->plan_offset, static_cast<const void*>(obtained.first), obtained.second, desc.is_written,
                                              desc.is_formatted, (a->plan_flags & Vulkan::RowShare::FlagStatic) != 0);
                                }
                            }
                            if ((a->plan_flags & Vulkan::RowShare::FlagStatic) != 0) {
                                c[9].fetch_add(1, std::memory_order_relaxed);
                            }
                            if (desc.is_written) {
                                c[10].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }
                if (cover_on) [[unlikely]] {
                    w3_row = row_served;
                    w3_b = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                    t_bseg[4] = w3_b;
                }
                const auto [buffer, offset] = obtained;
                Core::NativeContent::ClearBindContext();
                recipe_ranges.emplace_back(vsharp.address, static_cast<u32>(std::min<u64>(size, 0xffffffffull)));
                if (desc.is_written || size > 262144) {
                    recipe_reusable = false;
                }
                if (Core::LightRecorder::On() && !Core::LightRecorder::Recording() && draw_class_now == 0 && size <= 1024 && !desc.is_written) {
                    Core::LightRecorder::ShadowBuffer(Core::DebugServer::PresentCount(), vsharp.address, size); // a z / shadow draw's view constants candidate
                }
                if (Core::LightRecorder::On() && Core::LightRecorder::Recording()) {
                    Core::LightRecorder::Entry note{};
                    note.kind = 1;
                    note.via = immutable_hint != nullptr ? 1 : 0;
                    note.written = desc.is_written ? 1 : 0;
                    note.id = record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0;
                    note.address = vsharp.address;
                    note.size = size;
                    u64 h = 0xcbf29ce484222325ull;
                    const u64 hashed = std::min<u64>(size, 2048); // a light constant block is below 2 KB (more costs too much with the copy dispatches recorded too)
                    const u64 bytes = memory->IsValidMapping(vsharp.address, hashed) ? hashed : 0;
                    for (u64 b = 0; b < bytes; b += 4) {
                        u32 word = 0;
                        std::memcpy(&word, reinterpret_cast<const void*>(vsharp.address + b), std::min<u64>(4, bytes - b));
                        if (b < 704) {
                            note.words[b / 4] = word;
                        }
                        h = (h ^ word) * 0x100000001b3ull;
                    }
                    note.hash = h;
                    Core::LightRecorder::Note(note);
                }
                if (Core::NativeRegistry::AuditBuffer(vsharp.address, size, desc.is_written, desc.post_op == Shader::SharpFetchPostOp::OffsetByProgramBase)) {
                    {
                        // where did the shader get this descriptor from, and did the record write the registers it sits in?
                        thread_local u32 site_tick = 0;
                        if ((site_tick++ & 3) == 0) {
                            const u32 offset = desc.sharp_fetch.offsets[0];
                            const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
                            u32 set_by_record = 2, table_described = 2;
                            if (static_cast<u32>(desc.sharp_fetch.summary) != 0) {
                                const auto flat = stage.FlatUserData();
                                if (offset + 2 <= flat.size()) {
                                    const u64 table = u64{flat[offset]} | (u64{flat[offset + 1] & 0xfff} << 32);
                                    table_described = Core::NativeRegistry::DrawDescribes(seq, table, 1) ? 1 : 0;
                                }
                            } else if (offset < 16 && Core::NativeRegistry::CurrentDrawFromPlan()) {
                                std::array<u32, 16> rvs{}, rps{};
                                const u32 masks = Core::NativeDraw::RecordUserData(seq, rvs.data(), rps.data());
                                const u32 mask = stage.sw_stage == Shader::SwStage::Fragment ? masks >> 16 : masks & 0xffff;
                                set_by_record = (mask >> offset & 1) != 0 ? 1 : 0;
                            }
                            // where is the buffer relative to the program of the stage (shader constants sit next to the code), and which post operation builds the descriptor
                            const u64 program = stage.ProgramBase();
                            const u64 distance = vsharp.address >= program ? vsharp.address - program : program - vsharp.address;
                            const u32 below = vsharp.address >= program ? 0u : 1u;
                            const u32 bucket = distance == 0 ? 0u : static_cast<u32>(std::bit_width(distance));
                            Core::NativeRegistry::NoteUnexplainedSite(static_cast<u32>(stage.sw_stage), static_cast<u32>(desc.sharp_fetch.summary) != 0 ? 1u : 0u, offset, set_by_record, table_described,
                                                                      (static_cast<u32>(desc.post_op) << 8) | (below << 7) | std::min(bucket, 63u));
                        }
                    }
                    // the draw's description did not name this buffer: where did the shader's descriptor come from (user data register, flattened table offset)?
                    static std::atomic<u64> details{0};
                    static std::atomic<u32> logged_details{0};
                    if (details.fetch_add(1) % 60000 == 0 && logged_details.fetch_add(1) < 30) {
                        std::string text = fmt::format("unexplained buffer {:#x}+{} stage {} pgm {:#x} plan_from_record {} index {} sharp_fetch summary {} load_mask {:#x} flat_size {} offsets:", vsharp.address,
                                                       size, static_cast<u32>(stage.sw_stage), stage.pgm_hash, Core::NativeRegistry::CurrentDrawFromPlan(), buffer_index,
                                                       static_cast<u32>(desc.sharp_fetch.summary), desc.sharp_fetch.load_mask, stage.FlatUserData().size());
                        for (u32 k = 0; k < 4; ++k) {
                            const u32 offset = desc.sharp_fetch.offsets[k];
                            text += fmt::format(" {}:{:08x}", offset, offset < stage.FlatUserData().size() ? stage.FlatUserData()[offset] : 0u);
                        }
                        text += " user_data:";
                        for (const u32 word : stage.UserData()) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " flat:";
                        for (const u32 word : stage.FlatUserData()) {
                            text += fmt::format(" {:x}", word);
                        }
                        u32 record_registers = 0;
                        const u32 at = Core::NativeDraw::RecordRegisterWith(Core::NativeRegistry::CurrentDrawSeq(), static_cast<u32>(vsharp.address), &record_registers);
                        text += fmt::format(" | record has base word at register {:#x} of {}", at, record_registers);
                        std::array<u32, 16> rvs{}, rps{};
                        const u32 masks = Core::NativeDraw::RecordUserData(Core::NativeRegistry::CurrentDrawSeq(), rvs.data(), rps.data());
                        text += fmt::format(" | record user data masks vs {:#x} ps {:#x}: vs", masks & 0xffff, masks >> 16);
                        for (const u32 word : rvs) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " ps";
                        for (const u32 word : rps) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " | " + Core::NativeRegistry::DescribeCurrentDraw();
                        Core::NativeRegistry::NoteDebug(text);
                    }
                }
                if (cover_on) [[unlikely]] {
                    t_bseg[5] = BsegNow();
                }
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, src_access);
                if (cover_on && W1ShadowOn()) [[unlikely]] {
                    W31After(buffer, offset, vsharp.address, stage.sw_stage == Shader::SwStage::Compute, desc.is_written);
                }
                if (cover_on && stage.sw_stage == Shader::SwStage::Compute) [[unlikely]] {
                    t_w3.upload += VideoCore::BufferCache::UploadBytes();
                    t_w3.with_upload += VideoCore::BufferCache::UploadBytes() != 0 ? 1u : 0u;
                    t_w3.written += desc.is_written ? 1u : 0u;
                    t_w3.stream += buffer_cache.IsStreamBuffer(buffer) ? 1u : 0u;
                    const u32 why = immutable_hint != nullptr ? 0u
                                    : binder_plan == nullptr ? 1u
                                    : desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad ? 2u
                                    : record_binding == ~0u ? 3u
                                    : !binder_plan->has_handles ? 4u
                                    : binder_plan->handles[record_binding] == nullptr ? 5u
                                                                                       : 6u;
                    ++t_w3.why[why & 7];
                    if (desc.is_written) {
                        ++t_w3.written_id[(why == 0 || why >= 5) ? 1 : 0];
                    }
                    if (W1ShadowOn() && !buffer_cache.IsStreamBuffer(buffer)) {
                        const u64 row_id = immutable_hint != nullptr ? immutable_hint->id : record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0;
                        W1ShadowRow(row_id, vsharp.address, buffer, offset, desc.is_written ? 84u : 80u);
                    }
                }
                if (cover_on && W1ShadowOn() && stage.sw_stage != Shader::SwStage::Compute) [[unlikely]] {
                    const u32 why = immutable_hint != nullptr ? 0u
                                    : binder_plan == nullptr ? 1u
                                    : desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad ? 2u
                                    : record_binding == ~0u ? 3u
                                    : !binder_plan->has_handles ? 4u
                                    : binder_plan->handles[record_binding] == nullptr ? 5u
                                                                                       : 6u;
                    W1ShadowBinding(immutable_hint, record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0, vsharp.address, buffer, offset, desc.is_written, why, desc.is_formatted);
                }
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    if (size >= (8u << 20)) {
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1) < 6) {
                            LOG_INFO(Render_Vulkan, "stage {:#x} (user data {}) writes {} bytes at {:#x} through a storage buffer, caller {:#x}",
                                     stage.pgm_hash, stage.user_data.size(), size, vsharp.address,
                                     dispatch_caller ? dispatch_caller - Core::NativeDraw::ImageBase() + 0x1000000 : 0);
                        }
                    }
                    InvalidateWrittenBuffer(record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0, vsharp.address, size, false);
                }
                Census(census_base + CBufferBarrierQuery);
                if (BufferBarrierQuery(buffer, offset, size, desc.is_written)) {
                    Census(census_base + CBufferBarrierTrue);
                    needs_barrier = true;
                }
                }
            }
        }

    buffer_bound:
        if (own_compare != nullptr) [[unlikely]] {
            // design review 83: the own memory's answer against the old path's
            namespace OM = Core::OwnMemoryCensus;
            const auto& info = buffer_infos.back();
            const u32 cls = own_compare_cls;
            if (buffer_cache.IsStreamHandle(info.buffer)) {
                OM::NoteOwn(OM::OwnOldStream, cls);
            } else {
                OM::NoteOwn(OM::OwnCompared, cls);
                const u64 aligned = Common::AlignDown(own_compare_offset, alignment);
                const u64 adjust = own_compare_offset - aligned;
                if (info.buffer != own_compare->Handle() || info.offset != aligned || info.range != own_compare_size + adjust || push_data.buf_offsets[own_compare_slot] != adjust) {
                    OM::NoteOwn(OM::OwnDifferent, cls);
                    static std::atomic<u32> logged{0};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 20) {
                        LOG_ERROR(Render_Vulkan, "own memory differs (class {}): stage {:#x} buffer {}: own {:#x}+{:#x}/{:#x} adjust {}, old {:#x}+{:#x}/{:#x} adjust {}", cls, stage.pgm_hash, buffer_index,
                                  reinterpret_cast<u64>(static_cast<VkBuffer>(own_compare->Handle())), aligned, own_compare_size + adjust, adjust,
                                  reinterpret_cast<u64>(static_cast<VkBuffer>(info.buffer)), info.offset, info.range, push_data.buf_offsets[own_compare_slot]);
                    }
                } else if (VideoCore::BufferCache::UploadBytes() != 0) {
                    OM::NoteOwn(OM::OwnOldPathWork, cls);
                }
            }
            own_compare = nullptr;
        }
        if (t_fseg_on) [[unlikely]] {
            t_fseg[5] = BsegNow();
        }
        if (cover_on) [[unlikely]] {
            const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
            const u32 slot = (census_base != 0 ? 128u : 0u) + (cover_cause & 15u) * 8u + (t_cover_kind & 7u);
            cover_buf_n[slot].fetch_add(1, std::memory_order_relaxed);
            cover_buf_ns[slot].fetch_add(now - cover_t0, std::memory_order_relaxed);
            if (t_cover_kind == 4 && t_bseg[1] != 0 && t_bseg[2] != 0 && t_bseg[3] != 0 && t_bseg[4] != 0 && t_bseg[5] != 0) {
                const u64 marks[7] = {cover_t0, t_bseg[1], t_bseg[2], t_bseg[3], t_bseg[4], t_bseg[5], now};
                auto& seg = (t_bseg_served ? cover_handle_seg_served : cover_handle_seg)[census_base != 0 ? 1 : 0];
                for (u32 k = 0; k < 6; ++k) {
                    seg[k].fetch_add(marks[k + 1] - marks[k], std::memory_order_relaxed);
                }
                seg[6].fetch_add(1, std::memory_order_relaxed);
            }
            if (census_base != 0 && w3_a != 0 && w3_b != 0) {
                t_w3.buf[0] += w3_a - cover_t0;
                t_w3.buf[1] += w3_b - w3_a;
                if (desc.is_written) {
                    t_w3.wr_ns += w3_b - w3_a;
                    ++t_w3.wr_n;
                } else if (w3_row) {
                    t_w3.rd_row_ns += w3_b - w3_a;
                    ++t_w3.rd_row_n;
                } else {
                    t_w3.rd_ns += w3_b - w3_a;
                    ++t_w3.rd_n;
                }
                t_w3.buf[2] += now - w3_b;
                ++t_w3.buf_n;
            }
            cover_buf_sync[slot].fetch_add(::VideoCore::t_cover_sync_ns - cover_sync0, std::memory_order_relaxed);
            cover_buf_copies[slot].fetch_add(::VideoCore::t_cover_sync_copies - cover_copies0, std::memory_order_relaxed);
            if (cover_cause == 0 && t_cover_kind == 4) {
                const u32 code = (desc.is_written ? 1u : 0u) | (desc.is_formatted ? 2u : 0u) | (desc.post_op == Shader::SharpFetchPostOp::OffsetByProgramBase ? 4u : 0u) |
                                 (desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad ? 8u : 0u);
                const u32 s = (census_base != 0 ? 16u : 0u) + code;
                cover_skip_n[s].fetch_add(1, std::memory_order_relaxed);
                cover_skip_ns[s].fetch_add(now - cover_t0, std::memory_order_relaxed);
                cover_skip_sync[s].fetch_add(::VideoCore::t_cover_sync_ns - cover_sync0, std::memory_order_relaxed);
            }
        }
        if (indexed_active && stage.indexed && Shader::Indexed::IsIndexedBufferType(desc.buffer_type)) {
            auto* indexed = IndexedData::Get();
            const auto& info = buffer_infos.back();
            const u64 made_before = indexed->stats.buffer_new.load(std::memory_order_relaxed);
            const u32 slot = indexed->BufferSlot(info.buffer, info.offset, info.range);
            if (indexed->stats.buffer_new.load(std::memory_order_relaxed) != made_before) {
                // R4 census: which buffers make new slots (0 stream ring, 1 an arena, 2 other)
                const bool stream = buffer_cache.IsStreamHandle(info.buffer);
                slot_census[stream ? 0 : 2].fetch_add(1, std::memory_order_relaxed);
            } else {
                slot_census[3].fetch_add(1, std::memory_order_relaxed);
            }
            NoteIndexedRow(slot);
            if (verify_slots_on) {
                verify_slots.push_back(slot);
            }
            if (t_fseg_on) [[unlikely]] {
                NoteFastSegments(census_base != 0 ? 1 : 0, true);
            }
            ++binding.buffer;
            continue;
        }
        if (indexed_active && stage.indexed) {
            auto& st = IndexedData::Get()->stats;
            (desc.buffer_type == Shader::BufferType::Flatbuf ? st.legacy_flat : desc.buffer_type == Shader::BufferType::ClipPlanes ? st.legacy_clip : st.legacy_lds).fetch_add(1, std::memory_order_relaxed);
        }
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        if (t_fseg_on) [[unlikely]] {
            NoteFastSegments(census_base != 0 ? 1 : 0, false);
        }
        ++binding.buffer;
    }
}

bool Rasterizer::BufferBarrierQuery(const VideoCore::Buffer* buffer, u64 offset, u64 size, bool written) {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_BARRIER_GUARD"); // default 2 (real) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    if (mode == 0 || written) {
        return runtime.IsBufferAccessed(buffer, offset, size, written);
    }
    const bool no_writes = !runtime.PlanHasWrites();
    if (mode == 1) {
        const bool asked = runtime.IsBufferAccessed(buffer, offset, size, false);
        barrier_guard[0].fetch_add(1, std::memory_order_relaxed);
        if (no_writes) {
            barrier_guard[1].fetch_add(1, std::memory_order_relaxed);
            if (asked) {
                barrier_guard[2].fetch_add(1, std::memory_order_relaxed);
            }
        }
        return asked;
    }
    if (!no_writes) {
        barrier_guard[3].fetch_add(1, std::memory_order_relaxed);
        return runtime.IsBufferAccessed(buffer, offset, size, false);
    }
    static thread_local u32 tick = 0;
    if ((++tick & 63u) == 0) {
        barrier_guard[5].fetch_add(1, std::memory_order_relaxed);
        const bool asked = runtime.IsBufferAccessed(buffer, offset, size, false);
        if (asked) {
            barrier_guard[6].fetch_add(1, std::memory_order_relaxed);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
                LOG_ERROR(Render_Vulkan, "barrier guard: a read with no planned write still needed a barrier ({:p}+{:#x}, {:#x} bytes)", static_cast<const void*>(buffer), offset, size);
            }
        }
        return asked;
    }
    barrier_guard[4].fetch_add(1, std::memory_order_relaxed);
    return false;
}

void Rasterizer::NoteFastSegments(u32 side, bool indexed) noexcept {
    // design review 66: 0 validity (plan, epoch, version, identity walk), 1 bookkeeping (clock, census, writer, written effects, recipe), 2 the binding written, 3 the barrier query,
    // 4 the jump and the cover census at buffer_bound, 5 the loop tail (the indexed data slot or the descriptor write)
    t_fseg[6] = BsegNow();
    t_fseg_on = false;
    auto& seg = rows_ahead_fseg[side];
    for (u32 k = 0; k < 6; ++k) {
        seg[k].fetch_add(t_fseg[k + 1] - t_fseg[k], std::memory_order_relaxed);
    }
    seg[6].fetch_add(1, std::memory_order_relaxed);
    seg[7].fetch_add(indexed ? 1 : 0, std::memory_order_relaxed);
}

u32 Rasterizer::BufferSlotCounted(const vk::DescriptorBufferInfo& info) {
    auto* indexed = IndexedData::Get();
    const u64 made_before = indexed->stats.buffer_new.load(std::memory_order_relaxed);
    const u32 slot = indexed->BufferSlot(info.buffer, info.offset, info.range);
    if (indexed->stats.buffer_new.load(std::memory_order_relaxed) != made_before) {
        // R4 census: which buffers make new slots (0 stream ring, 1 an arena, 2 other)
        const bool stream = buffer_cache.IsStreamHandle(info.buffer);
        slot_census[stream ? 0 : 2].fetch_add(1, std::memory_order_relaxed);
    } else {
        slot_census[3].fetch_add(1, std::memory_order_relaxed);
    }
    return slot;
}

void Rasterizer::ResolveBufferBinding(const Shader::Info& stage, u32 buffer_index, u64 alignment, u32 census_base, BufferBind& out) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    ResolveBufferBindingBody(stage, buffer_index, alignment, census_base, out);
    // the descriptor slot of a buffer that is already final (not a stream ring copy made by the record pass): the slot table is state of the resolve side
    if (out.kind == BufferBind::Info && indexed_active && stage.indexed && Shader::Indexed::IsIndexedBufferType(stage.buffers[buffer_index].buffer_type)) {
        out.slot = BufferSlotCounted(out.info);
        out.has_slot = true;
    }
}

void Rasterizer::ResolveBufferBindingBody(const Shader::Info& stage, u32 buffer_index, u64 alignment, u32 census_base, BufferBind& out) {
    // Plan S step P1: the resolve half of one buffer binding. Decides what is bound (ring row, special buffer, resolved object, in place, ObtainBuffer) and fills `out`;
    // it does not touch the descriptor writes, the push data offsets, the stream ring rows or the indexed descriptor set (the record half does).
    const auto& desc = stage.buffers[buffer_index];
    Census(census_base + CBufferTotal);
    const auto src_access =
        vk::AccessFlagBits2::eShaderRead |
        (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        // TLG_RING_KIND (bisect): 1 draws only, 2 dispatches only, 0 both
        static const u32 ring_kind = [] { const char* e = std::getenv("TLG_RING_KIND"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }(); // default 0 (draws and dispatches) since 2026-11-07, owner play good; 1 draws only
        // kind 3: draws and the dispatches whose stage is not indexed
        const bool dispatch_ok = ring_use.dispatch && (ring_kind == 0 || ring_kind == 2 || (ring_kind == 3 && !(indexed_active && stage.indexed)));
        const RingBinding* ring = const_ring_level >= 4 && ((ring_use.draw && ring_kind != 2) || dispatch_ok) ? FindRingBinding(stage, buffer_index) : nullptr;
        if (ring != nullptr && desc.IsSpecial() && indexed_active && stage.indexed) {
            // an indexed stage has no binding for its flat user data buffer (the shader reads the descriptor words from the row): a ring copy of it would write a binding the layout does not have
            ring = nullptr;
        }
        if (ring != nullptr && ring->formatted) {
            // N5f: a texel buffer may be the memory of an image (a surface a shader reads as a buffer, HTile, CMask): only the raster thread knows
            if (texture_cache.AnyImageOverlaps(ring->address, ring->size) || texture_cache.IsMeta(ring->address)) {
                Core::NativeUploads::NoteRingPrecise(5);
                ring = nullptr;
            } else if (buffer_cache.IsRangeGpuWritten(ring->address, ring->size)) {
                Core::NativeUploads::NoteRingPrecise(6);
                ring = nullptr;
            } else {
                Core::NativeUploads::NoteRingPrecise(4);
            }
        }
        if (ring) {
            Census(census_base + CBufferRing);
            // Copied by the command processor thread into the constant ring (read-only, small, not written by queued work).
            if (ring->hook && hook_ring_buffer) {
                // F1 1a: bound straight from the hook ring (see the other binder)
                const u64 aligned = Common::AlignDown(ring->offset, alignment);
                const u64 adjust = ring->offset - aligned;
                out.has_offset = true; out.adjust = static_cast<u32>(adjust);
                out.info = vk::DescriptorBufferInfo(hook_ring_buffer->Handle(), aligned, ring->size + adjust);
            } else {
                if (!desc.IsSpecial()) {
                    out.has_offset = true; out.adjust = static_cast<u32>(0);
                }
                out.info = vk::DescriptorBufferInfo(constant_ring->Handle(), ring->offset, ring->size);
            }
            ring_use.bound.fetch_add(1, std::memory_order_relaxed);
            static const bool ring_verify = [] { const char* e = std::getenv("TLG_RING_VERIFY"); return e != nullptr && e[0] == '1'; }();
            if (ring_verify && !desc.IsSpecial()) {
                // bisect: the descriptor the raster thread's own path would choose against the one the command processor copied for
                AmdGpu::Buffer own_sharp;
                bool own_record = false;
                if (binder_plan != nullptr && desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
                    u32 record_binding = ~0u;
                    if (const u32* words = BinderWords(stage, desc.sharp_fetch.offsets[0], 4, 1, &record_binding)) {
                        own_sharp = desc.FromWords(words, stage);
                        own_record = true;
                    }
                }
                if (!own_record) {
                    own_sharp = desc.GetSharp(stage);
                }
                const Core::NativeDesc::BufferDesc own = Core::NativeDesc::BufferDescOf(own_sharp);
                static std::atomic<u64> sharp_checked[2]{}, sharp_differ[2]{};
                static std::atomic<u32> sharp_logged{0};
                const u32 kind = ring_use.dispatch ? 1u : 0u;
                sharp_checked[kind].fetch_add(1, std::memory_order_relaxed);
                if (own.address != ring->address || own.size != ring->size) {
                    sharp_differ[kind].fetch_add(1, std::memory_order_relaxed);
                    if (sharp_logged.fetch_add(1) < 40) {
                        LOG_WARNING(Render_Vulkan, "ring verify sharp: {} stage {:#x} buffer {} ring {:#x}+{:#x} own {:#x}+{:#x} (from record {}); differ {}/{} of {}/{} (draw/dispatch)", kind ? "dispatch" : "draw",
                                    stage.pgm_hash, buffer_index, ring->address, ring->size, own.address, own.size, own_record, sharp_differ[0].load(), sharp_differ[1].load(),
                                    sharp_checked[0].load(), sharp_checked[1].load());
                    }
                }
            }
            if (ring_verify && !ring->hook && ring->address != 0 && memory->IsValidMapping(ring->address, ring->size)) { // (a hook ring binding is checked by TLG_HOOK_RING itself)
                // bisect: the ring copy (made by the command processor) against guest memory now (what the raster thread's own copy would read)
                std::vector<u8> now(ring->size);
                memory->CopySparseMemory(ring->address, now.data(), ring->size);
                static std::atomic<u64> checked[2]{}, differ[2]{};
                static std::atomic<u32> logged{0};
                const u32 kind = ring_use.dispatch ? 1u : 0u;
                const u64 n_checked = checked[kind].fetch_add(1, std::memory_order_relaxed) + 1;
                if ((n_checked & (n_checked - 1)) == 0) {
                    LOG_WARNING(Render_Vulkan, "ring verify: {} checked {} differ {}", kind ? "dispatch" : "draw", n_checked, differ[kind].load());
                }
                if (std::memcmp(now.data(), constant_ring->Data(ring->offset), ring->size) != 0) {
                    differ[kind].fetch_add(1, std::memory_order_relaxed);
                    if (logged.fetch_add(1) < 40) {
                        size_t first = 0;
                        while (first < ring->size && now[first] == constant_ring->Data(ring->offset)[first]) {
                            ++first;
                        }
                        LOG_WARNING(Render_Vulkan, "ring verify: {} stage {:#x} buffer {} addr {:#x}+{:#x} differs at +{:#x} (ring {:#x}, memory {:#x}); checked {}/{} differ {}/{} (draw/dispatch)",
                                    kind ? "dispatch" : "draw", stage.pgm_hash, buffer_index, ring->address, ring->size, first, constant_ring->Data(ring->offset)[first], now[first],
                                    checked[0].load(), checked[1].load(), differ[0].load(), differ[1].load());
                    }
                }
            }
            if (Core::NativeUploads::RingPrecise() && ring->address != 0 && buffer_cache.IsRangeGpuWritten(ring->address, ring->size)) {
                Core::NativeUploads::NoteRingPrecise(1); // the GPU wrote bytes of this range after the command processor copied it
            }
        } else if (desc.IsSpecial()) {
            Census(census_base + CBufferSpecial);
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                out.info = vk::DescriptorBufferInfo(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                Core::ResourceAudit::Note(Core::ResourceAudit::BufferUse, 0xfffff0000000ull, gds_buf->SizeBytes(), desc.is_written);
                out.SetQuery(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written, false);
                out.SetBound(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                out.flat.assign(stage.flattened_ud_buf.begin(), stage.flattened_ud_buf.end()); // P3e.3: a copy; the shader info is rewritten by the next draw's plan
                if (indexed_active && stage.indexed) {
                    out.kind = BufferBind::Skip; // the record pass notes the flat user data of an indexed stage
                    return;
                }
                out.kind = BufferBind::FlatCopy; // the record pass copies the flattened user data into the stream ring
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (clip_plane_mask == 0) {
                    out.info = vk::DescriptorBufferInfo(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    out.kind = BufferBind::ClipCopy;
                    auto& planes = out.planes;
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = draw_inputs->clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                }
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                if (indexed_active && stage.indexed) {
                    Core::RareEvents::Note("c9_shared_memory", fmt::format("compute shader {:#x}: the shared memory is a storage buffer with a slot of its own per dispatch", stage.pgm_hash));
                }
                const auto& cs_program = *current_cs;
                out.kind = BufferBind::LdsMap; // the record pass maps and clears the stream ring rows
                out.lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            bool cp_no_object = false;
            if (const StagePlan* ns = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr) {
                cp_no_object = buffer_index < ns->resolved.size() && ns->resolved[buffer_index].no_object && ns->resolved_epoch == Core::NativeResources::ImmutableEpoch();
            }
            if (const StagePlan* rs = resolved_draw != nullptr && static_cast<u32>(stage.sw_stage) < MaxShaderStages ? &resolved_draw->stages[static_cast<u32>(stage.sw_stage)]
                                      : resolved_dispatch != nullptr && stage.sw_stage == Shader::SwStage::Compute ? &resolved_dispatch->stage
                                                                                                                  : nullptr;
                rs != nullptr && buffer_index < rs->resolved.size() && rs->resolved[buffer_index].object != nullptr) {
                // R2: resolved by the command processor from the record's engine object; bound when its stamps hold
                const ResolvedBuffer& r = rs->resolved[buffer_index];
                if (rs->resolved_epoch == Core::NativeResources::ImmutableEpoch()) {
                    const auto* object = r.object;
                    u64 copy_offset = 0;
                    if (const auto* copy = object->usage != 1 && object->alive.load(std::memory_order_acquire) && object->generation.load(std::memory_order_acquire) == r.generation
                                               ? buffer_cache.ObjectCopy(*object, r.address, copy_offset)
                                               : nullptr) {
                        // R1b: the object's own copy of this generation in the stream ring
                        const u64 offset_aligned = Common::AlignDown(copy_offset, alignment);
                        const u64 adjust = copy_offset - offset_aligned;
                        out.has_offset = true; out.adjust = static_cast<u32>(adjust);
                        out.info = vk::DescriptorBufferInfo(copy->Handle(), offset_aligned, r.size + adjust);
                        Census(census_base + CBufferResolved);
                        resolve_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        recipe_ranges.emplace_back(r.address, r.size);
                        return;
                    }
                    const bool ready = object->alive.load(std::memory_order_acquire) && object->synced.load(std::memory_order_acquire); // the arena fields are stored before `synced`
                    const auto* arena = ready ? static_cast<const VideoCore::Buffer*>(object->resident_buffer.load(std::memory_order_relaxed)) : nullptr;
                    if (ready && object->generation.load(std::memory_order_acquire) == r.generation && arena != nullptr &&
                        object->resident_epoch.load(std::memory_order_acquire) == buffer_cache.PlacementEpoch() &&
                        (object->usage == 1 || !buffer_cache.IsRegionCpuModified(r.address, r.size))) { // as ObtainBuffer: the tracker's dirty bit is the safety net of a resident buffer
                        const u64 offset = r.address - object->resident_base.load(std::memory_order_relaxed);
                        const u64 offset_aligned = Common::AlignDown(offset, alignment);
                        const u64 adjust = offset - offset_aligned;
                        out.has_offset = true; out.adjust = static_cast<u32>(adjust);
                        out.info = vk::DescriptorBufferInfo(arena->Handle(), offset_aligned, r.size + adjust);
                        out.SetBound(arena, offset, r.size, src_access);
                        out.SetQuery(arena, offset, r.size, false, true); // asked by the record pass: only when the plan has writes
                        Census(census_base + CBufferResolved);
                        resolve_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        recipe_ranges.emplace_back(r.address, r.size);
                        return;
                    }
                    // why: 0 epoch, 1 gone, 2 not synchronized, 3 generation moved, 4 arena, 5 tracker dirty
                    const u32 why = !object->alive.load(std::memory_order_relaxed) ? 1 : !object->synced.load(std::memory_order_relaxed) ? 2
                                    : object->generation.load(std::memory_order_relaxed) != r.generation                       ? 3
                                    : arena == nullptr || object->resident_epoch.load(std::memory_order_relaxed) != buffer_cache.PlacementEpoch() ? 4 : 5;
                    resolve_stats.stale_why[why].fetch_add(1, std::memory_order_relaxed);
                } else {
                    resolve_stats.stale_why[0].fetch_add(1, std::memory_order_relaxed);
                }
                resolve_stats.stale.fetch_add(1, std::memory_order_relaxed);
            }
            AmdGpu::Buffer chosen_sharp;
            bool from_record = false;
            u32 record_binding = ~0u;
            if (binder_plan != nullptr && desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad) {
                if (const u32* words = BinderWords(stage, desc.sharp_fetch.offsets[0], 4, 1, &record_binding)) {
                    chosen_sharp = desc.FromWords(words, stage);
                    from_record = true;
                }
            }
            if (from_record) {
                Core::NativeRegistry::NoteBinder(0);
            } else if (StageFlatSkipped(stage) && desc.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Buffer>::Summary::MultiLoad) {
                Core::NativeRegistry::NoteBinder(6); // must not happen: the plan skipped the flat user data for a stage the record covers
                if (stage.sw_stage == Shader::SwStage::Compute) {
                    DumpDispatchRecord(stage, 1, desc.sharp_fetch.offsets[0], "no record entry: buffer");
                }
                chosen_sharp = AmdGpu::Buffer::Null();
            } else {
                chosen_sharp = desc.GetSharp(stage);
                if (desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Buffer>::Summary::MultiLoad && desc.sharp_fetch.load_mask == 0) {
                    Core::NativeRegistry::NoteBinder(12); // F5: built from the shader's immediates, GetSharp reads no flat dword (not a flat user data read)
                } else {
                    if (binder_plan != nullptr) {
                        Core::NativeRegistry::NoteBinder(2);
                    }
                }
            }
            // F4: from here on the buffer is a plain description (the descriptor type stays at the recompiler's interface above)
            const Core::NativeDesc::BufferDesc vsharp = Core::NativeDesc::BufferDescOf(chosen_sharp);
            if (vsharp.address == 0 || vsharp.size == 0) {
                Census(census_base + CBufferNull);
                out.info = vk::DescriptorBufferInfo(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else if (vsharp.size >= (1ull << 30) && !memory->IsValidMapping(vsharp.address)) {
                LOG_ERROR(Render, "buffer sharp at unmapped address {:#x} of {} bytes for stage {:#x}: bound as null", vsharp.address,
                          vsharp.size, stage.pgm_hash);
                out.info = vk::DescriptorBufferInfo(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.address, vsharp.size);
                if (size != vsharp.size) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.size, size, stage.pgm_hash);
                }
                // N7.5: a read-only buffer of guest memory bound where it is (imported view of the runtime's backing), no copy, no dirty tracking
                bool in_place = false;
                if (inplace_on) {
                    inplace_offered.fetch_add(1, std::memory_order_relaxed);
                    if (desc.is_written) {
                        inplace_written.fetch_add(1, std::memory_order_relaxed);
                    } else if (size > inplace_cap) {
                        inplace_too_large.fetch_add(1, std::memory_order_relaxed);
                    } else if (buffer_cache.IsRangeGpuWritten(vsharp.address, size)) {
                        inplace_gpu_written.fetch_add(1, std::memory_order_relaxed);
                    } else if (desc.is_formatted && (texture_cache.AnyImageOverlaps(vsharp.address, size) || texture_cache.IsMeta(vsharp.address))) {
                        inplace_image_or_meta.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        const auto window = host_import->Find(vsharp.address, size);
                        const u64 aligned = Common::AlignDown(window.offset, alignment);
                        if (window.buffer && window.offset - aligned + size <= window.chunk_remaining + (window.offset - aligned)) {
                            in_place = true;
                            Census(census_base + CBufferInPlace);
                            if (cover_on) [[unlikely]] {
                                t_w1.bad |= W1BadInPlace;
                            }
                            const u64 adjust = window.offset - aligned;
                            out.has_offset = true; out.adjust = static_cast<u32>(adjust);
                            out.info = vk::DescriptorBufferInfo(window.buffer, aligned, size + adjust);
                            inplace_taken.fetch_add(1, std::memory_order_relaxed);
                            inplace_bytes.fetch_add(size, std::memory_order_relaxed);
                        } else {
                            inplace_not_in_backing.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                if (!in_place) {
                // N3c: the immutable engine buffer named by the record's binding (by the engine's id), when the plan resolved one and nothing was registered since
                Core::NativeResources::ImmutableBuffer* immutable_hint = nullptr;
                if (record_binding != ~0u && binder_plan->has_handles && (binder_plan->handles[record_binding] != nullptr)) {
                    if (binder_plan->handle_epoch == Core::NativeResources::ImmutableEpoch()) {
                        immutable_hint = binder_plan->handles[record_binding];
                    } else {
                        Core::NativeResources::NoteHandle(1);
                        // (the life census that read the plan's handle here is gone: the handle was made before a create / delete, and the object it names may already have left the
                        // graveyard, so its fields cannot be read; a crash in bloom, 2026-11-09, `alive.load` of a freed object)
                    }
                }
                if (immutable_hint != nullptr) {
                    Census(census_base + CBufferHandle);
                } else if (desc.is_written) {
                    Census(census_base + CBufferAddrWritten);
                } else {
                    Census(census_base + CBufferAddrRead);
                    Census(census_base + (size <= 256 ? CBufferAddrReadLe256 : size <= 4096 ? CBufferAddrReadLe4k : size <= 65536 ? CBufferAddrReadLe64k : CBufferAddrReadBig));
                }
                Core::NativeRegistry::SetBufferWriter(stage.pgm_hash, static_cast<u32>(stage.sw_stage));
                Core::NativeContent::SetBindContext({"shader binding", record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0,
                                                     record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].view_id : 0, stage.pgm_hash,
                                                     static_cast<u32>(stage.sw_stage), immutable_hint != nullptr, desc.is_written, vsharp.address, size});
                // D4 sizing (2026-11-07): the same call, timed by what kind of binding it serves
                const auto [buffer, offset] = [&] {
                    VideoCore::BufferCache::UploadBytesReset();
                    if (immutable_hint != nullptr) {
                        TLG_PROF_LEVEL("bind.obtain_by_handle", 3);
                        if (stage.sw_stage == Shader::SwStage::Compute) {
                            if (desc.is_written) { TLG_PROF_COUNT("bind.handle.compute.written"); } else if (desc.is_formatted) { TLG_PROF_COUNT("bind.handle.compute.texel"); } else { TLG_PROF_COUNT("bind.handle.compute.plain"); }
                        } else {
                            if (desc.is_written) { TLG_PROF_COUNT("bind.handle.graphics.written"); } else if (desc.is_formatted) { TLG_PROF_COUNT("bind.handle.graphics.texel"); } else { TLG_PROF_COUNT("bind.handle.graphics.plain"); }
                        }
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_handle", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_handle", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    } else if (desc.is_written) {
                        TLG_PROF_LEVEL("bind.obtain_by_address_written", 3);
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_addr_written", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_addr_written", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    } else {
                        TLG_PROF_LEVEL("bind.obtain_by_address_read", 3);
                        const auto r = buffer_cache.ObtainBuffer(vsharp.address, size, desc.is_written, desc.is_formatted, immutable_hint, NativeHandleMode() == 2, cp_no_object);
                        if (stage.sw_stage == Shader::SwStage::Compute) { TLG_PROF_ADD("bind.uploaded_bytes_compute_addr_read", VideoCore::BufferCache::UploadBytes()); } else { TLG_PROF_ADD("bind.uploaded_bytes_graphics_addr_read", VideoCore::BufferCache::UploadBytes()); }
                        return r;
                    }
                }();
                Core::NativeContent::ClearBindContext();
                recipe_ranges.emplace_back(vsharp.address, static_cast<u32>(std::min<u64>(size, 0xffffffffull)));
                if (desc.is_written || size > 262144) {
                    recipe_reusable = false;
                }
                if (Core::LightRecorder::On() && !Core::LightRecorder::Recording() && draw_class_now == 0 && size <= 1024 && !desc.is_written) {
                    Core::LightRecorder::ShadowBuffer(Core::DebugServer::PresentCount(), vsharp.address, size); // a z / shadow draw's view constants candidate
                }
                if (Core::LightRecorder::On() && Core::LightRecorder::Recording()) {
                    Core::LightRecorder::Entry note{};
                    note.kind = 1;
                    note.via = immutable_hint != nullptr ? 1 : 0;
                    note.written = desc.is_written ? 1 : 0;
                    note.id = record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0;
                    note.address = vsharp.address;
                    note.size = size;
                    u64 h = 0xcbf29ce484222325ull;
                    const u64 hashed = std::min<u64>(size, 2048); // a light constant block is below 2 KB (more costs too much with the copy dispatches recorded too)
                    const u64 bytes = memory->IsValidMapping(vsharp.address, hashed) ? hashed : 0;
                    for (u64 b = 0; b < bytes; b += 4) {
                        u32 word = 0;
                        std::memcpy(&word, reinterpret_cast<const void*>(vsharp.address + b), std::min<u64>(4, bytes - b));
                        if (b < 704) {
                            note.words[b / 4] = word;
                        }
                        h = (h ^ word) * 0x100000001b3ull;
                    }
                    note.hash = h;
                    Core::LightRecorder::Note(note);
                }
                if (Core::NativeRegistry::AuditBuffer(vsharp.address, size, desc.is_written, desc.post_op == Shader::SharpFetchPostOp::OffsetByProgramBase)) {
                    {
                        // where did the shader get this descriptor from, and did the record write the registers it sits in?
                        thread_local u32 site_tick = 0;
                        if ((site_tick++ & 3) == 0) {
                            const u32 offset = desc.sharp_fetch.offsets[0];
                            const u64 seq = Core::NativeRegistry::CurrentDrawSeq();
                            u32 set_by_record = 2, table_described = 2;
                            if (static_cast<u32>(desc.sharp_fetch.summary) != 0) {
                                const auto flat = stage.FlatUserData();
                                if (offset + 2 <= flat.size()) {
                                    const u64 table = u64{flat[offset]} | (u64{flat[offset + 1] & 0xfff} << 32);
                                    table_described = Core::NativeRegistry::DrawDescribes(seq, table, 1) ? 1 : 0;
                                }
                            } else if (offset < 16 && Core::NativeRegistry::CurrentDrawFromPlan()) {
                                std::array<u32, 16> rvs{}, rps{};
                                const u32 masks = Core::NativeDraw::RecordUserData(seq, rvs.data(), rps.data());
                                const u32 mask = stage.sw_stage == Shader::SwStage::Fragment ? masks >> 16 : masks & 0xffff;
                                set_by_record = (mask >> offset & 1) != 0 ? 1 : 0;
                            }
                            // where is the buffer relative to the program of the stage (shader constants sit next to the code), and which post operation builds the descriptor
                            const u64 program = stage.ProgramBase();
                            const u64 distance = vsharp.address >= program ? vsharp.address - program : program - vsharp.address;
                            const u32 below = vsharp.address >= program ? 0u : 1u;
                            const u32 bucket = distance == 0 ? 0u : static_cast<u32>(std::bit_width(distance));
                            Core::NativeRegistry::NoteUnexplainedSite(static_cast<u32>(stage.sw_stage), static_cast<u32>(desc.sharp_fetch.summary) != 0 ? 1u : 0u, offset, set_by_record, table_described,
                                                                      (static_cast<u32>(desc.post_op) << 8) | (below << 7) | std::min(bucket, 63u));
                        }
                    }
                    // the draw's description did not name this buffer: where did the shader's descriptor come from (user data register, flattened table offset)?
                    static std::atomic<u64> details{0};
                    static std::atomic<u32> logged_details{0};
                    if (details.fetch_add(1) % 60000 == 0 && logged_details.fetch_add(1) < 30) {
                        std::string text = fmt::format("unexplained buffer {:#x}+{} stage {} pgm {:#x} plan_from_record {} index {} sharp_fetch summary {} load_mask {:#x} flat_size {} offsets:", vsharp.address,
                                                       size, static_cast<u32>(stage.sw_stage), stage.pgm_hash, Core::NativeRegistry::CurrentDrawFromPlan(), buffer_index,
                                                       static_cast<u32>(desc.sharp_fetch.summary), desc.sharp_fetch.load_mask, stage.FlatUserData().size());
                        for (u32 k = 0; k < 4; ++k) {
                            const u32 offset = desc.sharp_fetch.offsets[k];
                            text += fmt::format(" {}:{:08x}", offset, offset < stage.FlatUserData().size() ? stage.FlatUserData()[offset] : 0u);
                        }
                        text += " user_data:";
                        for (const u32 word : stage.UserData()) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " flat:";
                        for (const u32 word : stage.FlatUserData()) {
                            text += fmt::format(" {:x}", word);
                        }
                        u32 record_registers = 0;
                        const u32 at = Core::NativeDraw::RecordRegisterWith(Core::NativeRegistry::CurrentDrawSeq(), static_cast<u32>(vsharp.address), &record_registers);
                        text += fmt::format(" | record has base word at register {:#x} of {}", at, record_registers);
                        std::array<u32, 16> rvs{}, rps{};
                        const u32 masks = Core::NativeDraw::RecordUserData(Core::NativeRegistry::CurrentDrawSeq(), rvs.data(), rps.data());
                        text += fmt::format(" | record user data masks vs {:#x} ps {:#x}: vs", masks & 0xffff, masks >> 16);
                        for (const u32 word : rvs) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " ps";
                        for (const u32 word : rps) {
                            text += fmt::format(" {:x}", word);
                        }
                        text += " | " + Core::NativeRegistry::DescribeCurrentDraw();
                        Core::NativeRegistry::NoteDebug(text);
                    }
                }
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                out.has_offset = true; out.adjust = static_cast<u32>(adjust);
                out.info = vk::DescriptorBufferInfo(buffer->Handle(), offset_aligned, size + adjust);
                out.SetBound(buffer, offset, size, src_access);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    if (size >= (8u << 20)) {
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1) < 6) {
                            LOG_INFO(Render_Vulkan, "stage {:#x} (user data {}) writes {} bytes at {:#x} through a storage buffer, caller {:#x}",
                                     stage.pgm_hash, stage.user_data.size(), size, vsharp.address,
                                     dispatch_caller ? dispatch_caller - Core::NativeDraw::ImageBase() + 0x1000000 : 0);
                        }
                    }
                    InvalidateWrittenBuffer(record_binding != ~0u && binder_plan != nullptr ? binder_plan->bindings[record_binding].id : 0, vsharp.address, size, false);
                }
                out.SetQuery(buffer, offset, size, desc.is_written, false); // the barrier query belongs to the record pass (it reads the barrier state the recording writes)
                out.census_query = true;
                }
            }
        }
}

void Rasterizer::RecordBufferBindings(const Shader::Info& stage, Shader::Backend::Bindings& binding, Shader::PushData& push_data, BufferBind* binds, u32 count, u64 alignment, u32 census_base) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    // Plan S step P1: the record half. Everything here writes the command stream state (descriptor writes, push data offsets, stream ring rows, indexed descriptor set).
    for (u32 buffer_index = 0; buffer_index < count; ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        BufferBind& out = binds[buffer_index];
        if (out.kind == BufferBind::Skip) {
            NoteIndexedFlat(stage, out.flat.data(), static_cast<u32>(out.flat.size()));
            ++binding.buffer;
            continue;
        }
        if (out.kind == BufferBind::FlatCopy) {
            auto& vk_buffer = buffer_cache.GetStreamBuffer();
            const u32 ubo_size = out.flat.size() * sizeof(u32);
            const u64 offset = vk_buffer.Copy(out.flat.data(), ubo_size, alignment);
            out.info = vk::DescriptorBufferInfo(vk_buffer.Handle(), offset, ubo_size);
        } else if (out.kind == BufferBind::ClipCopy) {
            auto& vk_buffer = buffer_cache.GetStreamBuffer();
            const u32 ubo_size = static_cast<u32>(sizeof(out.planes));
            const u64 offset = vk_buffer.Copy(out.planes.data(), ubo_size, alignment);
            out.info = vk::DescriptorBufferInfo(vk_buffer.Handle(), offset, ubo_size);
        } else if (out.kind == BufferBind::LdsMap) {
            auto& lds_buffer = buffer_cache.GetStreamBuffer();
            const auto [data, offset] = lds_buffer.Map(out.lds_size, alignment);
            std::memset(data, 0, out.lds_size);
            lds_buffer.Commit();
            out.info = vk::DescriptorBufferInfo(lds_buffer.Handle(), offset, out.lds_size);
        }
        if (out.has_offset) {
            push_data.AddOffset(binding.buffer, out.adjust);
        }
        buffer_infos.emplace_back(out.info);
        if (out.has_bound) {
            bound_buffers.emplace_back(out.bound_buffer, out.bound_offset, out.bound_size, out.bound_access);
        }
        if (out.has_query) {
            if (out.census_query) {
                Census(census_base + CBufferBarrierQuery);
            }
            if ((!out.query_plan_writes || runtime.PlanHasWrites()) && runtime.IsBufferAccessed(out.query_buffer, out.query_offset, out.query_size, out.query_written)) {
                if (out.census_query) {
                    Census(census_base + CBufferBarrierTrue);
                }
                needs_barrier = true;
            }
        }
        if (record_indexed_active && stage.indexed && Shader::Indexed::IsIndexedBufferType(desc.buffer_type)) {
            // a slot made by the resolve pass, or (a stream ring copy: its offset is only known here) made now
            const u32 slot = out.has_slot ? out.slot : BufferSlotCounted(buffer_infos.back());
            NoteIndexedRow(slot);
            ++binding.buffer;
            continue;
        }
        if (record_indexed_active && stage.indexed) {
            auto& st = IndexedData::Get()->stats;
            (desc.buffer_type == Shader::BufferType::Flatbuf ? st.legacy_flat : desc.buffer_type == Shader::BufferType::ClipPlanes ? st.legacy_clip : st.legacy_lds).fetch_add(1, std::memory_order_relaxed);
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

void Rasterizer::BindBuffersBody(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                                 Shader::PushData& push_data) {
    // Plan S step P1: TLG_SPLIT_BIND 0 = the old single pass (default), 1 = resolve pass then record pass, 2 = old pass, then the resolve pass again and compare (the oracle)
    static const u32 mode = [] { const char* e = std::getenv("TLG_SPLIT_BIND"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 0u; }();
    if (mode == 0) {
        BindBuffersBodyLegacy(stage, binding, push_data);
        return;
    }
    const u64 alignment = instance.StorageMinAlignment();
    const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
    const u32 count = static_cast<u32>(stage.buffers.size());
    if (mode == 2) {
        VerifyBufferBindings(stage, binding, push_data, alignment, census_base, count);
        return;
    }
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_buffers", 2);
    boost::container::small_vector<BufferBind, 16> binds(count);
    // P1d: the synchronization copies decided by the resolve pass are issued by the record pass; what else the resolve pass records is counted
    VideoCore::BufferCache::UploadSink uploads;
    const auto* list = scheduler.CommandBuffer().List();
    const u64 commands_before = list != nullptr ? list->Size() : 0;
    VideoCore::BufferCache::SetUploadSink(&uploads);
    for (u32 i = 0; i < count; ++i) {
        ResolveBufferBinding(stage, i, alignment, census_base, binds[i]);
    }
    VideoCore::BufferCache::SetUploadSink(nullptr);
    {
        static std::atomic<u64> calls{0}, with_commands{0}, commands{0}, deferred{0};
        const u64 n = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        const u64 made = (list != nullptr ? list->Size() : 0) - commands_before;
        deferred.fetch_add(uploads.size(), std::memory_order_relaxed);
        if (made != 0) {
            with_commands.fetch_add(1, std::memory_order_relaxed);
            commands.fetch_add(made, std::memory_order_relaxed);
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind: {} stage calls, {} copies deferred to the record pass, the resolve pass still recorded commands in {} calls ({} commands)", n, deferred.load(),
                        with_commands.load(), commands.load());
        }
    }
    {
        VideoCore::HalfPhase::Scope record_scope{VideoCore::HalfPhase::Record};
        buffer_cache.RunDeferredUploads(uploads);
    }
    RecordBufferBindings(stage, binding, push_data, binds.data(), count, alignment, census_base);
}

void Rasterizer::VerifyBufferBindings(const Shader::Info& stage, Shader::Backend::Bindings& binding, Shader::PushData& push_data, u64 alignment, u32 census_base, u32 count) {
    // Plan S step P1 oracle: run the old pass for real, then the resolve half on the same draw and compare what each decided. The resolve half runs after the whole old pass
    // (the order of the split path: every ObtainBuffer before any barrier query), so a difference can also come from that order. The record half does not run.
    static std::atomic<u64> checked{0}, bindings_checked{0}, differ{0}, extra_barrier{0}, slots_checked{0};
    static std::atomic<u32> logged{0};
    const size_t infos_before = buffer_infos.size();
    const bool barrier_before = needs_barrier;
    const u32 first_binding = binding.buffer;
    needs_barrier = false;
    verify_slots.clear();
    verify_slots_on = true;
    BindBuffersBodyLegacy(stage, binding, push_data);
    verify_slots_on = false;
    const bool legacy_barrier = needs_barrier;
    needs_barrier = barrier_before || legacy_barrier;
    const size_t ranges_after_legacy = recipe_ranges.size();
    const bool reusable_after_legacy = recipe_reusable;
    std::vector<vk::DescriptorBufferInfo> seen(buffer_infos.begin() + infos_before, buffer_infos.end());
    boost::container::small_vector<BufferBind, 16> binds(count);
    for (u32 i = 0; i < count; ++i) {
        ResolveBufferBinding(stage, i, alignment, census_base, binds[i]);
    }
    recipe_ranges.resize(ranges_after_legacy);
    recipe_reusable = reusable_after_legacy;
    const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
    bindings_checked.fetch_add(count, std::memory_order_relaxed);
    u32 diffs = 0;
    bool resolve_barrier = false;
    std::vector<u32> kept;
    for (u32 i = 0; i < count; ++i) {
        if (binds[i].has_query && (!binds[i].query_plan_writes || runtime.PlanHasWrites()) &&
            runtime.IsBufferAccessed(binds[i].query_buffer, binds[i].query_offset, binds[i].query_size, binds[i].query_written)) {
            resolve_barrier = true; // what the record pass would decide, asked here after the whole old pass
        }
        if (binds[i].kind != BufferBind::Skip) {
            kept.push_back(i);
        }
    }
    const auto stream_handle = buffer_cache.GetStreamBuffer().Handle();
    std::string what;
    if (kept.size() != seen.size()) {
        ++diffs;
        what += fmt::format(" entries legacy {} split {};", seen.size(), kept.size());
    } else {
        for (size_t i = 0; i < kept.size(); ++i) {
            const BufferBind& b = binds[kept[i]];
            const vk::DescriptorBufferInfo& a = seen[i];
            if (b.kind != BufferBind::Info) {
                if (a.buffer != stream_handle) {
                    ++diffs;
                    what += fmt::format(" #{} legacy not on the stream ring;", kept[i]);
                }
                continue;
            }
            const bool both_stream = a.buffer == stream_handle && b.info.buffer == stream_handle;
            if (a.buffer != b.info.buffer || a.range != b.info.range || (!both_stream && a.offset != b.info.offset)) {
                ++diffs;
                what += fmt::format(" #{} info legacy {:#x}+{:#x}/{:#x} split {:#x}+{:#x}/{:#x};", kept[i], reinterpret_cast<u64>(static_cast<VkBuffer>(a.buffer)), a.offset, a.range,
                                    reinterpret_cast<u64>(static_cast<VkBuffer>(b.info.buffer)), b.info.offset, b.info.range);
            }
            if (b.has_offset && push_data.buf_offsets[first_binding + kept[i]] != b.adjust) {
                ++diffs;
                what += fmt::format(" #{} push offset legacy {} split {};", kept[i], push_data.buf_offsets[first_binding + kept[i]], b.adjust);
            }
        }
    }
    if (legacy_barrier && !resolve_barrier) {
        ++diffs;
        what += " needs_barrier legacy true split false;";
    } else if (!legacy_barrier && resolve_barrier) {
        extra_barrier.fetch_add(1, std::memory_order_relaxed); // allowed: a query asked later sees more accesses (a barrier too many, never one too few)
    }
    {
        size_t at = 0;
        for (u32 i = 0; i < count; ++i) {
            if (binds[i].kind == BufferBind::Skip || !(indexed_active && stage.indexed && Shader::Indexed::IsIndexedBufferType(stage.buffers[i].buffer_type))) {
                continue;
            }
            if (at >= verify_slots.size()) {
                ++diffs;
                what += " fewer slots noted by the old pass;";
                break;
            }
            if (binds[i].has_slot && binds[i].info.buffer != stream_handle) { // (a stream ring copy is made again by this pass: another offset, another slot)
                slots_checked.fetch_add(1, std::memory_order_relaxed);
                if (binds[i].slot != verify_slots[at]) {
                    ++diffs;
                    what += fmt::format(" #{} slot legacy {} split {};", i, verify_slots[at], binds[i].slot);
                }
            }
            ++at;
        }
    }
    if (diffs != 0) {
        differ.fetch_add(1, std::memory_order_relaxed);
        if (logged.fetch_add(1) < 40) {
            LOG_WARNING(Render_Vulkan, "split bind verify: stage {:#x} {} {} buffers:{} (differing stage calls {} of {})", stage.pgm_hash, stage.sw_stage == Shader::SwStage::Compute ? "compute" : "graphics", count,
                        what, differ.load(), n);
        }
    }
    if ((n & (n - 1)) == 0) {
        LOG_WARNING(Render_Vulkan, "split bind verify: {} stage calls, {} bindings, {} slots compared, {} differ ({} stage calls with a barrier only the later query sees)", n, bindings_checked.load(), slots_checked.load(), differ.load(), extra_barrier.load());
    }
}


namespace {
u64 ImageUseFlags(const Shader::ImageResource& image_desc) {
    return u64(image_desc.is_depth) | u64(image_desc.is_atomic) << 8 | u64(image_desc.is_array) << 16 | u64(image_desc.is_written) << 24 |
           u64(image_desc.is_r128) << 32 | u64(image_desc.constant_mip_index) << 40 | u64(static_cast<u8>(image_desc.mip_fallback_mode)) << 48 |
           u64(static_cast<u8>(image_desc.post_op)) << 56;
}
} // namespace

namespace {
u32 TexsetByIdMode() noexcept {
    static const u32 mode = [] {
        // default 2 (real) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        const char* e = std::getenv("TLG_TEXSET_BY_ID");
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}
} // namespace

Rasterizer::TexsetShadowState& Rasterizer::TexsetShadow() {
    static thread_local TexsetShadowState state; // the slot the by-id check called still right (shadow) and its entries then
    return state;
}

bool Rasterizer::TextureSetGenerationMiss(const TextureSet& set) {
    const u32 mode = TexsetByIdMode();
    if (mode == 0 || set.generation == ~0ull) {
        return false;
    }
    const auto unchanged = [&](VideoCore::ImageId id, u64 uid) {
        return texture_cache.ImageUnchangedSince(id, uid, set.generation, texture_cache.GuestAddress(id), texture_cache.GuestSize(id));
    };
    for (u32 i = 0; i < set.count; ++i) {
        const auto& e = set.entries[i];
        if (e.null_entry) {
            continue;
        }
        if (!unchanged(e.found, e.found_uid) || (e.id != e.found && !unchanged(e.id, e.id_uid))) {
            return false;
        }
    }
    ++texture_set_stats.by_id_predicted;
    if (mode == 2) {
        const_cast<TextureSet&>(set).generation = texture_cache.ImageGeneration(); // (the set's images are what it holds now: its clock moves on)
        ++texture_set_stats.by_id_served;
        return true;
    }
    auto& shadow = TexsetShadow();
    shadow.slot = const_cast<TextureSet*>(&set);
    shadow.count = set.count;
    std::copy(set.entries.begin(), set.entries.begin() + set.count, shadow.entries.begin());
    return false; // shadow: the slow path answers and the new entries are compared
}

void Rasterizer::TextureSetShadowCompare(const TextureSet& set) {
    auto& shadow = TexsetShadow();
    if (shadow.slot != &set) {
        return;
    }
    shadow.slot = nullptr;
    bool same = set.count == shadow.count;
    for (u32 i = 0; same && i < set.count; ++i) {
        const auto& a = set.entries[i];
        const auto& b = shadow.entries[i];
        same = a.null_entry == b.null_entry && (a.null_entry || (a.id == b.id && a.found == b.found && a.view == b.view && a.backing == b.backing));
    }
    if (same) {
        ++texture_set_stats.by_id_same;
    } else {
        ++texture_set_stats.by_id_DIFFERENT;
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
            LOG_ERROR(Render_Vulkan, "texture set by id: the slow path answered differently from a set the by-id check called unchanged ({} images)", set.count);
        }
    }
}

bool Rasterizer::BindTexturesFromSet(const Shader::Info& stage, TextureSet& set, const std::array<u32, 8>* raws, const u64* flags, u32 count,
                                     Shader::Backend::Bindings& binding) {
    if (set.count != count) {
        ++texture_set_stats.no_entry;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (set.entries[i].flags != flags[i] || set.entries[i].raw != raws[i]) {
            ++texture_set_stats.no_entry;
            return false;
        }
    }
    if (set.generation != texture_cache.ImageGeneration() && !TextureSetGenerationMiss(set)) {
        ++texture_set_stats.generation;
        return false;
    }
    if (!BindTexturesFromEntries(stage, set, count, binding)) {
        ++texture_set_stats.image_check;
        return false;
    }
    ++texture_set_stats.hits;
    return true;
}

bool Rasterizer::BindTexturesFromEntries(const Shader::Info& stage, const TextureSet& set, u32 count, Shader::Backend::Bindings& binding) {
    // Everything the slow path does to an image before it looks at the view: touch and refresh. Idempotent, so a miss after it costs nothing.
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (entry.null_entry) {
            continue;
        }
        texture_cache.TouchFoundImage(entry.found);
        texture_cache.UpdateImage(entry.id, Core::ImageCaller::RasterizerBindTextures);
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
            return false;
        }
    }
    const u32 first_image_idx = image_infos.size();
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (entry.null_entry) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        if (recipe_active) {
            recipe_image_ranges.emplace_back(image.info.guest_address, static_cast<u32>(std::min<u64>(image.info.guest_size, 0xffffffffull)));
        }
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        const auto new_layout = image.info.props.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, entry.range);
        image.usage.texture |= 1u;
        image_infos.emplace_back(VK_NULL_HANDLE, entry.view, runtime.CurrentLayout(image));
    }
    for (u32 i = 0; i < count; ++i) {
        if (indexed_active && stage.indexed && Shader::Indexed::IsIndexedImage(stage.images[i])) {
            NoteIndexedImage(image_infos[first_image_idx + i], stage.images[i].array_key);
            continue;
        }
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

void Rasterizer::BindTexturesLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_textures", 2);
    static const bool tex_census_on = [] { const char* e = std::getenv("TLG_TEXTURE_CENSUS"); return e != nullptr && e[0] == '1'; }();
    struct TexCensusScope {
        Rasterizer* r;
        u64 t0;
        u32 cls;
        u32 side;
        u32 images;
        ~TexCensusScope() {
            if (t0 != 0) {
                auto& c = r->tex_census[side][cls < 10 ? cls : 9];
                c[0].fetch_add(1, std::memory_order_relaxed);
                c[1].fetch_add(BsegNow() - t0, std::memory_order_relaxed);
                c[2].fetch_add(images, std::memory_order_relaxed);
            }
        }
    } tex_census{this, tex_census_on ? BsegNow() : 0, 9, stage.sw_stage == Shader::SwStage::Compute ? 1u : 0u, static_cast<u32>(stage.images.size())};
    if (tex_census_on) [[unlikely]] {
        if (stage.images.empty()) {
            tex_census.cls = 0;
        } else if (stage.images.size() > TextureSet::Limit()) {
            tex_census.cls = 1;
        } else if (!texture_set_on) {
            tex_census.cls = 2;
        } else {
            bool written = false, fallback = false;
            for (const auto& image_desc : stage.images) {
                written |= image_desc.is_written;
                fallback |= image_desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
            }
            tex_census.cls = written ? 3 : fallback ? 4 : 9;
        }
    }
    const u32 first_image_idx = image_infos.size();
    const size_t view_table_bound_before = bound_images.size(); // note 95 shadow
    const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    // Texture set memo: the set of resolved images for these T#s, when the stage qualifies (no written image, no mip fallback binding, at most 16 images).
    if (Core::ViewTable::Mode() == 2 && ViewTableBind(stage, binding)) {
        Census(census_base + CImageTotal, static_cast<u32>(stage.images.size()));
        BindSamplers(stage, binding);
        return;
    }
    // note 96/97: the old path's answer per image binding, kept for the view table (`vt_ok`: the answer can be bound from the table)
    static thread_local std::array<TextureSetEntry, TextureSet::MaxImages> vt_answers;
    static thread_local std::array<bool, TextureSet::MaxImages> vt_ok;
    const bool vt_on = Core::ViewTable::Mode() != 0;
    if (vt_on) {
        vt_ok.fill(false);
    }
    TextureSet* set_slot = nullptr;
    std::array<std::array<u32, 8>, TextureSet::MaxImages> set_raws;
    std::array<u64, TextureSet::MaxImages> set_flags{};
    std::array<VideoCore::ImageId, TextureSet::MaxImages> set_found{};
    bool set_record = false;
    u64 set_generation = 0;
    const u32 stage_images = static_cast<u32>(stage.images.size());
    if (texture_set_on && stage_images != 0 && stage_images <= TextureSet::Limit()) {
        bool eligible = true;
        for (const auto& image_desc : stage.images) {
            eligible &= !image_desc.is_written && image_desc.mip_fallback_mode == Shader::MipStorageFallbackMode::None;
        }
        if (eligible) {
            TLG_PROF_NAMED_LEVEL(key_scope, "texture.set_key", 3);
            u64 key = 0x9E3779B97F4A7C15ull;
            const auto flat = stage.FlatUserData();
            for (u32 i = 0; i < stage_images; ++i) {
                AmdGpu::Image raw{};
                const auto& res = stage.images[i];
                const u32* record_words = binder_plan != nullptr && res.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad
                                              ? BinderWords(stage, res.sharp_fetch.offsets[0], res.is_r128 ? 4 : 8, 2)
                                              : nullptr;
                if (record_words != nullptr) {
                    res.RawFromWords(record_words, &raw);
                } else if (StageFlatSkipped(stage)) {
                    Core::NativeRegistry::NoteBinder(6); // must not happen
                    raw = {};
                } else if (!res.Fetch(flat.data(), &raw)) {
                    raw = {};
                }
                static_assert(sizeof(AmdGpu::Image) == sizeof(set_raws[0]));
                std::memcpy(set_raws[i].data(), &raw, sizeof(raw));
                set_flags[i] = ImageUseFlags(stage.images[i]);
                std::array<u64, 5> words{};
                std::memcpy(words.data(), &raw, sizeof(AmdGpu::Image));
                words[4] = set_flags[i];
                for (const u64 word : words) {
                    key = (key ^ word) * 0xC2B2AE3D27D4EB4Full;
                    key ^= key >> 29;
                }
            }
            key_scope.End();
            set_slot = &texture_sets[key % TextureSetSlots];
            TLG_PROF_NAMED_LEVEL(try_scope, "texture.set_try", 3);
            const TextureSetStats census_before = texture_set_stats;
            if (BindTexturesFromSet(stage, *set_slot, set_raws.data(), set_flags.data(), stage_images, binding)) {
                tex_census.cls = 5;
                try_scope.End();
                TLG_PROF_COUNT("texture.set_hit");
                Census(census_base + CImageTotal, stage_images);
                Census(census_base + CImageSetHit, stage_images);
                if (vt_on) [[unlikely]] {
                    vt_ok.fill(true); // (a recorded set holds no target, storage or forced-general image)
                    ViewTableShadow(stage, first_image_idx, view_table_bound_before, set_slot->entries.data(), vt_ok.data());
                }
                BindSamplers(stage, binding);
                return;
            }
            try_scope.End();
            tex_census.cls = texture_set_stats.no_entry != census_before.no_entry ? 6 : texture_set_stats.generation != census_before.generation ? 7
                             : texture_set_stats.image_check != census_before.image_check ? 8 : 9;
            TLG_PROF_COUNT("texture.set_miss");
            set_record = true;
            set_slot->count = 0; // the entries are rewritten below: no hit on a half written set
            set_slot->generation = ~0ull;
            set_generation = texture_cache.ImageGeneration();
        } else {
            ++texture_set_stats.not_eligible;
        }
    }

    TLG_PROF_LEVEL("texture.slow_path", 3);
    u32 num_images{};
    std::array<u32, Shader::NUM_IMAGES> image_keys{}; // tlg C9: the descriptor array of each image of the stage
    for (const auto& image_desc : stage.images) {
        AmdGpu::Image chosen_tsharp;
        bool image_from_record = false;
        u64 image_view_id = 0;   // E2: the engine's id of the view object the descriptor was built from
        u64 image_engine_id = 0; // L1(b): the engine's id of the view object the record's binding names
        if (binder_plan != nullptr && image_desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad) {
            u32 identity_binding = ~0u;
            const u32* planned_words = nullptr;
            const u32 image_index = static_cast<u32>(&image_desc - stage.images.data());
            if (!PlannedImageWords(stage, image_index, image_desc.is_r128 ? 4 : 8, planned_words, identity_binding)) {
                planned_words = BinderWords(stage, image_desc.sharp_fetch.offsets[0], image_desc.is_r128 ? 4 : 8, 2, &identity_binding);
            }
            if (const u32* words = planned_words) {
                if (identity_binding != ~0u) {
                    image_engine_id = binder_plan->bindings[identity_binding].id;
                    image_view_id = binder_plan->bindings[identity_binding].view_id;
                }
                chosen_tsharp = image_desc.FromWords(words);
                image_from_record = true;
            }
        }
        if (image_from_record) {
            Core::NativeRegistry::NoteBinder(1);
        } else if (StageFlatSkipped(stage)) {
            Core::NativeRegistry::NoteBinder(6); // must not happen
            if (stage.sw_stage == Shader::SwStage::Compute) {
                DumpDispatchRecord(stage, 2, image_desc.sharp_fetch.offsets[0], "no record entry: image");
            }
            chosen_tsharp = AmdGpu::Image::Null(image_desc.is_depth);
        } else {
            chosen_tsharp = image_desc.GetSharp(stage);
            if (binder_plan != nullptr) {
                Core::NativeRegistry::NoteBinder(3);
            }
        }
        const auto tsharp = chosen_tsharp;
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (indexed_active && stage.indexed) {
            const u32 desc_index = static_cast<u32>(&image_desc - stage.images.data());
            image_keys[desc_index] = image_desc.array_key;
            auto* indexed = IndexedData::Get();
        }
        Census(census_base + CImageTotal);
        if (tsharp.Address() == 0 || data_fmt == Ps4Layout::DataFormat::FormatInvalid) {
            Census(census_base + CImageNull);
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        Core::NativeRegistry::AuditImage(tsharp.Address());
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
        // F4.3: the host infos of this binding come from the engine's plain description when it has one (the T# stays for the shader specialization and the lookup key)
        Core::NativeDesc::ImageDesc native_image_desc;
        const bool from_desc = image_from_record && image_engine_id != 0 && !image_desc.is_r128 && Core::NativeImageDesc::FromDescOn() &&
                               Core::NativeImageDesc::DescForBinding(image_engine_id, image_view_id, image_desc.post_op == Shader::SharpFetchPostOp::ConvertCubeTo2DArray, image_desc.is_depth, native_image_desc);
        // the plain description every host consumer below reads: the engine's, or (descriptors the engine has no object for) the one decode of the T# words
        const Core::NativeDesc::ImageDesc binding_image = from_desc ? native_image_desc : VideoCore::ImageDescFromSharp(tsharp);
        const auto make_image_desc = [&]() { return VideoCore::TextureCache::ImageDesc(binding_image, image_desc); };

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            // Same T# and use as an earlier lookup, no image registered or removed since: the description and the
            // image FindImage returned are the same again (native_draw_mode >= 2).
            TextureKey key{binding_image};
            key.flags = u64(image_desc.is_depth) | u64(image_desc.is_atomic) << 8 |
                     u64(image_desc.is_array) << 16 | u64(image_desc.is_written) << 24 |
                     u64(image_desc.is_r128) << 32 | u64(image_desc.constant_mip_index) << 40 |
                     u64(static_cast<u8>(image_desc.mip_fallback_mode)) << 48 |
                     u64(static_cast<u8>(image_desc.post_op)) << 56;
            key.index = u64(i) | (image_from_record ? image_engine_id << 16 : 0);
            const u64 hash = Core::NativeDesc::HashImageDesc(binding_image, key.flags * 0x9E3779B97F4A7C15ull ^ key.index);
            auto& cached = texture_lookup_cache[hash & 8191];
            const bool native_reuse =
                Core::NativeConfig::DrawMode >=
                2;
            // the mip fallback of the shader's view (applied to a fresh description before FindImage)
            const auto finish_desc = [&](VideoCore::TextureCache::ImageDesc& d) {
                d.engine_id = image_engine_id; // E3 census: the engine object the record names
                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    d.view_info.range.base.level += image_desc.constant_mip_index;
                    d.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    d.view_info.range.base.level += i;
                    d.view_info.range.extent.levels = 1;
                }
            };
            const u64 generation_now = texture_cache.ImageGeneration();
            // L1(b): the same description and engine object as an earlier lookup, and the generation moved: the answer is still the answer when its image is registered and no registration or
            // unregistration touched the range since (the texture cache's event ring says so exactly); the image is found by what it is, not by "nothing changed anywhere"
            const bool same_generation = native_reuse && cached.valid && cached.key == key && cached.generation == generation_now;
            const bool by_id = native_reuse && !same_generation && cached.valid && cached.key == key && TextureIdsOn() &&
                               texture_cache.ImageUnchangedSince(cached.image_id, cached.image_uid, cached.generation, cached.desc.info.guest_address, cached.desc.info.guest_size);
            if (native_reuse && cached.valid && cached.key == key && !same_generation && !by_id && TextureIdsOn()) {
            }
            // B3: the image of an engine object is found by the object's id (and the binding's use), not by its address; the T# the record holds for the object is the
            // guard that the object still describes the same image
            NativeImage* native_entry = nullptr;
            bool native_hit = false;
            if (B3ImagesOn() && image_from_record && image_engine_id != 0) {
                native_entry = &native_images[(image_engine_id << 8) ^ (key.flags * 0x9E3779B97F4A7C15ull) ^ i];
                native_hit = native_entry->valid && native_entry->id == image_engine_id && native_entry->flags == key.flags && native_entry->index == i &&
                             native_entry->image_key == binding_image && texture_cache.ImageUid(native_entry->image_id) == native_entry->image_uid &&
                             True(texture_cache.GetImage(native_entry->image_id).flags & VideoCore::ImageFlagBits::Registered);
                b3_stats[native_hit ? 0 : native_entry->valid ? 2 : 1].fetch_add(1, std::memory_order_relaxed);
            }
            if (native_hit) {
                std::construct_at(&desc, native_entry->desc);
                image_id = native_entry->image_id;
                texture_cache.TouchFoundImage(image_id);
            } else if (same_generation || by_id) {
                Census(census_base + CImageLookupHit);
                std::construct_at(&desc, cached.desc);
                image_id = cached.image_id;
                texture_cache.TouchFoundImage(image_id);
                if (by_id) {
                    cached.generation = generation_now;
                }
            } else {
                std::construct_at(&desc, make_image_desc());
                finish_desc(desc);

                Census(census_base + CImageFind);
                if (image_engine_id == 0) {
                    Core::NativeUpdates::NoteE3UnnamedReason(image_from_record ? 1u : binder_plan == nullptr ? 2u : 3u);
                }
                image_id = texture_cache.FindImage(desc);
                if (native_reuse) {
                    cached.valid = true;
                    cached.generation = generation_now;
                    cached.image_id = image_id;
                    cached.image_uid = texture_cache.ImageUid(image_id);
                    cached.key = key;
                    cached.desc = desc;
                }
            }
            if (native_entry != nullptr && !native_hit && image_id) {
                if (!native_entry->valid) {
                    native_image_keys[image_engine_id].push_back((image_engine_id << 8) ^ (key.flags * 0x9E3779B97F4A7C15ull) ^ i);
                }
                native_entry->valid = true;
                native_entry->id = image_engine_id;
                native_entry->flags = key.flags;
                native_entry->index = i;
                native_entry->image_key = binding_image;
                native_entry->image_id = image_id;
                native_entry->image_uid = texture_cache.ImageUid(image_id);
                native_entry->desc = desc;
            }
            if (Core::LightRecorder::On() && image_desc.is_written) {
                Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), desc.info.guest_address, desc.info.guest_size, texture_cache.ImageUid(image_id), 3);
            }
            if (Core::LightRecorder::On() && Core::LightRecorder::Recording()) {
                Core::LightRecorder::Entry note{};
                note.kind = 2;
                note.via = native_hit ? 3 : same_generation ? 1 : by_id ? 2 : 0;
                note.id = image_engine_id;
                note.address = desc.info.guest_address;
                note.size = desc.info.guest_size;
                note.slot = image_id.index;
                note.uid = texture_cache.ImageUid(image_id);
                u64 h = 0xcbf29ce484222325ull;
                u32 words[8] = {};
                std::memcpy(words, &tsharp, std::min<size_t>(sizeof(words), sizeof(AmdGpu::Image)));
                for (const u32 word : words) {
                    h = (h ^ word) * 0x100000001b3ull;
                }
                note.hash = h;
                Core::LightRecorder::Note(note);
                if (image_desc.is_written) {
                    Core::LightRecorder::Watch(desc.info.guest_address, desc.info.guest_size);
                }
            }
            if (set_record || (vt_on && num_images - 1 < TextureSet::MaxImages)) {
                set_found[num_images - 1] = image_id;
            }
            auto* image = &texture_cache.GetImage(image_id);
            {
                // Dark / bright frame check (2026-10-29): a sampled texture served by a cache image of another address or size than the texture asked for (an overlap resolved
                // into a view of a bigger image). Inside the frame ring this is how a new small dynamic texture (the R32F 4x4 / 16x16 the CPU writes every frame) can be read
                // from an old image after the ring wrapped. Counted always, logged for the frame ring and the first other cases.
                const bool other_image = image->info.guest_address != desc.info.guest_address || image->info.size.width != desc.info.size.width ||
                                         image->info.size.height != desc.info.size.height;
                if (other_image && !image_desc.is_written) {
                    const auto& ring = VideoCore::UntrackedRing::Get();
                    const bool in_ring = ring.Active() && desc.info.guest_address >= ring.Start() && desc.info.guest_address < ring.End();
                    static std::atomic<u64> served_other{0}, served_other_ring{0};
                    const u64 n = (in_ring ? served_other_ring : served_other).fetch_add(1, std::memory_order_relaxed) + 1;
                    if ((in_ring && (n <= 40 || (n & (n - 1)) == 0)) || (!in_ring && n <= 10)) {
                        LOG_WARNING(Render_Vulkan, "texture served by another image ({} {}): asked {:#x} {}x{} format {}, got image {:#x} {}x{} format {} view level {} layer {}, present {}, program {:#x}, caller {:#x}",
                                    in_ring ? "frame ring" : "other", n, desc.info.guest_address, desc.info.size.width, desc.info.size.height, static_cast<u32>(desc.info.pixel_format), image->info.guest_address,
                                    image->info.size.width, image->info.size.height, static_cast<u32>(image->info.pixel_format), desc.view_info.range.base.level, desc.view_info.range.base.layer,
                                    Core::DebugServer::PresentCount(), stage.pgm_hash, binder_plan != nullptr ? Core::NativeDraw::RecordCallerGhidra(binder_plan->native_seq) : 0);
                    }
                }
            }
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
            bool vt_rebound = false;
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
                set_record = false;
                vt_rebound = true;
            }

            bound_images.emplace_back(image_id);
            const auto transit = [&](auto&&... args) {
                Census(census_base + CImageTransit);
                const bool result = runtime.Transit(std::forward<decltype(args)>(args)...);
                if (result) {
                    Census(census_base + CImageTransitTrue);
                }
                return result;
            };
            if (is_storage) {
                Census(census_base + CImageStorage);
            }

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            vk::ImageLayout transit_layout = vk::ImageLayout::eGeneral; // R3 step 2: the layout the transit leaves the image in
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    transit_layout = vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT;
                    needs_barrier |= transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    transit_layout = new_layout;
                    needs_barrier |= transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (is_storage) {
                Core::UiTrace::NoteGpuWrite("a storage image binding", image.info.guest_address, image.info.guest_size);
            }
            if (Core::UiTrace::Active() && !is_storage && image.info.props.is_block && !image.info.props.is_depth && image.info.guest_size <= (256u << 10) && image.info.resources.layers == 1 &&
                image.info.resources.levels >= 1 && Core::UiTrace::WantsShot(image.info.guest_address, static_cast<u32>(Core::DebugServer::PresentCount()))) {
                // read back the GPU's copy of this picture (raw compressed blocks of mip 0): the freeze report compares it with the guest memory
                auto ref = runtime.GetStagingPool().Request(image.info.guest_size, VideoCore::MemoryType::HostCached, 16, false);
                const vk::BufferImageCopy shot_copy = {
                    .bufferOffset = ref.offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {image.info.size.width, image.info.size.height, 1},
                };
                runtime.DownloadImage(&image, ref.buffer, std::span{&shot_copy, 1});
                needs_barrier |= transit(&image, transit_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                ui_shots_pending.push_back({image.info.guest_address, ref, scheduler.CurrentTick(), static_cast<u32>(Core::DebugServer::PresentCount()),
                                            fmt::format("{} {}x{}", vk::to_string(image.info.pixel_format), image.info.size.width, image.info.size.height)});
            }
            if (Core::UiTrace::Active()) {
                Core::UiTrace::NoteImage(image.info.guest_address, image.info.guest_size,
                                         fmt::format("{} {}x{} {} target={} storage={} bound_layout={}", vk::to_string(image.info.pixel_format), image.info.size.width, image.info.size.height,
                                                     AmdGpu::NameOf(image.info.tile_mode), binding.is_target, is_storage, vk::to_string(transit_layout)));
            }

            vk::ImageLayout bound_layout = transit_layout;
            if (!image.backing || !runtime.PlannerDecidesLayouts()) {
                bound_layout = runtime.CurrentLayout(image);
            }
            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view, bound_layout);
            if (recipe_active) {
                recipe_image_ranges.emplace_back(image.info.guest_address, static_cast<u32>(std::min<u64>(image.info.guest_size, 0xffffffffull)));
            }
            if (vt_on && i < TextureSet::MaxImages) {
                auto& ve = vt_answers[i];
                ve = {};
                ve.id = image_id;
                ve.found = vt_rebound ? image_id : set_found[i];
                ve.found_uid = texture_cache.Uid(ve.found);
                ve.id_uid = texture_cache.Uid(image_id);
                ve.view = *image_view.image_view;
                ve.backing = image.backing;
                ve.range = desc.view_info.range;
                ve.array_key = image_keys[std::min<u32>(i, Shader::NUM_IMAGES - 1)];
                vt_ok[i] = !(binding.force_general || binding.is_target || is_storage || vt_rebound);
            }
            if (set_record && i < TextureSet::MaxImages) {
                if (binding.force_general || binding.is_target || is_storage) {
                    set_record = false;
                } else {
                    auto& entry = set_slot->entries[i];
                    entry.id = image_id;
                    entry.found = set_found[i];
                    entry.found_uid = texture_cache.Uid(set_found[i]);
                    entry.id_uid = texture_cache.Uid(image_id);
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
    u32 image_desc_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (indexed_active && stage.indexed && Shader::Indexed::IsIndexedImage(stage.images[image_desc_idx])) {
            if (stage.images[image_desc_idx].is_atomic) {
                Core::RareEvents::Note("c9_atomic_image", fmt::format("shader {:#x} ({}): atomic image in array key {} (format {})", stage.pgm_hash, static_cast<u32>(stage.sw_stage), image_keys[image_desc_idx],
                                                                       (image_keys[image_desc_idx] - Shader::Indexed::FirstAtomicKey) / 4));
            }
            if (stage.images[image_desc_idx].mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                Core::RareEvents::Note("c9_mip_level_run", fmt::format("shader {:#x} ({}): image bound once per mip level, {} levels, array key {}", stage.pgm_hash, static_cast<u32>(stage.sw_stage), array_size,
                                                                       image_keys[image_desc_idx]));
                NoteIndexedRun(&image_infos[image_info_idx], array_size, image_keys[image_desc_idx]);
            } else {
                NoteIndexedImage(image_infos[image_info_idx], image_keys[image_desc_idx]);
            }
            image_info_idx += array_size;
            image_binding_idx += array_size;
            ++image_desc_idx;
            continue;
        }
        ++image_desc_idx;
        if (indexed_active && stage.indexed) {
            IndexedData::Get()->stats.legacy_images.fetch_add(1, std::memory_order_relaxed);
        }
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
            entry.array_key = image_keys[i];
        }
        set_slot->count = num_images;
        set_slot->generation = set_generation;
        ++texture_set_stats.recorded;
        TextureSetShadowCompare(*set_slot);
    }
    if (vt_on) [[unlikely]] {
        ViewTableShadow(stage, first_image_idx, view_table_bound_before, vt_answers.data(), vt_ok.data());
    }
    BindSamplers(stage, binding);
}

bool Rasterizer::ViewTableBind(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    using namespace Core::ViewTable;
    const u32 n = static_cast<u32>(stage.images.size());
    if (n == 0 || n > TextureSet::MaxImages || binder_plan == nullptr) {
        return false;
    }
    static thread_local TextureSet from_table;
    static_assert(sizeof(TextureSetEntry) <= sizeof(Entry::payload) && std::is_trivially_copyable_v<TextureSetEntry>);
    for (u32 i = 0; i < n; ++i) {
        const auto& res = stage.images[i];
        if (res.is_written || res.mip_fallback_mode != Shader::MipStorageFallbackMode::None || res.sharp_fetch.summary != Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad ||
            res.NumBindings(stage) != 1) {
            return false;
        }
        // the view the record names: an index into the plan, no T# word is read
        u32 identity_binding = ~0u;
        const u32* words = nullptr;
        if (!PlannedImageWords(stage, i, res.is_r128 ? 4 : 8, words, identity_binding)) {
            words = BinderWords(stage, res.sharp_fetch.offsets[0], res.is_r128 ? 4 : 8, 2, &identity_binding);
        }
        if (words == nullptr || identity_binding == ~0u) {
            return false;
        }
        const u64 view_id = binder_plan->bindings[identity_binding].view_id;
        if (view_id == 0) {
            return false;
        }
        const Entry* e = Find(view_id, ImageUseFlags(res));
        if (e == nullptr || !e->in_class || !e->has_payload) {
            return false;
        }
        TextureSetEntry& te = from_table.entries[i];
        std::memcpy(&te, e->payload, sizeof(TextureSetEntry));
        if (te.id_uid == 0 || texture_cache.ImageUid(te.id) != te.id_uid || False(texture_cache.GetImage(te.id).flags & VideoCore::ImageFlagBits::Registered) ||
            (te.found != te.id && texture_cache.ImageUid(te.found) != te.found_uid)) {
            return false;
        }
        te.array_key = res.array_key;
    }
    from_table.count = n;
    static thread_local u32 audit_tick = 0;
    if ((++audit_tick & 63u) == 0) {
        Note(RealAudited); // the old path binds this one; the shadow compare at its end must find the same images
        return false;
    }
    if (!BindTexturesFromEntries(stage, from_table, n, binding)) {
        Note(RealCheckFailed);
        return false;
    }
    Note(RealServed);
    Note(RealServedImages, n);
    return true;
}

void Rasterizer::ViewTableShadow(const Shader::Info& stage, u32 first_info, size_t bound_before, const TextureSetEntry* answers, const bool* answers_ok) {
    using namespace Core::ViewTable;
    const u32 n = static_cast<u32>(stage.images.size());
    if (n == 0) {
        return;
    }
    Note(StagesWithImages);
    if (image_infos.size() - first_info != n) {
        Note(NotClassUse, n); // (an image with several bindings: the infos do not line up one per image; the stage stays on the old path)
        return;
    }
    bool all_in = true, all_served = true, any_diff = false;
    size_t bound_at = bound_before;
    for (u32 i = 0; i < n; ++i) {
        const auto& res = stage.images[i];
        const vk::ImageView view = image_infos[first_info + i].imageView;
        VideoCore::ImageId bound{};
        if (view && bound_at < bound_images.size()) {
            bound = bound_images[bound_at++];
        }
        if (res.is_written || res.mip_fallback_mode != Shader::MipStorageFallbackMode::None || res.NumBindings(stage) != 1) {
            Note(NotClassUse);
            all_in = false;
            continue;
        }
        if (!view || !bound) {
            Note(NotClassNull);
            all_in = false;
            continue;
        }
        // the view the record names (as the slow path finds it)
        u64 view_id = 0;
        if (binder_plan != nullptr && res.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad) {
            u32 identity_binding = ~0u;
            const u32* words = nullptr;
            if (!PlannedImageWords(stage, i, res.is_r128 ? 4 : 8, words, identity_binding)) {
                words = BinderWords(stage, res.sharp_fetch.offsets[0], res.is_r128 ? 4 : 8, 2, &identity_binding);
            }
            if (words != nullptr && identity_binding != ~0u) {
                view_id = binder_plan->bindings[identity_binding].view_id;
            }
        }
        if (view_id == 0) {
            Note(NotClassNoView);
            all_in = false;
            continue;
        }
        const u64 flags = ImageUseFlags(res);
        const u64 view_bits = reinterpret_cast<u64>(static_cast<VkImageView>(view));
        const auto fill = [&](Entry& e) {
            e.image_index = bound.index;
            e.image_uid = texture_cache.ImageUid(bound);
            e.view_handle = view_bits;
            e.has_payload = answers_ok[i] && answers[i].id == bound && answers[i].view == view;
            if (e.has_payload) {
                std::memcpy(e.payload, &answers[i], sizeof(TextureSetEntry));
            }
            if (Core::TextureVersion::Mode() != 0 && e.resource_id != 0) [[unlikely]] {
                // note 100: the image of the view's texture keeps that texture's version slot; an image clean now starts at the current version
                if (auto* slot = Core::TextureVersion::Find(e.resource_id); slot != nullptr && slot->InClass()) {
                    const VideoCore::ImageId own = answers_ok[i] && answers[i].found ? answers[i].found : bound;
                    auto& image = texture_cache.GetImage(own);
                    if (image.tv_slot != slot) {
                        image.tv_slot = slot;
                        image.tv_uploaded = False(image.flags & VideoCore::ImageFlagBits::Dirty) ? slot->version.load(std::memory_order_acquire) : ~0ull;
                        Core::TextureVersion::Note(Core::TextureVersion::ImagesLinked);
                    }
                }
            }
        };
        Entry* e = Find(view_id, flags);
        if (e == nullptr) {
            Entry& made = Put(view_id, flags);
            Core::NativeDevice::Object v{}, t{};
            made.in_class = Core::NativeDevice::Lookup(view_id, v) && v.resource_id != 0 && Core::NativeDevice::Lookup(v.resource_id, t) && t.usage != 2;
            made.resource_id = v.resource_id;
            if (made.resource_id != 0) {
                LinkResource(made.resource_id, view_id);
            }
            fill(made);
            if (!made.in_class) {
                Note(NotClassDynamic);
                all_in = false;
                continue;
            }
            Note(ImagesInClass);
            Note(ImagesNoEntry);
            all_served = false;
            continue;
        }
        if (!e->in_class) {
            Note(NotClassDynamic);
            all_in = false;
            continue;
        }
        Note(ImagesInClass);
        const VideoCore::ImageId entry_image{e->image_index};
        const bool valid = e->image_uid != 0 && texture_cache.ImageUid(entry_image) == e->image_uid &&
                           True(texture_cache.GetImage(entry_image).flags & VideoCore::ImageFlagBits::Registered);
        if (!valid) {
            Note(ImagesStale);
            all_served = false;
            fill(*e);
            continue;
        }
        if (entry_image == bound && e->view_handle == view_bits) {
            Note(ImagesSame);
            continue;
        }
        Note(ImagesDifferent);
        any_diff = true;
        all_served = false;
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
            const auto& old_image = texture_cache.GetImage(bound);
            const auto& table_image = texture_cache.GetImage(entry_image);
            LOG_ERROR(Render_Vulkan, "view table: view {:#x} (use {:#x}) answered image {} {:#x} {}x{} view {:#x}; the old path bound image {} {:#x} {}x{} view {:#x} (program {:#x})", view_id, flags,
                      entry_image.index, table_image.info.guest_address, table_image.info.size.width, table_image.info.size.height, e->view_handle, bound.index,
                      old_image.info.guest_address, old_image.info.size.width, old_image.info.size.height, view_bits, stage.pgm_hash);
        }
        fill(*e);
    }
    if (all_in) {
        Note(StagesAllInClass);
        if (all_served) {
            Note(StagesWouldServe);
        }
    }
    if (any_diff) {
        Note(StagesWithDifference);
    }
}

void Rasterizer::ResolveTextures(const Shader::Info& stage, TextureResolved& r) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2: the resolve half of the texture bindings (T# words to images and views); the record half does the layout transitions and the descriptor writes
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_textures", 2);
    const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    // Texture set memo: the set of resolved images for these T#s, when the stage qualifies (no written image, no mip fallback binding, at most 16 images).
    TextureSet* set_slot = nullptr;
    std::array<std::array<u32, 8>, TextureSet::MaxImages> set_raws;
    std::array<u64, TextureSet::MaxImages> set_flags{};
    std::array<VideoCore::ImageId, TextureSet::MaxImages> set_found{};
    bool set_record = false;
    u64 set_generation = 0;
    const u32 stage_images = static_cast<u32>(stage.images.size());
    if (texture_set_on && stage_images != 0 && stage_images <= TextureSet::Limit()) {
        bool eligible = true;
        for (const auto& image_desc : stage.images) {
            eligible &= !image_desc.is_written && image_desc.mip_fallback_mode == Shader::MipStorageFallbackMode::None;
        }
        if (eligible) {
            TLG_PROF_NAMED_LEVEL(key_scope, "texture.set_key", 3);
            u64 key = 0x9E3779B97F4A7C15ull;
            const auto flat = stage.FlatUserData();
            for (u32 i = 0; i < stage_images; ++i) {
                AmdGpu::Image raw{};
                const auto& res = stage.images[i];
                const u32* record_words = binder_plan != nullptr && res.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad
                                              ? BinderWords(stage, res.sharp_fetch.offsets[0], res.is_r128 ? 4 : 8, 2)
                                              : nullptr;
                if (record_words != nullptr) {
                    res.RawFromWords(record_words, &raw);
                } else if (StageFlatSkipped(stage)) {
                    Core::NativeRegistry::NoteBinder(6); // must not happen
                    raw = {};
                } else if (!res.Fetch(flat.data(), &raw)) {
                    raw = {};
                }
                static_assert(sizeof(AmdGpu::Image) == sizeof(set_raws[0]));
                std::memcpy(set_raws[i].data(), &raw, sizeof(raw));
                set_flags[i] = ImageUseFlags(stage.images[i]);
                std::array<u64, 5> words{};
                std::memcpy(words.data(), &raw, sizeof(AmdGpu::Image));
                words[4] = set_flags[i];
                for (const u64 word : words) {
                    key = (key ^ word) * 0xC2B2AE3D27D4EB4Full;
                    key ^= key >> 29;
                }
            }
            key_scope.End();
            set_slot = &texture_sets[key % TextureSetSlots];
            TLG_PROF_NAMED_LEVEL(try_scope, "texture.set_try", 3);
            if (ResolveTextureSet(stage, *set_slot, set_raws.data(), set_flags.data(), stage_images, r)) {
                try_scope.End();
                TLG_PROF_COUNT("texture.set_hit");
                Census(census_base + CImageTotal, stage_images);
                Census(census_base + CImageSetHit, stage_images);
                r.set_hit = true;
                return;
            }
            try_scope.End();
            TLG_PROF_COUNT("texture.set_miss");
            set_record = true;
            set_slot->count = 0; // the entries are rewritten below: no hit on a half written set
            set_slot->generation = ~0ull;
            set_generation = texture_cache.ImageGeneration();
        } else {
            ++texture_set_stats.not_eligible;
        }
    }

    TLG_PROF_LEVEL("texture.slow_path", 3);
    u32 num_images{};
    std::array<u32, Shader::NUM_IMAGES> image_keys{}; // tlg C9: the descriptor array of each image of the stage
    for (const auto& image_desc : stage.images) {
        AmdGpu::Image chosen_tsharp;
        bool image_from_record = false;
        u64 image_view_id = 0;   // E2: the engine's id of the view object the descriptor was built from
        u64 image_engine_id = 0; // L1(b): the engine's id of the view object the record's binding names
        if (binder_plan != nullptr && image_desc.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad) {
            u32 identity_binding = ~0u;
            const u32* planned_words = nullptr;
            const u32 image_index = static_cast<u32>(&image_desc - stage.images.data());
            if (!PlannedImageWords(stage, image_index, image_desc.is_r128 ? 4 : 8, planned_words, identity_binding)) {
                planned_words = BinderWords(stage, image_desc.sharp_fetch.offsets[0], image_desc.is_r128 ? 4 : 8, 2, &identity_binding);
            }
            if (const u32* words = planned_words) {
                if (identity_binding != ~0u) {
                    image_engine_id = binder_plan->bindings[identity_binding].id;
                    image_view_id = binder_plan->bindings[identity_binding].view_id;
                }
                chosen_tsharp = image_desc.FromWords(words);
                image_from_record = true;
            }
        }
        if (image_from_record) {
            Core::NativeRegistry::NoteBinder(1);
        } else if (StageFlatSkipped(stage)) {
            Core::NativeRegistry::NoteBinder(6); // must not happen
            if (stage.sw_stage == Shader::SwStage::Compute) {
                DumpDispatchRecord(stage, 2, image_desc.sharp_fetch.offsets[0], "no record entry: image");
            }
            chosen_tsharp = AmdGpu::Image::Null(image_desc.is_depth);
        } else {
            chosen_tsharp = image_desc.GetSharp(stage);
            if (binder_plan != nullptr) {
                Core::NativeRegistry::NoteBinder(3);
            }
        }
        const auto tsharp = chosen_tsharp;
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (indexed_active && stage.indexed) {
            const u32 desc_index = static_cast<u32>(&image_desc - stage.images.data());
            image_keys[desc_index] = image_desc.array_key;
            auto* indexed = IndexedData::Get();
        }
        Census(census_base + CImageTotal);
        if (tsharp.Address() == 0 || data_fmt == Ps4Layout::DataFormat::FormatInvalid) {
            Census(census_base + CImageNull);
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        Core::NativeRegistry::AuditImage(tsharp.Address());
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
        // F4.3: the host infos of this binding come from the engine's plain description when it has one (the T# stays for the shader specialization and the lookup key)
        Core::NativeDesc::ImageDesc native_image_desc;
        const bool from_desc = image_from_record && image_engine_id != 0 && !image_desc.is_r128 && Core::NativeImageDesc::FromDescOn() &&
                               Core::NativeImageDesc::DescForBinding(image_engine_id, image_view_id, image_desc.post_op == Shader::SharpFetchPostOp::ConvertCubeTo2DArray, image_desc.is_depth, native_image_desc);
        // the plain description every host consumer below reads: the engine's, or (descriptors the engine has no object for) the one decode of the T# words
        const Core::NativeDesc::ImageDesc binding_image = from_desc ? native_image_desc : VideoCore::ImageDescFromSharp(tsharp);
        const auto make_image_desc = [&]() { return VideoCore::TextureCache::ImageDesc(binding_image, image_desc); };

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            // Same T# and use as an earlier lookup, no image registered or removed since: the description and the
            // image FindImage returned are the same again (native_draw_mode >= 2).
            TextureKey key{binding_image};
            key.flags = u64(image_desc.is_depth) | u64(image_desc.is_atomic) << 8 |
                     u64(image_desc.is_array) << 16 | u64(image_desc.is_written) << 24 |
                     u64(image_desc.is_r128) << 32 | u64(image_desc.constant_mip_index) << 40 |
                     u64(static_cast<u8>(image_desc.mip_fallback_mode)) << 48 |
                     u64(static_cast<u8>(image_desc.post_op)) << 56;
            key.index = u64(i) | (image_from_record ? image_engine_id << 16 : 0);
            const u64 hash = Core::NativeDesc::HashImageDesc(binding_image, key.flags * 0x9E3779B97F4A7C15ull ^ key.index);
            auto& cached = texture_lookup_cache[hash & 8191];
            const bool native_reuse =
                Core::NativeConfig::DrawMode >=
                2;
            // the mip fallback of the shader's view (applied to a fresh description before FindImage)
            const auto finish_desc = [&](VideoCore::TextureCache::ImageDesc& d) {
                d.engine_id = image_engine_id; // E3 census: the engine object the record names
                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    d.view_info.range.base.level += image_desc.constant_mip_index;
                    d.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    d.view_info.range.base.level += i;
                    d.view_info.range.extent.levels = 1;
                }
            };
            const u64 generation_now = texture_cache.ImageGeneration();
            // L1(b): the same description and engine object as an earlier lookup, and the generation moved: the answer is still the answer when its image is registered and no registration or
            // unregistration touched the range since (the texture cache's event ring says so exactly); the image is found by what it is, not by "nothing changed anywhere"
            const bool same_generation = native_reuse && cached.valid && cached.key == key && cached.generation == generation_now;
            const bool by_id = native_reuse && !same_generation && cached.valid && cached.key == key && TextureIdsOn() &&
                               texture_cache.ImageUnchangedSince(cached.image_id, cached.image_uid, cached.generation, cached.desc.info.guest_address, cached.desc.info.guest_size);
            if (native_reuse && cached.valid && cached.key == key && !same_generation && !by_id && TextureIdsOn()) {
            }
            // B3: the image of an engine object is found by the object's id (and the binding's use), not by its address; the T# the record holds for the object is the
            // guard that the object still describes the same image
            NativeImage* native_entry = nullptr;
            bool native_hit = false;
            if (B3ImagesOn() && image_from_record && image_engine_id != 0) {
                native_entry = &native_images[(image_engine_id << 8) ^ (key.flags * 0x9E3779B97F4A7C15ull) ^ i];
                native_hit = native_entry->valid && native_entry->id == image_engine_id && native_entry->flags == key.flags && native_entry->index == i &&
                             native_entry->image_key == binding_image && texture_cache.ImageUid(native_entry->image_id) == native_entry->image_uid &&
                             True(texture_cache.GetImage(native_entry->image_id).flags & VideoCore::ImageFlagBits::Registered);
                b3_stats[native_hit ? 0 : native_entry->valid ? 2 : 1].fetch_add(1, std::memory_order_relaxed);
            }
            if (native_hit) {
                std::construct_at(&desc, native_entry->desc);
                image_id = native_entry->image_id;
                texture_cache.TouchFoundImage(image_id);
            } else if (same_generation || by_id) {
                Census(census_base + CImageLookupHit);
                std::construct_at(&desc, cached.desc);
                image_id = cached.image_id;
                texture_cache.TouchFoundImage(image_id);
                if (by_id) {
                    cached.generation = generation_now;
                }
            } else {
                std::construct_at(&desc, make_image_desc());
                finish_desc(desc);

                Census(census_base + CImageFind);
                if (image_engine_id == 0) {
                    Core::NativeUpdates::NoteE3UnnamedReason(image_from_record ? 1u : binder_plan == nullptr ? 2u : 3u);
                }
                image_id = texture_cache.FindImage(desc);
                if (native_reuse) {
                    cached.valid = true;
                    cached.generation = generation_now;
                    cached.image_id = image_id;
                    cached.image_uid = texture_cache.ImageUid(image_id);
                    cached.key = key;
                    cached.desc = desc;
                }
            }
            if (native_entry != nullptr && !native_hit && image_id) {
                if (!native_entry->valid) {
                    native_image_keys[image_engine_id].push_back((image_engine_id << 8) ^ (key.flags * 0x9E3779B97F4A7C15ull) ^ i);
                }
                native_entry->valid = true;
                native_entry->id = image_engine_id;
                native_entry->flags = key.flags;
                native_entry->index = i;
                native_entry->image_key = binding_image;
                native_entry->image_id = image_id;
                native_entry->image_uid = texture_cache.ImageUid(image_id);
                native_entry->desc = desc;
            }
            if (Core::LightRecorder::On() && image_desc.is_written) {
                Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), desc.info.guest_address, desc.info.guest_size, texture_cache.ImageUid(image_id), 3);
            }
            if (Core::LightRecorder::On() && Core::LightRecorder::Recording()) {
                Core::LightRecorder::Entry note{};
                note.kind = 2;
                note.via = native_hit ? 3 : same_generation ? 1 : by_id ? 2 : 0;
                note.id = image_engine_id;
                note.address = desc.info.guest_address;
                note.size = desc.info.guest_size;
                note.slot = image_id.index;
                note.uid = texture_cache.ImageUid(image_id);
                u64 h = 0xcbf29ce484222325ull;
                u32 words[8] = {};
                std::memcpy(words, &tsharp, std::min<size_t>(sizeof(words), sizeof(AmdGpu::Image)));
                for (const u32 word : words) {
                    h = (h ^ word) * 0x100000001b3ull;
                }
                note.hash = h;
                Core::LightRecorder::Note(note);
                if (image_desc.is_written) {
                    Core::LightRecorder::Watch(desc.info.guest_address, desc.info.guest_size);
                }
            }
            if (set_record) {
                set_found[num_images - 1] = image_id;
            }
            auto* image = &texture_cache.GetImage(image_id);
            {
                // Dark / bright frame check (2026-10-29): a sampled texture served by a cache image of another address or size than the texture asked for (an overlap resolved
                // into a view of a bigger image). Inside the frame ring this is how a new small dynamic texture (the R32F 4x4 / 16x16 the CPU writes every frame) can be read
                // from an old image after the ring wrapped. Counted always, logged for the frame ring and the first other cases.
                const bool other_image = image->info.guest_address != desc.info.guest_address || image->info.size.width != desc.info.size.width ||
                                         image->info.size.height != desc.info.size.height;
                if (other_image && !image_desc.is_written) {
                    const auto& ring = VideoCore::UntrackedRing::Get();
                    const bool in_ring = ring.Active() && desc.info.guest_address >= ring.Start() && desc.info.guest_address < ring.End();
                    static std::atomic<u64> served_other{0}, served_other_ring{0};
                    const u64 n = (in_ring ? served_other_ring : served_other).fetch_add(1, std::memory_order_relaxed) + 1;
                    if ((in_ring && (n <= 40 || (n & (n - 1)) == 0)) || (!in_ring && n <= 10)) {
                        LOG_WARNING(Render_Vulkan, "texture served by another image ({} {}): asked {:#x} {}x{} format {}, got image {:#x} {}x{} format {} view level {} layer {}, present {}, program {:#x}, caller {:#x}",
                                    in_ring ? "frame ring" : "other", n, desc.info.guest_address, desc.info.size.width, desc.info.size.height, static_cast<u32>(desc.info.pixel_format), image->info.guest_address,
                                    image->info.size.width, image->info.size.height, static_cast<u32>(image->info.pixel_format), desc.view_info.range.base.level, desc.view_info.range.base.layer,
                                    Core::DebugServer::PresentCount(), stage.pgm_hash, binder_plan != nullptr ? Core::NativeDraw::RecordCallerGhidra(binder_plan->native_seq) : 0);
                    }
                }
            }
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

    // Second pass to re-bind images that were updated after binding, and the view of each image
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        TextureBind& tb = r.binds.emplace_back();
        tb.is_storage = is_storage; // (a null entry keeps the type the old code read from the binding slot)
        if (!image_id) {
            tb.null_entry = true;
            continue;
        }
        if (auto& old_image = texture_cache.GetImage(image_id); old_image.binding.needs_rebind) {
            old_image.binding = {};
            image_id = texture_cache.FindImage(desc);
            set_record = false;
        }
        bound_images.emplace_back(image_id);
        auto& image = texture_cache.GetImage(image_id);
        auto& image_view = texture_cache.FindTexture(image_id, desc);
        const auto bind_flags = image.binding;
        tb.image_id = image_id;
        tb.image = &image;
        tb.view = *image_view.image_view;
        tb.range = desc.view_info.range;
        tb.is_storage = is_storage;
        tb.is_depth = image.info.props.is_depth;
        tb.is_target = bind_flags.is_target;
        tb.general = (bind_flags.force_general || bind_flags.is_target) && !image.info.props.is_depth;
        tb.feedback = tb.general && instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target;
        if (set_record && i < TextureSet::MaxImages) {
            if (bind_flags.force_general || bind_flags.is_target || is_storage) {
                set_record = false;
            } else {
                auto& entry = set_slot->entries[i];
                entry.id = image_id;
                entry.found = set_found[i];
                entry.found_uid = texture_cache.Uid(set_found[i]);
                entry.id_uid = texture_cache.Uid(image_id);
                entry.view = *image_view.image_view;
                entry.backing = image.backing;
                entry.range = desc.view_info.range;
                entry.null_entry = false;
            }
        }
    }
    r.array_sizes = image_descriptor_array_sizes;
    r.image_keys = image_keys;
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
            entry.array_key = image_keys[i];
        }
        set_slot->count = num_images;
        set_slot->generation = set_generation;
        ++texture_set_stats.recorded;
        TextureSetShadowCompare(*set_slot);
    }
}

bool Rasterizer::ResolveTextureSet(const Shader::Info& stage, TextureSet& set, const std::array<u32, 8>* raws, const u64* flags, u32 count, TextureResolved& r) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2: the checks of the texture set memo (resolve), see BindTexturesFromSet
    if (set.count != count) {
        ++texture_set_stats.no_entry;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (set.entries[i].flags != flags[i] || set.entries[i].raw != raws[i]) {
            ++texture_set_stats.no_entry;
            return false;
        }
    }
    if (set.generation != texture_cache.ImageGeneration() && !TextureSetGenerationMiss(set)) {
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
        texture_cache.UpdateImage(entry.id, Core::ImageCaller::RasterizerBindTextures);
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
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        TextureBind& tb = r.binds.emplace_back();
        if (entry.null_entry) {
            tb.null_entry = true;
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        tb.image_id = entry.id;
        tb.image = &image;
        tb.view = entry.view;
        tb.range = entry.range;
        tb.is_depth = image.info.props.is_depth;
    }
    return true;
}

void Rasterizer::RecordTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding, TextureResolved& r) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    // Plan S step P2: the record half of the texture bindings (layout transitions, descriptor infos and writes, the indexed row)
    const u32 first_image_idx = image_infos.size();
    const u32 census_base = stage.sw_stage == Shader::SwStage::Compute ? 32u : 0u;
    if (r.set_hit) {
        const u32 count = static_cast<u32>(r.binds.size());
        for (u32 i = 0; i < count; ++i) {
            const TextureBind& tb = r.binds[i];
            if (tb.null_entry) {
                image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
                continue;
            }
            auto& image = *tb.image;
            if (recipe_active) {
                recipe_image_ranges.emplace_back(image.info.guest_address, static_cast<u32>(std::min<u64>(image.info.guest_size, 0xffffffffull)));
            }
            const auto new_layout = tb.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
            needs_barrier |= runtime.Transit(&image, new_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, tb.range);
            image.usage.texture |= 1u;
            image_infos.emplace_back(VK_NULL_HANDLE, tb.view, runtime.CurrentLayout(image));
        }
        for (u32 i = 0; i < count; ++i) {
            if (record_indexed_active && stage.indexed && Shader::Indexed::IsIndexedImage(stage.images[i])) {
                NoteIndexedImage(image_infos[first_image_idx + i], stage.images[i].array_key);
                continue;
            }
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified++;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampledImage;
            set_write.pImageInfo = &image_infos[first_image_idx + i];
        }
        return;
    }
    const u32 num_images = static_cast<u32>(r.binds.size());
    for (u32 i = 0; i < num_images; ++i) {
        TextureBind& tb = r.binds[i];
        if (tb.null_entry) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        const bool is_storage = tb.is_storage;
        const auto transit = [&](auto&&... args) {
            Census(census_base + CImageTransit);
            const bool result = runtime.Transit(std::forward<decltype(args)>(args)...);
            if (result) {
                Census(census_base + CImageTransitTrue);
            }
            return result;
        };
        if (is_storage) {
            Census(census_base + CImageStorage);
        }
        auto& image = *tb.image;
        const VideoCore::SubresourceRange& range = tb.range;
        vk::ImageLayout transit_layout = vk::ImageLayout::eGeneral; // R3 step 2: the layout the transit leaves the image in
        if (tb.general) {
            if (tb.feedback) {
                transit_layout = vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT;
                needs_barrier |= transit(&image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT, vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
            } else {
                needs_barrier |= transit(&image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range);
            }
        } else {
            if (is_storage) {
                needs_barrier |= transit(&image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range);
            } else {
                const auto new_layout = tb.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
                transit_layout = new_layout;
                needs_barrier |= transit(&image, new_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, range);
            }
        }
        image.usage.storage |= is_storage;
        image.usage.texture |= !is_storage;
        if (is_storage) {
            Core::UiTrace::NoteGpuWrite("a storage image binding", image.info.guest_address, image.info.guest_size);
        }
        if (Core::UiTrace::Active() && !is_storage && image.info.props.is_block && !image.info.props.is_depth && image.info.guest_size <= (256u << 10) && image.info.resources.layers == 1 &&
            image.info.resources.levels >= 1 && Core::UiTrace::WantsShot(image.info.guest_address, static_cast<u32>(Core::DebugServer::PresentCount()))) {
            auto ref = runtime.GetStagingPool().Request(image.info.guest_size, VideoCore::MemoryType::HostCached, 16, false);
            const vk::BufferImageCopy shot_copy = {
                .bufferOffset = ref.offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {image.info.size.width, image.info.size.height, 1},
            };
            runtime.DownloadImage(&image, ref.buffer, std::span{&shot_copy, 1});
            needs_barrier |= transit(&image, transit_layout, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead, range);
            ui_shots_pending.push_back({image.info.guest_address, ref, scheduler.CurrentTick(), static_cast<u32>(Core::DebugServer::PresentCount()),
                                        fmt::format("{} {}x{}", vk::to_string(image.info.pixel_format), image.info.size.width, image.info.size.height)});
        }
        if (Core::UiTrace::Active()) {
            Core::UiTrace::NoteImage(image.info.guest_address, image.info.guest_size,
                                     fmt::format("{} {}x{} {} target={} storage={} bound_layout={}", vk::to_string(image.info.pixel_format), image.info.size.width, image.info.size.height,
                                                 AmdGpu::NameOf(image.info.tile_mode), tb.is_target, is_storage, vk::to_string(transit_layout)));
        }
        vk::ImageLayout bound_layout = transit_layout;
        if (!image.backing || !runtime.PlannerDecidesLayouts()) {
            bound_layout = runtime.CurrentLayout(image);
        }
        image_infos.emplace_back(VK_NULL_HANDLE, tb.view, bound_layout);
        if (recipe_active) {
            recipe_image_ranges.emplace_back(image.info.guest_address, static_cast<u32>(std::min<u64>(image.info.guest_size, 0xffffffffull)));
        }
    }
    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    u32 image_desc_idx = 0;
    for (u32 array_size : r.array_sizes) {
        const bool is_storage = r.binds[image_binding_idx].is_storage;
        if (record_indexed_active && stage.indexed && Shader::Indexed::IsIndexedImage(stage.images[image_desc_idx])) {
            if (stage.images[image_desc_idx].is_atomic) {
                Core::RareEvents::Note("c9_atomic_image", fmt::format("shader {:#x} ({}): atomic image in array key {} (format {})", stage.pgm_hash, static_cast<u32>(stage.sw_stage), r.image_keys[image_desc_idx],
                                                                       (r.image_keys[image_desc_idx] - Shader::Indexed::FirstAtomicKey) / 4));
            }
            if (stage.images[image_desc_idx].mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                Core::RareEvents::Note("c9_mip_level_run", fmt::format("shader {:#x} ({}): image bound once per mip level, {} levels, array key {}", stage.pgm_hash, static_cast<u32>(stage.sw_stage), array_size,
                                                                       r.image_keys[image_desc_idx]));
                NoteIndexedRun(&image_infos[image_info_idx], array_size, r.image_keys[image_desc_idx]);
            } else {
                NoteIndexedImage(image_infos[image_info_idx], r.image_keys[image_desc_idx]);
            }
            image_info_idx += array_size;
            image_binding_idx += array_size;
            ++image_desc_idx;
            continue;
        }
        ++image_desc_idx;
        if (record_indexed_active && stage.indexed) {
            IndexedData::Get()->stats.legacy_images.fetch_add(1, std::memory_order_relaxed);
        }
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

}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 mode = SplitBindMode();
    if (mode == 0) {
        BindTexturesLegacy(stage, binding);
        return;
    }
    static std::atomic<u64> checked{0}, differ{0}, set_hits{0};
    if (mode == 2) {
        const size_t infos_before = image_infos.size();
        const size_t bound_before = bound_images.size();
        BindTexturesLegacy(stage, binding);
        const size_t infos_after = image_infos.size();
        const size_t bound_after = bound_images.size();
        const std::vector<vk::DescriptorImageInfo> seen(image_infos.begin() + infos_before, image_infos.end());
        const std::vector<VideoCore::ImageId> seen_bound(bound_images.begin() + bound_before, bound_images.end());
        TextureResolved r;
        ResolveTextures(stage, r);
        bound_images.resize(bound_after); // (the resolve pass put the same images in the list a second time)
        const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        set_hits.fetch_add(r.set_hit ? 1 : 0, std::memory_order_relaxed);
        bool same = r.binds.size() <= seen.size();
        std::string what;
        size_t bound_at = 0;
        for (size_t i = 0; same && i < r.binds.size(); ++i) {
            const TextureBind& tb = r.binds[i];
            if (tb.null_entry) {
                same = !seen[i].imageView;
                what = "null entry has a view";
                continue;
            }
            if (!(seen[i].imageView == tb.view)) {
                same = false;
                what = "view";
            } else if (bound_at >= seen_bound.size() || !(seen_bound[bound_at] == tb.image_id)) {
                same = false;
                what = "image id";
            }
            ++bound_at;
        }
        if (same && bound_at != seen_bound.size()) {
            same = false;
            what = "bound image count";
        }
        if (!same && differ.fetch_add(1) < 20) {
            LOG_WARNING(Render_Vulkan, "split bind verify: textures of stage {:#x} differ: {} ({} legacy infos, {} split bindings)", stage.pgm_hash, what, seen.size(), r.binds.size());
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind verify: {} texture stage calls ({} from the set memo), {} differ", n, set_hits.load(), differ.load());
        }
        (void)infos_after;
        return;
    }
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_textures", 2);
    TextureResolved r;
    ResolveTextures(stage, r);
    RecordTextures(stage, binding, r);
    BindSamplers(stage, binding);
}

void Rasterizer::BindSamplersLegacy(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_samplers", 2);
    for (const auto& sampler : stage.samplers) {
        AmdGpu::Sampler ssharp;
        const u32* record_words = nullptr;
        u32 aniso_dw3 = 0; // L1: dword 3 of the T# the sampler is used with, from the record's image entry (the aniso post operation reads it)
        if (binder_plan != nullptr && sampler.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Sampler>::Summary::SingleLoad) {
            record_words = BinderWords(stage, sampler.sharp_fetch.offsets[0], 4, 3);
            if (record_words != nullptr && sampler.post_op == Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
                const u32 off = sampler.post_op_tsharp_dw3_off;
                const u32* tsharp = off >= 3 && off != Shader::UNKNOWN_LOCATION ? BinderWords(stage, off - 3, 8, 2) : nullptr;
                if (tsharp == nullptr && off >= 3 && off != Shader::UNKNOWN_LOCATION) {
                    tsharp = BinderWords(stage, off - 3, 4, 2);
                }
                if (tsharp != nullptr) {
                    aniso_dw3 = tsharp[3];
                } else {
                    record_words = nullptr;
                }
            }
        }
        if (record_words != nullptr) {
            AmdGpu::Sampler raw{};
            std::memcpy(&raw, record_words, sizeof(raw));
            ssharp = sampler.PostOp(raw, aniso_dw3);
            Core::NativeRegistry::NoteBinder(7);
        } else if (sampler.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Sampler>::Summary::MultiLoad && sampler.sharp_fetch.load_mask == 0 &&
                   sampler.post_op != Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
            // F5: the sampler is the shader's own immediates (no flat user data dword is read), so a stage whose flat user data was skipped still has it
            ssharp = sampler.GetSharp(stage);
            Core::NativeRegistry::NoteBinder(11);
        } else if (StageFlatSkipped(stage)) {
            Core::NativeRegistry::NoteBinder(6); // must not happen
            if (stage.sw_stage == Shader::SwStage::Compute) {
                DumpDispatchRecord(stage, 3, sampler.sharp_fetch.offsets[0], "no record entry: sampler");
            }
            ssharp = AmdGpu::Sampler{};
        } else {
            ssharp = sampler.GetSharp(stage);
        }
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
        // F4.1 / F4.3: the host sampler is made from the plain description of the S# words (after the shader's post operation); the shadow compares it with the emulator's own decode
        const auto from_t_sharp = VideoCore::SamplerDescFromSharp(ssharp);
        const auto from_words = Core::NativeImageDesc::SamplerDescFromWords(reinterpret_cast<const u32*>(&ssharp));
        const auto vk_sampler = texture_cache.GetSampler(Core::NativeImageDesc::FromDescOn() ? from_words : from_t_sharp, *current_bc_base, sampler.is_depth);
        if (verify_slots_on) {
            verify_samplers.push_back(vk_sampler);
        }
        if (indexed_active && stage.indexed) {
            NoteIndexedSampler(vk_sampler);
            continue;
        }
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

void Rasterizer::ResolveSamplers(const Shader::Info& stage, boost::container::small_vector<vk::Sampler, 8>& out) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2: the resolve half of the sampler bindings (S# words to the host sampler)
    for (const auto& sampler : stage.samplers) {
        AmdGpu::Sampler ssharp;
        const u32* record_words = nullptr;
        u32 aniso_dw3 = 0; // L1: dword 3 of the T# the sampler is used with, from the record's image entry (the aniso post operation reads it)
        if (binder_plan != nullptr && sampler.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Sampler>::Summary::SingleLoad) {
            record_words = BinderWords(stage, sampler.sharp_fetch.offsets[0], 4, 3);
            if (record_words != nullptr && sampler.post_op == Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
                const u32 off = sampler.post_op_tsharp_dw3_off;
                const u32* tsharp = off >= 3 && off != Shader::UNKNOWN_LOCATION ? BinderWords(stage, off - 3, 8, 2) : nullptr;
                if (tsharp == nullptr && off >= 3 && off != Shader::UNKNOWN_LOCATION) {
                    tsharp = BinderWords(stage, off - 3, 4, 2);
                }
                if (tsharp != nullptr) {
                    aniso_dw3 = tsharp[3];
                } else {
                    record_words = nullptr;
                }
            }
        }
        if (record_words != nullptr) {
            AmdGpu::Sampler raw{};
            std::memcpy(&raw, record_words, sizeof(raw));
            ssharp = sampler.PostOp(raw, aniso_dw3);
            Core::NativeRegistry::NoteBinder(7);
        } else if (sampler.sharp_fetch.summary == Shader::SharpFetch<AmdGpu::Sampler>::Summary::MultiLoad && sampler.sharp_fetch.load_mask == 0 &&
                   sampler.post_op != Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
            // F5: the sampler is the shader's own immediates (no flat user data dword is read), so a stage whose flat user data was skipped still has it
            ssharp = sampler.GetSharp(stage);
            Core::NativeRegistry::NoteBinder(11);
        } else if (StageFlatSkipped(stage)) {
            Core::NativeRegistry::NoteBinder(6); // must not happen
            if (stage.sw_stage == Shader::SwStage::Compute) {
                DumpDispatchRecord(stage, 3, sampler.sharp_fetch.offsets[0], "no record entry: sampler");
            }
            ssharp = AmdGpu::Sampler{};
        } else {
            ssharp = sampler.GetSharp(stage);
        }
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
        // F4.1 / F4.3: the host sampler is made from the plain description of the S# words (after the shader's post operation); the shadow compares it with the emulator's own decode
        const auto from_t_sharp = VideoCore::SamplerDescFromSharp(ssharp);
        const auto from_words = Core::NativeImageDesc::SamplerDescFromWords(reinterpret_cast<const u32*>(&ssharp));
        const auto vk_sampler = texture_cache.GetSampler(Core::NativeImageDesc::FromDescOn() ? from_words : from_t_sharp, *current_bc_base, sampler.is_depth);
        out.push_back(vk_sampler);
    }
}

void Rasterizer::RecordSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding, const vk::Sampler* samplers, u32 count) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    // Plan S step P2: the record half (descriptor infos, writes, the indexed row)
    for (u32 i = 0; i < count; ++i) {
        if (record_indexed_active && stage.indexed) {
            NoteIndexedSampler(samplers[i]);
            continue;
        }
        image_infos.emplace_back(samplers[i], VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

void Rasterizer::BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 mode = SplitBindMode();
    if (mode == 0) {
        BindSamplersLegacy(stage, binding);
        return;
    }
    static std::atomic<u64> checked{0}, differ{0};
    if (mode == 2) {
        verify_samplers.clear();
        verify_slots_on = true;
        BindSamplersLegacy(stage, binding);
        verify_slots_on = false;
        boost::container::small_vector<vk::Sampler, 8> again;
        ResolveSamplers(stage, again);
        const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        if (again.size() != verify_samplers.size() || !std::equal(again.begin(), again.end(), verify_samplers.begin())) {
            if (differ.fetch_add(1) < 20) {
                LOG_WARNING(Render_Vulkan, "split bind verify: samplers of stage {:#x} differ ({} legacy, {} split)", stage.pgm_hash, verify_samplers.size(), again.size());
            }
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind verify: {} sampler stage calls, {} differ", n, differ.load());
        }
        return;
    }
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseMixed};
    TLG_PROF_LEVEL("raster.bind_samplers", 2);
    boost::container::small_vector<vk::Sampler, 8> resolved;
    ResolveSamplers(stage, resolved);
    RecordSamplers(stage, binding, resolved.data(), static_cast<u32>(resolved.size()));
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
    sig.depth_control = regs.raw_targets.depth_control; // the memo key keeps the whole register word (G1b: raw until F6b)
    sig.depth_valid = regs.depth_buffer.DepthValid();
    sig.stencil_valid = regs.depth_buffer.StencilValid();
    return sig;
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    TLG_PROF_LEVEL("raster.begin_rendering", 2);
    if (!RenderStateMemoEnabled()) {
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
        attachment_feedback_loop = false;
        return begin_memo.state;
    }
    begin_memo.valid = false;
    RenderState state = BeginRenderingFull(pipeline);
    if (!attachment_feedback_loop) {
        // A state with a clear is not remembered: the next draw's differs (the clear is done).
        bool clears = state.depth_stencil_attachment.depth_clear || state.depth_stencil_attachment.stencil_clear;
        for (u32 i = 0; i < state.num_color_attachments; ++i) {
            clears |= state.color_attachments[i].is_clear != 0;
        }
        if (clears) {
            Core::NativeDevice::NoteClearUse(Core::NativeDevice::UseRenderPassClear, engine_clear_on);
        }
        if (!clears) {
            begin_memo = {true, MakeBeginSignature(pipeline), state};
        }
    }
    return state;
}

RenderState Rasterizer::BeginRenderingLegacy(const GraphicsPipeline* pipeline) {
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
        texture_cache.UpdateImage(image_id, Core::ImageCaller::RasterizerBeginRendering);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        if (Core::ViewTable::TargetMode() != 0) [[unlikely]] {
            TargetViewShadow(cb, *image_view.image_view);
        }
        if (Core::LightRecorder::On()) {
            Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), image->info.guest_address, image->info.guest_size, texture_cache.ImageUid(image_id), 1);
        }
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        // B11: an engine fast clear pending on the image is the clear of the next render pass on it (or of the eliminate), as in the original port. Its colour is the
        // one in the target's registers (the hardware's clear words), not the colour of the engine's clear call (see EliminateFastClear). `TLG_B11_FAST_AT_PASS=0`
        // leaves it to the eliminate (study).
        static const bool fast_clear_at_pass = [] {
            const char* e = std::getenv("TLG_B11_FAST_AT_PASS");
            return e == nullptr || *e != '0';
        }();
        bool is_clear = false;
        std::optional<vk::ClearValue> engine_clear_value;
        if (image->pending_fast_clear && fast_clear_at_pass) {
            engine_clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);
            image->pending_fast_clear.reset();
            is_clear = true;
            Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeFastClearAtRenderPass);
        }

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

        const auto clear_value = engine_clear_value ? *engine_clear_value : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = runtime.CurrentLayout(*image);
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
        if (Core::ViewTable::TargetMode() != 0) [[unlikely]] {
            TargetViewShadow(8, *image_view.image_view);
        }
        if (Core::LightRecorder::On()) {
            Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), image.info.guest_address, image.info.guest_size, texture_cache.ImageUid(image_id), 2);
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable && regs.depth_control.depth_write_enable;
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
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
        attachment.image_layout = runtime.CurrentLayout(image);
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

void Rasterizer::ResolveRenderTargets(const GraphicsPipeline* pipeline, RenderResolved& r) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Resolve};
    // Plan S step P2c: the resolve half of the render pass setup (images, views, clear values, sizes); the record half transits the images and reads their layouts
    attachment_feedback_loop = false;
    const auto& regs = *draw_inputs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState& state = r.state;
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
        r.color_used[cb] = true;
        auto* image = &texture_cache.GetImage(image_id);
        r.color[cb].image = image;
        if (image->binding.needs_rebind) {
            rt_cache[cb].valid = false;
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
            r.color[cb].image = image;
        }
        texture_cache.UpdateImage(image_id, Core::ImageCaller::RasterizerBeginRendering);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        if (Core::LightRecorder::On()) {
            Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), image->info.guest_address, image->info.guest_size, texture_cache.ImageUid(image_id), 1);
        }
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        // B11: an engine fast clear pending on the image is the clear of the next render pass on it (or of the eliminate), as in the original port. Its colour is the
        // one in the target's registers (the hardware's clear words), not the colour of the engine's clear call (see EliminateFastClear). `TLG_B11_FAST_AT_PASS=0`
        // leaves it to the eliminate (study).
        static const bool fast_clear_at_pass = [] {
            const char* e = std::getenv("TLG_B11_FAST_AT_PASS");
            return e == nullptr || *e != '0';
        }();
        bool is_clear = false;
        std::optional<vk::ClearValue> engine_clear_value;
        if (image->pending_fast_clear && fast_clear_at_pass) {
            engine_clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);
            image->pending_fast_clear.reset();
            is_clear = true;
            Core::NativeDevice::NoteClearNative(Core::NativeDevice::NativeFastClearAtRenderPass);
        }

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            r.color[cb].feedback = true;
            attachment_feedback_loop = true;
        }
        r.color[cb].range = desc.view_info.range;

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value = engine_clear_value ? *engine_clear_value : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
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
        if (Core::LightRecorder::On()) {
            Core::LightRecorder::ImageWritten(Core::DebugServer::PresentCount(), image.info.guest_address, image.info.guest_size, texture_cache.ImageUid(image_id), 2);
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable && regs.depth_control.depth_write_enable;
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
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
        r.depth.image = &image;
        r.depth.range = desc.view_info.range;
        r.depth.new_layout = new_layout;

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
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
    r.feedback_loop = attachment_feedback_loop;
}


RenderState Rasterizer::RecordRenderTargets(RenderResolved& r) {
    VideoCore::HalfPhase::Scope half_scope{VideoCore::HalfPhase::Record};
    VideoCore::ResolveCommands::Run();
    // Plan S step P2c: the record half of the render pass setup: layout transitions of the targets, and the layout each attachment is in afterwards
    RenderState& state = r.state;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        auto& target = r.color[cb];
        if (target.image == nullptr || !r.color_used[cb]) {
            continue;
        }
        auto* image = target.image;
        if (target.feedback) {
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             target.range);
        }
        state.color_attachments[cb].image_layout = runtime.CurrentLayout(*image);
        image->usage.render_target = 1u;
    }
    if (r.depth.image != nullptr) {
        auto& image = *r.depth.image;
        needs_barrier |= runtime.Transit(&image, r.depth.new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         r.depth.range);
        state.depth_stencil_attachment.image_layout = runtime.CurrentLayout(image);
        image.usage.depth_target = true;
    }
    return state;
}

RenderState Rasterizer::BeginRenderingFull(const GraphicsPipeline* pipeline) {
    const u32 mode = SplitBindMode();
    if (mode == 0) {
        return BeginRenderingLegacy(pipeline);
    }
    if (mode == 2) {
        static std::atomic<u64> checked{0}, differ{0};
        const RenderState seen = BeginRenderingLegacy(pipeline);
        const bool loop_after_legacy = attachment_feedback_loop;
        const size_t bound_before = bound_images.size();
        RenderResolved r;
        ResolveRenderTargets(pipeline, r); // (the fast clear is consumed already: compare views, sizes and the feedback decision, not the clear)
        bound_images.resize(bound_before);
        bool same = r.state.width == seen.width && r.state.height == seen.height && r.state.num_layers == seen.num_layers &&
                    r.state.num_color_attachments == seen.num_color_attachments && attachment_feedback_loop == loop_after_legacy;
        for (u32 cb = 0; same && cb < seen.num_color_attachments; ++cb) {
            same = r.state.color_attachments[cb].image_view == seen.color_attachments[cb].image_view;
        }
        same = same && r.state.depth_stencil_attachment.image_view == seen.depth_stencil_attachment.image_view &&
               r.state.depth_stencil_attachment.has_depth == seen.depth_stencil_attachment.has_depth &&
               r.state.depth_stencil_attachment.has_stencil == seen.depth_stencil_attachment.has_stencil;
        const u64 n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!same && differ.fetch_add(1) < 20) {
            LOG_WARNING(Render_Vulkan, "split bind verify: render pass setup differs from the old code ({} colour targets)", seen.num_color_attachments);
        }
        if ((n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan, "split bind verify: {} render pass setups, {} differ", n, differ.load());
        }
        return seen;
    }
    RenderResolved r;
    ResolveRenderTargets(pipeline, r);
    return RecordRenderTargets(r);
}

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

void Rasterizer::Resolve(const DrawInputs& inputs, const AmdGpu::CbDbExtent& mrt0_hint, const AmdGpu::CbDbExtent& mrt1_hint) {
    VideoCore::TextureCache::ImageDesc mrt0_desc{inputs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{inputs.color_buffers[1], mrt1_hint};
    mrt0_desc.engine_id = (*current_target_resources)[0]; // E3
    mrt1_desc.engine_id = (*current_target_resources)[1];
    mrt0_desc.engine_site = 5;
    mrt1_desc.engine_site = 6;
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
    read_desc.engine_site = 7;
    read_desc.engine_id =(*current_target_resources)[8]; // E3: the bound depth view's resource (the write side is another buffer: left unnamed)
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
    if (!is_gds && texture_cache.GetMetaInfo(address) && SkipForEngineClear()) {
        // B11: the meta fill of an engine clear that ran natively (the CMASK / HTILE are not the image's state any more)
        return;
    }
    if (!is_gds) {
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
        Core::NativeRegistry::SetBufferWriter(u64(value), 101); // diagnostics: a command processor fill (value)
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
    texture_cache.UpdateImage(src_id, Core::ImageCaller::RasterizerTryCopyImages);
    const bool ui_readback = Core::UiTrace::Armed() && num_bytes >= 16384 && num_bytes <= (256u << 10) && a.props.is_block && !a.props.is_depth && a.resources.layers == 1;
    const auto ui_queue_readback = [&](VideoCore::Image& image, int which) {
        auto ref = runtime.GetStagingPool().Request(image.info.guest_size, VideoCore::MemoryType::HostCached, 16, false);
        const vk::BufferImageCopy readback_copy = {
            .bufferOffset = ref.offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {image.info.size.width, image.info.size.height, 1},
        };
        runtime.DownloadImage(&image, ref.buffer, std::span{&readback_copy, 1});
        ui_copies_pending.push_back({Core::UiTrace::LastCopyNumber(), which, ref, scheduler.CurrentTick()});
        Core::UiTrace::NoteCopyQueued(Core::UiTrace::LastCopyNumber());
    };
    if (ui_readback) {
        ui_queue_readback(source, 1); // what the source image holds before the copy
    }
    runtime.CopyImage(&source, &destination);
    if (ui_readback) {
        ui_queue_readback(destination, 0); // and the destination after it
    }
    if (Core::UiTrace::Armed()) {
        Core::UiTrace::NoteCopyPath(fmt::format("image copy (readback {})", ui_readback ? "queued" : "not queued"));
    }
    destination.flags |= VideoCore::ImageFlagBits::GpuModified;
    destination.flags &= ~VideoCore::ImageFlagBits::Dirty;
    ++recorded_work;
    Core::NativeResources::NoteImageCopy(true, 0);
    return true;
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds, bool staging_readback) {
    if (!dst_gds && !src_gds) {
        Core::UiTrace::NoteGpuCopy(dst, src, num_bytes);
    }
    if (!dst_gds && !src_gds && Core::OwnMemoryCensus::On()) [[unlikely]] {
        Core::OwnMemoryCensus::NoteCopy(dst, src, num_bytes); // design review 79, step 1a
    }
    const u64 ui_copy_number = Core::UiTrace::LastCopyNumber(); // taken now: work in between must not change which copy a result belongs to
    Core::ResourceAudit::Note(dst_gds ? Core::ResourceAudit::GdsFill : Core::ResourceAudit::MemoryCopyDst, dst, num_bytes, true);
    Core::ResourceAudit::Note(src_gds ? Core::ResourceAudit::GdsFill : Core::ResourceAudit::MemoryCopySrc, src, num_bytes, false);
    if (!dst_gds && !src_gds) {
        Core::NativeContent::NoteCopy(dst, src, num_bytes); // F2.2 shadow: the destination's content is the source's, resolved from the store
    }
    if (!dst_gds && !src_gds && Core::NativeResources::ImageCopyEnabled() && TryCopyImages(dst, src, num_bytes)) {
        return;
    }
    // B5 (`TLG_B5_GPU_COPIES=1`): every copy runs on the GPU between the two ranges; the CPU copy inside PS4 memory below was the emulation of the console's DMA
    static const bool b5_gpu_copies = [] {
        const char* e = std::getenv("TLG_B5_GPU_COPIES");
        return e == nullptr || *e != '0'; // B7: default on (owner play 2026-10-29)
    }();
    if (b5_gpu_copies) {
        b5_copies_on_gpu.fetch_add(1, std::memory_order_relaxed);
    } else if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory. The destination is
            // invalidated as one range first: written page by page the copy would take a write fault per page.
            if (Core::NativeUpdates::enabled.load(std::memory_order_relaxed)) {
                Core::NativeUpdates::NoteWriterRange(3, dst, num_bytes); // a copy of the command stream (the engine's own CsCopyBufferDwords / Map copies), not a game write
            }
            InvalidateMemory(dst, num_bytes, false);
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            buffer_cache.NoteCpGuestWrite(dst, num_bytes);
            b5_copies_on_cpu.fetch_add(1, std::memory_order_relaxed);
            Core::UiTrace::NoteCopyPath("cpu memcpy");
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
        Core::NativeRegistry::SetBufferWriter(src, src_gds ? 103 : 100); // diagnostics: a GPU copy (source address)
        Core::NativeContent::SetBindContext({"copy destination", 0, 0, 0, 0, false, true, dst, num_bytes});
        const auto copy_destination = buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
        Core::NativeContent::ClearBindContext();
        return copy_destination;
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    if (Core::UiTrace::Armed()) {
        Core::UiTrace::NoteCopyPath(fmt::format("gpu buffer copy (staging_readback={} explicit_readback={} src_gds={} dst_gds={})", staging_readback, explicit_readback, src_gds, dst_gds));
    }
    if (!dst_gds && !src_gds && !explicit_readback && num_bytes >= 16384 && num_bytes <= (256u << 10) && Core::UiTrace::Armed()) {
        // what the source holds right before the copy (the first hop that reads zeros where the previous hop wrote text)
        auto ref = runtime.GetStagingPool().Request(num_bytes, VideoCore::MemoryType::HostCached, 16, false);
        const vk::BufferCopy before = {.srcOffset = src_offset, .dstOffset = ref.offset, .size = num_bytes};
        runtime.CopyBuffer(src_buffer, ref.buffer, std::span{&before, 1});
        ui_copies_pending.push_back({ui_copy_number, 1, ref, scheduler.CurrentTick()});
    }
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
    if (!dst_gds && !explicit_readback && num_bytes >= 16384 && num_bytes <= (256u << 10) && Core::UiTrace::Armed()) {
        // read back what the destination holds now (a picture of the interface text moves through copies of copies; the report shows the hop where it goes blank or wrong)
        auto ref = runtime.GetStagingPool().Request(num_bytes, VideoCore::MemoryType::HostCached, 16, false);
        const vk::BufferCopy back = {.srcOffset = dst_offset, .dstOffset = ref.offset, .size = num_bytes};
        runtime.CopyBuffer(dst_buffer, ref.buffer, std::span{&back, 1});
        ui_copies_pending.push_back({ui_copy_number, 0, ref, scheduler.CurrentTick()});
        Core::UiTrace::NoteCopyQueued(ui_copy_number);
    }
    if (!dst_gds && VideoCore::ReadbackPolicy::Get().dma_async.load(std::memory_order_relaxed)) {
        // B6: only where the CPU reads the result (an explicit readback, a resource the engine made for CPU reads, or memory no engine resource owns)
        const u32 reason = explicit_readback ? 3u : Core::NativeDevice::CpuReadReason(dst, num_bytes);
        if (!Core::NativeDevice::B6NativeOn() || reason != 0) {
            buffer_cache.RecordDmaDownload(dst_buffer, dst, num_bytes, explicit_readback);
            b6_downloads[reason == 0 ? 0 : reason + 1].fetch_add(1, std::memory_order_relaxed);
        } else {
            b6_downloads[1].fetch_add(1, std::memory_order_relaxed);
        }
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
    if (Core::OwnMemoryCensus::On()) [[unlikely]] {
        Core::OwnMemoryCensus::NoteFault(addr, size); // design review 79, step 1a
    }
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
    // F2.4: no page is watched, so the only callers are a guest mprotect and the AV player; the images are not dirtied by an address any more (owner play 2026-11-03 with a counter: 0 calls). Written down if it ever happens.
    Core::RareEvents::Note("f24_invalidate_memory_on_mapped_range", fmt::format("{:#x} + {:#x}: images over it are not dirtied by the address any more", addr, size));
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
    // E4: no page is watched any more (the engine's announcements tell what the CPU wrote)
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
    TLG_PROF_LEVEL("raster.dynamic_state", 2);
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
    return executor->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return executor->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
