// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <map>
#include <string>

#include "common/types.h"
#include "native/native_timeline.h"

/// The raster thread profiler (key Shift+F9, endpoint /raster_profile): what the raster thread's time goes to, inside one draw or dispatch.
///
/// F8 (stall profiler) says which thread limits the frame; F9 (frame profiler) times the game's jobs and the stage threads as a whole. Neither looks inside `GpuRaster`. This does:
/// a capture of `seconds` seconds during which the raster thread times named phases (`Phase`: pipeline lookup, buffer binding, texture lookup, image refresh, barriers, the
/// descriptor push, the Vulkan recording calls, ...) as nested scopes. The time is accumulated on the raster thread itself (no event per scope: a draw has dozens), nested scopes give
/// inclusive and exclusive time, and the report (user/log/raster_<time>.html and .txt, opened in the browser) shows, per frame: the phase tree, the phases ranked by exclusive time,
/// the draws and dispatches by engine caller, and the slowest single operations with their own breakdown.
///
/// Cost: a scope is two timestamp reads and a few stores, only while a raster profile runs (otherwise one relaxed load). The report says how many scopes ran per frame; compare shapes, not
/// absolute numbers, as with the other profilers. Only the raster thread of the split pipeline is measured (the thread named shadPS4:GpuRaster).
namespace Core::RasterProf {

enum class Phase : u8 {
    None,
    // one operation of the raster thread's queue (top level)
    OpDraw,
    OpDrawIndirect,
    OpDispatch,
    OpDispatchIndirect,
    OpClosure,
    Callbacks,       ///< commands from other threads run between operations (ProcessCommands)
    RingUse,         ///< BeginRingUse / EndRingUse around a call
    FlushRecorded,   ///< MaybeFlushRecorded after a call
    // draw and dispatch
    PopPending,      ///< Scheduler::PopPendingOperations
    FilterDraw,
    ResolvePipeline, ///< ResolveDrawPipeline / ResolveDispatchPipeline: the pipeline of the draw
    PlanInputs,      ///< FixedFor / InputsFor: the fixed state and the draw inputs
    PrepareRenderState,
    PendingFill,     ///< a dispatch that is a fill: TryClearImage
    HleCheck,        ///< ExecuteShaderHLE
    BindResources,
    BindBuffers,
    ObtainBuffer,    ///< BufferCache::ObtainBuffer
    BindTextures,
    FindImage,       ///< TextureCache::FindImage
    FindView,        ///< FindTexture / FindRenderTarget / FindDepthTarget (the image view)
    RefreshImage,    ///< TextureCache::RefreshImage (all of it, including the upload below)
    ObtainForImage,  ///< RefreshImage: BufferCache::ObtainBufferForImage (the copy of the guest bytes)
    Detile,          ///< RefreshImage: TileManager::DetileImage
    UploadImage,     ///< RefreshImage: Runtime::UploadImage
    BindSamplers,
    GetSampler,
    IndexedRow,      ///< the indexed data path's table copy in BindResources
    BeginRendering,  ///< Rasterizer::BeginRendering: the render targets
    BindVertexBuffers,
    BindIndexBuffer,
    FlushBarriers,
    DescriptorPush,  ///< Pipeline::BindResources: the descriptor writes
    DynamicState,
    SchedBeginRendering,
    SchedEndRendering,
    RecordDraw,      ///< bindPipeline and the draw call
    RecordDispatch,  ///< bindPipeline and the dispatch call
    ResetBindings,
    HandoffWait,     ///< the raster thread waits for room in the encoder's queue
    Count
};

inline std::atomic<bool> g_active{false};
inline thread_local bool t_raster = false;

/// Called once by the raster thread when it starts.
inline void MarkRasterThread() {
    t_raster = true;
}

[[nodiscard]] inline bool Active() noexcept {
    return g_active.load(std::memory_order_relaxed) && t_raster;
}

void Enter(Phase phase) noexcept;
void Leave() noexcept;

/// A named phase of the raster thread's work, timed while a raster profile runs.
struct Scope {
    explicit Scope(Phase phase) noexcept : live{Active()} {
        if (live) [[unlikely]] {
            Enter(phase);
        }
    }
    ~Scope() {
        End();
    }
    /// Ends the scope before the end of the block (for a phase whose results are declared inside it).
    void End() noexcept {
        if (live) [[unlikely]] {
            live = false;
            Leave();
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    bool live;
};

/// Names the operation that is being measured (the engine function that issued it, the pass class): set inside Draw / Dispatch, read when the operation ends.
void NoteOp(u64 caller_ghidra, u32 op_class) noexcept;

/// Starts a raster profile of `seconds` seconds (false when a capture is running). With `open_in_browser` the report opens when it is done.
bool Start(u32 seconds, bool open_in_browser = false);
/// Stops a running raster profile early; it is analysed as if its time had run out.
bool Stop();
/// True while a raster profile runs.
[[nodiscard]] bool Recording();
/// True when the capture that just ran (or runs) is a raster profile; read by the timeline writer.
[[nodiscard]] bool ThisIsRasterCapture();

/// Called by the timeline writer when the capture ended.
void Analyze(const FrameTimeline::CapEvent* events, u64 count, u64 start_ns, u64 end_ns);

[[nodiscard]] std::string FullText();
[[nodiscard]] std::string ReportJson();
[[nodiscard]] double ReportAgeSeconds();

} // namespace Core::RasterProf
