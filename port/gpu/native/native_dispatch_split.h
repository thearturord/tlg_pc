// SPDX-License-Identifier: GPL-2.0-or-later
// Dispatch executor census (design review section 71, `TLG_DISPATCH_SPLIT=1`, default off; counters only, no behaviour change).
//
// One direct dispatch in `TLG_DISPATCH_SPLIT_EVERY` (default 8) is stamped with the time stamp counter at every step from the raster thread's ExecRasterCall to its end, so the work
// around the bind (about 2.2 us per dispatch in the census of section 66) is split by step. A dispatch that leaves early (cleared, HLE, no pipeline, bound nothing) is dropped.
// Split by whether the front planned the dispatch's bindings (a binder plan). Published as `dispatch_split` (cumulative ns per step and the count).

#pragma once

#include <atomic>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

namespace Core::DispatchSplit {

enum Point : u32 {
    PExecBegin = 0,      ///< ExecRasterCall entered (a direct dispatch)
    PRingUse,            ///< the plan's inputs and BeginRingUse
    PRecordDrain,        ///< RecordQueue::Drain(NonDraw)
    PCallerFill,         ///< NativeInvalidate, the caller, the pending fill, the record verify
    PReadbackFlag,       ///< SetExplicitImageReadback
    PEntry,              ///< DispatchDirect entered (the marker scope and the census scope opened)
    PClassTags,          ///< the clear checks, the draw class, the current draw, the profiler tags
    PPopPending,         ///< scheduler.PopPendingOperations
    PFill,               ///< the pending fill
    POpGroup,            ///< BarrierCensus::SetOp, the independent group
    PPipeline,           ///< ResolveDispatchPipeline
    PLightHle,           ///< the light recorder's frame op, ExecuteShaderHLE
    PBinderPlan,         ///< DispatchBinderFor, the light barrier switch, the light recorder's begin
    PBind,               ///< BindResources
    PAfterBindCensus,    ///< the light recorder's end, BarrierCensus::Dispatch, the independent group, the particle census
    PFlushBarriers,      ///< runtime.FlushBarriers (when needed)
    PEndRendering,       ///< scheduler.EndRendering
    PDescriptorPush,     ///< pipeline->BindResources (the descriptor set and push constants into the command buffer)
    PMarkerRow,          ///< NoteMarkerRow
    PRecord,             ///< the dispatch stamps, bindPipeline, dispatch
    PCount,              ///< DebugState and the recorded work count
    PResetResolveCmds,   ///< ResetBindings: ResolveCommands::Run
    PResetImages,        ///< ResetBindings: the bound images' binding flags cleared
    PResetBindings,      ///< ResetBindings: RecordBufferAccess (the barrier plan records every bound buffer's access)
    PExit,               ///< back in ExecRasterCall (the census and marker scopes closed)
    PStats,              ///< SetExplicitImageReadback(false), DispatchStatFor
    PRingEnd,            ///< EndRingUse
    PFlushRecorded,      ///< MaybeFlushRecorded
    PointCount
};

extern thread_local bool t_active;
extern thread_local u64 t_stamp[PointCount];

[[nodiscard]] inline u64 Tsc() noexcept {
    return __builtin_ia32_rdtsc();
}

/// Starts the census for the next direct dispatch on this thread (sampled).
void Begin() noexcept;
inline void Stamp(Point p) noexcept {
    if (t_active) [[unlikely]] {
        t_stamp[p] = Tsc();
    }
}
/// Marks the sampled dispatch as planned (the front's binder plan was found).
void NotePlanned(bool planned) noexcept;
/// RecordBufferAccess of the sampled dispatch: 0 a stream read skipped, 1 a read recorded, 2 a write recorded.
void NoteAccess(u32 what) noexcept;
/// Ends the sampled dispatch and adds its steps (dropped unless every point was stamped in order).
void End() noexcept;
[[nodiscard]] bool On() noexcept;
[[nodiscard]] nlohmann::json Report();

// ---- barrier planner census (design review section 72, `TLG_PLANNER_CENSUS=1`, default off; counters only): one access record and one query in 8 split by step,
// every effective flush timed with the size of what it clears. Published inside `dispatch_split` as `planner`.
[[nodiscard]] bool PlannerOn() noexcept;
extern thread_local bool t_planner_sampled; ///< set while a sampled access or query runs (LiveObjectSpace times its oracle part then)
extern thread_local u64 t_oracle_ticks;
/// Decides whether the next access or query on this thread is sampled (sets t_planner_sampled, clears t_oracle_ticks).
bool PlannerSample() noexcept;
/// One sampled access record: ticks of the object space lookup (oracle part apart), of the map lookup, of the interval adds, of the rest; whether the buffer is placed (own Vulkan
/// buffer), the map made a new node, it was a write; the sizes of the read and write lists after.
void NotePlannerAccess(u64 space, u64 oracle, u64 lookup, u64 add, u64 rest, bool placed, bool new_node, bool write, u32 read_size, u32 write_size) noexcept;
/// One sampled query: ticks of the object space lookup (oracle apart) and of the map lookup with the overlap tests; whether the map made a new node; the answer.
void NotePlannerQuery(u64 space, u64 oracle, u64 lookup, bool new_node, bool answer) noexcept;
/// One flush: ticks of the planner clears and of the Vulkan barrier record; the keys the object, address and group maps held; whether a barrier was recorded.
void NotePlannerFlush(u64 clear, u64 record, u32 object_keys, u32 address_keys, u32 group_keys, bool actual) noexcept;

// ---- draw census (design review section 75, `TLG_DRAW_SPLIT=1`, default off; counters only): one direct draw in `TLG_DISPATCH_SPLIT_EVERY` (default 8) stamped at every step
// from ExecRasterCall to its end, by draw class (the tag's class: 0 z prepass and shadow, 1 g-buffer, ...; 7 any other). Published inside `dispatch_split` as `draws`.
enum DrawPoint : u32 {
    DExecBegin = 0,
    DRingUse,          ///< the plan's inputs and BeginRingUse
    DSetTag,           ///< SetNativeTag
    DEntry,            ///< Draw entered (the marker scope and the census scope opened)
    DClearChecks,      ///< BarrierCensus::SetOp, the clear checks
    DPopPending,       ///< scheduler.PopPendingOperations
    DTagClassStats,    ///< the tag, the draw class, the current draw, the profiler tags, the native counters
    DFilter,           ///< FilterDraw
    DPipeline,         ///< ResolveDrawPipeline
    DDiagInputs,       ///< the UI trace, the light recorder, the plan's inputs, the extents
    DPrepareState,     ///< PrepareRenderState
    DBinderPlan,       ///< NativeBinderFor
    DBind,             ///< BindResources
    DBeginRenderingState, ///< BeginRendering(pipeline): the render state of the targets
    DVertexBuffers,    ///< BindVertexBuffers
    DIndexBuffer,      ///< BindIndexBuffer (when indexed)
    DPassDifference,   ///< PassStateDifference
    DFlushBarriers,    ///< runtime.FlushBarriers (when needed)
    DDescriptorPush,   ///< pipeline->BindResources
    DDynamicState,     ///< UpdateDynamicState
    DSchedBeginRendering, ///< scheduler.BeginRendering
    DDrawOffsets,      ///< GetDrawOffsets
    DMarkerRow,        ///< NoteMarkerRow
    DRecord,           ///< bindPipeline (unless the same), draw
    DCounts,           ///< DebugState, NoteDraw, the recorded work count
    DResetResolveCmds, ///< ResetBindings: ResolveCommands::Run
    DResetImages,      ///< ResetBindings: the bound images' flags
    DResetBufferAccess, ///< ResetBindings: RecordBufferAccess
    DExit,             ///< back in ExecRasterCall (native_last, the scopes closed)
    DRingEnd,          ///< EndRingUse
    DFlushRecorded,    ///< MaybeFlushRecorded
    DrawPointCount
};
extern thread_local bool t_draw_active;
extern thread_local u64 t_draw_stamp[DrawPointCount];
inline void DStamp(DrawPoint p) noexcept {
    if (t_draw_active) [[unlikely]] {
        t_draw_stamp[p] = Tsc();
    }
}
[[nodiscard]] bool DrawOn() noexcept;
void BeginDraw() noexcept;
/// The sampled draw's class (the tag's), whether it is indexed and whether the front's binder plan was found.
void NoteDrawKind(u32 draw_class, bool indexed, bool planned) noexcept;
void EndDraw() noexcept;

} // namespace Core::DispatchSplit
