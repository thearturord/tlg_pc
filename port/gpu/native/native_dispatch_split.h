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

} // namespace Core::DispatchSplit
