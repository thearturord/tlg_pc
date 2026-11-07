// SPDX-License-Identifier: GPL-2.0-or-later
// Plan S, step S0 (documents/tlg/native plan/native-data-path-notes.md, "Plan S: the thread split"): which shared state does each raster thread operation touch?
//
// Off by default (`TLG_SPLIT_CENSUS=1`). A thread-local bit mask is cleared before every operation of the raster thread; the entry points of the shared caches (buffer cache, texture
// cache, pipeline cache, barrier state, indexed data, stream buffer, label timeline, the command list of the scheduler) set their bit. After the operation the raster thread adds
// (group, mask, time) to a table. The report shows, per group, how long the operations took and which mask they had, and the derived number the split depends on: how much time sits in
// operations that touch no cache at all (they could run on another thread with no lock at all), and how much time each cache class is touched by.

#pragma once

#include <chrono>

#include "common/types.h"

namespace Core::SplitCensus {

enum Bit : u32 {
    BufObtain = 0,   ///< BufferCache::ObtainBuffer / ObtainBufferForImage / BufferFor: bind memo, arena, placed tables
    BufSync,         ///< BufferCache::SynchronizeMemory*: dirty sets, memory tracker, the upload copy
    BufDirty,        ///< BufferCache::ApplyCpuWrite / MarkBuffersChanged: object dirty sets, image index
    BufMemoErase,    ///< BufferCache::BindMemoErase
    BufPlaced,       ///< BufferCache::Place / DropPlaced
    BufStream,       ///< StreamBuffer rows (constant ring copies)
    TexFind,         ///< TextureCache::Find*: lookup tables (may register)
    TexRefresh,      ///< TextureCache::RefreshImage (upload, layout)
    TexSampler,      ///< TextureCache::GetSampler
    TexRegister,     ///< TextureCache::RegisterImage / UnregisterImage
    Pipeline,        ///< PipelineCache::Get*Pipeline
    Barrier,         ///< Runtime::FlushBarriers (barrier state)
    Indexed,         ///< IndexedData slots (global descriptor set)
    GpuRanges,       ///< BufferCache::SupersedeGpuWritten (GPU written ranges)
    ObjectNotes,     ///< NativeResources::NoteCpuWrite / NoteAnnouncedWrite (resident and immutable buffers)
    ImageIndex,      ///< NativeImageIndex announced write notes
    Label,           ///< LabelTimeline
    Commands,        ///< the scheduler's command list was asked for (the operation records Vulkan commands on a stream)
    GuestWrite,      ///< the operation writes guest memory (writes_memory)
    BitCount,
};

/// The classes that are shared caches (what a lane needs a lock for). Record, GuestWrite and Label are not caches.
constexpr u32 CacheMask = (1u << BufObtain) | (1u << BufSync) | (1u << BufDirty) | (1u << BufMemoErase) | (1u << BufPlaced) | (1u << BufStream) | (1u << TexFind) |
                          (1u << TexRefresh) | (1u << TexSampler) | (1u << TexRegister) | (1u << Pipeline) | (1u << Barrier) | (1u << Indexed) | (1u << GpuRanges) | (1u << ObjectNotes) | (1u << ImageIndex);

extern bool g_on;
extern thread_local u32 t_mask;
extern thread_local u32 t_phase;
extern thread_local u32 t_touch[8][32];

inline void Touch(Bit bit) noexcept {
    if (g_on) {
        t_mask |= 1u << bit;
        ++t_touch[t_phase][bit];
    }
}

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

// ---- P0 (phase split, notes S.6): exclusive time per phase of one operation. The default phase is Record (everything not inside a scope). A PhaseScope charges the time since the last
// change to the phase that was running, so nested scopes do not count twice. Timer cost is calibrated at the first report and taken off the buckets.
enum Phase : u32 {
    PhaseRecord = 0,   ///< everything outside a scope: recording, descriptor writes, state, the draw call itself
    PhaseBufResolve,   ///< ObtainBuffer / BufferFor / SynchronizeMemory decisions (resolve side)
    PhaseImgResolve,   ///< TextureCache::Find*, GetSampler
    PhaseImgRefresh,   ///< TextureCache::RefreshImage (image upload and layout)
    PhaseUpload,       ///< the upload part of SynchronizeMemory (copy into staging / the arena, copy record)
    PhaseBarrier,      ///< Runtime::FlushBarriers
    PhaseMixed,        ///< inside BindBuffersBody / BindTextures / BindSamplers / vertex / index binding, outside the scopes above: lookup glue and descriptor assembly
    PhaseAnnounce,     ///< the announce batch
    PhaseCount,
};
enum Counter : u32 {
    CtBarrierQuery = 0, ///< Runtime::IsBufferAccessed
    CtTickIsFree,       ///< SchedulerCore::IsFree
    CtTickCurrent,      ///< SchedulerCore::CurrentTick
    CtGpuWrittenCheck,  ///< BufferCache::IsRangeGpuWritten
    CtImageOverlap,     ///< TextureCache::AnyImageOverlaps
    CounterCount,
};

extern thread_local u64 t_phase_at;
extern thread_local u64 t_phase_ns[PhaseCount];
extern thread_local u32 t_phase_scopes[PhaseCount];
extern thread_local u32 t_counts[PhaseCount][CounterCount];

[[nodiscard]] inline u64 PhaseNow() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

class PhaseScope {
public:
    explicit PhaseScope(Phase phase) noexcept : active{g_on} {
        if (active) {
            const u64 now = PhaseNow();
            t_phase_ns[t_phase] += now - t_phase_at;
            previous = t_phase;
            t_phase = phase;
            t_phase_at = now;
            ++t_phase_scopes[phase];
        }
    }
    ~PhaseScope() {
        if (active) {
            const u64 now = PhaseNow();
            t_phase_ns[t_phase] += now - t_phase_at;
            t_phase = previous;
            t_phase_at = now;
        }
    }
    PhaseScope(const PhaseScope&) = delete;
    PhaseScope& operator=(const PhaseScope&) = delete;

private:
    u32 previous{};
    bool active;
};

inline void Count(Counter counter) noexcept {
    if (g_on) {
        ++t_counts[t_phase][counter];
    }
}

inline void Begin() noexcept {
    t_mask = 0;
    if (g_on) {
        for (u32 i = 0; i < PhaseCount; ++i) {
            t_phase_ns[i] = 0;
            t_phase_scopes[i] = 0;
        }
        t_phase = PhaseRecord;
        t_phase_at = PhaseNow();
    }
}

[[nodiscard]] inline u32 Current() noexcept {
    return t_mask;
}

/// What kind of operation: 0 graphics draw, 1 depth-only draw, 2 dispatch, 3 closure.
/// Raster thread only.
void Record(u32 kind, u32 queue_class, const char* file, u32 line, u32 mask, u64 ns);

/// Publishes `/stats` `split_census` for the interval and starts a new one. Raster thread, every ~5 s (does nothing when the census is off).
void PublishAndReset();

} // namespace Core::SplitCensus
