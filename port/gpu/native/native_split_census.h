// SPDX-License-Identifier: GPL-2.0-or-later
// Plan S, step S0 (documents/tlg/native plan/native-data-path-notes.md, "Plan S: the thread split"): which shared state does each raster thread operation touch?
//
// Off by default (`TLG_SPLIT_CENSUS=1`). A thread-local bit mask is cleared before every operation of the raster thread; the entry points of the shared caches (buffer cache, texture
// cache, pipeline cache, barrier state, indexed data, stream buffer, label timeline, the command list of the scheduler) set their bit. After the operation the raster thread adds
// (group, mask, time) to a table. The report shows, per group, how long the operations took and which mask they had, and the derived number the split depends on: how much time sits in
// operations that touch no cache at all (they could run on another thread with no lock at all), and how much time each cache class is touched by.

#pragma once

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

inline void Touch(Bit bit) noexcept {
    if (g_on) {
        t_mask |= 1u << bit;
    }
}

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

inline void Begin() noexcept {
    t_mask = 0;
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
