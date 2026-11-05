// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The synchronization state of the command processor (G4c, documents/tlg/native plan/executor-consumer-plan.md): the label timeline of the fences the command processor has
// recorded (what a wait on a label may pass on while the fence write is held back), the fence latency statistics, the reports of a wait that hangs, the writes of a fence from
// another thread, and the DMA and write-data operations of the command stream. The code moved here from Liverpool unchanged; what it needs from the rasterizer comes in as
// arguments or through `DmaHost`.

#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"
#include "native/native_sync.h"
#include "video_core/readback_stats.h"

namespace Core::NativeExecutorSync {

/// The label (and the fence recorded last for it) a wait matched.
struct FenceMatch {
    bool satisfied{};
    VAddr label_address{}; ///< The matched label and the fence recorded last for it.
    u64 seq{};
    u64 gpu_tick{};
    u32 queue{};
    u32 value{};
    u32 initial{};
};

/// Label timeline: per fence label, the latest value a fence packet the command processor
/// already recorded will write, while fence writes are held back (async readback fences
/// write them once the GPU finished and the readbacks landed). A command stream wait only
/// needs its producer to be recorded ahead of it in the single Vulkan queue, not executed,
/// so it may pass on that value. When everything recorded has landed, guest memory is
/// authoritative again, and so it is as soon as someone else writes the label meanwhile.
class LabelTimeline {
public:
    /// Records a fence the command processor parsed; returns its sequence number.
    u64 Record(VAddr address, u64 value, u32 num_bytes, u32 queue);
    /// Raster thread: the scheduler tick holding the work before fence `seq` of a label.
    void SetTick(VAddr address, u64 seq, u64 tick);
    [[nodiscard]] u64 Tick(VAddr address, u64 seq);
    /// Front thread: the raster queue position of the closure that signals fence `seq` (compute lane: operations that wait for the label run after it).
    void SetPosition(VAddr address, u64 seq, u64 position);
    /// The queue position a command stream wait on this label depends on: 0 = nothing queued (landed already), ~0 = unknown (everything before the wait).
    [[nodiscard]] u64 Position(VAddr address);
    /// The write of fence `seq` landed.
    void Retire(VAddr address, u64 seq);
    /// What the timeline says about the label a wait polls (not satisfied, and no label, when the wait must poll memory).
    [[nodiscard]] FenceMatch Match(const NativeSync::LabelWait& wait);
    /// A fence write to a label overlapping the range is still held back.
    [[nodiscard]] bool HeldBack(VAddr begin, u32 num_bytes);
    /// Logging (capped): a write queued behind a held back fence, a compute ring that passed a wait on a recorded fence.
    void NoteDeferredWrite(VAddr begin, u32 num_bytes, u32 queue);
    void NoteComputePass(const NativeSync::LabelWait& wait, const FenceMatch& match, u32 queue);
    /// The pending fence writes as text (hang report).
    [[nodiscard]] std::string PendingText();

private:
    struct LabelState {
        VAddr address;
        u32 num_bytes;
        u32 queue; ///< Queue of the latest fence: 0 = graphics, n = compute ring n - 1.
        u64 latest_value;
        u64 latest_seq; ///< Order in which the fence packets were processed.
        u64 gpu_tick;   ///< Scheduler tick holding the GPU work before the latest fence.
        u64 gpu_position; ///< Raster queue position of the closure of the latest fence (0 until SetPosition).
        u32 pending;    ///< Recorded fences whose write hasn't landed yet.
        bool overridden;
        /// Guest memory before the first held back fence, then every value recorded since.
        std::vector<u64> known_values;
    };
    static constexpr size_t MaxKnownLabelValues = 64;
    std::mutex mutex;
    std::vector<LabelState> label_states;
    u64 next_fence_seq{};
    std::unordered_set<VAddr> logged_compute_passes;
    std::unordered_set<VAddr> logged_overrides;
    std::unordered_set<VAddr> logged_deferred_writes;
};

/// Fence latency per FenceKind (index = bit position): the time from parsing the fence packet
/// to the raster thread starting it ("queue"), from there to the moment its signal is about to
/// run, after the GPU and the readbacks ("gpu"), and the sum. Published by the debug server.
class FenceLatency {
public:
    void Note(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns, u64 front_ns = 0);
    /// Per kind: counts and averages since the last report (the counters are reset).
    [[nodiscard]] nlohmann::json Report();

private:
    struct Stat {
        std::atomic<u64> count{};
        std::atomic<u64> queue_ns{};
        std::atomic<u64> front_ns{}; ///< parse to the end of the push (the front thread blocked in a full queue)
        std::atomic<u64> gpu_ns{};
        std::atomic<u64> total_ns{};
        std::atomic<u64> max_total_ns{};
    };
    std::array<Stat, VideoCore::NumFenceKinds> stats;
};

/// Logs a command stream wait that has been blocked for more than 2 seconds (once per wait: `reported`).
void ReportLongWait(LabelTimeline& timeline, const NativeSync::LabelWait& wait, const char* queue, u64 wait_start, bool& reported);
/// Logs a MemSemaphore/Rewind wait that has been blocked for more than 2 seconds.
void ReportLongSpin(const char* what, const void* address, u64 wait_start, bool& reported);

/// Fence writes (the callbacks `Fence::Signal` takes): through the memory backing, or none at all (the interrupt only).
void WriteFenceMemory(void* address, u64 data, u32 num_bytes);
void NoFenceWrite(void*, u64, u32);
/// Writes from outside the GPU thread. Memory without a GPU mapping is never page protected by the tracker, so it can be written directly (fence labels often live in plain game
/// memory). `not_mapped(address, bytes)` says the range is known to have no GPU mapping. Returns false when the write must be done on the GPU thread instead.
bool TryWriteOffThread(void* address, const void* data, u32 num_bytes, const std::function<bool(VAddr, u32)>& not_mapped);
/// The write of a deferred write-data packet, done on the GPU thread (memory backing first).
void WriteBackingOrMemory(void* address, const void* data, u32 num_bytes);
/// A command processor write-data packet that lands in order with the draws: the write, and the timeline record for the capture.
void ApplyCpWrite(void* address, const std::vector<u32>& words, u32 queue);

/// The rasterizer's part of a command processor DMA.
class DmaHost {
public:
    virtual ~DmaHost() = default;
    virtual void FillBuffer(u64 destination, u32 bytes, u32 fill, bool destination_is_gds) = 0;
    virtual void CopyBuffer(u64 destination, u64 source, u32 bytes, bool destination_is_gds, bool source_is_gds) = 0;
};
/// What a DMA needs before it is queued: false when there is nothing to do. Marks the next queued write for the drain audit.
[[nodiscard]] bool PrepareDma(const NativeSync::DmaCopy& dma);
/// The DMA itself, run on the raster thread.
void RunDma(const NativeSync::DmaCopy& dma, DmaHost& host);

} // namespace Core::NativeExecutorSync
