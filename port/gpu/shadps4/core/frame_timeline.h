// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

#include "common/types.h"

namespace Core::FrameTimeline {

/// Diagnostics (study): timestamps of the events that make up a frame, from any thread, in one ring read by the debug server (/timeline).
enum Type : u32 {
    FirstHook = 1,    ///< the first draw / dispatch hook after a submit (render thread)
    Submit = 2,       ///< sceGnmSubmit*: arg = time of the last hook before it
    SubmitFlip = 3,   ///< sceGnmSubmitAndFlip*: arg = time of the last hook before it
    FlipCp = 4,       ///< VideoOutDriver::SubmitFlip entry: arg = 1 for an end of pipe flip
    FlipPrepared = 5, ///< SubmitFlipInternal done (frame prepared, request queued)
    PresentStart = 6, ///< the present thread takes the request (a vblank tick)
    PresentEnd = 7,   ///< the present thread is done with the flip
    CpWait = 8,       ///< the command processor waited on a memory label: arg = duration ns | site << 56, timestamp = end of the wait
    FlipWait = 9,     ///< the command processor slept on the flip label (WaitFlipDone): arg = duration ns, timestamp = end of the wait
    GameWait = 10,    ///< a guest thread blocked: arg = duration ns | kind << 56 (1 condition variable, 2 semaphore, 3 sleep, 4 event queue), arg2 = the object
    CpIdleQueued = 12, ///< the command processor queued the GpuIdle signal (it parsed everything of the submit)
    RasterIdleSignal = 13, ///< the raster thread executed the GpuIdle signal (all queued operations before it are done)
    SubmitDone = 15,   ///< sceGnmSubmitDone / sceGnmRequestFlipAndSubmitDone: arg = 1 if the submission lock was set, +2 if the GPU was idle
    DingDongWait = 16, ///< sceGnmDingDong waited for the submission lock: arg = duration ns
    StreamPublish = 14, ///< a span of the command buffer was handed to the command processor while the frame is recorded: arg = dwords
    EopParsed = 17,    ///< the command processor parsed an end of pipe event of the graphics queue (arg = label address)
    EopSignaled = 18,  ///< the end of pipe signal ran: arg = ns from the parse to the start on the raster thread, arg2 = ns from there to the signal
    GameSignal = 11,  ///< a guest thread signalled: arg = kind (5 condition variable signal, 6 broadcast, 7 semaphore post), arg2 = the object
    // Capture only (documents/tlg/frame-timeline-capture-design.md): the types below are recorded while a capture runs.
    StageSpan = 20,   ///< a span on a stage thread, written when it ends: arg = duration ns | stage << 56, arg2 = detail (see CaptureStage)
    GpuCommandBuffer = 21, ///< a command buffer ran on the GPU: ts = end (CPU clock), arg = duration ns, arg2 = submission tick
    GpuPass = 22,     ///< a render pass or dispatch ran on the GPU: ts = end (CPU clock), arg = duration ns | kind << 56, arg2 = draws | width << 24 | height << 44
    ThreadCpu = 23,   ///< sampler (1 kHz): arg = CPU cycles the thread named in `thread` ran since the previous sample, taken at ts; cycles / cycles_per_ns (in the .json) = ns
    GuestSpan = 25,   ///< a timed guest function (spans file): ts = end, arg = duration ns | span id << 48, arg2 = start ns
    GpuClockSync = 24, ///< calibration: arg = GPU timestamp ticks, ts = the CPU clock at the same moment, arg2 = timestamp period (ns * 1000)
};

/// The stages of StageSpan (arg >> 56).
enum CaptureStage : u32 {
    StageFrontOp = 1,      ///< front: one operation planned and pushed (arg2: 1 draw, 2 dispatch)
    StageFrontQueueFull = 2, ///< front: waited for room in the raster queue
    StageFrontDrain = 3,   ///< front: waited for queued memory writes (DrainRasterWrites)
    StageRasterOp = 4,     ///< raster: one draw / dispatch / closure (arg2: 1 draw, 2 dispatch, 3 closure)
    StageRasterIdle = 5,   ///< raster: waited for work
    StageRasterPass = 6,   ///< raster: the draws of one render pass (arg2 = draw count)
    StageEncoderReplay = 7, ///< encoder: replayed a command list into a command buffer
    StageEncoderSubmit = 8, ///< encoder: queue submit
    StageEncoderIdle = 9,  ///< encoder: waited for work
    StageRasterHandoffWait = 10, ///< raster: waited for the encoder's queue to have room
};

struct Event {
    std::atomic<u64> ns{};
    std::atomic<u64> arg{};
    std::atomic<u64> arg2{};
    std::atomic<u32> type{};
    std::atomic<u32> thread{};
};

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId();
extern "C" __declspec(dllimport) int __stdcall QueryThreadCycleTime(void* thread, unsigned long long* cycles);
#endif

constexpr u32 RingSize = 65536;
inline std::array<Event, RingSize> ring{};
inline std::atomic<u64> head{};
inline std::atomic<u64> last_hook_ns{};
inline std::atomic<bool> armed{};

/// The capture buffer: while a capture runs every event is also appended here (a plain record, 32 bytes). Filled by any thread, read after the capture stops.
struct CapEvent {
    u64 ns;
    u64 arg;
    u64 arg2;
    u32 type;
    u32 thread;
};
inline std::atomic<bool> capture_active{};
inline CapEvent* capture_events = nullptr;
inline u64 capture_capacity = 0;
inline std::atomic<u64> capture_count{};
inline void CaptureAppend(Type type, u64 arg, u64 ns, u64 arg2, u32 thread) {
    const u64 at = capture_count.fetch_add(1, std::memory_order_relaxed);
    if (at < capture_capacity) {
        capture_events[at] = {ns, arg, arg2, static_cast<u32>(type), thread};
    }
}
/// Start a capture of `seconds` seconds into the file timeline_<tag>.bin (and .json) in the log directory. False if one is running.
bool StartCapture(u32 seconds, const std::string& tag);
/// A short JSON text: whether a capture runs, events so far, the last file written.
std::string CaptureStatus();
/// Stop a running capture now (it is written as if its time had run out). False when none runs.
bool StopCapture();
/// Name of a guest span id, written into the capture file (see guest_hooks.cpp, the spans file).
void RegisterSpan(u32 id, const std::string& name);
/// Names for the capture file: an object (a guest wait object, a stage) gets a text.
void CaptureNote(u64 id, const char* text);

inline u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// CPU cycles the calling thread has run so far (not time spent blocked or descheduled).
inline u64 ThreadCycles() {
#ifdef _WIN32
    unsigned long long cycles = 0;
    QueryThreadCycleTime(reinterpret_cast<void*>(static_cast<intptr_t>(-2)), &cycles);
    return cycles;
#else
    return 0;
#endif
}

inline u32 ThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return static_cast<u32>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

inline void Record(Type type, u64 arg, u64 ns, u64 arg2 = 0) {
    const u32 thread = ThreadId();
    auto& e = ring[head.fetch_add(1, std::memory_order_relaxed) & (RingSize - 1)];
    e.type.store(0, std::memory_order_relaxed);
    e.ns.store(ns, std::memory_order_relaxed);
    e.arg.store(arg, std::memory_order_relaxed);
    e.arg2.store(arg2, std::memory_order_relaxed);
    e.thread.store(thread, std::memory_order_relaxed);
    e.type.store(type, std::memory_order_release);
    if (capture_active.load(std::memory_order_relaxed)) [[unlikely]] {
        CaptureAppend(type, arg, ns, arg2, thread);
    }
}

inline void Mark(Type type, u64 arg = 0) {
    Record(type, arg, Now());
}

/// One draw / dispatch hook on the render thread.
inline void Hook() {
    last_hook_ns.store(Now(), std::memory_order_relaxed);
    if (armed.load(std::memory_order_relaxed) && armed.exchange(false)) {
        Record(FirstHook, 0, last_hook_ns.load(std::memory_order_relaxed), ThreadCycles());
    }
}

inline void Submitted(Type type) {
    Record(type, last_hook_ns.load(std::memory_order_relaxed), Now(), ThreadCycles());
    armed.store(true, std::memory_order_relaxed);
}

/// An event with a duration, written when it ends.
struct Span {
    explicit Span(Type type_, u64 site_ = 0) : type{type_}, site{site_}, start{Now()} {}
    ~Span() {
        const u64 end = Now();
        Record(type, (end - start) | (site << 56), end);
    }
    Type type;
    u64 site;
    u64 start;
};

/// A blocking call of a guest thread; only waits of at least 100 us are recorded.
struct WaitSpan {
    WaitSpan(u64 kind_, const void* object_) : kind{kind_}, object{reinterpret_cast<u64>(object_)}, start{Now()} {}
    ~WaitSpan() {
        const u64 end = Now();
        if (end - start >= 100'000) {
            Record(GameWait, (end - start) | (kind << 56), end, object);
        }
    }
    u64 kind;
    u64 object;
    u64 start;
};

/// A span on a stage thread (capture only): written when it ends, only if it lasted `min_ns` or more. `detail` can be set until then.
struct StageScope {
    StageScope(CaptureStage stage_, u64 detail_ = 0, u64 min_ns_ = 5'000)
        : active{capture_active.load(std::memory_order_relaxed)}, stage{stage_}, detail{detail_}, min_ns{min_ns_}, start{active ? Now() : 0} {}
    ~StageScope() {
        if (active) {
            const u64 end = Now();
            if (end - start >= min_ns) {
                Record(StageSpan, (end - start) | (u64(stage) << 56), end, detail);
            }
        }
    }
    bool active;
    CaptureStage stage;
    u64 detail;
    u64 min_ns;
    u64 start;
};

inline void Signal(u64 kind, const void* object) {
    Record(GameSignal, kind, Now(), reinterpret_cast<u64>(object));
}

} // namespace Core::FrameTimeline
