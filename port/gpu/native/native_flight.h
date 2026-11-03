// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <string>

#include "common/types.h"

/// L1 of documents/tlg/native plan/diagnostics-logging-plan.md: the flight recorder. One ring of compact events per thread (what was recorded, planned, uploaded, waited for,
/// signalled), always on unless TLG_FLIGHT=0, read by the debug server (/flight), the crash report and the glitch marker. Writing an event is one clock read, five stores and one
/// release store, no formatting, no allocation, no lock. Strings are never stored: kinds and scope names are ids and static strings.
namespace Core::Flight {

/// One line per kind: the enum, the name the reports print, and what a / b / seq hold. Add new kinds at the end of a group, never reuse an id for another meaning.
#define TLG_FLIGHT_KINDS(X)                                                                                                                                                    \
    X(None, "none")                                                                                                                                                            \
    X(Bench, "bench")                               /* the microbenchmark of /flight?bench=1 */                                                                                \
    X(RecordMade, "record made")                    /* seq; a = record kind, b = game call site (Ghidra address) */                                                           \
    X(SubmitBegin, "submit begin")                  /* a = entries or dwords, b = queue (0 graphics, 1+ compute) */                                                            \
    X(SubmitEnd, "submit end")                      /* a = queue */                                                                                                            \
    X(PlanTaken, "plan taken")                      /* seq; a = draw class, b = 1 planned from the record, 2 learning draw */                                                  \
    X(PlanLeft, "plan left to the game")            /* seq; a = reason id of the command processor's left(n) */                                                                \
    X(PassBegin, "pass begin")                      /* a = pass kind, b = render target address */                                                                             \
    X(PassEnd, "pass end")                          /* a = pass kind, b = draws in the pass */                                                                                 \
    X(Upload, "upload")                             /* a = guest address, b = size; flags = path */                                                                            \
    X(ImageRefresh, "image refreshed")              /* a = guest address, b = size; flags = class */                                                                           \
    X(GateWaitBegin, "gate wait begin")             /* a = gate id (see GateId) */                                                                                             \
    X(GateWaitEnd, "gate wait end")                 /* a = gate id, b = waited ns */                                                                                           \
    X(QueueSubmit, "queue submit")                  /* a = submission tick, b = queue stream (0 graphics, 1 compute) */                                                        \
    X(FenceSignaled, "fence signalled")             /* a = fence kind, b = address of the label written */                                                                     \
    X(FlipRequest, "flip request")                  /* a = display buffer, b = flip argument */                                                                                \
    X(GeneratorDeclined, "generator declined")      /* a = generator id, b = reason id */                                                                                      \
    X(RareEvent, "rare event")                      /* a = hash of the kind text, b = how many so far */                                                                       \
    X(Mark, "marker pressed")                       /* a = marker number, b = 1 full, 2 light */                                                                               \
    X(DeviceLost, "device lost")                    /* a = submission tick */                                                                                                  \
    X(Hook, "hook fired")                           /* a = game call site (Ghidra address), b = hook id */                                                                     \
    X(LogError, "error logged")                     /* a = hash of the source line text, b = line */                                                           \
    X(Invariant, "invariant failed")                /* seq; a = invariant id (native_invariant.h), b = how many so far */

enum Kind : u16 {
#define X(id, text) id,
    TLG_FLIGHT_KINDS(X)
#undef X
        KindCount
};

/// Gates named in GateWaitBegin / GateWaitEnd.
enum GateId : u32 {
    GateFrame = 1,   ///< the frame gate of the raster thread (frames in flight)
    GateGpuIdle = 2, ///< a wait for the GPU to be idle
    GateEop = 3,     ///< the end of pipe gate
    GatePresent = 4, ///< the present thread waits for its request
    GateQueue = 5,   ///< the front thread waits for room in the raster queue
};

const char* KindName(u32 kind);

struct Entry {
    u64 tick; ///< the processor's time stamp counter (converted to ns by the report)
    u16 kind;
    u8 slot;  ///< the thread's ring number
    u8 flags;
    u32 aux;
    u64 seq;  ///< the record the event is about, 0 if none
    u64 a;
    u64 b;
};
static_assert(sizeof(Entry) == 40);

constexpr u32 RingEvents = 16384; ///< per thread, a power of two
constexpr u32 MaxThreads = 64;

struct Ring {
    std::atomic<u64> head{0};               ///< events written so far
    std::atomic<const char*> scope{nullptr}; ///< what the thread is doing (a static string), FLIGHT_SCOPE
    char name[32]{};
    u32 thread_id{};
    u32 slot{};
    Entry events[RingEvents];
};

inline std::atomic<bool> g_enabled{true}; ///< TLG_FLIGHT=0 turns the recorder off
inline std::atomic<u32> g_ring_count{0};
inline Ring* g_rings[MaxThreads]{};
inline thread_local Ring* t_ring = nullptr;

/// Registers the calling thread (first event, or NameThisThread). Null when the table is full.
Ring* RegisterThisThread();

inline u64 Tick() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_ia32_rdtsc();
#else
    return 0;
#endif
}

/// Writes one event into the calling thread's ring.
inline void Add(Kind kind, u64 seq = 0, u64 a = 0, u64 b = 0, u8 flags = 0) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    Ring* ring = t_ring;
    if (ring == nullptr) {
        ring = RegisterThisThread();
        if (ring == nullptr) {
            return;
        }
    }
    const u64 head = ring->head.load(std::memory_order_relaxed);
    Entry& entry = ring->events[head & (RingEvents - 1)];
    entry.tick = Tick();
    entry.kind = kind;
    entry.slot = static_cast<u8>(ring->slot);
    entry.flags = flags;
    entry.aux = 0;
    entry.seq = seq;
    entry.a = a;
    entry.b = b;
    ring->head.store(head + 1, std::memory_order_release);
}

/// The name the reports print for the calling thread (the engine's thread names are given by Common::SetCurrentThreadName, which calls this).
void NameThisThread(const char* name);

/// "What is this thread doing": a static string, set and restored by an object on the stack. One store each way.
class Scope {
public:
    explicit Scope(const char* what) {
        if (!g_enabled.load(std::memory_order_relaxed)) {
            return;
        }
        ring = t_ring != nullptr ? t_ring : RegisterThisThread();
        if (ring != nullptr) {
            previous = ring->scope.load(std::memory_order_relaxed);
            ring->scope.store(what, std::memory_order_relaxed);
        }
    }
    ~Scope() {
        if (ring != nullptr) {
            ring->scope.store(previous, std::memory_order_relaxed);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Ring* ring = nullptr;
    const char* previous = nullptr;
};

#define TLG_FLIGHT_CAT2(a, b) a##b
#define TLG_FLIGHT_CAT(a, b) TLG_FLIGHT_CAT2(a, b)
#define FLIGHT_SCOPE(what) ::Core::Flight::Scope TLG_FLIGHT_CAT(flight_scope_, __LINE__) { what }

/// The /flight report: every thread (name, id, what it is doing, events written) and the latest `per_thread` events of each, merged by time, as JSON. `bench` first runs the
/// microbenchmark (one million events on the calling thread) and reports ns per event. `max_age_ms` > 0 leaves out events older than that.
std::string Report(u32 per_thread, bool bench, double max_age_ms = 0.0);

/// Milliseconds between a Tick() value and now (negative: in the past), with the calibration the reports use.
double MsAgo(u64 tick);

/// Total events written by all threads (for the stats line).
u64 EventsWritten();

/// L2: the folder the crash files go to (the debug server's start gives the user log folder; without it "user/log" next to the working directory).
void SetLogDirectory(const char* utf8_path);

/// L2: writes crash_<time>_<n>.txt and .flight.json in the log folder for the process state of this moment: why, what every thread is doing, the last event of every kind, the last rare
/// events and the last 64 events of every thread in time order. For the fault handler and the device lost branches: it takes no lock and allocates nothing (static buffers, WriteFile).
/// `note` is printed under the reason (the faulting address, for example) and may be null. At most 8 reports per run.
void DumpCrash(const char* reason, const char* note = nullptr);

} // namespace Core::Flight
