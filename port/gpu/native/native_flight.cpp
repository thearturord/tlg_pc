// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_flight.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#undef near
#endif

namespace Core::Flight {

namespace {

const char* const kind_names[] = {
#define X(id, text) text,
    TLG_FLIGHT_KINDS(X)
#undef X
};

std::mutex register_mutex;

u32 CurrentThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return 0;
#endif
}

u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// The processor's time stamp counter against the steady clock, taken when the first thread registers.
struct Calibration {
    u64 tick0;
    u64 ns0;
};
Calibration start{};

const bool env_read = [] {
    const char* e = std::getenv("TLG_FLIGHT");
    if (e != nullptr && *e == '0') {
        g_enabled.store(false);
    }
    return true;
}();

} // namespace

const char* KindName(u32 kind) {
    return kind < KindCount ? kind_names[kind] : "?";
}

Ring* RegisterThisThread() {
    if (t_ring != nullptr) {
        return t_ring;
    }
    std::scoped_lock lock{register_mutex};
    const u32 slot = g_ring_count.load(std::memory_order_relaxed);
    if (slot >= MaxThreads) {
        return nullptr; // a thread beyond the table writes no events
    }
    if (slot == 0) {
        start = {Tick(), SteadyNs()};
    }
    auto* ring = new Ring();
    ring->slot = slot;
    ring->thread_id = CurrentThreadId();
    std::snprintf(ring->name, sizeof(ring->name), "thread %u", ring->thread_id);
    g_rings[slot] = ring;
    g_ring_count.store(slot + 1, std::memory_order_release);
    t_ring = ring;
    return ring;
}

void NameThisThread(const char* name) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    Ring* ring = t_ring != nullptr ? t_ring : RegisterThisThread();
    if (ring != nullptr) {
        std::snprintf(ring->name, sizeof(ring->name), "%s", name);
    }
}

double MsAgo(u64 tick) {
    const u64 tick_now = Tick();
    const u64 ns_now = SteadyNs();
    double ticks_per_ns = 1.0;
    if (ns_now > start.ns0 + 5'000'000 && tick_now > start.tick0) {
        ticks_per_ns = static_cast<double>(tick_now - start.tick0) / static_cast<double>(ns_now - start.ns0);
    }
    return -static_cast<double>(static_cast<s64>(tick_now - tick)) / ticks_per_ns / 1e6;
}

u64 EventsWritten() {
    u64 total = 0;
    const u32 count = g_ring_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        total += g_rings[i]->head.load(std::memory_order_relaxed);
    }
    return total;
}

std::string Report(u32 per_thread, bool bench, double max_age_ms) {
    std::string out = "{";
    if (bench) {
        // the events go to a ring of their own that no report reads, so the thread's own ring stays clean
        static Ring* scratch = new Ring();
        scratch->slot = 0;
        Ring* const own = t_ring;
        t_ring = scratch;
        const bool was = g_enabled.exchange(true);
        constexpr u32 events = 1'000'000;
        const u64 t0 = SteadyNs();
        for (u32 i = 0; i < events; ++i) {
            Add(Bench, i, i, i);
        }
        const u64 t1 = SteadyNs();
        g_enabled.store(was);
        t_ring = own;
        out += fmt::format("\"bench_ns_per_event\": {:.2f}, \"bench_events\": {}, ", static_cast<double>(t1 - t0) / events, events);
    }
    const u64 tick_now = Tick();
    const u64 ns_now = SteadyNs();
    // ticks per ns from the span since the first thread registered (at least a few ms long, else 1)
    double ticks_per_ns = 1.0;
    if (ns_now > start.ns0 + 5'000'000 && tick_now > start.tick0) {
        ticks_per_ns = static_cast<double>(tick_now - start.tick0) / static_cast<double>(ns_now - start.ns0);
    }
    const auto ms_ago = [&](u64 tick) { return -static_cast<double>(static_cast<s64>(tick_now - tick)) / ticks_per_ns / 1e6; };
    out += fmt::format("\"enabled\": {}, \"threads\": {}, \"events_written\": {}, \"ring_events\": {}, \"ticks_per_ns\": {:.3f}, \"unit\": \"event time = ms before this report\", ",
                       g_enabled.load() ? "true" : "false", g_ring_count.load(), EventsWritten(), RingEvents, ticks_per_ns);

    struct Line {
        u64 tick;
        const Entry* entry;
    };
    std::vector<Line> lines;
    out += "\"thread_list\": [";
    const u32 count = g_ring_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        const Ring& ring = *g_rings[i];
        const u64 head = ring.head.load(std::memory_order_acquire);
        const char* scope = ring.scope.load(std::memory_order_relaxed);
        out += fmt::format("{}{{\"slot\": {}, \"name\": \"{}\", \"thread_id\": {}, \"doing\": \"{}\", \"events\": {}}}", i == 0 ? "" : ", ", i, ring.name, ring.thread_id,
                           scope != nullptr ? scope : "", head);
        const u64 first = head > per_thread ? head - per_thread : 0;
        for (u64 h = first; h < head && head - h <= RingEvents; ++h) {
            const Entry& entry = ring.events[h & (RingEvents - 1)];
            lines.push_back({entry.tick, &entry});
        }
    }
    out += "], \"events\": [";
    std::sort(lines.begin(), lines.end(), [](const Line& x, const Line& y) { return x.tick < y.tick; });
    bool first_line = true;
    for (const Line& line : lines) {
        const Entry& e = *line.entry;
        if (max_age_ms > 0.0 && -ms_ago(e.tick) > max_age_ms) {
            continue;
        }
        out += fmt::format("{}[{:.3f}, {}, \"{}\", {}, {}, {}, {}]", first_line ? "" : ", ", ms_ago(e.tick), e.slot, KindName(e.kind), e.seq, e.a, e.b, e.flags);
        first_line = false;
    }
    out += "], \"columns\": \"ms before report, thread slot, kind, seq, a, b, flags\"}";
    return out;
}

// ---- L2: crash context -----------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace {

char log_directory[512] = "user/log";

/// One output file built in a static buffer (no heap, so a crashed thread that holds the allocator's lock cannot block the report).
struct Out {
    char* data;
    size_t size;
    size_t used = 0;
    void Put(const char* format, ...) __attribute__((format(printf, 2, 3))) {
        if (used + 1 >= size) {
            return;
        }
        va_list args;
        va_start(args, format);
        const int n = std::vsnprintf(data + used, size - used, format, args);
        va_end(args);
        if (n > 0) {
            used = std::min(used + static_cast<size_t>(n), size - 1);
        }
    }
};

constexpr u32 TailPerThread = 64;
constexpr u32 TailMax = MaxThreads * TailPerThread;
constexpr u32 RareMax = 8;
char text_buffer[1 << 20];
char json_buffer[1 << 20];
const Entry* tail[TailMax];
const Entry* rare[RareMax];
const Entry* last_of_kind[64];
std::atomic<u32> crash_counter{0};

bool WriteWholeFile(const char* path, const char* data, size_t size) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD written = 0;
    const bool ok = WriteFile(file, data, static_cast<DWORD>(size), &written, nullptr) != 0;
    CloseHandle(file);
    return ok;
#else
    FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    return ok;
#endif
}

} // namespace

void SetLogDirectory(const char* utf8_path) {
    std::snprintf(log_directory, sizeof(log_directory), "%s", utf8_path);
}

void DumpCrash(const char* reason, const char* note) {
    const u32 number = crash_counter.fetch_add(1);
    if (number >= 8) {
        return; // a loop of device lost reports must not fill the disk
    }
    const u64 tick_now = Tick();
    const u64 ns_now = SteadyNs();
    double ticks_per_ns = 1.0;
    if (ns_now > start.ns0 + 5'000'000 && tick_now > start.tick0) {
        ticks_per_ns = static_cast<double>(tick_now - start.tick0) / static_cast<double>(ns_now - start.ns0);
    }
    const auto ms_ago = [&](u64 tick) { return -static_cast<double>(static_cast<s64>(tick_now - tick)) / ticks_per_ns / 1e6; };

    char stamp[64];
#ifdef _WIN32
    SYSTEMTIME time;
    GetLocalTime(&time);
    std::snprintf(stamp, sizeof(stamp), "%04u%02u%02u_%02u%02u%02u", time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
#else
    std::snprintf(stamp, sizeof(stamp), "%llu", static_cast<unsigned long long>(ns_now));
#endif

    Out text{text_buffer, sizeof(text_buffer)};
    Out json{json_buffer, sizeof(json_buffer)};
    const u32 count = g_ring_count.load(std::memory_order_acquire);
    const Ring* mine = t_ring;
    text.Put("TLG crash context %s: %s\n", stamp, reason);
    if (note != nullptr) {
        text.Put("  %s\n", note);
    }
    text.Put("  threads with a ring: %u, events written: %llu, reporting thread: %s\n\n", count, static_cast<unsigned long long>(EventsWritten()), mine != nullptr ? mine->name : "(none)");

    // 1. what every thread was doing
    text.Put("== threads ==\n");
    json.Put("{\"reason\": \"%s\", \"stamp\": \"%s\", \"unit\": \"event time = ms before the crash\", \"thread_list\": [", reason, stamp);
    for (u32 i = 0; i < count; ++i) {
        const Ring& ring = *g_rings[i];
        const char* scope = ring.scope.load(std::memory_order_relaxed);
        text.Put("  slot %2u  %-40s tid %-6u events %-8llu doing: %s%s\n", i, ring.name, ring.thread_id, static_cast<unsigned long long>(ring.head.load(std::memory_order_relaxed)),
                 scope != nullptr ? scope : "-", &ring == mine ? "   <-- reporting thread" : "");
        json.Put("%s{\"slot\": %u, \"name\": \"%s\", \"thread_id\": %u, \"doing\": \"%s\", \"events\": %llu, \"reporting\": %s}", i == 0 ? "" : ", ", i, ring.name, ring.thread_id,
                 scope != nullptr ? scope : "", static_cast<unsigned long long>(ring.head.load(std::memory_order_relaxed)), &ring == mine ? "true" : "false");
    }
    json.Put("], ");

    // 2. gather: the last event of every kind, the last rare events, the tail of every thread
    std::memset(last_of_kind, 0, sizeof(last_of_kind));
    u32 tail_count = 0;
    u32 rare_count = 0;
    for (u32 i = 0; i < count; ++i) {
        const Ring& ring = *g_rings[i];
        const u64 head = ring.head.load(std::memory_order_acquire);
        const u64 first_index = head > RingEvents ? head - RingEvents : 0;
        for (u64 h = first_index; h < head; ++h) {
            const Entry& e = ring.events[h & (RingEvents - 1)];
            if (e.kind >= 64 || e.kind == Bench || e.kind == None) {
                continue;
            }
            if (last_of_kind[e.kind] == nullptr || last_of_kind[e.kind]->tick < e.tick) {
                last_of_kind[e.kind] = &e;
            }
            if (e.kind == RareEvent) {
                // keep the RareMax latest by time
                if (rare_count < RareMax) {
                    rare[rare_count++] = &e;
                } else {
                    u32 oldest = 0;
                    for (u32 k = 1; k < RareMax; ++k) {
                        if (rare[k]->tick < rare[oldest]->tick) {
                            oldest = k;
                        }
                    }
                    if (rare[oldest]->tick < e.tick) {
                        rare[oldest] = &e;
                    }
                }
            }
            if (h + TailPerThread >= head && tail_count < TailMax) {
                tail[tail_count++] = &e;
            }
        }
    }
    std::sort(tail, tail + tail_count, [](const Entry* x, const Entry* y) { return x->tick < y->tick; });
    std::sort(rare, rare + rare_count, [](const Entry* x, const Entry* y) { return x->tick < y->tick; });

    const auto line = [&](const Entry& e) {
        text.Put("  %10.3f ms  slot %2u  %-22s seq %-10llu a %-18llx b %-18llx flags %u\n", ms_ago(e.tick), e.slot, KindName(e.kind), static_cast<unsigned long long>(e.seq),
                 static_cast<unsigned long long>(e.a), static_cast<unsigned long long>(e.b), e.flags);
    };
    const auto json_line = [&](const Entry& e, bool first) {
        json.Put("%s[%.3f, %u, \"%s\", %llu, %llu, %llu, %u]", first ? "" : ", ", ms_ago(e.tick), e.slot, KindName(e.kind), static_cast<unsigned long long>(e.seq),
                 static_cast<unsigned long long>(e.a), static_cast<unsigned long long>(e.b), e.flags);
    };

    text.Put("\n== the last event of every kind (any thread) ==\n");
    json.Put("\"last_of_kind\": [");
    bool first = true;
    for (u32 k = 1; k < KindCount && k < 64; ++k) {
        if (last_of_kind[k] != nullptr) {
            line(*last_of_kind[k]);
            json_line(*last_of_kind[k], first);
            first = false;
        }
    }
    json.Put("], \"rare_events\": [");
    text.Put("\n== the last %u rare events / refusals ==\n", rare_count);
    for (u32 i = 0; i < rare_count; ++i) {
        line(*rare[i]);
        json_line(*rare[i], i == 0);
    }
    text.Put("\n== the last %u events of every thread, in time order ==\n", TailPerThread);
    json.Put("], \"events\": [");
    for (u32 i = 0; i < tail_count; ++i) {
        line(*tail[i]);
        json_line(*tail[i], i == 0);
    }
    json.Put("], \"columns\": \"ms before the crash, thread slot, kind, seq, a, b, flags\"}");

    char path[640];
    std::snprintf(path, sizeof(path), "%s/crash_%s_%u.txt", log_directory, stamp, number);
    WriteWholeFile(path, text.data, text.used);
    char json_path[640];
    std::snprintf(json_path, sizeof(json_path), "%s/crash_%s_%u.flight.json", log_directory, stamp, number);
    WriteWholeFile(json_path, json.data, json.used);
    std::fputs("TLG crash context written to ", stderr);
    std::fputs(path, stderr);
    std::fputs("\n", stderr);
}

} // namespace Core::Flight

extern "C" void tlggpu_dump_crash(const char* reason, const char* note) {
    Core::Flight::DumpCrash(reason, note);
}
