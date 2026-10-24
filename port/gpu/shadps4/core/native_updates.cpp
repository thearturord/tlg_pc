// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
#include <nlohmann/json.hpp>
#include "core/guest_hooks.h"
#include "core/native_draw.h"
#include "core/native_resources.h"
#include "core/native_uploads.h"
#include "core/native_updates.h"

namespace Core::NativeUpdates {

std::atomic<bool> enabled{[] {
    const char* e = std::getenv("TLG_UPDATE_AUDIT");
    const char* u = std::getenv("TLG_NATIVE_UPLOADS");
    const char* s = std::getenv("TLG_SNAPSHOTS");
    // the announcements are also what the native uploads and the snapshots are made of
    return (e != nullptr && *e != 0 && *e != '0') || (u != nullptr && *u != 0 && *u != '0') || (s != nullptr && *s != 0 && *s != '0');
}()};

namespace {

constexpr size_t WindowCount = 4096;
constexpr u64 NowWindowNs = 200'000'000; ///< a write this soon after the Map of its own thread belongs to that Map

struct Window {
    u64 begin{}, end{};
    u64 id{};
    u64 time_ns{};
    u64 thread{};
    u32 usage{};
    u32 type{};
    bool in_ring{}; ///< the engine handed out a window of the frame ring, not the buffer's own memory
    u32 writer{};   ///< 0: a Map, else the announcing engine function
};

std::mutex mutex;
std::array<Window, WindowCount> windows{};
u64 window_count = 0;

/// Which thread announces: the thread of `RenderDev::BeginRecording` (the render thread, where the frame's commands are recorded) or another one. Per announcer 0..7;
/// 0 is a Map. Threads other than the render thread are numbered in the order they first announce.
std::atomic<u64> render_thread{0};
std::array<std::atomic<u64>, 8> announced_on_render_thread{}, announced_elsewhere{};
std::mutex thread_mutex;
std::map<u64, u32> other_threads;                                    ///< thread key -> number
std::array<std::array<std::atomic<u64>, 8>, 8> other_thread_counts{}; ///< [announcer][thread number], 8 threads at most
void NoteAnnouncerThread(u32 writer) {
    const u64 thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const u64 render = render_thread.load(std::memory_order_relaxed);
    if (render != 0 && thread == render) {
        announced_on_render_thread[writer & 7].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    announced_elsewhere[writer & 7].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{thread_mutex};
    const auto [it, added] = other_threads.try_emplace(thread, static_cast<u32>(other_threads.size()));
    if (it->second < 8) {
        other_thread_counts[writer & 7][it->second].fetch_add(1, std::memory_order_relaxed);
    }
}

struct MapCounters {
    std::array<std::atomic<u64>, 8> by_type{};   ///< D3D11 map type 1..5 (0 and 6, 7 other)
    std::array<std::atomic<u64>, 4> by_usage{};  ///< 0 default 1 immutable 2 dynamic 3 staging
    std::atomic<u64> in_ring{}, own_memory{};
    std::atomic<u64> total{};
} maps;

struct FaultCounters {
    std::atomic<u64> total{};
    std::atomic<u64> own_host_writes{}; ///< written by our own native code (tags into the game's command buffers), not by the game
    std::atomic<u64> covered_now{}, covered_earlier{}, covered_other_thread{}, uncovered{};
    std::array<std::atomic<u64>, 8> covered_by_writer{}; ///< covered_now by the announcing writer (0: a Map)
    std::array<std::atomic<u64>, 16> uncovered_by_class{};
    std::array<std::atomic<u64>, 16> covered_by_class{};
} faults;

struct Site {
    u64 count{};
    u64 sample_address{};
    u32 usage_class{};
};
struct PendingFault {
    u64 address, size, time_ns, thread;
    u32 usage_class;
    u64 rip, st0, st1;
};
std::array<PendingFault, 8192> pending{}; ///< faults that no announcement covered yet (a writer announces after it wrote): recent ones, newest last
u64 pending_count = 0;

struct SiteKey {
    u64 rip, st0, st1;
    bool operator<(const SiteKey& o) const {
        return std::tie(rip, st0, st1) < std::tie(o.rip, o.st0, o.st1);
    }
};
std::map<SiteKey, Site> map_sites; ///< the writers that fill the memory a Map handed out (faults inside a Map window of the same thread)
std::map<SiteKey, Site> uncovered_sites; ///< by faulting instruction and the first two stack words (the caller of a leaf function such as memcpy)

/// Faults that come without a faulting instruction are explicit invalidations (the runtime reports a protection change or a system call write): where from?
struct NoInstruction {
    u64 count{}, bytes{};
    std::array<u64, 12> stack{};
    u32 frames{};
    u64 first_address{};
};
std::map<u64, NoInstruction> no_instruction; ///< by the first frame of the host stack outside this module
std::array<std::atomic<u64>, 8> no_instruction_sizes{}; ///< size buckets: 1, <=8, <=64, <=4K, <=64K, <=1M, more

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

u64 ThreadKey() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace

namespace {
struct OpenMap {
    u64 address{}, size{};
};
std::mutex open_map_mutex;
std::map<std::pair<u64, u64>, OpenMap> open_maps; ///< (thread, resource id) -> the window of the Map that has not been unmapped yet
} // namespace

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread);

void NoteRenderThread() {
    render_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()), std::memory_order_relaxed);
    // a Map that was never unmapped (the engine keeps some pointers for the whole frame): its window is announced at the next frame start, for the thread that mapped it
    std::vector<std::pair<u64, OpenMap>> stale;
    {
        std::scoped_lock lock{open_map_mutex};
        for (const auto& [key, map] : open_maps) {
            stale.emplace_back(key.first, map);
        }
        open_maps.clear();
    }
    for (const auto& [thread, map] : stale) {
        NoteWriterRangeOf(0, map.address, map.size, thread);
    }
    NativeUploads::PublishPending(NativeDraw::NextSeq());
}

void Start() {
    enabled.store(true, std::memory_order_relaxed);
}


void NoteUnmap(u64 resource_id) {
    OpenMap map;
    {
        std::scoped_lock lock{open_map_mutex};
        const auto it = open_maps.find({ThreadKey(), resource_id});
        if (it == open_maps.end()) {
            return;
        }
        map = it->second;
        open_maps.erase(it);
    }
    NativeUploads::SnapshotTake(map.address, map.size, NativeDraw::NextSeq());
    NoteWriterRange(0, map.address, map.size);
}

void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size) {
    (void)caller;
    NoteAnnouncerThread(0);
    maps.total.fetch_add(1, std::memory_order_relaxed);
    maps.by_type[map_type < 8 ? map_type : 7].fetch_add(1, std::memory_order_relaxed);
    maps.by_usage[usage < 4 ? usage : 3].fetch_add(1, std::memory_order_relaxed);
    const bool in_ring = address != memory;
    (in_ring ? maps.in_ring : maps.own_memory).fetch_add(1, std::memory_order_relaxed);
    if (address == 0 || size == 0 || map_type == 1) {
        return;
    }
    // the written part: the requested size when the caller gave one, else everything the Map handed out; it is announced when the thread unmaps the resource
    const u64 written = requested_size != 0 && requested_size < size ? requested_size : size;
    std::scoped_lock lock{open_map_mutex};
    open_maps[{ThreadKey(), resource_id}] = {address, written};
    (void)usage, (void)in_ring;
}

void NoteWriterRange(u32 writer, u64 address, u64 size) {
    NoteWriterRangeOf(writer, address, size, ThreadKey());
}

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread) {
    if (address == 0 || size == 0) {
        return;
    }
    NoteAnnouncerThread(writer);
    NativeUploads::Announce(writer, address, size, thread == render_thread.load(std::memory_order_relaxed), NativeDraw::NextSeq());
    const u64 now = NowNs();
    std::scoped_lock lock{mutex};
    windows[window_count++ % WindowCount] = {address, address + size, 0, now, thread, 0, 0, false, writer};
    // faults of this thread inside the range that came first (the writer announces after it wrote) are covered by it
    const u64 count = std::min<u64>(pending_count, pending.size());
    for (u64 i = 0; i < count; ++i) {
        PendingFault& p = pending[(pending_count - count + i) % pending.size()];
        if (p.address == 0 || p.thread != thread || now - p.time_ns > NowWindowNs || p.address < address || p.address + p.size > address + size) {
            continue;
        }
        faults.uncovered.fetch_sub(1, std::memory_order_relaxed);
        faults.covered_now.fetch_add(1, std::memory_order_relaxed);
        faults.covered_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
        faults.uncovered_by_class[p.usage_class].fetch_sub(1, std::memory_order_relaxed);
        faults.covered_by_class[p.usage_class].fetch_add(1, std::memory_order_relaxed);
        if (const auto site = uncovered_sites.find(SiteKey{p.rip, p.st0, p.st1}); site != uncovered_sites.end() && site->second.count > 0) {
            --site->second.count;
        }
        p.address = 0;
    }
}

void NoteFault(u64 address, u64 size) {
    faults.total.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    {
        const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
        const u64 rip = GuestHooks::FaultRip();
        if (rip >= exe && rip < exe + 0x10000000) {
            faults.own_host_writes.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
#endif
    const u32 usage_class = NativeResources::UsageClassOf(address) & 15;
    const u64 now = NowNs();
    const u64 thread = ThreadKey();
    enum class Kind { None, Now, Earlier, Other } kind = Kind::None;
    u32 writer = 0;
    {
        std::scoped_lock lock{mutex};
        const u64 count = std::min<u64>(window_count, WindowCount);
        for (u64 i = 0; i < count && kind != Kind::Now; ++i) {
            const Window& w = windows[(window_count - 1 - i) % WindowCount];
            if (address < w.begin || address + size > w.end) {
                continue;
            }
            if (w.thread != thread) {
                kind = kind == Kind::None ? Kind::Other : kind;
            } else if (now - w.time_ns <= NowWindowNs) {
                kind = Kind::Now;
                writer = w.writer;
            } else if (kind != Kind::Now) {
                kind = Kind::Earlier;
            }
        }
        if (kind == Kind::None && GuestHooks::FaultRip() == 0) {
            u32 bucket = size <= 1 ? 0 : size <= 8 ? 1 : size <= 64 ? 2 : size <= 4096 ? 3 : size <= 65536 ? 4 : size <= (1u << 20) ? 5 : 6;
            no_instruction_sizes[bucket].fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
            void* frames[12] = {};
            const USHORT n = CaptureStackBackTrace(1, 12, frames, nullptr);
            const u64 key = n > 2 ? reinterpret_cast<u64>(frames[2]) : 0;
            NoInstruction& entry = no_instruction[key];
            if (entry.count++ == 0) {
                entry.frames = n;
                for (USHORT i = 0; i < n; ++i) {
                    entry.stack[i] = reinterpret_cast<u64>(frames[i]);
                }
                entry.first_address = address;
            }
            entry.bytes += size;
#endif
        }
        if (kind == Kind::Now && writer == 0) {
            Site& site = map_sites[{GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)}];
            if (site.count++ == 0) {
                site.sample_address = address;
                site.usage_class = usage_class;
            }
        }
        if (kind == Kind::None) {
            pending[pending_count++ % pending.size()] = {address, size, now, thread, usage_class, GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)};
            Site& site = uncovered_sites[{GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)}];
            if (site.count++ == 0) {
                site.sample_address = address;
                site.usage_class = usage_class;
            }
        }
    }
    switch (kind) {
    case Kind::Now:
        faults.covered_now.fetch_add(1, std::memory_order_relaxed);
        faults.covered_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Earlier:
        faults.covered_earlier.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Other:
        faults.covered_other_thread.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::None:
        faults.uncovered.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    (kind == Kind::None ? faults.uncovered_by_class : faults.covered_by_class)[usage_class].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = enabled.load();
    auto& m = j["maps_of_buffers"] = nlohmann::json::object();
    m["total"] = maps.total.load();
    m["into_the_buffers_own_memory"] = maps.own_memory.load();
    m["into_a_frame_ring_window"] = maps.in_ring.load();
    constexpr const char* type_names[8] = {"0", "read", "write", "read_write", "write_discard", "write_no_overwrite", "6", "other"};
    for (u32 i = 0; i < 8; ++i) {
        if (const u64 n = maps.by_type[i].load()) {
            m["by_map_type"][type_names[i]] = n;
        }
    }
    constexpr const char* usage_names[4] = {"default", "immutable", "dynamic", "staging"};
    for (u32 i = 0; i < 4; ++i) {
        if (const u64 n = maps.by_usage[i].load()) {
            m["by_buffer_usage"][usage_names[i]] = n;
        }
    }
    {
        constexpr const char* names[8] = {"map", "skinning", "light_constants", "command_stream_copy", "begin_recording", "buffer_creation", "texture_initialization", "7"};
        auto& t = j["announcements_by_thread"] = nlohmann::json::object();
        t["render_thread_key"] = render_thread.load();
        for (u32 w = 0; w < 7; ++w) {
            auto& o = t[names[w]] = nlohmann::json::object();
            o["on_the_render_thread"] = announced_on_render_thread[w].load();
            o["on_other_threads"] = announced_elsewhere[w].load();
            for (u32 n = 0; n < 8; ++n) {
                if (const u64 c = other_thread_counts[w][n].load()) {
                    o["other_thread_" + std::to_string(n)] = c;
                }
            }
        }
    }
    auto& f = j["cpu_write_faults"] = nlohmann::json::object();
    const u64 total = faults.total.load();
    f["total"] = total;
    f["inside_a_map_window_of_the_same_thread_within_200ms"] = faults.covered_now.load();
    f["inside_a_window_by_announcer"] = {{"map", faults.covered_by_writer[0].load()}, {"skinning_FUN_01c93730", faults.covered_by_writer[1].load()}, {"light_constants_FUN_01eeb610", faults.covered_by_writer[2].load()},
                                                       {"command_stream_memory_copy", faults.covered_by_writer[3].load()}, {"begin_recording_direct_stores", faults.covered_by_writer[4].load()},
                                                       {"buffer_creation_with_initial_data", faults.covered_by_writer[5].load()},
                                                       {"texture_initialization", faults.covered_by_writer[6].load()}};
    f["inside_an_older_map_window_of_the_same_thread"] = faults.covered_earlier.load();
    f["inside_a_map_window_of_another_thread"] = faults.covered_other_thread.load();
    f["in_no_map_window"] = faults.uncovered.load();
    f["written_by_our_own_native_code_tags_in_command_buffers"] = faults.own_host_writes.load();
    if (total != 0) {
        f["percent_announced_by_a_map_on_the_same_thread_within_200ms"] = 100.0 * static_cast<double>(faults.covered_now.load()) / static_cast<double>(total);
    }
    for (u32 i = 0; i < 16; ++i) {
        const u64 covered = faults.covered_by_class[i].load();
        const u64 uncovered = faults.uncovered_by_class[i].load();
        if (covered + uncovered != 0) {
            const std::string name = NativeResources::UsageClassNameOf(i);
            f["by_memory_class"][name] = {{"in_a_map_window", covered}, {"in_no_map_window", uncovered}};
        }
    }
    std::scoped_lock lock{mutex};
    {
#ifdef _WIN32
        const u64 base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
#else
        const u64 base = 0;
#endif
        auto& n = f["faults_without_an_instruction"] = nlohmann::json::object();
        constexpr const char* names[7] = {"1_byte", "up_to_8", "up_to_64", "up_to_4K", "up_to_64K", "up_to_1M", "more"};
        for (u32 i = 0; i < 7; ++i) {
            n["by_size"][names[i]] = no_instruction_sizes[i].load();
        }
        auto& list = n["by_host_caller"] = nlohmann::json::array();
        for (const auto& [key, e] : no_instruction) {
            nlohmann::json stack = nlohmann::json::array();
            for (u32 i = 0; i < e.frames; ++i) {
                stack.push_back(e.stack[i] - base);
            }
            list.push_back({{"faults", e.count}, {"bytes", e.bytes}, {"first_address", e.first_address}, {"exe_rva_stack", stack}});
        }
    }
    std::vector<std::pair<SiteKey, Site>> sites(uncovered_sites.begin(), uncovered_sites.end());
    std::ranges::sort(sites, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    {
        std::vector<std::pair<SiteKey, Site>> mapped(map_sites.begin(), map_sites.end());
        std::ranges::sort(mapped, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
        auto& out = f["writers_into_map_windows"] = nlohmann::json::array();
        for (size_t i = 0; i < mapped.size() && i < 32; ++i) {
            out.push_back({{"rip", mapped[i].first.rip}, {"stack0", mapped[i].first.st0}, {"stack1", mapped[i].first.st1}, {"faults", mapped[i].second.count},
                           {"sample_address", mapped[i].second.sample_address}});
        }
    }
    auto& list = f["unannounced_writes_by_faulting_instruction"] = nlohmann::json::array();
    for (size_t i = 0; i < sites.size() && i < 24; ++i) {
#ifdef _WIN32
        const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
        const u64 rip_rva = sites[i].first.rip >= exe && sites[i].first.rip < exe + 0x10000000 ? sites[i].first.rip - exe : 0;
#else
        const u64 rip_rva = 0;
#endif
        list.push_back({{"rip", sites[i].first.rip}, {"host_exe_rva", rip_rva}, {"stack0", sites[i].first.st0}, {"stack1", sites[i].first.st1}, {"faults", sites[i].second.count}, {"sample_address", sites[i].second.sample_address},
                        {"memory_class", NativeResources::UsageClassNameOf(sites[i].second.usage_class)}});
    }
    return j;
}

} // namespace Core::NativeUpdates
