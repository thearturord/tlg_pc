// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <httplib.h>
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_stream_probe.h"
#include "native/native_timeline.h"
#include "native/native_pass.h"
#include "native/native_registry.h"
#include "native/native_census.h"
#include "native/native_frame_gate.h"
#include "native/native_updates.h"
#include "native/native_drain_audit.h"
#include "native/native_uploads.h"
#include "native/native_zpass_audit.h"
#include "native/native_zpass_gen.h"
#include "native/native_gbuffer_audit.h"
#include "native/native_resources.h"
#include "core/memory.h"
#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_entry_dump.h"
#include "native/native_audit.h"
#include "video_core/renderdoc.h"
#include "native/native_readbacks.h"
#include "native/native_raycast_cpu.h"
#include "native/native_stall.h"
#include "native/shader_env.h"
#include "native/native_frame_events.h"
#include "native/native_flip.h"
#include "native/native_display.h"
#include "native/native_presenter.h"
#include "native/native_shader_env.h"
#include "native/native_skin_cpu.h"
#include "native/native_readback_policy.h"
#include "video_core/thread_profiler.h"
#include "video_core/thread_usage.h"

namespace Core::DebugServer {

// E8: the shader boundary header must compile and stay plain data
static_assert(std::is_trivially_copyable_v<Core::ShaderEnv::FixedFunctionForShader>);
static_assert(std::is_trivially_copyable_v<Core::ShaderEnv::FragmentIo>);

namespace {

struct Entry {
    std::string description;
    Handler handler;
};

std::mutex registry_mutex;
std::map<std::string, Entry> registry;
std::unique_ptr<httplib::Server> server;
std::jthread server_thread;
std::jthread snapshot_thread;
std::atomic<bool> snapshot_now{false};
std::mutex rare_mutex;
std::map<std::string, u64> rare_counts;
std::string rare_snapshot_kind; // the tag of the snapshot the next pass of the snapshot thread writes at once
std::mutex lifecycle_mutex;
std::mutex published_mutex;
std::map<std::string, nlohmann::json> published;

const auto start_time = std::chrono::steady_clock::now();
std::atomic<u64> presents{0};
constexpr u64 TimeRing = 1024;
std::array<std::atomic<u64>, TimeRing> present_times{};

constexpr u64 FaultRegions = 1u << 14; // 4 MB regions, aliased above 64 GB
std::array<std::atomic<u32>, FaultRegions> fault_regions{};
std::array<std::atomic<u64>, FaultRegions> fault_ns{};
std::atomic<u64> faults_read_at_ns{0};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start_time)
                                .count());
}

u64 ParseNumber(const Params& params, const std::string& key, u64 fallback) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return fallback;
    }
    return std::stoull(it->second, nullptr, 0);
}

double FpsOver(double seconds) {
    const u64 count = presents.load(std::memory_order_acquire);
    const u64 now = NowNs();
    const u64 window = static_cast<u64>(seconds * 1e9);
    u64 frames = 0;
    u64 oldest = now;
    for (u64 i = 0; i < std::min<u64>(count, TimeRing); ++i) {
        const u64 t = present_times[(count - 1 - i) & (TimeRing - 1)].load(std::memory_order_relaxed);
        if (t == 0 || now - t > window) {
            break;
        }
        ++frames;
        oldest = t;
    }
    if (frames < 2) {
        return 0.0;
    }
    return static_cast<double>(frames - 1) / (static_cast<double>(now - oldest) / 1e9);
}

nlohmann::json Status(const Params&) {
    nlohmann::json j;
    j["uptime_s"] = static_cast<double>(NowNs()) / 1e9;
    j["presents"] = presents.load();
    j["fps_1s"] = FpsOver(1.0);
    j["fps_5s"] = FpsOver(5.0);
    j["game_serial"] = MemoryPatcher::g_game_serial;
    return j;
}

nlohmann::json EntryDumpEndpoint(const Params& params) {
    if (params.find("frames") != params.end()) {
        Core::EntryDump::Request(static_cast<u32>(std::min<u64>(ParseNumber(params, "frames", 1), 16)));
    }
    return Core::EntryDump::Status();
}

nlohmann::json Native(const Params&) {
    auto& live = NativeDraw::Live();
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    nlohmann::json j;
    j["note"] = "cumulative counters of the native renderer since start; take differences for rates";
    j["mode"] = policy.native_draw_mode.load();
    j["exec"] = policy.native_draw_exec.load();
    j["audit_every"] = policy.native_audit_every.load();
    j["sdk_strip"] = policy.native_sdk_strip.load();
    j["gfx_tags"] = live.gfx_tags.load();
    j["gfx_exec"] = live.gfx_exec.load();
    j["compute_tags"] = live.compute_tags.load();
    j["compute_exec"] = live.compute_exec.load();
    j["asc_exec"] = live.asc_exec.load();
    j["asc_packet"] = live.asc_packet.load();
    j["asc_no_record"] = live.asc_no_record.load();
    j["fill_exec"] = live.fill_exec.load();
    j["mode_switch_exec"] = live.mode_switch_exec.load();
    j["asc_barrier_exec"] = live.asc_barrier_exec.load();
    j["clear_exec"] = live.clear_exec.load();
    j["asc_begin_stripped"] = live.asc_begin_stripped.load();
    j["counter_readback"] = live.counter_readback.load();
    j["async_copy_exec"] = live.async_copy_exec.load();
    j["device_begin_stripped"] = live.device_begin_stripped.load();
    j["sync_units_stripped"] = live.sync_units_stripped.load();
    j["sync_acquires_stripped"] = live.sync_acquires_stripped.load();
    j["sync_events_stripped"] = live.sync_events_stripped.load();
    j["barrier_calls_skipped"] = live.barrier_calls_skipped.load();
    {
        nlohmann::json audit;
        audit["dispatches"] = live.cs_audit_dispatches.load();
        audit["differing"] = live.cs_audit_differing.load();
        nlohmann::json dwords = nlohmann::json::object();
        for (size_t i = 0; i < live.cs_audit_dword.size(); ++i) {
            if (const u64 count = live.cs_audit_dword[i].load(); count != 0) {
                dwords[std::to_string(i)] = count;
            }
        }
        audit["differing_dwords"] = dwords;
        j["cs_audit"] = audit;
    }
    j["packet_path_skipped"] = {{"buffers", live.packet_path_skipped_buffers.load()}, {"packets", live.packet_path_skipped_packets.load()}};
    j["overlay"] = {{"records", live.overlay_records.load()}, {"offered", live.overlay_offered.load()}, {"unmapped", live.overlay_unmapped.load()}, {"same", live.overlay_same.load()}, {"changed", live.overlay_changed.load()}, {"group_skipped_values_checked", live.overlay_group_checked.load()}, {"group_skip_violations", live.overlay_group_violations.load()}};
    {
        std::vector<std::pair<u32, u32>> top;
        for (u32 i = 0; i < live.overlay_changed_by_index.size(); ++i) {
            if (const u32 c = live.overlay_changed_by_index[i].load(std::memory_order_relaxed); c != 0) {
                top.emplace_back(c, i);
            }
        }
        std::sort(top.rbegin(), top.rend());
        nlohmann::json list = nlohmann::json::array();
        for (size_t i = 0; i < top.size() && i < 60; ++i) {
            list.push_back({{"index", fmt::format("{:#x}", top[i].second)}, {"changed", top[i].first}});
        }
        j["overlay_changed_top"] = list;
    }
    j["map_native"] = {{"tried", live.map_tried.load()}, {"patched", live.map_patched.load()}, {"refused", live.map_refused.load()}, {"executed", live.map_exec.load()}};
    {
        nlohmann::json by_group = nlohmann::json::array();
        for (const auto& violations : live.group_sig_violations_by_group) {
            by_group.push_back(violations.load());
        }
        j["group_sig"] = {{"skipped", live.group_sig_skipped.load()}, {"applied", live.group_sig_applied.load()}, {"audited", live.group_sig_audited.load()},
                          {"violations", live.group_sig_violations.load()}, {"violations_by_group", by_group}};
    }
    j["stage_key"] = {{"hits", live.stage_key_hits.load()}, {"misses", live.stage_key_misses.load()}, {"uncached", live.stage_key_uncached.load()},
                      {"audits", live.stage_key_audits.load()}, {"violations", live.stage_key_violations.load()}};
    j["exec_inputs"] = {{"typed", live.exec_typed.load()}, {"fallback", live.exec_fallback.load()}, {"audits", live.exec_audits.load()}, {"violations", live.exec_violations.load()},
                        {"fast_clear", live.exec_fast_clear.load()}, {"inline", live.exec_inline.load()}, {"inline_seen", live.exec_inline_seen.load()},
                        {"reasons", {live.exec_reason[0].load(), live.exec_reason[1].load(), live.exec_reason[2].load(), live.exec_reason[3].load(),
                                     live.exec_reason[4].load(), live.exec_reason[5].load(), live.exec_reason[6].load(), live.exec_reason[7].load()}},
                        {"fixed_null_by_exit", [&live] { nlohmann::json a = nlohmann::json::array(); for (const auto& c : live.fixed_null) { a.push_back(c.load()); } return a; }()},
                        {"plan_null_stale_inputs_pipeline", {live.plan_null[0].load(), live.plan_null[1].load(), live.plan_null[2].load()}}};
    j["exec_state"] = {{"planned", live.state_planned.load()}, {"audits", live.state_audits.load()}, {"violations", live.state_violations.load()},
                       {"group_violations", live.state_group_violations.load()}};
    j["fetch_rebuild"] = {{"tried", live.fetch_rebuild_tried.load()}, {"rebuilt", live.fetch_rebuilt.load()}, {"refused", live.fetch_rebuild_refused.load()}, {"executed", live.fetch_rebuild_exec.load()}};
    j["sync_converted_hook"] = live.sync_converted_hook.load();
    j["sync_converted_sweep"] = live.sync_converted_sweep.load();
    j["sync_converted_submit"] = live.sync_converted_submit.load();
    j["sync_packets_kept"] = live.fence_packets_kept.load();
    j["sync_converted"] = {{"eop", live.sync_converted[1].load()}, {"release", live.sync_converted[2].load()}, {"wait", live.sync_converted[3].load()}, {"write", live.sync_converted[4].load()}, {"event", live.sync_converted[5].load()}, {"inert", live.sync_converted[6].load()}, {"dma", live.sync_converted[7].load()}, {"query", live.sync_converted[8].load()}};
    j["sync_exec"] = {{"eop", live.sync_exec[1].load()}, {"release", live.sync_exec[2].load()}, {"wait", live.sync_exec[3].load()}, {"write", live.sync_exec[4].load()}, {"event", live.sync_exec[5].load()}, {"dma", live.sync_exec[7].load()}, {"query", live.sync_exec[8].load()}};
    j["direct_built"] = live.direct_built.load();
    j["direct_pure"] = live.direct_pure.load();
    j["direct_mixed"] = live.direct_mixed.load();
    j["direct_entries"] = live.direct_entries.load();
    j["direct_matched"] = live.direct_matched.load();
    j["direct_entry_mismatch"] = live.direct_entry_mismatch.load();
    j["direct_purity_mismatch"] = live.direct_purity_mismatch.load();
    j["direct_incomplete"] = live.direct_incomplete.load();
    j["direct_executed"] = live.direct_executed.load();
    j["direct_executed_asc"] = live.direct_executed_asc.load();
    j["shadow_anchors"] = live.shadow_anchors.load();
    j["shadow_spans"] = live.shadow_spans.load();
    j["shadow_pure"] = live.shadow_pure.load();
    j["shadow_mixed"] = live.shadow_mixed.load();
    j["shadow_matched"] = live.shadow_matched.load();
    j["shadow_extra"] = live.shadow_extra.load();
    j["shadow_missing"] = live.shadow_missing.load();
    j["shadow_nested"] = live.shadow_nested.load();
    j["shadow_stale"] = live.shadow_stale.load();
    {
        nlohmann::json kinds = nlohmann::json::object();
        for (u32 kind = 0; kind < 64; ++kind) {
            if (const u64 n = Core::NativeDraw::ShadowExtraByKind(kind); n != 0) {
                kinds[std::to_string(kind)] = n;
            }
        }
        j["shadow_extra_by_kind"] = std::move(kinds);
    }
    {
        nlohmann::json other = nlohmann::json::object();
        for (u32 opcode = 0; opcode < 256; ++opcode) {
            if (const u64 n = Core::NativeDraw::ShadowNonEntryCount(opcode); n != 0) {
                other[fmt::format("{:#x}", opcode)] = n;
            }
        }
        j["shadow_non_entry"] = std::move(other);
    }
    j["fixed_native"] = live.fixed_native.load();
    j["fixed_checked"] = live.fixed_checked.load();
    j["fixed_diff"] = live.fixed_diff.load();
    j["track_native"] = live.track_native.load();
    j["track_fce"] = live.track_fce.load();
    j["track_game"] = live.track_game_second.load() + live.track_game_depth.load();
    auto& packets = j["packets"] = nlohmann::json::array();
    for (const auto& p : live.packets) {
        packets.push_back(p.load());
    }
    j["asc_packets"] = live.asc_packets.load();
    return j;
}

nlohmann::json Threads(const Params&) {
    static VideoCore::ThreadUsageSampler sampler;
    static std::mutex mutex;
    std::scoped_lock lk{mutex};
    double total = 0.0;
    const auto usage = sampler.Sample(&total);
    nlohmann::json j;
    j["note"] = "percent of one core since the previous call of this endpoint";
    j["total_percent"] = total;
    auto& list = j["threads"] = nlohmann::json::array();
    for (const auto& entry : usage) {
        list.push_back({{"name", entry.name}, {"percent", entry.percent}});
    }
    return j;
}

nlohmann::json Memory(const Params& params) {
    // ?ghidra=0x... reads the game image at a Ghidra address (runtime address = image base + ghidra - 0x1000000)
    u64 address = ParseNumber(params, "addr", 0);
    if (params.find("ghidra") != params.end()) {
        const u64 base = NativeDraw::ImageBase();
        if (base == 0) {
            return {{"error", "the image base is not known yet"}};
        }
        address = base + ParseNumber(params, "ghidra", 0) - 0x1000000;
    }
    const u64 size = ParseNumber(params, "size", 64);
    if (size == 0 || size > 0x10000) {
        return {{"error", "size must be 1..65536"}};
    }
    auto* memory = Core::Memory::Instance();
    std::vector<u8> data(size);
    if (address != 0 && memory->IsValidMapping(address, size)) {
        memory->CopySparseMemory(address, data.data(), size);
    } else {
        // the game image and the engine's own memory are plain host memory of this process (not in the GPU memory manager): read it when it is committed and readable
        bool readable = address != 0;
#ifdef _WIN32
        for (u64 at = address; readable && at < address + size;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0 || info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
                readable = false;
                break;
            }
            at = reinterpret_cast<u64>(info.BaseAddress) + info.RegionSize;
        }
#else
        readable = false;
#endif
        if (!readable) {
            return {{"error", "address range is not mapped"}, {"addr", address}, {"size", size}};
        }
        std::memcpy(data.data(), reinterpret_cast<const void*>(address), size);
    }
    static constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(size * 2);
    for (const u8 byte : data) {
        hex.push_back(digits[byte >> 4]);
        hex.push_back(digits[byte & 15]);
    }
    nlohmann::json j;
    j["addr"] = fmt::format("{:#x}", address);
    j["image_base"] = fmt::format("{:#x}", NativeDraw::ImageBase());
    j["size"] = size;
    j["hex"] = hex;
    if (size % 4 == 0 && size <= 256) {
        auto& words = j["u32"] = nlohmann::json::array();
        for (u64 i = 0; i < size; i += 4) {
            u32 word;
            std::memcpy(&word, data.data() + i, 4);
            words.push_back(fmt::format("{:#010x}", word));
        }
    }
    return j;
}

nlohmann::json Log(const Params& params) {
    const u64 lines = std::min<u64>(ParseNumber(params, "lines", 50), 2000);
    std::string name = "shad_log.txt";
    if (const auto it = params.find("file"); it != params.end()) {
        name = it->second;
    }
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find("..") != std::string::npos) {
        return {{"error", "bad file name"}};
    }
    std::string filter;
    if (const auto it = params.find("filter"); it != params.end()) {
        filter = it->second;
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {{"error", "cannot open " + path.string()}};
    }
    in.seekg(0, std::ios::end);
    const auto end = static_cast<s64>(in.tellg());
    const s64 chunk = std::min<s64>(end, 1 << 20);
    in.seekg(end - chunk);
    std::string text(static_cast<size_t>(chunk), '\0');
    in.read(text.data(), chunk);
    std::vector<std::string> result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) {
            eol = text.size();
        }
        std::string line = text.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (filter.empty() || line.find(filter) != std::string::npos) {
            result.push_back(std::move(line));
        }
        pos = eol + 1;
    }
    if (result.size() > lines) {
        result.erase(result.begin(), result.end() - static_cast<s64>(lines));
    }
    return {{"file", name}, {"lines", result}};
}

namespace {
// Write faults by the instruction that took them (Ghidra address): small open addressed table, emptied by /faults.
constexpr u32 FaultRipSlots = 512;
std::array<std::atomic<u64>, FaultRipSlots> fault_rip_keys{};
std::array<std::atomic<u64>, FaultRipSlots> fault_rip_counts{};
} // namespace

nlohmann::json Faults(const Params& params) {
    const u64 top = std::min<u64>(ParseNumber(params, "top", 15), 100);
    std::vector<std::pair<u32, u32>> counts; // {count, region}
    u64 total = 0;
    u64 total_ns = 0;
    std::array<u64, FaultRegions> region_ns{};
    for (u32 i = 0; i < FaultRegions; ++i) {
        const u32 count = fault_regions[i].exchange(0, std::memory_order_relaxed);
        region_ns[i] = fault_ns[i].exchange(0, std::memory_order_relaxed);
        total_ns += region_ns[i];
        if (count != 0) {
            counts.emplace_back(count, i);
            total += count;
        }
    }
    const u64 now = NowNs();
    const u64 previous = faults_read_at_ns.exchange(now, std::memory_order_relaxed);
    const double seconds = previous == 0 ? 0.0 : static_cast<double>(now - previous) / 1e9;
    std::ranges::sort(counts, std::greater<>{});
    nlohmann::json j;
    {
        const u64 base = NativeDraw::ImageBase();
        std::vector<std::pair<u64, u64>> rips;
        for (u32 i = 0; i < FaultRipSlots; ++i) {
            const u64 key = fault_rip_keys[i].load(std::memory_order_relaxed);
            if (key == 0) {
                continue;
            }
            const u64 count = fault_rip_counts[i].exchange(0, std::memory_order_relaxed);
            fault_rip_keys[i].store(0, std::memory_order_relaxed);
            if (count != 0) {
                rips.emplace_back(count, key);
            }
        }
        std::ranges::sort(rips, std::greater<>{});
        auto& list = j["top_instructions"] = nlohmann::json::array();
        for (size_t i = 0; i < rips.size() && i < top; ++i) {
            list.push_back({{"ghidra", fmt::format("{:#x}", rips[i].second >= base ? rips[i].second - base + 0x1000000 : rips[i].second)}, {"faults", rips[i].first}});
        }
    }
    j["note"] = "CPU write faults on tracked pages since the previous call, by 4 MB region and by faulting instruction (Ghidra address)";
    j["total"] = total;
    j["seconds_since_previous_call"] = seconds;
    j["handler_ms_total"] = static_cast<double>(total_ns) / 1e6;
    j["handler_cores"] = seconds > 0.0 ? static_cast<double>(total_ns) / 1e9 / seconds : 0.0;
    j["handler_us_per_fault"] = total != 0 ? static_cast<double>(total_ns) / 1e3 / static_cast<double>(total) : 0.0;
    j["faults_per_second"] = seconds > 0.0 ? static_cast<double>(total) / seconds : 0.0;
    auto& list = j["top_regions"] = nlohmann::json::array();
    for (size_t i = 0; i < counts.size() && i < top; ++i) {
        list.push_back({{"addr", fmt::format("{:#x}", static_cast<u64>(counts[i].second) << 22)},
                        {"faults", counts[i].first},
                        {"handler_ms", static_cast<double>(region_ns[counts[i].second]) / 1e6}});
    }
    return j;
}

nlohmann::json Profile(const Params& params) {
    std::string thread = "GpuRaster";
    if (const auto it = params.find("thread"); it != params.end() && !it->second.empty()) {
        thread = it->second;
    }
    const double seconds = static_cast<double>(ParseNumber(params, "seconds", 5));
    const u32 hz = static_cast<u32>(ParseNumber(params, "hz", 500));
    const u32 top = static_cast<u32>(std::min<u64>(ParseNumber(params, "top", 25), 200));
    static std::mutex profile_mutex; // one profile at a time
    std::scoped_lock lk{profile_mutex};
    return VideoCore::ProfileThread(thread, seconds, hz, top);
}

nlohmann::json Policy(const Params&) {
    return VideoCore::ReadbackPolicy::Dump();
}

nlohmann::json SetPolicy(const Params& params) {
    nlohmann::json j;
    for (const auto& [name, value] : params) {
        const bool ok = VideoCore::ReadbackPolicy::Set(name, static_cast<s64>(std::stoll(value, nullptr, 0)));
        j[name] = ok ? "ok" : "unknown name";
    }
    j["policy"] = VideoCore::ReadbackPolicy::Dump();
    return j;
}

std::mutex periodic_mutex;
std::map<std::string, std::vector<std::pair<double, double>>> periodic; // name -> {at_ms, cost_ms}

nlohmann::json Timeline(const Params& params) {
    const u64 end = FrameTimeline::head.load(std::memory_order_acquire);
    const u64 count = std::min<u64>({ParseNumber(params, "n", 8000), static_cast<u64>(FrameTimeline::RingSize), end});
    std::vector<std::array<u64, 5>> list;
    list.reserve(count);
    for (u64 i = end - count; i < end; ++i) {
        const auto& e = FrameTimeline::ring[i & (FrameTimeline::RingSize - 1)];
        const u32 type = e.type.load(std::memory_order_acquire);
        if (type == 0) {
            continue;
        }
        list.push_back({e.ns.load(std::memory_order_relaxed), type, e.arg.load(std::memory_order_relaxed), e.thread.load(std::memory_order_relaxed),
                        e.arg2.load(std::memory_order_relaxed)});
    }
    std::ranges::sort(list, [](const auto& a, const auto& b) { return a[0] < b[0]; });
    nlohmann::json j;
    const u64 t0 = list.empty() ? 0 : list.front()[0];
    j["note"] = "events [ms since the first, type, arg, thread id, arg2]; types: 10 GameWait (arg: ns | kind << 56, 1 cond 2 sema 3 sleep 4 equeue; arg2: object), 11 GameSignal (arg: kind 5 signal 6 broadcast; arg2: object), 1 FirstHook, 2 Submit, 3 SubmitFlip (arg: last hook ns), 4 FlipCp, 5 FlipPrepared, 6 PresentStart (arg: vblank count), 7 PresentEnd, 8 CpWait (arg: ns | site << 56), 9 FlipWait (arg: ns)";
    j["t0_ns"] = t0;
    auto& events = j["events"] = nlohmann::json::array();
    for (const auto& e : list) {
        events.push_back({static_cast<double>(e[0] - t0) / 1e6, e[1], e[2], e[3], e[4]});
    }
    auto& names = j["threads"] = nlohmann::json::object();
    for (const auto& [id, name] : VideoCore::ThreadNames()) {
        names[std::to_string(id)] = name;
    }
    return j;
}

nlohmann::json TimelineCapture(const Params& params) {
    nlohmann::json j;
    if (params.contains("seconds")) {
        const u32 seconds = static_cast<u32>(std::clamp<u64>(ParseNumber(params, "seconds", 10), 1, 60));
        const auto tag = params.contains("tag") ? params.at("tag") : std::string("capture");
        j["started"] = FrameTimeline::StartCapture(seconds, tag);
    }
    j["status"] = nlohmann::json::parse(FrameTimeline::CaptureStatus());
    return j;
}

nlohmann::json StreamProbeReport(const Params&) {
    return StreamProbe::Report();
}

nlohmann::json Frames(const Params& params) {
    const u64 want = std::min<u64>(ParseNumber(params, "n", 600), TimeRing - 1);
    const u64 count = presents.load(std::memory_order_acquire);
    const u64 available = std::min<u64>(count, TimeRing - 1);
    const u64 n = std::min(want, available);
    nlohmann::json j;
    j["now_ms"] = UptimeMs();
    auto& frames = j["frames"] = nlohmann::json::array();
    for (u64 i = n; i >= 1; --i) {
        const u64 t = present_times[(count - i) & (TimeRing - 1)].load(std::memory_order_relaxed);
        const u64 previous =
            present_times[(count - i - 1) & (TimeRing - 1)].load(std::memory_order_relaxed);
        if (t == 0 || previous == 0) {
            continue;
        }
        frames.push_back({t / 1e6, (t - previous) / 1e6});
    }
    return j;
}

nlohmann::json Periodic(const Params&) {
    std::scoped_lock lk{periodic_mutex};
    nlohmann::json j;
    j["now_ms"] = UptimeMs();
    for (const auto& [name, runs] : periodic) {
        auto& array = j[name] = nlohmann::json::array();
        for (const auto& [at, cost] : runs) {
            array.push_back({{"at_ms", at}, {"cost_ms", cost}});
        }
    }
    return j;
}

nlohmann::json Stats(const Params&) {
    std::scoped_lock lk{published_mutex};
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [group, value] : published) {
        j[group] = value;
    }
    return j;
}

nlohmann::json Index(const Params&) {
    nlohmann::json j;
    std::scoped_lock lk{registry_mutex};
    for (const auto& [path, entry] : registry) {
        j[path] = entry.description;
    }
    return j;
}

void RegisterBuiltins() {
    Register("/", "list of endpoints", Index);
    Register("/status", "uptime, presented frames, fps", Status);
    Register("/native", "native renderer counters (cumulative)", Native);
    Register("/frame_gate", "frames in flight: the setting, frames submitted, end-of-pipe interrupts, waits of the gate",
             [](const Params&) { return Core::FrameGate::Status(); });
    Register("/emulation", "E0 census (cumulative): raw PM4 packets that still execute and their registers, constant engine submissions, image refresh / detile / overlap calls",
             [](const Params&) { return Core::NativeCensus::Report(); });
    Register("/capture",
             "?frames=24&x0=0&y0=0&x1=1&y1=1: save the guest output of the next frames (every game flip once), cropped (fractions), to log/frame_sequence/f<index>_p<present>.png; "
             "without frames: how many are left",
             [](const Params& params) {
                 if (params.find("frames") != params.end()) {
                     const auto number = [&](const char* key, float fallback) {
                         const auto it = params.find(key);
                         return it == params.end() || it->second.empty() ? fallback : std::stof(it->second);
                     };
                     VideoCore::RequestFrameSequence(static_cast<u32>(std::min<u64>(ParseNumber(params, "frames", 24), 600)), number("x0", 0.0f),
                                                     number("y0", 0.0f), number("x1", 1.0f), number("y1", 1.0f));
                 }
                 return nlohmann::json{{"remaining", VideoCore::FrameSequenceRemaining()}};
             });
    Register("/audit", "tracker audit: how often the buffer cache called a page clean that the CPU had changed (native_sdk_strip bit 67108864)",
             [](const Params&) { return Core::NativeAudit::Report(); });
    Register("/readbacks", "explicit readbacks, step 1: destinations of the engine's copy / counter read-back records, AsyncMap ranges, CPU read faults",
             [](const Params&) { return Core::NativeReadbacks::Report(); });
    Register("/raycast", "CPU port of the ray against triangle shader (0xe15f3156), shadow compare against the GPU result (TLG_RAYCAST_SHADOW=1)",
             [](const Params&) { auto j = Core::NativeRaycast::Report(); j["skin"] = Core::NativeSkinCpu::Report(); return j; });
    Register("/stall_capture", "?seconds=10 starts the stall profiler (the same as F8); without arguments it shows whether it runs",
             [](const Params& params) {
                 if (params.contains("seconds")) {
                     Core::Stall::Start(static_cast<u32>(std::min<u64>(ParseNumber(params, "seconds", 10), 60)), ParseNumber(params, "open", 0) != 0);
                 }
                 return nlohmann::json{{"recording", Core::Stall::Recording()}, {"report_age_s", Core::Stall::ReportAgeSeconds()}};
             });
    Register("/frame_events", "E6.2a: the console's frame events (end of pipe, compute releases, GPU idle, flip ready): signals and subscriber runs",
             [](const Params&) { auto j = Core::NativeFrameEvents::Report(); j["flip_request"] = Core::NativeFlip::Report(); j["display"] = Core::NativeDisplay::Report(); j["presenter"] = Core::NativePresenter::Report(); j["shader_env"] = Core::NativeShaderEnv::Report(); return j; });
    Register("/stall", "the report of the last stall profiler capture (F8): the waits and our threads ranked by how much more the slow frames contain",
             [](const Params&) { return nlohmann::json::parse(Core::Stall::ReportJson()); });
    Register("/entrydump",
             "?frames=1: dump the next frames of the command stream as entries to log/entry_dump.txt (without frames: the state of the last request)",
             EntryDumpEndpoint);
    Register("/resources", "?start=1 turns the resource census on; the engine's GPU memory blocks and who uses, uploads and writes them",
             [](const Params& params) {
                 if (ParseNumber(params, "start", 0) != 0) {
                     Core::NativeResources::Start();
                 }
                 return Core::NativeResources::Report();
             });
    Register("/registry", "?start=1 turns the native resource registry (shadow mode) on; objects bound by draws, their kinds, tags and raw bytes, and the audit against the legacy buffer / image resolution",
             [](const Params& params) {
                 if (ParseNumber(params, "start", 0) != 0) {
                     Core::NativeRegistry::Start();
                 }
                 return Core::NativeRegistry::Report();
             });
    Register("/passes", "?start=1 turns the pass audit on (N4a): rendering instances of the scheduler, their draws, what ends them and whether they are plain runs of draws",
             [](const Params& params) {
                 if (ParseNumber(params, "start", 0) != 0) {
                     Core::NativePass::Start();
                 }
                 return Core::NativePass::Report();
             });
    Register("/updates", "?start=1 turns the update audit on (N5a): buffer Maps the engine announces, and which CPU write faults they cover",
             [](const Params& params) {
                 if (ParseNumber(params, "start", 0) != 0) {
                     Core::NativeUpdates::Start();
                 }
                 return Core::NativeUpdates::Report();
             });
    Register("/uploads", "(N5c, TLG_NATIVE_UPLOADS=1) announced writes, how many were published, queued before records and uploaded before the draws that use them",
             [](const Params&) { return Core::NativeUploads::Report(); });
    Register("/drain", "(N6a, TLG_DRAIN_AUDIT=1) the command processor's wait for queued memory writes: did any guest read of the planning touch a pending write",
             [](const Params&) { return Core::DrainAudit::Report(); });
    Register("/zpass", "(N6.3a, TLG_ZPASS_AUDIT=N) the depth pass loop against the draws built from the snapshot it reads: expected, issued, matched, missing, extra per view type",
             [](const Params&) { return Core::NativeZPass::Report(); });
    Register("/zpassgen", "(N6.3i, TLG_ZPASS_GEN=1) the cascade generator: passes generated, run by the loop, declined (and why), verify runs and their differences, the learned variant table",
             [](const Params&) { return Core::NativeZPassGen::Report(); });
    Register("/gbuffer", "(N6.6, TLG_GBUFFER_AUDIT=1) the arguments and shapes the G-buffer loop sees: passes, batches, primitives",
             [](const Params&) { return Core::NativeGBufferAudit::Report(); });
    Register("/threads", "CPU use of the busiest threads since the previous call", Threads);
    Register("/memory", "?addr=0x..&size=N (max 65536): guest memory as hex", Memory);
    Register("/log", "?lines=N&filter=text&file=shad_log.txt: tail of a log file", Log);
    Register("/profile",
             "?thread=GpuRaster&seconds=5&hz=500&top=25: stack sampling of one thread (functions as "
             "offsets, resolve with the linker map)",
             Profile);
    Register("/policy", "readback policy switches (video_core/readback_policy.h)", Policy);
    Register("/set", "?name=value changes readback policy switches (development only)",
             SetPolicy);
    Register("/stream_probe", "(study) the shape of the submitted command buffers and what the game changes after the draw hook saw it", StreamProbeReport);
    Register("/timeline", "?n=8000: the latest frame events (study): hooks, submits, flips, command processor waits, presents", Timeline);
    Register("/frames", "?n=600: the last n frame intervals (time stamp and length in ms)", Frames);
    Register("/timeline_capture", "?seconds=10&tag=name: records every frame timeline event for that long into log/timeline_<tag>.bin (+ .json); without arguments: the state", TimelineCapture);
    Register("/periodic", "last runs of periodic jobs (when they ended, how long they took)",
             Periodic);
    Register("/stats", "figures published by the subsystems (latest interval)", Stats);
    Register("/faults", "CPU write faults on tracked pages by 4 MB region since the previous call",
             Faults);
}

} // namespace

void Register(const std::string& path, const std::string& description, Handler handler) {
    std::scoped_lock lk{registry_mutex};
    registry[path] = Entry{description, std::move(handler)};
}

void Unregister(const std::string& path) {
    std::scoped_lock lk{registry_mutex};
    registry.erase(path);
}

u64 SincePresentNs() {
    const u64 count = presents.load(std::memory_order_acquire);
    if (count == 0) {
        return 0;
    }
    const u64 last = present_times[(count - 1) & (TimeRing - 1)].load(std::memory_order_relaxed);
    const u64 now = NowNs();
    return now > last ? now - last : 0;
}

u64 PresentCount() {
    return presents.load(std::memory_order_acquire);
}

double UptimeMs() {
    return static_cast<double>(NowNs()) / 1e6;
}

void RecordPeriodic(const std::string& name, double cost_ms) {
    const double at = UptimeMs();
    std::scoped_lock lk{periodic_mutex};
    auto& runs = periodic[name];
    runs.emplace_back(at, cost_ms);
    if (runs.size() > 24) {
        runs.erase(runs.begin());
    }
}

void Publish(const std::string& group, nlohmann::json value) {
    std::scoped_lock lk{published_mutex};
    published[group] = std::move(value);
}

void NotePresent() {
    const u64 index = presents.fetch_add(1, std::memory_order_acq_rel);
    present_times[index & (TimeRing - 1)].store(NowNs(), std::memory_order_relaxed);
}

void NoteWriteFault(VAddr address) {
    fault_regions[(address >> 22) & (FaultRegions - 1)].fetch_add(1, std::memory_order_relaxed);
    if (const u64 rip = GuestHooks::FaultRip(); rip != 0) {
        u32 slot = static_cast<u32>((rip * 0x9E3779B97F4A7C15ull) >> 55) % FaultRipSlots;
        for (u32 probe = 0; probe < 8; ++probe, slot = (slot + 1) % FaultRipSlots) {
            u64 key = fault_rip_keys[slot].load(std::memory_order_relaxed);
            if (key == rip || (key == 0 && fault_rip_keys[slot].compare_exchange_strong(key, rip, std::memory_order_relaxed))) {
                fault_rip_counts[slot].fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (key == rip) {
                fault_rip_counts[slot].fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }
}

void NoteWriteFaultTime(VAddr address, u64 nanoseconds) {
    fault_ns[(address >> 22) & (FaultRegions - 1)].fetch_add(nanoseconds, std::memory_order_relaxed);
}

void Start(u16 port) {
    std::scoped_lock lk{lifecycle_mutex};
    if (port == 0 || server) {
        return;
    }
    RegisterBuiltins();
    server = std::make_unique<httplib::Server>();
    server->Get(R"(/.*)", [](const httplib::Request& req, httplib::Response& res) {
        Handler handler;
        {
            std::scoped_lock lk{registry_mutex};
            auto it = registry.find(req.path);
            if (it != registry.end()) {
                handler = it->second.handler;
            }
        }
        if (!handler) {
            res.status = 404;
            res.set_content(Index({}).dump(2), "application/json");
            return;
        }
        Params params;
        for (const auto& [key, value] : req.params) {
            params[key] = value;
        }
        try {
            res.set_content(handler(params).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    if (!server->bind_to_port("127.0.0.1", port)) {
        LOG_ERROR(Core, "debug server: cannot listen on 127.0.0.1:{}", port);
        server.reset();
        return;
    }
    server_thread = std::jthread{[] { server->listen_after_bind(); }};
    LOG_INFO(Core, "debug server listening on 127.0.0.1:{}", port);
    // The counters of a play session survive the game: every 20 s (TLG_COUNTER_SNAPSHOT=0 turns it off) the reports of /status, /zpassgen, /native, /registry and /stats are written to
    // user/log/last_counters.json (written aside and renamed), so they can be read after the game is closed.
    const char* snapshot_env = std::getenv("TLG_COUNTER_SNAPSHOT");
    if (snapshot_env == nullptr || *snapshot_env != '0') {
        snapshot_thread = std::jthread{[](std::stop_token stop) {
            const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "last_counters.json";
            const auto write_snapshot = [&](const std::filesystem::path& target_path) {
                auto temp = target_path;
                temp += ".tmp";
                nlohmann::json out;
                {
                    std::scoped_lock lk{rare_mutex};
                    out["rare_events"] = rare_counts;
                }
                out["written_unix_s"] = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                for (const char* name : {"/status", "/zpassgen", "/native", "/registry", "/stats", "/resources"}) {
                    Handler handler;
                    {
                        std::scoped_lock lk{registry_mutex};
                        const auto it = registry.find(name);
                        if (it != registry.end()) {
                            handler = it->second.handler;
                        }
                    }
                    if (!handler) {
                        continue;
                    }
                    try {
                        out[name] = handler(Params{});
                    } catch (const std::exception& e) {
                        out[name] = nlohmann::json{{"error", e.what()}};
                    }
                }
                {
                    std::ofstream file{temp, std::ios::binary | std::ios::trunc};
                    file << out.dump(1);
                }
                std::error_code ec;
                std::filesystem::rename(temp, target_path, ec);
            };
            const bool self_test = std::getenv("TLG_RARE_EVENT_TEST") != nullptr; // development: one made up event after 3 s, to see the file and the snapshot
            u32 ticks = 0;
            while (!stop.stop_requested()) {
                for (int i = 0; i < 200 && !stop.stop_requested(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (self_test && ++ticks == 30) {
                        Core::RareEvents::Note("self_test", "made up by TLG_RARE_EVENT_TEST");
                    }
                    if (snapshot_now.exchange(false)) {
                        std::string kind;
                        {
                            std::scoped_lock lk{rare_mutex};
                            kind = rare_snapshot_kind;
                        }
                        write_snapshot(path.parent_path() / ("rare_" + kind + "_counters.json"));
                    }
                }
                write_snapshot(path);
            }
        }};
    }
}

void Stop() {
    std::scoped_lock lk{lifecycle_mutex};
    if (!server) {
        return;
    }
    if (snapshot_thread.joinable()) {
        snapshot_thread.request_stop();
        snapshot_thread.join();
    }
    server->stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }
    server.reset();
}

} // namespace Core::DebugServer

namespace Core::RareEvents {

void Note(const std::string& kind, const std::string& detail) {
    u64 n;
    {
        std::scoped_lock lk{DebugServer::rare_mutex};
        n = ++DebugServer::rare_counts[kind];
        if (n == 1) {
            DebugServer::rare_snapshot_kind = kind;
        }
    }
    if (n > 5) {
        return; // counted, not written again
    }
    const auto unix_s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto line = fmt::format("[{}] uptime {:.1f} s, {} #{}: {}\n", unix_s, DebugServer::UptimeMs() / 1000.0, kind, n, detail);
    {
        std::scoped_lock lk{DebugServer::rare_mutex};
        std::ofstream file{Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "rare_events.txt", std::ios::app | std::ios::binary};
        file << line;
    }
    LOG_ERROR(Core, "rare event {} (#{}), saved in user/log/rare_events.txt: {}", kind, n, detail);
    if (n == 1) {
        DebugServer::snapshot_now.store(true);
    }
}

} // namespace Core::RareEvents
