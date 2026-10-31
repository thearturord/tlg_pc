// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_raster_prof.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <x86intrin.h>
#endif

#include "common/logging/log.h"
#include "common/path_util.h"

namespace Core::RasterProf {

namespace {

constexpr u32 PhaseCount = static_cast<u32>(Phase::Count);
static_assert(PhaseCount < 64, "a phase id has 6 bits in a path key");

constexpr std::array<const char*, PhaseCount> PhaseNames{
    "-",
    "op: draw",
    "op: indirect draw",
    "op: dispatch",
    "op: indirect dispatch",
    "op: closure (fences, flips, write-backs)",
    "callbacks between operations (ProcessCommands)",
    "ring use (BeginRingUse / EndRingUse)",
    "flush of recorded work (MaybeFlushRecorded)",
    "scheduler: PopPendingOperations",
    "FilterDraw",
    "pipeline lookup (ResolveDrawPipeline / ResolveDispatchPipeline)",
    "fixed state and draw inputs (FixedFor / InputsFor)",
    "PrepareRenderState",
    "dispatch fill (TryClearImage)",
    "shader HLE check (ExecuteShaderHLE)",
    "BindResources",
    "BindBuffers",
    "BufferCache::ObtainBuffer",
    "BindTextures",
    "TextureCache::FindImage",
    "image view (FindTexture / FindRenderTarget / FindDepthTarget)",
    "TextureCache::RefreshImage",
    "RefreshImage: ObtainBufferForImage (guest bytes copied)",
    "RefreshImage: DetileImage",
    "RefreshImage: UploadImage",
    "BindSamplers",
    "TextureCache::GetSampler",
    "indexed data path: row table copy",
    "BeginRendering (render targets)",
    "BindVertexBuffers",
    "BindIndexBuffer",
    "FlushBarriers",
    "descriptor push (Pipeline::BindResources)",
    "UpdateDynamicState",
    "scheduler: BeginRendering",
    "scheduler: EndRendering",
    "record: bindPipeline + draw call",
    "record: bindPipeline + dispatch call",
    "ResetBindings",
    "wait for the encoder queue (hand-off)",
};
static_assert(PhaseNames.size() == PhaseCount);

constexpr const char* ClassNames[16] = {"z prepass / shadow", "g-buffer", "forward", "compute / other", "indirect", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "untagged"};
constexpr const char* KindNames[5] = {"?", "draw", "indirect draw", "dispatch", "indirect dispatch"};

struct Stat {
    u64 count{0}, incl{0}, excl{0}, max{0};
};

struct Slot {
    u64 key{0}; ///< path: the phase ids from the top level down, 6 bits each, the innermost in the low bits (0: empty)
    Stat stat{};
};
constexpr u32 SlotCount = 4096;
constexpr u32 MaxDepth = 10; // 60 bits of path

struct Frame {
    u64 t0;
    u64 child;
    u64 path;
    Phase phase;
};
constexpr u32 StackSize = 64;

struct CallerStat {
    u64 count{0}, ticks{0}, max{0};
    u32 kind{0}, cls{0};
    u64 caller{0};
};

struct SlowOp {
    u64 ticks{0};
    u32 kind{0}, cls{0};
    u64 caller{0};
    std::array<u64, PhaseCount> excl{};
};
constexpr u32 SlowCount = 12;

// Written by the raster thread only, read by the analysis after the capture stopped.
std::array<Slot, SlotCount> g_slots{};
std::array<Stat, PhaseCount> g_flat{};
std::array<Frame, StackSize> g_stack{};
u32 g_depth = 0;
std::array<u64, PhaseCount> g_op_excl{};
u64 g_op_caller = 0;
u32 g_op_class = 0;
std::unordered_map<u64, CallerStat> g_callers;
std::array<SlowOp, SlowCount> g_slow{};
u64 g_scopes = 0;
u64 g_slot_overflow = 0;

std::mutex g_mutex;
std::string g_full, g_json;
std::atomic<bool> g_raster_capture{false};
std::atomic<bool> g_open{false};
std::atomic<u64> g_report_ns{0};
u64 g_tick0 = 0, g_ns0 = 0;

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall ShellExecuteA(void*, const char*, const char*, const char*, const char*, int);
#endif

inline u64 Tick() {
#ifdef _WIN32
    return __rdtsc();
#else
    return FrameTimeline::Now();
#endif
}

Slot& SlotFor(u64 path) {
    u32 at = static_cast<u32>((path * 0x9E3779B97F4A7C15ull) >> 52) & (SlotCount - 1);
    for (u32 n = 0; n < SlotCount; ++n, at = (at + 1) & (SlotCount - 1)) {
        Slot& s = g_slots[at];
        if (s.key == path || s.key == 0) {
            s.key = path;
            return s;
        }
    }
    ++g_slot_overflow;
    return g_slots[0];
}

void Reset() {
    g_slots.fill({});
    g_flat.fill({});
    g_op_excl.fill(0);
    g_op_caller = 0;
    g_op_class = 0;
    g_callers.clear();
    g_callers.reserve(1024);
    g_slow.fill({});
    g_scopes = 0;
    g_slot_overflow = 0;
}

u32 KindOf(Phase p) {
    switch (p) {
    case Phase::OpDraw:
        return 1;
    case Phase::OpDrawIndirect:
        return 2;
    case Phase::OpDispatch:
        return 3;
    case Phase::OpDispatchIndirect:
        return 4;
    default:
        return 0;
    }
}

void FinishOp(Phase phase, u64 ticks) {
    const u32 kind = KindOf(phase);
    if (kind == 0) {
        return; // closures and callbacks have no caller
    }
    const u64 key = (u64(kind) << 56) | (u64(g_op_class & 0xff) << 48) | (g_op_caller & 0xffffffffffffull);
    CallerStat& c = g_callers[key];
    c.kind = kind;
    c.cls = g_op_class & 0xff;
    c.caller = g_op_caller;
    ++c.count;
    c.ticks += ticks;
    c.max = std::max(c.max, ticks);
    u32 least = 0;
    for (u32 i = 1; i < SlowCount; ++i) {
        if (g_slow[i].ticks < g_slow[least].ticks) {
            least = i;
        }
    }
    if (ticks > g_slow[least].ticks) {
        SlowOp& s = g_slow[least];
        s.ticks = ticks;
        s.kind = kind;
        s.cls = g_op_class & 0xff;
        s.caller = g_op_caller;
        s.excl = g_op_excl;
    }
}

std::string Escape(const std::string& text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        case '&':
            out += "&amp;";
            break;
        default:
            out += c;
        }
    }
    return out;
}

} // namespace

void Enter(Phase phase) noexcept {
    ++g_scopes;
    Frame f{};
    f.phase = phase;
    f.child = 0;
    if (g_depth == 0) {
        g_op_excl.fill(0);
        g_op_caller = 0;
        g_op_class = 0;
        f.path = static_cast<u64>(phase);
    } else if (g_depth < MaxDepth && g_depth <= StackSize) {
        f.path = (g_stack[g_depth - 1].path << 6) | static_cast<u64>(phase);
    } else {
        f.path = g_stack[(g_depth < StackSize ? g_depth : StackSize) - 1].path; // too deep: counted into the enclosing path
    }
    if (g_depth < StackSize) {
        f.t0 = Tick();
        g_stack[g_depth] = f;
    }
    ++g_depth;
}

void Leave() noexcept {
    if (g_depth == 0) {
        return;
    }
    const u32 at = g_depth - 1;
    --g_depth;
    if (at >= StackSize) {
        return;
    }
    const Frame f = g_stack[at];
    const u64 dur = Tick() - f.t0;
    const u64 excl = dur > f.child ? dur - f.child : 0;
    if (at > 0) {
        g_stack[at - 1].child += dur;
    }
    Slot& s = SlotFor(f.path);
    ++s.stat.count;
    s.stat.incl += dur;
    s.stat.excl += excl;
    s.stat.max = std::max(s.stat.max, dur);
    Stat& flat = g_flat[static_cast<u32>(f.phase)];
    ++flat.count;
    flat.incl += dur;
    flat.excl += excl;
    flat.max = std::max(flat.max, dur);
    g_op_excl[static_cast<u32>(f.phase)] += excl;
    if (at == 0) {
        FinishOp(f.phase, dur);
    }
}

void NoteOp(u64 caller_ghidra, u32 op_class) noexcept {
    g_op_caller = caller_ghidra;
    g_op_class = op_class;
}

bool Start(u32 seconds, bool open_in_browser) {
    g_raster_capture.store(true);
    g_open.store(open_in_browser);
    if (!FrameTimeline::StartCapture(seconds, fmt::format("raster_{}", static_cast<long long>(std::time(nullptr))))) {
        g_raster_capture.store(false);
        return false;
    }
    Reset();
    g_ns0 = FrameTimeline::Now();
    g_tick0 = Tick();
    g_active.store(true, std::memory_order_release);
    return true;
}

bool Stop() {
    return FrameTimeline::StopCapture();
}

bool Recording() {
    return g_raster_capture.load() && FrameTimeline::capture_active.load(std::memory_order_relaxed);
}

bool ThisIsRasterCapture() {
    return g_raster_capture.load();
}

double ReportAgeSeconds() {
    const u64 at = g_report_ns.load();
    return at == 0 ? 1e9 : static_cast<double>(FrameTimeline::Now() - at) / 1e9;
}

std::string FullText() {
    std::scoped_lock lock{g_mutex};
    return g_full;
}

std::string ReportJson() {
    std::scoped_lock lock{g_mutex};
    return g_json.empty() ? "{}" : g_json;
}

void Analyze(const FrameTimeline::CapEvent* events, u64 count, u64 start_ns, u64 end_ns) {
    g_active.store(false, std::memory_order_release);
    g_raster_capture.store(false);
    const u64 tick_end = Tick();
    const u64 ns_end = FrameTimeline::Now();
    const double ticks_per_ns = ns_end > g_ns0 ? static_cast<double>(tick_end - g_tick0) / static_cast<double>(ns_end - g_ns0) : 1.0;
    const auto ms = [&](u64 ticks) { return static_cast<double>(ticks) / ticks_per_ns / 1e6; };
    const auto us = [&](u64 ticks) { return static_cast<double>(ticks) / ticks_per_ns / 1e3; };

    // frames: the game's submits with a flip, else the present thread's flips
    u64 frames = 0, present = 0;
    double idle_ms = 0.0;
    for (u64 i = 0; i < count; ++i) {
        if (events[i].type == FrameTimeline::SubmitFlip) {
            ++frames;
        } else if (events[i].type == FrameTimeline::PresentEnd) {
            ++present;
        } else if (events[i].type == FrameTimeline::StageSpan && (events[i].arg >> 56) == FrameTimeline::StageRasterIdle) {
            idle_ms += static_cast<double>(events[i].arg & ((1ull << 56) - 1)) / 1e6;
        }
    }
    const char* frame_source = "game submits (SubmitFlip)";
    if (frames < 5) {
        frames = present;
        frame_source = "present thread (PresentEnd)";
    }
    const double seconds = static_cast<double>(end_ns - start_ns) / 1e9;
    std::string text;
    if (frames < 5 || g_scopes == 0) {
        text = fmt::format("Raster profiler: {} frames and {} scopes in {:.1f} s: nothing to report (is the game in a scene, and is the split pipeline on? Only the thread shadPS4:GpuRaster is measured).\n", frames, g_scopes, seconds);
        std::scoped_lock lock{g_mutex};
        g_full = text;
        g_json = "{}";
        g_report_ns.store(FrameTimeline::Now());
        return;
    }
    const double nframes = static_cast<double>(frames);

    // the tree, by path
    struct Node {
        u64 path;
        Stat stat;
        std::vector<size_t> children;
    };
    std::vector<Node> nodes;
    std::unordered_map<u64, size_t> index;
    for (const Slot& s : g_slots) {
        if (s.key != 0 && s.stat.count != 0) {
            index[s.key] = nodes.size();
            nodes.push_back({s.key, s.stat, {}});
        }
    }
    std::vector<size_t> roots;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const u64 parent = nodes[i].path >> 6;
        if (parent == 0) {
            roots.push_back(i);
        } else if (const auto it = index.find(parent); it != index.end()) {
            nodes[it->second].children.push_back(i);
        } else {
            roots.push_back(i); // the parent was lost (table overflow): shown at the top
        }
    }
    const auto by_incl = [&](size_t a, size_t b) { return nodes[a].stat.incl > nodes[b].stat.incl; };
    std::sort(roots.begin(), roots.end(), by_incl);
    for (auto& n : nodes) {
        std::sort(n.children.begin(), n.children.end(), by_incl);
    }
    u64 busy_ticks = 0;
    for (const size_t r : roots) {
        busy_ticks += nodes[r].stat.incl;
    }
    const double busy_ms_frame = ms(busy_ticks) / nframes;
    const double frame_ms = seconds * 1000.0 / nframes;
    const double scopes_frame = static_cast<double>(g_scopes) / nframes;

    const auto leaf_name = [&](u64 path) { return std::string(PhaseNames[path & 63]); };
    const auto depth_of = [&](u64 path) {
        u32 d = 0;
        for (; path > 63; path >>= 6) {
            ++d;
        }
        return d;
    };

    // flat list by exclusive time
    std::vector<u32> flat_order;
    for (u32 p = 1; p < PhaseCount; ++p) {
        if (g_flat[p].count != 0) {
            flat_order.push_back(p);
        }
    }
    std::sort(flat_order.begin(), flat_order.end(), [&](u32 a, u32 b) { return g_flat[a].excl > g_flat[b].excl; });

    // callers
    std::vector<const CallerStat*> callers;
    for (const auto& [key, c] : g_callers) {
        callers.push_back(&c);
    }
    std::sort(callers.begin(), callers.end(), [](const CallerStat* a, const CallerStat* b) { return a->ticks > b->ticks; });
    std::array<u64, 5> kind_count{}, kind_ticks{};
    for (const CallerStat* c : callers) {
        kind_count[c->kind] += c->count;
        kind_ticks[c->kind] += c->ticks;
    }
    std::vector<const SlowOp*> slow;
    for (const SlowOp& s : g_slow) {
        if (s.ticks != 0) {
            slow.push_back(&s);
        }
    }
    std::sort(slow.begin(), slow.end(), [](const SlowOp* a, const SlowOp* b) { return a->ticks > b->ticks; });

    // text
    text += fmt::format("RASTER THREAD PROFILE  {:.1f} s, {} frames from {} ({:.1f} fps, {:.2f} ms per frame)\n", seconds, frames, frame_source, nframes / seconds, frame_ms);
    text += fmt::format("raster thread: busy {:.2f} ms per frame in the measured operations ({:.0f} % of the frame), idle waiting for work {:.2f} ms per frame\n", busy_ms_frame, busy_ms_frame / frame_ms * 100.0,
                        idle_ms / nframes);
    text += fmt::format("{:.0f} scopes per frame were timed; each costs roughly 20 to 40 ns (an estimate, not measured), so about {:.1f} ms per frame of this profile is the profile itself\n\n", scopes_frame,
                        scopes_frame * 30e-6);
    text += "OPERATIONS (per frame)\n";
    for (u32 k = 1; k <= 4; ++k) {
        if (kind_count[k] != 0) {
            text += fmt::format("  {:<18} {:>8.1f} per frame  {:>7.2f} ms per frame  {:>7.1f} us each\n", KindNames[k], static_cast<double>(kind_count[k]) / nframes, ms(kind_ticks[k]) / nframes,
                                us(kind_ticks[k]) / static_cast<double>(kind_count[k]));
        }
    }
    text += "\nPHASES RANKED BY OWN TIME (exclusive; ms per frame, calls per frame, mean us, max us)\n";
    for (const u32 p : flat_order) {
        const Stat& s = g_flat[p];
        if (ms(s.excl) / nframes < 0.005) {
            break;
        }
        text += fmt::format("  {:>7.3f} ms  {:>6.1f} %  {:>9.1f} calls  mean {:>8.1f}  max {:>9.1f}  {}\n", ms(s.excl) / nframes, ms(s.excl) / nframes / busy_ms_frame * 100.0, static_cast<double>(s.count) / nframes,
                            us(s.excl) / static_cast<double>(s.count), us(s.max), PhaseNames[p]);
    }
    text += "\nTREE (inclusive ms per frame / exclusive ms per frame / calls per frame)\n";
    const auto print_node = [&](auto&& self, size_t i) -> void {
        const Node& n = nodes[i];
        text += fmt::format("  {}{:>7.3f} / {:>7.3f} ms  {:>8.1f} calls  {}\n", std::string(depth_of(n.path) * 2, ' '), ms(n.stat.incl) / nframes, ms(n.stat.excl) / nframes,
                            static_cast<double>(n.stat.count) / nframes, leaf_name(n.path));
        for (const size_t c : n.children) {
            if (ms(nodes[c].stat.incl) / nframes >= 0.002) {
                self(self, c);
            }
        }
    };
    for (const size_t r : roots) {
        print_node(print_node, r);
    }
    text += "\nOPERATIONS BY ENGINE CALLER (Ghidra address of the function that issued them)\n";
    for (size_t i = 0; i < std::min<size_t>(30, callers.size()); ++i) {
        const CallerStat& c = *callers[i];
        text += fmt::format("  {:<15} {:>8}  class {:<18} {:>8.2f} per frame  {:>7.3f} ms per frame  {:>8.1f} us each  max {:>9.1f}\n", KindNames[c.kind], c.caller == 0 ? std::string("-") : fmt::format("{:#x}", c.caller),
                            ClassNames[c.cls & 15], static_cast<double>(c.count) / nframes, ms(c.ticks) / nframes, us(c.ticks) / static_cast<double>(c.count), us(c.max));
    }
    text += "\nSLOWEST OPERATIONS (with the three phases that took the most of each)\n";
    for (const SlowOp* s : slow) {
        std::vector<u32> order;
        for (u32 p = 1; p < PhaseCount; ++p) {
            if (s->excl[p] != 0) {
                order.push_back(p);
            }
        }
        std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return s->excl[a] > s->excl[b]; });
        std::string parts;
        for (size_t k = 0; k < std::min<size_t>(3, order.size()); ++k) {
            parts += fmt::format("{}{} {:.0f} us", k ? ", " : "", PhaseNames[order[k]], us(s->excl[order[k]]));
        }
        text += fmt::format("  {:>9.1f} us  {} {}  class {}: {}\n", us(s->ticks), KindNames[s->kind], s->caller == 0 ? std::string("-") : fmt::format("{:#x}", s->caller), ClassNames[s->cls & 15], parts);
    }
    if (g_slot_overflow != 0) {
        text += fmt::format("\nnote: {} scope results did not fit the path table and were added to the first slot\n", g_slot_overflow);
    }

    // the page
    std::string html;
    html += "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Raster thread profile</title><style>"
            ":root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa3ad;--card:#1d2127;--line:#2c323a;--blue:#4da3ff;--orange:#ffb347}"
            "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1280px;margin:0 auto;padding:20px 16px 60px}"
            "h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 8px}.sub{color:var(--mut);margin-bottom:14px}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}"
            ".card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:12px}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}"
            "th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--mut);font-weight:600}td:first-child,th:first-child{text-align:left}"
            "td.n{text-align:left;white-space:normal}.bar{display:inline-block;height:10px;background:var(--blue);border-radius:2px;vertical-align:middle;margin-right:6px}"
            ".note{background:var(--card);border-left:4px solid var(--orange);padding:8px 12px;border-radius:4px;margin:12px 0;color:var(--mut)}</style></head><body><main>";
    html += "<h1>Raster thread profile</h1>";
    html += fmt::format("<div class=\"sub\">{:.1f} s, {} frames from {}; only the thread shadPS4:GpuRaster is measured. Times are per frame.</div>", seconds, frames, frame_source);
    html += "<div class=\"cards\">";
    html += fmt::format("<div class=\"card\"><b>{:.1f} fps</b><span>frame {:.2f} ms</span></div>", nframes / seconds, frame_ms);
    html += fmt::format("<div class=\"card\"><b>{:.2f} ms</b><span>raster thread busy per frame ({:.0f} % of the frame)</span></div>", busy_ms_frame, busy_ms_frame / frame_ms * 100.0);
    html += fmt::format("<div class=\"card\"><b>{:.2f} ms</b><span>raster thread idle per frame</span></div>", idle_ms / nframes);
    for (u32 k = 1; k <= 4; ++k) {
        if (kind_count[k] != 0) {
            html += fmt::format("<div class=\"card\"><b>{:.0f} / {:.2f} ms</b><span>{} per frame / their time ({:.1f} us each)</span></div>", static_cast<double>(kind_count[k]) / nframes, ms(kind_ticks[k]) / nframes, KindNames[k],
                                us(kind_ticks[k]) / static_cast<double>(kind_count[k]));
        }
    }
    html += "</div>";
    html += fmt::format("<div class=\"note\">{:.0f} scopes per frame were timed. A scope costs roughly 20 to 40 ns (an estimate, not measured): about {:.1f} ms per frame of this profile is the profile itself. Compare shapes, not absolute numbers.</div>",
                        scopes_frame, scopes_frame * 30e-6);
    html += "<h2>Where the raster thread's time goes (each phase's own time, without the phases inside it)</h2><table><tr><th>phase</th><th>ms / frame</th><th>% of busy</th><th>calls / frame</th><th>mean us</th><th>max us</th></tr>";
    const double top_ms = flat_order.empty() ? 1.0 : ms(g_flat[flat_order[0]].excl) / nframes;
    for (const u32 p : flat_order) {
        const Stat& s = g_flat[p];
        const double v = ms(s.excl) / nframes;
        if (v < 0.005) {
            break;
        }
        html += fmt::format("<tr><td><span class=\"bar\" style=\"width:{:.0f}px\"></span>{}</td><td>{:.3f}</td><td>{:.1f}</td><td>{:.1f}</td><td>{:.1f}</td><td>{:.1f}</td></tr>", v / top_ms * 220.0, Escape(PhaseNames[p]), v,
                            v / busy_ms_frame * 100.0, static_cast<double>(s.count) / nframes, us(s.excl) / static_cast<double>(s.count), us(s.max));
    }
    html += "</table><h2>The phase tree (inclusive / own time)</h2><table><tr><th>phase</th><th>inclusive ms / frame</th><th>own ms / frame</th><th>calls / frame</th><th>mean us</th><th>max us</th></tr>";
    const auto html_node = [&](auto&& self, size_t i) -> void {
        const Node& n = nodes[i];
        if (ms(n.stat.incl) / nframes < 0.002) {
            return;
        }
        html += fmt::format("<tr><td style=\"padding-left:{}px\">{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.1f}</td><td>{:.1f}</td><td>{:.1f}</td></tr>", 8 + depth_of(n.path) * 18, Escape(leaf_name(n.path)),
                            ms(n.stat.incl) / nframes, ms(n.stat.excl) / nframes, static_cast<double>(n.stat.count) / nframes, us(n.stat.incl) / static_cast<double>(n.stat.count), us(n.stat.max));
        for (const size_t c : n.children) {
            self(self, c);
        }
    };
    for (const size_t r : roots) {
        html_node(html_node, r);
    }
    html += "</table><h2>Operations by engine caller (Ghidra address of the function that issued them)</h2><table><tr><th>kind</th><th>caller</th><th>pass class</th><th>per frame</th><th>ms / frame</th><th>us each</th><th>max us</th></tr>";
    for (size_t i = 0; i < std::min<size_t>(40, callers.size()); ++i) {
        const CallerStat& c = *callers[i];
        html += fmt::format("<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.1f}</td><td>{:.1f}</td></tr>", KindNames[c.kind], c.caller == 0 ? std::string("-") : fmt::format("{:#x}", c.caller),
                            ClassNames[c.cls & 15], static_cast<double>(c.count) / nframes, ms(c.ticks) / nframes, us(c.ticks) / static_cast<double>(c.count), us(c.max));
    }
    html += "</table><h2>The slowest single operations</h2><table><tr><th>us</th><th>kind</th><th>caller</th><th>pass class</th><th>the phases that took the most</th></tr>";
    for (const SlowOp* s : slow) {
        std::vector<u32> order;
        for (u32 p = 1; p < PhaseCount; ++p) {
            if (s->excl[p] != 0) {
                order.push_back(p);
            }
        }
        std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return s->excl[a] > s->excl[b]; });
        std::string parts;
        for (size_t k = 0; k < std::min<size_t>(4, order.size()); ++k) {
            parts += fmt::format("{}{} {:.0f} us", k ? "; " : "", PhaseNames[order[k]], us(s->excl[order[k]]));
        }
        html += fmt::format("<tr><td>{:.1f}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"n\">{}</td></tr>", us(s->ticks), KindNames[s->kind], s->caller == 0 ? std::string("-") : fmt::format("{:#x}", s->caller),
                            ClassNames[s->cls & 15], Escape(parts));
    }
    html += "</table></main></body></html>";

    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    const auto stamp = static_cast<long long>(std::time(nullptr));
    const auto html_path = dir / fmt::format("raster_{}.html", stamp);
    const auto text_path = dir / fmt::format("raster_{}.txt", stamp);
    std::ofstream(html_path, std::ios::binary) << html;
    std::ofstream(text_path, std::ios::binary) << text;
    std::ofstream(dir / "raster_latest.txt", std::ios::binary) << text;
    std::ofstream(dir / "raster_latest.html", std::ios::binary) << html;
    text += fmt::format("\n(written to {} and {})\n", html_path.string(), text_path.string());
#ifdef _WIN32
    if (g_open.load()) {
        ShellExecuteA(nullptr, "open", html_path.string().c_str(), nullptr, nullptr, 1);
    }
#endif

    nlohmann::json j;
    j["seconds"] = seconds;
    j["frames"] = frames;
    j["busy_ms_per_frame"] = busy_ms_frame;
    j["idle_ms_per_frame"] = idle_ms / nframes;
    j["scopes_per_frame"] = scopes_frame;
    auto& phases = j["phases_by_own_time"] = nlohmann::json::array();
    for (const u32 p : flat_order) {
        const Stat& s = g_flat[p];
        phases.push_back({{"phase", PhaseNames[p]}, {"ms_per_frame", ms(s.excl) / nframes}, {"calls_per_frame", static_cast<double>(s.count) / nframes}, {"max_us", us(s.max)}});
    }
    auto& by_caller = j["by_caller"] = nlohmann::json::array();
    for (size_t i = 0; i < std::min<size_t>(40, callers.size()); ++i) {
        const CallerStat& c = *callers[i];
        by_caller.push_back({{"kind", KindNames[c.kind]}, {"caller", fmt::format("{:#x}", c.caller)}, {"class", ClassNames[c.cls & 15]}, {"per_frame", static_cast<double>(c.count) / nframes},
                             {"ms_per_frame", ms(c.ticks) / nframes}, {"us_each", us(c.ticks) / static_cast<double>(c.count)}});
    }
    {
        std::scoped_lock lock{g_mutex};
        g_full = text;
        g_json = j.dump(1);
    }
    g_report_ns.store(FrameTimeline::Now());
    LOG_INFO(Render, "raster profiler: {} frames, busy {:.2f} ms per frame, report {}", frames, busy_ms_frame, html_path.string());
}

} // namespace Core::RasterProf
