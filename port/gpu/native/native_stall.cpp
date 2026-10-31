// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_stall.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_draw.h"

namespace Core::Stall {

namespace {

using FrameTimeline::CapEvent;

std::mutex g_mutex;
std::string g_screen, g_full, g_json;
std::atomic<bool> g_stall_capture{false};
std::atomic<u64> g_report_ns{0};
std::atomic<bool> g_open_in_browser{false};

const char* const kPage =
#include "native/native_stall_page.inc"
    ;

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall ShellExecuteA(void*, const char*, const char*, const char*, const char*, int);
#endif

struct Wait {
    u32 thread{};
    u64 end_ns{};
    u64 dur_ns{};
    u32 kind{};
    u64 object{};
    u64 caller{};
};

constexpr const char* KindNames[] = {"?", "condition variable", "semaphore", "sleep", "event queue", "mutex", "other"};

/// Our own waits and the GPU, per frame (nanoseconds).
enum Metric : u32 {
    MFrontQueueFull,
    MFrontDrain,
    MRasterIdle,
    MEncoderIdle,
    MRasterHandoff,
    MCpWait,
    MFlipWait,
    MDingDong,
    MGpuBusy,
    MGameWaitTotal,
    MRasterWork,
    MFrontWork,
    MEncoderWork,
    MetricCount,
};
constexpr const char* MetricNames[MetricCount] = {
    "front thread waits for room in the raster queue", "front thread waits for queued writes (drain)", "raster thread idle (no work from the game)",
    "encoder thread idle", "raster thread waits for the encoder queue", "command processor waits on a memory label", "command processor waits on the flip label",
    "Gnm submit waits (submission lock / GPU idle)", "GPU busy (sum of pass times)", "all game thread waits (sum over threads)", "raster thread working (draws, dispatches, copies)", "front thread working (planning draws)", "encoder thread working (replay, submit)",
};

struct Frame {
    u64 start{}, end{};
    double ms() const { return static_cast<double>(end - start) / 1e6; }
    std::array<u64, MetricCount> metric{};
    bool slow{};
};

struct Group {
    std::string label;
    double normal_ms{}, slow_ms{};
    u64 normal_n{}, slow_n{};
};

double Percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5))];
}

std::string GhidraAddress(u64 runtime) {
    if (runtime == 0) {
        return "-";
    }
    const u64 base = NativeDraw::ImageBase();
    if (runtime < base || runtime >= base + 0x7000000) {
        return fmt::format("host {:#x}", runtime); // inside the runtime or a library, not the game's own code
    }
    return fmt::format("{:#x}", runtime - base + 0x1000000);
}

} // namespace

bool Start(u32 seconds, bool open_in_browser) {
    g_stall_capture.store(true);
    g_open_in_browser.store(open_in_browser);
    if (!FrameTimeline::StartCapture(seconds, fmt::format("stall_{}", static_cast<long long>(std::time(nullptr))))) {
        g_stall_capture.store(false);
        return false;
    }
    return true;
}

bool Stop() {
    return FrameTimeline::StopCapture();
}

bool Recording() {
    return g_stall_capture.load() && FrameTimeline::capture_active.load(std::memory_order_relaxed);
}

bool ThisIsStallCapture() {
    return g_stall_capture.load();
}

double ReportAgeSeconds() {
    const u64 at = g_report_ns.load();
    if (at == 0) {
        return 1e9;
    }
    return static_cast<double>(FrameTimeline::Now() - at) / 1e9;
}

std::string ScreenText() {
    std::scoped_lock lock{g_mutex};
    return g_screen;
}
std::string FullText() {
    std::scoped_lock lock{g_mutex};
    return g_full;
}
std::string ReportJson() {
    std::scoped_lock lock{g_mutex};
    return g_json.empty() ? "{}" : g_json;
}

void Analyze(const CapEvent* events, u64 count, u64 start_ns, u64 end_ns, const std::map<u32, std::string>& threads, const std::map<u64, std::string>& object_names) {
    g_stall_capture.store(false);
    const auto thread_name = [&](u32 id) {
        const auto it = threads.find(id);
        return it != threads.end() ? it->second : fmt::format("tid {}", id);
    };

    // 1. The frames: one per game submit with a flip. Fall back to the end of the present thread's flips.
    std::vector<u64> marks;
    for (u64 i = 0; i < count; ++i) {
        if (events[i].type == FrameTimeline::SubmitFlip) {
            marks.push_back(events[i].ns);
        }
    }
    const char* frame_source = "game submits (SubmitFlip)";
    if (marks.size() < 20) {
        marks.clear();
        for (u64 i = 0; i < count; ++i) {
            if (events[i].type == FrameTimeline::PresentEnd) {
                marks.push_back(events[i].ns);
            }
        }
        frame_source = "present thread (PresentEnd)";
    }
    std::sort(marks.begin(), marks.end());
    std::string full;
    if (marks.size() < 20) {
        full = fmt::format("Stall profiler: only {} frames in {:.1f} s, too few to analyse (is the game in a scene?).\n", marks.size(), static_cast<double>(end_ns - start_ns) / 1e9);
        std::scoped_lock lock{g_mutex};
        g_full = full;
        g_screen = full;
        g_json = "{}";
        g_report_ns.store(FrameTimeline::Now());
        return;
    }
    std::vector<Frame> frames;
    frames.reserve(marks.size());
    for (size_t i = 0; i + 1 < marks.size(); ++i) {
        frames.push_back({marks[i], marks[i + 1], {}, false});
    }
    std::vector<double> frame_ms;
    for (const auto& f : frames) {
        frame_ms.push_back(f.ms());
    }
    const double median = Percentile(frame_ms, 0.5), p95 = Percentile(frame_ms, 0.95), p99 = Percentile(frame_ms, 0.99);
    const double max_ms = *std::max_element(frame_ms.begin(), frame_ms.end());
    const double threshold = std::max(median * 1.3, median + 2.0);
    size_t slow_count = 0;
    double slow_excess_ms = 0;
    for (auto& f : frames) {
        f.slow = f.ms() > threshold;
        if (f.slow) {
            ++slow_count;
            slow_excess_ms += f.ms() - median;
        }
    }
    const size_t normal_count = frames.size() - slow_count;

    const auto frame_of = [&](u64 ns) -> Frame* {
        if (ns < frames.front().start || ns >= frames.back().end) {
            return nullptr;
        }
        const auto it = std::upper_bound(frames.begin(), frames.end(), ns, [](u64 v, const Frame& f) { return v < f.end; });
        return it == frames.end() ? nullptr : &*it;
    };

    // 2. The waits of the game's threads with their call sites, and our own waits and the GPU time.
    std::vector<Wait> waits;
    std::unordered_map<u32, size_t> pending; // thread -> index of the GameWait that waits for its site event
    for (u64 i = 0; i < count; ++i) {
        const CapEvent& e = events[i];
        switch (e.type) {
        case FrameTimeline::GameWait: {
            Wait w;
            w.thread = e.thread;
            w.end_ns = e.ns;
            w.dur_ns = e.arg & 0x00ffffffffffffffull;
            w.kind = static_cast<u32>(e.arg >> 56);
            w.object = e.arg2;
            pending[e.thread] = waits.size();
            waits.push_back(w);
            if (Frame* f = frame_of(e.ns)) {
                f->metric[MGameWaitTotal] += w.dur_ns;
            }
            break;
        }
        case FrameTimeline::GameWaitSite:
            if (const auto it = pending.find(e.thread); it != pending.end()) {
                waits[it->second].caller = e.arg;
                pending.erase(it);
            }
            break;
        case FrameTimeline::StageSpan: {
            const u32 stage = static_cast<u32>(e.arg >> 56);
            const u64 dur = e.arg & 0x00ffffffffffffffull;
            Metric m = MetricCount;
            switch (stage) {
            case FrameTimeline::StageFrontQueueFull: m = MFrontQueueFull; break;
            case FrameTimeline::StageFrontDrain: m = MFrontDrain; break;
            case FrameTimeline::StageRasterIdle: m = MRasterIdle; break;
            case FrameTimeline::StageEncoderIdle: m = MEncoderIdle; break;
            case FrameTimeline::StageRasterHandoffWait: m = MRasterHandoff; break;
            case FrameTimeline::StageRasterOp: m = MRasterWork; break;
            case FrameTimeline::StageFrontOp: m = MFrontWork; break;
            case FrameTimeline::StageEncoderReplay:
            case FrameTimeline::StageEncoderSubmit: m = MEncoderWork; break;
            default: break;
            }
            if (m != MetricCount) {
                if (Frame* f = frame_of(e.ns)) {
                    f->metric[m] += dur;
                }
            }
            break;
        }
        case FrameTimeline::CpWait:
            if (Frame* f = frame_of(e.ns)) {
                f->metric[MCpWait] += e.arg & 0x00ffffffffffffffull;
            }
            break;
        case FrameTimeline::FlipWait:
            if (Frame* f = frame_of(e.ns)) {
                f->metric[MFlipWait] += e.arg & 0x00ffffffffffffffull;
            }
            break;
        case FrameTimeline::DingDongWait:
            if (Frame* f = frame_of(e.ns)) {
                f->metric[MDingDong] += e.arg & 0x00ffffffffffffffull;
            }
            break;
        case FrameTimeline::GpuPass:
            if (Frame* f = frame_of(e.ns)) {
                f->metric[MGpuBusy] += e.arg & 0x00ffffffffffffffull;
            }
            break;
        default:
            break;
        }
    }

    // 3. The frame thread (the thread that submits the frames) and what it waits on; then who releases each wait (the chain).
    std::unordered_map<u32, u32> submit_threads;
    for (u64 i = 0; i < count; ++i) {
        if (events[i].type == FrameTimeline::SubmitFlip) {
            ++submit_threads[events[i].thread];
        }
    }
    u32 frame_thread = 0;
    u32 best = 0;
    for (const auto& [tid, n] : submit_threads) {
        if (n > best) {
            best = n;
            frame_thread = tid;
        }
    }
    struct Signal {
        u64 ns;
        u32 thread;
        u64 object;
    };
    struct Release {
        u64 ns;
        u32 type;
        u32 thread;
    };
    std::vector<Signal> signals;
    std::vector<Release> releases; // events that end an event queue wait: the end of pipe signal, fences, flips
    for (u64 i = 0; i < count; ++i) {
        const CapEvent& e = events[i];
        if (e.type == FrameTimeline::GameSignal) {
            signals.push_back({e.ns, e.thread, e.arg2});
        } else if (e.type == FrameTimeline::EopSignaled || e.type == FrameTimeline::FenceSignaled || e.type == FrameTimeline::FlipPrepared || e.type == FrameTimeline::PresentEnd) {
            releases.push_back({e.ns, e.type, e.thread});
        }
    }
    std::sort(signals.begin(), signals.end(), [](const Signal& a, const Signal& b) { return a.ns < b.ns; });
    std::sort(releases.begin(), releases.end(), [](const Release& a, const Release& b) { return a.ns < b.ns; });
    std::unordered_map<u32, std::vector<const Wait*>> waits_by_thread;
    for (const auto& w : waits) {
        waits_by_thread[w.thread].push_back(&w);
    }
    for (auto& [tid, list] : waits_by_thread) {
        std::sort(list.begin(), list.end(), [](const Wait* a, const Wait* b) { return a->end_ns < b->end_ns; });
    }
    const auto object_text = [&](u64 object) {
        const auto it = object_names.find(object);
        return (it != object_names.end() ? it->second : fmt::format("{:#x}", object)).substr(0, 30);
    };
    const auto wait_text = [&](const Wait& w) {
        return fmt::format("{} waits ({}) on {} [call site {}]", thread_name(w.thread), KindNames[std::min<u32>(w.kind, 6)], object_text(w.object), GhidraAddress(w.caller));
    };
    // Follows a wait to what released it, and what that thread was doing before: returns the hops as text and a signature of the thread/object names.
    const auto chain_of = [&](const Wait& first, std::string& signature) {
        std::string text;
        const Wait* w = &first;
        for (int depth = 0; depth < 6 && w; ++depth) {
            text += fmt::format("{}{} ({:.1f} ms)", depth ? "\n          <- released by " : "", wait_text(*w), static_cast<double>(w->dur_ns) / 1e6);
            signature += fmt::format("{}{}:{}", depth ? " <- " : "", thread_name(w->thread), object_text(w->object));
            u32 releaser = 0;
            u64 release_ns = 0;
            std::string how;
            if (w->kind == 4) {
                // event queue: the end of pipe signal, a fence or the flip that was queued just before the wait ended
                const auto it = std::upper_bound(releases.begin(), releases.end(), w->end_ns + 200'000, [](u64 v, const Release& r) { return v < r.ns; });
                for (auto r = it; r != releases.begin();) {
                    --r;
                    if (w->end_ns > r->ns + 3'000'000) {
                        break;
                    }
                    releaser = r->thread;
                    release_ns = r->ns;
                    how = r->type == FrameTimeline::EopSignaled ? "end of pipe signal" : r->type == FrameTimeline::FenceSignaled ? "fence signal" : r->type == FrameTimeline::FlipPrepared ? "flip prepared" : "present end";
                    break;
                }
            } else {
                const auto it = std::upper_bound(signals.begin(), signals.end(), w->end_ns + 200'000, [](u64 v, const Signal& s) { return v < s.ns; });
                for (auto r = it; r != signals.begin();) {
                    --r;
                    if (w->end_ns > r->ns + 3'000'000) {
                        break;
                    }
                    if (r->object == w->object && r->thread != w->thread) {
                        releaser = r->thread;
                        release_ns = r->ns;
                        how = "signal";
                        break;
                    }
                }
            }
            if (releaser == 0) {
                text += "\n          <- (no release event found)";
                signature += " <- ?";
                break;
            }
            // what the releasing thread did just before: a wait that ended shortly before the release
            const Wait* next = nullptr;
            if (const auto list = waits_by_thread.find(releaser); list != waits_by_thread.end()) {
                for (auto r = list->second.rbegin(); r != list->second.rend(); ++r) {
                    if ((*r)->end_ns <= release_ns + 100'000 && release_ns - (*r)->end_ns < 1'500'000) {
                        next = *r;
                        break;
                    }
                    if ((*r)->end_ns + 1'500'000 < release_ns) {
                        break;
                    }
                }
            }
            if (!next) {
                text += fmt::format("\n          <- {} by {}, which was running (not waiting)", how, thread_name(releaser));
                signature += fmt::format(" <- {}:running", thread_name(releaser));
                break;
            }
            w = next;
        }
        return text;
    };

    // The frame thread's waits per frame, grouped by object and call site: ms per normal frame against per slow frame.
    std::unordered_map<std::string, Group> groups;
    double ft_wait_normal = 0, ft_wait_slow = 0;
    for (const auto& w : waits) {
        if (w.thread != frame_thread) {
            continue;
        }
        Frame* f = frame_of(w.end_ns);
        if (!f) {
            continue;
        }
        const std::string label = fmt::format("{:<18} {:<30} call site {}", KindNames[std::min<u32>(w.kind, 6)], object_text(w.object), GhidraAddress(w.caller));
        auto& g = groups[label];
        g.label = label;
        const double ms = static_cast<double>(w.dur_ns) / 1e6;
        (f->slow ? g.slow_ms : g.normal_ms) += ms;
        (f->slow ? ft_wait_slow : ft_wait_normal) += ms;
        ++(f->slow ? g.slow_n : g.normal_n);
    }
    struct Ranked {
        const Group* g;
        double normal_per, slow_per, total_per;
    };
    std::vector<Ranked> ranked;
    for (const auto& [label, g] : groups) {
        const double normal_per = normal_count ? g.normal_ms / static_cast<double>(normal_count) : 0.0;
        const double slow_per = slow_count ? g.slow_ms / static_cast<double>(slow_count) : 0.0;
        ranked.push_back({&g, normal_per, slow_per, (g.normal_ms + g.slow_ms) / static_cast<double>(frames.size())});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) { return a.total_per > b.total_per; });

    // All the chains: for every frame, the frame thread's longest wait and what released it; grouped by the shape of the chain.
    struct ChainGroup {
        std::string text;
        u64 normal_n{}, slow_n{};
        double normal_ms{}, slow_ms{};
    };
    std::unordered_map<std::string, ChainGroup> chain_groups;
    std::vector<std::string> frame_chain(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        const Wait* longest = nullptr;
        if (const auto list = waits_by_thread.find(frame_thread); list != waits_by_thread.end()) {
            for (const Wait* w : list->second) {
                if (w->end_ns >= frames[i].start && w->end_ns < frames[i].end && (!longest || w->dur_ns > longest->dur_ns)) {
                    longest = w;
                }
            }
        }
        if (!longest) {
            continue;
        }
        std::string signature;
        const std::string text = chain_of(*longest, signature);
        frame_chain[i] = text;
        auto& cg = chain_groups[signature];
        if (cg.text.empty()) {
            cg.text = text;
        }
        (frames[i].slow ? cg.slow_n : cg.normal_n)++;
        (frames[i].slow ? cg.slow_ms : cg.normal_ms) += static_cast<double>(longest->dur_ns) / 1e6;
    }
    std::vector<const ChainGroup*> chains;
    for (const auto& [sig, cg] : chain_groups) {
        chains.push_back(&cg);
    }
    std::sort(chains.begin(), chains.end(), [](const ChainGroup* a, const ChainGroup* b) { return a->normal_n + a->slow_n > b->normal_n + b->slow_n; });

    struct MetricRow {
        u32 metric;
        double normal_per, slow_per, excess_total;
    };
    std::vector<MetricRow> metric_rows;
    for (u32 m = 0; m < MetricCount; ++m) {
        u64 normal = 0, slow = 0;
        for (const auto& f : frames) {
            (f.slow ? slow : normal) += f.metric[m];
        }
        const double normal_per = normal_count ? static_cast<double>(normal) / 1e6 / static_cast<double>(normal_count) : 0.0;
        const double slow_per = slow_count ? static_cast<double>(slow) / 1e6 / static_cast<double>(slow_count) : 0.0;
        metric_rows.push_back({m, normal_per, slow_per, (slow_per - normal_per) * static_cast<double>(slow_count)});
    }

    // 4. The text.
    const double seconds = static_cast<double>(end_ns - start_ns) / 1e9;
    double normal_total = 0;
    for (const auto& f : frames) {
        if (!f.slow) {
            normal_total += f.ms();
        }
    }
    const double normal_mean = normal_count ? normal_total / static_cast<double>(normal_count) : median;
    const double slow_mean = slow_count ? (slow_excess_ms + median * static_cast<double>(slow_count)) / static_cast<double>(slow_count) : 0.0;
    std::string screen;
    full += fmt::format("STALL PROFILER  {:.1f} s, {} frames from {}, median {:.2f} ms ({:.0f} fps), p95 {:.2f}, p99 {:.2f}, worst {:.2f} ms\n", seconds, frames.size(), frame_source, median,
                        1000.0 / median, p95, p99, max_ms);
    std::string limiter_screen, limiter_text;
    nlohmann::json shares = nlohmann::json::array();
    {
        const auto share = [&](u32 metric, bool slow) {
            u64 total = 0;
            size_t n = 0;
            for (const auto& f : frames) {
                if (f.slow == slow) {
                    total += f.metric[metric];
                    ++n;
                }
            }
            const double frame = slow ? slow_mean : normal_mean;
            return n && frame > 0 ? static_cast<double>(total) / 1e6 / static_cast<double>(n) / frame * 100.0 : 0.0;
        };
        const struct {
            const char* name;
            u32 metric;
        } workers[] = {{"GPU", MGpuBusy}, {"raster thread", MRasterWork}, {"front thread", MFrontWork}, {"encoder thread", MEncoderWork}};
        std::string line = "share of a normal frame: ";
        double top = 0;
        const char* top_name = "";
        for (const auto& w : workers) {
            const double v = share(w.metric, false);
            shares.push_back({{"name", w.name}, {"pct", v}});
            line += fmt::format("{} {:.0f} %  ", w.name, v);
            if (v > top) {
                top = v;
                top_name = w.name;
            }
        }
        line += fmt::format("| {} waits {:.0f} %\n", thread_name(frame_thread), normal_count ? ft_wait_normal / static_cast<double>(normal_count) / normal_mean * 100.0 : 0.0);
        const std::string verdict = top > 70.0 ? fmt::format("LIMITER: the {} works {:.0f} % of every frame, so it sets the frame time.\n", top_name, top)
                                               : std::string("LIMITER: no single worker is saturated (all below 70 %); the chain of waits below decides.\n");
        shares.push_back({{"name", thread_name(frame_thread) + " (frame thread) waiting"}, {"pct", normal_count ? ft_wait_normal / static_cast<double>(normal_count) / normal_mean * 100.0 : 0.0}});
        limiter_text = verdict.substr(0, verdict.size() - 1);
        full += line + verdict;
        limiter_screen = line + verdict;
    }
    full += fmt::format("slow frames: {} of {} ({:.1f} %), longer than {:.2f} ms (mean {:.2f} ms against {:.2f} ms for the others); the capture itself slows the game, compare shapes\n\n", slow_count,
                        frames.size(), 100.0 * static_cast<double>(slow_count) / static_cast<double>(frames.size()), threshold, slow_mean, normal_mean);
    screen += fmt::format("STALL PROFILER  {} frames, median {:.1f} ms ({:.0f} fps), p99 {:.1f}, worst {:.1f}; slow (> {:.1f} ms): {} ({:.1f} %)\n", frames.size(), median, 1000.0 / median, p99, max_ms,
                          threshold, slow_count, 100.0 * static_cast<double>(slow_count) / static_cast<double>(frames.size()));
    screen += limiter_screen;

    full += fmt::format("THE FRAME THREAD ({}) spends {:.2f} ms of a normal frame ({:.2f} ms) waiting, {:.2f} ms of a slow one ({:.2f} ms). What it waits on (ms per frame: normal -> slow):\n",
                        thread_name(frame_thread), normal_count ? ft_wait_normal / static_cast<double>(normal_count) : 0.0, normal_mean, slow_count ? ft_wait_slow / static_cast<double>(slow_count) : 0.0,
                        slow_mean);
    screen += fmt::format("frame thread {}: waits {:.1f} of {:.1f} ms (normal), {:.1f} of {:.1f} ms (slow)\n", thread_name(frame_thread),
                          normal_count ? ft_wait_normal / static_cast<double>(normal_count) : 0.0, normal_mean, slow_count ? ft_wait_slow / static_cast<double>(slow_count) : 0.0, slow_mean);
    int shown = 0;
    for (const auto& r : ranked) {
        if (shown >= 8 || r.total_per < 0.05) {
            break;
        }
        const std::string line = fmt::format("  {:>7.2f} -> {:>7.2f} ms  {}\n", r.normal_per, r.slow_per, r.g->label);
        full += line;
        if (shown < 3) {
            screen += line;
        }
        ++shown;
    }

    full += "\nWHY IT WAITS: the frame thread's longest wait of every frame, followed to what released it (how many normal / slow frames have this shape):\n";
    screen += "why (the longest wait of each frame, and who released it):\n";
    int chain_shown = 0;
    for (const auto* cg : chains) {
        if (chain_shown >= 4) {
            break;
        }
        full += fmt::format("  [{} normal, {} slow frames; mean first wait {:.1f} / {:.1f} ms]\n      {}\n", cg->normal_n, cg->slow_n, cg->normal_n ? cg->normal_ms / static_cast<double>(cg->normal_n) : 0.0,
                            cg->slow_n ? cg->slow_ms / static_cast<double>(cg->slow_n) : 0.0, cg->text);
        if (chain_shown < 2) {
            screen += fmt::format("  [{} normal / {} slow] {}\n", cg->normal_n, cg->slow_n, cg->text);
        }
        ++chain_shown;
    }

    full += "\nOur threads and the GPU (ms per frame: normal -> slow):\n";
    std::vector<MetricRow> sorted_rows = metric_rows;
    std::sort(sorted_rows.begin(), sorted_rows.end(), [](const MetricRow& a, const MetricRow& b) { return a.excess_total > b.excess_total; });
    int row_shown = 0;
    for (const auto& r : sorted_rows) {
        full += fmt::format("  {:>7.2f} -> {:>7.2f} ms  {}\n", r.normal_per, r.slow_per, MetricNames[r.metric]);
        if (row_shown < 3 && r.metric != MGameWaitTotal) {
            screen += fmt::format("  {:>6.2f} -> {:>6.2f} ms  {}\n", r.normal_per, r.slow_per, MetricNames[r.metric]);
            ++row_shown;
        }
    }

    full += "\nThe worst frames, each with the chain of its longest frame thread wait:\n";
    std::vector<size_t> order(frames.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return frames[a].ms() > frames[b].ms(); });
    for (size_t k = 0; k < std::min<size_t>(6, order.size()); ++k) {
        const Frame& f = frames[order[k]];
        full += fmt::format("  frame at +{:.2f} s: {:.2f} ms  GPU busy {:.1f} ms, raster idle {:.1f} ms\n      {}\n", static_cast<double>(f.start - start_ns) / 1e9, f.ms(), static_cast<double>(f.metric[MGpuBusy]) / 1e6,
                            static_cast<double>(f.metric[MRasterIdle]) / 1e6, frame_chain[order[k]].empty() ? "(no wait of the frame thread)" : frame_chain[order[k]]);
    }

    full += "\nEVERY game thread wait (idle workers included; ms per frame: normal -> slow):\n";
    {
        std::unordered_map<std::string, Group> all;
        for (const auto& w : waits) {
            if (Frame* f = frame_of(w.end_ns)) {
                const std::string label = fmt::format("{:<22} {:<18} {:<30} call site {}", thread_name(w.thread), KindNames[std::min<u32>(w.kind, 6)], object_text(w.object), GhidraAddress(w.caller));
                auto& g = all[label];
                g.label = label;
                (f->slow ? g.slow_ms : g.normal_ms) += static_cast<double>(w.dur_ns) / 1e6;
            }
        }
        std::vector<const Group*> list;
        for (const auto& [label, g] : all) {
            list.push_back(&g);
        }
        std::sort(list.begin(), list.end(), [](const Group* a, const Group* b) { return a->normal_ms + a->slow_ms > b->normal_ms + b->slow_ms; });
        for (size_t i = 0; i < std::min<size_t>(25, list.size()); ++i) {
            full += fmt::format("  {:>7.2f} -> {:>7.2f} ms  {}\n", normal_count ? list[i]->normal_ms / static_cast<double>(normal_count) : 0.0,
                                slow_count ? list[i]->slow_ms / static_cast<double>(slow_count) : 0.0, list[i]->label);
        }
    }

    // JSON for the debug server
    nlohmann::json j;
    j["seconds"] = seconds;
    j["frames"] = frames.size();
    j["frame_source"] = frame_source;
    j["median_ms"] = median;
    j["p95_ms"] = p95;
    j["p99_ms"] = p99;
    j["max_ms"] = max_ms;
    j["slow_threshold_ms"] = threshold;
    j["slow_frames"] = slow_count;
    j["slow_excess_ms"] = slow_excess_ms;
    j["frame_thread"] = thread_name(frame_thread);
    j["frame_thread_waits"] = nlohmann::json::array();
    for (size_t i = 0; i < ranked.size() && i < 20; ++i) {
        j["frame_thread_waits"].push_back({{"what", ranked[i].g->label}, {"normal_ms_per_frame", ranked[i].normal_per}, {"slow_ms_per_frame", ranked[i].slow_per}});
    }
    j["why_it_waits"] = nlohmann::json::array();
    for (size_t i = 0; i < chains.size() && i < 8; ++i) {
        j["why_it_waits"].push_back({{"chain", chains[i]->text}, {"normal_frames", chains[i]->normal_n}, {"slow_frames", chains[i]->slow_n}});
    }
    j["our_threads_and_gpu"] = nlohmann::json::array();
    for (const auto& r : sorted_rows) {
        j["our_threads_and_gpu"].push_back({{"what", MetricNames[r.metric]}, {"normal_ms_per_frame", r.normal_per}, {"slow_ms_per_frame", r.slow_per}, {"excess_total_ms", r.excess_total}});
    }

    const long long stamp = static_cast<long long>(std::time(nullptr));
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    const auto path = dir / fmt::format("stall_{}.txt", stamp);
    std::ofstream(path) << full;

    // The brief: about 25 lines, overwritten on every capture (user/log/stall_latest.txt), for a reader that wants the answer without the long report.
    {
        const auto one_line = [](std::string text) {
            const std::string hop = "\n          <- ";
            for (size_t at = text.find(hop); at != std::string::npos; at = text.find(hop, at)) {
                text.replace(at, hop.size(), " <- ");
            }
            return text;
        };
        std::string brief = fmt::format("STALL BRIEF {:.1f} s, {} frames ({}), median {:.2f} ms ({:.0f} fps), p99 {:.2f}, worst {:.2f}; slow (> {:.2f} ms): {} ({:.1f} %), mean {:.2f} against {:.2f} ms\n", seconds,
                                        frames.size(), frame_source, median, 1000.0 / median, p99, max_ms, threshold, slow_count, 100.0 * static_cast<double>(slow_count) / static_cast<double>(frames.size()),
                                        slow_mean, normal_mean);
        brief += limiter_screen;
        brief += fmt::format("frame thread {} waits {:.1f} of {:.1f} ms (normal), {:.1f} of {:.1f} ms (slow)\n", thread_name(frame_thread), normal_count ? ft_wait_normal / static_cast<double>(normal_count) : 0.0,
                             normal_mean, slow_count ? ft_wait_slow / static_cast<double>(slow_count) : 0.0, slow_mean);
        brief += "its waits (ms per frame, normal -> slow):\n";
        for (size_t i = 0; i < ranked.size() && i < 3; ++i) {
            brief += fmt::format("  {:.2f} -> {:.2f}  {}\n", ranked[i].normal_per, ranked[i].slow_per, ranked[i].g->label);
        }
        brief += "why it waits (longest wait of each frame, who released it; frames normal / slow):\n";
        for (size_t i = 0; i < chains.size() && i < 2; ++i) {
            brief += fmt::format("  [{} / {}] {}\n", chains[i]->normal_n, chains[i]->slow_n, one_line(chains[i]->text));
        }
        brief += "biggest growth in slow frames (ms per frame, normal -> slow):\n";
        int rows = 0;
        for (const auto& r : sorted_rows) {
            if (r.metric == MGameWaitTotal) {
                continue;
            }
            brief += fmt::format("  {:.2f} -> {:.2f}  {}\n", r.normal_per, r.slow_per, MetricNames[r.metric]);
            if (++rows >= 5) {
                break;
            }
        }
        brief += "worst frames:\n";
        for (size_t k = 0; k < std::min<size_t>(3, order.size()); ++k) {
            const Frame& f = frames[order[k]];
            brief += fmt::format("  +{:.2f} s {:.1f} ms, raster {:.1f}, front {:.1f}, GPU {:.1f}: {}\n", static_cast<double>(f.start - start_ns) / 1e9, f.ms(), static_cast<double>(f.metric[MRasterWork]) / 1e6,
                                 static_cast<double>(f.metric[MFrontWork]) / 1e6, static_cast<double>(f.metric[MGpuBusy]) / 1e6, frame_chain[order[k]].empty() ? "(no wait)" : one_line(frame_chain[order[k]]));
        }
        brief += fmt::format("more: stall_{}.txt (long) and stall_{}.html (charts) in the same folder\n", static_cast<long long>(std::time(nullptr)), static_cast<long long>(std::time(nullptr)));
        std::ofstream(dir / "stall_latest.txt") << brief;
    }

    // The page: one self-contained HTML file with the data inside (charts are drawn by its own script).
    nlohmann::json page;
    page["seconds"] = seconds;
    page["frame_count"] = frames.size();
    page["frame_source"] = frame_source;
    page["frame_thread"] = thread_name(frame_thread);
    page["median_ms"] = median;
    page["p99_ms"] = p99;
    page["max_ms"] = max_ms;
    page["slow_frames"] = slow_count;
    page["slow_threshold_ms"] = threshold;
    page["normal_mean"] = normal_mean;
    page["slow_mean"] = slow_mean;
    page["limiter"] = limiter_text;
    page["shares"] = shares;
    std::vector<double> ft_wait_frame(frames.size(), 0.0);
    for (const auto& w : waits) {
        if (w.thread == frame_thread) {
            if (Frame* f = frame_of(w.end_ns)) {
                ft_wait_frame[static_cast<size_t>(f - frames.data())] += static_cast<double>(w.dur_ns) / 1e6;
            }
        }
    }
    page["frames"] = nlohmann::json::array();
    for (size_t i = 0; i < frames.size(); ++i) {
        const Frame& f = frames[i];
        page["frames"].push_back({static_cast<double>(f.start - start_ns) / 1e9, f.ms(), f.slow ? 1 : 0, static_cast<double>(f.metric[MGpuBusy]) / 1e6, static_cast<double>(f.metric[MRasterWork]) / 1e6,
                                  static_cast<double>(f.metric[MFrontWork]) / 1e6, static_cast<double>(f.metric[MEncoderWork]) / 1e6, ft_wait_frame[i]});
    }
    page["frame_chains"] = frame_chain;
    page["frame_thread_waits"] = nlohmann::json::array();
    for (size_t i = 0; i < ranked.size() && i < 16; ++i) {
        page["frame_thread_waits"].push_back({{"what", ranked[i].g->label}, {"normal", ranked[i].normal_per}, {"slow", ranked[i].slow_per}});
    }
    page["why_it_waits"] = nlohmann::json::array();
    for (size_t i = 0; i < chains.size() && i < 8; ++i) {
        page["why_it_waits"].push_back({{"chain", chains[i]->text}, {"normal", chains[i]->normal_n}, {"slow", chains[i]->slow_n}});
    }
    page["metrics"] = nlohmann::json::array();
    for (const auto& r : sorted_rows) {
        page["metrics"].push_back({{"what", MetricNames[r.metric]}, {"normal", r.normal_per}, {"slow", r.slow_per}});
    }
    page["worst"] = nlohmann::json::array();
    for (size_t k = 0; k < std::min<size_t>(8, order.size()); ++k) {
        page["worst"].push_back(order[k]);
    }
    page["all_waits"] = nlohmann::json::array();
    {
        std::unordered_map<std::string, Group> all;
        for (const auto& w : waits) {
            if (Frame* f = frame_of(w.end_ns)) {
                const std::string label = fmt::format("{} | {} | {} | call site {}", thread_name(w.thread), KindNames[std::min<u32>(w.kind, 6)], object_text(w.object), GhidraAddress(w.caller));
                auto& g = all[label];
                g.label = label;
                (f->slow ? g.slow_ms : g.normal_ms) += static_cast<double>(w.dur_ns) / 1e6;
            }
        }
        std::vector<const Group*> list;
        for (const auto& [label, g] : all) {
            list.push_back(&g);
        }
        std::sort(list.begin(), list.end(), [](const Group* a, const Group* b) { return a->normal_ms + a->slow_ms > b->normal_ms + b->slow_ms; });
        for (size_t i = 0; i < std::min<size_t>(40, list.size()); ++i) {
            page["all_waits"].push_back({{"what", list[i]->label}, {"normal", normal_count ? list[i]->normal_ms / static_cast<double>(normal_count) : 0.0},
                                         {"slow", slow_count ? list[i]->slow_ms / static_cast<double>(slow_count) : 0.0}});
        }
    }
    std::string data = page.dump();
    for (size_t at = data.find("</"); at != std::string::npos; at = data.find("</", at + 2)) {
        data.replace(at, 2, "<\\/"); // never close the script tag from inside the data
    }
    std::string html = kPage;
    const std::string token = "/*DATA*/null";
    html.replace(html.find(token), token.size(), data);
    const auto html_path = dir / fmt::format("stall_{}.html", stamp);
    std::ofstream(html_path, std::ios::binary) << html;
    full += fmt::format("\n(written to {} and {})\n", path.string(), html_path.string());
    screen += "report: user/log/stall_<time>.html\n";
#ifdef _WIN32
    if (g_open_in_browser.load()) {
        ShellExecuteA(nullptr, "open", html_path.string().c_str(), nullptr, nullptr, 1);
    }
#endif
    {
        std::scoped_lock lock{g_mutex};
        g_full = full;
        g_screen = screen;
        g_json = j.dump(1);
    }
    g_report_ns.store(FrameTimeline::Now());
    LOG_INFO(Render, "stall profiler: {} frames, {} slow, report {}", frames.size(), slow_count, html_path.string());
}

} // namespace Core::Stall
