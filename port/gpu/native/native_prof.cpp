// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_prof.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <x86intrin.h>
#endif

#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/thread_usage.h"

namespace Core::Prof {

namespace {

constexpr u32 MaxSites = 1024;
constexpr u32 NodeSlots = 8192;
constexpr u32 StackSize = 48;
constexpr u32 SlowCount = 12;
constexpr u32 MaxTags = 4;

struct Stat {
    u64 count{0}, incl{0}, excl{0}, max{0};
    u64 child_calls{0}; ///< scopes that ran directly inside this one
    u64 inner{0};       ///< scopes that ran inside this one at any depth
};

struct Node {
    u64 hash{0}; ///< the path from the outermost scope down (0: empty slot)
    u64 parent{0};
    u16 site{0};
    Stat stat{};
};

struct Frame {
    u64 t0{0}, child{0}, hash{0}, inner{0}, child_calls{0};
    u16 site{0};
};

struct CounterStat {
    u64 count{0}, sum{0}, max{0};
};

struct TagValue {
    u16 kind{0};
    u64 value{0};
    const char* label{nullptr};
};

struct TagStat {
    u16 root{0}, kind{0};
    u64 value{0};
    const char* label{nullptr};
    u64 count{0}, ticks{0}, max{0}, inner{0};
};

struct SlowOp {
    u64 ticks{0};
    u16 site{0};
    u32 tag_count{0};
    std::array<TagValue, MaxTags> tags{};
    std::array<u16, 4> top_site{};
    std::array<u64, 4> top_ticks{};
};

}  // namespace

struct ThreadData {
    ThreadData() : nodes(NodeSlots), flat(MaxSites), root_excl(MaxSites), counters(MaxSites) {}
    void Reset() {
        std::fill(nodes.begin(), nodes.end(), Node{});
        std::fill(flat.begin(), flat.end(), Stat{});
        std::fill(root_excl.begin(), root_excl.end(), 0);
        std::fill(counters.begin(), counters.end(), CounterStat{});
        depth = 0;
        touched.clear();
        tag_count = 0;
        tag_stats.clear();
        slow.fill({});
        scopes = 0;
        node_overflow = 0;
    }
    u32 tid{0};
    std::string name;
    std::vector<Node> nodes;
    std::vector<Stat> flat;
    std::vector<u64> root_excl; ///< own time of each probe inside the operation that is open (reset when it ends)
    std::vector<CounterStat> counters;
    std::array<Frame, StackSize> stack{};
    u32 depth{0};
    std::vector<u16> touched;
    std::array<TagValue, MaxTags> tags{};
    u32 tag_count{0};
    std::unordered_map<u64, TagStat> tag_stats;
    std::array<SlowOp, SlowCount> slow{};
    u64 scopes{0};
    u64 node_overflow{0};
};

namespace {

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

std::mutex g_mutex;
std::vector<Site*>& Sites() {
    static std::vector<Site*> sites(1, nullptr);
    return sites;
}
std::vector<std::string>& Groups() {
    static std::vector<std::string> groups;
    return groups;
}
std::vector<const TagKind*>& TagKinds() {
    static std::vector<const TagKind*> kinds(1, nullptr);
    return kinds;
}
std::vector<std::unique_ptr<ThreadData>>& Threads() {
    static std::vector<std::unique_ptr<ThreadData>> threads;
    return threads;
}
std::unordered_set<u32> g_selected;
std::unordered_map<u32, std::string> g_tid_names;
bool g_select_all = false;
std::atomic<u32> g_epoch{1};
u32 g_cfg_level = 0;
u64 g_cfg_mask = 0;
double g_cost_ticks = 0.0; ///< the measured cost of one scope (Enter + Leave) in timestamp counter ticks
u64 g_tick0 = 0, g_ns0 = 0;

std::mutex g_report_mutex;
std::string g_full, g_json;
std::atomic<bool> g_prof_capture{false};
std::atomic<bool> g_open{false};
std::atomic<u64> g_report_ns{0};
Config g_config;

thread_local ThreadData* t_data = nullptr;
thread_local u32 t_epoch = 0;

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

inline u64 Mix(u64 parent, u16 site) {
    u64 h = (parent ^ (u64(site) + 1)) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    return h == 0 ? 1 : h;
}

u64 GroupBit(const std::string& name) {
    auto& groups = Groups();
    const auto it = std::find(groups.begin(), groups.end(), name);
    size_t index = static_cast<size_t>(it - groups.begin());
    if (it == groups.end()) {
        groups.push_back(name);
        index = groups.size() - 1;
    }
    return 1ull << std::min<size_t>(index, 63);
}

std::string GroupOf(const char* name) {
    const char* dot = std::strchr(name, '.');
    return dot ? std::string(name, dot) : std::string(name);
}

/// Whether the running capture records the probe. Called with g_mutex held.
bool Wanted(const Site& site) {
    if (!g_running.load(std::memory_order_relaxed) || site.id == 0) {
        return false;
    }
    return site.level <= g_cfg_level && (g_cfg_mask & GroupBit(GroupOf(site.name))) != 0;
}

std::vector<std::string> Split(const std::string& text) {
    std::vector<std::string> out;
    std::string current;
    for (const char c : text) {
        if (c == ',' || c == ' ' || c == ';') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        } else {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

std::string Lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

ThreadData* Attach() {
    t_epoch = g_epoch.load(std::memory_order_relaxed);
    t_data = nullptr;
    const u32 tid = FrameTimeline::ThreadId();
    std::scoped_lock lock{g_mutex};
    if (!g_select_all && !g_selected.contains(tid)) {
        return nullptr;
    }
    for (auto& thread : Threads()) {
        if (thread->tid == tid) {
            t_data = thread.get();
            return t_data;
        }
    }
    auto data = std::make_unique<ThreadData>();
    data->tid = tid;
    const auto it = g_tid_names.find(tid);
    data->name = it != g_tid_names.end() ? it->second : fmt::format("thread {}", tid);
    t_data = data.get();
    Threads().push_back(std::move(data));
    return t_data;
}

inline ThreadData* Current() {
    if (t_epoch != g_epoch.load(std::memory_order_relaxed)) [[unlikely]] {
        return Attach();
    }
    return t_data;
}

Node& NodeFor(ThreadData& td, u64 hash, u64 parent, u16 site) {
    u32 at = static_cast<u32>(hash >> 40) & (NodeSlots - 1);
    for (u32 n = 0; n < NodeSlots; ++n, at = (at + 1) & (NodeSlots - 1)) {
        Node& node = td.nodes[at];
        if (node.hash == hash) {
            return node;
        }
        if (node.hash == 0) {
            node.hash = hash;
            node.parent = parent;
            node.site = site;
            return node;
        }
    }
    ++td.node_overflow;
    return td.nodes[0];
}

void EnterTD(ThreadData& td, u16 site) {
    ++td.scopes;
    if (td.depth == 0) {
        td.touched.clear();
        td.tag_count = 0;
    }
    if (td.depth < StackSize) {
        Frame& f = td.stack[td.depth];
        f = {};
        f.site = site;
        f.hash = Mix(td.depth > 0 ? td.stack[td.depth - 1].hash : 0, site);
        f.t0 = Tick();
    }
    ++td.depth;
}

void FinishRoot(ThreadData& td, const Frame& f, u64 dur) {
    // the tags of the operation
    for (u32 i = 0; i < td.tag_count; ++i) {
        const TagValue& t = td.tags[i];
        const u64 key = Mix(Mix(Mix(0, f.site), t.kind), static_cast<u16>(t.value)) ^ (t.value * 0xC2B2AE3D27D4EB4Full);
        TagStat& s = td.tag_stats[key];
        s.root = f.site;
        s.kind = t.kind;
        s.value = t.value;
        if (t.label != nullptr) {
            s.label = t.label;
        }
        ++s.count;
        s.ticks += dur;
        s.max = std::max(s.max, dur);
        s.inner += f.inner;
    }
    // the slowest operations
    u32 least = 0;
    for (u32 i = 1; i < SlowCount; ++i) {
        if (td.slow[i].ticks < td.slow[least].ticks) {
            least = i;
        }
    }
    if (dur > td.slow[least].ticks) {
        SlowOp& s = td.slow[least];
        s = {};
        s.ticks = dur;
        s.site = f.site;
        s.tag_count = td.tag_count;
        s.tags = td.tags;
        for (const u16 site : td.touched) {
            const u64 own = td.root_excl[site];
            for (u32 k = 0; k < 4; ++k) {
                if (own > s.top_ticks[k]) {
                    for (u32 m = 3; m > k; --m) {
                        s.top_ticks[m] = s.top_ticks[m - 1];
                        s.top_site[m] = s.top_site[m - 1];
                    }
                    s.top_ticks[k] = own;
                    s.top_site[k] = site;
                    break;
                }
            }
        }
    }
    for (const u16 site : td.touched) {
        td.root_excl[site] = 0;
    }
    td.touched.clear();
}

void LeaveTD(ThreadData& td) {
    const u64 now = Tick();
    if (td.depth == 0) {
        return;
    }
    const u32 at = --td.depth;
    if (at >= StackSize) {
        return;
    }
    Frame& f = td.stack[at];
    const u64 dur = now - f.t0;
    const u64 excl = dur > f.child ? dur - f.child : 0;
    u64 parent_hash = 0;
    if (at > 0) {
        Frame& p = td.stack[at - 1];
        p.child += dur;
        p.inner += 1 + f.inner;
        ++p.child_calls;
        parent_hash = p.hash;
    }
    Node& node = NodeFor(td, f.hash, parent_hash, f.site);
    node.stat.count += 1;
    node.stat.incl += dur;
    node.stat.excl += excl;
    node.stat.max = std::max(node.stat.max, dur);
    node.stat.child_calls += f.child_calls;
    node.stat.inner += f.inner;
    Stat& flat = td.flat[f.site];
    flat.count += 1;
    flat.incl += dur;
    flat.excl += excl;
    flat.max = std::max(flat.max, dur);
    flat.child_calls += f.child_calls;
    flat.inner += f.inner;
    if (td.root_excl[f.site] == 0) {
        td.touched.push_back(f.site);
    }
    td.root_excl[f.site] += excl + 1;
    if (at == 0) {
        FinishRoot(td, f, dur);
    }
}

double CalibrateTicks() {
    static Site cal_a{"prof.calibrate_a", 9};
    static Site cal_b{"prof.calibrate_b", 9};
    auto td = std::make_unique<ThreadData>();
    constexpr u32 rounds = 20000;
    const u64 t0 = Tick();
    for (u32 i = 0; i < rounds; ++i) {
        EnterTD(*td, cal_a.id);
        EnterTD(*td, cal_b.id);
        LeaveTD(*td);
        LeaveTD(*td);
    }
    return static_cast<double>(Tick() - t0) / (2.0 * rounds);
}

}  // namespace

Site::Site(const char* name_, u8 level_, Kind kind_) : name{name_}, level{level_}, kind{kind_} {
    std::scoped_lock lock{g_mutex};
    auto& sites = Sites();
    if (sites.size() < MaxSites) {
        id = static_cast<u16>(sites.size());
        sites.push_back(this);
        enabled.store(Wanted(*this), std::memory_order_relaxed);
    }
}

TagKind::TagKind(const char* name_, bool hex_) : name{name_}, hex{hex_} {
    std::scoped_lock lock{g_mutex};
    auto& kinds = TagKinds();
    id = static_cast<u16>(kinds.size());
    kinds.push_back(this);
}

ThreadData* EnterSlow(Site& site) noexcept {
    ThreadData* td = Current();
    if (td == nullptr) {
        return nullptr;
    }
    EnterTD(*td, site.id);
    return td;
}

void Leave(ThreadData* data) noexcept {
    LeaveTD(*data);
}

void CountSlow(Site& site, u64 value) noexcept {
    ThreadData* td = Current();
    if (td == nullptr) {
        return;
    }
    CounterStat& c = td->counters[site.id];
    ++c.count;
    c.sum += value;
    c.max = std::max(c.max, value);
}

void TagSlow(const TagKind& kind, u64 value, const char* label) noexcept {
    ThreadData* td = Current();
    if (td == nullptr || td->depth == 0 || td->tag_count >= MaxTags) {
        return;
    }
    td->tags[td->tag_count++] = {kind.id, value, label};
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// control
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Config DefaultConfig() {
    Config c;
    if (const char* e = std::getenv("TLG_PROF_SECONDS")) {
        c.seconds = static_cast<u32>(std::clamp(std::atoi(e), 1, 60));
    }
    if (const char* e = std::getenv("TLG_PROF_LEVEL")) {
        c.level = static_cast<u32>(std::clamp(std::atoi(e), 1, 9));
    }
    if (const char* e = std::getenv("TLG_PROF_GROUPS")) {
        c.groups = e;
    }
    if (const char* e = std::getenv("TLG_PROF_THREADS")) {
        c.threads = e;
    }
    c.open = true;
    return c;
}

bool Start(const Config& config) {
    g_prof_capture.store(true);
    if (!FrameTimeline::StartCapture(config.seconds, fmt::format("prof_{}", static_cast<long long>(std::time(nullptr))))) {
        g_prof_capture.store(false);
        return false;
    }
    g_open.store(config.open);
    const double cost = CalibrateTicks();
    const auto names = VideoCore::ThreadNames();
    std::scoped_lock lock{g_mutex};
    g_config = config;
    g_cost_ticks = cost;
    g_tid_names.clear();
    for (const auto& [id, name] : names) {
        g_tid_names[id] = name;
    }
    g_selected.clear();
    g_select_all = names.empty();
    std::vector<std::string> wanted = Split(config.threads);
    if (wanted.empty()) {
        wanted.push_back("gpuraster");
    }
    for (const auto& w : wanted) {
        if (w == "all") {
            g_select_all = true;
        }
    }
    for (const auto& [id, name] : names) {
        const std::string lower = Lower(name);
        for (const auto& w : wanted) {
            if (lower.find(w) != std::string::npos) {
                g_selected.insert(id);
            }
        }
    }
    const std::vector<std::string> groups = Split(config.groups);
    g_cfg_mask = ~0ull;
    if (!groups.empty()) {
        g_cfg_mask = 0;
        for (const auto& g : groups) {
            g_cfg_mask |= GroupBit(g);
        }
    }
    g_cfg_level = config.level;
    for (auto& thread : Threads()) {
        thread->Reset();
    }
    g_epoch.fetch_add(1, std::memory_order_relaxed);
    g_ns0 = FrameTimeline::Now();
    g_tick0 = Tick();
    g_running.store(true, std::memory_order_release);
    for (Site* site : Sites()) {
        if (site != nullptr) {
            site->enabled.store(Wanted(*site), std::memory_order_relaxed);
        }
    }
    return true;
}

bool Stop() {
    return FrameTimeline::StopCapture();
}

bool Recording() {
    return g_prof_capture.load() && FrameTimeline::capture_active.load(std::memory_order_relaxed);
}

bool ThisIsProfCapture() {
    return g_prof_capture.load();
}

double ReportAgeSeconds() {
    const u64 at = g_report_ns.load();
    return at == 0 ? 1e9 : static_cast<double>(FrameTimeline::Now() - at) / 1e9;
}

std::string FullText() {
    std::scoped_lock lock{g_report_mutex};
    return g_full;
}

std::string ReportJson() {
    std::scoped_lock lock{g_report_mutex};
    return g_json.empty() ? "{}" : g_json;
}

nlohmann::json SitesJson() {
    nlohmann::json list = nlohmann::json::array();
    std::scoped_lock lock{g_mutex};
    for (const Site* site : Sites()) {
        if (site != nullptr && std::strncmp(site->name, "prof.", 5) != 0) {
            list.push_back({{"name", site->name}, {"group", GroupOf(site->name)}, {"level", site->level}, {"kind", site->kind == Kind::Scope ? "scope" : "counter"}, {"recorded_now", site->enabled.load()}});
        }
    }
    std::sort(list.begin(), list.end(), [](const nlohmann::json& a, const nlohmann::json& b) { return a["name"].get<std::string>() < b["name"].get<std::string>(); });
    return {{"probes", list}, {"tags", [] {
                nlohmann::json tags = nlohmann::json::array();
                for (const TagKind* kind : TagKinds()) {
                    if (kind != nullptr) {
                        tags.push_back(kind->name);
                    }
                }
                return tags;
            }()}};
}

namespace {

std::filesystem::path LogDir() {
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct Table {
    std::string title;
    std::string note;
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
    std::vector<u32> indent; ///< per row, levels of indentation of the first column (empty: none)
};

struct Card {
    std::string value, label;
};

struct ThreadReport {
    std::string name;
    std::vector<Card> cards;
    std::vector<Table> tables;
    nlohmann::json json;
};

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

std::string TextOf(const Table& t) {
    std::vector<size_t> width(t.headers.size(), 0);
    const auto cell = [&](size_t r, size_t c) {
        std::string text = t.rows[r][c];
        if (c == 0 && r < t.indent.size()) {
            text = std::string(t.indent[r] * 2, ' ') + text;
        }
        return text;
    };
    for (size_t c = 0; c < t.headers.size(); ++c) {
        width[c] = t.headers[c].size();
        for (size_t r = 0; r < t.rows.size(); ++r) {
            width[c] = std::max(width[c], cell(r, c).size());
        }
    }
    std::string out = t.title + "\n";
    if (!t.note.empty()) {
        out += "  (" + t.note + ")\n";
    }
    const auto line = [&](const std::vector<std::string>& cells) {
        std::string l = " ";
        for (size_t c = 0; c < cells.size(); ++c) {
            l += " ";
            l += c == 0 ? fmt::format("{:<{}}", cells[c], width[c]) : fmt::format("{:>{}}", cells[c], width[c]);
        }
        return l + "\n";
    };
    out += line(t.headers);
    for (size_t r = 0; r < t.rows.size(); ++r) {
        std::vector<std::string> cells;
        for (size_t c = 0; c < t.headers.size(); ++c) {
            cells.push_back(cell(r, c));
        }
        out += line(cells);
    }
    return out + "\n";
}

std::string HtmlOf(const Table& t) {
    std::string out = fmt::format("<h3>{}</h3>", Escape(t.title));
    if (!t.note.empty()) {
        out += fmt::format("<div class=\"note\">{}</div>", Escape(t.note));
    }
    out += "<table><tr>";
    for (const auto& h : t.headers) {
        out += "<th>" + Escape(h) + "</th>";
    }
    out += "</tr>";
    for (size_t r = 0; r < t.rows.size(); ++r) {
        out += "<tr>";
        for (size_t c = 0; c < t.rows[r].size(); ++c) {
            if (c == 0 && r < t.indent.size()) {
                out += fmt::format("<td style=\"padding-left:{}px\">{}</td>", 8 + t.indent[r] * 18, Escape(t.rows[r][c]));
            } else {
                out += "<td>" + Escape(t.rows[r][c]) + "</td>";
            }
        }
        out += "</tr>";
    }
    return out + "</table>";
}

std::string Base(const char* path) {
    const char* slash = std::strrchr(path, '/');
    const char* back = std::strrchr(path, '\\');
    const char* cut = std::max(slash, back);
    return cut != nullptr ? std::string(cut + 1) : std::string(path);
}

}  // namespace

bool SetBaseline(bool clear) {
    const auto dir = LogDir();
    std::error_code ec;
    if (clear) {
        std::filesystem::remove(dir / "prof_baseline.json", ec);
        return true;
    }
    if (!std::filesystem::exists(dir / "prof_latest.json")) {
        return false;
    }
    std::filesystem::copy_file(dir / "prof_latest.json", dir / "prof_baseline.json", std::filesystem::copy_options::overwrite_existing, ec);
    return !ec;
}

void Analyze(const FrameTimeline::CapEvent* events, u64 count, u64 start_ns, u64 end_ns) {
    g_running.store(false, std::memory_order_release);
    g_prof_capture.store(false);
    {
        std::scoped_lock lock{g_mutex};
        for (Site* site : Sites()) {
            if (site != nullptr) {
                site->enabled.store(false, std::memory_order_relaxed);
            }
        }
    }
    const u64 tick_end = Tick();
    const u64 ns_end = FrameTimeline::Now();
    const double ticks_per_ns = ns_end > g_ns0 ? static_cast<double>(tick_end - g_tick0) / static_cast<double>(ns_end - g_ns0) : 1.0;
    const double c = g_cost_ticks;
    const auto ms = [&](double ticks) { return ticks / ticks_per_ns / 1e6; };
    const auto us = [&](double ticks) { return ticks / ticks_per_ns / 1e3; };

    u64 frames = 0, present = 0;
    for (u64 i = 0; i < count; ++i) {
        if (events[i].type == FrameTimeline::SubmitFlip) {
            ++frames;
        } else if (events[i].type == FrameTimeline::PresentEnd) {
            ++present;
        }
    }
    const char* frame_source = "game submits (SubmitFlip)";
    if (frames < 5) {
        frames = present;
        frame_source = "present thread (PresentEnd)";
    }
    const double seconds = static_cast<double>(end_ns - start_ns) / 1e9;

    std::vector<ThreadData*> threads;
    {
        std::scoped_lock lock{g_mutex};
        for (auto& t : Threads()) {
            if (t->scopes != 0 || std::any_of(t->counters.begin(), t->counters.end(), [](const CounterStat& s) { return s.count != 0; })) {
                threads.push_back(t.get());
            }
        }
    }
    if (frames < 5 || threads.empty()) {
        const std::string text = fmt::format("Profiler: {} frames and {} threads with data in {:.1f} s: nothing to report (is the game in a scene, and does the selection '{}' / groups '{}' name a thread and probes that ran?).\n", frames,
                                             threads.size(), seconds, g_config.threads.empty() ? "GpuRaster" : g_config.threads, g_config.groups.empty() ? "all" : g_config.groups);
        std::scoped_lock lock{g_report_mutex};
        g_full = text;
        g_json = "{}";
        g_report_ns.store(FrameTimeline::Now());
        return;
    }
    const double nframes = static_cast<double>(frames);
    const double frame_ms = seconds * 1000.0 / nframes;

    // the baseline: thread name -> probe name -> own ms per frame
    std::unordered_map<std::string, std::unordered_map<std::string, double>> baseline;
    bool have_baseline = false;
    {
        std::ifstream in(LogDir() / "prof_baseline.json");
        if (in) {
            const auto j = nlohmann::json::parse(in, nullptr, false);
            if (!j.is_discarded() && j.contains("threads")) {
                for (const auto& t : j["threads"]) {
                    for (const auto& p : t.value("probes", nlohmann::json::array())) {
                        baseline[t.value("name", std::string{})][p.value("name", std::string{})] = p.value("own_ms_per_frame", 0.0);
                    }
                }
                have_baseline = true;
            }
        }
    }

    std::vector<ThreadReport> reports;
    for (ThreadData* td : threads) {
        ThreadReport rep;
        rep.name = td->name;
        const std::string base_key = td->name;
        const auto site_name = [&](u16 id) -> std::string {
            const auto& sites = Sites();
            return id < sites.size() && sites[id] != nullptr ? sites[id]->name : fmt::format("probe {}", id);
        };

        // nodes and their children
        struct N {
            const Node* node;
            std::vector<size_t> children;
        };
        std::vector<N> nodes;
        std::unordered_map<u64, size_t> index;
        for (const Node& n : td->nodes) {
            if (n.hash != 0 && n.stat.count != 0) {
                index[n.hash] = nodes.size();
                nodes.push_back({&n, {}});
            }
        }
        std::vector<size_t> roots;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].node->parent == 0) {
                roots.push_back(i);
            } else if (const auto it = index.find(nodes[i].node->parent); it != index.end()) {
                nodes[it->second].children.push_back(i);
            } else {
                roots.push_back(i);
            }
        }
        const auto own_c = [&](const Stat& s) { return std::max(0.0, static_cast<double>(s.excl) - 0.5 * c * static_cast<double>(s.count + s.child_calls)); };
        const auto incl_c = [&](const Stat& s) { return std::max(0.0, static_cast<double>(s.incl) - c * (0.5 * static_cast<double>(s.count) + static_cast<double>(s.inner))); };
        const auto by_incl = [&](size_t a, size_t b) { return nodes[a].node->stat.incl > nodes[b].node->stat.incl; };
        std::sort(roots.begin(), roots.end(), by_incl);
        for (auto& n : nodes) {
            std::sort(n.children.begin(), n.children.end(), by_incl);
        }
        double busy_raw = 0, busy = 0;
        for (const size_t r : roots) {
            busy_raw += static_cast<double>(nodes[r].node->stat.incl);
            busy += incl_c(nodes[r].node->stat);
        }
        const double scopes_frame = static_cast<double>(td->scopes) / nframes;
        const double cost_ms_frame = scopes_frame * ms(c);

        rep.cards.push_back({fmt::format("{:.2f} ms", ms(busy) / nframes), fmt::format("busy per frame, probe cost removed ({:.0f} % of the frame)", ms(busy) / nframes / frame_ms * 100.0)});
        rep.cards.push_back({fmt::format("{:.2f} ms", ms(busy_raw) / nframes), "busy per frame as measured"});
        rep.cards.push_back({fmt::format("{:.0f}", scopes_frame), fmt::format("scopes per frame, {:.0f} ns each when measured warm = {:.2f} ms per frame at least", ms(c) * 1e6, cost_ms_frame)});

        // the probes ranked by own time
        struct F {
            u16 site;
            double own, own_raw, incl;
            const Stat* stat;
        };
        std::vector<F> flat;
        for (u32 s = 1; s < MaxSites; ++s) {
            const Stat& st = td->flat[s];
            if (st.count != 0) {
                flat.push_back({static_cast<u16>(s), own_c(st), static_cast<double>(st.excl), incl_c(st), &st});
            }
        }
        std::sort(flat.begin(), flat.end(), [](const F& a, const F& b) { return a.own > b.own; });
        {
            Table t;
            t.title = "Probes ranked by their own time (time inside the probe without the probes inside it)";
            t.note = "ms per frame; the cost of the probes is subtracted";
            t.headers = {"probe", "own ms/frame", "% of busy", "calls/frame", "mean us", "max us"};
            if (have_baseline) {
                t.headers.push_back("baseline ms");
                t.headers.push_back("change");
            }
            const auto base_threads = baseline.find(base_key);
            for (const F& f : flat) {
                const double v = ms(f.own) / nframes;
                if (v < 0.002) {
                    break;
                }
                std::vector<std::string> row = {site_name(f.site), fmt::format("{:.3f}", v), fmt::format("{:.1f}", ms(f.own) / ms(busy) * 100.0), fmt::format("{:.1f}", static_cast<double>(f.stat->count) / nframes),
                                                fmt::format("{:.2f}", us(f.own) / static_cast<double>(f.stat->count)), fmt::format("{:.1f}", us(static_cast<double>(f.stat->max)))};
                if (have_baseline) {
                    double before = 0;
                    bool known = false;
                    if (base_threads != baseline.end()) {
                        if (const auto it = base_threads->second.find(site_name(f.site)); it != base_threads->second.end()) {
                            before = it->second;
                            known = true;
                        }
                    }
                    row.push_back(known ? fmt::format("{:.3f}", before) : "-");
                    row.push_back(known ? fmt::format("{:+.3f}", v - before) : "new");
                }
                t.rows.push_back(std::move(row));
            }
            rep.tables.push_back(std::move(t));
        }

        // changes against the baseline (including probes that no longer run)
        if (have_baseline) {
            if (const auto base_threads = baseline.find(base_key); base_threads != baseline.end()) {
                std::unordered_map<std::string, double> now;
                for (const F& f : flat) {
                    now[site_name(f.site)] = ms(f.own) / nframes;
                }
                std::vector<std::tuple<double, std::string, double, double>> changes;
                for (const auto& [name, before] : base_threads->second) {
                    const double after = now.contains(name) ? now[name] : 0.0;
                    changes.emplace_back(after - before, name, before, after);
                }
                for (const auto& [name, after] : now) {
                    if (!base_threads->second.contains(name)) {
                        changes.emplace_back(after, name, 0.0, after);
                    }
                }
                std::sort(changes.begin(), changes.end(), [](const auto& a, const auto& b) { return std::abs(std::get<0>(a)) > std::abs(std::get<0>(b)); });
                Table t;
                t.title = "Biggest changes against the baseline (own ms per frame)";
                t.headers = {"probe", "before", "now", "change"};
                for (size_t i = 0; i < std::min<size_t>(15, changes.size()); ++i) {
                    t.rows.push_back({std::get<1>(changes[i]), fmt::format("{:.3f}", std::get<2>(changes[i])), fmt::format("{:.3f}", std::get<3>(changes[i])), fmt::format("{:+.3f}", std::get<0>(changes[i]))});
                }
                rep.tables.push_back(std::move(t));
            }
        }

        // the tree
        {
            Table t;
            t.title = "How the probes nest (inclusive / own time, probe cost removed)";
            t.headers = {"probe", "inclusive ms/frame", "own ms/frame", "calls/frame", "mean us", "max us"};
            const auto walk = [&](auto&& self, size_t i, u32 depth) -> void {
                const N& n = nodes[i];
                const Stat& st = n.node->stat;
                if (ms(incl_c(st)) / nframes < 0.002) {
                    return;
                }
                t.rows.push_back({site_name(n.node->site), fmt::format("{:.3f}", ms(incl_c(st)) / nframes), fmt::format("{:.3f}", ms(own_c(st)) / nframes), fmt::format("{:.1f}", static_cast<double>(st.count) / nframes),
                                  fmt::format("{:.2f}", us(incl_c(st)) / static_cast<double>(st.count)), fmt::format("{:.1f}", us(static_cast<double>(st.max)))});
                t.indent.push_back(depth);
                for (const size_t child : n.children) {
                    self(self, child, depth + 1);
                }
            };
            for (const size_t r : roots) {
                walk(walk, r, 0);
            }
            rep.tables.push_back(std::move(t));
        }

        // the operations by tag
        {
            std::unordered_map<u16, std::vector<const TagStat*>> by_kind;
            for (const auto& [key, s] : td->tag_stats) {
                by_kind[s.kind].push_back(&s);
            }
            std::vector<u16> kinds;
            for (const auto& [kind, list] : by_kind) {
                kinds.push_back(kind);
            }
            std::sort(kinds.begin(), kinds.end());
            for (const u16 kind : kinds) {
                auto& list = by_kind[kind];
                std::sort(list.begin(), list.end(), [](const TagStat* a, const TagStat* b) { return a->ticks > b->ticks; });
                const TagKind* tk = kind < TagKinds().size() ? TagKinds()[kind] : nullptr;
                Table t;
                t.title = fmt::format("Operations ranked by tag '{}'", tk != nullptr ? tk->name : "?");
                t.note = "the outermost probe each operation ran under; ms per frame, probe cost removed";
                t.headers = {"operation", tk != nullptr ? tk->name : "tag", "per frame", "ms/frame", "us each", "max us"};
                for (size_t i = 0; i < std::min<size_t>(30, list.size()); ++i) {
                    const TagStat& s = *list[i];
                    const double ticks = std::max(0.0, static_cast<double>(s.ticks) - c * (0.5 * static_cast<double>(s.count) + static_cast<double>(s.inner)));
                    const std::string shown = s.label != nullptr ? fmt::format("{}:{}", Base(s.label), s.value) : (tk != nullptr && tk->hex ? fmt::format("{:#x}", s.value) : fmt::format("{}", s.value));
                    t.rows.push_back({site_name(s.root), shown, fmt::format("{:.2f}", static_cast<double>(s.count) / nframes), fmt::format("{:.3f}", ms(ticks) / nframes),
                                      fmt::format("{:.1f}", us(ticks) / static_cast<double>(s.count)), fmt::format("{:.1f}", us(static_cast<double>(s.max)))});
                }
                rep.tables.push_back(std::move(t));
            }
        }

        // counters
        {
            Table t;
            t.title = "Counters";
            t.headers = {"counter", "events/frame", "sum/frame", "max", "mean"};
            std::vector<u32> ids;
            for (u32 s = 1; s < MaxSites; ++s) {
                if (td->counters[s].count != 0) {
                    ids.push_back(s);
                }
            }
            std::sort(ids.begin(), ids.end(), [&](u32 a, u32 b) { return td->counters[a].count > td->counters[b].count; });
            for (const u32 s : ids) {
                const CounterStat& cs = td->counters[s];
                t.rows.push_back({site_name(static_cast<u16>(s)), fmt::format("{:.1f}", static_cast<double>(cs.count) / nframes), fmt::format("{:.1f}", static_cast<double>(cs.sum) / nframes), fmt::format("{}", cs.max),
                                  fmt::format("{:.2f}", static_cast<double>(cs.sum) / static_cast<double>(cs.count))});
            }
            if (!t.rows.empty()) {
                rep.tables.push_back(std::move(t));
            }
        }

        // the slowest operations
        {
            std::vector<const SlowOp*> slow;
            for (const SlowOp& s : td->slow) {
                if (s.ticks != 0) {
                    slow.push_back(&s);
                }
            }
            std::sort(slow.begin(), slow.end(), [](const SlowOp* a, const SlowOp* b) { return a->ticks > b->ticks; });
            Table t;
            t.title = "The slowest single operations";
            t.headers = {"us", "operation", "tags", "what took the time (own us)"};
            for (const SlowOp* s : slow) {
                std::string tags, parts;
                for (u32 i = 0; i < s->tag_count; ++i) {
                    const TagValue& tv = s->tags[i];
                    const TagKind* tk = tv.kind < TagKinds().size() ? TagKinds()[tv.kind] : nullptr;
                    tags += fmt::format("{}{}={}", i ? " " : "", tk != nullptr ? tk->name : "?",
                                        tv.label != nullptr ? fmt::format("{}:{}", Base(tv.label), tv.value) : (tk != nullptr && tk->hex ? fmt::format("{:#x}", tv.value) : fmt::format("{}", tv.value)));
                }
                for (u32 k = 0; k < 4; ++k) {
                    if (s->top_ticks[k] != 0) {
                        parts += fmt::format("{}{} {:.0f}", k ? "; " : "", site_name(s->top_site[k]), us(static_cast<double>(s->top_ticks[k])));
                    }
                }
                t.rows.push_back({fmt::format("{:.1f}", us(static_cast<double>(s->ticks))), site_name(s->site), tags, parts});
            }
            rep.tables.push_back(std::move(t));
        }

        // json
        nlohmann::json j;
        j["name"] = rep.name;
        j["busy_ms_per_frame"] = ms(busy) / nframes;
        j["busy_ms_per_frame_as_measured"] = ms(busy_raw) / nframes;
        j["scopes_per_frame"] = scopes_frame;
        auto& probes = j["probes"] = nlohmann::json::array();
        for (const F& f : flat) {
            probes.push_back({{"name", site_name(f.site)}, {"own_ms_per_frame", ms(f.own) / nframes}, {"own_ms_per_frame_as_measured", ms(f.own_raw) / nframes}, {"inclusive_ms_per_frame", ms(f.incl) / nframes},
                              {"calls_per_frame", static_cast<double>(f.stat->count) / nframes}, {"max_us", us(static_cast<double>(f.stat->max))}});
        }
        auto& counters = j["counters"] = nlohmann::json::array();
        for (u32 s = 1; s < MaxSites; ++s) {
            if (td->counters[s].count != 0) {
                counters.push_back({{"name", site_name(static_cast<u16>(s))}, {"events_per_frame", static_cast<double>(td->counters[s].count) / nframes}, {"sum_per_frame", static_cast<double>(td->counters[s].sum) / nframes},
                                    {"max", td->counters[s].max}});
            }
        }
        if (td->node_overflow != 0) {
            j["node_table_overflow"] = td->node_overflow;
        }
        rep.json = std::move(j);
        reports.push_back(std::move(rep));
    }

    // compose
    const auto stamp = static_cast<long long>(std::time(nullptr));
    const std::string groups_text = g_config.groups.empty() ? "all groups" : g_config.groups;
    const std::string threads_text = g_config.threads.empty() ? "GpuRaster" : g_config.threads;
    std::string text = fmt::format("PROFILE  {:.1f} s, {} frames from {} ({:.1f} fps, {:.2f} ms per frame); level {}, {}, threads: {}{}\n\n", seconds, frames, frame_source, nframes / seconds, frame_ms, g_config.level, groups_text,
                                   threads_text, have_baseline ? "; compared with prof_baseline.json" : "");
    std::string html =
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Profile</title><style>"
        ":root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa3ad;--card:#1d2127;--line:#2c323a;--blue:#4da3ff;--orange:#ffb347}"
        "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1320px;margin:0 auto;padding:20px 16px 60px}"
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 8px;border-top:1px solid var(--line);padding-top:14px}h3{font-size:15px;margin:22px 0 6px}.sub{color:var(--mut);margin-bottom:14px}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}"
        ".card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:12px}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}"
        "th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--mut);font-weight:600}td:first-child,th:first-child{text-align:left}"
        ".note{color:var(--mut);margin:2px 0 6px;font-size:12px}</style></head><body><main>";
    html += "<h1>Profile</h1>";
    html += fmt::format("<div class=\"sub\">{:.1f} s, {} frames from {} ({:.1f} fps, {:.2f} ms per frame); level {}, {}, threads: {}{}. Times are per frame, with the measured cost of the probes removed.</div>", seconds, frames,
                        frame_source, nframes / seconds, frame_ms, g_config.level, Escape(groups_text), Escape(threads_text), have_baseline ? "; compared with prof_baseline.json" : "");
    nlohmann::json json;
    json["seconds"] = seconds;
    json["frames"] = frames;
    json["fps"] = nframes / seconds;
    json["level"] = g_config.level;
    json["groups"] = groups_text;
    json["threads_selected"] = threads_text;
    json["scope_cost_ns_warm"] = ms(c) * 1e6;
    auto& jt = json["threads"] = nlohmann::json::array();
    for (const ThreadReport& rep : reports) {
        text += fmt::format("=== thread {} ===\n", rep.name);
        html += fmt::format("<h2>thread {}</h2><div class=\"cards\">", Escape(rep.name));
        for (const Card& card : rep.cards) {
            text += fmt::format("  {}  {}\n", card.value, card.label);
            html += fmt::format("<div class=\"card\"><b>{}</b><span>{}</span></div>", Escape(card.value), Escape(card.label));
        }
        text += "\n";
        html += "</div>";
        for (const Table& t : rep.tables) {
            text += TextOf(t);
            html += HtmlOf(t);
        }
        jt.push_back(rep.json);
    }
    html += "</main></body></html>";

    const auto dir = LogDir();
    const auto html_path = dir / fmt::format("prof_{}.html", stamp);
    const auto text_path = dir / fmt::format("prof_{}.txt", stamp);
    const auto json_path = dir / fmt::format("prof_{}.json", stamp);
    std::ofstream(html_path, std::ios::binary) << html;
    std::ofstream(text_path, std::ios::binary) << text;
    std::ofstream(json_path, std::ios::binary) << json.dump(1);
    std::ofstream(dir / "prof_latest.html", std::ios::binary) << html;
    std::ofstream(dir / "prof_latest.txt", std::ios::binary) << text;
    std::ofstream(dir / "prof_latest.json", std::ios::binary) << json.dump(1);
    // the same files under the name the raster thread reports had (raster_latest.*), next to the others
    std::ofstream(dir / "raster_latest.html", std::ios::binary) << html;
    std::ofstream(dir / "raster_latest.txt", std::ios::binary) << text;
    std::ofstream(dir / "raster_latest.json", std::ios::binary) << json.dump(1);
    text += fmt::format("(written to {}, {} and {}; /prof_baseline keeps this report to compare the next ones against)\n", html_path.string(), text_path.string(), json_path.string());
#ifdef _WIN32
    if (g_open.load()) {
        ShellExecuteA(nullptr, "open", html_path.string().c_str(), nullptr, nullptr, 1);
    }
#endif
    {
        std::scoped_lock lock{g_report_mutex};
        g_full = text;
        g_json = json.dump(1);
    }
    g_report_ns.store(FrameTimeline::Now());
    LOG_INFO(Render, "profiler: {} frames, {} threads, report {}", frames, reports.size(), html_path.string());
}

}  // namespace Core::Prof
