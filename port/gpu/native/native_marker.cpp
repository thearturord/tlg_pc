// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_marker.h"
#include "native/native_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_flight.h"

#ifdef _WIN32
extern "C" char** _environ;
#else
extern char** environ;
#endif

namespace Core::Marker {

namespace {

constexpr u32 MaxMarkers = 20; ///< folders kept; the oldest is deleted when a new one is written
constexpr double FlightSecondsFull = 5.0;
constexpr double FlightSecondsLight = 3.0;
constexpr u32 FramesInDrawTable = 2;

const bool enabled = [] {
    const char* e = std::getenv("TLG_MARKER");
    return e == nullptr || *e != '0';
}();

Row rows[TableRows];
std::atomic<u64> head{0};

struct Job {
    u32 mode{};
    std::shared_ptr<std::promise<std::string>> done;
};

std::mutex queue_mutex;
std::condition_variable queue_cv;
std::deque<Job> queue;
std::jthread writer;
std::once_flag writer_started;
Hooks hooks;
std::atomic<u32> written{0};

std::filesystem::path LogDir() {
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir);
}

/// The next marker number: one above the highest glitch_<n>_ folder in the log folder (so numbers do not repeat after a restart).
u32 NextNumber() {
    u32 highest = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(LogDir(), ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("glitch_", 0) == 0) {
            highest = std::max<u32>(highest, static_cast<u32>(std::strtoul(name.c_str() + 7, nullptr, 10)));
        }
    }
    return highest + 1;
}

/// Keeps the newest MaxMarkers folders.
void PruneOld() {
    std::vector<std::filesystem::path> folders;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(LogDir(), ec)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("glitch_", 0) == 0) {
            folders.push_back(entry.path());
        }
    }
    std::ranges::sort(folders);
    while (folders.size() > MaxMarkers) {
        std::filesystem::remove_all(folders.front(), ec);
        folders.erase(folders.begin());
    }
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file << text;
}

/// Numbers of `now` minus `before`, by path; only leaves that are numbers in both and differ.
void Flatten(const nlohmann::json& j, const std::string& path, std::vector<std::pair<std::string, double>>& out) {
    if (j.is_number()) {
        out.emplace_back(path, j.get<double>());
    } else if (j.is_object()) {
        for (const auto& [key, value] : j.items()) {
            Flatten(value, path.empty() ? key : path + "/" + key, out);
        }
    } else if (j.is_array() && j.size() <= 64) {
        for (size_t i = 0; i < j.size(); ++i) {
            Flatten(j[i], path + "/" + std::to_string(i), out);
        }
    }
}

nlohmann::json CounterDelta(const nlohmann::json& now, const nlohmann::json& before, double age_s) {
    std::vector<std::pair<std::string, double>> a, b;
    Flatten(now, "", a);
    Flatten(before, "", b);
    std::ranges::sort(b);
    struct Moved {
        std::string path;
        double now, before, delta;
    };
    std::vector<Moved> moved;
    for (const auto& [path, value] : a) {
        const auto it = std::ranges::lower_bound(b, std::pair<std::string, double>{path, -1e300});
        const double before_value = it != b.end() && it->first == path ? it->second : 0.0;
        if (path.find("written_unix_s") != std::string::npos || path.find("uptime") != std::string::npos) {
            continue;
        }
        if (value != before_value) {
            moved.push_back({path, value, before_value, value - before_value});
        }
    }
    std::ranges::sort(moved, [](const Moved& x, const Moved& y) { return std::abs(x.delta) > std::abs(y.delta); });
    nlohmann::json out;
    out["baseline_age_s"] = age_s;
    out["counters_moved"] = moved.size();
    out["note"] = "the 400 counters that moved most since the 20 s snapshot: [path, now, before, delta]";
    auto& list = out["moved"] = nlohmann::json::array();
    for (size_t i = 0; i < moved.size() && i < 400; ++i) {
        list.push_back({moved[i].path, moved[i].now, moved[i].before, moved[i].delta});
    }
    return out;
}

std::string Stamp() {
    const std::time_t t = std::time(nullptr);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buffer;
}

std::string DrawTableJson(u32 marker) {
    const u64 end = head.load(std::memory_order_acquire);
    const u64 first = end > TableRows ? end - TableRows : 0;
    u32 newest_frame = 0;
    for (u64 i = first; i < end; ++i) {
        newest_frame = std::max(newest_frame, rows[i & (TableRows - 1)].frame);
    }
    const u32 oldest_frame = newest_frame >= FramesInDrawTable - 1 ? newest_frame - (FramesInDrawTable - 1) : 0;
    std::string out = fmt::format("{{\"marker\": {}, \"frames\": [{}, {}], \"rows_in_table\": {}, \"columns\": \"ms before the marker, seq, caller (Ghidra address), frame, kind (0 draw, 1 indirect draw, 2 dispatch, 3 indirect dispatch), "
                                  "draw class, flags (1 from a record, 2 native plan), vs or compute hash, ps hash, colour target, depth target, colour targets, width, height, count, instances\", \"rows\": [",
                                  marker, oldest_frame, newest_frame, end - first);
    bool any = false;
    for (u64 i = first; i < end; ++i) {
        const Row r = rows[i & (TableRows - 1)]; // a copy: the raster thread may be overwriting the oldest rows
        if (r.tick == 0 || r.frame < oldest_frame) {
            continue;
        }
        out += fmt::format("{}[{:.3f}, {}, {}, {}, {}, {}, {}, \"{:x}\", \"{:x}\", \"{:x}\", \"{:x}\", {}, {}, {}, {}, {}]", any ? ", " : "", Flight::MsAgo(r.tick), r.seq, r.caller, r.frame, r.kind,
                           r.draw_class, r.flags, r.vs_hash, r.ps_hash, r.color0, r.depth, r.colors, r.width, r.height, r.count, r.instances);
        any = true;
    }
    out += "]}";
    return out;
}

std::string SwitchesNotAtDefault() {
    std::string out;
#ifdef _WIN32
    char** env = _environ;
#else
    char** env = environ;
#endif
    for (; env != nullptr && *env != nullptr; ++env) {
        if (std::strncmp(*env, "TLG_", 4) == 0) {
            out += fmt::format("  {}\n", *env);
        }
    }
    return out.empty() ? "  (none)\n" : out;
}

std::string WriteMarker(u32 mode) {
    const u32 number = NextNumber();
    const std::string stamp = Stamp();
    const auto dir = LogDir() / fmt::format("glitch_{:03}_{}", number, stamp);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const bool full = mode == 1;

    // the flight recorder's tail
    WriteText(dir / "flight.json", Flight::Report(Flight::RingEvents, false, (full ? FlightSecondsFull : FlightSecondsLight) * 1000.0));

    // the counters that moved against the 20 s snapshot
    nlohmann::json now = DebugServer::CountersSnapshot();
    nlohmann::json before;
    double age = 0;
    if (DebugServer::LastPeriodicSnapshot(before, age)) {
        WriteText(dir / "counters_delta.json", CounterDelta(now, before, age).dump(1));
    } else {
        WriteText(dir / "counters_delta.json", "{\"note\": \"no 20 s snapshot yet (the game ran less than 20 s, or the debug server is off)\"}");
    }
    if (full) {
        WriteText(dir / "counters.json", now.dump(1));
        WriteText(dir / "draws.json", DrawTableJson(number));
        if (hooks.screenshot) {
            hooks.screenshot((dir / "frame_presented.png").string());
        }
    }

    std::string note = fmt::format("glitch marker {} ({}), written {}\n", number, full ? "full: F6" : "light: F7", stamp);
    note += fmt::format("uptime {:.1f} s, frames presented {}\n", DebugServer::UptimeMs() / 1000.0, DebugServer::PresentCount());
    if (now.contains("/status")) {
        const auto& status = now["/status"];
        for (const char* key : {"fps_5s", "fps", "uptime_s"}) {
            if (status.contains(key)) {
                note += fmt::format("status.{}: {}\n", key, status[key].dump());
            }
        }
    }
    note += fmt::format("flight events written: {}, draw table rows: {}, native.log lines: {} (rate limit dropped {})\n", Flight::EventsWritten(), head.load(), Log::Written(), Log::Suppressed());
    note += "TLG_ environment (switches that were set for this run):\n" + SwitchesNotAtDefault();
    note += "\nWhat was seen (write one line here):\n";
    WriteText(dir / "note.txt", note);

    PruneOld();
    written.fetch_add(1);
    LOG_INFO(Render, "glitch marker {} written to {}", number, dir.string());
    return dir.string();
}

void WriterMain(std::stop_token stop) {
    Common::SetCurrentThreadName("tlg:marker");
    while (!stop.stop_requested()) {
        Job job;
        {
            std::unique_lock lock{queue_mutex};
            queue_cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return !queue.empty(); });
            if (queue.empty()) {
                continue;
            }
            job = std::move(queue.front());
            queue.pop_front();
        }
        std::string folder;
        try {
            folder = WriteMarker(job.mode);
            if (hooks.toast) {
                hooks.toast(job.mode == 1 ? 7 : 8);
            }
        } catch (const std::exception& e) {
            TLG_LOG(RUNTIME, Error, "glitch marker failed: {}", e.what());
            folder = std::string{"failed: "} + e.what();
        }
        if (job.done) {
            job.done->set_value(folder);
        }
    }
}

void Enqueue(Job job) {
    std::call_once(writer_started, [] { writer = std::jthread{WriterMain}; });
    {
        std::scoped_lock lock{queue_mutex};
        queue.push_back(std::move(job));
    }
    queue_cv.notify_one();
}

} // namespace

bool Enabled() {
    return enabled;
}

u64 RowsWritten() {
    return head.load(std::memory_order_relaxed);
}

void NoteRow(const Row& row) {
    if (!enabled) {
        return;
    }
    const u64 index = head.fetch_add(1, std::memory_order_relaxed);
    Row& slot = rows[index & (TableRows - 1)];
    slot = row;
    slot.tick = Flight::Tick();
}

void SetHooks(const Hooks& h) {
    hooks = h;
}

void Request(u32 mode) {
    if (!enabled) {
        return;
    }
    Flight::Add(Flight::Mark, 0, written.load() + 1, mode);
    Enqueue({mode, nullptr});
}

u32 Written() {
    return written.load();
}

std::string RequestForTest(u32 mode) {
    Flight::Add(Flight::Mark, 0, written.load() + 1, mode);
    auto done = std::make_shared<std::promise<std::string>>();
    auto future = done->get_future();
    Enqueue({mode, done});
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return "timeout";
    }
    return future.get();
}

} // namespace Core::Marker
