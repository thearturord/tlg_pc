// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_log.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_flight.h"

namespace Core::Log {

namespace {

constexpr const char* cat_names[] = {"CP", "RASTER", "UPLOAD", "IMAGE", "BUFFER", "GEN", "HOOK", "GATE", "PRESENT", "GPU", "RUNTIME", "SAVE"};
constexpr const char* level_names[] = {"error", "warn", "info", "trace"};
constexpr u32 MaxPerSecond = 10;
constexpr u64 RotateBytes = 20ull << 20;
constexpr u32 FilesKept = 5;

struct Config {
    Level level[static_cast<size_t>(Cat::Count)];
    Config() {
        for (auto& l : level) {
            l = Level::Warn;
        }
        // TLG_LOG=GEN:trace,UPLOAD:info,ALL:info
        const char* env = std::getenv("TLG_LOG");
        if (env == nullptr) {
            return;
        }
        std::string text = env;
        size_t at = 0;
        while (at < text.size()) {
            size_t end = text.find(',', at);
            if (end == std::string::npos) {
                end = text.size();
            }
            const std::string item = text.substr(at, end - at);
            at = end + 1;
            const size_t colon = item.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const std::string name = item.substr(0, colon);
            const std::string value = item.substr(colon + 1);
            int wanted = -1;
            for (int i = 0; i < 4; ++i) {
                if (value == level_names[i]) {
                    wanted = i;
                }
            }
            if (wanted < 0) {
                continue;
            }
            for (size_t c = 0; c < static_cast<size_t>(Cat::Count); ++c) {
                if (name == "ALL" || name == cat_names[c]) {
                    level[c] = static_cast<Level>(wanted);
                }
            }
        }
    }
};

const Config& GetConfig() {
    static const Config config;
    return config;
}

std::mutex file_mutex;
std::ofstream file;
u64 file_bytes = 0;
std::atomic<u64> written{0};
std::atomic<u64> suppressed_total{0};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

const u64 start_ns = NowNs();

void Rotate(const std::filesystem::path& base) {
    std::error_code ec;
    file.close();
    for (u32 i = FilesKept - 1; i >= 1; --i) {
        auto older = base;
        older += "." + std::to_string(i);
        auto newer = base;
        if (i > 1) {
            newer += "." + std::to_string(i - 1);
        }
        std::filesystem::remove(older, ec);
        std::filesystem::rename(newer, older, ec);
    }
    file.open(base, std::ios::binary | std::ios::trunc);
    file_bytes = 0;
}

} // namespace

bool Enabled(Cat cat, Level level) {
    return static_cast<u8>(level) <= static_cast<u8>(GetConfig().level[static_cast<size_t>(cat)]);
}

void Write(Cat cat, Level level, Site& site, const char* file_name, int line, const std::string& text) {
    // the rate limit: at most MaxPerSecond per call site and second
    const u64 now = NowNs();
    u64 window = site.window_start_ns.load(std::memory_order_relaxed);
    if (now - window >= 1'000'000'000ull) {
        if (site.window_start_ns.compare_exchange_strong(window, now)) {
            site.in_window.store(0, std::memory_order_relaxed);
        }
    }
    if (site.in_window.fetch_add(1, std::memory_order_relaxed) >= MaxPerSecond) {
        site.suppressed.fetch_add(1, std::memory_order_relaxed);
        suppressed_total.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 dropped = site.suppressed.exchange(0, std::memory_order_relaxed);

    const char* base = std::strrchr(file_name, '/');
    const char* base2 = std::strrchr(file_name, '\\');
    if (base2 != nullptr && (base == nullptr || base2 > base)) {
        base = base2;
    }
    base = base != nullptr ? base + 1 : file_name;

    Flight::Add(Flight::LogError, 0, level <= Level::Warn ? std::hash<std::string_view>{}(std::string_view{base}) ^ static_cast<u64>(line) : 0, static_cast<u64>(line));
    std::string out = fmt::format("{:10.3f} f{} [{} {}] {}:{} {}{}\n", static_cast<double>(now - start_ns) / 1e6, DebugServer::PresentCount(), cat_names[static_cast<size_t>(cat)],
                                  level_names[static_cast<size_t>(level)], base, line, dropped != 0 ? fmt::format("({} like this suppressed) ", dropped) : std::string{}, text);
    {
        std::scoped_lock lock{file_mutex};
        static const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "native.log";
        if (!file.is_open()) {
            file.open(path, std::ios::binary | std::ios::app);
            std::error_code ec;
            file_bytes = std::filesystem::exists(path, ec) ? static_cast<u64>(std::filesystem::file_size(path, ec)) : 0;
        }
        if (file_bytes >= RotateBytes) {
            Rotate(path);
        }
        file << out;
        file.flush();
        file_bytes += out.size();
    }
    written.fetch_add(1, std::memory_order_relaxed);
    // the engine's own log gets the line too
    if (level == Level::Error) {
        LOG_ERROR(Core, "{}", text);
    } else if (level == Level::Warn) {
        LOG_WARNING(Core, "{}", text);
    }
}

u64 Written() {
    return written.load(std::memory_order_relaxed);
}

u64 Suppressed() {
    return suppressed_total.load(std::memory_order_relaxed);
}

} // namespace Core::Log
