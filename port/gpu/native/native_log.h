// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/types.h"

/// L4 of documents/tlg/native plan/diagnostics-logging-plan.md: the native layer's messages. A category and a level per message, a rate limit per call site (10 per second, the number of
/// suppressed messages is printed with the next one), one file user/log/native.log (20 MB, 5 files kept) whose lines carry time, thread, frame and call site, and every warning or error
/// is also a flight recorder event. Default level: warn for all categories; TLG_LOG=GEN:trace,UPLOAD:info raises single categories (ALL:info raises everything). The lines are also sent
/// to the engine's normal log (shad_log) so the old habits keep working.
namespace Core::Log {

enum class Cat : u8 { CP, RASTER, UPLOAD, IMAGE, BUFFER, GEN, HOOK, GATE, PRESENT, GPU, RUNTIME, SAVE, Count };
enum class Level : u8 { Error, Warn, Info, Trace };

struct Site {
    std::atomic<u64> window_start_ns{0};
    std::atomic<u32> in_window{0};
    std::atomic<u32> suppressed{0};
};

[[nodiscard]] bool Enabled(Cat cat, Level level);

void Write(Cat cat, Level level, Site& site, const char* file, int line, const std::string& text);

/// Messages written to the file so far, and messages dropped by the rate limit (for the stats).
[[nodiscard]] u64 Written();
[[nodiscard]] u64 Suppressed();

} // namespace Core::Log

#define TLG_LOG(cat, level, ...)                                                                                                                                               \
    do {                                                                                                                                                                       \
        if (::Core::Log::Enabled(::Core::Log::Cat::cat, ::Core::Log::Level::level)) {                                                                                          \
            static ::Core::Log::Site tlg_log_site;                                                                                                                             \
            ::Core::Log::Write(::Core::Log::Cat::cat, ::Core::Log::Level::level, tlg_log_site, __FILE__, __LINE__, fmt::format(__VA_ARGS__));                                   \
        }                                                                                                                                                                      \
    } while (false)
