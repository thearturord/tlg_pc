// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "common/path_util.h"
#include "native/native_content.h"
#include "native/native_counter_ring.h"
#include "native/native_marker.h"

namespace Core::CounterRing {
namespace {
std::mutex mutex;
FILE* csv = nullptr;
FILE* marks = nullptr;
u64 rows = 0;
std::vector<u64> values;

std::filesystem::path Dir() {
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir);
}
} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_COUNTER_RING");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}

void NotePresent(u64 tick) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (csv == nullptr) {
        csv = std::fopen((Dir() / "counter_ring.csv").string().c_str(), "wb");
        marks = std::fopen((Dir() / "counter_ring_marks.txt").string().c_str(), "wb");
        if (csv == nullptr) {
            return;
        }
        std::string header = "row,tick,draw_rows";
        for (const std::string& name : NativeContent::CounterNames()) {
            header += "," + name;
        }
        header += "\n";
        std::fwrite(header.data(), 1, header.size(), csv);
    }
    NativeContent::CounterSample(values);
    std::string line = std::to_string(rows) + "," + std::to_string(tick) + "," + std::to_string(Marker::RowsWritten());
    for (const u64 v : values) {
        line += "," + std::to_string(v);
    }
    line += "\n";
    std::fwrite(line.data(), 1, line.size(), csv);
    if (++rows % 60 == 0) {
        std::fflush(csv);
        if (marks != nullptr) {
            std::fflush(marks);
        }
    }
}

void Mark(const char* kind, u64 tick) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (marks != nullptr) {
        std::fprintf(marks, "%s,%llu\n", kind, static_cast<unsigned long long>(tick));
        std::fflush(marks);
    }
}

} // namespace Core::CounterRing
