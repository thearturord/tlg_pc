// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/native_light_recorder.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/path_util.h"

namespace Core::LightRecorder {

namespace {
struct Record {
    u64 present{}, caller{}, program{};
    std::vector<Entry> entries;
};
std::mutex mutex;
std::deque<Record> ring;
thread_local Record current;
thread_local bool recording = false;
constexpr size_t RingSize = 800;

const std::vector<u64>& Callers() {
    static const std::vector<u64> list = [] {
        std::vector<u64> out;
        const char* e = std::getenv("TLG_LIGHT_RECORDER_CALLERS");
        if (e == nullptr || *e == 0) {
            return std::vector<u64>{0x1c6456d, 0x1c67f59, 0x1c68036, 0x1c44f64};
        }
        for (const char* at = e; *at != 0;) {
            char* end = nullptr;
            const u64 value = std::strtoull(at, &end, 16);
            if (end == at) {
                break;
            }
            out.push_back(value);
            at = *end == ',' ? end + 1 : end;
        }
        return out;
    }();
    return list;
}
} // namespace

bool On() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_LIGHT_RECORDER");
        return e != nullptr && *e == '1';
    }();
    return on;
}

bool IsLightCaller(u64 caller) {
    for (const u64 c : Callers()) {
        if (c == caller) {
            return true;
        }
    }
    return false;
}

void Begin(u64 present, u64 caller, u64 program) {
    current = {};
    current.present = present;
    current.caller = caller;
    current.program = program;
    recording = true;
}

void Note(const Entry& entry) {
    if (recording && current.entries.size() < 48) {
        current.entries.push_back(entry);
    }
}

struct PendingDump {
    std::string tag;
    u64 detail{};
    u64 not_before{};
};
std::vector<PendingDump> pending_dumps;
void WriteDump(const std::string& tag, u64 detail);

void End() {
    if (!recording) {
        return;
    }
    recording = false;
    const u64 present = current.present;
    std::vector<PendingDump> due;
    {
        std::scoped_lock lock{mutex};
        ring.push_back(std::move(current));
        while (ring.size() > RingSize) {
            ring.pop_front();
        }
        for (auto it = pending_dumps.begin(); it != pending_dumps.end();) {
            if (present >= it->not_before) {
                due.push_back(std::move(*it));
                it = pending_dumps.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const PendingDump& d : due) {
        WriteDump(d.tag, d.detail);
    }
}

void Dump(const char* tag, u64 detail) {
    std::scoped_lock lock{mutex};
    pending_dumps.push_back({tag, detail, detail + 4});
}

void WriteDump(const std::string& tag, u64 detail) {
    std::string text;
    {
        std::scoped_lock lock{mutex};
        text += fmt::format("light recorder: {} {} ({} dispatches)\n", tag, detail, ring.size());
        for (const Record& r : ring) {
            text += fmt::format("present {} caller {:#x} program {:#x}\n", r.present, r.caller, r.program);
            for (const Entry& e : r.entries) {
                if (e.kind == 1) {
                    text += fmt::format("  buffer id {} addr {:#x} size {} {}{} hash {:016x} words {:08x} {:08x} {:08x} {:08x}\n", e.id, e.address, e.size, e.written ? "written " : "", e.via == 1 ? "resident-no-sync" : "normal",
                                        e.hash, e.words[0], e.words[1], e.words[2], e.words[3]);
                } else {
                    text += fmt::format("  image  id {} addr {:#x} slot {} uid {} via {} tsharp hash {:016x}\n", e.id, e.address, e.slot, e.uid, e.via == 0 ? "FindImage" : e.via == 1 ? "same-generation" : e.via == 2 ? "kept-across-generation" : "B3-engine-id-table", e.hash);
                }
            }
        }
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / fmt::format("light_recorder_{}.txt", tag);
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (file != nullptr) {
        std::fwrite(text.data(), 1, text.size(), file);
        std::fclose(file);
    }
}

} // namespace Core::LightRecorder
