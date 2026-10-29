// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/native_light_recorder.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <cstring>
#include <fmt/format.h>

#include "core/memory.h"

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
struct FrameSig {
    u64 present{~0ull};
    u32 counts[8]{};
    u64 signature{};
    u64 image_generation_first{}, image_generation_last{}; ///< the texture cache's image generation at the first and the last operation of the frame (an image registered or unregistered moves it)
    float watch_mean[8]{};
    u32 watch_nonzero[8]{};
};
struct Watched {
    u64 address{};
    u64 bytes{};
};
Watched watched[8];
u32 watched_count = 0;
u64 last_sampled_present = ~0ull;
constexpr size_t FrameRing = 1024;
FrameSig frame_ring[FrameRing];

const std::vector<u64>& Callers() {
    static const std::vector<u64> list = [] {
        std::vector<u64> out;
        const char* e = std::getenv("TLG_LIGHT_RECORDER_CALLERS");
        if (e == nullptr || *e == 0) {
            return std::vector<u64>{0x1c6456d, 0x1c67f59, 0x1c68036, 0x1c44f64, 0x247ea43, 0x1c98a17};
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

std::string (*event_describer)(void*, u64, u64) = nullptr;
void* event_describer_context = nullptr;

void SetEventDescriber(std::string (*describe)(void*, u64, u64), void* context) {
    event_describer_context = context;
    event_describer = describe;
}

void Watch(u64 address, u64 bytes) {
    if (!recording || current.caller != 0x247ea43 || bytes == 0 || bytes > 8192) {
        return;
    }
    for (u32 i = 0; i < watched_count; ++i) {
        if (watched[i].address == address) {
            watched[i].bytes = bytes;
            return;
        }
    }
    if (watched_count < 8) {
        watched[watched_count++] = {address, bytes};
    }
}

void FrameOp(u64 present, u32 kind, u64 key, u64 image_generation) {
    if (present != last_sampled_present) {
        // the previous present is complete: what the watched blocks hold now (the CPU's view of the luminance copies)
        FrameSig& done = frame_ring[last_sampled_present % FrameRing];
        if (last_sampled_present != ~0ull && done.present == last_sampled_present) {
            auto* memory = Core::Memory::Instance();
            for (u32 i = 0; i < watched_count; ++i) {
                if (!memory->IsValidMapping(watched[i].address, watched[i].bytes)) {
                    continue;
                }
                double sum = 0;
                u32 nonzero = 0;
                for (u64 b = 0; b + 4 <= watched[i].bytes; b += 4) {
                    float v = 0;
                    std::memcpy(&v, reinterpret_cast<const void*>(watched[i].address + b), 4);
                    if (v == v && v > -1e30f && v < 1e30f) {
                        sum += v;
                    }
                    nonzero += v != 0 ? 1 : 0;
                }
                done.watch_mean[i] = static_cast<float>(sum / std::max<u64>(watched[i].bytes / 4, 1));
                done.watch_nonzero[i] = nonzero;
            }
        }
        last_sampled_present = present;
    }
    FrameSig& f = frame_ring[present % FrameRing];
    if (f.present != present) {
        f = {};
        f.present = present;
    }
    if (f.counts[0] + f.counts[1] + f.counts[2] + f.counts[3] + f.counts[4] + f.counts[5] + f.counts[6] + f.counts[7] == 0) {
        f.image_generation_first = image_generation;
    }
    f.image_generation_last = image_generation;
    ++f.counts[kind & 7];
    u64 h = (key ^ (u64{kind} << 56)) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    f.signature += h;
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
        for (const FrameSig& f : frame_ring) {
            if (f.present != ~0ull && f.present + 220 >= detail && f.present <= detail + 8) {
                text += fmt::format("frame {} dispatches {} draws z/shadow {} gbuffer {} forward {} other {} indirect {} untagged {} signature {:016x}\n", f.present, f.counts[0], f.counts[1], f.counts[2], f.counts[3], f.counts[4],
                                    f.counts[5], f.counts[6], f.signature);
                text += fmt::format("imggen {} {} {}\n", f.present, f.image_generation_first, f.image_generation_last);
                std::string watch;
                for (u32 i = 0; i < watched_count; ++i) {
                    watch += fmt::format(" [{:#x} mean {:.6g} nonzero {}]", watched[i].address, f.watch_mean[i], f.watch_nonzero[i]);
                }
                if (!watch.empty()) {
                    text += fmt::format("watch {}{}\n", f.present, watch);
                }
            }
        }
        if (event_describer != nullptr) {
            u64 first = ~0ull, last = 0;
            for (const FrameSig& f : frame_ring) {
                if (f.present != ~0ull && f.present + 6 >= detail && f.present <= detail + 1 && f.image_generation_first != 0) {
                    first = std::min(first, f.image_generation_first);
                    last = std::max(last, f.image_generation_last);
                }
            }
            if (first != ~0ull) {
                text += fmt::format("image events between generation {} and {} (presents {} to {}):\n", first, last, detail - 6, detail + 1);
                text += event_describer(event_describer_context, first > 0 ? first - 1 : 0, last);
            }
        }
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
