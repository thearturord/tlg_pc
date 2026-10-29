// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/native_light_recorder.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <deque>
#include <unordered_map>
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
constexpr size_t RingSize = 400;
struct FrameSig {
    u64 present{~0ull};
    u32 counts[8]{};
    u64 signature{};
    u64 image_generation_first{}, image_generation_last{}; ///< the texture cache's image generation at the first and the last operation of the frame (an image registered or unregistered moves it)
    float watch_mean[8]{};
    u32 watch_nonzero[8]{};
    struct ShadowView {
        u64 address{}, size{};
        u32 draws{};
        u32 words[64]{};
    };
    ShadowView views[6]{};
    u32 view_count{};
    u32 indirect{}, indirect_cpu_modified{}, indirect_gpu_modified{}, indirect_gpu_written{}, indirect_conflict{};
    struct Written {
        u64 address{}, size{}, uid{};
        u32 kind{}, count{};
    };
    Written written[64]{};
    u32 written_count{};
};
struct ShadowSeen {
    u64 size{};
    u32 draws{};
    u32 words[64]{};
};
std::unordered_map<u64, ShadowSeen> shadow_seen; ///< raster thread: the z / shadow draws' small buffers of the frame being recorded
u64 shadow_present = ~0ull;
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

void IndirectArgs(u64 present, bool cpu_modified, bool gpu_modified, bool gpu_written) {
    FrameSig& f = frame_ring[present % FrameRing];
    if (f.present != present) {
        return; // FrameOp opens the frame first
    }
    ++f.indirect;
    f.indirect_cpu_modified += cpu_modified ? 1 : 0;
    f.indirect_gpu_modified += gpu_modified ? 1 : 0;
    f.indirect_gpu_written += gpu_written ? 1 : 0;
    f.indirect_conflict += cpu_modified && (gpu_modified || gpu_written) ? 1 : 0;
}

void ImageWritten(u64 present, u64 address, u64 size, u64 uid, u32 kind) {
    FrameSig& f = frame_ring[present % FrameRing];
    if (f.present != present) {
        return; // FrameOp opens the frame first
    }
    for (u32 i = 0; i < f.written_count; ++i) {
        auto& w = f.written[i];
        if (w.address == address && w.uid == uid && w.kind == kind) {
            ++w.count;
            return;
        }
    }
    if (f.written_count < 64) {
        f.written[f.written_count++] = {address, size, uid, kind, 1};
    }
}

bool Recording() {
    return recording;
}

void CloseShadowFrame(u64 present);

void ShadowBuffer(u64 present, u64 address, u64 size) {
    if (present != shadow_present) {
        CloseShadowFrame(present);
    }
    auto [it, is_new] = shadow_seen.try_emplace(address);
    ++it->second.draws;
    if (is_new) {
        it->second.size = size;
        const u64 bytes = std::min<u64>(size, 256);
        if (Core::Memory::Instance()->IsValidMapping(address, bytes)) {
            std::memcpy(it->second.words, reinterpret_cast<const void*>(address), bytes);
        }
    }
}

/// The frame `shadow_present` is complete: its most shared z / shadow buffers go into the frame census.
void CloseShadowFrame(u64 present) {
    if (shadow_present != ~0ull && !shadow_seen.empty()) {
        std::vector<std::pair<u32, u64>> order;
        for (const auto& [address, seen] : shadow_seen) {
            if (seen.draws >= 4) {
                order.emplace_back(seen.draws, address);
            }
        }
        std::sort(order.rbegin(), order.rend());
        FrameSig& f = frame_ring[shadow_present % FrameRing];
        if (f.present == shadow_present) {
            f.view_count = 0;
            for (const auto& [draws, address] : order) {
                if (f.view_count >= 6) {
                    break;
                }
                auto& v = f.views[f.view_count++];
                const ShadowSeen& seen = shadow_seen[address];
                v.address = address;
                v.size = seen.size;
                v.draws = draws;
                std::memcpy(v.words, seen.words, sizeof(v.words));
            }
        }
    }
    shadow_seen.clear();
    shadow_present = present;
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
                text += fmt::format("indirect {} dispatches {} args_cpu_modified {} args_gpu_modified {} args_gpu_written {} cpu_over_gpu {}\n", f.present, f.indirect, f.indirect_cpu_modified, f.indirect_gpu_modified,
                                    f.indirect_gpu_written, f.indirect_conflict);
                std::string watch;
                for (u32 i = 0; i < watched_count; ++i) {
                    watch += fmt::format(" [{:#x} mean {:.6g} nonzero {}]", watched[i].address, f.watch_mean[i], f.watch_nonzero[i]);
                }
                if (!watch.empty()) {
                    text += fmt::format("watch {}{}\n", f.present, watch);
                }
                for (u32 i = 0; i < f.written_count; ++i) {
                    const auto& w = f.written[i];
                    text += fmt::format("written {} {:#x} size {:#x} uid {} kind {} count {}\n", f.present, w.address, w.size, w.uid, w.kind, w.count);
                }
                for (u32 i = 0; i < f.view_count; ++i) {
                    const auto& v = f.views[i];
                    text += fmt::format("shadowview {} {:#x} size {} draws {} data", f.present, v.address, v.size, v.draws);
                    for (const u32 w : v.words) {
                        text += fmt::format(" {:08x}", w);
                    }
                    text += "\n";
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
                    text += "    data";
                    for (const u32 w : e.words) {
                        text += fmt::format(" {:08x}", w);
                    }
                    text += "\n";
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
