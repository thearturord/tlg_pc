// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native_ui_trace.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include <fmt/format.h>

#include "native_draw.h"
#include "native_debug_server.h"
#include "native_registry.h"

namespace Core::UiTrace {

namespace {

constexpr size_t FramesKept = 8;
constexpr size_t ReadsPerDraw = 24;
constexpr u64 HashBytesMax = 4096;

struct Read {
    std::string kind;
    u64 address{};
    u64 size{};
    u64 hash{};
    std::string hex;
    std::string floats;
};

struct Draw {
    u64 seq{};
    u64 ps{};
    u64 vs{};
    u64 color0{};
    u32 count{};
    u32 instances{};
    u32 caller{};
    std::vector<Read> reads;
    u32 reads_total{};
};

struct Frame {
    u32 frame{};
    std::vector<Draw> draws;
};

/// A change between two consecutive frames of traced draws (the title menu is static, so a change is what the owner sees as a glitch). The colour block (160 bytes: the blinking text) is left out of the comparison.
struct Change {
    Frame before;
    Frame after;
};
constexpr size_t ChangesKept = 40;

/// A GPU write into the memory of an image that a traced draw sampled.
struct Write {
    const char* by{};
    u64 address{};
    u64 size{};
    u64 seq{};
    u64 caller{};
    u32 frame{};
    u64 number{}; ///< running number: the order of the writes
};
constexpr size_t WritesKept = 3000;
constexpr size_t WatchMax = 64;
/// A memory copy of at least 64 KB (the text pictures are made by copies of 128 KB).
struct Copy {
    u64 dst{};
    u64 src{};
    u64 size{};
    u64 seq{};
    u64 caller{};
    u32 frame{};
    u64 number{};
};
constexpr size_t CopiesKept = 400;
constexpr size_t RecentKept = 6000;
std::deque<Copy> copies;
std::deque<Write> recent; ///< every GPU write of the last moments: the writes of a copy's source happen before the copy
std::deque<Write> writes;
std::vector<std::pair<u64, u64>> src_watch; ///< sources of the kept copies
std::vector<std::pair<u64, u64>> watch; ///< [address, address + size) of the images traced draws sampled
u64 write_number = 0;

std::atomic<bool> armed{false};
std::mutex mutex;
std::vector<u32> ps_filter;
std::deque<Frame> frames;
std::deque<Change> changes;
u64 changes_total = 0;
u32 stable_run = 0;
constexpr u32 StableFrames = 20;
thread_local bool t_active = false;
thread_local Draw t_draw;
thread_local u32 t_frame = 0;

u64 Fnv(const u8* data, u64 size) {
    u64 h = 1469598103934665603ull;
    for (u64 i = 0; i < size; ++i) {
        h = (h ^ data[i]) * 1099511628211ull;
    }
    return h;
}

std::vector<u64> Signature(const Frame& f) {
    std::vector<u64> sig;
    for (const Draw& d : f.draws) {
        u64 h = 1469598103934665603ull ^ d.count ^ (d.reads_total * 131ull);
        for (const Read& r : d.reads) {
            if (r.size == 160) {
                continue;
            }
            h = (h ^ r.hash ^ r.size) * 1099511628211ull;
        }
        sig.push_back(h);
    }
    return sig;
}

std::string FrameJson(const Frame& fr) {
    std::string out = fmt::format("{{\"frame\":{},\"draws\":[", fr.frame);
    for (size_t d = 0; d < fr.draws.size(); ++d) {
        const Draw& dr = fr.draws[d];
        out += fmt::format("{}{{\"seq\":{},\"ps\":\"{:x}\",\"vs\":\"{:x}\",\"rt0\":\"{:x}\",\"count\":{},\"instances\":{},\"caller\":\"{:x}\",\"reads_total\":{},\"reads\":[", d ? "," : "", dr.seq, dr.ps, dr.vs,
                           dr.color0, dr.count, dr.instances, dr.caller, dr.reads_total);
        for (size_t i = 0; i < dr.reads.size(); ++i) {
            const Read& r = dr.reads[i];
            out += fmt::format("{}{{\"kind\":\"{}\",\"address\":\"{:x}\",\"size\":{},\"hash\":\"{:016x}\",\"hex\":\"{}\",\"floats\":\"{}\"}}", i ? "," : "", r.kind, r.address, r.size, r.hash, r.hex, r.floats);
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

bool EnvOn() {
    const char* e = std::getenv("TLG_UI_TRACE");
    return e != nullptr && e[0] == '1';
}

std::vector<u32> ParseList(const std::string& text) {
    std::vector<u32> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find(',', pos);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string item = text.substr(pos, end - pos);
        if (!item.empty()) {
            out.push_back(static_cast<u32>(std::strtoull(item.c_str(), nullptr, 16)));
        }
        pos = end + 1;
    }
    return out;
}

void EnsureInit() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* ps = std::getenv("TLG_UI_TRACE_PS");
        std::scoped_lock lock{mutex};
        ps_filter = ParseList(ps != nullptr && *ps != 0 ? ps : "bb4d66bf");
        if (EnvOn()) {
            armed.store(true);
        }
    });
}

} // namespace

bool Active() {
    return t_active;
}

bool Armed() {
    EnsureInit();
    return armed.load(std::memory_order_relaxed);
}

void Arm(bool on, const std::string& ps_list) {
    EnsureInit();
    std::scoped_lock lock{mutex};
    if (!ps_list.empty()) {
        ps_filter = ParseList(ps_list);
    }
    armed.store(on);
    if (!on) {
        frames.clear();
    }
}

void BeginDraw(u64 seq, u64 ps_hash, u64 vs_hash, u64 color0, u32 count, u32 instances, u32 caller, u32 frame) {
    if (!Armed()) {
        t_active = false;
        return;
    }
    {
        std::scoped_lock lock{mutex};
        if (std::ranges::find(ps_filter, static_cast<u32>(ps_hash & 0xffffffffu)) == ps_filter.end()) {
            t_active = false;
            return;
        }
    }
    t_draw = Draw{};
    t_draw.seq = seq;
    t_draw.ps = ps_hash;
    t_draw.vs = vs_hash;
    t_draw.color0 = color0;
    t_draw.count = count;
    t_draw.instances = instances;
    t_draw.caller = caller;
    t_frame = frame;
    t_active = true;
}

void NoteRead(const char* kind, u64 address, u64 size) {
    if (!t_active) {
        return;
    }
    ++t_draw.reads_total;
    if (t_draw.reads.size() >= ReadsPerDraw || address == 0 || size == 0) {
        return;
    }
    Read r;
    r.kind = kind != nullptr ? kind : "?";
    r.address = address;
    r.size = size;
    const u64 hashed = std::min(size, HashBytesMax);
    const u8* data = reinterpret_cast<const u8*>(address);
    r.hash = Fnv(data, hashed);
    const u64 shown = std::min<u64>(size, 64);
    for (u64 i = 0; i < shown; ++i) {
        r.hex += fmt::format("{:02x}", data[i]);
    }
    for (u64 i = 0; i + 4 <= shown; i += 4) {
        float f;
        std::memcpy(&f, data + i, 4);
        r.floats += fmt::format("{}{:.4g}", i == 0 ? "" : " ", f);
    }
    t_draw.reads.push_back(std::move(r));
}

void NoteImage(u64 address, u64 size, const std::string& description) {
    if (!t_active) {
        return;
    }
    ++t_draw.reads_total;
    if (t_draw.reads.size() >= ReadsPerDraw) {
        return;
    }
    if (address != 0 && size != 0) {
        std::scoped_lock lock{mutex};
        const std::pair<u64, u64> range{address, address + size};
        if (watch.size() < WatchMax && std::ranges::find(watch, range) == watch.end()) {
            watch.push_back(range);
        }
    }
    Read r;
    r.kind = "image: " + description;
    r.address = address;
    r.size = size;
    t_draw.reads.push_back(std::move(r));
}

namespace {
bool Overlaps(const std::vector<std::pair<u64, u64>>& list, u64 address, u64 size) {
    for (const auto& [lo, hi] : list) {
        if (address < hi && address + size > lo) {
            return true;
        }
    }
    return false;
}

void KeepWrite(const Write& w) {
    writes.push_back(w);
    while (writes.size() > WritesKept) {
        writes.pop_front();
    }
}
} // namespace

void NoteGpuWrite(const char* by, u64 address, u64 size) {
    if (!Armed() || size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Write w;
    w.by = by != nullptr ? by : "?";
    w.address = address;
    w.size = size;
    w.seq = Core::NativeRegistry::CurrentDrawSeq();
    w.caller = w.seq != 0 ? Core::NativeDraw::RecordCallerGhidra(w.seq) : 0;
    w.frame = static_cast<u32>(Core::DebugServer::PresentCount());
    w.number = ++write_number;
    if (Overlaps(watch, address, size) || Overlaps(src_watch, address, size)) {
        KeepWrite(w);
    } else {
        recent.push_back(w);
        while (recent.size() > RecentKept) {
            recent.pop_front();
        }
    }
}

void NoteGpuCopy(u64 dst, u64 src, u64 size) {
    if (!Armed() || size < 65536) {
        return;
    }
    std::scoped_lock lock{mutex};
    Copy c;
    c.dst = dst;
    c.src = src;
    c.size = size;
    c.seq = Core::NativeRegistry::CurrentDrawSeq();
    c.caller = c.seq != 0 ? Core::NativeDraw::RecordCallerGhidra(c.seq) : 0;
    c.frame = static_cast<u32>(Core::DebugServer::PresentCount());
    c.number = ++write_number;
    copies.push_back(c);
    while (copies.size() > CopiesKept) {
        copies.pop_front();
    }
    const std::pair<u64, u64> range{src, src + size};
    if (src_watch.size() < 512 && std::ranges::find(src_watch, range) == src_watch.end()) {
        src_watch.push_back(range);
    }
    // the writes of the source that happened before this copy
    for (auto it = recent.begin(); it != recent.end();) {
        if (it->address < src + size && it->address + it->size > src) {
            KeepWrite(*it);
            it = recent.erase(it);
        } else {
            ++it;
        }
    }
}

void EndDraw() {
    if (!t_active) {
        return;
    }
    t_active = false;
    std::scoped_lock lock{mutex};
    if (frames.empty() || frames.back().frame != t_frame) {
        // the newest frame is complete: compare it with the one before
        if (frames.size() >= 2) {
            const Frame& done = frames[frames.size() - 1];
            const Frame& prev = frames[frames.size() - 2];
            if (done.frame == prev.frame + 1) {
                if (Signature(done) == Signature(prev)) {
                    ++stable_run;
                } else {
                    // only a change out of a steady menu counts: screen transitions change every frame and would fill the log
                    if (stable_run >= StableFrames) {
                        ++changes_total;
                        changes.push_back(Change{prev, done});
                        while (changes.size() > ChangesKept) {
                            changes.pop_front();
                        }
                    }
                    stable_run = 0;
                }
            } else {
                stable_run = 0;
            }
        }
        frames.push_back(Frame{t_frame, {}});
        while (frames.size() > FramesKept) {
            frames.pop_front();
        }
    }
    frames.back().draws.push_back(std::move(t_draw));
    t_draw = Draw{};
}

std::string Json() {
    EnsureInit();
    std::scoped_lock lock{mutex};
    std::string out = fmt::format("{{\"armed\":{},\"ps_filter\":[", armed.load() ? "true" : "false");
    for (size_t i = 0; i < ps_filter.size(); ++i) {
        out += fmt::format("{}\"{:x}\"", i ? "," : "", ps_filter[i]);
    }
    out += fmt::format("],\"changes_total\":{},\"changes\":[", changes_total);
    for (size_t c = 0; c < changes.size(); ++c) {
        out += fmt::format("{}{{\"before\":{},\"after\":{}}}", c ? "," : "", FrameJson(changes[c].before), FrameJson(changes[c].after));
    }
    out += fmt::format("],\"gpu_writes_total\":{},\"gpu_copies\":[", write_number);
    for (size_t i = 0; i < copies.size(); ++i) {
        const Copy& c = copies[i];
        out += fmt::format("{}{{\"n\":{},\"frame\":{},\"dst\":\"{:x}\",\"src\":\"{:x}\",\"size\":{},\"seq\":{},\"caller\":\"{:x}\"}}", i ? "," : "", c.number, c.frame, c.dst, c.src, c.size, c.seq, c.caller);
    }
    out += "],\"gpu_writes\":[";
    std::vector<Write> sorted(writes.begin(), writes.end());
    std::ranges::sort(sorted, {}, &Write::number);
    for (size_t i = 0; i < sorted.size(); ++i) {
        const Write& w = sorted[i];
        out += fmt::format("{}{{\"n\":{},\"frame\":{},\"by\":\"{}\",\"address\":\"{:x}\",\"size\":{},\"seq\":{},\"caller\":\"{:x}\"}}", i ? "," : "", w.number, w.frame, w.by, w.address, w.size, w.seq, w.caller);
    }
    out += "],\"frames\":[";
    for (size_t f = 0; f < frames.size(); ++f) {
        out += (f ? "," : "") + FrameJson(frames[f]);
    }
    out += "]}";
    return out;
}

} // namespace Core::UiTrace
