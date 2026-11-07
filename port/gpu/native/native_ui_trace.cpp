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

std::atomic<bool> armed{false};
std::mutex mutex;
std::vector<u32> ps_filter;
std::deque<Frame> frames;
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

void EndDraw() {
    if (!t_active) {
        return;
    }
    t_active = false;
    std::scoped_lock lock{mutex};
    if (frames.empty() || frames.back().frame != t_frame) {
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
    out += "],\"frames\":[";
    for (size_t f = 0; f < frames.size(); ++f) {
        const Frame& fr = frames[f];
        out += fmt::format("{}{{\"frame\":{},\"draws\":[", f ? "," : "", fr.frame);
        for (size_t d = 0; d < fr.draws.size(); ++d) {
            const Draw& dr = fr.draws[d];
            out += fmt::format("{}{{\"seq\":{},\"ps\":\"{:x}\",\"vs\":\"{:x}\",\"rt0\":\"{:x}\",\"count\":{},\"instances\":{},\"caller\":\"{:x}\",\"reads_total\":{},\"reads\":[", d ? "," : "", dr.seq, dr.ps, dr.vs, dr.color0,
                               dr.count, dr.instances, dr.caller, dr.reads_total);
            for (size_t i = 0; i < dr.reads.size(); ++i) {
                const Read& r = dr.reads[i];
                out += fmt::format("{}{{\"kind\":\"{}\",\"address\":\"{:x}\",\"size\":{},\"hash\":\"{:016x}\",\"hex\":\"{}\",\"floats\":\"{}\"}}", i ? "," : "", r.kind, r.address, r.size, r.hash, r.hex, r.floats);
            }
            out += "]}";
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

} // namespace Core::UiTrace
