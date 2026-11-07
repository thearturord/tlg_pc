// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native_ui_trace.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include <filesystem>
#include <fstream>

#include <fmt/format.h>

#include "common/path_util.h"

#include "native_content.h"
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
    u64 src_hash{}; ///< hash of the source memory when the raster thread took the copy (0: unreadable)
    u64 src_upload_hash{}; ///< hash of the memory the last image upload at the source address read (0: never uploaded)
    u32 src_upload_frame{};
    std::string path;
    u32 src_nonzero_chunks{};
    std::vector<std::array<u64, 5>> srcw; ///< who wrote the source range when this copy ran: {copy number, frame, 1 = copy, overlap start, overlap size} (time correct chain)
    int queued{0}; ///< readbacks queued for this copy (diagnostics: a result that never arrives)
};
constexpr size_t CopiesKept = 3000;

/// Hash of a guest range in 64 byte chunks (HashChunksGuarded) folded into one value; nonzero chunk count in `nonzero`.
u64 RangeHash(u64 address, u64 size, u32* nonzero) {
    const u64 chunks = size / 64;
    std::vector<u32> hashes(chunks);
    if (chunks == 0 || !Core::NativeContent::HashChunksGuarded(address, chunks, 64, hashes.data())) {
        return 0;
    }
    static u32 zero_hash = 0;
    static bool zero_known = false;
    if (!zero_known) {
        std::vector<u8> z(64, 0);
        Core::NativeContent::HashChunksGuarded(reinterpret_cast<u64>(z.data()), 1, 64, &zero_hash);
        zero_known = true;
    }
    u64 h = 1469598103934665603ull;
    u32 nz = 0;
    for (const u32 v : hashes) {
        h = (h ^ v) * 1099511628211ull;
        nz += v != zero_hash ? 1 : 0;
    }
    if (nonzero != nullptr) {
        *nonzero = nz;
    }
    return h;
}
constexpr size_t RecentKept = 6000;
std::deque<Copy> copies;
std::deque<Write> recent; ///< every GPU write of the last moments: the writes of a copy's source happen before the copy
std::deque<Write> writes;
std::vector<std::pair<u64, u64>> src_watch; ///< sources of the kept copies
std::vector<std::pair<u64, u64>> watch; ///< [address, address + size) of the images traced draws sampled
u64 write_number = 0;

/// Provenance (2026-11-07): who made the memory of every range the GPU wrote last: [lo, hi) -> writer. Ranges never overlap. A copy names its source, so the chain image <- copy <- source <- ... can be followed.
struct Prov {
    u64 hi{};
    const char* by{};
    u64 src{}; ///< for a copy: the source address (0 otherwise)
    u32 frame{};
    u64 seq{};
    u64 caller{};
    u64 number{};
};
std::map<u64, Prov> prov;
/// The last upload of every small image from guest memory: the memory hash before the bytes were read and after the upload, to compare with the memory when a draw samples the image.
struct Upload {
    u64 size{};
    u64 before{};
    u64 after{};
    u32 frame{};
    u32 count{};
    bool source_gpu{};
    std::string page_map; ///< one character per 4 KB page: G = marked GPU written, . = not (read from guest memory)
};
std::map<u64, Upload> uploads;
struct CopyResult {
    u32 nonzero_chunks{};
    u64 hash{};
    s64 src_nonzero_chunks{-1}; ///< the source image just before an image copy (-1 unknown)
    u64 src_hash{};
};
std::map<u64, CopyResult> copy_results; ///< by copy number
u64 last_copy_number = 0;
struct Shot {
    u32 frame{};
    u32 last_asked{};
    u64 size{};
    std::string file;
    std::string info;
};
std::map<u64, Shot> shots;
constexpr size_t ProvMax = 400000;

std::string SrcwJson(const Copy& c) {
    std::string out = "[";
    for (size_t i = 0; i < c.srcw.size(); ++i) {
        const auto& w = c.srcw[i];
        out += fmt::format("{}[{},{},{},\"{:x}\",{}]", i ? "," : "", w[0], w[1], w[2], w[3], w[4]);
    }
    return out + "]";
}

void ProvSet(u64 lo, u64 size, const Prov& p0) {
    if (size == 0) {
        return;
    }
    if (prov.size() > ProvMax) {
        prov.clear();
    }
    const u64 hi = lo + size;
    auto it = prov.upper_bound(lo);
    if (it != prov.begin()) {
        auto prev = std::prev(it);
        if (prev->second.hi > lo) {
            Prov tail = prev->second;
            const u64 prev_lo = prev->first;
            const bool has_tail = tail.hi > hi;
            prev->second.hi = lo;
            if (prev_lo == lo) {
                prov.erase(prev);
            }
            if (has_tail) {
                if (tail.src != 0) {
                    tail.src += hi - prev_lo;
                }
                prov[hi] = tail;
            }
        }
    }
    it = prov.lower_bound(lo);
    while (it != prov.end() && it->first < hi) {
        if (it->second.hi > hi) {
            Prov rest = it->second;
            if (rest.src != 0) {
                rest.src += hi - it->first;
            }
            it = prov.erase(it);
            prov[hi] = rest;
            break;
        }
        it = prov.erase(it);
    }
    Prov p = p0;
    p.hi = hi;
    prov[lo] = p;
}

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

u64 UploadHash(u64 address, u64 size) {
    if (!Armed() || size == 0 || size > (256u << 10)) {
        return 0;
    }
    return RangeHash(address, size, nullptr);
}

void NoteImageUpload(u64 address, u64 size, u64 hash_before, u64 hash_after, bool source_gpu, const std::string& gpu_page_map) {
    if (hash_before == 0 && hash_after == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Upload& u = uploads[address];
    u.source_gpu = source_gpu;
    u.page_map = gpu_page_map;
    u.size = size;
    u.before = hash_before;
    u.after = hash_after;
    u.frame = static_cast<u32>(Core::DebugServer::PresentCount());
    ++u.count;
}

void NoteState(const std::string& text) {
    if (!t_active) {
        return;
    }
    ++t_draw.reads_total;
    if (t_draw.reads.size() >= ReadsPerDraw) {
        return;
    }
    Read r;
    r.kind = "state: " + text;
    r.hash = Fnv(reinterpret_cast<const u8*>(text.data()), text.size());
    r.size = text.size();
    t_draw.reads.push_back(std::move(r));
}

void NoteImage(u64 address, u64 size, const std::string& description0) {
    std::string description = description0;
    if (t_active && address != 0 && size != 0 && size <= (256u << 10)) {
        const u64 now = RangeHash(address, size, nullptr);
        std::scoped_lock lock{mutex};
        if (const auto it = uploads.find(address); it != uploads.end()) {
            const Upload& u = it->second;
            description += fmt::format(" | uploaded {}x, last at frame {} from {} [{}]: memory before {:x} after {:x}, memory now {:x} => {}{}", u.count, u.frame, u.source_gpu ? "the GPU's data" : "GUEST MEMORY", u.page_map, u.before, u.after, now,
                                       u.after == now ? "image matches memory" : "MEMORY CHANGED SINCE THE UPLOAD", u.before != u.after ? " (memory changed DURING the upload)" : "");
        } else {
            description += fmt::format(" | never uploaded since the trace started (memory now {:x})", now);
        }
    }
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
    if (std::strstr(w.by, "copy") == nullptr) {
        ProvSet(address, size, Prov{0, w.by, 0, w.frame, w.seq, w.caller, w.number});
    }
    if (Overlaps(watch, address, size) || Overlaps(src_watch, address, size)) {
        KeepWrite(w);
    } else {
        recent.push_back(w);
        while (recent.size() > RecentKept) {
            recent.pop_front();
        }
    }
}

bool WantsShot(u64 address, u32 frame) {
    if (!t_active) {
        return false;
    }
    std::scoped_lock lock{mutex};
    Shot& s = shots[address];
    if (s.last_asked != 0 && frame < s.last_asked + 20) {
        return false;
    }
    s.last_asked = frame == 0 ? 1 : frame;
    return true;
}

void SaveShot(u64 address, u32 frame, const u8* data, u64 size, const std::string& info) {
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "ui_shots";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string file = fmt::format("shot_{:x}.bin", address);
    {
        std::ofstream out(dir / file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    std::scoped_lock lock{mutex};
    Shot& s = shots[address];
    s.frame = frame;
    s.size = size;
    s.file = file;
    s.info = info;
}

void NoteCopyPath(const std::string& path) {
    std::scoped_lock lock{mutex};
    if (!copies.empty() && copies.back().number == last_copy_number) {
        copies.back().path = path;
    }
}

u64 harvest_pending = 0, harvest_oldest = 0, harvest_current = 0, harvest_side = 0;
bool harvest_side_prev_free = false;
bool harvest_oldest_free = false;
void NoteHarvest(u64 pending, u64 oldest_tick, bool oldest_free, u64 current_tick, u64 side_tick, bool side_prev_free) {
    std::scoped_lock lock{mutex};
    harvest_side = side_tick;
    harvest_side_prev_free = side_prev_free;
    harvest_pending = pending;
    harvest_oldest = oldest_tick;
    harvest_oldest_free = oldest_free;
    harvest_current = current_tick;
}

void NoteCopyQueued(u64 number) {
    std::scoped_lock lock{mutex};
    for (auto it = copies.rbegin(); it != copies.rend(); ++it) {
        if (it->number == number) {
            ++it->queued;
            return;
        }
    }
}

u64 LastCopyNumber() {
    std::scoped_lock lock{mutex};
    return last_copy_number;
}

void NoteCopyResult(u64 number, u32 nonzero_chunks, u64 hash, int which) {
    std::scoped_lock lock{mutex};
    CopyResult& r = copy_results[number];
    if (which == 1) {
        r.src_nonzero_chunks = nonzero_chunks;
        r.src_hash = hash;
    } else {
        r.nonzero_chunks = nonzero_chunks;
        r.hash = hash;
        if (r.src_nonzero_chunks == -1 && false) {
        }
    }
    // keep a result as long as its copy is kept (the copies list holds the newest CopiesKept copies): dropping by count lost the results of the older hops of a chain
    const u64 oldest_copy = copies.empty() ? 0 : copies.front().number;
    while (!copy_results.empty() && (copy_results.begin()->first < oldest_copy || copy_results.size() > 40000)) {
        copy_results.erase(copy_results.begin());
    }
}

void NoteSupersede(u64 address, u64 size, u64 return_address) {
    if (!Armed() || size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Prov p{0, "GPU mark dropped (superseded)", 0, static_cast<u32>(Core::DebugServer::PresentCount()), Core::NativeRegistry::CurrentDrawSeq(), return_address, ++write_number};
    ProvSet(address, size, p);
    Write w;
    w.by = p.by;
    w.address = address;
    w.size = size;
    w.seq = p.seq;
    w.caller = return_address;
    w.frame = p.frame;
    w.number = p.number;
    if (Overlaps(watch, address, size) || Overlaps(src_watch, address, size)) {
        KeepWrite(w);
    }
}

void NoteGpuCopy(u64 dst, u64 src, u64 size) {
    if (!Armed() || size < 4096) {
        return;
    }
    std::scoped_lock lock{mutex};
    std::vector<std::array<u64, 5>> srcw;
    if (size >= 16384 && size <= (256u << 10)) {
        auto it = prov.upper_bound(src);
        if (it != prov.begin()) {
            --it;
        }
        for (; it != prov.end() && it->first < src + size && srcw.size() < 12; ++it) {
            if (it->second.hi > src) {
                const u64 a = std::max(it->first, src);
                const u64 b = std::min(it->second.hi, src + size);
                srcw.push_back({it->second.number, it->second.frame, std::string_view(it->second.by) == "copy" ? 1u : 0u, a, b - a});
            }
        }
    }
    ProvSet(dst, size, Prov{0, "copy", src, static_cast<u32>(Core::DebugServer::PresentCount()), Core::NativeRegistry::CurrentDrawSeq(),
                            Core::NativeRegistry::CurrentDrawSeq() != 0 ? Core::NativeDraw::RecordCallerGhidra(Core::NativeRegistry::CurrentDrawSeq()) : 0, write_number + 1});
    if (size < 16384) {
        ++write_number;
        return;
    }
    Copy c;
    c.dst = dst;
    c.src = src;
    c.size = size;
    c.seq = Core::NativeRegistry::CurrentDrawSeq();
    c.caller = c.seq != 0 ? Core::NativeDraw::RecordCallerGhidra(c.seq) : 0;
    c.frame = static_cast<u32>(Core::DebugServer::PresentCount());
    c.number = ++write_number;
    c.srcw = std::move(srcw);
    if (size <= (256u << 10)) {
        c.src_hash = RangeHash(src, size, &c.src_nonzero_chunks);
        if (const auto it = uploads.find(src); it != uploads.end() && it->second.size == size) {
            c.src_upload_hash = it->second.after;
            c.src_upload_frame = it->second.frame;
        }
    } else {
        return; // big copies (streaming, 33 MB per frame) are not text pictures and would push them out of the list
    }
    copies.push_back(c);
    last_copy_number = c.number;
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
    out += fmt::format("],\"harvest\":{{\"pending\":{},\"oldest_tick\":{},\"oldest_free\":{},\"current_tick\":{},\"side_tick\":{},\"side_prev_free\":{}}},\"gpu_writes_total\":{},\"gpu_copies\":[", harvest_pending, harvest_oldest, harvest_oldest_free, harvest_current, harvest_side, harvest_side_prev_free, write_number);
    for (size_t i = 0; i < copies.size(); ++i) {
        const Copy& c = copies[i];
        out += fmt::format("{}{{\"n\":{},\"frame\":{},\"dst\":\"{:x}\",\"src\":\"{:x}\",\"size\":{},\"seq\":{},\"caller\":\"{:x}\",\"src_hash_then\":\"{:x}\",\"src_hash_now\":\"{:x}\",\"src_nonzero_chunks\":{},\"result_nz\":{},\"result_hash\":\"{:x}\",\"src_img_nz\":{},\"src_img_hash\":\"{:x}\",\"src_upload_hash\":\"{:x}\",\"src_upload_frame\":{},\"queued\":{},\"srcw\":{},\"path\":\"{}\"}}", i ? "," : "", c.number, c.frame, c.dst, c.src, c.size, c.seq, c.caller, c.src_hash, RangeHash(c.src, c.size, nullptr), c.src_nonzero_chunks,
                           copy_results.contains(c.number) ? static_cast<s64>(copy_results[c.number].nonzero_chunks) : -1, copy_results.contains(c.number) ? copy_results[c.number].hash : 0ull,
                           copy_results.contains(c.number) ? copy_results[c.number].src_nonzero_chunks : -1, copy_results.contains(c.number) ? copy_results[c.number].src_hash : 0ull, c.src_upload_hash, c.src_upload_frame, c.queued, SrcwJson(c), c.path);
    }
    out += "],\"gpu_writes\":[";
    std::vector<Write> sorted(writes.begin(), writes.end());
    std::ranges::sort(sorted, {}, &Write::number);
    for (size_t i = 0; i < sorted.size(); ++i) {
        const Write& w = sorted[i];
        out += fmt::format("{}{{\"n\":{},\"frame\":{},\"by\":\"{}\",\"address\":\"{:x}\",\"size\":{},\"seq\":{},\"caller\":\"{:x}\"}}", i ? "," : "", w.number, w.frame, w.by, w.address, w.size, w.seq, w.caller);
    }
    out += "],\"shots\":[";
    {
        bool first_shot = true;
        for (const auto& [address, sh] : shots) {
            if (sh.file.empty()) {
                continue;
            }
            out += fmt::format("{}{{\"address\":\"{:x}\",\"frame\":{},\"size\":{},\"file\":\"ui_shots/{}\",\"info\":\"{}\"}}", first_shot ? "" : ",", address, sh.frame, sh.size, sh.file, sh.info);
            first_shot = false;
        }
    }
    out += "],\"provenance\":[";
    if (!frames.empty()) {
        bool first = true;
        for (size_t d = 0; d < frames.back().draws.size(); ++d) {
            for (const Read& r : frames.back().draws[d].reads) {
                if (r.kind.rfind("image", 0) != 0) {
                    continue;
                }
                out += fmt::format("{}{{\"draw\":{},\"image\":\"{:x}\",\"size\":{},\"chain\":[", first ? "" : ",", d, r.address, r.size);
                first = false;
                u64 lo = r.address, size = r.size;
                bool firstc = true;
                for (int level = 0; level < 40 && size != 0; ++level) {
                    u64 next_src = 0;
                    auto it = prov.upper_bound(lo);
                    if (it != prov.begin()) {
                        --it;
                    }
                    for (; it != prov.end() && it->first < lo + size; ++it) {
                        if (it->second.hi <= lo) {
                            continue;
                        }
                        const u64 a = std::max(it->first, lo);
                        const u64 b = std::min(it->second.hi, lo + size);
                        out += fmt::format("{}{{\"level\":{},\"range\":\"{:x}+{}\",\"by\":\"{}\",\"src\":\"{:x}\",\"frame\":{},\"seq\":{},\"caller\":\"{:x}\",\"n\":{}}}", firstc ? "" : ",", level, a, b - a, it->second.by,
                                           it->second.src != 0 ? it->second.src + (a - it->first) : 0, it->second.frame, it->second.seq, it->second.caller, it->second.number);
                        firstc = false;
                        if (it->second.src != 0 && next_src == 0) {
                            next_src = it->second.src + (a - it->first);
                            size = b - a;
                        }
                    }
                    if (next_src == 0) {
                        break;
                    }
                    lo = next_src;
                }
                out += "]}";
            }
        }
    }
    out += "],\"frames\":[";
    for (size_t f = 0; f < frames.size(); ++f) {
        out += (f ? "," : "") + FrameJson(frames[f]);
    }
    out += "]}";
    return out;
}

} // namespace Core::UiTrace
