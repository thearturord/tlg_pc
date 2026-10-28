// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <xxhash.h>

#include "common/logging/log.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/native_write_watch.h"
#include "video_core/untracked_ring.h"

extern "C" {
int runtime_memory_ww_enabled(void);
size_t runtime_memory_ww_regions(uintptr_t* starts, uint64_t* sizes, size_t capacity);
size_t runtime_memory_ww_harvest(uintptr_t address, uint64_t size, uintptr_t* pages, size_t capacity, int take_pending);
int runtime_memory_ww_pending(void);
void runtime_memory_ww_stats(uint64_t out[4]);
}

namespace Core::WriteWatch {

std::atomic<u32> epoch{1};
std::atomic<u32> base_epoch{1};

namespace {
thread_local u32 t_known_epoch = 0;
std::atomic<u64> known_skips{0};
} // namespace

void SetKnownEpoch(u32 known) noexcept {
    t_known_epoch = known;
}

namespace {
constexpr u64 ChunkBits = 16;
constexpr u64 PageSize = 4096;

/// a harvest covers whole blocks around the request (one kernel call costs about the same for 64 KiB as for 1 MiB, and the
/// neighbours of a requested chunk are usually requested next)
u64 BlockChunks() {
    static const u64 block_chunks = [] {
        const char* e = std::getenv("TLG_WRITE_WATCH_BLOCK_KB");
        const u64 kb = e != nullptr ? std::strtoull(e, nullptr, 10) : 1024;
        return std::max<u64>(1, (kb * 1024) >> ChunkBits);
    }();
    return block_chunks;
}

struct Region {
    VAddr start{};
    u64 size{};
    std::unique_ptr<std::atomic<u32>[]> harvested; ///< per 64 KiB chunk: the epoch read before its last harvest (0: never)
    std::unique_ptr<std::atomic<u32>[]> harvested_base; ///< per 64 KiB chunk: the base epoch read before its last harvest (0: never)
    std::unique_ptr<std::atomic<u32>[]> requested; ///< per block: the epoch of its last request (the background harvest keeps those fresh)
};

struct State {
    std::mutex mutex; ///< one harvest at a time (a harvest's pages are replayed before another caller may skip the chunk)
    static constexpr u32 MaxRegions = 32;
    Region regions[MaxRegions]; ///< only appended (read without the lock up to region_count)
    std::atomic<u32> region_count{0};
    size_t runtime_count{~size_t{0}};
    std::vector<uintptr_t> pages;
    std::atomic<u32> refreshed_at{0};
    Replay replay{nullptr}; ///< the buffer cache's marking (the background harvest uses it)
    void* context{nullptr};
    std::vector<uintptr_t> background_pages;
    std::atomic<u64> background_ns{0}, background_calls{0}, background_pages_marked{0};
    std::atomic<u64> ns_runtime{0}, ns_replay{0}, window_bumps{0}, calls{0}, skipped{0}, chunks{0}, runtime_calls{0}, replayed{0}, pending_replayed{0}, ns{0};
};

State& Get() {
    static State state;
    return state;
}

bool CheckEnabled() {
    static const bool on = runtime_memory_ww_enabled() != 0;
    return on;
}

void RefreshRegions(State& s) {
    uintptr_t starts[64];
    uint64_t sizes[64];
    const size_t n = runtime_memory_ww_regions(starts, sizes, 64);
    if (n == s.runtime_count) {
        return;
    }
    // new regions are appended; a region that left keeps its entry (its harvest finds nothing: the runtime skips unwatched memory)
    for (size_t i = 0; i < n && i < 64; ++i) {
        const u32 count = s.region_count.load(std::memory_order_relaxed);
        bool known = false;
        for (u32 k = 0; k < count; ++k) {
            known |= s.regions[k].start == starts[i] && s.regions[k].size == sizes[i];
        }
        if (known || count == State::MaxRegions) {
            continue;
        }
        Region& r = s.regions[count];
        r.start = starts[i];
        r.size = sizes[i];
        const u64 chunks = (r.size + (u64{1} << ChunkBits) - 1) >> ChunkBits;
        r.harvested = std::make_unique<std::atomic<u32>[]>(chunks);
        r.harvested_base = std::make_unique<std::atomic<u32>[]>(chunks);
        for (u64 c = 0; c < chunks; ++c) {
            r.harvested[c].store(0, std::memory_order_relaxed);
            r.harvested_base[c].store(0, std::memory_order_relaxed);
        }
        const u64 blocks = chunks / BlockChunks() + 1;
        r.requested = std::make_unique<std::atomic<u32>[]>(blocks);
        for (u64 k = 0; k < blocks; ++k) {
            r.requested[k].store(0, std::memory_order_relaxed);
        }
        s.region_count.store(count + 1, std::memory_order_release);
    }
    s.runtime_count = n;
}
} // namespace

bool Enabled() noexcept {
    return CheckEnabled();
}

bool LiftProtection() noexcept {
    static const bool lift = [] {
        if (!CheckEnabled()) {
            return false;
        }
        if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
            LOG_ERROR(Render, "write watch: readbacks are on, the page protection stays (a write fault must download first)");
            return false;
        }
        return true;
    }();
    return lift;
}

namespace {
u32 VerifyInterval() {
    static const u32 n = [] {
        const char* e = std::getenv("TLG_WRITE_WATCH_VERIFY");
        return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    return n;
}
struct VerifyState {
    std::mutex mutex;
    std::unordered_map<u64, u64> hashes; ///< page number -> hash of the bytes last uploaded
    std::unordered_map<u64, std::unique_ptr<u8[]>> copies; ///< page number -> the bytes last uploaded (diagnosis of a miss)
    std::atomic<u64> pages_checked{0}, misses{0}, late_writes{0}, requests{0};
};
VerifyState& Verify() {
    static VerifyState v;
    return v;
}
} // namespace

bool VerifyOn() noexcept {
    return CheckEnabled() && VerifyInterval() != 0;
}

void NoteUpload(VAddr address, const u8* data, u64 size) {
    if (!VerifyOn()) {
        return;
    }
    auto& v = Verify();
    std::scoped_lock lock{v.mutex};
    for (u64 off = 0; off + PageSize <= size; off += PageSize) {
        if (((address + off) & (PageSize - 1)) == 0) {
            v.hashes[(address + off) / PageSize] = XXH3_64bits(data + off, PageSize);
            auto& copy = v.copies[(address + off) / PageSize];
            if (!copy) {
                copy = std::make_unique<u8[]>(PageSize);
            }
            std::memcpy(copy.get(), data + off, PageSize);
        }
    }
}

void NoteGpuSideWrite(VAddr address, u64 size) {
    if (!VerifyOn() || size == 0) {
        return;
    }
    auto& v = Verify();
    std::scoped_lock lock{v.mutex};
    for (u64 page = address / PageSize, end = (address + size + PageSize - 1) / PageSize; page < end; ++page) {
        v.hashes.erase(page);
        v.copies.erase(page);
    }
}

bool GpuSideWrite(VAddr address, u64 size, const std::function<bool()>& write) {
    auto& v = Verify();
    std::scoped_lock lock{v.mutex};
    const bool result = write();
    for (u64 page = address / PageSize, end = (address + size + PageSize - 1) / PageSize; page < end; ++page) {
        v.hashes.erase(page);
        v.copies.erase(page);
    }
    return result;
}

namespace {
/// verify mode, under the harvest lock: pages of the range that are not CPU modified must hold what was uploaded
void VerifyRange(State& s, VAddr address, u64 size, Replay replay, IsCpuDirty dirty, IsCpuDirty gpu, void* context) {
    auto& v = Verify();
    if (v.requests.fetch_add(1, std::memory_order_relaxed) % VerifyInterval() != 0) {
        return;
    }
    const VAddr end = address + size;
    for (u32 i = 0, count = s.region_count.load(std::memory_order_acquire); i < count; ++i) {
        const Region& r = s.regions[i];
        const VAddr a = std::max(address, r.start) & ~(PageSize - 1);
        const VAddr b = std::min(end, r.start + r.size);
        const auto& ring = VideoCore::UntrackedRing::Get();
        for (VAddr page = a; page < b; page += PageSize) {
            if (ring.Active()) {
                // the untracked upload ring is uploaded on every use, its CPU state means nothing
                const auto [lo, hi] = ring.Clip(page, PageSize);
                if (lo < hi) {
                    continue;
                }
            }
            // the lock also keeps a GPU side write (GpuSideWrite) out of the check
            std::scoped_lock lock{v.mutex};
            const auto it = v.hashes.find(page / PageSize);
            if (it == v.hashes.end()) {
                continue;
            }
            const u64 expected = it->second;
            if (dirty(context, page, PageSize)) {
                continue;
            }
            v.pages_checked.fetch_add(1, std::memory_order_relaxed);
            if (XXH3_64bits(reinterpret_cast<const void*>(page), PageSize) == expected) {
                continue;
            }
            // changed since the upload and not marked: a write after the harvest has its bit set by now, or sits in the pending list;
            // neither is a miss
            if (s.pages.size() < 4096) {
                s.pages.resize(4096);
            }
            const size_t n = runtime_memory_ww_harvest(page, PageSize, s.pages.data(), s.pages.size(), 1);
            bool found = false;
            for (size_t k = 0; k < n; ++k) {
                found |= s.pages[k] == page;
                replay(context, s.pages[k], PageSize);
            }
            replay(context, page, PageSize);
            if (found) {
                v.late_writes.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            v.misses.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1) < 24) {
                if (const auto c = v.copies.find(page / PageSize); c != v.copies.end()) {
                const u8* old_bytes = c->second.get();
                const u8* new_bytes = reinterpret_cast<const u8*>(page);
                u32 differing = 0, first = PageSize, last = 0;
                for (u32 k = 0; k < PageSize; ++k) {
                    if (old_bytes[k] != new_bytes[k]) {
                        ++differing;
                        first = std::min(first, k);
                        last = k;
                    }
                }
                u64 old_word = 0, new_word = 0;
                if (first < PageSize) {
                    std::memcpy(&old_word, old_bytes + (first & ~7u), 8);
                    std::memcpy(&new_word, new_bytes + (first & ~7u), 8);
                }
                LOG_ERROR(Render, "write watch MISS: page {:#x} changed since its upload without a write watch record: {} bytes differ in [{:#x}, {:#x}], word at {:#x}: {:#018x} -> {:#018x}, gpu modified {}",
                          page, differing, first, last, first & ~7u, old_word, new_word, gpu(context, page, PageSize));
            }
            }
        }
    }
}

bool BackgroundOn() {
    static const bool on = [] {
        // off by default: measured 2026-10-28, a guest submission or a passed wait moves the epoch right before the requests that
        // follow it, so the requests still harvest about as much themselves and the thread's work comes on top (TLG_WRITE_WATCH_THREAD=1)
        const char* e = std::getenv("TLG_WRITE_WATCH_THREAD");
        return e != nullptr && *e == '1';
    }();
    return on;
}

/// The background harvest: once per time window it moves the epoch on (a write is seen at most that long after it happened) and
/// harvests every block requested in the last epochs, so the requests mostly find their chunks current and the kernel calls run on
/// this thread instead of the command processor and raster threads. A request that finds a stale chunk still harvests it itself: the
/// rule of the epoch does not change.
void BackgroundLoop() {
    State& s = Get();
    const s32 HotEpochs = [] {
        const char* e = std::getenv("TLG_WRITE_WATCH_HOT_EPOCHS");
        return e != nullptr ? static_cast<s32>(std::strtol(e, nullptr, 10)) : 4;
    }();
    const u64 window_us = [] {
        const char* e = std::getenv("TLG_WRITE_WATCH_WINDOW_US");
        return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    std::vector<uintptr_t> pages;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::microseconds(window_us != 0 ? window_us : 1000));
        if (window_us != 0) {
            epoch.fetch_add(1, std::memory_order_acq_rel);
            s.window_bumps.fetch_add(1, std::memory_order_relaxed);
        }
        const u32 now_base = base_epoch.load(std::memory_order_acquire);
        const u32 now = epoch.load(std::memory_order_acquire);
        const auto started = std::chrono::steady_clock::now();
        const u32 count = s.region_count.load(std::memory_order_acquire);
        const u64 block_chunks = BlockChunks();
        const auto hot = [now, HotEpochs](u32 last) { return last != 0 && static_cast<s32>(now - last) <= HotEpochs; };
        for (u32 i = 0; i < count; ++i) {
            Region& r = s.regions[i];
            const u64 chunks = ((r.size - 1) >> ChunkBits) + 1;
            const u64 blocks = (chunks + block_chunks - 1) / block_chunks;
            u64 k = 0;
            while (k < blocks) {
                if (!hot(r.requested[k].load(std::memory_order_relaxed))) {
                    ++k;
                    continue;
                }
                // a run of hot blocks, one kernel call
                u64 k_end = k;
                while (k_end + 1 < blocks && hot(r.requested[k_end + 1].load(std::memory_order_relaxed))) {
                    ++k_end;
                }
                const u64 c = k * block_chunks;
                const u64 run_end = std::min(chunks - 1, (k_end + 1) * block_chunks - 1);
                const VAddr ra = r.start + (c << ChunkBits);
                const VAddr rb = std::min(r.start + ((run_end + 1) << ChunkBits), r.start + r.size);
                const size_t capacity = (rb - ra) / PageSize + 1;
                if (pages.size() < capacity) {
                    pages.resize(capacity);
                }
                {
                    std::scoped_lock lock{s.mutex};
                    if (s.replay != nullptr) {
                        const size_t n = runtime_memory_ww_harvest(ra, rb - ra, pages.data(), capacity, 0);
                        size_t p = 0;
                        while (p < n) {
                            size_t q = p + 1;
                            while (q < n && pages[q] == pages[q - 1] + PageSize) {
                                ++q;
                            }
                            s.replay(s.context, pages[p], (q - p) * PageSize);
                            p = q;
                        }
                        for (u64 ch = c; ch <= run_end; ++ch) {
                            r.harvested[ch].store(now, std::memory_order_release);
                            r.harvested_base[ch].store(now_base, std::memory_order_release);
                        }
                        s.background_calls.fetch_add(1, std::memory_order_relaxed);
                        s.background_pages_marked.fetch_add(n, std::memory_order_relaxed);
                    }
                }
                k = k_end + 1;
            }
        }
        s.background_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()),
                                  std::memory_order_relaxed);
    }
}

void StartBackground() {
    if (!BackgroundOn()) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            Common::SetCurrentThreadName("tlg:WriteWatch");
            BackgroundLoop();
        }).detach();
    });
}
} // namespace

bool LiftImages() noexcept {
    static const bool lift = [] {
        return LiftProtection();
    }();
    return lift;
}

namespace {
std::atomic<u64> image_checks{0}, image_misses{0}, image_samples{0};
} // namespace

bool SampleImageCheck() noexcept {
    return VerifyOn() && image_samples.fetch_add(1, std::memory_order_relaxed) % VerifyInterval() == 0;
}

void NoteImageCheck(bool changed_without_a_record, VAddr address, u64 size) {
    image_checks.fetch_add(1, std::memory_order_relaxed);
    if (changed_without_a_record) {
        image_misses.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 24) {
            LOG_ERROR(Render, "write watch image MISS: image {:#x}+{:#x} changed since its upload and is not marked dirty", address, size);
        }
    }
}

void Harvest(VAddr address, u64 size, Replay replay, IsCpuDirty dirty, IsCpuDirty gpu, void* context, bool image) {
    if (!CheckEnabled() || size == 0) {
        return;
    }
    State& s = Get();
    s.calls.fetch_add(1, std::memory_order_relaxed);
    if (const u32 known = t_known_epoch; known != 0 && !image && runtime_memory_ww_pending() == 0) {
        // harvested ahead by the command processor when it planned this operation
        bool current = true;
        const VAddr end = address + size;
        const u32 count = s.region_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < count && current; ++i) {
            const Region& r = s.regions[i];
            const VAddr a = std::max(address, r.start);
            const VAddr b = std::min(end, r.start + r.size);
            for (u64 c = a < b ? (a - r.start) >> ChunkBits : 1, c1 = a < b ? (b - 1 - r.start) >> ChunkBits : 0; c <= c1; ++c) {
                if (static_cast<s32>(r.harvested[c].load(std::memory_order_acquire) - known) < 0) {
                    current = false;
                    break;
                }
            }
        }
        if (current && VerifyInterval() == 0) {
            known_skips.fetch_add(1, std::memory_order_relaxed);
            s.skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    static const bool every = [] {
        const char* e = std::getenv("TLG_WRITE_WATCH_EPOCH");
        return e != nullptr && *e == '0';
    }();
    {
        // a write is seen at most this long after it happened even inside one epoch (the fault saw it at once)
        static const u64 window_ns = [] {
            const char* e = std::getenv("TLG_WRITE_WATCH_WINDOW_US");
            return (e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull) * 1000ull; // P2: off by default (2026-10-28), only submissions and passed waits
        }();
        static std::atomic<u64> last_bump{0};
        const u64 t = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count());
        u64 last = last_bump.load(std::memory_order_relaxed);
        if (!image && !BackgroundOn() && window_ns != 0 && t - last > window_ns && last_bump.compare_exchange_strong(last, t, std::memory_order_relaxed)) {
            epoch.fetch_add(1, std::memory_order_acq_rel);
            s.window_bumps.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const u32 now_base = base_epoch.load(std::memory_order_acquire);
    const u32 now = epoch.load(std::memory_order_acquire);
    const bool pending = runtime_memory_ww_pending() != 0;
    const VAddr end = address + size;
    // a chunk harvested at this epoch or later (its pages replayed before the epoch was stored) needs nothing; an image looks at the
    // base epoch only
    const auto current_at = [image, now, now_base](const Region& r, u64 c) {
        return image ? static_cast<s32>(r.harvested_base[c].load(std::memory_order_acquire) - now_base) >= 0
                     : static_cast<s32>(r.harvested[c].load(std::memory_order_acquire) - now) >= 0;
    };
    if (!every && !pending) {
        bool stale = false;
        const u32 count = s.region_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < count && !stale; ++i) {
            const Region& r = s.regions[i];
            const VAddr a = std::max(address, r.start);
            const VAddr b = std::min(end, r.start + r.size);
            for (u64 c = a < b ? (a - r.start) >> ChunkBits : 1, c1 = a < b ? (b - 1 - r.start) >> ChunkBits : 0; c <= c1; ++c) {
                if (!current_at(r, c)) {
                    stale = true;
                    break;
                }
            }
        }
        if (!stale && s.refreshed_at.load(std::memory_order_relaxed) == now_base) {
            s.skipped.fetch_add(1, std::memory_order_relaxed);
            if (VerifyInterval() == 0) {
                return;
            }
        }
    }
    std::scoped_lock lock{s.mutex};
    s.replay = replay;
    s.context = context;
    StartBackground();
    const auto started = std::chrono::steady_clock::now();
    if (s.refreshed_at.load(std::memory_order_relaxed) != now_base) {
        // mappings change before the submissions that use them
        RefreshRegions(s);
        s.refreshed_at.store(now_base, std::memory_order_relaxed);
    }
    // consecutive pages are marked as one range
    const auto replay_pages = [&](size_t n) {
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;
            while (j < n && s.pages[j] == s.pages[j - 1] + PageSize) {
                ++j;
            }
            replay(context, s.pages[i], (j - i) * PageSize);
            i = j;
        }
    };
    bool any = false;
    const u32 count = s.region_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        Region& r = s.regions[i];
        const VAddr a = std::max(address, r.start);
        const VAddr b = std::min(end, r.start + r.size);
        if (a >= b) {
            continue;
        }
        const u64 block_chunks = BlockChunks();
        const u64 last_chunk = (r.size - 1) >> ChunkBits;
        const u64 c0 = ((a - r.start) >> ChunkBits) / block_chunks * block_chunks;
        const u64 c1 = std::min(last_chunk, (((b - 1 - r.start) >> ChunkBits) / block_chunks + 1) * block_chunks - 1);
        for (u64 k = c0 / block_chunks; k <= c1 / block_chunks; ++k) {
            r.requested[k].store(now, std::memory_order_relaxed);
        }
        // runs of chunks not harvested at this epoch, one runtime call per run
        u64 c = c0;
        while (c <= c1) {
            if (!every && current_at(r, c)) {
                ++c;
                continue;
            }
            u64 run_end = c;
            while (run_end + 1 <= c1 && (every || !current_at(r, run_end + 1))) {
                ++run_end;
            }
            const VAddr ra = r.start + (c << ChunkBits);
            const VAddr rb = std::min(r.start + ((run_end + 1) << ChunkBits), r.start + r.size);
            const size_t capacity = (rb - ra) / PageSize + 1;
            if (s.pages.size() < capacity) {
                s.pages.resize(capacity);
            }
            const auto t0 = std::chrono::steady_clock::now();
            const size_t n = runtime_memory_ww_harvest(ra, rb - ra, s.pages.data(), capacity, 0);
            const auto t1 = std::chrono::steady_clock::now();
            s.runtime_calls.fetch_add(1, std::memory_order_relaxed);
            replay_pages(n);
            s.ns_runtime.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()), std::memory_order_relaxed);
            s.ns_replay.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t1).count()), std::memory_order_relaxed);
            s.replayed.fetch_add(n, std::memory_order_relaxed);
            for (u64 k = c; k <= run_end; ++k) {
                r.harvested[k].store(now, std::memory_order_release);
                r.harvested_base[k].store(now_base, std::memory_order_release);
            }
            s.chunks.fetch_add(run_end - c + 1, std::memory_order_relaxed);
            any = true;
            c = run_end + 1;
        }
    }
    if (pending) {
        // writes the runtime found outside a harvest (a GPU side write that raced with the CPU, a released region)
        if (s.pages.size() < 4096) {
            s.pages.resize(4096);
        }
        for (;;) {
            const size_t n = runtime_memory_ww_harvest(0, 0, s.pages.data(), s.pages.size(), 1);
            std::sort(s.pages.begin(), s.pages.begin() + n);
            replay_pages(n);
            s.pending_replayed.fetch_add(n, std::memory_order_relaxed);
            if (n < s.pages.size()) {
                break;
            }
        }
        any = true;
    }
    if (VerifyInterval() != 0) {
        VerifyRange(s, address, size, replay, dirty, gpu, context);
    }
    if (!any) {
        s.skipped.fetch_add(1, std::memory_order_relaxed);
    } else {
        s.ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()),
                       std::memory_order_relaxed);
    }
}

bool Watched(VAddr address, u64 size) {
    if (!CheckEnabled()) {
        return false;
    }
    uintptr_t starts[64];
    uint64_t sizes[64];
    const size_t n = runtime_memory_ww_regions(starts, sizes, 64);
    for (size_t i = 0; i < n && i < 64; ++i) {
        if (address >= starts[i] && address + size <= starts[i] + sizes[i]) {
            return true;
        }
    }
    return false;
}

nlohmann::json Report() {
    State& s = Get();
    uint64_t runtime[4] = {};
    runtime_memory_ww_stats(runtime);
    return {{"enabled", CheckEnabled()},
            {"protection_lifted", LiftProtection()},
            {"watched_bytes", runtime[0]},
            {"epoch", epoch.load(std::memory_order_relaxed)},
            {"epoch_bumps_by_the_time_window", s.window_bumps.load(std::memory_order_relaxed)},
            {"harvest_requests", s.calls.load(std::memory_order_relaxed)},
            {"requests_harvested_ahead_by_the_command_processor", known_skips.load(std::memory_order_relaxed)},
            {"requests_with_nothing_to_harvest", s.skipped.load(std::memory_order_relaxed)},
            {"chunks_harvested", s.chunks.load(std::memory_order_relaxed)},
            {"runtime_calls", s.runtime_calls.load(std::memory_order_relaxed)},
            {"written_pages_marked", s.replayed.load(std::memory_order_relaxed)},
            {"pending_pages_marked", s.pending_replayed.load(std::memory_order_relaxed)},
            {"harvest_ms", s.ns.load(std::memory_order_relaxed) / 1'000'000},
            {"harvest_ms_in_the_kernel_call", s.ns_runtime.load(std::memory_order_relaxed) / 1'000'000},
            {"harvest_ms_marking_pages", s.ns_replay.load(std::memory_order_relaxed) / 1'000'000},
            {"background", {{"on", BackgroundOn()}, {"ms", s.background_ns.load(std::memory_order_relaxed) / 1'000'000},
                            {"kernel_calls", s.background_calls.load(std::memory_order_relaxed)},
                            {"written_pages_marked", s.background_pages_marked.load(std::memory_order_relaxed)}}},
            {"gpu_side_writes_pages", runtime[1]},
            {"gpu_side_writes_raced_by_the_cpu", runtime[2]},
            {"regions_released", runtime[3]},
            {"images_unprotected", LiftImages()},
            {"image_verify", {{"images_checked", image_checks.load(std::memory_order_relaxed)},
                              {"MISSES_changed_without_being_marked", image_misses.load(std::memory_order_relaxed)}}},
            {"verify", {{"interval", VerifyInterval()},
                        {"pages_checked", Verify().pages_checked.load(std::memory_order_relaxed)},
                        {"MISSES_changed_without_a_record", Verify().misses.load(std::memory_order_relaxed)},
                        {"writes_after_the_harvest", Verify().late_writes.load(std::memory_order_relaxed)}}},
            {"note", "L3 write watch, cumulative since start: CPU writes found by the kernel's write watch instead of page faults"}};
}

} // namespace Core::WriteWatch
