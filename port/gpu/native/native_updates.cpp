// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_row_version.h"
#include <set>
#include "native/native_switch.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
#include <nlohmann/json.hpp>
#include "native/native_hooks.h"
#include "native/native_n0_census.h"
#include "native/native_ring_mirror.h"
#include "native/native_draw.h"
#include "native/native_registry.h"
#include "native/native_resources.h"
#include "native/native_uploads.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_image_caller.h"
#include "native/native_content.h"
#include "native/native_prof.h"
#include "native/native_split_census.h"
#include "native/native_ring_census.h"
#include "native/native_hook_ring.h"

namespace Core::NativeUpdates {

std::atomic<bool> enabled{[] {
    const char* e = std::getenv("TLG_UPDATE_AUDIT");
    const char* u = std::getenv("TLG_NATIVE_UPLOADS");
    const char* s = std::getenv("TLG_SNAPSHOTS");
    const char* r = std::getenv("TLG_RING_SUPERSEDE");
    const char* res = std::getenv("TLG_RESIDENT_BUFFERS");
    // the announcements are also what the native uploads and the snapshots are made of
    return (e != nullptr && *e != 0 && *e != '0') || (u != nullptr && *u != 0 && *u != '0') || (s != nullptr && *s != 0 && *s != '0') ||
           (r != nullptr && *r != 0 && *r != '0') || res == nullptr || *res != '0'; // resident buffers are on by default (L1), so are their announcements
}()};

namespace {

/// The thread of `RenderDev::BeginRecording` (the render thread): a map made on it may supersede GPU written marks.
std::atomic<u64> render_thread{0};

struct MapCounters {
    std::array<std::atomic<u64>, 8> by_type{};   ///< D3D11 map type 1..5 (0 and 6, 7 other)
    std::array<std::atomic<u64>, 4> by_usage{};  ///< 0 default 1 immutable 2 dynamic 3 staging
    std::atomic<u64> in_ring{}, own_memory{};
    std::atomic<u64> total{};
} maps;

u64 ThreadKey() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace

namespace {
struct OpenMap {
    u64 address{}, size{};
    u64 serial{}; ///< order in which Maps were made (FlushOpenMapsSince)
    u64 stamp{}; ///< the number the next record got when the Map was made (F2.2: a draw of an earlier record cannot depend on the write)
    bool in_ring{};
    u64 caller{}; ///< the guest return address of the Map call (TLG_WRITER_SITES names the write site after it)
    bool texture{}; ///< A1.0: a texture's Map (its window is image content), else a buffer's
    u32 map_type{};  ///< 1b shadow: the D3D11 map type (4 write discard, 5 no overwrite)
    u64 resource{};  ///< 1b shadow: the engine resource id
    u32 frame_flushed{}; ///< TLG_MAP_KEEP_OPEN: announced at a frame start and kept open for its real Unmap (it is dropped at the next frame start)
};
thread_local bool t_next_map_texture = false;
thread_local bool t_announce_texture = true; ///< A1.0: set around the announcement of a Map window: false for a buffer's
std::mutex open_map_mutex;
std::mutex site_mutex;
std::map<std::string, std::array<u64, 13>> map_diff; ///< caller/path -> count by (context frame - resource stamp), 12 = 12 or more (and wrap-around)
u64 open_map_serial = 0;
std::map<std::pair<u64, u64>, OpenMap> open_maps; ///< (thread, resource id) -> the window of the Map that has not been unmapped yet
// ring census (design review section 9): Maps whose window is never announced: a second Map of the same resource on the same thread before its Unmap replaces the
// first entry (its window is lost), and an Unmap that finds no open Map
std::atomic<u64> c_open_replaced{0}, c_open_replaced_bytes{0}, c_open_replaced_in_ring{0}, c_unmap_unmatched{0};
std::atomic<u64> c_replace_announced{0}, c_replace_announced_bytes{0};
// frame start flush of the open Maps (design review section 17): every thread's open Maps are announced at the render thread's frame start. A worker that is
// still between its Map and its Unmap then loses its entry: the window is announced half written and the real Unmap finds nothing.
std::atomic<u64> c_fs_flushed_own{0}, c_fs_flushed_other{0}, c_fs_kept{0}, c_fs_dropped_after_keep{0}, c_unmap_after_fs_flush{0}, c_unmap_kept_closed{0};
std::set<std::pair<u64, u64>> g_fs_recent; // (thread, resource id) flushed at the last frame start (under open_map_mutex)

/// `TLG_MAP_KEEP_OPEN` (default on since 2026-11-10 after the owner's play; =0 turns it off): an open Map of another thread is announced at the frame start as before, and also kept open, so that its real Unmap
/// (or a second Map of it) announces the finished window again; a Map still open at the next frame start is announced once more and dropped. Adds announcements only.
bool KeepOpenOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_MAP_KEEP_OPEN");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

/// `TLG_MAP_REPLACE_ANNOUNCE` (design review section 10; default on since 2026-11-10 after the owner's play, =0 turns it off): a Map of a resource that this thread mapped before and has not unmapped
/// announces the earlier window first, as its Unmap would have (the thread has moved on to a new Map of the same resource, so the earlier window's writes are done).
/// Before, the new entry replaced the old one and the old window was never announced. Adds announcements (and uploads), removes none.
bool ReplaceAnnounceOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_MAP_REPLACE_ANNOUNCE");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}
} // namespace

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread, u64 stamp = 0, bool ring = false);

namespace {
/// TLG_WRITER_SITES: sets the call site of the write a map announces: for a map into the buffer's own memory the caller (guest address in Ghidra numbering) so the late read census names it
void SetMapSite(const OpenMap& map, bool sites_on) {
    if (!sites_on) {
        return;
    }
    if (map.in_ring) {
        NativeContent::SetSite(NativeContent::SiteId("Map into a frame ring window"));
        return;
    }
    char name[64];
    std::snprintf(name, sizeof(name), "Map own @%llx", static_cast<unsigned long long>(map.caller - NativeDraw::ImageBase() + 0x1000000ull));
    NativeContent::SetSite(NativeContent::SiteId(name));
}
void TagMapFill(const OpenMap& map, u32 how) {
    Core::HookRing::SetFillTag(how, static_cast<u32>(map.caller - NativeDraw::ImageBase() + 0x1000000ull), map.resource, map.map_type);
}
bool MapSitesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_WRITER_SITES");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}
} // namespace

void NoteRenderThread() {
    render_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()), std::memory_order_relaxed);
    NativeDraw::FlushSceneSlotAnnouncements(); // C4: the last scene slots of the frame that just ended
    // a Map that was never unmapped (the engine keeps some pointers for the whole frame): its window is announced at the next frame start, for the thread that mapped it
    std::vector<std::pair<u64, OpenMap>> stale;
    {
        std::scoped_lock lock{open_map_mutex};
        const u64 me = render_thread.load(std::memory_order_relaxed);
        g_fs_recent.clear();
        for (auto it = open_maps.begin(); it != open_maps.end();) {
            stale.emplace_back(it->first.first, it->second);
            (it->first.first == me ? c_fs_flushed_own : c_fs_flushed_other).fetch_add(1, std::memory_order_relaxed);
            g_fs_recent.insert(it->first);
            if (KeepOpenOn() && it->first.first != me && it->second.frame_flushed == 0) {
                it->second.frame_flushed = 1; // announced below, kept for its real Unmap
                c_fs_kept.fetch_add(1, std::memory_order_relaxed);
                ++it;
            } else {
                if (it->second.frame_flushed != 0) {
                    c_fs_dropped_after_keep.fetch_add(1, std::memory_order_relaxed);
                }
                it = open_maps.erase(it);
            }
        }
    }
    for (const auto& [thread, map] : stale) {
        SetMapSite(map, MapSitesOn());
        t_announce_texture = map.texture;
        TagMapFill(map, 3);
        NoteWriterRangeOf(0, map.address, map.size, thread, map.stamp, map.in_ring);
        Core::HookRing::ClearFillTag();
        t_announce_texture = true;
        NativeContent::SetSite(0);
    }
    NativeUploads::PublishPending(NativeDraw::NextSeq());
}

namespace {
void (*supersede_fn)(u64, u64) = nullptr;
std::atomic<u64> superseded_ranges{}, superseded_bytes{};
} // namespace

void SetSupersede(void (*fn)(u64, u64)) {
    supersede_fn = fn;
}

bool SupersedeEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_SUPERSEDE");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

namespace {
/// The Unmap's work: the thread's open Map of the resource is closed and its window announced. False when there is none.
bool CloseOpenMap(u64 resource_id, u32 how) {
    OpenMap map;
    {
        std::scoped_lock lock{open_map_mutex};
        const auto it = open_maps.find({ThreadKey(), resource_id});
        if (it == open_maps.end()) {
            return false;
        }
        map = it->second;
        open_maps.erase(it);
    }
    NativeUploads::SnapshotTake(map.address, map.size, NativeDraw::NextSeq());
    char own_name[64];
    const char* site = "Map into a frame ring window";
    if (!map.in_ring) {
        site = "Map into the buffer's own memory";
        if (MapSitesOn()) {
            std::snprintf(own_name, sizeof(own_name), "Map own @%llx", static_cast<unsigned long long>(map.caller - NativeDraw::ImageBase() + 0x1000000ull));
            site = own_name;
        }
    }
    t_announce_texture = map.texture;
    TagMapFill(map, how);
    NoteWriterRange(0, map.address, map.size, site, map.in_ring);
    Core::HookRing::ClearFillTag();
    t_announce_texture = true;
    return true;
}
} // namespace

namespace {
/// Diagnostics (`TLG_WATCH_RESOURCES=0x181a,0x180b`): Maps and Unmaps of these engine buffers are logged, and the writer probe is armed on the window at the Unmap.
bool WatchedResource(u64 id) noexcept {
    static const std::vector<u64> ids = [] {
        std::vector<u64> v;
        const char* e = std::getenv("TLG_WATCH_RESOURCES");
        while (e != nullptr && *e != 0) {
            char* end = nullptr;
            const u64 x = std::strtoull(e, &end, 0);
            if (end == e) {
                break;
            }
            v.push_back(x);
            e = *end == ',' ? end + 1 : end;
        }
        return v;
    }();
    return !ids.empty() && std::find(ids.begin(), ids.end(), id) != ids.end();
}
std::atomic<u32> watch_logged{0};
} // namespace

void NoteUnmap(u64 resource_id) {
    bool kept = false;
    u64 watch_address = 0, watch_size = 0;
    {
        std::scoped_lock lock{open_map_mutex};
        const auto it = open_maps.find({ThreadKey(), resource_id});
        kept = it != open_maps.end() && it->second.frame_flushed != 0;
        if (it != open_maps.end()) {
            watch_address = it->second.address;
            watch_size = it->second.size;
        }
    }
    if (WatchedResource(resource_id)) [[unlikely]] {
        if (watch_logged.fetch_add(1, std::memory_order_relaxed) < 160) {
            std::fprintf(stderr, "watch res: Unmap %#llx window %#llx+%#llx seq %u thread %llu\n", static_cast<unsigned long long>(resource_id), static_cast<unsigned long long>(watch_address),
                         static_cast<unsigned long long>(watch_size), static_cast<u32>(NativeDraw::NextSeq()), static_cast<unsigned long long>(ThreadKey()));
        }
        // (no writer probe here: arming it at every Unmap of these buffers stopped the game in `wide4`)
    }
    if (CloseOpenMap(resource_id, 0)) {
        if (kept) {
            c_unmap_kept_closed.fetch_add(1, std::memory_order_relaxed); // the fix at work: the finished window is announced now
        }
        return;
    }
    c_unmap_unmatched.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{open_map_mutex};
    if (g_fs_recent.count({ThreadKey(), resource_id}) != 0) {
        c_unmap_after_fs_flush.fetch_add(1, std::memory_order_relaxed); // the race: the frame start announced and dropped this thread's open Map before its Unmap
    }
}

u64 OpenMapSerial() {
    std::scoped_lock lock{open_map_mutex};
    return open_map_serial;
}

void FlushOpenMapsSince(u64 serial) {
    const u64 thread = ThreadKey();
    std::vector<OpenMap> done;
    {
        std::scoped_lock lock{open_map_mutex};
        for (auto it = open_maps.begin(); it != open_maps.end();) {
            if (it->first.first == thread && it->second.serial > serial) {
                done.push_back(it->second);
                it = open_maps.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const OpenMap& map : done) {
        SetMapSite(map, MapSitesOn());
        t_announce_texture = map.texture;
        TagMapFill(map, 2);
        NoteWriterRangeOf(0, map.address, map.size, thread, map.stamp, map.in_ring);
        Core::HookRing::ClearFillTag();
        t_announce_texture = true;
        NativeContent::SetSite(0);
    }
}

bool OpenMapOverlaps(u64 address, u64 size) {
    std::scoped_lock lock{open_map_mutex};
    for (const auto& [key, map] : open_maps) {
        if (address < map.address + map.size && map.address < address + size) {
            return true;
        }
    }
    return false;
}

void MarkNextMapTexture() {
    t_next_map_texture = true;
}

void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size, u32 context_frame, u32 resource_stamp, u32 bind_flags) {
    const bool texture_map = t_next_map_texture;
    t_next_map_texture = false;
    maps.total.fetch_add(1, std::memory_order_relaxed);
    maps.by_type[map_type < 8 ? map_type : 7].fetch_add(1, std::memory_order_relaxed);
    maps.by_usage[usage < 4 ? usage : 3].fetch_add(1, std::memory_order_relaxed);
    const bool in_ring = address != memory;
    if (MapSitesOn() && map_type >= 2 && address != 0) {
        // the game's Map rule: ring memory when (frame - stamp) < 3, else in place. Per caller and path: how many frames since the buffer was last stamped
        const u32 diff = context_frame - resource_stamp;
        std::scoped_lock diff_lock{site_mutex};
        char name[72];
        std::snprintf(name, sizeof(name), "@%llx type%u %s bind%x usage%u", static_cast<unsigned long long>(caller - NativeDraw::ImageBase() + 0x1000000ull), map_type, in_ring ? "ring" : "inplace", bind_flags & 0xff, usage);
        ++map_diff[name][std::min<u32>(diff, 12u)];
    }
    (in_ring ? maps.in_ring : maps.own_memory).fetch_add(1, std::memory_order_relaxed);
    if (address == 0 || size == 0 || map_type == 1) {
        return;
    }
    // the written part: the requested size when the caller gave one, else everything the Map handed out; it is announced when the thread unmaps the resource
    const u64 written = requested_size != 0 && requested_size < size ? requested_size : size;
    if (WatchedResource(resource_id)) [[unlikely]] {
        if (watch_logged.fetch_add(1, std::memory_order_relaxed) < 160) {
            std::fprintf(stderr, "watch res: Map %#llx type %u returned %#llx+%#llx (buffer size %#x) own memory %#llx in ring %d frame %u stamp %u caller %#llx seq %u thread %llu\n",
                         static_cast<unsigned long long>(resource_id), map_type, static_cast<unsigned long long>(address), static_cast<unsigned long long>(written), size,
                         static_cast<unsigned long long>(memory), in_ring ? 1 : 0, context_frame, resource_stamp,
                         static_cast<unsigned long long>(caller - NativeDraw::ImageBase() + 0x1000000ull), static_cast<u32>(NativeDraw::NextSeq()), static_cast<unsigned long long>(ThreadKey()));
        }
    }
    if (ReplaceAnnounceOn()) {
        u64 old_size = 0;
        {
            std::scoped_lock lock{open_map_mutex};
            const auto it = open_maps.find({ThreadKey(), resource_id});
            old_size = it != open_maps.end() ? it->second.size : 0;
        }
        if (old_size != 0 && CloseOpenMap(resource_id, 1)) {
            c_replace_announced.fetch_add(1, std::memory_order_relaxed);
            c_replace_announced_bytes.fetch_add(old_size, std::memory_order_relaxed);
        }
    }
    NativeContent::NoteMapOpen(address, written); // M3: before the write
    std::scoped_lock lock{open_map_mutex};
    auto& slot = open_maps[{ThreadKey(), resource_id}];
    if (slot.size != 0) {
        c_open_replaced.fetch_add(1, std::memory_order_relaxed);
        c_open_replaced_bytes.fetch_add(slot.size, std::memory_order_relaxed);
        if (slot.in_ring) {
            c_open_replaced_in_ring.fetch_add(1, std::memory_order_relaxed);
        }
    }
    slot = {address, written, ++open_map_serial, NativeDraw::NextSeq(), in_ring, caller, texture_map, map_type, resource_id};
    (void)usage, (void)in_ring;
}

namespace {
struct SiteStat {
    u64 count{}, bytes{}, largest{};
    u64 by_object[3]{}; ///< announcements that named no buffer (0), a block over several (1), one buffer (2)
    u64 by_size_log2[40]{};
};
std::map<std::pair<u32, std::string>, SiteStat> site_stats;
bool SitesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_WRITER_SITES");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}
} // namespace

namespace {
thread_local u8 t_announce_site_class = 0; // N1: 8 while NoteWriterRange runs for a caller that gave no site name
thread_local u64 t_announce_object = 0;    // N1.2: the engine buffer id the caller named
std::array<std::atomic<u64>, BufferIdCounts> buffer_id_counts{};
} // namespace

u32 BufferIdMode() noexcept {
    static const u32 mode = [] {
        return SwitchValue("TLG_BUFFER_ID", 1);
    }();
    return mode;
}

u32 ApplyCoalesceMode() noexcept {
    static const u32 mode = [] {
        return SwitchValue("TLG_APPLY_COALESCE", 1);
    }();
    return mode;
}

namespace {
std::atomic<u64> coalesce_gap_out[4]{}, coalesce_objects{0};
std::atomic<u64> coalesce_guard[2][2]{}; // [gap 4 KiB / 64 KiB][pieces in groups the guard lets merge / pieces in groups it refuses]
std::atomic<u64> coalesce_guard_verdict[2][5]{};
std::atomic<u64> coalesce_class[3][2]{}; // [object: 0 unknown, buffers-only, named][pieces in groups the 4 KiB guard accepts / refuses]
std::atomic<u64> coalesce_in{0}, coalesce_sequential{0}, coalesce_out{0}, coalesce_batches{0}, coalesce_bytes{0};
using Piece = AnnouncedPiece;
bool (*merge_check_fn)(void*, u64, u64, u64, const AnnouncedPiece*, size_t) = nullptr;
void* merge_check_context = nullptr;
std::atomic<u64> coalesce_why[4][2]{};
std::atomic<u64> coalesce_merged_groups{0}, coalesce_refused_groups{0}, coalesce_merged_pieces{0}, coalesce_refused_pieces{0}, coalesce_gap_bytes{0}, coalesce_apply_calls{0};
} // namespace

u64 ApplyCoalesceGap() noexcept {
    static const u64 gap = [] {
        const char* e = std::getenv("TLG_APPLY_COALESCE_GAP");
        return e != nullptr ? static_cast<u64>(std::atoll(e)) : u64{4096};
    }();
    return gap;
}

void NoteMergeRefused(u32 why, u64 pieces) noexcept {
    coalesce_why[why & 3][0].fetch_add(1, std::memory_order_relaxed);
    coalesce_why[why & 3][1].fetch_add(pieces, std::memory_order_relaxed);
}

void SetMergeCheck(bool (*fn)(void*, u64, u64, u64, const AnnouncedPiece*, size_t), void* context) {
    merge_check_context = context;
    merge_check_fn = fn;
}

u32 SyncByIdMode() noexcept {
    static const u32 mode = [] {
        return SwitchValue("TLG_SYNC_BY_ID", 1);
    }();
    return mode;
}

void NoteBufferId(BufferIdCount what) {
    buffer_id_counts[what].fetch_add(1, std::memory_order_relaxed);
}

void NoteWriterRange(u32 writer, u64 address, u64 size, const char* site, bool ring, u64 object) {
    if (SitesOn() && address != 0 && size != 0) {
        std::scoped_lock lock{site_mutex};
        SiteStat& stat = site_stats[{writer, site != nullptr ? site : "?"}];
        ++stat.count;
        ++stat.by_object[object == 0 ? 0 : object == kBuffersOnly ? 1 : 2];
        stat.bytes += size;
        stat.largest = std::max(stat.largest, size);
        u32 log = 0;
        while ((size >> log) > 1 && log < 39) {
            ++log;
        }
        ++stat.by_size_log2[log];
    }
    NativeContent::SetSite(SitesOn() ? NativeContent::SiteId(site != nullptr ? site : "?") : 0);
    t_announce_site_class = site == nullptr ? 8 : 0;
    t_announce_object = object;
    NoteWriterRangeOf(writer, address, size, ThreadKey(), 0, ring);
    t_announce_site_class = 0;
    t_announce_object = 0;
    NativeContent::SetSite(0);
}

namespace {
std::mutex pending_mutex;
std::atomic<u64> announced_by_object_class[8][3][2]{}; // [writer][named: no, block, one buffer][count, bytes]
std::vector<std::pair<u64, u64>> pending_ranges, applied_last, pending_ring_ranges, applied_ring_last;
std::vector<Tag> pending_writers, applied_last_writers, pending_ring_writers, applied_ring_last_writers; // N1: the writer (N1.2: and the buffer id) of each range above
std::vector<std::array<u64, 4>> pending_deleted;
void (*delete_fn)(void*, u64, u64, u64, u32) = nullptr;
void (*create_fn)(void*, u64, u64, u64, bool, u32, u32) = nullptr;
void* create_context = nullptr;
void* delete_context = nullptr;
void (*apply_fn)(void*, u64, u64, bool, u64) = nullptr;
void* apply_context = nullptr;
std::atomic<u64> b2_applied_ranges{0}, b2_applied_bytes{0}, b2_reapplied_ranges{0};
/// TLG_REAPPLY_AUDIT=1 (raster-optimization-plan.md phase 1): does the second application of a range ever matter? The memory of each range is hashed when it is applied first and again
/// when it comes back as `again` one batch later. A change is explained when the newer batch announced bytes over the range (the game reused the memory and said so); a change
/// nothing announced is what the second application exists for (an announcer that fires before its write).
std::atomic<u64> ra_audited{0}, ra_unhashed{0}, ra_misaligned_batches{0}, ra_unchanged{0}, ra_changed_announced{0}, ra_changed_unannounced{0}, ra_unannounced_bytes{0}, ra_batches{0};
} // namespace

/// TLG_REAPPLY: 1 (default) applies a range a second time only when its memory changed since the first application (hash compare); 0 applies every range twice, as before.
/// The audit above found 99.9% of the second applications unchanged (3.71 M ranges), the rest are what the second application exists for.
bool ReapplyGateOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_REAPPLY");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

u32 AnnounceSkipOn() noexcept {
    static const u32 on = [] {
        const char* e = std::getenv("TLG_ANNOUNCE_SKIP");
        return e != nullptr && e[0] == '1' ? 1u : 0u;
    }();
    return on;
}

/// N1 (`TLG_AGAIN_SKIP`): 0 off (default). 1: a range announced by a *trusted writer* is not hashed and not applied a second time. A tag is the writer id (1 skinning matrices, 2 light constants,
/// 4 BeginRecording buffers, 5 CreateBuffer / UpdateSubresource / allocations, ...) plus 8 when the caller gave no site name (4 + 8 = the scene slot blocks). Trusted by default (`TLG_AGAIN_TRUST`
/// is the bit mask over the tags, default 0x1006 = tags 1, 2 and 12): the hooks that stand at the game's write sites. N1 census, three scenes at depth 1, 5.2 M second applications of those
/// tags: 0 changed. Tag 4 with a site name (BeginRecording) changed in 0.1 to 0.5 % and tag 5 (CreateBuffer and the like, announced before the fill) in 1 to 5 %, so they keep the hash.
/// 2: shadow, the gate runs as before and counts every trusted range whose memory changed between its two applications (`TRUSTED_WRONG_MUST_BE_0`).
u32 AgainSkipMode() noexcept {
    static const u32 mode = [] {
        return SwitchValue("TLG_AGAIN_SKIP", 1);
    }();
    return mode;
}
u32 AgainTrustMask() noexcept {
    static const u32 mask = [] {
        const char* e = std::getenv("TLG_AGAIN_TRUST");
        return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 0)) : 0x1006u;
    }();
    return mask;
}
std::array<std::atomic<u64>, 16> again_unchanged_by_writer{}, again_changed_by_writer{}, again_trusted_skipped_by_writer{}, again_trusted_wrong_by_writer{}; // by tag: writer id + 8 when no site name

bool ReapplyAuditOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_REAPPLY_AUDIT");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

namespace {
struct FirstHash {
    u64 address, size, hash;
    bool ok;
};
std::vector<FirstHash> ra_last_first;
std::atomic<u64> ra_gate_skipped{0}, ra_gate_applied{0};
/// TLG_ANNOUNCE_SKIP=1 (2026-11-07, default 0): a plain announced range whose bytes hash the same as when the same address and size were last announced and applied, and that no GPU
/// write and no image overlaps, is not applied again: the caches already hold exactly those bytes (the census found 59 % of the ranges in that state). Counters below.
std::atomic<u64> sk_considered{0}, sk_skipped{0}, sk_skipped_bytes{0}, sk_no_entry{0}, sk_changed{0}, sk_size_differs{0}, sk_unhashed{0}, sk_check_refused{0}, sk_map_cleared{0};
std::map<u64, std::pair<u64, u64>> sk_entries; // address -> size, hash of the last applied announcement (raster thread only); ordered, so a create or delete erases only what it overlaps
std::atomic<u64> sk_refused_clears{0}, sk_refused_gpu{0}, sk_refused_image{0}, sk_refused_tracker{0}, sk_erased_by_deletes{0};
u32 (*skip_check_fn)(void*, u64, u64) = nullptr;
void* skip_check_context = nullptr;

std::atomic<u64> ra_changed_announced_by_tag[16]{}, ra_changed_unannounced_by_tag[16]{};
void ReapplyAuditBefore(const Batch& batch) {
    std::vector<Tag> again_tags = batch.again_writer;
    again_tags.insert(again_tags.end(), batch.ring_again_writer.begin(), batch.ring_again_writer.end());
    std::vector<std::pair<u64, u64>> again = batch.again;
    again.insert(again.end(), batch.ring_again.begin(), batch.ring_again.end());
    std::vector<std::pair<u64, u64>> newer = batch.first;
    newer.insert(newer.end(), batch.ring_first.begin(), batch.ring_first.end());
    std::sort(newer.begin(), newer.end());
    // (start, size) sorted by start; the longest range seen so far bounds how far back an overlapping one can start
    u64 longest = 0;
    for (const auto& r : newer) {
        longest = std::max(longest, r.second);
    }
    const auto announced_over = [&](u64 a, u64 n) {
        auto it = std::lower_bound(newer.begin(), newer.end(), std::pair<u64, u64>{a > longest ? a - longest : 0, 0});
        for (; it != newer.end() && it->first < a + n; ++it) {
            if (it->first + it->second > a) {
                return true;
            }
        }
        return false;
    };
    ra_batches.fetch_add(1, std::memory_order_relaxed);
    if (again.size() != ra_last_first.size()) {
        ra_misaligned_batches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    static std::atomic<u32> logged{0};
    for (size_t i = 0; i < again.size(); ++i) {
        const FirstHash& before = ra_last_first[i];
        if (before.address != again[i].first || before.size != again[i].second) {
            ra_misaligned_batches.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!before.ok) {
            ra_unhashed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        u64 now = 0;
        if (!NativeContent::HashGuarded64(before.address, before.size, now)) {
            ra_unhashed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ra_audited.fetch_add(1, std::memory_order_relaxed);
        const bool cand = N0::On() && i < again_tags.size() && again_tags[i].ring_candidate != 0;
        if (cand) {
            N0::NoteRingOut(now == before.hash ? N0::OutAgainUnchanged : announced_over(before.address, before.size) ? N0::OutAgainChangedAnnounced : N0::OutAgainChangedUnannounced);
        }
        if (now == before.hash) {
            ra_unchanged.fetch_add(1, std::memory_order_relaxed);
        } else if (announced_over(before.address, before.size)) {
            ra_changed_announced.fetch_add(1, std::memory_order_relaxed);
            ra_changed_announced_by_tag[i < again_tags.size() ? again_tags[i].writer & 15 : 15].fetch_add(1, std::memory_order_relaxed);
        } else {
            ra_changed_unannounced_by_tag[i < again_tags.size() ? again_tags[i].writer & 15 : 15].fetch_add(1, std::memory_order_relaxed);
            ra_changed_unannounced.fetch_add(1, std::memory_order_relaxed);
            ra_unannounced_bytes.fetch_add(before.size, std::memory_order_relaxed);
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                std::fprintf(stderr, "reapply audit: %#llx+%#llx changed between its two applications and no newer announcement covers it\n", static_cast<unsigned long long>(before.address),
                             static_cast<unsigned long long>(before.size));
            }
        }
    }
}

/// Remembers the memory of the ranges just applied first; the next batch finds them as `again`.
void RememberFirst(const Batch& batch);
void ReapplyAuditAfter(const Batch& batch) {
    RememberFirst(batch);
}
void RememberFirst(const Batch& batch) {
    ra_last_first.clear();
    const auto add = [&](const std::vector<std::pair<u64, u64>>& list) {
        for (const auto& [address, size] : list) {
            FirstHash h{address, size, 0, false};
            if (size != 0 && size <= (u64{1} << 20)) {
                h.ok = NativeContent::HashGuarded64(address, size, h.hash);
            }
            ra_last_first.push_back(h);
        }
    };
    add(batch.first);
    add(batch.ring_first);
}
} // namespace


void SetSkipCheck(u32 (*fn)(void*, u64, u64), void* context) {
    skip_check_context = context;
    skip_check_fn = fn;
}

void SetApply(void (*fn)(void*, u64, u64, bool, u64), void* context) {
    apply_context = context;
    apply_fn = fn;
}

Batch TakeBatch() {
    Batch batch;
    if (B2NativeMode() == 0) {
        return batch;
    }
    std::scoped_lock lock{pending_mutex};
    batch.first.swap(pending_ranges);
    batch.ring_first.swap(pending_ring_ranges);
    batch.first_writer.swap(pending_writers);
    batch.ring_first_writer.swap(pending_ring_writers);
    batch.deleted.swap(pending_deleted);
    batch.again = applied_last;
    applied_last = batch.first;
    batch.ring_again = applied_ring_last;
    applied_ring_last = batch.ring_first;
    batch.again_writer = applied_last_writers;
    applied_last_writers = batch.first_writer;
    batch.ring_again_writer = applied_ring_last_writers;
    applied_ring_last_writers = batch.ring_first_writer;
    return batch;
}

void SetDelete(void (*fn)(void*, u64, u64, u64, u32), void* context) {
    delete_context = context;
    delete_fn = fn;
}

void SetCreate(void (*fn)(void*, u64, u64, u64, bool, u32, u32), void* context) {
    create_context = context;
    create_fn = fn;
}

void NoteEngineBufferCreate(u64 id, u64 address, u64 bytes, u32 usage, u32 cpu_access) {
    if (!B9PlaceOn() || B2NativeMode() == 0) {
        return;
    }
    std::scoped_lock lock{pending_mutex};
    pending_deleted.push_back({id, address, bytes, 5u | (u64{usage} << 8) | (u64{cpu_access} << 32)}); // the buffer's usage and CPU access ride along: the ordered view of the raster thread needs them
}

void NoteEngineCreate(u64 id, u64 address, u64 bytes) {
    if ((B8FindMode() == 0 && !B9PlaceOn()) || B2NativeMode() == 0) {
        return;
    }
    std::scoped_lock lock{pending_mutex};
    pending_deleted.push_back({id, address, bytes, 2u});
}

void NoteEngineDelete(u64 id, u64 address, u64 bytes, u32 what) {
    if ((what == 4 ? B9LifetimeMode() : B8LifetimeMode()) == 0 || B2NativeMode() == 0) {
        return;
    }
    std::scoped_lock lock{pending_mutex};
    pending_deleted.push_back({id, address, bytes, what});
}

bool RingLeanApplyOn() noexcept {
    static const bool on = [] {
        // default on since 2026-11-10 after the owner's play; =0 off
        const char* e = std::getenv("TLG_RING_LEAN_APPLY");
        return e == nullptr || e[0] == 0 || e[0] != '0';
    }();
    return on;
}

bool RingOutOn() noexcept {
    static const bool on = [] {
        // default on since 2026-11-10 after the owner's play; =0 off
        const char* e = std::getenv("TLG_RING_OUT");
        return e == nullptr || e[0] == 0 || e[0] != '0';
    }();
    return on;
}
std::atomic<u64> c_ring_out[2]{}; // 0 kept out of the batch, 1 sent to the batch as the sampled full check

std::atomic<u64> c_ring_lean[4]{};
void NoteRingLean(u32 what) noexcept {
    if (what < 4) {
        c_ring_lean[what].fetch_add(1, std::memory_order_relaxed);
    }
}

bool EarlyAnnounceOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_EARLY_ANNOUNCE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void ApplyBatch(const Batch& batch) {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseAnnounce};
    TLG_PROF_LEVEL("updates.apply_batch", 1);
    TLG_PROF_ADD("updates.ranges_first", batch.first.size());
    TLG_PROF_ADD("updates.ranges_again", batch.again.size());
    TLG_PROF_ADD("updates.deleted_or_created", batch.deleted.size());
    if (delete_fn != nullptr) {
        TLG_PROF_LEVEL("updates.deletions", 2);
        // deletions first: the work of this submission comes after them in the game's order, and no announcement of it touches a deleted object
        for (const auto& d : batch.deleted) {
            const u32 what = static_cast<u32>(d[3] & 0xff);
            if (what == 2 || what == 5) {
                if (create_fn != nullptr) {
                    create_fn(create_context, d[0], d[1], d[2], what == 2, static_cast<u32>((d[3] >> 8) & 0xffffff), static_cast<u32>(d[3] >> 32));
                }
            } else {
                delete_fn(delete_context, d[0], d[1], d[2], what);
            }
        }
    }
    if (apply_fn == nullptr) {
        return;
    }
    const bool reapply_audit = ReapplyAuditOn();
    if (reapply_audit) {
        ReapplyAuditBefore(batch);
    }
    const bool gate = ReapplyGateOn();
    {
        TLG_PROF_LEVEL("updates.apply_again", 2);
        // gate: the ranges of this list that came back unchanged are not applied a second time (ra_last_first holds the previous batch's first list, the ring ranges after the plain ones)
        const bool aligned = gate && ra_last_first.size() == batch.again.size() + batch.ring_again.size();
        size_t index = 0;
        const u32 again_mode = AgainSkipMode();
        const u32 trust = AgainTrustMask();
        const auto again_list = [&](const std::vector<std::pair<u64, u64>>& list, const std::vector<Tag>& writers) {
            size_t at = 0;
            for (const auto& [address, size] : list) {
                const u32 writer = at < writers.size() ? writers[at].writer : 0u;
                const u64 object = at < writers.size() ? writers[at].object : 0u;
                ++at;
                if (aligned && again_mode == 1 && (trust >> writer & 1) != 0) {
                    ++index; // (the first list holds an entry for every range, hashed or not)
                    again_trusted_skipped_by_writer[writer].fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (aligned) {
                    const FirstHash& before = ra_last_first[index];
                    ++index;
                    u64 now = 0;
                    bool unchanged = false;
                    N0::NoteApplied(size, true);
                    {
                        TLG_PROF_LEVEL("updates.again_hash", 3); // D2: the hash that decides whether a range is applied a second time
                        unchanged = before.ok && before.address == address && before.size == size && NativeContent::HashGuarded64(address, size, now) && now == before.hash;
                    }
                    if (before.ok && now != 0) {
                        (unchanged ? again_unchanged_by_writer : again_changed_by_writer)[writer].fetch_add(1, std::memory_order_relaxed);
                        if (!unchanged && again_mode == 2 && (trust >> writer & 1) != 0) {
                            again_trusted_wrong_by_writer[writer].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    if (unchanged) {
                        ra_gate_skipped.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                }
                ra_gate_applied.fetch_add(1, std::memory_order_relaxed);
                N0::t_ring_candidate = at - 1 < writers.size() && writers[at - 1].ring_candidate != 0;
                apply_fn(apply_context, address, size, false, object);
                N0::t_ring_candidate = false;
            }
        };
        again_list(batch.again, batch.again_writer);
        again_list(batch.ring_again, batch.ring_again_writer);
    }
    TLG_PROF_LEVEL("updates.apply_first", 2);
    const bool skip_on = AnnounceSkipOn() != 0 && skip_check_fn != nullptr;
    const bool hash_here = skip_on || (gate && !reapply_audit); // the hash of a range is taken before it is applied now; it is the one the next batch's gate compares against
    static std::vector<FirstHash> next_first;
    next_first.clear();
    if (skip_on) {
        if (sk_entries.size() > 2'000'000) {
            sk_entries.clear();
            sk_map_cleared.fetch_add(1, std::memory_order_relaxed);
        }
        // an engine buffer or texture was created or deleted over some bytes: what the caches hold for them is not the last announcement any more
        for (const auto& d : batch.deleted) {
            const u64 begin = d[1], end = d[1] + std::max<u64>(d[2], 1);
            auto it = sk_entries.lower_bound(begin > (u64{1} << 20) ? begin - (u64{1} << 20) : 0); // an entry is at most 1 MiB long
            while (it != sk_entries.end() && it->first < end) {
                if (it->first + it->second.first > begin) {
                    it = sk_entries.erase(it);
                    sk_erased_by_deletes.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ++it;
                }
            }
        }
    }
    const u32 first_mode = AgainSkipMode();
    const u32 first_trust = AgainTrustMask();
    const u32 coalesce_mode = ApplyCoalesceMode();
    static std::vector<Piece> todo;
    todo.clear();
    const auto first_one = [&](u64 address, u64 size, bool plain, u32 writer, u64 object, bool ring_candidate) {
        N0::NoteApplied(size, false);
        FirstHash h{address, size, 0, false};
        if (hash_here && size != 0 && size <= (u64{1} << 20) && !(first_mode == 1 && (first_trust >> writer & 1) != 0)) {
            h.ok = NativeContent::HashGuarded64(address, size, h.hash);
        }
        bool skipped = false;
        if (skip_on && plain) {
            sk_considered.fetch_add(1, std::memory_order_relaxed);
            const auto it = sk_entries.find(address);
            if (!h.ok) {
                sk_unhashed.fetch_add(1, std::memory_order_relaxed);
                if (it != sk_entries.end()) {
                    sk_entries.erase(it);
                }
            } else if (it == sk_entries.end()) {
                sk_no_entry.fetch_add(1, std::memory_order_relaxed);
                sk_entries.emplace(address, std::pair<u64, u64>{size, h.hash});
            } else if (it->second.first != size) {
                sk_size_differs.fetch_add(1, std::memory_order_relaxed);
                it->second = {size, h.hash};
            } else if (it->second.second != h.hash) {
                sk_changed.fetch_add(1, std::memory_order_relaxed);
                it->second.second = h.hash;
            } else if (const u32 verdict = skip_check_fn(skip_check_context, address, size); verdict == 0) {
                skipped = true;
                sk_skipped.fetch_add(1, std::memory_order_relaxed);
                sk_skipped_bytes.fetch_add(size, std::memory_order_relaxed);
            } else {
                sk_check_refused.fetch_add(1, std::memory_order_relaxed);
                (verdict == 1 ? sk_refused_gpu : verdict == 2 ? sk_refused_image : verdict == 4 ? sk_refused_clears : sk_refused_tracker).fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!skipped) {
            TLG_PROF_ADD("updates.bytes_first", size);
            if (coalesce_mode != 0 && size != 0) {
                todo.push_back({object, address, size});
            }
            if (coalesce_mode != 1) {
                N0::t_ring_candidate = ring_candidate;
                apply_fn(apply_context, address, size, true, object);
                N0::t_ring_candidate = false;
            }
            b2_applied_bytes.fetch_add(size, std::memory_order_relaxed);
        }
        if (hash_here) {
            next_first.push_back(h);
        }
    };
    {
        size_t at = 0;
        for (const auto& [address, size] : batch.first) {
            first_one(address, size, true, at < batch.first_writer.size() ? batch.first_writer[at].writer : 0u, at < batch.first_writer.size() ? batch.first_writer[at].object : 0u,
                      at < batch.first_writer.size() && batch.first_writer[at].ring_candidate != 0);
            ++at;
        }
        at = 0;
        for (const auto& [address, size] : batch.ring_first) {
            first_one(address, size, false, at < batch.ring_first_writer.size() ? batch.ring_first_writer[at].writer : 0u, at < batch.ring_first_writer.size() ? batch.ring_first_writer[at].object : 0u,
                      at < batch.ring_first_writer.size() && batch.ring_first_writer[at].ring_candidate != 0);
            ++at;
        }
    }
    if (coalesce_mode != 0 && !todo.empty()) {
        // L1: pieces of the same engine buffer that touch or overlap are one range (sorted by buffer, then address). The merged range stays inside what the pieces covered, so the sets the passes
        // keep (dirty bytes, GPU written bytes, resident generations, the image index) end the same; the announcer that named no buffer (0) merges by address alone.
        TLG_PROF_LEVEL("updates.coalesce", 2);
        u64 sequential = 0;
        for (size_t i = 1; i < todo.size(); ++i) {
            sequential += todo[i].object == todo[i - 1].object && todo[i].address >= todo[i - 1].address && todo[i].address <= todo[i - 1].address + todo[i - 1].size ? 1 : 0;
        }
        std::sort(todo.begin(), todo.end(), [](const Piece& a, const Piece& b) { return a.object != b.object ? a.object < b.object : a.address < b.address; });
        u64 out = 0, merged_bytes = 0;
        constexpr u64 MaxMerged = u64{1} << 20;
        const u64 gap = coalesce_mode == 1 ? ApplyCoalesceGap() : 0;
        for (size_t i = 0; i < todo.size();) {
            const u64 object = todo[i].object, begin = todo[i].address;
            u64 end = begin + todo[i].size;
            size_t j = i + 1;
            while (j < todo.size() && todo[j].object == object && todo[j].address <= end + gap && std::max(end, todo[j].address + todo[j].size) - begin <= MaxMerged) {
                end = std::max(end, todo[j].address + todo[j].size);
                ++j;
            }
            merged_bytes += end - begin;
            if (coalesce_mode == 1) {
                // a group the buffer cache refuses is cut in halves and each half asked again (an image next to the pieces refuses the whole run, not its two ends); pieces left alone are applied as they are
                const auto apply_run = [&](const auto& self, size_t a, size_t b, u32 depth) -> void {
                    if (b - a > 1) {
                        u64 run_end = todo[a].address + todo[a].size, piece_bytes = 0;
                        for (size_t k = a; k < b; ++k) {
                            run_end = std::max(run_end, todo[k].address + todo[k].size);
                            piece_bytes += todo[k].size;
                        }
                        const u64 run_begin = todo[a].address;
                        if (merge_check_fn != nullptr && merge_check_fn(merge_check_context, object, run_begin, run_end - run_begin, &todo[a], b - a)) {
                            coalesce_merged_groups.fetch_add(1, std::memory_order_relaxed);
                            coalesce_merged_pieces.fetch_add(b - a, std::memory_order_relaxed);
                            coalesce_gap_bytes.fetch_add((run_end - run_begin) > piece_bytes ? (run_end - run_begin) - piece_bytes : 0, std::memory_order_relaxed);
                            apply_fn(apply_context, run_begin, run_end - run_begin, true, object);
                            ++out;
                            return;
                        }
                        if (depth < 5 && b - a >= 4) {
                            const size_t mid = a + (b - a) / 2;
                            self(self, a, mid, depth + 1);
                            self(self, mid, b, depth + 1);
                            return;
                        }
                        coalesce_refused_groups.fetch_add(1, std::memory_order_relaxed);
                        coalesce_refused_pieces.fetch_add(b - a, std::memory_order_relaxed);
                    }
                    for (size_t k = a; k < b; ++k) {
                        apply_fn(apply_context, todo[k].address, todo[k].size, true, todo[k].object);
                        ++out;
                    }
                };
                apply_run(apply_run, i, j, 0);
            } else {
                ++out;
            }
            i = j;
        }
        if (coalesce_mode == 2) {
            // (census only) how many ranges would be left if pieces up to 256 B, 4 KiB, 64 KiB apart were merged too, and how many different engine buffers the batch names
            static const u64 gaps[4] = {256, 4096, 65536, 1u << 20};
            for (int g = 0; g < 4; ++g) {
                u64 n = 0;
                for (size_t i = 0; i < todo.size();) {
                    const u64 object = todo[i].object;
                    u64 end = todo[i].address + todo[i].size;
                    size_t j = i + 1;
                    while (j < todo.size() && todo[j].object == object && todo[j].address <= end + gaps[g]) {
                        end = std::max(end, todo[j].address + todo[j].size);
                        ++j;
                    }
                    ++n;
                    i = j;
                }
                coalesce_gap_out[g].fetch_add(n, std::memory_order_relaxed);
            }
            if (skip_check_fn != nullptr) {
                for (int g = 1; g <= 2; ++g) {
                    for (size_t i = 0; i < todo.size();) {
                        const u64 object = todo[i].object, begin = todo[i].address;
                        u64 end = begin + todo[i].size;
                        size_t j = i + 1;
                        while (j < todo.size() && todo[j].object == object && todo[j].address <= end + gaps[g]) {
                            end = std::max(end, todo[j].address + todo[j].size);
                            ++j;
                        }
                        if (j - i > 1) {
                            const u32 verdict = skip_check_fn(skip_check_context, begin, end - begin);
                            coalesce_guard[g - 1][verdict == 0 ? 0 : 1].fetch_add(j - i, std::memory_order_relaxed);
                            coalesce_guard_verdict[g - 1][verdict < 5 ? verdict : 4].fetch_add(1, std::memory_order_relaxed);
                            if (g == 1) {
                                coalesce_class[object == 0 ? 0 : object == kBuffersOnly ? 1 : 2][verdict == 0 ? 0 : 1].fetch_add(j - i, std::memory_order_relaxed);
                            }
                        }
                        i = j;
                    }
                }
            }
            u64 objects = 0;
            for (size_t i = 0; i < todo.size(); ++i) {
                objects += i == 0 || todo[i].object != todo[i - 1].object ? 1 : 0;
            }
            coalesce_objects.fetch_add(objects, std::memory_order_relaxed);
        }
        coalesce_batches.fetch_add(1, std::memory_order_relaxed);
        coalesce_in.fetch_add(todo.size(), std::memory_order_relaxed);
        coalesce_sequential.fetch_add(sequential, std::memory_order_relaxed);
        coalesce_out.fetch_add(out, std::memory_order_relaxed);
        coalesce_bytes.fetch_add(merged_bytes, std::memory_order_relaxed);
    }
    b2_applied_ranges.fetch_add(batch.first.size() + batch.ring_first.size(), std::memory_order_relaxed);
    b2_reapplied_ranges.fetch_add(batch.again.size(), std::memory_order_relaxed);
    if (reapply_audit) {
        ReapplyAuditAfter(batch);
    } else if (gate) {
        if (hash_here) {
            ra_last_first.swap(next_first);
        } else {
            RememberFirst(batch);
        }
    }
}

namespace {
std::array<std::atomic<u64>, B8Counts> b8_counts{};
} // namespace





void NoteB8(B8Count what) noexcept {
    b8_counts[what].fetch_add(1, std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<u64>, 4 * 16> e3_overlaps{};
} // namespace
std::array<std::atomic<u64>, 5 * 8 * 4> e3_unnamed{};
std::array<std::atomic<u64>, 4> e3_unnamed_reason{};
void NoteE3UnnamedReason(u32 reason) noexcept {
    e3_unnamed_reason[reason & 3].fetch_add(1, std::memory_order_relaxed);
}
void NoteE3Unnamed(u32 type, u32 site, u32 owner) noexcept {
    e3_unnamed[((type < 5 ? type : 4) * 8 + (site & 7)) * 4 + (owner & 3)].fetch_add(1, std::memory_order_relaxed);
}
void NoteE3Overlap(u32 cls, u32 outcome) noexcept {
    e3_overlaps[(cls & 3) * 16 + (outcome & 15)].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json B8Report() {
    const auto c = [](B8Count what) { return b8_counts[what].load(std::memory_order_relaxed); };
    return {{"mode", B8ImageMode()},
            {"announcements_on_images", c(B8AnnouncedOnImage)},
            {"second_applications_kept_for_gpu", c(B8KeptForGpu)},
            {"lifetime", {{"mode", B8LifetimeMode()}, {"textures_deleted", c(B8TexturesDeleted)}, {"alias_kept", c(B8TextureAliasKept)},
                          {"their_images", c(B8DeletedImages)}, {"images_bigger_left", c(B8DeletedImagesBigger)}, {"images_freed", c(B8DeletedImagesFreed)},
                          {"used_after_delete_new_owner", c(B8UsedAfterDeleteReused)}, {"USED_AFTER_DELETE_NO_OWNER", c(B8UsedAfterDeleteNoOwner)},
                          {"view_ids_dropped", c(B8ViewIdsDropped)}}},
            {"find", {{"mode", B8FindMode()}, {"by_engine_id", c(B8FindById)}, {"AUDIT_OTHER_IMAGE_BY_ADDRESS", c(B8FindByIdOther)},
                      {"no_image_yet", c(B8FindNoImageYet)}, {"no_engine_owner", c(B8FindNoOwner)}, {"images_owned", c(B8FindOwned)},
                      {"by_remembered_answer", c(B8FindByAnswer)}, {"AUDIT_OTHER_ANSWER_BY_ADDRESS", c(B8FindAnswerOther)}, {"answer_stale", c(B8FindAnswerStale)},
                      {"page_table_lookups_no_overlap", c(B8FirstNoOverlap)}, {"page_table_lookups_with_overlap", c(B8FirstOverlap)}}},
            {"route", {{"announcements_by_engine_textures", c(B8RouteAnnounce)}, {"announcements_page_table", c(B8RouteAnnounceFallback)},
                       {"audit_images_missed_not_looked_up", c(B8RouteMissed)}, {"AUDIT_IMAGES_MISSED_LISTED", c(B8RouteMissedListed)},
                       {"images_adopted", c(B8Adopted)}, {"images_adopted_gpu_written", c(B8AdoptedGpu)},
                       {"textures_created", c(B8CreatedTextures)}, {"older_images_unlisted_at_creation", c(B8CreatedUnlisted)},
                       {"first_lookups_by_engine_textures", c(B8FirstByEngine)}, {"first_lookups_page_table", c(B8FirstFallback)},
                       {"AUDIT_FIRST_CANDIDATE_MISSED", c(B8FirstMissedListed)}, {"audit_first_stale_left_out", c(B8FirstStaleLeft)}, {"copies_by_engine_texture", c(B8RouteCopy)}, {"copies_page_table", c(B8RouteCopyFallback)},
                       {"AUDIT_COPY_OTHER", c(B8RouteCopyOther)}}},
            {"buffer_images", {{"mode", B8BufferImagesMode()}, {"announcements_by_engine_buffers", c(B8BufAnnounce)}, {"copies_by_engine_buffers", c(B8BufCopy)},
                               {"images_listed", c(B8BufListed)}, {"AUDIT_MISSED", c(B8BufMissed)}, {"not_covered_page_table", c(B8BufNotCovered)}}},
            {"e3_overlap", [&] {
                 static constexpr const char* classes[4] = {"own_image", "other_engine_texture", "unlisted", "request_has_no_engine_texture"};
                 static constexpr const char* outcomes[16] = {"pool_free",    "depth",           "expand_block",       "expand_3d",    "view_reuse",  "view_incompatible",
                                                              "expand_grow",  "tile_free",       "assert_free",        "right_subres", "right_freed", "right_kept",
                                                              "left_rebind",  "left_copy_mip",   "left_nothing",       "left_none"};
                 nlohmann::json out = nlohmann::json::object();
                 for (u32 k = 0; k < 4; ++k) {
                     for (u32 o = 0; o < 16; ++o) {
                         if (const u64 n = e3_overlaps[k * 16 + o].load(std::memory_order_relaxed); n != 0) {
                             out[classes[k]][outcomes[o]] = n;
                         }
                     }
                 }
                 return out;
             }()},
            {"e3_unnamed_sampled_reason", {{"record_names_binding_with_id_0", e3_unnamed_reason[1].load()}, {"not_from_record_no_plan", e3_unnamed_reason[2].load()}, {"not_from_record_plan_present", e3_unnamed_reason[3].load()}}},
            {"e3_unnamed", [] {
                 static constexpr const char* types[5] = {"texture", "storage", "render_target", "depth_target", "video_out"};
                 static constexpr const char* owners[4] = {"no_engine_resource", "texture_starts_here", "inside_a_texture", "buffer_memory"};
                 nlohmann::json out = nlohmann::json::object();
                 for (u32 t = 0; t < 5; ++t) {
                     for (u32 s = 0; s < 8; ++s) {
                         for (u32 o = 0; o < 4; ++o) {
                             if (const u64 n = e3_unnamed[(t * 8 + s) * 4 + o].load(std::memory_order_relaxed); n != 0) {
                                 out[std::string(types[t]) + " site " + std::to_string(s)][owners[o]] = n;
                             }
                         }
                     }
                 }
                 return out;
             }()},
            {"e3_find", {{"unnamed", c(E3FindUnnamed)}, {"named", c(E3FindNamed)}, {"named_same_owner", c(E3NamedSameOwner)}, {"named_OTHER_OWNER", c(E3NamedOtherOwner)},
                         {"named_interior", c(E3NamedInterior)}, {"named_no_owner", c(E3NamedNoOwner)}, {"named_unknown_object", c(E3NamedUnknownObject)},
                         {"owner_by_id", c(E3OwnerById)}, {"owner_by_id_missed", c(E3OwnerByIdMissed)}, {"id_decides", E3IdDecides()}}},
            {"note", "B8.1, cumulative: images refreshed by the engine's announcements instead of write faults"}};
}

namespace {
std::array<std::atomic<u64>, B9Counts> b9_counts{};
} // namespace







void NoteB9(B9Count what, u64 n) noexcept {
    b9_counts[what].fetch_add(n, std::memory_order_relaxed);
}

u64 ImageBase() noexcept {
    return reinterpret_cast<u64>(GetModuleHandleW(nullptr));
}

u32 LeanDirtyMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_LEAN_DIRTY");
        return e != nullptr && e[0] >= '0' && e[0] <= '2' ? static_cast<u32>(e[0] - '0') : 2u; // default 2 since 2026-11-07 (owner); =0 is the old audit set, =1 verifies
    }();
    return mode;
}

nlohmann::json B9Report() {
    const auto c = [](B9Count what) { return b9_counts[what].load(std::memory_order_relaxed); };
    return {{"mode", B9BufferMode()},
            {"announcements", {{"inside_one_buffer", c(B9AnnouncedOnBuffer)}, {"running_over_several_buffers", c(B9AnnouncedClipped)}, {"not_in_a_buffer", c(B9AnnouncedNoBuffer)}}},
            {"uploads", {{"inside_one_buffer", c(B9SyncByObject)}, {"other_ranges", c(B9SyncNoObject)}}},
            {"TRACKER_ONLY_RANGES", c(B9TrackerOnlyRanges)}, {"tracker_ranges_announced_outside_the_request", c(B9TrackerElsewhere)}, {"tracker_only_bytes", c(B9TrackerOnlyBytes)}, {"announced_only_ranges", c(B9ObjectOnlyRanges)},
            {"bytes_by_page_tracker", c(B9BytesByTracker)}, {"bytes_by_announcements", c(B9BytesByObject)},
            {"placement", {{"on", B9PlaceOn()}, {"placed", c(B9Placed)}, {"placed_bytes", c(B9PlacedBytes)}, {"refused_alias", c(B9PlaceRefusedAlias)},
                           {"refused_size", c(B9PlaceRefusedSize)}, {"requests_served", c(B9PlacedServed)}, {"placements_with_gpu_bytes", c(B9PlacedGpuCopied)},
                           {"moved_back_to_the_arena", c(B9Demoted)}, {"of_those_by_a_request_over_the_edge", c(B9DemotedByRequest)}, {"freed_at_deletion", c(B9PlacedFreed)},
                           {"READBACKS_OVER_PLACED_MEMORY", c(B9ReadbackOverPlaced)}, {"blocks_of_several_buffers", c(B9Groups)},
                           {"buffers_in_those_blocks", c(B9GroupMembers)}, {"placed_buffers_taken_into_a_block", c(B9GroupReused)}, {"new_buffers_joining_a_block", c(B9Adopted)},
                           {"members_deleted_out_of_a_block", c(B9MemberLeft)}, {"requests_on_the_arena_over_engine_buffers", c(B9ArenaOverBuffers)},
                           {"aliases", {{"on", B9AliasesOn()}, {"blocks_with_overlapping_members", c(B9BlockAliased)}, {"blocks_with_textures", c(B9BlockWithTextures)},
                                        {"textures_created_inside_a_block", c(B9TextureInBlockKept)}, {"buffers_created_over_a_member", c(B9AdoptedOverMember)}}}}},
            {"page_tracker", {{"off_for_engine_buffers", B9TrackerOff()}, {"marks_skipped", c(B9TrackerMarksSkipped)}, {"uploads_without_it", c(B9TrackerUploadsSkipped)},
                              {"other_cpu_changes_given_to_buffers", c(B9OtherChanges)}}},
            {"queries", {{"mode", B9QueryMode()},
                         {"cpu_changed", {{"asked", c(B9CpuQueries)}, {"same_answer", c(B9CpuAgree)}, {"tracker_yes_neighbor_bytes", c(B9CpuTrackerNeighbor)},
                                          {"TRACKER_YES_NO_ANNOUNCED_BYTE", c(B9CpuTrackerOnly)}, {"buffer_yes_tracker_no", c(B9CpuObjectOnly)}}},
                         {"gpu_written", {{"asked", c(B9GpuQueries)}, {"same_answer", c(B9GpuAgree)}, {"tracker_yes_bytes_elsewhere_on_the_page", c(B9GpuTrackerPage)},
                                          {"tracker_yes_no_gpu_byte_on_the_pages", c(B9GpuTrackerOnly)}, {"bytes_yes_tracker_no", c(B9GpuBytesOnly)}}}}},
            {"lifetime", {{"mode", B9LifetimeMode()}, {"buffers_deleted", c(B9BuffersDeleted)}, {"with_announced_bytes_never_uploaded", c(B9DeletedPending)},
                          {"with_gpu_written_bytes", c(B9DeletedGpuWritten)}, {"memory_shared_with_a_live_buffer", c(B9DeletedAlias)},
                          {"gpu_written_bytes_dropped", c(B9DeletedGpuDropped)}}},
            {"e4_no_arena", {{"announced_uploads_bytes", c(E4AnnouncedUploads)}, {"blocks_freed_with_foreign_gpu_bytes", c(E4FreedWithForeignGpu)}, {"foreign_gpu_bytes", c(E4FreedForeignGpuBytes)}, {"gpu_bytes_lost_at_placement", c(E4GpuBytesLost)}, {"pseudo_placements", c(E4PseudoPlaced)}, {"pseudo_ring", c(E4PseudoRing)}, {"demotes_kept", c(E4DemoteKept)}}},
            {"lean_dirty", {{"mode", LeanDirtyMode()}, {"answers_compared", c(LeanChecked)}, {"LEAN_MISSING_BYTES_MUST_BE_0", c(LeanMissing)}, {"lean_extra_bytes", c(LeanExtra)}, {"bytes_moved_from_deleted_buffers", c(LeanMoved)}}},
            {"note", "B9.1, cumulative: engine buffers uploaded by their announced bytes instead of the page tracker's pages"}};
}

nlohmann::json B2Report() {
    return {{"mode", B2NativeMode()}, {"announcements_applied", b2_applied_ranges.load()}, {"bytes_applied", b2_applied_bytes.load()},
            {"announcements_applied_again", b2_reapplied_ranges.load()}, {"announce_skip", {{"mode", AnnounceSkipOn()}, {"considered", sk_considered.load()}, {"skipped", sk_skipped.load()}, {"skipped_bytes", sk_skipped_bytes.load()}, {"no_entry", sk_no_entry.load()},
                                    {"bytes_changed", sk_changed.load()}, {"size_differs", sk_size_differs.load()}, {"unhashed", sk_unhashed.load()}, {"same_bytes_but_check_refused", sk_check_refused.load()},
                                    {"refused_gpu_wrote_the_range", sk_refused_gpu.load()}, {"refused_an_image_covers_it", sk_refused_image.load()}, {"refused_a_clear_waits_for_an_image", sk_refused_clears.load()}, {"refused_tracker_mode", sk_refused_tracker.load()},
                                    {"entries_erased_by_creates_or_deletes", sk_erased_by_deletes.load()}, {"map_cleared_when_too_big", sk_map_cleared.load()}}},
                  {"ring_mirror", RingMirror::Report()},
                  {"buffer_id", {{"mode", BufferIdMode()}, {"told_whole", buffer_id_counts[IdToldWhole].load()}, {"told_partial", buffer_id_counts[IdToldPartial].load()}, {"unknown", buffer_id_counts[IdUnknown].load()},
                                  {"parts_match", buffer_id_counts[IdPartsMatch].load()}, {"PARTS_DIFFER_MUST_BE_0", buffer_id_counts[IdPartsDiffer].load()}, {"image_skipped", buffer_id_counts[IdImageSkipped].load()},
                                  {"IMAGE_FOUND_OVER_TOLD_MUST_BE_0", buffer_id_counts[IdImageFoundOverTold].load()}, {"told_block", buffer_id_counts[IdToldBlock].load()},
                                  {"coalesce", {{"mode", ApplyCoalesceMode()}, {"batches", coalesce_batches.load()}, {"pieces_in", coalesce_in.load()}, {"touching_the_previous_piece", coalesce_sequential.load()}, {"ranges_out", coalesce_out.load()}, {"gap", ApplyCoalesceGap()}, {"merged_groups", coalesce_merged_groups.load()}, {"merged_pieces", coalesce_merged_pieces.load()}, {"refused_groups", coalesce_refused_groups.load()}, {"refused_why_mode_namedbuffer_gpuwritten_image_groups_pieces", {{coalesce_why[0][0].load(), coalesce_why[0][1].load()}, {coalesce_why[1][0].load(), coalesce_why[1][1].load()}, {coalesce_why[2][0].load(), coalesce_why[2][1].load()}, {coalesce_why[3][0].load(), coalesce_why[3][1].load()}}}, {"refused_pieces", coalesce_refused_pieces.load()}, {"gap_bytes_marked_extra", coalesce_gap_bytes.load()}, {"ranges_if_gap_256", coalesce_gap_out[0].load()}, {"ranges_if_gap_4k", coalesce_gap_out[1].load()}, {"ranges_if_gap_64k", coalesce_gap_out[2].load()}, {"ranges_if_gap_1m", coalesce_gap_out[3].load()}, {"distinct_buffers_summed_over_batches", coalesce_objects.load()},
                                              {"guard_4k_pieces_in_mergeable_groups", coalesce_guard[0][0].load()}, {"guard_4k_pieces_in_refused_groups", coalesce_guard[0][1].load()}, {"guard_64k_pieces_in_mergeable_groups", coalesce_guard[1][0].load()}, {"guard_64k_pieces_in_refused_groups", coalesce_guard[1][1].load()},
                                              {"guard_4k_pieces_by_object_unknown_buffersonly_named_accepted_refused", {{coalesce_class[0][0].load(), coalesce_class[0][1].load()}, {coalesce_class[1][0].load(), coalesce_class[1][1].load()}, {coalesce_class[2][0].load(), coalesce_class[2][1].load()}}},
                                              {"guard_4k_verdicts_0_to_4", {coalesce_guard_verdict[0][0].load(), coalesce_guard_verdict[0][1].load(), coalesce_guard_verdict[0][2].load(), coalesce_guard_verdict[0][3].load(), coalesce_guard_verdict[0][4].load()}}, {"bytes_out", coalesce_bytes.load()}}},
                                  {"sync_by_id", {{"mode", SyncByIdMode()}, {"used", buffer_id_counts[IdSyncToldUsed].load()}, {"held_match", buffer_id_counts[IdSyncHeldMatch].load()}, {"HELD_DIFFER_MUST_BE_0", buffer_id_counts[IdSyncHeldDiffer].load()},
                                                  {"parts_match", buffer_id_counts[IdSyncPartsMatch].load()}, {"PARTS_DIFFER_MUST_BE_0", buffer_id_counts[IdSyncPartsDiffer].load()}}}}},
                  {"again_by_writer", [] {
                       nlohmann::json j = nlohmann::json::object();
                       j["mode"] = AgainSkipMode();
                       j["trust_mask"] = AgainTrustMask();
                       for (u32 w = 0; w < 16; ++w) {
                           j[(w >= 8 ? "no_site_" : "") + std::to_string(w & 7)] = {{"unchanged", again_unchanged_by_writer[w].load()}, {"changed", again_changed_by_writer[w].load()}, {"trusted_skipped", again_trusted_skipped_by_writer[w].load()}, {"TRUSTED_WRONG_MUST_BE_0", again_trusted_wrong_by_writer[w].load()}};
                       }
                       return j;
                   }()},
                  {"reapply_gate", {{"on", ReapplyGateOn()}, {"skipped_unchanged", ra_gate_skipped.load()}, {"applied_changed_or_unknown", ra_gate_applied.load()}}},
            {"reapply_audit", {{"on", ReapplyAuditOn()}, {"batches", ra_batches.load()}, {"misaligned_batches", ra_misaligned_batches.load()}, {"ranges_audited", ra_audited.load()}, {"unhashed", ra_unhashed.load()},
                               {"unchanged", ra_unchanged.load()}, {"changed_and_announced_again", ra_changed_announced.load()}, {"changed_nobody_announced", ra_changed_unannounced.load()}, {"changed_nobody_announced_by_tag", {ra_changed_unannounced_by_tag[0].load(), ra_changed_unannounced_by_tag[1].load(), ra_changed_unannounced_by_tag[2].load(), ra_changed_unannounced_by_tag[3].load(), ra_changed_unannounced_by_tag[4].load(), ra_changed_unannounced_by_tag[5].load(), ra_changed_unannounced_by_tag[6].load(), ra_changed_unannounced_by_tag[7].load()}}, {"changed_announced_again_by_tag", {ra_changed_announced_by_tag[0].load(), ra_changed_announced_by_tag[1].load(), ra_changed_announced_by_tag[2].load(), ra_changed_announced_by_tag[3].load(), ra_changed_announced_by_tag[4].load(), ra_changed_announced_by_tag[5].load(), ra_changed_announced_by_tag[6].load(), ra_changed_announced_by_tag[7].load()}},
                               {"changed_nobody_announced_bytes", ra_unannounced_bytes.load()}}},
            {"note", "B2, cumulative: the engine's announced CPU writes applied to the caches"}};
}

namespace {
std::array<std::atomic<u64>, 8> announced_ranges_by_writer{}, announced_bytes_by_writer{};
std::atomic<u64> announced_in_ring[8][2]{}; // phase 0b: [writer][ranges, bytes] of the announcements that lie inside the frame ring
} // namespace

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread, u64 stamp, bool ring) {
    if (address == 0 || size == 0) {
        return;
    }
    {
        const u32 cls = t_announce_object == 0 ? 0u : t_announce_object == kBuffersOnly ? 1u : 2u;
        announced_by_object_class[writer & 7][cls][0].fetch_add(1, std::memory_order_relaxed);
        announced_by_object_class[writer & 7][cls][1].fetch_add(size, std::memory_order_relaxed);
    }
    announced_ranges_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
    if (NativeContent::InRing(address, size)) {
        announced_in_ring[writer & 7][0].fetch_add(1, std::memory_order_relaxed);
        announced_in_ring[writer & 7][1].fetch_add(size, std::memory_order_relaxed);
    }
    Core::RingCensus::NoteAnnounce(writer, address, size); // TLG_RING_CENSUS (counters only)
    if (RowVersion::Watched(address, size)) [[unlikely]] {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 40) {
            std::fprintf(stderr, "watch: announced writer %u %#llx+%#llx object %#llx texture %d ring %d site class %u thread %llu render thread %d\n", writer, static_cast<unsigned long long>(address),
                         static_cast<unsigned long long>(size), static_cast<unsigned long long>(t_announce_object), t_announce_texture ? 1 : 0, ring ? 1 : 0, t_announce_site_class,
                         static_cast<unsigned long long>(thread), thread == render_thread.load(std::memory_order_relaxed) ? 1 : 0);
        }
    }
    if (Core::HookRing::FillsWriter(writer)) {
        Core::HookRing::Fill(address, size, writer); // TLG_HOOK_RING (F1 1a): the write is finished, copy it now
    }
    announced_bytes_by_writer[writer & 7].fetch_add(size, std::memory_order_relaxed);
    if (RingMirror::ShadowOn()) {
        RingMirror::NoteWriter(address, size, (writer & 7) | t_announce_site_class);
    }
    bool ring_out = false;
    const bool ring_candidate = B2NativeMode() != 0 && (N0::On() || RingLeanApplyOn() || RingOutOn()) && NativeContent::InRing(address, size) &&
                                ((writer & 7) == 3 || ((writer & 7) == 0 && !t_announce_texture)); // F1 1c
    if (ring_candidate && RingOutOn()) {
        static thread_local u32 out_count = 0;
        ring_out = (++out_count & 63) != 0;
        c_ring_out[ring_out ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
    }
    if (B2NativeMode() != 0 && !ring_out) {
        std::scoped_lock lock{pending_mutex};
        // memory the engine writes fresh every frame (the frame rings): applied early on the raster thread
        const bool fresh = EarlyAnnounceOn() && (writer == 1 || writer == 2 || writer == 4 || ring);
        (fresh ? pending_ring_ranges : pending_ranges).emplace_back(address, size);
        (fresh ? pending_ring_writers : pending_writers).push_back(Tag{static_cast<u8>((writer & 7) | t_announce_site_class), t_announce_object, static_cast<u8>(ring_candidate ? 1 : 0)});
    }
    NativeResources::NoteAnnouncedWrite(address, size); // C1: resident buffers on those pages are not synchronized any more
    NativeUploads::Announce(writer, address, size, thread == render_thread.load(std::memory_order_relaxed), NativeDraw::NextSeq());
    // A1.0: the constant buffer copies (writer 3) and the Maps of buffers are not image content: nothing reads their bytes from the store
    const bool image_content = !(writer == 3 || (writer == 0 && !t_announce_texture));
    NativeContent::Announce(writer, address, size, stamp != 0 ? stamp : NativeDraw::NextSeq(), image_content); // F2.2: the bytes now, on the writing thread
    if (supersede_fn != nullptr && SupersedeEnabled() && (writer == 5 || writer == 6 || (writer == 0 && thread == render_thread.load(std::memory_order_relaxed)))) {
        supersede_fn(address, size);
        superseded_ranges.fetch_add(1, std::memory_order_relaxed);
        superseded_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = enabled.load();
    j["ring_out_of_batch_real"] = {{"on", RingOutOn()}, {"kept_out", c_ring_out[0].load()}, {"sent_as_sampled_full_check", c_ring_out[1].load()}};
    j["ring_lean_apply"] = {{"on", RingLeanApplyOn()}, {"applied_lean", c_ring_lean[0].load()}, {"sampled_full", c_ring_lean[1].load()},
                            {"SAMPLED_WITH_BUFFER_MEMO_RECIPE_OR_RESIDENT_EFFECT_MUST_BE_0", c_ring_lean[2].load()}, {"rows_not_learned_in_ring", c_ring_lean[3].load()}};
    for (u32 w = 0; w < 8; ++w) {
        j["announced_by_writer"][std::to_string(w)] = {{"ranges", announced_ranges_by_writer[w].load()}, {"bytes", announced_bytes_by_writer[w].load()}};
        j["announced_by_writer"][std::to_string(w)]["in_frame_ring"] = {{"ranges", announced_in_ring[w][0].load()}, {"bytes", announced_in_ring[w][1].load()}};
        j["announced_by_writer"][std::to_string(w)]["object_unnamed"] = {{"ranges", announced_by_object_class[w][0][0].load()}, {"bytes", announced_by_object_class[w][0][1].load()}};
        j["announced_by_writer"][std::to_string(w)]["object_block"] = {{"ranges", announced_by_object_class[w][1][0].load()}, {"bytes", announced_by_object_class[w][1][1].load()}};
        j["announced_by_writer"][std::to_string(w)]["object_named"] = {{"ranges", announced_by_object_class[w][2][0].load()}, {"bytes", announced_by_object_class[w][2][1].load()}};
    }
    j["gpu_written_marks_superseded"] = {{"enabled", SupersedeEnabled()}, {"ranges", superseded_ranges.load()}, {"bytes", superseded_bytes.load()}};
    auto& m = j["maps_of_buffers"] = nlohmann::json::object();
    m["open_map_REPLACED_before_unmap"] = {{"count", c_open_replaced.load()}, {"bytes", c_open_replaced_bytes.load()}, {"in_ring", c_open_replaced_in_ring.load()}};
    m["unmap_without_open_map"] = c_unmap_unmatched.load();
    m["replace_announce"] = {{"on", ReplaceAnnounceOn()}, {"announced", c_replace_announced.load()}, {"bytes", c_replace_announced_bytes.load()}};
    m["frame_start_flush"] = {{"keep_open", KeepOpenOn()},
                              {"flushed_render_thread", c_fs_flushed_own.load()},
                              {"flushed_other_threads", c_fs_flushed_other.load()},
                              {"UNMAP_AFTER_ITS_MAP_WAS_FLUSHED", c_unmap_after_fs_flush.load()},
                              {"kept_open", c_fs_kept.load()},
                              {"kept_then_closed_by_unmap", c_unmap_kept_closed.load()},
                              {"kept_then_dropped_at_next_frame_start", c_fs_dropped_after_keep.load()}};
    {
        std::scoped_lock lock{open_map_mutex};
        u64 bytes = 0;
        for (const auto& [key, map] : open_maps) {
            bytes += map.size;
        }
        m["open_now"] = {{"count", open_maps.size()}, {"bytes", bytes}};
    }
    m["total"] = maps.total.load();
    m["into_the_buffers_own_memory"] = maps.own_memory.load();
    m["into_a_frame_ring_window"] = maps.in_ring.load();
    constexpr const char* type_names[8] = {"0", "read", "write", "read_write", "write_discard", "write_no_overwrite", "6", "other"};
    for (u32 i = 0; i < 8; ++i) {
        if (const u64 n = maps.by_type[i].load()) {
            m["by_map_type"][type_names[i]] = n;
        }
    }
    constexpr const char* usage_names[4] = {"default", "immutable", "dynamic", "staging"};
    for (u32 i = 0; i < 4; ++i) {
        if (const u64 n = maps.by_usage[i].load()) {
            m["by_buffer_usage"][usage_names[i]] = n;
        }
    }
    if (SitesOn()) {
        std::scoped_lock lock{site_mutex};
        for (const auto& [name, diffs] : map_diff) {
            auto& e = j["map_frame_minus_stamp"][name];
            for (u32 i = 0; i < 13; ++i) {
                if (diffs[i] != 0) {
                    e[std::to_string(i)] = diffs[i];
                }
            }
        }
        auto& sites = j["writer_sites"] = nlohmann::json::object();
        for (const auto& [key, stat] : site_stats) {
            auto& e = sites[std::to_string(key.first) + " " + key.second];
            e = {{"count", stat.count}, {"bytes", stat.bytes}, {"largest", stat.largest}, {"unnamed", stat.by_object[0]}, {"block", stat.by_object[1]}, {"named", stat.by_object[2]}};
            for (u32 i = 0; i < 40; ++i) {
                if (stat.by_size_log2[i] != 0) {
                    e["by_size_log2"][std::to_string(i)] = stat.by_size_log2[i];
                }
            }
        }
    }
    return j;
}

} // namespace Core::NativeUpdates
