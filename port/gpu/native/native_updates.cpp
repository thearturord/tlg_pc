// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
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
#include "native/native_draw.h"
#include "native/native_registry.h"
#include "native/native_resources.h"
#include "native/native_uploads.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_image_caller.h"
#include "native/native_content.h"
#include "native/native_prof.h"

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
};
std::mutex open_map_mutex;
u64 open_map_serial = 0;
std::map<std::pair<u64, u64>, OpenMap> open_maps; ///< (thread, resource id) -> the window of the Map that has not been unmapped yet
} // namespace

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread, u64 stamp = 0);

void NoteRenderThread() {
    render_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()), std::memory_order_relaxed);
    NativeDraw::FlushSceneSlotAnnouncements(); // C4: the last scene slots of the frame that just ended
    // a Map that was never unmapped (the engine keeps some pointers for the whole frame): its window is announced at the next frame start, for the thread that mapped it
    std::vector<std::pair<u64, OpenMap>> stale;
    {
        std::scoped_lock lock{open_map_mutex};
        for (const auto& [key, map] : open_maps) {
            stale.emplace_back(key.first, map);
        }
        open_maps.clear();
    }
    for (const auto& [thread, map] : stale) {
        NoteWriterRangeOf(0, map.address, map.size, thread, map.stamp);
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

void NoteUnmap(u64 resource_id) {
    OpenMap map;
    {
        std::scoped_lock lock{open_map_mutex};
        const auto it = open_maps.find({ThreadKey(), resource_id});
        if (it == open_maps.end()) {
            return;
        }
        map = it->second;
        open_maps.erase(it);
    }
    NativeUploads::SnapshotTake(map.address, map.size, NativeDraw::NextSeq());
    NoteWriterRange(0, map.address, map.size, map.in_ring ? "Map into a frame ring window" : "Map into the buffer's own memory");
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
        NoteWriterRangeOf(0, map.address, map.size, thread, map.stamp);
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

void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size) {
    (void)caller;
    maps.total.fetch_add(1, std::memory_order_relaxed);
    maps.by_type[map_type < 8 ? map_type : 7].fetch_add(1, std::memory_order_relaxed);
    maps.by_usage[usage < 4 ? usage : 3].fetch_add(1, std::memory_order_relaxed);
    const bool in_ring = address != memory;
    (in_ring ? maps.in_ring : maps.own_memory).fetch_add(1, std::memory_order_relaxed);
    if (address == 0 || size == 0 || map_type == 1) {
        return;
    }
    // the written part: the requested size when the caller gave one, else everything the Map handed out; it is announced when the thread unmaps the resource
    const u64 written = requested_size != 0 && requested_size < size ? requested_size : size;
    NativeContent::NoteMapOpen(address, written); // M3: before the write
    std::scoped_lock lock{open_map_mutex};
    open_maps[{ThreadKey(), resource_id}] = {address, written, ++open_map_serial, NativeDraw::NextSeq(), in_ring};
    (void)usage, (void)in_ring;
}

namespace {
struct SiteStat {
    u64 count{}, bytes{}, largest{};
    u64 by_size_log2[40]{};
};
std::mutex site_mutex;
std::map<std::pair<u32, std::string>, SiteStat> site_stats;
bool SitesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_WRITER_SITES");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}
} // namespace

void NoteWriterRange(u32 writer, u64 address, u64 size, const char* site) {
    if (SitesOn() && address != 0 && size != 0) {
        std::scoped_lock lock{site_mutex};
        SiteStat& stat = site_stats[{writer, site != nullptr ? site : "?"}];
        ++stat.count;
        stat.bytes += size;
        stat.largest = std::max(stat.largest, size);
        u32 log = 0;
        while ((size >> log) > 1 && log < 39) {
            ++log;
        }
        ++stat.by_size_log2[log];
    }
    NoteWriterRangeOf(writer, address, size, ThreadKey());
}

namespace {
std::mutex pending_mutex;
std::vector<std::pair<u64, u64>> pending_ranges, applied_last;
std::vector<std::array<u64, 4>> pending_deleted;
void (*delete_fn)(void*, u64, u64, u64, u32) = nullptr;
void (*create_fn)(void*, u64, u64, u64, bool, u32, u32) = nullptr;
void* create_context = nullptr;
void* delete_context = nullptr;
void (*apply_fn)(void*, u64, u64, bool) = nullptr;
void* apply_context = nullptr;
std::atomic<u64> b2_applied_ranges{0}, b2_applied_bytes{0}, b2_reapplied_ranges{0};
} // namespace


void SetApply(void (*fn)(void*, u64, u64, bool), void* context) {
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
    batch.deleted.swap(pending_deleted);
    batch.again = applied_last;
    applied_last = batch.first;
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

void ApplyBatch(const Batch& batch) {
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
    {
        TLG_PROF_LEVEL("updates.apply_again", 2);
        for (const auto& [address, size] : batch.again) {
            apply_fn(apply_context, address, size, false);
        }
    }
    TLG_PROF_LEVEL("updates.apply_first", 2);
    for (const auto& [address, size] : batch.first) {
        TLG_PROF_ADD("updates.bytes_first", size);
        apply_fn(apply_context, address, size, true);
        b2_applied_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    b2_applied_ranges.fetch_add(batch.first.size(), std::memory_order_relaxed);
    b2_reapplied_ranges.fetch_add(batch.again.size(), std::memory_order_relaxed);
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
            {"note", "B9.1, cumulative: engine buffers uploaded by their announced bytes instead of the page tracker's pages"}};
}

nlohmann::json B2Report() {
    return {{"mode", B2NativeMode()}, {"announcements_applied", b2_applied_ranges.load()}, {"bytes_applied", b2_applied_bytes.load()},
            {"announcements_applied_again", b2_reapplied_ranges.load()},
            {"note", "B2, cumulative: the engine's announced CPU writes applied to the caches"}};
}

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread, u64 stamp) {
    if (address == 0 || size == 0) {
        return;
    }
    if (B2NativeMode() != 0) {
        std::scoped_lock lock{pending_mutex};
        pending_ranges.emplace_back(address, size);
    }
    NativeResources::NoteAnnouncedWrite(address, size); // C1: resident buffers on those pages are not synchronized any more
    NativeUploads::Announce(writer, address, size, thread == render_thread.load(std::memory_order_relaxed), NativeDraw::NextSeq());
    NativeContent::Announce(writer, address, size, stamp != 0 ? stamp : NativeDraw::NextSeq()); // F2.2: the bytes now, on the writing thread
    if (supersede_fn != nullptr && SupersedeEnabled() && (writer == 5 || writer == 6 || (writer == 0 && thread == render_thread.load(std::memory_order_relaxed)))) {
        supersede_fn(address, size);
        superseded_ranges.fetch_add(1, std::memory_order_relaxed);
        superseded_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = enabled.load();
    j["gpu_written_marks_superseded"] = {{"enabled", SupersedeEnabled()}, {"ranges", superseded_ranges.load()}, {"bytes", superseded_bytes.load()}};
    auto& m = j["maps_of_buffers"] = nlohmann::json::object();
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
        auto& sites = j["writer_sites"] = nlohmann::json::object();
        for (const auto& [key, stat] : site_stats) {
            auto& e = sites[std::to_string(key.first) + " " + key.second];
            e = {{"count", stat.count}, {"bytes", stat.bytes}, {"largest", stat.largest}};
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
