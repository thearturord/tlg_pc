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

constexpr size_t WindowCount = 4096;
constexpr u64 NowWindowNs = 200'000'000; ///< a write this soon after the Map of its own thread belongs to that Map

struct Window {
    u64 begin{}, end{};
    u64 id{};
    u64 time_ns{};
    u64 thread{};
    u32 usage{};
    u32 type{};
    bool in_ring{}; ///< the engine handed out a window of the frame ring, not the buffer's own memory
    u32 writer{};   ///< 0: a Map, else the announcing engine function
};

std::mutex mutex;
std::array<Window, WindowCount> windows{};
u64 window_count = 0;

/// Which thread announces: the thread of `RenderDev::BeginRecording` (the render thread, where the frame's commands are recorded) or another one. Per announcer 0..7;
/// 0 is a Map. Threads other than the render thread are numbered in the order they first announce.
std::atomic<u64> render_thread{0};
std::array<std::atomic<u64>, 8> announced_on_render_thread{}, announced_elsewhere{};
std::mutex thread_mutex;
std::map<u64, u32> other_threads;                                    ///< thread key -> number
std::array<std::array<std::atomic<u64>, 8>, 8> other_thread_counts{}; ///< [announcer][thread number], 8 threads at most
void NoteAnnouncerThread(u32 writer) {
    const u64 thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const u64 render = render_thread.load(std::memory_order_relaxed);
    if (render != 0 && thread == render) {
        announced_on_render_thread[writer & 7].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    announced_elsewhere[writer & 7].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{thread_mutex};
    const auto [it, added] = other_threads.try_emplace(thread, static_cast<u32>(other_threads.size()));
    if (it->second < 8) {
        other_thread_counts[writer & 7][it->second].fetch_add(1, std::memory_order_relaxed);
    }
}

struct MapCounters {
    std::array<std::atomic<u64>, 8> by_type{};   ///< D3D11 map type 1..5 (0 and 6, 7 other)
    std::array<std::atomic<u64>, 4> by_usage{};  ///< 0 default 1 immutable 2 dynamic 3 staging
    std::atomic<u64> in_ring{}, own_memory{};
    std::atomic<u64> total{};
} maps;

struct FaultCounters {
    std::atomic<u64> total{};
    std::atomic<u64> own_host_writes{}; ///< written by our own native code (tags into the game's command buffers), not by the game
    std::atomic<u64> covered_now{}, covered_earlier{}, covered_other_thread{}, uncovered{};
    std::array<std::atomic<u64>, 8> covered_by_writer{}; ///< covered_now by the announcing writer (0: a Map)
    std::array<std::atomic<u64>, 16> uncovered_by_class{};
    std::array<std::atomic<u64>, 16> covered_by_class{};
    std::array<std::atomic<u64>, 4> uncovered_by_registry{}; ///< C4: the faults no announcement covers, by whether the written bytes are inside a registered buffer
} faults;

struct Site {
    u64 count{};
    u64 sample_address{};
    u32 usage_class{};
    u32 registry{}; ///< BufferContaining of the sample
    u64 in_registered{}; ///< faults of this site inside a registered buffer (the ones that can change what the GPU reads)
    std::array<u64, 8> host_stack{}; ///< exe offsets of the host frames at the first fault of a site that has no guest frame (a copy made by the runtime)
    u32 host_frames{};
};
struct PendingFault {
    u64 address, size, time_ns, thread;
    u32 usage_class;
    u64 rip, st0, st1;
    u32 registry;
};
std::array<PendingFault, 8192> pending{}; ///< faults that no announcement covered yet (a writer announces after it wrote): recent ones, newest last
u64 pending_count = 0;

struct SiteKey {
    u64 rip, st0, st1;
    bool operator<(const SiteKey& o) const {
        return std::tie(rip, st0, st1) < std::tie(o.rip, o.st0, o.st1);
    }
};
std::map<SiteKey, Site> map_sites; ///< the writers that fill the memory a Map handed out (faults inside a Map window of the same thread)
std::map<SiteKey, Site> uncovered_sites; ///< by faulting instruction and the first two stack words (the caller of a leaf function such as memcpy)

/// Faults that come without a faulting instruction are explicit invalidations (the runtime reports a protection change or a system call write): where from?
struct NoInstruction {
    u64 count{}, bytes{};
    std::array<u64, 12> stack{};
    u32 frames{};
    u64 first_address{};
};
std::map<u64, NoInstruction> no_instruction; ///< by the first frame of the host stack outside this module
std::array<std::atomic<u64>, 8> no_instruction_sizes{}; ///< size buckets: 1, <=8, <=64, <=4K, <=64K, <=1M, more

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

u64 ThreadKey() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace

namespace {
struct OpenMap {
    u64 address{}, size{};
};
std::mutex open_map_mutex;
std::map<std::pair<u64, u64>, OpenMap> open_maps; ///< (thread, resource id) -> the window of the Map that has not been unmapped yet
} // namespace

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread);

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
        NoteWriterRangeOf(0, map.address, map.size, thread);
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
std::atomic<bool> audit_faults{[] {
    const char* e = std::getenv("TLG_UPDATE_AUDIT");
    return e != nullptr && *e != 0 && *e != '0';
}()};
} // namespace

bool AuditFaults() {
    return audit_faults.load(std::memory_order_relaxed);
}

void Start() {
    enabled.store(true, std::memory_order_relaxed);
    audit_faults.store(true, std::memory_order_relaxed);
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
    NoteWriterRange(0, map.address, map.size);
}

void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size) {
    (void)caller;
    if (AuditFaults()) {
        NoteAnnouncerThread(0);
    }
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
    std::scoped_lock lock{open_map_mutex};
    open_maps[{ThreadKey(), resource_id}] = {address, written};
    (void)usage, (void)in_ring;
}

void NoteWriterRange(u32 writer, u64 address, u64 size) {
    NoteWriterRangeOf(writer, address, size, ThreadKey());
}

namespace {
constexpr u64 MarkSlots = u64{1} << 20;
std::unique_ptr<std::atomic<u64>[]> page_marks;
std::once_flag page_marks_once;
std::array<std::atomic<u64>, 2> harvested_ring{}, harvested_other{}; // [announced, unannounced]
std::array<std::atomic<u64>, 64> unannounced_by_class{};
std::mutex owner_mutex;
std::unordered_map<std::string, u64> unannounced_by_owner;
std::atomic<u32> unannounced_logged{0};
std::atomic<u64>* MarkOf(u64 page) {
    std::call_once(page_marks_once, [] { page_marks = std::make_unique<std::atomic<u64>[]>(MarkSlots); });
    return &page_marks[((page >> 12) * 0x9E3779B97F4A7C15ull >> 40) & (MarkSlots - 1)];
}
void MarkAnnounced(u64 address, u64 size) {
    for (u64 page = address & ~u64{4095}; page < address + size; page += 4096) {
        MarkOf(page)->store(page | 1, std::memory_order_relaxed);
    }
}
} // namespace

namespace {
std::mutex pending_mutex;
std::vector<std::pair<u64, u64>> pending_ranges, applied_last;
std::vector<std::array<u64, 4>> pending_deleted;
void (*delete_fn)(void*, u64, u64, u64, u32) = nullptr;
void (*create_fn)(void*, u64, u64, u64, bool) = nullptr;
void* create_context = nullptr;
void* delete_context = nullptr;
void (*apply_fn)(void*, u64, u64, bool) = nullptr;
void* apply_context = nullptr;
void (*queue_fn)(void*, Batch&&) = nullptr;
void* queue_context = nullptr;
std::atomic<u64> b2_applied_ranges{0}, b2_applied_bytes{0}, b2_reapplied_ranges{0}, b2_misses{0}, b2_miss_pages{0}, b2_skipped_pages{0};
std::mutex miss_mutex;
std::unordered_map<std::string, u64> misses_by_owner;
std::atomic<u32> misses_logged{0};
} // namespace


void SetApply(void (*fn)(void*, u64, u64, bool), void* context) {
    apply_context = context;
    apply_fn = fn;
}

void SetQueue(void (*fn)(void*, Batch&&), void* context) {
    queue_context = context;
    queue_fn = fn;
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

void SetCreate(void (*fn)(void*, u64, u64, u64, bool), void* context) {
    create_context = context;
    create_fn = fn;
}

void NoteEngineBufferCreate(u64 id, u64 address, u64 bytes) {
    if (!B9PlaceOn() || B2NativeMode() == 0) {
        return;
    }
    std::scoped_lock lock{pending_mutex};
    pending_deleted.push_back({id, address, bytes, 5u});
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
    if (delete_fn != nullptr) {
        // deletions first: the work of this submission comes after them in the game's order, and no announcement of it touches a deleted object
        for (const auto& d : batch.deleted) {
            if (d[3] == 2 || d[3] == 5) {
                if (create_fn != nullptr) {
                    create_fn(create_context, d[0], d[1], d[2], d[3] == 2);
                }
            } else {
                delete_fn(delete_context, d[0], d[1], d[2], static_cast<u32>(d[3]));
            }
        }
    }
    if (apply_fn == nullptr) {
        return;
    }
    for (const auto& [address, size] : batch.again) {
        apply_fn(apply_context, address, size, false);
    }
    for (const auto& [address, size] : batch.first) {
        apply_fn(apply_context, address, size, true);
        b2_applied_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    b2_applied_ranges.fetch_add(batch.first.size(), std::memory_order_relaxed);
    b2_reapplied_ranges.fetch_add(batch.again.size(), std::memory_order_relaxed);
}

void ApplyPending() {
    if (B2NativeMode() == 0) {
        return;
    }
    Batch batch = TakeBatch();
    if (batch.Empty()) {
        return;
    }
    if (queue_fn != nullptr) {
        queue_fn(queue_context, std::move(batch));
    } else {
        ApplyBatch(batch);
    }
}

bool ConsumeMark(u64 page) {
    const bool announced = MarkOf(page)->exchange(0, std::memory_order_relaxed) == (page | 1);
    if (announced) {
        b2_skipped_pages.fetch_add(1, std::memory_order_relaxed);
    }
    return announced;
}

void NoteMiss(u64 page, u64 bytes) {
    b2_misses.fetch_add(1, std::memory_order_relaxed);
    b2_miss_pages.fetch_add((bytes + 4095) / 4096, std::memory_order_relaxed);
    const std::string owner = NativeDevice::OwnerName(page);
    {
        std::scoped_lock lock{miss_mutex};
        ++misses_by_owner[owner];
    }
    if (misses_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Core, "B2 native: {:#x}+{:#x} written with no announcement ({})", page, bytes, owner);
    }
}

bool HarvestAuditOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_B2_AUDIT");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void NoteHarvested(u64 address, u64 bytes, bool in_upload_ring) {
    for (u64 page = address & ~u64{4095}; page < address + bytes; page += 4096) {
        const bool announced = MarkOf(page)->exchange(0, std::memory_order_relaxed) == (page | 1);
        (in_upload_ring ? harvested_ring : harvested_other)[announced ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        if (!announced && !in_upload_ring) {
            const std::string owner = NativeDevice::OwnerName(page);
            {
                std::scoped_lock lock{owner_mutex};
                ++unannounced_by_owner[owner];
            }
            static std::atomic<u32> none_logged{0};
            if (owner == "none" ? none_logged.fetch_add(1, std::memory_order_relaxed) < 24 : unannounced_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Core, "B2 audit: page {:#x} written with no announcement ({}; {}; usage class {})", page, owner, NativeResources::DescribeAddress(page),
                            NativeResources::UsageClassNameOf(NativeResources::UsageClassOf(page)));
            }
        }
    }
}

namespace {
std::array<std::atomic<u64>, B8Counts> b8_counts{};
std::atomic<u64> b8_samples{0};
u32 B8VerifyEvery() {
    static const u32 n = [] {
        const char* e = std::getenv("TLG_B8_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::max(std::atoi(e), 0)) : 0u;
    }();
    return n;
}
} // namespace





bool B8VerifyOn() noexcept {
    return B8ImageMode() == 2 && B8VerifyEvery() != 0;
}

bool B8VerifySample() noexcept {
    return B8VerifyOn() && b8_samples.fetch_add(1, std::memory_order_relaxed) % B8VerifyEvery() == 0;
}

void NoteB8(B8Count what) noexcept {
    b8_counts[what].fetch_add(1, std::memory_order_relaxed);
}

void NoteB8Unannounced(u64 address, u64 size, bool by_fault) noexcept {
    NoteB8(B8FoundWithout);
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
        LOG_WARNING(Core, "B8: image {:#x}+{:#x} changed by the CPU with no announcement ({}), owner {}", address, size, by_fault ? "write fault" : "verify hash",
                    Core::NativeDevice::OwnerName(address));
    }
}

namespace {
std::array<std::atomic<u64>, 4 * 16> e3_overlaps{};
} // namespace
void NoteE3Overlap(u32 cls, u32 outcome) noexcept {
    e3_overlaps[(cls & 3) * 16 + (outcome & 15)].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json B8Report() {
    const auto c = [](B8Count what) { return b8_counts[what].load(std::memory_order_relaxed); };
    const u64 found = c(B8FoundWithout), later = c(B8AnnouncedLater);
    return {{"mode", B8ImageMode()},
            {"refreshes_by_announcement", c(B8RefreshAnnounced)},
            {"refreshes_by_fault_and_announcement", c(B8RefreshFaultAndAnnounced)},
            {"changes_found_without_announcement", found},
            {"of_those_announced_later", later},
            {"MISSES_never_announced", found > later ? found - later : 0},
            {"announcements_on_images", c(B8AnnouncedOnImage)},
            {"second_applications_kept_for_gpu", c(B8KeptForGpu)},
            {"verify_checked", c(B8VerifyChecked)},
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
            {"e3_find", {{"unnamed", c(E3FindUnnamed)}, {"named", c(E3FindNamed)}, {"named_same_owner", c(E3NamedSameOwner)}, {"named_OTHER_OWNER", c(E3NamedOtherOwner)},
                         {"named_interior", c(E3NamedInterior)}, {"named_no_owner", c(E3NamedNoOwner)}, {"named_unknown_object", c(E3NamedUnknownObject)},
                         {"owner_by_id", c(E3OwnerById)}, {"owner_by_id_missed", c(E3OwnerByIdMissed)}, {"id_decides", E3IdDecides()}}},
            {"note", "B8.1, cumulative: images refreshed by the engine's announcements instead of write faults"}};
}

namespace {
std::array<std::atomic<u64>, B9Counts> b9_counts{};
} // namespace







u32 B9VerifyEvery() noexcept {
    static const u32 n = [] {
        const char* e = std::getenv("TLG_B9_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::max(std::atoi(e), 0)) : 0u;
    }();
    return B9BufferMode() == 2 ? n : 0u;
}

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
            {"e4_no_arena", {{"pseudo_placements", c(E4PseudoPlaced)}, {"pseudo_ring", c(E4PseudoRing)}, {"fault_readback_windows", c(E4ReadbackWindows)}, {"demotes_kept", c(E4DemoteKept)}}},
            {"verify", {{"every_nth_page", B9VerifyEvery()}, {"pages_checked", c(B9VerifyChecked)}, {"differing", c(B9VerifyMiss)},
                        {"of_those_announced_later", c(B9VerifyLate)}, {"MISSES_never_announced", c(B9VerifyMiss) - std::min(c(B9VerifyMiss), c(B9VerifyLate))}}},
            {"note", "B9.1, cumulative: engine buffers uploaded by their announced bytes instead of the page tracker's pages"}};
}

nlohmann::json HarvestAuditReport() {
    nlohmann::json j;
    j["enabled"] = HarvestAuditOn();
    j["upload_ring_pages"] = {{"announced", harvested_ring[0].load()}, {"UNANNOUNCED", harvested_ring[1].load()}};
    j["other_pages"] = {{"announced", harvested_other[0].load()}, {"UNANNOUNCED", harvested_other[1].load()}};
    nlohmann::json by_class = nlohmann::json::object();
    for (u32 c = 0; c < 64; ++c) {
        if (const u64 n = unannounced_by_class[c].load()) {
            by_class[NativeResources::UsageClassNameOf(c)] = n;
        }
    }
    j["unannounced_other_pages_by_memory_class"] = std::move(by_class);
    {
        std::scoped_lock lock{owner_mutex};
        std::vector<std::pair<u64, std::string>> rows;
        for (const auto& [name, n] : unannounced_by_owner) {
            rows.emplace_back(n, name);
        }
        std::sort(rows.rbegin(), rows.rend());
        nlohmann::json owners = nlohmann::json::object();
        for (size_t i = 0; i < rows.size() && i < 24; ++i) {
            owners[rows[i].second] = rows[i].first;
        }
        j["unannounced_other_pages_by_owner"] = std::move(owners);
    }
    j["note"] = "B2 gate: pages the write watch found written, against the engine's announced updates since their last harvest (cumulative)";
    j["native"] = {{"mode", B2NativeMode()}, {"announcements_applied", b2_applied_ranges.load()}, {"bytes_applied", b2_applied_bytes.load()},
                   {"announcements_applied_again", b2_reapplied_ranges.load()}, {"harvested_pages_announced_not_replayed", b2_skipped_pages.load()},
                   {"MISSES", b2_misses.load()}, {"miss_pages", b2_miss_pages.load()}};
    {
        std::scoped_lock lock{miss_mutex};
        nlohmann::json owners = nlohmann::json::object();
        for (const auto& [name, n] : misses_by_owner) {
            owners[name] = n;
        }
        j["native"]["misses_by_owner"] = std::move(owners);
    }
    return j;
}

void NoteWriterRangeOf(u32 writer, u64 address, u64 size, u64 thread) {
    if (address == 0 || size == 0) {
        return;
    }
    if (HarvestAuditOn() || B2NativeMode() != 0) {
        MarkAnnounced(address, size);
    }
    if (B2NativeMode() != 0) {
        std::scoped_lock lock{pending_mutex};
        pending_ranges.emplace_back(address, size);
    }
    NativeResources::NoteAnnouncedWrite(address, size); // C1: resident buffers on those pages are not synchronized any more
    if (AuditFaults()) {
        NoteAnnouncerThread(writer);
    }
    NativeUploads::Announce(writer, address, size, thread == render_thread.load(std::memory_order_relaxed), NativeDraw::NextSeq());
    if (supersede_fn != nullptr && SupersedeEnabled() && (writer == 5 || writer == 6 || (writer == 0 && thread == render_thread.load(std::memory_order_relaxed)))) {
        supersede_fn(address, size);
        superseded_ranges.fetch_add(1, std::memory_order_relaxed);
        superseded_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    if (!AuditFaults()) {
        return; // the windows below only serve the fault audit
    }
    const u64 now = NowNs();
    std::scoped_lock lock{mutex};
    windows[window_count++ % WindowCount] = {address, address + size, 0, now, thread, 0, 0, false, writer};
    // faults of this thread inside the range that came first (the writer announces after it wrote) are covered by it
    const u64 count = std::min<u64>(pending_count, pending.size());
    for (u64 i = 0; i < count; ++i) {
        PendingFault& p = pending[(pending_count - count + i) % pending.size()];
        if (p.address == 0 || p.thread != thread || now - p.time_ns > NowWindowNs || p.address < address || p.address + p.size > address + size) {
            continue;
        }
        faults.uncovered.fetch_sub(1, std::memory_order_relaxed);
        faults.covered_now.fetch_add(1, std::memory_order_relaxed);
        faults.covered_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
        faults.uncovered_by_class[p.usage_class].fetch_sub(1, std::memory_order_relaxed);
        faults.covered_by_class[p.usage_class].fetch_add(1, std::memory_order_relaxed);
        faults.uncovered_by_registry[p.registry & 3].fetch_sub(1, std::memory_order_relaxed);
        if (const auto site = uncovered_sites.find(SiteKey{p.rip, p.st0, p.st1}); site != uncovered_sites.end() && site->second.count > 0) {
            --site->second.count;
            if ((p.registry == 1 || p.registry == 2) && site->second.in_registered > 0) {
                --site->second.in_registered;
            }
        }
        p.address = 0;
    }
}

void NoteFault(u64 address, u64 size) {
    faults.total.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    {
        const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
        const u64 rip = GuestHooks::FaultRip();
        if (rip >= exe && rip < exe + 0x10000000) {
            faults.own_host_writes.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
#endif
    const u32 usage_class = NativeResources::UsageClassOf(address) & 15;
    const u32 registry_class = NativeRegistry::BufferContaining(address, size);
    const u64 now = NowNs();
    const u64 thread = ThreadKey();
    enum class Kind { None, Now, Earlier, Other } kind = Kind::None;
    u32 writer = 0;
    {
        std::scoped_lock lock{mutex};
        const u64 count = std::min<u64>(window_count, WindowCount);
        for (u64 i = 0; i < count && kind != Kind::Now; ++i) {
            const Window& w = windows[(window_count - 1 - i) % WindowCount];
            if (address < w.begin || address + size > w.end) {
                continue;
            }
            if (w.thread != thread) {
                kind = kind == Kind::None ? Kind::Other : kind;
            } else if (now - w.time_ns <= NowWindowNs) {
                kind = Kind::Now;
                writer = w.writer;
            } else if (kind != Kind::Now) {
                kind = Kind::Earlier;
            }
        }
        if (kind == Kind::None && GuestHooks::FaultRip() == 0) {
            u32 bucket = size <= 1 ? 0 : size <= 8 ? 1 : size <= 64 ? 2 : size <= 4096 ? 3 : size <= 65536 ? 4 : size <= (1u << 20) ? 5 : 6;
            no_instruction_sizes[bucket].fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
            void* frames[12] = {};
            const USHORT n = CaptureStackBackTrace(1, 12, frames, nullptr);
            const u64 key = n > 2 ? reinterpret_cast<u64>(frames[2]) : 0;
            NoInstruction& entry = no_instruction[key];
            if (entry.count++ == 0) {
                entry.frames = n;
                for (USHORT i = 0; i < n; ++i) {
                    entry.stack[i] = reinterpret_cast<u64>(frames[i]);
                }
                entry.first_address = address;
            }
            entry.bytes += size;
#endif
        }
        if (kind == Kind::Now && writer == 0) {
            Site& site = map_sites[{GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)}];
            if (site.count++ == 0) {
                site.sample_address = address;
                site.usage_class = usage_class;
            }
        }
        if (kind == Kind::None) {
            pending[pending_count++ % pending.size()] = {address, size, now, thread, usage_class, GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1), registry_class};
            Site& site = uncovered_sites[{GuestHooks::FaultRip(), GuestHooks::FaultStack(0), GuestHooks::FaultStack(1)}];
            if (site.count++ == 0) {
                site.sample_address = address;
                site.usage_class = usage_class;
                site.registry = registry_class;
#ifdef _WIN32
                if (GuestHooks::FaultStack(0) == 0) {
                    void* frames[16] = {};
                    const USHORT n = CaptureStackBackTrace(1, 16, frames, nullptr);
                    const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
                    for (USHORT i = 0; i < n && site.host_frames < 8; ++i) {
                        const u64 at = reinterpret_cast<u64>(frames[i]);
                        if (at >= exe && at < exe + 0x10000000) {
                            site.host_stack[site.host_frames++] = at - exe;
                        }
                    }
                }
#endif
            }
            if (registry_class == 1 || registry_class == 2) {
                ++site.in_registered;
            }
        }
    }
    switch (kind) {
    case Kind::Now:
        faults.covered_now.fetch_add(1, std::memory_order_relaxed);
        faults.covered_by_writer[writer & 7].fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Earlier:
        faults.covered_earlier.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::Other:
        faults.covered_other_thread.fetch_add(1, std::memory_order_relaxed);
        break;
    case Kind::None:
        faults.uncovered.fetch_add(1, std::memory_order_relaxed);
        faults.uncovered_by_registry[registry_class & 3].fetch_add(1, std::memory_order_relaxed);
        break;
    }
    (kind == Kind::None ? faults.uncovered_by_class : faults.covered_by_class)[usage_class].fetch_add(1, std::memory_order_relaxed);
}

namespace {
struct DirtierKey {
    u64 rip, a, b, c;
    bool operator<(const DirtierKey& o) const {
        return std::tie(rip, a, b, c) < std::tie(o.rip, o.a, o.b, o.c);
    }
};
struct Dirtier {
    u64 count{}, bytes{}, sample{};
};
std::mutex dirtier_mutex;
std::map<DirtierKey, Dirtier> dirtiers;
} // namespace

void NoteGpuWrittenDirtied(u64 address, u64 size) {
    DirtierKey key{GuestHooks::FaultRip(), 0, 0, 0};
    if (key.rip == 0) {
#ifdef _WIN32
        void* frames[8] = {};
        const USHORT n = CaptureStackBackTrace(2, 8, frames, nullptr);
        const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
        u64 rva[3] = {};
        for (USHORT i = 0, found = 0; i < n && found < 3; ++i) {
            const u64 at = reinterpret_cast<u64>(frames[i]);
            if (at >= exe && at < exe + 0x10000000) {
                rva[found++] = at - exe;
            }
        }
        key.a = rva[0];
        key.b = rva[1];
        key.c = rva[2];
#endif
    } else {
        key.a = GuestHooks::FaultStack(0);
    }
    std::scoped_lock lock{dirtier_mutex};
    Dirtier& d = dirtiers[key];
    if (d.count++ == 0) {
        d.sample = address;
    }
    d.bytes += size;
}

u32 WhoWrote(u64 address, u64 size) {
    const u64 now = NowNs();
    const u64 render = render_thread.load(std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    const u64 count = std::min<u64>(window_count, WindowCount);
    for (u64 i = 0; i < count; ++i) {
        const Window& w = windows[(window_count - 1 - i) % WindowCount];
        if (address >= w.begin && address + size <= w.end) {
            const u64 age_ms = std::min<u64>((now - w.time_ns) / 1'000'000, 60000);
            return (w.writer & 0xff) | (w.thread == render ? 0x100u : 0u) | static_cast<u32>(age_ms << 16);
        }
    }
    return ~0u;
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
    {
        constexpr const char* names[8] = {"map", "skinning", "light_constants", "command_stream_copy", "begin_recording", "buffer_creation", "texture_initialization", "instance_group_upload"};
        auto& t = j["announcements_by_thread"] = nlohmann::json::object();
        t["render_thread_key"] = render_thread.load();
        for (u32 w = 0; w < 8; ++w) {
            auto& o = t[names[w]] = nlohmann::json::object();
            o["on_the_render_thread"] = announced_on_render_thread[w].load();
            o["on_other_threads"] = announced_elsewhere[w].load();
            for (u32 n = 0; n < 8; ++n) {
                if (const u64 c = other_thread_counts[w][n].load()) {
                    o["other_thread_" + std::to_string(n)] = c;
                }
            }
        }
    }
    auto& f = j["cpu_write_faults"] = nlohmann::json::object();
    const u64 total = faults.total.load();
    f["total"] = total;
    f["inside_a_map_window_of_the_same_thread_within_200ms"] = faults.covered_now.load();
    f["inside_a_window_by_announcer"] = {{"map", faults.covered_by_writer[0].load()}, {"skinning_FUN_01c93730", faults.covered_by_writer[1].load()}, {"light_constants_FUN_01eeb610", faults.covered_by_writer[2].load()},
                                                       {"command_stream_memory_copy", faults.covered_by_writer[3].load()}, {"begin_recording_direct_stores", faults.covered_by_writer[4].load()},
                                                       {"buffer_creation_and_host_file_reads", faults.covered_by_writer[5].load()},
                                                       {"texture_initialization", faults.covered_by_writer[6].load()}, {"instance_group_upload_FUN_01ee4e50", faults.covered_by_writer[7].load()}};
    f["inside_an_older_map_window_of_the_same_thread"] = faults.covered_earlier.load();
    f["inside_a_map_window_of_another_thread"] = faults.covered_other_thread.load();
    f["in_no_map_window"] = faults.uncovered.load();
    f["unannounced_by_registered_buffer"] = {{"outside_every_registered_buffer", faults.uncovered_by_registry[0].load()}, {"inside_a_registered_buffer", faults.uncovered_by_registry[1].load()},
                                             {"partly_inside_a_registered_buffer", faults.uncovered_by_registry[2].load()}, {"registry_off", faults.uncovered_by_registry[3].load()}};
    f["written_by_our_own_native_code_tags_in_command_buffers"] = faults.own_host_writes.load();
    if (total != 0) {
        f["percent_announced_by_a_map_on_the_same_thread_within_200ms"] = 100.0 * static_cast<double>(faults.covered_now.load()) / static_cast<double>(total);
    }
    for (u32 i = 0; i < 16; ++i) {
        const u64 covered = faults.covered_by_class[i].load();
        const u64 uncovered = faults.uncovered_by_class[i].load();
        if (covered + uncovered != 0) {
            const std::string name = NativeResources::UsageClassNameOf(i);
            f["by_memory_class"][name] = {{"in_a_map_window", covered}, {"in_no_map_window", uncovered}};
        }
    }
    std::scoped_lock lock{mutex};
    {
#ifdef _WIN32
        const u64 base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
#else
        const u64 base = 0;
#endif
        auto& n = f["faults_without_an_instruction"] = nlohmann::json::object();
        constexpr const char* names[7] = {"1_byte", "up_to_8", "up_to_64", "up_to_4K", "up_to_64K", "up_to_1M", "more"};
        for (u32 i = 0; i < 7; ++i) {
            n["by_size"][names[i]] = no_instruction_sizes[i].load();
        }
        auto& list = n["by_host_caller"] = nlohmann::json::array();
        for (const auto& [key, e] : no_instruction) {
            nlohmann::json stack = nlohmann::json::array();
            for (u32 i = 0; i < e.frames; ++i) {
                stack.push_back(e.stack[i] - base);
            }
            list.push_back({{"faults", e.count}, {"bytes", e.bytes}, {"first_address", e.first_address}, {"exe_rva_stack", stack}});
        }
    }
    {
        std::scoped_lock dlock{dirtier_mutex};
        std::vector<std::pair<DirtierKey, Dirtier>> list(dirtiers.begin(), dirtiers.end());
        std::ranges::sort(list, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
        auto& out = j["who_dirtied_gpu_written_bytes"] = nlohmann::json::array();
        for (size_t i = 0; i < list.size() && i < 24; ++i) {
            out.push_back({{"guest_rip", list[i].first.rip}, {"stack_or_host_rva_0", list[i].first.a}, {"host_rva_1", list[i].first.b}, {"host_rva_2", list[i].first.c},
                           {"invalidations", list[i].second.count}, {"bytes", list[i].second.bytes}, {"sample_address", list[i].second.sample}});
        }
    }
    std::vector<std::pair<SiteKey, Site>> sites(uncovered_sites.begin(), uncovered_sites.end());
    std::ranges::sort(sites, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    {
        std::vector<std::pair<SiteKey, Site>> mapped(map_sites.begin(), map_sites.end());
        std::ranges::sort(mapped, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
        auto& out = f["writers_into_map_windows"] = nlohmann::json::array();
        for (size_t i = 0; i < mapped.size() && i < 32; ++i) {
            out.push_back({{"rip", mapped[i].first.rip}, {"stack0", mapped[i].first.st0}, {"stack1", mapped[i].first.st1}, {"faults", mapped[i].second.count},
                           {"sample_address", mapped[i].second.sample_address}});
        }
    }
    {
        std::vector<std::pair<SiteKey, Site>> relevant;
        for (const auto& s : sites) {
            if (s.second.in_registered > 0) {
                relevant.push_back(s);
            }
        }
        std::ranges::sort(relevant, [](const auto& a, const auto& b) { return a.second.in_registered > b.second.in_registered; });
        auto& out = f["unannounced_writes_into_registered_buffers"] = nlohmann::json::array();
        for (size_t i = 0; i < relevant.size() && i < 32; ++i) {
            out.push_back({{"rip", relevant[i].first.rip}, {"stack0", relevant[i].first.st0}, {"stack1", relevant[i].first.st1}, {"faults_inside_registered_buffers", relevant[i].second.in_registered},
                           {"faults", relevant[i].second.count}, {"sample_address", relevant[i].second.sample_address},
                           {"host_exe_rva_stack", std::vector<u64>(relevant[i].second.host_stack.begin(), relevant[i].second.host_stack.begin() + relevant[i].second.host_frames)}});
        }
    }
    auto& list = f["unannounced_writes_by_faulting_instruction"] = nlohmann::json::array();
    for (size_t i = 0; i < sites.size() && i < 24; ++i) {
#ifdef _WIN32
        const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
        const u64 rip_rva = sites[i].first.rip >= exe && sites[i].first.rip < exe + 0x10000000 ? sites[i].first.rip - exe : 0;
#else
        const u64 rip_rva = 0;
#endif
        list.push_back({{"rip", sites[i].first.rip}, {"host_exe_rva", rip_rva}, {"stack0", sites[i].first.st0}, {"stack1", sites[i].first.st1}, {"faults", sites[i].second.count}, {"sample_address", sites[i].second.sample_address},
                        {"memory_class", NativeResources::UsageClassNameOf(sites[i].second.usage_class)}, {"sample_in_registered_buffer", sites[i].second.registry}});
    }
    return j;
}

} // namespace Core::NativeUpdates
