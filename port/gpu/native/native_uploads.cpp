// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include "native/native_resources.h"
#include <deque>
#include <map>
#include <utility>
#include <mutex>
#include <nlohmann/json.hpp>
#include "native/native_uploads.h"

namespace Core::NativeUploads {

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_NATIVE_UPLOADS");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

namespace {

struct Entry {
    u64 begin, size, stamp;
};

std::mutex mutex;
std::vector<Entry> pending; ///< announced on other threads, waiting for the render thread's next BeginRecording
std::deque<Entry> journal;  ///< stamped, in nondecreasing stamp order

std::atomic<u64> announced{}, announced_bytes{}, published{}, taken_batches{}, taken_ranges{}, taken_bytes{}, merged_away{};
std::atomic<u64> applied_ranges{}, applied_bytes{}, presync_uploaded_bytes{}, on_demand_uploaded_bytes{}, journal_peak{};
thread_local bool presync_active = false;
std::mutex block_mutex;
std::map<u64, u64> demand_blocks, presync_blocks; ///< bytes copied per 256 KB block of guest memory, on demand at a draw / by the pre-synchronization
std::atomic<u64> ring_upload_bytes{}, ring_upload_count{}, ring_skipped_bytes{}, ring_upload_presync_bytes{};

constexpr bool Kept(u32 writer) {
    // 0 Map / Unmap, 1 skinning, 2 light constants, 4 BeginRecording, 5 buffer creation: memory of buffers. 3 (command stream copies) are ordered by the stream
    // itself and 6 (texture initialization) is image memory. 7 (instance group data, N6.4) is the constant data of instanced batches.
    return writer == 0 || writer == 1 || writer == 2 || writer == 4 || writer == 5 || writer == 7;
}

} // namespace

void Announce(u32 writer, u64 begin, u64 size, bool on_render_thread, u64 stamp_now) {
    if (!Enabled() || !Kept(writer) || size == 0) {
        return;
    }
    announced.fetch_add(1, std::memory_order_relaxed);
    announced_bytes.fetch_add(size, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (on_render_thread) {
        journal.push_back({begin, size, stamp_now});
        published.fetch_add(1, std::memory_order_relaxed);
        journal_peak.store(std::max<u64>(journal_peak.load(std::memory_order_relaxed), journal.size()), std::memory_order_relaxed);
    } else {
        pending.push_back({begin, size, 0});
    }
}

void PublishPending(u64 stamp_now) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    for (Entry& e : pending) {
        e.stamp = stamp_now;
        journal.push_back(e);
    }
    published.fetch_add(pending.size(), std::memory_order_relaxed);
    pending.clear();
    journal_peak.store(std::max<u64>(journal_peak.load(std::memory_order_relaxed), journal.size()), std::memory_order_relaxed);
}

bool TakeDue(u64 seq, std::vector<Range>& out) {
    std::vector<Entry> due;
    {
        std::scoped_lock lock{mutex};
        while (!journal.empty() && journal.front().stamp <= seq) {
            due.push_back(journal.front());
            journal.pop_front();
        }
    }
    if (due.empty()) {
        return false;
    }
    std::ranges::sort(due, [](const Entry& a, const Entry& b) { return a.begin < b.begin; });
    constexpr u64 Gap = 4096; // ranges closer than a page are one
    out.clear();
    for (const Entry& e : due) {
        if (!out.empty() && e.begin <= out.back().begin + out.back().size + Gap) {
            out.back().size = std::max(out.back().begin + out.back().size, e.begin + e.size) - out.back().begin;
            merged_away.fetch_add(1, std::memory_order_relaxed);
        } else {
            out.push_back({e.begin, e.size});
        }
    }
    taken_batches.fetch_add(1, std::memory_order_relaxed);
    taken_ranges.fetch_add(out.size(), std::memory_order_relaxed);
    u64 bytes = 0;
    for (const Range& r : out) {
        bytes += r.size;
    }
    taken_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

void NoteApplied(u64 ranges, u64 bytes, u64 uploaded_bytes) {
    applied_ranges.fetch_add(ranges, std::memory_order_relaxed);
    applied_bytes.fetch_add(bytes, std::memory_order_relaxed);
    (void)uploaded_bytes;
}

void SetPreSyncActive(bool active) {
    presync_active = active;
}

void NoteRingUpload(u64 bytes) {
    (presync_active ? ring_upload_presync_bytes : ring_upload_bytes).fetch_add(bytes, std::memory_order_relaxed);
    ring_upload_count.fetch_add(1, std::memory_order_relaxed);
}

void NoteUploadedRange(u64 address, u64 bytes) {
    std::scoped_lock lock{block_mutex};
    auto& table = presync_active ? presync_blocks : demand_blocks;
    u64 end = address + bytes;
    for (u64 at = address; at < end;) {
        const u64 block = at >> 18;
        const u64 next = std::min<u64>(end, (block + 1) << 18);
        table[block] += next - at;
        at = next;
    }
}

void NoteRingSkipped(u64 bytes) {
    ring_skipped_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteUploaded(u64 bytes) {
    if (bytes == 0) {
        return;
    }
    (presync_active ? presync_uploaded_bytes : on_demand_uploaded_bytes).fetch_add(bytes, std::memory_order_relaxed);
}

// ---- snapshots ---------------------------------------------------------------------------------------------------------------------
namespace {
constexpr u64 SnapshotBytes = 64ull << 20;         // ring of host memory
constexpr u32 SnapshotEntries = 1u << 16;          // entries, circular
constexpr u64 SnapshotMaxRange = 64 << 10;
constexpr u32 PageSlots = 1u << 16;                // page (4 KB) -> the newest entries that touch it
constexpr u32 PerPage = 4;

struct SnapEntry {
    u64 id{};     // 1 + index of the entry in the order taken; 0 = empty
    u64 begin{}, size{}, stamp{}, offset{}; // offset: absolute position in the ring (monotonic)
};

struct SnapStore {
    std::mutex mutex;
    std::unique_ptr<u8[]> bytes = std::make_unique<u8[]>(SnapshotBytes);
    std::unique_ptr<SnapEntry[]> entries = std::make_unique<SnapEntry[]>(SnapshotEntries);
    std::unique_ptr<std::array<u64, PerPage>[]> pages = std::make_unique<std::array<u64, PerPage>[]>(PageSlots); // entry ids, newest first
    u64 next_id = 1;
    u64 head = 0; // absolute bytes written
};
SnapStore& Snap() {
    static SnapStore store;
    return store;
}
std::atomic<u64> ring_snapshot_counters[8]{};
std::atomic<u64> snap_taken{}, snap_taken_bytes{}, snap_too_big{}, snap_found{}, snap_partial{}, snap_none{}, snap_lapped{};
} // namespace

bool SnapshotsEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_SNAPSHOTS");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

void SnapshotTake(u64 begin, u64 size, u64 stamp) {
    if (!SnapshotsEnabled() || size == 0) {
        return;
    }
    if (size > SnapshotMaxRange) {
        snap_too_big.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto& s = Snap();
    std::scoped_lock lock{s.mutex};
    u64 offset = s.head;
    if ((offset % SnapshotBytes) + size > SnapshotBytes) {
        offset += SnapshotBytes - (offset % SnapshotBytes); // never straddle the end of the ring
    }
    s.head = offset + ((size + 15) & ~u64{15});
    std::memcpy(s.bytes.get() + (offset % SnapshotBytes), reinterpret_cast<const void*>(begin), size);
    const u64 id = s.next_id++;
    s.entries[id % SnapshotEntries] = {id, begin, size, stamp, offset};
    for (u64 page = begin >> 12; page <= (begin + size - 1) >> 12; ++page) {
        auto& slot = s.pages[page % PageSlots];
        for (u32 i = PerPage - 1; i > 0; --i) {
            slot[i] = slot[i - 1];
        }
        slot[0] = id;
    }
    snap_taken.fetch_add(1, std::memory_order_relaxed);
    snap_taken_bytes.fetch_add(size, std::memory_order_relaxed);
}

namespace {
std::atomic<u64> diff_share[5]{}; ///< by how much of the range differs: < 10 %, < 50 %, < 100 %, all, and ranges of 16 bytes or less
std::atomic<u64> diff_by_who[9]{};
std::mutex diff_mutex;
std::vector<std::string> diff_samples;
} // namespace

void NoteSnapshotDiff(u64 address, u64 size, u64 first_offset, u64 diff_bytes, u32 snapshot_word, u32 guest_word, u32 who) {
    const double share = size ? static_cast<double>(diff_bytes) / static_cast<double>(size) : 1.0;
    diff_share[size <= 16 ? 4 : share < 0.1 ? 0 : share < 0.5 ? 1 : share < 1.0 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
    diff_by_who[who < 8 ? who : 8].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{diff_mutex};
    if (diff_samples.size() < 12) {
        diff_samples.push_back(fmt::format("{:#x}+{:#x}: {} of {} bytes differ, first at +{:#x} snapshot {:#010x} guest {:#010x}, last announcer {}", address, size, diff_bytes, size, first_offset, snapshot_word,
                                           guest_word, who < 8 ? std::to_string(who) : "none"));
    }
}

void NoteSnapshotRing(u32 what) {
    ring_snapshot_counters[what & 7].fetch_add(1, std::memory_order_relaxed);
}

SnapshotResult SnapshotRead(u64 begin, u64 size, u64 seq, void* data) {
    auto& s = Snap();
    std::scoped_lock lock{s.mutex};
    const SnapEntry* best = nullptr;
    for (u64 page = begin >> 12; page <= (begin + size - 1) >> 12; ++page) {
        for (const u64 id : s.pages[page % PageSlots]) {
            if (id == 0) {
                continue;
            }
            const SnapEntry& e = s.entries[id % SnapshotEntries];
            if (e.id != id || e.stamp > seq || e.begin >= begin + size || begin >= e.begin + e.size) {
                continue; // overwritten entry, newer than the record, or not touching the range
            }
            if (s.head - e.offset > SnapshotBytes) {
                snap_lapped.fetch_add(1, std::memory_order_relaxed);
                continue; // its bytes were overwritten by a later lap of the ring
            }
            if (best == nullptr || e.id > best->id) {
                best = &e;
            }
        }
    }
    if (best == nullptr) {
        snap_none.fetch_add(1, std::memory_order_relaxed);
        return SnapshotResult::None;
    }
    if (best->begin > begin || best->begin + best->size < begin + size) {
        snap_partial.fetch_add(1, std::memory_order_relaxed);
        return SnapshotResult::Partial;
    }
    std::memcpy(data, s.bytes.get() + (best->offset % SnapshotBytes) + (begin - best->begin), size);
    snap_found.fetch_add(1, std::memory_order_relaxed);
    return SnapshotResult::Found;
}

// ---- on-demand upload audit ----------------------------------------------------------------------------------------------------------
namespace {
thread_local u32 current_caller = 0;
thread_local u32 current_detail = 0;
thread_local u32 current_who = ~0u;
std::array<std::atomic<u64>, 9> who_by_writer{};          ///< uploads on GPU-written bytes by the announcer that wrote them last (index 8: none found)
std::array<std::atomic<u64>, 9> who_on_render_thread{};   ///< of those, announced on the render thread
std::array<std::atomic<u64>, 6> who_age_ms{};             ///< age of that announcement: <1, <4, <16, <64, <512, more
std::array<std::atomic<u64>, 256> detail_counts{}; ///< in-pass uploads of draw buffers by detail bits
struct OnDemandBucket {
    std::atomic<u64> requests{}, uploaded{};
};
constexpr const char* CallerNames[5] = {"other", "draw_buffer", "dispatch_buffer", "vertex_buffer", "index_buffer"};
constexpr const char* SizeNames[7] = {"up_to_256", "up_to_4K", "up_to_16K", "up_to_64K", "up_to_1M", "up_to_16M", "more"};
std::array<std::array<OnDemandBucket, 7>, 5> demand_by_caller_size_in_pass;   // [caller][request size]: uploads while a render pass was open
std::array<OnDemandBucket, 5> demand_by_caller_outside_pass;
std::array<OnDemandBucket, 5> demand_immutable_by_caller;
std::mutex class_mutex;
std::map<std::string, std::pair<u64, u64>> demand_in_pass_by_class; // memory class -> {requests, uploaded}
} // namespace

bool OnDemandAudit() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_UPLOAD_AUDIT");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

void SetCaller(u32 caller) {
    current_caller = caller;
}

u32 Caller() {
    return current_caller;
}

void SetDetail(u32 detail) {
    current_detail = detail;
}

void SetWriter(u32 who) {
    current_who = who;
}

void NoteOnDemand(bool pass_open, u64 request_size, u64 uploaded_bytes, u64 address, bool immutable_whole) {
    const u32 caller = current_caller < 5 ? current_caller : 0;
    if (immutable_whole) {
        demand_immutable_by_caller[caller].requests.fetch_add(1, std::memory_order_relaxed);
        demand_immutable_by_caller[caller].uploaded.fetch_add(uploaded_bytes, std::memory_order_relaxed);
        return;
    }
    if (!pass_open) {
        demand_by_caller_outside_pass[caller].requests.fetch_add(1, std::memory_order_relaxed);
        demand_by_caller_outside_pass[caller].uploaded.fetch_add(uploaded_bytes, std::memory_order_relaxed);
        return;
    }
    const u32 bucket = request_size <= 256 ? 0 : request_size <= 4096 ? 1 : request_size <= 16384 ? 2 : request_size <= 65536 ? 3 : request_size <= (1u << 20) ? 4 : request_size <= (16u << 20) ? 5 : 6;
    if (caller == 1) {
        detail_counts[current_detail & 255].fetch_add(1, std::memory_order_relaxed);
        if ((current_detail & 64) != 0) { // bytes the GPU wrote: who wrote them from the CPU side?
            const u32 who = current_who;
            const u32 writer = who == ~0u ? 8 : (who & 0xff) % 8;
            who_by_writer[writer].fetch_add(1, std::memory_order_relaxed);
            if (who != ~0u) {
                if ((who & 0x100) != 0) {
                    who_on_render_thread[writer].fetch_add(1, std::memory_order_relaxed);
                }
                const u32 age = who >> 16;
                who_age_ms[age < 1 ? 0 : age < 4 ? 1 : age < 16 ? 2 : age < 64 ? 3 : age < 512 ? 4 : 5].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    demand_by_caller_size_in_pass[caller][bucket].requests.fetch_add(1, std::memory_order_relaxed);
    demand_by_caller_size_in_pass[caller][bucket].uploaded.fetch_add(uploaded_bytes, std::memory_order_relaxed);
    const std::string cls = NativeResources::UsageClassNameOf(NativeResources::UsageClassOf(address));
    std::scoped_lock lock{class_mutex};
    auto& entry = demand_in_pass_by_class[cls];
    entry.first += 1;
    entry.second += uploaded_bytes;
}

namespace {
std::atomic<u64> geometry_counters[40]{};
}
bool RingGeometry() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_GEOMETRY");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

u64 RingGeometryCap() {
    static const u64 cap = [] {
        const char* e = std::getenv("TLG_RING_GEOMETRY_CAP");
        return e != nullptr && *e != 0 ? static_cast<u64>(std::strtoull(e, nullptr, 10)) : u64{1} << 20;
    }();
    return cap;
}

void NoteGeometry(u32 what) {
    geometry_counters[what % 40].fetch_add(1, std::memory_order_relaxed);
}

bool RingFormatted() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_FORMATTED");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

bool RingPrecise() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RING_PRECISE");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

namespace {
std::atomic<u64> ring_precise_counters[8]{};
}
void NoteRingPrecise(u32 what) {
    ring_precise_counters[what & 7].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = Enabled();
    j["announced_ranges"] = announced.load();
    j["announced_bytes"] = announced_bytes.load();
    j["published_ranges"] = published.load();
    j["batches_queued_before_a_record"] = taken_batches.load();
    j["ranges_after_merging"] = taken_ranges.load();
    j["ranges_merged_into_a_neighbour"] = merged_away.load();
    j["bytes_after_merging"] = taken_bytes.load();
    j["ranges_applied_on_the_raster_thread"] = applied_ranges.load();
    j["bytes_applied"] = applied_bytes.load();
    j["uploaded_by_the_pre_synchronization_bytes"] = presync_uploaded_bytes.load();
    j["uploaded_on_demand_bytes_at_draw_time"] = on_demand_uploaded_bytes.load();
    j["snapshots"] = {{"enabled", SnapshotsEnabled()},
                      {"taken", snap_taken.load()},
                      {"taken_bytes", snap_taken_bytes.load()},
                      {"writes_over_64KB_not_kept", snap_too_big.load()},
                      {"reads_answered_from_a_snapshot", snap_found.load()},
                      {"reads_the_newest_write_only_partly_covers", snap_partial.load()},
                      {"reads_no_snapshot_touches", snap_none.load()},
                      {"entries_found_overwritten_by_a_later_lap", snap_lapped.load()},
                      {"ring_bindings_carrying_a_snapshot", ring_snapshot_counters[0].load()},
                      {"ring_bindings_refused_partly_covered", ring_snapshot_counters[1].load()},
                      {"snapshot_ring_bindings_compared_with_guest_memory", ring_snapshot_counters[2].load()},
                      {"snapshot_differs_from_guest_memory_at_bind_time", ring_snapshot_counters[3].load()},
                      {"snapshot_differed_from_guest_memory_when_planned_guest_bytes_used_instead", ring_snapshot_counters[4].load()}};
    j["snapshots"]["differing_share_of_range"] = {{"under_10_percent", diff_share[0].load()}, {"under_50_percent", diff_share[1].load()}, {"under_100_percent", diff_share[2].load()},
                                                  {"all", diff_share[3].load()}, {"range_of_16_bytes_or_less", diff_share[4].load()}};
    {
        auto& w = j["snapshots"]["differing_by_last_announcer"] = nlohmann::json::object();
        for (u32 i = 0; i < 9; ++i) {
            w[i < 8 ? std::to_string(i) : "none"] = diff_by_who[i].load();
        }
        std::scoped_lock lock{diff_mutex};
        j["snapshots"]["differing_samples"] = diff_samples;
    }
    {
        auto& a = j["on_demand_upload_audit"] = nlohmann::json::object();
        a["enabled"] = OnDemandAudit();
        auto& in_pass = a["uploads_while_a_render_pass_was_open_which_ended_it"] = nlohmann::json::object();
        u64 total_requests = 0, total_bytes = 0;
        for (u32 c = 0; c < 5; ++c) {
            auto& o = in_pass[CallerNames[c]] = nlohmann::json::object();
            u64 requests = 0, bytes = 0;
            for (u32 b = 0; b < 7; ++b) {
                const u64 r = demand_by_caller_size_in_pass[c][b].requests.load(), u = demand_by_caller_size_in_pass[c][b].uploaded.load();
                if (r != 0) {
                    o["by_request_size"][SizeNames[b]] = {{"uploads", r}, {"bytes", u}};
                }
                requests += r;
                bytes += u;
            }
            o["uploads"] = requests;
            o["bytes"] = bytes;
            total_requests += requests;
            total_bytes += bytes;
        }
        {
            auto& d = in_pass["draw_buffer_uploads_by_what_the_buffer_is"] = nlohmann::json::object();
            for (u32 i = 0; i < 256; ++i) {
                if (const u64 n = detail_counts[i].load()) {
                    std::string name;
                    name += (i & 1) ? "written " : "read_only ";
                    name += (i & 2) ? "formatted " : "";
                    name += (i & 4) ? "over_16K " : "up_to_16K ";
                    name += (i & 16) ? "draw_has_ring_plan " : "no_ring_plan ";
                    name += (i & 32) ? "page_is_gpu_modified " : "page_not_gpu_modified ";
                    name += (i & 64) ? "BYTES_GPU_WRITTEN " : "bytes_not_gpu_written ";
                    name += (i & 128) ? "image_overlaps " : "no_image ";
                    name += (i & 8) ? "ring_binding_existed" : "";
                    d[name] = n;
                }
            }
        }
        {
            auto& w = in_pass["draw_buffer_uploads_on_gpu_written_bytes_by_the_cpu_writer"] = nlohmann::json::object();
            constexpr const char* names[9] = {"map_unmap", "skinning", "light_constants", "command_stream_copy", "begin_recording", "buffer_creation", "texture_initialization", "instance_group_upload", "no_announcement_found"};
            for (u32 i = 0; i < 9; ++i) {
                if (const u64 n = who_by_writer[i].load()) {
                    w[names[i]] = {{"uploads", n}, {"on_the_render_thread", who_on_render_thread[i].load()}};
                }
            }
            constexpr const char* ages[6] = {"under_1ms", "under_4ms", "under_16ms", "under_64ms", "under_512ms", "older"};
            for (u32 i = 0; i < 6; ++i) {
                if (const u64 n = who_age_ms[i].load()) {
                    w["announcement_age"][ages[i]] = n;
                }
            }
        }
        in_pass["total_uploads"] = total_requests;
        in_pass["total_bytes"] = total_bytes;
        auto& outside = a["uploads_with_no_render_pass_open"] = nlohmann::json::object();
        auto& immutable = a["whole_immutable_buffer_synchronizations"] = nlohmann::json::object();
        for (u32 c = 0; c < 5; ++c) {
            outside[CallerNames[c]] = {{"uploads", demand_by_caller_outside_pass[c].requests.load()}, {"bytes", demand_by_caller_outside_pass[c].uploaded.load()}};
            immutable[CallerNames[c]] = {{"uploads", demand_immutable_by_caller[c].requests.load()}, {"bytes", demand_immutable_by_caller[c].uploaded.load()}};
        }
        std::scoped_lock lock{class_mutex};
        auto& classes = a["uploads_while_a_render_pass_was_open_by_memory_class"] = nlohmann::json::object();
        for (const auto& [name, v] : demand_in_pass_by_class) {
            classes[name] = {{"uploads", v.first}, {"bytes", v.second}};
        }
    }
    j["ring_precise"] = {{"enabled", RingPrecise()},
                         {"ranges_the_page_flag_refuses_that_went_through_the_ring", ring_precise_counters[0].load()},
                         {"ring_bound_ranges_the_gpu_wrote_between_plan_and_bind", ring_precise_counters[1].load()},
                         {"ranges_refused_because_their_bytes_are_gpu_written", ring_precise_counters[2].load()},
                         {"formatted_buffers_bound_from_the_ring", ring_precise_counters[4].load()},
                         {"formatted_ring_copies_ignored_at_bind_an_image_or_metadata_covers_the_range", ring_precise_counters[5].load()},
                         {"formatted_ring_copies_ignored_at_bind_the_gpu_wrote_the_bytes", ring_precise_counters[6].load()}};
    j["ring_geometry"] = {{"enabled", RingGeometry()},
                          {"vertex_ranges_copied_by_the_front", geometry_counters[0].load()},
                          {"index_ranges_copied_by_the_front", geometry_counters[1].load()},
                          {"vertex_ranges_bound_from_the_ring", geometry_counters[2].load()},
                          {"index_ranges_bound_from_the_ring", geometry_counters[3].load()},
                          {"ring_copies_ignored_at_bind_the_gpu_wrote_the_bytes", geometry_counters[4].load()},
                          {"refused_over_the_cap", geometry_counters[5].load()},
                          {"refused_bytes_are_gpu_written", geometry_counters[6].load()},
                          {"refused_immutable_or_pending_write_or_ring_full", geometry_counters[7].load()},
                          {"bound_ranges_compared_with_guest_memory", geometry_counters[8].load()},
                          {"bound_ranges_differing_from_guest_memory", geometry_counters[9].load()},
                          {"index_bind_matched_the_ring_entry", geometry_counters[14].load()},
                          {"index_bind_had_an_entry_that_did_not_match_address_or_size", geometry_counters[15].load()},
                          {"index_bind_had_no_entry_at_all", geometry_counters[29].load()},
                          {"vertex_draws_seen", geometry_counters[10].load()},
                          {"vertex_stopped_no_fetch_shader_or_stage", geometry_counters[11].load()},
                          {"vertex_stopped_bad_table_pointer", geometry_counters[12].load()},
                          {"vertex_merged_ranges_considered", geometry_counters[13].load()},
                          {"vertex_refused_immutable", geometry_counters[16].load()},
                          {"vertex_refused_gpu_written", geometry_counters[17].load()},
                          {"vertex_refused_pending_write", geometry_counters[18].load()},
                          {"vertex_refused_ring_full", geometry_counters[19].load()},
                          {"vertex_refused_over_cap", geometry_counters[20].load()},
                          {"index_refused_immutable", geometry_counters[24].load()},
                          {"index_refused_gpu_written", geometry_counters[25].load()},
                          {"index_refused_pending_write", geometry_counters[26].load()},
                          {"index_refused_ring_full", geometry_counters[27].load()},
                          {"index_refused_over_cap", geometry_counters[28].load()}};
    j["journal_peak_entries"] = journal_peak.load();
    j["frame_ring_uploads_at_draw_time_bytes"] = ring_upload_bytes.load();
    j["frame_ring_uploads_by_the_pre_synchronization_bytes"] = ring_upload_presync_bytes.load();
    j["frame_ring_reads_found_uploaded_in_this_command_buffer_bytes"] = ring_skipped_bytes.load();
    j["frame_ring_upload_operations"] = ring_upload_count.load();
    {
        std::scoped_lock lock{block_mutex};
        for (const auto& [name, table] : {std::pair{"biggest_blocks_uploaded_on_demand", &demand_blocks}, std::pair{"biggest_blocks_uploaded_by_the_pre_synchronization", &presync_blocks}}) {
            std::vector<std::pair<u64, u64>> v(table->begin(), table->end());
            std::ranges::sort(v, [](const auto& a, const auto& b) { return a.second > b.second; });
            auto& list = j[name] = nlohmann::json::array();
            for (size_t i = 0; i < v.size() && i < 12; ++i) {
                list.push_back({{"address", v[i].first << 18}, {"bytes", v[i].second}});
            }
        }
    }
    {
        std::scoped_lock lock{mutex};
        j["pending_now"] = pending.size();
        j["journal_now"] = journal.size();
    }
    return j;
}

} // namespace Core::NativeUploads
