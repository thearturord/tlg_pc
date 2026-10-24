// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <map>
#include <utility>
#include <mutex>
#include <nlohmann/json.hpp>
#include "core/native_uploads.h"

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
    // itself and 6 (texture initialization) is image memory.
    return writer == 0 || writer == 1 || writer == 2 || writer == 4 || writer == 5;
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
