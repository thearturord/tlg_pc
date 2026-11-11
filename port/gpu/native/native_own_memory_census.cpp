// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_own_memory_census.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"

namespace Core::OwnMemoryCensus {

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_OWN_MEMORY_CENSUS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

namespace {
constexpr u32 Writers = 17;

struct Info {
    u64 address{}, size{};
    u32 usage{}, bind{}, misc{};
    u64 binds_read{}, binds_written{}, binds_compute{}, ring_copies{}, noid_binds{}, copy_src{}, copy_dst{}, geometry_reads{}, hook_binds{};
    std::array<u64, Writers> applies{};
    u64 apply_bytes{}, faults{};
    bool alive{true};
};

std::mutex mutex;
std::unordered_map<u64, Info> by_id;
std::map<u64, u64> id_at; ///< begin address -> id (live buffers)
std::array<std::atomic<u64>, BindClassCount> binds{};
std::atomic<u64> applies_no_buffer{}, faults_no_buffer{}, applies_total{}, faults_total{}, resources{}, copies_total{}, copies_src_no_buffer{}, copies_dst_no_buffer{}, geometry_in_buffer{}, geometry_no_buffer{}, hook_in_buffer{};

Info* Containing(u64 address, u64 size) {
    auto it = id_at.upper_bound(address);
    if (it == id_at.begin()) {
        return nullptr;
    }
    --it;
    const auto found = by_id.find(it->second);
    if (found == by_id.end() || !found->second.alive || address + size > found->second.address + found->second.size) {
        return nullptr;
    }
    return &found->second;
}
} // namespace

void NoteResource(u64 id, u64 address, u64 size, u32 usage, u32 bind, u32 misc) {
    if (id == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    auto& info = by_id[id];
    info = Info{};
    info.address = address;
    info.size = size;
    info.usage = usage;
    info.bind = bind;
    info.misc = misc;
    id_at[address] = id;
    resources.fetch_add(1, std::memory_order_relaxed);
}

void NoteResourceGone(u64 address) {
    std::scoped_lock lock{mutex};
    if (const auto it = id_at.find(address); it != id_at.end()) {
        if (const auto found = by_id.find(it->second); found != by_id.end()) {
            found->second.alive = false;
        }
        id_at.erase(it);
    }
}

void NoteBind(u32 bind_class) noexcept {
    if (bind_class < BindClassCount) {
        binds[bind_class].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteBindRing(u64 address, u64 size, bool hook) {
    binds[Ring].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (hook) {
        binds[RingHook].fetch_add(1, std::memory_order_relaxed);
        if (Info* in = Containing(address, size != 0 ? size : 1)) {
            ++in->hook_binds;
            hook_in_buffer.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    Info* info = Containing(address, size);
    if (info == nullptr) {
        binds[RingCopyNoEngineBuffer].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    binds[RingCopyInEngineBuffer].fetch_add(1, std::memory_order_relaxed);
    ++info->ring_copies;
}

void NoteBindNoId(u64 address, u64 size, bool written) {
    if (address == 0) {
        binds[NoIdNoWords].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::scoped_lock lock{mutex};
    Info* info = Containing(address, size != 0 ? size : 1);
    if (info == nullptr) {
        binds[NoIdNoEngineBuffer].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    binds[NoIdInEngineBuffer].fetch_add(1, std::memory_order_relaxed);
    ++info->noid_binds;
    if (written) {
        ++info->binds_written;
    }
}

void NoteGeometryRead(u64 address, u64 size) {
    std::scoped_lock lock{mutex};
    if (Info* info = Containing(address, size != 0 ? size : 1)) {
        ++info->geometry_reads;
        geometry_in_buffer.fetch_add(1, std::memory_order_relaxed);
    } else {
        geometry_no_buffer.fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteCopy(u64 dst, u64 src, u64 size) {
    std::scoped_lock lock{mutex};
    if (Info* s = Containing(src, size)) {
        ++s->copy_src;
    } else {
        copies_src_no_buffer.fetch_add(1, std::memory_order_relaxed);
    }
    if (Info* d = Containing(dst, size)) {
        ++d->copy_dst;
    } else {
        copies_dst_no_buffer.fetch_add(1, std::memory_order_relaxed);
    }
    copies_total.fetch_add(1, std::memory_order_relaxed);
}

void NoteBindId(u64 id, u64 address, u64 size, bool written, bool compute) {
    std::scoped_lock lock{mutex};
    const auto it = by_id.find(id);
    if (it == by_id.end()) {
        binds[IdUnknown].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Info& info = it->second;
    const bool inside = address >= info.address && address + size <= info.address + info.size;
    binds[inside ? IdInside : IdCrossing].fetch_add(1, std::memory_order_relaxed);
    ++(written ? info.binds_written : info.binds_read);
    if (compute) {
        ++info.binds_compute;
    }
}

void NoteApply(u64 address, u64 size, u32 writer) {
    applies_total.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    Info* info = Containing(address, size);
    if (info == nullptr) {
        applies_no_buffer.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ++info->applies[writer < Writers ? writer : Writers - 1];
    info->apply_bytes += size;
}

void NoteFault(u64 address, u64 size) {
    faults_total.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    Info* info = Containing(address, size);
    if (info == nullptr) {
        faults_no_buffer.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ++info->faults;
}

u32 DropDirtyMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_DROP_DIRTY_APPLIES"); // default 2 (real, 1 in 64 audited) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

namespace {
std::array<std::atomic<u64>, DropCount> drops{};
}

u32 OwnImmutableMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_IMMUTABLE"); // default 2 (real, 1 in 64 audited) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnMutableMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_MUTABLE"); // default 2 (real, 1 in 64 audited) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnLargeMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_LARGE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 119); LEGACY(own_large): =0 buffers above the resident cap through ObtainBuffer, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnRangeMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_RANGE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 124); LEGACY(own_range): =0 through ObtainBuffer, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnGeometryMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_GEOMETRY");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 119); LEGACY(own_geometry): =0 index and vertex reads by rows and ObtainBuffer, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnExactEndMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_EXACT_END");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 124); LEGACY(own_exact_end): =0 the rounded end only, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 OwnVertexSplitMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_VERTEX_SPLIT");
        return e != nullptr && (e[0] == '1' || e[0] == '2') ? static_cast<u32>(e[0] - '0') : 0u; // default 0 (note 121)
    }();
    return mode;
}

u32 OwnPlaceMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_PLACE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 124); LEGACY(own_place): =0 small reads copied into the stream ring, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

namespace {
std::array<std::atomic<u64>, PlaceCount> places{};
std::mutex place_mutex;
std::unordered_map<u64, u64> place_last_version; ///< id -> the write version at its last bind
} // namespace

void NotePlace(u64 id, u64 bytes, u64 version, bool zero_copy) noexcept {
    places[PlaceBinds].fetch_add(1, std::memory_order_relaxed);
    if (zero_copy) {
        places[PlaceZeroCopy].fetch_add(1, std::memory_order_relaxed);
    }
    bool changed = true;
    {
        std::scoped_lock lock{place_mutex};
        const auto [it, inserted] = place_last_version.try_emplace(id, version);
        if (inserted) {
            places[PlaceBuffers].fetch_add(1, std::memory_order_relaxed);
            places[PlaceBytes].fetch_add(bytes, std::memory_order_relaxed);
        } else {
            changed = it->second != version;
            it->second = version;
        }
    }
    places[changed ? PlaceDirty : PlaceClean].fetch_add(1, std::memory_order_relaxed);
}

bool DiagOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_F1_DIAG");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

namespace {
std::array<std::atomic<u64>, DiagCount> diags{};
std::array<std::atomic<u32>, DiagCount> diag_logged{};
constexpr const char* DiagNames[DiagCount] = {"outside_starts_below", "outside_starts_past_end", "outside_crossing", "outside_dead_record", "no_record_caller_memory",
                                              "no_record_staging", "no_record_other_refused", "no_record_not_a_buffer", "no_record_unknown_id", "no_own_no_block_meets",
                                              "no_own_blocks_meet_none_whole", "no_id_in_frame_ring", "no_id_inside_one_engine_buffer", "no_id_over_several_engine_buffers",
                                              "no_id_no_engine_buffer", "no_id_reason_offset_by_program_base", "no_id_reason_not_single_load", "no_id_reason_no_record_words",
                                              "no_id_reason_id_zero", "no_id_reason_other", "no_own_old_path_stream_ring", "no_own_old_path_placed_it_now", "no_own_old_path_other"};
} // namespace

void NoteDiag(Diag what, const char* detail) noexcept {
    if (what >= DiagCount) {
        return;
    }
    diags[what].fetch_add(1, std::memory_order_relaxed);
    if (detail != nullptr && diag_logged[what].fetch_add(1, std::memory_order_relaxed) < 6) {
        LOG_WARNING(Render_Vulkan, "F1 diag {}: {}", DiagNames[what], detail);
    }
}

u32 OwnCloseMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_OWN_CLOSE"); // default 2 (real, 1 in 64 audited) since 2026-11-10 after the owner's play; =0 off, =1 shadow
        if (e == nullptr || e[0] == 0) {
            return 2u;
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

namespace {
std::array<std::array<std::atomic<u64>, OwnCount>, 10> owns{};
}

void NoteOwn(u32 what, u32 cls) noexcept {
    if (what < OwnCount && cls < 10) {
        owns[cls][what].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteDrop(u32 what) noexcept {
    if (what < DropCount) {
        drops[what].fetch_add(1, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    {
        static constexpr const char* D[DropCount] = {"asked", "no_registered_buffer", "set_held_the_range", "flag_set_would_drop", "FLAG_SET_BUT_NOT_DIRTY_MUST_BE_0",
                                                     "set_held_but_flag_clear", "DROP_WITH_LIVE_COPY_MUST_BE_0", "memo_erase_removed_entries_kept", "uploads_cleared_the_flag", "dropped_real", "audited_real"};
        if (OwnPlaceMode() != 0) {
            static constexpr const char* P[PlaceCount] = {"stream_binds_no_own_block", "distinct_buffers", "binds_with_no_write_since_the_last_bind", "binds_after_a_write_or_first_uploads",
                                                          "unused", "binds_of_zero_copy_buffers", "bytes_of_the_buffers"};
            for (u32 k = 0; k < PlaceCount; ++k) {
                j["place_shadow"][P[k]] = places[k].load();
            }
        }
        if (DiagOn()) {
            for (u32 k = 0; k < DiagCount; ++k) {
                j["f1_diag"][DiagNames[k]] = diags[k].load();
            }
        }
        j["drop_dirty"]["mode"] = DropDirtyMode();
        for (u32 k = 0; k < DropCount; ++k) {
            j["drop_dirty"][D[k]] = drops[k].load();
        }
    }
    {
        static constexpr const char* O[OwnCount] = {"asked", "immutable_inside", "no_own_buffer", "not_synced", "bound_from_own", "compared", "DIFFERENT_MUST_BE_0",
                                                    "OLD_PATH_DID_WORK_MUST_BE_0", "old_answer_stream_ring", "own_set", "own_cleared", "not_gpu_marked_yet", "own_resolved_lazily",
                                                    "bindings_valid_only_by_a_resolved_pointer"};
        static constexpr const char* C[10] = {"own_immutable", "own_mutable_read", "own_mutable_written", "own_large_read", "own_large_written", "own_range_read", "own_range_written",
                                             "own_index", "own_vertex", "own_vertex_split"};
        for (u32 c = 0; c < 10; ++c) {
            j[C[c]]["mode"] = c == 0 ? OwnImmutableMode() : c >= 7 ? OwnGeometryMode() : c >= 5 ? OwnRangeMode() : c >= 3 ? OwnLargeMode() : OwnMutableMode();
            j[C[c]]["close_mode"] = OwnCloseMode();
            for (u32 k = 0; k < OwnCount; ++k) {
                j[C[c]][O[k]] = owns[c][k].load();
            }
        }
    }
    if (!On()) {
        return j;
    }
    static constexpr const char* B[BindClassCount] = {"ring", "special", "engine_id_inside_its_buffer", "engine_id_crossing_its_buffer", "engine_id_not_in_registry", "no_engine_id", "ring_from_hook_ring", "ring_copy_of_an_engine_buffer", "ring_copy_no_engine_buffer", "no_id_inside_an_engine_buffer", "no_id_no_engine_buffer", "no_id_no_record_words"};
    static constexpr const char* U[4] = {"default", "immutable", "dynamic", "staging"};
    for (u32 k = 0; k < BindClassCount; ++k) {
        j["binds"][B[k]] = binds[k].load();
    }
    j["resources_registered"] = resources.load();
    j["applies_total"] = applies_total.load();
    j["applies_no_engine_buffer"] = applies_no_buffer.load();
    j["faults_total"] = faults_total.load();
    j["copies_total"] = copies_total.load();
    j["hook_ring_binds_inside_an_engine_buffer"] = hook_in_buffer.load();
    j["geometry_reads_in_engine_buffer"] = geometry_in_buffer.load();
    j["geometry_reads_no_engine_buffer"] = geometry_no_buffer.load();
    j["copies_src_no_engine_buffer"] = copies_src_no_buffer.load();
    j["copies_dst_no_engine_buffer"] = copies_dst_no_buffer.load();
    j["faults_no_engine_buffer"] = faults_no_buffer.load();
    // per usage, over the buffers bound at least once (cumulative since start)
    struct Sum {
        u64 buffers{}, bytes{}, gpu_written{}, cpu_written{}, faulted{}, gpu_and_cpu{}, never_written_after_creation{}, caller_memory{}, binds{}, ring_copies{}, noid_binds{}, copy_src{}, copy_dst{}, geometry{}, hook{}, hook_binds{};
        std::array<u64, Writers> by_writer_buffers{}, by_writer_applies{};
    };
    std::array<Sum, 5> sums{}; // 0-3 usage, 4 other
    std::array<Sum, 5> ring_only{}; // buffers reached only through constant ring copies
    std::array<Sum, 5> unbound_written{}; // buffers no binding reached, written after creation
    u64 largest = 0;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, info] : by_id) {
            const u64 bound = info.binds_read + info.binds_written;
            u64 cpu_after = 0;
            for (u32 w = 0; w < Writers; ++w) {
                cpu_after += w != 5 && w != 13 ? info.applies[w] : 0;
            }
            if (bound == 0 && info.ring_copies == 0 && cpu_after == 0 && info.noid_binds == 0 && info.copy_src == 0 && info.copy_dst == 0 && info.geometry_reads == 0 && info.hook_binds == 0) {
                continue;
            }
            Sum& s = (bound != 0 ? sums : info.ring_copies != 0 ? ring_only : unbound_written)[info.usage < 4 ? info.usage : 4];
            s.ring_copies += info.ring_copies;
            s.noid_binds += info.noid_binds;
            s.copy_src += info.copy_src != 0 ? 1 : 0;
            s.copy_dst += info.copy_dst != 0 ? 1 : 0;
            s.geometry += info.geometry_reads != 0 ? 1 : 0;
            s.hook += info.hook_binds != 0 ? 1 : 0;
            s.hook_binds += info.hook_binds;
            ++s.buffers;
            s.bytes += info.size;
            s.binds += bound;
            largest = std::max(largest, info.size);
            u64 cpu = 0;
            for (u32 w = 0; w < Writers; ++w) {
                // writer 5 is creation and UpdateSubresource: the initial fill is counted, but a buffer written only by it is "never written after creation"
                if (info.applies[w] != 0) {
                    ++s.by_writer_buffers[w];
                    s.by_writer_applies[w] += info.applies[w];
                    if (w != 5 && w != 13) {
                        cpu += info.applies[w];
                    }
                }
            }
            s.gpu_written += info.binds_written != 0 ? 1 : 0;
            s.cpu_written += cpu != 0 ? 1 : 0;
            s.faulted += info.faults != 0 ? 1 : 0;
            s.gpu_and_cpu += info.binds_written != 0 && (cpu != 0 || info.faults != 0) ? 1 : 0;
            s.never_written_after_creation += info.binds_written == 0 && cpu == 0 && info.faults == 0 ? 1 : 0;
            s.caller_memory += (info.misc & 0x80000) != 0 ? 1 : 0;
        }
    }
    for (u32 u = 0; u < 15; ++u) {
        const Sum& s = (u < 5 ? sums : u < 10 ? ring_only : unbound_written)[u % 5];
        if (s.buffers == 0) {
            continue;
        }
        auto& o = j[u < 5 ? "bound_buffers" : u < 10 ? "ring_copied_only_buffers" : "written_unbound_buffers"][u % 5 < 4 ? U[u % 5] : "other"];
        o["ring_copies"] = s.ring_copies;
        o["binds_without_engine_id"] = s.noid_binds;
        o["buffers_copy_source"] = s.copy_src;
        o["buffers_copy_destination"] = s.copy_dst;
        o["buffers_read_as_geometry"] = s.geometry;
        o["buffers_read_from_hook_ring"] = s.hook;
        o["hook_ring_binds"] = s.hook_binds;
        o["buffers"] = s.buffers;
        o["bytes"] = s.bytes;
        o["binds"] = s.binds;
        o["gpu_written"] = s.gpu_written;
        o["cpu_written_after_creation"] = s.cpu_written;
        o["write_faulted"] = s.faulted;
        o["gpu_and_cpu_written"] = s.gpu_and_cpu;
        o["never_written_after_creation"] = s.never_written_after_creation;
        o["on_caller_memory"] = s.caller_memory;
        for (u32 w = 0; w < Writers; ++w) {
            if (s.by_writer_buffers[w] != 0) {
                o["by_writer"][std::to_string(w)] = {{"buffers", s.by_writer_buffers[w]}, {"applies", s.by_writer_applies[w]}};
            }
        }
    }
    j["largest_bound_buffer_bytes"] = largest;
    return j;
}

} // namespace Core::OwnMemoryCensus
