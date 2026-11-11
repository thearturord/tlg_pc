// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include "native/native_coverage.h"
#include "native/native_debug_server.h"

namespace Core::Coverage {

bool g_on = [] {
    const char* e = std::getenv("TLG_COVERAGE_CENSUS");
    return e != nullptr && e[0] == '1';
}();

namespace {
// raster thread only (one writer); the report reads them relaxed
struct Kind {
    std::atomic<u64> ops{0}, covered{0};
    std::array<std::atomic<u64>, OldCount> ops_with{};   ///< operations with at least one binding of this class on the old path
    std::array<std::atomic<u64>, OldCount> only{};       ///< operations whose only old-path class is this one
    std::array<std::atomic<u64>, OldCount> bindings{};   ///< old-path bindings of this class
};
std::array<Kind, 2> kinds; // 0 draws, 1 dispatches
bool t_open = false;
bool t_compute = false;
std::array<u32, OldCount> t_old{};
u64 t_since_publish = 0;
std::array<std::array<std::atomic<u64>, 6>, OldCount> details{}; // [class][id class * 3 + answer]
std::array<std::array<std::atomic<u64>, 6>, OldCount> detail_bytes{};
std::mutex g_program_mutex;
std::unordered_map<u64, u64> g_programs; // (class << 60) ^ program -> old-path bindings
std::array<std::atomic<u64>, IrCount> image_reasons{};
std::unordered_map<u64, u64> g_reason_programs; // (reason << 56) ^ program -> stages (under g_program_mutex)

nlohmann::json Report() {
    static constexpr const char* names[OldCount] = {"buffer_read", "buffer_written", "image_stage", "vertex", "index"};
    nlohmann::json j = nlohmann::json::object();
    for (u32 k = 0; k < 2; ++k) {
        auto& o = j[k == 0 ? "draws" : "dispatches"];
        const u64 ops = kinds[k].ops.load(std::memory_order_relaxed);
        const u64 covered = kinds[k].covered.load(std::memory_order_relaxed);
        o["operations"] = ops;
        o["every_binding_native"] = covered;
        o["share_native_percent"] = ops != 0 ? 100.0 * static_cast<double>(covered) / static_cast<double>(ops) : 0.0;
        for (u32 c = 0; c < OldCount; ++c) {
            o["operations_with_old_path"][names[c]] = kinds[k].ops_with[c].load(std::memory_order_relaxed);
            o["operations_blocked_only_by"][names[c]] = kinds[k].only[c].load(std::memory_order_relaxed);
            o["old_path_bindings"][names[c]] = kinds[k].bindings[c].load(std::memory_order_relaxed);
        }
    }
    static constexpr const char* dnames[6] = {"no_id_stream_copy", "no_id_other_buffer", "no_id_meets_frame_ring", "id_stream_copy", "id_other_buffer", "id_meets_frame_ring"};
    for (u32 c = 0; c < OldCount; ++c) {
        for (u32 d = 0; d < 6; ++d) {
            if (const u64 n = details[c][d].load(std::memory_order_relaxed); n != 0) {
                j["detail"][names[c]][dnames[d]] = {{"bindings", n}, {"bytes", detail_bytes[c][d].load(std::memory_order_relaxed)}};
            }
        }
    }
    {
        std::scoped_lock lock{g_program_mutex};
        std::vector<std::pair<u64, u64>> top(g_programs.begin(), g_programs.end());
        std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        for (size_t i = 0; i < top.size() && i < 16; ++i) {
            j["top_programs"].push_back({{"class", names[top[i].first >> 60]}, {"program", fmt::format("{:#x}", top[i].first & 0x0fffffffffffffffull)}, {"bindings", top[i].second}});
        }
    }
    {
        static constexpr const char* rnames[IrCount] = {"no_images_or_no_plan", "written_storage_not_real", "mip_fallback", "sharp_not_single_load", "several_bindings",
                                                       "no_record_words", "no_view_id", "no_entry_yet", "not_in_class_dynamic_or_unknown", "no_payload_answer_not_bindable",
                                                       "entry_image_stale", "audit_one_in_64", "tail_same_image_twice", "tail_check_failed", "storage_mark_mismatch", "view_table_off"};
        for (u32 r = 0; r < IrCount; ++r) {
            if (const u64 n = image_reasons[r].load(std::memory_order_relaxed); n != 0) {
                j["image_stage_reasons_note_151"][rnames[r]] = n;
            }
        }
        std::scoped_lock lock{g_program_mutex};
        std::vector<std::pair<u64, u64>> top(g_reason_programs.begin(), g_reason_programs.end());
        std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        for (size_t i = 0; i < top.size() && i < 16; ++i) {
            j["image_stage_reason_top_programs"].push_back({{"reason", rnames[std::min<u64>(top[i].first >> 56, IrCount - 1)]},
                                                            {"program", fmt::format("{:#x}", top[i].first & 0x00ffffffffffffffull)}, {"stages", top[i].second}});
        }
    }
    j["note"] = "note 143, cumulative; an operation counts as native when no binding of it took the old path (own-memory audits and hook ring answers count as native)";
    return j;
}

void Close() noexcept {
    if (!t_open) {
        return;
    }
    t_open = false;
    auto& k = kinds[t_compute ? 1 : 0];
    k.ops.fetch_add(1, std::memory_order_relaxed);
    u32 classes = 0, last = 0;
    for (u32 c = 0; c < OldCount; ++c) {
        if (t_old[c] != 0) {
            ++classes;
            last = c;
            k.ops_with[c].fetch_add(1, std::memory_order_relaxed);
            k.bindings[c].fetch_add(t_old[c], std::memory_order_relaxed);
        }
    }
    if (classes == 0) {
        k.covered.fetch_add(1, std::memory_order_relaxed);
    } else if (classes == 1) {
        k.only[last].fetch_add(1, std::memory_order_relaxed);
    }
    if (++t_since_publish >= 4096) {
        t_since_publish = 0;
        Core::DebugServer::Publish("coverage", Report());
    }
}
} // namespace

void Begin(bool compute) noexcept {
    if (!g_on) {
        return;
    }
    Close();
    t_open = true;
    t_compute = compute;
    t_old.fill(0);
}

void NoteOldDetail(Old what, u32 cls, u32 answer, u64 program, u64 size) noexcept {
    if (!g_on || !t_open || what >= OldCount) {
        return;
    }
    const u32 d = (cls != 0 ? 3u : 0u) + (answer < 3 ? answer : 1u);
    details[what][d].fetch_add(1, std::memory_order_relaxed);
    detail_bytes[what][d].fetch_add(size, std::memory_order_relaxed);
    std::scoped_lock lock{g_program_mutex};
    if (g_programs.size() < 4096) {
        ++g_programs[(static_cast<u64>(what) << 60) ^ (program & 0x0fffffffffffffffull)];
    }
}

void NoteImageReason(u32 reason, u64 program) noexcept {
    if (!g_on || !t_open) {
        return;
    }
    const u32 r = reason < IrCount ? reason : IrCount - 1;
    image_reasons[r].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{g_program_mutex};
    if (g_reason_programs.size() < 4096) {
        ++g_reason_programs[(static_cast<u64>(r) << 56) ^ (program & 0x00ffffffffffffffull)];
    }
}

void NoteOld(Old what) noexcept {
    if (!g_on || !t_open || what >= OldCount) {
        return;
    }
    ++t_old[what];
}

} // namespace Core::Coverage
