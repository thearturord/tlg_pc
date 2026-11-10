// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_view_table.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace Core::ViewTable {

u32 Mode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_VIEW_TABLE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 99); LEGACY(view_table): =0 finds every image by its T# again, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 TargetMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_TARGET_TABLE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 106); LEGACY(target_table): =0 finds every target by its register words again, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

u32 WrittenMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_WRITTEN_NO_IMAGE");
        if (e == nullptr || e[0] == 0) {
            return 2u; // default 2 since 2026-11-11 after the owner's play (note 110); LEGACY(written_no_image): =0 walks the texture cache for every written buffer, =1 shadow
        }
        return e[0] == '1' || e[0] == '2' ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

namespace {
std::mutex mutex; // (the raster thread binds and applies the batch's events; the lock only guards a second recording thread)
std::unordered_map<u64, Entry> entries;                       // key: view id and use flags
std::unordered_map<u64, std::vector<u64>> keys_by_view;      // view id -> keys
std::unordered_map<u64, std::vector<u64>> views_by_resource; // texture id -> view ids
std::array<std::atomic<u64>, Count_> counts{};

u64 Key(u64 view_id, u64 use_flags) noexcept {
    u64 k = view_id * 0x9E3779B97F4A7C15ull ^ (use_flags + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full;
    return k ^ (k >> 31);
}

void DropView(u64 view_id) {
    const auto it = keys_by_view.find(view_id);
    if (it == keys_by_view.end()) {
        return;
    }
    for (const u64 k : it->second) {
        if (entries.erase(k) != 0) {
            counts[DroppedByEvent].fetch_add(1, std::memory_order_relaxed);
        }
    }
    keys_by_view.erase(it);
}
} // namespace

Entry* Find(u64 view_id, u64 use_flags) noexcept {
    std::scoped_lock lock{mutex};
    const auto it = entries.find(Key(view_id, use_flags));
    return it != entries.end() && it->second.view_id == view_id ? &it->second : nullptr;
}

Entry& Put(u64 view_id, u64 use_flags) noexcept {
    std::scoped_lock lock{mutex};
    const u64 k = Key(view_id, use_flags);
    auto [it, inserted] = entries.try_emplace(k);
    if (inserted) {
        keys_by_view[view_id].push_back(k);
    }
    it->second = Entry{};
    it->second.view_id = view_id;
    return it->second;
}

void Event(u32 what, u64 id) noexcept {
    if ((Mode() == 0 && TargetMode() == 0) || (what != 0 && what != 1 && what != 3)) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (what == 0) {
        DropView(id);
        return;
    }
    if (const auto it = views_by_resource.find(id); it != views_by_resource.end()) {
        for (const u64 v : it->second) {
            DropView(v);
        }
        views_by_resource.erase(it);
    }
}

void Note(Count c, u64 n) noexcept {
    counts[c].fetch_add(n, std::memory_order_relaxed);
}

void LinkResource(u64 resource_id, u64 view_id) noexcept {
    std::scoped_lock lock{mutex};
    auto& v = views_by_resource[resource_id];
    for (const u64 x : v) {
        if (x == view_id) {
            return;
        }
    }
    v.push_back(view_id);
}

nlohmann::json Report() {
    const auto c = [](Count k) { return counts[k].load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["mode"] = Mode();
    {
        std::scoped_lock lock{mutex};
        j["entries"] = entries.size();
    }
    j["stages"] = {{"with_images", c(StagesWithImages)}, {"all_in_class", c(StagesAllInClass)}, {"would_be_served_by_the_table", c(StagesWouldServe)},
                   {"WITH_DIFFERENCE_MUST_BE_0", c(StagesWithDifference)}};
    j["images"] = {{"in_class", c(ImagesInClass)}, {"same", c(ImagesSame)}, {"VIEW_TABLE_DIFFERENT_MUST_BE_0", c(ImagesDifferent)}, {"no_entry_made", c(ImagesNoEntry)},
                   {"stale_remade", c(ImagesStale)}};
    j["not_in_class"] = {{"no_view", c(NotClassNoView)}, {"dynamic_or_unknown_texture", c(NotClassDynamic)}, {"written_fallback_or_multi", c(NotClassUse)}, {"null", c(NotClassNull)}};
    j["entries_dropped_by_delete_events"] = c(DroppedByEvent);
    j["targets"] = {{"mode", TargetMode()}, {"draws", c(TgDraws)}, {"draws_all_in_class", c(TgDrawsAllInClass)}, {"draws_would_be_served_by_the_table", c(TgDrawsWouldServe)},
                    {"draws_in_class_where_the_old_path_ran_findimage", c(TgDrawsOldFind)}, {"targets_in_class", c(TgTargets)}, {"same", c(TgSame)},
                    {"TARGET_TABLE_DIFFERENT_MUST_BE_0", c(TgDifferent)}, {"no_entry_made", c(TgNoEntry)}, {"stale_remade", c(TgStale)}, {"no_view", c(TgNoView)},
                    {"view_same", c(TgViewSame)}, {"TARGET_VIEW_DIFFERENT_MUST_BE_0", c(TgViewDifferent)}, {"view_recorded", c(TgViewNoEntry)},
                    {"TARGET_DESC_DIFFERENT_MUST_BE_0", c(TgDescDifferent)}, {"real_targets_taken", c(TgTaken)}, {"real_draws_taken", c(TgDrawsTaken)},
                    {"real_draws_audited", c(TgAudits)}};
    j["written_buffers"] = {{"mode", WrittenMode()}, {"walks", c(WbCalls)}, {"named_whole", c(WbNamed)}, {"id_says_no_image", c(WbIdNone)}, {"walk_marked_an_image", c(WbWalkFound)},
                            {"ID_SAID_NO_IMAGE_BUT_WALK_FOUND_MUST_BE_0", c(WbIdNoneWalkFound)}, {"not_named", c(WbNotNamed)}, {"real_walks_skipped", c(WbSkipped)},
                            {"real_audited", c(WbAudited)}, {"AUDIT_WALK_FOUND_MUST_BE_0", c(WbAuditFound)}};
    j["real"] = {{"stages_bound_from_the_table", c(RealServed)}, {"image_bindings_bound_from_the_table", c(RealServedImages)}, {"stages_audited_by_the_old_path", c(RealAudited)},
                 {"stages_whose_image_checks_failed", c(RealCheckFailed)}};
    return j;
}

} // namespace Core::ViewTable
