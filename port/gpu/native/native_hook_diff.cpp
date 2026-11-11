// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_hook_diff.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "native/native_debug_server.h"

namespace Core::HookDiff {

namespace {
const unsigned g_diff = [] {
    const char* e = std::getenv("TLG_HOOK_DIFF");
    return e == nullptr || e[0] == 0 ? 2u : e[0] == '1' || e[0] == '2' ? static_cast<unsigned>(e[0] - '0') : 0u; // default 2 since 2026-11-11 (note 157); LEGACY(hook_diff): =0 looks every fixed state up under the lock
}();
const bool g_pointers = [] {
    const char* e = std::getenv("TLG_PLAN_POINTERS");
    return e != nullptr && e[0] == '1';
}();

enum C : u32 {
    FxReused, FxRecords, FxNoPrev, FxSameKey, FxSameKeyOtherState, FxOtherKey, FxOtherKeySameState,
    BdRecords, BdSameCount, BdAllSame, BdSlots, BdSameObject, BdSameObjectSameBinding, BdSameObjectOtherBinding, BdOtherObjectSameBinding,
    PfChecked, PfSame, PfDiffer,
    PbSame, PbReused, PbOther, PbNoRecord,
    CCount
};
std::array<std::atomic<u64>, CCount> c{};
std::atomic<u64> since_publish{0};

void Add(C k, u64 n = 1) noexcept {
    c[k].fetch_add(n, std::memory_order_relaxed);
}

nlohmann::json Report() {
    const auto r = [](C k) { return c[k].load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["hook_diff_mode"] = g_diff;
    j["fixed_state_reused_without_lock_note_157"] = r(FxReused);
    j["plan_pointers_mode"] = g_pointers ? 1 : 0;
    j["fixed_state"] = {{"records", r(FxRecords)}, {"no_previous", r(FxNoPrev)}, {"same_key_as_previous", r(FxSameKey)},
                        {"SAME_KEY_OTHER_STATE_MUST_BE_0", r(FxSameKeyOtherState)}, {"other_key", r(FxOtherKey)}, {"other_key_same_state", r(FxOtherKeySameState)}};
    j["bindings"] = {{"records", r(BdRecords)}, {"same_count_as_previous", r(BdSameCount)}, {"every_binding_same_as_previous", r(BdAllSame)}, {"slots", r(BdSlots)},
                     {"slots_same_object", r(BdSameObject)}, {"same_object_same_binding", r(BdSameObjectSameBinding)},
                     {"same_object_other_binding_needs_a_version", r(BdSameObjectOtherBinding)}, {"other_object_same_binding", r(BdOtherObjectSameBinding)}};
    j["plan_pointers"] = {{"fixed_checked", r(PfChecked)}, {"fixed_same_as_cached", r(PfSame)}, {"fixed_differs_from_cached", r(PfDiffer)},
                          {"bindings_slot_still_holds_the_record", r(PbSame)}, {"bindings_slot_reused_before_execution", r(PbReused)},
                          {"bindings_same_record_other_bindings", r(PbOther)}, {"bindings_no_record", r(PbNoRecord)}};
    j["note"] = "notes 153 and 154 shadows; nothing here changes what is drawn";
    return j;
}

void MaybePublish() noexcept {
    if (((since_publish.fetch_add(1, std::memory_order_relaxed) + 1) & 4095) == 0) {
        Core::DebugServer::Publish("hook_diff", Report());
    }
}
} // namespace

bool DiffOn() noexcept {
    return g_diff != 0;
}

bool DiffShadow() noexcept {
    return g_diff == 1;
}

bool DiffReal() noexcept {
    return g_diff == 2;
}

void NoteReused() noexcept {
    Add(FxReused);
}

bool PointersOn() noexcept {
    return g_pointers;
}

void NoteFixed(bool have_prev, bool same_key, bool same_state) noexcept {
    Add(FxRecords);
    if (!have_prev) {
        Add(FxNoPrev);
    } else if (same_key) {
        Add(same_state ? FxSameKey : FxSameKeyOtherState);
        if (!same_state) {
            Add(FxSameKey);
        }
    } else {
        Add(FxOtherKey);
        if (same_state) {
            Add(FxOtherKeySameState);
        }
    }
}

void NoteBindings(u32 count, bool same_count, u32 same_object_slots, u32 same_object_same_binding, u32 other_object_same_binding, bool all_same) noexcept {
    Add(BdRecords);
    Add(BdSlots, count);
    if (same_count) {
        Add(BdSameCount);
    }
    if (all_same) {
        Add(BdAllSame);
    }
    Add(BdSameObject, same_object_slots);
    Add(BdSameObjectSameBinding, same_object_same_binding);
    Add(BdSameObjectOtherBinding, same_object_slots - same_object_same_binding);
    Add(BdOtherObjectSameBinding, other_object_same_binding);
    MaybePublish();
}

void NotePlanFixed(bool same) noexcept {
    Add(PfChecked);
    Add(same ? PfSame : PfDiffer);
    MaybePublish();
}

void NotePlanBindings(u32 state) noexcept {
    Add(state == 0 ? PbSame : state == 1 ? PbReused : state == 2 ? PbOther : PbNoRecord);
}

} // namespace Core::HookDiff
