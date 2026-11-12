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
const unsigned g_pointers = [] {
    const char* e = std::getenv("TLG_PLAN_POINTERS");
    return e == nullptr || e[0] == 0 ? 2u : e[0] == '1' || e[0] == '2' ? static_cast<unsigned>(e[0] - '0') : 0u; // default 2 since 2026-11-12 (note 161); LEGACY(plan_pointers): =0 the plan copies the fixed state, =1 shadow
}();

enum C : u32 {
    FxReused, FxRecords, FxNoPrev, FxSameKey, FxSameKeyOtherState, FxOtherKey, FxOtherKeySameState,
    BdRecords, BdSameCount, BdAllSame, BdSlots, BdSameObject, BdSameObjectSameBinding, BdSameObjectOtherBinding, BdOtherObjectSameBinding,
    PfChecked, PfSame, PfDiffer,
    PbSame, PbReused, PbOther, PbNoRecord,
    BvSameVersion, BvSameVersionOther,
    TbAsked, TbSame, TbDiffer, TbNotView, TbNotSrvBuffer, TbOutOfRange,
    TrUsed, TrSame, TrDiffer, TrNoTyped,
    CCount
};
std::array<std::atomic<u64>, CCount> c{};
std::array<std::atomic<u32>, 65536> g_versions{}; // note 164
std::array<std::atomic<u64>, 32> g_ver_field{};
std::atomic<u64> since_publish{0};

void Add(C k, u64 n = 1) noexcept {
    c[k].fetch_add(n, std::memory_order_relaxed);
}

nlohmann::json Report() {
    const auto r = [](C k) { return c[k].load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["hook_diff_mode"] = g_diff;
    j["fixed_state_reused_without_lock_note_157"] = r(FxReused);
    j["plan_pointers_mode"] = g_pointers;
    j["fixed_state"] = {{"records", r(FxRecords)}, {"no_previous", r(FxNoPrev)}, {"same_key_as_previous", r(FxSameKey)},
                        {"SAME_KEY_OTHER_STATE_MUST_BE_0", r(FxSameKeyOtherState)}, {"other_key", r(FxOtherKey)}, {"other_key_same_state", r(FxOtherKeySameState)}};
    j["bindings"] = {{"records", r(BdRecords)}, {"same_count_as_previous", r(BdSameCount)}, {"every_binding_same_as_previous", r(BdAllSame)}, {"slots", r(BdSlots)},
                     {"slots_same_object", r(BdSameObject)}, {"same_object_same_binding", r(BdSameObjectSameBinding)},
                     {"same_object_other_binding_needs_a_version", r(BdSameObjectOtherBinding)}, {"other_object_same_binding", r(BdOtherObjectSameBinding)}};
    j["binding_versions_note_164"] = {{"same_object_same_version_slots", r(BvSameVersion)}, {"SAME_VERSION_OTHER_BINDING_MUST_BE_0", r(BvSameVersionOther)}};
    {
        static constexpr const char* fields[8] = {"words_0_1", "words_2_3", "words_4_7", "table_base", "table_register_or_entry", "dwords", "id_or_view", "other"};
        static constexpr const char* kinds[4] = {"kind0", "buffer", "image", "sampler"};
        for (u32 k = 0; k < 4; ++k) {
            for (u32 f = 0; f < 8; ++f) {
                if (const u64 n = g_ver_field[(k << 3) | f].load(std::memory_order_relaxed); n != 0) {
                    j["binding_versions_note_164"]["first_field_that_differs"][kinds[k]][fields[f]] = n;
                }
            }
        }
    }
    j["plan_pointers"] = {{"fixed_checked", r(PfChecked)}, {"fixed_same_as_cached", r(PfSame)}, {"fixed_differs_from_cached", r(PfDiffer)},
                          {"bindings_slot_still_holds_the_record", r(PbSame)}, {"bindings_slot_reused_before_execution", r(PbReused)},
                          {"bindings_same_record_other_bindings", r(PbOther)}, {"bindings_no_record", r(PbNoRecord)}};
    j["typed_buffers_note_175"] = {{"mode", TypedBuffersMode()}, {"asked", r(TbAsked)}, {"same_as_the_words", r(TbSame)}, {"different_from_the_words", r(TbDiffer)},
                                   {"not_an_engine_view", r(TbNotView)}, {"not_a_buffer_srv", r(TbNotSrvBuffer)}, {"outside_the_engine_range", r(TbOutOfRange)}};
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
    return g_pointers == 1;
}

bool FixedPointerReal() noexcept {
    return g_pointers == 2;
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

void BumpVersion(u64 id) noexcept {
    if (id != 0) {
        g_versions[(id * 0x9E3779B97F4A7C15ull) >> 48].fetch_add(1, std::memory_order_acq_rel);
    }
}

u32 VersionOf(u64 id) noexcept {
    return id != 0 ? g_versions[(id * 0x9E3779B97F4A7C15ull) >> 48].load(std::memory_order_acquire) : 1u;
}

void NoteVersionDiffField(u32 field, u32 kind) noexcept {
    g_ver_field[(std::min<u32>(kind, 3) << 3) | (field & 7)].fetch_add(1, std::memory_order_relaxed);
}

void NoteBindingVersions(u32 same_version_slots, u32 same_version_other_binding) noexcept {
    Add(BvSameVersion, same_version_slots);
    Add(BvSameVersionOther, same_version_other_binding);
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

unsigned TypedBuffersMode() noexcept {
    static const unsigned mode = [] {
        const char* e = std::getenv("TLG_TYPED_BUFFERS");
        return e != nullptr && (e[0] == '1' || e[0] == '2') ? static_cast<unsigned>(e[0] - '0') : 0u; // note 175: 0 off, 1 shadow; note 176: 2 the readers take the typed fields
    }();
    return mode;
}

nlohmann::json TypedReport() {
    const auto r = [](C k) { return c[k].load(std::memory_order_relaxed); };
    return {{"mode", TypedBuffersMode()}, {"asked", r(TbAsked)}, {"same_as_the_words", r(TbSame)}, {"different_from_the_words", r(TbDiffer)},
            {"not_an_engine_view", r(TbNotView)}, {"not_a_buffer_srv", r(TbNotSrvBuffer)}, {"outside_the_engine_range", r(TbOutOfRange)},
            {"readers_note_176", {{"used_without_the_v_sharp", r(TrUsed)}, {"compared_same", r(TrSame)}, {"TYPED_DIFFERS_FROM_THE_DECODE_MUST_BE_0", r(TrDiffer)},
                                  {"no_typed_fields_decoded", r(TrNoTyped)}}}};
}

void NoteTypedReader(u32 what) noexcept {
    if (what < 4) {
        Add(static_cast<C>(TrUsed + what));
    }
}

void NoteTypedBuffer(u32 what) noexcept {
    if (what < 6) {
        Add(static_cast<C>(TbAsked + what));
    }
}

void NotePlanBindings(u32 state) noexcept {
    Add(state == 0 ? PbSame : state == 1 ? PbReused : state == 2 ? PbOther : PbNoRecord);
}

} // namespace Core::HookDiff
