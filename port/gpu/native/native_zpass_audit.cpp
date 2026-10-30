// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3a: read-only audit of the engine's depth pass loop, see native_zpass_audit.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "native/native_draw.h"
#include "native/native_zpass_audit.h"

namespace Core::NativeZPass {

namespace {

constexpr u64 ZPassBegin = 0x01c74a00, ZPassEnd = 0x01c76710; // DrawZPass and the next function (DrawGBufferPrimitives)

template <class T>
bool Rd(u64 address, T& out) {
#ifdef _WIN32
    SIZE_T got = 0;
    return address != 0 && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), &out, sizeof(T), &got) && got == sizeof(T);
#else
    return false;
#endif
}

bool RdBytes(u64 address, void* out, size_t size) {
#ifdef _WIN32
    SIZE_T got = 0;
    return address != 0 && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), out, size, &got) && got == size;
#else
    return false;
#endif
}

enum ExpectedKind : u32 { Plain = 0, Instanced = 1, Indirect = 2 };

constexpr size_t CtxBytes = 0x14000; // the part of the device context the record builder reads

/// A write into the shadow context: the field the loop's per primitive binding sets, taken from the snapshot.
struct Overlay {
    u64 offset;
    std::vector<u8> bytes;
};

struct Item {
    u32 kind;
    u64 count, first, instances;
    u32 batch, primitive;
    u64 material{};      // expected side: the primitive's material
    u64 material_sig{};  // expected side: a hash of the material fields the variant search reads
    u32 material_flag{}; // expected side: material + 0xd4 & 1 (the per-primitive resource set is bound only for these)
    u64 batch_object{};  // expected side: the batch object
    u32 triangles{};       // expected side: the triangle count of the primitive (the loop adds it to renderDev + 0x14 for every draw)
    u64 prim_record{};     // expected side: the address of the primitive record (the loop stores the chosen variant behind it: *(*(record + 0x28) + 0x48))
    u64 instance_object{}; // expected side: the instance group object of an instanced / indirect batch (FUN_01ee4e50 binds it)
    u64 batch_special{}; // expected side: batch + 0x48 (a special entry that fills the resource table itself)
    u32 mesh_word{};     // expected side: the mesh's vertex format word (mesh + 0x78), part of the variant key
    u32 batch_flags{};   // expected side: the batch flag word (bit 0x10 goes into the variant key)
    u64 vertex_shader{}, pixel_shader{}; // observed side: the shader objects the device context holds at the draw
};

/// Audit b: is the pipeline the loop chooses a function of the snapshot? One entry per key.
struct KeyStat {
    u64 vertex_shader, pixel_shader;
    u64 draws{};
    u64 conflicting_draws{};
    bool conflict{false};
};
constexpr u32 KeyKinds = 5;
constexpr const char* key_names[KeyKinds] = {"material only", "view type + material", "view type + material + batch flag 0x10 + draw kind", "view type + the material fields the variant search reads + batch flag 0x10 + draw kind", "the same + the mesh vertex format word (mesh + 0x78) + the pass flag bytes (+0x14, +0x1c)"};

struct ViewStats {
    u64 passes{}, batches{}, batches_skipped_by_flag{}, batches_skipped_instance{}, batches_skipped_no_instances_with_flag{}, batches_skipped_no_mesh{}, primitives{}, primitives_filtered{};
    u64 expected_plain{}, expected_instanced{}, expected_indirect{};
    u64 observed_plain{}, observed_indirect{};
    u64 matched_plain{}, matched_indirect{};
    u64 missing_plain{}, missing_indirect{}, extra_plain{}, extra_indirect{};
    u64 unreadable{};
    // N6.3g: would the pass qualify for the generator (nothing in it that the generator does not model)?
    u64 qualifying_passes{}, qualifying_draws{}, all_draws{};
    std::map<std::string, u64> decline_passes; // reason -> passes it applies to
    std::map<std::string, u64> decline_draws;  // reason -> draws of the passes it applies to
};

struct State {
    std::mutex mutex;
    bool active{false};
    u32 view_type{0};
    u32 pass_flags{0};
    u64 render_device{0};
    size_t cursor{0};
    // audit c: the per-primitive binding. checks: 0 ctx[0x980..0x9c8) = obj[0x450..0x498), 1 ctx[0x9d0..0xa18) = obj[0x498..0x4e0), 2 shader resources at ctx + 0x790, 3 at ctx + 0xb38,
    // 4 constant buffer slots ctx + 0x290 / 0x310, 5 the resource set (renderDev + 0xe80 = material + 0x178)
    static constexpr u32 Checks = 10;
    std::array<u64, Checks> bind_checked{}, bind_equal{}, bind_not_applicable{}, bind_unreadable{};
    std::vector<std::string> bind_samples;
    std::array<u32, Checks> bind_sample_count{};
    u64 bind_unmatched_draws{0};
    // shadow generator (N6.3e)
    std::vector<u8> baseline, shadow;
    u64 pass_ctx{0};
    bool current_valid{false};
    bool current_ctx_differs{false};
    Item current;
    std::vector<Overlay> overlay;
    std::unordered_map<u64, std::pair<u64, u64>> variant_table; // key -> (vertex shader, pixel shader), learned from the loop
    // first-use rule (N6.3f): the variant object the loop chose for a key; its shader pair is read from the object when the draw is generated
    struct VariantEntry {
        u64 variant{0};
        u64 last_vs{0}, last_ps{0};
        u64 pair_changes{0};
        u64 draws{0};
    };
    std::unordered_map<u64, VariantEntry> variants;
    u64 draws_with_special_entry{0}, draws_with_special_entry_pair_right{0};
    u64 draws_keeping_the_previous_shaders{0}, draws_keeping_right{0};
    u64 prev_live_vs{0}, prev_live_ps{0}, wrong_equal_previous{0}, wrong_total{0};
    // fields of the primitive record and the batch of one family (view 0, batch flags 0x6, mesh word 0x1a40): value sets of right and of wrong draws
    struct FieldValues {
        std::map<u32, u64> right, wrong;
    };
    std::map<std::string, FieldValues> family_fields;
    u64 last_caller{0};
    // end state audit (N6.3h)
    s32 rd_draws_before{0}, rd_triangles_before{0};
    u32 batches_prepared{0}, groups_processed{0}, groups_skipped{0}, last_group_batch_ordinal{0}, last_group_draws{0}, indirect_pass{0};
    std::map<std::string, std::pair<u64, u64>> group_patterns; // pattern -> {passes, passes whose live 0x70 stayed set}
    bool pass_has_special{false}, pass_has_resource_set{false};
    struct PassDiff {
        std::vector<u32> offsets;
        bool special;
    };
    std::vector<PassDiff> end_pass_diffs; // the differing words of every compared pass (classified when the report is made, with the final draw call noise rates)
    std::map<std::string, std::pair<u64, u64>> slot2e0_patterns; // pass kind -> {passes, passes where ctx +0x2e0 differs at the end}
    bool prev_instanced{false};
    u64 prev_batch{0};
    bool any_group{false}; // an instanced batch was prepared in this pass: its end clears the instance state
    // do the draw calls themselves write these words? live value at the previous draw entry against this one, for two plain draws of one batch with no loop write in between
    static constexpr std::array<u32, 4> WatchOffsets{0x70, 0x170, 0x2e0, 0x360};
    std::array<u64, 4> watch_previous{};
    u64 watch_previous_batch{0};
    bool watch_previous_valid{false};
    std::array<u64, 4> watch_pairs{}, watch_changes{};
    u64 last_mesh_batch{0}, last_mesh{0}, last_group{0}; // the last batch the loop prepared (it sets the batch level state before it knows whether the batch draws anything)
    u64 end_passes{0}, end_passes_clean{0}, end_unreadable{0};
    u64 end_counter_ok{0}, end_counter_bad{0};
    struct EndDiff {
        u64 passes{0}, live{0}, shadow{0};
    };
    std::map<u32, EndDiff> end_diffs;
    std::vector<std::string> end_samples;
    u64 entry_vs{0}, entry_ps{0}; // the shader objects the context held when the draw call was entered (before the record builder's resource tracking could draw internal quads)
    u64 pair_changed_inside_tag{0};
    u64 draws_on_another_context{0}, wrong_on_another_context{0}, wrong_on_the_pass_context{0};
    std::map<u64, std::array<u64, 2>> callers; // call site -> {draws, wrong draws}
    std::map<u64, std::pair<u64, u64>> wrong_over_time; // frame / 64 -> {draws, wrong draws}
    std::map<std::string, std::pair<u64, u64>> wrong_by_fields; // (view type, batch flags, mesh word, material flag) -> {draws, wrong draws}
    u64 variant_right{0}, variant_wrong{0}, variant_pair_right{0}, variant_pair_wrong{0}, variant_unreadable{0}, variant_cold{0};
    std::map<std::string, u64> variant_wrong_kinds;
    std::vector<std::string> variant_samples;
    u64 shadow_built{0}, shadow_failed{0}, shadow_equal{0}, shadow_differ{0}, key_hit{0}, key_cold{0}, key_wrong{0}, shadow_no_baseline{0};
    std::map<std::string, u64> shadow_difference_kinds;
    // which 8-byte words of the device context differ between the live context at the draw and the patched copy: offset -> {draws, a sample live value, a sample shadow value}
    struct WordDiff { u64 draws{0}, live{0}, shadow{0}; };
    std::map<u32, WordDiff> context_word_diffs;
    std::vector<u8> live_copy;
    std::vector<std::string> shadow_samples;
    u64 slot_differs_same_object{0}, slot_differs_foreign{0};
    std::map<u64, u64> unused_texture_slot_values; // value of the slots past the material's texture count -> draws (one value expected: the engine's null texture)
    u64 unused_slots_not_uniform{0};
    std::map<std::string, u64> slot_differs_by;
    std::vector<std::string> slot_samples;    std::vector<Item> expected;
    std::vector<Item> observed;
    u64 frame{0};
    std::map<u32, ViewStats> views;
    std::vector<std::string> samples;
    std::array<std::unordered_map<u64, KeyStat>, KeyKinds> keys;
    std::map<std::pair<u64, u64>, u64> pipelines; // (vertex shader, pixel shader) -> draws
    std::vector<std::string> conflict_samples;
};

State& S() {
    static State state;
    return state;
}

u32 Every() {
    static const u32 every = [] {
        const char* e = std::getenv("TLG_ZPASS_AUDIT");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 0u;
    }();
    return every;
}

bool InPass(u64 caller) {
    return caller >= ZPassBegin && caller < ZPassEnd;
}

void FinishLocked(State& s) {
    if (!s.active) {
        return;
    }
    s.active = false;
    auto& v = s.views[s.view_type];
    // greedy in-order matching: for every expected draw the next equal observed draw
    size_t j = 0;
    std::vector<bool> used(s.observed.size(), false);
    for (const auto& e : s.expected) {
        if (e.kind == Instanced) {
            continue; // the instanced draw (DrawIndexedInstanced) is not hooked: nothing to compare
        }
        const u32 want_kind = e.kind == Indirect ? 1u : 0u;
        size_t found = s.observed.size();
        for (size_t k = j; k < s.observed.size() && k < j + 256; ++k) {
            const auto& o = s.observed[k];
            if (o.kind == want_kind && o.count == e.count && o.first == e.first && (want_kind == 1 || o.instances == e.instances)) {
                found = k;
                break;
            }
        }
        if (found == s.observed.size()) {
            (e.kind == Indirect ? v.missing_indirect : v.missing_plain)++;
            if (s.samples.size() < 24) {
                s.samples.push_back(fmt::format("view type {}: expected {} count {} first {} instances {} (batch {}, primitive {}) was not issued", s.view_type,
                                                e.kind == Indirect ? "indirect" : "plain", e.count, e.first, e.instances, e.batch, e.primitive));
            }
        } else {
            (e.kind == Indirect ? v.matched_indirect : v.matched_plain)++;
            {
                const auto& o = s.observed[found];
                ++s.pipelines[{o.vertex_shader, o.pixel_shader}];
                const u64 ks[KeyKinds] = {
                    e.material,
                    e.material * 0x9E3779B97F4A7C15ull ^ s.view_type,
                    (e.material * 0x9E3779B97F4A7C15ull ^ s.view_type) * 31 + (e.batch_flags & 0x10) * 8 + e.kind,
                    (e.material_sig * 0x9E3779B97F4A7C15ull ^ s.view_type) * 31 + (e.batch_flags & 0x10) * 8 + e.kind,
                    ((e.material_sig * 0x9E3779B97F4A7C15ull ^ s.view_type) * 31 + (e.batch_flags & 0x10) * 8 + e.kind) * 0x100000001b3ull + e.mesh_word * 131u + s.pass_flags};
                for (u32 k = 0; k < KeyKinds; ++k) {
                    auto [it, fresh] = s.keys[k].try_emplace(ks[k]);
                    auto& stat = it->second;
                    if (fresh) {
                        stat.vertex_shader = o.vertex_shader;
                        stat.pixel_shader = o.pixel_shader;
                    }
                    ++stat.draws;
                    if (stat.vertex_shader != o.vertex_shader || stat.pixel_shader != o.pixel_shader) {
                        stat.conflict = true;
                        ++stat.conflicting_draws;
                        if (k == KeyKinds - 1 && s.conflict_samples.size() < 8) {
                            s.conflict_samples.push_back(fmt::format("view type {} material {:#x} batch flags {:#x} kind {}: the key first saw {:#x}/{:#x}, now {:#x}/{:#x}", s.view_type, e.material, e.batch_flags, e.kind, stat.vertex_shader,
                                                                     stat.pixel_shader, o.vertex_shader, o.pixel_shader));
                        }
                    }
                }
            }
            for (size_t k = j; k < found; ++k) {
                (s.observed[k].kind == 1 ? v.extra_indirect : v.extra_plain)++;
            }
            used[found] = true;
            j = found + 1;
        }
    }
    for (size_t k = j; k < s.observed.size(); ++k) {
        (s.observed[k].kind == 1 ? v.extra_indirect : v.extra_plain)++;
    }
    for (const auto& o : s.observed) {
        (o.kind == 1 ? v.observed_indirect : v.observed_plain)++;
    }
    s.expected.clear();
    s.observed.clear();
}

} // namespace

namespace {
void CheckBinding(State& s, u32 kind, u64 count, u64 first, u64 instances, u64 ctx) {
    s.current_valid = false;
    const u32 want = kind == 1 ? Indirect : Plain;
    size_t found = s.expected.size();
    for (size_t k = s.cursor; k < s.expected.size() && k < s.cursor + 64; ++k) {
        const auto& e = s.expected[k];
        if (e.kind == Instanced || e.kind != want) {
            continue;
        }
        if (e.count == count && e.first == first && (kind == 1 || e.instances == instances)) {
            found = k;
            break;
        }
    }
    if (found == s.expected.size()) {
        ++s.bind_unmatched_draws;
        return;
    }
    s.cursor = found + 1;
    const Item& e = s.expected[found];
    s.current = e;
    s.current_valid = true;
    s.overlay.clear();
    {
        std::array<u64, 4> now{};
        for (u32 w = 0; w < 4; ++w) {
            Rd(ctx + State::WatchOffsets[w], now[w]);
        }
        if (s.watch_previous_valid && e.batch_object == s.watch_previous_batch && e.instance_object == 0 && e.batch_special == 0 && e.material_flag == 0) {
            for (u32 w = 0; w < 4; ++w) {
                ++s.watch_pairs[w];
                s.watch_changes[w] += now[w] != s.watch_previous[w] ? 1 : 0;
            }
        }
        s.watch_previous = now;
        s.watch_previous_batch = e.batch_object;
        s.watch_previous_valid = true;
    }
    const auto not_applicable = [&](u32 check) { ++s.bind_not_applicable[check]; };
    const auto compare = [&](u32 check, u64 ctx_address, u64 source, size_t bytes, const char* what) {
        std::array<u8, 512> a{}, b{};
        if (bytes > a.size() || !RdBytes(ctx_address, a.data(), bytes) || !RdBytes(source, b.data(), bytes)) {
            ++s.bind_unreadable[check];
            return;
        }
        if (check != 5) {
            s.overlay.push_back({ctx_address - ctx, std::vector<u8>(b.begin(), b.begin() + bytes)});
        }
        ++s.bind_checked[check];
        if (std::memcmp(a.data(), b.data(), bytes) == 0) {
            ++s.bind_equal[check];
        } else if (s.bind_sample_count[check]++ < 3) {
            std::string differing;
            for (size_t i = 0; i + 8 <= bytes && differing.size() < 120; i += 8) {
                if (std::memcmp(a.data() + i, b.data() + i, 8) != 0) {
                    u64 x = 0, y = 0;
                    std::memcpy(&x, a.data() + i, 8);
                    std::memcpy(&y, b.data() + i, 8);
                    differing += fmt::format(" +{:#x}: context {:#x} object {:#x};", i, x, y);
                }
            }
            s.bind_samples.push_back(fmt::format("{} (view type {}, material {:#x}):{}", what, s.view_type, e.material, differing));
        }
    };
    {
        // the raster state object follows the material flags (renderDev + 0x3bf0, or + 0x3bf8 when material + 0xd4 has bits 0x2e), the dirty byte is set
        u32 word = 0;
        Rd(e.material + 0xd4, word);
        u64 state_object = 0;
        Rd(s.render_device + ((word & 0x2e) != 0 ? 0x3bf8 : 0x3bf0), state_object);
        std::vector<u8> bytes(8), dirty(8, 0);
        std::memcpy(bytes.data(), &state_object, 8);
        dirty[0] = 1;
        s.overlay.push_back({0x46c8, std::move(bytes)});
        s.overlay.push_back({0x46d8, std::move(dirty)});
    }
    // the loop's per batch flag bVar54: the mesh render data's first entry of +0x60 is neither the default object nor null and has a render object (FUN_01c92f30 is called only when it is false)
    bool bvar54 = false;
    {
        u64 mesh_b = 0, rd_b = 0, inner_b = 0, array_b = 0, first_b = 0, target_b = 0, g_ctx_b = 0, g1_b = 0, g2_b = 0;
        u32 count_b = 0;
        Rd(e.batch_object + 0x20, mesh_b);
        Rd(mesh_b + 0x38, rd_b);
        Rd(rd_b + 0x20, inner_b);
        Rd(inner_b + 0x68, count_b);
        Rd(inner_b + 0x60, array_b);
        Rd(NativeDraw::ImageBase() + (0x069a1b40ull - 0x1000000), g_ctx_b);
        Rd(g_ctx_b + 0x38, g1_b);
        Rd(g1_b + 0x20, g2_b);
        if (count_b != 0 && Rd(array_b, first_b) && first_b != g2_b && first_b != 0) {
            Rd(first_b + 0x38, target_b);
            bvar54 = target_b != 0;
        }
    }
    u64 obj = 0;
    if (!Rd(e.material + 0x38, obj) || obj == 0) {
        for (u32 c = 0; c < State::Checks; ++c) {
            ++s.bind_unreadable[c];
        }
        return;
    }
    compare(0, ctx + 0x980, obj + 0x450, 0x48, "block 1");
    u32 flag444 = 0;
    Rd(obj + 0x444, flag444);
    if (flag444 != 0) {
        compare(1, ctx + 0x9d0, obj + 0x498, 0x48, "block 2");
    } else {
        not_applicable(1);
    }
    u64 views = 0;
    u32 view_count = 0;
    Rd(obj + 0x4f0, views);
    Rd(obj + 0x4f8, view_count);
    if (views != 0 && view_count != 0 && view_count <= 64) {
        compare(2, ctx + 0x790, views, size_t{view_count} * 8, "views at +0x790");
    } else {
        not_applicable(2);
    }
    views = 0;
    view_count = 0;
    Rd(obj + 0x4e0, views);
    Rd(obj + 0x4e8, view_count);
    if (views != 0 && view_count != 0 && view_count <= 64) {
        compare(3, ctx + 0xb38, views, size_t{view_count} * 8, "views at +0xb38");
    } else {
        not_applicable(3);
    }
    u64 index_word = 0;
    Rd(obj + 0x440, index_word);
    if ((index_word & 0xffffffffull) < 64) {
        const u64 slot = obj + 0xc0 + (index_word & 0xffffffffull) * 0xb0;
        if ((index_word >> 32) != 0) {
            u64 second = 0;
            if (Rd(obj + 0x2d0 + (index_word & 0xffffffffull) * 0xb0, second)) {
                std::vector<u8> bytes(8);
                std::memcpy(bytes.data(), &second, 8);
                s.overlay.push_back({0x298, bytes});
                s.overlay.push_back({0x318, std::move(bytes)});
            }
        }
        compare(4, ctx + 0x290, slot, 8, "constant buffer slot at +0x290");
        compare(4, ctx + 0x310, slot, 8, "constant buffer slot at +0x310");
        // where does a different value come from? Is it another slot of the same render object, or something the object does not hold?
        u64 held = 0, wanted = 0;
        if (Rd(ctx + 0x290, held) && Rd(slot, wanted) && held != wanted) {
            s32 other = -1;
            for (u32 j = 0; j < 64; ++j) {
                u64 candidate = 0;
                if (Rd(obj + 0xc0 + u64{j} * 0xb0, candidate) && candidate == held) {
                    other = static_cast<s32>(j);
                    break;
                }
            }
            (other >= 0 ? s.slot_differs_same_object : s.slot_differs_foreign)++;
            u64 ring_pos = 0, high = 0;
            Rd(obj + 0x440, ring_pos);
            Rd(ctx + 0x298, high);
            const std::string key = fmt::format("view {} batch_special {} material_flag {} index {} other {}", s.view_type, e.batch_special != 0, e.material_flag, index_word & 0xffffffffull, other);
            ++s.slot_differs_by[key];
            if (s.slot_samples.size() < 6) {
                u64 expected_second = 0;
                Rd(obj + 0x2d0 + (index_word & 0xffffffffull) * 0xb0, expected_second);
                s.slot_samples.push_back(fmt::format("{}: context {:#x} object slot {:#x}; the context's second qword {:#x}, the object's second {:#x}, object word 0x440 {:#x}", key, held, wanted, high, expected_second, ring_pos));
            }
        }
    } else {
        not_applicable(4);
    }
    u32 token = 0;
    u64 cached = 0, set = 0;
    u32 set_count = 0;
    Rd(e.material + 0x1bc, token);
    Rd(e.material + 0x48, cached);
    Rd(e.material + 0x178, set);
    Rd(e.material + 0x180, set_count);
    if (e.material_flag == 1 && !bvar54 && e.batch_special == 0 && (token != 0 || cached != 0) && set != 0 && set_count != 0) {
        compare(5, s.render_device + 0x3790, set, size_t{std::min<u32>(set_count, 7)} * 16, "resource set at renderDev + 0x3790");
        {
            // FUN_01c92f30 ends with ctx[0x5c] = ctx[0x6c] = the constant buffer at renderDev + 0x2910 + 0xef0
            u64 constant_buffer = 0;
            Rd(s.render_device + 0x3800, constant_buffer);
            std::vector<u8> bytes(8);
            std::memcpy(bytes.data(), &constant_buffer, 8);
            s.overlay.push_back({0x2e0, bytes});
            s.overlay.push_back({0x360, std::move(bytes)});
        }
    } else {
        not_applicable(5);
    }

    // ---- audit d: the batch level and the texture pointers
    u64 mesh = 0, render_data = 0;
    Rd(e.batch_object + 0x20, mesh);
    Rd(mesh + 0x38, render_data);
    {
        // FUN_01c932d0: ctx[0x288] = ctx[0x308] = *(*(batch + 0x68) + 0x130)
        u64 pass_object = 0, expected_value = 0;
        if (Rd(e.batch_object + 0x68, pass_object) && Rd(pass_object + 0x130, expected_value)) {
            std::array<u8, 8> want{};
            std::memcpy(want.data(), &expected_value, 8);
            s.overlay.push_back({0x288, std::vector<u8>(want.begin(), want.end())});
            s.overlay.push_back({0x308, std::vector<u8>(want.begin(), want.end())});
            u64 a = 0, b = 0;
            if (Rd(ctx + 0x288, a) && Rd(ctx + 0x308, b)) {
                ++s.bind_checked[6];
                if (a == expected_value && b == expected_value) {
                    ++s.bind_equal[6];
                } else if (s.bind_sample_count[6]++ < 3) {
                    s.bind_samples.push_back(fmt::format("batch pass object (view type {}): context {:#x} / {:#x}, expected {:#x}", s.view_type, a, b, expected_value));
                }
            } else {
                ++s.bind_unreadable[6];
            }
        } else {
            ++s.bind_unreadable[6];
        }
    }
    if (render_data != 0) {
        u32 entries = 0;
        Rd(mesh + 0x68, entries);
        const u32 n = std::min<u32>(entries, 16);
        u64 inner = 0, defaults_holder = 0, default_object = 0;
        Rd(render_data + 0x20, inner);
        {
            const auto put64 = [&](u64 offset, u64 value) {
                std::vector<u8> bytes(8);
                std::memcpy(bytes.data(), &value, 8);
                s.overlay.push_back({offset, std::move(bytes)});
            };
            if (s.prev_instanced && e.batch_object != s.prev_batch) {
                // the end of an instanced batch clears these (ctx + 0x70.., 0x170, 0x1f0, 0x340, 0x800..0x830)
                s.overlay.push_back({0x70, std::vector<u8>(16, 0)});
                put64(0x170, 0);
                put64(0x1f0, 0);
                put64(0x340, 0);
                s.overlay.push_back({0x800, std::vector<u8>(0x30, 0)});
            }
            if (e.instance_object != 0) {
                // FUN_01ee4e50(inner, instance group, view type)
                const u64 group_object = e.instance_object;
                s32 ring_index = 0;
                u64 ring_array = 0, ring_entry = 0, view_holder = 0, view_object = 0, next_holder = 0;
                Rd(group_object + 0x20, ring_index);
                Rd(group_object + 0xb8, ring_array);
                Rd(ring_array + u64(ring_index) * 8, ring_entry);
                Rd(inner + 0x48, view_holder);
                Rd(view_holder + 0x38, view_object);
                u64 a = 0, b = 0;
                u32 c = 0, dd = 0;
                Rd(view_object + 0x20, a);
                Rd(ring_entry + 0x20, b);
                Rd(view_object + 0x40, c);
                Rd(ring_entry + 0x40, dd);
                put64(0x70, a);
                put64(0x78, b);
                {
                    std::vector<u8> bytes(8);
                    std::memcpy(bytes.data(), &c, 4);
                    std::memcpy(bytes.data() + 4, &dd, 4);
                    s.overlay.push_back({0x170, std::move(bytes)});
                }
                put64(0x1f0, 0);
                u64 uploader = 0;
                Rd(group_object + 0xf0, uploader);
                if (uploader != 0) {
                    u64 first = 0, buffer = 0;
                    Rd(uploader, first);
                    Rd(first + 0x10, buffer);
                    put64(0x340, buffer);
                }
                std::vector<u64> table;
                u64 v = 0;
                Rd(group_object + 0x1b0 + u64{s.view_type} * 8, v);
                table.push_back(v);
                v = 0;
                Rd(group_object + 0x1f8 + u64(ring_index) * 8, v);
                table.push_back(v);
                v = 0;
                Rd(group_object + 0x1f8 + u64(1 - ring_index) * 8, v);
                table.push_back(v);
                u32 views_count = 0;
                u64 views_array = 0;
                Rd(group_object + 0xe0, views_count);
                Rd(group_object + 0xd8, views_array);
                for (u32 i = 0; i < views_count && i < 60; ++i) {
                    u64 holder = 0, value = 0;
                    Rd(views_array + u64{i} * 8, holder);
                    if (holder != 0) {
                        Rd(holder + 0x28, value);
                    }
                    table.push_back(value);
                }
                std::vector<u8> bytes(table.size() * 8);
                std::memcpy(bytes.data(), table.data(), bytes.size());
                s.overlay.push_back({0x800, std::move(bytes)});
            }
            s.prev_instanced = e.instance_object != 0;
            s.prev_batch = e.batch_object;
        }
        {
            // per batch: ctx + 0x580 = *(*(*(inner + 0x48) + 0x38) + 0x28)
            u64 x = 0, y = 0, z = 0;
            Rd(inner + 0x48, x);
            Rd(x + 0x38, y);
            Rd(y + 0x28, z);
            std::vector<u8> bytes(8);
            std::memcpy(bytes.data(), &z, 8);
            s.overlay.push_back({0x580, std::move(bytes)});
        }
        const u64 global = NativeDraw::ImageBase() + (0x069a1b40ull - 0x1000000);
        u64 context_global = 0;
        Rd(global, context_global);
        Rd(context_global + 0x38, defaults_holder);
        Rd(defaults_holder + 0x20, default_object);
        u64 array = 0;
        u32 array_count = 0;
        Rd(inner + 0x60, array);
        Rd(inner + 0x68, array_count);
        bool any = false, all = true, unreadable = false;
        for (u32 j = 0; j < n; ++j) {
            u64 entry = 0, holder = 0, value = 0;
            if (j >= array_count || !Rd(array + u64{j} * 8, entry) || entry == default_object || entry == 0 || !Rd(entry + 0x38, holder) || holder == 0 || !Rd(holder + 0x28, value)) {
                continue; // the loop writes nothing for this entry
            }
            {
                std::vector<u8> bytes(8);
                std::memcpy(bytes.data(), &value, 8);
                s.overlay.push_back({0x6c0 + u64{j} * 8, std::move(bytes)});
            }
            u64 held = 0;
            if (!Rd(ctx + 0x6c0 + u64{j} * 8, held)) {
                unreadable = true;
                continue;
            }
            any = true;
            all &= held == value;
        }
        if (unreadable) {
            ++s.bind_unreadable[7];
        } else if (any) {
            ++s.bind_checked[7];
            if (all) {
                ++s.bind_equal[7];
            } else if (s.bind_sample_count[7]++ < 3) {
                s.bind_samples.push_back(fmt::format("batch table entries (view type {}, material {:#x}): the context differs from the mesh render data", s.view_type, e.material));
            }
        } else {
            ++s.bind_not_applicable[7];
        }
        // the index buffer view: *(ctx + 0x58) = *(*(*(inner + 0x50) + 0x38) + 0x20) (0 when inner + 0x50 is null), *(ctx + 0x60) = 0x39
        u64 view_holder = 0, view_inner = 0, view_value = 0;
        Rd(inner + 0x50, view_holder);
        if (view_holder != 0) {
            Rd(view_holder + 0x38, view_inner);
            Rd(view_inner + 0x20, view_value);
        }
        {
            std::vector<u8> bytes(8), type_bytes(8);
            std::memcpy(bytes.data(), &view_value, 8);
            const u64 index_type = 0x39;
            std::memcpy(type_bytes.data(), &index_type, 8);
            s.overlay.push_back({0x58, std::move(bytes)});
            s.overlay.push_back({0x60, std::move(type_bytes)});
        }
        u64 held_view = 0, held_type = 0;
        if (Rd(ctx + 0x58, held_view) && Rd(ctx + 0x60, held_type)) {
            ++s.bind_checked[8];
            if (held_view == view_value && held_type == 0x39) {
                ++s.bind_equal[8];
            } else if (s.bind_sample_count[8]++ < 3) {
                s.bind_samples.push_back(fmt::format("index buffer view (view type {}, material {:#x}): context {:#x} type {:#x}, expected {:#x} type 0x39", s.view_type, e.material, held_view, held_type, view_value));
            }
        } else {
            ++s.bind_unreadable[8];
        }
    } else {
        ++s.bind_unreadable[7];
        ++s.bind_unreadable[8];
    }
    {
        // FUN_01c92f30: the eight texture pointers go to ctx + 0xa40 (material + 0x170 & 1) and ctx + 0x740 (& 2)
        u32 texture_flags = 0, texture_count = 0;
        u64 texture_array = 0;
        Rd(e.material + 0x170, texture_flags);
        Rd(e.material + 0x198, texture_count);
        Rd(e.material + 400, texture_array);
        u32 token_d = 0;
        u64 cached_d = 0;
        Rd(e.material + 0x1bc, token_d);
        Rd(e.material + 0x48, cached_d);
        if (e.material_flag == 1 && !bvar54 && e.batch_special == 0 && (token_d != 0 || cached_d != 0) && (texture_flags & 3) != 0 && texture_array != 0) {
            const u64 global = NativeDraw::ImageBase() + (0x069a1b40ull - 0x1000000);
            u64 context_global = 0, h1 = 0, h2 = 0, default_texture_object = 0, default_render = 0, default_pointer = 0;
            Rd(global, context_global);
            Rd(context_global + 0x38, h1);
            Rd(h1 + 0x38, h2);
            Rd(h2 + 0x38, default_render);
            Rd(default_render + 0x48, default_pointer);
            (void)default_texture_object;
            std::array<u64, 8> expected_pointers;
            expected_pointers.fill(default_pointer);
            const u32 n = std::min<u32>(texture_count, 8);
            bool readable = true;
            for (u32 i = 0; i < n; ++i) {
                u64 object = 0, render = 0, pointer = 0;
                if (!Rd(texture_array + u64{i} * 8, object) || !Rd(object + 0x38, render)) {
                    readable = false;
                    break;
                }
                if (render != 0) {
                    Rd(render + 0x48, pointer);
                }
                expected_pointers[i] = pointer != 0 ? pointer : default_pointer;
            }
            if (!readable) {
                ++s.bind_unreadable[9];
            } else {
                for (u32 which = 0; which < 2; ++which) {
                    if ((texture_flags & (1u << which)) == 0) {
                        continue;
                    }
                    {
                        std::vector<u8> bytes(sizeof(u64) * 8);
                        std::memcpy(bytes.data(), expected_pointers.data(), bytes.size());
                        s.overlay.push_back({which == 0 ? 0xa40u : 0x740u, std::move(bytes)});
                    }
                    std::array<u64, 8> held{};
                    if (!RdBytes(ctx + (which == 0 ? 0xa40 : 0x740), held.data(), sizeof(held))) {
                        ++s.bind_unreadable[9];
                        continue;
                    }
                    ++s.bind_checked[9];
                    bool used_equal = true, unused_uniform = true;
                    for (u32 i = 0; i < 8; ++i) {
                        if (i < n) {
                            used_equal &= held[i] == expected_pointers[i];
                        } else {
                            unused_uniform &= held[i] == held[n < 8 ? n : 7];
                        }
                    }
                    if (n < 8) {
                        ++s.unused_texture_slot_values[held[n]];
                        if (!unused_uniform) {
                            ++s.unused_slots_not_uniform;
                        }
                    }
                    if (used_equal && unused_uniform) {
                        ++s.bind_equal[9];
                    } else if (s.bind_sample_count[9]++ < 3) {
                        std::string text;
                        for (u32 i = 0; i < 8; ++i) {
                            if (held[i] != expected_pointers[i]) {
                                text += fmt::format(" [{}] context {:#x} expected {:#x};", i, held[i], expected_pointers[i]);
                            }
                        }
                        s.bind_samples.push_back(fmt::format("texture pointers at +{:#x} (view type {}, material {:#x}, {} textures):{}", which == 0 ? 0xa40 : 0x740, s.view_type, e.material, n, text));
                    }
                }
            }
        } else {
            ++s.bind_not_applicable[9];
        }
    }
}
} // namespace

bool Enabled() {
    return Every() != 0;
}


void PassEnter(u64 render_device, u64 desc) {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    FinishLocked(s);
    if (s.frame % Every() != 0) {
        return;
    }
    u64 list = 0, flags = 0;
    u32 mask = 0;
    if (!Rd(desc, list) || !Rd(desc + 8, flags) || !Rd(desc + 0x20, mask) || list == 0) {
        return;
    }
    s.active = true;
    s.cursor = 0;
    s.render_device = render_device;
    s.current_valid = false;
    s.last_mesh_batch = s.last_mesh = s.last_group = 0;
    s.any_group = false;
    s.pass_has_special = s.pass_has_resource_set = false;
    s.prev_instanced = false;
    s.prev_batch = 0;
    s.batches_prepared = s.groups_processed = s.groups_skipped = s.last_group_batch_ordinal = s.last_group_draws = 0;
    s.baseline.clear();
    s.shadow.clear(); // a new pass: the shadow context starts from the live one and then follows the loop's writes
    if (Rd(render_device + 0x2d8, s.pass_ctx) && s.pass_ctx != 0) {
        s.baseline.resize(CtxBytes);
        if (!RdBytes(s.pass_ctx, s.baseline.data(), CtxBytes)) {
            s.baseline.clear();
        }
    }
    if (!s.baseline.empty()) {
        // the loop's prologue (DrawZPass, first lines): ctx[0x4480] = 4 with the dirty bit ctx[0x447c] |= 1, and ctx[0x68] = 0
        s.shadow = s.baseline;
        const u32 four = 4;
        std::memcpy(s.shadow.data() + 0x4480, &four, 4);
        s.shadow[0x447c] |= 1;
        const u64 zero = 0;
        std::memcpy(s.shadow.data() + 0x68, &zero, 8);
    }
    Rd(render_device + 0x10, s.rd_draws_before);
    Rd(render_device + 0x14, s.rd_triangles_before);
    s.view_type = static_cast<u32>(flags);
    {
        u8 flag14 = 0, flag1c = 0;
        Rd(desc + 0x14, flag14);
        Rd(desc + 0x1c, flag1c);
        s.pass_flags = u32{flag14} * 2 + u32{flag1c} * 4;
    }
    auto& v = s.views[s.view_type];
    ++v.passes;
    s32 indirect_mode = 0;
    if (!Rd(render_device + (s.view_type == 0 ? 0x38e0 : 0x38e4), indirect_mode)) {
        ++v.unreadable;
    }
    u64 batch_array = 0;
    u32 batch_count = 0;
    if (!Rd(list, batch_array) || !Rd(list + 8, batch_count) || batch_array == 0 || batch_count > 100000) {
        ++v.unreadable;
        return;
    }
    std::set<std::string> reasons;
    u64 default_kind = 0;
    Rd(NativeDraw::ImageBase() + (0x069ae968ull - 0x1000000), default_kind);
    const u8 flag_hi = static_cast<u8>(flags >> 32);
    const u8 flag_cvar = static_cast<u8>(flags >> 40);
    std::unordered_map<u64, u32> material_flags;
    std::unordered_map<u64, u64> material_sigs;
    std::vector<u8> prims;
    for (u32 b = 0; b < batch_count; ++b) {
        u64 batch = 0;
        if (!Rd(batch_array + u64(b) * 8, batch) || batch == 0) {
            ++v.unreadable;
            continue;
        }
        u16 batch_flags = 0, prim_count = 0;
        u64 group = 0, prim_array = 0;
        u32 base_instance = 0;
        if (!Rd(batch + 0x66, batch_flags) || !Rd(batch + 0x38, group) || !Rd(batch + 0x58, prim_array) || !Rd(batch + 100, prim_count) || !Rd(batch + 0x60, base_instance)) {
            ++v.unreadable;
            continue;
        }
        ++v.batches;
        if (flag_hi != 0 && (batch_flags & 2) != 0) {
            ++v.batches_skipped_by_flag;
            continue;
        }
        u64 mesh = 0;
        if (!Rd(batch + 0x20, mesh) || mesh == 0) {
            ++v.batches_skipped_no_mesh;
            continue;
        }
        s.last_mesh_batch = batch;
        s.last_mesh = mesh;
        s.last_group = group;
        ++s.batches_prepared;
        if (group == 0 && flag_cvar != 0) {
            ++v.batches_skipped_no_instances_with_flag; // the loop draws nothing for a batch without instance group when the pass flag byte 5 is set
            continue;
        }
        u16 mesh_word = 0;
        Rd(mesh + 0x78, mesh_word);
        {
            u64 render_data_q = 0, special_q = 0;
            Rd(mesh + 0x38, render_data_q);
            Rd(batch + 0x48, special_q);
            if (render_data_q == 0) {
                reasons.insert("mesh render data not created yet (the loop creates it at first use)");
            }
            s.pass_has_special |= special_q != 0;
            if (special_q != 0) {
                reasons.insert("batch with a special entry (a virtual call picks the material index)");
            }
            if (group != 0) {
                reasons.insert("batch with an instance group (the instance data upload of FUN_01ee4e50 is not modeled)");
            }
        }
        u32 instances = 0;
        u64 indirect_buffer = 0;
        u64 instance_object = 0;
        if (group != 0) {
            u64 inner = 0, object = 0, table = 0;
            u32 count = 0;
            const bool ok = Rd(group + 8, inner) && (flag_cvar == 0 || inner != 0) && Rd(inner + 0x38, object) && object != 0 && Rd(object + 0x28, table) && Rd(table + 0x48, count);
            if (!ok || count == 0) {
                ++v.batches_skipped_instance;
                ++s.groups_skipped;
                continue;
            }
            ++s.groups_processed;
            s.last_group_batch_ordinal = s.batches_prepared;
            s.last_group_draws = 0;
            instances = count;
            instance_object = object;
            s.any_group = true; // the batch is processed to its end: the clear after it applies
            if (indirect_mode != 0) {
                u64 args_holder = 0;
                Rd(inner + 0x38, args_holder);
                Rd(args_holder + 0x110, indirect_buffer); // the argument buffer object of the instance group
            }
        }
        if (prim_count == 0) {
            continue;
        }
        prims.resize(size_t{prim_count} * 0x38);
        if (!RdBytes(prim_array, prims.data(), prims.size())) {
            ++v.unreadable;
            continue;
        }
        for (u32 p = 0; p < prim_count; ++p) {
            const u8* record = prims.data() + size_t{p} * 0x38;
            u64 material = 0;
            std::memcpy(&material, record + 0x18, 8);
            ++v.primitives;
            u32 material_flag = 0;
            if (const auto it = material_flags.find(material); it != material_flags.end()) {
                material_flag = it->second;
            } else {
                u32 word = 0;
                if (!Rd(material + 0xd4, word)) {
                    ++v.unreadable;
                    continue;
                }
                material_flag = word & 1;
                material_flags.emplace(material, material_flag);
            }
            if ((mask & (2 - material_flag)) == 0) {
                ++v.primitives_filtered;
                continue;
            }
            s32 triangles = 0;
            u32 first = 0;
            std::memcpy(&first, record + 0x10, 4);
            std::memcpy(&triangles, record + 0x14, 4);
            Item item{Plain, u64(triangles) * 3, first, 1, b, p};
            item.material = material;
            item.triangles = static_cast<u32>(triangles);
            Rd(batch + 0x48, item.batch_special);
            item.batch_object = batch;
            item.instance_object = instance_object;
            item.prim_record = prim_array + u64{p} * 0x38;
            item.material_flag = material_flag;
            item.mesh_word = mesh_word;
            item.batch_flags = batch_flags;
            {
                const auto sig = material_sigs.find(material);
                if (sig != material_sigs.end()) {
                    item.material_sig = sig->second;
                } else {
                    // the fields of the material the variant search reads: +0x24, +0x28, +0xd0, +0xd4, +0x158, +0x15c, +0x164, +0x1bc (dwords)
                    u64 h = 0xcbf29ce484222325ull;
                    for (const u32 offset : {0x24u, 0x28u, 0xd0u, 0xd4u, 0x158u, 0x15cu, 0x164u, 0x1bcu}) {
                        u32 word = 0;
                        Rd(material + offset, word);
                        h = (h ^ word) * 0x100000001b3ull;
                    }
                    material_sigs.emplace(material, h);
                    item.material_sig = h;
                }
            }
            if (group != 0) {
                item.kind = indirect_mode != 0 ? Indirect : Instanced;
                item.instances = instances;
                if (item.kind == Indirect) {
                    item.count = indirect_buffer; // DrawIndexedIndirect(ctx, buffer, (primitive * 9 + view type) * 0x18)
                    item.first = (u64(p) * 9 + s.view_type) * 0x18;
                }
            }
            (item.kind == Plain ? v.expected_plain : item.kind == Instanced ? v.expected_instanced : v.expected_indirect)++;
            {
                const u64 key_q = ((item.material_sig * 0x9E3779B97F4A7C15ull ^ s.view_type) * 31 + (item.batch_flags & 0x10) * 8 + item.kind) * 0x100000001b3ull + item.mesh_word * 131u + s.pass_flags;
                if (s.variants.find(key_q) == s.variants.end()) {
                    reasons.insert("variant key not seen yet (the loop would create the variant)");
                }
                s.pass_has_resource_set |= material_flag == 1;
                u64 kind18 = 0, obj_q = 0;
                Rd(material + 0x18, kind18);
                Rd(material + 0x38, obj_q);
                if (kind18 == default_kind) {
                    reasons.insert("material of the engine's default kind");
                }
                if (obj_q == 0) {
                    reasons.insert("material without a render object");
                }
            }
            s.last_group_draws += group != 0 ? 1 : 0;
            s.expected.push_back(item);
        }
    }
    v.all_draws += s.expected.size();
    if (reasons.empty()) {
        ++v.qualifying_passes;
        v.qualifying_draws += s.expected.size();
    } else {
        for (const auto& reason : reasons) {
            ++v.decline_passes[reason];
            v.decline_draws[reason] += s.expected.size();
        }
    }
}

void NoteDraw(u64 caller, u32 kind, u64 count, u64 first, u64 instances, u64 vertex_shader, u64 pixel_shader, u64 ctx) {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (!InPass(caller)) {
        s.current_valid = false; // a draw of another pass: the shadow generator has nothing for it
        return;
    }
    s.last_caller = caller;
    s.entry_vs = vertex_shader;
    s.entry_ps = pixel_shader;
    if (s.active) {
        CheckBinding(s, kind, count, first, instances, ctx);
        Item item{kind, count, first, instances, 0, 0};
        item.vertex_shader = vertex_shader;
        item.pixel_shader = pixel_shader;
        s.observed.push_back(item);
    }
}

void RealRecord(u64 ctx, u64 seq, u32 kind, u64 a1, u64 a2, u64 a3) {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (!s.active || !s.current_valid) {
        return;
    }
    if (s.baseline.empty()) {
        ++s.shadow_no_baseline;
        return;
    }
    if (s.shadow.size() != s.baseline.size()) {
        s.shadow = s.baseline;
    }
    for (const auto& o : s.overlay) {
        if (o.offset + o.bytes.size() <= s.shadow.size()) {
            std::memcpy(s.shadow.data() + o.offset, o.bytes.data(), o.bytes.size());
        }
    }
    // the shader pair: from the key table the audit learns from the loop; a key the table does not know yet is learned from this draw (and counted)
    u64 real_vs = s.entry_vs, real_ps = s.entry_ps;
    {
        u64 after_vs = 0, after_ps = 0;
        Rd(ctx + 0x4698, after_vs);
        Rd(ctx + 0x4690, after_ps);
        s.pair_changed_inside_tag += (after_vs != real_vs || after_ps != real_ps) ? 1 : 0;
    }
    const Item& e = s.current;
    s.current_ctx_differs = ctx != s.pass_ctx;
    s.draws_on_another_context += s.current_ctx_differs ? 1 : 0;
    const u64 key = ((e.material_sig * 0x9E3779B97F4A7C15ull ^ s.view_type) * 31 + (e.batch_flags & 0x10) * 8 + e.kind) * 0x100000001b3ull + e.mesh_word * 131u + s.pass_flags;
    // the variant object the loop chose for this primitive (it stored it behind the primitive record); the generator reads the shader pair from the object of the key
    u64 live_variant = 0, record_data = 0;
    const bool variant_readable = Rd(e.prim_record + 0x28, record_data) && Rd(record_data + 0x48, live_variant) && live_variant != 0;
    u64 vs = real_vs, ps = real_ps;
    // the loop selects a variant only for a material whose +0x18 is not the engine's default (DAT_069ae968); for the others the context keeps the shader pair of the previous draw
    u64 material_kind = 0, default_kind = 0;
    Rd(e.material + 0x18, material_kind);
    Rd(NativeDraw::ImageBase() + (0x069ae968ull - 0x1000000), default_kind);
    const bool keeps_previous_shaders = material_kind == default_kind;
    if (keeps_previous_shaders) {
        ++s.draws_keeping_the_previous_shaders;
        u64 held_vs = 0, held_ps = 0;
        Rd(reinterpret_cast<u64>(s.shadow.data()) + 0x4698, held_vs);
        Rd(reinterpret_cast<u64>(s.shadow.data()) + 0x4690, held_ps);
        vs = held_vs;
        ps = held_ps;
        s.draws_keeping_right += (vs == real_vs && ps == real_ps) ? 1 : 0;
    } else if (!variant_readable) {
        ++s.variant_unreadable;
    } else {
        auto [it, fresh] = s.variants.try_emplace(key);
        auto& entry = it->second;
        if (fresh) {
            entry.variant = live_variant;
            ++s.variant_cold;
        }
        ++entry.draws;
        u64 object_vs = 0, object_ps = 0;
        if (Rd(entry.variant + 8, object_vs) && Rd(entry.variant + 0x18, object_ps)) {
            if (entry.last_vs != 0 && (entry.last_vs != object_vs || entry.last_ps != object_ps)) {
                ++entry.pair_changes;
            }
            entry.last_vs = object_vs;
            entry.last_ps = object_ps;
            const bool variant_right = entry.variant == live_variant;
            const bool pair_right = object_vs == real_vs && object_ps == real_ps;
            ++(variant_right ? s.variant_right : s.variant_wrong);
            ++(pair_right ? s.variant_pair_right : s.variant_pair_wrong);
            {
                auto& site = s.callers[s.last_caller];
                ++site[0];
                site[1] += pair_right ? 0 : 1;
            }
            {
                auto& bucket = s.wrong_over_time[s.frame / 64];
                ++bucket.first;
                bucket.second += pair_right ? 0 : 1;
            }
            {
                auto& cell = s.wrong_by_fields[fmt::format("view {} batch flags {:#x} mesh word {:#x} material flag {} kind {}", s.view_type, e.batch_flags, e.mesh_word, e.material_flag, e.kind)];
                ++cell.first;
                if (s.view_type == 0 && e.batch_flags == 0x6 && e.mesh_word == 0x1a40) {
                    std::array<u32, 14> record_words{};
                    std::array<u32, 40> batch_words{};
                    RdBytes(e.prim_record, record_words.data(), 14 * 4);
                    RdBytes(e.batch_object, batch_words.data(), 40 * 4);
                    for (u32 w = 0; w < 14; ++w) {
                        (pair_right ? s.family_fields[fmt::format("primitive record +{:#x}", w * 4)].right : s.family_fields[fmt::format("primitive record +{:#x}", w * 4)].wrong)[record_words[w]]++;
                    }
                    for (u32 w = 0; w < 40; ++w) {
                        (pair_right ? s.family_fields[fmt::format("batch +{:#x}", w * 4)].right : s.family_fields[fmt::format("batch +{:#x}", w * 4)].wrong)[batch_words[w]]++;
                    }
                }
                if (!pair_right) {
                    (s.current_ctx_differs ? s.wrong_on_another_context : s.wrong_on_the_pass_context)++;
                    ++cell.second;
                    ++s.wrong_total;
                    s.wrong_equal_previous += (real_vs == s.prev_live_vs && real_ps == s.prev_live_ps) ? 1 : 0;
                }
            }
            if (e.batch_special != 0) {
                ++s.draws_with_special_entry;
                s.draws_with_special_entry_pair_right += pair_right ? 1 : 0;
            }
            if (!variant_right || !pair_right) {
                const std::string kind = fmt::format("{}{}; batch special entry {}; material flag {}; draw kind {}", variant_right ? "same variant object" : "a different variant object than the key first had", pair_right ? "" : ", the object's pair is not the context's", e.batch_special != 0 ? "yes" : "no", e.material_flag, e.kind);
                ++s.variant_wrong_kinds[kind];
                if (!pair_right && s.variant_samples.size() < 10) {
                    // does another primitive of the same batch hold a variant whose pair is the context's pair? (the loop may have bound that one)
                    u64 prim_array_base = e.prim_record - u64{e.primitive} * 0x38;
                    u16 prim_count = 0;
                    Rd(e.batch_object + 100, prim_count);
                    std::string holder_text = "no primitive of the batch";
                    for (u32 q = 0; q < prim_count && q < 4096; ++q) {
                        u64 data = 0, variant_q = 0, qvs = 0, qps = 0;
                        if (Rd(prim_array_base + u64{q} * 0x38 + 0x28, data) && Rd(data + 0x48, variant_q) && variant_q != 0 && Rd(variant_q + 8, qvs) && Rd(variant_q + 0x18, qps) && qvs == real_vs && qps == real_ps) {
                            u32 k[4] = {};
                            Rd(variant_q + 0x20, k[0]);
                            Rd(variant_q + 0x24, k[1]);
                            Rd(variant_q + 0x28, k[2]);
                            Rd(variant_q + 0x2c, k[3]);
                            holder_text = fmt::format("primitive {} of the batch (variant {:#x}, key {:#x} {:#x} {:#x} {:#x})", q, variant_q, k[0], k[1], k[2], k[3]);
                            break;
                        }
                    }
                    u32 k[4] = {};
                    Rd(live_variant + 0x20, k[0]);
                    Rd(live_variant + 0x24, k[1]);
                    Rd(live_variant + 0x28, k[2]);
                    Rd(live_variant + 0x2c, k[3]);
                    std::array<u64, 4> head{};
                    RdBytes(live_variant, head.data(), 32);
                    // the per batch flag bVar54 of the loop, and whether the context's pair is an entry of the engine's default shader tables (renderDev + 0x3f0 / + 0x8e0, 16 bytes per entry)
                    u64 mesh_q = 0, render_data_q = 0, inner_q = 0, array_q = 0, first_q = 0, holder_q = 0, target_q = 0, g_ctx = 0, g1 = 0, g2 = 0;
                    u32 array_count_q = 0;
                    Rd(e.batch_object + 0x20, mesh_q);
                    Rd(mesh_q + 0x38, render_data_q);
                    Rd(render_data_q + 0x20, inner_q);
                    Rd(inner_q + 0x68, array_count_q);
                    Rd(inner_q + 0x60, array_q);
                    Rd(NativeDraw::ImageBase() + (0x069a1b40ull - 0x1000000), g_ctx);
                    Rd(g_ctx + 0x38, g1);
                    Rd(g1 + 0x20, g2);
                    bool bvar54 = false;
                    if (array_count_q != 0 && Rd(array_q, first_q) && first_q != g2 && first_q != 0) {
                        Rd(first_q + 0x38, target_q);
                        bvar54 = target_q != 0;
                    }
                    (void)holder_q;
                    std::string table_text = "not in the default tables";
                    for (u32 t = 0; t < 64; ++t) {
                        u64 tv = 0, tp = 0;
                        const bool vs_hit = Rd(s.render_device + 0x3f8 + u64{t} * 0x10, tv) && tv == real_vs;
                        const bool ps_hit = Rd(s.render_device + 0x8e8 + u64{t} * 0x10, tp) && tp == real_ps;
                        if (vs_hit || ps_hit) {
                            table_text = fmt::format("{}{} at index {}", vs_hit ? "vs in the table at +0x3f0" : "", ps_hit ? " ps in the table at +0x8e0" : "", t);
                            break;
                        }
                    }
                    u64 dbg_m18 = 0, dbg_dat = 0; Rd(e.material + 0x18, dbg_m18); Rd(NativeDraw::ImageBase() + (0x069ae968ull - 0x1000000), dbg_dat);
                    s.variant_samples.push_back(fmt::format("material+0x18 {:#x} DAT_069ae968 {:#x}; bVar54 {}; {}; primitive {} of {}; the context's pair is held by {}; this primitive's variant {:#x} key {:#x} {:#x} {:#x} {:#x} head {:#x} {:#x} {:#x} {:#x}", dbg_m18, dbg_dat, bvar54, table_text, e.primitive, prim_count, holder_text, live_variant, k[0], k[1], k[2], k[3], head[0], head[1], head[2], head[3]));
                }
                if (false && s.variant_samples.size() < 6) {
                    // which variant of the engine's cache holds the context's pair, and what are the key fields of it and of the stored one?
                    u64 cache = 0;
                    u32 cache_count = 0;
                    Rd(s.render_device + 0x1c60, cache);
                    Rd(s.render_device + 0x1c68, cache_count);
                    std::string holder = "none in the cache";
                    for (u32 i = 0; i < cache_count && i < 20000; ++i) {
                        u64 v = 0, cvs = 0, cps = 0;
                        if (Rd(cache + u64{i} * 8, v) && Rd(v + 8, cvs) && Rd(v + 0x18, cps) && cvs == real_vs && cps == real_ps) {
                            u32 k0 = 0, k1 = 0, k2 = 0, k3 = 0;
                            Rd(v + 0x10, k0);
                            Rd(v + 0x14, k1);
                            Rd(v + 0x18, k2);
                            Rd(v + 0x1c, k3);
                            holder = fmt::format("variant {:#x} (cache entry {}) key fields {:#x} {:#x} {:#x} {:#x}", v, i, k0, k1, k2, k3);
                            break;
                        }
                    }
                    u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    Rd(live_variant + 0x10, s0);
                    Rd(live_variant + 0x14, s1);
                    Rd(live_variant + 0x18, s2);
                    Rd(live_variant + 0x1c, s3);
                    s.variant_samples.push_back(fmt::format("the context's pair {:#x}/{:#x} is held by: {}; the stored variant {:#x} has key fields {:#x} {:#x} {:#x} {:#x}; cache size {}", real_vs, real_ps, holder, live_variant, s0, s1, s2, s3, cache_count));
                }
                if (false) {
                    s.variant_samples.push_back(fmt::format("{}: key {:#x} view type {} first variant {:#x} live variant {:#x}; object pair {:#x}/{:#x} live pair {:#x}/{:#x}", kind, key, s.view_type, entry.variant, live_variant, object_vs, object_ps, real_vs, real_ps));
                }
            }
            vs = object_vs;
            ps = object_ps;
        }
        // the old key table (shader pair learned from the loop) is kept for comparison
        if (const auto old = s.variant_table.find(key); old != s.variant_table.end()) {
            ++(old->second.first == real_vs && old->second.second == real_ps ? s.key_hit : s.key_wrong);
        } else {
            s.variant_table.emplace(key, std::make_pair(real_vs, real_ps));
            ++s.key_cold;
        }
    }
    s.prev_live_vs = real_vs;
    s.prev_live_ps = real_ps;
    std::memcpy(s.shadow.data() + 0x4698, &vs, 8);
    std::memcpy(s.shadow.data() + 0x4690, &ps, 8);
    {
        s.live_copy.resize(CtxBytes);
        if (RdBytes(ctx, s.live_copy.data(), CtxBytes)) {
            for (size_t off = 0; off + 8 <= CtxBytes; off += 8) {
                u64 x, y;
                std::memcpy(&x, s.live_copy.data() + off, 8);
                std::memcpy(&y, s.shadow.data() + off, 8);
                if (x != y) {
                    auto& w = s.context_word_diffs[static_cast<u32>(off)];
                    if (w.draws++ == 0) {
                        w.live = x;
                        w.shadow = y;
                    }
                }
            }
        }
    }
    const u64 shadow_seq = NativeDraw::ShadowBuildRecord(reinterpret_cast<u64>(s.shadow.data()), kind, a1, a2, a3);
    if (shadow_seq == 0) {
        ++s.shadow_failed;
        return;
    }
    ++s.shadow_built;
    std::vector<std::string> differences;
    if (NativeDraw::CompareRecords(seq, shadow_seq, differences) == 0) {
        ++s.shadow_equal;
        return;
    }
    ++s.shadow_differ;
    for (const auto& text : differences) {
        // group by the kind of difference: the text up to the first number
        std::string group = text;
        for (size_t i = 0; i < group.size(); ++i) {
            if (group[i] >= '0' && group[i] <= '9') {
                group.resize(i);
                break;
            }
        }
        ++s.shadow_difference_kinds[group];
    }
    if (s.shadow_samples.size() < 12) {
        std::string joined = fmt::format("view type {} material {:#x}:", s.view_type, e.material);
        for (size_t i = 0; i < differences.size() && i < 4; ++i) {
            joined += " [" + differences[i] + "]";
        }
        s.shadow_samples.push_back(joined);
    }
}

bool WrapEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_ZPASS_ENDSTATE");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on && Enabled();
}

void PassExit(u64 render_device) {
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (!s.active || s.shadow.empty() || s.pass_ctx == 0) {
        return;
    }
    // the loop's own side effect on the render device: the draw and triangle counters
    {
        s32 draws = 0, triangles = 0;
        Rd(render_device + 0x10, draws);
        Rd(render_device + 0x14, triangles);
        u64 expected_triangles = 0, expected_draws = 0;
        for (const auto& e : s.expected) {
            if (e.kind == Instanced) {
                continue;
            }
            ++expected_draws;
            expected_triangles += e.triangles;
        }
        if (static_cast<u64>(draws - s.rd_draws_before) == expected_draws && static_cast<u64>(triangles - s.rd_triangles_before) == expected_triangles) {
            ++s.end_counter_ok;
        } else {
            ++s.end_counter_bad;
            if (s.end_samples.size() < 4) {
                s.end_samples.push_back(fmt::format("counters: view type {}: draws +{} (expected {}), triangles +{} (expected {})", s.view_type, draws - s.rd_draws_before, expected_draws, triangles - s.rd_triangles_before, expected_triangles));
            }
        }
    }
    if (s.last_mesh_batch != 0) {
        // the batch level state of the last batch the loop prepared, and the clear after an instanced batch
        u64 render_data = 0, inner = 0, x = 0, y = 0, z = 0;
        Rd(s.last_mesh + 0x38, render_data);
        Rd(render_data + 0x20, inner);
        Rd(inner + 0x48, x);
        Rd(x + 0x38, y);
        Rd(y + 0x28, z);
        std::memcpy(s.shadow.data() + 0x580, &z, 8);
        if (s.any_group) {
            std::memset(s.shadow.data() + 0x70, 0, 16);
            std::memset(s.shadow.data() + 0x170, 0, 8);
            std::memset(s.shadow.data() + 0x1f0, 0, 8);
            std::memset(s.shadow.data() + 0x340, 0, 8);
            std::memset(s.shadow.data() + 0x800, 0, 0x30);
        }
    }
    {
        // the loop's epilogue (the last lines of DrawZPass): ctx[0x980] = 0; ctx[0x46c8] = renderDev[0x3bf0] with the dirty byte ctx[0x46d8] = 1 when it changed
        const u64 zero = 0;
        std::memcpy(s.shadow.data() + 0x980, &zero, 8);
        u64 default_state = 0, held_state = 0;
        Rd(render_device + 0x3bf0, default_state);
        std::memcpy(&held_state, s.shadow.data() + 0x46c8, 8);
        if (held_state != default_state) {
            std::memcpy(s.shadow.data() + 0x46c8, &default_state, 8);
            s.shadow[0x46d8] = 1;
        }
    }
    {
        u64 live70 = 0;
        Rd(s.pass_ctx + 0x70, live70);
        u64 shadow70 = 0;
        std::memcpy(&shadow70, s.shadow.data() + 0x70, 8);
        const std::string pattern = fmt::format("instanced batches processed {}, skipped {}, the last processed one is {} the last prepared batch, draws of the last processed one {}", s.groups_processed > 0 ? "some" : "none", s.groups_skipped > 0 ? "some" : "none",
                                                s.last_group_batch_ordinal == s.batches_prepared ? "also" : "not", s.last_group_draws > 0 ? "some" : "none");
        auto& cell = s.group_patterns[pattern];
        ++cell.first;
        cell.second += (live70 != 0 && shadow70 == 0) ? 1 : 0;
    }
    {
        u64 live2e0 = 0, shadow2e0 = 0;
        Rd(s.pass_ctx + 0x2e0, live2e0);
        std::memcpy(&shadow2e0, s.shadow.data() + 0x2e0, 8);
        auto& cell = s.slot2e0_patterns[fmt::format("pass with a special batch entry: {}, with a material that has a resource set: {}", s.pass_has_special ? "yes" : "no", s.pass_has_resource_set ? "yes" : "no")];
        ++cell.first;
        cell.second += live2e0 != shadow2e0 ? 1 : 0;
    }
    s.live_copy.resize(CtxBytes);
    if (!RdBytes(s.pass_ctx, s.live_copy.data(), CtxBytes)) {
        ++s.end_unreadable;
        return;
    }
    ++s.end_passes;
    bool clean = true;
    State::PassDiff pass_diff{{}, s.pass_has_special};
    for (size_t off = 0; off + 8 <= CtxBytes; off += 8) {
        u64 x, y;
        std::memcpy(&x, s.live_copy.data() + off, 8);
        std::memcpy(&y, s.shadow.data() + off, 8);
        if (x == y) {
            continue;
        }
        pass_diff.offsets.push_back(static_cast<u32>(off));
        auto& w = s.end_diffs[static_cast<u32>(off)];
        if (w.passes++ == 0) {
            w.live = x;
            w.shadow = y;
        }
        // words the draw calls themselves keep changing (differing at more than a fifth of the draw entries) are not the loop's: they do not make a pass unclean
        const auto noise = s.context_word_diffs.find(static_cast<u32>(off));
        if (noise == s.context_word_diffs.end() || noise->second.draws * 5 < s.shadow_built) {
            clean = false;
        }
    }
    s.end_passes_clean += clean ? 1 : 0;
    if (s.end_pass_diffs.size() < 100000) {
        s.end_pass_diffs.push_back(std::move(pass_diff));
    }
}

void FrameBoundary() {
    if (!Enabled()) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    FinishLocked(s);
    ++s.frame;
}

nlohmann::json Report() {
    nlohmann::json out;
    out["every_nth_frame"] = Every();
    if (!Enabled()) {
        out["note"] = "start the game with TLG_ZPASS_AUDIT=1 (every frame) or N (one frame in N)";
        return out;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    auto& views = out["by_view_type"] = nlohmann::json::object();
    for (const auto& [type, v] : s.views) {
        views[std::to_string(type)] = {{"passes", v.passes},
                                       {"batches", v.batches},
                                       {"batches_skipped_by_the_view_flag", v.batches_skipped_by_flag},
                                       {"batches_skipped_instance_count_unreadable_or_zero", v.batches_skipped_instance},
                                       {"batches_skipped_no_instance_group_while_the_pass_flag_byte_5_is_set", v.batches_skipped_no_instances_with_flag},
                                       {"batches_skipped_no_mesh", v.batches_skipped_no_mesh},
                                       {"primitives", v.primitives},
                                       {"primitives_filtered_by_the_pass_mask", v.primitives_filtered},
                                       {"expected_plain_draws", v.expected_plain},
                                       {"expected_instanced_draws_not_hooked", v.expected_instanced},
                                       {"expected_indirect_draws", v.expected_indirect},
                                       {"observed_plain_draws_from_the_loop", v.observed_plain},
                                       {"observed_indirect_draws_from_the_loop", v.observed_indirect},
                                       {"plain_matched", v.matched_plain},
                                       {"plain_expected_but_not_issued", v.missing_plain},
                                       {"plain_issued_but_not_expected", v.extra_plain},
                                       {"indirect_matched", v.matched_indirect},
                                       {"indirect_expected_but_not_issued", v.missing_indirect},
                                       {"indirect_issued_but_not_expected", v.extra_indirect},
                                       {"unreadable_snapshot_fields", v.unreadable},
                                       {"generator_would_take_passes", v.qualifying_passes},
                                       {"generator_would_take_draws", v.qualifying_draws},
                                       {"draws_in_all_passes", v.all_draws},
                                       {"passes_declined_by_reason", v.decline_passes},
                                       {"draws_of_those_passes_by_reason", v.decline_draws}};
    }
    out["first_mismatches"] = s.samples;
    {
        auto& shadow = out["shadow_generator"] = nlohmann::json::object();
        shadow["records_built_from_the_patched_pass_context"] = s.shadow_built;
        shadow["equal_to_the_record_of_the_live_context"] = s.shadow_equal;
        shadow["different"] = s.shadow_differ;
        shadow["build_failed"] = s.shadow_failed;
        shadow["no_baseline_copy_of_the_context"] = s.shadow_no_baseline;
        shadow["shader_pair_from_the_learned_key_table"] = {{"keys_learned", s.variant_table.size()}, {"draws_predicted_right", s.key_hit}, {"draws_predicted_wrong", s.key_wrong}, {"draws_on_a_key_not_seen_before", s.key_cold}};
        shadow["difference_kinds"] = s.shadow_difference_kinds;
        {
            auto& end = out["end_state_audit"] = nlohmann::json::object();
            end["passes_compared"] = s.end_passes;
            {
                // classified now, with the final rates: a word is draw call noise when it differs at more than a fifth of the draw entries
                u64 identical = 0, only_noise = 0, unclean = 0, unclean_special = 0, unclean_plain = 0, plain_passes = 0, special_passes = 0;
                for (const auto& pd : s.end_pass_diffs) {
                    (pd.special ? special_passes : plain_passes)++;
                    if (pd.offsets.empty()) {
                        ++identical;
                        continue;
                    }
                    bool noise_only = true;
                    for (const u32 off : pd.offsets) {
                        const auto n = s.context_word_diffs.find(off);
                        if (n == s.context_word_diffs.end() || n->second.draws * 5 < s.shadow_built) {
                            noise_only = false;
                            break;
                        }
                    }
                    if (noise_only) {
                        ++only_noise;
                    } else {
                        ++unclean;
                        (pd.special ? unclean_special : unclean_plain)++;
                    }
                }
                end["final_classification"] = {{"passes", s.end_pass_diffs.size()},
                                               {"context_identical_to_the_generators", identical},
                                               {"differs_only_in_words_the_draw_calls_change_themselves", only_noise},
                                               {"differs_in_other_words", unclean},
                                               {"of_those_passes_with_a_special_batch_entry", unclean_special},
                                               {"of_those_passes_without_one", unclean_plain},
                                               {"passes_without_a_special_batch_entry", plain_passes},
                                               {"passes_with_a_special_batch_entry", special_passes}};
            }
            end["passes_where_every_differing_word_is_one_the_draw_calls_change_anyway_at_the_time"] = s.end_passes_clean;
            end["passes_with_the_render_device_counters_as_predicted"] = s.end_counter_ok;
            end["passes_with_other_counters"] = s.end_counter_bad;
            end["unreadable"] = s.end_unreadable;
            nlohmann::json only_end = nlohmann::json::array(), also_noise = nlohmann::json::array();
            std::vector<std::pair<u32, State::EndDiff>> words(s.end_diffs.begin(), s.end_diffs.end());
            std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) { return a.second.passes > b.second.passes; });
            for (const auto& [off, w] : words) {
                nlohmann::json entry = {{"context_offset", fmt::format("{:#x}", off)}, {"passes", w.passes}, {"live_after_the_loop", fmt::format("{:#x}", w.live)}, {"generator_state", fmt::format("{:#x}", w.shadow)}};
                const auto noise = s.context_word_diffs.find(off);
                if (noise == s.context_word_diffs.end() || noise->second.draws * 5 < s.shadow_built) {
                    if (only_end.size() < 40) only_end.push_back(entry);
                } else if (also_noise.size() < 20) {
                    also_noise.push_back(entry);
                }
            }
            end["words_that_differ_at_the_end_but_not_at_the_draw_calls"] = only_end;
            end["words_that_also_differ_at_the_draw_calls"] = also_noise;
            end["samples"] = s.end_samples;
            nlohmann::json patterns = nlohmann::json::array();
            for (const auto& [name, counts] : s.group_patterns) {
                patterns.push_back(fmt::format("{}: {} passes, {} of them end with context +0x70 still set", name, counts.first, counts.second));
            }
            end["instanced_batch_patterns"] = patterns;
            nlohmann::json slots = nlohmann::json::array();
            for (const auto& [name, counts] : s.slot2e0_patterns) {
                slots.push_back(fmt::format("{}: {} passes, {} end with context +0x2e0 different", name, counts.first, counts.second));
            }
            end["constant_buffer_slot_0x2e0_by_pass_kind"] = slots;
            nlohmann::json watch = nlohmann::json::array();
            for (u32 w = 0; w < 4; ++w) {
                watch.push_back(fmt::format("context +{:#x}: changed between {} of {} consecutive plain draws of a batch (the loop writes nothing there between them)", State::WatchOffsets[w], s.watch_changes[w], s.watch_pairs[w]));
            }
            end["words_the_draw_calls_change_themselves"] = watch;
        }
        {
            u64 multi = 0, changing = 0, changes = 0;
            for (const auto& [key, v] : s.variants) {
                changing += v.pair_changes != 0 ? 1 : 0;
                changes += v.pair_changes;
                (void)multi;
            }
            {
                nlohmann::json fields = nlohmann::json::array();
                std::vector<std::pair<std::string, std::pair<u64, u64>>> cells(s.wrong_by_fields.begin(), s.wrong_by_fields.end());
                std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
                for (size_t i = 0; i < cells.size() && i < 8 && cells[i].second.second != 0; ++i) {
                    fields.push_back({{"fields", cells[i].first}, {"draws", cells[i].second.first}, {"wrong", cells[i].second.second}});
                }
                shadow["wrong_pair_by_fields"] = fields;
            }
            {
                nlohmann::json fields = nlohmann::json::array();
                for (const auto& [name, v] : s.family_fields) {
                    // a field separates right from wrong draws when no value is in both sets
                    bool shared = false;
                    for (const auto& [value, count] : v.wrong) {
                        shared |= v.right.count(value) != 0;
                    }
                    if (!shared && !v.wrong.empty() && !v.right.empty()) {
                        nlohmann::json w = nlohmann::json::array(), r = nlohmann::json::array();
                        for (const auto& [value, count] : v.wrong) {
                            w.push_back(fmt::format("{:#x} x{}", value, count));
                            if (w.size() >= 5) break;
                        }
                        for (const auto& [value, count] : v.right) {
                            r.push_back(fmt::format("{:#x} x{}", value, count));
                            if (r.size() >= 5) break;
                        }
                        fields.push_back({{"field", name}, {"values_of_wrong_draws", w}, {"values_of_right_draws", r}});
                    }
                }
                shadow["fields_that_separate_the_wrong_draws_of_the_family"] = fields;
            }
            {
                nlohmann::json over_time = nlohmann::json::array();
                for (const auto& [bucket, counts] : s.wrong_over_time) {
                    over_time.push_back(fmt::format("frames {}..{}: {} draws, {} wrong", bucket * 64, bucket * 64 + 63, counts.first, counts.second));
                }
                shadow["wrong_pairs_over_time"] = over_time;
            }
            {
                nlohmann::json sites = nlohmann::json::array();
                for (const auto& [site, counts] : s.callers) {
                    sites.push_back(fmt::format("call site {:#x}: {} draws, {} wrong", site, counts[0], counts[1]));
                }
                shadow["draw_call_sites_in_the_pass"] = sites;
            }
            shadow["first_use_rule"] = {{"keys_with_a_variant_object", s.variants.size()},
                                        {"draws_where_the_keys_variant_object_is_the_one_the_loop_chose", s.variant_right},
                                        {"draws_where_it_is_another_object", s.variant_wrong},
                                        {"draws_where_the_pair_read_from_the_object_is_the_contexts_pair", s.variant_pair_right},
                                        {"draws_where_it_is_not", s.variant_pair_wrong},
                                        {"draws_without_a_readable_variant", s.variant_unreadable},
                                        {"keys_whose_object_changed_its_shader_pair_over_time", changing},
                                        {"changes_of_the_pair_in_all_objects", changes},
                                        {"draws_where_the_context_pair_changed_between_the_call_entry_and_the_end_of_the_tag", s.pair_changed_inside_tag},
                                        {"wrong_draws", s.wrong_total},
                                        {"draws_issued_on_another_device_context_than_the_one_at_the_pass_start", s.draws_on_another_context},
                                        {"wrong_draws_on_another_context", s.wrong_on_another_context},
                                        {"wrong_draws_on_the_pass_context", s.wrong_on_the_pass_context},
                                        {"wrong_draws_whose_pair_is_the_previous_draws_pair", s.wrong_equal_previous},
                                        {"draws_of_a_material_that_keeps_the_previous_draws_shaders", s.draws_keeping_the_previous_shaders},
                                        {"of_those_the_shadow_contexts_previous_pair_is_the_contexts_pair", s.draws_keeping_right},
                                        {"draws_of_batches_with_a_special_entry", s.draws_with_special_entry},
                                        {"of_those_with_the_pair_from_the_object_right", s.draws_with_special_entry_pair_right},
                                        {"wrong_by_kind", s.variant_wrong_kinds},
                                        {"samples", s.variant_samples}};
        }
        {
            std::vector<std::pair<u32, State::WordDiff>> words(s.context_word_diffs.begin(), s.context_word_diffs.end());
            std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) { return a.second.draws > b.second.draws; });
            nlohmann::json list = nlohmann::json::array();
            for (size_t i = 0; i < words.size() && i < 60; ++i) {
                list.push_back({{"context_offset", fmt::format("{:#x}", words[i].first)}, {"draws_that_differ", words[i].second.draws}, {"live_sample", fmt::format("{:#x}", words[i].second.live)}, {"patched_copy_sample", fmt::format("{:#x}", words[i].second.shadow)}});
            }
            shadow["context_words_that_differ_most_often"] = list;
        }
        shadow["first_differences"] = s.shadow_samples;
    }
    {
        static constexpr const char* names[State::Checks] = {"material block 1: context 0x980..0x9c8 = material render object 0x450..0x498", "material block 2: context 0x9d0..0xa18 = object 0x498..0x4e0 (when object + 0x444 is set)",
                                                             "shader resource views: context + 0x790 = the array at object + 0x4f0", "shader resource views 2: context + 0xb38 = the array at object + 0x4e0",
                                                             "constant buffer slots: context + 0x290 / 0x310 = object 0xc0 + index * 0xb0", "resource set: render device + 0xe80 = material + 0x178 (no special batch entry, bound by FUN_01c92f30)",
                                                             "batch level (FUN_01c932d0): context + 0x288 / 0x308 = the batch's pass object + 0x130", "batch level: the sampler / resource table entries at context + 0x6c0 + 8 j from the mesh render data",
                                                             "batch level: the index buffer view, context + 0x58 and the index type +0x60 = 0x39", "texture pointers of the resource set (FUN_01c92f30) in the context slots (+0xa40 and +0x740)"};
        auto& binding = out["binding_audit"] = nlohmann::json::object();
        for (u32 c = 0; c < State::Checks; ++c) {
            binding[names[c]] = {{"checked", s.bind_checked[c]}, {"equal", s.bind_equal[c]}, {"differ", s.bind_checked[c] - s.bind_equal[c]}, {"not_applicable", s.bind_not_applicable[c]}, {"unreadable", s.bind_unreadable[c]}};
        }
        binding["draws_without_a_matching_expected_primitive"] = s.bind_unmatched_draws;
        binding["first_differences"] = s.bind_samples;
        {
            nlohmann::json values = nlohmann::json::object();
            for (const auto& [value, draws] : s.unused_texture_slot_values) {
                values[fmt::format("{:#x}", value)] = draws;
            }
            binding["texture_slots_past_the_material_texture_count"] = {{"distinct_values_and_draws", values}, {"draws_where_they_are_not_all_equal", s.unused_slots_not_uniform}};
        }
        binding["constant_buffer_slot_differences"] = {{"value_is_another_slot_of_the_same_render_object", s.slot_differs_same_object}, {"value_is_not_in_the_render_object", s.slot_differs_foreign}, {"by_cause", s.slot_differs_by}, {"samples", s.slot_samples}};
    }
    {
        auto& variant = out["variant_audit"] = nlohmann::json::object();
        variant["distinct_pipelines_seen_(vertex_shader,_pixel_shader)"] = s.pipelines.size();
        auto& by_key = variant["is_the_pipeline_a_function_of_the_key"] = nlohmann::json::object();
        for (u32 k = 0; k < KeyKinds; ++k) {
            u64 draws = 0, conflicting_keys = 0, conflicting_draws = 0;
            for (const auto& [key, stat] : s.keys[k]) {
                draws += stat.draws;
                conflicting_keys += stat.conflict ? 1 : 0;
                conflicting_draws += stat.conflicting_draws;
            }
            by_key[key_names[k]] = {{"distinct_keys", s.keys[k].size()}, {"keys_that_saw_two_pipelines", conflicting_keys}, {"draws", draws}, {"draws_after_the_first_pipeline_of_a_conflicting_key_that_differ", conflicting_draws}};
        }
        variant["conflict_samples"] = s.conflict_samples;
    }
    return out;
}

} // namespace Core::NativeZPass
