// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3a: read-only audit of the engine's depth pass loop, see native_zpass_audit.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
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
#include "core/native_zpass_audit.h"

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

struct Item {
    u32 kind;
    u64 count, first, instances;
    u32 batch, primitive;
    u64 material{};      // expected side: the primitive's material
    u64 material_sig{};  // expected side: a hash of the material fields the variant search reads
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
};

struct State {
    std::mutex mutex;
    bool active{false};
    u32 view_type{0};
    u32 pass_flags{0};
    std::vector<Item> expected;
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
            if (o.kind == want_kind && (want_kind == 1 || (o.count == e.count && o.first == e.first && o.instances == e.instances))) {
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
        if (group == 0 && flag_cvar != 0) {
            ++v.batches_skipped_no_instances_with_flag; // the loop draws nothing for a batch without instance group when the pass flag byte 5 is set
            continue;
        }
        u16 mesh_word = 0;
        Rd(mesh + 0x78, mesh_word);
        u32 instances = 0;
        if (group != 0) {
            u64 inner = 0, object = 0, table = 0;
            u32 count = 0;
            const bool ok = Rd(group + 8, inner) && (flag_cvar == 0 || inner != 0) && Rd(inner + 0x38, object) && object != 0 && Rd(object + 0x28, table) && Rd(table + 0x48, count);
            if (!ok || count == 0) {
                ++v.batches_skipped_instance;
                continue;
            }
            instances = count;
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
            }
            (item.kind == Plain ? v.expected_plain : item.kind == Instanced ? v.expected_instanced : v.expected_indirect)++;
            s.expected.push_back(item);
        }
    }
}

void NoteDraw(u64 caller, u32 kind, u64 count, u64 first, u64 instances, u64 vertex_shader, u64 pixel_shader) {
    if (!Enabled() || !InPass(caller)) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (s.active) {
        Item item{kind, count, first, instances, 0, 0};
        item.vertex_shader = vertex_shader;
        item.pixel_shader = pixel_shader;
        s.observed.push_back(item);
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
                                       {"unreadable_snapshot_fields", v.unreadable}};
    }
    out["first_mismatches"] = s.samples;
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
