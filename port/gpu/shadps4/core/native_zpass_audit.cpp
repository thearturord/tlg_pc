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
#include "core/native_draw.h"
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
    u32 material_flag{}; // expected side: material + 0xd4 & 1 (the per-primitive resource set is bound only for these)
    u64 batch_object{};  // expected side: the batch object
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
    const auto not_applicable = [&](u32 check) { ++s.bind_not_applicable[check]; };
    const auto compare = [&](u32 check, u64 ctx_address, u64 source, size_t bytes, const char* what) {
        std::array<u8, 512> a{}, b{};
        if (bytes > a.size() || !RdBytes(ctx_address, a.data(), bytes) || !RdBytes(source, b.data(), bytes)) {
            ++s.bind_unreadable[check];
            return;
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
    if (e.material_flag == 1 && e.batch_special == 0 && (token != 0 || cached != 0) && set != 0 && set_count != 0) {
        compare(5, s.render_device + 0x3790, set, size_t{std::min<u32>(set_count, 7)} * 16, "resource set at renderDev + 0x3790");
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
        if (e.material_flag == 1 && e.batch_special == 0 && (token_d != 0 || cached_d != 0) && (texture_flags & 3) != 0 && texture_array != 0) {
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
        u64 indirect_buffer = 0;
        if (group != 0) {
            u64 inner = 0, object = 0, table = 0;
            u32 count = 0;
            const bool ok = Rd(group + 8, inner) && (flag_cvar == 0 || inner != 0) && Rd(inner + 0x38, object) && object != 0 && Rd(object + 0x28, table) && Rd(table + 0x48, count);
            if (!ok || count == 0) {
                ++v.batches_skipped_instance;
                continue;
            }
            instances = count;
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
            Rd(batch + 0x48, item.batch_special);
            item.batch_object = batch;
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
            s.expected.push_back(item);
        }
    }
}

void NoteDraw(u64 caller, u32 kind, u64 count, u64 first, u64 instances, u64 vertex_shader, u64 pixel_shader, u64 ctx) {
    if (!Enabled() || !InPass(caller)) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (s.active) {
        CheckBinding(s, kind, count, first, instances, ctx);
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
