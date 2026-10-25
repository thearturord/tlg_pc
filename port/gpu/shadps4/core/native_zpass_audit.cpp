// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3a: read-only audit of the engine's depth pass loop, see native_zpass_audit.h.

#include <algorithm>
#include <atomic>
#include <cstdlib>
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
};

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
    std::vector<Item> expected;
    std::vector<Item> observed;
    u64 frame{0};
    std::map<u32, ViewStats> views;
    std::vector<std::string> samples;
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
            if (group != 0) {
                item.kind = indirect_mode != 0 ? Indirect : Instanced;
                item.instances = instances;
            }
            (item.kind == Plain ? v.expected_plain : item.kind == Instanced ? v.expected_instanced : v.expected_indirect)++;
            s.expected.push_back(item);
        }
    }
}

void NoteDraw(u64 caller, u32 kind, u64 count, u64 first, u64 instances) {
    if (!Enabled() || !InPass(caller)) {
        return;
    }
    auto& s = S();
    std::scoped_lock lk{s.mutex};
    if (s.active) {
        s.observed.push_back({kind, count, first, instances, 0, 0});
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
    return out;
}

} // namespace Core::NativeZPass
