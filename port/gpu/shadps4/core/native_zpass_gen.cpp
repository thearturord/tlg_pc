// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3i: the cascade generator, see native_zpass_gen.h and documents/tlg/n6-pass-generator-design.md.
//
// N6.4 adds the instance group batches (views 0, 1, 2): FUN_01ee4e50 (the binding of the group and the upload of its data) is the engine's own function, called by the generator and
// announced to the update list by its hook; the draws are DrawIndexedIndirect through the draw hook; the clear after the batch is plain stores.
// What the engine's loop does for a pass without special batch entries (RenderDev_DrawZPass, ghidra 0x01c74a00), as native code:
//   prologue   ctx + 0x4480 = 4 (dirty bit ctx + 0x447c |= 1 when it changed), ctx + 0x68 = 0
//   per batch  ctx + 0x580, the index buffer view (ctx + 0x58, + 0x60 = 0x39), FUN_01c932d0 (ctx + 0x288 / 0x308 and the table at ctx + 0x6c0)
//   per prim   the material render object into the context (FUN_01c92b60, and its early copy of the resource slot), the resource set (FUN_01c92f30, the engine's own function, for the
//              materials that have one), the variant's shader pair with the engine's reference counting, the raster state object, the draw through the draw hook, the counters
//   epilogue   ctx + 0x980 = 0, ctx + 0x46c8 = the default raster state object
// The variant of a primitive (the engine searches its cache with a key built from the material, the mesh and the pass flags and stores the result behind the primitive record) is a
// function of fields of the snapshot (audit N6.3b, f). The generator learns key -> variant object from the passes the engine's loop runs (the first use of every key, the verify passes) and
// declines a pass that has a key it has not learned or that two variants answered for. Nothing the loop does is guessed: a pass the generator does not model is run by the loop.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "core/native_draw.h"
#include "core/native_zpass_gen.h"

namespace Core::NativeZPassGen {

namespace {

constexpr u64 ZPassBegin = 0x01c74a00, ZPassEnd = 0x01c76710; // DrawZPass and the next function (DrawGBufferPrimitives)
constexpr u64 GlobalDefaultKind = 0x069ae968;                 // DAT_069ae968: the material kind of the engine's default material
constexpr u64 GlobalContext = 0x069a1b40;                     // DAT_069a1b40: the render context object (its +0x38 holder has the default objects)
constexpr u64 FunctionBindResources = 0x01c92f30;             // FUN_01c92f30(renderDev + 0x2910, ctx, material, special)
constexpr size_t CtxBytes = 0x14000;
constexpr u32 OrMask = 0x80000000u; // arena entry size flag: the entry is an OR of bits (dirty bytes), compared as "these bits are set"

template <class T>
T R(u64 address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}
template <class T>
void W(u64 address, T value) {
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(T));
}
u64 Global(u64 ghidra_address) {
    return NativeDraw::ImageBase() + (ghidra_address - 0x1000000);
}

struct Config {
    bool on{false};
    std::array<bool, 16> views{};
    u32 verify_every{32};
};
const Config& Cfg() {
    static const Config config = [] {
        Config c;
        const char* e = std::getenv("TLG_ZPASS_GEN");
        c.on = e != nullptr && *e != 0 && *e != '0';
        const char* v = std::getenv("TLG_ZPASS_GEN_VIEWS");
        std::string list = v != nullptr && *v != 0 ? v : "0,1,2,5,6,7";
        size_t at = 0;
        while (at < list.size()) {
            const size_t comma = list.find(',', at);
            const std::string item = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            const int view = std::atoi(item.c_str());
            if (view >= 0 && view < 16) {
                c.views[view] = true;
            }
            if (comma == std::string::npos) {
                break;
            }
            at = comma + 1;
        }
        if (const char* n = std::getenv("TLG_ZPASS_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

/// Everything about one batch that does not change inside a pass.
struct BatchLevel {
    u64 batch{};
    u64 value_580{}, view_58{}, pass_object_288{};
    bool bvar54{false};
    u32 table_count{0};
    std::array<u64, 16> table_entry{}; // the table entry object each value comes from
    std::array<std::pair<u32, u64>, 16> table{}; // (index j, value for ctx + 0x6c0 + 8 j), ascending j; the engine's loop fills them downwards, the values do not depend on each other
    u32 first_step{0}, step_count{0};
    // instance group batches (N6.4)
    u64 special{0};         // batch + 0x48, the special entry (class 0x26dbfc8: the vertex modifier entry)
    bool special_modifier{false}; // special + 0x40 != 0
    bool visibility{false}; // pass kind 3 with desc + 0x1d clear and batch flag 0x8: the primitive is skipped when the visibility cache says culled
    bool group{false};
    bool only_580{false};   // the loop writes ctx + 0x580 and nothing else for this batch (its group has no instances or no object)
    u64 inner{0};           // the mesh render data's inner object (first argument of FUN_01ee4e50)
    u64 group_object{0};
    u64 indirect_buffer{0}; // the argument buffer object of the group (group object + 0x110)
    // what FUN_01ee4e50 writes into the context (the verify run applies these; the generator calls the function)
    u64 g70{0}, g78{0}, g340{0};
    u32 g170{0}, g174{0};
    bool g340_set{false};
    u32 table_dwords{0};
    std::array<u64, 64> gtable{};
};

struct Step {
    u32 index{};    // the primitive's index in its batch (the indirect arguments are addressed with it)
    u64 record{};   // the primitive record (0x38 bytes)
    u64 material{};
    u64 key{};
    u64 variant{};  // the variant object learned for the key
    u32 flag{};     // material + 0xd4 & 1
};

struct Plan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u32 view_type{0}, kind{0};
    u64 render_device{0}, ctx{0};
    u32 flag14{0}, flag1c{0};
    s32 indirect_mode{0};
    u64 visibility_owner{0}; // desc + 0x28: the object whose +0x790 is the id the cache is keyed with
    std::vector<BatchLevel> batches;
    std::vector<Step> steps;
    u32 group_batches{0}, group_batches_skipped{0};
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* reason) {
        for (const char* r : reasons) {
            if (r == reason) {
                return;
            }
        }
        reasons.push_back(reason);
    }
};

struct TableEntry {
    u64 variant{0};
    bool ambiguous{false};
};

struct ViewStats {
    u64 passes{}, generated{}, generated_draws{}, generated_batches{}, eligible{};
    u64 loop_passes{}, verify_passes{}, verify_clean{}, verify_mismatched{}, verify_draws{};
    u64 not_scanned{};
    bool disabled{false};
    std::map<std::string, u64> declines;
    std::map<u32, u64> kinds; // passes by desc + 0x10
    u64 group_batches{}, group_batches_skipped{}, verify_raced{};
};

struct State {
    std::mutex mutex;
    std::map<u32, ViewStats> views;
    std::unordered_map<u64, TableEntry> table;
    std::vector<u64> cache_seen;
    u64 table_resets{0}, learned{0}, conflicts{0}, learn_not_in_cache{0}, learn_faults{0}, scan_faults{0};
    std::map<std::string, u64> mismatch_kinds;
    std::map<std::string, u64> special_classes;
    std::map<std::string, u64> raced_words; // verify: words where the live value differed from the shadow read but equals what the object holds when compared
    std::vector<std::string> samples;
};
State& S() {
    static State state;
    return state;
}

// ------------------------------------------------------------------------------------------------------------------------------- the target of the generator

struct Mark {
    size_t arena_end;
    u32 kind;
    u64 count, first;
    u64 vertex_shader, pixel_shader;
};

/// Where the generator writes: the live device context, or a private copy of it that logs every write (the verify run).
struct Target {
    bool live{true};
    u8* base{nullptr};
    std::vector<u8> arena; // shadow: entries [u32 offset][u32 size][bytes]
    std::vector<Mark> marks;
    u64 batch_owner{0};                          // shadow: the batch whose level words (ctx + 0x580, 0x58, 0x288, 0x308) are being written
    std::unordered_map<size_t, u64> batch_owner_map;
    u64 table_entry{0};                          // shadow: the table entry object whose value is being written (ctx + 0x6c0 + 8 j)
    std::unordered_map<size_t, u64> table_owner; // shadow: arena position of an entry -> that table entry object
    u64 slot_material{0};                       // shadow: the material whose resource slot words are being written
    std::unordered_map<size_t, u64> slot_owner; // shadow: arena position of an entry -> that material
    s32 draws{0};
    s64 triangles{0};

    template <class T>
    T Get(u32 offset) const {
        T value;
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }
    void Log(u32 offset, u32 size, const void* bytes) {
        const size_t at = arena.size();
        if (slot_material != 0) {
            slot_owner[at] = slot_material;
        }
        if (table_entry != 0) {
            table_owner[at] = table_entry;
        }
        if (batch_owner != 0) {
            batch_owner_map[at] = batch_owner;
        }
        arena.resize(at + 8 + (size & ~OrMask));
        std::memcpy(arena.data() + at, &offset, 4);
        std::memcpy(arena.data() + at + 4, &size, 4);
        std::memcpy(arena.data() + at + 8, bytes, size & ~OrMask);
    }
    void Put(u32 offset, const void* bytes, u32 size) {
        std::memcpy(base + offset, bytes, size);
        if (!live) {
            Log(offset, size, bytes);
        }
    }
    template <class T>
    void Set(u32 offset, T value) {
        Put(offset, &value, sizeof(T));
    }
    void Or8(u32 offset, u8 bits) {
        base[offset] |= bits;
        if (!live) {
            Log(offset, 1 | OrMask, &bits);
        }
    }
};

void ReaderLock(u64 address) {
    // FUN_0120d030: increment the low half of the dword when the high half (the writer flag) is clear
    auto* word = reinterpret_cast<std::atomic<u32>*>(address);
    for (;;) {
        u32 expected = word->load(std::memory_order_relaxed) & 0xffff;
        if (word->compare_exchange_strong(expected, expected + 1, std::memory_order_acquire)) {
            return;
        }
        std::this_thread::yield();
    }
}
void ReaderUnlock(u64 address) {
    reinterpret_cast<std::atomic<u32>*>(address)->fetch_sub(1, std::memory_order_release);
}

using GuestBind = void(PS4_SYSV_ABI*)(u64, u64, u64, u64);
using GuestRelease = void(PS4_SYSV_ABI*)(u64);
using GuestGroupBind = void(PS4_SYSV_ABI*)(u64, u64, u64);
constexpr u64 FunctionGroupBind = 0x01ee4e50; // FUN_01ee4e50(inner, instance group object, view type)

/// The engine's reference counted shader slot (ctx + 0x4698 vertex, ctx + 0x4690 pixel): release the old object, store the new one, count it, set the dirty bit.
void SetShader(Target& t, u32 offset, u64 object, u8 dirty_bit) {
    const u64 current = t.Get<u64>(offset);
    if (current == object) {
        return;
    }
    if (t.live) {
        if (current != 0) {
            auto* count = reinterpret_cast<std::atomic<s64>*>(current + 8);
            if (count->fetch_sub(1) == 1) {
                reinterpret_cast<GuestRelease>(R<u64>(R<u64>(current) + 0x10))(current);
            }
        }
        t.Set<u64>(offset, object);
        if (object != 0) {
            reinterpret_cast<std::atomic<s64>*>(object + 8)->fetch_add(1);
        }
    } else {
        t.Set<u64>(offset, object);
    }
    t.Or8(0x46a0, dirty_bit);
}

/// FUN_01c92b60 with the early copy of the resource slot the loop makes before the resource set: the material's render object into the context.
void CopyEarlyBlock(Target& t, u64 material) {
    const u64 object = R<u64>(material + 0x38);
    const u64 index_word = R<u64>(object + 0x440);
    const u64 index = (index_word & 0xffffffffull) * 0xb0;
    const u64 slot = R<u64>(object + 0xc0 + index);
    t.Put(0x980, reinterpret_cast<const void*>(object + 0x450), 8);
    if ((index_word >> 32) == 0) {
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x310, slot);
    } else {
        const u64 second = R<u64>(object + 0x2d0 + index);
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x298, second);
        t.Set<u64>(0x310, slot);
        t.Set<u64>(0x318, second);
    }
}

void BindMaterial(Target& t, u64 material) {
    if (t.live) {
        ReaderLock(material + 0x1e0);
    }
    const u64 object = R<u64>(material + 0x38);
    t.Put(0x980, reinterpret_cast<const void*>(object + 0x450), 0x48);
    if (R<u32>(object + 0x444) != 0) {
        t.Put(0x9d0, reinterpret_cast<const void*>(object + 0x498), 0x48);
    }
    if (const u64 views = R<u64>(object + 0x4f0); views != 0) {
        t.Put(0x790, reinterpret_cast<const void*>(views), R<u32>(object + 0x4f8) * 8);
    }
    if (const u64 views = R<u64>(object + 0x4e0); views != 0) {
        t.Put(0xb38, reinterpret_cast<const void*>(views), R<u32>(object + 0x4e8) * 8);
    }
    const u64 index_word = R<u64>(object + 0x440);
    const u64 index = (index_word & 0xffffffffull) * 0xb0;
    const u64 slot = R<u64>(object + 0xc0 + index);
    if ((index_word >> 32) == 0) {
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x310, slot);
    } else {
        const u64 second = R<u64>(object + 0x2d0 + index);
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x298, second);
        t.Set<u64>(0x310, slot);
        t.Set<u64>(0x318, second);
    }
    if (t.live) {
        ReaderUnlock(material + 0x1e0);
    }
}

/// The key the engine's variant search is a function of, as far as the snapshot says (audit N6.3b and f), plus what the audit did not need to look at (bVar54).
u64 Mix(u64 h, u64 value) {
    return (h ^ value) * 0x100000001b3ull + 0x9E3779B97F4A7C15ull;
}

struct MatInfo {
    bool ok{false};
    const char* reason{nullptr};
    u32 flag{0};
    u64 hash{0};
};

MatInfo ReadMaterial(u64 material, u64 default_kind) {
    MatInfo info;
    if (R<u64>(material + 0x18) == default_kind) {
        info.reason = "material of the engine's default kind (the loop keeps the previous shaders)";
        return info;
    }
    const u64 object = R<u64>(material + 0x38);
    if (object == 0) {
        info.reason = "material without a render object";
        return info;
    }
    const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
    if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72))) {
        info.reason = "material with a shader resource count the engine asserts on";
        return info;
    }
    if ((R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
        info.reason = "material with a constant buffer slot index out of range";
        return info;
    }
    (void)R<u64>(object + 0xc0 + (R<u64>(object + 0x440) & 0xffffffffull) * 0xb0);
    info.flag = R<u32>(material + 0xd4) & 1;
    u64 h = 0xcbf29ce484222325ull;
    for (const u32 offset : {0x24u, 0x28u, 0xd0u, 0xd4u, 0x158u, 0x15cu, 0x164u, 0x1bcu}) {
        h = Mix(h, R<u32>(material + offset));
    }
    // the variant search also reads two fields of the cached object behind the material (+0x48)
    if (const u64 cached = R<u64>(material + 0x48); cached != 0) {
        h = Mix(h, R<u32>(cached + 0x24));
        h = Mix(h, R<u32>(cached + 0x28));
    } else {
        h = Mix(h, 0xdeadbeefull);
    }
    info.hash = h;
    info.ok = true;
    return info;
}

void FillBatchLevel(BatchLevel& level, u64 batch, u64 default_object) {
    const u64 mesh = R<u64>(batch + 0x20);
    const u64 render_data = R<u64>(mesh + 0x38);
    const u64 inner = R<u64>(render_data + 0x20);
    level.batch = batch;
    level.inner = inner;
    level.value_580 = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28);
    level.pass_object_288 = R<u64>(R<u64>(batch + 0x68) + 0x130);
    level.view_58 = 0;
    if (const u64 holder = R<u64>(inner + 0x50); holder != 0) {
        level.view_58 = R<u64>(R<u64>(holder + 0x38) + 0x20);
    }
    const u32 array_count = R<u32>(inner + 0x68);
    const u64 array = R<u64>(inner + 0x60);
    level.bvar54 = false;
    if (array_count != 0) {
        const u64 first = R<u64>(array);
        if (first != default_object && first != 0) {
            level.bvar54 = R<u64>(first + 0x38) != 0;
        }
    }
    const u32 entries = std::min<u32>(R<u32>(mesh + 0x68), 16);
    level.table_count = 0;
    for (u32 j = 0; j < entries; ++j) {
        if (j >= array_count) {
            continue;
        }
        const u64 entry = R<u64>(array + u64{j} * 8);
        if (entry == default_object || entry == 0) {
            continue;
        }
        const u64 holder = R<u64>(entry + 0x38);
        if (holder == 0) {
            continue;
        }
        level.table_entry[level.table_count] = entry;
        level.table[level.table_count++] = {j, R<u64>(holder + 0x28)};
    }
}

u64 StepKey(u64 material_hash, const Plan& plan, u16 batch_flags, bool bvar54, u16 mesh_word, u64 group_bits) {
    u64 h = material_hash;
    h = Mix(h, plan.view_type);
    h = Mix(h, plan.flag14);
    h = Mix(h, plan.flag1c);
    h = Mix(h, batch_flags & 0x10);
    h = Mix(h, bvar54 ? 1 : 0);
    h = Mix(h, mesh_word);
    h = Mix(h, group_bits);
    return h;
}

/// The visibility cache of pass kind 3 (the loop, after the raster state of a primitive of a batch with flag 0x8): the primitive's data object holds 8 hash chains at +0x68 keyed with the id
/// at owner + 0x790 (the node's first qword is the id in both halves). Returns 0 when there is no node, else the node.
u64 VisibilityNode(u64 owner, u64 record) {
    const u32 id = R<u32>(owner + 0x790);
    const u64 data = R<u64>(record + 0x28);
    for (u64 node = R<u64>(data + 0x68 + u64(id & 7) * 8); node != 0; node = R<u64>(node + 0x10)) {
        const u64 key = R<u64>(node);
        if (static_cast<u32>(key) == id && static_cast<u32>(key >> 32) == id) {
            return node;
        }
    }
    return 0;
}

/// True when the loop would not create a cache entry for the primitive: a node exists, or the list of the current slot is full (then it draws without one).
bool VisibilityKnown(const Plan& plan, u64 rd, u64 record) {
    if (VisibilityNode(plan.visibility_owner, record) != 0) {
        return true;
    }
    const u32 list = R<u32>(rd + 0x10978);
    return R<u32>(rd + 0x10988 + u64{list} * 0x18) > 0x7f;
}

// ------------------------------------------------------------------------------------------------------------------------------- scan (read only, guarded)

struct ScanArgs {
    Plan* plan;
    u64 render_device, desc;
    const std::unordered_map<u64, TableEntry>* table;
};

void ScanImpl(ScanArgs& args) {
    Plan& plan = *args.plan;
    const u64 rd = args.render_device, desc = args.desc;
    plan.render_device = rd;
    const u64 list = R<u64>(desc);
    const u64 view_word = R<u64>(desc + 8);
    plan.view_type = static_cast<u32>(view_word);
    plan.kind = R<u32>(desc + 0x10);
    const u32 mask = R<u32>(desc + 0x20);
    plan.flag14 = R<u8>(desc + 0x14);
    plan.flag1c = R<u8>(desc + 0x1c);
    const u8 flag_hi = static_cast<u8>(view_word >> 32);
    const u8 flag_cvar = static_cast<u8>(view_word >> 40);
    plan.ctx = R<u64>(rd + 0x2d8);
    plan.indirect_mode = R<s32>(rd + (plan.view_type == 0 ? 0x38e0 : 0x38e4));
    if (list == 0 || plan.ctx == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    const u8 flag_1d = R<u8>(desc + 0x1d);
    if (plan.kind == 1 || plan.kind == 2 || plan.kind == 5) {
        plan.Decline("pass kind with queries or a target switch (desc + 0x10 in 1, 2, 5)");
    }
    if (R<u64>(rd + 0x8930) != 0) {
        plan.Decline("the engine's per draw register packet is on (renderDev + 0x8930)");
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    const u64 default_kind = R<u64>(Global(GlobalDefaultKind));
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(list);
    const u32 batch_count = R<u32>(list + 8);
    if (batch_array == 0 || batch_count > 100000) {
        plan.Decline("batch list unreadable");
        return;
    }
    std::unordered_map<u64, MatInfo> materials;
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        const u16 batch_flags = R<u16>(batch + 0x66);
        if (flag_hi != 0 && (batch_flags & 2) != 0) {
            continue; // the loop skips the batch before it touches anything
        }
        const u64 mesh = R<u64>(batch + 0x20);
        if (mesh == 0) {
            continue;
        }
        const u64 group = R<u64>(batch + 0x38);
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        BatchLevel level;
        FillBatchLevel(level, batch, default_object);
        level.first_step = static_cast<u32>(plan.steps.size());
        if (group == 0 && flag_cvar != 0) {
            // pass flag byte 5: the loop draws only instance group batches; it has written ctx + 0x580 and leaves
            level.only_580 = true;
            plan.batches.push_back(level);
            continue;
        }
        u64 special_bits = 0;
        if (const u64 special = R<u64>(batch + 0x48); special != 0) {
            // the only class in use (live data): its virtuals are +0x48 (the token the variant key uses: the material's, or the VERTEX_MODIFIER token when special + 0x40 is set; a
            // reference counted handle, net zero over the loop's sequence), +0x50 (special + 0x40 != 0) and +0x58 (the resource table fill inside FUN_01c92f30, the engine's own)
            const u64 vtable = R<u64>(special);
            if (vtable - NativeDraw::ImageBase() + 0x1000000 != 0x26dbfc8) {
                S().special_classes[fmt::format("vtable {:#x}", vtable - NativeDraw::ImageBase() + 0x1000000)]++;
                plan.Decline("batch with a special entry of an unknown class");
                continue;
            }
            level.special = special;
            level.special_modifier = R<u64>(special + 0x40) != 0;
            special_bits = (1ull << 48) | (u64(level.special_modifier) << 49);
        }
        if (plan.kind == 3 && flag_1d == 0 && (batch_flags & 8) != 0) {
            level.visibility = true;
            plan.visibility_owner = R<u64>(desc + 0x28);
            if (plan.visibility_owner == 0) {
                plan.Decline("pass kind 3 without the visibility owner (desc + 0x28)");
                continue;
            }
        }
        u64 group_bits = special_bits;
        if (group != 0) {
            if (plan.indirect_mode == 0) {
                plan.Decline("instance group batch without the indirect mode (DrawIndexedInstanced is not hooked)");
                continue;
            }
            const u64 group_inner = R<u64>(group + 8);
            if (group_inner == 0) {
                if (flag_cvar != 0) {
                    level.only_580 = true;
                    plan.batches.push_back(level);
                } else {
                    plan.Decline("instance group without its inner object");
                }
                continue;
            }
            level.group = true;
            level.bvar54 = false; // the loop's flag for the batch of a group
            ++plan.group_batches;
            const u64 group_object = R<u64>(group_inner + 0x38);
            u32 instances = 0;
            if (group_object != 0) {
                instances = R<u32>(R<u64>(group_object + 0x28) + 0x48);
            }
            if (group_object == 0 || instances == 0) {
                level.only_580 = true; // the loop leaves such a batch after writing ctx + 0x580
                ++plan.group_batches_skipped;
                level.step_count = 0;
                plan.batches.push_back(level);
                continue;
            }
            level.group_object = group_object;
            level.indirect_buffer = R<u64>(group_object + 0x110);
            const u32 views_count = R<u32>(group_object + 0xe0);
            if (views_count > 45) {
                plan.Decline("instance group with more shader resource views than the engine accepts");
                continue;
            }
            // what FUN_01ee4e50 writes (read now for the verify run, and to find stale pointers under the guard)
            const s32 ring_index = R<s32>(group_object + 0x20);
            if (ring_index != 0 && ring_index != 1) {
                plan.Decline("instance group with a ring index outside 0 and 1");
                continue;
            }
            const u64 ring_entry = R<u64>(R<u64>(group_object + 0xb8) + u64(ring_index) * 8);
            const u64 view_object = R<u64>(R<u64>(level.inner + 0x48) + 0x38);
            level.g70 = R<u64>(view_object + 0x20);
            level.g78 = R<u64>(ring_entry + 0x20);
            level.g170 = R<u32>(view_object + 0x40);
            level.g174 = R<u32>(ring_entry + 0x40);
            if (const u64 uploader = R<u64>(group_object + 0xf0); uploader != 0) {
                level.g340 = R<u64>(R<u64>(uploader) + 0x10);
                level.g340_set = true;
            }
            level.gtable[0] = R<u64>(group_object + 0x1b0 + u64(plan.view_type) * 8);
            level.gtable[1] = R<u64>(group_object + 0x1f8 + u64(ring_index) * 8);
            level.gtable[2] = R<u64>(group_object + 0x1f8 + u64(1 - ring_index) * 8);
            const u64 views_array = R<u64>(group_object + 0xd8);
            for (u32 i = 0; i < views_count; ++i) {
                const u64 holder = R<u64>(views_array + u64{i} * 8);
                level.gtable[3 + i] = holder != 0 ? R<u64>(holder + 0x28) : 0;
            }
            level.table_dwords = 3 + views_count;
            group_bits = (1ull << 40) | (u64(plan.indirect_mode != 0) << 32) | R<u32>(group_inner + 0x40);
        }
        const u16 mesh_word = R<u16>(mesh + 0x78);
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            const u64 material = R<u64>(record + 0x18);
            auto found = materials.find(material);
            if (found == materials.end()) {
                found = materials.emplace(material, ReadMaterial(material, default_kind)).first;
            }
            const MatInfo& info = found->second;
            if (!info.ok) {
                // the flag of the material decides whether the loop looks at the primitive at all: read it for the mask test even for materials the generator cannot take
                if ((mask & (2 - (R<u32>(material + 0xd4) & 1))) != 0) {
                    plan.Decline(info.reason);
                }
                continue;
            }
            if ((mask & (2 - info.flag)) == 0) {
                continue;
            }
            if (R<u64>(record + 0x28) == 0) {
                plan.Decline("primitive record without its per primitive data");
                continue;
            }
            Step step;
            step.record = record;
            step.material = material;
            step.flag = info.flag;
            step.key = StepKey(info.hash, plan, batch_flags, level.bvar54, mesh_word, group_bits);
            step.index = p;
            if (level.visibility && !VisibilityKnown(plan, rd, record)) {
                plan.Decline("visibility cache entry missing (the loop creates it, with a query around the draw)");
            }
            const auto entry = args.table->find(step.key);
            if (entry == args.table->end()) {
                plan.Decline("variant key not learned yet (the loop picks the variant at first use)");
            } else if (entry->second.ambiguous) {
                plan.Decline("variant key with two different variants");
            } else {
                step.variant = entry->second.variant;
                if (R<u64>(step.variant + 8) == 0 || R<u64>(step.variant + 0x18) == 0) {
                    plan.Decline("variant without a shader pair");
                }
            }
            plan.steps.push_back(step);
        }
        level.step_count = static_cast<u32>(plan.steps.size()) - level.first_step;
        plan.batches.push_back(level);
    }
    plan.scanned = true;
}

// ------------------------------------------------------------------------------------------------------------------------------- the generator

void IssueDraw(Target& t, u32 kind, u64 count, u64 first) {
    if (t.live) {
        if (kind == 0) {
            NativeDraw::IssueIndexedDraw(reinterpret_cast<u64>(t.base), count, first);
        } else {
            NativeDraw::IssueIndirectDraw(reinterpret_cast<u64>(t.base), count, first);
        }
    } else {
        t.marks.push_back({t.arena.size(), kind, count, first, t.Get<u64>(0x4698), t.Get<u64>(0x4690)});
    }
}

void Generate(Target& t, const Plan& plan) {
    const u64 rd = plan.render_device;
    // prologue
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    t.Set<u64>(0x68, 0);
    bool raster_flag = false; // bVar15: which of the two raster state objects the last primitive asked for
    for (const BatchLevel& level : plan.batches) {
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        t.batch_owner = 0;
        if (level.only_580) {
            continue;
        }
        if (level.group) {
            if (t.live) {
                reinterpret_cast<GuestGroupBind>(Global(FunctionGroupBind))(level.inner, level.group_object, plan.view_type);
            } else {
                t.Set<u64>(0x70, level.g70);
                t.Set<u64>(0x78, level.g78);
                t.Set<u32>(0x170, level.g170);
                t.Set<u32>(0x174, level.g174);
                t.Set<u64>(0x1f0, 0);
                if (level.g340_set) {
                    t.Set<u64>(0x340, level.g340);
                }
                t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
            }
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x58, level.view_58);
        t.Set<u64>(0x60, 0x39);
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const Step& step = plan.steps[s];
            t.slot_material = step.material;
            if (step.flag != 0) {
                CopyEarlyBlock(t, step.material);
                if (!level.bvar54 && t.live) {
                    reinterpret_cast<GuestBind>(Global(FunctionBindResources))(rd + 0x2910, reinterpret_cast<u64>(t.base), step.material, level.special);
                }
            }
            if (level.special != 0) {
                // the loop copies the material block first when its own conditions say so, then binds the resource set through the special entry (always), then the block again below
                const u64 cached = R<u64>(step.material + 0x48);
                const bool early = cached == 0 ? (R<u32>(step.material + 0x158) != 0 || (R<u8>(step.material + 0xd5) & 1) != 0)
                                               : ((R<u32>(cached + 0x24) != 0 && R<u32>(cached + 0x28) != 0) || (R<u8>(step.material + 0xd5) & 1) != 0);
                if (early) {
                    BindMaterial(t, step.material);
                }
                if (t.live) {
                    reinterpret_cast<GuestBind>(Global(FunctionBindResources))(rd + 0x2910, reinterpret_cast<u64>(t.base), step.material, level.special);
                }
            }
            BindMaterial(t, step.material);
            t.slot_material = 0;
            if (t.live) {
                W<u64>(R<u64>(step.record + 0x28) + 0x48, step.variant); // the loop stores the variant it chose behind the primitive record
            }
            SetShader(t, 0x4698, R<u64>(step.variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(step.variant + 0x18), 2);
            const u32 raster_bits = R<u32>(step.material + 0xd4) & 0x2e;
            const bool raster = raster_bits != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            if (level.visibility) {
                if (const u64 node = VisibilityNode(plan.visibility_owner, step.record); node != 0 && R<s32>(node + 8) == 2) {
                    continue; // culled by the cache: the loop leaves the primitive after its state is set, before the draw and the counters
                }
            }
            const s32 triangles = R<s32>(step.record + 0x14);
            if (level.group) {
                IssueDraw(t, 1, level.indirect_buffer, (u64{step.index} * 9 + plan.view_type) * 0x18);
            } else {
                IssueDraw(t, 0, static_cast<u32>(triangles * 3), R<u32>(step.record + 0x10));
            }
            if (t.live) {
                W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                W<s32>(rd + 0x14, R<s32>(rd + 0x14) + triangles);
            } else {
                ++t.draws;
                t.triangles += triangles;
            }
        }
        if (level.group) {
            // the end of an instanced batch clears the instance state
            const std::array<u8, 0x30> zeros{};
            t.Put(0x70, zeros.data(), 16);
            t.Set<u64>(0x170, 0);
            t.Set<u64>(0x1f0, 0);
            t.Set<u64>(0x340, 0);
            t.Put(0x800, zeros.data(), 0x30);
        }
    }
    // epilogue
    t.Set<u64>(0x980, 0);
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------- learning

struct LearnArgs {
    const Plan* plan;
    State* state;
};

void LearnImpl(LearnArgs& args) {
    State& s = *args.state;
    std::unordered_set<u64> cache(s.cache_seen.begin(), s.cache_seen.end());
    for (const Step& step : args.plan->steps) {
        const u64 data = R<u64>(step.record + 0x28);
        const u64 variant = data != 0 ? R<u64>(data + 0x48) : 0;
        if (variant == 0) {
            continue;
        }
        if (cache.find(variant) == cache.end()) {
            ++s.learn_not_in_cache;
            continue;
        }
        const auto [it, fresh] = s.table.try_emplace(step.key, TableEntry{variant, false});
        if (fresh) {
            ++s.learned;
        } else if (it->second.variant != variant && !it->second.ambiguous) {
            it->second.ambiguous = true;
            ++s.conflicts;
        }
    }
}

struct CacheArgs {
    State* state;
    u64 render_device;
};
/// The engine's variant cache (renderDev + 0x1c60, count + 0x1c68) only grows: a learned object stays valid as long as the entries we saw are still there.
void CheckCacheImpl(CacheArgs& args) {
    State& s = *args.state;
    const u64 array = R<u64>(args.render_device + 0x1c60);
    const u32 count = R<u32>(args.render_device + 0x1c68);
    bool same = array != 0 && count >= s.cache_seen.size() && count < 100000;
    for (size_t i = 0; same && i < s.cache_seen.size(); ++i) {
        same = R<u64>(array + i * 8) == s.cache_seen[i];
    }
    if (!same) {
        if (!s.table.empty()) {
            ++s.table_resets;
        }
        s.table.clear();
        s.cache_seen.clear();
    }
    if (array != 0 && count < 100000) {
        s.cache_seen.resize(count);
        for (u32 i = 0; i < count; ++i) {
            s.cache_seen[i] = R<u64>(array + u64{i} * 8);
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------- verify

struct Verify {
    std::atomic<bool> active{false};
    std::thread::id thread;
    Target shadow;
    size_t cursor{0}, arena_pos{0};
    u32 view{0};
    u64 ctx{0};
    u64 compared_draws{0};
    u64 mismatches{0};
    u64 raced{0};
    std::vector<s16> expect;
};
Verify& V() {
    static Verify verify;
    return verify;
}

void NoteMismatch(State& s, u32 view, const std::string& kind, const std::string& detail) {
    ++s.mismatch_kinds[kind];
    if (s.samples.size() < 24) {
        s.samples.push_back(fmt::format("view type {}: {}: {}", view, kind, detail));
        LOG_ERROR(Render, "cascade generator verify: view type {}: {}: {}", view, kind, detail);
    }
}

/// Compares the writes the generator made between two draws with the live context. Writes to the same bytes inside the span are applied in order (only the last counts: a batch
/// that draws nothing rewrites ctx + 0x580, the clear after an instanced batch overwrites the start of the table before it).
u32 CompareDelta(State& s, Verify& v, size_t from, size_t to, u64 live_ctx, const char* where) {
    u32 differences = 0;
    const u8* arena = v.shadow.arena.data();
    if (v.expect.empty()) {
        v.expect.assign(CtxBytes, -1);
    }
    std::vector<u32> touched;
    std::map<u32, u8> or_bits;
    struct SlotWord {
        u64 expected;
        u64 material;
    };
    std::map<u32, SlotWord> slot_words, table_words, batch_words;
    size_t at = from;
    while (at < to) {
        u32 offset, size;
        std::memcpy(&offset, arena + at, 4);
        std::memcpy(&size, arena + at + 4, 4);
        const bool is_or = (size & OrMask) != 0;
        const u32 bytes = size & ~OrMask;
        const u8* expected = arena + at + 8;
        const size_t entry_at = at;
        at += 8 + bytes;
        if (offset + bytes > CtxBytes) {
            continue;
        }
        if (is_or) {
            or_bits[offset] |= expected[0];
            continue;
        }
        if (const auto bowner = v.shadow.batch_owner_map.find(entry_at); bowner != v.shadow.batch_owner_map.end() && bytes == 8 && (offset == 0x580 || offset == 0x58 || offset == 0x288 || offset == 0x308)) {
            u64 value = 0;
            std::memcpy(&value, expected, 8);
            batch_words[offset] = {value, bowner->second};
            continue;
        }
        if (const auto towner = v.shadow.table_owner.find(entry_at); towner != v.shadow.table_owner.end() && bytes == 8) {
            u64 value = 0;
            std::memcpy(&value, expected, 8);
            table_words[offset] = {value, towner->second};
            continue;
        }
        const auto owner = v.shadow.slot_owner.find(entry_at);
        if (bytes == 8 && owner != v.shadow.slot_owner.end() && (offset == 0x290 || offset == 0x310 || offset == 0x298 || offset == 0x318)) {
            u64 value = 0;
            std::memcpy(&value, expected, 8);
            slot_words[offset] = {value, owner->second};
            continue;
        }
        for (u32 i = 0; i < bytes; ++i) {
            if (v.expect[offset + i] < 0) {
                touched.push_back(offset + i);
            }
            v.expect[offset + i] = expected[i];
        }
    }
    std::sort(touched.begin(), touched.end());
    for (size_t i = 0; i < touched.size();) {
        size_t j = i;
        while (j + 1 < touched.size() && touched[j + 1] == touched[j] + 1) {
            ++j;
        }
        // one run [touched[i], touched[j]]: report the first differing byte of it
        for (size_t k = i; k <= j; ++k) {
            const u32 at_byte = touched[k];
            const u8 live = R<u8>(live_ctx + at_byte);
            if (live != static_cast<u8>(v.expect[at_byte])) {
                u64 a = 0, b = 0;
                const u32 base = at_byte & ~7u;
                std::memcpy(&a, reinterpret_cast<const void*>(live_ctx + base), 8);
                for (u32 q = 0; q < 8; ++q) {
                    const s16 e = v.expect[base + q];
                    const u8 value = e >= 0 ? static_cast<u8>(e) : R<u8>(live_ctx + base + q);
                    b |= u64{value} << (q * 8);
                }
                ++differences;
                NoteMismatch(s, v.view, fmt::format("context word {:#x}", base), fmt::format("{}: the loop left {:#x}, the generator writes {:#x}", where, a, b));
                break;
            }
        }
        i = j + 1;
    }
    for (const u32 at_byte : touched) {
        v.expect[at_byte] = -1;
    }
    for (const auto& [offset, bits] : or_bits) {
        const u8 live = R<u8>(live_ctx + offset);
        if ((live & bits) != bits) {
            ++differences;
            NoteMismatch(s, v.view, fmt::format("context word {:#x}", offset), fmt::format("{}: the loop left bits {:#x}, the generator sets {:#x}", where, live, bits));
        }
    }
    for (const auto& [offset, word] : batch_words) {
        const u64 live = R<u64>(live_ctx + offset);
        if (live == word.expected) {
            continue;
        }
        // the batch level words come from the mesh render data and the pass object; the engine can replace them (a streamed index buffer) between the shadow read and the loop read
        struct Recompute {
            u64 batch;
            u32 offset;
            u64 value;
        } recompute{word.material, offset, ~0ull};
        NativeDraw::GuardedCall(
            [](void* p) {
                auto* r = static_cast<Recompute*>(p);
                const u64 context_global = R<u64>(Global(GlobalContext));
                const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
                BatchLevel now;
                FillBatchLevel(now, r->batch, default_object);
                r->value = r->offset == 0x580 ? now.value_580 : r->offset == 0x58 ? now.view_58 : now.pass_object_288;
            },
            &recompute);
        if (live == recompute.value) {
            ++v.raced;
            ++s.raced_words[fmt::format("view {} {} word {:#x}", v.view, "batch", offset)];
            continue;
        }
        ++differences;
        NoteMismatch(s, v.view, fmt::format("context word {:#x}", offset), fmt::format("{}: the loop left {:#x}, the generator writes {:#x} (the batch gives {:#x} now)", where, live, word.expected, recompute.value));
    }
    for (const auto& [offset, word] : table_words) {
        const u64 live = R<u64>(live_ctx + offset);
        if (live == word.expected) {
            continue;
        }
        // the table entry's resource view can be replaced by the engine between the shadow read and the loop read
        const u64 holder = R<u64>(word.material + 0x38);
        const u64 now = holder != 0 ? R<u64>(holder + 0x28) : 0;
        if (live == now) {
            ++v.raced;
            ++s.raced_words[fmt::format("view {} {} word {:#x}", v.view, "table", offset)];
            continue;
        }
        ++differences;
        NoteMismatch(s, v.view, fmt::format("context word {:#x}", offset), fmt::format("{}: the loop left {:#x}, the generator writes {:#x} (the table entry holds {:#x} now)", where, live, word.expected, now));
    }
    for (const auto& [offset, word] : slot_words) {
        const u64 live = R<u64>(live_ctx + offset);
        if (live == word.expected) {
            continue;
        }
        // the resource slot of a material's render object moves when the engine's writer thread advances the object's ring (the reader lock exists for it): the shadow read it before the
        // loop ran, the loop after. If the live value is what the object holds now, the loop and the generator agree and the difference is the time between their reads.
        const u64 object = R<u64>(word.material + 0x38);
        const u64 index_word = R<u64>(object + 0x440);
        const u64 index = (index_word & 0xffffffffull) * 0xb0;
        const u64 now = (offset == 0x290 || offset == 0x310) ? R<u64>(object + 0xc0 + index) : R<u64>(object + 0x2d0 + index);
        if (live == now) {
            ++v.raced;
            ++s.raced_words[fmt::format("view {} {} word {:#x}", v.view, "slot", offset)];
            continue;
        }
        ++differences;
        NoteMismatch(s, v.view, fmt::format("context word {:#x}", offset), fmt::format("{}: the loop left {:#x}, the generator writes {:#x} (the object holds {:#x} now)", where, live, word.expected, now));
    }
    return differences;
}

} // namespace

bool Enabled() {
    return Cfg().on;
}

bool VerifyActive() {
    return V().active.load(std::memory_order_relaxed);
}

void NoteLoopDraw(u64 caller, u32 kind, u64 ctx, u64 count, u64 first) {
    Verify& v = V();
    if (!v.active.load(std::memory_order_relaxed) || caller < ZPassBegin || caller >= ZPassEnd || std::this_thread::get_id() != v.thread) {
        return;
    }
    State& s = S();
    std::scoped_lock lock{s.mutex};
    const size_t k = v.cursor++;
    ++v.compared_draws;
    if (k >= v.shadow.marks.size()) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "extra draw", fmt::format("the loop issued draw {} but the generator has {}", k, v.shadow.marks.size()));
        return;
    }
    const Mark& mark = v.shadow.marks[k];
    if (mark.kind != kind || mark.count != count || mark.first != first) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "draw arguments", fmt::format("draw {}: the loop kind {} {:#x} / {:#x}, the generator kind {} {:#x} / {:#x}", k, kind, count, first, mark.kind, mark.count, mark.first));
    }
    if (ctx != v.ctx) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "context", fmt::format("draw {}: another device context", k));
        return;
    }
    const u64 vs = R<u64>(ctx + 0x4698), ps = R<u64>(ctx + 0x4690);
    if (vs != mark.vertex_shader || ps != mark.pixel_shader) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "shader pair", fmt::format("draw {}: the loop {:#x} / {:#x}, the generator {:#x} / {:#x}", k, vs, ps, mark.vertex_shader, mark.pixel_shader));
    }
    v.mismatches += CompareDelta(s, v, v.arena_pos, mark.arena_end, ctx, fmt::format("draw {}", k).c_str());
    v.arena_pos = mark.arena_end;
}

bool HandlePass(u64 render_device, u64 desc, u64 original_loop) {
    const Config& cfg = Cfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64);
    State& s = S();
    Plan plan;
    ScanArgs scan_args{&plan, render_device, desc, &s.table};
    // the scan and the learning read the engine's objects: guarded, and under the state lock for the table
    bool handled_view = false;
    {
        std::scoped_lock lock{s.mutex};
        u32 view = 0xffffffffu;
        struct HeaderArgs {
            u64 desc;
            u32 view;
        } header{desc, view};
        if (!NativeDraw::GuardedCall([](void* p) { auto* h = static_cast<HeaderArgs*>(p); h->view = static_cast<u32>(R<u64>(h->desc + 8)); }, &header)) {
            return false;
        }
        view = header.view;
        handled_view = view < cfg.views.size() && cfg.views[view];
        if (!handled_view) {
            return false;
        }
        CacheArgs cache_args{&s, render_device};
        if (!NativeDraw::GuardedCall([](void* p) { CheckCacheImpl(*static_cast<CacheArgs*>(p)); }, &cache_args)) {
            ++s.scan_faults;
        }
        ViewStats& vs = s.views[view];
        ++vs.passes;
        struct KindArgs {
            u64 desc;
            u32 kind;
        } kind_args{desc, 0xffffffffu};
        if (NativeDraw::GuardedCall([](void* p) { auto* k = static_cast<KindArgs*>(p); k->kind = R<u32>(k->desc + 0x10); }, &kind_args)) {
            ++vs.kinds[kind_args.kind];
        }
        if (vs.disabled) {
            plan.Decline("generator switched off for this view type by a verify difference");
        } else if (!NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &scan_args)) {
            ++s.scan_faults;
            ++vs.not_scanned;
            plan.scanned = false;
            plan.steps.clear();
            plan.batches.clear();
            plan.Decline("scan faulted");
        }
        plan.view_type = view;
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++vs.declines[r];
            }
        }
    }
    ViewStats* stats = nullptr;
    bool verify_pass = false;
    {
        std::scoped_lock lock{s.mutex};
        stats = &s.views[plan.view_type];
        if (plan.Ok()) {
            ++stats->eligible;
            verify_pass = cfg.verify_every != 0 && stats->eligible % cfg.verify_every == cfg.verify_every - 1 && !V().active.load();
        }
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Generate(live, plan);
        std::scoped_lock lock{s.mutex};
        ++stats->generated;
        stats->generated_batches += plan.batches.size();
        stats->generated_draws += plan.steps.size();
        stats->group_batches += plan.group_batches;
        stats->group_batches_skipped += plan.group_batches_skipped;
        return true;
    }
    Verify& v = V();
    if (verify_pass) {
        v.shadow = Target{};
        v.shadow.live = false;
        std::vector<u8> copy(CtxBytes);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(plan.ctx), CtxBytes);
        v.shadow.base = copy.data();
        Generate(v.shadow, plan);
        v.cursor = 0;
        v.arena_pos = 0;
        v.view = plan.view_type;
        v.ctx = plan.ctx;
        v.compared_draws = 0;
        v.mismatches = 0;
        v.raced = 0;
        v.thread = std::this_thread::get_id();
        const s32 draws_before = R<s32>(render_device + 0x10), triangles_before = R<s32>(render_device + 0x14);
        v.active.store(true);
        reinterpret_cast<Loop>(original_loop)(render_device, desc);
        v.active.store(false);
        std::scoped_lock lock{s.mutex};
        // the draws the generator has and the loop did not issue, the epilogue, the counters
        if (v.cursor < v.shadow.marks.size()) {
            ++v.mismatches;
            NoteMismatch(s, v.view, "missing draw", fmt::format("the generator has {} draws, the loop issued {}", v.shadow.marks.size(), v.cursor));
        } else {
            v.mismatches += CompareDelta(s, v, v.arena_pos, v.shadow.arena.size(), plan.ctx, "epilogue");
        }
        const s32 draws = R<s32>(render_device + 0x10) - draws_before;
        const s64 triangles = R<s32>(render_device + 0x14) - triangles_before;
        if (draws != v.shadow.draws || triangles != v.shadow.triangles) {
            ++v.mismatches;
            NoteMismatch(s, v.view, "counters", fmt::format("draws +{} (generator {}), triangles +{} (generator {})", draws, v.shadow.draws, triangles, v.shadow.triangles));
        }
        ++stats->verify_passes;
        stats->verify_draws += v.compared_draws;
        stats->verify_raced += v.raced;
        stats->group_batches += plan.group_batches;
        if (v.mismatches == 0) {
            ++stats->verify_clean;
        } else {
            ++stats->verify_mismatched;
            stats->disabled = true;
            LOG_ERROR(Render, "cascade generator: view type {} switched off, {} differences against the engine's loop", v.view, v.mismatches);
        }
        v.shadow = Target{};
    } else {
        reinterpret_cast<Loop>(original_loop)(render_device, desc);
    }
    // learn what the loop chose
    {
        std::scoped_lock lock{s.mutex};
        if (!verify_pass) {
            ++stats->loop_passes;
        }
        if (plan.scanned && !plan.steps.empty()) {
            CacheArgs cache_args{&s, render_device};
            if (!NativeDraw::GuardedCall([](void* p) { CheckCacheImpl(*static_cast<CacheArgs*>(p)); }, &cache_args)) {
                ++s.scan_faults;
            }
            LearnArgs learn_args{&plan, &s};
            if (!NativeDraw::GuardedCall([](void* p) { LearnImpl(*static_cast<LearnArgs*>(p)); }, &learn_args)) {
                ++s.learn_faults;
            }
        }
    }
    return true;
}

nlohmann::json Report() {
    auto& s = S();
    std::scoped_lock lock{s.mutex};
    const Config& cfg = Cfg();
    nlohmann::json out = nlohmann::json::object();
    out["enabled"] = cfg.on;
    out["verify_every_n_passes"] = cfg.verify_every;
    nlohmann::json views_on = nlohmann::json::array();
    for (u32 i = 0; i < cfg.views.size(); ++i) {
        if (cfg.views[i]) {
            views_on.push_back(i);
        }
    }
    out["view_types_handled"] = views_on;
    u64 ambiguous = 0;
    for (const auto& [key, entry] : s.table) {
        ambiguous += entry.ambiguous ? 1 : 0;
    }
    out["variant_table"] = {{"keys", s.table.size()}, {"ambiguous_keys", ambiguous}, {"learned", s.learned}, {"conflicts", s.conflicts}, {"resets_after_a_cache_change", s.table_resets},
                            {"variants_not_in_the_engine_cache", s.learn_not_in_cache}, {"learn_faults", s.learn_faults}, {"scan_faults", s.scan_faults}, {"cache_entries", s.cache_seen.size()}};
    auto& views = out["views"] = nlohmann::json::object();
    for (const auto& [view, v] : s.views) {
        nlohmann::json entry = {{"passes", v.passes},
                                {"generated_passes_without_the_loop", v.generated},
                                {"generated_batches", v.generated_batches},
                                {"generated_draws", v.generated_draws},
                                {"passes_run_by_the_loop", v.loop_passes},
                                {"verify_passes", v.verify_passes},
                                {"verify_clean", v.verify_clean},
                                {"verify_with_differences", v.verify_mismatched},
                                {"verify_draws_compared", v.verify_draws},
                                {"verify_slot_words_that_moved_between_the_shadow_read_and_the_loop_read", v.verify_raced},
                                {"switched_off_by_verify", v.disabled},
                                {"scan_faults", v.not_scanned},
                                {"instance_group_batches_generated", v.group_batches},
                                {"instance_group_batches_with_no_instances", v.group_batches_skipped}};
        nlohmann::json kinds = nlohmann::json::object();
        for (const auto& [kind, count] : v.kinds) {
            kinds[std::to_string(kind)] = count;
        }
        entry["passes_by_kind_desc_0x10"] = kinds;
        nlohmann::json declines = nlohmann::json::object();
        for (const auto& [reason, count] : v.declines) {
            declines[reason] = count;
        }
        entry["declined_passes_by_reason"] = declines;
        views[std::to_string(view)] = entry;
    }
    out["verify_difference_kinds"] = s.mismatch_kinds;
    out["special_entry_classes"] = s.special_classes;
    out["verify_words_that_moved_by_view_kind_offset"] = s.raced_words;
    out["verify_difference_samples"] = s.samples;
    return out;
}

} // namespace Core::NativeZPassGen
