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
constexpr u32 DecalStageBytes = 0x1f0; // A7: the decal loop's constant block (renderDev + 0x3990 .. 0x3b80)
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

/// Cost census of what the generators call (`TLG_GEN_CENSUS=1`, /zpassgen key `engine_call_census`): calls and wall time of every engine function the generators still call, and of the
/// parts of the generator itself, per call kind. The time of a call includes everything it calls.
enum class CallId : u32 {
    ResourceSet,       // FUN_01c92f30
    GroupBind,         // FUN_01ee4e50
    ForwardPrologue,   // FUN_01c710f0
    FindVariant,       // FUN_01c822c0
    HandleLookup,      // FUN_01c59060 (the G-buffer prologue's)
    SpecialToken,      // the special entry's virtual +0x48
    HandleFree,        // FUN_01e45a60
    DrawIssue,         // the draw hook (tag, the engine's draw function)
    MaterialBlock,     // FUN_01c92b60 as plain stores (with the reader lock)
    ShaderPair,        // SetShader x2
    PassDepth,         // a whole generated depth pass
    PassGBuffer,       // a whole generated G-buffer pass
    PassForward,       // a whole generated forward pass
    ScanDepth,
    ScanGBuffer,
    ScanForward,
    PassDecal,
    ScanDecal,
    DecalUpdate, // the decal loop's UpdateSubresource (context virtual + 0xd0)
    PassSkin,
    ScanSkin,
    SkinUpdate,
    Count
};
constexpr const char* CallNames[static_cast<u32>(CallId::Count)] = {"FUN_01c92f30 resource set", "FUN_01ee4e50 instance group bind", "FUN_01c710f0 forward prologue", "FUN_01c822c0 variant find", "FUN_01c59060 handle lookup (G-buffer prologue)",
                                                                      "special entry virtual +0x48", "FUN_01e45a60 handle free", "draw issue (hook, tag, engine draw)", "FUN_01c92b60 material block (plain stores)",
                                                                      "shader pair (2 x SetShader)", "generated depth pass (whole)", "generated G-buffer pass (whole)", "generated forward pass (whole)", "scan, depth", "scan, G-buffer", "scan, forward", "generated decal pass (whole)", "scan, decal", "decal constant buffer update (UpdateSubresource)", "generated skinning loop (whole)", "scan, skinning", "skinning constant buffer update (UpdateSubresource)"};
struct CallCensus {
    std::atomic<u64> calls{0}, ticks{0};
};
bool CensusOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_CENSUS");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}
std::array<CallCensus, static_cast<u32>(CallId::Count)>& Census() {
    static std::array<CallCensus, static_cast<u32>(CallId::Count)> census;
    return census;
}
u64 Ticks() {
#ifdef _WIN32
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return static_cast<u64>(value.QuadPart);
#else
    return 0;
#endif
}
template <class F>
auto Timed(CallId id, F&& body) {
    if (!CensusOn()) {
        return body();
    }
    const u64 t0 = Ticks();
    struct Done {
        CallId id;
        u64 t0;
        ~Done() {
            auto& c = Census()[static_cast<u32>(id)];
            c.calls.fetch_add(1, std::memory_order_relaxed);
            c.ticks.fetch_add(Ticks() - t0, std::memory_order_relaxed);
        }
    } done{id, t0};
    return body();
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
    u64 count, first, a3;
    u64 vertex_shader, pixel_shader;
    // A3 (FUN_01c92f30 natively): what the resource set before this draw left in the device (the matrices block and the constant buffer it fills), for the verify run
    bool rs_valid{false};
    u32 rs_n{0}, rs_size{0};
    std::array<u8, 112> rs_bytes{};
    // A3 (FUN_01ee4e50 natively): the uploader buffer the group bind filled: the node's buffer and the staged data it was filled from
    u64 gb_buffer{0}, gb_src{0};
    u32 gb_n{0};
    // A7: the decal loop's constant block (renderDev + 0x3990) at this draw, index into Target::stages
    s32 stage{-1};
    // B3: the constant blocks (live address, bytes) the skinning loop updated buffers from before this dispatch
    std::vector<std::pair<u64, std::vector<u8>>> blocks;
    // B3: dword 3 of the block at flag_block is a flag computed from the float at flag_source, which the engine's main thread changes: the verify run re-reads it
    u64 flag_source{0}, flag_block{0};
};

/// Where the generator writes: the live device context, or a private copy of it that logs every write (the verify run).
struct Target {
    bool live{true};
    u8* base{nullptr};
    std::vector<u8> arena; // shadow: entries [u32 offset][u32 size][bytes]
    std::vector<Mark> marks;
    u64 batch_owner{0};                          // shadow: the batch whose level words (ctx + 0x580, 0x58, 0x288, 0x308) are being written
    std::unordered_map<size_t, u64> batch_owner_map;
    u64 indexed_site{0x01c76433}, indirect_site{0x01c7640f}, indexed_a4{1}; // the loop's call sites (ghidra return addresses) and the fifth argument of its indexed draw
    u64 table_entry{0};                          // shadow: the table entry object whose value is being written (ctx + 0x6c0 + 8 j)
    std::unordered_map<size_t, u64> table_owner; // shadow: arena position of an entry -> that table entry object
    u64 direct_object{0};                        // shadow: the engine object whose word (at direct_offset) the value being written was read from
    u32 direct_offset{0};
    std::unordered_map<size_t, std::pair<u64, u32>> direct_owner; // shadow: arena position of an entry -> (object, offset) it was read from
    u64 slot_material{0};                       // shadow: the material whose resource slot words are being written
    std::unordered_map<size_t, u64> slot_owner; // shadow: arena position of an entry -> that material
    s32 draws{0};
    s64 triangles{0};
    std::vector<std::array<u8, DecalStageBytes>> stages; // shadow (A7): the decal loop's constant block as the generator built it, one per updated primitive
    s32 stage_pending{-1};
    std::vector<std::pair<u64, std::vector<u8>>> blocks; // shadow (B3): blocks staged since the last dispatch
    bool rs_pending{false}; // shadow: the native resource set ran since the last draw
    u32 rs_n{0}, rs_size{0};
    std::array<u8, 112> rs_bytes{};
    u64 gb_buffer{0}, gb_src{0}; // shadow: the native group bind ran since the last draw
    u32 gb_n{0};

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
        if (direct_object != 0) {
            direct_owner[at] = {direct_object, direct_offset};
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

void BindMaterialImpl(Target& t, u64 material);
void BindMaterial(Target& t, u64 material) {
    if (!t.live) {
        BindMaterialImpl(t, material);
        return;
    }
    Timed(CallId::MaterialBlock, [&] { BindMaterialImpl(t, material); });
}
void BindMaterialImpl(Target& t, u64 material) {
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

/// Reuse census (`TLG_GEN_CENSUS=1`): how many draws find the device context, as far as the generators and the record builder look at it, exactly as the previous draw of the pass
/// left it (ctx bytes 0 to 0xd80, the shader pair, the raster state object); only the draw arguments differ then.
struct ReuseCensus {
    std::atomic<u64> draws{0}, same_as_previous{0}, same_as_one_of_last_8{0};
};
ReuseCensus& Reuse() {
    static ReuseCensus census;
    return census;
}
std::array<std::atomic<u64>, 0xd80 / 8> reuse_changed_words;
void CountReuse(const Target& t) {
    thread_local std::array<u64, 0xd80 / 8> previous{};
    for (u32 i = 0; i < 0xd80 / 8; ++i) {
        u64 word;
        std::memcpy(&word, t.base + i * 8, 8);
        if (word != previous[i]) {
            reuse_changed_words[i].fetch_add(1, std::memory_order_relaxed);
            previous[i] = word;
        }
    }
    thread_local std::array<u64, 8> recent{};
    thread_local u32 next = 0;
    thread_local u64 last = 0;
    u64 h = 0xcbf29ce484222325ull;
    for (u32 offset = 0; offset < 0xd80; offset += 8) {
        u64 word;
        std::memcpy(&word, t.base + offset, 8);
        h = (h ^ word) * 0x100000001b3ull;
    }
    for (const u32 offset : {0x4690u, 0x4698u, 0x46c8u}) {
        u64 word;
        std::memcpy(&word, t.base + offset, 8);
        h = (h ^ word) * 0x100000001b3ull;
    }
    auto& r = Reuse();
    r.draws.fetch_add(1, std::memory_order_relaxed);
    if (h == last) {
        r.same_as_previous.fetch_add(1, std::memory_order_relaxed);
    }
    bool seen = false;
    for (const u64 x : recent) {
        seen |= x == h;
    }
    if (seen) {
        r.same_as_one_of_last_8.fetch_add(1, std::memory_order_relaxed);
    }
    last = h;
    recent[next++ & 7] = h;
}

void IssueDraw(Target& t, u32 kind, u64 count, u64 first, u64 a3 = 0) {
    if (t.live && CensusOn()) {
        CountReuse(t);
    }
    if (t.live) {
        Timed(CallId::DrawIssue, [&] {
        if (kind == 0) {
            NativeDraw::IssueIndexedDraw(reinterpret_cast<u64>(t.base), count, first, t.indexed_site, t.indexed_a4, a3);
        } else {
            NativeDraw::IssueIndirectDraw(reinterpret_cast<u64>(t.base), count, first, t.indirect_site);
        }
        });
    } else {
        t.marks.push_back({t.arena.size(), kind, count, first, a3, t.Get<u64>(0x4698), t.Get<u64>(0x4690)});
        if (t.rs_pending) {
            Mark& m = t.marks.back();
            m.rs_valid = true;
            m.rs_n = t.rs_n;
            m.rs_size = t.rs_size;
            m.rs_bytes = t.rs_bytes;
            t.rs_pending = false;
        }
        if (t.gb_buffer != 0) {
            Mark& m = t.marks.back();
            m.gb_buffer = t.gb_buffer;
            m.gb_src = t.gb_src;
            m.gb_n = t.gb_n;
            t.gb_buffer = 0;
        }
    }
}

u64 HandleLookup(u64 render_device, u64 object);
void ResourceSetStep(Target& t, u64 rd, u64 material, u64 special, bool shadow_ok);
bool GroupBindNative(Target& t, const BatchLevel& level);

void Generate(Target& t, const Plan& plan) {
    const u64 rd = plan.render_device;
    const bool pure_pass = std::none_of(plan.batches.begin(), plan.batches.end(), [](const BatchLevel& b) { return b.special != 0; });
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
            if (Timed(CallId::GroupBind, [&] { return GroupBindNative(t, level); })) {
                // done natively (the context words below the engine's function would have written, and the uploader)
            } else if (t.live) {
                Timed(CallId::GroupBind, [&] { reinterpret_cast<GuestGroupBind>(Global(FunctionGroupBind))(level.inner, level.group_object, plan.view_type); });
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
                if (!level.bvar54) {
                    Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, level.special, pure_pass); });
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
                Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, level.special, pure_pass); });
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
    // one verify pass at a time over the one shadow: taken when a pass is chosen for verify (before the shadow is built, long before `active`), given back when it is done
    std::atomic<bool> claimed{false};
    std::thread::id thread;
    Target shadow;
    size_t cursor{0}, arena_pos{0};
    u32 view{0};
    u64 lo{ZPassBegin}, hi{ZPassEnd}; // the call sites of the loop under verify
    u64 ctx{0};
    u64 rd{0};
    u64 compared_draws{0};
    u64 mismatches{0};
    u64 raced{0};
    std::vector<s16> expect;
    std::vector<u64> owner_of; // per context byte: the material whose render object the shadow copied the byte from (0: none)
    std::array<u8, 0x340> forward_block{}; // A3: the block the native forward prologue built in the shadow (compared with the engine's at the first draw)
    bool forward_block_set{false};
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

u64 RenderDataOf(u64 render_device, u64 object);
constexpr u64 GlobalObjectTableForRace = 0x0689eed8, GlobalContextForRace = 0x069a1b40;
/// The byte the native resource set would put in the context byte `offset` (the texture render data slots at 0xa40 and 0x740) from the material's list as it is now, or -1. The
/// engine's other threads swap textures of a material (streaming): shadow and loop then read the list at different times.
int ResourceSlotByte(u64 material, u32 offset) {
    int index = -1;
    if (offset >= 0xa40 && offset < 0xa80) {
        index = static_cast<int>((offset - 0xa40) / 8);
    } else if (offset >= 0x740 && offset < 0x780) {
        index = static_cast<int>((offset - 0x740) / 8);
    }
    if (index < 0) {
        return -1;
    }
    const u64 table = R<u64>(Global(GlobalObjectTableForRace));
    const u64 list = R<u64>(R<u64>(Global(GlobalContextForRace)) + 0x38);
    u64 value = RenderDataOf(table, R<u64>(list + 0x30));
    if (static_cast<u32>(index) < std::min<u32>(8, R<u32>(material + 0x198))) {
        value = RenderDataOf(table, R<u64>(R<u64>(material + 400) + u64{static_cast<u32>(index)} * 8));
        if (value == 0) {
            value = RenderDataOf(table, R<u64>(list + 0x38));
        }
    }
    return static_cast<int>((value >> ((offset & 7) * 8)) & 0xff);
}

/// The byte the render object of `material` holds now for the context byte `offset` (the block copies of FUN_01c92b60), or -1.
int MaterialByte(u64 material, u32 offset) {
    const u64 object = R<u64>(material + 0x38);
    if (object == 0) {
        return -1;
    }
    if (offset >= 0x980 && offset < 0x9c8 + 8) {
        return R<u8>(object + 0x450 + (offset - 0x980));
    }
    if (offset >= 0x9d0 && offset < 0xa18) {
        return R<u8>(object + 0x498 + (offset - 0x9d0));
    }
    if (const u64 views = R<u64>(object + 0x4f0); views != 0 && offset >= 0x790 && offset < 0x790 + R<u32>(object + 0x4f8) * 8) {
        return R<u8>(views + (offset - 0x790));
    }
    if (const u64 views = R<u64>(object + 0x4e0); views != 0 && offset >= 0xb38 && offset < 0xb38 + R<u32>(object + 0x4e8) * 8) {
        return R<u8>(views + (offset - 0xb38));
    }
    return -1;
}

/// Compares the writes the generator made between two draws with the live context. Writes to the same bytes inside the span are applied in order (only the last counts: a batch
/// that draws nothing rewrites ctx + 0x580, the clear after an instanced batch overwrites the start of the table before it).
u32 CompareDelta(State& s, Verify& v, size_t from, size_t to, u64 live_ctx, const char* where) {
    u32 differences = 0;
    const u8* arena = v.shadow.arena.data();
    if (v.expect.empty()) {
        v.expect.assign(CtxBytes, -1);
        v.owner_of.assign(CtxBytes, 0);
    }
    std::vector<u32> touched;
    std::map<u32, u8> or_bits;
    struct SlotWord {
        u64 expected;
        u64 material;
    };
    std::map<u32, SlotWord> slot_words, table_words, batch_words;
    struct DirectWord {
        u64 expected;
        u64 object;
        u32 offset;
    };
    std::map<u32, DirectWord> direct_words;
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
        if (const auto downer = v.shadow.direct_owner.find(entry_at); downer != v.shadow.direct_owner.end() && bytes == 8) {
            u64 value = 0;
            std::memcpy(&value, expected, 8);
            direct_words[offset] = {value, downer->second.first, downer->second.second};
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
        const u64 material_owner = owner != v.shadow.slot_owner.end() ? owner->second : 0;
        for (u32 i = 0; i < bytes; ++i) {
            if (v.expect[offset + i] < 0) {
                touched.push_back(offset + i);
            }
            v.expect[offset + i] = expected[i];
            v.owner_of[offset + i] = material_owner;
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
                // the dynamic constants of a material's render object are rewritten by the engine's other threads: if the live byte is what the object holds now, shadow and loop only read at
                // different times
                if (const u64 material = v.owner_of[at_byte]; material != 0) {
                    int now = MaterialByte(material, at_byte);
                    if (now < 0) {
                        now = ResourceSlotByte(material, at_byte);
                    }
                    if (now >= 0 && now == live) {
                        ++v.raced;
                        s.raced_words[fmt::format("view {} material block byte {:#x}", v.view, at_byte & ~7u)]++;
                        continue;
                    }
                }
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
        v.owner_of[at_byte] = 0;
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
    for (const auto& [offset, word] : direct_words) {
        const u64 live = R<u64>(live_ctx + offset);
        if (live == word.expected) {
            continue;
        }
        // a view the engine's other threads replace in the object between the shadow read and the loop read
        const u64 now = R<u64>(word.object + word.offset);
        if (live == now) {
            ++v.raced;
            ++s.raced_words[fmt::format("view {} {} word {:#x}", v.view, "object", offset)];
            continue;
        }
        ++differences;
        NoteMismatch(s, v.view, fmt::format("context word {:#x}", offset), fmt::format("{}: the loop left {:#x}, the generator writes {:#x} (the object holds {:#x} now)", where, live, word.expected, now));
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


// ------------------------------------------------------------------------------------------------------------------------------- the G-buffer loop (N6.6)
// RenderDev_DrawGBufferPrimitives(renderDev, mode, list, args, flags, query mode, p7), ghidra 0x01c76710. What the live scenes ask of it (audit /gbuffer): mode 0, query mode 0, the
// flags 0x20; batches with and without an instance group, with and without a second render data (batch + 0x28); no special entries; every primitive has its variant cached behind its
// data (+0x40) and a negative +0x24. Everything else is declined. The variant is the cached one, so there is no key table here.

constexpr u64 GBufferBegin = 0x01c76710, GBufferEnd = 0x01c779a0; // DrawGBufferPrimitives and the next function (DrawForwardPass)
constexpr u64 GBufferIndexedSite = 0x01c777a1, GBufferIndirectSite = 0x01c77788;
constexpr u64 FunctionHandleLookup = 0x01c59060; // FUN_01c59060(render device, object) -> the object's render data or 0 (creates it at first use)

// A3 (documents/tlg/native plan/stages/a3-engine-functions.md): FUN_01c59060 is, once the object has its render data, three reads: `data = object[7]`; when `(int)object[4]` is
// not zero the virtual at `vtable + 0x28` is called with the object (a pending update; the value read before the call is the one returned); `data` is returned. The creation
// of the render data (operator new, vtable 0x026e7670, FUN_01c96930, FUN_01c98930) stays the engine's. `TLG_GEN_NATIVE_LEAF=1` takes the first case natively;
// `TLG_GEN_NATIVE_LEAF_VERIFY=N` (default 64) runs the engine's function instead for one call in N and compares its result with the native answer.
struct LeafStats {
    std::atomic<u64> native{}, engine_creating{}, engine_null{}, verified{}, differing{};
    std::atomic<u64> prologue_native{}, block_compared{}, block_differing{};
};
LeafStats& Leaf() {
    static LeafStats stats;
    return stats;
}
u64 HandleLookup(u64 render_device, u64 object) {
    static const bool native_on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    static const u32 verify_every = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 64u;
    }();
    const auto engine = [&] { return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64)>(Global(FunctionHandleLookup))(render_device, object); };
    if (!native_on) {
        return engine();
    }
    if (object == 0) {
        Leaf().engine_null.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u64 existing = R<u64>(object + 0x38);
    if (existing == 0) {
        Leaf().engine_creating.fetch_add(1, std::memory_order_relaxed);
        return engine();
    }
    thread_local u32 counter = 0;
    if (verify_every != 0 && (++counter % verify_every) == 0) {
        const u64 result = engine();
        Leaf().verified.fetch_add(1, std::memory_order_relaxed);
        if (result != existing) {
            Leaf().differing.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    if (R<s32>(object + 0x20) != 0) {
        reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(R<u64>(R<u64>(object) + 0x28))(object);
    }
    Leaf().native.fetch_add(1, std::memory_order_relaxed);
    return existing;
}
constexpr u64 GlobalObjectTable = 0x0689eed8;    // DAT_0689eed8

struct GBatch {
    BatchLevel level;     // value_580, view_58, pass_object_288, the table; group fields (inner, group_object, g70.., gtable)
    u64 batch{};
    u32 flags_value{0};   // ((batch flags & 4) << 4) | the pass flags
    bool has_second{false};
    u64 v840{0};
    bool has_mesh{false};
    bool pair{false};     // no group: the first table entry is a valid object: ctx + 0x588 / 0x590
    u64 v588{0}, v590{0};
    u64 e588{0}, e590{0}; // the table entry objects the two values come from
    u64 v68_group{0};     // group: ctx + 0x68
};

struct GStep {
    u32 index{};
    u64 record{}, material{}, variant{};
};

struct GPlan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0};
    u32 mode{0}, flags{0}, query_mode{0};
    s32 indirect_mode{0};
    u64 v_b60{0}, v_3c28{0}, v_68{0};
    std::vector<GBatch> batches;
    std::vector<GStep> steps;
    u32 group_batches{0};
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct GScanArgs {
    GPlan* plan;
    u64 render_device, list, args;
};

void ScanGBuffer(GScanArgs& a) {
    GPlan& plan = *a.plan;
    const u64 rd = a.render_device;
    plan.render_device = rd;
    plan.ctx = R<u64>(rd + 0x2d8);
    plan.indirect_mode = R<s32>(rd + 0x38e0);
    if (plan.ctx == 0 || a.list == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    if (plan.mode == 1) {
        plan.Decline("mode 1 (the other binding order, never seen)");
    }
    if (plan.query_mode == 1 || plan.query_mode == 2 || plan.query_mode == 5) {
        plan.Decline("query mode 1, 2, 5 (queries or a target switch)");
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    plan.v_3c28 = R<u64>(rd + 0x3c28);
    plan.v_68 = R<u64>(rd + 0x3d90 + u64(plan.mode == 1) * 8);
    {
        const u64 table = R<u64>(Global(GlobalObjectTable));
        const u64 handle = R<u64>(R<u64>(a.args + 0x2d0) + 0x30);
        const u64 object = Timed(CallId::HandleLookup, [&] { return HandleLookup(table, handle); });
        plan.v_b60 = object != 0 ? R<u64>(object + 0x48) : 0;
    }
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    if (batch_count != 0 && (batch_array == 0 || batch_count > 100000)) {
        plan.Decline("batch list unreadable");
        return;
    }
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        GBatch gb;
        gb.batch = batch;
        const u16 batch_flags = R<u16>(batch + 0x66);
        gb.flags_value = (u32{static_cast<u32>(batch_flags & 4)} << 4) | plan.flags;
        if (R<u64>(batch + 0x48) != 0) {
            plan.Decline("batch with a special entry");
            continue;
        }
        if (const u64 second = R<u64>(batch + 0x28); second != 0) {
            const u64 second_data = R<u64>(second + 0x38);
            if (second_data == 0 || R<u64>(second_data + 0x20) == 0) {
                plan.Decline("second render data not created yet");
                continue;
            }
            gb.has_second = true;
            gb.v840 = R<u64>(R<u64>(R<u64>(R<u64>(second_data + 0x20) + 0x48) + 0x38) + 0x28);
        }
        const u64 mesh = R<u64>(batch + 0x20);
        gb.has_mesh = mesh != 0;
        gb.level.first_step = static_cast<u32>(plan.steps.size());
        if (!gb.has_mesh) {
            plan.batches.push_back(gb);
            continue;
        }
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        FillBatchLevel(gb.level, batch, default_object);
        gb.level.first_step = static_cast<u32>(plan.steps.size());
        const u64 inner = gb.level.inner;
        const u64 group = R<u64>(batch + 0x38);
        if (group == 0) {
            const u32 count = R<u32>(inner + 0x68);
            if (count != 0) {
                const u64 array = R<u64>(inner + 0x60);
                const u64 first = R<u64>(array);
                if (first != default_object && first != 0 && R<u64>(first + 0x38) != 0) {
                    gb.pair = true;
                    gb.v588 = R<u64>(R<u64>(first + 0x38) + 0x28);
                    gb.e588 = first;
                    u64 second_entry = R<u64>(array + 8);
                    if (second_entry == default_object) {
                        second_entry = 0;
                    }
                    if (second_entry == 0) {
                        plan.Decline("table with a valid first entry and no second one (the loop reads through a null pointer)");
                        continue;
                    }
                    gb.v590 = R<u64>(R<u64>(second_entry + 0x38) + 0x28);
                    gb.e590 = second_entry;
                }
            }
        } else {
            const u64 group_inner = R<u64>(group + 8);
            if (group_inner == 0) {
                plan.Decline("instance group without its inner object");
                continue;
            }
            if (plan.indirect_mode == 0) {
                plan.Decline("instance group batch without the indirect mode (DrawIndexedInstanced is not hooked)");
                continue;
            }
            const u64 group_object = R<u64>(group_inner + 0x38);
            u32 instances = 0;
            if (group_object != 0) {
                instances = R<u32>(R<u64>(group_object + 0x28) + 0x48);
            }
            gb.level.group = true;
            ++plan.group_batches;
            if (group_object == 0 || instances == 0) {
                gb.level.only_580 = true; // the loop leaves the batch after ctx + 0x580
                plan.batches.push_back(gb);
                continue;
            }
            gb.level.group_object = group_object;
            gb.level.indirect_buffer = R<u64>(group_object + 0x110);
            const u32 views_count = R<u32>(group_object + 0xe0);
            const s32 ring_index = R<s32>(group_object + 0x20);
            if (views_count > 45 || (ring_index != 0 && ring_index != 1)) {
                plan.Decline("instance group with an unusual view count or ring index");
                continue;
            }
            const u64 ring_entry = R<u64>(R<u64>(group_object + 0xb8) + u64(ring_index) * 8);
            const u64 view_object = R<u64>(R<u64>(inner + 0x48) + 0x38);
            gb.level.g70 = R<u64>(view_object + 0x20);
            gb.level.g78 = R<u64>(ring_entry + 0x20);
            gb.level.g170 = R<u32>(view_object + 0x40);
            gb.level.g174 = R<u32>(ring_entry + 0x40);
            if (const u64 uploader = R<u64>(group_object + 0xf0); uploader != 0) {
                gb.level.g340 = R<u64>(R<u64>(uploader) + 0x10);
                gb.level.g340_set = true;
            }
            gb.level.gtable[0] = R<u64>(group_object + 0x1b0); // view type 0
            gb.level.gtable[1] = R<u64>(group_object + 0x1f8 + u64(ring_index) * 8);
            gb.level.gtable[2] = R<u64>(group_object + 0x1f8 + u64(1 - ring_index) * 8);
            const u64 views_array = R<u64>(group_object + 0xd8);
            for (u32 i = 0; i < views_count; ++i) {
                const u64 holder = R<u64>(views_array + u64{i} * 8);
                gb.level.gtable[3 + i] = holder != 0 ? R<u64>(holder + 0x28) : 0;
            }
            gb.level.table_dwords = 3 + views_count;
            gb.v68_group = plan.indirect_mode == 0 ? R<u64>(rd + 0x3da8) : R<u64>(rd + 0x3d90);
        }
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            GStep step;
            step.index = p;
            step.record = record;
            step.material = R<u64>(record + 0x18);
            const u64 data = R<u64>(record + 0x28);
            if (data == 0 || R<s32>(record + 0x24) >= 0) {
                plan.Decline(data == 0 ? "primitive record without its per primitive data" : "primitive with a non-negative +0x24 (another draw path)");
                continue;
            }
            step.variant = R<u64>(data + 0x40);
            if (step.variant == 0 || R<u32>(record + 0x30) != 0) {
                plan.Decline("primitive without a cached variant (the loop creates it)");
                continue;
            }
            if (R<u64>(step.variant + 8) == 0 || R<u64>(step.variant + 0x18) == 0) {
                plan.Decline("variant without a shader pair");
                continue;
            }
            const u64 object = R<u64>(step.material + 0x38);
            if (object == 0) {
                plan.Decline("material without a render object");
                continue;
            }
            const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
            if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72)) || (R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
                plan.Decline("material with a resource count or slot index the engine asserts on");
                continue;
            }
            plan.steps.push_back(step);
        }
        gb.level.step_count = static_cast<u32>(plan.steps.size()) - gb.level.first_step;
        plan.batches.push_back(gb);
    }
    plan.scanned = true;
}

void GenerateGBuffer(Target& t, const GPlan& plan, u32 pass_flags) {
    const u64 rd = plan.render_device;
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    t.Set<u64>(0x68, plan.v_68);
    t.Set<u64>(0xb60, plan.v_b60);
    bool raster_flag = false;
    u32 local_158 = 0, local_150 = pass_flags;
    for (const GBatch& gb : plan.batches) {
        const BatchLevel& level = gb.level;
        if (pass_flags != 0) {
            local_150 = local_158;
            if (local_158 != gb.flags_value) {
                t.Set<u64>(0x46e0, plan.v_3c28);
                t.Set<u32>(0x46e8, gb.flags_value);
                t.Set<u8>(0x46ec, 1);
                local_158 = gb.flags_value;
                local_150 = gb.flags_value;
            }
        }
        if (gb.has_second) {
            t.Set<u64>(0x840, gb.v840);
        }
        if (!gb.has_mesh) {
            continue;
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        t.batch_owner = 0;
        if (!level.group) {
            t.Set<u64>(0x68, 0);
            t.batch_owner = level.batch;
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
            if (gb.pair) {
                t.Set<u64>(0x580, level.value_580);
                t.table_entry = gb.e588;
                t.Set<u64>(0x588, gb.v588);
                t.table_entry = gb.e590;
                t.Set<u64>(0x590, gb.v590);
                t.table_entry = 0;
            }
        } else {
            if (level.only_580) {
                continue;
            }
            if (Timed(CallId::GroupBind, [&] { return GroupBindNative(t, level); })) {
                // done natively
            } else if (t.live) {
                Timed(CallId::GroupBind, [&] { reinterpret_cast<GuestGroupBind>(Global(FunctionGroupBind))(level.inner, level.group_object, 0); });
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
            t.Set<u64>(0x68, gb.v68_group);
            t.batch_owner = level.batch;
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
        }
        // FUN_01c932d0
        t.batch_owner = level.batch;
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const GStep& step = plan.steps[s];
            Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, 0, true); });
            const bool raster = (R<u32>(step.material + 0xd4) & 0x26) != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            SetShader(t, 0x4698, R<u64>(step.variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(step.variant + 0x18), 2);
            t.slot_material = step.material;
            BindMaterial(t, step.material);
            t.slot_material = 0;
            const s32 triangles = R<s32>(step.record + 0x14);
            if (level.group) {
                IssueDraw(t, 1, level.indirect_buffer, u64{step.index} * 0xd8);
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
            const std::array<u8, 0x30> zeros{};
            t.Put(0x70, zeros.data(), 16);
            t.Set<u64>(0x170, 0);
            t.Set<u64>(0x1f0, 0);
            t.Set<u64>(0x340, 0);
            t.Put(0x800, zeros.data(), 0x30);
        }
    }
    // epilogue: memset(ctx + 0x580, 0, 0x400), memset(ctx + 0x980, 0, 0x400)
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x580, zeros.data(), 0x400);
        t.Put(0x980, zeros.data(), 0x400);
    }
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
    t.Set<u64>(0x46e0, plan.v_3c28);
    t.Set<u32>(0x46e8, local_150);
    t.Set<u8>(0x46ec, 1);
}

struct GConfig {
    bool on{false};
    u32 verify_every{32};
};
const GConfig& GCfg() {
    static const GConfig config = [] {
        GConfig c;
        const char* e = std::getenv("TLG_GBUFFER_GEN");
        c.on = e != nullptr && *e != 0 && *e != '0';
        if (const char* n = std::getenv("TLG_GBUFFER_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

struct GStats {
    ViewStats v;
    std::map<u32, u64> arguments; // mode * 1000 + query mode -> passes
    u64 eligible{0};
};
GStats& GS() {
    static GStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the forward loop (N6.7)
// RenderDev_DrawForwardPass(renderDev, a1, a2, list, mode), ghidra 0x01c779a0. Batches with an instance group are skipped by the loop; the others are drawn plain. The variant of every
// primitive is found or created by the engine's own FUN_01c822c0 (called here, in both the live run and the verify run's shadow, so there is no key table). Live census: mode 0, no
// special entries, material + 0x158 == 0, the first table entry never a valid object.

constexpr u64 ForwardBegin = 0x01c779a0, ForwardEnd = 0x01c782a0;
constexpr u64 ForwardIndexedSite = 0x01c7819d;
constexpr u64 FunctionForwardPrologue = 0x01c710f0; // FUN_01c710f0(renderDev, a1, a2, mode)
constexpr u64 FunctionFindVariant = 0x01c822c0;     // FUN_01c822c0(renderDev + 1000, material, flags, 0, mesh word, &token, 0) -> variant or 0
constexpr u64 StringTable = 0x02716230;             // PTR_DAT_02716230: 0x20 bytes per entry, the name pointer first

// A3: FUN_01c822c0, the case the engine answers from `material + 0x1d0` (the variant it found last time) without touching its caches or any lock: the key is made of material
// fields, the flags and two FNV-1a hashes of names from the string table; when the cached variant carries the same four key words it is returned. Everything else (search of the
// engine's cache lists under its lock, creation) stays the engine's. `TLG_GEN_NATIVE_LEAF=3` (3 includes 1 and 2); verify: the engine's function is called for one hit in
// `TLG_GEN_NATIVE_LEAF_VERIFY` and must return the same variant (it has no side effect in this case).
u32 Fnv1a(const char* text) {
    u32 hash = 0x811c9dc5u;
    for (; *text != 0; ++text) {
        hash = (static_cast<u32>(static_cast<s32>(static_cast<s8>(*text))) ^ hash) * 0x1000193u;
    }
    return hash;
}
std::atomic<u64> last_forward_ctx{0}; // study: the device context seen by the forward generator (the report lists its vtable)
struct VariantStats {
    std::atomic<u64> native{}, engine{}, verified{}, differing{};
};
VariantStats& VStats() {
    static VariantStats stats;
    return stats;
}
u64 FindVariant(u64 cache, u64 material, u32 flags, u32 flags2, u32 mesh_word, u32* token) {
    static const bool native_on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && std::atoi(e) >= 3;
    }();
    static const u32 verify_every = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF_VERIFY");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 64u;
    }();
    const auto engine = [&] { return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u32, u32, u32, void*, u64)>(Global(FunctionFindVariant))(cache, material, flags, flags2, mesh_word, token, 0); };
    if (!native_on) {
        return engine();
    }
    const u64 cached = R<u64>(material + 0x1d0);
    if (cached != 0) {
        const s32 selector_164 = R<s32>(material + 0x164);
        const bool is_default = R<u64>(material + 0x18) == R<u64>(Global(GlobalDefaultKind));
        u32 key = flags | flags2 | (is_default ? 0x200u : 0u);
        const s32 mode = R<s32>(material + 0x15c);
        if (mode == 2) {
            key = ((flags | flags2) & 0xffffff6fu) | (is_default ? 0x200u : 0u) | 0x80u;
        } else {
            if (mode == 4) {
                key |= 0x20u;
            }
            if (mode == 3) {
                key |= 0x8080u;
            }
        }
        if (R<s32>(material + 0x158) != 0) {
            key |= 0x20000000u;
        }
        u32 name_hash = 0;
        const u64 shared = R<u64>(material + 0x48);
        if (shared != 0) {
            if (const u32 index = R<u32>(shared + 0x24); index != 0) {
                if (const u64 name = R<u64>(Global(StringTable) + u64{index} * 0x20); name != 0) {
                    name_hash = Fnv1a(reinterpret_cast<const char*>(name));
                }
            }
        }
        if (token[0] != 0) {
            name_hash = Fnv1a(reinterpret_cast<const char*>(R<u64>(Global(StringTable) + u64{token[0]} * 0x20)));
        }
        if (R<s32>(cached + 0x20) == selector_164 && R<u32>(cached + 0x24) == key && R<u32>(cached + 0x28) == (mesh_word & 0xffffu) && R<u32>(cached + 0x2c) == name_hash) {
            thread_local u32 counter = 0;
            if (verify_every != 0 && (++counter % verify_every) == 0) {
                const u64 result = engine();
                VStats().verified.fetch_add(1, std::memory_order_relaxed);
                if (result != cached) {
                    VStats().differing.fetch_add(1, std::memory_order_relaxed);
                }
                return result;
            }
            VStats().native.fetch_add(1, std::memory_order_relaxed);
            return cached;
        }
    }
    VStats().engine.fetch_add(1, std::memory_order_relaxed);
    return engine();
}
constexpr u64 FunctionHandleFree = 0x01e45a60;      // FUN_01e45a60(handle): the last reference went
constexpr u64 HandleRefTable = 0x02716240;          // DAT_02716240: 0x20 bytes per handle, the reference count first

void HandleRefInc(u64 handle) {
    reinterpret_cast<std::atomic<s32>*>(Global(HandleRefTable) + handle * 0x20)->fetch_add(1);
}
void HandleRefDec(u64 handle) {
    if (reinterpret_cast<std::atomic<s32>*>(Global(HandleRefTable) + handle * 0x20)->fetch_sub(1) == 1) {
        Timed(CallId::HandleFree, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(Global(FunctionHandleFree))(handle); });
    }
}

// A3: FUN_01c92f30(renderDev + 0x2910, ctx, material, special), the per primitive resource set, read off the decompilation (stages/a3-engine-functions.md). `TLG_GEN_NATIVE_LEAF=4`
// makes it natively for a primitive without a special entry, with the constant buffer of the matrices updated the way the context's UpdateSubresource (vtable + 0xd0,
// FUN_02480f90) does it for the dynamic buffer kind; any other shape runs the engine's function. In the verify run's shadow the effects on the device (the matrices block, the
// buffer's descriptor and ring block) cannot be applied: the shadow computes the expected bytes, the first draw after the call compares them with what the engine's loop left
// (`NoteLoopDraw`), and a pass with a special entry is not modelled in the shadow at all (as before).
constexpr u64 FunctionAllocFrameGarlic = 0x02476350; // JsDevice_AllocFrameGarlic(allocator, size, align)
constexpr u64 GlobalDeviceHolder = 0x06b03558;       // DAT_06b03558: holder of the frame allocator (+ 0x10238)
u64 RenderDataOf(u64 render_device, u64 object);
struct RsStatsT {
    std::atomic<u64> native{}, engine_special{}, engine_shape{}, compared{}, differing{};
};
RsStatsT& RsStats() {
    static RsStatsT stats;
    return stats;
}
bool NativeResourceSetOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && std::atoi(e) >= 4;
    }();
    return on;
}
void ResourceSetStep(Target& t, u64 rd, u64 material, u64 special, bool shadow_ok) {
    const u64 rd2910 = rd + 0x2910;
    const auto engine = [&] { reinterpret_cast<GuestBind>(Global(FunctionBindResources))(rd2910, reinterpret_cast<u64>(t.base), material, special); };
    const bool native = NativeResourceSetOn() && special == 0;
    if (!native) {
        if (special != 0 && NativeResourceSetOn()) {
            RsStats().engine_special.fetch_add(1, std::memory_order_relaxed);
        }
        if (t.live) {
            engine();
        }
        return;
    }
    if (!t.live && !shadow_ok) {
        return;
    }
    if (!(R<s32>(material + 0x1bc) != 0 || R<u64>(material + 0x48) != 0)) {
        RsStats().native.fetch_add(1, std::memory_order_relaxed); // the engine's function does nothing for this material
        return;
    }
    const u64 buffer = R<u64>(rd2910 + 0xef0);
    if (buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) < 112) {
        RsStats().engine_shape.fetch_add(1, std::memory_order_relaxed);
        if (t.live) {
            engine();
        }
        return;
    }
    const u64 table = R<u64>(Global(GlobalObjectTable));
    const u64 list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38);
    std::array<u64, 8> slot;
    slot.fill(RenderDataOf(table, R<u64>(list + 0x30)));
    // the matrices: up to seven 16 byte entries from the material into the block at renderDev + 0x3790 (the shadow keeps its own copy)
    const u32 n = std::min<u32>(7, R<u32>(material + 0x180));
    std::array<u8, 112> matrices{};
    if (n != 0) {
        std::memcpy(matrices.data(), reinterpret_cast<const void*>(R<u64>(material + 0x178)), n * 16);
    }
    // the textures: the material's list is read under its spin lock
    auto* lock = reinterpret_cast<std::atomic<s32>*>(material + 0x1a4);
    for (;;) {
        s32 expected = 0;
        if (lock->compare_exchange_strong(expected, 1)) {
            break;
        }
        std::this_thread::yield();
    }
    const u32 count = std::min<u32>(8, R<u32>(material + 0x198));
    const u64 entries = R<u64>(material + 400);
    for (u32 i = 0; i < count; ++i) {
        u64 data = RenderDataOf(table, R<u64>(entries + u64{i} * 8));
        if (data == 0) {
            data = RenderDataOf(table, R<u64>(list + 0x38));
        }
        slot[i] = data;
    }
    {
        s32 expected = 1;
        lock->compare_exchange_strong(expected, 0);
    }
    const u32 mask = R<u32>(material + 0x170);
    if ((mask & 1) != 0) {
        for (u32 i = 0; i < 8; ++i) {
            t.Set<u64>(0xa40 + i * 8, slot[i]);
        }
    }
    if ((mask & 2) != 0) {
        for (u32 i = 0; i < 8; ++i) {
            t.Set<u64>(0x740 + i * 8, slot[i]);
        }
    }
    const u32 size = R<u32>(buffer + 0x180);
    if (t.live) {
        // UpdateSubresource on the dynamic buffer: the whole block at renderDev + 0x3790 goes to a block of the frame ring, the buffer's descriptor points at it
        if (n != 0) {
            std::memcpy(reinterpret_cast<void*>(rd2910 + 0xe80), matrices.data(), n * 16);
        }
        W<u32>(buffer + 0x2c, t.Get<u32>(0x47ac));
        const u64 holder = R<u64>(Global(GlobalDeviceHolder));
        void* block = reinterpret_cast<void*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), size, 0x10));
        std::memcpy(block, reinterpret_cast<const void*>(rd2910 + 0xe80), size);
        const u64 address = reinterpret_cast<u64>(block);
        W<u32>(buffer + 0x60, static_cast<u32>(address));
        W<u32>(buffer + 0x64, static_cast<u32>(address >> 32) & 0xfffu | 0x100000u);
        W<u32>(buffer + 0x6c, 0x20077facu);
        W<u32>(buffer + 0x68, (size + 0xfu) >> 4);
        const s64 version = reinterpret_cast<std::atomic<s64>*>(holder + 0x302a0)->fetch_add(1);
        W<s64>(buffer + 0x160, version);
    } else {
        t.rs_pending = true;
        t.rs_n = n;
        t.rs_size = size;
        t.rs_bytes = matrices;
    }
    t.Set<u64>(0x2e0, buffer);
    t.Set<u64>(0x360, buffer);
    RsStats().native.fetch_add(1, std::memory_order_relaxed);
}

// A3: FUN_01ee4e50(inner, group, view type), the instance group bind. The context words and the table are the values the scan already read (the shadow writes the same); what is
// left is the uploader: the staged instance data goes into the uploader's ring block, the node's buffer is updated like the context's UpdateSubresource does it for a dynamic
// buffer, and the double buffer index toggles. `TLG_GEN_NATIVE_LEAF=5`. A buffer of another shape runs the engine's function (returns false).
int LeafLevel() {
    static const int level = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 ? std::atoi(e) : 0;
    }();
    return level;
}
struct GbStatsT {
    std::atomic<u64> native{}, engine_shape{}, compared{}, differing{};
};
GbStatsT& GbStats() {
    static GbStatsT stats;
    return stats;
}
bool GroupBindNative(Target& t, const BatchLevel& level) {
    if (LeafLevel() < 5) {
        return false;
    }
    const u64 uploader = R<u64>(level.group_object + 0xf0);
    u64 node = 0, buffer = 0, value = 0, staged = 0;
    s32 staged_bytes = 0;
    if (uploader != 0) {
        node = R<u64>(uploader);
        buffer = R<u64>(node + 0x10);
        const s32 index = R<s32>(uploader + 0x28);
        value = R<u64>(uploader + 0x18 + static_cast<u64>(index & 1) * 8);
        staged = R<u64>(uploader + 8);
        staged_bytes = R<s32>(uploader + 0x10);
        if ((index != 0 && index != 1) || buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) == 0 || staged_bytes < 0 ||
            level.g340 != buffer) {
            GbStats().engine_shape.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    t.Set<u64>(0x70, level.g70);
    t.Set<u64>(0x78, level.g78);
    t.Set<u32>(0x170, level.g170);
    t.Set<u32>(0x174, level.g174);
    t.Set<u64>(0x1f0, 0);
    if (uploader != 0) {
        if (t.live) {
            std::memcpy(reinterpret_cast<void*>(value), reinterpret_cast<const void*>(staged), static_cast<size_t>(staged_bytes));
            W<u64>(node, value);
            const u32 size = R<u32>(buffer + 0x180);
            W<u32>(buffer + 0x2c, t.Get<u32>(0x47ac));
            const u64 holder = R<u64>(Global(GlobalDeviceHolder));
            void* block = reinterpret_cast<void*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), size, 0x10));
            std::memcpy(block, reinterpret_cast<const void*>(value), size);
            const u64 address = reinterpret_cast<u64>(block);
            W<u32>(buffer + 0x60, static_cast<u32>(address));
            W<u32>(buffer + 0x64, static_cast<u32>(address >> 32) & 0xfffu | 0x100000u);
            W<u32>(buffer + 0x6c, 0x20077facu);
            W<u32>(buffer + 0x68, (size + 0xfu) >> 4);
            W<s64>(buffer + 0x160, reinterpret_cast<std::atomic<s64>*>(holder + 0x302a0)->fetch_add(1));
            W<u8>(uploader + 0x28, R<u8>(uploader + 0x28) ^ 1);
        } else {
            t.gb_buffer = buffer;
            t.gb_src = staged;
            t.gb_n = std::min<u32>(static_cast<u32>(staged_bytes), R<u32>(buffer + 0x180));
        }
        t.Set<u64>(0x340, level.g340);
    }
    t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
    GbStats().native.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// A3: FUN_0198c730, the special entry's virtual +0x48 (forward pass): the handle token and its reference count. The case with the static VERTEX_MODIFIER handle already created
// is answered natively (`TLG_GEN_NATIVE_LEAF=5`); the first use (the function-static guard still clear) is the engine's.
bool SpecialTokenNative(u32* out, u64 special, u64 material);

// A3: FUN_01c710f0 and FUN_01c70cc0 (the forward prologue), read off the decompilation (documents/tlg/native plan/stages/a3-engine-functions.md). `TLG_GEN_NATIVE_LEAF=2`
// (the lookup too): the prologue is made natively, in the live run and in the verify run's shadow. The shadow cannot take a block from the frame ring: it builds the block in
// `Verify::forward_block` and the first draw of the engine's loop compares it with the block the engine's prologue put in the ring (`NoteLoopDraw`).
bool NativePrologueOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && std::atoi(e) >= 2;
    }();
    return on;
}
u64 RenderDataOf(u64 render_device, u64 object) {
    const u64 data = object != 0 ? HandleLookup(render_device, object) : 0;
    return data != 0 ? R<u64>(data + 0x48) : 0;
}
/// FUN_01c70cc0(renderDev, ctx, 0, desc): ten render targets -> ctx + 0xac0 .. 0xb08.
void TargetSlotsNative(Target& t, u64 rd, u64 desc) {
    const u64 list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38); // DAT_069a1b40 holds the pointer
    const u32 count = R<u32>(list + 0x88);
    const u64 array = R<u64>(list + 0x80);
    std::array<u64, 10> slot{};
    for (u32 i = 0; i < 10; ++i) {
        slot[i] = RenderDataOf(rd, i < count ? R<u64>(array + u64{i} * 8) : 0);
    }
    if (const u64 d = R<u64>(desc + 0x458); d != 0 && R<s32>(d + 0x270) == 3 && R<s32>(d + 0x258) == 7) {
        const u64 colour = R<u64>(d + 0x250), depth = R<u64>(d + 0x268);
        for (u32 i = 0; i < 7; ++i) {
            slot[i] = RenderDataOf(rd, R<u64>(colour + u64{i} * 8));
        }
        for (u32 i = 0; i < 3; ++i) {
            slot[7 + i] = RenderDataOf(rd, R<u64>(depth + u64{i} * 8));
        }
    }
    for (u32 i = 0; i < 10; ++i) {
        t.Set<u64>(0xac0 + i * 8, slot[i]);
    }
}
/// FUN_01c710f0(renderDev, a1, a2, mode).
void ForwardPrologueNative(Target& t, u64 rd, u64 a1, u64 a2, u32 mode) {
    Leaf().prologue_native.fetch_add(1, std::memory_order_relaxed);
    TargetSlotsNative(t, rd, a2);
    const u64 second = mode == 2 ? 1 : 0;
    t.Set<u64>(0xa80, R<u64>(R<u64>(R<u64>(a1 + 0x20) + 0x20) + 0x48));
    t.Set<u64>(0xa88, R<u64>(rd + 0x4448));
    t.Set<u64>(0xa90, R<u64>(rd + 0x4458));
    t.Set<u64>(0xa98, R<u64>(R<u64>(a1 + 0x38) + 0x18));
    t.Set<u64>(0xaa0, R<u64>(rd + 0x87e8 + second * 0x28));
    t.Set<u64>(0xaa8, R<u64>(rd + 0x4460 + second * 8));
    t.Set<u64>(0xab0, R<u64>(R<u64>(a1 + 0x38) + 0x38 + second * 8));
    const u64 target_desc = R<u64>(a2 + 0x448);
    const u64 object5 = R<u64>(target_desc + 0x240), object6 = R<u64>(target_desc + 0x248);
    const u64 data5 = RenderDataOf(rd, object5), data6 = RenderDataOf(rd, object6);
    std::array<u8, 0x340> local{};
    u8* block = local.data();
    if (t.live) {
        const u64 holder = R<u64>(Global(GlobalDeviceHolder));
        block = reinterpret_cast<u8*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), 0x340, 4));
    }
    std::memcpy(block, reinterpret_cast<const void*>(rd + 0x2910), 0x2b0);
    std::memcpy(block + 0x2b0, reinterpret_cast<const void*>(rd + 0x2dc0), 0x50);
    if (data5 != 0) {
        std::memcpy(block + 0x300, reinterpret_cast<const void*>(data5 + 0x58), 0x20);
    } else {
        std::memset(block + 0x300, 0, 0x20);
    }
    if (data6 != 0) {
        std::memcpy(block + 0x320, reinterpret_cast<const void*>(data6 + 0x58), 0x20);
    } else {
        std::memset(block + 0x320, 0, 0x20);
    }
    if (t.Get<u32>(0x13088) < 2) {
        t.Set<u32>(0x13088, 2);
    }
    if (t.Get<u32>(0x13060) < 2) {
        t.Set<u32>(0x13060, 2);
    }
    if (t.live) {
        t.Set<u64>(0x13070, reinterpret_cast<u64>(block));
        t.Set<u64>(0x13048, reinterpret_cast<u64>(block));
    } else {
        Verify& v = V();
        v.forward_block = local;
        v.forward_block_set = true;
    }
    t.Set<u64>(0xa28, data5);
    t.Set<u64>(0xa30, data6);
    t.Set<u64>(0xbb0, 0);
    t.Set<u64>(0xbb8, R<u64>(rd + 0x87c0));
}

constexpr u64 GlobalModifierGuard = 0x06888938, GlobalModifierHandle = 0x06888940; // DAT_06888938 / DAT_06888940: the function-static VERTEX_MODIFIER handle
bool SpecialTokenNative(u32* out, u64 special, u64 material) {
    if (LeafLevel() < 5) {
        return false;
    }
    if (R<u64>(special + 0x40) == 0) {
        out[0] = R<u32>(material + 0x1bc);
    } else if (R<u8>(Global(GlobalModifierGuard)) != 0) {
        out[0] = R<u32>(Global(GlobalModifierHandle));
    } else {
        return false;
    }
    HandleRefInc(out[0]);
    return true;
}

struct FBatch {
    BatchLevel level;
    u64 batch{};
    bool has_second{false};
    u64 v840{0};
    bool valid_first{false}; // the mesh's first table entry is a valid object (the other branch of the loop)
    u64 view_g70{0};
    u32 view_g170{0};
    u32 mesh_word{0}, base_vertex{0};
    u32 variant_flags{0};
    u64 special{0}; // batch + 0x48 (class 0x26dbfc8)
};

struct FStep {
    u64 record{}, material{};
};

struct FPlan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0}, a1{0}, a2{0};
    u32 mode{0};
    u64 v_3ca8{0}, v_3d90{0};
    std::vector<FBatch> batches;
    std::vector<FStep> steps;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct FScanArgs {
    FPlan* plan;
    u64 list;
};

void ScanForwardPass(FScanArgs& a) {
    FPlan& plan = *a.plan;
    const u64 rd = plan.render_device;
    plan.ctx = R<u64>(rd + 0x2d8);
    if (plan.ctx == 0 || a.list == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    plan.v_3ca8 = R<u64>(rd + 0x3ca8);
    plan.v_3d90 = R<u64>(rd + 0x3d90);
    const u32 base_flags = plan.mode == 2 ? 0x10000010u : 0x10000000u;
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    if (batch_count != 0 && (batch_array == 0 || batch_count > 100000)) {
        plan.Decline("batch list unreadable");
        return;
    }
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        if (R<u64>(batch + 0x38) != 0) {
            continue; // the forward loop leaves batches with an instance group alone
        }
        const u64 mesh = R<u64>(batch + 0x20);
        if (mesh == 0) {
            continue;
        }
        FBatch fb;
        fb.batch = batch;
        fb.variant_flags = base_flags;
        if (const u64 special = R<u64>(batch + 0x48); special != 0) {
            if (R<u64>(special) - NativeDraw::ImageBase() + 0x1000000 != 0x26dbfc8) {
                plan.Decline("batch with a special entry of an unknown class");
                continue;
            }
            fb.special = special;
            fb.variant_flags |= 0x40;
        }
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        if (const u64 second = R<u64>(batch + 0x28); second != 0) {
            const u64 second_data = R<u64>(second + 0x38);
            if (second_data == 0 || R<u64>(second_data + 0x20) == 0) {
                plan.Decline("second render data not created yet");
                continue;
            }
            fb.has_second = true;
            fb.variant_flags |= 0x80000000u;
            fb.v840 = R<u64>(R<u64>(R<u64>(R<u64>(second_data + 0x20) + 0x48) + 0x38) + 0x28);
        }
        FillBatchLevel(fb.level, batch, default_object);
        const u64 inner = fb.level.inner;
        {
            const u32 count = R<u32>(inner + 0x68);
            const u64 first = count != 0 ? R<u64>(R<u64>(inner + 0x60)) : 0;
            fb.valid_first = count != 0 && first != default_object && first != 0 && R<u64>(first + 0x38) != 0;
            if (fb.valid_first) {
                fb.variant_flags |= 0x100;
            } else {
                const u64 view_object = R<u64>(R<u64>(inner + 0x48) + 0x38);
                fb.view_g70 = R<u64>(view_object + 0x20);
                fb.view_g170 = R<u32>(view_object + 0x40);
            }
        }
        fb.mesh_word = R<u32>(mesh + 0x78);
        fb.base_vertex = R<u32>(batch + 0x60);
        fb.level.first_step = static_cast<u32>(plan.steps.size());
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            FStep step;
            step.record = record;
            step.material = R<u64>(record + 0x18);
            const u32 selector = R<u32>(step.material + 0x158);
            if (selector > 3) {
                plan.Decline("material + 0x158 above 3");
                continue;
            }
            const u64 object = R<u64>(step.material + 0x38);
            if (object == 0) {
                plan.Decline("material without a render object");
                continue;
            }
            const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
            if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72)) || (R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
                plan.Decline("material with a resource count or slot index the engine asserts on");
                continue;
            }
            plan.steps.push_back(step);
        }
        fb.level.step_count = static_cast<u32>(plan.steps.size()) - fb.level.first_step;
        plan.batches.push_back(fb);
    }
    plan.scanned = true;
}

void GenerateForward(Target& t, const FPlan& plan) {
    const u64 rd = plan.render_device;
    const bool pure_pass = std::none_of(plan.batches.begin(), plan.batches.end(), [](const FBatch& b) { return b.special != 0; });
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    if (NativePrologueOn() && R<u64>(rd + 0x2d8) == plan.ctx) {
        Timed(CallId::ForwardPrologue, [&] { ForwardPrologueNative(t, rd, plan.a1, plan.a2, plan.mode); });
    } else if (t.live) {
        Timed(CallId::ForwardPrologue, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64)>(Global(FunctionForwardPrologue))(rd, plan.a1, plan.a2, plan.mode); });
    }
    const auto state_words = [&](u64 entry) {
        t.Set<u64>(0x46f0, entry);
        t.Set<u64>(0x46f8, 0x3f8000003f800000ull);
        t.Set<u64>(0x4700, 0x3f8000003f800000ull);
        t.Set<u32>(0x4708, 0xffffffffu);
        t.Set<u8>(0x470c, 1);
    };
    state_words(plan.v_3ca8);
    t.Set<u64>(0x68, 0);
    bool raster_flag = false;
    u32 last_state = 0;
    for (const FBatch& fb : plan.batches) {
        const BatchLevel& level = fb.level;
        if (fb.has_second) {
            t.Set<u64>(0x840, fb.v840);
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        if (!fb.valid_first) {
            t.Set<u64>(0x70, fb.view_g70);
            t.Set<u32>(0x170, fb.view_g170);
            t.Set<u32>(0x1f0, 0);
            t.Set<u64>(0x58, level.view_58);
            t.Set<u64>(0x60, 0x39);
            t.batch_owner = 0;
            t.Set<u64>(0x68, plan.v_3d90);
        } else {
            t.Set<u64>(0x68, 0);
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const FStep& step = plan.steps[s];
            const u32 selector = R<u32>(step.material + 0x158);
            const bool cached_object = R<u64>(step.material + 0x48) != 0;
            u32 state = 0;
            if (selector == 3) {
                state = cached_object ? 0xf : 4;
            } else if (selector == 2) {
                state = cached_object ? 0xe : 3;
            } else if (selector == 1) {
                state = cached_object ? 0xd : 2;
            }
            if (state != last_state) {
                state_words(R<u64>(rd + 0x3ca8 + u64{state} * 8));
                last_state = state;
            }
            Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, fb.special, pure_pass); });
            // the handle sequence of the loop: the material's handle is counted, a special entry replaces it by the handle its virtual +0x48 returns (counted by the virtual), the variant search
            // gets the resulting handle
            u64 token = R<u32>(step.material + 0x1bc);
            HandleRefInc(token);
            if (fb.special != 0) {
                std::array<u32, 2> out{0, 0};
                Timed(CallId::SpecialToken, [&] {
                    if (!SpecialTokenNative(out.data(), fb.special, step.material)) {
                        reinterpret_cast<void(PS4_SYSV_ABI*)(void*, u64, u64)>(R<u64>(R<u64>(fb.special) + 0x48))(out.data(), fb.special, step.material);
                    }
                });
                u64 released = token;
                if (token != out[0]) {
                    HandleRefInc(out[0]);
                    if (token != 0) {
                        HandleRefDec(token);
                    }
                    released = out[0];
                    token = out[0];
                }
                if (released != 0) {
                    HandleRefDec(released);
                }
            }
            const bool raster = (R<u32>(step.material + 0xd4) & 0x26) != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            HandleRefInc(token);
            std::array<s32, 2> token_slot{static_cast<s32>(token), 0};
            const u64 variant = Timed(CallId::FindVariant, [&] { return FindVariant(rd + 1000, step.material, fb.variant_flags, 0, fb.mesh_word, reinterpret_cast<u32*>(token_slot.data())); });
            if (token != 0) {
                HandleRefDec(token);
            }
            if (variant != 0) {
                SetShader(t, 0x4698, R<u64>(variant + 8), 1);
                SetShader(t, 0x4690, R<u64>(variant + 0x18), 2);
                t.slot_material = step.material;
                BindMaterial(t, step.material);
                t.slot_material = 0;
                const s32 triangles = R<s32>(step.record + 0x14);
                IssueDraw(t, 0, static_cast<u32>(triangles * 3), R<u32>(step.record + 0x10), (fb.variant_flags & 0x100) == 0 ? fb.base_vertex : 0);
                if (t.live) {
                    W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                    W<s32>(rd + 0x14, R<s32>(rd + 0x14) + triangles);
                } else {
                    ++t.draws;
                    t.triangles += triangles;
                }
            }
            if (token != 0) {
                HandleRefDec(token);
            }
        }
    }
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x980, zeros.data(), 0x400);
        t.Put(0x580, zeros.data(), 0x400);
    }
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
}

struct FConfig {
    bool on{false};
    u32 verify_every{32};
};
const FConfig& FCfg() {
    static const FConfig config = [] {
        FConfig c;
        const char* e = std::getenv("TLG_FORWARD_GEN");
        c.on = e != nullptr && *e != 0 && *e != '0';
        if (const char* n = std::getenv("TLG_FORWARD_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

struct FStats {
    ViewStats v;
    u64 eligible{0};
};
FStats& FS() {
    static FStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the decal pass loop (A7)
// FUN_01c71680(renderDev, view, a3, a4, desc, flags, type, width, height) draws the decals of one view type: `desc + 0x480 + type * 0x18` batches (list at + 0x478), every primitive of a batch
// is one decal object (record + 0x18) drawn with an inlined `DrawIndexAuto`. The loop writes the pass constants once (the block at renderDev + 0x39f0 .. 0x3a3c and 0x3b40 .. 0x3b7c), then per
// primitive the block at renderDev + 0x3990 .. 0x3b3c from the decal object, updates the constant buffer at renderDev + 0x3b80 from the whole block (UpdateSubresource, the context's virtual +
// 0xd0), puts five render data views into the context (0xa38 .. 0xa58), asks the variant cache for the shader pair and draws. Decompiled in documents/tlg/native plan/stages/a7-decal-pass.md.
// The generator makes the same context writes (live: into the device context; verify: into the shadow copy, compared word by word at every draw), the same block (compared byte by byte at
// every draw against the block the engine's loop left) and the same draws. What stays the engine's: the handle lookups that create render data at first use (`FUN_01c59060`), the variant
// search, the buffer update (UpdateSubresource) and the index buffer call (`FUN_02488430`).
constexpr u64 DecalBegin = 0x01c71680, DecalEnd = 0x01c72a20; // the loop and the next function
constexpr u64 DecalDrawSite = 0x01c727a7;                     // the return address of its inlined PrepareDraw
constexpr u64 FunctionSetIndexBuffer = 0x02488430;            // FUN_02488430(ctx, index buffer): emits a SET_INDEX_BASE when the base changed (the loop passes 0)
constexpr u32 DecalStageBase = 0x3990;                        // renderDev + 0x3990 .. + 0x3b80: the constant block the buffer at renderDev + 0x3b80 is updated from

struct DStep {
    u64 record{}, decal{};
    u32 index{};
};
struct DBatch {
    u64 batch{};
    bool set_580{false};
    u64 value_580{};
    u32 first_step{0}, step_count{0};
};
struct DPlan {
    bool scanned{false};
    bool no_work{false}; // desc + 0x480 + type * 0x18 is zero: the function does nothing at all
    bool any_mesh{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0}, view{0}, a3{0}, a4{0}, desc{0};
    u32 flags{0}, width{0}, height{0};
    s32 type{0};
    std::vector<DBatch> batches;
    std::vector<DStep> steps;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct DScanArgs {
    DPlan* plan;
};

void ScanDecalPass(DScanArgs& a) {
    DPlan& plan = *a.plan;
    const u64 rd = plan.render_device;
    const u64 desc = plan.desc;
    if (plan.type < 0 || plan.type > 64) {
        plan.Decline("view type out of range");
        return;
    }
    const u64 list_count_at = desc + 0x480 + u64{static_cast<u32>(plan.type)} * 0x18, list_array_at = desc + 0x478 + u64{static_cast<u32>(plan.type)} * 0x18;
    const u32 count = R<u32>(list_count_at);
    if (count == 0) {
        plan.no_work = true;
        plan.scanned = true;
        return;
    }
    plan.ctx = R<u64>(rd + 0x2d8);
    if (plan.ctx == 0) {
        plan.Decline("no device context");
        return;
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    if (!NativeDraw::InlineAutoDrawTagged(1)) {
        plan.Decline("native draw execution is off (the draw hook does not take the inlined auto draw)");
    }
    if (R<u64>(desc + 0x298) == 0 || R<u64>(desc + 0x448) == 0 || R<u64>(R<u64>(desc + 0x448) + 0x2b0) == 0) {
        plan.Decline("pass description without its constants object or target description");
        return;
    }
    const u64 array = R<u64>(list_array_at);
    if (array == 0 || count > 100000) {
        plan.Decline("batch list unreadable");
        return;
    }
    u64 last_mesh = 0;
    for (u32 b = 0; b < count; ++b) {
        const u64 batch = R<u64>(array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        const u64 mesh = R<u64>(batch + 0x20);
        DBatch db;
        db.batch = batch;
        bool enter = false;
        if (mesh == last_mesh) {
            if (mesh == 0) {
                plan.Decline("first batch without a mesh");
                continue;
            }
            enter = true;
        } else if (mesh != 0) {
            const u64 render_data = R<u64>(mesh + 0x38);
            if (render_data == 0) {
                plan.Decline("mesh render data not created yet (the loop creates it at first use)");
                continue;
            }
            const u64 inner = R<u64>(render_data + 0x20);
            if (inner == 0 || R<u64>(inner + 0x48) == 0 || R<u64>(R<u64>(inner + 0x48) + 0x38) == 0) {
                plan.Decline("mesh render data without its inner object");
                continue;
            }
            if (R<u64>(inner + 0x50) == 0) {
                db.set_580 = true;
                db.value_580 = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28);
                last_mesh = mesh;
                enter = true;
            }
        }
        if (!enter) {
            continue; // the loop leaves the batch alone
        }
        db.first_step = static_cast<u32>(plan.steps.size());
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            if (R<s32>(record + 0x14) == 0) {
                continue;
            }
            const u64 decal = R<u64>(record + 0x18);
            if (decal == 0) {
                continue;
            }
            if (R<u64>(decal + 0x1e8) == 0 || R<u64>(decal + 0x230) == 0 || R<u64>(decal + 0x248) == 0 || R<u64>(decal + 0x218) == 0 || (R<u32>(decal + 0x2b8) >= 2 && R<u64>(decal + 0x200) == 0)) {
                plan.Decline("decal object without one of its resource lists");
                continue;
            }
            plan.steps.push_back({record, decal, p});
        }
        db.step_count = static_cast<u32>(plan.steps.size()) - db.first_step;
        plan.batches.push_back(db);
    }
    plan.any_mesh = last_mesh != 0;
    plan.scanned = true;
}

void GenerateDecal(Target& t, const DPlan& plan) {
    const u64 rd = plan.render_device, desc = plan.desc;
    const bool wide_state = plan.type == 10 || plan.type == 7; // the loop's two state object families
    std::array<u8, DecalStageBytes> shadow_stage{};
    u8* stage = shadow_stage.data();
    if (t.live) {
        stage = reinterpret_cast<u8*>(rd + DecalStageBase);
    } else {
        std::memcpy(stage, reinterpret_cast<const void*>(rd + DecalStageBase), DecalStageBytes);
    }
    const auto put32 = [&](u32 offset, u32 value) { std::memcpy(stage + (offset - DecalStageBase), &value, 4); };
    const auto putf = [&](u32 offset, float value) { std::memcpy(stage + (offset - DecalStageBase), &value, 4); };
    const auto put_from = [&](u32 offset, u64 source, u32 bytes) { std::memcpy(stage + (offset - DecalStageBase), reinterpret_cast<const void*>(source), bytes); };
    const auto raster_object = [&](u64 object) {
        if (t.Get<u64>(0x46c8) != object) {
            t.Set<u64>(0x46c8, object);
            t.Set<u8>(0x46d8, 1);
        }
    };
    const auto state_words = [&](u64 entry) {
        t.Set<u64>(0x46f0, entry);
        t.Set<u64>(0x46f8, 0x3f8000003f800000ull);
        t.Set<u64>(0x4700, 0x3f8000003f800000ull);
        t.Set<u32>(0x4708, 0xffffffffu);
        t.Set<u8>(0x470c, 1);
    };
    // ----- prologue
    raster_object(R<u64>(rd + 0x3bf8));
    state_words(R<u64>(rd + (wide_state ? 0x3cb0 : 0x3d30)));
    t.Set<u64>(0x46e0, R<u64>(rd + 0x3c40));
    t.Set<u32>(0x46e8, 0);
    t.Set<u8>(0x46ec, 1);
    t.Set<u64>(0x68, 0);
    const u64 object_list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38);
    HandleLookup(rd, R<u64>(object_list + 0x50)); // the loop calls it for its side effect (the render data of the default object)
    const auto view_of = [&](u64 object) -> u64 {
        const u64 data = HandleLookup(rd, object);
        return data != 0 ? R<u64>(data + 0x48) : 0;
    };
    const u64 desc_targets = R<u64>(desc + 0x448);
    const u64 target_set = R<u64>(desc_targets + 0x2b0);
    const u64 w_988 = R<u64>(rd + 0x41b0);
    const u64 w_990 = R<u64>(R<u64>(R<u64>(plan.view + 0x40) + 0x90) + 0x48);
    const u64 w_998 = view_of(R<u64>(target_set + 0x140));
    const u64 w_9a0 = view_of(R<u64>(target_set + 0x148));
    u64 w_9c8 = 0;
    if (const u64 list = R<u64>(plan.view + 0x50); list != 0) {
        w_9c8 = R<u64>(R<u64>(list + 0x18 + u64{R<u32>(list + 0x14)} * 8) + 0x48);
    }
    const std::array<u64, 5> w_9d0 = {R<u64>(rd + 0x4448), R<u64>(rd + 0x4458), R<u64>(rd + 0x8898), R<u64>(rd + 0x88a0), R<u64>(rd + 0x4060)};
    const std::array<u64, 5> w_a10 = {R<u64>(rd + 0x87e8), R<u64>(rd + 0x4460), R<u64>(rd + 0x88a8), R<u64>(rd + 0x4080), R<u64>(R<u64>(R<u64>(rd + 0x3a8) + 0x50) + 0x48)};
    const std::array<u64, 4> w_9a8 = {R<u64>(rd + 0x4188), R<u64>(rd + 0x4190), R<u64>(rd + 0x4198), R<u64>(rd + 0x41a0)};
    // the pass constants
    const u64 constants = R<u64>(desc + 0x298);
    const float depth_a = R<float>(constants + 0x1f8), depth_b = R<float>(constants + 0x1fc);
    const std::array<u64, 6> q = {R<u64>(constants + 0x10), R<u64>(constants + 0x18), R<u64>(constants + 0x20), R<u64>(constants + 0x28), R<u64>(constants + 0x30), R<u64>(constants + 0x38)};
    const u32 one = 0x3f800000u;
    put32(0x39f0, static_cast<u32>(q[0]));
    put32(0x39f4, static_cast<u32>(q[0] >> 32));
    put32(0x39f8, static_cast<u32>(q[1]));
    put32(0x39fc, one);
    put32(0x3a00, static_cast<u32>(q[2]));
    put32(0x3a04, static_cast<u32>(q[2] >> 32));
    put32(0x3a08, static_cast<u32>(q[3]));
    put32(0x3a0c, one);
    put32(0x3a10, static_cast<u32>(q[4]));
    put32(0x3a14, static_cast<u32>(q[4] >> 32));
    put32(0x3a18, static_cast<u32>(q[5]));
    put32(0x3a1c, one);
    putf(0x3a20, depth_a);
    putf(0x3a24, depth_b);
    putf(0x3a28, 1.0f / depth_a);
    putf(0x3a2c, (depth_a - depth_b) / (depth_b * depth_a));
    put_from(0x3b40, desc + 0x210, 0x40);
    putf(0x3a30, static_cast<float>(plan.width));
    putf(0x3a34, static_cast<float>(plan.height));
    putf(0x3a38, 1.0f / static_cast<float>(plan.width));
    putf(0x3a3c, 1.0f / static_cast<float>(plan.height));
    {
        const u64 buffer_a = R<u64>(rd + 0x3980), buffer_b = R<u64>(rd + 0x3b80);
        t.Set<u64>(0x2e0, buffer_a);
        t.Set<u64>(0x2e8, buffer_b);
        t.Set<u64>(0x360, buffer_a);
        t.Set<u64>(0x368, buffer_b);
    }
    const float scale = (plan.flags & 2) != 0 ? 1.0f : 0.0f;
    if (NativePrologueOn() && R<u64>(rd + 0x2d8) == plan.ctx) {
        Timed(CallId::ForwardPrologue, [&] { ForwardPrologueNative(t, rd, plan.view, desc, 0); });
    } else if (t.live) {
        Timed(CallId::ForwardPrologue, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64)>(Global(FunctionForwardPrologue))(rd, plan.view, desc, 0); });
    }
    const u64 desc_targets_after = R<u64>(desc + 0x448);
    const u64 w_9f8 = R<u64>(desc_targets_after + 0x240) != 0 ? view_of(R<u64>(desc_targets_after + 0x240)) : 0;
    const u64 w_a00 = R<u64>(desc_targets_after + 0x248) != 0 ? view_of(R<u64>(desc_targets_after + 0x248)) : 0;
    t.Set<u64>(0x980, plan.a3);
    t.Set<u64>(0x988, w_988);
    t.Set<u64>(0x990, w_990);
    t.Set<u64>(0x998, w_998);
    t.Set<u64>(0x9a0, w_9a0);
    for (u32 i = 0; i < 4; ++i) {
        t.Set<u64>(0x9a8 + i * 8, w_9a8[i]);
    }
    t.Set<u64>(0x9c8, w_9c8);
    for (u32 i = 0; i < 5; ++i) {
        t.Set<u64>(0x9d0 + i * 8, w_9d0[i]);
    }
    t.Set<u64>(0x9f8, w_9f8);
    t.Set<u64>(0xa00, w_a00);
    t.Set<u64>(0xa08, plan.a4);
    for (u32 i = 0; i < 5; ++i) {
        t.Set<u64>(0xa10 + i * 8, w_a10[i]);
    }
    // ----- the batches
    u32 last_state = 1;
    for (const DBatch& db : plan.batches) {
        if (db.set_580) {
            t.batch_owner = db.batch;
            t.Set<u64>(0x580, db.value_580);
            t.batch_owner = 0;
        }
        for (u32 s = db.first_step; s < db.first_step + db.step_count; ++s) {
            const DStep& step = plan.steps[s];
            const u64 d = step.decal, record = step.record;
            const u64 p = step.index;
            const u64 image_object = R<u64>(R<u64>(d + 0x1e8) + p * 8);
            const u64 view_0 = view_of(image_object);
            const u64 view_1 = view_of(R<u64>(R<u64>(d + 0x230) + p * 8));
            const u64 view_2 = view_of(R<u64>(R<u64>(d + 0x248) + p * 8));
            if (view_0 == 0) {
                continue;
            }
            u64 view_d = view_0;
            if (R<u32>(d + 0x2b8) >= 2) {
                const u64 object = R<u64>(R<u64>(d + 0x200) + p * 8);
                if (object == 0) {
                    continue;
                }
                view_d = view_of(object);
            }
            const u64 data_e = HandleLookup(rd, R<u64>(R<u64>(d + 0x218) + p * 8));
            if (data_e == 0) {
                continue;
            }
            const u64 view_e = R<u64>(data_e + 0x48);
            if (view_e == 0) {
                continue;
            }
            // the state object the primitive's flag byte selects
            const u32 flag_byte = R<u8>(record + 0x21);
            if (flag_byte != last_state) {
                u64 object;
                if (wide_state) {
                    object = (flag_byte & 1) != 0 ? R<u64>(rd + 0x3cb0) : (flag_byte & 2) != 0 ? R<u64>(rd + 0x3b90) : (flag_byte & 4) != 0 ? R<u64>(rd + 0x3b98) : R<u64>(rd + 0x3ca8);
                } else {
                    object = (flag_byte & 1) != 0 ? R<u64>(rd + 0x3d30) : (flag_byte & 2) != 0 ? R<u64>(rd + 0x3d38) : (flag_byte & 4) != 0 ? R<u64>(rd + 0x3d40) : R<u64>(rd + 0x3d48);
                }
                state_words(object);
                last_state = flag_byte;
            }
            // the constant block
            put_from(0x39c0, d + 0x110, 16);
            const u64 target_view = R<u64>(image_object + 0x68);
            const u32 size_a = target_view != 0 ? R<u32>(target_view + 0x10) : R<u32>(image_object + 0x84);
            const u32 size_b = target_view != 0 ? R<u32>(target_view + 0x14) : R<u32>(image_object + 0x88);
            putf(0x39d0, static_cast<float>(R<s32>(d + 0x2cc)));
            putf(0x39d4, static_cast<float>(size_a));
            putf(0x39d8, static_cast<float>(size_b));
            put32(0x39dc, 0);
            const u32 vertices = R<u32>(d + (R<s32>(d + 0x2c4) != 0 ? 0x2c0 : 0x2b4));
            const float radius = R<float>(d + 0xa4);
            put32(0x3a40, R<u32>(d + 0x2c8));
            putf(0x3a44, static_cast<float>(vertices));
            putf(0x3a48, radius * radius);
            put32(0x3a4c, R<u32>(d + 0x5bc));
            put_from(0x3a50, d + 0x540, 16);
            put32(0x3a60, R<u32>(d + 0x5c0));
            put32(0x3a64, R<u32>(d + 0x5c4));
            put32(0x3a68, 0);
            put32(0x3a6c, 0);
            put_from(0x3a90, d + 0x2d0, 0x40);
            const u32 first = R<u32>(record + 0x10);
            put32(0x3990, R<u32>(d + 0x29c));
            put32(0x3994, R<u32>(d + 0x284));
            put32(0x3998, R<u32>(d + 0x294));
            put32(0x399c, R<u32>(d + 0x298));
            const u64 holder = R<u64>(d + 0x1d8);
            const s32 holder_value = holder != 0 ? R<s32>(holder + 0x24) : 0;
            const u64 word_288 = R<u64>(d + 0x288);
            put32(0x39a0, static_cast<u32>(word_288));
            put32(0x39a4, static_cast<u32>(word_288 >> 32));
            putf(0x39a8, static_cast<float>(holder_value));
            put32(0x39ac, R<u32>(d + 0x2a0));
            put32(0x39b0, R<u32>(target_set + 0x2d4));
            put32(0x39b4, R<u32>(target_set + 0x2d8));
            put32(0x39b8, R<u32>(target_set + 0x2dc));
            put32(0x39bc, R<u32>(d + 0x290));
            put32(0x39e0, R<u32>(d + 0x278));
            put32(0x39e4, R<u32>(d + 0x27c));
            put32(0x39e8, R<u32>(d + 0x280));
            putf(0x39ec, static_cast<float>(first));
            put32(0x3a70, R<u32>(d + 0x260));
            put32(0x3a74, R<u32>(d + 0x264));
            put32(0x3a78, R<u32>(d + 0x270));
            put32(0x3a7c, R<u32>(d + 0x274));
            putf(0x3a80, scale * R<float>(d + 0x268));
            putf(0x3a84, scale * R<float>(d + 0x26c));
            put32(0x3a88, R<u32>(d + 0x2a8));
            put32(0x3a8c, R<u32>(d + 0x2a4));
            put32(0x3ad0, R<u32>(d + 0x550));
            put32(0x3ad4, R<u32>(d + 0x554));
            put32(0x3ad8, R<u32>(d + 0x558));
            put32(0x3adc, 0);
            put32(0x3ae0, R<u32>(d + 0x55c));
            put32(0x3ae4, R<u32>(d + 0x560));
            put32(0x3ae8, R<u32>(d + 0x564));
            put32(0x3aec, 0);
            put_from(0x3af0, d + 0x570, 0x40);
            put32(0x3b30, first);
            put32(0x3b34, 0);
            put32(0x3b38, 0);
            put32(0x3b3c, 0);
            if (t.live) {
                Timed(CallId::DecalUpdate, [&] {
                    reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64)>(R<u64>(R<u64>(plan.ctx) + 0xd0))(plan.ctx, R<u64>(rd + 0x3b80), 0, 0, rd + DecalStageBase, 0, 0);
                });
            } else {
                t.stages.push_back(shadow_stage);
                t.stage_pending = static_cast<s32>(t.stages.size()) - 1;
            }
            t.Set<u64>(0xa38, view_0);
            t.Set<u64>(0xa40, view_d);
            t.direct_object = data_e;
            t.direct_offset = 0x48;
            t.Set<u64>(0xa48, view_e);
            t.direct_object = 0;
            t.Set<u64>(0xa50, view_1);
            t.Set<u64>(0xa58, view_2);
            t.Set<u64>(0x58, 0);
            t.Set<u64>(0x60, 0x39);
            if (t.Get<u32>(0x4480) != 5) {
                t.Set<u32>(0x4480, 5);
                t.Or8(0x447c, 1);
            }
            std::array<s32, 2> token{0, 0};
            const u64 variant = Timed(CallId::FindVariant, [&] {
                return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u32, u32, u32, void*, u64)>(Global(FunctionFindVariant))(rd + 1000, d, 0, 0, 0, token.data(), 0);
            });
            if (variant == 0) {
                continue;
            }
            SetShader(t, 0x4698, R<u64>(variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(variant + 0x18), 2);
            const s32 instances = R<s32>(record + 0x14);
            if (t.Get<s32>(0x4724) != instances) {
                t.Set<s32>(0x4724, instances);
            }
            if (t.live) {
                reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64)>(Global(FunctionSetIndexBuffer))(plan.ctx, 0);
            } else if (t.Get<u32>(0x278) != 0x200) {
                // FUN_02488430(ctx, 0): the index buffer base and type the loop last set go back to zero (the packets it emits are not modelled)
                if (t.Get<u64>(0x4470) != 0) {
                    t.Set<u64>(0x4470, 0);
                }
                if (t.Get<s32>(0x4478) != 0) {
                    t.Set<s32>(0x4478, 0);
                }
            }
            t.Set<u64>(0x479c, 0);
            if (t.live) {
                Timed(CallId::DrawIssue, [&] { NativeDraw::IssueInlineAutoDraw(plan.ctx, vertices, DecalDrawSite); });
            } else if (NativeDraw::InlineAutoDrawTagged(vertices)) {
                t.marks.push_back({t.arena.size(), 2, vertices, 0, static_cast<u64>(instances), t.Get<u64>(0x4698), t.Get<u64>(0x4690)});
                if (t.stage_pending >= 0) {
                    t.marks.back().stage = t.stage_pending;
                }
            }
            t.stage_pending = -1;
            if (t.live) {
                W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                W<s32>(rd + 0x14, R<s32>(rd + 0x14) + instances * 2);
            } else {
                ++t.draws;
                t.triangles += instances * 2;
            }
        }
    }
    // ----- epilogue
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x580, zeros.data(), 0x400);
        t.Put(0x980, zeros.data(), 0x400);
        if (plan.any_mesh) {
            t.Set<u64>(0x580, 0);
            t.Put(0x980, zeros.data(), 0xe0);
            raster_object(R<u64>(rd + 0x3bf0));
        }
    }
}

struct DConfig {
    bool on{false};
    u32 verify_every{32};
};
const DConfig& DCfg() {
    static const DConfig config = [] {
        DConfig c;
        const char* e = std::getenv("TLG_DECAL_GEN");
        c.on = e != nullptr && *e != 0 && *e != '0';
        if (const char* n = std::getenv("TLG_DECAL_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

struct DStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
DStats& DS() {
    static DStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the skinning loop (B3)
// FUN_01c93730(skin system S, scene F, async flag): for every skinned object of the scene (the list at F + 0x29e00) one skinning dispatch (CS_Skinning, 24 variants), then, for the objects
// with the normal flag, three more (CS_FixVertexNormal: clear, calculate, normalize). It talks to the device context only through virtual calls that are table writes
// (SetShader: ctx + 0x4670; SetConstantBuffers: ctx + 0x500; SetShaderResources: ctx + 0x1980; SetUnorderedAccessViews: ctx + 0x2098) and UpdateSubresource of its own constant block (S + 0x50),
// plus the dispatches. The generator makes the same table writes, the same constant blocks (compared byte by byte at every dispatch in the verify run), the same dispatches
// (through the dispatch hook, from the loop's own call sites). Decompiled and annotated in documents/tlg/native plan/stages/b3-skinning.md.
constexpr u64 SkinBegin = 0x01c93730, SkinEnd = 0x01c95550; // the loop and the next function
constexpr u64 GlobalDevice = 0x0689eed8;                    // DAT_0689eed8: the render device (the value)
constexpr u64 SkinSiteDispatch = 0x01c94f44, SkinSiteClear = 0x01c951ea, SkinSiteCalculate = 0x01c952f3, SkinSiteNormalize = 0x01c95426;
constexpr u32 SkinBlockOffset = 0x50, SkinBlockBytes = 0x120; // the constant block UpdateSubresource sends from S + 0x50 (the buffer's byte width is checked at the scan)

struct SkinNode {
    u64 node{};
    bool active{false}; // objects 1, 2 and 8 exist: the loop makes bone matrices and a dispatch for it
    u64 rd1{}, rd2{}, rd3{}, rd8{}, rd5{}, rd35{};
    u64 inner1{};
    u64 v20{}, v21{}, v59{}, v60{}, v63{}, v64{}, v65{}, v66{}, v69{}, v180{}, v50{};
    bool has_object4{false}, b70{false}, b74{false}, b75{false};
    u64 v_a0{0}; // SRV 8 (only with the object at +0x108)
    s32 count46{0};
    u32 variant{0};
    bool normals{false};
    u64 normals_rd{0}; // render data of the object at node + 0x20 (pass 3)
};

struct SkinPlan {
    bool scanned{false};
    bool no_work{false};
    std::vector<const char*> reasons;
    u64 device{0}, ctx{0}, system{0}, scene{0}, def{0}, context_global{0};
    u64 camera{0}, camera_object{0}, view{0};
    bool async{false};
    bool run_normals{false};
    std::vector<SkinNode> nodes;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct SkinScanArgs {
    SkinPlan* plan;
};

void ScanSkin(SkinScanArgs& a) {
    SkinPlan& p = *a.plan;
    p.device = R<u64>(Global(GlobalDevice));
    p.context_global = R<u64>(Global(GlobalContext));
    p.def = R<u64>(R<u64>(p.context_global + 0x38) + 0x20);
    // the graphics device context, or the async compute context (its tables differ: shader + 0x1d10, constant buffers + 0x21f8, shader resources + 0x1d18, unordered access + 0x2118)
    p.ctx = p.async ? R<u64>(R<u64>(p.device + 0x2c8) + 0x30258) : R<u64>(p.device + 0x2d8);
    const u64 first = R<u64>(p.scene + 0x29e00);
    if (first == 0) {
        p.no_work = true;
        p.scanned = true;
        return;
    }
    if (p.ctx == 0) {
        p.Decline("no device context");
        return;
    }
    const u64 S = p.system;
    for (const u32 offset : {0x1d0u, 0x230u, 0x340u, 0x360u, 0x380u}) {
        if (R<u64>(S + offset) == 0) {
            p.Decline("the system's constant buffers are not created");
            return;
        }
    }
    if (R<u32>(R<u64>(S + 0x1d0) + 0x180) < SkinBlockBytes) {
        p.Decline("the constant buffer is smaller than the block");
        return;
    }
    p.camera_object = R<u64>(R<u64>(p.context_global + 0x108) + 0x2c8);
    p.camera = R<u64>(R<u64>(R<u64>(p.context_global + 0x108) + 0x2e0) + 0x2c0);
    if (p.camera == 0 || p.camera_object == 0) {
        p.Decline("no camera");
        return;
    }
    const u64 slot_count = R<u32>(p.scene + 0x29dd0);
    p.view = 0;
    if (slot_count != 0) {
        u64 slot = p.scene + 0x5d0;
        for (u64 k = 1;; ++k) {
            const bool taken = R<s32>(R<u64>(slot + 0x290) + 0x40) != 0;
            if (!taken) {
                p.view = slot;
                break;
            }
            if (slot_count <= k) {
                break;
            }
            slot += 0xa60;
        }
    }
    const auto render_data = [&](u64 object) { return object != 0 ? R<u64>(object + 0x38) : 0ull; };
    const auto zero_or_default = [&](u64 object) { return object == p.def ? 0ull : object != 0 ? R<u64>(object + 0x38) : 0ull; };
    u32 guard = 0;
    for (u64 node = first; node != 0 && guard < 100000; node = R<u64>(node), ++guard) {
        SkinNode n;
        n.node = node;
        const u64 o1 = R<u64>(node + 8), o2 = R<u64>(node + 0x10), o3 = R<u64>(node + 0x18), o4 = R<u64>(node + 0x20), o5 = R<u64>(node + 0x28), o8 = R<u64>(node + 0x40),
                  o35 = R<u64>(node + 0x118);
        if (o1 != 0) {
            n.rd1 = render_data(o1);
            if (n.rd1 == 0) {
                p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
            }
        }
        if (o1 != 0 && o2 != 0) {
            n.rd2 = render_data(o2);
            if (n.rd2 == 0) {
                p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
            }
            if (o3 != 0) {
                n.rd3 = render_data(o3);
                if (n.rd3 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (o8 != 0) {
                n.rd8 = render_data(o8);
                if (n.rd8 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
        }
        n.active = o1 != 0 && o2 != 0 && o8 != 0;
        if (n.active && !p.reasons.empty()) {
            p.nodes.push_back(n);
            continue;
        }
        if (n.active) {
            if (o5 != 0) {
                n.rd5 = render_data(o5);
                if (n.rd5 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (o35 != 0) {
                n.rd35 = render_data(o35);
                if (n.rd35 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            u64 object4_data = 0;
            if (o4 != 0) {
                object4_data = render_data(o4);
                if (object4_data == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (!p.reasons.empty()) {
                p.nodes.push_back(n);
                continue;
            }
            n.inner1 = R<u64>(n.rd1 + 0x20);
            n.has_object4 = o4 != 0;
            u64 inner4 = 0;
            if (n.has_object4) {
                inner4 = R<u64>(object4_data + 0x20);
                const u32 count = R<u32>(inner4 + 0x68);
                if (count >= 9) {
                    const u64 array = R<u64>(inner4 + 0x60);
                    n.v69 = zero_or_default(R<u64>(array + 0x40));
                    if (count >= 0x24) {
                        n.v180 = zero_or_default(R<u64>(array + 0x118));
                    }
                    if (count >= 0xb) {
                        const u64 x = R<u64>(array + 0x50);
                        n.v50 = (x == p.def || x == 0) ? 0 : R<u64>(x + 0x38);
                    }
                }
            }
            if (n.inner1 == 0 || R<u64>(n.rd2 + 0x20) == 0) {
                p.Decline("render data without its inner object");
                p.nodes.push_back(n);
                continue;
            }
            if (R<u32>(n.inner1 + 0x68) > 0xb) {
                const u64 x = R<u64>(R<u64>(n.inner1 + 0x60) + 0x58);
                n.v63 = zero_or_default(x);
            }
            n.v20 = R<u64>(R<u64>(n.inner1 + 0x48) + 0x38);
            n.v21 = R<u64>(R<u64>(R<u64>(n.rd2 + 0x20) + 0x48) + 0x38);
            n.v59 = n.rd3 != 0 ? R<u64>(R<u64>(R<u64>(n.rd3 + 0x20) + 0x48) + 0x38) : 0;
            if (n.has_object4) {
                n.v66 = R<u64>(R<u64>(inner4 + 0x48) + 0x38);
                const u32 count = R<u32>(inner4 + 0x68);
                if (count >= 0x25) {
                    n.v60 = zero_or_default(R<u64>(R<u64>(inner4 + 0x60) + 0x120));
                    if (count >= 0x26) {
                        const u64 x = R<u64>(R<u64>(inner4 + 0x60) + 0x128);
                        n.v65 = (x == p.def || x == 0) ? 0 : R<u64>(x + 0x38);
                    }
                }
            }
            n.v64 = R<u64>(n.inner1 + 0x58) != 0 ? R<u64>(R<u64>(n.inner1 + 0x58) + 0x38) : 0;
            n.count46 = R<s32>(n.rd8 + 0x40);
            n.b74 = !p.async && R<u64>(node + 0xa0) != 0;
            n.b75 = n.v69 != 0 && R<u64>(n.v69 + 0x30) != 0;
            n.b70 = R<u64>(node + 0x108) != 0 && !n.b74;
            if (n.b70) {
                if (!n.has_object4) {
                    p.Decline("constants object without the object at +0x20");
                } else {
                    const u64 candidate = R<u64>(R<u64>(inner4 + 0x60) + 0x108);
                    if (candidate == p.def || candidate == 0 || R<u64>(candidate + 0x38) == 0) {
                        p.Decline("constants object without render data");
                    } else {
                        n.v_a0 = R<u64>(R<u64>(candidate + 0x38) + 0x28);
                    }
                }
            }
            const bool flag54 = (R<u32>(node + 0x54) >> 1 & 1) != 0;
            const bool o5set = n.rd5 != 0, o35set = n.rd35 != 0, o4set = n.has_object4;
            (void)o4set;
            u32 v;
            const bool all_three = n.rd5 != 0 && n.v64 != 0 && n.rd3 != 0;
            if (all_three) {
                v = (n.count46 == 0x40 ? 1u : 0u) | 0x5a;
            } else if (R<u64>(node + 0x108) == 0 || n.b74) {
                if (n.b74) {
                    if (n.b75) {
                        if (n.v180 == 0) {
                            v = (o35set ? 2u : 0u) + 0x4d;
                            if (n.count46 == 0x40) {
                                v = (o35set ? 2u : 0u) + 0x4e;
                            }
                        } else {
                            v = (o35set ? 2u : 0u) + 0x51;
                            if (n.count46 == 0x40) {
                                v = (o35set ? 2u : 0u) + 0x52;
                            }
                        }
                    } else {
                        v = 0x4c;
                        if (n.count46 != 0x40) {
                            v = (flag54 ? 1u : 0u) | 0x4a;
                        }
                    }
                } else if (n.count46 == 0x40) {
                    v = 0x49;
                    if (!flag54) {
                        u32 high = (n.v60 != 0 ? 1u : 0u) + 0x58;
                        v = (n.v60 != 0 ? 1u : 0u) | 0x56;
                        if (n.v180 != 0) {
                            v = high;
                        }
                    }
                } else {
                    v = 0x48;
                    if (!flag54) {
                        v = n.v60 != 0 ? 0x55u : 0x47u;
                    }
                }
                (void)o5set;
            } else {
                const u32 high = (n.v60 != 0 ? 1u : 0u) + 0x5e;
                v = (n.v60 != 0 ? 1u : 0u) | 0x5c;
                if (n.v180 != 0) {
                    v = high;
                }
            }
            n.variant = v;
            if (R<u64>(p.device + 0x1008 + u64{v} * 0x10) == 0) {
                p.Decline("shader variant not loaded");
            }
        }
        p.nodes.push_back(n);
    }
    if (guard >= 100000) {
        p.Decline("node list unreadable");
    }
    // pass 3: the normals
    p.run_normals = !p.async && R<u32>(p.camera + 0x24) != 0;
    if (p.run_normals) {
        for (SkinNode& n : p.nodes) {
            const u64 node = n.node;
            if (R<u64>(node + 0xa0) != 0 && (R<u8>(node + 0x54) & 8) != 0) {
                const u64 object = R<u64>(node + 0x20);
                if (object != 0) {
                    n.normals = true;
                    n.normals_rd = R<u64>(object + 0x38);
                    if (n.normals_rd == 0) {
                        p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                    } else if (R<u64>(n.normals_rd + 0x20) == 0) {
                        p.Decline("render data without its inner object");
                    }
                }
            }
        }
        for (const u32 offset : {0x1448u, 0x1458u, 0x1468u}) {
            if (R<u64>(p.device + offset) == 0) {
                p.Decline("normal shader not loaded");
            }
        }
    }
    p.scanned = true;
}

/// The 4x4 inverse the loop makes of a node's matrix (node + 0xb0, row major) with the engine's own order of operations (read off the disassembly: cofactors as `(a*b - c*d) + e*f` per lane,
/// scaled by 1 / determinant with alternating signs); a determinant under 1e-5 gives the engine's constant block. `out` is 16 floats (S + 0xe0 .. 0x11f).
void SkinInverse(u64 node, float* out) {
    const float m00 = R<float>(node + 0xb0), m01 = R<float>(node + 0xb4), m02 = R<float>(node + 0xb8), m03 = R<float>(node + 0xbc);
    const float m10 = R<float>(node + 0xc0), m11 = R<float>(node + 0xc4), m12 = R<float>(node + 0xc8), m13 = R<float>(node + 0xcc);
    const float m20 = R<float>(node + 0xd0), m21 = R<float>(node + 0xd4), m22 = R<float>(node + 0xd8), m23 = R<float>(node + 0xdc);
    const float m30 = R<float>(node + 0xe0), m31 = R<float>(node + 0xe4), m32 = R<float>(node + 0xe8), m33 = R<float>(node + 0xec);
    const float a1 = m13 * m02 - m12 * m03;
    const float a2 = m23 * m02 - m22 * m03;
    const float x = m33 * m02 - m32 * m03;
    const float a3 = m23 * m12 - m22 * m13;
    const float b3 = m33 * m12 - m32 * m13;
    const float a5 = m33 * m22 - m32 * m23;
    const float c0 = (m11 * a5 - m21 * b3) + m31 * a3;
    const float c1 = (m31 * a2 - m21 * x) + m01 * a5;
    const float c2 = (m31 * a1 - m11 * x) + m01 * b3;
    const float c3 = (m21 * a1 - m11 * a2) + m01 * a3;
    const float sum_a = c2 * m20 + c0 * m00;
    const float sum_b = c1 * m10 + c3 * m30;
    const float det = sum_a - sum_b;
    const float abs_det = 0.0f < det ? det : sum_b - sum_a;
    const float threshold = R<float>(Global(0x024f15dc));
    if (!(threshold < abs_det)) {
        std::memcpy(out, reinterpret_cast<const void*>(Global(0x024ff720)), 32);      // S + 0xe0
        std::memcpy(out + 8, reinterpret_cast<const void*>(Global(0x024ff700)), 32); // S + 0x100
        return;
    }
    const float e1 = m11 * m00 - m01 * m10;
    const float e2 = m21 * m00 - m20 * m01;
    const float e3 = m31 * m00 - m30 * m01;
    const float e4 = m10 * m21 - m20 * m11;
    const float e5 = m10 * m31 - m30 * m11;
    const float e6 = m20 * m31 - m30 * m21;
    const float c4 = (m10 * a5 - m20 * b3) + m30 * a3;
    const float c5 = (m00 * a5 - m20 * x) + m30 * a2;
    const float c6 = (m00 * b3 - m10 * x) + m30 * a1;
    const float c7 = (m00 * a3 - m10 * a2) + m20 * a1;
    const float d0 = (e6 * m13 + e4 * m33) - e5 * m23;
    const float d1 = (e6 * m03 + e2 * m33) - e3 * m23;
    const float d2 = (e5 * m03 + e1 * m33) - e3 * m13;
    const float d3 = (e4 * m03 + e1 * m23) - e2 * m13;
    const float d4 = (e6 * m12 + e4 * m32) - e5 * m22;
    const float d5 = (e6 * m02 + e2 * m32) - e3 * m22;
    const float d6 = (e5 * m02 + e1 * m32) - e3 * m12;
    const float d7 = (e4 * m02 + e1 * m22) - e2 * m12;
    const float r = R<float>(Global(0x024f15e0)) / det;
    const float n = -r;
    out[0] = r * c0;
    out[1] = n * c1;
    out[2] = r * c2;
    out[3] = n * c3;
    out[4] = n * c4;
    out[5] = r * c5;
    out[6] = n * c6;
    out[7] = r * c7;
    out[8] = r * d0;
    out[9] = d1 * n;
    out[10] = r * d2;
    out[11] = d3 * n;
    out[12] = d4 * n;
    out[13] = r * d5;
    out[14] = d6 * n;
    out[15] = r * d7;
}

void GenerateSkin(Target& t, const SkinPlan& p) {
    const u64 device = p.device, scene = p.scene, S_live = p.system;
    const u64 ctx = p.ctx;
    // the system object: live, the loop's own; verify, a private copy of the first 0x400 bytes (its staging area is written by the loop)
    std::array<u8, 0x400> shadow_system{};
    u8* sb = reinterpret_cast<u8*>(S_live);
    if (!t.live) {
        std::memcpy(shadow_system.data(), reinterpret_cast<const void*>(S_live), shadow_system.size());
        sb = shadow_system.data();
    }
    const auto sget = [&]<class T>(u32 offset) { T v; std::memcpy(&v, sb + offset, sizeof(T)); return v; };
    const auto sput = [&]<class T>(u32 offset, T value) { std::memcpy(sb + offset, &value, sizeof(T)); };
    const auto sput_from = [&](u32 offset, u64 source, u32 bytes) { std::memcpy(sb + offset, reinterpret_cast<const void*>(source), bytes); };
    // the ring index and the bone matrices
    const s32 old_index = sget.template operator()<s32>(0x40);
    const s32 index = (old_index + 1) % 4;
    if (t.live) {
        W<s32>(S_live + 0x40, index);
    }
    const u64 ring_memory = R<u64>(R<u64>(S_live + u64{static_cast<u32>(index)} * 8) + 0x30);
    if (t.live) {
        u32 offset = 0;
        for (const SkinNode& n : p.nodes) {
            if (n.active) {
                std::memcpy(reinterpret_cast<void*>(ring_memory + u64{offset} * 0x40), reinterpret_cast<const void*>(R<u64>(n.node + 0x48)), u64{R<u32>(n.node + 0x50)} << 6);
                offset += R<u32>(n.node + 0x50);
            }
        }
    }
    const std::array<u64, 3> cb_array{R<u64>(S_live + 0x1d0), R<u64>(S_live + 0x230), R<u64>(S_live + 0x340)};
    const u32 set_shader_offset = p.async ? 0x1d10 : 0x4670, cb_table = p.async ? 0x21f8 : 0x500, srv_table = p.async ? 0x1d18 : 0x1980, uav_table = p.async ? 0x2118 : 0x2098;
    const auto update_subresource = [&](u64 buffer, u32 source_offset, u32 bytes) {
        if (t.live) {
            Timed(CallId::SkinUpdate, [&] {
                reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64)>(R<u64>(R<u64>(ctx) + 0xd0))(ctx, buffer, 0, 0, S_live + source_offset, 0, 0);
            });
        } else {
            std::vector<u8> block(bytes);
            std::memcpy(block.data(), sb + source_offset, bytes);
            t.blocks.push_back({S_live + source_offset, std::move(block)});
        }
    };
    const auto dispatch = [&](u32 x, bool fenced, u64 site) {
        if (t.live) {
            Timed(CallId::DrawIssue, [&] { NativeDraw::IssueDispatch(ctx, x, 1, 1, fenced, site, p.async); });
        } else {
            t.marks.push_back({t.arena.size(), p.async ? 4u : 3u, x, 1, 1, t.Get<u64>(set_shader_offset), 0});
            t.marks.back().blocks = std::move(t.blocks);
            t.blocks.clear();
            t.marks.back().flag_source = p.context_global + 0x19c;
            t.marks.back().flag_block = S_live + 0x50;
        }
        ++t.draws;
    };
    const auto flush_wait = [&] {
        if (t.live) {
            reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(R<u64>(R<u64>(ctx) + 0x78))(ctx);
        }
    };
    const auto set_cb = [&](u32 start, u32 count, const u64* values) { t.Put(cb_table + start * 8, values, count * 8); };
    const auto set_srv = [&](u32 start, u32 count, const u64* values) { t.Put(srv_table + start * 8, values, count * 8); };
    const auto set_uav = [&](u32 start, u32 count, const u64* values) {
        t.Put(uav_table + start * 8, values, count * 8);
        if (p.async && t.live) {
            for (u32 i = 0; i < count; ++i) {
                if (values[i] != 0) {
                    W<u32>(values[i] + 0x88, 0xffffffffu); // the async context resets the counter index of every view it is given (no counter array)
                }
            }
        }
    };
    const auto set_shader = [&](u64 shader) { t.Set<u64>(set_shader_offset, shader); };
    // pass 2: one dispatch per object
    u32 matrix_offset = 0;
    const float one_e5 = R<float>(Global(0x024f15dc));
    for (const SkinNode& n : p.nodes) {
        if (!n.active) {
            continue;
        }
        const u64 node = n.node;
        set_shader(R<u64>(device + 0x1008 + u64{n.variant} * 0x10));
        if (n.b70) {
            u32 dwords = R<u32>(node + 0x110);
            if (dwords > 0x3f) {
                dwords = 0x40;
            }
            sput_from(0x240, R<u64>(node + 0x108), dwords << 2);
            update_subresource(cb_array[2], 0x240, std::min<u32>(R<u32>(cb_array[2] + 0x180), 0x100));
        }
        if (p.view == 0) {
            const u64 cam = p.camera_object;
            const float f2 = R<float>(cam + 0x90), f3 = R<float>(cam + 0x98), f77 = -R<float>(cam + 0x94);
            sput.template operator()<float>(0x60, (f77 * R<float>(node + 0xb4) - f2 * R<float>(node + 0xb0)) - f3 * R<float>(node + 0xb8));
            sput.template operator()<float>(0x64, (f77 * R<float>(node + 0xc4) - f2 * R<float>(node + 0xc0)) - f3 * R<float>(node + 0xc8));
            sput.template operator()<float>(0x68, (f77 * R<float>(node + 0xd4) - f2 * R<float>(node + 0xd0)) - f3 * R<float>(node + 0xd8));
            sput.template operator()<u32>(0x6c, 0x3f800000u);
            sput_from(0x170, cam + 0xf0, 16);
        } else {
            const u64 view = p.view;
            sput_from(0x190, view + 0x80, 64);
            sput.template operator()<u32>(0x60, R<u32>(view + 0x2a0) ^ 0x80000000u);
            sput.template operator()<u32>(0x64, R<u32>(view + 0x2a4) ^ 0x80000000u);
            sput.template operator()<u32>(0x68, R<u32>(view + 0x2a8) ^ 0x80000000u);
            sput.template operator()<u32>(0x6c, R<u32>(view + 0x2ac));
            sput_from(0x170, view + 0x140, 16);
        }
        sput_from(0x180, p.camera + 0x30, 16);
        sput.template operator()<u32>(0x50, R<u32>(node + 0x30));
        sput.template operator()<u32>(0x54, R<u32>(node + 0x50));
        sput.template operator()<u32>(0x58, matrix_offset);
        sput.template operator()<u32>(0x5c, !(R<float>(p.context_global + 0x19c) < one_e5) ? 1u : 0u);
        const u64 camera_flags = R<u64>(p.camera + 0x40);
        sput.template operator()<u32>(0x70, static_cast<u32>(camera_flags));
        sput.template operator()<u32>(0x74, static_cast<u32>(camera_flags >> 32));
        sput.template operator()<u64>(0x78, 0);
        sput_from(0x90, node + 0xf0, 16);
        sput_from(0xa0, node + 0xb0, 64);
        {
            std::array<float, 16> inverse{};
            SkinInverse(node, inverse.data());
            std::memcpy(sb + 0xe0, inverse.data(), 64);
        }
        if (const u64 holder = R<u64>(node + 0xa0); holder != 0) {
            sput.template operator()<float>(0x80, 1.0f / R<float>(holder + 0x30));
            sput.template operator()<u32>(0x84, R<u32>(holder + 0x34));
            sput.template operator()<u32>(0x88, 0);
            sput.template operator()<u32>(0x8c, R<u32>(holder + 0x38));
            sput_from(0x120, node + 0x70, 16);
            const u64 w80 = R<u64>(node + 0x80), w88 = R<u64>(node + 0x88);
            sput.template operator()<u32>(0x130, static_cast<u32>(w80));
            sput.template operator()<u32>(0x134, static_cast<u32>(w80 >> 32));
            sput.template operator()<u32>(0x138, static_cast<u32>(w88));
            sput.template operator()<u32>(0x13c, R<u32>(holder + 0x48));
            sput_from(0x140, node + 0x90, 16);
            const u64 h40 = R<u64>(holder + 0x40), h58 = R<u64>(holder + 0x58);
            sput.template operator()<u32>(0x150, static_cast<u32>(h40));
            sput.template operator()<u32>(0x154, static_cast<u32>(h58));
            sput.template operator()<u32>(0x158, static_cast<u32>(h40 >> 32));
            sput.template operator()<u32>(0x15c, static_cast<u32>(h58 >> 32));
        }
        sput.template operator()<u32>(0x160, R<u32>(node + 0x100));
        sput.template operator()<u64>(0x164, 0);
        sput.template operator()<u32>(0x16c, 0);
        update_subresource(cb_array[0], 0x50, std::min<u32>(R<u32>(cb_array[0] + 0x180), SkinBlockBytes));
        matrix_offset += R<u32>(node + 0x50);
        set_cb(0, 3, cb_array.data());
        // the views and the targets
        std::array<u64, 11> srv{};
        srv[0] = R<u64>(n.v20 + 0x28);
        srv[1] = R<u64>(n.rd8 + 0x28);
        srv[2] = R<u64>(S_live + 0x20 + u64{static_cast<u32>(index)} * 8);
        srv[3] = n.v64 != 0 ? R<u64>(n.v64 + 0x28) : 0;
        srv[4] = n.v59 != 0 ? R<u64>(n.v59 + 0x28) : 0;
        if (const u64 holder = R<u64>(node + 0xa0); holder != 0) {
            const u64 data = HandleLookup(device, R<u64>(holder + 0x50));
            srv[5] = R<u64>(data + 0x48);
        }
        srv[6] = n.v63 != 0 ? R<u64>(n.v63 + 0x28) : 0;
        srv[7] = n.rd35 != 0 ? R<u64>(n.rd35 + 0x28) : 0;
        srv[8] = n.b70 ? n.v_a0 : 0;
        srv[9] = n.v180 != 0 ? R<u64>(n.v180 + 0x28) : 0;
        srv[10] = n.v60 != 0 ? R<u64>(n.v60 + 0x28) : 0;
        set_srv(4, 11, srv.data());
        std::array<u64, 8> uav{};
        uav[0] = R<u64>(n.v21 + 0x30);
        uav[2] = n.has_object4 ? R<u64>(n.v66 + 0x30) : 0;
        uav[3] = n.rd5 != 0 ? R<u64>(n.rd5 + 0x30) : 0;
        uav[5] = n.v69 != 0 ? R<u64>(n.v69 + 0x30) : 0;
        uav[6] = n.b70 ? R<u64>(n.v50 + 0x30) : 0;
        uav[7] = n.v65 != 0 ? R<u64>(n.v65 + 0x30) : 0;
        set_uav(0, 8, uav.data());
        dispatch((R<u32>(node + 0x34) + 0x3f) >> 6, false, SkinSiteDispatch);
    }
    flush_wait();
    {
        const std::array<u64, 11> zeros{};
        set_srv(0, 4, zeros.data());
        set_srv(4, 11, zeros.data());
        set_uav(0, 8, zeros.data());
        set_cb(0, 3, zeros.data());
        set_shader(0);
    }
    // pass 3
    if (p.run_normals) {
        for (const SkinNode& n : p.nodes) {
            if (!n.normals) {
                continue;
            }
            const u64 node = n.node;
            const u64 object = R<u64>(node + 0x20);
            const u64 rd = n.normals_rd;
            const u64 w38 = R<u64>(node + 0x38);
            const u32 triangles = static_cast<u32>(w38 >> 32), vertices = R<u32>(node + 0x34);
            sput.template operator()<u32>(0x350, static_cast<u32>(w38));
            sput.template operator()<u32>(0x354, 0);
            const u64 cb360 = R<u64>(S_live + 0x360);
            update_subresource(cb360, 0x350, std::min<u32>(R<u32>(cb360 + 0x180), 0x10));
            set_cb(0, 1, &cb360);
            const u64 uav380 = R<u64>(S_live + 0x380);
            set_uav(0, 1, &uav380);
            set_shader(R<u64>(device + 0x1448));
            const u32 groups = (vertices + 0x3f) >> 6;
            dispatch(groups, true, SkinSiteClear);
            const u64 inner = R<u64>(rd + 0x20);
            const std::array<u64, 2> srv_a{R<u64>(R<u64>(R<u64>(inner + 0x50) + 0x38) + 0x28), R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28)};
            set_srv(0, 2, srv_a.data());
            set_shader(R<u64>(device + 0x1458));
            dispatch(((triangles / 3) + 0x3f) >> 6, true, SkinSiteCalculate);
            const u64 zero = 0;
            set_uav(0, 1, &zero);
            const std::array<u64, 2> srv_b{R<u64>(S_live + 0x378), 0};
            set_srv(0, 2, srv_b.data());
            const u64 uav_normal = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x30);
            set_uav(0, 1, &uav_normal);
            set_shader(R<u64>(device + 0x1468));
            dispatch(groups, true, SkinSiteNormalize);
            set_uav(0, 1, &zero);
            const std::array<u64, 2> srv_c{};
            set_srv(0, 2, srv_c.data());
            (void)object;
        }
        flush_wait();
        const std::array<u64, 11> zeros{};
        set_srv(0, 11, zeros.data());
        set_uav(0, 8, zeros.data());
        const u64 zero = 0;
        set_cb(0, 1, &zero);
        set_shader(0);
    }
}

struct SConfig {
    bool on{false};
    u32 verify_every{32};
};
const SConfig& SCfg() {
    static const SConfig config = [] {
        SConfig c;
        const char* e = std::getenv("TLG_SKIN_GEN");
        c.on = e != nullptr && *e != 0 && *e != '0';
        if (const char* n = std::getenv("TLG_SKIN_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

struct SStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
SStats& SS() {
    static SStats stats;
    return stats;
}


} // namespace

bool Enabled() {
    return Cfg().on;
}

bool VerifyActive() {
    return V().active.load(std::memory_order_relaxed);
}

void NoteLoopDraw(u64 caller, u32 kind, u64 ctx, u64 count, u64 first, u64 a3) {
    Verify& v = V();
    if (!v.active.load(std::memory_order_relaxed) || caller < v.lo || caller >= v.hi || std::this_thread::get_id() != v.thread) {
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
    if (mark.kind != kind || mark.count != count || mark.first != first || ((kind == 0 || kind == 2) && mark.a3 != a3)) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "draw arguments", fmt::format("draw {}: the loop kind {} {:#x} / {:#x} / {:#x}, the generator kind {} {:#x} / {:#x} / {:#x}", k, kind, count, first, a3, mark.kind, mark.count, mark.first, mark.a3));
    }
    if (ctx != v.ctx) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "context", fmt::format("draw {}: another device context", k));
        return;
    }
    if (kind == 3 || kind == 4) {
        // B3: a dispatch: the compute shader of the context and the constant blocks the loop updated its buffers from
        const u32 shader_offset = kind == 4 ? 0x1d10 : 0x4670;
        if (R<u64>(ctx + shader_offset) != mark.vertex_shader) {
            ++v.mismatches;
            NoteMismatch(s, v.view, "compute shader", fmt::format("draw {}: the loop {:#x}, the generator {:#x}", k, R<u64>(ctx + shader_offset), mark.vertex_shader));
        }
        for (const auto& [address, generated] : mark.blocks) {
            std::vector<u8> bytes = generated;
            if (mark.flag_source != 0 && address == mark.flag_block && bytes.size() >= 0x10) {
                // the flag in dword 3 comes from a float the main thread updates: if the live value is what the float gives now, the loop and the generator read it at different times
                u32 live_flag = 0, generated_flag = 0;
                std::memcpy(&live_flag, reinterpret_cast<const void*>(address + 0xc), 4);
                std::memcpy(&generated_flag, bytes.data() + 0xc, 4);
                const u32 now = !(R<float>(mark.flag_source) < R<float>(Global(0x024f15dc))) ? 1u : 0u;
                if (live_flag != generated_flag && live_flag == now) {
                    std::memcpy(bytes.data() + 0xc, &live_flag, 4);
                    ++v.raced;
                    ++s.raced_words[fmt::format("view {} skinning flag word {:#x}", v.view, 0xc)];
                }
            }
            if (std::memcmp(reinterpret_cast<const void*>(address), bytes.data(), bytes.size()) != 0) {
                u32 at = 0;
                while (at + 4 <= bytes.size() && std::memcmp(reinterpret_cast<const void*>(address + at), bytes.data() + at, 4) == 0) {
                    at += 4;
                }
                u32 a = 0, b = 0;
                std::memcpy(&a, reinterpret_cast<const void*>(address + at), 4);
                std::memcpy(&b, bytes.data() + at, 4);
                ++v.mismatches;
                NoteMismatch(s, v.view, "skinning constants", fmt::format("dispatch {}: dword {:#x} of the block at {:#x}: the loop {:#x}, the generator {:#x}", k, at, address, a, b));
            }
        }
    }
    const u64 vs = R<u64>(ctx + 0x4698), ps = R<u64>(ctx + 0x4690);
    if (kind != 3 && kind != 4 && (vs != mark.vertex_shader || ps != mark.pixel_shader)) {
        ++v.mismatches;
        NoteMismatch(s, v.view, "shader pair", fmt::format("draw {}: the loop {:#x} / {:#x}, the generator {:#x} / {:#x}", k, vs, ps, mark.vertex_shader, mark.pixel_shader));
    }
    if (k == 0 && v.forward_block_set && (v.view == 200 || v.view == 201)) {
        // A3: the 0x340 byte block the engine's forward prologue put in the frame ring (ctx + 0x13048) against the one the native prologue built
        v.forward_block_set = false;
        const u64 block = R<u64>(ctx + 0x13048);
        Leaf().block_compared.fetch_add(1, std::memory_order_relaxed);
        if (block == 0 || std::memcmp(reinterpret_cast<const void*>(block), v.forward_block.data(), v.forward_block.size()) != 0) {
            Leaf().block_differing.fetch_add(1, std::memory_order_relaxed);
            ++v.mismatches;
            NoteMismatch(s, v.view, "forward prologue block", fmt::format("the engine's block at {:#x} differs from the native one", block));
        }
    }
    if (mark.stage >= 0) {
        // A7: the constant block of the decal loop (renderDev + 0x3990) at this draw against the block the generator built
        const auto& built = v.shadow.stages[static_cast<size_t>(mark.stage)];
        const u8* now = reinterpret_cast<const u8*>(v.rd + DecalStageBase);
        if (std::memcmp(now, built.data(), built.size()) != 0) {
            u32 at = 0;
            while (at + 4 <= built.size() && std::memcmp(now + at, built.data() + at, 4) == 0) {
                at += 4;
            }
            u32 a = 0, b = 0;
            std::memcpy(&a, now + at, 4);
            std::memcpy(&b, built.data() + at, 4);
            ++v.mismatches;
            NoteMismatch(s, v.view, "decal constants", fmt::format("draw {}: dword {:#x} of the block: the loop {:#x}, the generator {:#x}", k, DecalStageBase + at, a, b));
        }
    }
    if (mark.rs_valid) {
        // A3: what the engine's FUN_01c92f30 left in the device before this draw against what the native resource set computed
        RsStats().compared.fetch_add(1, std::memory_order_relaxed);
        const u64 matrices = v.rd + 0x3790, buffer = R<u64>(v.rd + 0x3800);
        bool same = std::memcmp(reinterpret_cast<const void*>(matrices), mark.rs_bytes.data(), mark.rs_n * 16) == 0;
        std::string why = same ? "" : "the matrices block";
        if (same && buffer != 0) {
            const u64 block = u64{R<u32>(buffer + 0x60)} | (u64{R<u32>(buffer + 0x64) & 0xfffu} << 32);
            same = R<u32>(buffer + 0x68) == ((mark.rs_size + 0xfu) >> 4) && R<u32>(buffer + 0x6c) == 0x20077facu && block != 0 &&
                   std::memcmp(reinterpret_cast<const void*>(block), mark.rs_bytes.data(), mark.rs_n * 16) == 0;
            why = same ? "" : "the constant buffer";
        }
        if (!same) {
            RsStats().differing.fetch_add(1, std::memory_order_relaxed);
            ++v.mismatches;
            NoteMismatch(s, v.view, "resource set", fmt::format("draw {}: {} differs from the native one", k, why));
        }
    }
    if (mark.gb_buffer != 0) {
        // A3: the instance group's uploader buffer after the engine's FUN_01ee4e50 against the staged data the native group bind would have copied
        GbStats().compared.fetch_add(1, std::memory_order_relaxed);
        const u64 buffer = mark.gb_buffer;
        const u64 block = u64{R<u32>(buffer + 0x60)} | (u64{R<u32>(buffer + 0x64) & 0xfffu} << 32);
        const bool same = block != 0 && R<u32>(buffer + 0x6c) == 0x20077facu && R<u32>(buffer + 0x68) == ((R<u32>(buffer + 0x180) + 0xfu) >> 4) &&
                          std::memcmp(reinterpret_cast<const void*>(block), reinterpret_cast<const void*>(mark.gb_src), mark.gb_n) == 0;
        if (!same) {
            GbStats().differing.fetch_add(1, std::memory_order_relaxed);
            ++v.mismatches;
            NoteMismatch(s, v.view, "group bind", fmt::format("draw {}: the uploader buffer differs from the staged data", k));
        }
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
        } else if (!Timed(CallId::ScanDepth, [&] { return NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &scan_args); })) {
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
            verify_pass = cfg.verify_every != 0 && stats->eligible % cfg.verify_every == cfg.verify_every - 1 && !V().claimed.exchange(true);
        }
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassDepth, [&] { Generate(live, plan); });
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
        v.lo = ZPassBegin;
        v.hi = ZPassEnd;
        v.ctx = plan.ctx;
        v.rd = plan.render_device;
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
        v.claimed.store(false);
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

bool GBufferEnabled() {
    return GCfg().on;
}

bool HandleGBuffer(u64 render_device, u64 mode, u64 list, u64 args, u64 flags, u64 query_mode, u64 param7, u64 original_loop) {
    const GConfig& cfg = GCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    (void)param7;
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
    State& s = S();
    GStats& gs = GS();
    GPlan plan;
    plan.mode = static_cast<u32>(mode);
    plan.flags = static_cast<u32>(flags);
    plan.query_mode = static_cast<u32>(query_mode);
    GScanArgs scan_args{&plan, render_device, list, args};
    {
        std::scoped_lock lock{s.mutex};
        ++gs.v.passes;
        ++gs.arguments[static_cast<u32>(mode) * 1000 + static_cast<u32>(query_mode)];
        if (gs.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanGBuffer, [&] { return NativeDraw::GuardedCall([](void* p) { ScanGBuffer(*static_cast<GScanArgs*>(p)); }, &scan_args); })) {
            ++gs.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++gs.v.declines[r];
            }
        }
    }
    bool verify_pass = false;
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++gs.eligible;
        verify_pass = cfg.verify_every != 0 && gs.eligible % cfg.verify_every == cfg.verify_every - 1 && !V().claimed.exchange(true);
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        live.indexed_site = GBufferIndexedSite;
        live.indirect_site = GBufferIndirectSite;
        live.indexed_a4 = 0;
        Timed(CallId::PassGBuffer, [&] { GenerateGBuffer(live, plan, plan.flags); });
        std::scoped_lock lock{s.mutex};
        ++gs.v.generated;
        gs.v.generated_batches += plan.batches.size();
        gs.v.generated_draws += plan.steps.size();
        gs.v.group_batches += plan.group_batches;
        return true;
    }
    Verify& v = V();
    if (verify_pass) {
        v.shadow = Target{};
        v.shadow.live = false;
        v.shadow.indexed_site = GBufferIndexedSite;
        v.shadow.indirect_site = GBufferIndirectSite;
        v.shadow.indexed_a4 = 0;
        std::vector<u8> copy(CtxBytes);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(plan.ctx), CtxBytes);
        v.shadow.base = copy.data();
        GenerateGBuffer(v.shadow, plan, plan.flags);
        v.cursor = 0;
        v.arena_pos = 0;
        v.view = 100;
        v.lo = GBufferBegin;
        v.hi = GBufferEnd;
        v.ctx = plan.ctx;
        v.rd = plan.render_device;
        v.compared_draws = 0;
        v.mismatches = 0;
        v.raced = 0;
        v.thread = std::this_thread::get_id();
        const s32 draws_before = R<s32>(render_device + 0x10), triangles_before = R<s32>(render_device + 0x14);
        v.active.store(true);
        reinterpret_cast<Loop>(original_loop)(render_device, mode, list, args, flags, query_mode, param7);
        v.active.store(false);
        std::scoped_lock lock{s.mutex};
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
        ++gs.v.verify_passes;
        gs.v.verify_draws += v.compared_draws;
        gs.v.verify_raced += v.raced;
        gs.v.group_batches += plan.group_batches;
        if (v.mismatches == 0) {
            ++gs.v.verify_clean;
        } else {
            ++gs.v.verify_mismatched;
            gs.v.disabled = true;
            LOG_ERROR(Render, "G-buffer generator switched off, {} differences against the engine's loop", v.mismatches);
        }
        v.shadow = Target{};
        v.lo = ZPassBegin;
        v.hi = ZPassEnd;
        v.claimed.store(false);
    } else {
        reinterpret_cast<Loop>(original_loop)(render_device, mode, list, args, flags, query_mode, param7);
        std::scoped_lock lock{s.mutex};
        ++gs.v.loop_passes;
    }
    return true;
}

bool ForwardEnabled() {
    return FCfg().on;
}

bool HandleForward(u64 render_device, u64 a1, u64 a2, u64 list, u64 mode, u64 original_loop) {
    const FConfig& cfg = FCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64);
    State& s = S();
    FStats& fs = FS();
    FPlan plan;
    plan.render_device = render_device;
    plan.a1 = a1;
    plan.a2 = a2;
    plan.mode = static_cast<u32>(mode);
    last_forward_ctx.store(R<u64>(render_device + 0x2d8), std::memory_order_relaxed);
    FScanArgs scan_args{&plan, list};
    {
        std::scoped_lock lock{s.mutex};
        ++fs.v.passes;
        if (fs.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanForward, [&] { return NativeDraw::GuardedCall([](void* p) { ScanForwardPass(*static_cast<FScanArgs*>(p)); }, &scan_args); })) {
            ++fs.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++fs.v.declines[r];
            }
        }
    }
    bool verify_pass = false;
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++fs.eligible;
        verify_pass = cfg.verify_every != 0 && fs.eligible % cfg.verify_every == cfg.verify_every - 1 && !V().claimed.exchange(true);
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        live.indexed_site = ForwardIndexedSite;
        live.indexed_a4 = 1;
        Timed(CallId::PassForward, [&] { GenerateForward(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++fs.v.generated;
        fs.v.generated_batches += plan.batches.size();
        fs.v.generated_draws += plan.steps.size();
        return true;
    }
    Verify& v = V();
    if (verify_pass) {
        v.shadow = Target{};
        v.shadow.live = false;
        v.shadow.indexed_site = ForwardIndexedSite;
        v.shadow.indexed_a4 = 1;
        std::vector<u8> copy(CtxBytes);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(plan.ctx), CtxBytes);
        v.shadow.base = copy.data();
        GenerateForward(v.shadow, plan);
        v.cursor = 0;
        v.arena_pos = 0;
        v.view = 200;
        v.lo = ForwardBegin;
        v.hi = ForwardEnd;
        v.ctx = plan.ctx;
        v.rd = plan.render_device;
        v.compared_draws = 0;
        v.mismatches = 0;
        v.raced = 0;
        v.thread = std::this_thread::get_id();
        const s32 draws_before = R<s32>(render_device + 0x10), triangles_before = R<s32>(render_device + 0x14);
        v.active.store(true);
        reinterpret_cast<Loop>(original_loop)(render_device, a1, a2, list, mode);
        v.active.store(false);
        v.forward_block_set = false;
        std::scoped_lock lock{s.mutex};
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
        ++fs.v.verify_passes;
        fs.v.verify_draws += v.compared_draws;
        fs.v.verify_raced += v.raced;
        if (v.mismatches == 0) {
            ++fs.v.verify_clean;
        } else {
            ++fs.v.verify_mismatched;
            fs.v.disabled = true;
            LOG_ERROR(Render, "forward generator switched off, {} differences against the engine's loop", v.mismatches);
        }
        v.shadow = Target{};
        v.lo = ZPassBegin;
        v.hi = ZPassEnd;
        v.claimed.store(false);
    } else {
        reinterpret_cast<Loop>(original_loop)(render_device, a1, a2, list, mode);
        std::scoped_lock lock{s.mutex};
        ++fs.v.loop_passes;
    }
    return true;
}

bool DecalEnabled() {
    return DCfg().on;
}

bool HandleDecal(u64 render_device, u64 view, u64 a3, u64 a4, u64 desc, u32 flags, s32 type, u32 width, u32 height, u64 original_loop) {
    const DConfig& cfg = DCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64, u64, u64);
    const auto run_loop = [&] {
        reinterpret_cast<Loop>(original_loop)(render_device, view, a3, a4, desc, flags, static_cast<u64>(static_cast<u32>(type)), width, height);
    };
    State& s = S();
    DStats& ds = DS();
    DPlan plan;
    plan.render_device = render_device;
    plan.view = view;
    plan.a3 = a3;
    plan.a4 = a4;
    plan.desc = desc;
    plan.flags = flags;
    plan.type = type;
    plan.width = width;
    plan.height = height;
    DScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (ds.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanDecal, [&] { return NativeDraw::GuardedCall([](void* p) { ScanDecalPass(*static_cast<DScanArgs*>(p)); }, &scan_args); })) {
            ++ds.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++ds.loops_without_work;
        } else {
            ++ds.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++ds.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop(); // the engine's function does nothing for an empty list
        return true;
    }
    bool verify_pass = false;
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++ds.eligible;
        verify_pass = cfg.verify_every != 0 && ds.eligible % cfg.verify_every == cfg.verify_every - 1 && !V().claimed.exchange(true);
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassDecal, [&] { GenerateDecal(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++ds.v.generated;
        ds.v.generated_batches += plan.batches.size();
        ds.v.generated_draws += plan.steps.size();
        return true;
    }
    Verify& v = V();
    if (verify_pass) {
        v.shadow = Target{};
        v.shadow.live = false;
        std::vector<u8> copy(CtxBytes);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(plan.ctx), CtxBytes);
        v.shadow.base = copy.data();
        GenerateDecal(v.shadow, plan);
        v.cursor = 0;
        v.arena_pos = 0;
        v.view = 201;
        v.lo = DecalBegin;
        v.hi = DecalEnd;
        v.ctx = plan.ctx;
        v.rd = plan.render_device;
        v.compared_draws = 0;
        v.mismatches = 0;
        v.raced = 0;
        v.thread = std::this_thread::get_id();
        const s32 draws_before = R<s32>(render_device + 0x10), triangles_before = R<s32>(render_device + 0x14);
        v.active.store(true);
        run_loop();
        v.active.store(false);
        v.forward_block_set = false;
        std::scoped_lock lock{s.mutex};
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
        ++ds.v.verify_passes;
        ds.v.verify_draws += v.compared_draws;
        ds.v.verify_raced += v.raced;
        if (v.mismatches == 0) {
            ++ds.v.verify_clean;
        } else {
            ++ds.v.verify_mismatched;
            ds.v.disabled = true;
            LOG_ERROR(Render, "decal generator switched off, {} differences against the engine's loop", v.mismatches);
        }
        v.shadow = Target{};
        v.lo = ZPassBegin;
        v.hi = ZPassEnd;
        v.claimed.store(false);
    } else {
        run_loop();
        std::scoped_lock lock{s.mutex};
        ++ds.v.loop_passes;
    }
    return true;
}

bool SkinEnabled() {
    return SCfg().on;
}

bool HandleSkin(u64 system, u64 scene, u64 async_flag, u64 original_loop) {
    const SConfig& cfg = SCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64);
    State& s = S();
    SStats& ss = SS();
    SkinPlan plan;
    plan.system = system;
    plan.scene = scene;
    plan.async = (async_flag & 0xff) != 0;
    const auto run_loop = [&] { reinterpret_cast<Loop>(original_loop)(system, scene, async_flag); };
    SkinScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (ss.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanSkin, [&] { return NativeDraw::GuardedCall([](void* p) { ScanSkin(*static_cast<SkinScanArgs*>(p)); }, &scan_args); })) {
            ++ss.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++ss.loops_without_work;
        } else {
            ++ss.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++ss.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop();
        return true;
    }
    bool verify_pass = false;
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++ss.eligible;
        verify_pass = cfg.verify_every != 0 && ss.eligible % cfg.verify_every == cfg.verify_every - 1 && !V().claimed.exchange(true);
    }
    if (plan.Ok() && !verify_pass) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassSkin, [&] { GenerateSkin(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++ss.v.generated;
        ss.v.generated_batches += plan.nodes.size();
        ss.v.generated_draws += live.draws;
        return true;
    }
    Verify& v = V();
    if (verify_pass) {
        v.shadow = Target{};
        v.shadow.live = false;
        std::vector<u8> copy(CtxBytes);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(plan.ctx), CtxBytes);
        v.shadow.base = copy.data();
        GenerateSkin(v.shadow, plan);
        v.cursor = 0;
        v.arena_pos = 0;
        v.view = 202;
        v.lo = SkinBegin;
        v.hi = SkinEnd;
        v.ctx = plan.ctx;
        v.rd = plan.device;
        v.compared_draws = 0;
        v.mismatches = 0;
        v.raced = 0;
        v.thread = std::this_thread::get_id();
        v.active.store(true);
        run_loop();
        v.active.store(false);
        std::scoped_lock lock{s.mutex};
        if (v.cursor < v.shadow.marks.size()) {
            ++v.mismatches;
            NoteMismatch(s, v.view, "missing draw", fmt::format("the generator has {} dispatches, the loop issued {}", v.shadow.marks.size(), v.cursor));
        } else {
            v.mismatches += CompareDelta(s, v, v.arena_pos, v.shadow.arena.size(), plan.ctx, "epilogue");
        }
        ++ss.v.verify_passes;
        ss.v.verify_draws += v.compared_draws;
        ss.v.verify_raced += v.raced;
        if (v.mismatches == 0) {
            ++ss.v.verify_clean;
        } else {
            ++ss.v.verify_mismatched;
            ss.v.disabled = true;
            LOG_ERROR(Render, "skinning generator switched off, {} differences against the engine's loop", v.mismatches);
        }
        v.shadow = Target{};
        v.lo = ZPassBegin;
        v.hi = ZPassEnd;
        v.claimed.store(false);
    } else {
        run_loop();
        std::scoped_lock lock{s.mutex};
        ++ss.v.loop_passes;
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
    {
        const GStats& gs = GS();
        const ViewStats& v = gs.v;
        nlohmann::json g = {{"enabled", GCfg().on},
                            {"passes", v.passes},
                            {"generated_passes_without_the_loop", v.generated},
                            {"generated_batches", v.generated_batches},
                            {"generated_draws", v.generated_draws},
                            {"instance_group_batches", v.group_batches},
                            {"passes_run_by_the_loop", v.loop_passes},
                            {"verify_passes", v.verify_passes},
                            {"verify_clean", v.verify_clean},
                            {"verify_with_differences", v.verify_mismatched},
                            {"verify_draws_compared", v.verify_draws},
                            {"verify_words_that_moved", v.verify_raced},
                            {"switched_off_by_verify", v.disabled},
                            {"scan_faults", v.not_scanned},
                            {"passes_by_mode_times_1000_plus_query_mode", gs.arguments}};
        nlohmann::json declines = nlohmann::json::object();
        for (const auto& [reason, count] : v.declines) {
            declines[reason] = count;
        }
        g["declined_passes_by_reason"] = declines;
        out["gbuffer"] = g;
        const ViewStats& f = FS().v;
        nlohmann::json fj = {{"enabled", FCfg().on},
                             {"passes", f.passes},
                             {"generated_passes_without_the_loop", f.generated},
                             {"generated_batches", f.generated_batches},
                             {"generated_draws", f.generated_draws},
                             {"passes_run_by_the_loop", f.loop_passes},
                             {"verify_passes", f.verify_passes},
                             {"verify_clean", f.verify_clean},
                             {"verify_with_differences", f.verify_mismatched},
                             {"verify_draws_compared", f.verify_draws},
                             {"verify_words_that_moved", f.verify_raced},
                             {"switched_off_by_verify", f.disabled},
                             {"scan_faults", f.not_scanned}};
        nlohmann::json fdeclines = nlohmann::json::object();
        for (const auto& [reason, count] : f.declines) {
            fdeclines[reason] = count;
        }
        fj["declined_passes_by_reason"] = fdeclines;
        out["forward"] = fj;
        const ViewStats& dj = DS().v;
        nlohmann::json dd = {{"enabled", DCfg().on},
                             {"passes", dj.passes},
                             {"loops_without_work", DS().loops_without_work},
                             {"generated_passes_without_the_loop", dj.generated},
                             {"generated_batches", dj.generated_batches},
                             {"generated_draws", dj.generated_draws},
                             {"passes_run_by_the_loop", dj.loop_passes},
                             {"verify_passes", dj.verify_passes},
                             {"verify_clean", dj.verify_clean},
                             {"verify_with_differences", dj.verify_mismatched},
                             {"verify_draws_compared", dj.verify_draws},
                             {"verify_words_that_moved", dj.verify_raced},
                             {"switched_off_by_verify", dj.disabled},
                             {"scan_faults", dj.not_scanned}};
        nlohmann::json ddeclines = nlohmann::json::object();
        for (const auto& [reason, count] : dj.declines) {
            ddeclines[reason] = count;
        }
        dd["declined_passes_by_reason"] = ddeclines;
        out["decal"] = dd;
        const ViewStats& sj = SS().v;
        nlohmann::json sk = {{"enabled", SCfg().on},
                             {"passes", sj.passes},
                             {"loops_without_work", SS().loops_without_work},
                             {"generated_passes_without_the_loop", sj.generated},
                             {"generated_dispatches", sj.generated_draws},
                             {"passes_run_by_the_loop", sj.loop_passes},
                             {"verify_passes", sj.verify_passes},
                             {"verify_clean", sj.verify_clean},
                             {"verify_with_differences", sj.verify_mismatched},
                             {"verify_dispatches_compared", sj.verify_draws},
                             {"verify_draws_compared", sj.verify_draws},
                             {"switched_off_by_verify", sj.disabled},
                             {"scan_faults", sj.not_scanned}};
        nlohmann::json sdeclines = nlohmann::json::object();
        for (const auto& [reason, count] : sj.declines) {
            sdeclines[reason] = count;
        }
        sk["declined_passes_by_reason"] = sdeclines;
        out["skinning"] = sk;
    }
    if (CensusOn()) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        nlohmann::json census = nlohmann::json::object();
        const double to_ms = 1000.0 / static_cast<double>(frequency.QuadPart);
        for (u32 i = 0; i < static_cast<u32>(CallId::Count); ++i) {
            const u64 calls = Census()[i].calls.load(), ticks = Census()[i].ticks.load();
            if (calls != 0) {
                census[CallNames[i]] = {{"calls", calls}, {"total_ms", ticks * to_ms}, {"microseconds_per_call", ticks * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        for (u32 kind = 0; kind < 2; ++kind) {
            u64 calls, tag, original, taken;
            NativeDraw::IssueCensusRead(kind, calls, tag, original, taken);
            if (calls != 0) {
                census[kind == 0 ? "draw issue, indexed: the tag (TagIndexed)" : "draw issue, indirect: the tag (TagIndirect)"] = {{"calls", calls}, {"total_ms", tag * to_ms}, {"microseconds_per_call", tag * to_ms * 1000.0 / static_cast<double>(calls)}, {"draws_the_tag_took_over", taken}};
                census[kind == 0 ? "draw issue, indexed: the engine's draw function after the tag" : "draw issue, indirect: the engine's draw function after the tag"] = {{"calls", calls - taken}, {"total_ms", original * to_ms}, {"microseconds_per_call", original * to_ms * 1000.0 / static_cast<double>(std::max<u64>(1, calls - taken))}};
            }
        }
        {
            std::array<u64, 5> ticks{};
            u64 calls = 0;
            NativeDraw::TagCensusRead(ticks, calls);
            constexpr const char* names[5] = {"tag phase 0: probes, policy, class checks", "tag phase 1: resource tracking (the engine's, RefreshTargets) and the registry", "tag phase 2: signature of the context (Compute)",
                                              "tag phase 3: record build (MakeRecord, SignRecord)", "tag phase 4: tag words, execution eligibility, native execution side effects"};
            for (u32 i = 0; i < 5 && calls != 0; ++i) {
                census[names[i]] = {{"calls", calls}, {"total_ms", ticks[i] * to_ms}, {"microseconds_per_call", ticks[i] * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        {
            std::vector<std::pair<u64, u32>> changed;
            for (u32 i = 0; i < reuse_changed_words.size(); ++i) {
                if (const u64 n = reuse_changed_words[i].load()) {
                    changed.emplace_back(n, i * 8);
                }
            }
            std::sort(changed.rbegin(), changed.rend());
            nlohmann::json words = nlohmann::json::object();
            for (size_t i = 0; i < changed.size() && i < 40; ++i) {
                words[fmt::format("{:#x}", changed[i].second)] = changed[i].first;
            }
            census["context words (0 to 0xd80) that changed between consecutive draws: offset -> draws"] = words;
        }
        census["context state equal to the previous draw's (only the draw arguments differ)"] = {{"draws", Reuse().draws.load()}, {"same_as_previous", Reuse().same_as_previous.load()}, {"same_as_one_of_the_last_8", Reuse().same_as_one_of_last_8.load()}};
        {
            std::array<u64, 6> ticks{};
            u64 calls = 0;
            NativeDraw::MakeRecordCensusRead(ticks, calls);
            constexpr const char* names[6] = {"record build 0: setup", "record build 1: Synthesize (registers)", "record build 2: Fixed::Build (fixed function state)", "record build 3: user data bindings, vertex stage",
                                              "record build 4: user data bindings, pixel stage", "record build 5: draw argument registers and the rest"};
            for (u32 i = 0; i < 6 && calls != 0; ++i) {
                census[names[i]] = {{"calls", calls}, {"total_ms", ticks[i] * to_ms}, {"microseconds_per_call", ticks[i] * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        {
            u64 lookups, hits_big, hits_small;
            NativeDraw::RecordCacheCensusRead(lookups, hits_big, hits_small);
            census["a cache of records keyed by the whole signature (all groups but the draw arguments), direct mapped"] = {{"lookups", lookups}, {"hits_with_1M_entries", hits_big}, {"hits_with_64K_entries", hits_small}};
        }
        {
            u64 scan, fill, tables, slots;
            NativeDraw::UserDataCensusRead(scan, fill, tables, slots);
            if (tables != 0) {
                census["user data tables (resource, constant buffer, sampler pointers): the slot scan"] = {{"tables", tables}, {"total_ms", scan * to_ms}, {"microseconds_per_table", scan * to_ms * 1000.0 / static_cast<double>(tables)}};
                census["user data tables: filling the descriptors"] = {{"tables", tables}, {"slots_filled", slots}, {"total_ms", fill * to_ms}, {"microseconds_per_table", fill * to_ms * 1000.0 / static_cast<double>(tables)},
                                                                       {"nanoseconds_per_slot", fill * to_ms * 1e6 / static_cast<double>(std::max<u64>(1, slots))}};
            }
        }
        {
            std::array<u64, 6> m{};
            NativeDraw::MergeCensusRead(m);
            if (m[0] != 0) {
                census["records equal to an earlier record of the thread except for the draw arguments"] = {{"draws", m[0]}, {"equal_to_the_previous_record", m[1]}, {"equal_to_one_of_the_last_16", m[2]},
                                                                                                                   {"registers_and_fixed_state_equal_to_the_previous", m[3]}, {"registers_equal_to_the_previous", m[4]}};
            }
        }
        out["engine_call_census"] = census;
    }
    if (CensusOn()) {
        nlohmann::json audit = nlohmann::json::object();
        NativeDraw::InputTableAuditRead(audit);
        out["input_table_audit"] = audit;
    }
    {
        nlohmann::json sites = nlohmann::json::object();
        NativeDraw::DrawSiteCensusRead(sites);
        if (sites["total_draws"].get<u64>() != 0) {
            out["draw_site_census"] = std::move(sites);
        }
    }
    {
        nlohmann::json list = nlohmann::json::object();
        NativeDraw::NativeListRead(list);
        out["native_list"] = std::move(list);
    }
    {
        nlohmann::json audit = nlohmann::json::object();
        NativeDraw::SynthAuditRead(audit);
        if (audit["records"].get<u64>() != 0) {
            out["synthesize_audit"] = std::move(audit);
        }
    }
    out["native_handle_lookup_FUN_01c59060"] = {{"answered_natively", Leaf().native.load()}, {"engine_creates_the_render_data", Leaf().engine_creating.load()},
                                                {"null_object", Leaf().engine_null.load()}, {"verify_calls_of_the_engine", Leaf().verified.load()},
                                                {"verify_differences", Leaf().differing.load()},
                                                {"forward_prologues_made_natively_live_and_shadow", Leaf().prologue_native.load()},
                                                {"forward_prologue_blocks_compared_with_the_engines", Leaf().block_compared.load()},
                                                {"forward_prologue_blocks_differing", Leaf().block_differing.load()}};
    out["native_variant_find_FUN_01c822c0"] = {{"answered_natively_from_the_cached_variant", VStats().native.load()}, {"engine_calls_search_or_create", VStats().engine.load()},
                                               {"verify_calls_of_the_engine", VStats().verified.load()}, {"verify_differences", VStats().differing.load()}};
    if (const u64 ctx = last_forward_ctx.load(); ctx != 0) {
        nlohmann::json vt = nlohmann::json::object();
        const u64 table = R<u64>(ctx);
        for (u32 slot = 0; slot < 0x120 / 8; ++slot) {
            vt[fmt::format("{:#x}", slot * 8)] = fmt::format("{:#x}", R<u64>(table + slot * 8) - NativeDraw::ImageBase() + 0x1000000);
        }
        out["study_device_context_vtable_as_ghidra_addresses"] = vt;
    }
    out["native_resource_set_FUN_01c92f30"] = {{"made_natively", RsStats().native.load()}, {"engine_special_entry", RsStats().engine_special.load()},
                                              {"engine_other_buffer_shape", RsStats().engine_shape.load()}, {"verify_draws_compared", RsStats().compared.load()},
                                              {"verify_draws_differing", RsStats().differing.load()}};
    out["native_group_bind_FUN_01ee4e50"] = {{"made_natively", GbStats().native.load()}, {"engine_other_buffer_shape", GbStats().engine_shape.load()},
                                            {"verify_draws_compared", GbStats().compared.load()}, {"verify_draws_differing", GbStats().differing.load()}};
    out["verify_difference_kinds"] = s.mismatch_kinds;
    out["special_entry_classes"] = s.special_classes;
    out["verify_words_that_moved_by_view_kind_offset"] = s.raced_words;
    out["verify_difference_samples"] = s.samples;
    return out;
}

} // namespace Core::NativeZPassGen
