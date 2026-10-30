// SPDX-License-Identifier: GPL-2.0-or-later
// N6.6 (read only): what the engine's G-buffer loop (`RenderDev_DrawGBufferPrimitives`, ghidra 0x01c76710) is asked to do in the live scenes: pass arguments, batch shapes, primitive shapes.
// `TLG_GBUFFER_AUDIT=1`, report at /gbuffer. Before the generator for this loop is written the shapes it has to cover are counted here.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "native/native_draw.h"
#include "native/native_gbuffer_audit.h"

namespace Core::NativeGBufferAudit {

namespace {

template <class T>
T R(u64 address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}

struct State {
    std::mutex mutex;
    std::map<std::string, u64> passes;       // arguments
    std::map<std::string, u64> batches;      // batch shapes
    std::map<std::string, u64> prims;        // primitive shapes
    std::map<std::string, u64> forward_passes, forward_batches, forward_prims;
    u64 faults{0};
};
State& S() {
    static State state;
    return state;
}

struct Args {
    u64 rd, mode, list, flags, qmode, p7;
};

void Scan(Args& a) {
    State& s = S();
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    const s32 indirect_mode = R<s32>(a.rd + 0x38e0);
    s.passes[fmt::format("mode {} flags {:#x} query mode {} p7 {} indirect {} batches>0 {}", a.mode, a.flags, a.qmode, a.p7, indirect_mode, batch_count != 0)]++;
    for (u32 b = 0; b < batch_count && b < 100000; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        const u16 bflags = R<u16>(batch + 0x66);
        const u64 mesh = R<u64>(batch + 0x20), second = R<u64>(batch + 0x28), group = R<u64>(batch + 0x38), field40 = R<u64>(batch + 0x40), special = R<u64>(batch + 0x48);
        const u16 prim_count = R<u16>(batch + 100);
        u32 negative = 0, nonneg = 0, no_data40 = 0;
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count && p < 4096; ++p) {
            const u64 rec = prim_array + u64{p} * 0x38;
            const s32 v24 = R<s32>(rec + 0x24);
            (v24 < 0 ? negative : nonneg)++;
            const u64 data = R<u64>(rec + 0x28);
            if (data != 0 && R<u64>(data + 0x40) == 0) {
                ++no_data40;
            }
            const u32 v30 = R<u32>(rec + 0x30);
            s.prims[fmt::format("mode {} +0x24 {} +0x30 {} material flag {:#x}", a.mode, v24 < 0 ? "negative" : "non-negative", v30 != 0 ? "set" : "0", R<u32>(R<u64>(rec + 0x18) + 0xd4) & 0x2e)]++;
        }
        s.batches[fmt::format("mode {} flags {:#x} mesh {} second {} group {} field40 {} special {} +0x38 flags {:#x} prims {} neg {} cachedvariant-missing {}", a.mode, bflags & 0x14, mesh != 0, second != 0, group != 0,
                              field40 != 0, special != 0, bflags & 0x1fff, prim_count != 0, negative != 0, no_data40 != 0)]++;
    }
}

struct ForwardArgs {
    u64 rd, list, mode;
};

void ScanForward(ForwardArgs& a) {
    State& s = S();
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    s.forward_passes[fmt::format("mode {} indirect {} batches>0 {}", a.mode, R<s32>(a.rd + 0x38e0), batch_count != 0)]++;
    for (u32 b = 0; b < batch_count && b < 100000; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        const u16 bflags = R<u16>(batch + 0x66);
        const u64 mesh = R<u64>(batch + 0x20), second = R<u64>(batch + 0x28), group = R<u64>(batch + 0x38), special = R<u64>(batch + 0x48);
        const u16 prim_count = R<u16>(batch + 100);
        const u64 render_data = mesh != 0 ? R<u64>(mesh + 0x38) : 0;
        u64 first_entry_valid = 0;
        if (render_data != 0) {
            const u64 inner = R<u64>(render_data + 0x20);
            const u64 context_global = R<u64>(NativeDraw::ImageBase() + (0x069a1b40ull - 0x1000000));
            const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
            if (inner != 0 && R<u32>(inner + 0x68) != 0) {
                const u64 first = R<u64>(R<u64>(inner + 0x60));
                first_entry_valid = (first != default_object && first != 0 && R<u64>(first + 0x38) != 0) ? 1 : 0;
            }
        }
        s.forward_batches[fmt::format("mode {} group {} (skipped by the loop) mesh {} render data {} second {} special {} flags {:#x} first table entry valid {} prims {}", a.mode, group != 0, mesh != 0, render_data != 0, second != 0, special != 0,
                                      bflags & 0x1fff, first_entry_valid, prim_count != 0)]++;
        if (group != 0 || mesh == 0) {
            continue;
        }
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count && p < 4096; ++p) {
            const u64 rec = prim_array + u64{p} * 0x38;
            const u64 material = R<u64>(rec + 0x18);
            const u64 data = R<u64>(rec + 0x28);
            s.forward_prims[fmt::format("material +0x158 {} cached object +0x48 {} flags {:#x} token {} prim +0x24 {} +0x30 {} data +0x40 {}", R<u32>(material + 0x158), R<u64>(material + 0x48) != 0, R<u32>(material + 0xd4) & 0x2e,
                                        R<u32>(material + 0x1bc) != 0, R<s32>(rec + 0x24) < 0 ? "negative" : "non-negative", R<u32>(rec + 0x30) != 0, data != 0 && R<u64>(data + 0x40) != 0)]++;
        }
    }
}

} // namespace

void EnterForward(u64 render_device, u64 list, u64 mode) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GBUFFER_AUDIT");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    if (!on || list == 0) {
        return;
    }
    static std::atomic<u32> counter{0};
    if (counter.fetch_add(1) % 16 != 0) {
        return;
    }
    ForwardArgs args{render_device, list, mode};
    std::scoped_lock lock{S().mutex};
    if (!NativeDraw::GuardedCall([](void* p) { ScanForward(*static_cast<ForwardArgs*>(p)); }, &args)) {
        ++S().faults;
    }
}

void Enter(u64 render_device, u64 mode, u64 list, u64 flags, u64 query_mode, u64 param7) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GBUFFER_AUDIT");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    if (!on || list == 0) {
        return;
    }
    static std::atomic<u32> counter{0};
    if (counter.fetch_add(1) % 16 != 0) {
        return;
    }
    Args args{render_device, mode, list, flags, query_mode, param7};
    std::scoped_lock lock{S().mutex};
    if (!NativeDraw::GuardedCall([](void* p) { Scan(*static_cast<Args*>(p)); }, &args)) {
        ++S().faults;
    }
}

nlohmann::json Report() {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    return {{"passes_by_arguments", s.passes}, {"batch_shapes", s.batches}, {"primitive_shapes", s.prims}, {"forward_passes", s.forward_passes}, {"forward_batch_shapes", s.forward_batches},
            {"forward_primitive_shapes", s.forward_prims}, {"faults", s.faults}};
}

} // namespace Core::NativeGBufferAudit
