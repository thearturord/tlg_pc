// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Helpers of Vulkan::Rasterizer's native path that both vk_rasterizer.cpp and native/native_rasterizer.cpp use (include after the rasterizer's own includes).

#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

#include "native/native_draw.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

/// N3 native binder switch (`TLG_NATIVE_BINDER`): 0 off, 1 the depth passes (class 0), 2 every class with a complete binding list, 3 like 2 and every descriptor is compared with
/// the legacy one (the legacy one wins on a difference). Modes 1 and 2 also leave the flat user data out of the plan of a stage the record covers completely (N3b).
/// N3c handle switch (`TLG_NATIVE_HANDLES`): 0 off, 1 the immutable engine buffers of a draw are found by the engine's id (resolved when the plan is made),
/// 2 like 1 and every handle is compared with the lookup by address (the lookup wins on a difference). Needs the binder (the record's bindings) and the immutable residency.
inline u32 NativeHandleMode() {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_NATIVE_HANDLES");
        return e != nullptr && *e != 0 ? static_cast<u32>(std::atoi(e)) : 2u; // default 2 since 2026-10-27 (the owner's play test); 0 turns it off
    }();
    return mode;
}

/// L2 census: the record's binding list of the dispatch being bound, and a dump of it when a descriptor of a dispatch differs from the guest's (once per call site, program, kind, offset; at most 60 dumps)
struct DispatchDumpContext {
    const Core::NativeDraw::RecordBinding* list{};
    u32 count{};
    u64 seq{};
};
inline thread_local DispatchDumpContext g_dump_context;
inline void DumpDispatchRecord(const Shader::Info& stage, u32 kind, u32 offset, const char* why) {
    static std::mutex mutex;
    static std::set<std::tuple<u64, u64, u32, u32>> seen;
    static u32 dumps = 0;
    const u64 caller = Core::NativeDraw::RecordCallerGhidra(g_dump_context.seq);
    std::scoped_lock lock{mutex};
    if (dumps >= 60 || !seen.insert({caller, u64{stage.pgm_hash}, kind, offset}).second) {
        return;
    }
    ++dumps;
    u32 root = 0, src = 0;
    const bool table = stage.srt_info.TableSource(offset, root, src);
    LOG_ERROR(Render_Vulkan, "L2 dump ({}): caller {:#x} program {:#x} kind {} flat offset {} table {} root sgpr {} src {} entries {}", why, caller, u64{stage.pgm_hash}, kind, offset, table, root, src, g_dump_context.count);
    for (u32 i = 0; i < g_dump_context.count && i < 64; ++i) {
        const auto& b = g_dump_context.list[i];
        LOG_ERROR(Render_Vulkan, "L2 entry {}: stage {} kind {} slot {} offset {:#x} dwords {} table_sgpr {} entry_dwords {} id {:#x} words {:08x} {:08x} {:08x} {:08x}", i, b.stage, b.kind, b.slot, b.offset, b.dwords, b.table_sgpr,
                  b.entry_dwords, b.id, b.words[0], b.words[1], b.words[2], b.words[3]);
    }
}

} // namespace Vulkan
