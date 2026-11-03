// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Helpers of Vulkan::Rasterizer's engine path that both vk_rasterizer.cpp and vk_rasterizer_engine.cpp use (include after the rasterizer's own includes).

#include <mutex>
#include <set>
#include <tuple>

#include "native/native_consumer_support.h"
#include "native/native_draw.h"
#include "native/native_log.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

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
    TLG_LOG(RASTER, Error, "L2 dump ({}): caller {:#x} program {:#x} kind {} flat offset {} table {} root sgpr {} src {} entries {}", why, caller, u64{stage.pgm_hash}, kind, offset, table, root, src, g_dump_context.count);
    for (u32 i = 0; i < g_dump_context.count && i < 64; ++i) {
        const auto& b = g_dump_context.list[i];
        TLG_LOG(RASTER, Error, "L2 entry {}: stage {} kind {} slot {} offset {:#x} dwords {} table_sgpr {} entry_dwords {} id {:#x} words {:08x} {:08x} {:08x} {:08x}", i, b.stage, b.kind, b.slot, b.offset, b.dwords, b.table_sgpr,
                  b.entry_dwords, b.id, b.words[0], b.words[1], b.words[2], b.words[3]);
    }
}

} // namespace Vulkan
