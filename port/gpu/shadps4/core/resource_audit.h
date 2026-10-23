// SPDX-License-Identifier: GPL-2.0-or-later
// Resource audit (documents/tlg/async-compute-queue-plan.md, step 1): while a timeline capture runs, every buffer and image the raster thread
// binds is written to the capture as a ResourceUse event with the queue it belongs to, so the resources of the asynchronous compute queue 1
// can be compared with those of the frame's graphics. Diagnostics only, no effect on the rendering.

#pragma once

#include <unordered_map>

#include <xxhash.h>

#include "common/types.h"
#include "core/frame_timeline.h"

namespace Core::ResourceAudit {

inline constexpr u8 NoClass = 0xff;
/// Queue class of the commands being parsed on the command processor thread (curr_qid; NoClass outside a queue task).
inline thread_local u8 front_class = NoClass;
/// Queue class of the operation the raster thread is executing.
inline thread_local u8 raster_class = NoClass;

enum Kind : u8 { BufferUse = 1, ImageUse = 2, GdsFill = 3, MemoryCopyDst = 4, MemoryCopySrc = 5 };

/// Raster thread only. Dedupes by (class, kind, address, size, write) within 4 ms.
inline void Note(Kind kind, u64 address, u64 size, bool written) {
    if (raster_class == NoClass || !FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
        return;
    }
    static thread_local std::unordered_map<u64, u64> last_emit;
    const u64 now = FrameTimeline::Now();
    u64 key = (address * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full) ^ (u64(raster_class) << 56) ^ (u64(kind) << 52) ^ (u64(written) << 51);
    key ^= key >> 31;
    auto& at = last_emit[key];
    if (now - at < 4'000'000) {
        return;
    }
    at = now;
    if (last_emit.size() > 400'000) {
        last_emit.clear();
    }
    if (raster_class == 1 && kind == BufferUse && !written && address > 0x100000000ull && size <= (8u << 20)) {
        // does the content change from frame to frame (are the inputs of queue 1 written by the CPU every frame)?
        FrameTimeline::Record(FrameTimeline::ResourceHash, address, now, XXH64(reinterpret_cast<const void*>(address), size, 0));
    }
    FrameTimeline::Record(FrameTimeline::ResourceUse, (address & 0xffffffffffffull) | (u64(raster_class) << 48) | (u64(kind) << 56) | (u64(written) << 63), now, size);
}

} // namespace Core::ResourceAudit
