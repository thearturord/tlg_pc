// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Private to liverpool.cpp and native/native_liverpool.cpp: the nested types of the raster thread's operation queue (RasterCall, RasterOp, RasterSplit)
// and IsFixedStateRegister. Include after video_core/amdgpu/liverpool.h and its usual companions.

#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

#include "common/polyfill_thread.h"
#include "native/native_draw_tag.h"
#include "video_core/amdgpu/liverpool.h"

namespace AmdGpu {

// ---- Command processor split (documents/tlg/cp-split-design.md) ----

/// A draw or dispatch for the rasterizer, with everything it needs besides the registers.
struct Liverpool::RasterCall {
    enum class Kind : u8 { Draw, DrawIndirect, DispatchDirect, DispatchIndirect };
    Kind kind{};
    bool asc{};       ///< Marker names come from a compute ring (vqid argument).
    bool no_marker{}; ///< Run without a ScopeMarker.
    u32 vqid{};
    const void* cmd_address{}; ///< Only printed in markers, never dereferenced.
    const ComputeProgram* cs_override{}; ///< Dispatch: this program instead of the queue's (valid until RasterDispatch returns).
    const char* marker{};      ///< Format string literal.
    bool indexed{};
    u32 index_offset{};
    VAddr args_addr{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_addr{};
    u16 vertex_sgpr_offset{};
    u16 instance_sgpr_offset{};
    u32 dispatch_size{};
    VideoCore::NativeTag tag{}; ///< From the game's native draw hook, draws only.
    VAddr fill_destination{};   ///< A dispatch of the fill shader: the pattern fill it performs (the raster thread may clear an image instead).
    u32 fill_bytes{};
    u32 fill_dwords{};
    std::array<u32, 4> fill_pattern{};
};

struct Liverpool::RasterOp {
    bool has_call{};
    bool writes_memory{};
    u8 audit_class{0xff}; ///< Core::ResourceAudit: the queue whose command stream pushed this operation
    u32 write_line{}; ///< Source line of the RasterRun call that queued a writes_memory closure.
    u64 position{};   ///< The queue position (Liverpool::RasterSplit::pushed after the push)
    RasterCall call{};
    Common::UniqueFunction<void> closure{};
    ComputeProgram cs{};
    std::unique_ptr<Vulkan::DrawPlan> draw_plan{};
    std::unique_ptr<Vulkan::DispatchPlan> dispatch_plan{};
};

struct Liverpool::RasterSplit {
    static constexpr size_t MaxQueued = 256;

    std::jthread thread;
    std::jthread housekeeping; ///< Slow periodic work that must not stall the raster thread.
    std::mutex mutex;
    std::condition_variable_any work_cv; ///< Raster thread: work arrived.
    std::condition_variable_any done_cv; ///< Front: space in the queue or operations done.
    std::vector<RasterOp> queue; // swapped with the raster thread's batch: no per-op allocation
    std::atomic<u64> pushed{};
    std::atomic<u64> done{};
    std::atomic<u32> pending_writes{}; ///< Queued closures with writes_memory.
    std::atomic<u64> write_done{};     ///< The queue position of the newest writes_memory closure the raster thread has finished (`done` moves in batches)
    /// What the front thread is doing (for the raster thread's idle statistics): 0 working,
    /// 1 waiting for the game to submit, 2 waiting for queued memory writes, 3 queue full.
    std::atomic<u8> front_state{0};
    std::array<std::atomic<u64>, 4> raster_idle_ns_by_front_state{};
    /// Idle time by the kind of the operation that ended it: draw, indirect draw, dispatch,
    /// indirect dispatch, closure (fence, write, flip, ...).
    std::array<std::atomic<u64>, 5> raster_idle_ns_by_next_op{};
    std::array<std::atomic<u64>, 5> raster_idle_count_by_next_op{};
    /// Statistics per RasterRun call site that queues memory-writing closures: how many, and how
    /// long the command processor waited for them before a pipeline lookup.
    struct WriteSite {
        std::atomic<u32> line{};
        std::atomic<u64> pushes{};
        std::atomic<u64> wait_ns{};
    };
    std::array<WriteSite, 24> write_sites;
    u32 last_write_line{}; ///< Command processor thread: line of the newest queued write closure.
    WriteSite* SiteFor(u32 line) {
        for (auto& site : write_sites) {
            u32 expected = 0;
            if (site.line.load(std::memory_order_relaxed) == line ||
                (site.line.compare_exchange_strong(expected, line) || expected == line)) {
                return &site;
            }
        }
        return &write_sites.back();
    }
    // Statistics, logged every few seconds.
    u64 stat_ops{};
    u64 stat_calls{};
    u64 stat_write_drains{};
    std::atomic<u64> stat_front_blocked{}; ///< Pushes that found the queue full.
    std::atomic<u64> stat_raster_idle_ns{};
    u64 stat_draws{};
    u64 stat_draws_no_reg_change{};
    // Raster thread wall time by kind of work (ns), logged every few seconds.
    u64 ns_draws{};
    u64 ns_depth_only_draws{}; ///< part of ns_draws spent in draws without a color target (z-prepass, shadows)
    u64 n_draws{};
    u64 n_depth_only_draws{};
    u64 ns_dispatches{};
    u64 ns_write_closures{};
    u64 ns_other_closures{};
    u64 ns_callbacks{};
    u64 stat_last_log_ns{};
    bool verify{}; ///< cp_split_mode 2: every operation is waited for (synchronous)
};

/// Registers whose only reader is the translation to the fixed-function state (blend, stencil, depth range of the viewport,
/// screen scissor, pixel shader colour mask).
inline bool IsFixedStateRegister(u32 index) {
    if (index < Regs::ContextRegWordOffset) {
        return false;
    }
    const u32 reg = index - Regs::ContextRegWordOffset;
    return (reg >= 0x1e0 && reg <= 0x1e7) || (reg >= 0x105 && reg <= 0x108) || (reg >= 0x10b && reg <= 0x10d) ||
           reg == 0xb4 || reg == 0xb5 || reg == 0xc || reg == 0xd || reg == 0x8f || reg == 0x8e;
}

} // namespace AmdGpu
