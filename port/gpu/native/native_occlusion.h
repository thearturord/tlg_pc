// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

/// Real occlusion queries (documents/tlg/native plan/no-emulation-plan.md, E5, owner decision 2026-10-30). The game asks "can you see this object?" by
/// writing a begin and an end into its command stream, with a result block in memory (per counter pair a begin word and an end word, 16 bytes apart,
/// bit 63 = valid; the answer is the sum of end - begin). Until 2026-10-30 the answer was made up (always visible). Here a Vulkan occlusion query counts
/// what the draws between begin and end really draw, and the answer is written into the block when the graphics card has finished, so the game reads
/// the previous frame's answer, never waits, and a block that has never been answered reads "visible".
///
/// The answer is yes/no: the counters of a block differ by 0x2FFFFFF per pair when something was seen (what the made up answer always was), by 0 when not.
/// `TLG_REAL_OCCLUSION=0` brings back the made up answer.
namespace Core::NativeOcclusion {

/// False when `TLG_REAL_OCCLUSION=0`
[[nodiscard]] bool Enabled() noexcept;
/// The report of the live Queries object (or {"enabled": false})
[[nodiscard]] nlohmann::json ReportAll();

class Queries {
public:
    Queries(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler, std::function<void(VAddr, u64)> note_guest_write);
    ~Queries();

    /// Raster thread, in order with the draws: the game begins counting for the block at `address` (`pairs` counter pairs, 8, or 16 on a Neo)
    void Open(VAddr address, u32 pairs);
    /// ... and ends it. Draws between Open and Close are counted, also across render pass changes.
    void Close(VAddr address);
    /// Raster thread, once per submission: writes the answers of the blocks the graphics card has finished
    void Poll();

    [[nodiscard]] nlohmann::json Report() const;

private:
    static constexpr u32 NumSlots = 256;

    struct Block {
        u64 base{0x100000};        ///< the counters the next answer starts from (they only grow, as the made up ones did)
        u32 pairs{8};
        bool answered{};           ///< a made up first answer was written
    };
    struct Slot {
        VAddr address{};
        u64 tick{};                ///< the scheduler tick that ends the query; the answer exists once the graphics card reached it
        enum State : u8 { Free, Open, Pending } state{Free};
    };

    void WriteAnswer(VAddr address, bool seen);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    std::function<void(VAddr, u64)> note_guest_write;
    vk::QueryPool pool{};
    std::array<Slot, NumSlots> slots{};
    u32 next_slot{};
    VAddr open_address{};
    u32 open_slot{~0u};
    std::unordered_map<VAddr, Block> blocks;
    mutable std::mutex mutex;
    std::atomic<u64> opened{}, closed{}, seen{}, hidden{}, dropped{}, first_answers{}, not_ready{};
};

} // namespace Core::NativeOcclusion
