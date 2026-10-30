// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>

#include "common/logging/log.h"
#include "native/native_occlusion.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Core::NativeOcclusion {

namespace {
constexpr u64 ValidBit = 1ull << 63;
constexpr u64 PlainlyVisible = 0x2FFFFFFull; // the difference per counter pair the made up answer had
std::atomic<Queries*> live{nullptr};
} // namespace

bool Enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_REAL_OCCLUSION");
        const char* tags = std::getenv("TLG_E5_TAGS"); // the real queries run on the query tags
        return (e == nullptr || *e != '0') && (tags == nullptr || *tags != '0');
    }();
    return on;
}

Queries::Queries(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, std::function<void(VAddr, u64)> note_guest_write_)
    : instance{instance_}, scheduler{scheduler_}, note_guest_write{std::move(note_guest_write_)} {
    vk::QueryPoolCreateInfo info{};
    info.queryType = vk::QueryType::eOcclusion;
    info.queryCount = NumSlots;
    pool = instance.GetDevice().createQueryPool(info).value;
    live.store(this, std::memory_order_release);
}

Queries::~Queries() {
    live.store(nullptr, std::memory_order_release);
    instance.GetDevice().destroyQueryPool(pool);
}

nlohmann::json ReportAll() {
    Queries* queries = live.load(std::memory_order_acquire);
    return queries != nullptr ? queries->Report() : nlohmann::json{{"enabled", Enabled()}, {"opened", 0}};
}

void Queries::WriteAnswer(VAddr address, bool seen_now) {
    Block& block = blocks[address];
    // Both words of a pair move on each answer, like the made up counters did (they only grow). The end words go first: a reader that comes in the middle
    // sees end - begin as three steps, which reads "visible", never "hidden" by mistake.
    block.base += 2 * PlainlyVisible;
    note_guest_write(address, block.pairs * 16ull);
    auto* words = reinterpret_cast<u64*>(address);
    for (u32 pair = 0; pair < block.pairs; ++pair) {
        words[2 * pair + 1] = ((block.base + (seen_now ? PlainlyVisible : 0)) & ~ValidBit) | ValidBit;
    }
    for (u32 pair = 0; pair < block.pairs; ++pair) {
        words[2 * pair] = (block.base & ~ValidBit) | ValidBit;
    }
}

void Queries::Open(VAddr address, u32 pairs) {
    std::scoped_lock lock{mutex};
    Block& block = blocks[address];
    block.pairs = pairs;
    if (!block.answered) {
        // nothing has been measured for this block yet: it reads "visible" until the first real answer
        block.answered = true;
        WriteAnswer(address, true);
        first_answers.fetch_add(1, std::memory_order_relaxed);
    }
    if (open_slot != ~0u) {
        return; // a block is already open (the game does not nest them); the new one gets no query and keeps its last answer
    }
    Slot& slot = slots[next_slot];
    if (slot.state == Slot::Pending) {
        dropped.fetch_add(1, std::memory_order_relaxed);
    }
    const u32 index = next_slot;
    next_slot = (next_slot + 1) % NumSlots;
    slot.address = address;
    slot.state = Slot::Open;
    open_slot = index;
    open_address = address;
    scheduler.EndRendering();
    const vk::QueryPool query_pool = pool;
    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.resetQueryPool(query_pool, index, 1);
    cmdbuf.record([query_pool, index](vk::CommandBuffer c) { c.beginQuery(query_pool, index, {}); });
    opened.fetch_add(1, std::memory_order_relaxed);
}

void Queries::Close(VAddr address) {
    std::scoped_lock lock{mutex};
    if (open_slot == ~0u || open_address != address) {
        return;
    }
    scheduler.EndRendering();
    const vk::QueryPool query_pool = pool;
    const u32 index = open_slot;
    scheduler.CommandBuffer().record([query_pool, index](vk::CommandBuffer c) { c.endQuery(query_pool, index); });
    slots[index].state = Slot::Pending;
    slots[index].tick = scheduler.CurrentTick();
    open_slot = ~0u;
    open_address = 0;
    closed.fetch_add(1, std::memory_order_relaxed);
}

void Queries::Poll() {
    std::scoped_lock lock{mutex};
    for (u32 index = 0; index < NumSlots; ++index) {
        Slot& slot = slots[index];
        if (slot.state != Slot::Pending || !scheduler.IsFree(slot.tick)) {
            continue;
        }
        std::array<u64, 2> result{}; // the sample count, then its availability
        const vk::Result status = instance.GetDevice().getQueryPoolResults(pool, index, 1, sizeof(result), result.data(), sizeof(result),
                                                                           vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (status != vk::Result::eSuccess || result[1] == 0) {
            not_ready.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        slot.state = Slot::Free;
        const bool seen_now = result[0] != 0;
        (seen_now ? seen : hidden).fetch_add(1, std::memory_order_relaxed);
        WriteAnswer(slot.address, seen_now);
    }
}

nlohmann::json Queries::Report() const {
    return {{"enabled", true},
            {"blocks", [&] {
                 std::scoped_lock lock{mutex};
                 return blocks.size();
             }()},
            {"opened", opened.load()},
            {"closed", closed.load()},
            {"answered_seen", seen.load()},
            {"answered_hidden", hidden.load()},
            {"first_answers_made_up", first_answers.load()},
            {"answers_not_ready_at_a_poll", not_ready.load()},
            {"slots_overwritten_before_answered", dropped.load()}};
}

} // namespace Core::NativeOcclusion
