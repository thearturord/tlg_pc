// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_hook_owned.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "native/native_device.h"
#include "native/native_n0_census.h"
#include "native/native_updates.h"

namespace Core::HookOwned {

u32 Mode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_HOOK_OWNED");
        return e != nullptr && e[0] == '1' ? 1u : 0u; // (2, real, is not built yet)
    }();
    return mode;
}

thread_local u8 t_mark = MarkNone;

namespace {

// ---- hook side: one slot per engine buffer id that a light or slot write or a texture creation named (open addressing, never removed; a full probe answers "not a member") ----
constexpr u32 Bits = 17;
constexpr u32 Probe = 64;
enum State : u32 { Empty = 0, Pending = 1, Owned = 2, Excluded = 3, Gone = 4, Proposed = 5 }; ///< Pending: the hook is checking; Proposed: waiting for the raster thread
struct Slot {
    std::atomic<u64> id{0};
    std::atomic<u32> state{Empty};
    std::atomic<u64> begin{0}, end{0}; ///< the buffer's whole memory (set before the state becomes Owned)
};
Slot* Table() noexcept {
    static const std::unique_ptr<Slot[]> table{new Slot[std::size_t{1} << Bits]};
    return table.get();
}
u32 Home(u64 id) noexcept {
    return static_cast<u32>((id * 0x9E3779B97F4A7C15ull) >> (64 - Bits));
}
Slot* Find(u64 id) noexcept {
    Slot* t = Table();
    const u32 h = Home(id);
    for (u32 i = 0; i < Probe; ++i) {
        Slot& s = t[(h + i) & ((1u << Bits) - 1)];
        const u64 k = s.id.load(std::memory_order_acquire);
        if (k == id) {
            return &s;
        }
        if (k == 0) {
            return nullptr;
        }
    }
    return nullptr;
}
/// The slot of `id`, inserted with `initial` when absent (`inserted` true). nullptr: the probe is full.
Slot* FindOrInsert(u64 id, u32 initial, bool& inserted) noexcept {
    inserted = false;
    Slot* t = Table();
    const u32 h = Home(id);
    for (u32 i = 0; i < Probe; ++i) {
        Slot& s = t[(h + i) & ((1u << Bits) - 1)];
        u64 k = s.id.load(std::memory_order_acquire);
        if (k == 0) {
            u64 expected = 0;
            if (s.id.compare_exchange_strong(expected, id, std::memory_order_acq_rel)) {
                u32 empty = Empty;
                s.state.compare_exchange_strong(empty, initial, std::memory_order_acq_rel); // (an exclusion that found the slot first wins)
                inserted = true;
                return &s;
            }
            k = expected;
        }
        if (k == id) {
            return &s;
        }
    }
    return nullptr;
}

enum HookCount : u32 {
    HkNotClass,          ///< writes of tags 2 and 12 with no single buffer named
    HkOwnedWrites,       ///< writes of a member: the real mode does not announce them (by tag below)
    HkOwnedOutside,      ///< a member's write outside its recorded range (announced)
    HkExcludedWrites,    ///< writes of an excluded or deleted buffer (announced)
    HkPendingWrites,     ///< writes while another thread was joining the buffer (announced)
    HkJoins,             ///< buffers joined (their first write is announced, marked join)
    HkJoinNotInside,     ///< the write is not inside the buffer it names (announced, not joined)
    HkJoinTexture,       ///< the buffer's memory holds a texture: excluded
    HkJoinLost,          ///< an exclusion came between the join's check and its decision
    HkTableFull,         ///< the probe was full (announced)
    HkExcludedByTexture, ///< members excluded by a texture created over them
    HkExcludeCalls,      ///< buffers a texture creation named
    HkGone,              ///< named buffers deleted
    HkAwaiting,          ///< writes of a proposed buffer before the raster thread confirmed it (announced)
    HkCount
};
std::array<std::atomic<u64>, HkCount> hk{};
std::array<std::atomic<u64>, 16> hk_owned_by_tag{};

// ---- raster side ----
std::mutex rmutex;
std::unordered_map<u64, std::pair<u64, u64>> members; // id -> [begin, end)
std::map<u64, std::pair<u64, u64>> by_address;        // begin -> (end, id)
u64 max_len = 0;
std::atomic<u64> member_count{0};

enum RasterCount : u32 {
    RcJoins,                ///< members confirmed at their join apply
    RcJoinNoRange,          ///< a join apply whose buffer had no recorded range
    RcRefusedImage,         ///< joins refused: an image or a waiting clear is in the buffer's memory
    RcRefusedGpu,           ///< joins refused: GPU-written bytes are in it
    RcRefusedUnknown,       ///< joins refused: the buffer's image list (B8.5) holds a live image, or the index does not know the buffer
    RcConfirmLost,          ///< confirmed, but the hook had excluded the buffer meanwhile
    RcOwnedApplies,         ///< applies of writes the real mode would not announce (member, mark owned)
    RcOwnedImageTouched,    ///< ... whose image step touched an image or a waiting clear (MUST_BE_0)
    RcOwnedMemoErased,      ///< ... whose memo erase removed an entry (the real mode never makes that entry)
    RcOwnedNotMember,       ///< mark owned but the raster thread had removed the member (a texture or delete event came first in batch order)
    RcPlainOfMember,        ///< an announced (unmarked) write of a member
    RcLeftTexture,          ///< members removed by a texture created over them
    RcLeftDelete,           ///< members removed by their deletion
    RcGpuWriteOverMember,   ///< a GPU-written request over a member (MUST_BE_0)
    RcImageOverMember,      ///< an image registered over a member (MUST_BE_0)
    RcRequests,             ///< read requests of the address path that meet a member (the real mode copies their member bytes each time)
    RcRequestMemberBytes,   ///< ... the member bytes they cover
    RcCount
};
std::array<std::atomic<u64>, RcCount> rc{};
thread_local bool t_captured = false;

bool IntersectsLocked(u64 address, u64 size, u64* overlap = nullptr) {
    auto it = by_address.lower_bound(address >= max_len ? address - max_len : 0);
    bool any = false;
    const u64 end = address + size;
    for (; it != by_address.end() && it->first < end; ++it) {
        if (it->second.first > address) {
            any = true;
            if (overlap == nullptr) {
                return true;
            }
            *overlap += std::min(end, it->second.first) - std::max(address, it->first);
        }
    }
    return any;
}

void RemoveLocked(u64 id) {
    const auto it = members.find(id);
    if (it == members.end()) {
        return;
    }
    if (const auto at = by_address.find(it->second.first); at != by_address.end() && at->second.second == id) {
        by_address.erase(at);
    }
    members.erase(it);
    member_count.store(members.size(), std::memory_order_relaxed);
}

void LogFew(std::atomic<u32>& logged, const char* what, u64 address, u64 size) {
    if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
        LOG_ERROR(Render_Vulkan, "hook owned: {} {:#x}+{:#x}", what, address, size);
    }
}

} // namespace

u8 OnWrite(u32 tag, u64 id, u64 address, u64 size) noexcept {
    if ((tag != 2 && tag != 12) || id == 0 || id == Core::NativeUpdates::kBuffersOnly || size == 0) {
        if (tag == 2 || tag == 12) {
            hk[HkNotClass].fetch_add(1, std::memory_order_relaxed);
        }
        return MarkNone;
    }
    const auto known = [&](Slot& s) -> u8 {
        switch (s.state.load(std::memory_order_acquire)) {
        case Owned:
            if (address >= s.begin.load(std::memory_order_relaxed) && address + size <= s.end.load(std::memory_order_relaxed)) {
                hk[HkOwnedWrites].fetch_add(1, std::memory_order_relaxed);
                hk_owned_by_tag[tag & 15].fetch_add(1, std::memory_order_relaxed);
                return MarkOwned;
            }
            hk[HkOwnedOutside].fetch_add(1, std::memory_order_relaxed);
            return MarkNone;
        case Excluded:
        case Gone:
            hk[HkExcludedWrites].fetch_add(1, std::memory_order_relaxed);
            return MarkNone;
        case Proposed:
            hk[HkAwaiting].fetch_add(1, std::memory_order_relaxed);
            return MarkNone;
        default:
            hk[HkPendingWrites].fetch_add(1, std::memory_order_relaxed);
            return MarkNone;
        }
    };
    if (Slot* s = Find(id)) {
        return known(*s);
    }
    // the join: the write lies inside the buffer it names, and no texture is in the buffer's memory
    NativeDevice::BufferPart part{};
    if (!NativeDevice::BufferPartOf(id, address, size, part) || part.begin != address || part.end != address + size) {
        hk[HkJoinNotInside].fetch_add(1, std::memory_order_relaxed);
        return MarkNone;
    }
    bool inserted = false;
    Slot* s = FindOrInsert(id, Pending, inserted);
    if (s == nullptr) {
        hk[HkTableFull].fetch_add(1, std::memory_order_relaxed);
        return MarkNone;
    }
    if (!inserted) {
        return known(*s);
    }
    s->begin.store(part.object_begin, std::memory_order_relaxed);
    s->end.store(part.object_end, std::memory_order_relaxed);
    std::vector<u64> textures;
    (void)NativeDevice::TexturesIn(part.object_begin, part.object_end - part.object_begin, textures);
    if (!textures.empty()) {
        u32 pending = Pending;
        s->state.compare_exchange_strong(pending, Excluded, std::memory_order_acq_rel);
        hk[HkJoinTexture].fetch_add(1, std::memory_order_relaxed);
        return MarkNone;
    }
    u32 pending = Pending;
    if (!s->state.compare_exchange_strong(pending, Proposed, std::memory_order_acq_rel)) {
        hk[HkJoinLost].fetch_add(1, std::memory_order_relaxed);
        return MarkNone;
    }
    hk[HkJoins].fetch_add(1, std::memory_order_relaxed);
    return MarkJoin;
}

void ExcludeBuffer(u64 id) noexcept {
    if (Mode() == 0 || id == 0) {
        return;
    }
    hk[HkExcludeCalls].fetch_add(1, std::memory_order_relaxed);
    bool inserted = false;
    Slot* s = FindOrInsert(id, Excluded, inserted);
    if (s == nullptr || inserted) {
        return;
    }
    u32 st = s->state.load(std::memory_order_acquire);
    while (st != Excluded && st != Gone) {
        if (s->state.compare_exchange_weak(st, Excluded, std::memory_order_acq_rel)) {
            if (st == Owned) {
                hk[HkExcludedByTexture].fetch_add(1, std::memory_order_relaxed);
            }
            break;
        }
    }
}

void BufferGone(u64 id) noexcept {
    if (Mode() == 0 || id == 0) {
        return;
    }
    if (Slot* s = Find(id)) {
        s->state.store(Gone, std::memory_order_release);
        hk[HkGone].fetch_add(1, std::memory_order_relaxed);
    }
}

bool ApplyBegin(u64 object, u64 address, u64 size, u64& join_begin, u64& join_end) noexcept {
    join_begin = join_end = 0;
    const u8 mark = t_mark;
    t_captured = false;
    if (mark == MarkNone && member_count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::scoped_lock lock{rmutex};
    if (mark == MarkJoin) {
        if (!members.contains(object)) {
            const Slot* s = Find(object);
            const u64 b = s != nullptr ? s->begin.load(std::memory_order_relaxed) : 0, e = s != nullptr ? s->end.load(std::memory_order_relaxed) : 0;
            if (e <= b) {
                rc[RcJoinNoRange].fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            join_begin = b;
            join_end = e;
        }
        return false;
    }
    if (mark == MarkOwned) {
        if (!members.contains(object)) {
            rc[RcOwnedNotMember].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        rc[RcOwnedApplies].fetch_add(1, std::memory_order_relaxed);
        t_captured = true;
        return true;
    }
    if (object != 0 && members.contains(object)) {
        rc[RcPlainOfMember].fetch_add(1, std::memory_order_relaxed);
    }
    return false;
}

void ApplyEnd(u32 effects) noexcept {
    if (!t_captured) {
        return;
    }
    t_captured = false;
    if ((effects & N0::EffImage) != 0) {
        rc[RcOwnedImageTouched].fetch_add(1, std::memory_order_relaxed);
    }
    if ((effects & N0::EffMemo) != 0) {
        rc[RcOwnedMemoErased].fetch_add(1, std::memory_order_relaxed);
    }
}

void Confirm(u64 id, u64 begin, u64 end, u32 refuse) noexcept {
    Slot* s = Find(id);
    if (s == nullptr) {
        return;
    }
    if (refuse != RefuseNone) {
        s->state.store(Excluded, std::memory_order_release);
        rc[refuse == RefuseImage ? RcRefusedImage : refuse == RefuseGpuWritten ? RcRefusedGpu : RcRefusedUnknown].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    u32 proposed = Proposed;
    if (!s->state.compare_exchange_strong(proposed, Owned, std::memory_order_acq_rel)) {
        rc[RcConfirmLost].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::scoped_lock lock{rmutex};
    members[id] = {begin, end};
    by_address[begin] = {end, id};
    max_len = std::max(max_len, end - begin);
    member_count.store(members.size(), std::memory_order_relaxed);
    rc[RcJoins].fetch_add(1, std::memory_order_relaxed);
}

void Event(u32 what, u64 id, u64 address, u64 bytes) noexcept {
    if (member_count.load(std::memory_order_relaxed) == 0 || (what != 2 && what != 4)) {
        return;
    }
    std::scoped_lock lock{rmutex};
    if (what == 4) {
        if (members.contains(id)) {
            RemoveLocked(id);
            rc[RcLeftDelete].fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    const u64 end = address + std::max<u64>(bytes, 1);
    std::vector<u64> gone;
    for (auto it = by_address.lower_bound(address >= max_len ? address - max_len : 0); it != by_address.end() && it->first < end; ++it) {
        if (it->second.first > address) {
            gone.push_back(it->second.second);
        }
    }
    for (const u64 g : gone) {
        RemoveLocked(g);
        rc[RcLeftTexture].fetch_add(1, std::memory_order_relaxed);
    }
}

void GpuWrite(u64 address, u64 size) noexcept {
    if (member_count.load(std::memory_order_relaxed) == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{rmutex};
    if (IntersectsLocked(address, size)) {
        rc[RcGpuWriteOverMember].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        LogFew(logged, "a GPU-written request over a member", address, size);
    }
}

void ImageRegistered(u64 address, u64 size) noexcept {
    if (member_count.load(std::memory_order_relaxed) == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{rmutex};
    if (IntersectsLocked(address, size)) {
        rc[RcImageOverMember].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        LogFew(logged, "an image registered over a member", address, size);
    }
}

void Request(u64 address, u64 size) noexcept {
    if (member_count.load(std::memory_order_relaxed) == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{rmutex};
    u64 overlap = 0;
    if (IntersectsLocked(address, size, &overlap)) {
        rc[RcRequests].fetch_add(1, std::memory_order_relaxed);
        rc[RcRequestMemberBytes].fetch_add(overlap, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    const auto h = [](HookCount c) { return hk[c].load(std::memory_order_relaxed); };
    const auto r = [](RasterCount c) { return rc[c].load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["mode"] = Mode();
    j["hook"] = {{"writes_not_announced_by_real_mode", h(HkOwnedWrites)},
                 {"of_them_light_constants_tag_2", hk_owned_by_tag[2].load()},
                 {"of_them_scene_slots_tag_12", hk_owned_by_tag[12].load()},
                 {"joins_proposed", h(HkJoins)},
                 {"writes_awaiting_confirmation", h(HkAwaiting)},
                 {"join_refused_texture_in_memory", h(HkJoinTexture)},
                 {"join_refused_not_inside_buffer", h(HkJoinNotInside)},
                 {"join_lost_to_exclusion", h(HkJoinLost)},
                 {"member_write_outside_range", h(HkOwnedOutside)},
                 {"writes_of_excluded_or_deleted", h(HkExcludedWrites)},
                 {"writes_while_joining", h(HkPendingWrites)},
                 {"writes_naming_no_buffer", h(HkNotClass)},
                 {"probe_full", h(HkTableFull)},
                 {"buffers_named_by_texture_creation", h(HkExcludeCalls)},
                 {"members_excluded_by_texture", h(HkExcludedByTexture)},
                 {"named_buffers_deleted", h(HkGone)}};
    j["raster"] = {{"members_now", member_count.load()},
                   {"joins", r(RcJoins)},
                   {"join_without_range", r(RcJoinNoRange)},
                   {"applies_of_unannounced_writes", r(RcOwnedApplies)},
                   {"WOULD_SKIP_BUT_IMAGE_TOUCHED_MUST_BE_0", r(RcOwnedImageTouched)},
                   {"would_skip_memo_erased", r(RcOwnedMemoErased)},
                   {"owned_mark_after_member_left", r(RcOwnedNotMember)},
                   {"announced_write_of_member", r(RcPlainOfMember)},
                   {"left_by_texture_create", r(RcLeftTexture)},
                   {"left_by_delete", r(RcLeftDelete)},
                   {"joins_refused_image_or_clear_in_memory", r(RcRefusedImage)},
                   {"joins_refused_gpu_written", r(RcRefusedGpu)},
                   {"joins_refused_by_the_buffers_image_list", r(RcRefusedUnknown)},
                   {"confirm_lost_to_exclusion", r(RcConfirmLost)},
                   {"GPU_WRITE_OVER_HOOK_OWNED_MUST_BE_0", r(RcGpuWriteOverMember)},
                   {"IMAGE_OVER_HOOK_OWNED_MUST_BE_0", r(RcImageOverMember)},
                   {"address_requests_over_members", r(RcRequests)},
                   {"address_request_member_bytes", r(RcRequestMemberBytes)}};
    return j;
}

} // namespace Core::HookOwned
