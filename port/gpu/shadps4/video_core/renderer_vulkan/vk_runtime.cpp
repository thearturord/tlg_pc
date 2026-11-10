// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/half_phase.h"
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <map>
#include <deque>
#include <tuple>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "native/native_occlusion.h"
#include "native/native_config.h"
#include "native/native_registry.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "native/native_readback_policy.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "native/native_split_census.h"
#include "native/native_dispatch_split.h"

#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

namespace IndependentGroup {
namespace {
std::atomic<u64> g_queries{}, g_suppressed{}, g_aware_extra{}, g_dispatches{}, g_barrier_before{}, g_barrier_saved{}, g_other_cause{};
}
u32 Mode() {
    static const u32 mode = [] { const char* e = std::getenv("TLG_SIM_INDEPENDENT"); return e != nullptr ? static_cast<u32>(std::atoi(e)) : 2u; }(); // default 2 since 2026-11-09 (owner play good); =0 turns it off, =1 shadow
    return mode;
}
void NoteQuery(bool real, bool aware) {
    g_queries.fetch_add(1, std::memory_order_relaxed);
    g_suppressed.fetch_add(real && !aware ? 1 : 0, std::memory_order_relaxed);
    g_aware_extra.fetch_add(!real && aware ? 1 : 0, std::memory_order_relaxed);
}
void NoteDispatch(bool had_barrier, bool real_buffer_hazard, bool aware_buffer_hazard) {
    g_dispatches.fetch_add(1, std::memory_order_relaxed);
    g_barrier_before.fetch_add(had_barrier ? 1 : 0, std::memory_order_relaxed);
    // the barrier only existed for buffer hazards the group-aware answer does not have
    g_barrier_saved.fetch_add(had_barrier && real_buffer_hazard && !aware_buffer_hazard ? 1 : 0, std::memory_order_relaxed);
    g_other_cause.fetch_add(had_barrier && !real_buffer_hazard ? 1 : 0, std::memory_order_relaxed);
}
std::string Report() {
    nlohmann::json j = {{"mode", Mode()}, {"group_dispatches", g_dispatches.load()}, {"with_a_barrier_before_them", g_barrier_before.load()}, {"barriers_the_group_aware_answer_drops", g_barrier_saved.load()},
                        {"barriers_from_other_causes_(images)", g_other_cause.load()}, {"buffer_queries_in_group", g_queries.load()}, {"hazards_suppressed", g_suppressed.load()},
                        {"EXTRA_hazards_the_group_aware_answer_adds_(must_be_0)", g_aware_extra.load()}};
    return j.dump();
}
} // namespace IndependentGroup

namespace BarrierCensus {
namespace {
struct Pending {
    u64 key, begin, end, tag;
    bool write;
};
struct PairStats {
    u64 count{}, overlap{}, bound{}, full{};
};
struct ConsumerStats {
    u64 dispatches{}, with_barrier{}, buffer_hazard{}, prev_same{}, prev_same_barrier{}, pgm_hash{};
};
std::mutex g_mutex;
std::vector<Pending> g_pending;
std::map<std::tuple<u64, u64, u32>, PairStats> g_pairs; // (consumer, producer, kind 0 RAW 1 WAW 2 WAR)
std::unordered_map<u64, ConsumerStats> g_consumers;
u64 g_op_tag{}, g_prev_tag{}, g_overflow{}, g_queries_true_unmatched{};
bool g_op_hazard{};
constexpr size_t kPendingCap = 8192;
struct SimStats {
    u64 total{}, status[6]{}, from_ring{}, from_memory{};
    u64 pair_disjoint{}, pair_overlap{}, pair_first{}, pair_same_pool{}, window_same_pool{};
    u64 window_disjoint{}, window_overlap{};
    u64 count_sum{}, count_max{};
    u64 bytes_sum{};
};
std::unordered_map<u64, SimStats> g_sim;
SimStats g_sim_all;
struct SlotStats {
    u64 dispatches{}, unreadable{}, particles{}, equal_index{}, zero{}, dup_in_window{}, above_capacity{};
} g_slots;
std::deque<std::vector<u32>> g_slot_window;
std::map<std::tuple<u32, u64, bool>, u64> g_sim_bindings; // (position among the bound buffers, size, written) -> count
struct SimRecent {
    u64 key, begin, end;
};
std::deque<SimRecent> g_sim_recent; // the slices of the last 64 valid SimParticle dispatches
constexpr size_t kSimWindow = 64;
} // namespace

bool On() {
    static const bool on = [] { const char* e = std::getenv("TLG_BARRIER_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void SetOp(u64 tag) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    g_prev_tag = g_op_tag;
    g_op_tag = tag;
    g_op_hazard = false;
}

void Dispatch(u64 tag, bool barrier, u64 pgm_hash) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    auto& c = g_consumers[tag];
    c.pgm_hash = pgm_hash;
    ++c.dispatches;
    c.with_barrier += barrier ? 1 : 0;
    c.buffer_hazard += g_op_hazard ? 1 : 0;
    if (g_prev_tag == tag) {
        ++c.prev_same;
        c.prev_same_barrier += barrier ? 1 : 0;
    }
}

static void NoteAccess(u64 key, u64 begin, u64 size, bool write) {
    std::scoped_lock lock{g_mutex};
    if (g_pending.size() >= kPendingCap) {
        ++g_overflow;
        return;
    }
    g_pending.push_back({key, begin, begin + size, g_op_tag, write});
}

static void NoteHazard(u64 key, u64 begin, u64 size, bool new_access_writes) {
    std::scoped_lock lock{g_mutex};
    g_op_hazard = true;
    const u64 end = begin + size;
    u64 best = 0, best_overlap = 0;
    u32 best_kind = 0;
    bool found = false;
    for (const auto& p : g_pending) {
        if (p.key != key || p.begin >= end || p.end <= begin) {
            continue;
        }
        if (!p.write && !new_access_writes) {
            continue; // a pending read only conflicts with a write
        }
        const u64 overlap = std::min(p.end, end) - std::max(p.begin, begin);
        if (!found || overlap > best_overlap) {
            found = true;
            best = p.tag;
            best_overlap = overlap;
            best_kind = p.write ? (new_access_writes ? 1u : 0u) : 2u;
        }
    }
    if (!found) {
        ++g_queries_true_unmatched;
        return;
    }
    auto& s = g_pairs[{g_op_tag, best, best_kind}];
    ++s.count;
    s.overlap += std::min(best_overlap, size);
    s.bound += size;
    s.full += best_overlap * 10 >= size * 9 ? 1 : 0;
}

void SimSlice(u64 tag, u32 status, bool from_ring, u64 pool_key, u64 begin, u64 end, u32 count, u32 base) {
    if (!On()) {
        return;
    }
    (void)base;
    std::scoped_lock lock{g_mutex};
    for (SimStats* s : {&g_sim[tag], &g_sim_all}) {
        ++s->total;
        ++s->status[status < 6 ? status : 5];
        if (status == 0) {
            (from_ring ? s->from_ring : s->from_memory) += 1;
            s->count_sum += count;
            s->count_max = std::max<u64>(s->count_max, count);
            s->bytes_sum += end - begin;
        }
    }
    if (status != 0) {
        return;
    }
    // pair: against the previous valid SimParticle dispatch; window: against the last 64
    bool pair_overlap = false, window_overlap = false, pair_same = false, window_same = false;
    if (!g_sim_recent.empty()) {
        const auto& last = g_sim_recent.back();
        pair_same = last.key == pool_key;
        pair_overlap = pair_same && begin < last.end && last.begin < end;
    }
    for (const auto& r : g_sim_recent) {
        window_same |= r.key == pool_key;
        if (r.key == pool_key && begin < r.end && r.begin < end) {
            window_overlap = true;
            break;
        }
    }
    const bool first = g_sim_recent.empty();
    for (SimStats* s : {&g_sim[tag], &g_sim_all}) {
        if (first) {
            ++s->pair_first;
        } else {
            s->pair_same_pool += pair_same ? 1 : 0;
            s->window_same_pool += window_same ? 1 : 0;
            ++(pair_overlap ? s->pair_overlap : s->pair_disjoint);
            ++(window_overlap ? s->window_overlap : s->window_disjoint);
        }
    }
    g_sim_recent.push_back({pool_key, begin, end});
    if (g_sim_recent.size() > kSimWindow) {
        g_sim_recent.pop_front();
    }
}

void SimBinding(u64 tag, u32 position, u64 size, bool write) {
    (void)tag;
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    ++g_sim_bindings[{position, size, write}];
}

void SimSlots(const u32* slots, const u32* index, u32 n, bool memory_readable) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    ++g_slots.dispatches;
    if (!memory_readable) {
        ++g_slots.unreadable;
        return;
    }
    std::vector<u32> mine(slots, slots + n);
    for (u32 i = 0; i < n; ++i) {
        ++g_slots.particles;
        g_slots.equal_index += slots[i] == index[i] ? 1 : 0;
        g_slots.zero += slots[i] == 0 ? 1 : 0;
        g_slots.above_capacity += slots[i] >= 20480 ? 1 : 0;
        bool dup = false;
        for (const auto& w : g_slot_window) {
            for (const u32 s : w) {
                if (s == slots[i] && s != 0) {
                    dup = true;
                }
            }
            if (dup) {
                break;
            }
        }
        g_slots.dup_in_window += dup ? 1 : 0;
    }
    g_slot_window.push_back(std::move(mine));
    if (g_slot_window.size() > 16) {
        g_slot_window.pop_front();
    }
}

static void ClearPending() {
    std::scoped_lock lock{g_mutex};
    g_pending.clear();
}

std::string Report() {
    std::scoped_lock lock{g_mutex};
    nlohmann::json out = nlohmann::json::object();
    auto hex = [](u64 v) { char b[24]; std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v)); return std::string(b); };
    nlohmann::json consumers = nlohmann::json::array();
    std::vector<std::pair<u64, u64>> order;
    for (const auto& [tag, c] : g_consumers) {
        order.emplace_back(c.dispatches, tag);
    }
    std::sort(order.begin(), order.end(), std::greater<>());
    for (size_t i = 0; i < order.size() && i < 60; ++i) {
        const auto& c = g_consumers[order[i].second];
        consumers.push_back({{"program", hex(order[i].second)}, {"dispatches", c.dispatches}, {"with_barrier", c.with_barrier}, {"buffer_hazard", c.buffer_hazard}, {"previous_op_same_program", c.prev_same}, {"previous_same_and_barrier", c.prev_same_barrier}, {"pgm_hash", hex(c.pgm_hash)}});
    }
    nlohmann::json pairs = nlohmann::json::array();
    std::vector<std::pair<u64, std::tuple<u64, u64, u32>>> porder;
    for (const auto& [k, s] : g_pairs) {
        porder.emplace_back(s.count, k);
    }
    std::sort(porder.begin(), porder.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    static const char* kinds[3] = {"RAW", "WAW", "WAR"};
    for (size_t i = 0; i < porder.size() && i < 120; ++i) {
        const auto& k = porder[i].second;
        const auto& s = g_pairs[k];
        pairs.push_back({{"consumer", hex(std::get<0>(k))}, {"producer", hex(std::get<1>(k))}, {"kind", kinds[std::get<2>(k)]}, {"hazards", s.count}, {"overlap_bytes", s.overlap}, {"asked_bytes", s.bound}, {"overlap_covers_90pct_of_asked", s.full}});
    }
    {
        auto sim_json = [&](const SimStats& s) {
            return nlohmann::json{{"dispatches", s.total}, {"ok", s.status[0]}, {"no_control_buffer", s.status[1]}, {"control_buffer_too_short", s.status[2]}, {"count_zero", s.status[3]},
                                  {"slice_outside_pool", s.status[4]}, {"no_pool_buffer", s.status[5]}, {"control_from_ring_copy", s.from_ring}, {"control_from_guest_memory", s.from_memory},
                                  {"pair_disjoint", s.pair_disjoint}, {"pair_overlap", s.pair_overlap}, {"pair_same_pool_buffer", s.pair_same_pool}, {"window64_has_same_pool_buffer", s.window_same_pool}, {"window64_disjoint", s.window_disjoint}, {"window64_overlap", s.window_overlap},
                                  {"count_sum", s.count_sum}, {"count_max", s.count_max}, {"slice_bytes_sum", s.bytes_sum}};
        };
        nlohmann::json sim = nlohmann::json::object();
        sim["all"] = sim_json(g_sim_all);
        nlohmann::json per = nlohmann::json::object();
        for (const auto& [tag, s] : g_sim) {
            per[hex(tag)] = sim_json(s);
        }
        sim["by_program"] = std::move(per);
        sim["render_slots"] = {{"dispatches", g_slots.dispatches}, {"memory_unreadable", g_slots.unreadable}, {"particles_read", g_slots.particles}, {"slot_equals_particle_index", g_slots.equal_index},
                               {"slot_zero", g_slots.zero}, {"slot_repeats_in_last_16_dispatches", g_slots.dup_in_window}, {"slot_at_or_above_20480", g_slots.above_capacity}};
        nlohmann::json bindings = nlohmann::json::array();
        for (const auto& [k, n] : g_sim_bindings) {
            bindings.push_back({{"position", std::get<0>(k)}, {"size", std::get<1>(k)}, {"written", std::get<2>(k)}, {"count", n}});
        }
        sim["bound_buffers"] = std::move(bindings);
        out["sim_particle_slices"] = std::move(sim);
    }
    out["consumers"] = std::move(consumers);
    out["hazard_pairs"] = std::move(pairs);
    out["pending_overflow"] = g_overflow;
    out["hazards_without_a_matching_pending_entry"] = g_queries_true_unmatched;
    out["note"] = "Phase 0b, cumulative since start. Tag: program address of a compute op, 4000000000000000 = any graphics op, 0 = none. Overlap is on the BOUND ranges, not on the bytes the shader touches.";
    return out.dump();
}
} // namespace BarrierCensus

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("Unexpected image type {}", u32(type));
    }
}

static std::pair<u32, u32> SanitizeCopyLayers(const VideoCore::ImageInfo& src_info,
                                              const VideoCore::ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D source layers {} to 1.", src_layers);
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D destination layers {} to 1.", dst_layers);
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy source layers {} and destination layers {} to minimum.",
                        src_layers, dst_layers);
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D source layers {} to 3D destination depth {}", src_layers,
                        depth);
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D destination layers {} to 3D source depth {}", dst_layers,
                        depth);
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

static u32 BufferImageCopySize(const vk::BufferImageCopy& copy, const vk::Format pixel_format) {
    const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
    const u32 height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;

    const auto block = vk::blockExtent(pixel_format);
    const u32 block_size = vk::blockSize(pixel_format);
    const u32 row_pitch = (row_length / block[0]) * block_size;
    const u32 slice_pitch = (height / block[1]) * row_pitch;

    const u32 width_in_blocks = (copy.imageExtent.width + block[0] - 1) / block[0];
    const u32 height_in_blocks = (copy.imageExtent.height + block[1] - 1) / block[1];
    const u32 num_slices = copy.imageExtent.depth * copy.imageSubresource.layerCount;

    return (num_slices - 1) * slice_pitch + (height_in_blocks - 1) * row_pitch +
           width_in_blocks * block_size;
}

Runtime::Runtime(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, staging_pool{instance_, scheduler_} {
    blit_helper = std::make_unique<VideoCore::BlitHelper>(instance, scheduler);

    memory_barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    memory_barrier.dstAccessMask =
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    // C8 (2026-10-28): the planner always decides; the tracker runs next to it and is only compared
    live_plan_on = true;
    live_plan_decides = true;
    // B10: buffer hazards by object (no switch since the cleanup of 2026-10-30)
    b10_mode = 2;
}

namespace {
/// the planner's list rule: the same as the tracker's bounded lists (above 64 intervals one covering range replaces them, which can only add a barrier)
template <typename List>
void LivePlanAdd(List& list, u64 start, u64 end) {
    constexpr size_t MaxIntervals = 64;
    if (list.Contains(start, end)) {
        return;
    }
    if (list.Size() >= MaxIntervals) {
        u64 lo = start, hi = end;
        for (const auto& interval : list) {
            lo = std::min(lo, interval.start);
            hi = std::max(hi, interval.end);
        }
        list.Clear();
        list.Add({.start = lo, .end = hi});
        return;
    }
    list.Add({.start = start, .end = end});
}
} // namespace

vk::ImageLayout Runtime::CurrentLayout(const VideoCore::Image& image) const {
    if (live_plan_decides && image.backing) {
        if (const auto it = image_plan.find(ImagePlanKey(image, image.backing->num_samples)); it != image_plan.end()) {
            return it->second.state.layout;
        }
    }
    return image.backing ? image.backing->state.layout : vk::ImageLayout::eUndefined;
}

void Runtime::PlanImageBarriers(VideoCore::Image* image, VideoCore::Image::Barriers& barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                                vk::PipelineStageFlags2 dst_stage, std::optional<VideoCore::SubresourceRange> subres_range) {
    if (!image->backing) {
        return;
    }
    // 2026-11-07 (gpu-ordering-plan step 2): a depth image is written by its render pass after the transit that made it an attachment; the planner's state does not know that,
    // so a later layout transition got the stale source (shader read) and the layer reported WRITE_AFTER_WRITE. Every barrier that leaves a depth image waits for the depth writes.
    // TLG_DEPTH_BARRIER_SRC=0 turns it off.
    static const bool widen_depth_source = [] { const char* e = std::getenv("TLG_DEPTH_BARRIER_SRC"); return e == nullptr || e[0] != '0'; }();
    const size_t first_new_barrier = barriers.size();
    struct WidenOnExit {
        VideoCore::Image::Barriers& barriers;
        size_t first;
        bool on;
        ~WidenOnExit() {
            if (!on) {
                return;
            }
            for (size_t i = first; i < barriers.size(); ++i) {
                barriers[i].srcStageMask |= vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
                barriers[i].srcAccessMask |= vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            }
        }
    } widen_guard{barriers, first_new_barrier, widen_depth_source && image->info.props.is_depth};
    const auto [it, created] = image_plan.try_emplace(ImagePlanKey(*image, image->backing->num_samples));
    live_stats.images_planned += created ? 1 : 0;
    auto& last_state = it->second.state;
    auto& subresource_states = it->second.subresources;
    const auto& info = image->info;
    constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
    const bool needs_partial_transition =
        subres_range && (subres_range->base != VideoCore::SubresourceBase{} || subres_range->extent != info.resources);
    const bool partially_transited = !subresource_states.empty();
    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(info.resources.levels * info.resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
        }
        const u32 start_mip = needs_partial_transition ? subres_range->base.level : 0u;
        const u32 end_mip = needs_partial_transition ? subres_range->base.level + subres_range->extent.levels : info.resources.levels;
        const u32 start_layer = needs_partial_transition ? subres_range->base.layer : 0u;
        const u32 end_layer = needs_partial_transition ? subres_range->base.layer + subres_range->extent.layers : info.resources.layers;
        for (u32 mip = start_mip; mip < end_mip; ++mip) {
            for (u32 layer = start_layer; layer < end_layer; ++layer) {
                const auto subres_idx = mip * info.resources.layers + layer;
                if (subres_idx >= subresource_states.size()) {
                    continue;
                }
                auto& state = subresource_states[subres_idx];
                const bool is_write = static_cast<bool>(state.access_mask & write_flags);
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = image->GetImage(),
                        .subresourceRange{
                            .aspectMask = image->aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        },
                    });
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = dst_stage;
                }
            }
        }
        if (!needs_partial_transition) {
            subresource_states.clear();
        }
    } else {
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) {
            return;
        }
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->GetImage(),
            .subresourceRange{
                .aspectMask = image->aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }
    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    last_state.pl_stage = dst_stage;
}

u64 Runtime::LiveObjectSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin) {
    if (handle->CpuAddr() != 0) {
        // B10: a placed engine buffer (or block) is its own Vulkan buffer: its hazards are its own, in its own offset space
        ++live_stats.object_space_queries;
        const_cast<VideoCore::Buffer*>(handle)->NoteGpuUse(scheduler.CurrentTick()); // zero copy: the CPU writes it directly only once this is finished
        const u64 oracle_start = Core::DispatchSplit::t_planner_sampled ? Core::DispatchSplit::Tsc() : 0;
        struct OracleTicks {
            u64 start;
            ~OracleTicks() {
                if (start != 0) {
                    Core::DispatchSplit::t_oracle_ticks += Core::DispatchSplit::Tsc() - start;
                }
            }
        };
        {
            const OracleTicks oracle_ticks{oracle_start};
            // P3c oracle: a buffer the record side uses for an operation was asked for by the resolve side in that operation (or the one before: bound buffers released at its end)
            static std::atomic<u64> uses{0}, unannounced{0}, never{0};
            const u64 seq = VideoCore::OpSequence::recording != 0 ? VideoCore::OpSequence::recording : VideoCore::OpSequence::resolving.load(std::memory_order_relaxed);
            const u64 n = uses.fetch_add(1, std::memory_order_relaxed) + 1;
            if (handle->resolve_use_seq != seq && handle->resolve_prev_seq != seq && VideoCore::RecordQueue::Lag() == 0) {
                const u64 k = unannounced.fetch_add(1, std::memory_order_relaxed);
                static const bool split_only = [] { const char* e = std::getenv("TLG_SPLIT_BIND"); return e != nullptr && std::atoi(e) == 1; }();
                if (split_only && k < 5) {
                    // mode 1 is the real split path: here a use the resolve side did not announce is an error (mode 2 runs the old code too, which announces nothing)
                    LOG_ERROR(Render_Vulkan, "split bind: record side used buffer {:#x}+{:#x} that the resolve side did not announce in operation {} (last asked {}), rva: {}", handle->CpuAddr(),
                              handle->SizeBytes(), seq, handle->resolve_use_seq, Core::NativeOcclusion::CallerRvas(1, 7));
                }
                if (handle->resolve_use_seq == 0) {
                    never.fetch_add(1, std::memory_order_relaxed);
                }
                // 2026-11-09: the stack capture below walked the stack and took a mutex for every unannounced use (6.5 % of the raster thread's samples in the spears sample); it is a census, off unless TLG_P3C_STACKS=1
                static const bool stacks_on = [] { const char* e = std::getenv("TLG_P3C_STACKS"); return e != nullptr && e[0] == '1'; }();
                if (stacks_on) {
                    // which code paths: the stack (frames 1 to 7) counted, the top ones printed at the powers of two
                    static std::mutex stacks_mutex;
                    static std::unordered_map<std::string, std::pair<u64, u64>> stacks; // stack -> (uses, of them never asked)
                    std::scoped_lock lock{stacks_mutex};
                    auto& entry = stacks[Core::NativeOcclusion::CallerRvas(1, 7)];
                    ++entry.first;
                    entry.second += handle->resolve_use_seq == 0 ? 1 : 0;
                    if ((k + 1 >= 4096) && ((k + 1) & k) == 0) {
                        std::vector<std::pair<u64, const std::string*>> order;
                        for (const auto& [text, counts] : stacks) {
                            order.emplace_back(counts.first, &text);
                        }
                        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                        for (size_t i = 0; i < order.size() && i < 8; ++i) {
                            LOG_WARNING(Render_Vulkan, "zero copy census: unannounced x{} ({} never asked) rva: {}", order[i].first, stacks[*order[i].second].second, *order[i].second);
                        }
                    }
                }
            }
            if ((n & (n - 1)) == 0 && n >= 1024) {
                LOG_WARNING(Render_Vulkan, "zero copy census: {} record side uses of placed buffers, {} of them not asked for by the resolve side in that operation ({} never asked at all)", n, unannounced.load(), never.load());
            }
        }
        begin = offset;
        return reinterpret_cast<u64>(handle);
    }
    return LivePlanSpace(handle, offset, begin);
}

u64 Runtime::LivePlanSpace(const VideoCore::Buffer* handle, u64 offset, u64& begin) {
    const u64 base = handle->CpuAddr();
    const u64 size = handle->SizeBytes();
    if (handle == plan_last_handle && base == plan_last_base && size == plan_last_size) {
        begin = base != 0 ? base + offset : offset;
        return base != 0 ? 0 : reinterpret_cast<u64>(handle);
    }
    plan_last_handle = handle;
    plan_last_base = base;
    plan_last_size = size;
    auto& seen = live_seen[reinterpret_cast<u64>(handle)];
    if (seen.base != base || seen.size != size) {
        // the buffer cache moved or re-created the buffer: both guest ranges count as read and written by an unknown access (the tracker keeps lists in the buffer's own offset space)
        if (seen.size != 0) {
            ++live_stats.rebases;
            auto& space = live_plan.buffers[0];
            ++live_plan_writes;
            if (seen.base != 0) {
                LivePlanAdd(space.write, seen.base, seen.base + seen.size);
                LivePlanAdd(space.read, seen.base, seen.base + seen.size);
            }
            if (base != 0) {
                LivePlanAdd(space.write, base, base + size);
                LivePlanAdd(space.read, base, base + size);
            }
        }
        seen = {base, size};
    }
    begin = base != 0 ? base + offset : offset;
    return base != 0 ? 0 : reinterpret_cast<u64>(handle);
}

void Runtime::TickFrame() {
    staging_pool.TickFrame();
}

void Runtime::CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                         std::span<const vk::BufferCopy> copies) {
    scheduler.EndRendering();

    bool needs_flush{};
    for (const auto& copy : copies) {
        needs_flush |= IsBufferAccessed(src, copy.srcOffset, copy.size);
        needs_flush |= IsBufferAccessed(dst, copy.dstOffset, copy.size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(src->Handle(), dst->Handle(), copies);

    for (const auto& copy : copies) {
        AccessBuffer(src, copy.srcOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
        AccessBuffer(dst, copy.dstOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value) {
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, size, true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.fillBuffer(dst->Handle(), offset, size, value);

    AccessBuffer(dst, offset, size, vk::PipelineStageFlagBits2::eClear,
                 vk::AccessFlagBits2::eTransferWrite);
}

void Runtime::InlineData(VideoCore::Buffer* dst, u64 offset, u32 value) {
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, sizeof(value), true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.updateBuffer(dst->Handle(), offset, sizeof(value), &value);

    AccessBuffer(dst, offset, sizeof(value), vk::PipelineStageFlagBits2::eCopy,
                 vk::AccessFlagBits2::eTransferWrite);
}

bool Runtime::Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                      vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                      std::optional<VideoCore::SubresourceRange> subres_range) {
    VideoCore::RecordQueue::Guard(VideoCore::RecordQueue::Transit);
    if (VideoCore::HalfPhase::current == VideoCore::HalfPhase::Resolve) {
        VideoCore::ResolveCommands::Run(); // P3d: what was queued comes first
        Core::NativeOcclusion::NoteCrossing("Transit (layout state) in a resolve half");
    }
    const size_t prev_num_barriers = static_cast<size_t>(image_barriers.size());
    {
        // 2026-11-07 GPU ordering plan step 1 (TLG_TRACE_DISPLAY_LAYOUT=1): who moves the 4K colour images, in call order, next to the validation layer's own messages in the same log
        static const bool trace = [] { const char* e = std::getenv("TLG_TRACE_DISPLAY_LAYOUT"); return e != nullptr && e[0] == '1'; }();
        static std::atomic<u32> traced{0};
        if (trace && image->info.size.width == 3840 && !image->info.props.is_depth && image->backing && traced.load(std::memory_order_relaxed) < 600) {
            traced.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "layout trace: image {:#x} {} -> {} (planner), caller {:#x}, tick {}, rendering {}", image->info.guest_address, vk::to_string(CurrentLayout(*image)),
                        vk::to_string(dst_layout), reinterpret_cast<u64>(__builtin_return_address(0)), scheduler.CurrentTick(), scheduler.IsRendering());
        }
    }
    // L4, images: the planner's barriers (the image's own state is not consulted)
    VideoCore::Image::Barriers planned;
    PlanImageBarriers(image, planned, dst_layout, dst_access, dst_stage, subres_range);
    {
        static const bool trace = [] { const char* e = std::getenv("TLG_TRACE_DISPLAY_LAYOUT"); return e != nullptr && e[0] == '1'; }();
        static std::atomic<u32> traced{0};
        if (trace && image->info.size.width == 3840 && !image->info.props.is_depth && image->backing && traced.load(std::memory_order_relaxed) < 1500) {
            traced.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "layout trace: image {:#x} planned {} barrier(s), partial range {}, pending queue {}{}", image->info.guest_address, planned.size(), subres_range.has_value(),
                        image_barriers.size(), planned.empty() ? "" : [&] {
                            std::string text;
                            for (const auto& b : planned) {
                                text += fmt::format(" [{} -> {} src {}/{} dst {}/{} levels {} layers {}]", vk::to_string(b.oldLayout), vk::to_string(b.newLayout), vk::to_string(b.srcStageMask),
                                                    vk::to_string(b.srcAccessMask), vk::to_string(b.dstStageMask), vk::to_string(b.dstAccessMask), b.subresourceRange.levelCount, b.subresourceRange.layerCount);
                            }
                            return text;
                        }());
        }
    }
    ++live_stats.image_transits;
    live_stats.image_barriers += planned.size();
    for (const auto& barrier : planned) {
        image_barriers.push_back(barrier);
    }
    return image_barriers.size() != prev_num_barriers;
}

void Runtime::UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                          std::span<const vk::BufferImageCopy> upload_copies) {
    SetBackingSamples(dst, dst->info.num_samples, false);
    scheduler.EndRendering();

    bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite);
    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        needs_flush |= IsBufferAccessed(src, copy.bufferOffset, copy_size);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBufferToImage(src->Handle(), dst->GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             upload_copies);

    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        AccessBuffer(src, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
    }

    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferImageCopy> download_copies) {
    SetBackingSamples(src, src->info.num_samples);
    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        needs_flush |= IsBufferAccessed(dst, copy.bufferOffset, copy_size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->Handle(),
                             download_copies);

    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        AccessBuffer(dst, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::CopyImage(VideoCore::Image* src, VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src->info.pixel_format != dst->info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src->info.pixel_format), vk::to_string(dst->info.pixel_format));
    }

    const u32 base_width = src->info.size.width;
    const u32 base_height = src->info.size.height;
    const u32 base_depth =
        dst->info.type == AmdGpu::ImageType::Color3D ? dst->info.size.depth : src->info.size.depth;

    // Match sample count before copying
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    SmallVector<vk::ImageCopy, 8> regions;

    const vk::ImageAspectFlags src_aspect = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    const vk::ImageAspectFlags dst_aspect = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const bool src_is_2d = ConvertImageType(src->info.type) == vk::ImageType::e2D;
    const bool src_is_3d = ConvertImageType(src->info.type) == vk::ImageType::e3D;

    const bool dst_is_2d = ConvertImageType(dst->info.type) == vk::ImageType::e2D;
    const bool dst_is_3d = ConvertImageType(dst->info.type) == vk::ImageType::e3D;

    const bool is_2d_to_3d = src_is_2d && dst_is_3d;
    const bool is_3d_to_2d = src_is_3d && dst_is_2d;
    const bool is_same_type = !is_2d_to_3d && !is_3d_to_2d;

    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, mip_d);

        vk::ImageCopy region{};
        region.srcSubresource.aspectMask = src_aspect;
        region.srcSubresource.mipLevel = mip;
        region.srcSubresource.baseArrayLayer = 0;
        region.dstSubresource.aspectMask = dst_aspect;
        region.dstSubresource.mipLevel = mip;
        region.dstSubresource.baseArrayLayer = 0;

        if (is_same_type) {
            // 2D->2D OR 3D->3D
            if (src_is_3d) {
                // 3D images must use layerCount=1
                region.srcSubresource.layerCount = 1;
                region.dstSubresource.layerCount = 1;
                region.extent = vk::Extent3D(mip_w, mip_h, mip_d);
            } else {
                // Array images
                const u32 copy_layers = std::min(src_layers, dst_layers);
                region.srcSubresource.layerCount = copy_layers;
                region.dstSubresource.layerCount = copy_layers;
                region.extent = vk::Extent3D(mip_w, mip_h, 1);
            }
        } else if (is_2d_to_3d) {
            // 2D array -> 3D volume
            region.srcSubresource.layerCount = src_layers;
            region.dstSubresource.layerCount = 1;
            region.extent = vk::Extent3D(mip_w, mip_h, src_layers);
        } else if (is_3d_to_2d) {
            // 3D volume -> 2D array
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.layerCount = dst_layers;
            region.extent = vk::Extent3D(mip_w, mip_h, dst_layers);
        }

        regions.push_back(region);
    }

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, regions);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                  const VideoCore::Buffer* buffer, u64 offset) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);
    ASSERT(num_mips == 1);

    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    vk::BufferImageCopy buffer_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
            .mipLevel = 0u,
            .baseArrayLayer = 0,
            .layerCount = num_layers,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };
    const auto copy_size = BufferImageCopySize(buffer_copy, src->info.pixel_format);

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    needs_flush |= IsBufferAccessed(buffer, offset, copy_size);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             buffer->Handle(), buffer_copy);

    const vk::MemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &post_copy_barrier,
    });

    buffer_copy.imageSubresource.aspectMask = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    cmdbuf.copyBufferToImage(buffer->Handle(), dst->GetImage(),
                             vk::ImageLayout::eTransferDstOptimal, buffer_copy);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice) {
    const auto dst_dim = dst->info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(dst->info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(dst->info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max<u32>(dst->info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src->info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src->info.size.width >> src_dim));
    ASSERT(mip_block_h == (src->info.size.height >> src_dim));
    ASSERT(mip_block_p == (src->info.pitch >> src_dim));

    const auto [src_layers, dst_layers] =
        SanitizeCopyLayers(src->info, dst->info, src->info.size.depth);

    const vk::ImageCopy image_copy{
        .srcSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src_layers,
        },
        .dstSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = mip,
            .baseArrayLayer = slice,
            .layerCount = dst_layers,
        },
        .extent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };

    SetBackingSamples(dst, dst->info.num_samples);
    SetBackingSamples(src, src->info.num_samples);

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, image_copy);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.num_samples == 1 && dst->info.num_samples == 1) {
        if ((instance.IsMaintenance8Supported() ||
             src->info.props.is_depth == dst->info.props.is_depth) &&
            src->aspect_mask == dst->aspect_mask) {
            CopyImage(src, dst);
        } else {
            // Perform depth from/to color copy using the intermediate copy buffer.
            static constexpr size_t COPY_BUFFER_SIZE = 128_MB;
            const auto copy_ref =
                staging_pool.Request(COPY_BUFFER_SIZE, VideoCore::MemoryType::DeviceLocal);
            CopyImageWithBuffer(src, dst, copy_ref.buffer, copy_ref.offset);
        }
    } else if (src->info.num_samples == 1 && dst->info.num_samples > 1 &&
               dst->info.props.is_depth) {
        // Perform a rendering pass to transfer the channels of source as samples in dest.
        bool needs_flush =
            Transit(src, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
        needs_flush |= Transit(dst, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                               vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                   vk::PipelineStageFlagBits2::eLateFragmentTests,
                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
        if (needs_flush) {
            FlushBarriers();
        }

        blit_helper->ReinterpretColorAsMsDepth(
            dst->info.size.width, dst->info.size.height, dst->info.num_samples,
            src->info.pixel_format, dst->info.pixel_format, src->GetImage(), dst->GetImage());
    } else {
        LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
    }
}

void Runtime::CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                               const VideoCore::SubresourceRange& sub_range) {
    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto aspect_mask = src->aspect_mask & dst->aspect_mask;

    const vk::ImageCopy region = {
        .srcSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .dstOffset = {0, 0, 0},
        .extent = {dst->info.size.width, dst->info.size.height, 1},
    };

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                           const VideoCore::SubresourceRange& src_range,
                           const VideoCore::SubresourceRange& dst_range) {
    SetBackingSamples(dst, 1, false);
    scheduler.EndRendering();

    const bool needs_resolve = src->backing->num_samples != 1;
    const auto dst_stage =
        needs_resolve ? vk::PipelineStageFlagBits2::eResolve : vk::PipelineStageFlagBits2::eCopy;

    bool needs_flush = Transit(src, vk::ImageLayout::eTransferSrcOptimal, dst_stage,
                               vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal, dst_stage,
                           vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, 1);
    if (!needs_resolve) {
        const vk::ImageCopy region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->info.size.width, src->info.size.height, 1},
        };
        cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                         vk::ImageLayout::eTransferDstOptimal, region);
    } else {
        const vk::ImageResolve region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->info.size.width, src->info.size.height, 1},
        };
        cmdbuf.resolveImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                            vk::ImageLayout::eTransferDstOptimal, region);
    }

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                         const vk::ClearValue& clear_value) {
    scheduler.EndRendering();

    const bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eTransferWrite, range);
    if (needs_flush) {
        FlushBarriers();
    }

    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.clearColorImage(dst->GetImage(), vk::ImageLayout::eTransferDstOptimal, clear_value.color,
                           vk_range);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ClearDepthStencilImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range, bool clear_depth,
                                     float depth, bool clear_stencil, u32 stencil) {
    vk::ImageAspectFlags aspect{};
    if (clear_depth && (dst->aspect_mask & vk::ImageAspectFlagBits::eDepth)) {
        aspect |= vk::ImageAspectFlagBits::eDepth;
    }
    if (clear_stencil && (dst->aspect_mask & vk::ImageAspectFlagBits::eStencil)) {
        aspect |= vk::ImageAspectFlagBits::eStencil;
    }
    if (!aspect) {
        return;
    }
    scheduler.EndRendering();

    const bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eTransferWrite, range);
    if (needs_flush) {
        FlushBarriers();
    }

    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = aspect,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    const vk::ClearDepthStencilValue value{.depth = depth, .stencil = stencil};
    scheduler.CommandBuffer().clearDepthStencilImage(dst->GetImage(), vk::ImageLayout::eTransferDstOptimal, value,
                                                     vk_range);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing) {
    auto& backing = image->backing;
    auto& backing_images = image->backing_images;
    const auto& info = image->info;
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!image->info.props.is_depth, "Swapping samples is only valid for color images");
    VideoCore::RecordQueue::Guard(VideoCore::RecordQueue::Backing);
    if (VideoCore::HalfPhase::current == VideoCore::HalfPhase::Resolve) {
        VideoCore::ResolveCommands::Run();
        Core::NativeOcclusion::NoteCrossing("backing swap (SetBackingSamples) in a resolve half");
    }
    VideoCore::Image::BackingImage* new_backing;
    auto it = std::ranges::find(backing_images, num_samples,
                                &VideoCore::Image::BackingImage::num_samples);
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, image->supported_samples);

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->image = VideoCore::UniqueImage{instance.GetDevice(), instance.GetAllocator()};
        new_backing->image.Create(new_image_ci);

        Vulkan::SetObjectName(instance.GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (copy_backing) {
        scheduler.EndRendering();
        ASSERT(image->info.resources.levels == 1 && image->info.resources.layers == 1);

        // Transition current backing to shader read layout
        Transit(image, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);

        // Transition dest backing to color attachment layout, not caring of previous contents
        constexpr auto dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        constexpr auto dst_access =
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        constexpr auto dst_layout = vk::ImageLayout::eColorAttachmentOptimal;
        image_barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = new_backing->image,
            .subresourceRange{
                .aspectMask = image->aspect_mask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = image->info.resources.layers,
            },
        });
        FlushBarriers();

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(
            info.size.width, info.size.height, new_backing->num_samples, info.pixel_format,
            backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
        if (live_plan_on) {
            // the image planner learns the new backing's state the same way (no transition of its own made it)
            image_plan[ImagePlanKey(*image, new_backing->num_samples)] = {{dst_stage, dst_access, dst_layout}, {}};
        }
    }

    backing = new_backing;
}

u32 Runtime::EpochPlannerMode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_EPOCH_PLANNER");
        return e != nullptr && (e[0] == '1' || e[0] == '2') ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

bool Runtime::EpochQuery(u64 key, u64 begin, u64 size, bool check_read) const {
    const auto it = epoch_entries.find(key); // (a query makes no entry)
    if (it == epoch_entries.end() || it->second.epoch != plan_epoch) {
        return false;
    }
    const EpochEntry& e = it->second;
    const u64 end = begin + size;
    bool hit = e.write.Overlaps(begin, end) || (check_read && e.read.Overlaps(begin, end));
    // G2 mode 2: inside the group only what was accessed outside it counts; outside the group everything counts (the group's lists are filled only in mode 2)
    if (!hit && independent_group == 0) {
        hit = e.group_write.Overlaps(begin, end) || (check_read && e.group_read.Overlaps(begin, end));
    }
    return hit;
}

void Runtime::EpochAccess(u64 key, u64 begin, u64 size, bool write, bool read) {
    const size_t before = epoch_entries.size();
    EpochEntry& e = epoch_entries[key];
    if (epoch_entries.size() != before) {
        ++epoch_stats.entries_new;
    }
    if (e.epoch != plan_epoch) {
        if (e.epoch != 0) {
            ++epoch_stats.entries_reset;
        }
        e.epoch = plan_epoch;
        e.read.Clear(); // (the lists keep their storage)
        e.write.Clear();
        e.group_read.Clear();
        e.group_write.Clear();
    }
    const bool group = IndependentGroup::Mode() == 2 && independent_group != 0;
    if (write) {
        LivePlanAdd(group ? e.group_write : e.write, begin, begin + size);
    }
    if (read) {
        LivePlanAdd(group ? e.group_read : e.read, begin, begin + size);
    }
    ++epoch_stats.accesses;
}

bool Runtime::IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                               bool check_read_access) {
    Core::SplitCensus::Count(Core::SplitCensus::CtBarrierQuery);
    // B10: the planner by object answers (a placed engine buffer in its own offset space, the arena by guest address)
    const u32 epoch_mode = EpochPlannerMode();
    const bool run_old = epoch_mode != 2 || epoch_audit;
    const bool pc = Core::DispatchSplit::PlannerOn() && Core::DispatchSplit::PlannerSample(); // design review 72 census
    const u64 q0 = pc ? Core::DispatchSplit::Tsc() : 0;
    u64 begin = 0;
    const u64 census_key = LiveObjectSpace(handle, offset, begin);
    const u64 q1 = pc ? Core::DispatchSplit::Tsc() : 0;
    bool planned = false;
    if (run_old) {
        const size_t keys_before = pc ? live_object.buffers.size() : 0;
        auto& space = live_object.buffers[census_key];
        const bool planned_old = space.write.Overlaps(begin, begin + size) || (check_read_access && space.read.Overlaps(begin, begin + size));
        if (pc) {
            Core::DispatchSplit::NotePlannerQuery(q1 - q0, Core::DispatchSplit::t_oracle_ticks, Core::DispatchSplit::Tsc() - q1, live_object.buffers.size() != keys_before, planned_old);
        }
        planned = planned_old;
        if (IndependentGroup::Mode() != 0) {
            // the group-aware answer: outside a group, everything counts; inside the group, only what was accessed outside it (the group's own pending accesses are independent of each other)
            auto& normal = aware_normal.buffers[census_key];
            bool aware = normal.write.Overlaps(begin, begin + size) || (check_read_access && normal.read.Overlaps(begin, begin + size));
            if (independent_group == 0) {
                auto& grp = aware_group.buffers[census_key];
                aware = aware || grp.write.Overlaps(begin, begin + size) || (check_read_access && grp.read.Overlaps(begin, begin + size));
            }
            op_real_hazard |= planned_old;
            op_aware_hazard |= aware;
            if (independent_group != 0) {
                IndependentGroup::NoteQuery(planned_old, aware);
            }
            if (IndependentGroup::Mode() == 2) {
                planned = aware;
            }
        }
    }
    if (epoch_mode != 0) {
        const bool answer = EpochQuery(census_key, begin, size, check_read_access);
        ++epoch_stats.queries;
        if (run_old) {
            ++epoch_stats.compared;
            if (answer != planned) {
                ++epoch_stats.different;
                ++(answer ? epoch_stats.epoch_true_old_false : epoch_stats.epoch_false_old_true);
                static u32 logged = 0;
                if (logged++ < 10) {
                    LOG_ERROR(Render_Vulkan, "epoch planner: answer {} where the old planner answered {} (key {:#x}, {:#x}+{:#x}, read check {}, group {})", answer, planned, census_key, begin, size,
                              check_read_access, independent_group);
                }
            }
            if (epoch_mode == 2) {
                planned = planned || answer; // audit window: a barrier either planner wants is made
            }
        } else {
            planned = answer;
            if (IndependentGroup::Mode() != 0) {
                op_real_hazard |= answer; // (the group census sees the deciding answer only outside the audit windows)
                op_aware_hazard |= answer;
                if (independent_group != 0) {
                    IndependentGroup::NoteQuery(answer, answer);
                }
            }
        }
    }
    if (planned && BarrierCensus::On()) {
        BarrierCensus::NoteHazard(census_key, begin, size, check_read_access);
    }
    ++live_stats.object_queries;
    ++live_stats.queries;
    live_stats.answered_true += planned ? 1 : 0;
    return planned;
}

void Runtime::AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                           vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access) {
    constexpr static vk::AccessFlags2 READ_MASK =
        vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eVertexAttributeRead |
        vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eShaderRead |
        vk::AccessFlagBits2::eColorAttachmentRead |
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eTransferRead |
        vk::AccessFlagBits2::eMemoryRead;

    constexpr static vk::AccessFlags2 WRITE_MASK =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eTransferWrite |
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eTransformFeedbackWriteEXT;

    const u32 epoch_mode = EpochPlannerMode();
    const bool run_old = epoch_mode != 2 || epoch_audit;
    const bool is_write = (src_access & WRITE_MASK) != vk::AccessFlags2{};
    const bool is_read = (src_access & READ_MASK) != vk::AccessFlags2{};
    const bool pc = Core::DispatchSplit::PlannerOn() && Core::DispatchSplit::PlannerSample(); // design review 72 census
    u64 a0 = pc ? Core::DispatchSplit::Tsc() : 0, a1 = 0, a2 = 0, a3 = 0;
    bool new_node = false;
    u32 read_size = 0, write_size = 0;
    {
        u64 begin = 0;
        const u64 census_key = LiveObjectSpace(handle, offset, begin);
        if (pc) {
            a1 = Core::DispatchSplit::Tsc();
        }
        if (BarrierCensus::On()) {
            if (is_write) {
                BarrierCensus::NoteAccess(census_key, begin, size, true);
            }
            if (is_read) {
                BarrierCensus::NoteAccess(census_key, begin, size, false);
            }
        }
        if (is_write) {
            ++const_cast<VideoCore::Buffer*>(handle)->gpu_write_generation;
        }
        if (run_old) {
            const size_t keys_before = pc ? live_object.buffers.size() : 0;
            auto& space = live_object.buffers[census_key];
            if (pc) {
                a2 = Core::DispatchSplit::Tsc();
                new_node = live_object.buffers.size() != keys_before;
            }
            if (is_write) {
                LivePlanAdd(space.write, begin, begin + size);
            }
            if (is_read) {
                LivePlanAdd(space.read, begin, begin + size);
            }
            if (pc) {
                a3 = Core::DispatchSplit::Tsc();
                read_size = static_cast<u32>(space.read.Size());
                write_size = static_cast<u32>(space.write.Size());
            }
            if (IndependentGroup::Mode() != 0) {
                auto& aware = (independent_group != 0 ? aware_group : aware_normal).buffers[census_key];
                if (is_write) {
                    LivePlanAdd(aware.write, begin, begin + size);
                }
                if (is_read) {
                    LivePlanAdd(aware.read, begin, begin + size);
                }
            }
        } else if (pc) {
            a2 = a3 = Core::DispatchSplit::Tsc();
        }
        if (epoch_mode != 0) {
            EpochAccess(census_key, begin, size, is_write, is_read);
        }
        live_object.memory_pending |= src_stage != vk::PipelineStageFlags2{};
    }
    // B10 native: the address planner is not kept (only its write count, which R2 asks)
    if (is_write) {
        ++live_plan_writes;
    }
    live_plan.memory_pending |= src_stage != vk::PipelineStageFlags2{};
    memory_barrier.srcStageMask |= src_stage;
    memory_barrier.srcAccessMask |= src_access & WRITE_MASK;
    if (pc) {
        Core::DispatchSplit::NotePlannerAccess(a1 - a0, Core::DispatchSplit::t_oracle_ticks, a2 - a1, a3 - a2, Core::DispatchSplit::Tsc() - a3, handle->CpuAddr() != 0, new_node,
                                               is_write, read_size, write_size);
    }
}

void Runtime::FlushBarriers() {
    Core::SplitCensus::PhaseScope phase_scope{Core::SplitCensus::PhaseBarrier};
    Core::SplitCensus::Touch(Core::SplitCensus::Barrier);
    vk::DependencyInfo dep_info{};

    if (memory_barrier.srcStageMask) {
        dep_info.pMemoryBarriers = &memory_barrier;
        dep_info.memoryBarrierCount = 1U;
    }
    if (!image_barriers.empty()) {
        dep_info.pImageMemoryBarriers = image_barriers.data();
        dep_info.imageMemoryBarrierCount = static_cast<u32>(image_barriers.size());
    }

    const bool flush_census = Core::DispatchSplit::PlannerOn();
    const u64 f0 = flush_census ? Core::DispatchSplit::Tsc() : 0;
    const u32 object_keys = flush_census ? static_cast<u32>(live_object.buffers.size()) : 0u;
    const u32 address_keys = flush_census ? static_cast<u32>(live_plan.buffers.size()) : 0u;
    const u32 group_keys = flush_census ? static_cast<u32>(aware_normal.buffers.size() + aware_group.buffers.size()) : 0u;
    struct FlushCensus {
        bool on;
        u64 f0, f1{0};
        u32 object_keys, address_keys, group_keys;
        bool actual{false};
        ~FlushCensus() {
            if (on) {
                const u64 end = Core::DispatchSplit::Tsc();
                const u64 clear_end = f1 != 0 ? f1 : end;
                Core::DispatchSplit::NotePlannerFlush(clear_end - f0, f1 != 0 ? end - f1 : 0, object_keys, address_keys, group_keys, actual);
            }
        }
    } flush_scope{flush_census, f0, 0, object_keys, address_keys, group_keys};
    if (live_plan_on) {
        const bool actual = dep_info.memoryBarrierCount != 0 || dep_info.imageMemoryBarrierCount != 0;
        const bool planned = live_plan.memory_pending || !image_barriers.empty();
        ++live_stats.flushes;
        live_stats.flush_differences += planned != actual ? 1 : 0;
        flush_scope.actual = actual;
        if (actual) {
            live_plan.buffers.clear();
            live_plan_writes = 0;
            live_plan.memory_pending = false;
            live_object.buffers.clear();
            live_object.memory_pending = false;
            aware_normal.buffers.clear();
            aware_group.buffers.clear();
            if (EpochPlannerMode() != 0) {
                // design review 72: the epoch moves on (every entry counts as empty from now); the map is cleared only when it grew past its cap
                ++plan_epoch;
                ++epoch_flushes;
                ++epoch_stats.windows;
                if (epoch_entries.size() > (size_t{1} << 16)) {
                    epoch_entries.clear();
                    ++epoch_stats.cap_clears;
                }
                epoch_audit = EpochPlannerMode() != 2 || (epoch_flushes & 63) == 0;
                epoch_stats.audit_windows += epoch_audit && EpochPlannerMode() == 2 ? 1 : 0;
            }
            if (BarrierCensus::On()) {
                BarrierCensus::ClearPending();
            }
        }
    }
    if (flush_census) {
        flush_scope.f1 = Core::DispatchSplit::Tsc(); // the clears end here; the rest is the barrier's record
    }
    if (!dep_info.memoryBarrierCount && !dep_info.imageMemoryBarrierCount) {
        return;
    }

    ++live_stats.barriers_recorded;
    live_stats.barriers_memory_only += dep_info.imageMemoryBarrierCount == 0 ? 1 : 0;
    live_stats.barriers_ending_a_pass += scheduler.IsRendering() ? 1 : 0;
    live_stats.barriers_from_compute_writes += (memory_barrier.srcStageMask & vk::PipelineStageFlagBits2::eComputeShader) ? 1 : 0;
    if (scheduler.IsRendering()) {
        g_flush_cuts.Note(reinterpret_cast<u64>(__builtin_return_address(0)), dep_info.imageMemoryBarrierCount, dep_info.memoryBarrierCount == 0);
        if (t_flush_in_same_pass) {
            g_same_pass_flushes.flushes.fetch_add(1, std::memory_order_relaxed);
            g_same_pass_flushes.image_barriers.fetch_add(dep_info.imageMemoryBarrierCount, std::memory_order_relaxed);
            g_same_pass_flushes.with_memory_barrier.fetch_add(dep_info.memoryBarrierCount != 0 ? 1 : 0, std::memory_order_relaxed);
            for (const auto& b : image_barriers) {
                g_flush_cut_layouts.Note(0x1000000u | (static_cast<u32>(b.oldLayout) & 0xfff) << 12 | (static_cast<u32>(b.newLayout) & 0xfff), 0, false);
            }
        } else {
            g_same_pass_flushes.other_pass_changes.fetch_add(1, std::memory_order_relaxed);
        }
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    {
        static const bool trace = [] { const char* e = std::getenv("TLG_TRACE_DISPLAY_LAYOUT"); return e != nullptr && e[0] == '1'; }();
        static std::atomic<u32> traced{0};
        if (trace && dep_info.imageMemoryBarrierCount != 0 && traced.load(std::memory_order_relaxed) < 3000) {
            traced.fetch_add(1, std::memory_order_relaxed);
            std::string images;
            for (u32 i = 0; i < dep_info.imageMemoryBarrierCount; ++i) {
                const auto& b = dep_info.pImageMemoryBarriers[i];
                images += fmt::format(" {:#x}:{}>{}", reinterpret_cast<u64>(static_cast<VkImage>(b.image)), vk::to_string(b.oldLayout).substr(0, 5), vk::to_string(b.newLayout).substr(0, 5));
            }
            LOG_WARNING(Render_Vulkan, "layout trace: flush at tick {} recording {}:{}", scheduler.CurrentTick(), scheduler.IsRecordingCommands(), images);
        }
    }
    cmdbuf.pipelineBarrier2(dep_info);

    memory_barrier.srcStageMask = vk::PipelineStageFlagBits2::eNone;
    memory_barrier.srcAccessMask = vk::AccessFlagBits2::eNone;

    image_barriers.clear();
}

} // namespace Vulkan
