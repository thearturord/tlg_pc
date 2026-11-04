// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// The rasterizer as the native executor's host (native/native_raster_host.h): the plan accessors and the interval statistics. The rest of the interface is the rasterizer's own.

#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_device.h"
#include "native/native_resources.h"
#include "native/native_updates.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Core::NativeRasterQueue {
void PlanDelete::operator()(Vulkan::DrawPlan* plan) const {
    delete plan;
}

void PlanDelete::operator()(Vulkan::DispatchPlan* plan) const {
    delete plan;
}
} // namespace Core::NativeRasterQueue

namespace Vulkan {

Core::DrawPlanPtr Rasterizer::MakeDrawPlan(Core::DrawIndirectParams params, const FixedState& native, u64 exec_key, u64 inputs_key, const Core::NativeDraw::ExecDraw& record,
                                           u32 index_count) {
    return Core::DrawPlanPtr{PlanDraw(params, native, exec_key, inputs_key, record, index_count).release()};
}

Core::DrawPlanPtr Rasterizer::MakeExecDrawPlan(const FixedState& native, u64 exec_key, u64 inputs_key, std::span<const u32, 16> vertex_user_data,
                                               std::span<const u32, 16> pixel_user_data, const Core::NativeDraw::ExecDraw& draw, u32 index_count, u32 draw_class) {
    return Core::DrawPlanPtr{PlanExecDraw(native, exec_key, inputs_key, vertex_user_data, pixel_user_data, draw, index_count, draw_class).release()};
}

Core::DrawPlanPtr Rasterizer::MakeOperationPlan(const Core::NativeDraw::ExecDraw& draw) {
    return Core::DrawPlanPtr{PlanExecOperation(draw).release()};
}

Core::DispatchPlanPtr Rasterizer::MakeDispatchPlan(const Core::NativeDispatch::DispatchInputs& cs) {
    return Core::DispatchPlanPtr{PlanDispatch(cs).release()};
}

u64 Rasterizer::DrawPlanDigest(const DrawPlan& plan) {
    return PlanDigest(plan);
}

u64 Rasterizer::DispatchPlanDigest(const DispatchPlan& plan) {
    return PlanDigest(plan);
}

void Rasterizer::RecycleDrawPlan(Core::DrawPlanPtr&& plan) {
    RecyclePlan(std::unique_ptr<DrawPlan>{plan.release()});
}

void Rasterizer::RecycleDispatchPlan(Core::DispatchPlanPtr&& plan) {
    RecyclePlan(std::unique_ptr<DispatchPlan>{plan.release()});
}

u64 Rasterizer::DrawPlanSeq(const DrawPlan& plan) {
    return plan.native_seq;
}

bool Rasterizer::DrawPlanIsDepthOnly(const DrawPlan& plan) {
    return plan.fixed.NoColorTargets();
}

void Rasterizer::SetDrawPlanTargets(DrawPlan& plan, const ColorExtents& cb_extent, const DepthExtent& db_extent, const std::array<u64, 9>& target_resources) {
    plan.cb_extent = cb_extent;
    plan.db_extent = db_extent;
    plan.target_resources = target_resources;
}

const Core::NativeDispatch::DispatchInputs& Rasterizer::DispatchPlanInputs(const DispatchPlan& plan) {
    return plan.cs;
}

void Rasterizer::TagDispatchPlan(DispatchPlan& plan, u64 seq) {
    plan.native_seq = seq;
    plan.bindings_complete = Core::NativeDraw::TakeDispatchBindings(seq, plan.bindings.data(), static_cast<u32>(plan.bindings.size()), plan.binding_count);
}

bool Rasterizer::MarkersEnabled() const noexcept {
    return host_markers_enabled;
}

void Rasterizer::MarkerBegin(std::string_view text) {
    ScopeMarkerBegin(text);
}

void Rasterizer::MarkerEnd() {
    ScopeMarkerEnd();
}

Core::RasterHost::IntervalStats Rasterizer::PublishIntervalStats() {
    IntervalStats result;
    const auto plan_stats = TakePlanStats();
    const auto memo_stats = TakeBeginMemoStats();
    if (TextureSetMemoOn()) {
        const auto sets = TakeTextureSetStats();
        Core::DebugServer::Publish("texture_set_memo", {{"hits", sets.hits}, {"recorded", sets.recorded}, {"miss_no_entry", sets.no_entry}, {"miss_generation", sets.generation},
                                                      {"miss_image_check", sets.image_check}, {"not_eligible", sets.not_eligible}, {"note", "counts per ~5 s interval"}});
    }
    {
        const auto c = TakeConsumerCensus();
        nlohmann::json out = nlohmann::json::object();
        static constexpr const char* shader_names[] = {"buffers", "buffers_ring", "buffers_special", "buffers_null", "buffers_by_engine_handle", "buffers_by_address_read_only",
                                                       "buffers_by_address_written", "address_read_le256", "address_read_le4k", "address_read_le64k", "address_read_bigger",
                                                       "buffer_barrier_queries", "buffer_barrier_true", "images", "images_null", "images_from_texture_set_memo",
                                                       "images_from_lookup_cache", "images_find_image", "images_storage", "image_transits", "image_transits_true",
                                                       "bind_resources_calls", "buffers_in_place", "buffers_resolved_ahead"};
        for (u32 side = 0; side < 2; ++side) {
            nlohmann::json part = nlohmann::json::object();
            for (u32 k = 0; k < std::size(shader_names); ++k) {
                part[shader_names[k]] = c[(side ? 32 : 0) + k];
            }
            out[side ? "compute" : "graphics"] = std::move(part);
        }
        out["geometry"] = {{"vertex_ring", c[24]}, {"vertex_by_engine_handle", c[25]}, {"vertex_by_address", c[26]},
                           {"index_ring", c[27]}, {"index_by_engine_handle", c[28]}, {"index_by_address", c[29]}};
        out["note"] = "C0 census, counts per ~5 s interval (read only)";
        Core::DebugServer::Publish("consumer_census", std::move(out));
        const auto r = TakeResolveStats();
        Core::DebugServer::Publish("native_backend", {{"buffers_resolved_by_the_command_processor", r[0]}, {"bound_from_the_resolution", r[1]},
                                                      {"resolution_stale_at_bind", r[2]},
                                                      {"images", {{"record_words_resolved_by_the_command_processor", image_words_planned.exchange(0)},
                                                                  }},
                                                      {"buffer_slots", {{"new_for_stream_ring", slot_census[0].exchange(0)}, {"new_for_other_buffers", slot_census[2].exchange(0)},
                                                                        {"hits", slot_census[3].exchange(0)}}},
                                                      {"image_census", Core::NativeResources::ImageCensus()},
                                                      {"device", Core::NativeDevice::Report()},
                                                      {"b5_copies", {{"on_the_gpu", b5_copies_on_gpu.exchange(0)}, {"as_a_cpu_copy_in_ps4_memory", b5_copies_on_cpu.exchange(0)}}},
                                                      {"uploads_that_kept_gpu_written_bytes", UploadsKeptForGpu()},
                                                      {"b6_copy_results", {{"downloaded_b6_off", b6_downloads[0].exchange(0)}, {"left_on_the_gpu", b6_downloads[1].exchange(0)},
                                                                            {"downloaded_cpu_read_resource", b6_downloads[2].exchange(0)}, {"downloaded_no_engine_owner", b6_downloads[3].exchange(0)},
                                                                            {"downloaded_explicit_readback", b6_downloads[4].exchange(0)}}},
                                                      {"b3_images", {{"by_engine_id", b3_stats[0].exchange(0)}, {"first_sight", b3_stats[1].exchange(0)},
                                                                     {"image_changed", b3_stats[2].exchange(0)}}},
                                                      {"b2_applied", Core::NativeUpdates::B2Report()},
                                                      {"b8_images", Core::NativeUpdates::B8Report()},
                                                      {"b9_buffers", Core::NativeUpdates::B9Report()}, {"zero_copy", GetBufferCache().ZeroCopyReport()}, {"bind_memo", GetBufferCache().BindMemoReport()},
                                                      {"b11_clears", Core::NativeDevice::ClearReport()},
                                                      {"b9_raw_address_shaders", {{"binds_reading_memory_by_address", b9_dma_binds.exchange(0)}, {"all_binds", b9_all_binds.exchange(0)}}},
                                                      {"stale_why", {{"epoch", r[12]}, {"gone", r[13]}, {"not_synchronized", r[14]}, {"generation_moved", r[15]}, {"arena", r[16]}, {"tracker_dirty", r[17]}}},
                                                      {"not_resolved", {{"no_record_entry", r[3]}, {"null_or_huge", r[4]}, {"no_engine_object", r[5]}, {"not_immutable", r[6]},
                                                                        {"outside_the_object", r[7]}, {"not_synchronized", r[8]}, {"no_arena_yet", r[9]}, {"arena_moved", r[10]}, {"clamped", r[11]}}},
                                                      {"note", "R2, counts per ~5 s interval"}});
    }
    if (Core::NativeResources::ResidentBuffersEnabled()) {
        const auto r = Core::NativeResources::TakeResidentCounters();
        static std::array<u64, 8> resident_total{};
        for (u32 i = 0; i < 8; ++i) {
            resident_total[i] += r[i];
        }
        Core::DebugServer::Publish("resident_buffers",
                                   {{"answered_without_a_synchronization", r[0]}, {"synchronized_first_use_or_after_a_write", r[1]}, {"verify_runs", r[2]},
                                    {"verify_runs_that_found_something_to_upload_stale_answers", r[3]}, {"left_to_the_normal_path_gpu_modified", r[4]}, {"registered_buffers", r[5]},
                                    {"dropped_by_announced_writes", r[6]}, {"dropped_by_cpu_write_faults", r[7]},
                                    {"since_start", {{"answered_without_a_synchronization", resident_total[0]}, {"synchronized", resident_total[1]}, {"verify_runs", resident_total[2]}, {"stale_answers_found_by_verify", resident_total[3]}, {"registered_buffers", resident_total[5]}}},
                                    {"note", "C1 / C3, counts per ~5 s interval; since_start is cumulative"}});
    }
    {
        const auto& s = Vulkan::g_segment_census;
        nlohmann::json kinds = nlohmann::json::object();
        const char* names[3] = {"depth_only", "colour", "g_buffer_4_or_more_targets"};
        for (u32 k = 0; k < 3; ++k) {
            kinds[names[k]] = {{"rendering_instances", s.segments[k].load()}, {"commands", s.commands[k].load()}, {"draws", s.draws[k].load()}, {"largest_instance_commands", s.max_commands[k].load()}};
        }
        Core::DebugServer::Publish("segment_census", {{"by_kind_since_start", std::move(kinds)}, {"note", "C6 census: commands and draws recorded between a begin and an end of rendering, graphics stream, counts only"}});
    }
    if (LivePlannerOn()) {
        const auto p = ReadLivePlanStats();
        Core::DebugServer::Publish("barrier_planner",
                                   {{"queries_answered_by_the_planner", p.queries}, {"answered_true", p.answered_true}, {"misses_barrier_the_tracker_had_and_the_planner_lacked", p.planner_missed},
                                    {"extra_barriers_the_tracker_did_not_need", p.planner_extra}, {"tracker_true_only_from_its_own_range_collapse_(mode_2)", p.tracker_collapse_only}, {"flushes", p.flushes}, {"flush_decisions_differing", p.flush_differences},
                                    {"buffer_rebases_seen", p.rebases},
                                    {"pipeline_barriers", {{"recorded", p.barriers_recorded}, {"memory_only", p.barriers_memory_only}, {"ending_a_render_pass", p.barriers_ending_a_pass}, {"after_compute_writes", p.barriers_from_compute_writes}}},
                                    {"images", {{"transitions", p.image_transits}, {"barriers", p.image_barriers}, {"DIFFERENCES_from_the_image_state", p.image_differences}, {"images_planned", p.images_planned}}},
                                    {"b10_by_object", {{"mode", B10Mode()}, {"queries", p.object_queries}, {"in_a_placed_buffers_own_space", p.object_space_queries},
                                                       {"same_answer", p.object_same}, {"OBJECT_NO_ADDRESS_YES", p.object_no_address_yes}, {"object_yes_address_no", p.object_yes_address_no}}},
                                    {"note", "L4 TLG_BARRIER_PLANNER=1, cumulative since start; the planner decides, the tracker corrects a miss"}});
    }
    if (InPlaceOn()) {
        const auto p = TakeInPlaceStats();
        Core::DebugServer::Publish("in_place_buffers",
                                   {{"offered", p.offered}, {"bound_in_place", p.taken}, {"bytes", p.bytes},
                                    {"refused", {{"written_by_a_shader", p.written}, {"too_large", p.too_large}, {"gpu_written_bytes", p.gpu_written}, {"image_or_metadata", p.image_or_meta},
                                                 {"not_in_one_mapped_region_or_import_failed", p.not_in_backing}}},
                                    {"imported", {{"bytes", p.imported_bytes}, {"chunks", p.imported_chunks}, {"failed_chunks", p.failed_chunks}}},
                                    {"note", "counts per ~5 s interval; imported is cumulative"}});
    }
    if (ConstRingLevel() != 0) {
        const auto ring = TakeRingStats();
        Core::DebugServer::Publish(
            "const_ring",
            {{"level", ConstRingLevel()}, {"queued_ops", ring.queued}, {"written_buffers_noted", ring.written_buffers}, {"offered", ring.offered},
             {"taken", ring.taken}, {"bytes", ring.bytes}, {"bound_by_raster", ring.bound},
             {"refused", {{"no_sharp", ring.no_sharp}, {"written", ring.written}, {"formatted", ring.formatted}, {"too_large", ring.too_large},
                          {"immutable", ring.immutable}, {"gpu_modified", ring.gpu_modified}, {"pending_write", ring.pending_write},
                          {"ring_full", ring.ring_full}}},
             {"note", "counts per ~5 s interval"}});
    }

    result.plan_draws_all_stages_same = plan_stats.draws_all_stages_same;
    result.plan_draws = plan_stats.draws;
    result.plan_stages_same = plan_stats.stages_same;
    result.plan_stages = plan_stats.stages;
    result.memo_hits = memo_stats.first;
    result.memo_misses = memo_stats.second;
    const auto native = TakeNativeStats();
    result.native_tagged = native.tagged;
    result.native_untagged = native.untagged;
    result.native_dynamic_skipped = native.dynamic_skipped;
    result.native_bind_skipped = native.bind_skipped;
    return result;
}

} // namespace Vulkan
