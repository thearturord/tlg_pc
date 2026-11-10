// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// The rasterizer as the native executor's host (native/native_raster_host.h): the plan accessors and the interval statistics. The rest of the interface is the rasterizer's own.

#include "native/native_row_version.h"
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_device.h"
#include "native/native_n0_census.h"
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
    if (TextureSetMemoOn()) {
        const auto sets = TakeTextureSetStats();
        Core::DebugServer::Publish("texture_set_memo", {{"hits", sets.hits}, {"recorded", sets.recorded}, {"miss_no_entry", sets.no_entry}, {"miss_generation", sets.generation},
                                                      {"miss_image_check", sets.image_check}, {"not_eligible", sets.not_eligible},
                                                      {"by_id", {{"predicted_unchanged", sets.by_id_predicted}, {"served", sets.by_id_served}, {"shadow_same", sets.by_id_same},
                                                                 {"SHADOW_DIFFERENT_MUST_BE_0", sets.by_id_DIFFERENT}}},
                                                      {"note", "counts per ~5 s interval"}});
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
        {
            nlohmann::json ra = nlohmann::json::object();
            static constexpr const char* names[] = {"bindings_with_engine_id", "front_saw_none", "KEY_DIFFERS", "same_key_row_served_version_as_planned", "same_key_row_served_version_moved",
                                                    "same_key_no_row_served", "of_same_key_written", "of_same_key_texel"};
            for (u32 side = 0; side < 2; ++side) {
                nlohmann::json one = nlohmann::json::object();
                for (u32 k = 0; k < 8; ++k) {
                    one[names[k]] = row_ahead_census[side][k].load();
                }
                ra[side ? "compute" : "graphics"] = std::move(one);
            }
            out["row_ahead_census_cumulative"] = std::move(ra);
        }
        {
            nlohmann::json rs = nlohmann::json::object();
            static constexpr const char* names[] = {"bindings_with_engine_id", "planned_row_found", "planned_row_valid_at_bind", "valid_raster_row_served_same",
                                                    "VALID_RASTER_ROW_SERVED_DIFFERENT_MUST_BE_0", "valid_raster_old_path_same", "valid_raster_old_path_different",
                                                    "planned_row_stale_at_bind", "no_planned_row_but_raster_row_served", "of_valid_static", "of_valid_written", "x11"};
            for (u32 side = 0; side < 2; ++side) {
                nlohmann::json one = nlohmann::json::object();
                for (u32 k = 0; k < 12; ++k) {
                    one[names[k]] = rows_ahead_shadow[side][k].load();
                }
                rs[side ? "compute" : "graphics"] = std::move(one);
            }
            out["rows_ahead_shadow_cumulative"] = std::move(rs);
            static constexpr const char* real_names[] = {"bound_from_planned_row", "sampled_full_path", "SAMPLED_DIFFERENT_MUST_BE_0", "sampled_same", "refused_identity_walk",
                                                         "planned_row_stale", "static_row_full_path", "no_planned_row"};
            nlohmann::json rr = nlohmann::json::object();
            for (u32 side = 0; side < 2; ++side) {
                nlohmann::json one = nlohmann::json::object();
                for (u32 k = 0; k < 8; ++k) {
                    one[real_names[k]] = rows_ahead_real[side][k].load();
                }
                rr[side ? "compute" : "graphics"] = std::move(one);
            }
            out["rows_ahead_real_cumulative"] = std::move(rr);
            static constexpr const char* tex_names[] = {"no_image", "more_than_16_images", "memo_off", "written_image", "mip_fallback_image", "set_hit", "miss_no_entry",
                                                        "miss_generation", "miss_image_check", "other"};
            nlohmann::json tc = nlohmann::json::object();
            for (u32 side = 0; side < 2; ++side) {
                nlohmann::json one = nlohmann::json::object();
                for (u32 k = 0; k < 10; ++k) {
                    one[tex_names[k]] = {{"calls", tex_census[side][k][0].load()}, {"ns", tex_census[side][k][1].load()}, {"images", tex_census[side][k][2].load()}};
                }
                tc[side ? "compute" : "graphics"] = std::move(one);
            }
            out["texture_census_cumulative"] = std::move(tc);
        }
        out["geometry"] = {{"vertex_ring", c[24]}, {"vertex_by_engine_handle", c[25]}, {"vertex_by_address", c[26]},
                           {"index_ring", c[27]}, {"index_by_engine_handle", c[28]}, {"index_by_address", c[29]}};
        if (cover_on) {
            nlohmann::json cov = nlohmann::json::array();
            for (u32 slot = 0; slot < 256; ++slot) {
                if (const u64 n = cover_count[slot].load(); n != 0) {
                    cov.push_back({{"side", slot >= 128 ? "compute" : "graphics"}, {"mask", slot & 127}, {"ops", n}, {"bind_ns", cover_ns[slot].load()}, {"sync_ns", cover_sync_ns[slot].load()}});
                }
            }
            out["coverage_cumulative"] = std::move(cov);
            nlohmann::json bc = nlohmann::json::array();
            for (u32 slot = 0; slot < 256; ++slot) {
                if (const u64 n = cover_buf_n[slot].load(); n != 0) {
                    bc.push_back({{"side", slot >= 128 ? "compute" : "graphics"}, {"cause", (slot & 127) / 8}, {"kind", slot & 7}, {"n", n}, {"ns", cover_buf_ns[slot].load()},
                                  {"sync_ns", cover_buf_sync[slot].load()}, {"sync_calls_that_copied", cover_buf_copies[slot].load()}});
                }
            }
            out["buffer_causes_cumulative"] = std::move(bc);
            {
                // design review 53: the handle path of a binding in six segments (ns, cumulative)
                static constexpr const char* seg_names[] = {"0_resolve_ahead_check_and_descriptor", "1_hint_census_bind_context", "2_row_lookup", "3_answer_verify_learn",
                                                            "4_recipe_recorder_audit", "5_binding_write_barrier_query"};
                nlohmann::json hs = nlohmann::json::object();
                for (u32 served = 0; served < 2; ++served) {
                    for (u32 side = 0; side < 2; ++side) {
                        auto& seg = (served ? cover_handle_seg_served : cover_handle_seg)[side];
                        nlohmann::json one = {{"bindings", seg[6].load()}};
                        for (u32 k = 0; k < 6; ++k) {
                            one[seg_names[k]] = seg[k].load();
                        }
                        hs[std::string(served ? "row_served_" : "not_row_served_") + (side ? "compute" : "graphics")] = std::move(one);
                    }
                }
                out["handle_path_segments_ns_cumulative"] = std::move(hs);
            }
            nlohmann::json cc = nlohmann::json::array();
            for (u32 c = 0; c < 16; ++c) {
                if (const u64 n = cover_class_draws[c].load(); n != 0) {
                    cc.push_back({{"class", c}, {"draws", n}, {"bind_ns", cover_class_bind_ns[c].load()}, {"draw_total_ns", cover_class_total_ns[c].load()}});
                }
            }
            out["draw_classes_cumulative"] = std::move(cc);
            {
                static constexpr const char* names[] = {"passes", "draws", "sum_distinct_pipelines", "max_distinct_pipelines", "passes_pipelines_le8", "passes_pipelines_le16", "passes_pipelines_le24", "passes_pipelines_more",
                                                        "sum_distinct_geometry_sources", "passes_geom_ratio_le10", "passes_geom_ratio_le33", "passes_geom_ratio_le66", "passes_geom_ratio_more", "sum_distinct_other_buffer_sets",
                                                        "sum_distinct_ring_windows", "sum_distinct_states", "pairs", "pairs_same_pipeline", "pairs_same_geometry", "pairs_same_other_buffers", "pairs_same_push", "pairs_same_state",
                                                        "pairs_same_vertex", "pairs_same_index", "ring_windows_per_draw_sum", "other_buffers_per_draw_sum", "images_per_draw_sum", "index_count_sum", "instance_count_sum",
                                                        "passes_of_one_draw", "draws_in_passes_over_50", "sum_distinct_index_buffer_handles", "sum_distinct_other_buffer_handles"};
                nlohmann::json w1 = nlohmann::json::object();
                for (u32 i = 0; i < std::size(names); ++i) {
                    w1[names[i]] = w1_counts[i].load();
                }
                out["w1_census"] = std::move(w1);
            }
            {
                // W1.1 (rewrite plan section 4b)
                static constexpr const char* names[] = {"shadow_draws", "eligible_no_bad_bit", "eligible_if_front_makes_flat_constants", "eligible_if_stream_copies_allowed",
                                                        "draws_with_image", "draws_with_written_buffer", "draws_with_buffer_by_address", "draws_with_buffer_from_stream_ring", "draws_with_buffer_upload",
                                                        "draws_with_flat_constants_copy", "draws_with_geometry_by_address", "draws_with_geometry_from_stream_ring", "draws_with_geometry_upload",
                                                        "draws_with_in_place_buffer",
                                                        "only_image", "only_written_buffer", "only_buffer_by_address", "only_buffer_from_stream_ring", "only_buffer_upload", "only_flat_constants_copy",
                                                        "only_geometry_by_address", "only_geometry_from_stream_ring", "only_geometry_upload", "only_in_place_buffer",
                                                        "record_integers_sum", "record_integers_max",
                                                        "row_shader_same", "row_shader_DIFFERENT_MUST_BE_0", "row_shader_learned", "row_shader_epoch_moved",
                                                        "row_geometry_same", "row_geometry_DIFFERENT_MUST_BE_0", "row_geometry_learned", "row_geometry_epoch_moved",
                                                        "handle_buffers_from_stream_ring", "handle_buffers_resident", "handle_buffers_resident_with_upload",
                                                        "geometry_from_stream_ring", "geometry_resident", "geometry_resident_with_upload", "geometry_without_engine_handle",
                                                        "nohandle_0_has_handle", "nohandle_1_no_binder_plan", "nohandle_2_not_single_load", "nohandle_3_no_record_entry", "nohandle_4_plan_has_no_handles",
                                                        "nohandle_5_id_not_in_registry", "nohandle_6_handle_epoch_stale", "nohandle_7",
                                                        "texel_0_has_handle", "texel_1_no_binder_plan", "texel_2_not_single_load", "texel_3_no_record_entry", "texel_4_plan_has_no_handles",
                                                        "texel_5_id_not_in_registry", "texel_6_handle_epoch_stale", "texel_7",
                                                        "geo_nohandle_1_no_plan", "geo_nohandle_2_plan_has_no_handles", "geo_nohandle_3_epoch_stale", "geo_nohandle_4_no_handle_holds_request",
                                                        "eligible_draw_ns_before_bind", "eligible_draw_ns_in_bind", "eligible_draw_ns_after_bind", "eligible_draws_timed",
                                                        "other_draw_ns_before_bind", "other_draw_ns_in_bind", "other_draw_ns_after_bind", "other_draws_timed",
                                                        "draws_with_image_work", "only_image_work_or_image", "eligible_images_allowed_no_image_work", "eligible_images_and_image_work_allowed",
                                                        "shape_new_in_pass", "shape_same_as_previous_draw", "shape_seen_earlier_in_pass", "sum_distinct_shapes_per_pass",
                                                        "x77", "x78", "x79",
                                                        "row_compute_read_same", "row_compute_read_DIFFERENT_MUST_BE_0", "row_compute_read_learned", "row_compute_read_epoch_moved",
                                                        "row_compute_written_same", "row_compute_written_DIFFERENT_MUST_BE_0", "row_compute_written_learned", "row_compute_written_epoch_moved",
                                                        "stamp_graphics_read_0_good_valid_no_work_same", "stamp_graphics_read_1_VALID_BUT_WORK_MUST_BE_0", "stamp_graphics_read_2_valid_but_moved", "stamp_graphics_read_3_missed_not_predicted_no_work", "stamp_graphics_read_4_not_predicted_work", "stamp_graphics_read_5_no_row", "stamp_graphics_read_6_dirty", "stamp_graphics_read_7_epoch_moved",
                                                        "stamp_graphics_written_0_good_valid_no_work_same", "stamp_graphics_written_1_VALID_BUT_WORK_MUST_BE_0", "stamp_graphics_written_2_valid_but_moved", "stamp_graphics_written_3_missed_not_predicted_no_work", "stamp_graphics_written_4_not_predicted_work", "stamp_graphics_written_5_no_row", "stamp_graphics_written_6_dirty", "stamp_graphics_written_7_epoch_moved",
                                                        "stamp_compute_read_0_good_valid_no_work_same", "stamp_compute_read_1_VALID_BUT_WORK_MUST_BE_0", "stamp_compute_read_2_valid_but_moved", "stamp_compute_read_3_missed_not_predicted_no_work", "stamp_compute_read_4_not_predicted_work", "stamp_compute_read_5_no_row", "stamp_compute_read_6_dirty", "stamp_compute_read_7_epoch_moved",
                                                        "stamp_compute_written_0_good_valid_no_work_same", "stamp_compute_written_1_VALID_BUT_WORK_MUST_BE_0", "stamp_compute_written_2_valid_but_moved", "stamp_compute_written_3_missed_not_predicted_no_work", "stamp_compute_written_4_not_predicted_work", "stamp_compute_written_5_no_row", "stamp_compute_written_6_dirty", "stamp_compute_written_7_epoch_moved",
                                                        "stampwork_graphics_read_of_1_stream_choice_only", "stampwork_graphics_written_of_1_stream_choice_only", "stampwork_compute_read_of_1_stream_choice_only", "stampwork_compute_written_of_1_stream_choice_only",
                                                        "x124", "x125", "x126", "x127",
                                                        "rows_served", "rows_served_and_verified_by_the_old_path", "rows_verify_DIFFERENT_ANSWER_MUST_BE_0", "rows_verify_OLD_PATH_DID_WORK_MUST_BE_0", "rows_learned",
                                                        "rows_not_learned_texel_over_image_or_gpu_written", "rows_not_served_no_row", "rows_not_served_stale",
                                                        "rows_differ_no_handle", "rows_differ_handle_usage1", "rows_differ_handle_other_usage", "rows_differ_old_answer_in_stream_ring"};
                nlohmann::json w1s = nlohmann::json::object();
                for (u32 i = 0; i < std::size(names); ++i) {
                    w1s[names[i]] = w1s_counts[i].load();
                }
                out["w1_shadow"] = std::move(w1s);
            }
            {
                static constexpr const char* names[] = {"dispatches", "total_ns", "ns_before_bind", "ns_in_bind", "ns_after_bind", "sync_ns", "uploaded_bytes", "b_ring", "b_special", "b_null", "b_engine_handle",
                                                        "b_address_read", "b_address_written", "b_resolved_by_front", "images", "written_buffers", "stream_ring_answers", "requests_that_uploaded", "with_barrier", "groups",
                                                        "why0_has_handle", "why1_no_binder_plan", "why2_not_single_load", "why3_no_record_entry", "why4_plan_has_no_handles", "why5_id_not_in_registry", "why6_epoch_stale", "why7",
                                                        "written_without_record_id", "written_with_record_id", "x30", "x31",
                                                        "phase_entry_drains", "phase_compute_image_op_checks", "phase_setup", "phase_buffers", "phase_textures_and_samplers", "phase_finish",
                                                        "buf_ns_before_obtain", "buf_ns_obtain", "buf_ns_after_obtain", "buf_bindings_with_obtain",
                                                        "obtain_written_ns", "obtain_written_n", "obtain_read_old_ns", "obtain_read_old_n", "obtain_read_row_ns", "obtain_read_row_n"};
                nlohmann::json rows = nlohmann::json::array();
                for (const auto& [program, v] : W3Rows()) {
                    nlohmann::json row = {{"program", fmt::format("{:#x}", program)}};
                    for (u32 i = 0; i < std::size(names); ++i) {
                        row[names[i]] = v[i];
                    }
                    rows.push_back(std::move(row));
                }
                out["w3_dispatch"] = std::move(rows);
            }
            nlohmann::json sk = nlohmann::json::array();
            for (u32 s = 0; s < 32; ++s) {
                if (const u64 n = cover_skip_n[s].load(); n != 0) {
                    sk.push_back({{"side", s >= 16 ? "compute" : "graphics"}, {"code", s & 15}, {"n", n}, {"ns", cover_skip_ns[s].load()}, {"sync_ns", cover_skip_sync[s].load()}});
                }
            }
            out["handle_without_entry_by_skip_reason_cumulative"] = std::move(sk);
            out["read_only_texel_handle_bindings_cumulative"] = {{"graphics", {{"checked", cover_texel[0].load()}, {"image_overlaps", cover_texel[1].load()}, {"is_image_metadata", cover_texel[2].load()}, {"gpu_written", cover_texel[3].load()}}},
                                                                 {"compute", {{"checked", cover_texel[4].load()}, {"image_overlaps", cover_texel[5].load()}, {"is_image_metadata", cover_texel[6].load()}, {"gpu_written", cover_texel[7].load()}}}};
        }
        out["note"] = "C0 census, counts per ~5 s interval (read only)";
        Core::DebugServer::Publish("consumer_census", std::move(out));
        if (Core::N0::On()) {
            Core::DebugServer::Publish("n0_census", Core::N0::Report());
        }
        if (Core::RowVersion::Mode() != 0) {
            Core::DebugServer::Publish("row_version", Core::RowVersion::Report()); // F2 row version shadow
        }
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
        nlohmann::json cuts = nlohmann::json::array();
        const u64 base = Core::NativeUpdates::ImageBase();
        for (u32 i = 0; i < Vulkan::CutReasonCensus::kSlots; ++i) {
            const u64 a = Vulkan::g_cut_reasons.address[i].load();
            if (a != 0) {
                char rva[24];
                std::snprintf(rva, sizeof(rva), "%llx", static_cast<unsigned long long>(a - base));
                cuts.push_back({{"caller_rva", rva}, {"instances", Vulkan::g_cut_reasons.instances[i].load()}, {"colour_instances", Vulkan::g_cut_reasons.colour_instances[i].load()}, {"draws", Vulkan::g_cut_reasons.draws[i].load()}});
            }
        }
        nlohmann::json flush_cuts = nlohmann::json::array();
        for (u32 i = 0; i < Vulkan::CutReasonCensus::kSlots; ++i) {
            const u64 a = Vulkan::g_flush_cuts.address[i].load();
            if (a != 0) {
                char rva[24];
                std::snprintf(rva, sizeof(rva), "%llx", static_cast<unsigned long long>(a - base));
                flush_cuts.push_back({{"caller_rva", rva}, {"flushes", Vulkan::g_flush_cuts.instances[i].load()}, {"image_barriers", Vulkan::g_flush_cuts.draws[i].load()}, {"with_memory_barrier", Vulkan::g_flush_cuts.colour_instances[i].load()}});
            }
        }
        nlohmann::json layouts = nlohmann::json::array();
        for (u32 i = 0; i < Vulkan::CutReasonCensus::kSlots; ++i) {
            const u64 a = Vulkan::g_flush_cut_layouts.address[i].load();
            if (a != 0) {
                layouts.push_back({{"old_layout", (a >> 12) & 0xfff}, {"new_layout", a & 0xfff}, {"barriers", Vulkan::g_flush_cut_layouts.instances[i].load()}});
            }
        }
        nlohmann::json changes = nlohmann::json::array();
        for (u32 m = 1; m < 0x200; ++m) {
            const u64 n = Vulkan::g_pass_changes.count[m].load();
            if (n != 0) {
                changes.push_back({{"difference_mask", m}, {"draws_that_ended_a_pass", n}});
            }
        }
        nlohmann::json repeat = nlohmann::json::object();
        {
            const auto& r = Vulkan::g_repeat_census;
            const char* names[2] = {"draws", "dispatches"};
            for (u32 k = 0; k < 2; ++k) {
                repeat[names[k]] = {{"total", r.total[k].load()}, {"shape_seen_in_previous_frame", r.in_previous_frame[k].load()}, {"shape_seen_earlier_this_frame", r.in_this_frame[k].load()}, {"new_shape", r.new_shape[k].load()},
                                    {"ring_bindings", r.ring_bindings[k].load()}, {"other_buffers", r.other_buffers[k].load()}, {"images", r.images[k].load()},
                                    {"repeated_draws_ring_bindings", r.repeat_ring_bindings[k].load()}, {"repeated_draws_other_buffers", r.repeat_other_buffers[k].load()}, {"repeated_draws_images", r.repeat_images[k].load()}};
            }
        }
        nlohmann::json key_census = nlohmann::json::object();
        {
            const auto& r = Vulkan::g_key_census;
            const char* names[2] = {"draws", "dispatches"};
            for (u32 k = 0; k < 2; ++k) {
                key_census[names[k]] = {{"total", r.total[k].load()}, {"no_complete_record", r.no_record[k].load()}, {"keyed", r.keyed[k].load()}, {"key_seen_before", r.key_seen[k].load()},
                                        {"same_shape_as_last_time", r.same_shape[k].load()}, {"other_shape", r.other_shape[k].load()}, {"differ_in_buffers_only", r.differ_buffers_only[k].load()},
                                        {"differ_in_images_only", r.differ_images_only[k].load()}, {"differ_in_both", r.differ_both[k].load()}};
            }
        }
        const auto& sp = Vulkan::g_same_pass_flushes;
        nlohmann::json same_pass = {{"flushes_that_end_a_pass_the_draw_would_keep", sp.flushes.load()}, {"image_barriers_in_them", sp.image_barriers.load()}, {"with_a_memory_barrier", sp.with_memory_barrier.load()},
                                    {"flushes_with_a_pass_open_where_the_draw_changes_pass_anyway", sp.other_pass_changes.load()}, {"note", "Rasterizer::Draw only, other callers count as 'anyway'"}};
        Core::DebugServer::Publish("segment_census", {{"by_kind_since_start", std::move(kinds)}, {"flush_in_same_pass", std::move(same_pass)}, {"draw_shape_repeat", std::move(repeat)}, {"draw_key_census", std::move(key_census)}, {"pass_state_differences", std::move(changes)}, {"image_barriers_with_a_pass_open_by_layouts", std::move(layouts)}, {"ended_by_caller_since_start", std::move(cuts)}, {"flush_barriers_with_a_pass_open_by_caller", std::move(flush_cuts)}, {"note", "C6 census: commands and draws recorded between a begin and an end of rendering, graphics stream, counts only"}});
    }
    if (IndependentGroup::Mode() != 0) {
        Core::DebugServer::Publish("sim_independent", nlohmann::json::parse(IndependentGroup::Report()));
    }
    if (BarrierCensus::On()) {
        Core::DebugServer::Publish("barrier_census", nlohmann::json::parse(BarrierCensus::Report()));
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

    const auto native = TakeNativeStats();
    result.native_tagged = native.tagged;
    result.native_untagged = native.untagged;
    result.native_dynamic_skipped = native.dynamic_skipped;
    result.native_bind_skipped = native.bind_skipped;
    return result;
}

} // namespace Vulkan
