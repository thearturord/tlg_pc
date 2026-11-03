// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <bit>
#include <cstdlib>
#include <string>
#include <nlohmann/json.hpp>
#include "native/native_field_use.h"

namespace Core::NativeFieldUse {

namespace {
constexpr u64 Bits(u64 value, u32 from, u32 count) {
    return (value >> from) & ((1ull << count) - 1);
}
u64 Word64(const u32* words, u32 index) {
    return u64{words[index]} | (u64{words[index + 1]} << 32);
}
constexpr u32 IdentityDstSel = 4u | (5u << 3) | (6u << 6) | (7u << 9);

// T# (AmdGpu::Image, resource.h): qword 0 base_address 38, mtype_l2 2, min_lod 12, data_format 6, num_format 4, mtype 2; qword 1 width 14, height 14, perf_modulation 3, interlaced 1,
// dst_sel x y z w 3 bits each (32), base_level 4 (44), last_level 4 (48), tiling_index 5 (52), pow2pad 1, mtype2 1, atc 1, type 4 (60); qword 2 depth 13, pitch 14, 5, base_array 13 (32),
// last_array 13 (45), 6; qword 3 min_lod_warn 12, counter_bank_id 8 (12), lod_hw_cnt_en (20), compression_en (21), alpha_is_on_msb (22), color_transform (23), alt_tile_mode (24).
std::atomic<u64> c_images{0};
std::atomic<u64> c_image_alt_tile{0}, c_image_samples_over_1{0}, c_image_min_lod_nonzero{0}, c_image_base_level_nonzero{0}, c_image_base_array_nonzero{0}, c_image_mtype_l2{0}, c_image_mtype{0},
    c_image_mtype2{0}, c_image_perf_mod{0}, c_image_interlaced{0}, c_image_atc{0}, c_image_min_lod_warn{0}, c_image_counter_bank{0}, c_image_lod_hw_cnt{0}, c_image_compression{0},
    c_image_alpha_msb{0}, c_image_color_transform{0}, c_image_pow2pad{0}, c_image_dst_sel_not_identity{0}, c_image_upper_nonzero{0};
std::atomic<u64> c_image_type[16]{}, c_image_tiling[32]{}, c_image_dfmt[64]{}, c_image_nfmt[16]{}, c_image_dst_sel[4096]{};

// S# (AmdGpu::Sampler): qword 0 clamp_x/y/z 3 bits each, max_aniso (9), depth_compare (12), force_unnormalized (15), aniso_threshold 3 (16), mc_coord_trunc (19), force_degamma (20),
// aniso_bias 6 (21), trunc_coord (27), disable_cube_wrap (28), filter_mode 2 (29), min_lod 12 (32), max_lod 12 (44), perf_mip 4 (56), perf_z 4 (60); qword 1 lod_bias 14, lod_bias_sec 6 (14),
// xy_mag 2 (20), xy_min 2 (22), z_filter 2 (24), mip_filter 2 (26), mip_point_preclamp (28), disable_lsb_ceil (29), unused 2 (30), border_color_ptr 12 (32), unused 18 (42), border_color_type 2 (62).
std::atomic<u64> c_samplers{0};
std::atomic<u64> c_sampler_border_type[4]{}, c_sampler_clamp_x[8]{}, c_sampler_clamp_y[8]{}, c_sampler_clamp_z[8]{}, c_sampler_aniso[8]{}, c_sampler_compare[8]{}, c_sampler_filter_mode[4]{},
    c_sampler_mag[4]{}, c_sampler_min[4]{}, c_sampler_mip[4]{}, c_sampler_z[4]{};
std::atomic<u64> c_sampler_force_unnormalized{0}, c_sampler_aniso_threshold{0}, c_sampler_mc_trunc{0}, c_sampler_degamma{0}, c_sampler_aniso_bias{0}, c_sampler_trunc_coord{0},
    c_sampler_cube_wrap{0}, c_sampler_min_lod_nonzero{0}, c_sampler_max_lod_not_max{0}, c_sampler_perf_mip{0}, c_sampler_perf_z{0}, c_sampler_lod_bias_nonzero{0}, c_sampler_lod_bias_sec{0},
    c_sampler_preclamp{0}, c_sampler_lsb_ceil{0}, c_sampler_unused{0}, c_sampler_border_ptr_nonzero{0};

// V# (AmdGpu::Buffer): qword 0 base_address 40, padding 8 (40), stride 14 (48), cache_swizzle (62), swizzle_enable (63); dword 2 num_records; dword 3 dst_sel 4x3 (0), num_format 3 (12),
// data_format 4 (15), element_size 2 (19), index_stride 2 (21), add_tid_enable (23), padding 3 (24), mtype 3 (27), type 2 (30).
std::atomic<u64> c_buffers{0};
std::atomic<u64> c_buffer_padding0{0}, c_buffer_cache_swizzle{0}, c_buffer_swizzle_enable{0}, c_buffer_add_tid{0}, c_buffer_padding1{0}, c_buffer_mtype{0}, c_buffer_type_nonzero{0},
    c_buffer_dst_sel_not_identity{0};
std::atomic<u64> c_buffer_dfmt[16]{}, c_buffer_nfmt[8]{}, c_buffer_stride_width[16]{}, c_buffer_element_size[4]{}, c_buffer_index_stride[4]{};

void Bump(std::atomic<u64>& c, bool condition) {
    if (condition) {
        c.fetch_add(1, std::memory_order_relaxed);
    }
}
void Count(std::atomic<u64>& c) {
    c.fetch_add(1, std::memory_order_relaxed);
}
nlohmann::json Histogram(const std::atomic<u64>* values, u32 count) {
    nlohmann::json out = nlohmann::json::object();
    for (u32 i = 0; i < count; ++i) {
        if (const u64 n = values[i].load(std::memory_order_relaxed); n != 0) {
            out[std::to_string(i)] = n;
        }
    }
    return out;
}
} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_FIELD_USE");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void NoteImage(const u32* w) {
    if (!On()) {
        return;
    }
    const u64 q0 = Word64(w, 0), q1 = Word64(w, 2), q2 = Word64(w, 4), q3 = Word64(w, 6);
    Count(c_images);
    Bump(c_image_mtype_l2, Bits(q0, 38, 2) != 0);
    Bump(c_image_min_lod_nonzero, Bits(q0, 40, 12) != 0);
    Count(c_image_dfmt[Bits(q0, 52, 6)]);
    Count(c_image_nfmt[Bits(q0, 58, 4)]);
    Bump(c_image_mtype, Bits(q0, 62, 2) != 0);
    Bump(c_image_perf_mod, Bits(q1, 28, 3) != 0);
    Bump(c_image_interlaced, Bits(q1, 31, 1) != 0);
    const u32 dst = static_cast<u32>(Bits(q1, 32, 12));
    Count(c_image_dst_sel[dst]);
    Bump(c_image_dst_sel_not_identity, dst != IdentityDstSel);
    Bump(c_image_base_level_nonzero, Bits(q1, 44, 4) != 0);
    Count(c_image_tiling[Bits(q1, 52, 5)]);
    Bump(c_image_pow2pad, Bits(q1, 57, 1) != 0);
    Bump(c_image_mtype2, Bits(q1, 58, 1) != 0);
    Bump(c_image_atc, Bits(q1, 59, 1) != 0);
    Count(c_image_type[Bits(q1, 60, 4)]);
    Bump(c_image_base_array_nonzero, Bits(q2, 32, 13) != 0);
    Bump(c_image_min_lod_warn, Bits(q3, 0, 12) != 0);
    Bump(c_image_counter_bank, Bits(q3, 12, 8) != 0);
    Bump(c_image_lod_hw_cnt, Bits(q3, 20, 1) != 0);
    Bump(c_image_compression, Bits(q3, 21, 1) != 0);
    Bump(c_image_alpha_msb, Bits(q3, 22, 1) != 0);
    Bump(c_image_color_transform, Bits(q3, 23, 1) != 0);
    Bump(c_image_alt_tile, Bits(q3, 24, 1) != 0);
    Bump(c_image_upper_nonzero, Bits(q3, 25, 39) != 0);
}

void NoteHostImageFields(u32 alt_tile, u32 samples) {
    if (!On()) {
        return;
    }
    Bump(c_image_samples_over_1, samples > 1);
    (void)alt_tile;
}

void NoteSampler(const u32* w) {
    if (!On()) {
        return;
    }
    const u64 q0 = Word64(w, 0), q1 = Word64(w, 2);
    Count(c_samplers);
    Count(c_sampler_clamp_x[Bits(q0, 0, 3)]);
    Count(c_sampler_clamp_y[Bits(q0, 3, 3)]);
    Count(c_sampler_clamp_z[Bits(q0, 6, 3)]);
    Count(c_sampler_aniso[Bits(q0, 9, 3)]);
    Count(c_sampler_compare[Bits(q0, 12, 3)]);
    Bump(c_sampler_force_unnormalized, Bits(q0, 15, 1) != 0);
    Bump(c_sampler_aniso_threshold, Bits(q0, 16, 3) != 0);
    Bump(c_sampler_mc_trunc, Bits(q0, 19, 1) != 0);
    Bump(c_sampler_degamma, Bits(q0, 20, 1) != 0);
    Bump(c_sampler_aniso_bias, Bits(q0, 21, 6) != 0);
    Bump(c_sampler_trunc_coord, Bits(q0, 27, 1) != 0);
    Bump(c_sampler_cube_wrap, Bits(q0, 28, 1) != 0);
    Count(c_sampler_filter_mode[Bits(q0, 29, 2)]);
    Bump(c_sampler_min_lod_nonzero, Bits(q0, 32, 12) != 0);
    Bump(c_sampler_max_lod_not_max, Bits(q0, 44, 12) != 4095);
    Bump(c_sampler_perf_mip, Bits(q0, 56, 4) != 0);
    Bump(c_sampler_perf_z, Bits(q0, 60, 4) != 0);
    Bump(c_sampler_lod_bias_nonzero, Bits(q1, 0, 14) != 0);
    Bump(c_sampler_lod_bias_sec, Bits(q1, 14, 6) != 0);
    Count(c_sampler_mag[Bits(q1, 20, 2)]);
    Count(c_sampler_min[Bits(q1, 22, 2)]);
    Count(c_sampler_z[Bits(q1, 24, 2)]);
    Count(c_sampler_mip[Bits(q1, 26, 2)]);
    Bump(c_sampler_preclamp, Bits(q1, 28, 1) != 0);
    Bump(c_sampler_lsb_ceil, Bits(q1, 29, 1) != 0);
    Bump(c_sampler_unused, Bits(q1, 30, 2) != 0 || Bits(q1, 42, 20) != 0);
    Bump(c_sampler_border_ptr_nonzero, Bits(q1, 32, 12) != 0);
    Count(c_sampler_border_type[Bits(q1, 62, 2)]);
}

void NoteBuffer(const u32* w) {
    if (!On()) {
        return;
    }
    const u64 q0 = Word64(w, 0);
    const u32 d3 = w[3];
    Count(c_buffers);
    Bump(c_buffer_padding0, Bits(q0, 40, 8) != 0);
    Count(c_buffer_stride_width[std::bit_width(Bits(q0, 48, 14)) & 15]);
    Bump(c_buffer_cache_swizzle, Bits(q0, 62, 1) != 0);
    Bump(c_buffer_swizzle_enable, Bits(q0, 63, 1) != 0);
    Bump(c_buffer_dst_sel_not_identity, (d3 & 0xfff) != IdentityDstSel);
    Count(c_buffer_nfmt[(d3 >> 12) & 7]);
    Count(c_buffer_dfmt[(d3 >> 15) & 15]);
    Count(c_buffer_element_size[(d3 >> 19) & 3]);
    Count(c_buffer_index_stride[(d3 >> 21) & 3]);
    Bump(c_buffer_add_tid, ((d3 >> 23) & 1) != 0);
    Bump(c_buffer_padding1, ((d3 >> 24) & 7) != 0);
    Bump(c_buffer_mtype, ((d3 >> 27) & 7) != 0);
    Bump(c_buffer_type_nonzero, ((d3 >> 30) & 3) != 0);
}

nlohmann::json Report() {
    nlohmann::json out;
    out["on"] = On();
    out["switch"] = "TLG_FIELD_USE=1";
    const auto v = [](const std::atomic<u64>& c) { return c.load(std::memory_order_relaxed); };
    out["images"] = {{"bound", v(c_images)},
                     {"non_default", {{"alt_tile_mode", v(c_image_alt_tile)}, {"num_samples_over_1", v(c_image_samples_over_1)}, {"min_lod", v(c_image_min_lod_nonzero)},
                                      {"base_level", v(c_image_base_level_nonzero)}, {"base_array", v(c_image_base_array_nonzero)}, {"mtype_l2", v(c_image_mtype_l2)}, {"mtype", v(c_image_mtype)},
                                      {"mtype2", v(c_image_mtype2)}, {"perf_modulation", v(c_image_perf_mod)}, {"interlaced", v(c_image_interlaced)}, {"atc", v(c_image_atc)},
                                      {"min_lod_warn", v(c_image_min_lod_warn)}, {"counter_bank_id", v(c_image_counter_bank)}, {"lod_hw_cnt_en", v(c_image_lod_hw_cnt)},
                                      {"compression_en", v(c_image_compression)}, {"alpha_is_on_msb", v(c_image_alpha_msb)}, {"color_transform", v(c_image_color_transform)},
                                      {"pow2pad", v(c_image_pow2pad)}, {"dst_sel_not_identity", v(c_image_dst_sel_not_identity)}, {"bits_25_to_63_of_qword_3", v(c_image_upper_nonzero)}}},
                     {"type", Histogram(c_image_type, 16)},
                     {"tiling_index", Histogram(c_image_tiling, 32)},
                     {"data_format", Histogram(c_image_dfmt, 64)},
                     {"num_format", Histogram(c_image_nfmt, 16)},
                     {"dst_sel_packed_3_bits_each_xyzw", Histogram(c_image_dst_sel, 4096)}};
    out["samplers_created"] = {
        {"count", v(c_samplers)},
        {"border_color_type", Histogram(c_sampler_border_type, 4)},
        {"clamp_x", Histogram(c_sampler_clamp_x, 8)},
        {"clamp_y", Histogram(c_sampler_clamp_y, 8)},
        {"clamp_z", Histogram(c_sampler_clamp_z, 8)},
        {"max_aniso", Histogram(c_sampler_aniso, 8)},
        {"depth_compare", Histogram(c_sampler_compare, 8)},
        {"filter_mode", Histogram(c_sampler_filter_mode, 4)},
        {"xy_mag", Histogram(c_sampler_mag, 4)},
        {"xy_min", Histogram(c_sampler_min, 4)},
        {"z_filter", Histogram(c_sampler_z, 4)},
        {"mip_filter", Histogram(c_sampler_mip, 4)},
        {"non_default", {{"border_ptr", v(c_sampler_border_ptr_nonzero)}, {"force_unnormalized", v(c_sampler_force_unnormalized)}, {"aniso_threshold", v(c_sampler_aniso_threshold)},
                         {"mc_coord_trunc", v(c_sampler_mc_trunc)}, {"force_degamma", v(c_sampler_degamma)}, {"aniso_bias", v(c_sampler_aniso_bias)}, {"trunc_coord", v(c_sampler_trunc_coord)},
                         {"disable_cube_wrap", v(c_sampler_cube_wrap)}, {"min_lod", v(c_sampler_min_lod_nonzero)}, {"max_lod_below_4095", v(c_sampler_max_lod_not_max)},
                         {"perf_mip", v(c_sampler_perf_mip)}, {"perf_z", v(c_sampler_perf_z)}, {"lod_bias", v(c_sampler_lod_bias_nonzero)}, {"lod_bias_sec", v(c_sampler_lod_bias_sec)},
                         {"mip_point_preclamp", v(c_sampler_preclamp)}, {"disable_lsb_ceil", v(c_sampler_lsb_ceil)}, {"unused_bits", v(c_sampler_unused)}}}};
    out["buffers"] = {{"bound", v(c_buffers)},
                      {"non_default", {{"padding0", v(c_buffer_padding0)}, {"cache_swizzle", v(c_buffer_cache_swizzle)}, {"swizzle_enable", v(c_buffer_swizzle_enable)},
                                       {"add_tid_enable", v(c_buffer_add_tid)}, {"padding1", v(c_buffer_padding1)}, {"mtype", v(c_buffer_mtype)}, {"type_bits", v(c_buffer_type_nonzero)},
                                       {"dst_sel_not_identity", v(c_buffer_dst_sel_not_identity)}}},
                      {"data_format", Histogram(c_buffer_dfmt, 16)},
                      {"num_format", Histogram(c_buffer_nfmt, 8)},
                      {"element_size", Histogram(c_buffer_element_size, 4)},
                      {"index_stride", Histogram(c_buffer_index_stride, 4)},
                      {"stride_bit_width", Histogram(c_buffer_stride_width, 16)}};
    return out;
}

} // namespace Core::NativeFieldUse
