// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_shader_env.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "video_core/amdgpu/liverpool.h"

namespace Core::NativeShaderEnv {

namespace {

thread_local u64 g_vertex_gnm = 0;
thread_local u64 g_pixel_gnm = 0;
thread_local u64 g_fetch_object = 0;
thread_local bool g_active = false;

std::atomic<u64> g_compared[2]{}, g_differing[2]{}, g_unavailable{};
std::mutex g_mutex;
std::array<std::atomic<u64>, 24> g_field_counts{};
std::array<std::string, 24> g_field_names;
std::atomic<u32> g_samples{};

template <typename T>
T Rd(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

/// The program header: address and the two resource words, decoded with the console's own bit layout.
void FillProgram(ShaderEnv::Program& out, u32 address_lo, u32 rsrc1, u32 rsrc2) {
    AmdGpu::ShaderProgram program{};
    program.address = address_lo;
    static_assert(sizeof(program.settings) == 2 * sizeof(u32));
    const u32 words[2] = {rsrc1, rsrc2};
    std::memcpy(&program.settings, words, sizeof(words));
    out.address = static_cast<u64>(address_lo) << 8;
    out.vgprs = static_cast<u16>(program.NumVgprs());
    out.user_data_count = static_cast<u8>(program.settings.num_user_regs);
    out.input_vgpr_components = static_cast<u8>(program.settings.vgpr_comp_cnt);
    out.fp_denorm_mode32 = static_cast<u8>(program.settings.fp_denorm_mode32);
    out.fp_denorm_mode16_64 = static_cast<u8>(program.settings.fp_denorm_mode64);
    out.fp_round_mode32 = static_cast<u8>(program.settings.fp_round_mode32);
    out.fp_round_mode16_64 = static_cast<u8>(program.settings.fp_round_mode64);
    out.dx10_clamp = program.settings.dx10_clamp != 0;
    out.offchip_lds = program.settings.oc_lds_en != 0;
}

void BuildVertex(ShaderEnv::ShaderEnv& env, u64 gnm, u64 fetch_object) {
    const u64 regs = gnm + 8;
    env.stage = ShaderEnv::Stage::Vertex;
    // PrepareDraw merges the fetch object's modifier into the first resource word when it is not zero
    const u32 modifier = fetch_object ? Rd<u32>(fetch_object, 0x40) : 0;
    const u32 rsrc1 = Rd<u32>(regs, 8);
    FillProgram(env.program, Rd<u32>(regs, 0), modifier != 0 ? (rsrc1 & 0xfcfffc3fu) | modifier : rsrc1, Rd<u32>(regs, 12));
    env.exports.vs_out_control = Rd<u32>(regs, 24);
    // The step rates of the instanced attributes. The layout object keeps 0 for "no divisor"; the hardware holds the frame's default, 1 (a divisor of 1 and no
    // divisor give the same instance index, but the translator keys its variants on the value).
    env.rates.instance_step_rate_0 = 1;
    env.rates.instance_step_rate_1 = 1;
    if (fetch_object) {
        if (const u32 rate = Rd<u32>(fetch_object, 0x48); rate != 0) {
            env.rates.instance_step_rate_0 = rate;
        }
        if (const u32 rate = Rd<u32>(fetch_object, 0x4c); rate != 0) {
            env.rates.instance_step_rate_1 = rate;
        }
    }
}

void BuildPixel(ShaderEnv::ShaderEnv& env, u64 gnm, u64 vertex_gnm) {
    const u64 regs = gnm + 8;
    env.stage = ShaderEnv::Stage::Fragment;
    FillProgram(env.program, Rd<u32>(regs, 0), Rd<u32>(regs, 8), Rd<u32>(regs, 12));
    auto& f = env.fragment;
    const u32 z_format = Rd<u32>(regs, 16);
    const u32 col_format = Rd<u32>(regs, 20);
    f.z_export_format = static_cast<u8>(z_format & 0xf);
    for (u32 i = 0; i < 8; ++i) {
        f.color_export_format[i] = static_cast<u8>((col_format >> (i * 4)) & 0xf);
    }
    f.input_enable = Rd<u32>(regs, 24);
    f.input_address = Rd<u32>(regs, 28);
    f.num_interpolants = static_cast<u8>(Rd<u32>(regs, 32) & 0x3f);
    f.front_face_all_bits = ((Rd<u32>(regs, 36) >> 24) & 1) != 0;
    const auto depth = std::bit_cast<AmdGpu::DepthShaderControl>(Rd<u32>(regs, 40));
    f.depth_before_shader = depth.depth_before_shader != 0;
    const u8 stencil_ref_export = static_cast<u8>(depth.stencil_op_val_export_enable | depth.stencil_test_val_export_enable);
    f.depth_export_mask = static_cast<u8>(depth.z_export_enable | (stencil_ref_export << 1) | (depth.mask_export_enable << 2) | (depth.coverage_to_mask_enable << 3));
    f.sample_coverage = ((f.input_enable >> 14) & 1) != 0 && ((f.input_address >> 14) & 1) != 0;

    // SPI_PS_INPUT_CNTL_n: the interpolant table that follows the shader's input usage table, each entry matched with the vertex shader's output semantics
    const u64 interpolants = gnm + 0x3c + u64(Rd<u8>(gnm, 3)) * 4;
    if (vertex_gnm == 0) {
        return;
    }
    const u64 vertex_semantics = vertex_gnm + u64(Rd<u8>(vertex_gnm, 3)) * 4 + 0x28 + u64(Rd<u8>(vertex_gnm, 0x24)) * 4;
    const u32 vertex_semantic_count = Rd<u8>(vertex_gnm, 0x25);
    for (u32 k = 0; k < Rd<u8>(gnm, 0x38) && k < 32; ++k) {
        const u32 entry = Rd<u16>(interpolants, k * 2);
        bool found = false;
        u32 offset = 0;
        for (u32 j = 0; j < vertex_semantic_count && !found; ++j) {
            if (Rd<u8>(vertex_semantics, j * 2) == (entry & 0xff)) {
                found = true;
                offset = Rd<u8>(vertex_semantics, j * 2 + 1) & 0x1f;
            }
        }
        u32 value = 0x20;
        if (found) {
            value = ((entry >> 7) & 0x20) | offset;
            value |= (((entry >> 12) | (entry >> 10)) & 1) << 10;
        }
        const u32 reg = ((entry >> 8) & 3) << 8 | value;
        auto& in = f.interpolants[k];
        in.input_offset = static_cast<u8>(reg & 0x1f);
        in.use_default = ((reg >> 5) & 1) != 0;
        in.default_value = static_cast<u8>((reg >> 8) & 3);
        in.flat_shade = ((reg >> 10) & 1) != 0;
    }
}

} // namespace

Scope::Scope(u64 vertex_gnm, u64 pixel_gnm, u64 fetch_object) {
    g_vertex_gnm = vertex_gnm;
    g_pixel_gnm = pixel_gnm;
    g_fetch_object = fetch_object;
    g_active = true;
}

Scope::~Scope() {
    g_active = false;
}

bool Current(DrawEnv& out) {
    if (!g_active) {
        return false;
    }
    out = {};
    if (g_vertex_gnm != 0) {
        BuildVertex(out.vertex, g_vertex_gnm, g_fetch_object);
        out.has_vertex = true;
    }
    if (g_pixel_gnm != 0) {
        BuildPixel(out.pixel, g_pixel_gnm, g_vertex_gnm);
        out.has_pixel = true;
    }
    return true;
}

void NoteCompared(u32 stage, const std::string& differing) {
    stage &= 1;
    g_compared[stage].fetch_add(1, std::memory_order_relaxed);
    if (differing.empty()) {
        return;
    }
    g_differing[stage].fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{g_mutex};
        // count by field name (the names are separated by spaces)
        size_t at = 0;
        while (at < differing.size()) {
            const size_t end = differing.find(' ', at);
            const std::string name = differing.substr(at, end == std::string::npos ? std::string::npos : end - at);
            at = end == std::string::npos ? differing.size() : end + 1;
            if (name.empty()) {
                continue;
            }
            for (size_t i = 0; i < g_field_names.size(); ++i) {
                if (g_field_names[i] == name || g_field_names[i].empty()) {
                    g_field_names[i] = name;
                    g_field_counts[i].fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        }
    }
    if (g_samples.fetch_add(1, std::memory_order_relaxed) < 8) {
        Core::RareEvents::Note("shader_env_differs_from_registers", fmt::format("{} stage: {}", stage == 0 ? "vertex" : "pixel", differing));
    }
}

bool Decides() noexcept {
    return true; // played by the owner 2026-10-31, no switch any more
}

void NoteUnavailable() {
    g_unavailable.fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    nlohmann::json j;
    j["shader_env_decides"] = Decides();
    j["vertex_compared"] = g_compared[0].load();
    j["vertex_differing"] = g_differing[0].load();
    j["pixel_compared"] = g_compared[1].load();
    j["pixel_differing"] = g_differing[1].load();
    j["runtime_info_built_without_a_native_draw"] = g_unavailable.load();
    std::scoped_lock lock{g_mutex};
    auto& fields = j["differing_fields"] = nlohmann::json::object();
    for (size_t i = 0; i < g_field_names.size(); ++i) {
        if (!g_field_names[i].empty()) {
            fields[g_field_names[i]] = g_field_counts[i].load();
        }
    }
    return j;
}

} // namespace Core::NativeShaderEnv
