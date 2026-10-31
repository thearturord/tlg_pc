// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <span>

#include "common/types.h"

/// E8 (documents/tlg/native plan/shader-boundary.md): the contract between the engine's world and the shader translator. The GPU register file is not an input of the
/// translator and the pipeline cache. What they need comes in two parts, both defined here:
///
///  1. `ShaderEnv`: what belongs to the shader itself, its own ABI. The console's shader file carries a block of register words per stage (the program address, the
///     resource words with the register counts, the float modes and the user data count, the export and input-enable words, the thread counts of a compute shader, the
///     tessellation and geometry configuration). The game copies that block into registers when it sets the shader; here it is read where it lives, the shader's header.
///     These words stay in the design: they are the shader's interface, as the shader code itself is.
///  2. `FixedFunctionForShader`: the few pieces of fixed-function state the translator needs to compile a shader (the formats and swizzle of the render targets it exports
///     to, whether blending reads two sources, the primitive type, the clip state). They are state of the draw, not of the shader, and come from the draw's native state
///     (E1), not from a register image.
///
/// Rule (checked by `native/check_native_deps.py` once E1 and E2 are done): the translator (shader_recompiler/) and the pipeline cache include this header and
/// the shader-binary headers, and nothing of the register file (regs_*.h, `Liverpool::Regs`).
///
/// The plain integer widths below are the console's register fields as they are today; E2 maps the formats to the Vulkan side in one place. Numbers in comments name the
/// field's width in bits.
namespace Core::ShaderEnv {

/// The hardware stages, in the order the translator's `HwStage` has them.
enum class Stage : u8 { Fragment, Vertex, Geometry, Export, Hull, Local, Compute };

/// What every stage has: where the program is and how it asks to be run (SPI_SHADER_PGM_LO/HI, PGM_RSRC1, PGM_RSRC2; COMPUTE_PGM_* for compute).
struct Program {
    u64 address{};                    ///< guest address of the code
    u64 hash{};                       ///< hash of the code (the pipeline key's stage hash)
    std::span<const u32> code{};      ///< the code itself (up to the end-of-program, found by the translator's own scan)
    u8 vgprs{};                       ///< allocated vector registers (granule count * 4)
    u8 user_data_count{};             ///< 5 bits: how many of the 16 user data words the shader reads
    u8 input_vgpr_components{};       ///< 2 bits: vertex id, instance id, ... the stage's VGPR inputs
    u8 fp_denorm_mode32{};            ///< 2 bits
    u8 fp_denorm_mode16_64{};         ///< 2 bits
    u8 fp_round_mode32{};             ///< 2 bits
    u8 fp_round_mode16_64{};          ///< 2 bits
    bool dx10_clamp{};
    bool offchip_lds{};               ///< hull: the control point output goes to off chip LDS (PGM_RSRC2.OC_LDS_EN)
    u32 shared_memory_size{};         ///< compute: bytes of LDS (PGM_RSRC2.LDS_SIZE)
    std::array<bool, 3> tgid_enable{}; ///< compute: the workgroup id registers it reads
};

/// The user data of one draw or dispatch: the words the game wrote into the stage's SGPRs (descriptors, pointers, immediates). Shader ABI, as the code is.
struct UserData {
    std::array<u32, 16> words{};
    std::span<const u32> extended{}; ///< the flat user data beyond the 16 words (the engine's constant tables)
};

/// Vertex and geometry exports (PA_CL_VS_OUT_CNTL, 32 bits): which of point size, edge flag, kill flag, GS cut flag, render target index, viewport index and the clip and
/// cull distances the shader writes.
struct VertexExports {
    u32 vs_out_control{};
};

struct VertexInputRates {
    u32 instance_step_rate_0{}; ///< VGT_INSTANCE_STEP_RATE_0 / 1: the divisors of the instanced vertex attributes
    u32 instance_step_rate_1{};
};

struct DrawParameters {
    u16 vertex_sgpr_offset{};   ///< where the draw's base vertex and base instance reach the shader (indirect draws)
    u16 instance_sgpr_offset{};
    bool indirect{};
};

/// Fragment shader inputs and exports (SPI_PS_INPUT_ENA / ADDR, SPI_PS_IN_CONTROL, SPI_PS_INPUT_CNTL_n, SPI_BARYC_CNTL, DB_SHADER_CONTROL, SPI_SHADER_Z_FORMAT,
/// SPI_SHADER_COL_FORMAT).
struct FragmentIo {
    u32 input_enable{};
    u32 input_address{};
    u8 num_interpolants{};
    struct Interpolant {
        u8 input_offset{};      ///< 5 bits: which vertex shader parameter
        bool use_default{};
        bool flat_shade{};
        u8 default_value{};     ///< 2 bits
    };
    std::array<Interpolant, 32> interpolants{};
    bool front_face_all_bits{};
    bool depth_before_shader{};
    u8 z_export_format{};       ///< 4 bits
    u8 depth_export_mask{};     ///< z export, stencil ref export, mask export, coverage to mask (4 bits)
    std::array<u8, 8> color_export_format{}; ///< 4 bits for each of the 8 render target slots
    bool sample_coverage{};     ///< SPI_PS_INPUT_ENA and ADDR both name the sample coverage input
};

/// Hull and domain (tessellation) configuration (VGT_LS_HS_CONFIG, VGT_TF_PARAM).
struct Tessellation {
    u8 input_control_points{};  ///< 6 bits
    u8 output_control_points{}; ///< 6 bits: threads of the hull shader
    u8 type{};                  ///< 2 bits: isoline, triangle, quad
    u8 topology{};              ///< 3 bits
    u8 partitioning{};          ///< 3 bits
    u32 local_stride{};         ///< from the engine's tessellation constant buffer, which sits in the shader's user data
};

/// Export and geometry stage configuration (VGT_ESGS_RING_ITEMSIZE, VGT_GS_MAX_VERT_OUT, VGT_GS_INSTANCE_CNT, VGT_GS_OUT_PRIM_TYPE, VGT_GSVS_RING_ITEMSIZE, VGT_GS_MODE).
struct GeometryConfig {
    u32 esgs_ring_item_size{};
    u32 max_output_vertices{};
    u32 instance_count{};
    std::array<u8, 4> out_primitive{}; ///< per stream
    u32 out_vertex_item_size{};        ///< stream 0
    u8 mode{};                         ///< 3 bits
    bool on_chip{};                    ///< not supported by the translator
    bool stream_out{};                 ///< not supported by the translator
    Program copy_shader{};             ///< the vertex shader that copies the geometry shader's output (its own program)
};

/// Compute: the thread counts of a workgroup (COMPUTE_NUM_THREAD_X / Y / Z).
struct ComputeConfig {
    std::array<u32, 3> workgroup_size{};
};

/// The stages the pipeline uses (VGT_SHADER_STAGES_EN): which of local, hull, export, geometry run before the pixel shader.
struct StageSet {
    u32 raw{};
    [[nodiscard]] constexpr bool Has(Stage stage) const noexcept {
        return (raw & (1u << static_cast<u32>(stage))) != 0;
    }
};

/// Everything the translator takes from the shader's own world for one hardware stage.
struct ShaderEnv {
    Stage stage{};
    Program program{};
    UserData user_data{};
    StageSet stages{};           ///< the pipeline's stage set (every stage sees it)
    VertexExports exports{};     ///< vertex, export and geometry stages
    VertexInputRates rates{};    ///< vertex stages that read instanced attributes
    DrawParameters draw{};       ///< the vertex stage that reads base vertex and base instance
    FragmentIo fragment{};       ///< the pixel stage
    Tessellation tessellation{}; ///< hull and domain
    GeometryConfig geometry{};   ///< export and geometry
    ComputeConfig compute{};     ///< compute
};

/// The fixed-function state of the draw that reaches the translator today (read from `Liverpool::Regs` and `FixedState` by the pipeline cache). Source: the draw's native
/// state (E1). Nothing else of the register file may reach the translator.
struct FixedFunctionForShader {
    u8 primitive_type{};          ///< the console's primitive type number: the geometry shader's input and the rect / quad list emulation read it
    u32 user_clip_plane_mask{};   ///< PA_CL_CLIP_CNTL: user clip planes the vertex stage lowers
    bool clip_disabled{};
    bool negative_one_to_one{};   ///< the clip space depth range (PA_CL_VTE_CNTL); emulated when the driver cannot do it
    bool dual_source_blending{};  ///< the blend state reads the second pixel shader output
    u8 num_samples{1};            ///< samples of the render pass the pixel shader writes into
    struct ColorTarget {
        u8 data_format{};         ///< 6 bits, as CB_COLOR_INFO.FORMAT
        u8 number_format{};       ///< 4 bits
        u8 number_conversion{};   ///< 3 bits
        u16 swizzle{};            ///< the component mapping (4 fields)
        bool blend_self_scale{};  ///< min / max blending with source color on both sides: the shader must compensate
    };
    std::array<ColorTarget, 8> color_targets{};
};

} // namespace Core::ShaderEnv
