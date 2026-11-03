// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <string>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "native/shader_env.h"

/// E1c (documents/tlg/native plan/no-emulation-plan.md section 17): the shader environment of a draw, read from the engine's shader objects instead of from the
/// register image. A pixel or vertex shader object holds a pointer (object + 0x120) to the shader's Gnm block; the block's register part (block + 8) is what the
/// game copies into SPI_SHADER_*, SPI_PS_*, DB_SHADER_CONTROL and PA_CL_VS_OUT_CNTL when it binds the shader (`Synthesize` does exactly that copy, in register
/// numbers). The input layout's fetch object holds the vertex attribute step rates. This module reads the same words and presents them as `ShaderEnv` (shader_env.h).
///
/// Rollout: the translator's input `Shader::RuntimeInfo` is still built from the registers; every time it is built for a native draw it is also derived from the
/// `ShaderEnv` and compared field by field (`/frame_events` section shader_env). Native decides after a clean long run and the owner's play; then the register
/// reads of `BuildRuntimeInfo` and the shader words of `Synthesize` go.
namespace Core::NativeShaderEnv {

/// What a draw has: the vertex and pixel stage environments (the draws the native path executes are vertex plus pixel).
struct DrawEnv {
    ShaderEnv::ShaderEnv vertex{};
    ShaderEnv::ShaderEnv pixel{};
    bool has_vertex{};
    bool has_pixel{};
    /// F3.2: a hull, domain or geometry shader is bound in the device context (the native pipeline is vertex plus pixel only; such a draw is a counted refusal)
    bool other_stage{};
    /// F3.2b: the 16 user data words of each stage as the record's draws left them (persistent storage of the command processor); null when the draw has no typed user data
    const u32* vertex_user_words{};
    const u32* pixel_user_words{};
};

/// The shader objects of the draw that is being planned (set by the command processor around `PlanDraw`; nothing is read until `Current` asks).
struct Scope {
    Scope(u64 vertex_gnm, u64 pixel_gnm, u64 fetch_object, bool other_stage = false, const u32* vertex_user_words = nullptr, const u32* pixel_user_words = nullptr);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

/// The environment of the draw being planned on this thread; false when no native draw is being planned.
[[nodiscard]] bool Current(DrawEnv& out);

/// BuildRuntimeInfo ran for a stage with no native draw being planned (the packet path, the raster thread).
void NoteUnavailable();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeShaderEnv
