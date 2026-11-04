// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_shader_env.h"
#include "shader_recompiler/runtime_info.h"

namespace Vulkan::ShaderBoundary {

/// G2 (documents/tlg/native plan/executor-consumer-plan.md): the one place where the plain numbers of a draw or dispatch become the shader recompiler's own types
/// (`ShaderExportFormat`, `VsOutputControl`, `PsInput`, `FpRoundMode`, `FpDenormMode`, `PrimitiveType`). The pipeline cache calls these and names none of the six.

/// The fields of a graphics stage's translator input that belong to the shader's own environment (program modes, vertex outputs, pixel inputs and export formats), from the shader
/// objects (native/native_shader_env.h). Everything else in `info` is left as it is.
void ApplyShaderEnv(Shader::RuntimeInfo& info, const Core::NativeShaderEnv::DrawEnv& draw, Shader::HwStage stage, u32 aa_samples);

/// The translator input of a compute stage, from the dispatch's plain inputs.
void FillComputeInfo(Shader::RuntimeInfo& info, const Core::NativeDispatch::DispatchInputs& cs);

/// True when the vertex shader's output control enables guest clip distances.
bool GuestClipDistances(const Core::ShaderEnv::ShaderEnv& vertex);

/// The export format of a colour target: the pixel shader's (SPI_SHADER_COL_FORMAT code), or none exported (a draw without a pixel shader).
void SetExportFormat(Shader::PsColorBuffer& color_buffer, u32 code);
void SetNoExport(Shader::PsColorBuffer& color_buffer);

} // namespace Vulkan::ShaderBoundary
