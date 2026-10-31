// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

/// E7: the switches of the native renderer that were set by the owner's overlay file (CUSA03627.readback.json) in every played build, as constants. They were
/// settings and live /set values; none of them is a choice any more: the path they turn off was deleted or is only kept where a note says so.
namespace Core::NativeConfig {

/// Bits of the former `native_sdk_strip`. All of 0..19, 21..25 and 31 were on; 20 (direct submission shadow index) and 26..30 (study switches: tracker audit, stale audit,
/// surface / clip / clear flags from the register mirror) were off.
inline constexpr u32 SdkStrip = 2213543935u;
inline constexpr u32 DrawMode = 3;
inline constexpr u32 DrawClasses = 15;
inline constexpr u32 SynthVerify = 1;
inline constexpr u32 DrawExec = 1;
inline constexpr u32 ExecInputs = 1;
inline constexpr u32 StageKey = 1;
inline constexpr u32 GroupSig = 1;
inline constexpr s32 MapNative = 2;

/// E7c (`TLG_TARGET_REGFREE=1`): the register-only fields of the target blocks as constants of this game. Every observed colour target and depth view had 0 in each (info_alt_tile_mode, info_compression,
/// info_linear_general, the depth view's read-only bits); the target description applies them unless the engine's own object or the resource's T# says otherwise (a guard trip: the register value is used
/// for that draw and the trip is counted with an example).
inline constexpr u32 TargetAltTileMode = 0;
inline constexpr u32 TargetCompression = 0;
inline constexpr u32 TargetLinearGeneral = 0;
inline constexpr u32 DepthViewReadOnly = 0;

} // namespace Core::NativeConfig
