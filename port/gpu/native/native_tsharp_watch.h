// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

/// Diagnostic (`TLG_TSHARP_WATCH=1`, off by default): finds the code that changes the T# a texture view keeps. A hardware write watch (debug registers, all threads) is armed on the
/// height dword of up to four shader resource views of a 1920 x 1088 texture when they are created; each write is logged with the instruction address (relative to the game image),
/// the return addresses found on the stack and the old and new dword. See documents/tlg/native plan/phase-notes/F4-notes.md, question (a).
namespace Core::NativeTsharpWatch {

[[nodiscard]] bool On() noexcept;
/// A view was created: `address` is the guest address of the T# dword to watch (armed when the texture is 1920 x 1088)
void Arm(u64 view_object, u64 address, u32 width, u32 height);
/// W1.5 census (`TLG_SILENT_CENSUS=1`): a write watch on the dword at `address`, whatever the switch above (four watches in all)
void ArmAddress(u64 address, const char* tag);

} // namespace Core::NativeTsharpWatch
