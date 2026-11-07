// SPDX-License-Identifier: GPL-2.0-or-later
// Lua frame-counted timers at 120 fps (documents/tlg/native plan/pc-port/lua-timers-120fps-design.md): load-time text rules for the game's scripts.

#pragma once

#include "common/types.h"
#include "native/native_hooks.h"

namespace Core::NativeLuaTime {

/// Called from the hksL_loadbuffer mid hook: if rules exist for the chunk (rdx = text, rcx = size, r8 = chunk name), points rdx / rcx at the patched text.
void Patch(GuestHooks::MidContext* context);
/// Publishes the counters to the debug server (/stats key "lua_time").
void Report();

} // namespace Core::NativeLuaTime
