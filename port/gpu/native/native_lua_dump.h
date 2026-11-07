// SPDX-License-Identifier: GPL-2.0-or-later
// L0 of the Lua timer design (documents/tlg/native plan/pc-port/lua-timers-120fps-design.md): writes every script chunk the game loads to user/log/lua_dump/.

#pragma once

#include "common/types.h"

namespace Core::NativeLuaDump {

/// Hook handler by name ("lua_dump"), 0 when the name is not it.
u64 FindHandler(const char* name);
/// Publishes the counters to the debug server (/stats key "lua_dump").
void Report();

} // namespace Core::NativeLuaDump
