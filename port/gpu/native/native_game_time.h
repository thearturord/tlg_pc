// SPDX-License-Identifier: GPL-2.0-or-later
// 120 fps game timing: the game's constant 1/30 s steps scaled by the real frame time (see native_game_time.cpp).

#pragma once

#include "common/types.h"

namespace Core::NativeGameTime {

/// Hook handler by name ("env_phase_dt", "particle_dt_a", "particle_dt_b", "particle_dt_c"), 0 when the name is not one of them.
u64 FindHandler(const char* name);
/// Publishes the counters to the debug server (/stats key "game_time").
void Report();

} // namespace Core::NativeGameTime
