// SPDX-License-Identifier: GPL-2.0-or-later
// N6.6: read only census of the arguments and shapes the engine's G-buffer loop sees (`TLG_GBUFFER_AUDIT=1`, /gbuffer).

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeGBufferAudit {

/// `DrawGBufferPrimitives(render_device, mode, list, a3, flags, query_mode, param7)` was entered.
void Enter(u64 render_device, u64 mode, u64 list, u64 flags, u64 query_mode, u64 param7);
/// `DrawForwardPass(render_device, a2, a3, list, mode)` was entered (ghidra 0x01c779a0).
void EnterForward(u64 render_device, u64 list, u64 mode);
nlohmann::json Report();

} // namespace Core::NativeGBufferAudit
