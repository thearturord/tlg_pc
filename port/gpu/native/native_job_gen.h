// SPDX-License-Identifier: GPL-2.0-or-later
// B7: the generators that replace inline loops of the engine's `RenderScene` through mid function hooks (core/guest_hooks.h, mode "mid"; documents/tlg/native plan/stages/b7-scene-jobs.md).
// The first one: the scene's small async jobs (the loop body at ghidra 0x01c606f0, 90 iterations per frame): per job the constants of the object go into the staged block of the render
// device, the block is sent with the context's UpdateSubresource, four resource views are set as the constant buffers of the async context, the job's shader is set, one dispatch
// of (count + 63) / 64 groups is issued and the resources and the shader are reset. Switches: `TLG_JOB_GEN=0` turns it off (default on), Report: /zpassgen key `job_loop`.

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::GuestHooks {
struct MidContext;
}

namespace Core::NativeJobGen {

[[nodiscard]] bool Enabled();
/// Hook handler (mode "mid"): the top of the loop body (0x01c606f0).
u64 PS4_SYSV_ABI JobLoopTop(GuestHooks::MidContext* context);

nlohmann::json Report();

} // namespace Core::NativeJobGen
