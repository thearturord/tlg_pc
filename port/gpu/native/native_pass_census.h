// SPDX-License-Identifier: GPL-2.0-or-later
// Native back end design P6/P7 gate (documents/tlg/native plan/native-backend-design.md, section 7 step 1): a census, no behaviour change.
//
// Off by default (`TLG_PASS_CENSUS=1`). The raster thread reports every draw and dispatch plan after it ran. For the depth-only (shadow) draws the census says how many distinct
// pipeline / state groups there are and what changes between consecutive draws; for the compute dispatches which kernels run, with what sizes, and how long the runs of the same kernel are.
// The result is published as `pass_census` (the debug server, /stats).

#pragma once

#include "common/types.h"

namespace Core::NativePlan {
struct DrawPlan;
struct DispatchPlan;
} // namespace Core::NativePlan

namespace Core::PassCensus {

extern bool g_on;

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

/// A draw (direct or indirect) that was just executed; `depth_only` is the classification the raster thread uses.
void NoteDraw(const NativePlan::DrawPlan& plan, bool depth_only, bool indirect);
/// A dispatch that was just executed.
void NoteDispatch(const NativePlan::DispatchPlan& plan, bool indirect);

} // namespace Core::PassCensus
