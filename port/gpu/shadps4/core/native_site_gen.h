// SPDX-License-Identifier: GPL-2.0-or-later
// B7: dispatch site generators on mid function hooks (core/guest_hooks.h mode "mid"), one per site (documents/tlg/native plan/stages/b7-scene-jobs.md section 6). A site is the tail of an
// engine loop or a straight line piece: the stores into the engine's objects, the staged block, the views set on the context and the dispatch. The top hook scans it; when it is the steady
// state the generator does what the engine's code does (the engine's own virtuals for UpdateSubresource, `NativeDraw::IssueDispatch` for the dispatch), sets the registers the code after
// the piece reads and continues at the address after it. One eligible execution in `TLG_SITE_GEN_VERIFY` (default 256) is run by the engine and compared. `TLG_SITE_GEN=0` turns all of
// them off (default on). Report: /zpassgen key `site_generators`.

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::GuestHooks {
struct MidContext;
}

namespace Core::NativeSiteGen {

/// Handler address of a site's hook by name ("site_top_0", "site_bottom_0", ...), 0 when unknown.
u64 FindHandler(const char* name);
/// A dispatch of the async or graphics context was issued by the engine: during a verify execution it is compared with the prediction.
void NoteDispatch(u64 context, u64 x, u64 y, u64 z, u64 site);
/// `TLG_SITE_CENSUS=1`: counts every dispatch by its call site (the return address), told apart by who issued it (the engine's own code or a generator): `site_generators` `dispatch_census`.
void CensusDispatch(u64 site_runtime, bool generated);

nlohmann::json Report();

} // namespace Core::NativeSiteGen
