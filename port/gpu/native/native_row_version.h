// SPDX-License-Identifier: GPL-2.0-or-later
// F2 item 2 (design review 2026-11-09, section 29): the row check becomes one integer compare. A version per engine buffer id, bumped wherever bytes of that buffer are marked
// changed (the places that add to `object_dirty`), and one version for the bytes no engine buffer holds (the places that add to `b9_unattributed`). A row stores the versions it
// was answered under; it is clean when they did not move. `TLG_ROW_VERSION`: 2 real (default since 2026-11-10), 1 shadow (the sets decide, every check compared), 0 off.
#pragma once

#include <atomic>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

namespace Core::RowVersion {

/// 0 off, 1 shadow (the sets decide, the versions are compared), 2 real (the version decides; rows are learned only inside their buffer; one check in 64 also asks
/// the sets, and a row they call dirty is not served; default since 2026-11-10).
[[nodiscard]] u32 Mode() noexcept;

/// Bytes of engine buffer `id` were marked changed. Any thread. (Ids share slots: a collision only makes a row look changed.)
void Bump(u64 id) noexcept;
/// Bytes no engine buffer holds were marked changed. Any thread.
void BumpUnattributed() noexcept;

[[nodiscard]] u64 Of(u64 id) noexcept;
[[nodiscard]] u64 Unattributed() noexcept;

/// One row check: what the versions say, what the sets said, whether the row lies inside its buffer (then the unattributed version is not asked), the time of the sets' answer.
void NoteCheck(bool version_clean, bool sets_dirty, bool inside, u64 sets_ns, u64 id, u64 address, u64 size) noexcept;
/// A row learned, inside its buffer or not.
void NoteLearn(bool inside) noexcept;
/// Real mode: 0 a row served by its version, 1 a check also asked of the sets, 2 the sets said dirty where the version said clean (MUST_BE_0; not served), 3 a row not learned
/// because it is not inside its buffer, 4 a row the version called changed.
void NoteReal(u32 what) noexcept;
/// Census of the identity walk per row check (`TLG_ROW_WALK_CENSUS=1`, counters only): on, then one walk's events stepped over, its time, its answer (0 none, 1 overlaps, 2 unknown).
[[nodiscard]] bool WalkCensusOn() noexcept;
void NoteWalk(u64 steps, u64 ns, u32 answer) noexcept;
/// Census (with the walk census): a read the old path answered and that leaves no row behind, by what answered it: 0 the hook ring, 1 the stream ring inside the frame ring,
/// 2 the stream ring outside the frame ring, 3 a buffer inside the frame ring, 4 a buffer outside it (learned or refused later), bytes summed per class.
void NoteOldAnswer(u32 what, u64 bytes) noexcept;
/// Static-row census (`TLG_STATIC_ROW_CENSUS=1`, counters only): a read of an engine buffer outside the frame ring that the stream ring answered with a fresh copy. Compared with
/// the previous copy of the same request: 0 no previous copy, 1 version same + bytes same (a row would have been right), 2 version same + bytes DIFFERENT (MUST_BE_0: a row would
/// have served stale bytes), 3 version moved + bytes same (an extra miss), 4 version moved + bytes different (a right miss), 5 not inside its buffer, 6 not hashed.
[[nodiscard]] bool StaticCensusOn() noexcept;
/// Diagnostics (`TLG_WATCH_ADDR=0x...`): does [address, address + size) overlap the watched address? (logs, nothing else)
[[nodiscard]] bool Watched(u64 address, u64 size) noexcept;
void NoteStatic(u32 what, u64 bytes) noexcept;

[[nodiscard]] nlohmann::json Report();

} // namespace Core::RowVersion
