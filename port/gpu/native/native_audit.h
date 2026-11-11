// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdlib>
#include "common/types.h"

/// Note 150 (plan R2.13, `TLG_AUDITS`: 1 on by default; 0 off): one switch for every real mode's "one in 64 takes the old path and is compared" audit (own memory, crossing,
/// placement, geometry, the view, storage and target tables, written buffers, hook-owned writes, the id marks, the drop of dirty applies, the barrier guard, the epoch planner).
/// On for plays (the audits are the correctness proof while the owner plays); off for the measuring set and releases.
namespace Core::Audit {

inline const bool g_on = [] {
    const char* e = std::getenv("TLG_AUDITS");
    return e == nullptr || e[0] != '0';
}();

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

/// Note 159: the audits that fired on this thread (the dispatch executor's shadow leaves out a binding the old body audited).
inline thread_local u32 t_fired = 0;

/// Advances the site's counter; true for the one call in 64 that is audited (never while `TLG_AUDITS=0`).
template <typename T>
[[nodiscard]] inline bool Due(T& tick) noexcept {
    if ((++tick & 63u) == 0 && g_on) {
        ++t_fired;
        return true;
    }
    return false;
}

} // namespace Core::Audit
