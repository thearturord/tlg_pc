// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdlib>

/// Switches of the raster thread work reduction (read once at start; environment variables):
///   TLG_DIAG_COUNTERS (default 1): the always-on counters of the bind path (consumer census, binder counters, resident / handle / immutable counters, the bind memo's hit and
///   miss counts). `TLG_DIAG_COUNTERS=0` makes them do nothing; the numbers they feed in /stats then stand still. Only pure counters are behind it.
///   TLG_COALESCE_ANNOUNCE (default 0): the announced writes of a batch are merged where they overlap or touch before they are applied (NativeUpdates::ApplyBatch). Ranges with
///   a gap between them are never merged.
namespace Core::NativeDiag {

inline bool ReadSwitch(const char* name, bool fallback) {
    const char* e = std::getenv(name);
    if (e == nullptr || *e == 0) {
        return fallback;
    }
    return *e != '0';
}

inline const bool counters = ReadSwitch("TLG_DIAG_COUNTERS", true);
inline const bool coalesce_announce = ReadSwitch("TLG_COALESCE_ANNOUNCE", false);

} // namespace Core::NativeDiag
