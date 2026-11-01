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

/// TLG_ABLATE=<list> (default empty, a bisection tool: the picture is WRONG when it is on): comma separated, leaves out work on the raster thread to see what the frame rate
/// does without it. draws (draw and indirect draw calls are not executed), dispatches (compute calls are not executed), announce (announced ranges are not applied, first
/// and again; deletions still run), again (only the re-apply of the previous batch), sync (SynchronizeMemory uploads nothing). Closures always run (they carry the EOP signals and
/// the write fences).
inline unsigned AblateMask() {
    const char* e = std::getenv("TLG_ABLATE");
    unsigned mask = 0;
    if (e == nullptr) {
        return 0;
    }
    const auto has = [&](const char* word) {
        for (const char* p = e; *p != 0;) {
            const char* end = p;
            while (*end != 0 && *end != ',') {
                ++end;
            }
            const char* w = word;
            const char* q = p;
            while (q < end && *w != 0 && *q == *w) {
                ++q, ++w;
            }
            if (q == end && *w == 0) {
                return true;
            }
            p = *end == ',' ? end + 1 : end;
        }
        return false;
    };
    mask |= has("draws") ? 1u : 0u;
    mask |= has("dispatches") ? 2u : 0u;
    mask |= has("announce") ? 4u : 0u;
    mask |= has("again") ? 8u : 0u;
    mask |= has("sync") ? 16u : 0u;
    return mask;
}
inline const unsigned ablate = AblateMask();
inline constexpr unsigned AblateDraws = 1, AblateDispatches = 2, AblateAnnounce = 4, AblateAgain = 8, AblateSync = 16;

} // namespace Core::NativeDiag
