// SPDX-License-Identifier: GPL-2.0-or-later
//
// `TLG_NATIVE_DATA=1`: the master switch of the native data path (Plan N+T, tracks N, L and O). It turns on, as their default, every switch that replaces an address-keyed step by the engine's
// own knowledge: TLG_AGAIN_SKIP=1, TLG_BUFFER_ID=1, TLG_RING_MIRROR=1, TLG_SYNC_BY_ID=1, TLG_APPLY_COALESCE=1, TLG_RING_FILL=1. A switch set by name still wins (`TLG_RING_FILL=0` keeps that one off).
// Without the master switch every one of them is off, as before.

#pragma once

#include <cstdlib>

#include "common/types.h"

namespace Core {

[[nodiscard]] inline bool NativeDataMaster() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_NATIVE_DATA");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

/// The value of the switch `name`: what the environment says, else `master_value` under TLG_NATIVE_DATA=1, else 0.
[[nodiscard]] inline u32 SwitchValue(const char* name, u32 master_value) noexcept {
    const char* e = std::getenv(name);
    if (e != nullptr) {
        return static_cast<u32>(std::atoi(e));
    }
    return NativeDataMaster() ? master_value : 0u;
}

} // namespace Core
