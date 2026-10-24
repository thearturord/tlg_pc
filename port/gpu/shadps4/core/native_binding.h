// SPDX-License-Identifier: GPL-2.0-or-later
// NativeDraw v2 (documents/tlg/native-renderer-plan.md, N2 / N3): the descriptors a draw's record put into the user data of its stages, with where they sit.
// The binder of the rasterizer takes a shader resource by (stage, flat user data offset) and reads its descriptor words here instead of from the flat user data.

#pragma once

#include "common/types.h"

namespace Core::NativeDraw {

constexpr u32 MaxRecordBindings = 24;

struct RecordBinding {
    u8 stage;    ///< 0 vertex, 1 pixel
    u8 kind;     ///< 1 buffer descriptor (V#), 2 image descriptor (T#), 3 sampler (S#)
    u16 offset;  ///< flat user data index of the first dword
    u16 slot;    ///< the engine slot the object was found in
    u16 dwords;
    u64 id;      ///< the engine's unique id of the resource (the registry's key)
    u32 words[8];
};

} // namespace Core::NativeDraw
