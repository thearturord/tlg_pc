// SPDX-License-Identifier: GPL-2.0-or-later
// NativeDraw v2 (documents/tlg/native-renderer-plan.md, N2 / N3): the descriptors a draw's record put into the user data of its stages, with where they sit.
// The binder of the rasterizer takes a shader resource by (stage, flat user data offset) and reads its descriptor words here instead of from the flat user data.

#pragma once

#include "common/types.h"

namespace Core::NativeDraw {

constexpr u32 MaxGeometryIds = 9; // N6.2: the engine ids of the vertex buffers of a draw (up to 8) and of its index buffer
constexpr u32 MaxRecordBindings = 64; // N6.1: the entries of the descriptor tables are in the list too

struct RecordBinding {
    u8 stage;    ///< 0 vertex, 1 pixel
    u8 kind;     ///< 1 buffer descriptor (V#), 2 image descriptor (T#), 3 sampler (S#)
    u16 offset;  ///< flat user data index of the first dword; 0xffff: an entry of a table behind a user data register (`table_sgpr`)
    u16 slot;    ///< the engine slot the object was found in (for a table entry: its index in the table)
    u16 dwords;
    u8 table_sgpr;    ///< table entry: the user data register that points to the table (0xff: not a table entry)
    u8 entry_dwords;  ///< table entry: dwords per entry of the table (8 resource table, 4 sampler / constant buffer table)
    u64 id;      ///< the engine's unique id of the resource (the registry's key)
    u32 words[8];
};

} // namespace Core::NativeDraw
