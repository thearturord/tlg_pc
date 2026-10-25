// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    /// N6.1: where the walker takes each dword of the flat user data from, for the dwords it copies out of a table that one user data register points to (one level, constant
    /// offset): flat offset `16 + i` comes from dword `table_src[i]` of the table behind user data register `table_root_plus1[i] - 1` (0: not such a copy). A record that holds the
    /// table's entries can serve the load without running the walker.
    static constexpr u32 MaxTableMap = 384;
    u8 table_root_plus1[MaxTableMap]{};
    u16 table_src[MaxTableMap]{};
    /// True and the register / dword when flat offset `flat_offset` is such a copy.
    bool TableSource(u32 flat_offset, u32& root_sgpr, u32& src_dword) const noexcept {
        if (flat_offset < 16 || flat_offset - 16 >= MaxTableMap || table_root_plus1[flat_offset - 16] == 0) {
            return false;
        }
        root_sgpr = table_root_plus1[flat_offset - 16] - 1u;
        src_dword = table_src[flat_offset - 16];
        return true;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
