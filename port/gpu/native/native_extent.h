// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The extent hint of a render target (width and height in pixels, 0 = none): plain numbers the target description carries (G6, replaces the PS4 register word header
// `amdgpu/cb_db_extent.h`; the host keeps the old name `AmdGpu::CbDbExtent` as an alias of this type).

#include "common/types.h"

namespace Core {

union TargetExtent {
    struct {
        u16 width;
        u16 height;
    };
    u32 raw;

    bool Valid() const {
        return raw != 0;
    }
};

} // namespace Core
