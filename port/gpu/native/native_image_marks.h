// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

namespace VideoCore {
/// Image::b8_marks: why the native path wants the image refreshed or forgotten (shared by the texture cache and native/native_texture_cache.cpp)
constexpr u8 B8Deleted = 8, B8Listed = 16;
/// F1 ring-out census (TLG_N0_CENSUS=1): dirtied by a ring candidate / by anything else since the last refresh
constexpr u8 B8CandDirty = 32, B8OtherDirty = 64;
} // namespace VideoCore
