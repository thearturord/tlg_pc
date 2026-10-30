// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

namespace VideoCore {
/// Image::b8_marks: why the native path wants the image refreshed or forgotten (shared by the texture cache and native/native_texture_cache.cpp)
constexpr u8 B8Fault = 1, B8Announced = 2, B8Owed = 4, B8Deleted = 8, B8Listed = 16;
} // namespace VideoCore
