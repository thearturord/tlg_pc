// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// F4.0b (`TLG_FIELD_USE=1`, default off, read once): which fields of the bound T#, S# and V# are anything but the default. The evidence for the "constant or refusal" rule of F4.2: a field that
/// is always the default in the default scene and a cutscene becomes a constant of the native description with a guard counter, a field that varies has to be produced.
/// Plain words only (no PS4 header): the callers pass the raw descriptor words (T# 8 dwords, S# 4, V# 4). Report: /stats native_backend.device.f4_field_use. Removed with F4.6.
namespace Core::NativeFieldUse {

[[nodiscard]] bool On() noexcept;
void NoteImage(const u32* words);
void NoteSampler(const u32* words);
void NoteBuffer(const u32* words);
/// Host fields of an image description that are not plain bits of the T# words: the alt tile mode and the sample count.
void NoteHostImageFields(u32 alt_tile, u32 samples);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeFieldUse
