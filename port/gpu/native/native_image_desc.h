// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "video_core/amdgpu/resource.h"

/// E2 (documents/tlg/native plan/no-emulation-plan.md): the description of an image a draw samples, derived from the engine's own objects instead of read from the
/// console descriptor (T#) words the engine builds for it. The inputs are the texture the engine created (its Direct3D 11 description, `native_device.h`) and the shader
/// resource / unordered access view it created for it (the description it was created with). The console format, component select, tiling index and type are functions of
/// those (tables below); the memory address, size, level range and layer range are copied.
///
/// Rollout (shadow first): every image binding of a native draw is derived and compared field by field with the T# the record holds (`/stats`, native_backend.device.e2_derive).
/// Native decides after a clean run and the owner's play; then the T# words of the record bindings are no longer read for images.
namespace Core::NativeImageDesc {

enum class Result : u8 {
    Derived,
    UnknownObject, ///< the engine table has no live object of the id
    UnknownFormat, ///< the engine's format has no entry in the table (the record's T# keeps deciding)
};

/// The image of the resource `resource_id` as seen through the view `view_id` (0: the resource itself).
[[nodiscard]] Result Derive(u64 resource_id, u64 view_id, AmdGpu::Image& out);

/// True: the image descriptor of a binding comes from `Derive` and the words the record holds only verify (default; `TLG_IMAGE_DESC=0` turns it off).
[[nodiscard]] bool Decides() noexcept;
/// The deciding call of the consumer: the image of the binding (the raw descriptor, before the shader's fix ups) in `out`; false when the engine table cannot say (the record's
/// words then decide). `record_words` (8 dwords, or null) are compared with the result; a difference is counted and goes to the rare events.
[[nodiscard]] bool Decide(u64 resource_id, u64 view_id, const u32* record_words, AmdGpu::Image& out);

/// `TLG_DESC_CHECKS=1`: derive and compare with the T# the record holds (`actual`); counts per field.
void Compare(u64 resource_id, u64 view_id, const AmdGpu::Image& actual);

/// E2 samplers: the S# words (four dwords) of a sampler binding built from the engine's sampler object (its Direct3D style description at +0x30: filter, address modes,
/// anisotropy, comparison, border color, level limits). False when the object holds a value the table has not met (the words the engine built then decide).
/// Called while the object is alive (the record builder); `record_words` (or null) are compared with the result and counted.
[[nodiscard]] bool DeriveSampler(u64 object, const u32* record_words, u32* out);

/// E2 study (`TLG_DESC_CHECKS=1`): a sampler object and the four S# words the engine built from it
void StudySampler(u64 object, const u32* words);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeImageDesc
