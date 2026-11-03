// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "native/native_desc.h"
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

/// F4.1: the same description as plain numbers (`native_desc.h`), from the engine's objects only (no T# word is read; the view's own copy of the descriptor at +0x58 is read as plain
/// bits for the padded rows and the limited level count). False when the object or its format is not in the tables.
[[nodiscard]] bool DeriveDesc(u64 resource_id, u64 view_id, NativeDesc::ImageDesc& out);
/// `TLG_DESC_CHECKS=1`: the caller builds the reference description from the T# (`ImageDescFromSharp`) and `CompareDesc` compares it field by field with `DeriveDesc`.
/// `accessor_mismatch` is the host side's check of the accessors of the engine's description against the T# helper functions (bit per accessor, `AccessorNames`).
[[nodiscard]] bool DescCompareOn() noexcept;
void CompareDesc(u64 resource_id, u64 view_id, const NativeDesc::ImageDesc& reference, u32 accessor_mismatch);
/// the description the accessor bit names (bit order of the mask `CompareDesc` receives)
constexpr const char* AccessorNames[] = {"NumLayers", "NumLevels", "NumSamples", "BaseType", "ViewType", "NumViewLevels", "NumViewLayers", "IsTiled", "IsVolume"};

/// F4.1 samplers: the plain description of a sampler from the four S# words (bit extraction, the host mapping tables written as numbers); the host's own decode of the same words
/// is `VideoCore::SamplerDescFromSharp`, compared in the shadow (`TLG_DESC_CHECKS=1`): `NoteSamplerDescCompare(equal)` counts, `example` names a difference.
[[nodiscard]] NativeDesc::SamplerDesc SamplerDescFromWords(const u32* words) noexcept;
void NoteSamplerDescCompare(bool equal, const char* example);
constexpr float SamplerDescLodScale = 1.0f / 256.0f;

/// F4.3 (`TLG_IMAGE_FROM_DESC`, default on since 2026-11-03, `=0` turns it off, read once): the host builds the `ImageInfo` / `ImageViewInfo` of a sampled or stored image from the plain
/// description instead of the T#. `FromDescOn()` is the switch; `DescForBinding` is the consumer's call: the engine's description of the binding with the shader's fix ups applied
/// (`cube_to_2d_array`: the cube is read as a 2D array; `is_depth`: the host reads a depth image only as a 16 or 32 bit single channel). False (counted, the T# path serves the binding)
/// when the engine has nothing for the binding, the alternative tiling is set, or the depth use does not fit.
[[nodiscard]] bool FromDescOn() noexcept;
[[nodiscard]] bool DescForBinding(u64 resource_id, u64 view_id, bool cube_to_2d_array, bool is_depth, NativeDesc::ImageDesc& out);

/// F4.2 shadow: the host `ImageInfo` / `ImageViewInfo` built from the plain description against the ones built from the T#: bit per differing field (names below).
/// `engine_description_available` false: `DeriveDesc` had nothing for the binding (or the alternative tiling is set): counted, not compared.
constexpr const char* InfoFieldNames[] = {"props", "pixel_format", "type", "levels", "layers", "size", "num_bits", "num_samples", "pitch", "tile_mode", "array_mode", "mips_layout", "micro_tiled_mips", "guest_address", "guest_size", "bank_swizzle_alt_tile", "meta_stencil"};
constexpr const char* ViewFieldNames[] = {"type", "format", "range", "mapping", "min_lod", "is_storage"};
void NoteHostInfoCompare(bool engine_description_available, u32 info_mask, u32 view_mask, const char* example);

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
