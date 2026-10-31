// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E7 Task 4 (port/gpu/native/E7-notes.md): who still needs the address keyed image code of the texture cache. Census only, `TLG_E3_CENSUS=1` (default off; with it off every call below returns at once).
///
/// `RefreshImage`, `ExpandImage`, `ResolveDepthOverlap`, `ResolveOverlap` and the `exact_match` test of `FindImage` are counted per caller (a named call site), and for `RefreshImage` the bytes it uploads
/// are hashed and compared with the bytes of the image's previous upload, so the report says how many uploads carried nothing new. Counters: /stats native_backend.device.e3_census (and last_counters.json);
/// "per_25s" is the count scaled to 25 seconds of observed time (the clock starts with the first counted call).
namespace Core::NativeE3Census {

[[nodiscard]] bool On() noexcept;

enum class Fn : u8 { RefreshImage, ExpandImage, ResolveDepthOverlap, ResolveOverlap, ExactMatch, Count };

/// The call sites. Names are where the call is made: the first group calls `UpdateImage` (which refreshes), the second are calls inside the texture cache.
enum class Caller : u8 {
    Unknown,
    PresenterPrepareFrame,      ///< vk_presenter.cpp, the video out image
    RasterizerBindTextures,     ///< BindTexturesFromSet: a sampled image
    RasterizerBeginRendering,   ///< BeginRenderingFull: a render or depth target
    RasterizerTryCopyImages,    ///< TryCopyImages: the source of an image to image copy
    RasterizerNativeClear,      ///< NativeClear: the rest of an image around an engine clear
    FindTexture,                ///< TextureCache::FindTexture
    FindRenderTarget,           ///< TextureCache::FindRenderTarget
    FindDepthTarget,            ///< TextureCache::FindDepthTarget
    FindImageClearRest,         ///< FindImage: the rest of a texture from memory before its pending engine clear is applied
    ExpandImageCopy,            ///< ExpandImage: the new, bigger image is filled from memory before the old one is copied in
    ResolveDepthOverlapNewImage, ///< ResolveDepthOverlap: the re-created image
    OverlapBlockView,           ///< ResolveOverlap, equal address: a compressed view of an uncompressed image (outcome 2)
    OverlapVolume,              ///< ResolveOverlap, equal address: a 3D view of equal size (outcome 3)
    OverlapGrow,                ///< ResolveOverlap, equal address: the requested image has more resources (outcome 6)
    OverlapDepth,               ///< ResolveOverlap: depth overlap at an equal address
    FindImageOverlap,           ///< FindImage calls ResolveOverlap
    FindImageExactById,         ///< FindImage: exact match against the images of the engine texture found by id
    FindImageViewReuse,         ///< FindImage: exact match when a lookup answer is remembered
    Count
};

void NoteCall(Fn fn, Caller caller);

/// A `RefreshImage` upload from guest memory: `bytes` / `size` is the memory the image was filled from (null when the source was GPU written memory).
void NoteUpload(Caller caller, u64 image_uid, const u8* bytes, u64 size);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeE3Census
