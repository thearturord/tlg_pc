// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"

namespace Core {

/// The call sites. Names are where the call is made: the first group calls `UpdateImage` (which refreshes), the second are calls inside the texture cache.
enum class ImageCaller : u8 {
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

} // namespace Core
