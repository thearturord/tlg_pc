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

/// E7c Task 3 (`TLG_E3_ANNOUNCED=1`, default off, read once): what the engine's announcements say about the bytes a `RefreshImage` upload reads.
///
/// An announcement (NativeUpdates) is a memory range, not stored bytes. So the content is compared the only way it can be: when `ApplyBatch` applies an announced range, the bytes of every
/// 4 KB page it touches are hashed and remembered (`NoteAnnounced`, raster thread, ranges over 2 MB are remembered without a hash). When an image is uploaded, the pages of its memory are hashed
/// again (the guest memory is what the upload reads, before any detile: the same bytes on both sides, so no detile is needed to compare) and compared with the page's remembered hash:
/// `announced_equal` (every page that was announced still has the bytes it had when announced, at least one page announced), `announced_differs` (a page changed after its announcement: a write
/// the announcement did not cover, or a write that came after it) and `no_announcement` (no page of the image was ever announced). `NoteAnnouncedUpload` also takes what the upload was (first upload,
/// gpu source, repeat) so the report can answer per group and per caller, with the first examples.
[[nodiscard]] bool AnnouncedOn() noexcept;
void NoteAnnounced(u64 address, u64 size, bool first);

struct UploadInfo {
    Caller caller{Caller::Unknown};
    u64 image_uid{0};
    u64 address{0}, size{0};
    u32 width{0}, height{0}, format{0}; ///< the host format (vk::Format) of the image
    bool gpu_source{false};              ///< the upload read GPU written bytes (BufferCache::last_image_source_gpu), not guest memory
    bool gpu_modified_flag{false};       ///< the image's own GpuModified flag
    u32 usage_bits{0};                   ///< vk::ImageUsageFlags of the image (0x10 colour attachment, 0x20 depth stencil attachment, 0x8 storage)
};
void NoteAnnouncedUpload(const UploadInfo& info);

/// Who wrote the GPU written bytes (`gpu_modified_ranges`): set by the call sites that ask the buffer cache for a buffer to write (a shader's storage buffer, a fill, a copy), read at the one place
/// that adds the range. The upload with a GPU source finds the last site that wrote its memory.
enum class WriteSite : u8 { None, ShaderGraphics, ShaderCompute, FillBuffer, CopyBuffer, HleCopy, Count };
struct WriteSiteScope {
    explicit WriteSiteScope(WriteSite site) noexcept;
    ~WriteSiteScope();
    WriteSiteScope(const WriteSiteScope&) = delete;
    WriteSiteScope& operator=(const WriteSiteScope&) = delete;

private:
    WriteSite previous;
};
/// BufferCache::ObtainBuffer, where `gpu_modified_ranges.Add` is called
void NoteGpuWritten(u64 address, u64 size);

/// What `RefreshImage` did with the call (one per call, per caller). Entered counts every call; the rest are the exits.
enum class Outcome : u8 { NotDirty, Multisample, MaybeCpuDirtySameHash, NoCopies, Uploaded, Count };
void NoteOutcome(Caller caller, Outcome outcome);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeE3Census
