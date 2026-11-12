// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Design review notes 95 to 97, section 79 step 3a (`TLG_VIEW_TABLE`: 2 real by default since the owner's play, note 99, with 1 stage in 64 audited; 1 shadow; 0 off, the old path): read-only image bindings of non-dynamic engine textures
/// are found by the engine view id the record names, not by the T# words. The table lives on the raster thread: an entry is made from the old path's answer at the first bind
/// of a (view id, use) and holds while the image's uid is unchanged and registered and no delete event of the view or its texture dropped it.
namespace Core::ViewTable {

[[nodiscard]] u32 Mode() noexcept;
/// Notes 102 to 106 (`TLG_TARGET_TABLE`: 2 real by default since the owner's play, note 106, one draw in 64 audited; 1 shadow; 0 off, the old path): render and depth targets
/// by their view id, in this same table (use flags with bit 63).
[[nodiscard]] u32 TargetMode() noexcept;
/// Notes 107 to 110 (`TLG_WRITTEN_NO_IMAGE`: 2 real by default since the owner's play, note 110, one in 64 walked as the audit; 1 shadow; 0 off, the old path): a written binding of an engine buffer whose image list holds no live image and no
/// waiting clear needs no texture cache walk for images based in its memory (`NativeImageIndex::NoImageInBuffer`).
[[nodiscard]] u32 WrittenMode() noexcept;
/// Notes 111 and 147 (`TLG_STORAGE_TABLE`: 0 off, the old path; 1 shadow; 2 real, one stage in 64 audited; 0 also while `TLG_VIEW_TABLE=0`): storage (written) image bindings of non-dynamic
/// engine textures by their view id, in this same table (the use flags hold `is_written`).
[[nodiscard]] u32 StorageMode() noexcept;

/// Note 113: the engine's texture copy dispatches (`MakeUploadRecord`: CopyResource, Map uploads) bind a texture's subresource with no view object; their T# is a function
/// of the texture's own T# (+0x70, fixed at creation for non-dynamic textures), the subresource and read or write. Such a binding is named by this key in place of a view id:
/// bit 63 the mark, bit 62 written, bits 40..61 the subresource, bits 0..39 the texture id; 0 (no key) when a part does not fit. Made only while `StorageMode() != 0`.
[[nodiscard]] constexpr u64 SubresourceKey(u64 resource_id, u32 subresource, bool written) noexcept {
    if (resource_id == 0 || resource_id >= (1ull << 40) || subresource >= (1u << 22)) {
        return 0;
    }
    return 1ull << 63 | u64(written) << 62 | u64(subresource) << 40 | resource_id;
}
[[nodiscard]] constexpr bool IsSubresourceKey(u64 view_id) noexcept {
    return (view_id >> 63) != 0;
}
[[nodiscard]] constexpr u64 SubresourceKeyResource(u64 view_id) noexcept {
    return view_id & ((1ull << 40) - 1);
}

struct Entry {
    u64 view_id{};
    u64 resource_id{};
    u32 image_index{~0u};  ///< the texture cache slot of the bound image (after any depth redirect)
    u64 image_uid{};       ///< its uid when the entry was made
    u64 view_handle{};     ///< the VkImageView the binding used
    bool in_class{};       ///< false: the view's texture is dynamic or unknown (the old path keeps it)
    bool has_payload{};    ///< the rasterizer's per-image record (its texture set entry) is in `payload`
    alignas(8) unsigned char payload[160]{};
};

/// The entry of (view id, use flags), or nullptr.
[[nodiscard]] Entry* Find(u64 view_id, u64 use_flags) noexcept;
/// Make or replace the entry of (view id, use flags).
Entry& Put(u64 view_id, u64 use_flags) noexcept;
/// A batch event (`what` as in `NoteEngineDelete`: 0 view deleted, 1 and 3 texture deleted): the entries of that view or texture go.
void Event(u32 what, u64 id) noexcept;
/// The view `view_id` looks at texture `resource_id` (its entries go when the texture is deleted).
void LinkResource(u64 resource_id, u64 view_id) noexcept;
/// Note 146: texture `resource_id` got a new image in the texture cache's list (B8.3). The old path's lookup answers the newest matching image of a texture, so every entry
/// of the texture's views (sampled, storage and target) goes and is made again from the old path's next answer.
void ResourceImageAdded(u64 resource_id) noexcept;

enum Count : u32 {
    StagesWithImages,       ///< stages that bound at least one image (shadow)
    StagesAllInClass,       ///< ... every image binding in the class
    StagesWouldServe,       ///< ... and every entry valid and equal: the real mode binds them from the table
    StagesWithDifference,   ///< ... a valid entry answered another image or view (MUST_BE_0)
    ImagesInClass,          ///< image bindings in the class
    ImagesSame,             ///< ... with a valid entry that equals the old path's answer
    ImagesDifferent,        ///< ... with a valid entry that differs (VIEW_TABLE_DIFFERENT_MUST_BE_0)
    ImagesNoEntry,          ///< ... first bind of the (view, use): entry made
    ImagesStale,            ///< ... entry whose image uid moved or is unregistered: remade
    NotClassNoView,         ///< image bindings not in the class: the record names no view
    NotClassDynamic,        ///< ... the view's texture is dynamic (or the view is unknown)
    NotClassUse,            ///< ... a written image, a mip fallback, or more than one binding
    NotClassNull,           ///< ... a null or rejected image
    DroppedByEvent,         ///< entries dropped by a view or texture delete
    RealServed,             ///< real mode: stages bound from the table
    RealServedImages,       ///< ... their image bindings
    RealAudited,            ///< ... stages the table could serve that took the old path as the audit (compared at its end)
    RealCheckFailed,        ///< ... stages whose per-image checks (backing, rebind, target, depth association) failed: the old path bound them
    TgDraws,                ///< target table shadow: draws that bind targets
    TgDrawsAllInClass,      ///< ... every target names an engine view
    TgDrawsWouldServe,      ///< ... and every entry valid and equal (image and view): the real mode takes them from the table
    TgDrawsOldFind,         ///< ... draws of the class where the old path ran FindImage for a target (its register cache missed)
    TgTargets,              ///< targets in the class
    TgSame,                 ///< ... valid entry, same image
    TgDifferent,            ///< ... valid entry, another image (TARGET_TABLE_DIFFERENT_MUST_BE_0)
    TgNoEntry,              ///< ... entry made
    TgStale,                ///< ... entry remade (uid moved or unregistered)
    TgNoView,               ///< targets that name no engine view (video out, eliminate)
    TgViewSame,             ///< at the render pass: the target view equals the entry's
    TgViewDifferent,        ///< ... differs (TARGET_VIEW_DIFFERENT_MUST_BE_0)
    TgViewNoEntry,          ///< ... the entry had no view yet (recorded now)
    TgDescDifferent,        ///< same image, but the old path's description differs from the stored one (TARGET_DESC_DIFFERENT_MUST_BE_0)
    TgTaken,                ///< real mode: targets taken from the table
    TgDrawsTaken,           ///< real mode: draws whose every target came from the table
    TgAudits,               ///< real mode: draws audited by the old path (one in 64)
    WbCalls,                ///< written buffer bindings that took the texture cache's walk
    WbNamed,                ///< ... naming an engine buffer that holds the whole range
    WbIdNone,               ///< ... whose id says no image and no waiting clear is in it
    WbWalkFound,            ///< ... where the walk marked an image (any)
    WbIdNoneWalkFound,      ///< ... the id said none and the walk marked one (ID_SAID_NO_IMAGE_BUT_WALK_FOUND_MUST_BE_0)
    WbNotNamed,             ///< written bindings with no engine id, or a range outside its buffer (the walk stays)
    WbSkipped,              ///< real mode: walks skipped (the id says no image)
    WbAudited,              ///< real mode: ... walked anyway as the audit (one in 64)
    WbAuditFound,           ///< real mode: ... and the audit's walk marked an image (AUDIT_WALK_FOUND_MUST_BE_0)
    StStages,               ///< note 111 shadow: stages with a written image binding
    StStagesAllInClass,     ///< ... every image binding in the class
    StStagesWouldServe,     ///< ... every entry valid and equal, every answer bindable, no image twice: the real mode binds them from the table
    StStagesSameImageTwice, ///< ... refused: the same image bound twice, one binding written
    StStagesNotBindable,    ///< ... refused: an answer the tail cannot bind (a target, a rebind, a forced-general sampled image)
    StInClass,              ///< written image bindings in the class
    StSame,                 ///< ... valid entry equal to the old path's answer
    StDifferent,            ///< ... valid entry that differs (STORAGE_TABLE_DIFFERENT_MUST_BE_0)
    StNoEntry,              ///< ... entry made
    StStale,                ///< ... entry remade (uid moved or unregistered)
    StNoView,               ///< written bindings whose record names no view
    StDynamic,              ///< ... whose view's texture is dynamic or unknown
    StOtherUse,             ///< ... a mip fallback store or several bindings
    StNull,                 ///< ... a null or rejected image
    KeyedImages,            ///< note 113: sampled bindings named by a texture subresource key (copy dispatches)
    StKeyed,                ///< ... written bindings named by one
    KeyedUnknown,           ///< ... entries of a key whose texture the device table does not know
    KeyedDynamic,           ///< ... whose texture is dynamic (usage 2: its memory moves with every Map)
    KeyedOther,             ///< ... whose object is not a texture
    DroppedByNewImage,      ///< note 146: entries dropped because their texture got a new image
    StRealServed,           ///< note 147 real mode: stages with a written image bound from the table
    StRealServedWritten,    ///< ... their written image bindings
    StRealRefusedTwice,     ///< ... stages refused at the tail: the same image twice, one binding written
    NewImageEvents,         ///< ... the events (a texture with linked views got a new image)
    RealServedTargets,      ///< note 160: sampled render targets (feedback loops) bound by the tail with the old path's layout
    TwiceWouldServe,        ///< note 170 shadow: stages refused only for the same image twice (one binding written) that the real mode would bind
    TwiceServed,            ///< note 170 real: those stages bound by the tail with the old path's two passes
    ForceGeneralServed,     ///< note 170 real: sampled image bindings with force general set, bound with the old path's general transit
    DynServed,              ///< note 163: dynamic texture bindings bound from the texture's own image
    DynRefreshed,           ///< ... uploads into those images (the window's bytes differed from the last upload)
    DynRefused,             ///< ... refused (another shape, too large, invalid T#, GPU-modified memory)
    DynNotDynamic,          ///< ... entries not in class whose texture is not dynamic (unknown: the old path)
    DynShadowSame,          ///< shadow: the old path's image is at the window with the same shape
    DynShadowOldOther,      ///< shadow: the old path served another image (another address or shape: the overlap answer)
    DynWindowDifferent,     ///< the description does not name the T#'s window (must be 0)
    Count_
};
void Note(Count c, u64 n = 1) noexcept;

[[nodiscard]] nlohmann::json Report();
/// Note 163 (`TLG_DYNAMIC_TEXTURES`: 2 real by default, 1 shadow, 0 off): dynamic textures bound from their own image by id. LEGACY(dynamic_textures): =0.
[[nodiscard]] u32 DynamicMode() noexcept;

} // namespace Core::ViewTable
