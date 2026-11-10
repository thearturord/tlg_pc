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
/// Note 102/103 (`TLG_TARGET_TABLE`: 0 off (default), 1 shadow; 2 real is not built yet): render and depth targets by their view id, in this same table (use flags with bit 63).
[[nodiscard]] u32 TargetMode() noexcept;

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
    Count_
};
void Note(Count c, u64 n = 1) noexcept;

[[nodiscard]] nlohmann::json Report();

} // namespace Core::ViewTable
