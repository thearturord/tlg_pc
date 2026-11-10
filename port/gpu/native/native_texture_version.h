// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Design review note 100, section 79 step 3b (`TLG_TEXTURE_VERSION`: 0 off (default), 1 shadow; 2 real is not built yet): an engine texture's CPU content by its id.
/// The class: a texture that is not dynamic and whose memory holds no live engine buffer and no other live texture (decided by the engine object table, in game order; revoked for
/// good when a buffer or texture is later created over it). The writer 6 hooks (2D initialization, 1D/3D creation, UpdateSubresource of a texture), which all run after the write,
/// bump the texture's content version. An image of a class texture keeps the version it was uploaded at; the shadow compares, at each refresh, the version's answer with the flags'.
namespace Core::TextureVersion {

[[nodiscard]] u32 Mode() noexcept;

struct Slot {
    std::atomic<u64> version{1};
    std::atomic<u32> state{0}; ///< 1 in the class, 2 excluded
    [[nodiscard]] bool InClass() const noexcept {
        return state.load(std::memory_order_acquire) == 1;
    }
};

// ---- the engine object table (under its lock) and the hooks ----
/// Texture `id` was created: in the class or not.
void Created(u64 id, bool in_class) noexcept;
/// Texture `id` leaves the class for good (a buffer or texture was created over its memory).
void Exclude(u64 id) noexcept;
/// A writer 6 hook finished writing texture `id`.
void Bump(u64 id) noexcept;
/// The slot of texture `id` (nullptr: never created in the class).
[[nodiscard]] Slot* Find(u64 id) noexcept;

// ---- the texture cache and the rasterizer ----
enum Count : u32 {
    TexturesInClass,       ///< textures created in the class
    TexturesNotInClass,    ///< textures created outside it (dynamic, or memory shared with a buffer or texture)
    TexturesExcluded,      ///< class textures excluded later
    Bumps,                 ///< writer 6 writes of a class texture
    BumpsNotInClass,       ///< writer 6 writes of other textures
    ImagesLinked,          ///< images linked to their class texture (at a view table entry)
    RefreshChecked,        ///< refreshes of a linked image of a class texture (each bind)
    RefreshFirstUpload,    ///< ... never uploaded since linked: not compared
    RefreshGpuSkipped,     ///< ... the GPU wrote the image (3c): not compared
    BothClean,             ///< version and flags: unchanged
    BothDirty,             ///< version and flags: changed
    VersionOnly,           ///< the version moved, the flags are clean (an extra upload in the real mode: the hook's bump is seen before the batch's mark)
    FlagsOnly,             ///< the flags are dirty, the version did not move (VERSION_CLEAN_FLAGS_DIRTY_MUST_BE_0: the real mode would miss an upload)
    AddressWriteOnClass,   ///< an announcement of another writer dirtied an image of a class texture (ADDRESS_WRITE_ON_CLASS_TEXTURE_MUST_BE_0)
    Count_
};
void Note(Count c) noexcept;

[[nodiscard]] nlohmann::json Report();

} // namespace Core::TextureVersion
