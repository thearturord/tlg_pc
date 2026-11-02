// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// F2.2: the announced content store. One shared piece that holds the bytes of what the engine wrote, taken at the moment the writer finished (the announcement), so that a reader
/// (today the image refresh, later the buffer readers) can ask "what did the engine write into this range" without reading the memory the console would have had.
///
/// Two kinds of content:
///  - bytes: a writer announced a range after it wrote it (texture initialization, UpdateSubresource, Unmap, buffer creation): the bytes are copied then, on the writing thread. The newest
///    entry wins where entries overlap. Writers that announce before they write (skinning, lights, recording stores; writers 1, 2, 4) carry no image content and are not kept.
///  - copies: a GPU copy (`CopyBuffer`) between two ranges is a record that is resolved to bytes from the store when it runs (an approximation: the source is taken as the store holds it at that
///    moment, not as of the moment the copy was issued).
///
/// Shadow stage (`TLG_F2_STORE=1`, default off, read once): nothing reads the store for rendering. `RefreshImage` asks `CompareImage` after each upload and the report says how the store
/// compares with the memory the upload read. Cap: `TLG_F2_STORE_MB` (default 1024); the oldest entries are dropped first (counted). Report: /stats native_backend.device.f2_store.
namespace Core::NativeContent {

/// The store is active: the shadow (`TLG_F2_STORE=1`, keeps every announcement, compares uploads) and/or the ring swap (`TLG_F2_RING_STORE`).
[[nodiscard]] bool On() noexcept;
[[nodiscard]] bool ShadowOn() noexcept;
/// F2.2 swap (`TLG_F2_RING_STORE`, default on since 2026-11-03, `=0` turns it off, read once): the store keeps the announcements inside the frame ring (up to `TLG_F2_RING_STORE_KB`, default 256 KB, `TLG_F2_RING_STORE_MB` of
/// them, default 256), each stamped with the record number it was made at, and `RefreshImage` reads a ring image's bytes from the newest entry stamped at or before the draw's record instead of
/// from the ring's memory, which a later allocation may have reused by the time the raster thread uploads. Bytes no entry covers still come from memory.
[[nodiscard]] bool RingSwapOn() noexcept;
/// The frame allocator's ring (found by the engine's allocator hook).
void SetRing(u64 begin, u64 end);
/// `staged` holds the guest memory bytes of [address, address + size) (what the upload would read): if the range lies in the ring, the bytes of the entries the current draw may see replace them.
void OverlayImage(u64 address, u64 size, u8* staged);

/// The engine wrote [address, address + size) and has finished (any thread). `writer` is the number NativeUpdates::NoteWriterRange gets.
void Announce(u32 writer, u64 address, u64 size, u64 stamp = 0);
/// The raster thread runs a GPU copy of guest memory to guest memory.
void NoteCopy(u64 destination, u64 source, u64 size);
/// The engine deleted the memory's owner or reused it: nothing is dropped in the shadow stage, this only counts (so the report can say how much content outlived its owner).
void NoteDeleted(u64 address, u64 size);
/// A Map the hook saw (counted by resource kind, map type and the path taken, see Report "maps_note").
void CountMap(u32 kind, u32 type, u32 path);
/// A write whose bytes are not kept (it announces nothing or before it writes): only the range and the order are remembered, so a differing upload can be attributed (writers: 1 skin, 2 lights,
/// 3 constant buffer version in the frame ring written by UpdateSubresource, 4 recording stores).
void NoteMark(u32 writer, u64 address, u64 size);

enum class Group : u8 { FirstUpload, RepeatUpload, GpuSource };
struct ImageUpload {
    u64 image_uid{};
    u64 address{};
    u64 size{};
    u32 width{};
    u32 height{};
    u32 format{};
    Group group_hint{Group::FirstUpload}; ///< `GpuSource` when the upload read GPU written memory (the store is only asked how much it covers)
    u32 caller{};
};
/// `RefreshImage` (raster thread) uploaded the image from the memory [address, address + size): compare the store with that memory.
void CompareImage(const ImageUpload& upload);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeContent
