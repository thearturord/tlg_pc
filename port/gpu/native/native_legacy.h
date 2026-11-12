// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>

/// THE INDEX OF OLD-PATH SWITCHES (design review 89). Every switch that brings an old, emulator-shaped path back is listed here, so a long play can be bisected
/// without searching the code. All of them are read once at startup (environment variable, or the overlay the launcher passes). Each site the switch gates in the
/// code carries a comment `LEGACY(<name>)` so `grep -rn "LEGACY(" port/gpu` finds them all.
///
/// Deleted paths kept behind a switch: none. The rows for engine buffer bindings (`TLG_LEGACY_ENGINE_ROWS`, design review 89) and the index buffer rows
/// (`TLG_LEGACY_GEOMETRY_ROWS`, note 122) were deleted for good in note 141: they had rotted once the writers left the batch (note 140).
///
/// Steps whose old path is the switch's `=0` (each default on; listed so the whole set is in one place):
///   TLG_EPOCH_PLANNER=0        the barrier planner's per-flush maps (design review 72-74)
///   TLG_BARRIER_GUARD=0        the barrier query on every read binding (67-69)
///   TLG_DROP_DIRTY_APPLIES=0   the light and slot writes' dirty-set work (81-82)
///   TLG_HOOK_OWNED=0           the light and slot writes announced and applied on the raster thread, one batch entry per write (90-94)
///   TLG_HOOK_OWNED_RECORDING=0 the three BeginRecording buffers announced whole and applied on the raster thread every frame (129-131)
///   TLG_INSTANCE_SCRATCH=0     the instance uploader's double buffer (CPU scratch, no GPU reader) applied on the raster thread at every group bind (132-135)
///   TLG_VIEW_TABLE=0           every image binding found by its T# words (set memo key, lookup cache, B3, FindImage) instead of by the engine view id (95-99)
///   TLG_TARGET_TABLE=0         every render and depth target found by its register words (register cache, image generation, decode, FindImage) instead of by the view id (102-106)
///   TLG_WRITTEN_NO_IMAGE=0     every written buffer binding walks the texture cache for images based in its memory, even where the buffer's id says there is none (107-110)
///   TLG_OWN_IMMUTABLE=0        immutable buffers bound through rows and `ObtainBuffer` (83-84)
///   TLG_OWN_MUTABLE=0          default and dynamic buffers bound through rows and `ObtainBuffer` (85-86)
///   TLG_OWN_CLOSE=0            no lazy own pointer (87-88)
///   TLG_OWN_LARGE=0            default and dynamic engine buffers above the resident cap bound through ObtainBuffer (117-119)
///   TLG_OWN_GEOMETRY=0         index and vertex reads through the geometry rows and ObtainBuffer instead of the buffer's own memory by id (117-119)
///   TLG_OWN_RANGE=0            bindings their record does not hold (caller memory, moved dynamic buffers) through ObtainBuffer (120, 123-124)
///   TLG_OWN_PLACE=0            small reads of buffers with no own block copied into the stream ring at every bind (121, 123-124)
///   TLG_OWN_EXACT_END=0        own blocks found only by the rounded end: buffers whose size is not a multiple of 4 miss them (122-124)
///   TLG_ID_MARKS=0             light and slot applies by address (49-51)
///   TLG_IMAGE_BY_ID=0          the image step by address (63-65)
///   TLG_ROW_VERSION=0, TLG_STATIC_ROWS=0, TLG_WRITTEN_ROWS=0, TLG_WIDE_ROWS=0, TLG_TEXSET_BY_ID=0, TLG_HOOK_RING=0 (sections 13-47)
///
/// The play set, default on since 2026-11-11 (review notes A3: every fast-lane check and play ran with them, so the code default is now what was tested; before, only
/// play_start.ps1 set them):
///   TLG_MAP_RING=0             dynamic buffers the bind code never stamps written in place by the game's Map while the raster thread may read them
///   TLG_LIGHT_RING=0           the light constants read from the engine's three-buffer ring instead of a frame-ring copy
///   TLG_SLOT_RING=0            the scene slots read from their own memory instead of a frame-ring copy
///   TLG_SKIN_RING=0            the skinning matrices written into the slot's own memory instead of a frame-ring block
///   TLG_RECORDING_RING=0       the BeginRecording buffers written in place instead of renamed onto frame-ring blocks
///   TLG_FREE_HOLD=0            engine frees run at once instead of after the raster thread is done (TLG_FREE_HOLD_FRAMES default 3, =0 no frame rule)
///   TLG_ORDERED_VIEW=0         the buffer index without the ordered create and delete view
///
/// F1's last writers and leftover bindings, default on since 2026-11-11 (note 138):
///   TLG_COMPUTE_CONST_RING=0   writer 3's constants inside the compute command buffer announced and applied on the raster thread
///   TLG_SKIN_RING_OUT=0        writer 1's skinning matrices (frame-ring blocks) announced and applied on the raster thread
///   TLG_RING_OUT_UNSAMPLED=0   one ring write in 64 still applied in the batch as a check
///   TLG_OWN_GEOMETRY_RANGE=0   index reads that run past their buffer's end through ObtainBuffer
///   TLG_OWN_VERTEX_SPLIT=0     merged vertex ranges through ObtainBuffer instead of each member buffer's own memory
///   TLG_OWN_VERTEX_RANGE=0     merged vertex ranges past their buffer's end through ObtainBuffer (note 139)
///   TLG_OWN_VERTEX_SLACK=0     merged vertex ranges ending up to 64 bytes past their buffer (fetch rounding) through ObtainBuffer (note 139)
///   TLG_PROGRAM_CONSTS=0       constants inside shader memory (reads offset by the program base) copied into the stream ring at every bind (note 144)
///   TLG_OWN_INDEX_BY_ADDRESS=0 index lists of draws that name no geometry id copied into the stream ring at every draw (note 145)
///   TLG_TEXTURE_AT_CREATION=0  a texture enters the engine object table only at its first view (notes 114 to 116, 147)
///   TLG_STORAGE_TABLE=0        storage (written) images found by their T# words at every bind, outside the view table (notes 111, 112, 147)
///   TLG_HOOK_DIFF=0            every draw record looks its fixed state up in the shared cache under its lock (notes 153, 157)
///   TLG_DISPATCH_EXECUTOR=0    every buffer binding of a dispatch through the old body (BindBufferLegacy) instead of the executor's native sources (note 159)
///   TLG_TARGET_TAIL=0          a sampled render target of the draw (a feedback loop) is not learnt by the view table and is bound by the old texture path (note 160)
///   TLG_PLAN_POINTERS=0        every draw plan copies the record's fixed state (about 1 KB) instead of pointing at the cached state (note 161)
///   TLG_OWN_SYNC=0             a dispatch read of an engine buffer with CPU bytes waiting goes to the old body (the R1b copy of the object in the stream ring) (note 162)
///   TLG_DYNAMIC_TEXTURES=0     a dynamic texture (Map discard every frame) found by its T# address in the texture cache instead of its own image by id (note 163)
///   TLG_DRAW_EXECUTOR=0        every buffer binding of a draw through the old body (BindBufferLegacy) instead of the executor's native sources (note 165)
///   TLG_RESOLVED_ANSWERS=2     the R1b / R2 answers back: the front resolves an engine object per read binding, the binder takes its stream copy or resident arena (note 166)
///   TLG_WIDE_PAST_END=0        4 MB windows that start in the frame ring and run past its end through ObtainBuffer instead of the hook ring (note 167)
namespace Core::Legacy {


/// Every switch of the index with its value as read at startup ("unset" = the default), and whether a deleted path is back. Published as `legacy`.
[[nodiscard]] nlohmann::json Report();

} // namespace Core::Legacy
