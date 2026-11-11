// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>

/// THE INDEX OF OLD-PATH SWITCHES (design review 89). Every switch that brings an old, emulator-shaped path back is listed here, so a long play can be bisected
/// without searching the code. All of them are read once at startup (environment variable, or the overlay the launcher passes). Each site the switch gates in the
/// code carries a comment `LEGACY(<name>)` so `grep -rn "LEGACY(" port/gpu` finds them all.
///
/// Deleted paths kept behind a switch here (default off; `=1` brings the old path back):
///   TLG_LEGACY_ENGINE_ROWS=1   `engine_rows`: rows for shader buffer bindings of engine buffers, on the front (rows ahead: `RowShare::Find`) and the raster thread
///                              (`RowsServe`, `RowsLearn`, the rows-ahead bind). Deleted 2026-11-10 (design review 89): own memory by id answers them, the rest take
///                              `ObtainBuffer`, whose answer the rows cached. Not covered: the index buffer rows (`TLG_GEOMETRY_ROWS`), which stay.
///   TLG_LEGACY_GEOMETRY_ROWS=1 `geometry_rows`: the index buffer rows (design review 76-78). Deleted 2026-11-11 (note 122) while `TLG_OWN_GEOMETRY` is on: own memory by
///                              the draw's geometry ids answers index reads first, the rest take `ObtainBuffer`, whose answer the rows cached.
///
/// Steps whose old path is the switch's `=0` (each default on; listed so the whole set is in one place):
///   TLG_EPOCH_PLANNER=0        the barrier planner's per-flush maps (design review 72-74)
///   TLG_BARRIER_GUARD=0        the barrier query on every read binding (67-69)
///   TLG_GEOMETRY_ROWS=0        the index buffer by address (76-78)
///   TLG_DROP_DIRTY_APPLIES=0   the light and slot writes' dirty-set work (81-82)
///   TLG_HOOK_OWNED=0           the light and slot writes announced and applied on the raster thread, one batch entry per write (90-94)
///   TLG_VIEW_TABLE=0           every image binding found by its T# words (set memo key, lookup cache, B3, FindImage) instead of by the engine view id (95-99)
///   TLG_TARGET_TABLE=0         every render and depth target found by its register words (register cache, image generation, decode, FindImage) instead of by the view id (102-106)
///   TLG_WRITTEN_NO_IMAGE=0     every written buffer binding walks the texture cache for images based in its memory, even where the buffer's id says there is none (107-110)
///   TLG_OWN_IMMUTABLE=0        immutable buffers bound through rows and `ObtainBuffer` (83-84)
///   TLG_OWN_MUTABLE=0          default and dynamic buffers bound through rows and `ObtainBuffer` (85-86)
///   TLG_OWN_CLOSE=0            no lazy own pointer (87-88)
///   TLG_OWN_LARGE=0            default and dynamic engine buffers above the resident cap bound through ObtainBuffer (117-119)
///   TLG_OWN_GEOMETRY=0         index and vertex reads through the geometry rows and ObtainBuffer instead of the buffer's own memory by id (117-119)
///   TLG_ID_MARKS=0             light and slot applies by address (49-51)
///   TLG_IMAGE_BY_ID=0          the image step by address (63-65)
///   TLG_ROW_VERSION=0, TLG_STATIC_ROWS=0, TLG_WRITTEN_ROWS=0, TLG_WIDE_ROWS=0, TLG_TEXSET_BY_ID=0, TLG_HOOK_RING=0 (sections 13-47)
namespace Core::Legacy {

/// TLG_LEGACY_ENGINE_ROWS=1: the rows for shader buffer bindings of engine buffers (see above).
[[nodiscard]] bool EngineRows() noexcept;
/// TLG_LEGACY_GEOMETRY_ROWS=1: the index buffer rows (see above).
[[nodiscard]] bool GeometryRows() noexcept;

/// Every switch of the index with its value as read at startup ("unset" = the default), and whether a deleted path is back. Published as `legacy`.
[[nodiscard]] nlohmann::json Report();

} // namespace Core::Legacy
