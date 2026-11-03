// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

/// L3 of documents/tlg/native plan/diagnostics-logging-plan.md: the glitch marker. The owner presses F6 (full) or F7 (light) when something looks wrong; a thread of its own then writes
/// user/log/glitch_<n>_<time>/ with the flight recorder's tail, the draw table of the last two frames, the counters that moved (against the 20 s snapshot), the presented frame and a note.
/// The game is never paused or slowed: the key press sets a flag, the writer thread does the rest, the only thing on the render path is one table row per draw and dispatch.
/// Switch: TLG_MARKER=0 turns the table and the keys off.
namespace Core::Marker {

/// What the table says about one draw or dispatch (raster thread).
struct Row {
    u64 tick{};      ///< Flight::Tick() when the row was written
    u64 seq{};       ///< record sequence number (0: the game's own draw, no record)
    u64 vs_hash{};   ///< vertex (or compute) program hash
    u64 ps_hash{};   ///< pixel program hash, 0 if none
    u64 color0{};    ///< guest address of the first colour target
    u64 depth{};     ///< guest address of the depth target
    u32 caller{};    ///< game call site (Ghidra address) of the record, 0 if unknown
    u32 frame{};     ///< frames presented so far
    u32 count{};     ///< indices or vertices (compute: group count x)
    u32 instances{}; ///< instances (compute: group count y)
    u8 kind{};       ///< 0 draw, 1 indirect draw, 2 dispatch, 3 indirect dispatch
    u8 draw_class{}; ///< the native draw class (0 shadow depth, 1 g buffer, 2 forward, 3 compute, 4 other, 15 untagged)
    u8 flags{};      ///< 1 planned from a record, 2 pipeline resolved by the native plan
    u8 colors{};     ///< colour targets bound
    u16 width{};
    u16 height{};
    u32 pad{};
};
static_assert(sizeof(Row) == 80);

constexpr u32 TableRows = 16384; ///< power of two; about five frames of the busiest scene

[[nodiscard]] bool Enabled();

/// Raster thread (or whichever thread executes the draw): adds one row.
void NoteRow(const Row& row);

/// What the shim gives the marker: the screenshot into a path and the on screen toast (kind 7 full, 8 light). Called once at start.
struct Hooks {
    void (*screenshot)(const std::string& path){};
    void (*toast)(int kind){};
};
void SetHooks(const Hooks& hooks);

/// Any thread (the key handler): asks for a marker. `mode` 1 full, 2 light. Returns at once.
void Request(u32 mode);

/// Markers written so far.
[[nodiscard]] u32 Written();

/// The same writer, started by the debug server's /marker?mode=1|2 for headless runs; the answer names the folder the files go to.
std::string RequestForTest(u32 mode);

} // namespace Core::Marker
