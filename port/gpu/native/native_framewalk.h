// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

/// Frame walk: two keys to look at what happened in the frames before a glitch (documents/tlg/native plan/engine/frame-walk.md).
///   Shift+F5  starts recording into a ring (the frame timeline capture in ring mode: every event of the last seconds, including the resource uses of every draw)
///   F5        freezes the game at the next frame boundary (the game's render thread waits at the start of sceGnmSubmitAndFlipCommandBuffers), stops the recording, writes
///             user/log/framewalk_<time>/ (draw table of the last frames, flight recorder, counters, a note) and user/log/timeline_fw_<time>.bin/.json; while frozen the debug server
///             (/memory, /framewalk, /resources, ...) reads the live state of that moment. F5 again resumes.
/// Without a recording F5 freezes too (the folder then has no timeline). Switch: TLG_FRAMEWALK=0 turns the keys off.
namespace Core::FrameWalk {

enum class State : u32 { Idle, Recording, Freezing, Frozen };

/// Key handlers (any thread, return at once).
void KeyRecord();
void KeyFreeze();

/// Game's render thread, first thing in the submit and flip call: blocks while a freeze is requested or active. One relaxed load when nothing is going on.
void AtSubmit();

[[nodiscard]] State Current();

/// The on screen toast (kind 9 recording, 10 frozen, 11 resumed): set once by the shim.
void SetToastHook(void (*toast)(int kind));

/// JSON for the debug server's /framewalk: the state, the folders and files, what the keys do. `action` "record", "freeze" or "resume" does what the key would.
std::string Report(const std::string& action);

} // namespace Core::FrameWalk
