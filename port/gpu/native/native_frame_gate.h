// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json.hpp>
#include "common/types.h"

/// Frames in flight: one setting for every place that works on the "frame N+1" rule (documents/tlg/native plan/no-emulation-plan.md, section 13).
/// Endpoint `/frame_gate`. Not an in-game setting: `user/custom_configs/<serial>.frames.json` `{"frames_in_flight": 1}`, or the environment variable `TLG_FRAMES_IN_FLIGHT` (wins).
///
///   0  the game waits for its own frame: `sceGnmSubmitAndFlipCommandBuffers` returns when the frame's end-of-pipe interrupt came, the presenter waits for the
///      frame it just made (lowest input delay)
///   1  today (default): the game submits frame N after the end-of-pipe of frame N-1; the Gnm submit lock waits for an idle GPU; the presenter waits for the
///      frame one flip back (`frames_queued`, 1)
///   2  the game may be two frames ahead (the most its engine supports: 3 frame contexts and 3 display buffers, read in Ghidra): the Gnm idle wait goes and a frame submit waits for
///      "fewer than 2 unfinished frames", the game's own submit gate gets one credit (see CreditGuestGate), the presenter waits for the frame two flips back
///
/// The places that read the gate: (1) the game's submit counter (CreditGuestGate), (2) the Gnm submit lock (gnmdriver.cpp WaitSubmitAllowed / SubmitDone),
/// (3) the presenter's wait for frames ahead (vk_presenter.cpp), (4) the flip queue and display buffer labels (counted: PeakUnfinished, flips in the driver),
/// (5) fences, occlusion answers and readbacks (they are ordered on the raster thread, nothing to change; the age of an answer is `Depth()` frames at most),
/// (6) the double buffered host resources (GPU side only, ordered by the queue; checked, see the section).
namespace Core::FrameGate {

constexpr int MaxDepth = 2;

/// 0, 1 or 2 (clamped), read once
int Depth();

/// The two waits that depth 0 adds to depth 1, switchable on their own (to find which one matters): the game's thread waits for the end-of-pipe interrupt of its
/// own frame (`WaitOwnFrame`), and the presenter waits for the frame it just made (vk_presenter.cpp). Default: both on at depth 0, both off otherwise.
/// Json keys `game_waits_own_frame` (0, 1, 2) / `presenter_waits_own_frame` (0, 1), environment `TLG_FIF_GAME_WAIT` / `TLG_FIF_PRESENTER_WAIT` (win).
/// 0 off; 1 until the frame's end-of-pipe interrupt (the GPU is done and the readbacks landed); 2 until the raster thread has executed everything the frame recorded
/// (`NoteRasterReached`: our side has read all the guest memory the frame needs, the GPU may still run) - tells a late read of guest memory from a late result.
int GameWaitMode();
bool GameWaitsOwnFrame();
/// The raster thread passed the end of a frame (queued in order behind the frame's work, before its end-of-pipe signal)
void NoteRasterReached();
bool PresenterWaitsOwnFrame();

/// A frame was handed to the command processor (`sceGnmSubmitAndFlipCommandBuffers`, before the submit)
void NoteSubmitFlip();
/// The end-of-pipe interrupt of a frame (the one the game's submit thread waits for) was raised
void NoteEndOfPipe();

/// Depth 2: blocks until fewer than `Depth()` frames are unfinished (times out after 2 s, reported once as a rare event). Depth 0 and 1: returns at once.
void WaitForSlot();
/// Depth 0: blocks until every submitted frame has its end-of-pipe interrupt (times out like WaitForSlot).
void WaitOwnFrame();

/// How many flips back the presenter waits for: 2 at depth 2, the value of the `frames_queued` setting otherwise.
/// `queued_setting` is that setting (0 = unbounded).
u32 PresenterAhead(u32 queued_setting);

nlohmann::json Status();

} // namespace Core::FrameGate
