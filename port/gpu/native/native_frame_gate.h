// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json.hpp>
#include "common/types.h"

/// Frames in flight: one setting for every place that works on the "frame N+1" rule (documents/tlg/native plan/no-emulation-plan.md, section 13).
/// Endpoint `/frame_gate`. Not an in-game setting: `user/custom_configs/<serial>.frames.json` `{"frames_in_flight": 1}`, or the environment variable `TLG_FRAMES_IN_FLIGHT` (wins).
///
///   0  (default since 2026-10-31) the game waits for its own frame: `sceGnmSubmitAndFlipCommandBuffers` returns when the frame's end-of-pipe interrupt came, the presenter waits for the
///      frame it just made (lowest input delay)
///   1  the earlier default: the game submits frame N after the end-of-pipe of frame N-1; the Gnm submit lock waits for an idle GPU; the presenter waits for the
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
/// M5: `TLG_OWNERSHIP_GATE=1` (depth 2 and `TLG_FRAME_STORE=2` needed): the game's frame slot wait counts frames the raster thread has not reached, not frames the GPU has not
/// finished; the game's own end-of-pipe gate gets one more credit. Off by default.
bool OwnershipGate();
/// M5b: `TLG_OWNERSHIP_EARLY_IRQ=1` (with the ownership gate): the frame's end-of-pipe interrupt (the guest's "EOP QUEUE" event that releases the engine's submit-done thread and with it the
/// next submit) is raised when the raster thread has reached the frame end, not when the GPU is done. The label write stays at GPU completion (readers of GPU results wait for it).
bool EarlyInterrupt();

/// The two waits that depth 0 adds to depth 1, switchable on their own (to find which one matters): the game's thread waits for the end-of-pipe interrupt of its
/// own frame (`WaitOwnFrame`), and the presenter waits for the frame it just made (vk_presenter.cpp). Default: both on at depth 0, both off otherwise.
/// Json keys `game_waits_own_frame` (0, 1, 2) / `presenter_waits_own_frame` (0, 1), environment `TLG_FIF_GAME_WAIT` / `TLG_FIF_PRESENTER_WAIT` (win).
/// 0 off; 1 until the frame's end-of-pipe interrupt (the GPU is done and the readbacks landed); 2 until the raster thread has executed everything the frame recorded
/// (`NoteRasterReached`: our side has read all the guest memory the frame needs, the GPU may still run) - tells a late read of guest memory from a late result.
int GameWaitMode();
bool GameWaitsOwnFrame();
/// The raster thread passed the end of a frame (queued in order behind the frame's work, before its end-of-pipe signal)
void NoteRasterReached();
/// Frames the game has submitted, and frames the raster thread has reached the end of (loader gate: a writer that reuses freed memory waits until the raster thread has
/// executed every frame that was submitted when the memory was freed).
u64 SubmittedFrames();
u64 RasterReachedFrames();
/// Blocks (up to 2 s, counted and reported like the slot wait) until the raster thread has reached the end of `target_frames` frames.
void WaitRasterReachedFrames(u64 target_frames);
bool PresenterWaitsOwnFrame();

/// M5b early interrupt (`TLG_OWNERSHIP_EARLY_IRQ=1`). The end-of-pipe interrupt of a frame is raised on whichever comes first: (a) the raster thread has recorded the frame and the occlusion
/// answers of all earlier frames are written (the game reads them without waiting, so a frame it records must not see older ones than at the late interrupt: owner play 2026-11-04, black spots
/// with real occlusion), (b) a poll wrote the answers that frame was waiting for, (c) the GPU finished the frame (the late interrupt, as before). Each frame's interrupt is raised once, in order.
/// The number of the frame the next interrupt-carrying end of pipe belongs to (command processor thread, in order)
u64 NextEarlyOrdinal();
/// Raster thread: everything frame `ordinal` recorded was executed
void EarlyFrameRecorded(u64 ordinal);
/// Any thread: the GPU finished frame `ordinal`; raises its interrupt (and the earlier ones) if it is still waiting
void EarlyGpuDone(u64 ordinal);

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
