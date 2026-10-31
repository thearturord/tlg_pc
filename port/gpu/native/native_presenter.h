// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <chrono>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E6.2d (documents/tlg/native plan/no-emulation-plan.md section 12): the presenter's own state. The video-out driver's present thread used to hold it in local
/// variables and members of an emulated video-out port: the queue of flips the game asked for, the frame limit's time slots, and which display buffer is on screen.
/// Here they are one object. Timing model A of the plan: the present thread's timing model (one tick per vblank period, a flip taken at a slot of the frame limit) is
/// kept exactly, the object only owns it, so a later change of the pacing is a change of this file. The frames themselves are opaque to it (`frame`, `port` are
/// pointers the driver gave and takes back).
///
/// How the pieces fit with the rest of E6.2: the typed flip (native_flip.h) locks the display buffer when it is queued, the present thread takes the flip out of the
/// queue at its slot, shows it, and `NoteShown` tells which buffer the screen had before, which the display object (native_display.h) frees. The queue holds at most
/// as many flips as the display gate lets the game queue (the game cannot draw into a locked buffer: with 3 display buffers at most 2 are queued); `queue_peak` in the
/// report shows the real number.
namespace Core::NativePresenter {

struct Request {
    void* frame{};
    void* port{};
    s64 flip_arg{};
    s32 index{-1};
    bool eop{};

    explicit operator bool() const noexcept {
        return frame != nullptr;
    }
};

using Clock = std::chrono::steady_clock;

// ---- the queue of flips ------------------------------------------------------------------------------------------------------------
/// A flip was prepared (any thread).
void Push(const Request& request);
/// The oldest flip, if there is one.
[[nodiscard]] bool TryTake(Request& out);
/// Waits until a flip is queued or `deadline` passes; true when one is queued.
[[nodiscard]] bool WaitForRequest(Clock::time_point deadline);
[[nodiscard]] std::size_t Pending();

// ---- the frame limit's slots -------------------------------------------------------------------------------------------------------
/// The frame limit in frames per second (0: none); the next slot is now.
void SetFrameLimit(u32 fps, Clock::time_point now);
[[nodiscard]] u32 FrameLimit();
/// With a frame limit a queued flip is shown as soon as it arrives and its slot allows (between vblanks), not on the next vblank tick.
[[nodiscard]] bool ShowBetweenVblanks();
/// Is a flip allowed to be taken now? (no limit, or the slot has come)
[[nodiscard]] bool SlotOpen(Clock::time_point now);
[[nodiscard]] Clock::time_point NextSlot();
/// A flip was taken at `now`: the next slot is one period later, but never more than one period behind `now` (no drift, no catching up in a burst).
void NoteTaken(Clock::time_point now);
/// The present thread looked at a slot that was open and found no flip waiting.
void NoteStarved();

// ---- what is on screen -------------------------------------------------------------------------------------------------------------
/// `index` is shown now (-1 a blank frame); returns the display buffer the screen had before (-1 none), which is free from here on.
[[nodiscard]] s32 NoteShown(s32 index);
/// The video-out port was closed.
void Reset();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativePresenter
