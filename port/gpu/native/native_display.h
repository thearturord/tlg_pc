// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E6.2c (documents/tlg/native plan/no-emulation-plan.md section 12): the display buffer gate as a native object. The game starts each frame with
/// `sceGnmInsertWaitFlipDone` on the display buffer it is going to draw into: the GPU must not touch a buffer that is queued for the screen or on it. On the console
/// that was a wait of the command processor on a label word in memory (1 when the flip was queued, 0 when a later frame is shown). Here the state "this display buffer
/// is free" lives in this object: the typed flip locks it, the present thread releases it, and the typed wait (a tag in the stream, not a WAIT_REG_MEM packet) waits for
/// it. The label words in the video-out port's memory are still written at the same moments (they are the console's interface; the game never reads them), and every
/// wait that passes compares the state with the label word (`/frame_events`, section display: `label_disagrees`).
namespace Core::NativeDisplay {

/// Word 1 of the tail nop the Gnm driver writes in place of the wait packet ("NDDW"); word 2 is the display buffer index.
constexpr u32 WaitTagMagic = 0x5744444eu;
constexpr u32 MaxBuffers = 16;

/// The video-out port's label words (read for the comparison only).
void SetLabels(const volatile u64* labels);

/// The typed flip was queued: the buffer may not be drawn into.
void Lock(u32 buffer);
/// A later frame is shown (or the buffer was registered): the buffer is free.
void Release(u32 buffer);
[[nodiscard]] bool IsFree(u32 buffer);
/// Blocks the calling thread until the buffer is free (used when the command processor has nothing else to do).
void WaitFree(u32 buffer);
/// A wait passed: the label word of the buffer is compared with the state, counted.
void NoteWaitPassed(u32 buffer);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeDisplay
