// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E6.2a (documents/tlg/native plan/no-emulation-plan.md section 12): the frame events of the console's GPU, as named calls. They replaced `Platform::IrqC`, the
/// emulator's interrupt controller: the command processor and the raster thread say what happened (the end of pipe signal, a compute queue's release, the GPU is idle,
/// a flip is ready) and the OS side that the game talks to (the Gnm driver's event queues, the submission lock, the video-out flip) subscribes. The ordering is the
/// controller's: persistent subscribers in registration-independent order under the event's lock, then the oldest one-time subscriber (video-out flips). Every event is
/// counted (endpoint /frame_events), so a later step can show that it changed nothing: the end of pipe count equals the frame gate's interrupt count.
namespace Core::NativeFrameEvents {

/// The console's interrupt numbers (what the game's event queues name): compute queue release 0 to 6, flip 8, GPU idle 9, end of pipe 0x40.
constexpr u32 FlipReadyId = 0x08;
constexpr u32 GpuIdleId = 0x09;
constexpr u32 EndOfPipeId = 0x40;
constexpr u32 MaxId = 0x40;

using Handler = std::function<void(u32 id)>;

/// The graphics queue's end of pipe signal: it comes after the GPU's work and the readbacks of everything the game submitted before it.
void EndOfPipe();
/// A compute queue's release (`pipe` 0 to 6).
void ComputeRelease(u32 pipe);
/// The GPU finished everything it was given (the submission lock opens).
void GpuIdle();
/// A flip request that the game queued with its frame is ready to be shown.
void FlipReady();

/// A persistent subscriber (one per `uid` and event): the Gnm driver's event queues, the submission lock.
void Subscribe(u32 id, Handler handler, void* uid);
void Unsubscribe(u32 id, void* uid);
/// A one-time subscriber: runs for the next signal of the event, one signal per subscriber, oldest first (video-out's `SubmitEopFlip`).
void SubscribeOnce(u32 id, Handler handler);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeFrameEvents
