// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json.hpp>

#include "common/types.h"

// Streaming of the draws (documents/tlg/native-streaming-design.md): while the render thread records a frame, the part of the command buffer that is final is
// handed to the command processor as a submit of its own; the real submit then hands over only the rest. Setting cp_stream_submit, default 0.
namespace Core::StreamSubmit {

/// The draw hook (render thread): may publish the final part of the command buffer being recorded.
void AtHook(u64 ctx);

/// The real submit (render thread) of the command buffer that starts at `base`: the dwords already handed over (0: none). The part handed over is compared with
/// the memory first; a difference switches the feature off.
u32 Consumed(const u32* base, u32 dwords);

/// The real submit has been queued: the frame is over for the streaming.
void RealSubmitQueued();

/// Something showed that streaming is not safe (data or stream changed after it was handed over): the feature switches itself off.
void NoteViolation(const char* what);

nlohmann::json Report();

} // namespace Core::StreamSubmit
