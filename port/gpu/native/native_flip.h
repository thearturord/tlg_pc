// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "native/native_sync.h"

/// E6.2b (documents/tlg/native plan/no-emulation-plan.md section 12): the flip as a typed record. The game ends its frame with `sceGnmSubmitAndFlipCommandBuffers`
/// (video-out handle, display buffer index, flip mode and argument) and a 64 dword "prepareFlip" packet at the tail of the last command buffer that says what else
/// the flip needs: nothing, a label write, an end-of-pipe interrupt, or both. The Gnm driver used to rewrite that tail into PM4 packets (a label write that locks the
/// display buffer, a marker nop, an end-of-pipe packet, a label write) that the command processor parsed again. Here the driver publishes one `Request` and the tail
/// carries only the request's id; the command processor takes the request and does the same four things in the same order from the typed values.
///
/// Rolled out (2026-10-31): the typed request was compared with the old PM4 rewrite over thousands of flips (0 differences), the owner played it, the old rewrite
/// and the switch are gone.
namespace Core::NativeFlip {

/// Word 1 of the tail's nop: the flip request tag ("NDFR"); words 2 and 3 are the request id.
constexpr u32 TagMagic = 0x5246444eu;

struct Request {
    u32 video_out_handle{};
    u32 buffer_index{};
    u32 flip_mode{};
    s64 flip_arg{};
    u64 display_label{};         ///< the display buffer's label in guest memory: written 1 when the flip is queued (the GPU may not draw into the buffer any more)
    bool has_label{};            ///< the game also wants a label written after the flip ...
    u64 label_address{};
    u32 label_value{};
    bool has_fence{};            ///< ... and/or an end-of-pipe signal (with its interrupt)
    NativeSync::Fence fence{};
};

/// The request is built; returns its id (game thread).
[[nodiscard]] u64 Publish(const Request& request);
/// Command processor: the request of `id`; false when it is gone (the ring holds 512).
[[nodiscard]] bool Take(u64 id, Request& out);
/// The kind of the game's prepareFlip packet (the Nop payload type): counted, so the report says which of the four variants the game really uses.
void NoteVariant(u32 payload_type);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeFlip
