// SPDX-License-Identifier: GPL-2.0-or-later
// N4a (documents/tlg/native-renderer-plan.md): the pass audit. A pass is one rendering instance of the scheduler (BeginRendering .. EndRendering). The raster thread
// reports every draw it records and the scheduler reports where a pass starts and why it ends; the audit says which passes consist of draws only, how many
// draws they hold, and what stops a pass early. It decides how a pass recorder can work: only a pass that is a plain run of draws (no barrier, no resource written
// by a shader, no break) can be prepared in one go. Off by default: `TLG_PASS_AUDIT=1` or the debug server endpoint `/passes?start=1`.

#pragma once

#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativePass {

extern std::atomic<bool> enabled;

/// Why a pass ended.
enum class EndCause : u8 {
    Other,    ///< a copy, clear, dispatch or other operation ended it
    Barrier,  ///< the barrier tracker had to flush before a draw
    NewState, ///< the next draw needs other attachments
    Submit,   ///< the command list was submitted
};

void Start();
/// Scheduler (main stream): a rendering instance starts with `colors` colour attachments of the given extent.
void PassBegin(u32 colors, u32 width, u32 height);
/// Says why the pass that ends next ends (read by PassEnd, default Other).
void SetEndCause(EndCause cause);
/// `site`: the return address of whoever ended the pass (kept for the ends that are not a barrier, a state change or a submit: which operation cut the pass).
void PassEnd(const void* site = nullptr);
/// Raster thread: a draw was recorded into the open pass. `draw_class` is the game's tag class (15 untagged, 4 indirect), `barrier` that the barrier tracker
/// asked for a flush before it, `pipeline` identifies the pipeline.
void NoteDraw(u32 draw_class, bool indirect, bool barrier, const void* pipeline);
/// Raster thread: the draw's shaders write a buffer / an image.
void NoteWrite(bool image);
/// An operation that ends passes names its own caller (used instead of the return address of the scheduler call), e.g. the buffer copy.
void SetEndSite(const void* site);

nlohmann::json Report();

} // namespace Core::NativePass
