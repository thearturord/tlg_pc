// SPDX-License-Identifier: GPL-2.0-or-later
// Step 4 of note 79, part 1 (design review notes 137 C and 143): coverage per operation. Counters only, off by default (`TLG_COVERAGE_CENSUS=1`).
//
// An executor can run a draw or a dispatch without the old binding loop only when every one of its bindings is answered by a native path: own memory by engine id, the
// hook ring (frame ring windows), the front's constant ring, an engine view (view table). The raster thread marks each binding the old path answered (a buffer cache request
// that is not a hook ring answer and not an own-memory audit, a texture stage the view table did not bind, a vertex or index range from the buffer cache) and counts, per
// operation, whether any was left. Published as `coverage` (the debug server, /stats).

#pragma once

#include "common/types.h"

namespace Core::Coverage {

extern bool g_on;

[[nodiscard]] inline bool On() noexcept {
    return g_on;
}

/// What the old path answered for one binding of the operation being bound.
enum Old : u32 { OldBuffer, OldBufferWritten, OldImageStage, OldVertex, OldIndex, OldCount };

/// The raster thread starts binding an operation (`compute`: a dispatch). Closes the previous one.
void Begin(bool compute) noexcept;
/// The old path answered a binding of the current operation.
void NoteOld(Old what) noexcept;
/// The detail of an old-path answer: `cls` 0 no engine id / 1 an engine id, `answer` 0 a stream ring copy / 1 another buffer (arena, placed block) / 2 the range meets the frame ring,
/// `program` the stage's program hash, `size` the binding's size.
void NoteOldDetail(Old what, u32 cls, u32 answer, u64 program, u64 size) noexcept;
/// Note 151 (plan R2.5): why the view table did not bind a texture stage (`ImageReason`).
enum ImageReason : u32 {
    IrNoImagesOrPlan, IrWrittenNotReal, IrMipFallback, IrNotSingleLoad, IrSeveralBindings, IrNoRecordWords, IrNoViewId, IrNoEntry, IrNotInClass, IrNoPayload,
    IrStale, IrAudit, IrTailSameImageTwice, IrTailCheckFailed, IrStorageMarkMismatch, IrTableOff, IrCount
};
void NoteImageReason(u32 reason, u64 program) noexcept;

} // namespace Core::Coverage
