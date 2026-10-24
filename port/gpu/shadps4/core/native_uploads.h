// SPDX-License-Identifier: GPL-2.0-or-later
// N5c (documents/tlg/native-renderer-plan.md): uploads announced by the engine. The writers the update audit knows (Map / Unmap, the skinning and light constant
// functions, BeginRecording, buffer creation) say which guest memory they wrote. The announcements are collected here, published at a defined point of the command
// stream, and the raster thread uploads the announced ranges before the first draw or dispatch that follows the publication, so the pass that uses the data is not
// cut by an upload in its middle.
//
// Publication: an announcement made on the render thread (the thread of RenderDev::BeginRecording) is stamped at once with the number the next record will get; one made
// on another thread waits until the render thread starts recording the next frame (BeginRecording), then it is stamped. The front thread (command processor) takes the
// entries whose stamp is not above the number of the record it is about to queue and queues one upload operation in front of that record. The raster thread runs it
// (`Rasterizer::PreSynchronize`): the page tracker still marks what the CPU changed, so this step uploads the dirty pages of the announced ranges; whatever it does not
// find clean at the draw is counted as an upload "on demand" (announced too late or not at all). Off by default: `TLG_NATIVE_UPLOADS=1`.

#pragma once

#include <atomic>
#include <vector>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeUploads {

struct Range {
    u64 begin;
    u64 size;
};

bool Enabled();

/// A writer announced `size` bytes at `begin` (any thread). Only writers whose memory is a buffer are kept (`writer`: see NativeUpdates).
void Announce(u32 writer, u64 begin, u64 size, bool on_render_thread, u64 stamp_now);
/// The render thread starts recording a frame: the announcements of the other threads are stamped.
void PublishPending(u64 stamp_now);
/// Front thread: the entries due before the record `seq` is queued, merged. False when there are none.
bool TakeDue(u64 seq, std::vector<Range>& out);

/// Raster thread: counters.
void NoteApplied(u64 ranges, u64 bytes, u64 uploaded_bytes);
/// SynchronizeMemory of the buffer cache copied `bytes` from guest memory (on this thread, inside a pre-synchronization or on demand).
void NoteUploaded(u64 bytes);
void SetPreSyncActive(bool active);
/// The upload ring (UntrackedRing): a read uploaded `bytes` of it / found them uploaded already in this command buffer.
void NoteRingUpload(u64 bytes);
void NoteRingSkipped(u64 bytes);
/// Where SynchronizeMemory copied from (guest address): the on-demand uploads are kept by 256 KB block to name the biggest.
void NoteUploadedRange(u64 address, u64 bytes);

nlohmann::json Report();

} // namespace Core::NativeUploads
