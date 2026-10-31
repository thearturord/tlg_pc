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

// ---- snapshots (N5d) ---------------------------------------------------------------------------------------------------------------
// The writers' announcements carry the data: at the end of a write (Unmap) the written bytes are copied into a ring of host memory, stamped with the number of the next
// record. A record that reads a range of such memory takes the bytes as they were when it was issued, whatever the command processor's lag and whatever the render
// thread wrote into the same buffer for the next draw. `TLG_SNAPSHOTS=1` (needs the constant ring, `TLG_CONST_RING=3` or more, which carries the bytes to the GPU).

bool SnapshotsEnabled();
/// A write of [begin, begin + size) just ended (stamp = the number the next record gets). Ranges over 64 KB are not kept.
void SnapshotTake(u64 begin, u64 size, u64 stamp);

enum class SnapshotResult : u8 {
    None,    ///< no snapshot touches the range (it was never announced, or the snapshots are gone): the caller reads guest memory as before
    Partial, ///< the newest write before the record touches the range but does not cover it: unknown, the caller must not use guest memory
    Found,   ///< `data` holds the range as the newest write before the record left it
};
/// The range [begin, begin + size) as the writes before record `seq` left it. For `Found` the bytes are copied to `data` (size bytes).
SnapshotResult SnapshotRead(u64 begin, u64 size, u64 seq, void* data);

/// A snapshot differed from guest memory when the front thread planned the record: `diff_bytes` of `size` bytes differ, the first at `first_offset` (snapshot word, guest
/// word); `who` is the announcer that wrote the range last (NativeUpdates::WhoWrote). Counted by how much of the range differs, the first samples are kept for the report.
void NoteSnapshotDiff(u64 address, u64 size, u64 first_offset, u64 diff_bytes, u32 snapshot_word, u32 guest_word, u32 who);

/// The constant ring and the snapshots: 0 a ring binding carries a snapshot, 1 refused (partly covered), 2 verified against guest memory at bind time, 3 differs from it.
void NoteSnapshotRing(u32 what);

// ---- the on-demand upload audit (TLG_UPLOAD_AUDIT=1, changes nothing) ----------------------------------------------------------------
// Which buffer requests at draw or dispatch time copy bytes from guest memory, how big the requests are, who asks, and whether a render pass was open (the upload then
// ended it).
bool OnDemandAudit();
/// Who is asking the buffer cache (set by the rasterizer around its buffer requests): 0 other, 1 buffer of a draw, 2 buffer of a dispatch, 3 vertex buffer, 4 index buffer.
void SetCaller(u32 caller);
u32 Caller();
/// What the buffer being requested is (set by the rasterizer's BindBuffers): bit 0 written by the shader, bit 1 formatted (texel buffer), bit 2 larger than the ring takes,
/// bit 3 a ring binding existed for it (the request should not have come), bit 4 the draw has a ring plan.
void SetDetail(u32 detail);
/// The audit of the uploads on GPU-written bytes: the announcer that last wrote the range (NativeUpdates::WhoWrote), set together with the detail.
void SetWriter(u32 who);
void NoteOnDemand(bool pass_open, u64 request_size, u64 uploaded_bytes, u64 address, bool immutable_whole);

// ---- byte precise GPU-written ranges for the constant ring (N5e, `TLG_RING_PRECISE=1`) ----------------------------------------------------------------------
bool RingPrecise();
/// `TLG_RING_FORMATTED=1` (N5f): formatted (texel) buffers go through the ring too. The shader decodes the format itself (the port binds them as storage buffers); the
/// buffer cache treats a formatted read as a possible read of an image's memory (a depth or colour surface, HTile, CMask), so the raster thread checks at bind time that no
/// cached image covers the range and that it is no metadata surface, else it ignores the ring copy.
bool RingFormatted();
/// `TLG_RING_GEOMETRY=1` (N5g): the vertex buffers and the index buffer of a draw go through the constant ring too (up to `RingGeometryCap()` bytes each, default 1 MiB,
/// `TLG_RING_GEOMETRY_CAP`). Same rules as the shader buffers: not GPU written by bytes, not immutable, no pending GPU write; the raster thread checks the bytes again at bind.
bool RingGeometry();
u64 RingGeometryCap();
/// 0 vertex range copied by the front, 1 index range copied, 2 vertex range bound from the ring, 3 index range bound, 4 ignored at bind (the GPU wrote the bytes),
/// 5 refused: over the cap, 6 refused: GPU written bytes, 7 refused: immutable or pending write or ring full, 8 verified against guest memory, 9 differs from it.
void NoteGeometry(u32 what);
/// 0 a range the page flag would have refused went through the ring (its bytes were never written by the GPU), 1 the GPU wrote bytes of a ring-bound range between the
/// plan and the bind (the ring served guest bytes the GPU had changed since), 2 refused because its bytes are GPU written.
void NoteRingPrecise(u32 what);

nlohmann::json Report();

} // namespace Core::NativeUploads
