// SPDX-License-Identifier: GPL-2.0-or-later
// N5a (documents/tlg/native-renderer-plan.md): the update audit. Today the backend learns that the game changed GPU-visible memory from page faults (the pages of
// guest memory the buffer cache watches are write protected). The native way is that the engine says so: every dynamic buffer is written between a
// JsDeviceContext::Map and the draws that use it, and Map is hooked. This audit checks that claim: it records every Map of a buffer (resource id, the pointer the engine
// hands out, the size, the thread) and classifies every CPU write fault by whether an announced Map window covers it. A fault with no window is a write the engine does
// not announce, and has to be found (by the faulting instruction) before page protection can be left out for those buffers. Off by default: `TLG_UPDATE_AUDIT=1` or the
// debug server endpoint `/updates?start=1`.

#pragma once

#include <array>
#include <cstdlib>
#include <utility>
#include <vector>
#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeUpdates {

extern std::atomic<bool> enabled;
/// Called on the render thread (the thread of RenderDev::BeginRecording): announcements are counted as made on that thread or elsewhere.
void NoteRenderThread();
/// The engine mapped a buffer: `address` / `size` are what Map returned (the buffer's own memory, or a window of the frame ring), `memory` the buffer's own memory.
/// A1.0: the next NoteMap of the calling thread is a texture's (its Unmap announces image content)
void MarkNextMapTexture();
void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size = 0, u32 context_frame = 0, u32 resource_stamp = 0, u32 bind_flags = 0);
/// JsDeviceContext::Unmap(context, resource, subresource) (an empty function of the engine, hooked): the writes into the memory the thread's Map of this resource handed
/// out are finished; that range (the requested size or the whole mapped size) is announced.
void NoteUnmap(u64 resource_id);
/// F2.2: a Map of memory overlapping [address, address + size) is open (mapped, not yet unmapped or flushed at the frame start): its writer may still be writing
bool OpenMapOverlaps(u64 address, u64 size);
/// F2.2: a writer that maps resources and never unmaps ends: `serial = OpenMapSerial()` before it runs, `FlushOpenMapsSince(serial)` after it returned announces the Maps this thread made in
/// between (their writes are finished) instead of at the next frame start.
u64 OpenMapSerial();
void FlushOpenMapsSince(u64 serial);
/// A hooked engine function announces the memory it is about to write (`writer` names the function: 1 skinning matrices, 2 light constants, 3 a memory copy of the command stream, 4 the BeginRecording stores, 5 buffer creation with initial data, 6 texture initialization).
/// `site` names the call site for the census `TLG_WRITER_SITES=1` (`/stats` writer_sites: count, bytes, size classes per writer and site); it must be a string literal.
/// N1.2: `object` is the id (+0x10) of the engine buffer the hook knows it writes into (0: not told). The apply then bumps that buffer by id instead of finding it by address (`TLG_BUFFER_ID`).
void NoteWriterRange(u32 writer, u64 address, u64 size, const char* site = nullptr, bool ring = false, u64 object = 0);
/// TLG_EARLY_ANNOUNCE=1: announcements of memory the engine writes fresh every frame (frame rings: light constants, skinning matrices, scene slots, Map into a ring window) are applied on the raster
/// thread as soon as it can, not behind the graphics work queued before them (documents/tlg/native plan/engine/compute-lane-design.md section 13).
[[nodiscard]] bool EarlyAnnounceOn() noexcept;
/// F1 1c real (`TLG_RING_LEAN_APPLY`, default on since 2026-11-10, =0 off): an in-ring constant copy or buffer Map window is applied lean on the raster thread (image index and GPU-written
/// supersede only; no buffer marks, memo erase or resident note), one in 64 in full with its effects counted; rows do not learn ranges inside the frame ring.
[[nodiscard]] bool RingLeanApplyOn() noexcept;
/// F1 class 1 finished (`TLG_RING_OUT`, default on since 2026-11-10, =0 off): an in-ring constant copy or buffer Map window does not enter the raster thread's batch at all (the hook ring holds its bytes, the
/// census of design review section 26 found every remaining step of its apply without effect); one in 64 still goes through the batch and is applied in full with its effects counted.
[[nodiscard]] bool RingOutOn() noexcept;
/// Counters of the lean apply: 0 applied lean, 1 applied in full as the sampled check, 2 sampled with a buffer, memo, recipe or resident effect (MUST_BE_0), 3 rows not learned in the ring.
void NoteRingLean(u32 what) noexcept;
/// N5h (`TLG_RING_SUPERSEDE=1`): a write that gives memory new contents supersedes what the GPU wrote there before. The announcer is told (`fn`) for: the creation of a
/// buffer, the initialization of a texture (any thread) and the unmap of a buffer on the render thread. Safe because anything the GPU writes into that memory afterwards is
/// issued after the announcement; marks that older GPU writes add later are only too many.
void SetSupersede(void (*fn)(u64 address, u64 size));
bool SupersedeEnabled();


/// B2 cutover (`TLG_B2_NATIVE`): 0 off (the write watch decides what the CPU changed), 1 the engine's announcements decide and the write watch only replays the pages
/// no announcement covered (counted as misses), 2 the announcements alone (misses are counted, not replayed: the fully native mode).
/// Announcements are queued and applied when the command processor reaches the game's next submission or passed memory wait (the point the write watch harvests
/// from), and once more one submission later, so an announcer that fires before its write is covered too.
[[nodiscard]] constexpr u32 B2NativeMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// The buffer cache's marking of a range the CPU changed (what a harvest's replay does); `newer` (first application): the CPU's bytes are newer than anything the
/// GPU wrote there before (the GPU written marks of the range are dropped); false for the second application one submission later
void SetApply(void (*fn)(void* context, u64 address, u64 size, bool newer, u64 object), void* context);
/// N1.2 (`TLG_BUFFER_ID`): 0 off (default), 1 an announcement that names its engine buffer is applied by id (the buffer's part, no address walk; no image index lookup when no clear waits), 2 shadow:
/// the address path runs as before and every difference to the id path is counted (`buffer_id` in /updates, `*_MUST_BE_0`).
[[nodiscard]] u32 BufferIdMode() noexcept;
/// `object` value of an announcement whose memory the hook knows to be engine buffers only (a block over several buffers): no single id, but no image lives there
constexpr u64 kBuffersOnly = ~u64{0};
enum BufferIdCount : u32 { IdToldWhole, IdToldPartial, IdUnknown, IdPartsMatch, IdPartsDiffer, IdImageSkipped, IdImageFoundOverTold, IdToldBlock, IdSyncHeldMatch, IdSyncHeldDiffer, IdSyncPartsMatch, IdSyncPartsDiffer, IdSyncToldUsed, BufferIdCounts };
/// N3 (`TLG_SYNC_BY_ID`): 0 off (default), 1 `SynchronizeMemory` of a placed engine buffer takes the buffer's memory and its id from the buffer itself (set when it was placed) instead of walking the
/// placed map and the engine's address table, 2 shadow: both run, differences are counted (`sync_*` in `buffer_id`, `*_MUST_BE_0`).
[[nodiscard]] u32 SyncByIdMode() noexcept;
/// L1 (`TLG_APPLY_COALESCE`): 0 off (default), 1 the first-time announcements of a batch are applied as merged ranges (same engine buffer, touching or overlapping bytes: the byte set is the same, the
/// apply passes run once per merged range instead of once per piece), 2 counts what merging would do and applies as before (`coalesce` in `buffer_id`).
[[nodiscard]] u32 ApplyCoalesceMode() noexcept;
/// F1 class 2, marks by buffer id (`TLG_ID_MARKS`, design review sections 48 to 51): 2 real (default since 2026-11-10, owner's play), 1 shadow, 0 off. Concerns the batch applies of the light constants (tag 2) and the
/// scene slots (tag 12: writer 4 without a site name; with the switch on the slot announcer names each slot's engine buffer instead of announcing a block). The batch order stays.
/// 1: the full apply decides; beside it, what the lean apply would do is compared (parts, effects, resident buffers it would not bump). 2: a range that its named buffer holds is applied lean:
/// the bind memo erase, the image index (kept: a buffer's memory can still hold an image or a waiting clear), the buffer's dirty part and row version, its resident generation; no GPU-written
/// supersede, no address walk, no page-wide resident walk, no batch hashes,
/// no second application (census `ab0`: those steps did nothing for these writers in 2.37 M applies, the second application found 0 changed in 3.1 M). One in 64 runs in full and is compared.
[[nodiscard]] u32 IdMarksMode() noexcept;
[[nodiscard]] inline bool IdMarksTag(u32 tag) noexcept {
    return tag == 2 || tag == 12;
}
enum IdMarksCount : u32 {
    ImApplies,               ///< applies of tags 2 and 12 with the switch on
    ImLean,                  ///< applied lean (mode 2)
    ImSampledFull,           ///< mode 2: the one in 64 applied in full and compared
    ImNotNamed,              ///< the announcer named no buffer that holds the range (full path)
    ImBlock,                 ///< a block over several buffers (kBuffersOnly; full path)
    ImChecked,               ///< full applies compared (shadow, or the mode 2 sample)
    ImUnexpectedEffect,      ///< the full apply did something the lean one does not do: GPU-written supersede or the image index (MUST_BE_0)
    ImUnexpectedGpu,         ///< ... the GPU-written supersede
    ImUnexpectedImage,       ///< ... the image index
    ImSampledUnexpected,     ///< the same in a mode 2 sample (MUST_BE_0)
    ImUnattributed,          ///< the full apply marked bytes no buffer holds (gaps of a block)
    ImPartsMatch,            ///< named: the address walk found exactly the named part
    ImPartsDiffer,           ///< named: it found something else (MUST_BE_0)
    ImPartWhole,             ///< a part covers its whole buffer
    ImPartNotWhole,          ///< a part covers only some of its buffer
    ImResidentTold,          ///< the full apply bumped the resident generation of a buffer the range lies in
    ImResidentNeighbor,      ///< ... of another buffer on the same page (the lean apply does not)
    ImResidentNeighborSynced,///< ... that was synchronized at that moment
    ImResidentToldNone,      ///< named, and the full apply bumped no resident generation of the named buffer
    ImLeanResident,          ///< lean: the named buffer's resident generation bumped
    ImLeanNoResident,        ///< lean: the named buffer is not a resident buffer
    ImAgainSkipped,          ///< mode 2: second applications skipped
    ImHashSkipped,           ///< mode 2: batch hashes skipped
    ImLeanImageTouched,      ///< lean: the image index found an image or a waiting clear over the range (the step is kept: idmdef found 18 in 51 k samples on a later save)
    ImImgIdNone,             ///< TLG_IMAGE_BY_ID: the id says no image
    ImImgIdNoneAddrNone,     ///< ... and the address lookup agrees
    ImImgIdMissed,           ///< ... but the address lookup finds an image or a waiting clear (MUST_BE_0)
    ImImgIdMaybe,            ///< the id cannot say (clears wait, creation not handled, a live image listed)
    ImImgIdMaybeAddrNone,    ///< ... and the address lookup finds none
    ImImgIdClearsSum,        ///< the number of waiting clears, summed over the shadow's checks
    IdMarksCounts
};
void NoteIdMarks(IdMarksCount what, u64 n = 1) noexcept;
/// Design review 63 (`TLG_IMAGE_BY_ID`): 0 off (default), 1 shadow (the image step of a lean mark asks the id and the address and compares; the address decides).
[[nodiscard]] u32 ImageByIdMode() noexcept;
/// Batch census (`TLG_BATCH_CENSUS=1`, design review 62, counters only): the time of each apply of the batch by writer tag, and of the lean mark's steps (one mark in 16 timed:
/// 0 the named-buffer check, 1 the bind memo erase, 2 the image step, 3 the dirty sets under their lock, 4 the version and epochs, 5 the resident lookup and bump).
[[nodiscard]] bool BatchCensusOn() noexcept;
void NoteLeanStep(u32 step, u64 ns) noexcept;
/// L1: pieces of one engine buffer closer than this many bytes (`TLG_APPLY_COALESCE_GAP`, default 4096) are candidates for one applied range.
[[nodiscard]] u64 ApplyCoalesceGap() noexcept;
/// L1: why a group was refused (0 tracker/image mode, 1 named buffer does not hold the range, 2 GPU written, 3 clear waits or an image touches no piece), counted in groups and pieces.
void NoteMergeRefused(u32 why, u64 pieces) noexcept;
/// One announced piece of a batch.
struct AnnouncedPiece {
    u64 object, address, size;
    u32 writer{16}; ///< (announce batch census: the writer tag, 16 unknown)
};
/// L1: the buffer cache says whether the pieces (sorted, one engine buffer or none named) may be applied as the one range [begin, begin + size): true when the bytes between the pieces are
/// not GPU written, no clear waits for an image, no image is in the range that no piece touches, and a named buffer holds the whole range.
void SetMergeCheck(bool (*fn)(void*, u64 object, u64 begin, u64 size, const AnnouncedPiece* pieces, size_t count), void* context);
void NoteBufferId(BufferIdCount what);
/// N1: who announced a range: `writer` (the writer id, plus 8 when the caller gave no site name), `object` the engine buffer id N1.2 (0: not told)
struct Tag {
    u8 writer{};
    u64 object{};
    u8 ring_candidate{}; ///< F1 1c: inside the frame ring and a constant buffer copy (writer 3) or a buffer's Map window (writer 0, not a texture): a range that could leave the batch
};
/// The announcements of one submission: applied once (`first`) and the previous submission's again (`again`)
struct Batch {
    std::vector<std::pair<u64, u64>> first, again;
    std::vector<Tag> first_writer, again_writer, ring_first_writer, ring_again_writer; ///< N1: the writer (and N1.2 the buffer id) of every range of the four lists (same order)
    std::vector<std::pair<u64, u64>> ring_first, ring_again; ///< the part of first / again in frame ring memory (empty unless EarlyAnnounceOn)
    std::vector<std::array<u64, 4>> deleted; ///< B8.2: engine objects deleted or created before the submission (id, memory, bytes, what: 0 view, 1 texture, 2 created texture, 3 texture with an alias)
    [[nodiscard]] bool Empty() const noexcept {
        return first.empty() && again.empty() && ring_first.empty() && ring_again.empty() && deleted.empty();
    }
};
/// Takes the announcements made since the last batch (B2 cutover; empty when off)
[[nodiscard]] Batch TakeBatch();
/// Applies a batch (raster thread: in order with the GPU work)
void ApplyBatch(const Batch& batch);
/// TLG_ANNOUNCE_SKIP: the buffer cache says whether an announced range with the same bytes as last time may be left unapplied: 0 yes, 1 the GPU wrote it, 2 an image covers it, 3 not in tracker mode.
void SetSkipCheck(u32 (*fn)(void*, u64, u64), void* context);
[[nodiscard]] nlohmann::json B2Report();

/// B8.1 (`TLG_B8_IMAGES`): 0 off, 1 audit (texture memory stays write protected; a fault that dirtied an image no announcement explains is counted), 2 native (the
/// announcements alone refresh images, texture memory is not write protected)
[[nodiscard]] constexpr u32 B8ImageMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B8.2 (`TLG_B8_LIFETIME`): 0 off, 1 audit (images the engine's deleted textures leave behind are marked, a later use is counted), 2 native (an engine texture's
/// images are freed when the engine deletes it, in order with the GPU work; the B3 table drops the ids of deleted views in both modes)
[[nodiscard]] constexpr u32 B8LifetimeMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B8.3 (`TLG_B8_FIND`): 0 off, 1 audit (the image found by the engine texture's id is compared with the lookup by address), 2 native (an engine texture's
/// image is found by its id; the page table lookup only runs when the texture has no matching image yet or no engine texture owns the memory)
[[nodiscard]] constexpr u32 B8FindMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// E3.2: a named image description finds its engine texture by the engine's id (default on; `TLG_E3_ID=0` asks the address as before)
[[nodiscard]] inline bool E3IdDecides() noexcept {
    static const bool on = [] {
        const char* env = std::getenv("TLG_E3_ID");
        return env == nullptr || env[0] != '0';
    }();
    return on;
}
/// B8.5 (`TLG_B8_BUFFER_IMAGES`): images in engine buffer memory (formatted reads, render targets in a buffer's memory) are listed by the buffer; announcements
/// and copies over buffer memory find them there instead of in the page table. 0 off, 1 audit (page table kept, lists compared), 2 native (default)
[[nodiscard]] constexpr u32 B8BufferImagesMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B8.2: the engine deleted a view (`what` 0), a texture (1), a texture another live texture shares the memory with (3) or a buffer (4, B9.2); queued with the next
/// submission's batch
void NoteEngineDelete(u64 id, u64 address, u64 bytes, u32 what);
/// Where a deletion is carried out (raster thread, in order with the GPU work); set by the rasterizer
void SetDelete(void (*fn)(void* context, u64 id, u64 address, u64 bytes, u32 what), void* context);
/// B8.4: the engine created a texture (its memory gets a new owner); queued with the next submission's batch like a deletion
void NoteEngineCreate(u64 id, u64 address, u64 bytes);
void SetCreate(void (*fn)(void* context, u64 id, u64 address, u64 bytes, bool texture, u32 usage, u32 cpu_access), void* context);
/// B9.4: the engine created a buffer (a placed buffer under its memory moves back into the arena)
void NoteEngineBufferCreate(u64 id, u64 address, u64 bytes, u32 usage, u32 cpu_access);
enum B8Count : u32 {
    B8AnnouncedOnImage,         ///< announcements that touched an image
    B8KeptForGpu,               ///< second applications that left an image the GPU wrote since alone
    B8TexturesDeleted,          ///< B8.2: engine texture deletions carried out
    B8TextureAliasKept,         ///< ... skipped: another live engine texture has the same memory
    B8DeletedImages,            ///< images of a deleted texture (same memory, not bigger than the object)
    B8DeletedImagesBigger,      ///< images at a deleted texture's memory but bigger than the object (left alone)
    B8DeletedImagesFreed,       ///< native mode: images freed at the engine's deletion
    B8UsedAfterDeleteReused,    ///< audit: an image of a deleted texture used again while a new engine texture owns its memory (native mode makes a new image)
    B8UsedAfterDeleteNoOwner,   ///< audit: ... while no engine texture owns its memory (native mode would lose it: must stay 0)
    B8ViewIdsDropped,           ///< B3 entries of deleted views dropped
    B8FindById,                 ///< B8.3: image lookups answered by the engine texture's id
    B8FindByIdOther,            ///< ... audit: the lookup by address found another image
    B8FindNoImageYet,           ///< ... the texture has no matching image yet (page table lookup, then owned)
    B8FindNoOwner,              ///< ... no engine texture starts at the memory (page table lookup)
    B8FindOwned,                ///< images given to an engine texture
    B8FindByAnswer,             ///< B8.4: lookups answered by the engine texture's remembered answer (a view of another format or size, a mip or layer of a bigger image)
    B8FindAnswerOther,          ///< ... audit: the lookup by address gave another image, mip or layer
    B8FindAnswerStale,          ///< ... an image was registered or removed in the range since: the page table lookup runs again
    B8FirstNoOverlap,           ///< B8.4 census: a page table lookup for an engine texture that met no image in its memory
    B8FirstOverlap,             ///< ... that met images (overlap resolution ran)
    B8RouteAnnounce,            ///< B8.4: announcements whose images were found through the engine textures of the range
    B8RouteAnnounceFallback,    ///< ... the range is not covered by engine textures (page table)
    B8RouteMissed,              ///< ... audit: an image the page table finds and the engine textures' lists do not (not looked up since: refreshed when adopted)
    B8RouteMissedListed,        ///< ... of those, images a lookup already gave out (a gap in the lists: must stay 0)
    B8Adopted,                  ///< B8.4: an older image a page table lookup gave out for the first time (native mode: refreshed from memory, its announcements may have been missed)
    B8AdoptedGpu,               ///< ... the GPU wrote it: kept
    B8CreatedTextures,          ///< B8.4: engine texture creations carried out
    B8FirstByEngine,            ///< B8.4 step 3: a first lookup whose candidate images came from the engine textures covering the memory
    B8FirstFallback,            ///< ... the memory is not covered by engine textures (page table)
    B8FirstMissedListed,        ///< ... audit: a candidate the page table finds, a lookup gave out, and the engine textures' lists lack (must stay 0)
    B8FirstStaleLeft,           ///< ... audit: an older image no lookup gave out since (native mode leaves it out: the new image does not reuse a stale one)
    B8CreatedUnlisted,          ///< ... older images in the new texture's memory, unlisted (refreshed when a lookup gives them out again)
    B8RouteCopy,                ///< B8.4: copies whose image was found through the engine texture at the address
    B8RouteCopyFallback,        ///< ... no engine texture starts at the address (page table)
    B8RouteCopyOther,           ///< ... audit: the page table gives another answer
    B8BufAnnounce,              ///< B8.5: announcements over engine buffer memory whose images came from the buffers' lists
    B8BufCopy,                  ///< ... copies at engine buffer memory, same
    B8BufListed,                ///< ... images listed by an engine buffer (at registration, or at the buffer's creation)
    B8BufMissed,                ///< ... audit: an image the page table finds that the buffers' lists lack (must stay 0)
    B8BufNotCovered,            ///< ... ranges neither textures nor buffers cover whole (page table)
    E3FindUnnamed,              ///< E3 census: FindImage calls whose description names no engine object (targets, video out, copies, internal)
    E3FindNamed,                ///< ... that name an engine resource / view (a sampled or stored texture of a record binding)
    E3NamedSameOwner,           ///< ... named: the texture starting at the memory is the named resource (address and id agree)
    E3NamedOtherOwner,          ///< ... named: another texture starts at the memory (the id and the address disagree)
    E3NamedInterior,            ///< ... named: no texture starts at the memory but the named resource contains it (a view of a part: mip / slice offset)
    E3NamedNoOwner,             ///< ... named: no engine texture holds the memory at all
    E3NamedUnknownObject,       ///< ... named: the id is not in the engine table any more
    E3OwnerById,                ///< E3.2: the engine texture of a lookup came from the named id (not the address)
    E3OwnerByIdMissed,          ///< ... the id could not be resolved (the address asked instead)
    B8Counts,
};
void NoteB8(B8Count what) noexcept;
/// E3 census: an overlap resolution (`TextureCache::ResolveOverlap`) ended with `outcome` (0 pool free, 1 depth, 2 expand block, 3 expand 3D, 4 view reuse, 5 view incompatible,
/// 6 expand grow, 7 tile free, 8 assert free, 9 right subresource, 10 right chance freed, 11 right chance kept, 12 left rebind, 13 left copy mip, 14 left nothing, 15 left none)
/// for a candidate of class `cls` (0 the requested texture's own image, 1 another engine texture's, 2 unlisted, 3 the request names no engine texture)
void NoteE3Overlap(u32 cls, u32 outcome) noexcept;
/// F2.3 census: a `FindImage` call that names no engine object. `type` is the BindingType (0 texture, 1 storage, 2 render target, 3 depth target, 4 video out), `site` the engine site
/// (0 none), `owner` what the address lookup finds (0 no engine resource, 1 a texture starts at the address, 2 a texture contains it, 3 a buffer's memory).
void NoteE3Unnamed(u32 type, u32 site, u32 owner) noexcept;
/// F2.3 census: why a sampled or stored image lookup carries no engine id (1 the record names a binding with id 0, 2 not from the record and no plan, 3 not from the record with a plan).
void NoteE3UnnamedReason(u32 reason) noexcept;
[[nodiscard]] nlohmann::json B8Report();

/// B9.1 (`TLG_B9_BUFFERS`): 0 off, 1 audit (each engine buffer keeps the bytes the engine announced since their last upload; an upload compares them with the page
/// tracker's ranges), 2 native (an engine buffer's upload copies its announced bytes; the page tracker only keeps its marks for the other paths)
[[nodiscard]] constexpr u32 B9BufferMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B9.2 (`TLG_B9_LIFETIME`): 0 off, 1 audit (an engine buffer's deletion is carried out in order and counted: its announced bytes go, what it leaves behind is
/// counted), 2 native (also the GPU written bytes and stream copies of its memory go, unless another live buffer shares the memory)
[[nodiscard]] constexpr u32 B9LifetimeMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B9.3 (`TLG_B9_QUERIES`): 0 off, 1 audit (for a range inside one engine buffer, "did the CPU change it" and "did the GPU write it" are answered by the buffer's state
/// and compared with the page tracker), 2 native (the buffer's state answers: its announced bytes, the GPU written bytes)
[[nodiscard]] constexpr u32 B9QueryMode() noexcept {
    return 2u; // native, no switch since the cleanup of 2026-10-30
}
/// B9.3 step 2 (`TLG_B9_TRACKER=0`, with B9.1 and B9.3 native): the page tracker is not kept for engine buffers any more (announced bytes and CPU changes found
/// otherwise mark the buffers; uploads inside one engine buffer neither ask nor clear the tracker); other memory keeps it
[[nodiscard]] constexpr bool B9TrackerOff() noexcept {
    return true; // native, no switch since the cleanup of 2026-10-30
}
/// B9.4 (`TLG_B9_PLACE=1`, with B9.1 to B9.3 native): an engine buffer that shares its memory with no other live resource gets a Vulkan buffer of its own at its first
/// use; every request inside it is served from that buffer instead of the arena. A new resource over its memory, or a request that runs over its edge, moves it back
/// into the arena; the engine's deletion frees it.
[[nodiscard]] constexpr bool B9PlaceOn() noexcept {
    return true; // native, no switch since the cleanup of 2026-10-30
}
/// B9.4 step 4 (`TLG_B9_ALIASES`, default on, =0 off): a block may hold engine buffers that overlap each other and textures (one Vulkan buffer is one memory range, as the arena
/// was); a buffer created over a block's members joins it, a texture created inside a block leaves it in place
[[nodiscard]] constexpr bool B9AliasesOn() noexcept {
    return true; // native, no switch since the cleanup of 2026-10-30
}
enum B9Count : u32 {
    B9AnnouncedOnBuffer,   ///< announcements inside one engine buffer
    B9AnnouncedClipped,    ///< announcements that meet several buffers or reach past one (each buffer takes its part)
    B9AnnouncedNoBuffer,   ///< announcements not starting in an engine buffer (textures, unowned memory)
    B9SyncByObject,        ///< uploads of a range inside one engine buffer
    B9SyncNoObject,        ///< uploads of other ranges (page tracker)
    B9TrackerOnlyRanges,   ///< a page tracker range with no announced byte anywhere in it (native mode would not upload it: must stay 0)
    B9TrackerOnlyBytes,
    B9TrackerElsewhere,    ///< a page tracker range whose announced bytes lie outside the request (another buffer on the page, or later in this one: uploaded with that request)
    B9ObjectOnlyRanges,    ///< announced bytes no page tracker range covered
    B9BytesByTracker,      ///< bytes the page tracker would upload
    B9BytesByObject,       ///< bytes the announcements upload
    B9CpuQueries,          ///< B9.3: CPU changed? for a range inside one engine buffer
    B9CpuAgree,
    B9CpuTrackerNeighbor,  ///< ... tracker yes, buffer no: the page's announced bytes are another buffer's (page granularity)
    B9CpuTrackerOnly,      ///< ... tracker yes, buffer no, no announced byte on the pages (must stay 0)
    B9CpuObjectOnly,       ///< ... buffer yes, tracker no (the page was uploaded whole for a neighbor; native uploads again: safe)
    B9GpuQueries,          ///< B9.3: GPU written? for a range inside one engine buffer
    B9GpuAgree,
    B9GpuTrackerPage,      ///< ... tracker yes, bytes no: GPU written bytes elsewhere on the page (page granularity)
    B9GpuTrackerOnly,      ///< ... tracker yes, no GPU written byte on the pages (the tracker keeps a page's mark after the bytes were superseded or downloaded)
    B9GpuBytesOnly,        ///< ... bytes yes, tracker no (native answers yes: the arena's bytes are kept, the safe way round)
    B9TrackerMarksSkipped, ///< B9.3 step 2: CPU changes inside engine buffers not marked in the page tracker
    B9TrackerUploadsSkipped, ///< ... uploads inside one engine buffer that did not ask the tracker
    B9OtherChanges,        ///< ... CPU changes found otherwise (faults, decoders) given to the buffers
    B9Placed,              ///< B9.4: engine buffers given a Vulkan buffer of their own
    B9PlacedBytes,
    B9PlaceRefusedAlias,   ///< ... refused: another live buffer or texture shares the memory
    B9PlaceRefusedSize,    ///< ... refused: size unknown (0) or above the limit
    B9PlacedServed,        ///< requests served from an engine buffer's own Vulkan buffer
    B9PlacedGpuCopied,     ///< placements that copied GPU written bytes over from the arena
    B9Demoted,             ///< buffers moved back into the arena (a new resource over the memory, a request over the edge)
    B9DemotedByRequest,    ///< ... of those, by a request over the edge
    B9PlacedFreed,         ///< buffers freed at the engine's deletion
    B9ReadbackOverPlaced,  ///< a fault readback over a placed buffer's memory (it reads the arena: must stay 0)
    B9Groups,              ///< B9.4 step 2: placements of a block of several engine buffers (a request over several, or over a placed buffer's edge)
    B9GroupMembers,        ///< ... engine buffers in those blocks
    B9GroupReused,         ///< ... placed buffers whose contents a block took over
    B9Adopted,             ///< a new engine buffer inside a placed block that overlaps no live member: it joins the block
    B9MemberLeft,          ///< an engine buffer deleted out of a block that still has members
    B9ArenaOverBuffers,    ///< requests served by the arena although engine buffers meet them (aliased memory, textures in the block, too big)
    B9BlockAliased,        ///< B9.4 step 4: blocks placed with members that overlap each other
    B9BlockWithTextures,   ///< ... with textures inside
    B9TextureInBlockKept,  ///< a texture created inside a block (the block stays)
    B9AdoptedOverMember,   ///< a buffer created over a block's live member joins the block
    B9BuffersDeleted,      ///< B9.2: engine buffer deletions carried out
    B9DeletedPending,      ///< ... with announced bytes never uploaded (dropped)
    B9DeletedGpuWritten,   ///< ... whose memory held GPU written bytes
    B9DeletedAlias,        ///< ... whose memory another live buffer shares (its GPU written bytes stay)
    B9DeletedGpuDropped,   ///< native: GPU written bytes dropped with the buffer
    E4PseudoPlaced,        ///< E4: placements of a pseudo resource (memory no engine buffer owns)
    E4PseudoRing,          ///< ... of those, the frame allocator's ring
    E4DemoteKept,          ///< E4: a move back to the arena that was not made (the block stays)
    E4FreedWithForeignGpu, ///< E4: a block's last engine buffer was deleted while the GPU had written bytes of the block outside that buffer
    E4FreedForeignGpuBytes, ///< E4: those bytes
    E4GpuBytesLost,        ///< E4: GPU written bytes a new placement found in no placed buffer
    E4AnnouncedUploads, ///< E4: bytes of unowned memory uploaded from the announced set
    LeanChecked,        ///< D2 (`TLG_LEAN_DIRTY=1`): answers compared between the by-address audit set and the unattributed set alone
    LeanMissing,        ///< ... the audit set had announced bytes the lean set did not (a stale answer if the audit set were dropped: must stay 0)
    LeanExtra,          ///< ... the lean set had bytes the audit set did not (an over-upload, safe)
    LeanMoved,          ///< bytes a deleted buffer left behind, moved to the unattributed set
    B9Counts,
};
void NoteB9(B9Count what, u64 n = 1) noexcept;
[[nodiscard]] constexpr bool E4PseudoOn() noexcept {
    return true; // E4: memory no engine buffer owns is a pseudo resource, there is no arena to fall back to
}
[[nodiscard]] nlohmann::json B9Report();

/// D2 (`TLG_LEAN_DIRTY`): 0 keeps `all_object_dirty`, the by-address copy of every announced byte, as the answer for memory no engine buffer holds (today's behaviour);
/// 1 keeps it and compares each of its answers with the one the unattributed set alone gives (counters `Lean*`); 2 does not keep it (a deleted buffer's pending bytes move to
/// the unattributed set instead).
[[nodiscard]] u32 LeanDirtyMode() noexcept;

/// Load address of the executable (for turning code addresses in diagnostics into offsets).
[[nodiscard]] u64 ImageBase() noexcept;

nlohmann::json Report();

} // namespace Core::NativeUpdates
