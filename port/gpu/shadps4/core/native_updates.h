// SPDX-License-Identifier: GPL-2.0-or-later
// N5a (documents/tlg/native-renderer-plan.md): the update audit. Today the backend learns that the game changed GPU-visible memory from page faults (the pages of
// guest memory the buffer cache watches are write protected). The native way is that the engine says so: every dynamic buffer is written between a
// JsDeviceContext::Map and the draws that use it, and Map is hooked. This audit checks that claim: it records every Map of a buffer (resource id, the pointer the engine
// hands out, the size, the thread) and classifies every CPU write fault by whether an announced Map window covers it. A fault with no window is a write the engine does
// not announce, and has to be found (by the faulting instruction) before page protection can be left out for those buffers. Off by default: `TLG_UPDATE_AUDIT=1` or the
// debug server endpoint `/updates?start=1`.

#pragma once

#include <array>
#include <utility>
#include <vector>
#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeUpdates {

extern std::atomic<bool> enabled;
/// True only for the audit (`TLG_UPDATE_AUDIT=1`, `/updates?start=1`): the per fault classification, the announcement windows and the thread census. The announcements
/// themselves (uploads, supersede) run with `enabled` alone and cost much less.
[[nodiscard]] bool AuditFaults();

void Start();
/// Called on the render thread (the thread of RenderDev::BeginRecording): announcements are counted as made on that thread or elsewhere.
void NoteRenderThread();
/// The engine mapped a buffer: `address` / `size` are what Map returned (the buffer's own memory, or a window of the frame ring), `memory` the buffer's own memory.
void NoteMap(u64 resource_id, u32 usage, u32 map_type, u64 address, u32 size, u64 memory, u64 caller, u64 requested_size = 0);
/// JsDeviceContext::Unmap(context, resource, subresource) (an empty function of the engine, hooked): the writes into the memory the thread's Map of this resource handed
/// out are finished; that range (the requested size or the whole mapped size) is announced.
void NoteUnmap(u64 resource_id);
/// A hooked engine function announces the memory it is about to write (`writer` names the function: 1 skinning matrices, 2 light constants, 3 a memory copy of the command stream, 4 the BeginRecording stores, 5 buffer creation with initial data, 6 texture initialization).
void NoteWriterRange(u32 writer, u64 address, u64 size);
/// The game wrote tracked memory (called from the fault handler, on the writing thread).
void NoteFault(u64 address, u64 size);

/// The CPU marked [address, address + size) as changed (a fault or an explicit invalidation) while the byte set says the GPU wrote bytes of it: who did it (the faulting
/// instruction, or the host call stack for an explicit invalidation). Kept by site, reported by /updates.
void NoteGpuWrittenDirtied(u64 address, u64 size);
/// N5h (`TLG_RING_SUPERSEDE=1`): a write that gives memory new contents supersedes what the GPU wrote there before. The announcer is told (`fn`) for: the creation of a
/// buffer, the initialization of a texture (any thread) and the unmap of a buffer on the render thread. Safe because anything the GPU writes into that memory afterwards is
/// issued after the announcement; marks that older GPU writes add later are only too many.
void SetSupersede(void (*fn)(u64 address, u64 size));
bool SupersedeEnabled();
/// Who announced a write that covers `address` most recently: the announcer (0 Map / Unmap ... 6, see NoteWriterRange), whether on the render thread (bit 8), and how long ago
/// in milliseconds (bits 16 and up); ~0 when no announcement covers it (the window ring keeps the last 4096).
u32 WhoWrote(u64 address, u64 size);

/// B2 gate (`TLG_B2_AUDIT=1`): every page the write watch finds written is checked against the pages an announcement covered since the page's last harvest.
/// The pages of an announcement are marked (a direct mapped table of 1 M pages); a harvested page with no mark is a CPU write no engine update explains.
[[nodiscard]] bool HarvestAuditOn() noexcept;

/// B2 cutover (`TLG_B2_NATIVE`): 0 off (the write watch decides what the CPU changed), 1 the engine's announcements decide and the write watch only replays the pages
/// no announcement covered (counted as misses), 2 the announcements alone (misses are counted, not replayed: the fully native mode).
/// Announcements are queued and applied when the command processor reaches the game's next submission or passed memory wait (the point the write watch harvests
/// from), and once more one submission later, so an announcer that fires before its write is covered too.
[[nodiscard]] u32 B2NativeMode() noexcept;
/// The buffer cache's marking of a range the CPU changed (what a harvest's replay does); `newer` (first application): the CPU's bytes are newer than anything the
/// GPU wrote there before (the GPU written marks of the range are dropped); false for the second application one submission later
void SetApply(void (*fn)(void* context, u64 address, u64 size, bool newer), void* context);
/// The announcements of one submission: applied once (`first`) and the previous submission's again (`again`)
struct Batch {
    std::vector<std::pair<u64, u64>> first, again;
    std::vector<std::array<u64, 4>> deleted; ///< B8.2: engine objects deleted or created before the submission (id, memory, bytes, what: 0 view, 1 texture, 2 created texture, 3 texture with an alias)
    [[nodiscard]] bool Empty() const noexcept {
        return first.empty() && again.empty() && deleted.empty();
    }
};
/// Takes the announcements made since the last batch (B2 cutover; empty when off)
[[nodiscard]] Batch TakeBatch();
/// Applies a batch (raster thread: in order with the GPU work)
void ApplyBatch(const Batch& batch);
/// Where the command processor queues a batch in order (the raster thread); set by the command processor
void SetQueue(void (*fn)(void* context, Batch&& batch), void* context);
/// A passed memory wait (command processor): the announcements made so far are queued to the raster thread
void ApplyPending();
/// A harvested page: true when an announcement covered it since its last harvest (the mark is consumed)
[[nodiscard]] bool ConsumeMark(u64 page);
/// B2 cutover: a harvested page no announcement covered
void NoteMiss(u64 page, u64 bytes);
void NoteHarvested(u64 address, u64 bytes, bool in_upload_ring);
[[nodiscard]] nlohmann::json HarvestAuditReport();

/// B8.1 (`TLG_B8_IMAGES`): 0 off, 1 audit (texture memory stays write protected; a fault that dirtied an image no announcement explains is counted), 2 native (the
/// announcements alone refresh images, texture memory is not write protected; `TLG_B8_VERIFY=N` hashes every Nth bound image that is not dirty against its last refresh)
[[nodiscard]] u32 B8ImageMode() noexcept;
/// B8.2 (`TLG_B8_LIFETIME`): 0 off, 1 audit (images the engine's deleted textures leave behind are marked, a later use is counted), 2 native (an engine texture's
/// images are freed when the engine deletes it, in order with the GPU work; the B3 table drops the ids of deleted views in both modes)
[[nodiscard]] u32 B8LifetimeMode() noexcept;
/// B8.3 (`TLG_B8_FIND`): 0 off, 1 audit (the image found by the engine texture's id is compared with the lookup by address), 2 native (an engine texture's
/// image is found by its id; the page table lookup only runs when the texture has no matching image yet or no engine texture owns the memory)
[[nodiscard]] u32 B8FindMode() noexcept;
/// B8.2: the engine deleted a view (`what` 0), a texture (1), a texture another live texture shares the memory with (3) or a buffer (4, B9.2); queued with the next
/// submission's batch
void NoteEngineDelete(u64 id, u64 address, u64 bytes, u32 what);
/// Where a deletion is carried out (raster thread, in order with the GPU work); set by the rasterizer
void SetDelete(void (*fn)(void* context, u64 id, u64 address, u64 bytes, u32 what), void* context);
/// B8.4: the engine created a texture (its memory gets a new owner); queued with the next submission's batch like a deletion
void NoteEngineCreate(u64 id, u64 address, u64 bytes);
void SetCreate(void (*fn)(void* context, u64 id, u64 address, u64 bytes, bool texture), void* context);
/// B9.4: the engine created a buffer (a placed buffer under its memory moves back into the arena)
void NoteEngineBufferCreate(u64 id, u64 address, u64 bytes);
/// B8.1 verify: true for every Nth check (0: verify off)
[[nodiscard]] bool B8VerifySample() noexcept;
[[nodiscard]] bool B8VerifyOn() noexcept;
enum B8Count : u32 {
    B8RefreshAnnounced,         ///< a CPU refresh of an image an announcement dirtied (no fault)
    B8RefreshFaultAndAnnounced, ///< ... dirtied by a fault and an announcement
    B8FoundWithout,             ///< a CPU change of an image no announcement explained when it was refreshed (a fault, or the verify's hash)
    B8AnnouncedLater,           ///< ... of those, announced afterwards (the game wrote ahead of the raster thread: a timing case, not a miss)
    B8VerifyChecked,            ///< images the verify hashed
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
    B8Counts,
};
void NoteB8(B8Count what) noexcept;
/// An image a CPU change reached without an announcement (first few logged with the owner)
void NoteB8Unannounced(u64 address, u64 size, bool by_fault) noexcept;
[[nodiscard]] nlohmann::json B8Report();

/// B9.1 (`TLG_B9_BUFFERS`): 0 off, 1 audit (each engine buffer keeps the bytes the engine announced since their last upload; an upload compares them with the page
/// tracker's ranges), 2 native (an engine buffer's upload copies its announced bytes; the page tracker only keeps its marks for the other paths)
[[nodiscard]] u32 B9BufferMode() noexcept;
/// B9.1 verify (`TLG_B9_VERIFY=N`, native mode): every Nth page keeps a CPU copy of what the GPU copy holds; after a native upload the request's bytes on the page must
/// equal memory (a byte the engine wrote and did not announce shows up here)
[[nodiscard]] u32 B9VerifyEvery() noexcept;
/// B9.2 (`TLG_B9_LIFETIME`): 0 off, 1 audit (an engine buffer's deletion is carried out in order and counted: its announced bytes go, what it leaves behind is
/// counted), 2 native (also the GPU written bytes and stream copies of its memory go, unless another live buffer shares the memory)
[[nodiscard]] u32 B9LifetimeMode() noexcept;
/// B9.3 (`TLG_B9_QUERIES`): 0 off, 1 audit (for a range inside one engine buffer, "did the CPU change it" and "did the GPU write it" are answered by the buffer's state
/// and compared with the page tracker), 2 native (the buffer's state answers: its announced bytes, the GPU written bytes)
[[nodiscard]] u32 B9QueryMode() noexcept;
/// B9.3 step 2 (`TLG_B9_TRACKER=0`, with B9.1 and B9.3 native): the page tracker is not kept for engine buffers any more (announced bytes and CPU changes found
/// otherwise mark the buffers; uploads inside one engine buffer neither ask nor clear the tracker); other memory keeps it
[[nodiscard]] bool B9TrackerOff() noexcept;
/// B9.4 (`TLG_B9_PLACE=1`, with B9.1 to B9.3 native): an engine buffer that shares its memory with no other live resource gets a Vulkan buffer of its own at its first
/// use; every request inside it is served from that buffer instead of the arena. A new resource over its memory, or a request that runs over its edge, moves it back
/// into the arena; the engine's deletion frees it.
[[nodiscard]] bool B9PlaceOn() noexcept;
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
    B9VerifyChecked,       ///< verify: request pages compared
    B9VerifyMiss,          ///< ... differing from memory after the upload (fixed by a page upload)
    B9VerifyLate,          ///< ... of those, announced afterwards (the game wrote ahead of the raster thread: the page tracker had the same timing)
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
    B9BuffersDeleted,      ///< B9.2: engine buffer deletions carried out
    B9DeletedPending,      ///< ... with announced bytes never uploaded (dropped)
    B9DeletedGpuWritten,   ///< ... whose memory held GPU written bytes
    B9DeletedAlias,        ///< ... whose memory another live buffer shares (its GPU written bytes stay)
    B9DeletedGpuDropped,   ///< native: GPU written bytes dropped with the buffer
    B9Counts,
};
void NoteB9(B9Count what, u64 n = 1) noexcept;
[[nodiscard]] nlohmann::json B9Report();

nlohmann::json Report();

} // namespace Core::NativeUpdates
