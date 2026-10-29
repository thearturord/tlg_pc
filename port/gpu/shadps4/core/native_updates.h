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
/// B8.2: the engine deleted a view (`what` 0), a texture (1) or a texture another live texture shares the memory with (3); queued with the next submission's batch
void NoteEngineDelete(u64 id, u64 address, u64 bytes, u32 what);
/// Where a deletion is carried out (raster thread, in order with the GPU work); set by the rasterizer
void SetDelete(void (*fn)(void* context, u64 id, u64 address, u64 bytes, u32 what), void* context);
/// B8.4: the engine created a texture (its memory gets a new owner); queued with the next submission's batch like a deletion
void NoteEngineCreate(u64 id, u64 address, u64 bytes);
void SetCreate(void (*fn)(void* context, u64 id, u64 address, u64 bytes), void* context);
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

nlohmann::json Report();

} // namespace Core::NativeUpdates
