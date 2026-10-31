// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

// What the native resource model knows about the game's GPU memory: the engine allocates every buffer and texture from two
// heaps of its own (garlic: write combined GPU memory, onion: CPU coherent memory). The census scans those heaps once a
// second and attributes the backend's buffer requests, uploads, image refreshes and CPU write faults to the allocation they
// fall in. It answers which resources the GPU reads, how big they are, who rewrites them and how often (debug server
// endpoint /resources, `?start=1` turns it on).
namespace Core::NativeResources {

/// How a buffer request was answered.
enum class BufferPath : u8 {
    StreamCopy,  ///< read-only, small: copied into the stream buffer
    StreamReuse, ///< the same copy as earlier in the command buffer
    Memo,        ///< large or written, answered from the per-command-buffer memo
    Arena,       ///< large or written, looked up in the arenas (`uploaded` bytes were copied from guest memory)
};

extern bool census_enabled;

/// The engine created a buffer (called after JsBuffer creation, the object is complete).
void NoteBufferObject(u64 object);
/// The engine is initializing a texture (called before the initialization; the object is read on the next scan).
void NoteTextureObject(u64 object);
/// The engine destroys a buffer.
void NoteBufferDestroyed(u64 object);

void NoteBufferRequest(VAddr address, u32 size, bool written, BufferPath path, u64 uploaded);
/// An image was (re)uploaded from guest memory.
void NoteImageUpload(VAddr address, u64 size, bool from_gpu_memory);
/// B1 census: an image the texture cache registers, against the engine resource at its address (0 engine texture with the same size, 1 engine texture of
/// another size, 2 engine buffer, 3 no engine resource at that address); `render_target` / `depth`: how the cache made it
void NoteImageCensus(VAddr address, u32 width, u32 height, bool render_target, bool depth);
[[nodiscard]] nlohmann::json ImageCensus();
/// The CPU wrote tracked memory.
void NoteCpuWrite(VAddr address, u64 size);

/// An engine buffer with Usage IMMUTABLE: the engine writes its contents once, at creation, and never again. Once the backend
/// has synchronized the whole buffer with the GPU, requests for it need no tracker query and no copy. A CPU write fault on the
/// range (the page protection stays as the safety net) or the destruction of the buffer drops that state.
struct ImmutableBuffer {
    u64 begin{};
    u64 end{};
    u64 object{};
    u64 id{}; ///< the engine's unique id of the buffer (JsResource +0x10): the key of the native handles (N3c)
    std::atomic<bool> alive{true};
    std::atomic<bool> synced{false};
    std::atomic<u32> generation{0}; ///< bumped by every CPU write to the range
    std::atomic<u64> hash{0};       ///< hash of the guest memory at the first check after the synchronization (0: not taken yet)
    std::atomic<u32> requests{0};
    /// R1 (native back end): where the object is resident on the GPU since its last synchronization, written by the raster thread before `synced` (release):
    /// the buffer cache's arena (a `VideoCore::Buffer`), its guest base address and the arena epoch it was found at
    std::atomic<const void*> resident_buffer{nullptr};
    std::atomic<u64> resident_base{0};
    std::atomic<u64> resident_epoch{~u64{0}};
    std::atomic<u32> gpu_writes{0}; ///< R1b: bumped when a GPU write (a written storage buffer, a copy or fill) covers bytes of the object
    /// R1b (raster thread only): the object's own copy in the stream ring, made once per generation for a default / dynamic buffer: valid while the generation,
    /// `gpu_writes` and the ring's wrap count are the ones it was made at
    u64 copy_offset{~u64{0}};
    u64 copy_wraps{};
    u32 copy_generation{};
    u32 copy_gpu_writes{};
    u8 usage{1}; ///< 1 immutable (the engine writes it once); 0 default and 2 dynamic only with `TLG_RESIDENT_BUFFERS` (C1, see below)
};

/// C1 / C3: resident engine buffers (`TLG_RESIDENT_BUFFERS=1`). The residency of the immutable buffers (a buffer synchronized once is read from the arena with no tracker query)
/// is extended to the engine's `default` and `dynamic` usage buffers up to `TLG_RESIDENT_CAP` bytes (default 262144) that the GPU does not write. They are found by the engine id
/// of the record's binding (no address walk), and a request is answered from the arena while nothing announced a write and no CPU write fault touched the page of the buffer
/// since the last synchronization: the generation of the buffer is bumped by `NoteCpuWrite` (the whole 4 KiB page, as the page protection is per page) and by `NoteAnnouncedWrite`.
/// `TLG_RESIDENT_VERIFY=N` (default 64): every Nth answer without a synchronization is checked with one (nothing may have to be uploaded: else the answer would have been stale).
bool ResidentBuffersEnabled();
u32 ResidentCap();
u32 ResidentVerifyInterval();
/// An engine writer announced a write of [address, address + size): the resident buffers on those pages are not synchronized any more.
void NoteAnnouncedWrite(VAddr address, u64 size);
/// R1b: the GPU writes [address, address + size) (byte exact): the copies of the engine buffers it touches are stale.
void NoteGpuWritten(VAddr address, u64 size);
/// Counters (0 answered without a synchronization, 1 synchronized (first use or after a write), 2 verify run, 3 verify run that found something to upload (stale answer),
/// 4 request left to the normal path because the GPU modified the range, 5 buffer registered, 6 written by an announcement, 7 dropped by a CPU write fault).
void NoteResident(u32 what);
std::array<u64, 8> TakeResidentCounters();

/// True when native_sdk_strip bit 65536 is set.
bool ImmutableResidencyEnabled();
/// The live immutable buffer that contains [address, address + size), or null.
ImmutableBuffer* FindImmutable(VAddr address, u32 size);
/// N3c: the live immutable buffer of the engine id `id` that contains [address, address + size) (size 0: the id alone), or null. Per thread cache by id, valid while
/// `ImmutableEpoch` does not change (a buffer was registered or destroyed). The pointer is only good for the epoch read together with it.
ImmutableBuffer* FindImmutableById(u64 id);
u64 ImmutableEpoch();
/// N3c counters of the handle path (0 handle used, 1 handle stale because the epoch changed, 2 the id names no immutable buffer, 3 the request is outside the
/// handle's range, 4 verify mode: the handle differs from the address lookup).
void NoteHandle(u32 what);
/// The backend synchronized the whole buffer; `generation` is the one read before it started.
void ImmutableSynced(ImmutableBuffer& buffer, u32 generation);
/// Every 4096th request: true when the guest memory no longer matches what was synchronized (the state is dropped).
bool ImmutableCheck(ImmutableBuffer& buffer);
/// Counters of the residency: hits answered without a copy, whole buffer synchronizations, drops by CPU writes.
void NoteImmutableHit();
void NoteImmutableSync();

/// Image to image copies of the engine's CopyResource: done natively, or refused (reason: 0 no source image, 1 no destination
/// image, 2 the two images do not have the same layout).
void NoteImageCopy(bool done, u32 reason);
/// True when native_sdk_strip bit 131072 is set.
bool ImageCopyEnabled();
/// A memory fill of an image: done as an image clear, or refused (reason: 0 no image there, 1 the image is not a plain colour
/// image of the fill's size, 2 the pattern is not one texel or the format is not handled).
void NoteImageClear(bool done, u32 reason);

/// Usage of a JsBuffer / JsTexture object (0 default, 1 immutable, 2 dynamic, 3 staging), ~0u when it cannot be read.
u32 ResourceUsage(u64 object);

/// The usage class (0 unknown, else 1 + (kind - 1) * 4 + usage) of the heap block that contains `address`, from the census snapshot.
u32 UsageClassOf(VAddr address);
/// "buffer/dynamic" and the like.
std::string UsageClassNameOf(u32 usage_class);
/// The heap block (census snapshot) that contains `address`: its start and size. False when the census is off or the address is in no block.
bool BlockContaining(VAddr address, u64* start, u64* size);
/// The engine's frame allocator window (census), false when unknown.
bool FrameRing(u64* base, u64* size);
/// The heap block around `address`: class, begin and size ("unknown" when the census is off or the address is in no block).
std::string DescribeAddress(VAddr address);

void Start();
nlohmann::json Report();

} // namespace Core::NativeResources
