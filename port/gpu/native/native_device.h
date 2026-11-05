// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/types.h"
#include "native/native_hooks.h"

/// The native back end's device layer (documents/tlg/native plan/native-backend-100-plan.md, B1 onward). The engine renders through a Direct3D 11 style
/// device layer (`JsDevice`, `JsDeviceContext`); every object it creates (buffers, textures, the four kinds of views) carries a unique 64 bit id (+0x10,
/// never reused) and its Direct3D 11 description. This table holds those objects by id, filled from the creation hooks, so the back end can find what
/// a draw binds by what it is instead of by its PS4 address.
namespace Core::NativeDevice {

enum class Kind : u8 {
    Unknown = 0,
    Buffer,
    Texture1D,
    Texture2D,
    Texture3D,
    ShaderResourceView,
    UnorderedAccessView,
    RenderTargetView,
    DepthStencilView,
};

struct Object {
    u64 id{};
    u64 object{};     ///< the engine object (guest pointer); valid while `alive`
    Kind kind{};
    bool alive{};
    u32 usage{}, bind{}, misc{}, format{};
    u32 cpu_access{}; ///< D3D11 CPUAccessFlags (0x10000 write, 0x20000 read)
    u32 width{}, height{}, depth{}, mips{}, layers{};
    VAddr address{};  ///< resources: the memory (+0x30); views: the memory of their resource
    u64 bytes{};      ///< resources: the size of the memory (+0x38 for textures, ByteWidth for buffers)
    u64 resource_id{};    ///< views: the id of the resource they look at
    Kind resource_kind{}; ///< views: the kind of that resource
    u32 stride{};         ///< buffers: StructureByteStride
    bool has_tsharp{};    ///< texture views: `tsharp` is the T# the engine made for the view when it created it (shader resource view: object +0x58, unordered access view: +0x54)
    u32 tsharp[8]{};      ///< (the view creation copies the texture's T# and edits only the mip levels and the array range; later only the address refresh changes it: F4 notes, question a)
    u32 created_submit{}, deleted_submit{}; ///< the graphics submission count when the object was noted and when it was deleted (0: not deleted); late read samples name the owner of an address at a submission
    bool has_desc{};      ///< views: `desc` holds the first dwords of the description the view was created with (D3D11 layout: format, dimension, then the range)
    u32 desc[8]{};
};

/// `TLG_NATIVE_DEVICE=0` turns the table off (default on: it only observes)
[[nodiscard]] bool Enabled() noexcept;

/// The resource object after its creation returned (buffers from `CreateBuffer`, textures from their initialization)
void NoteResource(u64 object);
/// A view object after its creation returned
void NoteView(Kind kind, u64 view_object, u64 desc = 0);
/// `AddDelayDeleteObject`: the object leaves (its id is never used again)
void NoteDelete(u64 object);

/// The live object of an id, copied (false: unknown or deleted)
[[nodiscard]] bool Lookup(u64 id, Object& out);
/// The live shader resource views (`out`) and unordered access views (`out_uav`) (engine view objects) whose resource is engine object `resource_object`
void ViewsOf(u64 resource_object, std::vector<u64>& out, std::vector<u64>& out_uav);
/// E2 study (`TLG_DESC_CHECKS=1`): the first 0x400 bytes of the object as they were when its creation returned
[[nodiscard]] bool RawBytes(u64 id, u8* out, size_t size);
/// One line for the late read samples: the resources (alive or deleted, buffers and textures) whose memory meets [address, address + size) and that were alive at graphics submission `submit`
/// (created at or before it, deleted after it); at most 3, newest first
[[nodiscard]] std::string OwnersAt(VAddr address, u64 size, u32 submit);
/// The end address of the newest engine buffer alive at `submit` that contains `address` (0: none).
[[nodiscard]] u64 OwnerEndAt(VAddr address, u32 submit);
/// The live resource (buffer or texture) whose memory holds `address`, copied (false: none)
[[nodiscard]] bool ResourceAt(VAddr address, Object& out);
/// B8.4: the live textures whose memory meets [address, address + size) (ids appended to `out`); true when together they cover the whole range
[[nodiscard]] bool TexturesIn(VAddr address, u64 size, std::vector<u64>& out);
/// B9.1: the live buffers whose memory meets [address, address + size), each with the part of the range inside it (appended to `out`)
struct BufferPart {
    u64 id;
    VAddr begin, end;               ///< the part of the asked range inside the buffer
    u32 usage, cpu_access;
    VAddr object_begin, object_end; ///< the buffer's whole memory
};
void BuffersIn(VAddr address, u64 size, std::vector<BufferPart>& out);
/// A short name of what owns `address` (for audits): "buffer usage U bind B", "texture_2d usage U bind B", "none"
[[nodiscard]] std::string OwnerName(VAddr address);

/// E2: a Map of buffer `id` returned `address` (`size` bytes, `type` the Direct3D map type); the buffer's data is there until its next Map
void NoteMapped(u64 id, u64 address, u64 size, u32 type);
/// B1 shadow check: the texture cache bound an image at `address` of `width` x `height` for a binding whose record names engine object `id`
/// E2 census (`TLG_DESC_CHECKS=1`): the fields of the T# a binding of engine view / texture `id` carries, against the engine's own description of the object (and of
/// the resource a view looks at). Counts per field how often the two agree; the format pair is checked as a function of the engine format.
struct SharpFacts {
    u32 data_format{}, num_format{}, type{};
    u32 width{}, height{}, depth{};       ///< the T# fields as stored (width and height are size - 1)
    u32 base_level{}, last_level{}, base_array{}, last_array{};
    u32 pitch{};
    u32 dst_sel{};      ///< the four component selects packed x | y << 3 | z << 6 | w << 9
    u32 tiling_index{}, pow2pad{};
    VAddr address{};
};
void CheckImageSharp(u64 id, u64 view_id, const SharpFacts& facts);
/// E2 census for buffers: the fields of the V# a binding of buffer `id` (through view `view_id`, or 0) carries, against the engine's buffer and view descriptions
void CheckBufferSharp(u64 id, u64 view_id, u64 base_address, u32 stride, u32 num_records, u32 data_format, u32 num_format, bool is_constant_buffer_hint);
/// ... and a buffer range for a binding whose record names `id`
/// B6: does the CPU read [address, address + size)? True for memory of a resource the engine created for CPU reads (staging usage or CPU read access) and for memory
/// no engine resource owns (unknown: kept safe); false for every other engine resource (the GPU's own data)
[[nodiscard]] bool CpuReads(VAddr address, u64 size);
/// the same with the reason: 0 an engine resource the CPU does not read, 1 one it reads (staging / CPU read access), 2 no engine resource owns the range
[[nodiscard]] u32 CpuReadReason(VAddr address, u64 size);
/// B6 switch (`TLG_B6_NATIVE=1`): results go back into PS4 memory only where the CPU reads them
[[nodiscard]] bool B6NativeOn() noexcept;
/// B4 census: a color (`depth` false) or depth target the draw renders into, at `address` of `width` x `height`

[[nodiscard]] nlohmann::json Report();

/// Hook handlers (names in the hooks file: `native_view_srv`, `native_view_uav`, `native_view_rtv`, `native_view_dsv`, `native_delay_delete`)
u64 PS4_SYSV_ABI ViewCreatedSrv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedUav(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedRtv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedDsv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI DelayDelete(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret);
/// native-v3-writers (`TLG_DEFER_FREE=K`, hook `GomallocFree` at 0x248fd20, handler `native_gomalloc_free`): the engine's memory free (heap, address) of its GPU heaps (garlic / onion) is held for K game
/// frames so the loader cannot reuse memory a frame in flight (raster thread, up to 2 frames behind) still reads; `DrainDeferredFrees` (called at the start of every frame) frees what is due.
u64 PS4_SYSV_ABI GomallocFree(u64 heap, u64 address, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret);
void DrainDeferredFrees();
/// Hook `GomallocAlloc` (0x248fb10, `native_gomalloc_alloc`): the engine heap's allocation (heap, size, alignment, type). Runs the original; with `TLG_LOADER_GATE=1` remembers the block's size and
/// waits at once when the block lies on memory freed while the raster thread had not finished the submissions that could read it.
u64 PS4_SYSV_ABI GomallocAlloc(u64 heap, u64 size, u64 alignment, u64 type, u64 a4, u64 a5, u64 ret);
/// Hook `B0_SwapChainBeginFrame` (0x2495480, `native_begin_frame_tick`): runs the original and marks the Gomalloc frees inside it as the game's delete tick (W2, `TLG_W2_TICK=1`).
u64 PS4_SYSV_ABI BeginFrameTick(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret);
/// native-v3-writers (`TLG_LOADER_GATE=1`): the engine's allocation (`address`, `size`; hooks AllocGarlicMemory / AllocOnionMemory) overlaps memory that was freed (hook GomallocFree) when the game
/// had submitted `n` frames: the game thread waits until the raster thread has executed `n` frames, so the loader never overwrites memory a frame in flight still reads. Keeps normal memory reuse.
void LoaderGateAtAllocation(u64 address, u64 size, u64 ret = 0);
/// native-v3-writers (`TLG_SKIN_GATE=1`, called by the `SkinWrites` hook before the original): the skinning matrix ring has 4 slots and two passes per frame, so a slot is written again two frames
/// after its last write. When slot `index` is about to be written, the game thread waits until the raster thread has finished the submissions that could still read its old contents.
void SkinSlotAboutToWrite(u32 index);

/// B11 census: mid hooks at +6 (after the 6 bytes the B0 count hook displaced) of `JsDeviceContext::ClearRenderTargetView` (0x2483380: ctx, rtv, color[4]) and `ClearDepthStencilView` (0x24860c0: depth in
/// xmm0, ctx, dsv, flags, stencil). They record the target and the value and bracket the packets the clear writes into the context's command buffer (its write cursor at
/// ctx+0x47d0 before and after, through an exit stub); the packets are decoded by opcode (names in the hooks file: `b11_clear_rtv`, `b11_clear_dsv`)
u64 PS4_SYSV_ABI ClearRtvEnter(Core::GuestHooks::MidContext* c);
u64 PS4_SYSV_ABI ClearDsvEnter(Core::GuestHooks::MidContext* c);
[[nodiscard]] nlohmann::json ClearReport();
/// B11 step 2: an engine clear and the packets it wrote ([begin, end) in its command buffer)
struct ClearRecord {
    u64 begin{}, end{};
    u32 kind{};  ///< 0 color, 1 depth / stencil
    u32 path{};  ///< color: 1 the target has a meta surface; depth: 1 the target has an HTILE surface
    u32 flags{}, stencil{};
    u64 view_id{}, resource_id{}, address{};
    float value[4]{}; ///< color, or depth in [0]
    /// the view's subresources (B11 step 4, from the view's description at +0x40): `mip`, layers [first_layer, first_layer + layer_count);
    /// layer_count 0: the view is of a 3D texture (slices) or unknown
    u32 mip{}, first_layer{}, layer_count{1};
};
/// The clears recorded in [begin, end) (a submitted command buffer), in order; taken out of the pending list
void TakeClears(u64 begin, u64 end, std::vector<ClearRecord>& out);
/// What the back end did inside / outside an engine clear's packets (raster thread)
enum ClearUse : u32 { UseMetaClearDispatch, UseFastClearEliminate, UseFastClearEliminateNothing, UseDepthClearTarget, UseRenderPassClear, UseDraw, UseDispatch, ClearUses };
void NoteClearUse(ClearUse use, bool inside);
/// The command processor met a clear's packets: entries inside by kind (0 tag, 1 sync, 2 packet, 3 other)
void NoteClearEntered(const ClearRecord& clear, const u32 (&entries)[4]);
/// The depth / stencil clear the game is inside of right now (its ClearDepthStencilView call has not returned): false when none. Game thread.
[[nodiscard]] bool CurrentDepthClear(ClearRecord& out);
enum ClearNative : u32 { NativeDone, NativeImages, NativeNotEligible, NativeSkippedWork, NativeFastClearKept, NativeSubresource, NativeFastClearPending, NativeFastClearAtRenderPass, NativeFastClearAtEliminate, NativeClearBeforeImage, NativeClearAtFirstImage, NativeClearFirstImageMismatch, ClearNatives };
void NoteClearNative(ClearNative what, u64 n = 1);

} // namespace Core::NativeDevice
