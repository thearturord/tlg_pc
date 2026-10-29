// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/types.h"
#include "core/guest_hooks.h"

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
};

/// `TLG_NATIVE_DEVICE=0` turns the table off (default on: it only observes)
[[nodiscard]] bool Enabled() noexcept;

/// The resource object after its creation returned (buffers from `CreateBuffer`, textures from their initialization)
void NoteResource(u64 object);
/// A view object after its creation returned
void NoteView(Kind kind, u64 view_object);
/// `AddDelayDeleteObject`: the object leaves (its id is never used again)
void NoteDelete(u64 object);

/// The live object of an id, copied (false: unknown or deleted)
[[nodiscard]] bool Lookup(u64 id, Object& out);
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

/// B1 shadow check: the texture cache bound an image at `address` of `width` x `height` for a binding whose record names engine object `id`
void CheckImage(u64 id, VAddr address, u32 width, u32 height);
/// ... and a buffer range for a binding whose record names `id`
void CheckBuffer(u64 id, VAddr address, u64 size);
/// B6: does the CPU read [address, address + size)? True for memory of a resource the engine created for CPU reads (staging usage or CPU read access) and for memory
/// no engine resource owns (unknown: kept safe); false for every other engine resource (the GPU's own data)
[[nodiscard]] bool CpuReads(VAddr address, u64 size);
/// the same with the reason: 0 an engine resource the CPU does not read, 1 one it reads (staging / CPU read access), 2 no engine resource owns the range
[[nodiscard]] u32 CpuReadReason(VAddr address, u64 size);
/// B6 switch (`TLG_B6_NATIVE=1`): results go back into PS4 memory only where the CPU reads them
[[nodiscard]] bool B6NativeOn() noexcept;
/// B4 census: a color (`depth` false) or depth target the draw renders into, at `address` of `width` x `height`
void CheckTarget(bool depth, VAddr address, u32 width, u32 height);

[[nodiscard]] nlohmann::json Report();

/// Hook handlers (names in the hooks file: `native_view_srv`, `native_view_uav`, `native_view_rtv`, `native_view_dsv`, `native_delay_delete`)
u64 PS4_SYSV_ABI ViewCreatedSrv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedUav(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedRtv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedDsv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI DelayDelete(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret);

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
/// B11 step 3 (`TLG_B11_CLEARS=1`): the engine's clear runs as a Vulkan clear of the target's images at its place in the stream; the work inside its packets
/// (the meta fill, the fast clear eliminate, the depth target clear, its draws and dispatches) is skipped
[[nodiscard]] bool B11ClearsOn() noexcept;
enum ClearNative : u32 { NativeDone, NativeImages, NativeNotEligible, NativeNoImage, NativeSkippedWork, NativeFastClearKept, NativeSubresource, ClearNatives };
void NoteClearNative(ClearNative what, u64 n = 1);

} // namespace Core::NativeDevice
