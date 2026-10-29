// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>

#include <nlohmann/json.hpp>

#include "common/types.h"

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
/// A short name of what owns `address` (for audits): "buffer usage U bind B", "texture_2d usage U bind B", "none"
[[nodiscard]] std::string OwnerName(VAddr address);

/// B1 shadow check: the texture cache bound an image at `address` of `width` x `height` for a binding whose record names engine object `id`
void CheckImage(u64 id, VAddr address, u32 width, u32 height);
/// ... and a buffer range for a binding whose record names `id`
void CheckBuffer(u64 id, VAddr address, u64 size);

[[nodiscard]] nlohmann::json Report();

/// Hook handlers (names in the hooks file: `native_view_srv`, `native_view_uav`, `native_view_rtv`, `native_view_dsv`, `native_delay_delete`)
u64 PS4_SYSV_ABI ViewCreatedSrv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedUav(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedRtv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI ViewCreatedDsv(u64 resource, u64 desc, u64 out, u64 a3, u64 a4, u64 a5, u64 ret);
u64 PS4_SYSV_ABI DelayDelete(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret);

} // namespace Core::NativeDevice
