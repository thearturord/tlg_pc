// SPDX-License-Identifier: GPL-2.0-or-later
// Resource registry of the native renderer (documents/tlg/native-renderer-plan.md, phase N1, shadow mode): every engine object that a draw binds
// (shaders, input layout, buffers, shader resource / render target / depth views and the textures behind them, state objects) is entered by its
// engine id (object + 0x10) the first time a draw uses it. Nothing in the rendering depends on it; it is read by the audit in the backend, which
// asks for each buffer and image the legacy path resolves by guest address whether the registry knows the resource and agrees on its extent.
// The objects' layouts are in documents/tlg/render-engine/native-resource-model.md.

#pragma once

#include <atomic>
#include <string>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

namespace Core::NativeRegistry {

enum class Kind : u8 {
    Unknown,
    Buffer,
    Texture1D,
    Texture2D,
    Texture3D,
    ShaderResourceView,
    RenderTargetView,
    DepthStencilView,
    UnorderedAccessView,
    BlendState,
    RasterizerState,
    DepthStencilState,
    SamplerState,
    InputLayout,
    VertexShader,
    PixelShader,
    ComputeShader,
    NumKinds,
};

extern std::atomic<bool> enabled;

/// Guest render thread: an object found in a slot of the device context. `hint` says what the slot holds. Cheap when the object is known.
void Observe(u64 object, Kind hint);

/// The engine created a buffer (complete) / is initializing a texture (read later) / destroys a buffer. Hooked creation functions; cheap when the registry is off.
void NoteBufferCreated(u64 object);
void NoteTextureInitializing(u64 object);
void NoteBufferDestroyed(u64 object);

/// Raster thread: the legacy path resolved [address, address + size) as a buffer / the image that starts at `address`.
/// N2 (native description of a draw), step 1: the resources one draw binds according to the engine's own objects (buffers as [begin, end) of the
/// object's memory, textures as the start of their memory), collected on the guest thread when the draw is tagged and stored under the record's
/// sequence number. The audit (raster thread) checks that every buffer / image the legacy path resolves for that draw is one of them.
struct DrawResources {
    static constexpr u32 MaxBuffers = 64;
    static constexpr u32 MaxImages = 64;
    u64 seq{};
    u32 buffer_count{};
    u32 image_count{};
    u32 dropped{}; ///< entries that did not fit
    struct Range {
        u64 begin, end;
    } buffers[MaxBuffers];
    u64 images[MaxImages];

    void AddBuffer(u64 begin, u64 end);
    void AddImage(u64 start);
};
/// Reads the object (a Js resource) and adds its memory to the draw. Guarded by the caller.
void CollectResource(DrawResources& draw, u64 object);
void StoreDraw(const DrawResources& draw);
/// Diagnostics: a device context pointer-sized slot at `offset` held a pointer to a Js resource (buffer / texture) that the known context tables do not
/// explain; counted per offset and kind for the report.
void NoteContextSlot(u32 offset, u32 resource_kind);
/// Raster thread: the record sequence of the draw being bound (0 none).
void SetCurrentDraw(u64 seq);

/// The pass class of what the raster thread is binding for (0 z prepass / shadow, 1 g-buffer, 2 forward, 3 compute / other, 15 untagged).
void SetDrawClass(u32 draw_class);
void AuditBuffer(u64 address, u64 size, bool written);
void AuditImage(u64 address);

void Start();
nlohmann::json Report();

} // namespace Core::NativeRegistry
