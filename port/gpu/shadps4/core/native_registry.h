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
void AuditBuffer(u64 address, u64 size, bool written);
void AuditImage(u64 address);

void Start();
nlohmann::json Report();

} // namespace Core::NativeRegistry
