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
    u32 special{};    ///< 1 vertex shader has a variant object (+0x420), 2 / 4 context +0x4680 / +0x4688 set, 8 dispatch draw mode
    u32 vs_object_entries{}, ps_object_entries{}; ///< input usage entries of the shaders the context holds
    u32 vs_object_pgm{};
    u32 note_calls{}; ///< descriptors the record builder emitted for this draw (vertex buffer, resource, constant buffer, sampler descriptors)
    struct Range {
        u64 begin, end;
    } buffers[MaxBuffers];
    u64 images[MaxImages];

    void AddBuffer(u64 begin, u64 end);
    void AddImage(u64 start);

    /// NativeDraw v2: the descriptors the record builder put into the draw's user data, with where they came from. A native binder looks a shader resource up by
    /// (stage, flat user data offset) and gets the engine object, its resource and the descriptor words.
    static constexpr u32 MaxBindings = 160;
    struct Binding {
        u8 stage;    ///< 0 vertex, 1 pixel (the record builder's order)
        u8 kind;     ///< 1 buffer descriptor (V#), 2 image descriptor (T#), 3 sampler (S#)
        u16 offset;  ///< flat user data index of the first dword (the user data register, or 16 + the extended user data index); 0xffff: an entry of a table behind a pointer
        u16 slot;    ///< the engine's slot (constant buffer, shader resource, sampler) the object was found in
        u16 dwords;
        u64 object;   ///< the engine object in the slot (buffer, view, sampler)
        u64 resource; ///< the resource behind it (buffer or texture); the sampler itself
        u64 id;       ///< the engine's unique id of `resource` (object + 0x10), the key of the registry
        u32 words[8];
    };
    u32 binding_count{};
    u32 bindings_dropped{};
    Binding bindings[MaxBindings];
    void AddBinding(const Binding& binding);

    /// Diagnostics for the attribution of unexplained requests: buffers the context holds in slots the shaders' usage tables do not name
    /// (`where`: stage * 16 + slot of the constant buffer tables, 0x100 + stage * 128 + slot of the shader resource tables; +0x1000 for the
    /// object's own V# when it points elsewhere).
    static constexpr u32 MaxOthers = 48;
    u32 other_count{};
    struct Other {
        u64 begin, end;
        u32 where;
    } others[MaxOthers];
    void AddOther(u64 begin, u64 end, u32 where);
};
/// Adds the memory (and the V# range, when different) of a buffer object found in a context slot the description does not collect.
void CollectOtherBuffer(DrawResources& draw, u64 object, u32 where);
/// Reads the object (a Js resource) and adds its memory to the draw. Guarded by the caller.
void CollectResource(DrawResources& draw, u64 object);
void StoreDraw(const DrawResources& draw);
/// The constant update engine of the engine's GNM layer was given a resource: kind 0 constant buffer, 1 buffer, 2 texture, 3 user resource table; `caller` is the
/// Ghidra address of the call site. Counted per (kind, caller, stage, slot) for the report.
void NoteCueCall(u32 kind, u64 caller, u32 stage, u32 slot, u64 address, u64 bytes);
/// Diagnostics: a short text for the report (bounded).
void NoteDebug(const std::string& text);
/// Raster thread: the ranges the current draw's description holds (diagnostics).
std::string DescribeCurrentDraw();
/// Diagnostics: a device context pointer-sized slot at `offset` held a pointer to a Js resource (buffer / texture) that the known context tables do not
/// explain; counted per offset and kind for the report.
void NoteContextSlot(u32 offset, u32 resource_kind);
/// The engine's render device object (the `this` of the pass functions, seen by the hook of the per-primitive binding `FUN_01c92f30`): its buffer descriptor
/// table at +0xe80 (up to 7 V# of 16 bytes: the primitive's resource set, copied there by that function) is what the shaders that fetch their constants from a table read.
void SetRenderDevice(u64 render_device);
u64 RenderDevice();
/// Raster thread: the record sequence of the draw being bound (0 none).
void SetCurrentDraw(u64 seq, bool from_plan = false);
u64 CurrentDrawSeq();
/// True when the sequence number of the current draw is the one the plan was built from (an executed record), false when it is only the pending tag.
bool CurrentDrawFromPlan();
/// Number of draws set as current on this thread so far (diagnostics).
u64 CurrentDrawCallIndex();
/// True when [address, address + size) lies in a buffer the description of draw `seq` names.
bool DrawDescribes(u64 seq, u64 address, u64 size);
/// Aggregated for the report: an unexplained buffer request of the current draw, with where the shader got its descriptor: `sw_stage`, `summary` (0 the V# is
/// read straight from the user data / the flattened table, 1 the shader fetches it from a table in memory), the flat user data offset of the V#, whether the
/// record wrote those user data registers itself (`set_by_record`: 0 no, 1 yes, 2 not applicable), whether the table the shader fetches from is named by the description.
void NoteUnexplainedSite(u32 sw_stage, u32 summary, u32 offset, u32 set_by_record, u32 table_described, u32 plan_matches_record);
/// Command processor (front): a SET_SH_REG packet of the game's own command stream wrote `count` registers from SH register offset `reg_offset`. The user data windows of
/// the vertex (0x4c..0x5b) and pixel (0x0c..0x1b) stage are decoded for buffer descriptors (the engine's pass code writes per-pass constants there directly).
void NoteShPacket(u32 reg_offset, const u32* words, u32 count);

/// The pass class of what the raster thread is binding for (0 z prepass / shadow, 1 g-buffer, 2 forward, 3 compute / other, 15 untagged).
void SetDrawClass(u32 draw_class);
/// NativeDraw v2 audit (raster thread): the legacy backend reads the descriptor of a shader resource from the flat user data at `offset` (`words`, `dwords` of them;
/// offset 0xffff: it is not a single load of consecutive dwords). Compares it with the binding of the current draw's record at that offset: same words.
void AuditBinding(u32 sw_stage, bool image, u32 offset, const u32* words, u32 dwords);
/// The native binder (N3): 0 buffer descriptor taken from the record, 1 image descriptor taken from the record, 2 buffer fell back to the flat user data, 3 image fell back,
/// 4 buffer descriptor differed from the legacy one (verify mode), 5 image differed.
void NoteBinder(u32 what);
/// Returns true when the draw has a description and it does not explain this buffer.
/// `shader_constant`: the descriptor is built from immediates and the stage's program base (constants stored with the shader code), so it belongs to the shader
/// variant, not to the draw's engine objects.
bool AuditBuffer(u64 address, u64 size, bool written, bool shader_constant = false);
void AuditImage(u64 address);

void Start();
nlohmann::json Report();

} // namespace Core::NativeRegistry
