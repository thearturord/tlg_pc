// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// CPU port of the game's ray against triangle compute shader (program hash 0xe15f3156, 112 byte hit records), the first target of "compute the readback on the CPU"
/// (documents/tlg/native plan, readback elimination). The GPU shader tests every triangle of a mesh against up to 128 rays and appends a hit record with an atomic
/// counter; the game's worker `FUN_01c4c570` reads the records and the counter one frame later through a staging copy. The port does the same work from the same guest
/// memory, so the answer no longer has to come back from the GPU.
///
/// Step 1 (this file): SHADOW only, switch `TLG_RAYCAST_SHADOW=1`, default off. The GPU dispatch still runs. The CPU result of every dispatch is kept; when the engine's
/// copy of the output buffer is executed the list of the frame is closed, and when the game's worker has read the landed staging copy the two are compared (the records as
/// a set, keyed by ray and triangle; floats within a few ulp because the GPU's inverse square root is not bit exact). Counters on the debug server endpoint /raycast.
namespace Core::NativeRaycast {

/// The program hash of the shader this file ports.
constexpr u32 ShaderHash = 0xe15f3156;

/// True when `TLG_RAYCAST_SHADOW=1`.
[[nodiscard]] bool ShadowOn() noexcept;

/// The five buffers of the dispatch as guest memory (the sharps' base addresses); `out_*` is the GPU's output buffer.
struct Buffers {
    const u32* params{};   u64 params_dwords{};
    const u32* indices{};  u64 indices_dwords{};
    const u32* vertices{}; u64 vertices_dwords{};
    const u32* rays{};     u64 rays_dwords{};
    u64 out_base{};        u32 out_capacity{}; ///< guest address of the output buffer, records it holds
    u32 groups_x{};
    bool vertices_from_skin{}; ///< the vertices are the CPU skinning port's result (guest memory holds zeros there)
};

/// Raster thread (shader HLE point, before the GPU dispatch is recorded): runs the shader on the CPU and appends the hits to the list of the current frame.
void ShadowDispatch(const Buffers& buffers);

/// Raster thread, every other compute dispatch while the shadow is on: the dispatch with program `hash` writes the buffer [base, base + size). If it overlaps a vertex buffer
/// the ray dispatches read, the program is the producer of those vertices (reported under "vertex_writers").
void NoteComputeWrite(u32 hash, u64 base, u64 size, u32 groups_x);

/// Raster thread (the command processor's DMA, in stream order): the engine copied `bytes` from `source` (guest memory) to `destination`, or the counter at GDS offset
/// `source` to `destination`. The copy of the output buffer closes the list.
void NoteMemoryCopy(u64 source, u64 destination, u64 bytes);
void NoteCounterCopy(u64 gds_offset, u64 destination);

/// Game worker thread, after `FUN_01c4c570` returned (the landed staging data is there): compares the landed counter and list with the closed lists.
void CompareAfterRead();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeRaycast
