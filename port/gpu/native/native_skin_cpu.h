// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// CPU port of the game's skinning compute shader (program hash 0xb5a91460, 4 bone weights, 32 byte vertices), the second step of "compute the readback on the CPU":
/// the ray against triangle shader (native_raycast_cpu.h) reads vertices that the GPU skinning wrote, so the CPU needs the skinned positions. Shadow only, behind
/// `TLG_RAYCAST_SHADOW=1`: the GPU dispatch still runs; the CPU result of the two position outputs (x, y, z of each vertex) is kept by output buffer address and the
/// raycast port reads it instead of guest memory, which holds zeros there.
namespace Core::NativeSkinCpu {

constexpr u32 ShaderHash = 0xb5a91460;

struct Buffers {
    const u32* b0{}; u64 n0{}; ///< per vertex: 4 bone indices, 4 weights (8 dwords)
    const u32* b1{}; u64 n1{}; ///< constants: vertex base, bone base, transform
    const u32* b2{}; u64 n2{}; ///< bind pose vertices (8 dwords)
    const u32* b3{}; u64 n3{}; ///< bone matrices (16 dwords each)
    u64 out_a{}; u64 out_a_dwords{}; ///< output buffer 4 (object space positions), guest address
    u64 out_b{}; u64 out_b_dwords{}; ///< output buffer 5 (transformed positions)
    u32 groups_x{};
};

/// Raster thread, in stream order before the raycast that reads the result.
void ShadowDispatch(const Buffers& buffers);

/// The second skinning shader (program hash 0xbdcd8c92, the 13 buffer variant with more weights): the buffers by binding, the position outputs are bindings 6 and 7.
constexpr u32 ShaderHashWide = 0xbdcd8c92;
struct WideBuffers {
    const u32* p[13]{};
    u64 n[13]{};
    u64 out_a{}, out_a_dwords{}, out_b{}, out_b_dwords{};
    u32 groups_x{};
};
void ShadowDispatchWide(const WideBuffers& buffers);

/// The CPU skinned vertices (8 dwords per vertex, x y z filled) of the output buffer that starts at `base` and holds `dwords`; null when the CPU did not skin it.
[[nodiscard]] const u32* Lookup(u64 base, u64 dwords);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeSkinCpu
