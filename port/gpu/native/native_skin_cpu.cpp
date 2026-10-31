// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_skin_cpu.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

// The arithmetic is the shader's, statement by statement (native_skin_b5a91460.inc is generated from documents/tlg/shaders/cs_b5a91460.glsl by
// documents/tlg/tools/glsl_to_cpp_closure.py). Never fuse.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace Core::NativeSkinCpu {

namespace {

float F(u32 bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

u32 B(float value) {
    u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float max(float a, float b) {
    return std::fmax(a, b);
}

float clamp(float x, float lo, float hi) {
    return std::fmin(std::fmax(x, lo), hi);
}

float Rsqrt(float x) {
    return 1.0f / std::sqrt(x);
}

/// Reads past the end give zero, as the graphics card's robust buffer access does.
struct U32Array {
    const u32* p{};
    u64 n{};
    u32 operator[](u64 i) const { return i < n ? p[i] : 0u; }
};
struct FloatArray {
    const u32* p{};
    u64 n{};
    float operator[](u64 i) const { return F(i < n ? p[i] : 0u); }
};

struct Skinned {
    std::vector<u32> data;
};

std::mutex g_mutex;
std::unordered_map<u64, Skinned> g_store; ///< by output buffer address
std::atomic<u64> g_dispatches{}, g_vertices{}, g_lookups{}, g_lookup_hits{};

} // namespace

void ShadowDispatch(const Buffers& b) {
    if (b.b0 == nullptr || b.b1 == nullptr || b.b2 == nullptr || b.b3 == nullptr || b.out_a == 0) {
        return;
    }
    const U32Array A1{b.b0, b.n0}, A2{b.b1, b.n1}, A3{b.b2, b.n2}, A4{b.b3, b.n3};
    const FloatArray FA3{b.b2, b.n2};
    const u64 vertices = std::min<u64>(static_cast<u64>(b.groups_x) * 64u, b.out_a_dwords / 8);
    std::vector<u32> out_a(b.out_a_dwords, 0u), out_b(b.out_b_dwords, 0u);
    for (u32 gid = 0; gid < vertices; ++gid) {
#include "native/native_skin_b5a91460.inc"
        out_a[static_cast<u64>(gid) * 8 + 0] = B(_877);
        out_a[static_cast<u64>(gid) * 8 + 1] = B(_882);
        out_a[static_cast<u64>(gid) * 8 + 2] = B(_916);
        if (b.out_b != 0 && static_cast<u64>(gid) * 8 + 2 < out_b.size()) {
            out_b[static_cast<u64>(gid) * 8 + 0] = B(_1013);
            out_b[static_cast<u64>(gid) * 8 + 1] = B(_1030);
            out_b[static_cast<u64>(gid) * 8 + 2] = B(_1008);
        }
    }
    g_dispatches.fetch_add(1, std::memory_order_relaxed);
    g_vertices.fetch_add(vertices, std::memory_order_relaxed);
    std::lock_guard lock(g_mutex);
    g_store[b.out_a].data = std::move(out_a);
    if (b.out_b != 0) {
        g_store[b.out_b].data = std::move(out_b);
    }
}

void ShadowDispatchWide(const WideBuffers& b) {
    if (b.out_a == 0 || b.p[0] == nullptr) {
        return;
    }
    const auto arr = [&](u32 i) { return U32Array{b.p[i], b.n[i]}; };
    const U32Array A1 = arr(0), A2 = arr(1), A3 = arr(2), A4 = arr(3), A5 = arr(4), A6 = arr(5), A7 = arr(6), A8 = arr(7), A9 = arr(8), A10 = arr(9), A11 = arr(10), A12 = arr(11), A13 = arr(12);
    (void)A3; (void)A7; (void)A8; (void)A9; (void)A10; (void)A11; (void)A12; (void)A13; (void)A4; (void)A5; (void)A6; (void)A2;
    const u64 vertices = std::min<u64>(static_cast<u64>(b.groups_x) * 64u, b.out_a_dwords / 8);
    std::vector<u32> out_a(b.out_a_dwords, 0u), out_b(b.out_b_dwords, 0u);
    for (u32 gid = 0; gid < vertices; ++gid) {
#include "native/native_skin_bdcd8c92.inc"
        out_a[static_cast<u64>(gid) * 8 + 0] = _1310_x;
        out_a[static_cast<u64>(gid) * 8 + 1] = _1310_y;
        out_a[static_cast<u64>(gid) * 8 + 2] = _1310_z;
        if (b.out_b != 0 && static_cast<u64>(gid) * 8 + 2 < out_b.size()) {
            out_b[static_cast<u64>(gid) * 8 + 0] = _1542_x;
            out_b[static_cast<u64>(gid) * 8 + 1] = _1542_y;
            out_b[static_cast<u64>(gid) * 8 + 2] = _1542_z;
        }
    }
    g_dispatches.fetch_add(1, std::memory_order_relaxed);
    g_vertices.fetch_add(vertices, std::memory_order_relaxed);
    std::lock_guard lock(g_mutex);
    g_store[b.out_a].data = std::move(out_a);
    if (b.out_b != 0) {
        g_store[b.out_b].data = std::move(out_b);
    }
}

const u32* Lookup(u64 base, u64 dwords) {
    g_lookups.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(g_mutex);
    const auto it = g_store.find(base);
    if (it == g_store.end() || it->second.data.size() < dwords) {
        return nullptr;
    }
    g_lookup_hits.fetch_add(1, std::memory_order_relaxed);
    return it->second.data.data();
}

nlohmann::json Report() {
    nlohmann::json j;
    j["dispatches"] = g_dispatches.load();
    j["vertices"] = g_vertices.load();
    j["lookups"] = g_lookups.load();
    j["lookup_hits"] = g_lookup_hits.load();
    std::lock_guard lock(g_mutex);
    j["buffers_skinned"] = g_store.size();
    return j;
}

} // namespace Core::NativeSkinCpu
