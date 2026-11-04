// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_sync.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "native/native_wait_registers.h"
#include "common/uint128.h"
#include "core/libraries/kernel/time.h"

namespace Core::NativeSync {

u64 Fence::GpuClock64() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

u64 Fence::GpuPerfCounter() {
    const auto cpu_freq = Libraries::Kernel::sceKernelGetTscFrequency();
    constexpr u64 gpu_freq = 800'000'000; // the GPU core clock of a base PS4 (the port never reports Neo mode)
    return Common::MultiplyAndDivide64(Libraries::Kernel::sceKernelReadTsc(), gpu_freq, cpu_freq);
}

u64 DecodeEventAddress(const u32* packet) {
    return static_cast<u64>(packet[2] & 0xfffffff8u) | static_cast<u64>(packet[3] & 0xffu) << 32;
}

bool LabelWait::TestValue(u32 value) const {
    switch (function) {
    case Function::Always:
        return true;
    case Function::LessThan:
        return (value & mask) < ref;
    case Function::LessThanEqual:
        return (value & mask) <= ref;
    case Function::Equal:
        return (value & mask) == ref;
    case Function::NotEqual:
        return (value & mask) != ref;
    case Function::GreaterThanEqual:
        return (value & mask) >= ref;
    case Function::GreaterThan:
        return (value & mask) > ref;
    }
    return true;
}

bool LabelWait::Test() const {
    if (memory) {
        return TestValue(*Address());
    }
    return TestValue(NativeWaitRegisters::Read(reg));
}


namespace Typed {

namespace {

struct Pending {
    Kind kind{KindEop};
    Fence fence{};
    LabelWait wait{};
    LabelWrite write{};
    std::array<u32, 4> write_data{};
    DmaCopy dma{};
};

constexpr std::array<const char*, KindCount> KindNames{"eop", "release", "wait", "write", "dma"};

std::mutex g_mutex;
std::unordered_map<u64, Pending> g_pending;
std::atomic<u64> g_expected{0}, g_overwritten{0};

void Add(Pending& p, const std::unordered_map<u64, Pending>::key_type key) {
    std::scoped_lock lock{g_mutex};
    if (!g_pending.emplace(key, p).second) {
        g_pending[key] = p;
        g_overwritten.fetch_add(1, std::memory_order_relaxed);
    }
    g_expected.fetch_add(1, std::memory_order_relaxed);
    if (g_pending.size() > 65536) {
        g_pending.clear(); // packets that were never converted (a sweep that did not see them): forget, do not grow
    }
}

} // namespace

void Expect(u64 packet_address, const Fence& fence, Kind kind) {
    Pending p;
    p.kind = kind;
    p.fence = fence;
    Add(p, packet_address);
}

bool Expected(u64 packet_address) {
    std::scoped_lock lock{g_mutex};
    return g_pending.contains(packet_address);
}

void Expect(u64 packet_address, const LabelWait& wait) {
    Pending p;
    p.kind = KindWait;
    p.wait = wait;
    Add(p, packet_address);
}

void Expect(u64 packet_address, const LabelWrite& write) {
    Pending p;
    p.kind = KindWrite;
    p.write = write;
    p.write.data = nullptr;
    for (u32 i = 0; i < 4 && write.data != nullptr && i < write.bytes / 4; ++i) {
        p.write_data[i] = write.data[i];
    }
    Add(p, packet_address);
}

void Expect(u64 packet_address, const DmaCopy& dma) {
    Pending p;
    p.kind = KindDma;
    p.dma = dma;
    Add(p, packet_address);
}

namespace {
std::array<std::atomic<u64>, KindCount> g_untyped{}, g_converted{}, g_shape_mismatch{};

struct Shape {
    u32 opcode;
    u32 count_min, count_max;
};
constexpr std::array<Shape, KindCount> Shapes{{{0x47, 4, 4}, {0x49, 5, 5}, {0x3c, 5, 5}, {0x37, 3, 6}, {0x50, 5, 5}}};

/// The words of the typed tag for `p` (after the header and the magic): `count` of them
u32 TagPayload(const Pending& p, u32 count, std::array<u32, 6>& w) {
    w = {};
    const auto split = [&](u64 address, u32 flags) {
        w[0] = static_cast<u32>(address);
        w[1] = static_cast<u32>(address >> 32 & 0xffff) | flags;
    };
    switch (p.kind) {
    case KindEop:
    case KindRelease: {
        const Fence& f = p.fence;
        split(f.address, static_cast<u32>(f.source) << 16 | (f.interrupt ? 1u << 20 : 0u) | (f.valid ? 1u << 21 : 0u));
        w[2] = static_cast<u32>(f.value);
        w[3] = static_cast<u32>(f.value >> 32);
        if (p.kind == KindRelease) {
            w[4] = f.gds_index | static_cast<u32>(f.gds_dwords) << 16;
        }
        return count;
    }
    case KindWait: {
        const LabelWait& x = p.wait;
        split(x.memory ? x.address : x.reg, static_cast<u32>(x.function) << 16 | (x.memory ? 1u << 20 : 0u) | (x.pfp ? 1u << 21 : 0u));
        w[2] = x.ref;
        w[3] = x.mask;
        w[4] = x.poll_interval;
        return count;
    }
    case KindWrite:
        w[0] = static_cast<u32>(p.write.address);
        w[1] = static_cast<u32>(p.write.address >> 32);
        for (u32 i = 0; i < 4; ++i) {
            w[2 + i] = p.write_data[i];
        }
        return count;
    case KindDma:
        w[0] = static_cast<u32>(p.dma.destination);
        w[1] = static_cast<u32>(p.dma.destination >> 32);
        w[2] = static_cast<u32>(p.dma.source);
        w[3] = static_cast<u32>(p.dma.source >> 32);
        w[4] = p.dma.bytes;
        return count;
    default:
        return 0;
    }
}
} // namespace

u32 ReadTypedTag(const u32* tag, SyncRecord& out) {
    const u32 magic = tag[1];
    if ((magic & ~0xffu) != TypedTagMagic) {
        return 0;
    }
    const u32 count = (tag[0] >> 16) & 0x3fff;
    const u32 kind = magic & 0xff;
    const u32* w = tag + 2;
    const auto address = [&] { return static_cast<u64>(w[0]) | static_cast<u64>(w[1] & 0xffff) << 32; };
    switch (kind) {
    case KindEop:
    case KindRelease: {
        out.opcode = kind == KindEop ? 0x47 : 0x49;
        out.fence = {};
        out.fence.address = address();
        out.fence.source = static_cast<Source>(w[1] >> 16 & 0xf);
        out.fence.interrupt = (w[1] >> 20 & 1) != 0;
        out.fence.valid = (w[1] >> 21 & 1) != 0;
        out.fence.value = static_cast<u64>(w[2]) | static_cast<u64>(w[3]) << 32;
        if (kind == KindRelease) {
            out.fence.gds_index = static_cast<u16>(w[4] & 0xffff);
            out.fence.gds_dwords = static_cast<u16>(w[4] >> 16);
        }
        return out.opcode;
    }
    case KindWait: {
        out.opcode = 0x3c;
        out.wait = {};
        out.wait.memory = (w[1] >> 20 & 1) != 0;
        out.wait.pfp = (w[1] >> 21 & 1) != 0;
        out.wait.function = static_cast<LabelWait::Function>(w[1] >> 16 & 0xf);
        if (out.wait.memory) {
            out.wait.address = address();
        } else {
            out.wait.reg = w[0];
        }
        out.wait.ref = w[2];
        out.wait.mask = w[3];
        out.wait.poll_interval = w[4];
        return out.opcode;
    }
    case KindWrite:
        out.opcode = 0x37;
        out.write = {};
        out.write.address = static_cast<u64>(w[0]) | static_cast<u64>(w[1]) << 32;
        out.write.bytes = (count - 2) * 4;
        for (u32 i = 0; i < 4; ++i) {
            out.data[i] = w[2 + i];
        }
        out.write.data = out.data.data();
        return out.opcode;
    case KindDma:
        out.opcode = 0x50;
        out.dma = {};
        out.dma.kind = DmaCopy::Kind::MemoryToMemory;
        out.dma.destination = static_cast<u64>(w[0]) | static_cast<u64>(w[1]) << 32;
        out.dma.source = static_cast<u64>(w[2]) | static_cast<u64>(w[3]) << 32;
        out.dma.fill = w[2];
        out.dma.bytes = w[4];
        return out.opcode;
    default:
        return 0;
    }
}

bool Convert(u64 packet_address, Kind kind) {
    if ((packet_address & 3) != 0) {
        return false;
    }
    const u32 header = *reinterpret_cast<const u32*>(packet_address);
    const u32 count = (header >> 16) & 0x3fff;
    const Shape& shape = Shapes[kind];
    if ((header >> 30) != 3 || ((header >> 8) & 0xff) != shape.opcode || count < shape.count_min || count > shape.count_max) {
        return false;
    }
    Pending p;
    {
        std::scoped_lock lock{g_mutex};
        const auto it = g_pending.find(packet_address);
        if (it == g_pending.end() || it->second.kind != kind) {
            return false;
        }
        if (kind == KindWrite && it->second.write.bytes != (count - 2) * 4) {
            g_shape_mismatch[kind].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (kind == KindWait && !it->second.wait.memory) {
            return false;
        }
        p = it->second;
        g_pending.erase(it);
    }
    std::array<u32, 6> words;
    TagPayload(p, count, words);
    u32* tag = reinterpret_cast<u32*>(packet_address);
    for (u32 i = 0; i < count; ++i) {
        tag[2 + i] = words[i];
    }
    tag[1] = TypedTagMagic | kind;
    tag[0] = 0xC0001000u | count << 16;
    g_converted[kind].fetch_add(1, std::memory_order_relaxed);
    return true;
}

void NoteUntyped(Kind kind, u64 packet_address) {
    g_untyped[kind].fetch_add(1, std::memory_order_relaxed);
    char detail[96];
    std::snprintf(detail, sizeof detail, "%s packet at %llx has no typed record and stays raw", KindNames[kind], static_cast<unsigned long long>(packet_address));
    Core::RareEvents::Note(std::string("sync_untyped_") + KindNames[kind], detail);
}

nlohmann::json Report() {
    nlohmann::json j;
    j["note"] = "typed sync records (F1.2): converted_typed: packets turned into typed tags per kind; untyped: packets of that kind that had no typed record and stayed raw (must be 0; also a rare event "
                "sync_untyped_<kind>); expected_never_met: records left whose packet was not converted (end of run tail)";
    std::scoped_lock lock{g_mutex};
    auto& converted = j["converted_typed"] = nlohmann::json::object();
    auto& untyped = j["untyped"] = nlohmann::json::object();
    auto& mismatch = j["shape_mismatch"] = nlohmann::json::object();
    for (u32 k = 0; k < KindCount; ++k) {
        converted[KindNames[k]] = g_converted[k].load();
        untyped[KindNames[k]] = g_untyped[k].load();
        mismatch[KindNames[k]] = g_shape_mismatch[k].load();
    }
    j["expected_total"] = g_expected.load();
    j["expected_overwritten_before_met"] = g_overwritten.load();
    j["expected_never_met"] = g_pending.size();
    return j;
}

} // namespace Typed

} // namespace Core::NativeSync
