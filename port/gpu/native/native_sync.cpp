// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_sync.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <intrin.h>
#include <string>
#include <thread>
#include <unordered_map>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"
#include "video_core/amdgpu/pm4_cmds.h"

namespace Core::NativeSync {

using namespace AmdGpu;

u64 Fence::GpuClock64() {
    return ::AmdGpu::GetGpuClock64();
}

u64 Fence::GpuPerfCounter() {
    return ::AmdGpu::GetGpuPerfCounter();
}

namespace {

Source SourceOf(DataSelect select, bool& valid) {
    switch (select) {
    case DataSelect::None:
        return Source::None;
    case DataSelect::Data32Low:
        return Source::Immediate32;
    case DataSelect::Data64:
        return Source::Immediate64;
    case DataSelect::GpuClock64:
        return Source::GpuClock;
    case DataSelect::PerfCounter:
        return Source::PerfCounter;
    case DataSelect::GdsMemStore:
        return Source::GdsCopy;
    }
    valid = false;
    return Source::None;
}

void NoteInvalid(const char* what, u32 first, u32 second) {
    char detail[96];
    std::snprintf(detail, sizeof detail, "%s with fields the game never used: %08x %08x", what, first, second);
    Core::RareEvents::Note(std::string("sync_invalid_") + what, detail);
}

} // namespace

Fence DecodeEop(const u32* packet) {
    const auto& eop = *reinterpret_cast<const PM4CmdEventWriteEop*>(packet);
    Fence fence;
    fence.address = reinterpret_cast<u64>(eop.Address<u8>());
    fence.value = eop.DataQWord();
    fence.source = SourceOf(eop.data_sel.Value(), fence.valid);
    if (fence.source == Source::GdsCopy) {
        fence.valid = false; // an end-of-pipe event write has no GDS copy
    }
    switch (eop.int_sel.Value()) {
    case InterruptSelect::None:
        break;
    case InterruptSelect::IrqOnly:
        // the interrupt alone: no data is written with it
        fence.interrupt = true;
        fence.valid &= fence.source == Source::None;
        break;
    case InterruptSelect::IrqWhenWriteConfirm:
        fence.interrupt = true;
        break;
    default:
        fence.valid = false;
        break;
    }
    if (!fence.valid) {
        NoteInvalid("eop", packet[1], packet[3]);
    }
    return fence;
}

Fence DecodeRelease(const u32* packet) {
    const auto& release = *reinterpret_cast<const PM4CmdReleaseMem*>(packet);
    Fence fence;
    fence.address = release.Address<u64>();
    fence.value = release.DataQWord();
    fence.source = SourceOf(release.data_sel.Value(), fence.valid);
    fence.valid &= fence.source != Source::None; // a release memory always names its data
    if (fence.source == Source::GdsCopy) {
        fence.gds_index = release.gds_index;
        fence.gds_dwords = release.num_dw;
    }
    switch (release.int_sel.Value()) {
    case InterruptSelect::None:
        break;
    case InterruptSelect::IrqUndocumented:
    case InterruptSelect::IrqWhenWriteConfirm:
        fence.interrupt = true;
        break;
    default:
        fence.valid = false;
        break;
    }
    if (!fence.valid) {
        NoteInvalid("release", packet[1], packet[2]);
    }
    return fence;
}

LabelWait DecodeWait(const u32* packet) {
    const auto& wait = *reinterpret_cast<const PM4CmdWaitRegMem*>(packet);
    LabelWait out;
    out.memory = wait.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
    out.pfp = wait.engine.Value() == PM4CmdWaitRegMem::Engine::Pfp;
    out.function = static_cast<LabelWait::Function>(std::min<u32>(static_cast<u32>(wait.function.Value()), 6u));
    out.reg = wait.Reg();
    out.address = wait.Address<u64>();
    out.ref = wait.ref;
    out.mask = wait.mask;
    out.poll_interval = wait.poll_interval;
    return out;
}

LabelWrite DecodeWrite(const u32* packet) {
    const auto& write = *reinterpret_cast<const PM4CmdWriteData*>(packet);
    LabelWrite out;
    out.address = write.Address<u64>();
    out.data = write.data;
    out.bytes = (write.header.count.Value() - 2) * 4;
    return out;
}

DmaCopy DecodeDma(const u32* packet) {
    const auto& dma = *reinterpret_cast<const PM4DmaData*>(packet);
    DmaCopy out;
    out.bytes = dma.NumBytes();
    out.fill = dma.data;
    if (dma.dst_addr_lo == 0x3022C || dma.command.das != 0) {
        out.kind = DmaCopy::Kind::Ignored; // a write to a register the game does not need, or a destination the console does not map
        return out;
    }
    const bool to_memory = dma.dst_sel == DmaDataDst::Memory || dma.dst_sel == DmaDataDst::MemoryUsingL2;
    const bool from_memory = dma.src_sel == DmaDataSrc::Memory || dma.src_sel == DmaDataSrc::MemoryUsingL2;
    if (dma.src_sel == DmaDataSrc::Data && dma.dst_sel == DmaDataDst::Gds) {
        out.kind = DmaCopy::Kind::FillGds;
        out.destination = dma.dst_addr_lo;
    } else if (from_memory && dma.dst_sel == DmaDataDst::Gds) {
        out.kind = DmaCopy::Kind::MemoryToGds;
        out.destination = dma.dst_addr_lo;
        out.source = dma.SrcAddress<u64>();
    } else if (dma.src_sel == DmaDataSrc::Data && to_memory) {
        out.kind = DmaCopy::Kind::FillGuest;
        out.destination = dma.DstAddress<u64>();
    } else if (dma.src_sel == DmaDataSrc::Gds && to_memory) {
        out.kind = DmaCopy::Kind::GdsToMemory;
        out.destination = dma.DstAddress<u64>();
        out.source = dma.src_addr_lo;
    } else if (from_memory && to_memory) {
        out.kind = DmaCopy::Kind::MemoryToMemory;
        out.destination = dma.DstAddress<u64>();
        out.source = dma.SrcAddress<u64>();
    }
    return out;
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

bool LabelWait::Test(std::span<const u32> regs) const {
    if (memory) {
        return TestValue(*Address());
    }
    return TestValue(regs[reg]);
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

struct KindCounters {
    std::atomic<u64> evaluated{0}, differing{0}, without_typed_source{0};
    std::map<std::string, u64> fields; ///< field -> evaluations in which it differed (under the mutex)
};

std::mutex g_mutex;
std::unordered_map<u64, Pending> g_pending;
std::array<KindCounters, KindCount> g_counters;
std::atomic<u64> g_expected{0}, g_overwritten{0};
std::array<std::string, 6> g_examples;
u32 g_example_count = 0;

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

struct Diff {
    std::string fields;
    void Check(const char* name, u64 typed, u64 decoded) {
        if (typed != decoded) {
            fields += fmt::format("{}(typed {:#x}, decoded {:#x}) ", name, typed, decoded);
        }
    }
};

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_SYNC_TYPED");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void Expect(u64 packet_address, const Fence& fence, Kind kind) {
    if (!On()) {
        return;
    }
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
    if (!On()) {
        return;
    }
    Pending p;
    p.kind = KindWait;
    p.wait = wait;
    Add(p, packet_address);
}

void Expect(u64 packet_address, const LabelWrite& write) {
    if (!On()) {
        return;
    }
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
    if (!On()) {
        return;
    }
    Pending p;
    p.kind = KindDma;
    p.dma = dma;
    Add(p, packet_address);
}

std::atomic<u64> g_image_base{0}, g_image_size{0};
std::mutex g_stack_mutex;
std::map<std::string, u64> g_stack_tally;

void SetImage(u64 base, u64 size) {
    g_image_base.store(base);
    g_image_size.store(size);
}

/// The first game return addresses (a call instruction right before them) found on the stack of the thread, as Ghidra offsets
void TallyStack(Kind kind, const u32* packet) {
    static const bool on = [] {
        const char* e = std::getenv("TLG_SYNC_STACKS");
        return e != nullptr && *e == '1';
    }();
    const u64 base = g_image_base.load();
    if (!on || base == 0) {
        return;
    }
    const u64 size = g_image_size.load();
    u64 probe = 0;
    const u64* sp = &probe;
    std::string key = fmt::format("{} {:08x} ", KindNames[kind], packet[0]);
    int found = 0;
    const u64* stack_top = reinterpret_cast<const u64*>(__readgsqword(8));
    for (const u64* q = sp; q < stack_top && found < 3; ++q) {
        const u64 v = *q;
        if (v > base + 8 && v < base + size) {
            const u8* code = reinterpret_cast<const u8*>(v);
            if (code[-5] == 0xe8 || code[-2] == 0xff || code[-3] == 0xff || code[-6] == 0xff) {
                key += fmt::format(" {:x}", v - base + 0x1000000);
                ++found;
            }
        }
    }
    std::scoped_lock lock{g_stack_mutex};
    if (g_stack_tally.size() < 600 || g_stack_tally.contains(key)) {
        ++g_stack_tally[key];
    }
}

void Met(u64 packet_address, Kind kind, const u32* packet) {
    if (!On()) {
        return;
    }
    Pending expected;
    bool found = false;
    {
        std::scoped_lock lock{g_mutex};
        if (const auto it = g_pending.find(packet_address); it != g_pending.end()) {
            expected = it->second;
            g_pending.erase(it);
            found = true;
        }
    }
    KindCounters& counters = g_counters[kind];
    if (!found) {
        counters.without_typed_source.fetch_add(1, std::memory_order_relaxed);
        TallyStack(kind, packet);
        return;
    }
    counters.evaluated.fetch_add(1, std::memory_order_relaxed);
    Diff diff;
    switch (kind) {
    case KindEop:
    case KindRelease: {
        const Fence decoded = kind == KindEop ? DecodeEop(packet) : DecodeRelease(packet);
        const Fence& typed = expected.fence;
        diff.Check("address", typed.address, decoded.address);
        diff.Check("source", static_cast<u64>(typed.source), static_cast<u64>(decoded.source));
        const u64 mask = typed.source == Source::Immediate32 ? 0xffffffffull : ~0ull;
        if (typed.source == Source::Immediate32 || typed.source == Source::Immediate64) {
            diff.Check("value", typed.value & mask, decoded.value & mask);
        }
        diff.Check("interrupt", typed.interrupt, decoded.interrupt);
        diff.Check("valid", typed.valid, decoded.valid);
        diff.Check("gds_index", typed.gds_index, decoded.gds_index);
        diff.Check("gds_dwords", typed.gds_dwords, decoded.gds_dwords);
        break;
    }
    case KindWait: {
        const LabelWait decoded = DecodeWait(packet);
        const LabelWait& typed = expected.wait;
        diff.Check("memory", typed.memory, decoded.memory);
        diff.Check("pfp", typed.pfp, decoded.pfp);
        diff.Check("function", static_cast<u64>(typed.function), static_cast<u64>(decoded.function));
        if (!decoded.memory) { // the register of a memory wait is address bits of the packet, read by nothing
            diff.Check("reg", typed.reg, decoded.reg);
        }
        diff.Check("address", typed.address, decoded.address);
        diff.Check("ref", typed.ref, decoded.ref);
        diff.Check("mask", typed.mask, decoded.mask);
        diff.Check("poll_interval", typed.poll_interval, decoded.poll_interval);
        break;
    }
    case KindWrite: {
        const LabelWrite decoded = DecodeWrite(packet);
        diff.Check("address", expected.write.address, decoded.address);
        diff.Check("bytes", expected.write.bytes, decoded.bytes);
        for (u32 i = 0; i < 4 && i < decoded.bytes / 4; ++i) {
            diff.Check("data", expected.write_data[i], decoded.data[i]);
        }
        break;
    }
    case KindDma: {
        const DmaCopy decoded = DecodeDma(packet);
        diff.Check("kind", static_cast<u64>(expected.dma.kind), static_cast<u64>(decoded.kind));
        diff.Check("destination", expected.dma.destination, decoded.destination);
        diff.Check("source", expected.dma.source, decoded.source);
        diff.Check("fill", expected.dma.fill, decoded.fill);
        diff.Check("bytes", expected.dma.bytes, decoded.bytes);
        break;
    }
    default:
        break;
    }
    if (!diff.fields.empty()) {
        counters.differing.fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lock{g_mutex};
        // the first words of each differing field name, counted once per evaluation
        size_t from = 0;
        while (from < diff.fields.size()) {
            const size_t paren = diff.fields.find('(', from);
            if (paren == std::string::npos) {
                break;
            }
            ++counters.fields[diff.fields.substr(from, paren - from)];
            const size_t end = diff.fields.find(") ", paren);
            from = end == std::string::npos ? diff.fields.size() : end + 2;
        }
        if (g_example_count < g_examples.size()) {
            g_examples[g_example_count++] = fmt::format("{} at {:#x}: {}", KindNames[kind], packet_address, diff.fields);
        }
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["enabled"] = On();
    j["note"] = "TLG_SYNC_TYPED=1: typed records built by the engine functions' hooks against the decoder's records from the packet words, per kind; without_typed_source: packets of that kind met "
                "with no producer (the producers still to be hooked); expected_never_met: expectations left (their packet was not converted)";
    std::scoped_lock lock{g_mutex};
    auto& kinds = j["kinds"] = nlohmann::json::object();
    for (u32 k = 0; k < KindCount; ++k) {
        const KindCounters& c = g_counters[k];
        nlohmann::json fields = nlohmann::json::object();
        for (const auto& [name, n] : c.fields) {
            fields[name] = n;
        }
        kinds[KindNames[k]] = {{"evaluated", c.evaluated.load()}, {"differing", c.differing.load()}, {"without_typed_source", c.without_typed_source.load()}, {"differing_fields", std::move(fields)}};
    }
    {
        std::scoped_lock stack_lock{g_stack_mutex};
        if (!g_stack_tally.empty()) {
            j["untyped_stacks"] = g_stack_tally;
        }
    }
    j["expected_total"] = g_expected.load();
    j["expected_overwritten_before_met"] = g_overwritten.load();
    j["expected_never_met"] = g_pending.size();
    nlohmann::json examples = nlohmann::json::array();
    for (u32 i = 0; i < g_example_count; ++i) {
        examples.push_back(g_examples[i]);
    }
    j["first_differences"] = std::move(examples);
    return j;
}

} // namespace Typed

} // namespace Core::NativeSync
