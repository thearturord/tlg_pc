// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_sync.h"

#include "native/native_debug_server.h"
#include "native/native_write_watch.h"
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
        const bool passed = TestValue(*Address());
        if (passed) {
            // a label the CPU may have written after its data: data written before it matters from here (L3 write watch)
            Core::WriteWatch::base_epoch.fetch_add(1, std::memory_order_acq_rel);
            Core::WriteWatch::epoch.fetch_add(1, std::memory_order_acq_rel);
        }
        return passed;
    }
    return TestValue(regs[reg]);
}

} // namespace Core::NativeSync
