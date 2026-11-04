// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <array>
#include <span>

#include "common/types.h"

/// Synchronization records: what a fence, a wait, a label write and a CP DMA copy do, built by the hooks of the engine functions that write them (typed producers).
///
/// What the engine does with them (read in Ghidra, 2026-10-30):
///  - `FenceEop` (0x2480d60): takes 8 bytes of the frame ring, stores 0x11 there, writes an end-of-pipe packet that stores 0x10 there when the GPU has finished all earlier
///    work, and returns the address; the CPU polls it until it reads 0x10. That is a `Fence` with an immediate 32 bit value.
///  - `JsSwapChain_SubmitAndFlip` (0x1c4dfa0): an end-of-pipe write of the frame's fence value before the submit and one with the interrupt bit after it, then the
///    flip through the Gnm driver.
///  - the barrier units around transitions (end-of-pipe, wait on the label, cache acquire) only order the GPU against itself and are taken out before they get here.
namespace Core::NativeSync {

enum class Source : u8 {
    None,         ///< no data is written (an interrupt only)
    Immediate32,  ///< the low 32 bits of `value`
    Immediate64,  ///< `value`
    GpuClock,     ///< the GPU clock at the time of the signal
    PerfCounter,  ///< the GPU performance counter at the time of the signal
    GdsCopy,      ///< release memory only: `gds_dwords` dwords of GDS from byte `gds_index` to `address`
};

/// An end-of-pipe signal: graphics end-of-pipe event write or compute release memory.
struct Fence {
    u64 address{};
    u64 value{};
    Source source{Source::None};
    bool interrupt{};     ///< raise the interrupt of the queue after the write
    bool valid{true};     ///< false: a combination of fields the game never used (not executed, reported as a rare event)
    u16 gds_index{};
    u16 gds_dwords{};

    [[nodiscard]] u32 Bytes() const noexcept {
        return source == Source::Immediate32 ? 4u : source == Source::Immediate64 ? 8u : 0u;
    }

    /// Runs the signal: `write(void* address, u64 data, u32 bytes)` for the data, then `irq()` for the interrupt, `gds_to_mem(address, gds_index, dwords)` for a GDS copy.
    template <typename Write, typename Irq, typename Gds>
    void Signal(Write&& write, Irq&& irq, Gds&& gds_to_mem) const {
        switch (source) {
        case Source::None:
            break;
        case Source::Immediate32:
            write(reinterpret_cast<void*>(address), value & 0xffffffffull, 4u);
            break;
        case Source::Immediate64:
            write(reinterpret_cast<void*>(address), value, 8u);
            break;
        case Source::GpuClock:
            write(reinterpret_cast<void*>(address), GpuClock64(), 8u);
            break;
        case Source::PerfCounter:
            write(reinterpret_cast<void*>(address), GpuPerfCounter(), 8u);
            break;
        case Source::GdsCopy:
            gds_to_mem(static_cast<VAddr>(address), gds_index, gds_dwords);
            break;
        }
        if (interrupt) {
            irq();
        }
    }
    template <typename Write, typename Irq>
    void Signal(Write&& write, Irq&& irq) const {
        Signal(write, irq, [](VAddr, u16, u16) {});
    }

    static u64 GpuClock64();
    static u64 GpuPerfCounter();
};

/// A wait of the command processor on a memory label (or a register): continue when `(value & mask) <function> reference`.
struct LabelWait {
    enum class Function : u8 { Always, LessThan, LessThanEqual, Equal, NotEqual, GreaterThanEqual, GreaterThan };
    bool memory{true}; ///< false: a register (`reg`)
    bool pfp{};        ///< the prefetch parser (the game only uses the micro engine)
    Function function{Function::Always};
    u32 reg{};
    u64 address{};
    u32 ref{};
    u32 mask{};
    u32 poll_interval{};

    template <typename T = u32*>
    [[nodiscard]] T Address() const {
        return reinterpret_cast<T>(address);
    }
    [[nodiscard]] u32 Reg() const {
        return reg;
    }
    [[nodiscard]] bool TestValue(u32 value) const;
    /// Tests the label (or the register, see native_wait_registers.h)
    [[nodiscard]] bool Test() const;
};

/// A CPU style write of one to four dwords to a label, in order with the GPU work.
struct LabelWrite {
    u64 address{};
    const u32* data{}; ///< points into the words the record was decoded from
    u32 bytes{};
};

/// A CP DMA copy or fill.
struct DmaCopy {
    enum class Kind : u8 { FillGds, MemoryToGds, FillGuest, GdsToMemory, MemoryToMemory, Ignored, Unknown };
    Kind kind{Kind::Unknown};
    u64 destination{}; ///< a guest address, or a GDS byte offset
    u64 source{};      ///< a guest address, or a GDS byte offset
    u32 fill{};        ///< the fill value
    u32 bytes{};
};

/// A synchronization record read back from a sync tag (`ReadSyncTag`): `opcode` says which member is valid (0x47 end-of-pipe, 0x49 release, 0x3c wait, 0x37 write, 0x50 dma; 0x46 and 0x69 only
/// carry their raw words in `words`).
struct SyncRecord {
    u32 opcode{};
    Fence fence;
    LabelWait wait;
    LabelWrite write;
    DmaCopy dma;
    std::array<u32, 4> data{}; ///< the words `write.data` points to
    std::array<u32, 8> words{}; ///< event and query tags: the packet they stand for
};
/// Second dword of a typed sync tag: `TypedTagMagic | kind` (Typed::Kind)
constexpr u32 TypedTagMagic = 0x4E445400u; // "NDT" + kind

/// The result address of an occlusion query dump (PIXEL_PIPE_STAT_DUMP): words 2 and 3 of the event write
[[nodiscard]] u64 DecodeEventAddress(const u32* packet);

/// F1.2 typed sync records: a producer (a hook on the function that writes the packet) calls `Expect` with the address of the packet it wrote and the record it built from the engine's own
/// arguments; `ConvertSyncPacket` calls `Convert`, which rewrites the packet in place as a typed sync tag that carries the record. A packet with no record is left raw and counted (`NoteUntyped`).
/// Report: /stats native_backend.device.sync_typed.

namespace Typed {

enum Kind : u32 { KindEop, KindRelease, KindWait, KindWrite, KindDma, KindCount };

/// F1.2: the packet at `packet_address` has the shape of `kind` (opcode and size from its header) and a typed record waits for it: the packet is rewritten in place as a typed sync tag
/// (header = a no-op of the same size, magic `TypedTagMagic | kind`, the record's words). False: no record, or another
/// shape; the packet is left as it is.
[[nodiscard]] bool Convert(u64 packet_address, Kind kind);
/// Reads a typed sync tag (`TypedTagMagic`) into `out`; returns its opcode-like code (0x47, 0x49, 0x3c, 0x37, 0x50) or 0 when `tag` is not one.
[[nodiscard]] u32 ReadTypedTag(const u32* tag, SyncRecord& out);

void Expect(u64 packet_address, const Fence& fence, Kind kind = KindEop);
void Expect(u64 packet_address, const LabelWait& wait);
void Expect(u64 packet_address, const LabelWrite& write);
void Expect(u64 packet_address, const DmaCopy& dma);

/// True when a typed record is waiting for the packet at `packet_address` (census use)
[[nodiscard]] bool Expected(u64 packet_address);
/// A packet of `kind` reached the converter with no typed record: counted and noted as a rare event (it stays raw)
void NoteUntyped(Kind kind, u64 packet_address);

[[nodiscard]] nlohmann::json Report();

} // namespace Typed

} // namespace Core::NativeSync
