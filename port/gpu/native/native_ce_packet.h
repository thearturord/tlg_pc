// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The constant engine buffer of a submission is the one place where the game's command stream is read as packets (E0 census: write const RAM, dump const RAM, the
// counter packets, a NOP, an indirect buffer). This is the plain decode of those type 3 packets by bit extraction (G6: replaces the PS4 header `amdgpu/pm4_cmds.h`).
// Layouts as the console documents them: header = type (31:30) | count (29:16, body dwords - 1) | opcode (15:8) | shader type / predicate.

#include <cstddef>
#include "common/types.h"

namespace Core::NativeCePacket {

enum Opcode : u32 {
    Nop = 0x10,
    IndirectBufferConst = 0x33,
    WriteConstRam = 0x81,
    DumpConstRam = 0x83,
    IncrementCeCounter = 0x84,
    WaitOnDeCounterDiff = 0x88,
};

/// The size in bytes of the group count record a dispatch with indirect arguments reads (three dwords).
constexpr u32 DispatchIndirectArgsBytes = 3 * sizeof(u32);

/// The size in bytes of the argument record an indexed draw with indirect arguments reads (index count, instance count, start index, base vertex, start instance).
constexpr u32 DrawIndexedIndirectArgsBytes = 5 * sizeof(u32);

struct Header {
    u32 type;    ///< 3 for the packets this decodes
    u32 opcode;
    u32 dwords;  ///< the packet's length including the header
};

[[nodiscard]] constexpr Header DecodeHeader(const u32* packet) noexcept {
    const u32 raw = packet[0];
    return {raw >> 30, (raw >> 8) & 0xff, ((((raw >> 16) & 0x3fff) + 1) & 0x3fff) + 1};
}

/// Write const RAM: `size` bytes at `data` go to the constant RAM at byte offset `offset`.
struct WriteConst {
    u32 offset;
    u32 size;
    const u32* data;
};
[[nodiscard]] constexpr WriteConst DecodeWriteConst(const u32* packet) noexcept {
    return {packet[1] & 0xffff, ((packet[0] >> 16) & 0x3fff) << 2, packet + 2};
}

/// Dump const RAM: `size` bytes from the constant RAM at byte offset `offset` to guest memory `address`.
struct DumpConst {
    u32 offset;
    u32 size;
    u64 address;
};
[[nodiscard]] constexpr DumpConst DecodeDumpConst(const u32* packet) noexcept {
    return {packet[1] & 0xffff, (packet[2] & 0x7fff) << 2, (u64{packet[4]} << 32) | packet[3]};
}

/// Indirect buffer (constant engine): `dwords` dwords at guest memory `address`.
struct IndirectBuffer {
    u64 address;
    u32 dwords;
};
[[nodiscard]] constexpr IndirectBuffer DecodeIndirectBuffer(const u32* packet) noexcept {
    return {(u64{packet[2] & 0xffff} << 32) | packet[1], packet[3] & 0xfffff};
}

} // namespace Core::NativeCePacket
