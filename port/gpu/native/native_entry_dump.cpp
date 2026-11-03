// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_flip.h"
#include "native/native_log.h"
#include "native/native_draw.h"
#include "native/native_entry_dump.h"
#include "native/native_draw_tag.h"

namespace Core::EntryDump {

namespace {

enum State : u32 { Idle, Armed, Capturing };

constexpr size_t MaxLines = 400000;

std::atomic<u32> state{Idle};
std::atomic<u32> next_span{1};
std::mutex mutex;
std::vector<std::string> lines;
std::map<std::string, u32> by_label;
u32 capture_generation{};
u32 frames_wanted{};
u32 frames_done{};
u64 order{};
bool truncated{};
std::string last_file;
size_t last_lines{};
u64 last_dumps{};

const char* OpcodeName(u32 opcode) {
    switch (opcode) {
    case 0x10: return "NOP";
    case 0x15: return "DISPATCH_DIRECT";
    case 0x37: return "WRITE_DATA";
    case 0x3c: return "WAIT_REG_MEM";
    case 0x3f: return "INDIRECT_BUFFER";
    case 0x46: return "EVENT_WRITE";
    case 0x47: return "EVENT_WRITE_EOP";
    case 0x49: return "RELEASE_MEM";
    case 0x50: return "DMA_DATA";
    case 0x69: return "SET_CONTEXT_REG";
    case 0x76: return "SET_SH_REG";
    default: return "PACKET";
    }
}

/// A raw packet: its opcode and the first words
std::string DescribePacket(const u32* words, u32 opcode, u32 dwords) {
    std::string text = fmt::format("{} opcode={:#x} dwords={}:", OpcodeName(opcode), opcode, dwords);
    for (u32 i = 1; i < dwords && i < 8; ++i) {
        text += fmt::format(" {:08x}", words[i]);
    }
    return text;
}

const char* FunctionName(Core::NativeSync::LabelWait::Function function) {
    using F = Core::NativeSync::LabelWait::Function;
    switch (function) {
    case F::Always: return "always";
    case F::LessThan: return "<";
    case F::LessThanEqual: return "<=";
    case F::Equal: return "==";
    case F::NotEqual: return "!=";
    case F::GreaterThanEqual: return ">=";
    case F::GreaterThan: return ">";
    }
    return "?";
}

/// A sync tag (typed record, or the packet an older tag stands for)
std::string DescribeSync(const Core::NativeSync::SyncRecord& record) {
    switch (record.opcode) {
    case 0x47:
    case 0x49:
        return fmt::format("{} address={:#x} value={:#x} source={} interrupt={}", record.opcode == 0x47 ? "END_OF_PIPE" : "RELEASE_MEM", record.fence.address, record.fence.value,
                           static_cast<int>(record.fence.source), record.fence.interrupt);
    case 0x3c:
        if (record.wait.memory) {
            return fmt::format("WAIT_REG_MEM address={:#x} {} {:#x} mask={:#x}", record.wait.address, FunctionName(record.wait.function), record.wait.ref, record.wait.mask);
        }
        return fmt::format("WAIT_REG_MEM register={:#x} {} {:#x}", record.wait.reg, FunctionName(record.wait.function), record.wait.ref);
    case 0x37:
        return fmt::format("WRITE_DATA address={:#x} bytes={} first={:#x}", record.write.address, record.write.bytes, record.write.bytes >= 4 ? record.write.data[0] : 0u);
    case 0x50:
        return fmt::format("DMA_DATA destination={:#x} source={:#x} bytes={}", record.dma.destination, record.dma.source, record.dma.bytes);
    default:
        return DescribePacket(record.words.data(), (record.words[0] >> 8) & 0xff, (record.words[0] >> 16 & 0x3fff) + 2);
    }
}

void CountLabel(const std::string& text) {
    const size_t first = text.find(' ');
    if (text.starts_with("ENTRY SIGNAL")) {
        ++by_label[text.substr(0, text.find(' ', text.find(' ', first + 1) + 1))];
    } else if (first != std::string::npos && (text.starts_with("ENTRY") || text.starts_with("SIGNAL"))) {
        ++by_label[text.substr(0, text.find(' ', first + 1))];
    } else {
        ++by_label[text.substr(0, first)];
    }
}

void Finish() {
    state.store(Idle, std::memory_order_release);
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "entry_dump.txt";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        TLG_LOG(HOOK, Error, "entry dump: cannot write {}", path.string());
        lines.clear();
        by_label.clear();
        return;
    }
    out << fmt::format("entry dump: {} frame(s), {} lines{}\n", frames_done, lines.size(), truncated ? " (truncated)" : "");
    out << "columns: walk order, queue (0 graphics, 1+n async compute n), span number, offset in the span (bytes), what\n";
    out << "walk order is the order the command processor walks the buffers: the queues interleave\n";
    out << "ENTRY mode: record = executed from the record, no game packets; at-draw-packet = executed when the draw packet arrives; "
           "packets-follow = the game writes the packets too\n";
    out << "summary:\n";
    for (const auto& [label, count] : by_label) {
        out << fmt::format("  {:8} {}\n", count, label);
    }
    out << "----\n";
    for (const auto& line : lines) {
        out << line << '\n';
    }
    last_file = path.string();
    last_lines = lines.size();
    ++last_dumps;
    LOG_INFO(Core, "entry dump: {} lines written to {}", lines.size(), last_file);
    lines.clear();
    lines.shrink_to_fit();
    by_label.clear();
}

} // namespace

void Request(u32 frames) {
    std::scoped_lock lock{mutex};
    lines.clear();
    by_label.clear();
    ++capture_generation;
    frames_wanted = frames == 0 ? 1 : frames;
    frames_done = 0;
    order = 0;
    truncated = false;
    state.store(Armed, std::memory_order_release);
}

nlohmann::json Status() {
    std::scoped_lock lock{mutex};
    nlohmann::json j;
    const u32 current = state.load();
    j["state"] = current == Idle ? "idle" : current == Armed ? "armed, waiting for a flip" : "capturing";
    j["lines_so_far"] = lines.size();
    j["dumps"] = last_dumps;
    j["last_file"] = last_file;
    j["last_lines"] = last_lines;
    return j;
}

Span::Span(u32 queue_id, const u32* span_begin, u32 span_dwords)
    : tracked{state.load(std::memory_order_relaxed) != Idle}, queue{queue_id}, dwords{span_dwords}, id{next_span.fetch_add(1, std::memory_order_relaxed)},
      begin{reinterpret_cast<u64>(span_begin)} {}

void Span::Packet(const void* packet, u64 address, bool show) {
    if (!tracked || !show) {
        return;
    }
    const u64 packet_address = address != 0 ? address : reinterpret_cast<u64>(packet);
    if (state.load(std::memory_order_acquire) == Idle) {
        tracked = false;
        return;
    }
    const u32* header = static_cast<const u32*>(packet);
    const u32 opcode = (header[0] >> 8) & 0xff;
    const u32 packet_dwords = ((header[0] >> 16) & 0x3fff) + 2;
    std::string text;
    bool flip = false;
    if (opcode == 0x10) {
        const u32 count = (header[0] >> 16) & 0x3fff;
        if (count == 0) {
            return;
        }
        if (header[1] == VideoCore::NativeTagMagic && count + 1 >= VideoCore::NativeTagPayloadDwords) {
            if (state.load(std::memory_order_acquire) == Armed) {
                return;
            }
            text = Core::NativeDraw::DescribeEntry(header);
        } else if (Core::NativeSync::SyncRecord record; Core::NativeDraw::ReadSyncTag(header, record, false) != 0) {
            if (state.load(std::memory_order_acquire) == Armed) {
                return;
            }
            text = "ENTRY " + DescribeSync(record);
        } else if (header[1] == Core::NativeFlip::TagMagic) {
            flip = true;
            text = fmt::format("FLIP patched flip request");
        } else {
            return;
        }
    } else {
        if (state.load(std::memory_order_acquire) == Armed) {
            return;
        }
        text = DescribePacket(header, opcode, packet_dwords);
    }
    std::scoped_lock lock{mutex};
    const u32 current = state.load(std::memory_order_relaxed);
    if (current == Idle) {
        return;
    }
    if (flip && queue == 0 && current == Armed) {
        state.store(Capturing, std::memory_order_release);
        lines.emplace_back("---- frame start (flip seen) ----");
        return;
    }
    if (current != Capturing) {
        return;
    }
    if (generation != capture_generation) {
        generation = capture_generation;
        lines.push_back(fmt::format("        q{} s{:<5} span begins: address {:#x}, {} dwords", queue, id, begin, dwords));
    }
    if (lines.size() >= MaxLines) {
        truncated = true;
        Finish();
        return;
    }
    CountLabel(text);
    lines.push_back(fmt::format("{:7} q{} s{:<5} +{:06x}  {}", order++, queue, id, packet_address - begin, text));
    if (flip && queue == 0) {
        lines.emplace_back("---- frame end (flip) ----");
        if (++frames_done >= frames_wanted) {
            Finish();
        }
    }
}

} // namespace Core::EntryDump
