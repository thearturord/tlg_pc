// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <map>
#include <vector>

#include "native/native_census.h"
#include "native/native_occlusion.h"

namespace Core::NativeCensus {

namespace {

constexpr u32 NumKinds = 5;
constexpr const char* KindNames[NumKinds] = {"tag", "sync", "flip", "state_buffer", "packet"};
constexpr const char* QueueNames[2] = {"graphics", "async_compute"};
constexpr const char* ImageNames[NumImageEvents] = {"refresh_image", "detile_image", "resolve_overlap", "resolve_depth_overlap", "expand_image"};

std::array<std::array<std::atomic<u64>, NumKinds>, 2> entries{};
std::array<std::array<std::atomic<u64>, 256>, 2> packet_opcodes{};
// register packets: [0] SH (0x76), [1] context (0x69), [2] uconfig (0x79); by first register offset, 16 bits
std::array<std::array<std::atomic<u32>, 65536>, 3> first_registers{};
std::array<std::atomic<u64>, 3> register_packets{}, register_values{};
std::atomic<u64> ce_submissions{}, ce_dwords{};
std::array<std::atomic<u64>, 64 * 16> event_kinds{}; // EVENT_WRITE by event_type (6 bits) and event_index (4 bits)
std::array<std::atomic<u64>, 256> other_packets{};
struct SkipKey { u32 kind; u64 hash; bool operator<(const SkipKey& o) const { return kind != o.kind ? kind < o.kind : hash < o.hash; } };
std::map<SkipKey, std::pair<u64, u64>> skipped_in_clear; // (count, last size argument)
std::mutex skip_mutex;
std::array<std::atomic<u64>, 256> ce_opcodes{}, ce_opcode_dwords{};
std::mutex sample_mutex;
std::array<std::vector<std::string>, 256> samples; // the first words of the first packets of each opcode
std::array<std::atomic<u64>, NumImageEvents> image_events{}, image_bytes{};

u32 RegisterSlot(u32 opcode) {
    return opcode == 0x76 ? 0u : opcode == 0x69 ? 1u : opcode == 0x79 ? 2u : 3u;
}

} // namespace

void NoteEntry(u32 queue, u32 kind, u32 opcode) {
    const u32 q = queue == 0 ? 0u : 1u;
    if (kind < NumKinds) {
        entries[q][kind].fetch_add(1, std::memory_order_relaxed);
    }
    if (kind == 4) {
        packet_opcodes[q][opcode & 0xff].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteRegisterPacket(u32 opcode, u32 first_register, u32 count) {
    const u32 slot = RegisterSlot(opcode);
    if (slot > 2) {
        return;
    }
    first_registers[slot][first_register & 0xffff].fetch_add(1, std::memory_order_relaxed);
    register_packets[slot].fetch_add(1, std::memory_order_relaxed);
    register_values[slot].fetch_add(count, std::memory_order_relaxed);
}

void NoteRawPacket(u32 opcode, const u32* words, u32 dwords) {
    if (opcode == 0x46 && dwords >= 2) {
        event_kinds[(words[1] & 0x3f) * 16 + (words[1] >> 8 & 0xf)].fetch_add(1, std::memory_order_relaxed);
    }
    std::scoped_lock lk{sample_mutex};
    auto& list = samples[opcode & 0xff];
    if (list.size() < 6) {
        std::string text;
        char buf[16];
        for (u32 i = 0; i < dwords && i < 10; ++i) {
            std::snprintf(buf, sizeof buf, "%08x ", words[i]);
            text += buf;
        }
        list.push_back(text);
    }
}

void NoteOtherPacket(u32 opcode) {
    other_packets[opcode & 0xff].fetch_add(1, std::memory_order_relaxed);
}

void NoteSkippedInClear(u32 kind, u64 program_hash, u64 size) {
    std::scoped_lock lk{skip_mutex};
    auto& entry = skipped_in_clear[{kind, program_hash}];
    ++entry.first;
    entry.second = size;
}

void NoteCePacket(u32 opcode, u32 dwords) {
    ce_opcodes[opcode & 0xff].fetch_add(1, std::memory_order_relaxed);
    ce_opcode_dwords[opcode & 0xff].fetch_add(dwords, std::memory_order_relaxed);
}

void NoteCeSubmission(u32 dwords) {
    ce_submissions.fetch_add(1, std::memory_order_relaxed);
    ce_dwords.fetch_add(dwords, std::memory_order_relaxed);
}

void NoteImage(ImageEvent what, u64 bytes) {
    image_events[what].fetch_add(1, std::memory_order_relaxed);
    image_bytes[what].fetch_add(bytes, std::memory_order_relaxed);
}

nlohmann::json Report() {
    using json = nlohmann::json;
    json out;
    out["note"] = "E0 census, cumulative since start (read only). entries: what the command processor walks per submission; packet_opcodes: raw PM4 packets that execute from the game's own stream";
    json by_queue = json::object();
    for (u32 q = 0; q < 2; ++q) {
        json kinds = json::object();
        u64 total = 0;
        for (u32 k = 0; k < NumKinds; ++k) {
            const u64 n = entries[q][k].load();
            kinds[KindNames[k]] = n;
            total += n;
        }
        json opcodes = json::object();
        for (u32 o = 0; o < 256; ++o) {
            if (const u64 n = packet_opcodes[q][o].load()) {
                char key[8];
                std::snprintf(key, sizeof key, "0x%02x", o);
                opcodes[key] = n;
            }
        }
        by_queue[QueueNames[q]] = {{"entries", total}, {"by_kind", kinds}, {"packet_opcodes", opcodes}};
    }
    out["queues"] = by_queue;
    json regs = json::object();
    const char* names[3] = {"set_sh_reg_0x76", "set_context_reg_0x69", "set_uconfig_reg_0x79"};
    constexpr u32 bases[3] = {0x2c00, 0xa000, 0xc000};
    for (u32 s = 0; s < 3; ++s) {
        std::vector<std::pair<u32, u32>> top;
        for (u32 r = 0; r < 65536; ++r) {
            if (const u32 n = first_registers[s][r].load()) {
                top.emplace_back(n, r);
            }
        }
        std::sort(top.begin(), top.end(), std::greater<>());
        json list = json::array();
        for (size_t i = 0; i < top.size() && i < 24; ++i) {
            char reg[16];
            std::snprintf(reg, sizeof reg, "0x%04x", bases[s] + top[i].second);
            list.push_back({{"first_register", reg}, {"offset", top[i].second}, {"packets", top[i].first}});
        }
        regs[names[s]] = {{"packets", register_packets[s].load()}, {"values", register_values[s].load()}, {"distinct_first_registers", top.size()}, {"top", list}};
    }
    out["register_packets"] = regs;
    json events = json::object();
    for (u32 i = 0; i < event_kinds.size(); ++i) {
        if (const u64 n = event_kinds[i].load()) {
            char key[24];
            std::snprintf(key, sizeof key, "type_%u_index_%u", i / 16, i % 16);
            events[key] = n;
        }
    }
    out["event_write_kinds"] = events;
    json sample_json = json::object();
    {
        std::scoped_lock lk{sample_mutex};
        for (u32 o = 0; o < 256; ++o) {
            if (!samples[o].empty()) {
                char key[8];
                std::snprintf(key, sizeof key, "0x%02x", o);
                sample_json[key] = samples[o];
            }
        }
    }
    out["raw_packet_samples"] = sample_json;
    json other = json::object();
    for (u32 o = 0; o < 256; ++o) {
        if (const u64 n = other_packets[o].load()) {
            char key[8];
            std::snprintf(key, sizeof key, "0x%02x", o);
            other[key] = n;
        }
    }
    out["packets_that_are_not_entries"] = other;
    out["occlusion_queries"] = Core::NativeOcclusion::ReportAll();
    json skipped = json::array();
    {
        std::scoped_lock lk{skip_mutex};
        for (const auto& [key, v] : skipped_in_clear) {
            char hash[24];
            std::snprintf(hash, sizeof hash, "%#llx", static_cast<unsigned long long>(key.hash));
            skipped.push_back({{"kind", key.kind == 0 ? "draw skipped" : key.kind == 1 ? "dispatch skipped (the clear's own fill)" : key.kind == 2 ? "draw run (inside the range, not the clear's)" : "dispatch run (inside the range, not the clear's)"}, {"program_hash", hash}, {"count", v.first}, {"last_size", v.second}});
        }
    }
    out["skipped_inside_engine_clears"] = skipped;
    json ce_ops = json::object();
    for (u32 o = 0; o < 256; ++o) {
        if (const u64 n = ce_opcodes[o].load()) {
            char key[8];
            std::snprintf(key, sizeof key, "0x%02x", o);
            ce_ops[key] = {{"packets", n}, {"dwords", ce_opcode_dwords[o].load()}};
        }
    }
    out["constant_engine"] = {{"ccb_submissions", ce_submissions.load()}, {"ccb_dwords", ce_dwords.load()}, {"by_opcode", ce_ops}};
    json images = json::object();
    for (u32 i = 0; i < NumImageEvents; ++i) {
        images[ImageNames[i]] = {{"calls", image_events[i].load()}, {"bytes", image_bytes[i].load()}};
    }
    out["texture_cache"] = images;
    return out;
}

} // namespace Core::NativeCensus
