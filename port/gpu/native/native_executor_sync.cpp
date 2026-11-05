// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_executor_sync.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "core/memory.h"
#include "native/native_debug_server.h"
#include "native/native_drain_audit.h"
#include "native/native_raycast_cpu.h"
#include "native/native_timeline.h"
#include "native/native_wait_registers.h"

namespace Core::NativeExecutorSync {

namespace {
u64 ReadLabel(VAddr address, u32 num_bytes) {
    u64 value{};
    std::memcpy(&value, reinterpret_cast<const void*>(address), num_bytes);
    return value;
}
} // namespace

u64 LabelTimeline::Record(VAddr address, u64 value, u32 num_bytes, u32 queue) {
    // Read outside the lock, see Match.
    const u64 memory = ReadLabel(address, num_bytes);
    std::scoped_lock lk{mutex};
    const u64 seq = ++next_fence_seq;
    // A label reused with another size: forget the old state, waits then poll memory.
    std::erase_if(label_states, [&](const LabelState& label) {
        return label.address < address + num_bytes && address < label.address + label.num_bytes &&
               (label.address != address || label.num_bytes != num_bytes);
    });
    auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->overridden) {
        // Start tracking from what guest memory holds now.
        if (it == label_states.end()) {
            it = label_states.emplace(label_states.end());
            it->address = address;
            it->num_bytes = num_bytes;
        }
        it->overridden = false;
        it->known_values.clear();
        it->known_values.push_back(memory);
    }
    // Only the latest value can satisfy a later wait: the GPU runs the fences in the order they
    // were recorded, so older values are overwritten before any later wait could observe them.
    it->latest_value = value;
    it->latest_seq = seq;
    // Set by SetTick once the raster thread has recorded the work before this fence.
    it->gpu_tick = 0;
    it->gpu_position = 0;
    it->queue = queue;
    ++it->pending;
    if (it->known_values.size() < MaxKnownLabelValues) {
        it->known_values.push_back(value);
    }
    return seq;
}

void LabelTimeline::SetTick(VAddr address, u64 seq, u64 tick) {
    std::scoped_lock lk{mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it != label_states.end() && it->latest_seq == seq) {
        it->gpu_tick = tick;
    }
}

void LabelTimeline::SetPosition(VAddr address, u64 seq, u64 position) {
    std::scoped_lock lk{mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it != label_states.end() && it->latest_seq == seq) {
        it->gpu_position = position;
    }
}

u64 LabelTimeline::Position(VAddr address) {
    std::scoped_lock lk{mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->pending == 0) {
        return 0;
    }
    return it->gpu_position != 0 ? it->gpu_position : ~u64{0};
}

u64 LabelTimeline::Tick(VAddr address, u64 seq) {
    std::scoped_lock lk{mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->latest_seq < seq) {
        return 0; // Retired: the work finished, nothing to order against.
    }
    return it->gpu_tick;
}

void LabelTimeline::Retire(VAddr address, u64 seq) {
    std::scoped_lock lk{mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->latest_seq < seq) {
        return;
    }
    if (--it->pending == 0) {
        // Everything recorded has landed: guest memory is authoritative again.
        label_states.erase(it);
    }
}

FenceMatch LabelTimeline::Match(const NativeSync::LabelWait& wait_reg_mem) {
    if (!wait_reg_mem.memory) {
        return {};
    }
    const VAddr wait_address = wait_reg_mem.Address<VAddr>();
    // Read outside the lock: the read may fault into a readback that waits on the thread
    // retiring fences.
    const u32 memory = *wait_reg_mem.Address();
    std::scoped_lock lk{mutex};
    for (auto& label : label_states) {
        if (wait_address < label.address ||
            wait_address + sizeof(u32) > label.address + label.num_bytes) {
            continue;
        }
        if (label.overridden) {
            return {};
        }
        const u32 shift = static_cast<u32>(wait_address - label.address) * 8;
        const auto slice = [shift](u64 value) { return static_cast<u32>(value >> shift); };
        // Guest memory may only hold the value from before the first held back fence, or one
        // of the recorded values. Anything else was written by someone else (typically the
        // guest CPU resetting or releasing the label), and that write is newer than the
        // recorded values: from now on waits on this label poll guest memory.
        if (std::ranges::none_of(label.known_values,
                                 [&](u64 known) { return slice(known) == memory; })) {
            label.overridden = true;
            if (logged_overrides.insert(label.address).second) {
                LOG_INFO(Render,
                         "Fence timeline: label {:#x} changed to {:#x} outside the recorded "
                         "fences (held back value {:#x}, value before them {:#x}); waits on it "
                         "poll memory until the next fence",
                         label.address, memory, slice(label.latest_value),
                         slice(label.known_values.front()));
            }
            return {};
        }
        const u32 value = slice(label.latest_value);
        return {
            .satisfied = wait_reg_mem.TestValue(value),
            .label_address = label.address,
            .seq = label.latest_seq,
            .gpu_tick = label.gpu_tick,
            .queue = label.queue,
            .value = value,
            .initial = slice(label.known_values.front()),
        };
    }
    return {};
}

bool LabelTimeline::HeldBack(VAddr begin, u32 num_bytes) {
    std::scoped_lock lk{mutex};
    return std::ranges::any_of(label_states, [&](const LabelState& label) {
        return label.pending != 0 && label.address < begin + num_bytes &&
               begin < label.address + label.num_bytes;
    });
}

void LabelTimeline::NoteDeferredWrite(VAddr begin, u32 num_bytes, u32 queue) {
    if (logged_deferred_writes.size() < 64 && logged_deferred_writes.insert(begin).second) {
        LOG_INFO(Render,
                 "Fence timeline: queue {} writes {} bytes to label {:#x} while a fence write to "
                 "it is held back; the write is queued behind it",
                 queue, num_bytes, begin);
    }
}

void LabelTimeline::NoteComputePass(const NativeSync::LabelWait& wait_reg_mem, const FenceMatch& match, u32 queue) {
    const VAddr address = wait_reg_mem.Address<VAddr>();
    // Capped: every new label address logged made the console sink about 3% of the front thread.
    if (logged_compute_passes.size() < 64 && logged_compute_passes.insert(address).second) {
        LOG_INFO(Render,
                 "Fence timeline: compute ring {} passed a wait on {:#x} (func={} ref={:#x} "
                 "mask={:#x}) with value {:#x} recorded by queue {}; memory holds {:#x}, "
                 "held {:#x} before the fence",
                 queue - 1, address, static_cast<u32>(wait_reg_mem.function),
                 wait_reg_mem.ref, wait_reg_mem.mask, match.value, match.queue,
                 *wait_reg_mem.Address(), match.initial);
    }
}

std::string LabelTimeline::PendingText() {
    std::string pending;
    std::scoped_lock lk{mutex};
    for (const auto& label : label_states) {
        pending +=
            fmt::format(" [{:#x}={:#x} q{} pending={}{}]", label.address, label.latest_value,
                        label.queue, label.pending, label.overridden ? " overridden" : "");
    }
    return pending;
}

void FenceLatency::Note(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns, u64 front_ns) {
    auto& stat = stats[std::countr_zero(static_cast<u32>(kind)) % stats.size()];
    const u64 total = queue_ns + gpu_ns;
    stat.count.fetch_add(1, std::memory_order_relaxed);
    stat.queue_ns.fetch_add(queue_ns, std::memory_order_relaxed);
    stat.front_ns.fetch_add(front_ns, std::memory_order_relaxed);
    stat.gpu_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
    stat.total_ns.fetch_add(total, std::memory_order_relaxed);
    u64 max = stat.max_total_ns.load(std::memory_order_relaxed);
    while (total > max && !stat.max_total_ns.compare_exchange_weak(max, total)) {
    }
}

nlohmann::json FenceLatency::Report() {
    static constexpr const char* kind_names[] = {"gfx_eop", "gfx_eos", "compute_release",
                                                 "held_write", "submit_end"};
    nlohmann::json fences = nlohmann::json::object();
    for (size_t i = 0; i < stats.size(); ++i) {
        auto& stat = stats[i];
        const u64 count = stat.count.exchange(0);
        const u64 queue_ns = stat.queue_ns.exchange(0);
        const u64 front_ns = stat.front_ns.exchange(0);
        const u64 gpu_ns = stat.gpu_ns.exchange(0);
        const u64 total_ns = stat.total_ns.exchange(0);
        const u64 max_ns = stat.max_total_ns.exchange(0);
        if (count == 0) {
            continue;
        }
        fences[kind_names[i]] = {
            {"fences", count},
            {"avg_ms_parsed_to_raster_start", queue_ns / 1e6 / count},
            {"avg_ms_parsed_to_push_done", front_ns / 1e6 / count},
            {"avg_ms_raster_start_to_signal", gpu_ns / 1e6 / count},
            {"avg_ms_total", total_ns / 1e6 / count},
            {"max_ms_total", max_ns / 1e6}};
    }
    return fences;
}

void ReportLongWait(LabelTimeline& timeline, const NativeSync::LabelWait& wait_reg_mem, const char* queue,
                    u64 wait_start, bool& reported) {
    if (reported || VideoCore::ReadbackStats::NowNs() - wait_start < 2'000'000'000ULL) {
        return;
    }
    reported = true;
    const bool is_memory = wait_reg_mem.memory;
    const u32 value = is_memory ? *wait_reg_mem.Address() : NativeWaitRegisters::Read(wait_reg_mem.Reg());
    const std::string pending = timeline.PendingText();
    LOG_WARNING(Render,
                "Hang check: {} queue waiting >2s on {} {:#x}: value={:#x} func={} ref={:#x} "
                "mask={:#x}; pending fence writes:{}",
                queue, is_memory ? "memory" : "register",
                is_memory ? wait_reg_mem.Address<VAddr>() : VAddr{wait_reg_mem.Reg()}, value,
                static_cast<u32>(wait_reg_mem.function), wait_reg_mem.ref,
                wait_reg_mem.mask, pending.empty() ? " none" : pending);
}

void ReportLongSpin(const char* what, const void* address, u64 wait_start, bool& reported) {
    VideoCore::ReadbackStats::CpWaitYield(false);
    if (reported || VideoCore::ReadbackStats::NowNs() - wait_start < 2'000'000'000ULL) {
        return;
    }
    reported = true;
    LOG_WARNING(Render, "Hang check: {} waiting >2s on {}", what, fmt::ptr(address));
}

void WriteFenceMemory(void* address, u64 data, u32 num_bytes) {
    auto* memory = Core::Memory::Instance();
    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
        memcpy(address, &data, num_bytes);
    }
}

void NoFenceWrite(void*, u64, u32) {}

bool TryWriteOffThread(void* address, const void* data, u32 num_bytes, const std::function<bool(VAddr, u32)>& not_mapped) {
    if (Core::Memory::Instance()->TryWriteBacking(address, data, num_bytes)) {
        return true;
    }
    if (not_mapped(reinterpret_cast<VAddr>(address), num_bytes)) {
        std::memcpy(address, data, num_bytes);
        return true;
    }
    return false;
}

void WriteBackingOrMemory(void* address, const void* data, u32 num_bytes) {
    if (!Core::Memory::Instance()->TryWriteBacking(address, data, num_bytes)) {
        std::memcpy(address, data, num_bytes);
    }
}

void ApplyCpWrite(void* address, const std::vector<u32>& words, u32 queue) {
    std::memcpy(address, words.data(), words.size() * sizeof(u32));
    if (Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
        Core::FrameTimeline::Record(Core::FrameTimeline::LabelWrite, words[0] | (u64(queue) << 32), Core::FrameTimeline::Now(), reinterpret_cast<u64>(address));
    }
}

bool PrepareDma(const NativeSync::DmaCopy& dma) {
    using Kind = NativeSync::DmaCopy::Kind;
    if (dma.kind == Kind::Ignored) {
        return false;
    }
    if (dma.kind == Kind::Unknown) {
        Core::RareEvents::Note("dma_unknown_kind", "a CP DMA with a source / destination pair the game never used");
        return false;
    }
    if (dma.kind == Kind::FillGuest || dma.kind == Kind::GdsToMemory || dma.kind == Kind::MemoryToMemory) {
        Core::DrainAudit::SetNextWrite(static_cast<VAddr>(dma.destination), dma.bytes);
    } else {
        Core::DrainAudit::SetNextWrite(Core::DrainAudit::NoGuestMemory, 0);
    }
    return true;
}

void RunDma(const NativeSync::DmaCopy& dma, DmaHost& host) {
    using Kind = NativeSync::DmaCopy::Kind;
    switch (dma.kind) {
    case Kind::FillGds:
        host.FillBuffer(dma.destination, dma.bytes, dma.fill, true);
        break;
    case Kind::MemoryToGds:
        host.CopyBuffer(dma.destination, static_cast<VAddr>(dma.source), dma.bytes, true, false);
        break;
    case Kind::FillGuest:
        host.FillBuffer(static_cast<VAddr>(dma.destination), dma.bytes, dma.fill, false);
        break;
    case Kind::GdsToMemory:
        if (Core::NativeRaycast::ShadowOn()) {
            Core::NativeRaycast::NoteCounterCopy(dma.source, dma.destination);
        }
        host.CopyBuffer(static_cast<VAddr>(dma.destination), dma.source, dma.bytes, false, true);
        break;
    case Kind::MemoryToMemory:
        if (Core::NativeRaycast::ShadowOn()) {
            Core::NativeRaycast::NoteMemoryCopy(dma.source, dma.destination, dma.bytes);
        }
        host.CopyBuffer(static_cast<VAddr>(dma.destination), static_cast<VAddr>(dma.source), dma.bytes, false, false);
        break;
    default:
        break;
    }
}

} // namespace Core::NativeExecutorSync
