// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <functional>

#include <nlohmann/json.hpp>

#include "common/types.h"

/// L3, the CPU write tracker without page protection (`TLG_WRITE_WATCH=1`). The runtime maps the GPU visible direct memory as
/// private allocations with `MEM_WRITE_WATCH` (port/src/runtime_memory.c): the kernel records every page the CPU writes, for any
/// writer, announced or not. The buffer cache asks for the written pages of a range before it uses the range (`Harvest`) and marks
/// them CPU modified exactly as the write fault did; the page manager then grants write access to those pages whatever the
/// buffer cache tracks (`LiftProtection`), so they never fault. Pages an image watches stay protected (the texture cache still
/// learns writes from the fault).
///
/// When a range needs a harvest: a page the CPU wrote matters to the GPU from the guest's next submission or from a memory wait
/// the command processor passed (the game publishes data and then a label). The epoch counts both; a chunk of 64 KiB harvested at
/// the current epoch needs no new harvest (`TLG_WRITE_WATCH_EPOCH=0` harvests every request).
namespace Core::WriteWatch {

[[nodiscard]] bool Enabled() noexcept;
/// Enabled and the readbacks mode allows it (a write fault with readbacks on must download first, which a write watch cannot)
[[nodiscard]] bool LiftProtection() noexcept;

extern std::atomic<u32> epoch;
/// a guest submission (graphics or compute) or a passed wait on memory
inline void BumpEpoch() noexcept {
    epoch.fetch_add(1, std::memory_order_acq_rel);
}

using Replay = void (*)(void* context, VAddr page, u64 size);
using IsCpuDirty = bool (*)(void* context, VAddr page, u64 size);
/// The pages of [address, address + size) the CPU wrote since their last harvest, and the pending ones the runtime found, each given
/// to `replay` (4 KiB pages). Serialized: a caller that finds nothing to do returns only after a harvest in progress replayed.
/// Verify mode (`TLG_WRITE_WATCH_VERIFY=N`, one request in N): every page of the range that is not CPU modified after the harvest is
/// hashed and compared with the hash of its last upload (`NoteUpload`); a page that changed without a write watch record is a miss
/// (counted, logged, marked).
void Harvest(VAddr address, u64 size, Replay replay, IsCpuDirty dirty, IsCpuDirty gpu, void* context);

/// verify mode: what an upload sent to the GPU, per 4 KiB page (page aligned data)
void NoteUpload(VAddr address, const u8* data, u64 size);
/// verify mode: the GPU side wrote guest memory (no CPU write): its pages have no valid upload hash any more
void NoteGpuSideWrite(VAddr address, u64 size);
[[nodiscard]] bool VerifyOn() noexcept;
/// verify mode: runs `write` (a GPU side write of the range) with the upload hashes locked, then drops the range's hashes
bool GpuSideWrite(VAddr address, u64 size, const std::function<bool()>& write);

/// The watched regions, for the page manager (true when [address, address + size) lies in one)
[[nodiscard]] bool Watched(VAddr address, u64 size);

nlohmann::json Report();

} // namespace Core::WriteWatch
