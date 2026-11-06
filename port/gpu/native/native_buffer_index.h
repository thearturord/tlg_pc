// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The engine's buffers as the raster thread's in-order create and delete events know them (G5b, documents/tlg/native plan/executor-consumer-plan.md; B9 / M5 of the native
// back end). The live table of the game thread (`NativeDevice`) is ahead of the raster thread at frames in flight 2 (it has deleted what the raster thread still works on and
// created what it has not reached); this index is the raster thread's own view: begin -> {end, id, usage, cpu access} and id -> begin. Plain ids and numbers, no Vulkan.
//
// `TLG_ORDERED_VIEW=1` (see `On`): the buffer cache asks this view; otherwise it asks the live table. `TLG_ORDERED_VIEW_CHECK=1` counts the requests where the two disagree
// (`Check`, class 0 same, 1 the live table has buffers the view lacks, 2 the view has buffers the live table lacks, 3 both).
//
// Thread: writes from the raster thread (in order events); reads from any thread (shared lock).

#include <atomic>
#include <map>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <nlohmann/json.hpp>

#include "common/types.h"
#include "native/native_device.h"

namespace Core::NativeBufferIndex {

class Ordered {
public:
    /// TLG_ORDERED_VIEW=1 and the B9 modes it needs.
    [[nodiscard]] static bool On() noexcept;
    /// TLG_ORDERED_VIEW_CHECK=1.
    [[nodiscard]] static bool CheckOn() noexcept;

    /// A buffer of `bytes` (not 0) at `address` was created in order.
    void Create(u64 id, VAddr address, u64 bytes, u32 usage, u32 cpu_access);
    /// The buffer `id` was deleted in order.
    void Delete(u64 id);

    /// The buffers over [address, address + size) (the same rule as the live table: a few entries that start before the range are looked at).
    void BuffersIn(VAddr address, u64 size, std::vector<NativeDevice::BufferPart>& out) const;
    [[nodiscard]] bool ResourceAt(VAddr address, NativeDevice::Object& out) const;
    [[nodiscard]] bool Lookup(u64 id, NativeDevice::Object& out) const;

    /// The ids of the buffers over [address, address + size), for the comparison with the live table.
    void IdsIn(VAddr address, u64 size, boost::container::small_vector<u64, 8>& out) const;
    /// One compared request: `cls` as in the header; `written` splits reads and writes.
    void NoteCheck(bool written, u32 cls);
    [[nodiscard]] nlohmann::json CheckReport() const;

private:
    struct Entry {
        VAddr end;
        u64 id;
        u32 usage, cpu_access;
    };
    mutable std::shared_mutex mutex;
    std::map<VAddr, Entry> by_address;         ///< begin -> entry
    std::unordered_map<u64, VAddr> by_id;      ///< id -> begin
    std::atomic<u64> counts[2][5]{};           ///< [written][class 0..3, 4 = requests]
};

} // namespace Core::NativeBufferIndex
