// SPDX-License-Identifier: GPL-2.0-or-later
// G5b: see native_buffer_index.h.

#include <algorithm>
#include <cstdlib>
#include <mutex>

#include "native/native_buffer_index.h"
#include "native/native_updates.h"

namespace Core::NativeBufferIndex {

bool Ordered::On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_ORDERED_VIEW");
        // the ordered view needs the create and delete events of buffers (placement and lifetime modes of the native updates)
        return e != nullptr && e[0] == '1' && NativeUpdates::B9PlaceOn() && NativeUpdates::B2NativeMode() != 0 && NativeUpdates::B9LifetimeMode() != 0;
    }();
    return on;
}

bool Ordered::CheckOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_ORDERED_VIEW_CHECK");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void Ordered::Create(u64 id, VAddr address, u64 bytes, u32 usage, u32 cpu_access) {
    std::unique_lock lock{mutex};
    by_address[address] = {address + bytes, id, usage, cpu_access};
    by_id[id] = address;
}

void Ordered::Delete(u64 id) {
    std::unique_lock lock{mutex};
    if (const auto it = by_id.find(id); it != by_id.end()) {
        if (const auto at = by_address.find(it->second); at != by_address.end() && at->second.id == id) {
            by_address.erase(at);
        }
        by_id.erase(it);
    }
}

void Ordered::BuffersIn(VAddr address, u64 size, std::vector<NativeDevice::BufferPart>& out) const {
    if (size == 0) {
        return;
    }
    const VAddr end = address + size;
    std::shared_lock lock{mutex};
    auto it = by_address.lower_bound(end);
    u32 below = 0; // as the live table: a few entries that start before the range are looked at
    while (it != by_address.begin() && below < 16) {
        --it;
        if (it->first < address) {
            ++below;
        }
        if (it->second.end <= address) {
            continue;
        }
        out.push_back({it->second.id, std::max(it->first, address), std::min<VAddr>(it->second.end, end), it->second.usage, it->second.cpu_access, it->first, it->second.end});
    }
}

bool Ordered::ResourceAt(VAddr address, NativeDevice::Object& out) const {
    std::shared_lock lock{mutex};
    auto it = by_address.upper_bound(address);
    if (it == by_address.begin()) {
        return false;
    }
    --it;
    if (address >= std::max(it->second.end, it->first + 1)) {
        return false;
    }
    out = {};
    out.id = it->second.id;
    out.kind = NativeDevice::Kind::Buffer;
    out.alive = true;
    out.usage = it->second.usage;
    out.cpu_access = it->second.cpu_access;
    out.address = it->first;
    out.bytes = it->second.end - it->first;
    return true;
}

bool Ordered::Lookup(u64 id, NativeDevice::Object& out) const {
    std::shared_lock lock{mutex};
    const auto by = by_id.find(id);
    if (by == by_id.end()) {
        return false;
    }
    const auto at = by_address.find(by->second);
    if (at == by_address.end() || at->second.id != id) {
        return false;
    }
    out = {};
    out.id = id;
    out.kind = NativeDevice::Kind::Buffer;
    out.alive = true;
    out.usage = at->second.usage;
    out.cpu_access = at->second.cpu_access;
    out.address = at->first;
    out.bytes = at->second.end - at->first;
    return true;
}

void Ordered::IdsIn(VAddr address, u64 size, boost::container::small_vector<u64, 8>& out) const {
    std::shared_lock lock{mutex};
    auto it = by_address.upper_bound(address);
    if (it != by_address.begin()) {
        --it;
    }
    for (; it != by_address.end() && it->first < address + size; ++it) {
        if (it->second.end > address) {
            out.push_back(it->second.id);
        }
    }
}

void Ordered::NoteCheck(bool written, u32 cls) {
    counts[written ? 1 : 0][4].fetch_add(1, std::memory_order_relaxed);
    counts[written ? 1 : 0][cls].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Ordered::CheckReport() const {
    const auto side = [&](int w) {
        return nlohmann::json{{"requests", counts[w][4].load()}, {"same", counts[w][0].load()}, {"live_has_more", counts[w][1].load()}, {"ordered_has_more", counts[w][2].load()}, {"both_differ", counts[w][3].load()}};
    };
    return {{"read", side(0)}, {"written", side(1)}};
}

} // namespace Core::NativeBufferIndex
