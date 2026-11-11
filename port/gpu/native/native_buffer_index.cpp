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
        const char* e = std::getenv("TLG_ORDERED_VIEW"); // LEGACY(ordered_view): default on since 2026-11-11 (review notes A3), =0 old path
        // the ordered view needs the create and delete events of buffers (placement and lifetime modes of the native updates)
        return (e == nullptr || e[0] != '0') && NativeUpdates::B9PlaceOn() && NativeUpdates::B2NativeMode() != 0 && NativeUpdates::B9LifetimeMode() != 0;
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

size_t Ordered::LowerBound(VAddr address) const noexcept {
    return static_cast<size_t>(std::lower_bound(items.begin(), items.end(), address, [](const Item& item, VAddr a) { return item.begin < a; }) - items.begin());
}

void Ordered::Create(u64 id, VAddr address, u64 bytes, u32 usage, u32 cpu_access) {
    std::unique_lock lock{mutex};
    const size_t at = LowerBound(address);
    const Item item{address, address + bytes, id, usage, cpu_access};
    if (at < items.size() && items[at].begin == address) {
        items[at] = item; // (as the tree did: the same begin replaces the entry)
    } else {
        items.insert(items.begin() + static_cast<std::ptrdiff_t>(at), item);
    }
    by_id[id] = address;
}

void Ordered::Delete(u64 id) {
    std::unique_lock lock{mutex};
    if (const auto it = by_id.find(id); it != by_id.end()) {
        const size_t at = LowerBound(it->second);
        if (at < items.size() && items[at].begin == it->second && items[at].id == id) {
            items.erase(items.begin() + static_cast<std::ptrdiff_t>(at));
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
    size_t at = LowerBound(end);
    u32 below = 0; // as the live table: a few entries that start before the range are looked at
    while (at != 0 && below < 16) {
        const Item& item = items[--at];
        if (item.begin < address) {
            ++below;
        }
        if (item.end <= address) {
            continue;
        }
        out.push_back({item.id, std::max(item.begin, address), std::min<VAddr>(item.end, end), item.usage, item.cpu_access, item.begin, item.end});
    }
}

bool Ordered::ResourceAt(VAddr address, NativeDevice::Object& out) const {
    std::shared_lock lock{mutex};
    // the last item with begin <= address
    size_t at = LowerBound(address);
    if (at < items.size() && items[at].begin == address) {
        ++at;
    }
    if (at == 0) {
        return false;
    }
    const Item& item = items[at - 1];
    if (address >= std::max(item.end, item.begin + 1)) {
        return false;
    }
    out = {};
    out.id = item.id;
    out.kind = NativeDevice::Kind::Buffer;
    out.alive = true;
    out.usage = item.usage;
    out.cpu_access = item.cpu_access;
    out.address = item.begin;
    out.bytes = item.end - item.begin;
    return true;
}

bool Ordered::Lookup(u64 id, NativeDevice::Object& out) const {
    std::shared_lock lock{mutex};
    const auto by = by_id.find(id);
    if (by == by_id.end()) {
        return false;
    }
    const size_t at = LowerBound(by->second);
    if (at >= items.size() || items[at].begin != by->second || items[at].id != id) {
        return false;
    }
    const Item& item = items[at];
    out = {};
    out.id = id;
    out.kind = NativeDevice::Kind::Buffer;
    out.alive = true;
    out.usage = item.usage;
    out.cpu_access = item.cpu_access;
    out.address = item.begin;
    out.bytes = item.end - item.begin;
    return true;
}

void Ordered::IdsIn(VAddr address, u64 size, boost::container::small_vector<u64, 8>& out) const {
    std::shared_lock lock{mutex};
    size_t at = LowerBound(address);
    if (at < items.size() && items[at].begin == address) {
        ++at; // upper bound
    }
    if (at != 0) {
        --at;
    }
    for (; at < items.size() && items[at].begin < address + size; ++at) {
        if (items[at].end > address) {
            out.push_back(items[at].id);
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
