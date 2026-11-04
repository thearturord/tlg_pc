// SPDX-License-Identifier: GPL-2.0-or-later

// The engine's textures and buffers as the image cache sees them (G5a): the code of the TextureCache members that kept these lists (texture_cache_engine.cpp), by name only.

#include <algorithm>

#include "native/native_device.h"
#include "native/native_image_index.h"
#include "native/native_updates.h"

namespace Core {

void NativeImageIndex::NoteTextureCreate(VAddr address, u64 bytes) {
    NativeUpdates::NoteB8(NativeUpdates::B8CreatedTextures);
    host.ForEachImageIn(address, bytes, [&](ImageId image_id) {
        if (host.ClearListed(image_id)) {
            NativeUpdates::NoteB8(NativeUpdates::B8CreatedUnlisted);
        }
    });
}

void NativeImageIndex::NoteBufferCreate(u64 id, VAddr address, u64 bytes) {
    if (NativeUpdates::B8BufferImagesMode() == 0 || !buffers_known.insert(id).second) {
        return;
    }
    auto& list = buffer_images[id];
    host.ForEachImageIn(address, bytes, [&](ImageId image_id) {
        list.emplace_back(image_id, host.Uid(image_id));
        NativeUpdates::NoteB8(NativeUpdates::B8BufListed);
    });
    if (list.empty()) {
        buffer_images.erase(id);
    }
}

bool NativeImageIndex::BufferImagesIn(VAddr address, u64 size, SmallVector<ImageId, 8>& out) {
    static thread_local std::vector<NativeDevice::BufferPart> parts;
    parts.clear();
    NativeDevice::BuffersIn(address, size, parts);
    if (parts.empty()) {
        return false;
    }
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
    VAddr covered = address;
    for (const auto& part : parts) {
        if (part.begin > covered) {
            return false;
        }
        covered = std::max(covered, part.end);
    }
    if (covered < address + size) {
        return false;
    }
    for (const auto& part : parts) {
        if (!buffers_known.contains(part.id)) {
            // the buffer's creation is still queued (the guest thread made it, its announcements can come first): its older images are listed now
            NoteBufferCreate(part.id, part.object_begin, part.object_end - part.object_begin);
        }
        const auto it = buffer_images.find(part.id);
        if (it == buffer_images.end()) {
            continue;
        }
        auto& list = it->second;
        list.erase(std::remove_if(list.begin(), list.end(), [&](const auto& e) { return !host.Live(e.first, e.second) || !host.Registered(e.first); }), list.end());
        for (const auto& [image_id, uid] : list) {
            if (host.Overlaps(image_id, address, size) && std::ranges::find(out, image_id) == out.end()) {
                out.push_back(image_id);
            }
        }
    }
    return true;
}

void NativeImageIndex::ImagesIn(const std::vector<u64>& ids, VAddr address, u64 size, SmallVector<ImageId, 8>& out) const {
    const auto add = [&](ImageId image_id, u64 uid) {
        if (!host.Live(image_id, uid) || !host.Registered(image_id) || !host.Overlaps(image_id, address, size) || std::ranges::find(out, image_id) != out.end()) {
            return;
        }
        out.push_back(image_id);
    };
    for (const u64 id : ids) {
        if (const auto it = images.find(id); it != images.end()) {
            for (const auto& [image_id, uid] : it->second) {
                add(image_id, uid);
            }
        }
        if (const auto it = answers.find(id); it != answers.end()) {
            for (const auto& a : it->second) {
                add(a.image_id, a.uid);
            }
        }
    }
}

void NativeImageIndex::NoteAnnouncedWrite(VAddr addr, size_t size, bool newer) {
    if (pending_clears.empty() && !host.AnyImageIn(addr, size)) {
        return; // no image in this memory (most announcements: buffer contents); the page table answers that in a few lookups
    }
    SmallVector<ImageId, 8> image_ids;
    // B8.4: a range the engine's textures cover finds its images through them (their lists); B8.5: a range engine buffers cover, through the buffers' lists;
    // only memory neither covers whole asks the page table
    static thread_local std::vector<u64> ids;
    ids.clear();
    const bool covered_by_textures = NativeDevice::TexturesIn(addr, size, ids);
    if (!pending_clears.empty()) {
        // B11: a CPU write to a texture whose clear waits for its first image comes after the clear: the clear is gone
        for (const u64 id : ids) {
            pending_clears.erase(id);
        }
    }
    if (covered_by_textures) {
        NativeUpdates::NoteB8(NativeUpdates::B8RouteAnnounce);
        ImagesIn(ids, addr, size, image_ids);
    } else if (BufferImagesIn(addr, size, image_ids)) {
        NativeUpdates::NoteB8(NativeUpdates::B8BufAnnounce);
    } else {
        image_ids.clear();
        NativeUpdates::NoteB8(NativeUpdates::B8BufNotCovered);
        host.ForEachImageIn(addr, size, [&](ImageId image_id) { image_ids.push_back(image_id); });
    }
    for (const auto image_id : image_ids) {
        switch (host.ApplyAnnounced(image_id, addr, size, newer)) {
        case Host::Announced::NotOverlapping:
            break;
        case Host::Announced::KeptForGpu:
            // the GPU wrote the image after the first application: its bytes are newer than the CPU's
            NativeUpdates::NoteB8(NativeUpdates::B8AnnouncedOnImage);
            NativeUpdates::NoteB8(NativeUpdates::B8KeptForGpu);
            break;
        case Host::Announced::Dirtied:
            NativeUpdates::NoteB8(NativeUpdates::B8AnnouncedOnImage);
            break;
        }
    }
}

void NativeImageIndex::NoteTextureDelete(u64 id, VAddr address, u64 bytes, bool free) {
    NativeUpdates::NoteB8(NativeUpdates::B8TexturesDeleted);
    // B8.4: the images the texture's lists held are no longer kept current through it: the next lookup that gives one out refreshes it (adoption)
    const auto unlist = [&](ImageId image_id, u64 uid) {
        if (host.Live(image_id, uid)) {
            host.ClearListed(image_id);
        }
    };
    if (const auto it = images.find(id); it != images.end()) {
        for (const auto& [image_id, uid] : it->second) {
            unlist(image_id, uid);
        }
    }
    if (const auto it = answers.find(id); it != answers.end()) {
        for (const auto& a : it->second) {
            unlist(a.image_id, a.uid);
        }
    }
    images.erase(id);
    answers.erase(id);
    pending_clears.erase(id);
    if (bytes == 0) {
        return;
    }
    SmallVector<ImageId, 4> image_ids;
    host.ForEachImageIn(address, bytes, [&](ImageId image_id) {
        if (host.GuestAddress(image_id) != address) {
            return;
        }
        if (host.GuestSize(image_id) > bytes) {
            NativeUpdates::NoteB8(NativeUpdates::B8DeletedImagesBigger);
            return;
        }
        image_ids.push_back(image_id);
    });
    for (const auto image_id : image_ids) {
        NativeUpdates::NoteB8(NativeUpdates::B8DeletedImages);
        if (free) {
            host.Free(image_id);
            NativeUpdates::NoteB8(NativeUpdates::B8DeletedImagesFreed);
        } else {
            host.MarkDeleted(image_id);
        }
    }
}

} // namespace Core
