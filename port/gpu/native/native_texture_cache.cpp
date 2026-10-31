// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <fstream>
#include <limits>
#include <xxhash.h>
#include "native/native_image_marks.h"
#include "native/native_untracked_ring.h"

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "native/native_resources.h"
#include "native/native_device.h"
#include "native/native_updates.h"
#include "native/native_write_watch.h"
#include "native/native_resource_audit.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "native/native_readback_policy.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include <fmt/format.h>
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

// Native renderer: member functions of an emulator class that belong to the native path. They live here (not in the emulator's file)
// so that the emulator files only call into native/; the class stays declared in the emulator's header until the owning plan phase
// (documents/tlg/native plan/no-emulation-plan.md, E-1 part 2) gives the native side its own interface.

namespace VideoCore {

void TextureCache::NoteEngineTextureCreate(VAddr address, u64 bytes) {
    Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8CreatedTextures);
    ForEachImageInRegion(address, bytes, [&](ImageId, Image& image) {
        if ((image.b8_marks & B8Listed) != 0) {
            image.b8_marks &= ~B8Listed;
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8CreatedUnlisted);
        }
    });
}

void TextureCache::NoteEngineBufferCreate(u64 id, VAddr address, u64 bytes) {
    if (Core::NativeUpdates::B8BufferImagesMode() == 0 || !buffers_known.insert(id).second) {
        return;
    }
    auto& list = buffer_images[id];
    ForEachImageInRegion(address, bytes, [&](ImageId image_id, Image& image) {
        list.emplace_back(image_id, image.image_uid);
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8BufListed);
    });
    if (list.empty()) {
        buffer_images.erase(id);
    }
}

bool TextureCache::BufferImagesIn(VAddr address, u64 size, SmallVector<ImageId, 8>& out) {
    static thread_local std::vector<Core::NativeDevice::BufferPart> parts;
    parts.clear();
    Core::NativeDevice::BuffersIn(address, size, parts);
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
            NoteEngineBufferCreate(part.id, part.object_begin, part.object_end - part.object_begin);
        }
        const auto it = buffer_images.find(part.id);
        if (it == buffer_images.end()) {
            continue;
        }
        auto& list = it->second;
        list.erase(std::remove_if(list.begin(), list.end(),
                                  [&](const auto& e) {
                                      return !slot_images.IsAllocated(e.first) || slot_images[e.first].image_uid != e.second ||
                                             False(slot_images[e.first].flags & ImageFlagBits::Registered);
                                  }),
                   list.end());
        for (const auto& [image_id, uid] : list) {
            if (slot_images[image_id].Overlaps(address, size) && std::ranges::find(out, image_id) == out.end()) {
                out.push_back(image_id);
            }
        }
    }
    return true;
}

void TextureCache::EngineImagesIn(const std::vector<u64>& ids, VAddr address, u64 size, SmallVector<ImageId, 8>& out) const {
    const auto add = [&](ImageId image_id, u64 uid) {
        if (!slot_images.IsAllocated(image_id)) {
            return;
        }
        const Image& image = slot_images[image_id];
        if (image.image_uid != uid || False(image.flags & ImageFlagBits::Registered) || !image.Overlaps(address, size) ||
            std::ranges::find(out, image_id) != out.end()) {
            return;
        }
        out.push_back(image_id);
    };
    for (const u64 id : ids) {
        if (const auto it = engine_images.find(id); it != engine_images.end()) {
            for (const auto& [image_id, uid] : it->second) {
                add(image_id, uid);
            }
        }
        if (const auto it = engine_answers.find(id); it != engine_answers.end()) {
            for (const auto& a : it->second) {
                add(a.image_id, a.uid);
            }
        }
    }
}

void TextureCache::NoteAnnouncedWrite(VAddr addr, size_t size, bool newer) {
    if (engine_pending_clears.empty() && !AnyImageOverlaps(addr, size)) {
        return; // no image in this memory (most announcements: buffer contents); the page table answers that in a few lookups
    }
    SmallVector<ImageId, 8> image_ids;
    // B8.4: a range the engine's textures cover finds its images through them (their lists); B8.5: a range engine buffers cover, through the buffers' lists;
    // only memory neither covers whole asks the page table
    static thread_local std::vector<u64> ids;
    ids.clear();
    const bool covered_by_textures = Core::NativeDevice::TexturesIn(addr, size, ids);
    if (!engine_pending_clears.empty()) {
        // B11: a CPU write to a texture whose clear waits for its first image comes after the clear: the clear is gone
        for (const u64 id : ids) {
            engine_pending_clears.erase(id);
        }
    }
    if (covered_by_textures) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8RouteAnnounce);
        EngineImagesIn(ids, addr, size, image_ids);
    } else if (BufferImagesIn(addr, size, image_ids)) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8BufAnnounce);
    } else {
        image_ids.clear();
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8BufNotCovered);
        ForEachImageInRegion(addr, size, [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });
    }
    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};
        if (!image.Overlaps(addr, size)) {
            continue;
        }
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8AnnouncedOnImage);
        if ((image.b8_marks & B8Owed) != 0) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8AnnouncedLater);
            image.b8_marks &= ~B8Owed;
        }
        if (!newer && True(image.flags & ImageFlagBits::GpuModified)) {
            // the GPU wrote the image after the first application: its bytes are newer than the CPU's
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8KeptForGpu);
            continue;
        }
        image.b8_marks |= B8Announced;
        image.flags |= ImageFlagBits::CpuDirty;
        UntrackImage(image_id);
    }
}

void TextureCache::NoteEngineTextureDelete(u64 id, VAddr address, u64 bytes, bool free) {
    Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8TexturesDeleted);
    // B8.4: the images the texture's lists held are no longer kept current through it: the next lookup that gives one out refreshes it (adoption)
    const auto unlist = [&](ImageId image_id, u64 uid) {
        if (slot_images.IsAllocated(image_id) && slot_images[image_id].image_uid == uid) {
            slot_images[image_id].b8_marks &= ~B8Listed;
        }
    };
    if (const auto it = engine_images.find(id); it != engine_images.end()) {
        for (const auto& [image_id, uid] : it->second) {
            unlist(image_id, uid);
        }
    }
    if (const auto it = engine_answers.find(id); it != engine_answers.end()) {
        for (const auto& a : it->second) {
            unlist(a.image_id, a.uid);
        }
    }
    engine_images.erase(id);
    engine_answers.erase(id);
    engine_pending_clears.erase(id);
    if (bytes == 0) {
        return;
    }
    SmallVector<ImageId, 4> image_ids;
    ForEachImageInRegion(address, bytes, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (image.info.guest_size > bytes) {
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImagesBigger);
            return;
        }
        image_ids.push_back(image_id);
    });
    for (const auto image_id : image_ids) {
        Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImages);
        if (free) {
            FreeImage(image_id);
            Core::NativeUpdates::NoteB8(Core::NativeUpdates::B8DeletedImagesFreed);
        } else {
            slot_images[image_id].b8_marks |= B8Deleted;
        }
    }
}

vk::ClearValue TextureCache::EngineColorValue(vk::Format format, const float (&value)[4]) {
    vk::ClearValue out{};
    const std::string name = vk::to_string(format);
    if (name.find("Uint") != std::string::npos || name.find("Sint") != std::string::npos) {
        for (u32 i = 0; i < 4; ++i) {
            out.color.int32[i] = static_cast<s32>(value[i]);
        }
    } else {
        for (u32 i = 0; i < 4; ++i) {
            out.color.float32[i] = value[i];
        }
    }
    return out;
}

} // namespace VideoCore
