// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The engine's textures and buffers as the image cache sees them (G5a, documents/tlg/native plan/executor-consumer-plan.md; B8 of the native back end). Which cache images (slot id
// and uid) belong to an engine texture, which images lie in the memory of an engine buffer, the clear an engine texture still waits to apply to its first image, and what an
// announced CPU write, a created or a deleted engine texture does to those lists. Plain ids and numbers: the image cache (the host, `Host` below) is asked for the few facts about
// an image it holds and does the Vulkan side of the work (untracking, freeing).
//
// Thread: the raster thread (like the cache).

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/slot_vector.h"
#include "common/small_vector.h"
#include "common/types.h"

namespace Core {

class NativeImageIndex {
public:
    using ImageId = Common::SlotId;

    /// What the index asks of the image cache.
    class Host {
    public:
        virtual ~Host() = default;
        enum class Announced { NotOverlapping, KeptForGpu, Dirtied };
        /// The slot holds the image with this uid.
        [[nodiscard]] virtual bool Live(ImageId id, u64 uid) const = 0;
        /// The slot holds a registered image (the cache looks at it).
        [[nodiscard]] virtual bool Registered(ImageId id) const = 0;
        [[nodiscard]] virtual bool Overlaps(ImageId id, VAddr address, u64 size) const = 0;
        [[nodiscard]] virtual u64 Uid(ImageId id) const = 0;
        [[nodiscard]] virtual VAddr GuestAddress(ImageId id) const = 0;
        [[nodiscard]] virtual u64 GuestSize(ImageId id) const = 0;
        [[nodiscard]] virtual bool AnyImageIn(VAddr address, u64 size) = 0;
        virtual void ForEachImageIn(VAddr address, u64 size, const std::function<void(ImageId)>& func) = 0;
        /// Takes the image off the lists of the engine (the mark that says a list keeps it current); returns whether it was listed.
        virtual bool ClearListed(ImageId id) = 0;
        /// An announced CPU write of [address, address + size): the image is dirtied unless the GPU wrote it since (`newer` false).
        virtual Announced ApplyAnnounced(ImageId id, VAddr address, u64 size, bool newer) = 0;
        virtual void MarkDeleted(ImageId id) = 0;
        virtual void Free(ImageId id) = 0;
    };

    /// An engine clear of a texture that has no image yet; applied to its first image when the lookup lists it (B11).
    struct Clear {
        bool depth{}, fast{};
        u32 flags{}, stencil{};
        float value[4]{};
        u32 mip{}, first_layer{}, layer_count{}; ///< layer_count 0: every layer
    };
    /// The other answers of an engine texture's lookups (a view of another format or size, a mip or layer of a bigger image), by the request; valid while no image was
    /// registered or removed in the range since (the event ring) (B8.4).
    struct Answer {
        u64 key{};
        ImageId image_id{};
        u64 uid{};
        u64 generation{};
        s32 mip{-1}, slice{-1};
    };
    using ImageList = SmallVector<std::pair<ImageId, u64>, 2>;

    explicit NativeImageIndex(Host& host_) : host{host_} {}

    /// B8.1: the engine announced a CPU write of [addr, addr + size): the images it touches are marked and dirtied.
    void NoteAnnouncedWrite(VAddr addr, size_t size, bool newer);
    /// B8.2: the engine deleted the texture whose memory is [address, address + bytes): its images (same memory, not bigger) are freed (`free`) or marked.
    void NoteTextureDelete(u64 id, VAddr address, u64 bytes, bool free);
    /// B8.4: the engine created a texture at [address, address + bytes): older images in that memory are unlisted (a lookup that gives one out refreshes it).
    void NoteTextureCreate(VAddr address, u64 bytes);
    /// B8.4: the images listed for the engine textures `ids` (their own and their remembered answers) that meet [address, address + size), each once.
    void ImagesIn(const std::vector<u64>& ids, VAddr address, u64 size, SmallVector<ImageId, 8>& out) const;
    /// B8.5: an engine buffer was created (images already in its memory join its list) / deleted (its list goes).
    void NoteBufferCreate(u64 id, VAddr address, u64 bytes);
    void NoteBufferDelete(u64 id) {
        buffer_images.erase(id);
        buffers_known.erase(id);
    }
    /// B8.5: the images the engine buffers meeting [address, address + size) list (each once); false when live buffers do not cover the range whole.
    bool BufferImagesIn(VAddr address, u64 size, SmallVector<ImageId, 8>& out);
    /// B11: the engine texture's own images (its list, registered).
    void ImagesOf(u64 id, SmallVector<ImageId, 4>& out) const {
        if (const auto it = images.find(id); it != images.end()) {
            for (const auto& [image_id, uid] : it->second) {
                if (host.Live(image_id, uid) && host.Registered(image_id)) {
                    out.push_back(image_id);
                }
            }
        }
    }
    void SetPendingClear(u64 id, const Clear& clear) {
        pending_clears[id] = clear;
    }

    /// B8.3: the images of each engine texture (by its id; slot and uid, a stale entry is skipped).
    std::unordered_map<u64, ImageList> images;
    /// B8.4: the other answers of each engine texture's lookups.
    std::unordered_map<u64, SmallVector<Answer, 2>> answers;
    /// B11 step 6, by engine texture id.
    std::unordered_map<u64, Clear> pending_clears;
    /// B8.5, by engine buffer id: the images in its memory.
    std::unordered_map<u64, ImageList> buffer_images;
    /// B8.5: engine buffers whose creation was handled (their memory's older images listed).
    std::unordered_set<u64> buffers_known;

private:
    Host& host;
};

} // namespace Core
