// SPDX-License-Identifier: GPL-2.0-or-later
// tlg-native, N7.5: in place buffers. The emulator's guest memory is a pagefile backed section; the runtime keeps one always read/write view of all of it (the "backing", the
// view the host side reads and writes without touching the guest views' page protection). `VK_EXT_external_memory_host` lets the GPU read the pages of that view where they are:
// a buffer bound to such memory sees every CPU write at once, with no copy and no dirty page tracking (tools/hostptr_probe shows this works on the target device for exactly this
// kind of memory, with the 4 KiB import alignment, 1 GB imported in 0.3 s because the driver pins the pages).
//
// The backing is imported in chunks of 256 MiB on first use. `Find` answers where a guest range is: the chunk's buffer and the offset in it, or a null buffer (the range is not
// inside one mapped region, spans two chunks, or the import failed): the caller then takes its ordinary path.

#pragma once

#include <array>
#include <atomic>
#include <mutex>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

class HostImport {
public:
    static constexpr u64 ChunkBytes = 256ull << 20;
    static constexpr u32 MaxChunks = 64;

    explicit HostImport(const Instance& instance);
    ~HostImport();

    struct Window {
        vk::Buffer buffer{};
        u64 offset{0};
        u64 chunk_remaining{0}; ///< bytes from `offset` to the end of the chunk's buffer
    };
    /// A window for [address, address + size) of guest memory; `buffer` is null when there is none.
    Window Find(VAddr address, u64 size);

    [[nodiscard]] bool Available() const noexcept {
        return available;
    }
    [[nodiscard]] u64 ImportedBytes() const noexcept {
        return imported_bytes.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u32 ImportedChunks() const noexcept {
        return imported_chunks.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u32 FailedChunks() const noexcept {
        return failed_chunks.load(std::memory_order_relaxed);
    }

private:
    struct Chunk {
        vk::Buffer buffer{};
        vk::DeviceMemory memory{};
        u64 bytes{0};
        bool tried{false};
    };
    bool ImportChunk(u32 index, Chunk& chunk, u8* backing_base, u64 section_bytes);

    const Instance& instance;
    bool available{false};
    u32 memory_type_bits{0};
    u64 alignment{4096};
    std::array<Chunk, MaxChunks> chunks{};
    std::mutex mutex;
    std::atomic<u64> imported_bytes{0};
    std::atomic<u32> imported_chunks{0}, failed_chunks{0};
};

} // namespace Vulkan
