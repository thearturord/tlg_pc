// tlg C9: the indexed data path of the consumer (documents/tlg/native plan/stages/c9-indexed-data.md).
// Graphics shaders find the guest buffers of a draw through one global descriptor set instead of push descriptors per draw: a descriptor array of storage buffers (slots), and a
// row of slot numbers per draw in a table buffer, addressed by one value in the push constants. A slot stands for the exact descriptor the legacy path would push (buffer, offset,
// range), so the range checks of the device (robustBufferAccess2) stay what they were; equal descriptors share a slot; a slot lives until the GPU is done with the last draw
// that named it.

#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;
class CommandRecorder;

class IndexedData {
public:
    IndexedData(const Instance& instance, Scheduler& scheduler);
    ~IndexedData();

    /// `TLG_INDEXED_DATA=1` (development switch, off by default).
    [[nodiscard]] static bool RequestedByEnvironment();
    /// The live object, or null (switch off or the device lacks a feature).
    [[nodiscard]] static IndexedData* Get() noexcept {
        return live.load(std::memory_order_acquire);
    }
    /// Called by the buffer destructor: the slots that name the buffer are not looked up any more.
    static void NoteBufferDestroyed(vk::Buffer buffer) noexcept;

    [[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept {
        return *layout;
    }
    /// The table buffer (the stream buffer the rows are copied into) is named once, at the first draw.
    void SetTable(vk::Buffer buffer, u64 size_bytes);
    [[nodiscard]] bool TableReady() const noexcept {
        return table_ready;
    }

    /// The slot of a buffer descriptor (0: the null descriptor). Records the write when the slot is new.
    [[nodiscard]] u32 BufferSlot(vk::Buffer buffer, u64 offset, u64 range);
    /// What a slot stands for, for the verify (false: not a live slot).
    [[nodiscard]] bool BufferOf(u32 slot, vk::Buffer& buffer, u64& offset, u64& range) const;

    /// Writes the descriptors of the slots made since the last call (the draw's, before it is submitted).
    void FlushWrites();
    /// Frees the slots of draws the GPU has finished.
    void Maintain();
    /// Binds the global set to the command buffer that records now, once per command buffer.
    void BindSet(const CommandRecorder& cmdbuf, vk::PipelineLayout pipeline_layout);

    struct Stats {
        std::atomic<u64> draws{0}, rows_entries{0}, buffer_hits{0}, buffer_new{0}, buffer_freed{0}, exhausted{0}, flushes{0}, written{0}, set_binds{0};
        std::atomic<u64> verified{0}, verify_differences{0};
        std::atomic<u32> live_buffer_slots{0};
    };
    Stats stats;
    [[nodiscard]] u32 VerifyEvery() const noexcept {
        return verify_every;
    }

private:
    struct BufferKey {
        u64 buffer, offset, range;
        bool operator==(const BufferKey&) const = default;
    };
    struct BufferKeyHash {
        size_t operator()(const BufferKey& k) const noexcept {
            return static_cast<size_t>((k.buffer * 0x9E3779B97F4A7C15ull) ^ (k.offset * 0xC2B2AE3D27D4EB4Full) ^ (k.range * 0x165667B19E3779F9ull));
        }
    };
    struct BufferSlotInfo {
        BufferKey key{};
        u64 last_tick{0};
        bool used{false};
    };

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueDescriptorSetLayout layout;
    vk::UniqueDescriptorPool pool;
    vk::DescriptorSet set{};
    bool table_ready{false};
    u32 verify_every{64};

    std::mutex mutex;
    std::unordered_map<BufferKey, u32, BufferKeyHash> buffer_map;
    std::vector<BufferSlotInfo> buffer_slots;
    std::vector<u32> buffer_free;
    std::deque<std::pair<u32, u64>> buffer_uses; // (slot, tick of the use), oldest first

    std::vector<vk::DescriptorBufferInfo> pending_infos;
    std::vector<vk::WriteDescriptorSet> pending_writes;
    u64 bound_tick{~0ull};

    static inline std::atomic<IndexedData*> live{nullptr};
};

} // namespace Vulkan
