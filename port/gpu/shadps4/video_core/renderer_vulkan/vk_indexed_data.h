// tlg C9: the indexed data path of the consumer (documents/tlg/native plan/stages/c9-indexed-data.md).
// Graphics shaders find the guest buffers, sampled images and samplers of a draw through one global descriptor set instead of push descriptors per draw: descriptor arrays (slots), and a
// row of slot numbers per draw in a table buffer, addressed by one value in the push constants. A slot stands for the exact descriptor the legacy path would push (buffer, offset,
// range; image view and layout; sampler), so the range checks of the device (robustBufferAccess2) stay what they were; equal descriptors share a slot; a slot lives until the GPU is
// done with the last draw that named it.

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/resource.h"
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
    /// Called by the destructors of the buffers, image views and samplers: the slots that name the object are not looked up any more.
    static void NoteBufferDestroyed(vk::Buffer buffer) noexcept;
    static void NoteViewDestroyed(vk::ImageView view) noexcept;
    static void NoteSamplerDestroyed(vk::Sampler sampler) noexcept;

    /// Does the device have the room for the descriptor arrays (the limits of the update after bind descriptors).
    [[nodiscard]] static bool DeviceFits(const Instance& instance);

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
    /// The slot of a sampled image descriptor in the array of the image key (0: the null descriptor).
    [[nodiscard]] u32 ImageSlot(u32 image_key, vk::ImageView view, vk::ImageLayout layout);
    /// The slot of a sampler.
    [[nodiscard]] u32 SamplerSlot(vk::Sampler sampler);
    /// What a slot stands for, for the verify (false: not a live slot).
    [[nodiscard]] bool BufferOf(u32 slot, vk::Buffer& buffer, u64& offset, u64& range) const;
    [[nodiscard]] bool ImageOf(u32 image_key, u32 slot, vk::ImageView& view, vk::ImageLayout& layout) const;
    [[nodiscard]] bool SamplerOf(u32 slot, vk::Sampler& sampler) const;

    /// Writes the descriptors of the slots made since the last call (the draw's, before it is submitted).
    void FlushWrites();
    /// Frees the slots of draws the GPU has finished.
    void Maintain();
    /// Binds the global set to the command buffer that records now, once per command buffer.
    void BindSet(const CommandRecorder& cmdbuf, vk::PipelineLayout pipeline_layout);

    struct Stats {
        std::atomic<u64> draws{0}, rows_entries{0}, buffer_hits{0}, buffer_new{0}, buffer_freed{0}, exhausted{0}, flushes{0}, written{0}, set_binds{0};
        std::atomic<u64> image_hits{0}, image_new{0}, image_freed{0}, sampler_hits{0}, sampler_new{0}, sampler_freed{0};
        std::atomic<u64> verified{0}, verify_differences{0}, key_differences{0};
        std::atomic<u32> live_buffer_slots{0}, live_image_slots{0}, live_sampler_slots{0};
    };
    Stats stats;
    [[nodiscard]] u32 VerifyEvery() const noexcept {
        return verify_every;
    }

private:
    struct Key {
        u64 a, b, c;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            return static_cast<size_t>((k.a * 0x9E3779B97F4A7C15ull) ^ (k.b * 0xC2B2AE3D27D4EB4Full) ^ (k.c * 0x165667B19E3779F9ull));
        }
    };
    struct SlotInfo {
        Key key{};
        u64 last_tick{0};
        bool used{false};
    };
    /// One descriptor array: the buffers, the samplers, and one per image key.
    struct Table {
        u32 binding{};
        vk::DescriptorType type{};
        u32 capacity{};
        std::unordered_map<Key, u32, KeyHash> map;
        std::vector<SlotInfo> slots;
        std::vector<u32> free;
        std::deque<std::pair<u32, u64>> uses; // (slot, tick of the use), oldest first
    };
    static constexpr u32 BuffersTable = 0;
    static constexpr u32 SamplersTable = 1;
    static constexpr u32 FirstImageTable = 2;
    static constexpr u32 NumTables = FirstImageTable + Shader::Indexed::NumImageKeys;
    struct PendingWrite {
        u32 table, slot, info;
    };

    /// Finds or makes the slot of a descriptor in a table; `made` tells that a write must be recorded. Caller holds the mutex.
    u32 Slot(u32 table, const Key& key, bool& made);
    void ForgetHandle(u32 first_table, u32 last_table, u64 handle);

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueDescriptorSetLayout layout;
    vk::UniqueDescriptorPool pool;
    vk::UniqueSampler default_sampler;
    vk::DescriptorSet set{};
    bool table_ready{false};
    u32 verify_every{64};

    std::mutex mutex;
    std::array<Table, NumTables> tables;

    std::vector<vk::DescriptorBufferInfo> pending_buffer_infos;
    std::vector<vk::DescriptorImageInfo> pending_image_infos;
    std::vector<PendingWrite> pending;
    u64 bound_tick{~0ull};

    static inline std::atomic<IndexedData*> live{nullptr};
};

} // namespace Vulkan
