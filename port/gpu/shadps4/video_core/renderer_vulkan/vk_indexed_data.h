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

#include <absl/container/flat_hash_map.h>

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;

/// The census of the pipelines' stages, with or without the indexed path: stages, mip level images, atomic images, shared memory buffers.
struct IndexedCensus {
    std::atomic<u64> stages{0}, mip_run_images{0}, atomic_images{0}, shared_memory_buffers{0};
};
inline IndexedCensus g_indexed_census;
class CommandRecorder;

class IndexedData {
public:
    IndexedData(const Instance& instance, Scheduler& scheduler);
    ~IndexedData();

    /// `TLG_INDEXED_DATA=1` (development switch, off by default).
    [[nodiscard]] static bool RequestedByEnvironment();
    /// The live object, or null (switch off or the device lacks a feature).
    /// P4 (`TLG_INDEXED_FLUSH_PER_DRAW=1` goes back to a flush per draw): the pending writes and the release of slots run once per submission
    [[nodiscard]] static bool FlushPerSubmit() noexcept {
        static const bool on = [] {
            const char* e = std::getenv("TLG_INDEXED_FLUSH_PER_DRAW");
            return e == nullptr || *e != '1';
        }();
        return on;
    }
    [[nodiscard]] static IndexedData* Get() noexcept {
        return live.load(std::memory_order_acquire);
    }
    /// A pipeline was made (new or preloaded) with this stage: counts the rare kinds of resources of indexed shaders (the scenes of a test may never run them).
    static void NoteShaderKinds(const Shader::Info& info) noexcept;

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
    /// The slot of an image descriptor in the array of the array key (0: the null descriptor).
    [[nodiscard]] u32 ImageSlot(u32 image_key, vk::ImageView view, vk::ImageLayout layout);
    /// The first slot of a run of consecutive slots holding the images of the mip levels of one image (the array of a storage image array key); the null run when the images are null.
    [[nodiscard]] u32 ImageRun(u32 image_key, const vk::DescriptorImageInfo* infos, u32 count);
    [[nodiscard]] u32 NullRun(u32 image_key) const noexcept;
    [[nodiscard]] bool RunOf(u32 image_key, u32 base, u32 index, vk::ImageView& view, vk::ImageLayout& layout) const;
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
    /// Binds the global set to the command buffer that records now, once per command buffer and bind point.
    void BindSet(const CommandRecorder& cmdbuf, vk::PipelineBindPoint bind_point, vk::PipelineLayout pipeline_layout);

    struct Run {
        u32 base{}, count{};
        u64 hash{};
        std::vector<std::pair<u64, u64>> items; // (view, layout) of each level
        u64 last_tick[2]{0, 0};
        bool active{false};
    };
    struct Stats {
        std::atomic<u64> draws{0}, rows_entries{0}, buffer_hits{0}, buffer_new{0}, buffer_freed{0}, exhausted{0}, flushes{0}, written{0}, set_binds{0};
        std::atomic<u64> image_hits{0}, image_new{0}, image_freed{0}, sampler_hits{0}, sampler_new{0}, sampler_freed{0}, run_hits{0}, run_new{0}, run_freed{0};
        std::atomic<u64> verified{0}, verify_differences{0}, key_differences{0}, legacy_writes{0}, compute_dispatches{0}, legacy_flat{0}, legacy_clip{0}, legacy_lds{0}, legacy_images{0}, flat_dwords{0};
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
        u64 last_tick[2]{0, 0}; // the last draw that named the slot, in the graphics stream and in the asynchronous stream (the slot lives until both are done)
        bool used{false};
    };
    /// One descriptor array: the buffers, the samplers, and one per image key.
    struct Table {
        u32 binding{};
        vk::DescriptorType type{};
        u32 capacity{};
        absl::flat_hash_map<Key, u32, KeyHash> map; ///< P3: flat (about 230 k new buffer slots a second, the key holds the offset)
        std::vector<SlotInfo> slots;
        std::vector<u32> free;
        std::deque<std::pair<u32, u64>> uses; // (slot, tick of the use), oldest first
        // the runs of consecutive slots (storage image arrays): the last RunSlots slots
        u32 run_start{0};
        std::vector<Run> runs;
        std::vector<u32> run_free_ids;
        std::unordered_map<u64, u32> run_map; // hash of the levels -> index in runs
        std::vector<u8> run_bits;             // slot used, from run_start
        std::deque<std::pair<u32, u64>> run_uses;
    };
    static constexpr u32 BuffersTable = 0;
    static constexpr u32 SamplersTable = 1;
    static constexpr u32 FirstImageTable = 2;
    static constexpr u32 NumTables = FirstImageTable + Shader::Indexed::NumImageArrays;
    struct PendingWrite {
        u32 table, slot, info;
    };

    /// Finds or makes the slot of a descriptor in a table; `made` tells that a write must be recorded. Caller holds the mutex.
    u32 Slot(u32 table, const Key& key, bool& made);
    void ForgetHandle(u32 first_table, u32 last_table, u64 handle);
    void MaintainRuns(Table& t);

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
    u64 bound_tick[2]{~0ull, ~0ull}; // graphics, compute

    static inline std::atomic<IndexedData*> live{nullptr};
};

} // namespace Vulkan
