// tlg C9: see vk_indexed_data.h

#include <cstdlib>

#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_command_recorder.h"
#include "video_core/renderer_vulkan/vk_indexed_data.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

bool IndexedData::RequestedByEnvironment() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_INDEXED_DATA");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}

bool IndexedData::DeviceFits(const Instance& instance) {
    constexpr u32 buffers = 1 + Shader::Indexed::MaxBuffers;
    constexpr u32 images = Shader::Indexed::NumImageKeys * Shader::Indexed::MaxImages;
    constexpr u32 storage_images = 2 * Shader::Indexed::NumImageKeys * Shader::Indexed::MaxStorageImages;
    constexpr u32 samplers = Shader::Indexed::MaxSamplers;
    const auto props = instance.GetPhysicalDevice().getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan12Properties>();
    const auto& p = props.get<vk::PhysicalDeviceVulkan12Properties>();
    const bool fits = p.maxPerStageDescriptorUpdateAfterBindStorageBuffers >= buffers + 8 && p.maxPerStageDescriptorUpdateAfterBindSampledImages >= images + 64 && p.maxPerStageDescriptorUpdateAfterBindStorageImages >= storage_images + 64 &&
                      p.maxPerStageDescriptorUpdateAfterBindSamplers >= samplers + 64 && p.maxDescriptorSetUpdateAfterBindStorageBuffers >= buffers &&
                      p.maxDescriptorSetUpdateAfterBindSampledImages >= images && p.maxDescriptorSetUpdateAfterBindSamplers >= samplers &&
                      p.maxDescriptorSetUpdateAfterBindStorageImages >= storage_images && p.maxPerStageUpdateAfterBindResources >= buffers + images + storage_images + samplers + 256;
    LOG_INFO(Render_Vulkan, "indexed data path: device limits (update after bind, per stage) storage buffers {} sampled images {} storage images {} samplers {} resources {}: {}",
             p.maxPerStageDescriptorUpdateAfterBindStorageBuffers, p.maxPerStageDescriptorUpdateAfterBindSampledImages, p.maxPerStageDescriptorUpdateAfterBindStorageImages,
             p.maxPerStageDescriptorUpdateAfterBindSamplers, p.maxPerStageUpdateAfterBindResources, fits ? "enough" : "NOT enough");
    return fits;
}

IndexedData::IndexedData(const Instance& instance_, Scheduler& scheduler_) : instance{instance_}, scheduler{scheduler_} {
    const vk::Device device = instance.GetDevice();
    if (const char* e = std::getenv("TLG_INDEXED_VERIFY"); e != nullptr && *e != 0) {
        verify_every = static_cast<u32>(std::atoi(e));
    }
    auto setup = [&](u32 index, u32 binding, vk::DescriptorType type, u32 capacity) {
        auto& t = tables[index];
        t.binding = binding;
        t.type = type;
        t.capacity = capacity;
        t.slots.resize(capacity);
        t.free.reserve(capacity);
        for (u32 slot = capacity - 1; slot >= 1; --slot) {
            t.free.push_back(slot);
        }
        t.slots[0].used = true; // slot 0 is the null descriptor (the default sampler for samplers)
    };
    setup(BuffersTable, Shader::Indexed::BufferBinding, vk::DescriptorType::eStorageBuffer, Shader::Indexed::MaxBuffers);
    setup(SamplersTable, Shader::Indexed::SamplerBinding, vk::DescriptorType::eSampler, Shader::Indexed::MaxSamplers);
    for (u32 key = 0; key < Shader::Indexed::NumImageArrays; ++key) {
        const bool storage = key >= Shader::Indexed::NumImageKeys;
        setup(FirstImageTable + key, Shader::Indexed::ImageBinding + key, storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
              storage ? Shader::Indexed::MaxStorageImages : Shader::Indexed::MaxImages);
    }

    std::array<vk::DescriptorSetLayoutBinding, 1 + NumTables> bindings{};
    std::array<vk::DescriptorBindingFlags, 1 + NumTables> binding_flags{};
    const vk::ShaderStageFlags stages = AllGraphicsStageBits | vk::ShaderStageFlagBits::eCompute;
    bindings[0] = {.binding = Shader::Indexed::TableBinding, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = stages};
    binding_flags[0] = vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    for (u32 i = 0; i < NumTables; ++i) {
        bindings[1 + i] = {.binding = tables[i].binding, .descriptorType = tables[i].type, .descriptorCount = tables[i].capacity, .stageFlags = stages};
        binding_flags[1 + i] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    }
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .bindingCount = static_cast<u32>(binding_flags.size()),
        .pBindingFlags = binding_flags.data(),
    };
    const vk::DescriptorSetLayoutCreateInfo layout_ci{
        .pNext = &flags_ci,
        .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [layout_result, created_layout] = device.createDescriptorSetLayoutUnique(layout_ci);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "indexed data: the descriptor set layout failed: {}", vk::to_string(layout_result));
    layout = std::move(created_layout);

    const std::array<vk::DescriptorPoolSize, 4> pool_sizes{{
        {vk::DescriptorType::eStorageBuffer, 1 + Shader::Indexed::MaxBuffers},
        {vk::DescriptorType::eSampledImage, Shader::Indexed::NumImageKeys * Shader::Indexed::MaxImages},
        {vk::DescriptorType::eStorageImage, 2 * Shader::Indexed::NumImageKeys * Shader::Indexed::MaxStorageImages},
        {vk::DescriptorType::eSampler, Shader::Indexed::MaxSamplers},
    }};
    const vk::DescriptorPoolCreateInfo pool_ci{
        .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = 1,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };
    auto [pool_result, created_pool] = device.createDescriptorPoolUnique(pool_ci);
    ASSERT_MSG(pool_result == vk::Result::eSuccess, "indexed data: the descriptor pool failed: {}", vk::to_string(pool_result));
    pool = std::move(created_pool);
    const vk::DescriptorSetLayout set_layout = *layout;
    const vk::DescriptorSetAllocateInfo alloc_info{.descriptorPool = *pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    auto [set_result, sets] = device.allocateDescriptorSets(alloc_info);
    ASSERT_MSG(set_result == vk::Result::eSuccess, "indexed data: the descriptor set failed: {}", vk::to_string(set_result));
    set = sets[0];

    // samplers cannot be null descriptors: slot 0 of the sampler array is a plain sampler (a draw only names it when the array is full)
    const vk::SamplerCreateInfo sampler_ci{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    auto [sampler_result, created_sampler] = device.createSamplerUnique(sampler_ci);
    ASSERT_MSG(sampler_result == vk::Result::eSuccess, "indexed data: the default sampler failed: {}", vk::to_string(sampler_result));
    default_sampler = std::move(created_sampler);

    pending_buffer_infos.reserve(4096);
    pending_image_infos.reserve(4096);
    pending.reserve(4096);
    // slot 0 of every array: the null descriptor (robustness2 nullDescriptor: reads give zeros)
    pending_buffer_infos.push_back({VK_NULL_HANDLE, 0, VK_WHOLE_SIZE});
    pending.push_back({BuffersTable, 0, 0});
    pending_image_infos.push_back({*default_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
    pending.push_back({SamplersTable, 0, 0});
    for (u32 key = 0; key < Shader::Indexed::NumImageArrays; ++key) {
        pending_image_infos.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
        pending.push_back({FirstImageTable + key, 0, static_cast<u32>(pending_image_infos.size() - 1)});
    }
    FlushWrites();
    live.store(this, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "indexed data path: {} buffer slots, {} slots in each of {} sampled image arrays, {} in each of {} storage image arrays, {} sampler slots, verify one draw in {}",
             Shader::Indexed::MaxBuffers, Shader::Indexed::MaxImages, Shader::Indexed::NumImageKeys, Shader::Indexed::MaxStorageImages, 2 * Shader::Indexed::NumImageKeys,
             Shader::Indexed::MaxSamplers, verify_every);
}

IndexedData::~IndexedData() {
    live.store(nullptr, std::memory_order_release);
}

void IndexedData::SetTable(vk::Buffer buffer, u64 size_bytes) {
    if (table_ready) {
        return;
    }
    const vk::DescriptorBufferInfo info{buffer, 0, size_bytes};
    const vk::WriteDescriptorSet write{.dstSet = set,
                                       .dstBinding = Shader::Indexed::TableBinding,
                                       .dstArrayElement = 0,
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .pBufferInfo = &info};
    instance.GetDevice().updateDescriptorSets(write, {});
    table_ready = true;
}

u32 IndexedData::Slot(u32 table, const Key& key, bool& made) {
    auto& t = tables[table];
    const u64 tick = scheduler.CurrentTick();
    const u32 stream = (tick & Scheduler::SideTag) != 0 ? 1u : 0u;
    made = false;
    if (const auto it = t.map.find(key); it != t.map.end()) {
        auto& info = t.slots[it->second];
        if (info.last_tick[stream] != tick) {
            info.last_tick[stream] = tick;
            t.uses.emplace_back(it->second, tick);
        }
        return it->second;
    }
    if (t.free.empty()) {
        stats.exhausted.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u32 slot = t.free.back();
    t.free.pop_back();
    auto& info = t.slots[slot];
    info.key = key;
    info.last_tick[stream] = tick;
    info.last_tick[1 - stream] = 0;
    info.used = true;
    t.map.emplace(key, slot);
    t.uses.emplace_back(slot, tick);
    made = true;
    return slot;
}

u32 IndexedData::BufferSlot(vk::Buffer buffer, u64 offset, u64 range) {
    if (!buffer) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    bool made;
    const u32 slot = Slot(BuffersTable, {reinterpret_cast<u64>(static_cast<VkBuffer>(buffer)), offset, range}, made);
    if (made) {
        pending_buffer_infos.push_back({buffer, offset, range});
        pending.push_back({BuffersTable, slot, static_cast<u32>(pending_buffer_infos.size() - 1)});
        stats.buffer_new.fetch_add(1, std::memory_order_relaxed);
        stats.live_buffer_slots.fetch_add(1, std::memory_order_relaxed);
    } else if (slot != 0) {
        stats.buffer_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return slot;
}

u32 IndexedData::ImageSlot(u32 image_key, vk::ImageView view, vk::ImageLayout layout) {
    if (!view || image_key >= Shader::Indexed::NumImageArrays) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    bool made;
    const u32 table = FirstImageTable + image_key;
    const u32 slot = Slot(table, {reinterpret_cast<u64>(static_cast<VkImageView>(view)), static_cast<u64>(layout), 0}, made);
    if (made) {
        pending_image_infos.push_back({VK_NULL_HANDLE, view, layout});
        pending.push_back({table, slot, static_cast<u32>(pending_image_infos.size() - 1)});
        stats.image_new.fetch_add(1, std::memory_order_relaxed);
        stats.live_image_slots.fetch_add(1, std::memory_order_relaxed);
    } else if (slot != 0) {
        stats.image_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return slot;
}

u32 IndexedData::SamplerSlot(vk::Sampler sampler) {
    if (!sampler) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    bool made;
    const u32 slot = Slot(SamplersTable, {reinterpret_cast<u64>(static_cast<VkSampler>(sampler)), 0, 0}, made);
    if (made) {
        pending_image_infos.push_back({sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
        pending.push_back({SamplersTable, slot, static_cast<u32>(pending_image_infos.size() - 1)});
        stats.sampler_new.fetch_add(1, std::memory_order_relaxed);
        stats.live_sampler_slots.fetch_add(1, std::memory_order_relaxed);
    } else if (slot != 0) {
        stats.sampler_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return slot;
}

bool IndexedData::BufferOf(u32 slot, vk::Buffer& buffer, u64& offset, u64& range) const {
    const auto& t = tables[BuffersTable];
    if (slot == 0 || slot >= t.slots.size() || !t.slots[slot].used) {
        return false;
    }
    const auto& key = t.slots[slot].key;
    buffer = vk::Buffer{reinterpret_cast<VkBuffer>(key.a)};
    offset = key.b;
    range = key.c;
    return true;
}

bool IndexedData::ImageOf(u32 image_key, u32 slot, vk::ImageView& view, vk::ImageLayout& layout) const {
    if (image_key >= Shader::Indexed::NumImageArrays) {
        return false;
    }
    const auto& t = tables[FirstImageTable + image_key];
    if (slot == 0 || slot >= t.slots.size() || !t.slots[slot].used) {
        return false;
    }
    const auto& key = t.slots[slot].key;
    view = vk::ImageView{reinterpret_cast<VkImageView>(key.a)};
    layout = static_cast<vk::ImageLayout>(key.b);
    return true;
}

bool IndexedData::SamplerOf(u32 slot, vk::Sampler& sampler) const {
    const auto& t = tables[SamplersTable];
    if (slot == 0 || slot >= t.slots.size() || !t.slots[slot].used) {
        return false;
    }
    sampler = vk::Sampler{reinterpret_cast<VkSampler>(t.slots[slot].key.a)};
    return true;
}

void IndexedData::FlushWrites() {
    std::scoped_lock lock{mutex};
    if (pending.empty()) {
        return;
    }
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(pending.size());
    for (const auto& p : pending) {
        const auto& t = tables[p.table];
        vk::WriteDescriptorSet write{.dstSet = set, .dstBinding = t.binding, .dstArrayElement = p.slot, .descriptorCount = 1, .descriptorType = t.type};
        if (p.table == BuffersTable) {
            write.pBufferInfo = &pending_buffer_infos[p.info];
        } else {
            write.pImageInfo = &pending_image_infos[p.info];
        }
        writes.push_back(write);
    }
    instance.GetDevice().updateDescriptorSets(writes, {});
    stats.flushes.fetch_add(1, std::memory_order_relaxed);
    stats.written.fetch_add(pending.size(), std::memory_order_relaxed);
    pending.clear();
    pending_buffer_infos.clear();
    pending_image_infos.clear();
}

void IndexedData::Maintain() {
    std::scoped_lock lock{mutex};
    for (u32 index = 0; index < NumTables; ++index) {
        auto& t = tables[index];
        while (!t.uses.empty() && scheduler.IsFree(t.uses.front().second)) {
            const auto [slot, tick] = t.uses.front();
            t.uses.pop_front();
            auto& info = t.slots[slot];
            // the slot is free when the GPU is done with the last draw of each stream that named it (the last of the other stream has its own entry in the queue)
            if (!info.used || (info.last_tick[0] != 0 && !scheduler.IsFree(info.last_tick[0])) || (info.last_tick[1] != 0 && !scheduler.IsFree(info.last_tick[1]))) {
                continue;
            }
            if (const auto it = t.map.find(info.key); it != t.map.end() && it->second == slot) {
                t.map.erase(it);
            }
            info.used = false;
            t.free.push_back(slot);
            if (index == BuffersTable) {
                stats.buffer_freed.fetch_add(1, std::memory_order_relaxed);
                stats.live_buffer_slots.fetch_sub(1, std::memory_order_relaxed);
            } else if (index == SamplersTable) {
                stats.sampler_freed.fetch_add(1, std::memory_order_relaxed);
                stats.live_sampler_slots.fetch_sub(1, std::memory_order_relaxed);
            } else {
                stats.image_freed.fetch_add(1, std::memory_order_relaxed);
                stats.live_image_slots.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }
}

void IndexedData::ForgetHandle(u32 first_table, u32 last_table, u64 handle) {
    std::scoped_lock lock{mutex};
    for (u32 index = first_table; index <= last_table; ++index) {
        auto& map = tables[index].map;
        for (auto it = map.begin(); it != map.end();) {
            it = it->first.a == handle ? map.erase(it) : std::next(it);
        }
    }
}

void IndexedData::NoteBufferDestroyed(vk::Buffer buffer) noexcept {
    if (IndexedData* self = Get(); self != nullptr && buffer) {
        self->ForgetHandle(BuffersTable, BuffersTable, reinterpret_cast<u64>(static_cast<VkBuffer>(buffer)));
    }
}

void IndexedData::NoteViewDestroyed(vk::ImageView view) noexcept {
    if (IndexedData* self = Get(); self != nullptr && view) {
        self->ForgetHandle(FirstImageTable, NumTables - 1, reinterpret_cast<u64>(static_cast<VkImageView>(view)));
    }
}

void IndexedData::NoteSamplerDestroyed(vk::Sampler sampler) noexcept {
    if (IndexedData* self = Get(); self != nullptr && sampler) {
        self->ForgetHandle(SamplersTable, SamplersTable, reinterpret_cast<u64>(static_cast<VkSampler>(sampler)));
    }
}

void IndexedData::BindSet(const CommandRecorder& cmdbuf, vk::PipelineBindPoint bind_point, vk::PipelineLayout pipeline_layout) {
    const u64 tick = scheduler.CurrentTick();
    const u32 index = bind_point == vk::PipelineBindPoint::eCompute ? 1u : 0u;
    if (tick == bound_tick[index]) {
        return;
    }
    cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, Shader::Indexed::GlobalSet, set, {});
    bound_tick[index] = tick;
    stats.set_binds.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Vulkan
