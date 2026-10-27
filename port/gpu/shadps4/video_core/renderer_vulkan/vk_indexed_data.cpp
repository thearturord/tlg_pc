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

IndexedData::IndexedData(const Instance& instance_, Scheduler& scheduler_) : instance{instance_}, scheduler{scheduler_} {
    const vk::Device device = instance.GetDevice();
    if (const char* e = std::getenv("TLG_INDEXED_VERIFY"); e != nullptr && *e != 0) {
        verify_every = static_cast<u32>(std::atoi(e));
    }
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {.binding = Shader::Indexed::TableBinding, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = AllGraphicsStageBits},
        {.binding = Shader::Indexed::BufferBinding,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .descriptorCount = Shader::Indexed::MaxBuffers,
         .stageFlags = AllGraphicsStageBits},
    }};
    const std::array<vk::DescriptorBindingFlags, 2> binding_flags{
        vk::DescriptorBindingFlagBits::eUpdateAfterBind,
        vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
    };
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

    const vk::DescriptorPoolSize pool_size{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1 + Shader::Indexed::MaxBuffers};
    const vk::DescriptorPoolCreateInfo pool_ci{
        .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    auto [pool_result, created_pool] = device.createDescriptorPoolUnique(pool_ci);
    ASSERT_MSG(pool_result == vk::Result::eSuccess, "indexed data: the descriptor pool failed: {}", vk::to_string(pool_result));
    pool = std::move(created_pool);
    const vk::DescriptorSetLayout set_layout = *layout;
    const vk::DescriptorSetAllocateInfo alloc_info{.descriptorPool = *pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    auto [set_result, sets] = device.allocateDescriptorSets(alloc_info);
    ASSERT_MSG(set_result == vk::Result::eSuccess, "indexed data: the descriptor set failed: {}", vk::to_string(set_result));
    set = sets[0];

    buffer_slots.resize(Shader::Indexed::MaxBuffers);
    buffer_free.reserve(Shader::Indexed::MaxBuffers);
    for (u32 slot = Shader::Indexed::MaxBuffers - 1; slot >= 1; --slot) {
        buffer_free.push_back(slot);
    }
    pending_infos.reserve(4096);
    pending_writes.reserve(4096);
    // slot 0: the null descriptor (robustness2 nullDescriptor: reads give zeros)
    pending_infos.push_back({VK_NULL_HANDLE, 0, VK_WHOLE_SIZE});
    pending_writes.push_back({.dstSet = set, .dstBinding = Shader::Indexed::BufferBinding, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer});
    buffer_slots[0].used = true;
    FlushWrites();
    live.store(this, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "indexed data path: {} buffer slots, verify one draw in {}", Shader::Indexed::MaxBuffers, verify_every);
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

u32 IndexedData::BufferSlot(vk::Buffer buffer, u64 offset, u64 range) {
    if (!buffer) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    const u64 tick = scheduler.CurrentTick();
    const BufferKey key{reinterpret_cast<u64>(static_cast<VkBuffer>(buffer)), offset, range};
    if (const auto it = buffer_map.find(key); it != buffer_map.end()) {
        auto& info = buffer_slots[it->second];
        if (info.last_tick != tick) {
            info.last_tick = tick;
            buffer_uses.emplace_back(it->second, tick);
        }
        stats.buffer_hits.fetch_add(1, std::memory_order_relaxed);
        return it->second;
    }
    if (buffer_free.empty()) {
        stats.exhausted.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u32 slot = buffer_free.back();
    buffer_free.pop_back();
    auto& info = buffer_slots[slot];
    info.key = key;
    info.last_tick = tick;
    info.used = true;
    buffer_map.emplace(key, slot);
    buffer_uses.emplace_back(slot, tick);
    pending_infos.push_back({buffer, offset, range});
    pending_writes.push_back({.dstSet = set, .dstBinding = Shader::Indexed::BufferBinding, .dstArrayElement = slot, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer});
    stats.buffer_new.fetch_add(1, std::memory_order_relaxed);
    stats.live_buffer_slots.fetch_add(1, std::memory_order_relaxed);
    return slot;
}

bool IndexedData::BufferOf(u32 slot, vk::Buffer& buffer, u64& offset, u64& range) const {
    if (slot == 0 || slot >= buffer_slots.size() || !buffer_slots[slot].used) {
        return false;
    }
    const auto& key = buffer_slots[slot].key;
    buffer = vk::Buffer{reinterpret_cast<VkBuffer>(key.buffer)};
    offset = key.offset;
    range = key.range;
    return true;
}

void IndexedData::FlushWrites() {
    std::scoped_lock lock{mutex};
    if (pending_writes.empty()) {
        return;
    }
    for (size_t i = 0; i < pending_writes.size(); ++i) {
        pending_writes[i].pBufferInfo = &pending_infos[i];
    }
    instance.GetDevice().updateDescriptorSets(pending_writes, {});
    stats.flushes.fetch_add(1, std::memory_order_relaxed);
    stats.written.fetch_add(pending_writes.size(), std::memory_order_relaxed);
    pending_writes.clear();
    pending_infos.clear();
}

void IndexedData::Maintain() {
    std::scoped_lock lock{mutex};
    while (!buffer_uses.empty() && scheduler.IsFree(buffer_uses.front().second)) {
        const auto [slot, tick] = buffer_uses.front();
        buffer_uses.pop_front();
        auto& info = buffer_slots[slot];
        if (!info.used || info.last_tick != tick) {
            continue; // used again since
        }
        if (const auto it = buffer_map.find(info.key); it != buffer_map.end() && it->second == slot) {
            buffer_map.erase(it);
        }
        info.used = false;
        buffer_free.push_back(slot);
        stats.buffer_freed.fetch_add(1, std::memory_order_relaxed);
        stats.live_buffer_slots.fetch_sub(1, std::memory_order_relaxed);
    }
}

void IndexedData::NoteBufferDestroyed(vk::Buffer buffer) noexcept {
    IndexedData* self = Get();
    if (self == nullptr || !buffer) {
        return;
    }
    std::scoped_lock lock{self->mutex};
    const u64 handle = reinterpret_cast<u64>(static_cast<VkBuffer>(buffer));
    for (auto it = self->buffer_map.begin(); it != self->buffer_map.end();) {
        it = it->first.buffer == handle ? self->buffer_map.erase(it) : std::next(it);
    }
}

void IndexedData::BindSet(const CommandRecorder& cmdbuf, vk::PipelineLayout pipeline_layout) {
    const u64 tick = scheduler.CurrentTick();
    if (tick == bound_tick) {
        return;
    }
    cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, Shader::Indexed::GlobalSet, set, {});
    bound_tick = tick;
    stats.set_binds.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Vulkan
