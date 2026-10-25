// SPDX-License-Identifier: GPL-2.0-or-later
// tlg-native, N7.5: see vk_host_import.h.

#include <algorithm>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_host_import.h"
#include "video_core/renderer_vulkan/vk_instance.h"

extern "C" {
// runtime_memory.c: where [address, address + size) lives in the backing view (one mapped region), the view's base address and the section's size; 0 when it is not inside one.
int runtime_memory_backing_window(uintptr_t address, uint64_t size, void** base, uint64_t* offset, uint64_t* section_bytes);
}

namespace Vulkan {

HostImport::HostImport(const Instance& instance_) : instance{instance_} {
    if (!instance.IsExternalMemoryHostSupported()) {
        LOG_INFO(Render_Vulkan, "in place buffers: VK_EXT_external_memory_host is not available");
        return;
    }
    auto properties = instance.GetPhysicalDevice().getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
    alignment = properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>().minImportedHostPointerAlignment;
    available = alignment != 0 && ChunkBytes % alignment == 0;
    LOG_INFO(Render_Vulkan, "in place buffers: host pointer import available, alignment {}", alignment);
}

HostImport::~HostImport() {
    const auto device = instance.GetDevice();
    for (auto& chunk : chunks) {
        if (chunk.buffer) {
            device.destroyBuffer(chunk.buffer);
        }
        if (chunk.memory) {
            device.freeMemory(chunk.memory);
        }
    }
}

bool HostImport::ImportChunk(u32 index, Chunk& chunk, u8* backing_base, u64 section_bytes) {
    const u64 begin = u64{index} * ChunkBytes;
    chunk.bytes = std::min(ChunkBytes, section_bytes - begin);
    chunk.bytes &= ~(alignment - 1);
    if (chunk.bytes == 0) {
        return false;
    }
    u8* pointer = backing_base + begin;
    const auto device = instance.GetDevice();
    const auto host_properties = device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, pointer);
    if (host_properties.result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "in place buffers: chunk {} (backing {:#x}): host pointer properties failed ({})", index, begin, vk::to_string(host_properties.result));
        return false;
    }
    const vk::ExternalMemoryBufferCreateInfo external_info{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
    };
    const vk::BufferCreateInfo buffer_info{
        .pNext = &external_info,
        .size = chunk.bytes,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    auto buffer = device.createBuffer(buffer_info);
    if (buffer.result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "in place buffers: chunk {}: buffer creation failed ({})", index, vk::to_string(buffer.result));
        return false;
    }
    const auto requirements = device.getBufferMemoryRequirements(buffer.value);
    const u32 bits = host_properties.value.memoryTypeBits & requirements.memoryTypeBits;
    const auto& memory_properties = instance.GetMemoryProperties();
    u32 type = ~0u;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            type = i;
            break;
        }
    }
    if (type == ~0u) {
        LOG_ERROR(Render_Vulkan, "in place buffers: chunk {}: no host visible memory type for the imported pointer", index);
        device.destroyBuffer(buffer.value);
        return false;
    }
    const vk::ImportMemoryHostPointerInfoEXT import_info{
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        .pHostPointer = pointer,
    };
    const vk::MemoryAllocateInfo allocate_info{
        .pNext = &import_info,
        .allocationSize = (requirements.size + alignment - 1) & ~(alignment - 1),
        .memoryTypeIndex = type,
    };
    auto memory = device.allocateMemory(allocate_info);
    if (memory.result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "in place buffers: chunk {} (backing {:#x}, {} MiB): import failed ({})", index, begin, chunk.bytes >> 20, vk::to_string(memory.result));
        device.destroyBuffer(buffer.value);
        return false;
    }
    if (device.bindBufferMemory(buffer.value, memory.value, 0) != vk::Result::eSuccess) {
        device.destroyBuffer(buffer.value);
        device.freeMemory(memory.value);
        return false;
    }
    chunk.buffer = buffer.value;
    chunk.memory = memory.value;
    imported_bytes.fetch_add(chunk.bytes, std::memory_order_relaxed);
    imported_chunks.fetch_add(1, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "in place buffers: imported chunk {} (backing {:#x}, {} MiB)", index, begin, chunk.bytes >> 20);
    return true;
}

HostImport::Window HostImport::Find(VAddr address, u64 size) {
    if (!available || size == 0) {
        return {};
    }
    void* base = nullptr;
    u64 offset = 0, section_bytes = 0;
    if (runtime_memory_backing_window(address, size, &base, &offset, &section_bytes) == 0) {
        return {};
    }
    const u32 index = static_cast<u32>(offset / ChunkBytes);
    if (index >= MaxChunks || (offset + size - 1) / ChunkBytes != index) {
        return {}; // a range across two chunks stays on the ordinary path
    }
    std::scoped_lock lock{mutex};
    Chunk& chunk = chunks[index];
    if (!chunk.tried) {
        chunk.tried = true;
        if (!ImportChunk(index, chunk, static_cast<u8*>(base), section_bytes)) {
            failed_chunks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const u64 in_chunk = offset - u64{index} * ChunkBytes;
    if (!chunk.buffer || in_chunk + size > chunk.bytes) {
        return {};
    }
    return {chunk.buffer, in_chunk, chunk.bytes - in_chunk};
}

} // namespace Vulkan
