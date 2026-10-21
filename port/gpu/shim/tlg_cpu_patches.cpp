// tlg-native: executable memory near the game's code for the guest hook trampolines (core/guest_hooks.cpp).
#include <cstdint>
#include <mutex>
#include "core/cpu_patches.h"
#include "common/memory_patcher.h"
#include "common/logging/log.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace MemoryPatcher {
std::string g_game_serial = "UNKNOWN"; // set by tlggpu_init; names the per game config files (core/guest_hooks.cpp)
}

// tlg-native M1: the frame sequence capture of the old emulator (renderdoc.cpp) is not part of the port; the debug server's /capture answers empty.
#include "video_core/renderdoc.h"
namespace VideoCore {
void RequestFrameSequence(u32, float, float, float, float) {}
u32 FrameSequenceRemaining() {
    return 0;
}
} // namespace VideoCore

namespace Core {
namespace {
constexpr size_t AreaSize = 4 * 1024 * 1024;
constexpr uintptr_t Reach = 0x70000000; // below the 2 GB limit of rel32 so the whole area stays reachable
std::mutex area_mutex;
uint8_t* area_base = nullptr;
size_t area_used = 0;

uint8_t* AllocateNear(const void* code_ptr) {
#ifdef _WIN32
    const uintptr_t anchor = reinterpret_cast<uintptr_t>(code_ptr) & ~uintptr_t(0xffff);
    for (uintptr_t delta = 0x10000; delta < Reach; delta += 0x10000) {
        for (const uintptr_t candidate : {anchor + Reach - delta, anchor - Reach + delta}) {
            if (candidate < 0x10000) {
                continue;
            }
            void* p = VirtualAlloc(reinterpret_cast<void*>(candidate), AreaSize, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
            if (p) {
                return static_cast<uint8_t*>(p);
            }
        }
    }
#else
    (void)code_ptr;
#endif
    return nullptr;
}
} // namespace

bool EmitInTrampolineArea(const void* code_ptr, const std::function<bool(Xbyak::CodeGenerator&)>& emit) {
    std::scoped_lock lock{area_mutex};
    if (!area_base) {
        area_base = AllocateNear(code_ptr);
        if (!area_base) {
            LOG_ERROR(Core, "guest hooks: no executable memory near the game's code");
            return false;
        }
    }
    Xbyak::CodeGenerator gen(AreaSize - area_used, area_base + area_used);
    try {
        if (emit(gen)) {
            area_used += (gen.getSize() + 15) & ~size_t(15);
            return true;
        }
    } catch (const Xbyak::Error&) {
    }
    return false;
}
} // namespace Core
