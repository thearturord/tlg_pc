// tlg-native M1: counters of the old pipeline cache / draw capture that the port's frame statistics (videoout/driver.cpp) still read. The native core does not
// count these (shader compiles are in the debug server); they stay at zero until the port statistics are rebuilt for the native path (pc-port-plan.md section 10).
#include <atomic>
#include "common/types.h"

namespace Vulkan {
std::atomic<u64> g_tlg_compile_ns;
std::atomic<u32> g_tlg_compiles;
std::atomic<u64> g_tlg_capture_ns, g_tlg_async_ns, g_tlg_compile_wait_ns;
std::atomic<u64> g_tlg_snapshot_ns, g_tlg_guest_wait_ns;
std::atomic<u32> g_tlg_pending_draws;
} // namespace Vulkan
