// tlg-native: where the guest's heap lies in the host address space. The native render path reads the engine's objects through pointers it must not trust,
// and accepts only pointers into the heap. shadPS4 keeps the game's memory at 8 to 16 GiB (wide: up to 64 GiB); the port's runtime maps it from 64 GiB.
#pragma once
#include "common/types.h"

namespace Core {
struct GuestHeapRange {
    u64 heap_low = 0x200000000ull;  ///< engine objects live in [heap_low, heap_high)
    u64 heap_high = 0x400000000ull;
    u64 wide_low = 0x200000000ull;  ///< any guest data (tables, command buffers, records) lives in [wide_low, wide_high)
    u64 wide_high = 0x1000000000ull;
};
inline GuestHeapRange g_guest_heap;
} // namespace Core

inline bool IsHeapPointer(u64 pointer) {
    return pointer >= Core::g_guest_heap.heap_low && pointer < Core::g_guest_heap.heap_high;
}
inline bool IsGuestAddress(u64 address) {
    return address >= Core::g_guest_heap.wide_low && address < Core::g_guest_heap.wide_high;
}
