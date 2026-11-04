// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_wait_registers.h"

#include <array>
#include <atomic>

#include <fmt/format.h>

#include "native/native_debug_server.h"

namespace Core::NativeWaitRegisters {

namespace {
constexpr u32 NumRegisters = 0xD000;
std::array<std::atomic<u32>, NumRegisters> polls{};
} // namespace

u32 Read(u32 reg) {
    if (reg < NumRegisters && polls[reg].fetch_add(1, std::memory_order_relaxed) == 0) {
        RareEvents::Note(fmt::format("register_wait_polled_{:#x}", reg), "a WAIT_REG_MEM on a register: the executor keeps no register file and returned 0");
    }
    return 0;
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    for (u32 i = 0; i < NumRegisters; ++i) {
        if (const u32 n = polls[i].load(std::memory_order_relaxed)) {
            j[fmt::format("{:#x}", i)] = n;
        }
    }
    return j;
}

} // namespace Core::NativeWaitRegisters
