// SPDX-License-Identifier: GPL-2.0-or-later
// 120 fps game timing (documents/tlg/native plan/pc-port/game-timing-120fps.md): the game's own code advances two kinds of effects by a constant 1/30 s per frame, which is right at
// 30 fps and runs four times too fast at 120. The constant is read at four places; each is followed by a mid hook (core/guest_hooks.h, mode "mid") that multiplies the register that
// holds the constant by (frame time * 30), so the step is the real frame time and stays exactly 1 at 30 fps.
//
//   env_phase_dt   0x01e0f071  FUN_01e0eec0: phase = fract(phase + rate * (time scale * 1/30))        xmm1 = time scale (the constant is multiplied in the next instruction)
//   particle_dt_a  0x01bcb5d1  FUN_01bcac40: count = max(1/30 * emission rate, 0) + carry               xmm0 = the constant
//   particle_dt_b  0x01bcbf34  FUN_01bcac40: the same, the other emission mode                           xmm0
//   particle_dt_c  0x01bcb790  FUN_01bcac40: the spread of the new particles over the frame (1/30 * ..)  xmm6
//
// The frame time is the engine clock's raw frame time (`[0x069a1b40 (ghidra)] + 0x190`, written by FUN_01d16480 every frame, before smoothing, the snap and the pause / slow motion
// scale), so a paused game keeps its environment animating as the console does. Switch: `TLG_GAME_TIME=0` turns it off (the factor is 1).

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <nlohmann/json.hpp>

#include "common/types.h"
#include "native/native_debug_server.h"
#include "native/native_game_time.h"
#include "native/native_hooks.h"

namespace Core::NativeGameTime {

namespace {
constexpr u64 ClockSlotImageOffset = 0x069a1b40 - 0x1000000;
constexpr u32 RawFrameTimeOffset = 0x190;
constexpr float ConsoleStep = 1.0f / 30.0f;

std::atomic<u64> calls[4]{}, fallbacks{0};
std::atomic<u32> lua_dt_bits{0}, lua_edt_bits{0};
std::atomic<u32> last_dt_bits{0}, min_factor_bits{0x7f7fffff}, max_factor_bits{0};

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GAME_TIME");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

/// dt * 30 for this frame (1 when it cannot be read or is not a plausible frame time).
float Factor() {
    const u64 image = GuestHooks::EbootBase();
    if (!Enabled() || image == 0) {
        return 1.0f;
    }
    const u64 clock = *reinterpret_cast<const u64*>(image + ClockSlotImageOffset);
    if (clock == 0) {
        fallbacks.fetch_add(1, std::memory_order_relaxed);
        return 1.0f;
    }
    float dt;
    std::memcpy(&dt, reinterpret_cast<const void*>(clock + RawFrameTimeOffset), sizeof(dt));
    if (!(dt > 0.0001f && dt <= 0.1f)) {
        fallbacks.fetch_add(1, std::memory_order_relaxed);
        return 1.0f;
    }
    const float factor = dt / ConsoleStep;
    u32 bits;
    // the two values the Lua ticks get as (dt, edt): needed to pick the input of the Lua frame factor
    std::memcpy(&bits, reinterpret_cast<const void*>(clock + 0x19c), sizeof(bits));
    lua_dt_bits.store(bits, std::memory_order_relaxed);
    std::memcpy(&bits, reinterpret_cast<const void*>(clock + 0x194), sizeof(bits));
    lua_edt_bits.store(bits, std::memory_order_relaxed);
    std::memcpy(&bits, &dt, sizeof(bits));
    last_dt_bits.store(bits, std::memory_order_relaxed);
    std::memcpy(&bits, &factor, sizeof(bits));
    u32 low = min_factor_bits.load(std::memory_order_relaxed);
    while (bits < low && !min_factor_bits.compare_exchange_weak(low, bits)) {
    }
    u32 high = max_factor_bits.load(std::memory_order_relaxed);
    while (bits > high && !max_factor_bits.compare_exchange_weak(high, bits)) {
    }
    return factor;
}

template <u32 Site, u32 Reg>
u64 PS4_SYSV_ABI ScaleRegister(GuestHooks::MidContext* context) {
    const u64 n = calls[Site].fetch_add(1, std::memory_order_relaxed);
    float value;
    std::memcpy(&value, context->ymm[Reg], sizeof(value));
    value *= Factor();
    std::memcpy(context->ymm[Reg], &value, sizeof(value));
    if (Site == 0 && n % 600 == 0) {
        Report();
    }
    return 0;
}
} // namespace

u64 FindHandler(const char* name) {
    const std::string_view n{name};
    if (n == "env_phase_dt") {
        return reinterpret_cast<u64>(&ScaleRegister<0, 1>);
    }
    if (n == "particle_dt_a") {
        return reinterpret_cast<u64>(&ScaleRegister<1, 0>);
    }
    if (n == "particle_dt_b") {
        return reinterpret_cast<u64>(&ScaleRegister<2, 0>);
    }
    if (n == "particle_dt_c") {
        return reinterpret_cast<u64>(&ScaleRegister<3, 6>);
    }
    return 0;
}

void Report() {
    const auto f = [](u32 bits) {
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    DebugServer::Publish("game_time", {{"enabled", Enabled()},
                                       {"env_phase_calls", calls[0].load()},
                                       {"particle_a_calls", calls[1].load()},
                                       {"particle_b_calls", calls[2].load()},
                                       {"particle_c_calls", calls[3].load()},
                                       {"last_raw_frame_time_s", f(last_dt_bits.load())},
                                       {"last_lua_dt_s_clock_19c", f(lua_dt_bits.load())},
                                       {"last_lua_edt_s_clock_194", f(lua_edt_bits.load())},
                                       {"factor_min", f(min_factor_bits.load())},
                                       {"factor_max", f(max_factor_bits.load())},
                                       {"unreadable_or_implausible_frame_time", fallbacks.load()},
                                       {"note", "factor = raw frame time * 30; 1 at 30 fps, 0.25 at 120 fps"}});
}

} // namespace Core::NativeGameTime
