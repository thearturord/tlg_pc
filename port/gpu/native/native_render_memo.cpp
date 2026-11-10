// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_render_memo.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <nlohmann/json.hpp>

namespace Core::RenderMemo {

u32 Mode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_RENDER_STATE_MEMO"); // 0 off (default), 1 shadow, 2 real (until 2026-11-10 "1" was the real memo)
        return e != nullptr && (e[0] == '1' || e[0] == '2') ? static_cast<u32>(e[0] - '0') : 0u;
    }();
    return mode;
}

namespace {
std::atomic<u64> counts[Count]{};
std::atomic<u64> ticks[2]{}, calls[2]{};
std::atomic<u64> cal_tsc{0}, cal_ns{0};

u64 SteadyNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

void Note(u32 what) noexcept {
    if (what < Count) {
        counts[what].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteTicks(bool would_hit, u64 t) noexcept {
    if (cal_tsc.load(std::memory_order_relaxed) == 0) {
        cal_ns.store(SteadyNs(), std::memory_order_relaxed);
        cal_tsc.store(__builtin_ia32_rdtsc(), std::memory_order_relaxed);
    }
    ticks[would_hit ? 1 : 0].fetch_add(t, std::memory_order_relaxed);
    calls[would_hit ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

nlohmann::json Report() {
    static constexpr const char* N[Count] = {"asked", "memo_valid", "pass_open", "no_clear", "signature_same", "would_hit", "compared", "DIFFERENT_MUST_BE_0",
                                             "FULL_DID_WORK_barrier_MUST_BE_0", "FULL_DID_WORK_commands_MUST_BE_0", "FULL_DID_WORK_upload_MUST_BE_0", "feedback_loop_full_only",
                                             "stored", "not_stored_clear", "not_stored_feedback", "served"};
    nlohmann::json j = nlohmann::json::object();
    j["mode"] = Mode();
    for (u32 k = 0; k < Count; ++k) {
        j[N[k]] = counts[k].load();
    }
    double tpn = 0.0;
    if (const u64 t0 = cal_tsc.load(); t0 != 0) {
        const u64 dn = SteadyNs() - cal_ns.load();
        const u64 dt = __builtin_ia32_rdtsc() - t0;
        tpn = dn != 0 ? static_cast<double>(dt) / static_cast<double>(dn) : 0.0;
    }
    j["ticks_per_ns"] = tpn;
    static constexpr const char* K[2] = {"full_setup_memo_would_miss", "full_setup_memo_would_hit"};
    for (u32 k = 0; k < 2; ++k) {
        j[K[k]] = {{"calls", calls[k].load()}, {"ticks", ticks[k].load()}};
    }
    return j;
}

} // namespace Core::RenderMemo
