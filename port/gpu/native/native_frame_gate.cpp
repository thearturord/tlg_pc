// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_draw.h"
#include "native/native_frame_gate.h"

namespace Core::FrameGate {

namespace {

// The game's submit gate (Ghidra addresses, image base 0x1000000): `JsSwapChain_SubmitAndFlip` (0x1c4dfa0) skips its wait on its first call (flag 0x6b03748),
// then waits for `*(u64*)0x6b03798 > 0` and decrements it; `ps4render_submitDone_ThreadMain` (0x2494d40) adds one to it after every end-of-pipe event.
constexpr u64 GhidraBase = 0x1000000;
constexpr u64 GuestFirstCallFlag = 0x6b03748;
constexpr u64 GuestRequestCount = 0x6b03790;
constexpr u64 GuestDoneCount = 0x6b03798;
constexpr auto WaitLimit = std::chrono::seconds(2);

std::mutex mutex;
std::condition_variable changed;
u64 submitted = 0; // under `mutex`
u64 ended = 0;
std::atomic<u64> peak_unfinished{0};
u64 raster_reached = 0; // under `mutex`
std::atomic<u64> slot_waits{0}, slot_wait_ns{0}, own_frame_waits{0}, own_frame_wait_ns{0}, timeouts{0};
std::atomic<u64> credited{0};
std::atomic<int> credit_state{0}; // 0 not tried, 1 done, 2 refused
std::atomic<u64> unfinished_hist[4]{};   // frames unfinished at a submit (before it counts): 0, 1, 2, more

struct Config {
    int depth = 1;
    int game_wait = 0;
    bool presenter_waits = false;
};

Config LoadConfig() {
    Config config;
    int depth = 1;
    int game_waits = -1;
    int presenter_waits = -1;
    const char* source = "default";
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (MemoryPatcher::g_game_serial + ".frames.json");
    if (std::filesystem::exists(path)) {
        try {
            nlohmann::json config;
            std::ifstream in(path);
            in >> config;
            depth = config.value("frames_in_flight", 1);
            game_waits = config.value("game_waits_own_frame", -1);
            presenter_waits = config.value("presenter_waits_own_frame", -1);
            source = "json";
        } catch (const std::exception& e) {
            LOG_ERROR(Core, "frame gate: cannot read {}: {}", path.string(), e.what());
        }
    }
    if (const char* env = std::getenv("TLG_FRAMES_IN_FLIGHT"); env && env[0]) {
        depth = std::atoi(env);
        source = "TLG_FRAMES_IN_FLIGHT";
    }
    if (const char* env = std::getenv("TLG_FIF_GAME_WAIT"); env && env[0]) {
        game_waits = std::atoi(env);
    }
    if (const char* env = std::getenv("TLG_FIF_PRESENTER_WAIT"); env && env[0]) {
        presenter_waits = std::atoi(env) != 0;
    }
    const int clamped = std::clamp(depth, 0, MaxDepth);
    config.depth = clamped;
    config.game_wait = game_waits >= 0 ? std::min(game_waits, 2) : clamped == 0 ? 1 : 0;
    config.presenter_waits = presenter_waits >= 0 ? presenter_waits != 0 : clamped == 0;
    LOG_INFO(Core, "frame gate: frames in flight {} ({}{}), game wait mode {}, presenter waits for its own frame {}", clamped, source,
             clamped != depth ? ", clamped" : "", config.game_wait, config.presenter_waits);
    return config;
}

const Config& GetConfig() {
    static const Config config = LoadConfig();
    return config;
}

/// The first submit of the game: the game's own gate lets one more frame through per credit (depth 2 = one credit). Only when the guest state is exactly the
/// start state (first call flag cleared, no request, no done signal yet); anything else is refused and reported.
void CreditGuestGate(int credits) {
    const VAddr base = NativeDraw::ImageBase();
    if (base == 0) {
        credit_state = 2;
        RareEvents::Note("frame_gate_credit_refused", "no image base");
        return;
    }
    auto at = [&](u64 ghidra) { return reinterpret_cast<u8*>(base + (ghidra - GhidraBase)); };
    const u8 flag = *at(GuestFirstCallFlag);
    u64& request = *reinterpret_cast<u64*>(at(GuestRequestCount));
    u64& done = *reinterpret_cast<u64*>(at(GuestDoneCount));
    if (flag != 0 || request != 0 || done != 0) {
        credit_state = 2;
        char detail[96];
        std::snprintf(detail, sizeof detail, "guest state at the first submit: flag %u, requests %llu, done %llu", flag, static_cast<unsigned long long>(request),
                      static_cast<unsigned long long>(done));
        RareEvents::Note("frame_gate_credit_refused", detail);
        return;
    }
    std::atomic_ref<u64>(done).fetch_add(static_cast<u64>(credits), std::memory_order_acq_rel);
    credited = static_cast<u64>(credits);
    credit_state = 1;
}

} // namespace

int Depth() {
    return GetConfig().depth;
}

int GameWaitMode() {
    return GetConfig().game_wait;
}

bool GameWaitsOwnFrame() {
    return GetConfig().game_wait != 0;
}

bool PresenterWaitsOwnFrame() {
    return GetConfig().presenter_waits;
}

void NoteSubmitFlip() {
    u64 unfinished;
    bool first;
    {
        std::scoped_lock lock{mutex};
        unfinished = submitted > ended ? submitted - ended : 0;
        first = submitted == 0;
        ++submitted;
    }
    unfinished_hist[std::min<u64>(unfinished, 3)].fetch_add(1, std::memory_order_relaxed);
    const u64 now_unfinished = unfinished + 1;
    u64 peak = peak_unfinished.load(std::memory_order_relaxed);
    while (now_unfinished > peak && !peak_unfinished.compare_exchange_weak(peak, now_unfinished)) {
    }
    if (first && Depth() >= 2) {
        CreditGuestGate(Depth() - 1);
    }
}

void NoteEndOfPipe() {
    {
        std::scoped_lock lock{mutex};
        ++ended;
    }
    changed.notify_all();
}

namespace {

/// Waits until `done()` holds (called under the mutex); false on timeout
template <typename Done>
bool WaitFor(Done&& done, std::atomic<u64>& count, std::atomic<u64>& total_ns, const char* what) {
    std::unique_lock lock{mutex};
    if (done()) {
        return true;
    }
    const auto start = std::chrono::steady_clock::now();
    const bool ok = changed.wait_for(lock, WaitLimit, done);
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    count.fetch_add(1, std::memory_order_relaxed);
    total_ns.fetch_add(static_cast<u64>(waited), std::memory_order_relaxed);
    if (!ok) {
        timeouts.fetch_add(1, std::memory_order_relaxed);
        char detail[96];
        std::snprintf(detail, sizeof detail, "%s: submitted %llu, end of pipe %llu after 2 s", what, static_cast<unsigned long long>(submitted),
                      static_cast<unsigned long long>(ended));
        lock.unlock();
        RareEvents::Note("frame_gate_timeout", detail);
    }
    return ok;
}

} // namespace

void WaitForSlot() {
    const int depth = Depth();
    if (depth < 2) {
        return;
    }
    WaitFor([depth] { return submitted <= ended || submitted - ended < static_cast<u64>(depth); }, slot_waits, slot_wait_ns, "wait for a frame slot");
}

void NoteRasterReached() {
    {
        std::scoped_lock lock{mutex};
        ++raster_reached;
    }
    changed.notify_all();
}

void WaitOwnFrame() {
    const int mode = GameWaitMode();
    if (mode == 0) {
        return;
    }
    if (mode == 2) {
        WaitFor([] { return raster_reached >= submitted; }, own_frame_waits, own_frame_wait_ns, "wait for the raster thread to reach the frame end");
        return;
    }
    WaitFor([] { return ended >= submitted; }, own_frame_waits, own_frame_wait_ns, "wait for the own frame");
}

u32 PresenterAhead(u32 queued_setting) {
    return Depth() == 2 ? 2u : queued_setting;
}

nlohmann::json Status() {
    nlohmann::json j;
    j["note"] = "frames in flight (native_frame_gate.h): the setting, the frames handed to the command processor and their end-of-pipe interrupts, waits made by the gate";
    j["depth"] = Depth();
    j["game_wait_mode"] = GameWaitMode();
    j["presenter_waits_own_frame"] = PresenterWaitsOwnFrame();
    {
        std::scoped_lock lock{mutex};
        j["submitted_frames"] = submitted;
        j["end_of_pipe_interrupts"] = ended;
        j["raster_reached_frame_end"] = raster_reached;
        j["unfinished_now"] = submitted > ended ? submitted - ended : 0;
    }
    j["unfinished_peak"] = peak_unfinished.load();
    j["unfinished_at_submit"] = {{"0", unfinished_hist[0].load()}, {"1", unfinished_hist[1].load()}, {"2", unfinished_hist[2].load()}, {"3_or_more", unfinished_hist[3].load()}};
    j["slot_waits"] = {{"count", slot_waits.load()}, {"total_ms", slot_wait_ns.load() / 1e6}};
    j["own_frame_waits"] = {{"count", own_frame_waits.load()}, {"total_ms", own_frame_wait_ns.load() / 1e6}};
    j["timeouts"] = timeouts.load();
    j["guest_gate_credit"] = {{"state", credit_state.load() == 0 ? "not needed" : credit_state.load() == 1 ? "credited" : "refused"}, {"credits", credited.load()}};
    return j;
}

} // namespace Core::FrameGate
