// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_log.h"
#include "native/native_draw.h"
#include "native/native_frame_gate.h"
#include "native/native_frame_events.h"
#include "native/native_occlusion.h"
#include "native/native_content.h"
#include "native/native_flight.h"
#include "native/native_prof.h"

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
std::atomic<u64> raster_behind_hist[4]{}; // M5: frames the raster thread had not reached at a submit (before it counts): 0, 1, 2, more
std::atomic<u64> eop_cap_waits{0};
constexpr u64 OwnershipEopCap = 4;       // M5: the GPU may be this many frames behind before the game is held anyway

struct Config {
    int depth = 1;
    int game_wait = 0;
    bool presenter_waits = false;
};

Config LoadConfig() {
    Config config;
    int depth = 0; // default 0 since 2026-10-31 (the owner played it at 120 fps)
    int game_waits = -1;
    int presenter_waits = -1;
    const char* source = "default";
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (MemoryPatcher::g_game_serial + ".frames.json");
    if (std::filesystem::exists(path)) {
        try {
            nlohmann::json config;
            std::ifstream in(path);
            in >> config;
            depth = config.value("frames_in_flight", 0);
            game_waits = config.value("game_waits_own_frame", -1);
            presenter_waits = config.value("presenter_waits_own_frame", -1);
            source = "json";
        } catch (const std::exception& e) {
            TLG_LOG(GATE, Error, "frame gate: cannot read {}: {}", path.string(), e.what());
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

bool EarlyInterrupt() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_OWNERSHIP_EARLY_IRQ");
        if (e == nullptr || *e == '\0' || *e == '0') {
            return false;
        }
        if (!OwnershipGate()) {
            RareEvents::Note("ownership_early_irq_refused", "needs the ownership gate");
            return false;
        }
        LOG_INFO(Core, "frame gate: the end-of-pipe interrupt is raised when the raster thread reaches the frame end");
        return true;
    }();
    return on;
}

bool OwnershipGate() {
    // M5 (memory-ownership-plan.md): the game's slot wait counts frames the raster thread has not reached (our side owns the frame's data) instead of frames the GPU has not finished.
    // Needs depth 2 and the frame store serving the readers (TLG_FRAME_STORE=2), or the late reads would see the game's next frame.
    static const bool on = [] {
        const char* e = std::getenv("TLG_OWNERSHIP_GATE");
        if (e == nullptr || *e == '\0' || *e == '0') {
            return false;
        }
        if (GetConfig().depth < 2 || !NativeContent::FrameStoreServes()) {
            RareEvents::Note("ownership_gate_refused", "needs frames in flight 2 and TLG_FRAME_STORE=2");
            return false;
        }
        LOG_INFO(Core, "frame gate: ownership gate on (slot wait counts frames the raster thread has not reached)");
        return true;
    }();
    return on;
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
    {
        std::scoped_lock lock{mutex};
        const u64 behind = submitted > raster_reached + 1 ? submitted - 1 - raster_reached : 0; // `submitted` already counts this frame
        raster_behind_hist[std::min<u64>(behind, 3)].fetch_add(1, std::memory_order_relaxed);
    }
    const u64 now_unfinished = unfinished + 1;
    u64 peak = peak_unfinished.load(std::memory_order_relaxed);
    while (now_unfinished > peak && !peak_unfinished.compare_exchange_weak(peak, now_unfinished)) {
    }
    if (first && Depth() >= 2) {
        // ownership gate: the game's own gate (one end of pipe per frame) must not be the limit, the slot wait is
        int credits = Depth() - 1;
        if (OwnershipGate()) {
            credits = Depth() - 1; // the engine's most is two frames ahead (3 frame contexts): 3 credits gave a black scene in the owner's play (2026-11-04)
            if (const char* e = std::getenv("TLG_OWNERSHIP_CREDIT"); e && e[0]) { // experiment: the credits of the game's own gate with the ownership gate on
                credits = std::clamp(std::atoi(e), 0, 4);
            }
        }
        CreditGuestGate(credits);
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

std::mutex early_mutex;
u64 early_handed = 0;          // ordinals given out, under `early_mutex`
u64 early_raised = 0;          // interrupts raised: ordinals below this are done
std::deque<u64> early_waiting; // recorded by the raster thread, interrupt not yet raised (waiting for occlusion answers)
std::atomic<u64> early_by_raster{0}, early_by_answers{0}, early_by_gpu{0}, early_deferred{0};

/// Raises the interrupts of every frame up to `ordinal` that is still owed (once each, in order)
void RaiseUpTo(u64 ordinal, std::atomic<u64>& counter) {
    u64 n = 0;
    {
        std::scoped_lock lock{early_mutex};
        if (ordinal + 1 <= early_raised) {
            return;
        }
        n = ordinal + 1 - early_raised;
        early_raised = ordinal + 1;
        while (!early_waiting.empty() && early_waiting.front() <= ordinal) {
            early_waiting.pop_front();
        }
    }
    counter.fetch_add(n, std::memory_order_relaxed);
    for (u64 i = 0; i < n; ++i) {
        NoteEndOfPipe();
        NativeFrameEvents::EndOfPipe();
    }
}

/// Raster thread, after a poll wrote answers
void RetryEarly() {
    for (;;) {
        u64 front = 0;
        {
            std::scoped_lock lock{early_mutex};
            if (early_waiting.empty()) {
                return;
            }
            front = early_waiting.front();
        }
        if (!NativeOcclusion::AllAnsweredBefore(front)) {
            return;
        }
        RaiseUpTo(front, early_by_answers);
    }
}

} // namespace

u64 NextEarlyOrdinal() {
    static const bool hooked = [] {
        NativeOcclusion::SetAnswersHook(&RetryEarly);
        return true;
    }();
    (void)hooked;
    std::scoped_lock lock{early_mutex};
    return early_handed++;
}

void EarlyFrameRecorded(u64 ordinal) {
    NativeOcclusion::NoteFrameEnd(); // the queries opened from now on belong to the next frame
    if (NativeOcclusion::AllAnsweredBefore(ordinal)) {
        RaiseUpTo(ordinal, early_by_raster);
        return;
    }
    early_deferred.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{early_mutex};
    if (ordinal >= early_raised) {
        early_waiting.push_back(ordinal);
    }
}

void EarlyGpuDone(u64 ordinal) {
    RaiseUpTo(ordinal, early_by_gpu);
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
    Flight::Add(Flight::GateWaitBegin, 0, Flight::GateFrame, submitted - ended); // L1
    FLIGHT_SCOPE("waiting: the frame gate (frames in flight)");
    const bool ok = changed.wait_for(lock, WaitLimit, done);
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    Flight::Add(Flight::GateWaitEnd, 0, Flight::GateFrame, static_cast<u64>(waited), ok ? 0 : 1); // L1
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
    if (OwnershipGate()) {
        TLG_PROF_LEVEL("wait.frame_gate_slot_owned (game thread, ownership gate: fewer than 2 frames the raster thread has not reached)", 1);
        // `TLG_OWNERSHIP_RASTER_LEAD=n` (experiment, native-v3-writers): how many frames the raster thread may be behind the game's submits (default: the depth, 2). The census of late reads
        // shows the game's rings of 3 are rewritten before the raster thread has read them with a lead of 2; a lead of 1 leaves one frame of slack.
        static const int lead = [depth] {
            const char* e = std::getenv("TLG_OWNERSHIP_RASTER_LEAD");
            const int v = e != nullptr ? std::atoi(e) : depth;
            return v >= 1 && v <= depth ? v : depth;
        }();
        WaitFor([] {
            const bool raster_ok = submitted <= raster_reached || submitted - raster_reached < static_cast<u64>(lead);
            const bool eop_ok = submitted <= ended || submitted - ended < OwnershipEopCap;
            if (raster_ok && !eop_ok) {
                eop_cap_waits.fetch_add(1, std::memory_order_relaxed);
            }
            return raster_ok && eop_ok;
        }, slot_waits, slot_wait_ns, "wait for a frame slot (ownership gate)");
        return;
    }
    TLG_PROF_LEVEL("wait.frame_gate_slot (game thread, depth 2: fewer than 2 unfinished frames)", 1);
    WaitFor([depth] { return submitted <= ended || submitted - ended < static_cast<u64>(depth); }, slot_waits, slot_wait_ns, "wait for a frame slot");
}

void NoteRasterReached() {
    {
        std::scoped_lock lock{mutex};
        ++raster_reached;
    }
    changed.notify_all();
}

u64 SubmittedFrames() {
    std::scoped_lock lock{mutex};
    return submitted;
}

u64 RasterReachedFrames() {
    std::scoped_lock lock{mutex};
    return raster_reached;
}

void WaitRasterReachedFrames(u64 target_frames) {
    static std::atomic<u64> waits{0}, wait_ns{0};
    TLG_PROF_LEVEL("wait.loader_gate (game thread, a write into freed memory waits for the raster thread)", 1);
    WaitFor([target_frames] { return raster_reached >= target_frames; }, waits, wait_ns, "wait for the raster thread (loader gate)");
}

void WaitOwnFrame() {
    const int mode = GameWaitMode();
    if (mode == 0) {
        return;
    }
    if (mode == 2) {
        TLG_PROF_LEVEL("wait.frame_gate_own_frame_raster (game thread, mode 2: raster thread reached the frame end)", 1);
        WaitFor([] { return raster_reached >= submitted; }, own_frame_waits, own_frame_wait_ns, "wait for the raster thread to reach the frame end");
        return;
    }
    TLG_PROF_LEVEL("wait.frame_gate_own_frame_eop (game thread, depth 0: end of pipe of its own frame)", 1);
    WaitFor([] { return ended >= submitted; }, own_frame_waits, own_frame_wait_ns, "wait for the own frame");
}

u32 PresenterAhead(u32 queued_setting) {
    return Depth() == 2 ? 2u : queued_setting;
}

nlohmann::json Status() {
    nlohmann::json j;
    j["note"] = "frames in flight (native_frame_gate.h): the setting, the frames handed to the command processor and their end-of-pipe interrupts, waits made by the gate";
    j["depth"] = Depth();
    j["ownership_gate"] = OwnershipGate();
    j["early_interrupt"] = EarlyInterrupt();
    j["raster_behind_at_submit"] = {{"0", raster_behind_hist[0].load()}, {"1", raster_behind_hist[1].load()}, {"2", raster_behind_hist[2].load()}, {"3_or_more", raster_behind_hist[3].load()}};
    j["ownership_eop_cap_polls"] = eop_cap_waits.load();
    j["early_interrupts"] = {{"raised_at_raster", early_by_raster.load()}, {"raised_after_answers", early_by_answers.load()}, {"raised_by_gpu_done", early_by_gpu.load()}, {"deferred_for_occlusion", early_deferred.load()}};
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
