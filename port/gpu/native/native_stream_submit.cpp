// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/logging/log.h"
#include "native/native_timeline.h"
#include "native/native_stream_submit.h"
#include "native/native_readback_policy.h"

namespace Libraries::GnmDriver {
/// gnmdriver.cpp: the sweep of the submit and the queueing of a span on the graphics queue; false when the debug state forbids it.
bool StreamPublish(const u32* from, u32 dwords);
void StreamOpen(bool open);
bool StreamAllowed();
} // namespace Libraries::GnmDriver

namespace Core::StreamSubmit {

namespace {

constexpr u32 LagHooks = 2;        ///< the span ends at the stream position of the hook this many hooks back: later hooks may still rewrite earlier packets
constexpr u32 MinSpanDwords = 400; ///< shorter spans are not worth a submit (and could be taken for a state template)
constexpr u32 FramesBeforeStart = 5;

std::mutex mutex;
std::thread::id render_thread{};
u64 frame_end{};        ///< the end pointer of the buffer being recorded (identifies it)
u64 frame_base{};       ///< its start
std::vector<u64> cursors; ///< stream position at the entry of every hook of the frame
u32 published_dwords{};
u32 published_hooks{};  ///< index of the hook at which the last span was published
bool stopped{};         ///< no more spans in this frame
bool open{};
std::vector<u32> published_copy;
u64 real_submits{};

u64 stat_hook_frames{}, stat_no_span_frames{}, stat_not_plausible{}, stat_not_allowed{}, stat_max_hooks{}, stat_hooks{};
u64 stat_spans{}, stat_dwords{}, stat_frames{}, stat_stopped_frames{}, stat_violations{}, stat_chunk_changes{};
std::string last_violation;

} // namespace

void AtHook(u64 ctx) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.stream_submit.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (render_thread == std::thread::id{}) {
        render_thread = std::this_thread::get_id();
    }
    if (render_thread != std::this_thread::get_id()) {
        return;
    }
    const u64 begin = *reinterpret_cast<const u64*>(ctx + 0x47c0);
    const u64 end = *reinterpret_cast<const u64*>(ctx + 0x47c8);
    const u64 cursor = *reinterpret_cast<const u64*>(ctx + 0x47d0);
    if (end != frame_end) {
        if (open) {
            // the game moved on to another buffer in the middle of a frame (a chunk): what was handed over stays, the rest of the frame goes the normal way
            ++stat_chunk_changes;
            stopped = true;
            frame_end = end;
            return;
        }
        frame_end = end;
        frame_base = begin;
        cursors.clear();
        published_dwords = 0;
        published_hooks = 0;
        published_copy.clear();
        const bool plausible = begin != 0 && begin <= cursor && cursor - begin < (8u << 20);
        const bool allowed = Libraries::GnmDriver::StreamAllowed();
        stat_not_plausible += plausible ? 0 : 1;
        stat_not_allowed += allowed ? 0 : 1;
        ++stat_hook_frames;
        stopped = !plausible || real_submits < FramesBeforeStart || !allowed;
    }
    if (stopped || begin != frame_base) {
        return;
    }
    cursors.push_back(cursor);
    ++stat_hooks;
    stat_max_hooks = std::max<u64>(stat_max_hooks, cursors.size());
    const u32 hook = static_cast<u32>(cursors.size()) - 1;
    const u32 every = std::max<u32>(policy.stream_hooks.load(std::memory_order_relaxed), 4);
    if (hook < LagHooks || hook - published_hooks < every) {
        return;
    }
    const u64 to = cursors[hook - LagHooks];
    const u32 to_dwords = static_cast<u32>((to - frame_base) / 4);
    if (to_dwords < published_dwords + MinSpanDwords) {
        return;
    }
    const u32* from = reinterpret_cast<const u32*>(frame_base) + published_dwords;
    const u32 dwords = to_dwords - published_dwords;
    if (!open) {
        Libraries::GnmDriver::StreamOpen(true);
        open = true;
    }
    const u64 publish_start = FrameTimeline::Now();
    if (!Libraries::GnmDriver::StreamPublish(from, dwords)) {
        stopped = true;
        return;
    }
    const u64 publish_ns = FrameTimeline::Now() - publish_start;
    // what was handed over, after the sweep of the submit: the real submit compares the buffer with it
    published_copy.insert(published_copy.end(), from, from + dwords);
    published_dwords = to_dwords;
    published_hooks = hook;
    ++stat_spans;
    stat_dwords += dwords;
    FrameTimeline::Record(FrameTimeline::StreamPublish, dwords, FrameTimeline::Now(), publish_ns);
}

u32 Consumed(const u32* base, u32 dwords) {
    std::scoped_lock lock{mutex};
    if (!open) {
        return 0;
    }
    if (reinterpret_cast<u64>(base) != frame_base || published_dwords > dwords) {
        NoteViolation("the submitted buffer is not the one that was streamed");
        return 0;
    }
    if (std::memcmp(base, published_copy.data(), static_cast<size_t>(published_dwords) * 4) != 0) {
        size_t first = 0;
        while (first < published_dwords && base[first] == published_copy[first]) {
            ++first;
        }
        LOG_ERROR(Core, "streaming: the command buffer changed after it was handed over (first difference at dword {} of {}: {:#x} -> {:#x})", first,
                  published_dwords, published_copy[first], base[first]);
        NoteViolation("the command buffer changed after it was handed over");
    }
    return published_dwords;
}

void RealSubmitQueued() {
    std::scoped_lock lock{mutex};
    if (frame_end != 0 && !open) {
        ++stat_no_span_frames;
    }
    if (open) {
        ++stat_frames;
        stat_stopped_frames += stopped ? 1 : 0;
        Libraries::GnmDriver::StreamOpen(false);
    }
    ++real_submits;
    open = false;
    frame_end = 0;
    frame_base = 0;
    cursors.clear();
    published_dwords = 0;
    published_hooks = 0;
    published_copy.clear();
    stopped = false;
}

void NoteViolation(const char* what) {
    ++stat_violations;
    last_violation = what;
    LOG_ERROR(Core, "streaming switched off: {}", what);
    VideoCore::ReadbackPolicy::Get().stream_submit = 0;
}

nlohmann::json Report() {
    std::scoped_lock lock{mutex};
    return {{"enabled", VideoCore::ReadbackPolicy::Get().stream_submit.load()},
            {"every_hooks", VideoCore::ReadbackPolicy::Get().stream_hooks.load()},
            {"frames_streamed", stat_frames},
            {"frames_stopped_midway", stat_stopped_frames},
            {"spans_handed_over", stat_spans},
            {"dwords_handed_over", stat_dwords},
            {"buffer_changes_in_a_frame", stat_chunk_changes},
            {"violations", stat_violations},
            {"last_violation", last_violation},
            {"real_submits", real_submits},
            {"frames_with_hooks", stat_hook_frames},
            {"frames_without_span", stat_no_span_frames},
            {"frames_not_plausible", stat_not_plausible},
            {"frames_not_allowed", stat_not_allowed},
            {"hooks_in_streamed_frames", stat_hooks},
            {"max_hooks_in_a_frame", stat_max_hooks}};
}

} // namespace Core::StreamSubmit
