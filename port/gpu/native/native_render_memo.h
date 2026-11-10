// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// Design review 79: the render state memo (`TLG_RENDER_STATE_MEMO`, default 0 off; 1 shadow, 2 real). A draw that continues the render pass the draw
/// before it opened gets that draw's render state back instead of the full setup (target lookups, image updates, layout transitions, attachment
/// descriptions). Shadow: the memo decides, the full setup still runs, and its state is compared with the memo's; any work the full setup did on a
/// draw the memo would have answered (a barrier, a recorded command, an upload) is counted too.
namespace Core::RenderMemo {

[[nodiscard]] u32 Mode() noexcept;

enum What : u32 {
    Asked,
    MemoValid,
    PassOpen,
    NoClear,          ///< (with the pass open)
    SignatureSame,    ///< (with the pass open and no clear)
    WouldHit,         ///< and no target sampled by the draw
    Compared,
    Different,        ///< DIFFERENT_MUST_BE_0
    WorkBarrier,      ///< FULL_DID_WORK_MUST_BE_0: the full setup asked for a barrier
    WorkCommands,     ///< FULL_DID_WORK_MUST_BE_0: the full setup recorded commands
    WorkUpload,       ///< FULL_DID_WORK_MUST_BE_0: the full setup uploaded bytes
    FeedbackLoop,     ///< the full setup found a feedback loop the memo would not
    Stored,
    NotStoredClear,
    NotStoredFeedback,
    Served,           ///< real mode: answered by the memo (one would-hit draw in 64 is audited by the shadow instead)
    Count
};

void Note(u32 what) noexcept;
/// The time of one full setup (TSC ticks), on a draw the memo would have answered or not.
void NoteTicks(bool would_hit, u64 ticks) noexcept;
[[nodiscard]] nlohmann::json Report();

} // namespace Core::RenderMemo
