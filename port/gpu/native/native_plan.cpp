// SPDX-License-Identifier: GPL-2.0-or-later
// G5c: see native_plan.h.

#include "native/native_plan.h"

namespace Core::NativePlan {

namespace {
u64 Mix(u64 h, u64 v) {
    return (h ^ v) * 0x100000001b3ull;
}

u64 DigestStage(u64 h, const StagePlan& stage) {
    h = Mix(h, reinterpret_cast<u64>(stage.info));
    h = Mix(h, stage.pgm_base);
    h = Mix(h, stage.user_data_count);
    for (u32 i = 0; i < stage.user_data_count; ++i) {
        h = Mix(h, stage.user_data[i]);
    }
    h = Mix(h, stage.flat_skipped ? 1 : 0);
    for (const u32 word : stage.flat) {
        h = Mix(h, word);
    }
    return h;
}

constexpr size_t PoolLimit = 512;
} // namespace

void ResetStage(StagePlan& stage) {
    stage.info = nullptr;
    stage.ring.clear();
    stage.user_data_count = 0;
    stage.flat.clear();
    stage.flat_skipped = false;
    stage.pgm_base = 0;
    stage.resolved.clear();
    stage.image_words.clear();
}

u64 PlanDigest(const DrawPlan& plan) {
    u64 h = 0xcbf29ce484222325ull;
    h = Mix(h, reinterpret_cast<u64>(plan.pipeline));
    h = Mix(h, plan.op_class);
    for (const auto& stage : plan.stages) {
        h = DigestStage(h, stage);
    }
    return h;
}

u64 PlanDigest(const DispatchPlan& plan) {
    return DigestStage(Mix(0xcbf29ce484222325ull, reinterpret_cast<u64>(plan.pipeline)), plan.stage);
}

const RingBinding* RingUse::Find(const Shader::Info* info, u32 index) const noexcept {
    const StagePlan* plan_stage = nullptr;
    if (draw) {
        for (const auto& candidate : draw->stages) {
            if (candidate.info == info) {
                plan_stage = &candidate;
                break;
            }
        }
    } else if (dispatch && dispatch->stage.info == info) {
        plan_stage = &dispatch->stage;
    }
    if (!plan_stage) {
        return nullptr;
    }
    for (const auto& ring : plan_stage->ring) {
        if (ring.index == index) {
            return &ring;
        }
    }
    return nullptr;
}

const RingBinding* RingUse::FindGeometry(VAddr address, u64 size, u32 kind) const noexcept {
    if (draw == nullptr) {
        return nullptr;
    }
    for (const auto& ring : draw->geometry_ring) {
        if (ring.index == kind && ring.address == address && ring.size == size) {
            return &ring;
        }
    }
    return nullptr;
}

u64 RingUse::End() noexcept {
    const u64 end = draw ? draw->ring_end : dispatch ? dispatch->ring_end : 0;
    draw = nullptr;
    dispatch = nullptr;
    return end;
}

RingStats RingUse::Take() noexcept {
    RingStats s;
    s.queued = queued.exchange(0);
    s.written_buffers = written_buffers.exchange(0);
    s.offered = offered.exchange(0);
    s.taken = taken.exchange(0);
    s.bytes = bytes.exchange(0);
    s.no_sharp = no_sharp.exchange(0);
    s.written = written.exchange(0);
    s.formatted = formatted.exchange(0);
    s.too_large = too_large.exchange(0);
    s.immutable = immutable.exchange(0);
    s.gpu_modified = gpu_modified.exchange(0);
    s.pending_write = pending_write.exchange(0);
    s.ring_full = full.exchange(0);
    s.bound = bound.exchange(0);
    return s;
}

std::unique_ptr<DrawPlan> PlanPool::TakeDraw() {
    {
        std::scoped_lock lk{mutex};
        if (!draws.empty()) {
            auto plan = std::move(draws.back());
            draws.pop_back();
            return plan;
        }
    }
    return std::make_unique<DrawPlan>();
}

std::unique_ptr<DispatchPlan> PlanPool::TakeDispatch() {
    {
        std::scoped_lock lk{mutex};
        if (!dispatches.empty()) {
            auto plan = std::move(dispatches.back());
            dispatches.pop_back();
            return plan;
        }
    }
    return std::make_unique<DispatchPlan>();
}

void PlanPool::Recycle(std::unique_ptr<DrawPlan>&& plan) {
    if (!plan) {
        return;
    }
    plan->pipeline = nullptr;
    plan->executed = false;
    plan->ring_end = 0;
    for (auto& stage : plan->stages) {
        ResetStage(stage);
    }
    std::scoped_lock lk{mutex};
    if (draws.size() < PoolLimit) {
        draws.push_back(std::move(plan));
    }
}

void PlanPool::Recycle(std::unique_ptr<DispatchPlan>&& plan) {
    if (!plan) {
        return;
    }
    plan->pipeline = nullptr;
    plan->ring_end = 0;
    plan->native_seq = 0;
    ResetStage(plan->stage);
    std::scoped_lock lk{mutex};
    if (dispatches.size() < PoolLimit) {
        dispatches.push_back(std::move(plan));
    }
}

} // namespace Core::NativePlan
