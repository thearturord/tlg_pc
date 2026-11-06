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
