// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// E7b Task 3 (port/gpu/native/E7-notes.md section 10): how often a field of the register image that the register fallback of the draw path reads (the plan of a draw whose input or pipeline key
/// is seen for the first time: `PlanDraw` from `exec_state`, `MakeDrawInputs`, `PlanGraphics`, `RefreshGraphicsKey`, `BuildRuntimeInfo`) differs from the value the native side has for the same thing.
/// Counting only, `TLG_PLAN_SHADOW=1` (default off; with it off every call returns at once). The module knows nothing of the emulator's types: the callers hand over plain values.
/// Report: /stats native_backend.device.plan_shadow (and last_counters.json), per "group/field": evaluated, differs, and for register-only fields how often the value is not 0.
namespace Core::NativePlanShadow {

[[nodiscard]] bool On() noexcept;

/// A field the native side has a value for: `native` is the native value, `registers` the register image's.
void Compare(const char* group, const char* field, u64 native, u64 registers);
/// The same for a field of any size (equal bytes: no difference); the examples show the first eight bytes of each side.
void CompareBytes(const char* group, const char* field, const void* native, const void* registers, size_t size);
/// A register image field the native side has no value for: counts the evaluations and how often the value is not 0 (a field that is always 0 in the fallbacks is a candidate to drop).
void RegisterOnly(const char* group, const char* field, u64 registers);

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativePlanShadow
