// SPDX-License-Identifier: GPL-2.0-or-later
#include <atomic>
#include <nlohmann/json.hpp>
#include "native/native_dispatch_inputs.h"

namespace Core::NativeDispatch {

namespace {
std::atomic<u64> checked{0};
std::atomic<u64> differences{0};
} // namespace

void NoteChecked(bool equal) noexcept {
    checked.fetch_add(1, std::memory_order_relaxed);
    if (!equal) {
        differences.fetch_add(1, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    return {{"conversions_checked", checked.load()}, {"differences", differences.load()}};
}

} // namespace Core::NativeDispatch
