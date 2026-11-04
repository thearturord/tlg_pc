// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// The register values a WAIT_REG_MEM can poll (G4a, documents/tlg/native plan/executor-consumer-plan.md). The command processor used to keep a 0xD000 word register file
// (Liverpool::regs) that every state template and clear wrote into, for the sake of this one reader. The G0 census (/native register_wait_polls) shows that no scene
// ever polls a register (spears, furthest, bloom: empty), so the register file is gone: a register poll reads 0 here, is counted per register and is recorded as a rare
// event (`register_wait_polled_<reg>`, user/log/rare_events.txt), so the first game that does poll one names the register that needs a value.

#include <nlohmann/json.hpp>

#include "common/types.h"

namespace Core::NativeWaitRegisters {

/// The value a register wait sees for `reg` (a word index in the PS4 register file; always 0 until the census names a register that needs one).
[[nodiscard]] u32 Read(u32 reg);

/// Register number (hex) to the number of polls. Empty: no register wait ever ran. Report: /native register_wait_polls.
[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeWaitRegisters
