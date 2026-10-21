// tlg-native: the part of shadPS4's cpu_patches used by the guest hooks: code emitted near the game's code, reachable by a rel32 jump.
#pragma once
#include <functional>
#include <xbyak/xbyak.h>

namespace Core {
/// Emits code with `emit` into executable memory within +-2 GB of `code_ptr` (the code of the eboot). False when no such memory could be had or `emit` failed.
bool EmitInTrampolineArea(const void* code_ptr, const std::function<bool(Xbyak::CodeGenerator&)>& emit);
} // namespace Core
