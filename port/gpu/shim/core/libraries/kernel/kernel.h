// tlg: the parts of shadPS4's kernel.h used by vendored libraries.
#pragma once
#include "common/types.h"
#include "core/libraries/kernel/orbis_error.h"
namespace Libraries::Kernel {
int* PS4_SYSV_ABI __Error();
/// tlg-native: the port's runtime sets up its own memory (2816 MB Garlic pool); the PS4 Pro mode of the old emulator is not needed.
}
extern "C" void runtime_set_neo_mode_calls(unsigned calls);
namespace Libraries::Kernel {
/// While `calls` > 0 the game's sceKernelIsNeoMode answers are yes (PS4 Pro memory setup, see GarlicPoolPro in native_draw.cpp).
inline void SetNeoModeCalls(u32 calls) { runtime_set_neo_mode_calls(calls); }
} // namespace Libraries::Kernel
