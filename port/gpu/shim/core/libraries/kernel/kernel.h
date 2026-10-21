// tlg: the parts of shadPS4's kernel.h used by vendored libraries.
#pragma once
#include "common/types.h"
#include "core/libraries/kernel/orbis_error.h"
namespace Libraries::Kernel {
int* PS4_SYSV_ABI __Error();
/// tlg-native: the port's runtime sets up its own memory (2816 MB Garlic pool); the PS4 Pro mode of the old emulator is not needed.
inline void SetNeoModeCalls(u32) {}
} // namespace Libraries::Kernel
