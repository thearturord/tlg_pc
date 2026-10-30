#!/usr/bin/env python3
"""Dependency rule for native/: native code may include only native/, common/, third party and system headers,
plus the PS4-shaped headers listed in ALLOWED_PS4 below. The emulator (shadps4/) may include native/ freely.

Every entry in ALLOWED_PS4 is a debt: it goes when the phase of documents/tlg/native plan/no-emulation-plan.md
named next to it is done. When the list is empty, native/ is independent of the emulation code.

Run: python -I port/gpu/native/check_native_deps.py     (exit 1 on a violation, 0 otherwise)
"""
import os, re, sys

NATIVE = os.path.dirname(os.path.abspath(__file__))

ALLOWED_PS4 = {
    "video_core/amdgpu/pm4_cmds.h": "E5 (PM4 stream)",
    "video_core/amdgpu/pixel_format.h": "E2 (descriptions)",
    "video_core/amdgpu/draw_state.h": "E1 (draw state)",
    "video_core/amdgpu/liverpool.h": "E1/E5 (register file, command processor)",
    "video_core/renderer_vulkan/vk_fixed_state.h": "E1 (draw state)",
    "video_core/renderer_vulkan/liverpool_to_vk.h": "E1/E2 (format maps)",
    "video_core/renderer_vulkan/vk_indexed_data.h": "E2 (descriptors)",
    "video_core/renderer_vulkan/vk_identity_audit.h": "E7 (audits)",
    "core/libraries/gnmdriver/gnmdriver.h": "E5/E6 (Gnm submission HLE)",
}
# plain emulator infrastructure the native code shares (not GPU emulation); reviewed once, kept short
ALLOWED_SHARED = {
    "core/memory.h", "core/guest_heap.h", "core/platform.h", "core/emulator_settings.h",  # settings and the CPU's view of PS4 memory (by design)
    "core/address_space.h", "core/cpu_patches.h", "core/libraries/kernel/process.h",     # PS4 OS / CPU side (out of scope of the plan)
    "video_core/renderdoc.h", "video_core/thread_profiler.h", "video_core/thread_usage.h",  # profiling and capture tools, not GPU state
    "video_core/renderer_vulkan/vk_common.h", "video_core/renderer_vulkan/vk_instance.h", "video_core/renderer_vulkan/vk_scheduler.h",  # the Vulkan host layer (device, command recording), not PS4 state
}

# Member files: out-of-class definitions of emulator class members that belong to the native path (E-1 part 2). They must include the
# emulator's class headers, so they are exempt from the include rule; each one is a debt that goes when its class gets a native-side
# interface (see the plan row E-1). The list is reported so it cannot grow silently.
MEMBER_FILES = {
    "native_liverpool.cpp": "Liverpool (command processor): E1/E5",
    "native_texture_cache.cpp": "TextureCache: E3",
    "native_buffer_cache.cpp": "BufferCache: E4",
    "native_rasterizer.cpp": "Rasterizer (consumer): E2/E3/E4",
    "native_rasterizer_helpers.h": "Rasterizer helpers shared with vk_rasterizer.cpp: E2/E3/E4",
}

include = re.compile(r'#\s*include\s+"([^"]+)"')
bad = []
used = set()
for name in sorted(os.listdir(NATIVE)):
    if not name.endswith((".cpp", ".h")):
        continue
    if name in MEMBER_FILES:
        continue
    for number, line in enumerate(open(os.path.join(NATIVE, name), encoding="utf-8", errors="replace"), 1):
        m = include.search(line)
        if not m:
            continue
        target = m.group(1)
        if target.startswith(("native/", "common/")) or "/" not in target:
            continue
        if target in ALLOWED_PS4:
            used.add(target)
            continue
        if target in ALLOWED_SHARED:
            continue
        if target.startswith(("core/", "video_core/", "shader_recompiler/")):
            bad.append(f"native/{name}:{number}: includes {target} (not on the allowlist)")

for line in bad:
    print(line)
print(f"native/ includes {len(used)} of {len(ALLOWED_PS4)} allowlisted PS4 headers; {len(bad)} violation(s)")
for target in sorted(ALLOWED_PS4):
    print(f"  {'used  ' if target in used else 'unused'}  {target}  -> goes with {ALLOWED_PS4[target]}")
for name, owner in MEMBER_FILES.items():
    exists = os.path.exists(os.path.join(NATIVE, name))
    print(f"  member file {name:28s} {'present' if exists else 'absent '}  -> {owner}")
sys.exit(1 if bad else 0)
