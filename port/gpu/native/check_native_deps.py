#!/usr/bin/env python3
"""Dependency rule for native/: native code may include only native/, common/, third party and system headers,
plus the PS4-shaped headers listed in ALLOWED_PS4 below. The emulator (shadps4/) may include native/ freely.

Every entry in ALLOWED_PS4 is a debt: it goes when the phase of documents/tlg/native plan/native-completion-plan.md
named next to it is done. When the list is empty, native/ is independent of the emulation code.

Run: python -I port/gpu/native/check_native_deps.py     (exit 1 on a violation, 0 otherwise)
     --consumer-table              print the per file numbers of the consumer rule (G0)
     --write-consumer-baseline     write consumer_rule_baseline.json (refuses to raise any number)
"""
import json, os, re, sys

NATIVE = os.path.dirname(os.path.abspath(__file__))

ALLOWED_PS4 = {
    "video_core/amdgpu/resource.h": "by design: the shader recompiler reads T# words, so Derive writes them in the PS4 layout (pixel_format.h comes with it); goes when the recompiler interface changes",
    "video_core/amdgpu/regs.h": "by design: bit layouts of the colour/depth target words the shader and the host format code take (ColorBuffer, DepthBuffer, DepthView, DataFormat tables); native_target_desc.cpp only",
    "video_core/amdgpu/liverpool.h": "by design: the record format is a list of register pairs; native_draw.cpp uses the register group map and the register word types (DepthShaderControl, ShaderProgram) of the record, native_shader_env.cpp the same words",
    "video_core/renderer_vulkan/vk_fixed_state.h": "by design: the Vulkan fixed state struct (Vulkan terms; it includes regs.h only for NUM_VIEWPORTS and PrimitiveType)",
    "video_core/renderer_vulkan/vk_indexed_data.h": "by design: the Vulkan host class of the indexed data path (C9); native_draw.cpp only reads its counters",
}
# plain emulator infrastructure the native code shares (not GPU emulation); reviewed once, kept short
ALLOWED_SHARED = {
    "core/memory.h", "core/guest_heap.h", "core/platform.h", "core/emulator_settings.h",  # settings and the CPU's view of PS4 memory (by design)
    "core/address_space.h", "core/cpu_patches.h", "core/libraries/kernel/process.h",     # PS4 OS / CPU side (out of scope of the plan)
    "core/libraries/kernel/time.h",      # the OS time stamp counter (GPU perf counter source), a clock, not GPU state
    "video_core/renderdoc.h", "video_core/thread_profiler.h", "video_core/thread_usage.h",  # profiling and capture tools, not GPU state
    "video_core/renderer_vulkan/vk_common.h", "video_core/renderer_vulkan/vk_instance.h", "video_core/renderer_vulkan/vk_scheduler.h",  # the Vulkan host layer (device, command recording), not PS4 state
}

# Member files: none. The out-of-class definitions of emulator class members that served the native path (Liverpool, TextureCache, BufferCache,
# Rasterizer) lived here until 2026-11-03; they are now `*_engine.cpp` beside their class in shadps4/video_core/, where the emulator may include
# native/ freely. The mechanism stays so that a member file cannot be added to native/ without being reported.
MEMBER_FILES = {}

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

# ---- G0 consumer rule (documents/tlg/native plan/executor-consumer-plan.md section 8) -------------------------------------------------------------------
# The host side (shadps4/video_core/**, the emulator classes and the Vulkan consumers) may include a PS4-shaped header only where it did on the day of the
# baseline (consumer_rule_baseline.json, written once with --write-consumer-baseline). Lists may only shrink: a new include of a PS4 header in a file, or more
# `AmdGpu::` lines in a file than the baseline, is a violation. Direct includes only. The G phases lower the numbers; the baseline is rewritten
# downwards (--write-consumer-baseline refuses to raise any number) after each phase, G6 ends with the recompiler boundary only.
VIDEO_CORE = os.path.normpath(os.path.join(NATIVE, "..", "shadps4", "video_core"))
CONSUMER_BASELINE = os.path.join(NATIVE, "consumer_rule_baseline.json")
PS4_HEADER = re.compile(r"^(video_core/amdgpu/|shader_recompiler/|video_core/renderer_vulkan/(liverpool_to_vk|vk_fixed_state|vk_indexed_data)\.h$|core/libraries/(gnmdriver|videoout)/)")


def scan_consumers():
    result = {}
    for dirpath, _, names in os.walk(VIDEO_CORE):
        for name in names:
            if not name.endswith((".cpp", ".h")):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, VIDEO_CORE).replace(os.sep, "/")
            includes, lines = set(), 0
            for line in open(path, encoding="utf-8", errors="replace"):
                m = include.search(line)
                if m and PS4_HEADER.match(m.group(1)):
                    includes.add(m.group(1))
                if "AmdGpu::" in line:
                    lines += 1
            if includes or lines:
                result[rel] = {"includes": sorted(includes), "amdgpu_lines": lines}
    return result


consumers = scan_consumers()
if "--write-consumer-baseline" in sys.argv:
    old = json.load(open(CONSUMER_BASELINE)) if os.path.exists(CONSUMER_BASELINE) else None
    if old is not None:
        for rel, now in consumers.items():
            before = old.get(rel, {"includes": [], "amdgpu_lines": 0})
            if now["amdgpu_lines"] > before["amdgpu_lines"] or set(now["includes"]) - set(before["includes"]):
                print(f"consumer baseline: {rel} would go up, refused (lists only shrink)")
                sys.exit(1)
    with open(CONSUMER_BASELINE, "w") as f:
        json.dump(consumers, f, indent=1, sort_keys=True)
    print(f"consumer baseline written: {len(consumers)} files, {sum(c['amdgpu_lines'] for c in consumers.values())} AmdGpu:: lines")
    sys.exit(0)
if os.path.exists(CONSUMER_BASELINE):
    base = json.load(open(CONSUMER_BASELINE))
    for rel, now in sorted(consumers.items()):
        before = base.get(rel, {"includes": [], "amdgpu_lines": 0})
        for target in sorted(set(now["includes"]) - set(before["includes"])):
            bad.append(f"shadps4/video_core/{rel}: includes PS4 header {target} (not in consumer_rule_baseline.json)")
        if now["amdgpu_lines"] > before["amdgpu_lines"]:
            bad.append(f"shadps4/video_core/{rel}: {now['amdgpu_lines']} AmdGpu:: lines, baseline {before['amdgpu_lines']}")
    total_now = sum(c["amdgpu_lines"] for c in consumers.values())
    total_base = sum(c["amdgpu_lines"] for c in base.values())
    print(f"consumer rule: {len(consumers)} files with PS4 includes or AmdGpu:: lines (baseline {len(base)}), {total_now} AmdGpu:: lines (baseline {total_base}), "
          f"{sum(len(c['includes']) for c in consumers.values())} PS4 includes (baseline {sum(len(c['includes']) for c in base.values())})")
else:
    print("consumer rule: no consumer_rule_baseline.json yet (python check_native_deps.py --write-consumer-baseline)")
if "--consumer-table" in sys.argv:
    for rel, now in sorted(consumers.items(), key=lambda kv: -kv[1]["amdgpu_lines"]):
        print(f"  {now['amdgpu_lines']:4d} AmdGpu:: lines  {len(now['includes']):2d} PS4 includes  {rel}")

for line in bad:
    print(line)
print(f"native/ includes {len(used)} of {len(ALLOWED_PS4)} allowlisted PS4 headers; {len(bad)} violation(s)")
for target in sorted(ALLOWED_PS4):
    print(f"  {'used  ' if target in used else 'unused'}  {target}  -> goes with {ALLOWED_PS4[target]}")
for name, owner in MEMBER_FILES.items():
    exists = os.path.exists(os.path.join(NATIVE, name))
    print(f"  member file {name:28s} {'present' if exists else 'absent '}  -> {owner}")
sys.exit(1 if bad else 0)
