# DLAA / DLSS on the native renderer (plan, 2026-11-01)

## Start condition: only when the renderer is 100 % native

This plan does not start before the renderer is 100 % native. Reason: DLAA has to know exactly which pass draws the scene, which target
holds the scene color and depth, which pass makes the game's motion vectors and which pass is the game's anti-aliasing. The native
renderer will know all of that from the engine's own records; doing it earlier means guessing from target shapes and shader hashes (as
the fork had to) and redoing it later.

"100 % native" is checked, not assumed:
- a long play (default spot, cutscenes, loading screens) shows no draw or dispatch planned from the register file or run on the packet
  path (`/stats`: the native draw counters' fallback reasons all 0, `direct_executed` for every submit);
- the native plan's own last steps are done (no PS4 GPU emulation left in the draw path);
- the regression script passes.

Until then this file is a parked plan. The raster thread and memory ownership plans are independent of it.

## Goal

1. **DLAA** (NVIDIA DLSS at native resolution, used as anti-aliasing) instead of the game's own anti-aliasing (SMAA): foliage, feathers
   and fences stop shimmering. **FSR 3.1 Native AA** (AMD, through FSR-Vulkan, runs on any card) as the second choice.
2. Later, **DLSS / FSR upscaling** (render the scene below the output resolution).

## What is already in this fork

From upstream, not connected to the native renderer (README: "temporal upscalers, scene resolution and post effects are not ported to
this build yet"):
- `port/gpu/dlss_bridge/` (the bridge to NVIDIA's DLSS, a separate DLL),
- `renderer_vulkan/vk_dlss.{h,cpp}`, `vk_temporal_upscaler.{h,cpp}` (the presenter still references its `Display` struct),
- `third_party/fsr-vulkan` with this port's patches (`gpu/patches/fsr-vulkan`),
- settings and overlay entries for an upscaler.

## What the reference has (tabnul/tlg_pc, master; credit @tabnul, FSR-Vulkan by @FireBurn)

DLAA there is on by default on RTX cards; DLSS upscaling is a parked work in progress (`feat/dlss-upscaling`, notes on blockiness and
edges). Useful pieces:
- **Pipeline:** the upscaler runs on the HDR scene color right before the game's post-processing combine pass, with the scene depth and
  motion vectors, and writes the result back, so the game's own post, tonemap and UI continue unchanged.
- **Motion vectors:** the game's own motion pass, completed by `vk_camera_motion.cpp` / `camera_motion.comp` (camera motion where the
  game writes none, e.g. hair).
- **Reactive mask** for transparent draws (`upscale_reactive.comp`), merge and sharpening (`upscale_merge.comp`, `taa_sharpen.comp`,
  RCAS after DLAA, CAS).
- **Jitter rules** (`port/docs/dlaa-debugging.md`), found the hard way:
  1. only geometry drawn into the scene depth is jittered; full-screen passes are not;
  2. draws that sample a scene depth are not jittered (decals, light volumes);
  3. a redraw of a mesh takes the jitter of its first jittered depth pass (cave water);
  4. one jitter per game frame, advanced at the game's motion pass;
  5. hair writes no motion vectors; it takes camera motion at its depth.
- **Debugging kit:** jitter off live (a toggle bit), `TLG_JITTER_CONTROL` (map the eligible draws, jitter only a range or one shader,
  skip a shader), a test list (the boy up close, water in the first cave, grass while moving).
- **Build:** the DLSS bridge as a separate DLL built with MSVC and the DLSS SDK (`packaging/windows/build_dlss.sh`, SDK v310.9.16);
  the package ships `tlg_dlss.dll`, `nvngx_dlss.dll` and NVIDIA's license. FSR needs no extra files.
- About 520 changed lines in `vk_temporal_upscaler`, `vk_dlss`, `vk_camera_motion` against this fork, plus small bridge changes, docs and
  tests (`test_upscaler_assets.py`, `test_upscaler_support.cpp`).

## How it fits the native renderer (better inputs than the reference had)

| needed | reference (emulated GPU) | native renderer |
|---|---|---|
| which pass is the scene, the motion pass, the post combine, the SMAA | target shapes and shader hashes | the engine's records: the caller (`RenderDev_DrawGBufferPrimitives`, `DrawZPass`, `DrawForwardPass`, post functions) and the pass class |
| scene color and depth | a full-size RGBA16F target with the scene depth | the native target descriptions of those passes |
| jitter eligibility (rules 1 to 3) | per draw, from depth state and bound images | from the record: pass class, depth write, sampled depth |
| one jitter per game frame (rule 4) | the motion pass in the stream | the frame boundary the native renderer already knows (frame start record) |
| where to apply the jitter | the draw's viewport / projection | the draw plan, when the command processor plans it (no extra raster thread work) |

## Phases

**Phase 0: entry check and inventory** (start condition above). From the native records list the game's passes of one frame with their
targets: scene color, scene depth, motion vectors, the anti-aliasing pass, the post combine pass, transparent draws.

**Phase 1: bring the code in.** The reference's changes to `vk_temporal_upscaler`, `vk_dlss`, `vk_camera_motion`, the shaders, the bridge,
`build_dlss.sh`, packaging and tests, credited by @login; adapted to this fork's code. Builds with the upscaler off: no behaviour change.

**Phase 2: FSR 3.1 Native AA first.** It needs no extra DLL and proves the wiring: scene color and depth from the native target
descriptions, motion vectors from the game's motion pass, the reactive mask, the write back before the post combine, the game's anti-aliasing
pass skipped while it runs. Switch `TLG_UPSCALER=fsr3` (off by default).

**Phase 3: jitter.** The jitter sequence and the five rules, applied in the draw plan by pass class; the debugging kit ported
(`TLG_JITTER_CONTROL`, live jitter off). Check the known spots.

**Phase 4: DLAA.** The DLSS DLL build and package, `TLG_UPSCALER=dlss`; greyed out with a reason on cards or systems without DLSS.

**Phase 5: settings.** Launcher row and overlay entry (Off, DLAA, FSR 3.1 Native AA), the `tlg.ini` keys, and a key to switch against the
game's own anti-aliasing (not F8: in this build F8 is the stall profiler).

**Phase 6: validation and default.** The known spots, cutscenes, loading screens, long play; GPU cost per frame (Shift+F9 GPU busy);
owner plays; then the default on RTX cards (FSR elsewhere).

**Phase 7 (later): upscaling.** Scene targets below the output size, mip LOD bias for scene materials, UI and post at the output size.
The reference's parked branch and notes (blockiness, edges) are the starting point.

## Risks

- Jitter artifacts on passes the game never meant to be jittered: the rules and the debugging kit exist for this; new scenes may need
  new rules (fix the class of draw, not the shader hash).
- The DLSS DLL needs MSVC and NVIDIA's SDK; it is Windows only (FSR covers the rest).
- Licenses: NVIDIA's DLSS license ships with `nvngx_dlss.dll`; FSR-Vulkan's license as it is in `third_party`.
