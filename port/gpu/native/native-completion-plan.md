# Plan: finish the native renderer, no PS4 GPU emulation left (updated 2026-11-01)

**Goal (owner):** the renderer takes the engine's own description of a frame and nothing else. What cannot go stays by design, with a
named interface. This plan is only about finishing the native renderer; performance work, 120 fps game timing, add-ons and the heavy scene
are not part of it. The heavy scene is not a gate of any phase here.

**Basis:** this plan updates the plan of 2026-10-31 (which replaced `no-emulation-plan-E0-E7`, phases E-1 to E8, where all the numbers and
sections 1 to 23 live). Evidence: `port/gpu/native/E7-notes.md` (sections 12 to 19), `port/gpu/native/F4-census.md`. Branch `native-v2`
(port repo) after the E7c merge, plus `cloud-e7c` (the Shift+F9 probe profiler, the kernel semaphore fix in the OS layer, notes; the
experiments of that round were removed again). The older documents are in `history/`, `Legacy/`, `stages/`.

What changed since 2026-10-31 (checked against the code):
- The section 1 facts below were checked in the code: every switch and function this plan names exists; `check_native_deps.py` lists 10
  PS4 headers and 5 member files; the F4 numbers match `F4-census.md`.
- The PM4 packet walk is never executed (`NativeDirect::PacketPathOff()` returns true), but its code is still in the tree
  (`ProcessGraphics`, `ProcessCompute`, the packet decoders they use): now named for deletion in F5.
- "Arena" survives as a name in the buffer cache's counters and comments (`buffer.uncached.arena_synchronized`, `BufferFor`'s comments) after
  E4 removed the arena: renamed in F5.
- Counters can now also be read per thread with the probe profiler (Shift+F9, `/prof`); it is a tool, not a verifier, and stays (F6).
- The OS layer (`port/src`, the port's own kernel and libc side) is by design, but not exempt from fixes: the kernel semaphores had one
  lock for all of them (fixed, `frame-loop-findings.md`). Such fixes are outside this plan.

## 1. Where we are (code of 2026-11-01; counters of 2026-10-31, default scene, 25 s headless)

| layer | state |
|---|---|
| Command stream (PM4 executed) | gone (E5): the stream is our tags only. The packet walk's code is dead but present (F5). |
| Sync consumption and present | native (E6). The producer side still decodes PM4 sync packets: `SweepSyncPackets` / `ConvertSyncPacket` / the `native_sync` decoders. Only `FenceEop` has a typed producer (0 differences over 1,659). |
| Draw state (fixed function, shader environment, draw arguments) | native, played (E1). |
| Render target and depth description | native, played, default (E7). Register-only fields (alt tile, compression, linear general, depth view read-only, clear values, fmask) have a register-free version behind `TLG_TARGET_REGFREE=1`: 138k constants applied, 0 guard trips, 41.9k zeroed fmask sets, 1.5k clear values from the call. Not played yet. |
| Image and sampler descriptions | native (E2), played. |
| Buffer descriptions (V#), `AmdGpu::Image/Buffer/Sampler` types | census done (static, `F4-census.md`): three layers: the shader interface in `shader_recompiler/` stays by design; the host consumers are F4; `native/` builds words that the host decodes again. The host reads about 15 of 40 T# fields, 8 of 14 V# fields, 14 of 30 S# fields; 30 `GetSharp` call sites outside the recompiler (9 legacy fallback or verify, 5 address-only, 8 HLE shadows, 3 indexed path, 5 vertex or pipeline creation). Not built yet. |
| Image identity | native for named lookups (E3). Address-keyed leftovers: `RefreshImage` (about 2 M calls per 25 s, 353 uploads), `exact_match` (77k), `ResolveOverlap` / `ResolveDepthOverlap` (53), `ExpandImage` (11). |
| Image content | still copied from guest memory by `RefreshImage`. Evidence for announced content: 196 of 196 first uploads equal to the announced bytes; 154 of 156 GPU-sourced uploads equal (2 differ, all DMA copies). |
| Memory (arena, page protection, tracker) | gone (E4): CPU-changed queries come from the announcements (`NativeCpuQuery`, the engine buffers' dirty sets), GPU-written ranges from `gpu_modified_ranges`; the only page protection call left restores read/write for the frame ring once. The thin address-keyed `ObtainBuffer` entry points are kept; "arena" remains as a name (F5). |
| Register fallback | open, the largest piece. A new input or pipeline key (about 480 per 25 s) is learned through `PlanDraw(exec_state)`: `MakeDrawInputs`, `pipeline_cache.PlanGraphics(regs)`, `Synthesize`, `DrawRegs`, `ExecApplyRecord`, `MirrorRegs`, `front_state`, `ClassifyDraw`, `MakeFixedState`. The shadow (`TLG_PLAN_SHADOW=1`) says every field with a native value has 0 differences (fixed state, draw arguments, user data, program addresses, target formats, shader environment; the pixel input table differs only in the padding bits 9 to 15). Register-only fields are all zero or constant in the default scene: tessellation and geometry stage registers, clip user data, `ta_bc_base`, `index_offset`, clip distance emulation; `aa_config.NumSamples` is 1. |
| Mixed functions and allowlists | `BindBuffers`, `BindTextures`, `Draw`, `DispatchDirect/Indirect`, `ObtainBufferUncached`, `SynchronizeMemory`, `FindImage`, `RefreshImage`, `RasterDispatch/Main` still mix native and emulation. `check_native_deps.py` lists 10 PS4 headers (`pm4_cmds.h`, `pixel_format.h`, `resource.h`, `draw_state.h`, `liverpool.h`, `vk_fixed_state.h`, `liverpool_to_vk.h`, `vk_indexed_data.h`, `vk_identity_audit.h`, `gnmdriver.h`) and 5 member files (`native_liverpool.cpp`, `native_texture_cache.cpp`, `native_buffer_cache.cpp`, `native_rasterizer.cpp`, `native_rasterizer_helpers.h`). |
| Verifiers | default-sampled ones are off since 2026-10-30; opt-in ones remain in the code (list in F6). |

**By design (stays):** the shader ISA and its translator (`ShaderEnv`, `reference/shader-boundary.md`); `shader_recompiler/` keeps
`AmdGpu::Image/Buffer/Sampler` and `pixel_format.h` (the translator reads the descriptor words the shader names: stride, swizzle, add_tid,
mtype, view type, number conversion); guest memory as the shader's data source; the descriptor words the shader reads from its own user
data; the completion mailbox in guest memory; detile once at upload; the guest CPU and OS side (the port itself); the probe profiler.

## 2. Method (unchanged)

- **Shadow:** build the native answer next to the old one, compare on live data, counters, switch off by default.
- **Switch on by default**, old path kept as the way back (`=0`).
- **Owner plays** the default scene and a cutscene and says it looks good. Never ask for the heavy scene.
- **Delete** the old code and its allowlist entries in `native/check_native_deps.py`. A phase is not done while its entries remain.
- Where a function mixes both, the native part first becomes one call into `native/`, then the emulation side is deleted.
- **Evidence rule** for constants and "never used" cases: one scene is not proof. Every register-only field that becomes a constant or a
  refusal needs its counters read in the default scene and a cutscene before the register read is deleted. An unsupported case after
  deletion is a counted refusal plus a `rare_events.txt` entry, never a silent fallback.
- **Who does what:** cloud session (no binary, no game; can read code, run Ghidra headless on an uploaded eboot, write shadows, counters,
  notes, deletions guided by the compiler); local (this machine): Ghidra project, hooks (`CUSA03627.hooks.json`), builds, regression, game
  runs, merge. Prompts for cloud rounds go to `history/cloud-prompts/` after use. Nothing is pushed without the owner's word.

## 3. Phases

| # | phase | removes | native replacement | gate |
|---|---|---|---|---|
| F0 | Play what is built (owner, 1 session) | nothing deleted; turns the register-free target fields on | `TLG_TARGET_REGFREE=1` with `TLG_TARGET_REGFREE_VERIFY=1` in the default scene and a cutscene: alt tile, compression, linear general, depth view read-only as guarded constants, depth/stencil clear values from the clear call, fmask zeroed for one-sample targets. Then default (`=0` way back). | `constant_guard_trips` empty and verify differences 0 in both scenes; the owner says it looks good |
| F1 | Sync producers (local, Ghidra) | `SweepSyncPackets`, `ConvertSyncPacket`, the packet decoders in `native_sync`, sync tags that carry packet words; `pm4_cmds.h` leaves the allowlist | Hook the engine functions that write each remaining sync packet and build the typed record from their arguments (`TLG_SYNC_TYPED=1` compares with the decoder). Packets without a producer per 25 s run: end-of-pipe from other writers 11.6k, release memory 12.8k, label wait 19.1k, label write 3.8k, CP DMA 17. Writers to find: label write and wait wrappers, the compute release writer, async copy `FUN_024717c0`, CP DMA `FUN_024a4ab0` `FUN_01c6e670` `FUN_024a5e20` `FUN_01c6e320` `FUN_024a53e0`, the frame fence in `JsSwapChain_SubmitAndFlip` (0x1c4dfa0). Steps: F1.1 one kind at a time to 0 differences; F1.2 the typed record decides, decode only verifies, owner plays; F1.3 delete. | 0 differing per kind over a 25 s run and a longer run; no PM4 header left in `native/` except what F1.3 could not remove (named) |
| F2 | Image content and identity without guest memory (cloud shadow, local play) | `RefreshImage`, `UpdateImage`, `ResolveOverlap`, `ResolveDepthOverlap`, `ExpandImage`, `exact_match`; the address-keyed part of `FindImage` | F2.1 explain the 2 differing GPU-sourced uploads (DMA copies). F2.2 `RefreshImage` takes its bytes from the announced data (creation data, UpdateSubresource, Map/Unmap, file load), never from guest memory; GPU-written images are never refreshed; shadow first (announced upload against the guest-memory upload). The announced-content source is built as one shared piece (one store of announced bytes that other readers can use later), not as a private path of the texture cache. F2.3 the leftovers become engine-id rules: same texture grows, views, block views, 3D views of equal size (callers counted by `TLG_E3_CENSUS=1`: 53 and 11 per 25 s). F2.4 delete; `ImageInfo.guest_address` stays as data copied from the engine object. | announced equals guest for every first upload and every refresh in a default-scene run and a cutscene; owner plays; the image part of `resource.h` and `native_texture_cache.cpp` shrink |
| F3 | Register fallback replaced by native planning (the largest phase) | `exec_state`, `DrawRegs` for draws, `MakeDrawInputs`, `Synthesize` register output, `ExecApplyRecord`, `MirrorRegs`, `front_state`, `ClassifyDraw`, `MakeFixedState`, the register pairs of the record and the state package built from them; the register reads of `BuildRuntimeInfo` | Learning draws (about 480 per 25 s) plan from the record and the engine objects. F3.1 the pipeline key reads the target description, not `regs.color_buffers` / `regs.depth_buffer`; the colour/depth register types (`ColorBuffer`, `DepthBuffer`, `DepthView`, `CbDbExtent`) are a currency of `ImageInfo` / `ImageViewInfo` and are handled together with F4. F3.2 stages, program, user data from `ShaderEnv`; tessellation and geometry are always 0 in the default scene: read the same counters in a cutscene first, then an unsupported stage is a counted refusal. F3.3 `BuildRuntimeInfo` entirely from `ShaderEnv`; `aa_config.NumSamples` from the target sample count, clip distance emulation from the vertex export info, `clip_user_data` / `ta_bc_base` / `index_offset` as constants with guard counters. F3.4 `PlanDraw` takes a native description (record keys, `TargetSet`, fixed state, `ShaderEnv`). F3.5 delete. Each of F3.1 to F3.4 is shadow, switch, play. | `exec_inputs.fallback` counts only refusals (0 expected) in the default scene and a cutscene; grep for `exec_state`, `DrawRegs`, `MakeDrawInputs`, `Synthesize` finds nothing outside history; regression 3/3 PASS; `draw_state.h` leaves the allowlist; owner plays |
| F4 | Buffers and the PS4 resource types (detail: `F4-census.md`, phase notes `F4-notes.md`) | the host users of `AmdGpu::Image`, `AmdGpu::Buffer`, `AmdGpu::Sampler` (texture cache, pipelines, rasterizer, draw_prep, specialization fill), the five vertex table readers, `resource.h` and `pixel_format.h` from `native/` | F4.0 run the binder counters in the default scene and a cutscene; F4.0b field-use histogram of the bound T#, S#, V# (cloud counter round); F4.1 plain `ImageDesc` / `SamplerDesc` / `BufferDesc` in `native/` with no PS4 header; F4.2 `ImageInfo` / `ImageViewInfo` / `Sampler` constructors from them behind a switch (needs F4.0b); F4.3 rasterizer and draw_prep key by the description; F4.4 vertex description in the record replaces the five vertex-table readers; F4.5 `specialization.h` filled from the native description, key bytes unchanged; F4.6 delete | grep for `AmdGpu::Image`, `AmdGpu::Buffer`, `AmdGpu::Sampler` outside `shader_recompiler/` finds only by-design items; `resource.h` and `pixel_format.h` leave the allowlist; regression 3/3 PASS; owner plays |
| F5 | Split the mixed functions, empty the allowlists | the emulation side of `BindBuffers`, `BindTextures`, `Draw`, `DispatchDirect/Indirect`, `ObtainBufferUncached`, `SynchronizeMemory`, `FindImage`, `RasterDispatch/Main`; the dead packet walk (`ProcessGraphics`, `ProcessCompute` and the packet decoders only they use, `native_packet_path`); the 5 member files and 10 PS4 headers of `check_native_deps.py` | After F2 to F4 most of the emulation side is dead; what remains of the native part becomes the function itself (member files move into `native/` classes or free functions). Names that describe removed machinery go with it (the buffer cache's "arena" counters and comments). | `ALLOWED_PS4` and `MEMBER_FILES` empty, or every remaining entry named by design in this plan |
| F6 | Prune the verifiers and census tools | `TLG_ZPASS_AUDIT`, `TLG_GBUFFER_AUDIT`, `TLG_DRAIN_VERIFY` (keep while the lazy drain check from the flagged list is open), `TLG_B3/B8/B9_VERIFY`, `TLG_TEXTURE_IDS_VERIFY`, `TLG_RESIDENT_VERIFY`, `TLG_OBJECT_COPY_VERIFY`, `TLG_ZERO_COPY_VERIFY`, `TLG_INDEXED_VERIFY`, `TLG_CONST_RING_VERIFY`, `TLG_INPLACE_VERIFY`, `TLG_IDENTITY_AUDIT` + `vk_identity_audit.h`, `TLG_DISPATCH_AUDIT`, `TLG_HAZARD_AUDIT`, `TLG_PLAN_SHADOW`, `TLG_SYNC_TYPED`, `TLG_E3_*`, `TLG_TARGET_REGONLY` | Delete what only compares against deleted paths. The generator `*_GEN_VERIFY` shadows go last: the regression script uses them, so it gets a counter-based check first. The probe profiler (Shift+F9 and its probes) is a tool and stays: its probes cost one load while no capture runs. | warning-clean build; regression 3/3 PASS with the counter-based check |
| F7 | Final gate and release ("100 % native") | nothing new; consolidation | grep of the port tree for PM4, `AmdGpu::Image`, `exec_state`, `DrawRegs`, `RefreshImage`, arena finds only by-design items. Documents consolidated into one architecture page (section 1 rewritten as the final state). The push decision is the owner's. This gate is the definition of "100 % native" that other plans wait for. | build warning-clean; regression 3/3 PASS; `check_native_deps.py` clean with empty lists; a long play (default scene, cutscenes, loading screens) shows no draw or dispatch planned from registers and no packet path; the owner plays the default scene and a cutscene |

## 4. Order and sizes

```
F0 ---------------------------------------------------+
F1 (independent, local) -------------------------------+
F2 -----------------------------------------------------+--> F5 --> F6 --> F7
F4.0 -> F4.0b -> F4.1 -> F4.2 -> F4.3 -> F4.4 -> F4.5 -> F4.6 +
F3.1 -> F3.2 -> F3.3 -> F3.4 -> F3.5 ------------------+
```

F0, F1 and F2 run in parallel (F1 local, F2 cloud). The F4 census is done: F4.1 to F4.3 do not depend on F2 or F3; F4.4 needs the record
to name the vertex streams (local question); F4.5 is last. F3 is the long pole; its counters need F0's play session (target fields) and a
cutscene run for the register-only evidence. F5 to F7 wait for the rest.

Rough size in work blocks (a block = one cloud round or one local session; an estimate, not a measurement): F0 1, F1 4 to 6, F2 3 to 4,
F3 8 to 12, F4 3 to 5, F5 3 to 4, F6 2, F7 1: about 25 to 35 blocks.

## 5. Risks

- **One scene is not evidence.** Everything in section 1 about "constant" or "never used" comes from the default scene. The cutscene run
  of the shadow counters is a step of F3.2 and F3.3, not an extra.
- **Tessellation, geometry and multisample targets** may exist in scenes not played. After F3.5 they are refusals; a refusal in play means
  the guard missed a case. Keep the rare-event file in the play checklist.
- **GPU hangs:** paths that change which bytes the GPU reads (F2 content, F4 buffers) need `reference/gpu-driver-crashes.md` first.
- **Hidden state:** the engine patches words after creation and caches register copies (see `Synthesize`); F3 differences show up as
  flicker, not crashes (the 2026-10 surface group case). Counters first, play second.
- **Hook correctness (F1):** a wrong typed expectation looks like a bug in the decoder (`FenceEop` was 32 versus 64 bit). Read the packet
  writer in Ghidra before building the expectation.
- **F4 open questions** (local, Ghidra or counters): engine patches of the view's T# copy (view+0x58, min_lod), vertex streams in the record,
  texture alt_tile_mode, custom border colours, post operations on sharps, multisample sampled images, sharps without an engine texture; the
  seven items are in `F4-notes.md` section 3.
- **Verifier removal (F6)** removes the safety net of the regression: replace the generator verify with a counter check before deleting.
- **Dead code that looks alive:** the packet walk is unreachable but compiled; until F5 deletes it, a change to the PM4 decoders can look
  like it matters. Treat it as dead in reviews.

## 6. Parked (not part of this plan)

Performance and frame pacing work, 120 fps game timing (fixed 1/30 s steps and Lua timers), add-ons, E9 (textures decoded at load), the
lazy drain / CP split / stage key cache / replay thread list (owner, 2026-10-30), known visual bugs (`reference/known-visual-bugs.md`),
branches `tlg-upstream-picks`, `tlg-encoder-thread`. Fixes in the OS layer (`port/src`) are made when found and are not tracked here.
