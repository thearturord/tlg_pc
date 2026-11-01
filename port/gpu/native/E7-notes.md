# E7 notes (cloud session, Task 1: read-only analysis)

Base: `native-v2` at `8667426`. Nothing was built or run; every statement below comes from reading the code (file:line at that commit) or from the numbers in the task
description. Where I could not tell, it says so and it is a question in the last section.

## 0. What changes the plan (read this first)

1. **A typed draw does not read the register image for its targets at draw time.** `PlanExecDraw` (native/native_rasterizer.cpp:335) copies `DrawInputs` from the
   cache `exec_inputs[inputs_key]` (line ~380). That cache is filled by the slow path: `Rasterizer::PlanDraw` runs `plan->inputs = MakeDrawInputs(regs)` and
   `exec_inputs.insert_or_assign(inputs_key, plan->inputs)` (vk_rasterizer.cpp, end of `PlanDraw`). `inputs_key` is a hash of the register values of the Targets, Other and VsProgram groups of the record
   (native_draw.cpp, `TakeExecDrawImpl` / the state package, `values.h | 1`; erratum of round 5: it is not a hash of the view blocks, `TargetSignature` is a different hash). So the register image feeds
   the target description once per distinct block hash (a "learning" draw, `plan_null[1]` in native_draw.h:64), and everything after that replays the cached
   copy. The place where a native `TargetDesc` would replace the register decode is that one fill point, plus the three operations that never go through
   `exec_inputs` (section 2).
2. **The target size does not come from the pitch/slice words.** `ImageInfo(ColorBuffer, hint)` and `ImageInfo(DepthBuffer, ...)` take `size.width/height` from
   `hint` (the engine's own extent word: `view + 0x54 + 15*4` for colour, `dsv + 0x58 + 12*4` for depth, `CbDbExtent`) whenever `hint.Valid()`
   (texture_cache/image_info.cpp:64-93, 95-125). `Pitch()`, `Height()` and the slice size are still used for `pitch`, `guest_size` and `mips_layout[0]`.
3. **`TakeFromRegisters` is a no-op in the default configuration.** `PlanDraw` masks `native_stale` with the `native_sdk_strip` bits 28 to 30
   (vk_rasterizer.cpp, `PlanDraw`); the in-use value 2213543935 has those bits off, so `groups == 0`. It only runs when someone sets those bits for a comparison.
4. **The E1c shader environment comparison still runs on every vertex/pixel stage input build** (vk_pipeline_cache.cpp:378-392: `ApplyShaderEnv`, `DiffShaderEnvFields`,
   `NoteCompared`, then `Decides()`). It compares against the register decode, so it goes with the register image, not before.
5. **The engine's texture description in `native_device` has no sample count.** `ReadResourceObject` (native_device.cpp:91-127) reads `d(5)` (SampleCount, resource
   `+0x194` for a tex2d) into nothing; `Object` (native_device.h:33) has no `samples` field. Task 2 can read it from `Object::object + 0x194` while the
   object is alive, or add a field. I have not added anything.

## 1. Consumers of the target words after `MakeDrawInputs`

Classes: **V** needed for a Vulkan image / view / pipeline (format, extent, samples, layers, mips); **T** needed to detile CPU-visible content or to match cache
entries (tile mode, pitch, alt tile, sizes); **M** AMD metadata a Vulkan renderer never needs (cmask, fmask, htile, dcc); **C** clear values; **F** duplicate of
state that is already native elsewhere (`FixedState`); **N** no reader found by `grep` outside the register structs (`regs_color.h`, `regs_depth.h`, `draw_state.*`).

### 1.1 `AmdGpu::ColorBuffer` (regs_color.h:117, 15 dwords, `DrawInputs.color_buffers[8]`)

| field | class | readers after `MakeDrawInputs` | engine source (as far as I can tell) |
|---|---|---|---|
| `base_address`, `Address()` (low 8 bits from `view.slice_start` when `linear_general`) | T | `ImageInfo.guest_address` (image_info.cpp:82); `Presenter::IsVideoOutSurface` (vk_presenter.cpp:187); target markers (vk_rasterizer.cpp:3275, `EliminateFastClear` marker); `TargetInputsMatch` memcmp key (vk_rasterizer.cpp:585-600) | resource `+0x30` = `Object::address`. The `linear_general` low-bits case has no engine source seen. |
| `pitch.tile_max` (`Pitch()`) | T | `ImageInfo.pitch`, `size.width` when no hint, `mips_layout[0]` (image_info.cpp:64-93) | `Derive` pitch rule (native_image_desc.cpp:120-127); verified for sampled images only, not for targets |
| `pitch.fmask_tile_max` | N | none | - |
| `slice.tile_max` (`Height()`, `GetColorSliceSize()`) | T | `ImageInfo.guest_size`, `mips_layout[0]`, `size.height` when no hint | derivable from height and pitch; the 1088 vs 1080 padding case seen in E2 (section 19.1 of the plan) applies |
| `view.slice_start` / `slice_max` (`BaseSlice()`, `NumSlices()`) | V | `ImageInfo.resources.layers`; `ImageViewInfo(ColorBuffer)` range and type (image_view.cpp:76) | RTV description: dimension `+0x44`, then `+0x48/+0x4c/+0x50` (same reader as `ReadViewRange`, native_device.cpp ~700) |
| `info.format`, `number_type`, `comp_swap` (`GetDataFmt`, `GetNumberFmt`, `GetNumberConversion`, `Swizzle`) | V | `ImageInfo.pixel_format`, `num_bits`; `ImageViewInfo.format`; pipeline key `color_buffers[cb].data_format/num_format/num_conversion/swizzle` (vk_pipeline_cache.cpp:937-941) and from there the shader recompiler (frontend/translate/export.cpp:61, spirv_emit_context.cpp:734, emit_spirv_context_get_set.cpp:286): this one reaches shader variants, so it is GPU visible | DXGI format of the RTV (`+0x40`) or the resource. NOTE: the 5 bit colour-buffer format enum is not the T# data format (`GetDataFmt` goes through `RemapDataFormat`), so the E2 `FormatEntry` table cannot be reused as is; it gives the T# data/number format and `dst_sel`, not `comp_swap`/`number_type` |
| `info.endian` | N | none | - |
| `info.linear_general` | T | `Address()`, `BaseSlice()`, `GetTileMode()`, `IsTiled()` | unknown for targets (display surfaces are the case that matters) |
| `info.fast_clear` | M / behaviour | `meta_info.cmask_addr` (image_info.cpp:79); `EliminateFastClear` returns early without it (vk_rasterizer.cpp:686) | resource `+0x58` (cmask pointer) plus the cmask fast-clear enable bit; the bit's place in the block is not known to me |
| `info.compression` | M | `meta_info.fmask_addr` only (image_info.cpp:80) | - |
| `info.blend_bypass` | F/V | pipeline key blend setup (vk_pipeline_cache.cpp:945 and 975) | hardware property of the format; not checked |
| `info.alt_tile_mode` | T | `ImageInfo.alt_tile` on Neo (image_info.cpp:92) -> detile spec data, `texture_cache.cpp` cache keys (582-601, 761) | bit in the block |
| `info.blend_clamp`, `simple_float`, `round_mode`, `cmask_is_linear`, `blend_opt_*`, `fmask_*_ci`, `fmask_compress_1frag_only`, `dcc_enable`, `cmask_addr_type` | N | none (`grep`) | - |
| `attrib.tile_mode_index` (`GetTileMode()`) | T | `ImageInfo.tile_mode`/`array_mode` -> `tile_manager.cpp:91-118` (detile pipeline), `texture_cache.cpp:471,497,596` (cache matching) | `Derive` `tiling_index` (14 for colour targets, native_image_desc.cpp:116); the same GB_TILE_MODE index space if the engine uses 14 for both views of one resource (not verified) |
| `attrib.num_fragments_log2` (`NumSamples()`) | V | `ImageInfo.num_samples`; pipeline key `color_samples` (vk_pipeline_cache.cpp:984-986) | D3D11 `SampleDesc.Count` (resource `+0x194`, not stored, see 0.5) |
| `attrib.num_samples_log2`, `fmask_tile_mode_index`, `fmask_bank_height`, `force_dst_alpha_1` | N | none | - |
| `cmask_base_address`, `cmask_slice` | M | `CmaskAddress()` -> `surface_metas` -> `GetMetaInfo` -> `SkipForEngineClear` decides whether the engine's metadata fill is skipped (vk_rasterizer.cpp:3316; texture_cache.cpp:1122-1148) | resource `+0x58` / cmask address, per the SDK strings |
| `fmask_base_address`, `fmask_slice` | M | `FmaskAddress()` (same path) | no engine source seen (second 32 byte block at view `+0x74`?) |
| `clear_word0/1` | C | `LiverpoolToVK::ColorBufferClearValue` (liverpool_to_vk.cpp:830) -> pending fast clear colour, applied at the eliminate (vk_rasterizer.cpp:704) and at the render pass (BeginRenderingFull, vk_rasterizer.cpp:3125) | block words 11/12 = the prebuilt block of the resource (+0xdc/+0xe0) for mip 0, 0 above (section 6). The colour the engine's clear call packs lives at resource +0x150/+0x154 (valid when +0x14c != 0). The B11 note (registers 0,0,0,0 vs call 1,1,1,1) is explained by this: the view's words never carry the call's colour |
| `pad0`, `pad1[2]` | N | - | - |

`CbDbExtent hint` (not a field of `ColorBuffer`): carried separately (`current_cb_extent`), used for `size.width/height` (see 0.2). It is the engine's own word, not a register decode.

### 1.2 `AmdGpu::DepthBuffer` and the depth-side words (regs_depth.h:186; `DrawInputs.depth_*`)

| field | class | readers | engine source |
|---|---|---|---|
| `z_info.format`, `stencil_info.format` | V | `DepthValid()/StencilValid()` everywhere (vk_rasterizer.cpp:289, 384, 640-641, 3034-3035, 3212-3217; vk_depth_stencil_state.h:25-26); `ImageInfo.pixel_format`/`has_stencil`/`num_bits` (image_info.cpp:95-125; `num_bits` at 107, `stencil_addr` at 115); `ImageViewInfo` format (image_view.cpp:84); pipeline key `z_format`/`stencil_format` (vk_pipeline_cache.cpp:906-910); `LiverpoolToVK::DepthFormat` | DXGI depth format of the resource / DSV |
| `z_info.num_samples` (`NumSamples()`) | V | `ImageInfo.num_samples`; pipeline key `depth_samples` (vk_pipeline_cache.cpp:921) | `SampleDesc.Count` |
| `z_info.tile_mode_index` | T | `ImageInfo.tile_mode` -> detile and cache matching | `Derive` `tiling_index` 0 for depth (native_image_desc.cpp:116), if the DSV and the T# agree |
| `z_info.tile_surface_enable` | M | `meta_info.htile_addr` (image_info.cpp:113) | - |
| `z_info.tile_split`, `decompress_on_n_zplanes`, `allow_expclear`, `read_size`, `clear_disallowed`, `zrange_precision` | N | none | - |
| `z_read_base`, `stencil_read_base` | T | `DepthAddress()`/`StencilAddress()`: `ImageInfo.guest_address`/`stencil_addr`, `DepthValid()`, markers | resource `+0x30` |
| `z_write_base`, `stencil_write_base` | T | `DepthWriteAddress()`/`StencilWriteAddress()`: `write_buffer` variant of `ImageInfo` (depth copy, vk_rasterizer.cpp:3284-3305), `ClassifyDraw` (250-256) | unknown: when does the engine make the write base differ from the read base (DSV flags `+0x48` bits 1/2 read-only?) |
| `depth_size.pitch_tile_max/height_tile_max` | T | `Pitch()`, `Height()`: `ImageInfo.pitch`, size fallback, `ReadbackPolicy::NoteDepthOnly(pitch, height)` (vk_rasterizer.cpp:290-291, 384-385) | `Derive` pitch rule (256 alignment for depth) |
| `depth_slice.tile_max` | T | `GetDepthSliceSize()` -> `guest_size` | derivable |
| `depth_view.slice_start/slice_max` (`NumSlices()`) | V | `ImageDesc` ctor layer count (texture_cache.h:97-100), `ImageViewInfo(DepthBuffer, ...)` range (image_view.cpp:84-91), `DepthStencilCopy` sub range (vk_rasterizer.cpp:3299-3300) | DSV description (`+0x44` dimension, then the range) |
| `depth_view.z_read_only`, `stencil_read_only` | N | none | DSV flags `+0x48` |
| `depth_control` | F | `PrepareRenderState` gate (vk_rasterizer.cpp:640-641), `ImageViewInfo.is_storage = depth_write_enable` (image_view.cpp:87), `TargetInputsMatch`, begin signature (3033) | the depth-stencil state object (already in `FixedState`) |
| `depth_render_control` | F / op | `ClassifyDraw` (vk_rasterizer.cpp:230-256, takes `DrawRegs`, i.e. the register path only), `ClearDepthTarget` flags (729-730), begin memo (3051-3052), `FixedState.depth_clear/stencil_clear` via `TakeFromRegisters(StaleClear)` (off by default) | the engine's own clear/copy/decompress operations (B11); not draw state |
| `depth_htile_data_base` | M | `GetAddress()` -> `meta_info.htile_addr` (vk_rasterizer.cpp:642, 722, 3171) | resource, place unknown (resource `+0x48` is used as "has clear values" flag in `Synthesize`) |
| `depth_clear`, `stencil_clear` | C | `ClearDepthStencilImage` (vk_rasterizer.cpp, `ClearDepthTarget`, after line 731) | resource `+0x158` (float), `+0x15c` (byte), only when `resource+0x48 != 0` (native_draw.cpp:2001-2006 and the hash) |

### 1.3 Reading the table

* **Derivable now from the engine's resource and view objects** (candidate for the Task 2 comparison): colour/depth format (needs an RTV/DSV-specific mapping, see
  question Q3), extent and layers, mip slice, tile mode (via tiling index), pitch, base addresses, depth read base, clear values for depth.
* **Metadata that Vulkan never needs but the code still uses for one purpose:** cmask/fmask/htile addresses only feed `surface_metas` -> `GetMetaInfo` ->
  `SkipForEngineClear` (vk_rasterizer.cpp:3316): "is this fill the engine's CMASK/HTILE clear, then skip it". That one use can be kept with the resource addresses
  (cmask `+0x58`) or by recognising the fill from the engine's clear record (B11); I have not decided which.
* **No engine source known (before section 6):** `linear_general` for targets, `alt_tile_mode`, the fmask words. Colour `clear_word0/1`, `blend_bypass`, the cmask words, the depth write bases and the htile base were resolved by the Ghidra answers (section 6).
* **Dead fields (N rows):** about 25 colour and depth fields have no reader at all; they can be dropped with the structs, but not before the register image is.

## 2. Producers of `exec_state` / `DrawRegs`

`exec_state` is `Liverpool::exec_state` (liverpool.h:333), the command processor's private register block; `front_state` is the packet path's mirror.
Planning reads `exec_state` for typed fallbacks and `front_state` for the packet path (liverpool.cpp:603).

| # | writer | where | path |
|---|---|---|---|
| P1 | `SeedExecState` copies `front_state` once | liverpool.cpp:304-311, called from native_liverpool.cpp:279 | start |
| P2 | `MirrorRegsRaw` writes `front_state` and `exec_state` | liverpool.cpp:348-353; callers `MirrorRegs`/`MirrorObject` (liverpool.cpp:238, 365; native_liverpool.cpp:100-120 for state templates, 358, 360) | **old packet path / state templates** (the stream's `StateWrite` kinds Context, Sh, Uconfig, NumInstances, IndexType, ClearState) |
| P3 | `ExecApplyRecord` writes the record's register pairs per group, with group signatures and user data overlays | native_liverpool.cpp:124-190, called at 559 only when `!planned_from_record` | **typed path, fallback only** |
| P4 | `ExecSetWord` / `ExecRestore` / `ExecForget` overlay bookkeeping | liverpool.cpp:313-346 | typed fallback |
| P5 | depth/stencil clear (`draw.kind == 7`): record registers -> `exec_state` -> `MakeDrawInputs(exec_state)` -> `ClearDepthStencilTarget` | native_liverpool.cpp:338-350 | **always the register path** (the clear operation never uses `exec_inputs`) |
| P6 | `MirrorSetDefaults` -> `exec_state.SetDefaults()` | liverpool.cpp:370-378 | ClearState |
| P7 | `Synthesize` builds the ~100 register words per record | native_draw.cpp:1878 (single caller, 3000); `MaterializeRegs` (4762) for the A5.3 state packages | producer for P3, P5 and for `PlanExecOperation` |
| P8 | `PlanExecOperation` (fast clear eliminate, `draw.kind == 4`) applies the record's pairs straight into a fresh `DrawInputs` (`ApplyExecRegisters`, native_rasterizer.cpp:40-80, 440) | native_liverpool.cpp:491-493 | typed, but still register pairs (target words) |

Which of these the default scene uses, from the task numbers only: about 1.35 M typed draws are planned from the record (`PlanExecDraw`, so P3 is not run for them) against
about 1.4 k fallbacks per 20 s (fast clears, eliminates, a failed `Fixed::Build`). The fallback reasons are `exec_reason[0..7]` in native_draw.h
(0 switches off, 1 not typed, 2 no split, 3 fixed verify, 5 eliminate without a plan, 6 no stage key, 7 no fixed state). I did not guess counters for P2, P5 and P8.
The dump I need is listed in section 5.

## 3. Code that exists only for register-vs-native comparison

| item | callers outside the comparison | status |
|---|---|---|
| `MakeFixedState` (vk_fixed_state.cpp:433) | **in use**: native_rasterizer.cpp:549 (`FixedFor` when no plan), vk_pipeline_cache.cpp:480, vk_fixed_state.cpp:287 (`DefaultFixedState`), vk_rasterizer.cpp:495 (every non-native draw, `else plan->fixed = MakeFixedState(regs)`), 473 (verify) | in use (group (a)); deletable after the register image |
| `TakeFromRegisters` (vk_fixed_state.cpp:260) | vk_rasterizer.cpp:465 only; no-op unless `native_sdk_strip` bits 28-30 are set (0.3) | study switch; deletable with the register image (not "zero callers") |
| `DiffFixedState` (vk_fixed_state.cpp:347) | native_rasterizer.cpp:500 (`DiffExecPlans`), vk_rasterizer.cpp:467 (stale audit) and 482 (verify) | comparison only; deletable with the register image |
| `DiffDrawInputs` (vk_fixed_state.cpp:315) | native_rasterizer.cpp:499 only | comparison only; deletable with the register image |
| `DiffExecPlans` and the every-32nd-plan audit (`native_audit_plan`, native_liverpool.cpp:472-477, 592-606) | live: runs the register plan next to the typed plan | in use as a verifier; deletable with the register image |
| `TLG_FIXED_VERIFY_FIRST`, `native_fixed_verify` (native_liverpool.cpp:438-448) | default 0 | study switch; goes with `MakeFixedState` verify |
| `DiffShaderEnvFields`, `ApplyShaderEnv` compare (vk_pipeline_cache.cpp:175, 378-392) | runs per stage input build | comparison against `BuildRuntimeInfo(regs)`; deletable after the register image |
| `TLG_SYNTH_AUDIT` / A5.1 (`SynthAuditEntry`, `SynthAuditNote`, `SynthAuditRead`: native_draw.cpp:2563-2700, 9284; native_draw.h:138-140; read by native_zpass_gen.cpp:6150) | off by default | audits `Synthesize`; **deletable now** (no effect when off) but better together with `Synthesize` |
| `native_audit.{h,cpp}` (291 lines) | `StaleEnabled/NoteStale/PlanCaller` used in `PlanDraw` | the header describes the tracker audit (tracker deleted in E4): needs a read before Task 3, mixed live and dead |
| `native_census.{h,cpp}` (E0 census, `/emulation`) | 5 files outside | counters; keep until the last phase, then retire |
| `native_drain_audit`, `native_gbuffer_audit`, `native_resource_audit.h`, `native_zpass_audit`, `vk_identity_audit.h` | 2 to 5 files each | diagnostics, not comparisons against deleted paths; not in scope for E7 unless the owner says so |
| `MirrorRegs*`, `front_state`, `exec_state`, `Synthesize`, `DrawRegs`, `MakeDrawInputs` | many | wait for Task 2 numbers and an owner play (as the task says) |

Constants (native_updates.h): `B2NativeMode()` (line 56), `B8ImageMode()` (78) and `B8LifetimeMode()` (83) are `constexpr` returning 2, `B9TrackerOff()` (203) returns `true`; `B8FindMode`
(88), `B9BufferMode` (185), `B9LifetimeMode` (193), `B9QueryMode` (198), `B9PlaceOn` (209) and `B9AliasesOn` (214) are `constexpr` too (values not read by me yet). Their call sites
still contain the dead branches; that is Task 3 material, not touched.

## 4. Task 2 (built, not compiled)

`native/native_target_desc.{h,cpp}`, switch `TLG_TARGET_DESC=1` (default off; with it off the only cost is one `Enabled()` check per draw record). `Observe(ctx)` is called by the
record builder in native_draw.cpp right after the `Synthesize` / state package block, for draw records only (`kind <= 2`), so the engine objects are alive. It does not read the
register image: the console words are decoded straight from the view blocks (colour: 14 dwords at `view + 0x54` are the `ColorBuffer` layout, the extent word is dword 15;
depth: 8 dwords at `dsv + 0x58` are `DepthBuffer`, dword 8 is `DepthView`, dword 12 the extent), which is the same data `Synthesize` copies into the registers.

What it compares ("object" = engine objects, "words" = console words): data/number format (via `NativeImageDesc::Derive` of the resource and the view's own format), tile mode (tiling index),
pitch (mip 0 only), the hint extent against the mip extent, first slice and slice count (from the view description), sample count (resource `+0x194`), base address (mip 0 only),
colour cmask address and clear words 0/1 against resource `+0x58`, `+0x150`, `+0x154` (only when `info.fast_clear` is set), depth z format and stencil presence (small DXGI table: 40, 20, 55;
other formats are counted as `format_not_in_the_table` and show in the tuples).

Output: `/stats` -> native_backend.device.e2_targets (and the exit counters file). `colour` / `depth`: `observed` (every target of every record), `evaluated` (view or block changed since the thread last
saw the slot), `differing`, `differing_fields`, `extent_hint` (none / equal / larger / smaller than the mip extent). `tuples`: one entry per distinct `(dxgi_view, dxgi_resource, samples, info|z_info raw,
attrib|stencil_info raw, mip, dimension, aux)` with `count` = observations (answers Q2, Q3, Q5). `extent_shapes`: `(hint w/h, pitch, height from the words, resource w/h, mip)` with counts (Q4).
`first_differences`: the first 16 differing evaluations with both values of every differing field (also in the log and in user/log/rare_events.txt as `target_desc_differs`).
`aux` bits: colour 0 cmask words set, 1 fmask words set, 2 resource `+0x58` set; depth 0 write base = read base, 1 htile base word set, 2 resource `+0x48` set, 3-4 DSV flags (`+0x48` & 3).

Limits: operations (fast clear eliminate, depth clear, resolve) are not observed (their `ctx` targets are not the operation's); `Derive` counts a "no view" or "view with desc" in its own E2 counters
for each call, so `e2_derive` is slightly inflated in a run with this switch on; `static_assert`s check `sizeof(ColorBuffer) == 15 dwords`, `DepthBuffer == 8 dwords`, `DepthView == 1 dword`
(they hold by the register index asserts in regs.cpp, but I could not compile them). Not compiled: only `clang -fsyntax-only` on the new file, whose own errors were limited to the `LOG_WARNING`
line, which fails here because the container's spdlog is a different version from the port's.

### 4.1 Changes after the Ghidra answers (section 6)

The comparison now also uses the resource's own T# (`resource + 0x70`, 32 bytes, read as `AmdGpu::Image`): `t_data_format` (`cb.info.format` against the T# data format), `t_pitch` and `t_tile_mode`
(mip 0), plus `blend_bypass` (against the integer number types 4 and 5), `fast_clear_flag` (against resource `+0x58 != 0`), the cmask address (mip 0), the colour clear words (against the prebuilt block at
resource `+0xdc`/`+0xe0`, 0 above mip 0), and for depth `z_write_base`, `stencil_address` (resource `+0x40`), `htile_base` (resource `+0x48` >> 8, block word 9) and `htile_enable`.
`colour.clear_colour_in_resource` counts how often the block's clear words equal the packed colour at resource `+0x150`/`+0x154` (when `+0x14c != 0`) and how often not; that is the B11 effect in numbers.
Each tuple now carries the resource T# (`t_data_format`, `t_number_format`, `t_dst_sel`) next to the raw `info` and `attrib` words: from those I can build the T#-to-info table (number type and comp_swap
go through the small tables `DAT_025e1250` and `FUN_0246be80`).

## 5. What I need from the local side

Dumps (no Ghidra), from a headless default-scene run of 40 s with a new census switch if you want one, otherwise from the existing debug server:

* **Q1.** `/stats` `native_draw.exec_reason[0..7]`, `plan_null[0..2]`, `exec_typed`, `exec_fast_clear`, `exec_fallback`, `state_planned`, and how many kind-7 depth clears per 20 s. This settles
  which of P2, P5 and P8 run in the default scene.
* **Q2.** For every distinct render target seen: `(DXGI format of the RTV, ColorBuffer.info.raw, ColorBuffer.attrib.raw & 0x7fff, linear_general, resource sample count at +0x194)`. With this
  I can build the RTV format table (data format, number type, comp_swap, blend_bypass) from data instead of guessing, and see whether `attrib.tile_mode_index` equals the T# tiling index.
* **Q3.** Same for depth: `(DXGI format of the DSV, z_info raw, stencil_info raw, read base == write base?, sample count)` and whether `z_info.tile_surface_enable` is ever set.
* **Q4.** For the extent word: `hint.width/height` against `Object.width/height >> mip` and against `Pitch()/Height()` of the same target (equal, padded, or different), and how many RTV/DSV
  bindings have a mip slice above 0 or an array/MS dimension.
* **Q5.** How many draws bind a target with more than one sample, and whether any binds a target with `fmask_base_address != 0` (E0 counted FMASK reads, but I do not have the number).

Ghidra (addresses from the task text; I have no decompiler access and did not guess):

* **Q6.** `FUN_0246a420`: the mapping from the packed format word to `CB_COLORi_INFO.format`, `number_type` and `comp_swap`, per D3D11 format used. Alternative: Q2 gives it as data.
* **Q7.** Where does the fast-clear colour (`clear_word0/1`, block words 11 and 12) get written into `resource + 0xb0..0xef`, and from what? The B11 note says it differs from the colour passed
  to `ClearRenderTargetView` in 470 of 16,656 clears. This is the one field with no other source.
* **Q8.** `FUN_02492370` (depth block): how `z_write_base`/`stencil_write_base` are set against the read base (DSV flags `+0x48` bits 1 and 2?), where the htile base (block word 9) and
  `tile_surface_enable` come from in the resource, and what `resource + 0x48` means (it is the guard for the clear values).
* **Q9.** The second 32 byte block at RTV `+0x74` for MS targets (fmask words), and where the cmask fast-clear enable bit sits.
* **Q10.** `ctx + 0x130e0` (target 7): confirm it is an object of the same kind (tag `0x16`) as the other seven, since `Synthesize` treats it as a view.

## 6. Ghidra answers (owner, eboot, FUN_02491080 / 02491a30 / 02492370 / 02492ce0 / 0246a420 / 02483380)

* **Colour info word** (Q6): `FUN_0246a420` builds it from one packed word (bits 0-7 T# data format, 8-11 T# number format, 12-23 T# dst_sel), which `FUN_02491a30` takes from the resource T# at
  `resource + 0x70` (dword 1 data/number format, dword 3 dst_sel). Layout: `format = data format << 2`; `number_type = DAT_025e1250[number format] & 7` (bits 8-10); `comp_swap = FUN_0246be80(...) & 3`
  (bits 11-12); bit 15 for number types 0, 1, 6; `blend_bypass` (bit 16) for number types 4 and 5; the word is XORed with 0x40000. So the colour target format is a function of the resource T#, not of the DXGI
  format. The E2 tables (DXGI to T#) can be reused plus a second small table T# to info; one dump of `(T# data, num, dst_sel) -> (info.raw)` checks it (the tuples now carry both).
* **Clear words** (Q7): `ClearRenderTargetView` (0x2483380) packs the colour per format (sRGB included) into resource `+0x150/+0x154` and sets `+0x14c = 1` and a serial at `+0x160`. The RTV builder
  copies `+0x150` into view `+0x80/+0x84` first, then overwrites both with block words 11/12 (prebuilt block at `+0xb0`: `+0xdc/+0xe0` for mip 0; 0 for mip above 0). Consequence: do not derive `clear_word0/1`
  from the view; a native path takes the colour from the clear call or from resource `+0x150/+0x154`. B11 already clears natively from the call arguments.
* **Depth block** (Q8): `z_read_base` and `z_write_base` are both resource `+0x30 >> 8`; the stencil bases both resource `+0x40 >> 8` (read-only is a view flag, not a different address). The htile base is
  resource `+0x48 >> 8` (block word 9) and `tile_surface_enable` is set iff `+0x48 != 0` (the same field guards the clear values at `+0x158..0x17c`). For mip 0 / MS tex2d the block is copied from
  `resource + 0xf0..0x11f` and the slice range is patched (`FUN_0247da00`).
* **Cmask / fmask** (Q9): cmask base is resource `+0x58 >> 8` into block word 7 and `info.fast_clear` is set iff `+0x58 != 0`. `FUN_02491a30` never writes the fmask words (9, 10); a nonzero fmask can only come from
  the prebuilt block (mip 0, MS). Neo-mode DCC (`resource + 0x8a & 0x20`, `+0x8c` into word 13) is not needed for Vulkan.
* **Target 7** (Q10): not looked at. `Observe` reads `ctx + 0x130e0` as a view pointer, exactly as `Synthesize` does.

What this resolves from section 1: `blend_bypass` (function of the number type), the cmask words and `fast_clear` (resource `+0x58`), the depth write bases and the htile base (resource), and the clear words (a
resource field, not a view one). Still open for the target replacement: the number-type and comp_swap tables, `linear_general` and `alt_tile_mode` for targets, and the fmask words.

## 7. Round 2 (after the first local run: colour 32,501 evaluated / 1,610 differ, depth 9,246 / 825, 6,785 depth `unknown_format`)

Tile mode, display buffers (view ids 66/69/72, DXGI 28, attrib 0x14a, info 0x8028): the "14" came from `NativeImageDesc::Derive`, which guesses the tiling index from the bind flags (`0x20` -> 14,
native_image_desc.cpp:116). It cannot tell a display surface from another render target, and nothing in the D3D11 description I can see (bind flags, usage, cpu access) does either. What the engine
itself uses is the T# it keeps in the resource (`resource + 0x70`, the description `FUN_02491a30` builds the target block from), so the shadow's `tile_mode` now compares the words with that T# index,
at every mip, and `Derive`'s guess is only listed as a fact. I could not use Ghidra from here (no decompiler in this container), so I did not look at how the swap chain creates the resource. The new
`tile_cases` table lists, for every resource a target was bound on: bind flags (`+0x1a0`), MiscFlags (`+0x1a8` for a 2D texture), usage, cpu access, resource kind, the T# index, the words' tile mode and
`Derive`'s index, so one run shows whether any object field separates the display buffers from the other targets if the T# index turns out not to be 10 for them. Expectation: `tile_mode` differing = 0.

Depth `num_slices` (825 of 9,246): the shadow expected the single slice a plain 2D view names (dimension 3: MipSlice only). From Ghidra Q8 the block of a mip 0 plain view is copied from the resource's
prebuilt block (`resource + 0xf0`) and only the array dimensions patch the slice range, so the words carry the slice range of the whole resource. The shadow now expects `first 0, count = resource layers` for
dimension 3 at mip 0 (a hypothesis, not a fact). Two tables decide it: `prebuilt_block_cases` (which words of each view block differ from the resource's prebuilt block, as a bit mask, per dimension; for
depth 12 words at `+0xf0`, for colour 16 words at `+0xb0`) and `slice_cases_that_differ` (dimension, mip, resource layers, expected and words' first/count, the DB_DEPTH_VIEW word). If the mask is 0 for dimension
3 / mip 0 the words are the prebuilt block and the cache's image (which has the resource's layers) is counted right by the words; the view range for Vulkan still comes from the description (one slice).
If the hypothesis is wrong the table says what the words hold instead. Expectation: `num_slices` differing = 0.

Depth format table: D24_UNORM_S8_UINT (view 45, resource R24G8_TYPELESS 44) -> Z32 float + Stencil8 (seen: z_info 0xa0000003, stencil_info 0x20000001, T# data format 4, number format 7). The table now
covers the four formats a D3D11 depth stencil view can have (20, 40, 45, 55) and maps a view with format 0 to its resource's typeless format (19, 39, 44, 53). D32_FLOAT_S8X24_UINT (20) has not been
seen; it is assumed to go the same way as 45. `format_not_in_the_table` is 0 by construction for D3D11 formats; a value outside that set (the format switch in `FUN_02474fb0` may accept others, I could not
read it) would be counted there. Expectation: `format_not_in_the_table` = 0 and `z_format` / `stencil_format` differing = 0.

Colour clear words when `resource + 0x14c` is 0: new counters `colour.clear_colour_not_set_in_resource` (count, both block words zero, block words equal `+0x150/+0x154`) and the table
`clear_words_when_resource_14c_is_zero` (block words 11/12, resource +0x150/+0x154, prebuilt +0xdc/+0xe0, mip, fast_clear bit). The comparison with the prebuilt block (`clear_word0/1` differing) stays on.

Extent (FYI from the local run): the engine's extent word equals the mip extent in all cases and the register decode pads heights (1088 / 2176 / 640) because of tile alignment, so the extent word is the right
source for the image size and `Pitch()` / `Height()` only matter for `guest_size`.

### Fields the shadow cannot (yet) decide from the engine's objects
* `number_type` and `comp_swap` (colour info bits 8-12): they go through `DAT_025e1250` and `FUN_0246be80` from the T# number format and `dst_sel`. The shadow does not compare them; the tuples carry
  the T# (`t_number_format`, `t_dst_sel`) and the raw `info` word, which is what builds the table. Decidable from the T# once the table exists, not from the D3D11 description alone.
* `alt_tile_mode` (Neo only) and `linear_general`: no object field known. `linear_general` is 0 for the display buffers (info 0x8028 has bit 7 clear); the raw `info` in the tuples shows the rest.
* the fmask words (9, 10): written only by the prebuilt block; `prebuilt_block_cases` shows whether a view block ever differs from it there.
* the colour clear colour of the engine's clear call: resource `+0x150/+0x154` (valid when `+0x14c != 0`), not the view; a native path takes it from the call.

## 8. Round 3 (after round 2: colour 0 differences, depth format gaps 0)

* **Slices.** The words hold the first slice and the last slice (`slice_max` is an index; `NumSlices()` is last + 1, not the count: db_view word 0x2001 = first 1, last 1). The comparison is now
  `slice_start == first` and `slice_max == first + count - 1` for colour and depth (field `last_slice` replaces `num_slices`). The `dimension 3 / mip 0` rule of round 2 (whole resource range) stays; it
  agrees with the new formula because it has first 0. Expectation: `first_slice` and `last_slice` differing = 0. `slice_cases_that_differ` now lists `words_last`.
* **Display tiling.** The three display buffers are the resources with bind flags 0x128 (misc 0, T# index 10); every other colour target has T# index 14. The target description now says display tiling (index 10)
  for a colour target with the engine's bind flag 0x100 (a flag no Direct3D 11 bind flag uses) and `Derive`'s index otherwise; `NativeImageDesc::Derive` itself is NOT changed, because the sampled images
  that decide from it are played and a change there would reach the GPU. The comparison with the words is `tile_mode`; the T# comparison stays as `t_tile_mode` (colour and depth). Expectation: both 0.
* **Number type, comp_swap and the rest of the info word.** The tables behind `FUN_0246a420` (`DAT_025e1250`, `FUN_0246be80`) are not known to me, and the first-run tuples are not in this session, so the
  shadow learns them: the first value a key produces is the table, and a later different value for the same key is a difference.
  * `number_type` is learned by the T# number format (`learned_number_type`).
  * `comp_swap` is learned by (T# data format, T# number format, T# dst_sel) (`learned_comp_swap`); `learned_comp_swap_without_number_format` shows whether the data format and dst_sel are enough.
  * Closed-form bits from your Q6 text: `blend_clamp` (bit 15) = number type in {0, 1, 6}; `blend_bypass` (bit 16) = number type in {4, 5}; `info_bit18` = 1 (the XOR 0x40000).
  Differences are counted in `differing_fields` as `number_type`, `comp_swap`, `blend_clamp`, `blend_bypass`, `info_bit18`. The learned tables are in the report with their counts, so the next round can
  turn them into constants and a real comparison. Expectation: all of these 0 if the output really is a function of those keys; a nonzero `comp_swap` or `number_type` count says an input is missing.

Still not decidable from the engine's objects: `alt_tile_mode`, `linear_general` (0 for the display buffers), the fmask words (only the prebuilt block writes them), the clear colour of the engine's clear call
(resource `+0x150/+0x154`, not the view), and the bits of the info word above 18 (cmask_is_linear, blend_opt_*, fmask_*, dcc_enable, cmask_addr_type): the word's known bits are format (2-6), number type
(8-10), comp_swap (11-12), fast clear (13, resource `+0x58`), blend_clamp, blend_bypass and bit 18.

## 9. Checklist for the next step: replacing the register decode with this description

Verdicts: **decided** = built from the engine's objects, 0 differences in the local runs reported to me (rounds 1 to 3: colour only `info_bit18` differed in round 3, depth 0); **constant** = a
fixed value or a small table; **not decidable** = no engine field known (why); **not compared yet** = the shadow does not check it, so I cannot call it decided. Round 4 adds the constants for
`number_type`, `comp_swap` and bit 18 and one new check (`slice_size`), all unrun.

### `AmdGpu::ColorBuffer` (dwords in block order)
| field | verdict | how / why |
|---|---|---|
| `base_address` (word 0) | decided at mip 0 | `resource + 0x30` (`Object::address`), `address` 0 differences. Mip above 0: the block holds the mip's own address; not compared yet |
| `pitch.tile_max` (1) | decided at mip 0 | T# pitch (`resource + 0x70`) and `Derive`'s pitch rule, `pitch` / `t_pitch` 0 differences. Mip above 0: not compared yet |
| `pitch.fmask_tile_max` (1) | not needed | no reader; fmask only |
| `slice.tile_max` (2) | not compared yet (check added in round 4: `slice_size`) | the tile aligned height (1080 -> 1088, 2160 -> 2176, 540 -> 640: not one alignment) comes from the console's tiling size math, not from the D3D11 description. The engine keeps the result in the prebuilt block (`resource + 0xb0`, word 2) and the allocation size at `resource + 0x38`; `slice_size` compares `GetColorSliceSize() * layers` with `Object::bytes` for one-mip textures. The image size itself comes from the extent word (equal to the mip extent in all cases) |
| `view.slice_start` / `slice_max` (3) | decided | view description first slice and last slice (`first_slice`, `last_slice`, 0 differences); `slice_max` is the last index |
| `info.endian` (bits 0-1) | constant 0 | `FUN_0246a420` sets `format << 2` and nothing in bits 0-1; no reader |
| `info.format` (2-6) | decided | the T# data format (`resource + 0x70`), `t_data_format` 0 differences |
| `info.linear_general` (7) | not decidable | no engine field known; 0 in the three info words quoted (0x8028, 0xa028, 0x42718); read by `Address()`, `BaseSlice()`, `GetTileMode()` only |
| `info.number_type` (8-10) | constant (round 4) | T# number format -> number type: identity for 0-5 and 7, 9 (Srgb) -> 6; entries 6 (SnormNz), 8 and 10-15 unknown (not read in Ghidra by me); `number_type_constants` marks `observed` per entry, `number_type_not_in_the_constant_table` counts the rest; the learned table checks it |
| `info.comp_swap` (11-12) | constant rule (round 4) | from the T# data format (channel count) and dst_sel: the stored channel `c` is read by the component whose select is `4 + c`; the swap whose row in `ColorBuffer::Swizzle()`'s table has that component order is the answer (BGRA `0xf2e` -> 1, RGBA/RG/R/RGB -> 0 for all six cases seen). Derived from the repo's swizzle table, not from `FUN_0246be80`, which I did not read; rows for 1 to 3 channels other than standard are unseen. `comp_swap_rule_unmatched` / `_ambiguous` count the cases it cannot place |
| `info.fast_clear` (13) | decided | `resource + 0x58 != 0`, `fast_clear_flag` 0 differences |
| `info.compression` (14) | not decidable | no engine field known; only the fmask address reads it; 0 in the quoted words; needed only if multisample colour targets exist |
| `info.blend_clamp` (15) | constant rule | number type in {0, 1, 6}; `blend_clamp` 0 differences |
| `info.blend_bypass` (16) | constant rule | number type in {4, 5}; 0 differences |
| `info.simple_float` (17) | not needed | no reader; 0 in the quoted words |
| `info.round_mode` (18) | constant rule (round 4) | `(u ^ 1)` with `u = 1` for number types 0, 1, 6, else 1 -> 0 for those types, 1 for the others (your correction of Q6); `info_bit18` |
| `info.cmask_is_linear` (19), `blend_opt_*` (20-25), `fmask_compression_disable_ci` (26), `fmask_compress_1frag_only` (27), `dcc_enable` (28), `cmask_addr_type` (29-30) | not needed | no reader; 0 in the quoted words |
| `info.alt_tile_mode` (31) | not decidable | Neo only (`ImageInfo.alt_tile` and the detile / cache keys read it); no engine field known |
| `attrib.tile_mode_index` (0-4) | decided | T# tiling index, and display tiling 10 for bind flag `0x100`, 14 otherwise (`tile_mode`, `t_tile_mode` 0 differences) |
| `attrib.fmask_tile_mode_index`, `fmask_bank_height`, `num_samples_log2`, `force_dst_alpha_1` | not needed | no reader (`fmask_tile_mode_index` is 10 in the display buffer's word 0x14a: a copy of the tile mode) |
| `attrib.num_fragments_log2` | decided | `SampleDesc.Count` at `resource + 0x194`, `samples` 0 differences |
| `pad0` (6) | not needed | Neo DCC base (`resource + 0x8c`), not used by Vulkan |
| `cmask_base_address` (7) | decided at mip 0 | `resource + 0x58`, `cmask_address` 0 differences |
| `cmask_slice.tile_max` (8) | not needed | no reader |
| `fmask_base_address`, `fmask_slice` (9, 10) | not decidable | only the prebuilt block writes them (`FUN_02491a30` never does); no engine field named; no reader except the fmask address of `meta_info`; `prebuilt_block_cases` shows whether a view block ever differs from the prebuilt one |
| `clear_word0/1` (11, 12) | decided (round 6): the block the engine keeps | the view's own block words (`view + 0x80/+0x84`, = what the register decode copies) for a view, the resource's prebuilt block (`+0xdc/+0xe0`) for the eliminate; NOT the clear call's colour (`resource + 0x150/+0x154`), which differs from the registers in some clears (round 5 long play: 5,378 of 62,065 eliminates) |
| `pad1` (13, 14) | not needed | unused |
| the extent word (15), `CbDbExtent` | decided | mip extent of the resource (equal in 100 % of cases) |

### `AmdGpu::DepthBuffer` and the depth words around it
| field | verdict | how / why |
|---|---|---|
| `z_info.format` | decided | DXGI depth format of the view (or the resource's typeless one): 20 / 45 -> Z32 float + Stencil8, 40 -> Z32 float, 55 -> Z16; `format_not_in_the_table` 0 |
| `z_info.num_samples` | decided | `SampleDesc.Count`, `samples` 0 differences |
| `z_info.tile_mode_index` | decided | `Derive` index (0 for depth) and T# index, `tile_mode` / `t_tile_mode` 0 differences |
| `z_info.tile_surface_enable` | decided | `resource + 0x48 != 0`, `htile_enable` 0 differences |
| `z_info.tile_split`, `decompress_on_n_zplanes`, `allow_expclear`, `read_size`, `clear_disallowed`, `zrange_precision` | not needed | no reader (`zrange_precision` is 1 in the quoted word 0xa0000003) |
| `stencil_info.format` | decided | same table (`stencil_format` 0 differences) |
| `z_read_base`, `z_write_base` | decided | both `resource + 0x30 >> 8` (read-only is a view flag), `address` / `z_write_base` 0 differences |
| `stencil_read_base`, `stencil_write_base` | decided | both `resource + 0x40 >> 8`, `stencil_address` 0 differences |
| `depth_size.pitch_tile_max` | decided at mip 0 | T# pitch / `Derive` pitch, `pitch` / `t_pitch` 0 differences |
| `depth_size.height_tile_max` | not compared yet | tile aligned height, same remark as `slice.tile_max`; the image size comes from the extent word |
| `depth_slice.tile_max` | not compared yet (check added in round 4: `slice_size`) | `GetDepthSliceSize() * layers` against `Object::bytes` for one-mip textures; the stencil plane is a separate allocation (`resource + 0x40`) so it should not be in `bytes` (to be seen) |
| `depth_view.slice_start` / `slice_max` | decided | first and last slice of the view description; plain 2D views at mip 0 carry the whole resource's range (`last_slice` 0 differences) |
| `depth_view.z_read_only`, `stencil_read_only` | not compared yet | no reader; the DSV flags (`+0x48` bits 0 and 1) are the likely source |
| `depth_htile_data_base` | decided | `resource + 0x48 >> 8` (block word 9), `htile_base` 0 differences |
| `depth_clear`, `stencil_clear` | not decidable (round 6): registers keep it | `resource + 0x158` / `+0x15c` (valid when `resource + 0x48 != 0`) are compared and counted (`compare_only_differs`, table `clear_cases_depth_values_vs_registers`) but not applied: the long play had a few differing cases (1 in the depth clear path, 3 at the fill point) |
| the depth extent word (12) | decided | mip extent of the resource |
| `depth_control`, `depth_render_control` | out of scope here | draw state (`FixedState`) and the engine's own clear / copy / decompress operations (B11), not target description |

### What the replacement still needs that the shadow does not give
1. The tile aligned sizes (`slice.tile_max`, `height_tile_max`, `depth_slice.tile_max`) for `guest_size` and `mips_layout`: either the repo's tiling size function with the T# as input, or the engine's prebuilt block; `slice_size` will show which agrees with the allocation.
2. Mip above 0 targets: address, pitch and the recomputed block (`FUN_02491a30` recomputes them from the T#); not compared. Whether the game binds any is visible in `tuples` (`mip` column).
3. The clear colour of a fast clear: from the clear call or resource `+0x150/+0x154`.
4. Multisample colour targets (fmask, `compression`): not decidable; the `tuples` `samples` column says whether any exist.

## 10. Round 5: the description feeds the draw (`TLG_TARGET_DESC_USE=1`, off by default) and its verify mode (`TLG_TARGET_DESC_USE_VERIFY=1`)

**Switches** (each read once): `TLG_TARGET_DESC_USE=1` applies the description; `TLG_TARGET_DESC_USE_VERIFY=1` builds it and compares, and always keeps the register result (it also turns the build on by itself);
`TLG_TARGET_DESC=1` (the shadow of rounds 1 to 4) is independent. With none of them on, the only cost is the `UseOn()` / `Enabled()` checks. Nothing is deleted.

**Where the description is made and where it is used.** The engine's objects can be gone by the time the command processor plans a draw, so the description is made by the record builders on the game's thread and
travels with the record as an immutable `TargetSet` (equal sets are shared by content, 4,096 at most; the pointer is `NativeRecord::native_targets`, valid for `native_targets_seq == seq`, copied into
`ExecDraw::native_targets` by `TakeExecDrawImpl`):
* **fill point** (`PlanDraw`, vk_rasterizer.cpp, right after `plan->inputs = MakeDrawInputs(regs)` and before the `exec_inputs` insert): the set of the draw is read through `Core::NativeTargetDesc::Scope`, set by
  native_liverpool.cpp around the register-path plan (`make_candidate`). The set is made by `PrepareDraw(ctx)` in the draw record builder (kind <= 2), from the same `ctx` slots as `Observe` (colour 0-6 by the
  count, 7 at `ctx + 0x130e0`, depth at `ctx + 0x44d0`), cached per thread while the targets' hashes do not change.
* **depth clear** (kind 7, `ConvertClearQuad`): `PrepareDepthClear(target)`. `target` is a block pointer, not an object: the DSV is looked for at `target - 0x58` and the resource at `target - 0xf0` and either is
  accepted only if the engine's table says the object at that address is a depth stencil view / a texture. Applied in native_liverpool.cpp after `MakeDrawInputs(exec_state)`.
* **fast clear eliminate** (kind 4, `EmitFastClearEliminate`): `PrepareEliminate(resource)`, the colour target of the resource (mip 0, first slice 0, last slice layers - 1). Applied in `PlanExecOperation`
  (native_rasterizer.cpp) after the record's registers.

**What is replaced** (everything else keeps the register decode and is counted as `register_only`): the `Decided` fields of section 9, by dword group: colour words 0 (base), 1 (pitch), 2 (slice size, prebuilt), 3 (first and
last slice), 4 (format, number type, comp_swap, fast clear, blend_clamp, blend_bypass, bit 18), 5 (tile mode, fragments), 7 (cmask base), 11/12 (clear words); depth words 0 (format, samples, tile mode, tile surface),
1 (stencil format, tile surface), 2-5 (bases), 6 (pitch, tile aligned height from the prebuilt block), 7 (slice size, prebuilt), the depth view (first and last slice), the htile base, the depth and stencil clear values.
The bits are merged into the register words (`(register & ~mask) | description`), so a field the description does not cover stays as the register path made it; the `DrawInputs` members `depth_control` and
`depth_render_control` and the extent hints are not touched. A linear general colour target keeps the registers' `slice_start` (it holds address bits).

**Mip above 0, tile aligned sizes, clear colour** (item 3): the prebuilt block of the resource (colour `+0xb0`, depth `+0xf0`) holds mip 0 only, so at mip 0 the slice size (colour word 2) and the depth height and slice size come from it
(`source_prebuilt_block` counts each use); the clear words come from the clear call's colour (`resource + 0x150/+0x154`) when `resource + 0x14c != 0` (`source_clear_call_colour`), else from the prebuilt block (`+0xdc/+0xe0`).
For a mip above 0 there is no engine source (`FUN_02491a30` recomputes it with the SDK size functions): address, pitch, slice size, cmask and clear words stay with the registers and are counted as `register_only`;
`targets_above_mip_0_with_register_geometry` counts the targets.

**Counters** (/stats native_backend.device.e2_targets.feeds_the_draw): `fill_point`, `depth_clear`, `eliminate`, each with `colour_targets_applied`, `depth_targets_applied`, `targets_differing_from_registers`,
`description_without_registers`, `registers_without_description`, `no_set_for_the_record`, `object_not_found`, and per field of colour and depth (`colour_fields`, `depth_fields`): `class` (decided / not_needed /
not_decidable), `description_used`, `register_only`, `differs_from_registers`, `source_prebuilt_block`, `source_clear_call_colour`. Expectation with the verify run: `differs_from_registers` 0 everywhere,
`description_used` close to the number of applied targets for the decided fields, `register_only` only for the not-needed / not-decidable fields and for mip above 0 geometry.

**What the verify run cannot cover.**
1. Only the fill of an `exec_inputs` entry is compared: the 99.9 % of draws that hit the cache reuse the entry, so a later draw with the same register key but different engine objects is not seen.
2. Draws without a set: the packet path, draws the native path does not take (`no_set_for_the_record`), and any record whose set could not be made (`object_not_found`, `sets_overflow`).
3. Mip above 0 and multisample colour targets (fmask, compression): register source, counted, not verified; the eliminate's slice range (first 0, last layers - 1) is assumed for the whole resource.
4. The depth clear's target is found by address arithmetic and the engine's table; if neither candidate matches, the path stays on registers (`object_not_found`).
5. Verify compares with the register decode the description was already checked against in the shadow; a value both get wrong from the same prebuilt block is invisible, and what the picture looks like needs the owner's play with `TLG_TARGET_DESC_USE=1` (without verify), after a clean verify run.
6. Timing: the set is made when the record is built and applied when it is planned; a resource field that changes in between (the clear colour at `+0x150`) is the case `differs_from_registers` would show.

Syntax check of this round (container with Vulkan, xxhash, abseil, robin-map headers added; spdlog and magic_enum are other versions or absent): `native_target_desc.cpp` and `native_draw.cpp` compile with no error on the new lines
(the remaining errors are the container's spdlog `LOG_*` macros and Windows types); `native_liverpool.cpp`, `native_rasterizer.cpp` and `vk_rasterizer.cpp` stop at the missing magic_enum header, so their one-line
edits are checked by reading only.

## 11. Round 6: the clear words and the clear values (after the long play with `TLG_TARGET_DESC_USE=1`)

Long-play counters of round 5: eliminate path, `clear_word0` differed from the registers in 5,378 of 62,065 targets (all 62,065 taken from the clear call's colour); fill point 353 of 536,331 (304,237 from the
clear call, 232,094 from the prebuilt block); depth clear values: 1 case in the depth clear path, 3 at the fill point. The picture was fine (the cases are rare or harmless), but the rule was wrong in principle: B11 says
that where the clear call's colour and the register words disagree, the registers are right.

**What I did not have.** The round 5 build kept no table of the differing cases, so I did not see the tuples; the rule below follows from the facts known so far, and the new tables (below) will confirm or refute it
on the next long run. The facts: (1) the register decode of a view copies the view's block words 11/12 (`Synthesize`: 14 words from `view + 0x54`), and the eliminate record's registers are the resource's prebuilt block
(`EmitFastClearEliminate`: `resource + 0xb0 + k * 4`); (2) the clear call stores its packed colour at `resource + 0x150/+0x154` and sets `+0x14c` (a state: the eliminate is built when it is 1 or 2, so an eliminate always
comes after a clear call), and the Ghidra reading says the view's words are overwritten by the prebuilt block's; (3) in 5,378 of 62,065 eliminates `+0x150` is not the prebuilt word.

**The rule (applied).** The clear words come from the same block the registers copy, as an engine object field: `view + 0x54 + 11 * 4` and `12 * 4` for a view (fill point), the resource's `+0xdc/+0xe0` for the eliminate
(no view). The clear call's colour is no longer a source; it is only recorded. By construction `differs_from_registers` is 0 for these two fields (same memory, same moment); what remains to see in the tables is how
often the clear call's colour differs from them. Field class: decided (source: the engine's block, not a description derived from D3D11); for the next step (removing the register decode) the words must come from the view's
block or the prebuilt block, which is the engine's console data, not a D3D11-derived value: there is no D3D11 source for a clear word.

**Depth and stencil clear values.** The resource's `+0x158/+0x15c` do not always equal the register decode (the registers hold what the clear's setup wrote, or Synthesize's copy at the draw). They are no longer applied:
class not decidable, registers keep the field, the resource's values are compared (`compare_only`, `compare_only_differs`) and every differing case goes to a table.

**Tables** (/stats native_backend.device.e2_targets.feeds_the_draw; at most 64 distinct tuples each with counts, `overflow_beyond_64_tuples` and `total_cases` beside them):
* `clear_cases_description_vs_registers`: the description's clear words are not the register decode's (applied fields; expected empty).
* `clear_cases_clear_call_colour_vs_registers`: the clear call's packed colour (`+0x150/+0x154`) is not the register decode's, with `+0x14c`, the serial (`+0x160`, raw), the prebuilt words, the view block words, the register words, the DXGI format
  and the mip: this is the table that explains the 5,378 and 353 cases. Whether the eliminate came before or after the clear call: the eliminate record is only built for `+0x14c` 1 or 2, so it is always after; the raw state
  value is in `resource_14c`.
* `clear_cases_depth_values_vs_registers`: the resource's depth and stencil clear values against the register decode, with the htile word and the format. The "packed value the clear call would write" is the resource's own field, which
  is the packed value; I do not compute it separately.
Each key: path (fill_point / depth_clear / eliminate), kind, `resource_14c`, `serial_160`, `resource_150_or_158`, `resource_154_or_15c`, `prebuilt_dc`, `prebuilt_e0`, `register_word0/1`, `description_word0/1`, `dxgi_view`, `mip`,
`view_block_word11_or_htile`, `view_block_word12`, `count`. (The serial at `+0x160` is read as a raw dword; the first Ghidra reading also called `+0x160` a memory pointer, so treat it as unverified.)

**Cumulative counters for the whole session.** Every counter here is process-lifetime and never reset. A thread writes the full report to `user/log/target_desc_counters.json` (written aside and renamed) every 30 s from the first
time any of the three switches (`TLG_TARGET_DESC`, `TLG_TARGET_DESC_USE`, `TLG_TARGET_DESC_USE_VERIFY`) builds something, and once more at a normal exit (`atexit`); the debug server's own snapshot (`last_counters.json`, every 20 s) carries the same
report under `/stats`. Verify mode behaves as before: it compares and keeps the register result.

New counter columns per field: `source_view_block` (replaces `source_clear_call_colour`), `compare_only`, `compare_only_differs`.

## 12. E7b (branch cloud-e7b, from native-v2 5fbaf71): the four tasks of the third prompt

(The prompt called the Task 3 notes "section 10"; sections 10 and 11 were taken by rounds 5 and 6, so they are 12 to 16 here.) Everything below is shadow or census, off by default, nothing deleted.
Not compiled in full: each edited file was syntax-checked in a container with the Vulkan, xxhash, abseil, robin-map headers installed and stubs for magic_enum; the only errors left are the container's
spdlog version, an old Vulkan header (`eMesaKosmickrisp`) and Windows types; none is on a new line.

| switch | where the counters are |
|---|---|
| `TLG_SYNC_TYPED=1` | /stats native_backend.device.sync_typed |
| `TLG_TARGET_REGONLY=1` | /stats native_backend.device.e2_targets: `feeds_the_draw.<path>.colour_fields / depth_fields` (`compare_only`, `compare_only_differs`), `register_value_histograms`, `depth_clear_call` |
| `TLG_PLAN_SHADOW=1` | /stats native_backend.device.plan_shadow |
| `TLG_E3_CENSUS=1` | /stats native_backend.device.e3_census |
`last_counters.json` carries /stats every 20 s; `user/log/target_desc_counters.json` carries the target description report every 30 s.

## 13. Task 1: the sync producers (TLG_SYNC_TYPED=1)

**Not possible from the cloud as asked: hooking the engine functions.** A hook is an entry in the owner's hooks file (a name, an address and the expected bytes at it, read from the binary); this container has neither
the eboot nor a decompiler, so I cannot find the other writers, say which arguments carry their fields, or write a hook entry whose expect bytes are right. What is built, and what is left:

* **How the packets get to the tags today** (read in native_draw.cpp): only `FenceEop` is hooked on the producer side (`TagFenceSignal`, hook name `FenceEop`, Ghidra 0x2480d60): after the game's function returns, the hook
  converts the 6 dword end-of-pipe packet at the command buffer cursor it had before (`ctx + 0x47d0`). Every other sync packet (compute release memory 0x49, wait on label 0x3c, CPU write to a label 0x37, CP DMA 0x50, the
  occlusion dump event) is found by `SweepSyncPackets` when the next tag is written or the command buffer is submitted, and converted by `ConvertSyncPacket`. No function that writes them is hooked in this tree
  (`OriginalEntry` names in use: AppMemoryInit AscBegin AscDispatch AscDispatchFenced AsyncMap DeviceBegin DispatchIndirect Dispatch_02479810 Dispatch_0247e340 DrawDecalPass DrawForwardPass DrawGBufferPrimitives
  DrawIndexedCommon DrawIndexedIndirect DrawZPass FenceEop InstanceLoop Map PrepareDraw ResTransition ResourceBufferCreate SkinWrites StagingWorker WetSim; the async copy `FUN_024717c0` is served through the
  async dispatch hooks, not a packet hook).
* **The shadow harness (`native_sync.{h,cpp}`, namespace `Typed`).** A producer calls `Typed::Expect(packet_address, record)` with the record it believes it wrote (`Fence`, `LabelWait`, `LabelWrite`, `DmaCopy`);
  `ConvertSyncPacket` calls `Typed::Met(packet_address, kind, packet words)` right before it turns the packet into a tag; `Met` decodes the words with the existing decoder and compares field by field with the expectation.
  Counters per kind (`eop`, `release`, `wait`, `write`, `dma`): `evaluated`, `differing`, `differing_fields`, `without_typed_source`; plus `expected_total`, `expected_overwritten_before_met`, `expected_never_met`
  and the first differences with both values.
* **The one producer wired: `FenceEop`.** The hook builds `Fence{address = the label the function returns, value = 0x10, source = Immediate32, interrupt = false}` (from the documented behaviour in native_sync.h: it
  stores 0x11 in the label, the packet stores 0x10 there when the GPU has finished all earlier work) and expects it at the packet's address. Expectation: `eop.evaluated` > 0 and `eop.differing` 0.
* **What the counters will tell you:** `without_typed_source` per kind is the number of packets that still have no producer, i.e. the size of the remaining work per kind; `eop.without_typed_source` counts the
  end-of-pipe writes `FenceEop` did not write (for example the frame fence in `JsSwapChain_SubmitAndFlip`, 0x1c4dfa0).
* **For the local session, per remaining writer** (Ghidra): which packet it writes (opcode, 6 / 7 dwords), which arguments carry address, value or data, whether it writes through the command buffer cursor at `ctx + 0x47d0`
  (graphics, as `FenceEop`) or another cursor (the async queue); then a hook entry and a hook function after the pattern of `TagFenceSignal`: run the original, read the cursor before and after, build the typed record
  from the arguments, `Typed::Expect(cursor_before, record)`. The writers named in the plan (section 9): `FUN_024717c0` (async copy), `FUN_024a4ab0`, `FUN_01c6e670`, `FUN_024a5e20`, `FUN_01c6e320`, `FUN_024a53e0`
  (CP DMA, 17 per run in the default scene) and the label write / wait wrappers.
  Gate: 0 differing per kind over a 25 s regression run and a longer run. No deletion.

## 14. Task 2: the register-only target fields (TLG_TARGET_REGONLY=1)

No new Ghidra reading was possible (no binary, no decompiler here). The candidates come from the structures and the earlier Ghidra answers (section 6); each is compared with the register value and counted, never applied.
Where the objects cannot decide it, the report says so by the `compare_only_differs` count and the value histogram. A field is a constant of the game only when its histogram has one value; I cannot say that without a run, so no
constant is named yet.

| field | candidate from the engine's objects | evidence | counters |
|---|---|---|---|
| colour `info.linear_general` (bit 7) | the resource's T# tiling index is 31 (`DisplayLinearGeneral`) | `GetTileMode()` returns that mode exactly when the bit is set (regs_color.h); the three info words quoted from the local runs all have it 0 | `colour_fields.info_linear_general.compare_only(_differs)`, histogram `<path>/colour/info_linear_general` |
| colour `info.compression` (bit 14) | the resource's prebuilt block has an fmask base (resource `+0xb0 + 9 * 4` != 0, mip 0) | the only reader is the fmask address in `ImageInfo.meta_info` (image_info.cpp:80); fmask exists for multisample targets; Ghidra Q9: `FUN_02491a30` never writes the fmask words | `info_compression` |
| colour `info.alt_tile_mode` (bit 31) | the resource's T# `alt_tile_mode` bit (Neo mode) | `AmdGpu::Image::alt_tile_mode` is the Neo bit of the T#; `ImageInfo.alt_tile` reads the register bit only on Neo | `info_alt_tile_mode` |
| colour fmask words (9, 10) | the prebuilt block's words (`+0xd4`, `+0xd8`) at mip 0, 0 above | Ghidra Q9: only the prebuilt block writes them; the view's block is a copy of it | `fmask_base`, `fmask_slice` |
| depth stencil bases without a stencil (words 3, 5) | resource `+0x40 >> 8` | Ghidra Q8: both stencil bases are resource `+0x40 >> 8`; the question is what the registers hold when the stencil format is invalid | `stencil_read_base`, `stencil_write_base` (compare-only when no stencil) |
| depth view read-only bits (24, 25) | the DSV description's flags at `+0x48` (bit 0 depth, bit 1 stencil: D3D11_DSV_READ_ONLY_*) | no reader of the bits in the port (grep); the flags are in the D3D11 description | `view_read_only` |
| depth and stencil clear values (words 10, 11) | (a) resource `+0x158` / `+0x15c` (compare-only since round 6); (b) the arguments of the `ClearDepthStencilView` call: depth in `ymm0`, stencil in `rcx & 0xff`, flags in `rdx` (the existing hook, native_device.cpp `ClearDsvEnter`), against the values the clear's setup wrote into the clear registers | the registers of a depth clear record come from the clear function's setup packets (`TakeClearPrelude`: registers 0xa stencil, 0xb depth), i.e. from the call's arguments, not from an object; (a) differed in a few cases per long play | (a) `depth_clear`, `stencil_clear` compare-only; (b) `depth_clear_call`: `evaluated`, `depth_equal / differs / register_absent`, `stencil_equal / differs / register_absent`, histograms of the call's and the register's values and of the flags |

Expected: `compare_only_differs` 0 for the fields whose candidate is right; where it is not 0 the histogram and the examples say what the register holds instead. Fields where I expect the objects NOT to decide: the clear
values on the fill path (they are the clear call's, a call argument: (b) tests exactly that), and `alt_tile_mode` (Neo only; may be a constant 0 on this console model: the histogram will say).

## 15. Task 3: what the register fallback reads (TLG_PLAN_SHADOW=1)

**The fallback.** `PlanExecDraw` plans a draw from its record only when the render target inputs and the pipeline of its keys are cached (`exec_inputs`, `exec_pipelines`). The first sight of a key (`exec_reason` 4, about 480
per 25 s; a few of 6 and 7) goes through `PlanDraw(exec_state, ...)` in the register image: `MakeDrawInputs(regs)`, `pipeline_cache.PlanGraphics(regs, ...)`, which runs `GetGraphicsPipeline` (the stage lookup by
`ProgramForStage`), `RefreshGraphicsKey`, `RefreshGraphicsStages` and `BuildRuntimeInfo`. `ClassifyDraw(regs)` runs only for plans that are not `executed` (the packet path, verify). `MakeFixedState(regs)` runs only where the
record has no native fixed state. The register image `exec_state` is written from the record (`Synthesize`'s output) by `ExecApplyRecord`; so for a field the record carries, the register value is the engine's word as the
record builder read it, and the "native value" below is the typed value the record builder or the shader object gives for the same thing. The counters compare those two.

The members of `DrawRegs` read by these functions (found by listing `regs.<member>` in vk_pipeline_cache.cpp, vk_fixed_state.cpp and vk_rasterizer.cpp, then reading each use) and where the native side has the value:

| DrawRegs member | read by | native value | shadow counter group/field |
|---|---|---|---|
| `color_buffers[8]` | `MakeDrawInputs`; `RefreshGraphicsKey` (data / number format, conversion, swizzle, samples, blend_bypass); `PrepareRenderState`, `BeginRendering` through `DrawInputs` | the target description (default, played), applied to `DrawInputs` only: the pipeline key reads `regs.color_buffers` directly | `feeds_the_draw` (applied differs) and `pipeline_key_targets/color.*` (the key's fields, register vs native-applied inputs) |
| `depth_buffer` | `MakeDrawInputs`; `RefreshGraphicsKey` (z / stencil format, samples, validity) | target description (same remark) | `pipeline_key_targets/depth.*`, `stencil.*` |
| `depth_view`, `depth_htile_data_base` | `MakeDrawInputs` | target description | `feeds_the_draw` |
| `depth_clear`, `stencil_clear` | `MakeDrawInputs` | none that decides: registers (section 14) | `feeds_the_draw` compare-only |
| `depth_control` | `MakeDrawInputs` (`depth_enable`, `stencil_enable`, `depth_write_enable` gate the depth target and the depth image view) | `FixedState.depth_test`, `depth_write`, `stencil_test` | `inputs_vs_native_fixed/depth_control.*` |
| `depth_render_control` | `MakeDrawInputs` (clear flags) | `FixedState.depth_clear`, `stencil_clear` (the engine's own clear operations also arrive as records) | `inputs_vs_native_fixed/depth_render_control.*` |
| `num_indices`, `num_instances`, `index_base_address`, `index_buffer_type`, `max_index_size` | `MakeDrawInputs` | `DrawArgs` of the record (E1b, typed) | `draw_args/*` (only where the record sets the argument; `num_indices` only for non-inline draws) |
| `index_offset` | `MakeDrawInputs` (`GetDrawOffsets`) | none (VGT_INDX_OFFSET; the base vertex also arrives as user data) | `inputs/index_offset` (register only) |
| `vgt_instance_step_rate_0/1` | `MakeDrawInputs`, `BuildRuntimeInfo` | `ShaderEnv.rates` | `shader_env.vertex/step_rate_0, step_rate_1` |
| `ta_bc_base` | `MakeDrawInputs` (border colour table of the samplers) | none | `inputs/ta_bc_base` (register only) |
| `clip_user_data` | `MakeDrawInputs` | none (0 in the native path) | `inputs/clip_user_data` (register only) |
| `vs_program`, `ps_program` | `GetGraphicsPipeline` / `RefreshGraphicsStages` (`ProgramForStage`, `GetParams`: address, user data, flat table), `BuildRuntimeInfo` (`BuildCommon`: float modes, vgprs, user data count, vgpr components, dx10 clamp) | `ShaderEnv.program`, `user_data` (shader objects; typed user data of the record) | `program/vs_address, ps_address`, `user_data/*`, `shader_env.vertex/*`, `shader_env.pixel/*` |
| `vs_output_control` | `BuildRuntimeInfo` (`MapOutputs`), clip distance emulation | `ShaderEnv.exports.vs_out_control` (decides for the outputs; the clip distance emulation input still reads the register) | `shader_env.vertex/num_outputs, outputs`; `translator_input.fragment/clip_distance_emulation` (register only) |
| `ps_input_ena`, `ps_input_addr`, `num_interp`, `ps_inputs[32]`, `barycentric_control`, `depth_shader_control`, `z_export_format` | `BuildRuntimeInfo` (fragment) | `ShaderEnv.fragment` | `shader_env.pixel/*` |
| `color_export_format` | `RefreshGraphicsKey` | `ShaderEnv.fragment.color_export_format` | `shader_env.pixel/color_export_format` |
| `aa_config` | `BuildRuntimeInfo` (`num_samples` when the shader reads sample coverage) | none (the render pass sample count is in the target description) | `translator_input.fragment/aa_config.NumSamples` (register only) |
| `stage_enable` (`raw`, `IsStageEnabled`, `hs_en`) | `GetGraphicsPipeline` (exec cache entry), `RefreshGraphicsKey`, `RefreshGraphicsStages`, `BuildRuntimeInfo` | `ShaderEnv.stages` exists in the contract; the native draws are vertex + pixel only | `stages/stage_enable.raw` (register only: nonzero = a tessellation / geometry draw) |
| `ls_program`, `hs_program`, `es_program`, `gs_program`, `ls_hs_config`, `tess_config`, `vgt_gs_mode`, `vgt_gs_instance_cnt`, `vgt_gs_max_vert_out`, `vgt_gs_out_prim_type`, `vgt_gs_vert_itemsize`, `vgt_esgs_ring_itemsize`, `vgt_strmout_config` | `BuildRuntimeInfo` (local, hull, export, geometry), `RefreshGraphicsStages` | `ShaderEnv.tessellation`, `geometry` exist in the contract; the default scene does not use these stages | `stages/ls_hs_config, tess_config, vgt_gs_mode.onchip, vgt_strmout_config.raw` and `translator_input.stage_built/*` (register only: how often a draw builds them) |
| `clipper_control`, `color_control`, `polygon_control`, `viewport_control`, `viewports[]`, `viewport_scissors[]`, `screen_scissor`, `generic_scissor`, `window_*`, `stencil_*`, `blend_*`, `poly_offset`, `line_control`, `mode_control`, `primitive_*`, `color_target_mask`, `color_shader_mask`, `depth_bounds_*`, `depth_render_override` | `MakeFixedState(regs)` (only where the record has no native fixed state), `ClassifyDraw` (not executed draws) | the native fixed state (`Fixed::Build`, played), every `FixedState` field | `fixed_state/<field>` (every field of `FixedState`) |

Each group compares at the point both values exist: the fixed state, draw arguments and user data in `native_liverpool.cpp` right after the fallback plan; the key's target fields at the end of `PlanDraw`; the translator input
inside `BuildRuntimeInfo` (the register-built input before and after `ApplyShaderEnv`); the program addresses in `GetGraphicsPipeline` before the key is built. A field with `differs` 0 over a 25 s run and over a longer one is
safe to drop first; a `register_only` field with `nonzero` 0 is a candidate to drop (or to turn into a constant) as well.

## 16. Task 4: who still needs the address keyed image code (TLG_E3_CENSUS=1)

`native_e3_census.{h,cpp}`. Counted per named caller: `RefreshImage` (callers: Presenter::PrepareFrame, Rasterizer::BindTexturesFromSet, Rasterizer::BeginRenderingFull, Rasterizer::TryCopyImages, Rasterizer::NativeClear,
TextureCache::FindTexture / FindRenderTarget / FindDepthTarget through `UpdateImage`; and inside the cache FindImage (the rest of a texture before its engine clear), ExpandImage, ResolveDepthOverlap), `ExpandImage`
(ResolveOverlap equal address: block view of an uncompressed image, 3D view of equal size, grow), `ResolveDepthOverlap`, `ResolveOverlap` (FindImage) and `exact_match` (FindImage: the engine texture's list, the remembered
answer). Each with `count` and `per_25s` (scaled by the observed seconds). `refresh_uploads`: for each caller the uploads from guest memory, hashed (XXH3 of the source bytes) and compared with the previous upload of the same
image: `first_upload_of_the_image`, `bytes_differ_from_last_upload`, `bytes_same_as_last_upload` (an upload that carried nothing new), `source_gpu_written_not_hashed`, megabytes hashed and megabytes in the same-bytes uploads.
Read it as: a caller with `per_25s` 0 over the scenes you play does not need the function; a caller whose uploads are mostly `bytes_same_as_last_upload` is refreshing without a cause.

## 17. E7c (branch cloud-e7c, from native-v2 325c324): Task 1, the register-only target fields without registers (`TLG_TARGET_REGFREE=1`, off by default)

Code: `native_target_desc.{h,cpp}` (`BuildColour` / `BuildDepth` / `Merge` / `Report`), `native_config.h` (the constants), `native_draw.cpp` (`ConvertClearQuad` hands the depth clear call to `PrepareDepthClear`).
With the switch off nothing changes: `PlanDraw` and the fill point read the registers as before. `TLG_TARGET_REGFREE_VERIFY=1` builds the same values, compares them with the register decode at every fill point,
depth clear and eliminate, counts the difference per field (`ignored_changed` / `differences`), and keeps the registers (so verify never changes what reaches the GPU).

**What is applied, and the guard that goes with it**

| field | applied | guard (a trip is counted in `constant_guard_trips.<field>` with one example, and the registers decide that draw) |
|---|---|---|
| `info_alt_tile_mode` | `TargetAltTileMode` = 0 | the T# of the engine object says alt tile mode, or the engine's own colour block (`+0x54`) has bit 31 set |
| `info_linear_general` | `TargetLinearGeneral` = 0 | the T# tiling index is 31 (linear general), or the block has bit 7 set |
| `info_compression` | `TargetCompression` = 0 | the target has more than one sample, or the block has bit 14 set |
| depth `view_read_only` (DB_DEPTH_VIEW bits 24 and 25) | `DepthViewReadOnly` = 0 | the engine's depth view block has those bits set |
| fmask words (`fmask_base`, `fmask_slice`, `pitch_fmask`, attrib fmask bits 5..11) | 0 when `num_samples == 1` | more than one sample: the registers are kept and counted in `fmask_from_registers` |
| depth / stencil clear values of a depth clear record | the arguments of the engine's `ClearDepthStencilView` call, carried in the `TargetSet` | no call (the fill point's clear flag, a draw that clears through the flag): the registers are kept and counted in `clear_value_from_registers.<path>` |

**Why my old candidate rules gave 1.** `compression`: I derived it from "the target has an fmask address". The engine fills `fmask_base` for *every* target, also one-sample targets, so the candidate was 1 for all of them
while the hardware bit (`CB_COLOR_INFO.COMPRESSION`, bit 14) is 0 unless the target is multisample and uses fmask. The only reader of the fmask address on the host side is `image_info.cpp:80`
(`meta_info.fmask_addr = buffer.info.compression ? buffer.FmaskAddress() : 0`), so with compression 0 the fmask words are not read at all for a one-sample target; the only other fmask readers are the shader's own
`fmasks` list (`vk_pipeline_cache.cpp:715`, from a sampled T#, not from the target) and the `FmaskDecompress` draw class (decided from `color_control.mode`). `view_read_only`: I looked at the engine's DSV flags (`+0x48`).
Those flags reach the hardware through the depth control / stencil write mask that `Synthesize` builds (the "read only" views disable the depth write and the stencil write), not through DB_DEPTH_VIEW bits 24 / 25, which
the engine leaves 0. So the DSV flag is a true statement about the draw but the wrong register. The candidate rules in `TLG_TARGET_REGONLY=1` are corrected (compression candidate: `samples > 1`; read-only candidate: the constant);
the constants are the values your register histograms showed (one value in every draw); `TLG_TARGET_REGFREE_VERIFY=1` is the check that it holds.

**Depth / stencil clear values.** The clear function's setup writes the value into the clear registers through the register prelude; the same value is an argument of the call (the counters `depth_clear_call` say how often they agree). `ConvertClearQuad` reads it from the
call (`CurrentDepthClear`) and `NoteDepthClearCall(call_depth_bits, call_stencil, flags, reg_depth_bits, reg_stencil)` counts under `depth_clear_call` how often call and prelude agree. The fill point of a draw with the clear flag
set has no ClearDepthStencilView call record of its own: it stays register sourced, `clear_value_from_registers.fill_point_draws_with_the_clear_flag`
says how many draws that is. If that counter is 0 in the scenes you play, the registers are not needed for the clear values at all.

Counters: /stats `native_backend.device.e2_targets.feeds_the_draw.regfree` = `{on, verify_only, constants_applied, clear_value_from_call, clear_value_from_registers{fill_point, depth_clear, eliminate, fill_point_draws_with_the_clear_flag},
fmask_from_registers, fmask_zeroed_one_sample_targets, constant_guard_trips{alt_tile_mode, linear_general, compression, view_read_only}}` (and the same in `last_counters.json`, and `user/log/target_desc_counters.json` every 30 s).
Expected in the default scene: every difference 0, `constant_guard_trips` all 0, `fmask_from_registers` 0 (no multisample colour target is bound; if the game uses one, the count says how many draws still need the registers).

## 18. Task 2: why `shader_env.pixel/inputs` differs (329 of 1179) (`TLG_PLAN_SHADOW=1`)

Reading `HwFragmentRuntimeInfo::PsInput` (`runtime_info.h`): `param_index`, `is_default`, `is_flat`, `default_value` and unnamed bits 9..15 of its 16-bit bitfield. The native build leaves the padding 0; the register path
is `info.hw.fs.inputs[i] = {.param_index = ...}` after the whole array was filled from the register image, and in the cases seen the register side holds 0xfe00 in those bits (native 0x0). `operator==` of the struct compares the whole `u16` and so reports a difference where the named fields are equal.
`HwFragmentRuntimeInfo::operator==` compares `inputs` only up to `num_inputs` (the shader's own count), and the translator indexes inputs by the attribute of the instruction (`vector_interpolation.cpp`), so it
never reads past `num_inputs` and never the padding.

Shadow: for each pixel input slot `Compare` is now split into `inputs.named_bits` (bits 0..8), `inputs.unnamed_bits` (0xfe00) and `inputs.operator==` (the existing whole-word compare). `PS::Dump("shader_env.pixel.inputs", ...)` writes the first
16 differing cases (pixel shader gnm address and hash, slot, the native decode, the register decode, the raw `SPI_PS_INPUT_CNTL_n` word from `regs.ps_inputs[i]`, the env interpolant fields, the vertex shader outputs). Read:
`named_bits` differences 0 and `unnamed_bits` differences = the 329 means the difference is the padding of the struct, not a value: the native side is right (a zero padding is what the struct means), and it is not a stale register mirror
(the register words were current: the named bits agree). If `named_bits` is ever non-zero the dump shows the case and the vertex outputs, and the call is the user's: with the dump a value in the register that the engine did not set would
be a stale mirror.

## 19. Task 3: RefreshImage with announced content (`TLG_E3_ANNOUNCED=1`) and the exits of RefreshImage (`TLG_E3_CENSUS=1`)

Code: `native_e3_census.{h,cpp}`, hooks in `NativeUpdates::ApplyBatch`, `TextureCache::RefreshImage`, `BufferCache::ObtainBuffer` (where the GPU written range is added) and the three rasterizer call sites that ask for a written buffer.

**What an announcement is.** A range of guest memory (address, size), nothing more: the engine says "this memory is written"; the bytes are not stored. So "the announced content" can only be the content of that memory when the announcement
is applied. `NoteAnnounced` (raster thread, `ApplyBatch`) hashes every 4 KB page of a first-time announced range (ranges over 2 MB are marked announced without a hash) and remembers it. At the upload the pages of the image's memory are
hashed again and compared. Both sides are raw guest bytes before any detile (the detile is a function of the bytes), so no detile is needed to compare. Classes per upload and caller:
`announced_equal` (every announced page still has the bytes it had when announced; at least one page announced), `announced_differs` (a page changed after its announcement: a write the announcement did not cover or a write
that came after it, the game thread runs ahead of the raster thread), `no_announcement` (no page of the image was ever announced). Details: `every_page_announced_and_hashed`, `some_page_not_announced`, page totals, up to 8 examples
per group and class (caller, address, size, width, height, host format, pages, `first_differing_offset`, the engine texture that owns the memory, `announcements_ago`).
Groups: `first_upload_of_the_image`, `source_gpu_written_not_hashed`, `repeat_upload`.

**The 160 GPU sourced uploads.** `BufferCache::ObtainBufferForImage` takes the GPU buffer when `IsRegionGpuModified` (bytes a shader storage write, a GPU fill or a GPU copy wrote and nothing superseded since). The only place that adds
to that set is `ObtainBuffer(..., is_written = true)`. `WriteSiteScope` names the call site around the call (shader_graphics, shader_compute, fill_buffer, copy_buffer, hle_copy); `NoteGpuWritten` keeps the last 8192 ranges with the site, and the upload
looks up the last range that overlaps its memory: `gpu_source_uploads_by_last_writer`. Render targets are not in that set (they are images, written by draws); `gpu_source_uploads_image` counts the images with the GpuModified flag, a colour attachment,
a depth attachment or a storage usage, and whether an engine texture owns the memory. For these uploads the class only says whether the guest pages were announced and are unchanged since: the bytes uploaded are the GPU's, not the guest's.

**The exits of RefreshImage** (`TLG_E3_CENSUS=1`, `refresh_outcomes.by_reason`, per caller, with `per_25s`): `not_dirty` (no Dirty mark: the call returns at the first test), `multisample` (dirty, more than one sample, never uploaded),
`maybe_cpu_dirty_same_hash` (the 8x8 pixel hash is unchanged), `no_copies`, `uploaded`. Their sum is `functions.RefreshImage.total`. Nothing in RefreshImage was changed or removed; the B8 verify and the ring check stay as they were.

**What this can not decide.** Whether the engine's announcements cover everything the uploads read is answered for the pages hashed at announcement; a page announced by a range over 2 MB is counted as announced but is not compared
(`pages_not_hashed_range_over_2mb`). A write after the announcement and before the upload looks the same as a write the announcement did not cover (`announced_differs`): `announcements_ago` and the example's engine texture show which
it is; the answer needs the hook of the writer if the examples name one texture again and again.

## 20. The profiler (Shift+F9, `/prof`): one tool, probes in the code

`native_prof.{h,cpp}` (it replaced the raster-only first version). F8 says which thread limits the frame, F9 times the game's jobs and the stage threads as a whole; this looks inside any host thread, and what it measures is chosen by probes in the code and by the capture's settings, not by the report.

**Add a probe** (one line, anywhere, any thread; one relaxed load while no profile runs):

```cpp
TLG_PROF("raster.bind_buffers");              // times the rest of the block (level 2)
TLG_PROF_LEVEL("texture.find_view", 3);       // 1 coarse (an operation), 2 normal, 3 fine (called thousands of times a frame)
TLG_PROF_NAMED(scope, "raster.begin");        // a scope object that scope.End() can end early
TLG_PROF_COUNT("buffer.bind_memo.hit");       // counts an event
TLG_PROF_ADD("buffer.bytes_streamed", size);  // counts an event with a value (count, sum, max in the report)
TLG_PROF_TAG("caller", value, label);         // names the operation being timed (the outermost probe open on the thread); TLG_PROF_TAG_HEX shows it in hex
```

The name before the first `.` is the group. Nothing else needs a change: the report lists whatever ran.

**Run it.** Shift+F9 (stops again with Shift+F9 or F9): 10 s, level 2, every group, the raster thread, the report opens in the browser. `TLG_PROF_SECONDS`, `TLG_PROF_LEVEL`, `TLG_PROF_GROUPS`, `TLG_PROF_THREADS` change that default. From the debug server: `/prof?seconds=10&level=3&groups=raster,buffer&threads=GpuRaster,GpuCommandProcessor&open=1` (`threads=all` records every thread that runs a probe), `/prof_sites` lists every probe that has run with its group and level, `/prof_baseline?set=1` keeps the last report and every later report shows the change against it per probe (`?clear=1` forgets it). So an optimisation is measured by running the same capture before and after.

**The report** (`user/log/prof_<time>.html`, `.txt`, `.json`, and `prof_latest.*`; the charts are in the html only): at the top the frame time per frame (with the 120 / 60 / 30 fps lines), the CPU use of the busiest threads and the GPU busy, in 100 ms steps, as in the F9 report; then per selected thread a chart of its busy time by outermost probe over the capture, and: busy ms per frame (with the measured probe cost taken off, and as measured); the probes ranked by their own time (time without the probes inside), calls per frame, mean and max; the tree of how they nest; the operations ranked by every tag name used (the function that issued a draw or dispatch, the pass class, the source line of a closure); the counters (events per frame, sum, max); the 12 slowest single operations with the probes that took their time; and the change against the baseline. The probe cost is measured when the capture starts (a warm loop of empty scopes) and subtracted per probe as half a scope for the probe itself and for each probe it contains; it is a lower bound (in the game the caches are colder; the first version, which assumed 20 to 40 ns, was low by a factor of three: 66,000 scopes a frame cost about 7 ms).

**How it works.** A probe is a function-local static `Site` (registers once) and a `Scope`. While a capture runs, the capture's level and group choice enables the matching sites (`Site::enabled`), and a scope on a thread the capture selected reads the timestamp counter at entry and exit and adds the time to that thread's tables (by probe, and by the path of nesting); no event is written per scope. Counters and tags are written to the same per-thread tables. Frames are counted from the capture's `SubmitFlip` events (the same machinery as F8), so the capture also records the usual timeline events.

**What is probed now.** group `raster` (operations, binding, render targets, barriers, descriptor push, the record calls), `buffer` (ObtainBuffer and counters for each path it takes), `texture` (image lookup, refresh and its upload steps, views, samplers), `sched` (the wait for the encoder). Tags: `caller`, `pass_class` on draws and dispatches, `closure_site` (file:line of the `RasterRun` call) on closures.

**Limits.** Host threads only (the game's own threads are timed by the F9 guest spans); a thread created after the capture started is not selected by name; the draw and dispatch calls into the driver are inside the `record` probes and not split further.

## 21. Raster thread work reduction, switches A and B (2026-11-01, not played)

`native/native_diag.h`, both read once from the environment.

- **A, `TLG_COALESCE_ANNOUNCE=1` (default off):** `NativeUpdates::ApplyBatch` merges the ranges of each list (`first`, the replay `again`) where they overlap or touch before it applies them, and applies each merged range once. A gap between two ranges is never bridged, so no byte is applied that was not announced; zero size ranges are dropped; the two lists are not mixed (`newer` differs). The E3 announced census still sees the ranges as announced. Counters: `/stats` b2 report `coalesce` {on, ranges_in, ranges_applied} and the profile counter `updates.ranges_after_merge` next to `updates.ranges_first / again`. The B9 / B8 classification counters (`AnnouncedOnBuffer`, `AnnouncedClipped`, ...) count merged ranges when it is on, so they change; the dirty sets and epochs the caches read are the same. How much it merges is not known until a run: read `ranges_in` against `ranges_applied`, then `updates.apply_first / apply_again` in the profile.
- **B, `TLG_DIAG_COUNTERS=0` (default 1, no change):** the always-on pure counters of the bind path do nothing: `Rasterizer::Census`, `NativeRegistry::NoteBinder / NoteBinderReason`, `NativeResources::NoteResident / NoteHandle / NoteImmutableHit / NoteImmutableSync`, the bind memo's hit and miss counts. The numbers they feed in `/stats` (`consumer_census`, `native_binder`, resident and handle counters, `bind_memo`) stand still. Nothing that changes behaviour is behind it (`NoteGpuWritten`, `NoteCpuWrite` stay).

To measure: capture the same scene with the switches off and on (`/prof_baseline?set=1` after the first run) and read the change of `updates.apply_*`, `raster.bind_buffers`, `raster.bind_textures`, `buffer.obtain`.

## 22. Item C: the bind memo for texel buffers and bigger requests (TLG_BIND_MEMO_WIDE, 2026-11-01, not played)

The bind memo (`BufferCache::ObtainBuffer`) answered only read-only requests of at most 64 KB. In the profile of the default scene about 4 in 10 requests skipped it: about 1,300 written, 1,760 texel and 970 over 64 KB per frame. The memo's rules for staleness are unchanged: an answer is dropped by a write over its range (`BindMemoErase`, from announced writes and from written requests), a bump of the memo epoch (GPU written bytes change hands, the command processor's writes), the range epoch, the placement epoch, and a wrapped stream buffer; all answers that name the stream buffer are dropped at a new command buffer.

`TLG_BIND_MEMO_WIDE` (read once):
- **0 (default):** as before.
- **1:** read-only texel buffers and read-only requests from 64 KB up to 16 MB are also memoized, but the uncached answer is used and the memo's answer is compared with it (counter `bind_memo.wide`: `compared`, `compared_and_different`, `compared_stream_answers`; a difference is logged, the first 8). A stream copy is a new copy each time, so for those only the buffer (the stream buffer) is compared.
- **2:** the memo's answer is used for that class (`answers_used`).

What had to change for big entries: `BindMemoErase` looked back at most 64 KB from a write for an entry that covers it; it now looks back by the largest size ever stored (`BindMemo::max_size`). Entries carry a `texel` flag, so a texel request and another request at the same address and size do not share an answer.

Written requests are not in it: each one must mark its bytes GPU written, which is state, not a lookup.

To use: run mode 1 first in the default scene and a cutscene and read `compared_and_different` (it must stay 0, with `compared` in the thousands), then mode 2. The profile shows the effect in `buffer.obtain` own time and the counter `buffer.bind_memo.wide_hit`.

## 23. TLG_ABLATE: a bisection tool (the picture is wrong when it is on)

`TLG_ABLATE=<comma list>`, read once at start, default empty (`native/native_diag.h`). It leaves work out on the raster thread to see what the frame rate does without it:

| item | left out | where |
|---|---|---|
| `draws` | draw and indirect draw calls (binds, descriptors, the draw) | `liverpool.cpp`, before `ExecRasterCall` |
| `dispatches` | compute calls | same |
| `announce` | applying announced ranges (first and again); deletions still run | `NativeUpdates::ApplyBatch` |
| `again` | only the re-apply of the previous batch | `NativeUpdates::ApplyBatch` |
| `sync` | `BufferCache::SynchronizeMemory` uploads | `buffer_cache.cpp` |

Closures always run (EOP signals and write fences). A critical log line says when the switch is on. Run order: `draws,dispatches` first (the ceiling of everything upstream), then `announce`, then `sync`, each against a baseline of the same scene.
