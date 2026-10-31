# E7 notes (cloud session, Task 1: read-only analysis)

Base: `native-v2` at `8667426`. Nothing was built or run; every statement below comes from reading the code (file:line at that commit) or from the numbers in the task
description. Where I could not tell, it says so and it is a question in the last section.

## 0. What changes the plan (read this first)

1. **A typed draw does not read the register image for its targets at draw time.** `PlanExecDraw` (native/native_rasterizer.cpp:335) copies `DrawInputs` from the
   cache `exec_inputs[inputs_key]` (line ~380). That cache is filled by the slow path: `Rasterizer::PlanDraw` runs `plan->inputs = MakeDrawInputs(regs)` and
   `exec_inputs.insert_or_assign(inputs_key, plan->inputs)` (vk_rasterizer.cpp, end of `PlanDraw`). `inputs_key` is a hash of the target blocks by content
   (`view + 0x54`, 16 words per colour target; `dsv + 0x58`, 13 words, plus the depth resource's clear values; native_draw.cpp:2800-2835). So the register image feeds
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
