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
| `clear_word0/1` | C | `LiverpoolToVK::ColorBufferClearValue` (liverpool_to_vk.cpp:830) -> pending fast clear colour, applied at the eliminate (vk_rasterizer.cpp:704) and at the render pass (BeginRenderingFull, vk_rasterizer.cpp:3125) | **none known.** vk_rasterizer.cpp's own comment (B11) says the words in the block differ from the colour the engine passed to `ClearRenderTargetView` in 470 of 16,656 clears (registers 0,0,0,0 vs call 1,1,1,1) and the registers are the correct one. So these two dwords have no other source yet |
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
* **No engine source known:** colour `clear_word0/1` (see C row), `linear_general` for targets, `blend_bypass`, `alt_tile_mode`, the fmask words, the write bases.
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
