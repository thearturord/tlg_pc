# F4 census: who reads `AmdGpu::Image` (T#), `AmdGpu::Buffer` (V#), `AmdGpu::Sampler` (S#) and the format enums

Cloud round, 2026-10-31, branch `cloud-e7c` (from native-v2 325c324). **Static reading only**: nothing here was run, so "reads" means "the code reads it", not "the default scene reaches it".
The runtime question (which of these paths the default scene and a cutscene reach) is answered by counters that already exist, listed in section 8.
Plan reference: phase F4 of "Plan: finish the native renderer, no PS4 GPU emulation left (2026-10-31)", and the F4 rows of F5 (`check_native_deps.py`).

## 1. Result in ten lines

1. The three types are used in **three layers**, and only one of them is the F4 target.
   (A) the shader translator and its specialization (`shader_recompiler/`): reads the descriptor words the shader itself names (`SharpFetch<T>`) and decodes them. This is the shader ABI. **Stays by design.**
   (B) the host consumers of the decoded type: `ImageInfo` / `ImageViewInfo` / `Sampler` constructors, the texture binding code of the rasterizer, vertex input and buffer binding, pipeline key, `draw_prep`, the HLE shadows. **This is F4.**
   (C) `native/` itself: it builds T# / S# words from the engine's objects (E2) and the host then decodes them again. **This round trip is what F4 removes.**
2. The host reads **few fields**. Image: about 15 of the 40 (format pair, type, size triple, pitch, level and layer range, tiling index, pow2pad, alt tile, min_lod, component select, address). Buffer: 8 of 14. Sampler: 14 of 30. The rest are only ever produced (by `native_image_desc.cpp`, to reproduce the engine's words) and compared, never consumed.
3. **`GetSharp` has 30 call sites outside the recompiler** (section 4): 9 are the legacy fallback or verify of a path that already has a record binding (`binder_plan`); 5 are address-only reads in dispatch audits and utility code; 8 are the HLE debug shadows; 3 are the indexed data path; 5 are vertex / pipeline creation. Only the vertex ones have no record binding to fall back on.
4. **Vertex inputs are the one big open piece.** The vertex V# come from a descriptor table in guest memory that the fetch shader's user data points to (`attrib.sgpr_base` / `dword_offset`). Five places read it. The table is written by the native layer itself for native draws (the "table arena window"), then decoded again by the host.
5. `pixel_format.h` is two things: the **GCN data/number format vocabulary** used by the translator (153+ uses in `liverpool_to_vk.cpp`, 100+ in the recompiler; by design) and a **host currency** between the engine's DXGI format and `vk::Format` (a two-table detour: DXGI -> (dfmt, nfmt) in `native_image_desc.cpp`, then (dfmt, nfmt) -> `vk::Format` in `liverpool_to_vk.cpp`). Only `native_draw.cpp` includes it in `native/`; the other two route through `resource.h` / `draw_state.h`.
6. `draw_state.h` defines only `DrawRegs`. It leaves with F3, not F4. The colour/depth types (`AmdGpu::ColorBuffer`, `DepthBuffer`, `DepthView`, `CbDbExtent`) are a fourth currency not named in the plan: `ImageInfo` / `ImageViewInfo` have constructors from them, and `native_target_desc.cpp` builds them. They are F3.1 / F4 together.
7. The shader ABI boundary can be drawn at the directory: **`shader_recompiler/` keeps `AmdGpu::Image/Buffer/Sampler` and `pixel_format.h`; nothing else does.** The gate "no `AmdGpu::Image/Buffer/Sampler` outside the shader ABI" then becomes a grep.
8. One cost of that boundary: `specialization.h` is part of the recompiler but is **filled by the host** from the sharps (`FillImage` etc.). Its inputs must come from the native description (a plain struct) instead of the sharps, or the host keeps decoding the words for that one purpose. Section 6 recommends the first.
9. `native_image_desc.cpp:Derive` is not fully register-free: it copies `width`, `height` and `last_level` from **the engine's own T# copy in the view object (`view+0x58`)** when it still names the same texture. Those three fields are the engine's words, not derived (the D3D description cannot say padded rows or a limited level count).
10. Everything in section 7 (what the cloud cannot decide) needs a local look; none of it blocks starting F4.1.

## 2. The three layers, with files

### (A) Shader ABI, stays by design

| file | what it does with the types |
|---|---|
| `shader_recompiler/resource.h` | `SharpFetch<T>` (which user-data dwords form the sharp, immediates, load mask), `BufferResource / ImageResource / SamplerResource` with `GetSharp`, `FromWords`, `PostOp`, `Finish`, `RawFromWords`, the post operations (`BitwiseOrDw1WithImm`, `OffsetByProgramBase`, `ClearStrideAndBitshiftDw2WithImm`, `ConvertCubeTo2DArray`, `DisableAnisoIfSingleLod`, ...) |
| `shader_recompiler/ir/passes/resource_patching_pass.cpp` | at **translation time** reads the live sharps from the shader's own user data to patch the IR: `buffer.stride`, `swizzle_enable`, `add_tid_enable`, `GetDataFmt`; image `GetType`, `DstSelect`, `GetNumberConversion`, `GetViewType`; sampler `force_unnormalized`, `force_degamma`, `max_aniso`, `clamp_x/y/z` |
| `shader_recompiler/specialization.h` | the specialization key: buffer `stride`, `swizzle_enable`, `data_format`, `num_format`, `dst_select`, `num_conversion`, `index_stride`, `element_size`; image `GetViewType`, `is_integer`, `dst_select`, `is_srgb`, `num_conversion`; fmask `width`, `height`; sampler `force_unnormalized`, `force_degamma` |
| `shader_recompiler/backend/spirv/spirv_emit_context.cpp` | `GetHostViewType`, `GetNumberClass`, buffer `mtype == 3` (coherent) |
| `shader_recompiler/frontend/translate/translate.cpp`, `lower_buffer_format_to_raw.cpp` | `DstSelect`, `GetNumberConversion`, `GetNumberFmt` |
| `shader_recompiler/frontend/fetch_shader.h` | `VertexAttribute::GetSharp`: the vertex V# read from the table the fetch shader's user data points to |

These are the translator's own inputs: the same words the console shader would load. No change proposed.

### (B) Host consumers of the decoded types (F4 target)

Image (T#), fields read:

| consumer | fields |
|---|---|
| `texture_cache/image_info.cpp:131` `ImageInfo(Image, ImageResource)` | `GetDataFmt`, `GetNumberFmt`, `GetTileMode` (tiling_index), `GetBaseType` (type), `IsTiled`, `pow2pad`, `width`, `height`, `depth`, `Pitch`, `NumLevels`, `NumLayers`, `NumSamples`, `Address`, `alt_tile_mode` + `GetBankSwizzle`, plus `IsBlockCoded(data_fmt)`, `NumBitsPerBlock(data_fmt)` |
| `texture_cache/image_view.cpp:53` `ImageViewInfo(Image, ImageResource)` | `GetDataFmt`, `GetNumberFmt`, `base_level`, `base_array`, `NumLayers`, `NumViewLevels`, `NumViewLayers`, `GetViewType`, `min_lod`, `DstSelect` |
| `vk_rasterizer.cpp` `BindTextures*` (2305-2760) | whole 32 bytes as a cache key (`memcmp`, `ImageDescHash`, the texture set key), `Address()`, `GetDataFmt`, `GetNumberFmt`, validity test (`Address() == 0`, format invalid, `IsValidGpuMapping`), and a 13-field copy into `NativeDevice::CheckImageSharp` (the E2 census) |
| `vk_draw_prep.{h,cpp}` | `ImageDescHash` (whole 32 bytes), `GetSharp` of every image of a stage |
| `vk_pipeline_cache.h:93` `PreparedImage` | stores the `tsharp` next to its `ImageInfo` / `ImageViewInfo` |

Not read by any host consumer (only produced by `native_image_desc.cpp` and compared): `mtype_l2`, `mtype`, `mtype2`, `atc`, `perf_modulation`, `interlaced`, `min_lod_warn`, `counter_bank_id`, `lod_hw_cnt_en`, `compression_en`, `alpha_is_on_msb`, `color_transform`. (`mtype` of the **buffer** is read by the translator, see (A).)

Buffer (V#), fields read by the host: `base_address` (about 40 reads in `vk_rasterizer.cpp` / `native_rasterizer.cpp`), `GetSize()` (= `num_records`, or `stride * num_records`), `GetStride`, `GetDataFmt`, `GetNumberFmt` (vertex attribute format, `vk_graphics_pipeline.cpp:449`, `vk_pipeline_cache.cpp:1159`, `vk_draw_prep.cpp:137`), `num_records` (`ShadowRaycast` capacity), `data_format` / `num_format` raw (E2 census). The vertex streams also copy the whole V# into a `guest_buffers` list for the range merge. Not read: `dst_sel_*`, `swizzle_enable`, `index_stride`, `element_size`, `add_tid_enable`, `cache_swizzle`, `mtype` by the host (the translator reads them).

Sampler (S#), fields read by the host: `Sampler::Sampler` (`texture_cache/sampler.cpp`) reads `xy_mag_filter`, `xy_min_filter`, `mip_filter`, `clamp_x/y/z`, `max_aniso` (`MaxAniso`), `depth_compare_func`, `border_color_type`, `border_color_ptr`, `LodBias`, `MinLod`, `MaxLod`; `BindSamplers` reads `border_color_type`, `max_aniso`, `filter_mode`, `mip_filter`, `Valid()` (validity and the log). `TextureCache::GetSampler` keys the cache by a hash of the 16 bytes. Not read by the host: `aniso_threshold`, `mc_coord_trunc`, `aniso_bias`, `trunc_coord`, `disable_cube_wrap`, `filter_mode` (only logged), `perf_mip`, `perf_z`, `lod_bias_sec`, `z_filter`, `mip_point_preclamp`, `disable_lsb_ceil`. The translator reads `force_unnormalized`, `force_degamma`.

### (C) `native/`

| file | role |
|---|---|
| `native_image_desc.{h,cpp}` | `Derive` builds an `AmdGpu::Image` from the engine's texture + view (`Decides()` default on, played); `DeriveSampler` builds the four S# words from the engine's sampler object; both then feed the old decode. `Compare` / `StudySampler` are the checks. |
| `native_target_desc.cpp` | reads the T# of a resource (`AmdGpu::Image t` at the engine block) to describe render targets; `static_assert(sizeof(AmdGpu::Image) == 32)`; builds `ColorBuffer` / `DepthBuffer` / `DepthView` words. |
| `native_rasterizer.cpp` (member file) | `PlanBufferSharp` (V# of a planned stage from the record bindings), `desc.FromWords` for buffers, the vertex table read for the geometry ring (`AmdGpu::Buffer` copied out of the descriptor table at `table + dword_offset`), `ReadableGuestMemory(table_entry, sizeof(AmdGpu::Buffer))`. |
| `native_draw.cpp` | includes `pixel_format.h`, `liverpool.h`, `liverpool_to_vk.h`, `vk_fixed_state.h`, `vk_indexed_data.h`, `vk_identity_audit.h`, `gnmdriver.h` (the rest of the allowlist). |
| `native_device.{h,cpp}` | `CheckImageSharp` / `CheckBufferSharp`: the E2 census that compares the engine object with the decoded sharp (the 13-field struct above). |

## 3. Per field: what replaces it

"Engine" = available from the engine's objects through `NativeDevice` today. "Table" = a function of engine fields already in `native_image_desc.cpp`. "Held" = copied from the engine's own T# copy (needs a look, section 7).

### Image

| field | host reader | source in a native description | status |
|---|---|---|---|
| address (`base_address`) | ImageInfo, binding validity, key | Engine (`res.address`) | done (E2) |
| `data_format` + `num_format` | ImageInfo, ImageViewInfo, binding | Table: DXGI -> `vk::Format` directly, plus bits per block / block flag; the (dfmt, nfmt) detour is only needed for the translator | table exists, needs the direct column |
| `dst_sel_x..w` | ImageViewInfo `DstSelect` | Table (per format) | done (E2) |
| `width`, `height` | ImageInfo | Engine, but padded rows (1088 for 1080) are **Held** | partly Held |
| `depth` | ImageInfo (volumes), layer count | Engine | done |
| `pitch` | ImageInfo | Table (alignment rules in `Derive`) | done, rules encoded |
| `base_level`, `last_level` | ImageViewInfo, NumLevels | Engine (view desc); `last_level` limit for block-compressed cubes is **Held** | partly Held |
| `base_array`, `last_array` | ImageViewInfo, NumLayers | Engine (view desc) | done |
| `type` | ImageInfo base type, view type | Engine (dimension) | done |
| `tiling_index` | ImageInfo tile mode | Table: 19 volume, 0 depth, 14 colour target, 13 other | done, a 4-value rule |
| `pow2pad` | ImageInfo | Table (`mips > 1`) | done |
| `alt_tile_mode` | ImageInfo alt tile (Neo mode) | constant 0 for textures (target version: `TargetAltTileMode`, E7c) | needs the same evidence rule as the target field |
| `min_lod` | ImageViewInfo | not produced by `Derive`; read as 0 in the default scene? | **unknown, section 7** |
| `NumSamples` | ImageInfo | `res.samples` (`+0x194`) | engine |
| the other 12 | none | none | drop |

### Buffer

| field | host reader | source | status |
|---|---|---|---|
| `base_address`, size (`num_records`, `stride`) | every buffer binding, range merge, vertex ranges | Engine buffer address + view offset + size | `CheckBufferSharp` compares; not yet deciding |
| `stride` | vertex binding stride, specialization | Engine (`StructureByteStride`, input layout stride) | open |
| `data_format`, `num_format` | vertex attribute format; formatted buffer reads | Engine (input layout DXGI format, or the typed buffer view's format) | open |
| the rest | translator only | the descriptor words in user data (by design) | stay |

Constant buffers: V# are frame allocations from the ring, already pseudo resources (E4); their size is the engine's constant buffer size.

### Sampler

All 14 host-read fields are a function of the engine's sampler object (`DeriveSampler` already builds the words from it, `TLG_DESC_CHECKS=1` compared them). Border colour: `border_color_ptr` indexes a **table in guest memory** (`BorderColorBuffer`, `current_bc_base`); with a custom border colour the sampler reads from guest memory at creation. `BindSamplers` rejects an S# whose type is `Custom` while `current_bc_base->Address() == 0`.

## 4. The 30 `GetSharp` / `FromWords` call sites outside the recompiler

| group | sites | what they are | disposition after F2 / F3 |
|---|---|---|---|
| legacy fallback of a record binding | `vk_rasterizer.cpp` 1954, 1973, 1989, 2038, 2513, 2530, 2970, 2983 and `native_rasterizer.cpp:173` (`PlanBufferSharp`) | read the guest descriptor when the record has no binding; `NoteBinder(6)` "must not happen" counts it | dead when `binder_plan` is always set: delete with the counter proving 0 |
| binder verify | `binder_verify` blocks around the same sites | compare the legacy read with the record | F6 |
| vertex / pipeline creation | `vk_graphics_pipeline.cpp:445, 500`, `vk_compute_pipeline.cpp:48`, `vk_pipeline_cache.cpp:1159`, `vk_draw_prep.cpp:137` | vertex attribute format and stride, buffer sharp at pipeline creation | needs the vertex description (below) |
| indexed data path | `vk_draw_prep.cpp:101, 105, 108` | the sharps of every image / sampler / buffer of a stage into thread-local lists, hashed | replace with the binding's description; the key then hashes the description |
| HLE debug shadows | `vk_shader_hle.cpp` 24-26, 115, 142, 162, 166, 187 | `ShadowRaycast`, `ShadowSkin`, `ShadowSkinWide`, the copy shader: read base/size of buffers | the CPU-port shadows are F6 candidates; the copy HLE (24-26) needs only address and size |
| rasterizer utility reads | `vk_rasterizer.cpp` 1639, 1672-1673, 1729-1730 | `desc.GetSharp(info).base_address`, two-buffer comparisons in a dispatch audit | address only |

## 5. The vertex descriptor table

`native_rasterizer.cpp:1216-1262`, `vk_pipeline_cache.cpp:1150-1165`, `vk_graphics_pipeline.cpp:436-470`, `vk_draw_prep.cpp:125-160` and the vertex binding code of `vk_rasterizer.cpp` (~1408-1500) all do the same thing: for each fetch-shader attribute read `sgpr_base` / `dword_offset`, find the pointer in the vertex stage's user data, read 16 bytes from guest memory at that table entry, add `inst_offset` to the base, and use `base`, `size`, `stride`, `dfmt`, `nfmt`. There is a `DrainAudit::Read(ReadFetch, ...)` hook around the read, and the comment at `vk_draw_prep.cpp:~135` says the V# may come from user data the guest is still writing. For native draws the table sits in the table arena window written by the native layer. So the native layer **writes** these 16-byte entries and the host **reads and decodes** them back, five times. The replacement is a vertex description in the record (per attribute: buffer id or address+size, stride, `vk::Format`, step rate), built where the table is written; it is the engine's input layout and vertex buffer bindings. Whether the record already carries enough to build it is a local question (section 7).

## 6. Proposed F4 steps (cloud / local split)

| step | what | who | gate |
|---|---|---|---|
| F4.0 | run the section 8 counters in the default scene and a cutscene; record which `GetSharp` sites are reached | local | list of reached sites |
| F4.1 | introduce plain native descriptions: `Core::Native::ImageDesc` (vk::Format, bits per block, block flag, type, w/h/d, pitch, base/last level, base/last array, tile mode, pow2, alt tile, min lod, dst sel, samples, address), `SamplerDesc` (the 14 fields as Vulkan-ready values), `BufferDesc` (address, size, stride, `vk::Format`). They live in `native/` and include no PS4 header. | cloud shadow, local play | built next to the sharps, compared field by field (the existing `Compare` becomes the check) |
| F4.2 | `ImageInfo` / `ImageViewInfo` / `Sampler` get constructors from the native description; the sharp constructors stay behind a switch | cloud | 0 differences between the two `ImageInfo` for every binding in both scenes |
| F4.3 | the rasterizer and `draw_prep` key by the description (hash of the description, not of 32 sharp bytes) and carry it in `PreparedImage` | cloud + local | texture set key stable (no extra uploads or image creations) |
| F4.4 | vertex description in the record (section 5) replaces the five table readers | local (hook / record change) + cloud | 0 differences against the table decode |
| F4.5 | `specialization.h` fills from the native description for the fields in section 2(A) (`FillImage`, `FillBuffer`, `FillSampler`); the translator keeps reading the shader's own words | cloud | spec keys equal for every pipeline in both scenes (counter: `spec key differences` 0) |
| F4.6 | delete: host users of the three types, `PreparedImage::tsharp`, `ImageDescHash`, `CheckImageSharp`, the legacy fallback sites; `resource.h` / `pixel_format.h` leave the host and `native/` includes | cloud | grep of `AmdGpu::Image|Buffer|Sampler` outside `shader_recompiler/` finds nothing |

Order constraints: F4.1 to F4.3 do not depend on F3. F4.4 needs the record to name the vertex streams. F4.5 needs F4.1 and is the last because it touches the shader cache key (a pipeline cache invalidation if the key layout changes: keep the key bytes equal, only change who fills them).

## 7. What the cloud cannot decide (needs a local look)

| question | evidence | what to look at |
|---|---|---|
| Does the engine's T# copy at `view+0x58` carry fields the D3D description cannot (padded rows, level limit, **`min_lod`**)? `Derive` takes `width`, `height`, `last_level` from it. | `native_image_desc.cpp:Derive` last block | Ghidra: the function that builds the SRV and copies the T# into the view object; which fields it patches after creation. If `min_lod` or `mtype`-like fields are patched, `ImageDesc` needs them |
| Does the record carry the vertex streams (buffer id, stride, format per attribute)? | section 5; the table arena is written by the native layer | the record builder for the vertex input table: whether the engine's input layout and vertex buffer slots are already in hand there |
| `alt_tile_mode` of **textures** (Neo mode) | `ImageInfo` reads `image.alt_tile_mode` + `sceKernelIsNeoMode()`; the colour target version is `TargetAltTileMode` | counter: how many bound T# have `alt_tile_mode == 1` in both scenes (0 expected; needs a counter, section 8) |
| Custom border colours | `Sampler` reads a table in guest memory (`border_color_ptr`) | counter: samplers with `border_color_type == Custom` (the rasterizer rejects it when `current_bc_base` is 0) |
| Post operations on sharps (`ConvertCubeTo2DArray`, `DisableAnisoIfSingleLod` reads dword 3 of the T#, `ForceRepeatXyzClamp`, ...) | `resource.h` | these are shader-side corrections of the console words. A native `SamplerDesc` has to apply the same corrections from the description (aniso off for a single level image needs the **image's** level count) |
| Multisample textures (`NumSamples > 1`) | `RefreshImage` returns for them; `ImageInfo` takes `NumSamples()` | counter for sampled images with samples > 1 |
| Sharps whose address is not an engine texture (T# built by the shader or written by a compute pass) | `NoteBinder(6)`, `identity_binding == ~0u` | the `Decide` counters `g_decide_no_id` / `g_decide_unknown` |

## 8. Counters that already exist and answer F4.0

| counter | answers |
|---|---|
| /stats `native_backend.registry` -> `native_binder` (`NativeRegistry::Report`, debug server) | `buffers_from_the_record`, `images_from_the_record`, `samplers_from_the_record` versus `buffers_from_the_flat_user_data` / `images_from_the_flat_user_data` (the legacy read), `skipped_stage_needed_the_flat_user_data` (must stay 0), and the `*_differing_from_the_legacy_ones` verify counters; `kept_flat_user_data_by_class_and_reason` says why a stage still read the flat user data (a buffer that is not a single load, a flat constant buffer, FMasks, a sampler that cannot be built from words) |
| `device.e2_derive` (`NativeImageDesc::Report`: decided, unknown object, differing from the record) | how many image bindings the engine decides and what is left |
| `device.e2_census` (`TLG_DESC_CHECKS=1`; `CheckImageSharp` / `CheckBufferSharp`) | per-field agreement of the engine object with the decoded sharp |
| `NoteFlatRead(2 or 3, ...)` | reads of the flat user data for images (2) and samplers (3): where the legacy path still runs |
| `rare_events.txt` `image_desc_decided_differs_from_record` | the first differing derived T# |

(The exact /stats path of the registry report is the one `native_debug_server.cpp:602` serves; check `native_binder` there.)

New counters worth a cloud round (not written yet): a field-use histogram of the bound T#, S# and V# (alt tile, min_lod, samples, border colour, post operations, formats seen) per 25 s, in the default scene and a cutscene. This is the evidence the "constant or refusal" rule of the plan asks for before the register-style read is deleted.

## 9. Allowlist consequences (F5 rows this census feeds)

| entry | after F4 |
|---|---|
| `video_core/amdgpu/resource.h` | leaves `native/` (only `native_image_desc.h` includes it); stays for the recompiler |
| `video_core/amdgpu/pixel_format.h` | `native_draw.cpp` stops including it once formats go DXGI -> `vk::Format`; stays for the recompiler |
| `video_core/amdgpu/draw_state.h` | F3 (it only defines `DrawRegs`) |
| `liverpool_to_vk.h` | the format map keeps its (dfmt, nfmt) side for the recompiler's users; `native_draw.cpp` stops including it once the target fields are native |
| `vk_fixed_state.h`, `vk_indexed_data.h`, `vk_identity_audit.h`, `liverpool.h`, `pm4_cmds.h`, `gnmdriver.h` | not F4 (F1, F3, F5, F6) |
| `native_texture_cache.cpp`, `native_buffer_cache.cpp`, `native_rasterizer.cpp`, `native_liverpool.cpp`, `native_rasterizer_helpers.h` (member files) | not F4; but `native_rasterizer.cpp` is the file that holds the vertex table reader and `PlanBufferSharp`, so F4.4 removes the largest AmdGpu use from it |
