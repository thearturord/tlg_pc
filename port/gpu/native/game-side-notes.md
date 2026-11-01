# Game side of the frame (from the decompile, 2026-11-01)

Read with the headless Ghidra recipe in `port/tools/ghidra/README.md`, using the owner's names (`ghidra_names.tsv`). Addresses are Ghidra addresses (image base 0x1000000). These are the callers the raster profile names (`/prof`, tag `caller`; they are return addresses).

| profile caller | function | what the call site is |
|---|---|---|
| `0x1c76433`, `0x1c7640f` | `RenderDev_DrawZPass` (0x1c74a00) | the z prepass / shadow loop: one `JsDeviceContext_DrawIndexedCommon` per mesh batch (the instanced and indirect variants are the other two branches; the indirect one is `0x1c7640f`). Before each draw the loop may write a command word of its own into the command stream (`0xc0017900` = a SET_UCONFIG_REG packet, register 0x248). About 670 draws per frame. |
| `0x1c777a1`, `0x1c77788` | `RenderDev_DrawGBufferPrimitives` (0x1c76710) | the g-buffer loop, the same shape: one `DrawIndexedCommon` per primitive batch, with the instanced and indirect (stride 0xd8) variants. About 360 draws per frame. |
| `0x1c7819d` | `RenderDev_DrawForwardPass` (0x1c779a0) | the forward pass (about 30 draws per frame, 37 us each on the raster thread). |
| `0x1c60bb2` | `RenderDev_RenderScene` (0x1c593d0, 79 KB) | a compute dispatch through the device context's vtable (slot +0x58) inside a loop over a list of items: `Dispatch(ctx, (n + 63) >> 6, 1, 1)`, followed by two state resets (slots +0x28 and +0x38). About 160 small dispatches per frame, one per item. |

What this says: the engine submits one command per batch and one dispatch per item; nothing in these loops skips or merges work. So the raster thread's cost per command (11 us a draw, 12 us a dispatch in the profile) is the lever, not the game's count. Merging the per-item dispatches would be a change in the game's own code (the loop in `RenderScene` around `0x1c60b50..0x1c60c18`), which is possible but needs care (each item has its own buffers and count).

Not yet read: the task graph (`ComputeFrame` and the rest of the profile's "engine tasks": the owner's spans file maps their names to addresses and was not part of the export), the 1/30 s step logic.
