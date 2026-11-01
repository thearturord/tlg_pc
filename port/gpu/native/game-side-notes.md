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

## 120 fps: what counts updates instead of time (read 2026-11-01)

| item | finding |
|---|---|
| GPU particle emission (`0x24f109c` = float 1/30) | read by three instructions, all in `FUN_01bcac40` (it makes the particle compute dispatches `0x1bcb5e..0x1bcc3fc`): `0x1bcb5cd` and `0x1bcbf30` compute the count per update as `max(rate * 1/30, 0) + carried fraction` (the fraction is kept in the emitter, `+0x260`), `0x1bcb78c` feeds the emission speed (with the other 30 Hz constants `0x24f10a0` = 30.0, `0x24f10b4` = 1/900, `0x24f10b0` = 14.85). The simulation runs on the real frame time (`max(dt, 0.001)` goes into the compute constants). Patch `particle_emit_step` (opt-in) points the two count loads at the dt snap target `0x24f17fc`, so the count per update is rate / fps. The third read is left alone, it needs a design decision. |
| native timers (`MSCore_UpdateTimers`, `FUN_01984c10`) | time based: remaining time is reduced by the game delta (`game+0x194`). Not a problem at 120 fps. Related strings: `MSC_GetGameDelta` (3 users: `0x197fde0`, `0x1981160`, `0x1988b70`), `UseSystemDelta` (`0x15aa6a0`), `DeltaTimeModifier` (`0x164fee0`), `EngineDeltaTime` (`0x16818b0`). |
| tasks | the task registrations are in `FUN_01cff010` (`LuaTick` at `0x1d075c0`, `ScriptMgrGc` at `0x1d07910`, `WorldVolumes` at `0x1d01218`) and in the huge `FUN_01230810` (`TimerExtTick`, `WaterSamplerTick`, `WindSwingTick`, ...; the decompiler times out on it). `LuaTickBehindPhysics` is registered in `FUN_01cd6240`. |
| Lua timers that count updates | cannot be decided from the eboot: the counting is in the Lua scripts (game data). Two designs: (a) run the `LuaTick` task only on every 4th frame at 120 fps (every 2nd at 60) with the accumulated time, which fixes all update counting scripts at once but makes scripted things update at 30 Hz and could break scripts that expect a delta every frame; (b) patch the scripts that count (needs the script text). |
| environment shader phases | not located. They are shader constants that advance per update; candidates are the per frame tick tasks above. Needs the owner's observation of which effect runs fast (clouds, water, fog, foliage) to find the constant. |
