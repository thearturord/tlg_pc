<p align="center"><img src="port/packaging/icon/tlg-icon-1024.png" width="160" alt=""></p>

<h1 align="center">The Last Guardian — native PC port (native-v2)</h1>

An experimental fork of [HayatoG/tlg_pc](https://github.com/HayatoG/tlg_pc) that replaces the port's GPU layer with a **native renderer**. **Windows only**, tested on Windows 11 with an RTX 4090. No game files are included: you need your own copy of The Last Guardian (CUSA03627 / CUSA03745, update 01.03) dumped from your own PS4. For the game files, the launcher and everything the original port does, see the original project.

## What this fork is

The original port runs the game's code natively and translates the PS4 GPU commands to Vulkan. This fork stops translating them. The game's own renderer already talks to a Direct3D style device layer, and we hook that layer: every draw, dispatch, clear, copy, fence and flip becomes a typed record that a command processor plans and a raster thread runs. The PS4 command stream (PM4), the GPU registers and the PS4 address space mirror are being removed one layer at a time, so that the renderer ends up fully native and can then be optimized (the goal is 120 FPS).

## Status (2026-10-31, branch `native-v2`)

| Layer | State |
|---|---|
| Command stream (PM4) | **Done.** No PM4 packet is executed or interpreted; the stream carries only our records |
| Synchronization and present | **Done.** Fences, labels, flips and frame pacing are typed records and native objects, with a native display gate and presenter |
| Draw state (fixed function, shader environment, draw arguments) | **Done**, played, default |
| Resource descriptions (images, samplers) | **Done** for images and samplers; buffer descriptions still derive from the PS4 side |
| Image identity (which image a draw names) | **Done** for named lookups (targets, bindings, transitions) by the engine's resource id. Not done: content refresh from announced writes, overlap handling kept for same-memory images |
| Memory (the "arena") | **Done.** The sparse mirror of the PS4 address space, the BDA page table and the fault manager are deleted; memory no engine buffer owns is a pseudo resource |
| Page tracker, address-keyed buffer API, write watch | **Not done.** The tracker still decides uploads for memory the engine does not own and watches GPU-written pages for CPU reads. A census shows it adds almost nothing there (16 pages in 40 s), but it is not deleted until a cutscene run says the same |
| Prune (dead register and PM4 structures) | Not started |

The design, the phases (E0 to E9), every census and the decisions are written down in the plan this work follows.

## Features

- **120 FPS mode:** `TLG_FPS=120` (also launcher → Frame rate). Sets the game's time step and the vblank to 120 Hz.
- **Frames in flight:** `TLG_FRAMES_IN_FLIGHT=0|1|2` (default 1). 0 makes the game wait for its own frame.
- **4K patch:** the mod loader accepts the RELRO segment, which allows the 4K resolution patch in `user/mods`.
- **FPS counter** (F10) and **F11** switches between windowed and fullscreen (desktop mode, Alt+Tab works).
- **Debug server** on `127.0.0.1:7777` (JSON): `/stats`, `/status`, `/log`, `/timeline_capture?seconds=10&tag=x`, `/stall_capture`, `/resources` and more. Press F8 for the stall profiler.
- **Regression script:** a headless run that compares the native answers with the old ones (`PASS` means 0 differences).
- **Check switches** (off by default): `TLG_B9_VERIFY=N` compares every Nth page of GPU buffers with memory and counts writes nobody announced (read only); `TLG_E4_CENSUS=1` compares the page tracker with the announced writes.

## Known issues

- One dark or bright frame while walking (sunlight and fog missing for one frame); a bisect is open.
- The original port's temporal upscalers (FSR 3/4, TAA), scene resolution and post effects are **not ported** to this build yet.
- Not played yet after the memory change: other cutscenes and long sessions. Black patches on the ground after removing the arena were fixed (they came from freeing a block that held GPU-written bytes); if you see them again, please report where.
- GPU hangs happened earlier in the project when the CPU's data went stale. If the picture turns to garbage or the game stutters hard, close it and report the scene.

## Running it

Use the original port's instructions for the game files, the launcher and the settings. Start the game with the environment variables you want, for example:

```bat
set TLG_FPS=120
set TLG_VBLANK_HZ=120
set TLG_ENGINE_FPS=120
set TLG_FRAMES_IN_FLIGHT=0
set TLG_SHOW_FPS=1
set TLG_GAME_DIR=C:\path\to\CUSA03627
"The Last Guardian.exe" --play
```

## Credits and license

Built on the work of [HayatoG](https://github.com/HayatoG/tlg_pc) (the original port), [gow3_pc](https://github.com/FelipePrado0/gow3_pc), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) and [shadPS4](https://github.com/shadps4-emu/shadPS4), all GPL-2.0, as is this project ([LICENSE](port/LICENSE)). Not affiliated with Sony Interactive Entertainment. The Last Guardian is a trademark of Sony Interactive Entertainment. No game code, files or art are distributed here. The icon is original artwork of the original project.
