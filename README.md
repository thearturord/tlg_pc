<p align="center"><img src="port/packaging/icon/tlg-icon-1024.png" width="160" alt=""></p>

<h1 align="center">The Last Guardian — native PC port</h1>

An experimental fork of [HayatoG/tlg_pc](https://github.com/HayatoG/tlg_pc) that replaces the port's GPU layer with a **native renderer**. **Windows only**, tested on Windows 11 with an RTX 4090. No game files are included: you need your own copy of The Last Guardian (CUSA03627 / CUSA03745, update 01.03) dumped from your own PS4. For the game files, the launcher and everything the original port does, see the original project.

**The goal is 120 fps.** `main` always holds the latest stable build code.

## Progress (2026-11-08)

```
Road to 120 fps          [███████░░░░░░░░░░░░░]  37%   about 45 of 120 fps
Native renderer          [████████████████████] 100%   no PS4 GPU emulation left (named interfaces stay)
Native data path         [░░░░░░░░░░░░░░░░░░░░]   0%   0 of 7 steps
Thread split             [███░░░░░░░░░░░░░░░░░]  14%   1 of 7 steps
```

* **Road to 120 fps:** the fps is measured in headless test runs of the default scene at 4K with one frame in flight, on an RTX 4090 that is deliberately underclocked. The raster thread is the limit: it needs to come down from about 22 ms per frame to 14 ms for 60 fps and to 8.3 ms for 120 fps. 60 fps is the next milestone, 120 fps the goal.
* **Native renderer:** done. The PS4 command stream, the GPU registers and the address space mirror are gone; what is left is named and by design (the shader recompiler boundary, PS4 data layouts, the Gnm submit input).
* **Native data path (next):** the renderer still finds out what the game changed by address and then copies it. It should be told which object changed and its version instead. Steps: census, versions instead of guessing, copy exactly what changed, bind buffers by object, images by id, delete the address-keyed leftovers, optional shader change for the flat user data buffer. Measured on the raster thread today: about 14 % announcing and marking writes, about 16 % copying guest memory, about 18 % finding buffers and images by address.
* **Thread split:** done so far is a recorder thread (off by default, `TLG_SPLIT_BIND=1` with `TLG_SPLIT_THREAD=1`). It gave no gain alone because recording is only about 9 % of the raster thread. The next lanes (apply, compute, shadow draws) come after the data path, because tables keyed by object and version are safe to share between threads and the address-keyed caches are not.

## What this fork is

The original port runs the game's code natively and translates the PS4 GPU commands to Vulkan. This fork stops translating them. The game's own renderer already talks to a Direct3D style device layer, and we hook that layer: every draw, dispatch, clear, copy, fence and flip becomes a typed record that a command processor plans and a raster thread runs. The PS4 command stream (PM4), the GPU registers and the PS4 address space mirror are being removed one layer at a time, so that the renderer ends up fully native and can then be optimized (the goal is 120 FPS).

## Status of the layers (2026-11-08)

| Layer | State |
|---|---|
| Command stream (PM4) | **Done.** No PM4 packet is executed or interpreted; the stream carries only our records |
| Synchronization and present | **Done.** Fences, labels, flips and frame pacing are typed records and native objects, with a native display gate and presenter |
| Draw state (fixed function, shader environment, draw arguments) | **Done**, played, default |
| Resource descriptions (images, samplers, buffers) | **Done.** Plain descriptions made from the engine's own objects; the recompiler gets its PS4 shapes at one named boundary |
| Image identity (which image a draw names) | **Done** for named lookups (targets, bindings, transitions) by the engine's resource id. Not done: content refresh from announced writes, overlap handling kept for same-memory images |
| Memory (the "arena") | **Done.** The sparse mirror of the PS4 address space, the BDA page table and the fault manager are deleted; memory no engine buffer owns is a pseudo resource |
| Page protection, write watch, page tracker, address-keyed buffer API | **Done except the lookup entry points.** Page protection, the write watch (module, kernel write tracking, epochs), the GPU read watch, the fault readbacks and the page tracker are deleted: what the CPU changed comes from the engine's announced writes, what the GPU wrote from the GPU written bytes. `ObtainBuffer` / `BufferFor` stay as an address to placed-buffer lookup (the callers get addresses from the command stream) |
| Prune (dead register and PM4 structures) | **Done.** The emulated command processor classes are replaced by a native executor and raster host |
| Address-keyed data path (announce by address, mark, copy by walking guest memory, find by address) | **Not done**, the next phase (see Progress) |


## Features

- **120 FPS mode:** `TLG_FPS=120` (also launcher → Frame rate). Sets the game's time step and the vblank to 120 Hz.
- **Frames in flight:** `TLG_FRAMES_IN_FLIGHT=0|1|2` (default 0, 1 is what is tested for speed). 0 makes the game wait for its own frame; 2 is not stable yet.
- **4K patch:** the mod loader accepts the RELRO segment, which allows the 4K resolution patch in `user/mods`.
- **FPS counter** (F10) and **F11** switches between windowed and fullscreen (desktop mode, Alt+Tab works).
- **Debug server** on `127.0.0.1:7777` (JSON): `/stats`, `/status`, `/log`, `/timeline_capture?seconds=10&tag=x`, `/stall_capture`, `/resources` and more. Press F8 for the stall profiler, F9 for the frame profiler and Shift+F9 for the probe profiler (named probes in the code, any thread, report in `user/log/prof_<time>.html`; `/prof` configures it, `/prof_baseline` compares runs).
- **Regression script:** a headless run that compares the native answers with the old ones (`PASS` means 0 differences).
- **Experimental split of the render thread** (off by default): `TLG_SPLIT_BIND=1` resolves a draw first and records it second, `TLG_SPLIT_QUEUE=N` delays the recording by N draws, `TLG_SPLIT_THREAD=1` records on its own thread; `TLG_SPLIT_BIND=2` runs the old and the new path side by side and counts differences.
- **Check switches** (off by default): `TLG_B9_VERIFY=N` compares every Nth page of GPU buffers with memory and counts writes nobody announced (read only); `TLG_E4_CENSUS=1` compares the page tracker with the announced writes.

## Known issues

- One dark or bright frame while walking (sunlight and fog missing for one frame); a bisect is open.
- The original port's temporal upscalers (FSR 3/4, TAA), scene resolution and post effects are **not ported** to this build yet.
- Frames in flight 2 shows glitches (for example a one-frame slab in a cutscene); use 0 or 1. Long sessions and every cutscene have not been played on this build.
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
