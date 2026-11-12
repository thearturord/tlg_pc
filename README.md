<p align="center"><img src="port/packaging/icon/tlg-icon-1024.png" width="160" alt=""></p>

<h1 align="center">The Last Guardian — native PC port</h1>

An experimental fork of [HayatoG/tlg_pc](https://github.com/HayatoG/tlg_pc) that replaces the port's GPU layer with a **native renderer**. **Windows only**, tested on Windows 11 with an RTX 4090. No game files are included: you need your own copy of The Last Guardian (CUSA03627 / CUSA03745, update 01.03) dumped from your own PS4. For the game files, the launcher and everything the original port does, see the original project.

**This is a work in progress.** `main` holds the latest build that was played and works. **The goal is 120 fps.**

## Progress (2026-11-12)

```
Overall                  [███████████████░░░░░]  74%
Native renderer          [████████████████████] 100%   no PS4 GPU command stream, registers or address mirror
Data ownership           [████████████████████] 100%   GPU memory follows the engine's objects, not address tracking
Resource binding         [███████████████████░]  98%   draws and dispatches bound by engine id
Threads and submission   [██████████████░░░░░░]  70%   from the game's draw call to the GPU
Clean-up                 [░░░░░░░░░░░░░░░░░░░░]   0%   old emulator paths out of the build
```

The overall bar is the average of the five below it.

* **Native renderer:** done. The game's renderer already talks to a Direct3D style device layer, and the port hooks that layer. Every draw, dispatch, clear, copy, fence and flip becomes a typed record that the port plans and runs on Vulkan. The PS4 command stream (PM4), the GPU registers and the mirror of the PS4 address space are gone.
* **Data ownership:** done. The GPU copy of each buffer and texture belongs to the engine object that owns it, and the per-frame data is taken where the game writes it. Nothing has to watch memory pages or guess what changed.
* **Resource binding:** nearly done. Almost every draw and dispatch is bound straight from the engine's ids. The emulator-style lookups (by address and PS4 descriptor words) are left only for a few rare cases.
* **Threads and submission:** mostly done. Each draw is now planned on the game's own thread when the game records it, from read-only copies of the caches, and the command thread only takes it in order and passes it on (no record copy, no planning there). Buffers are described by the engine's objects instead of PS4 descriptor words. Still to do: compute work (physics, particles) on its own line and queue, so the game waits less for its results.
* **Clean-up:** after that. Every removed emulator path is still in the code behind a switch, so it can be compared and turned back on. Once they are retired, the code is deleted, and what is left of shadPS4's Vulkan code moves into the port's own renderer. The shader translator (the game ships only PS4 GPU binaries) and the system library layer stay.
* **Frame rate:** the frame rate is not tuned while the emulator logic is being removed. It is measured and optimized once that is finished.

## Features

- **120 FPS mode:** `TLG_FPS=120` (also launcher → Frame rate). Sets the game's time step and the vblank to 120 Hz.
- **Frames in flight:** `TLG_FRAMES_IN_FLIGHT=0|1|2` (default 0). 0 makes the game wait for its own frame; 1 is what is tested for speed.
- **4K patch:** the mod loader accepts the RELRO segment, which allows the 4K resolution patch in `user/mods`.
- **FPS counter** (F10) and **F11** switches between windowed and fullscreen (desktop mode, Alt+Tab works).
- **Debug server** (setting `debug_server_port`, off by default): JSON endpoints for stats, status, the log and timeline captures, plus in-game profilers.

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
