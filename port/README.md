# port/: loader, runtime and video core of The Last Guardian for PC

This directory holds the code that runs the original `eboot.bin` of TLG (PS4, CUSA03627 v1.03) on PC:

- `scripts/`: preparing and linking the eboot with the game's and the console's modules (`prepare.py`, `link_modules.py`);
- `src/`: the loader (`probe.c`) and the runtime that reimplements the PS4 libraries (`runtime_*.c`, `runtime_tlg.c`);
- `gpu/`: the GPU side: the shadPS4 core, the native renderer in `gpu/native/`, and the overlay and menu in `gpu/shim/`;
- `launcher/`: the graphical launcher (`tlg_launcher.py`) and the run core (`tlg_run.py`).

The code's origin (gow3_pc → bloodborne_pc, shadPS4) and the license (GPL-2.0-or-later) are in
[UPSTREAM.md](UPSTREAM.md) and [LICENSE](LICENSE). The native renderer's plan and record live outside this repo, in
`F:\1.Dev\DEV Shadps4\documents\tlg\native plan\`.
