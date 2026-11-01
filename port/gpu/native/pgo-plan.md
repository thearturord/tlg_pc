# PGO for the Windows build (plan, not started)

PGO (profile-guided optimization): build once with recording on, play a few minutes, then build again so the compiler optimizes the
code paths that actually run most. No code changes; typically a few percent up to about 15 % on CPU-bound work (the raster thread and
the game's threads are CPU-bound). Source of the idea: tabnul/tlg_pc `376f8ed` (from the SotC port), credit @tabnul.

Today: `build.sh` passes `-DTLG_PGO=off` on Windows ("GCC-only flags"); `gpu/CMakeLists.txt` and the profile writer in `shim/tlggpu.cpp`
only know GCC. The Windows build uses Clang (MSYS2 CLANG64), which needs other flags and another profile format.

## What changes (about 50 lines, the same as the fork's commit)

1. **`gpu/CMakeLists.txt`:** a Clang branch next to the GCC one:
   - generate: `-fprofile-generate=<pgo dir> -fprofile-update=atomic` (compile and link), defines `TLG_PGO_GENERATE` and `TLG_PGO_CLANG`;
   - use: `-fprofile-use=<pgo dir>/tlggpu.profdata` plus the warnings that a stale profile is allowed to cause turned off.
2. **`shim/tlggpu.cpp`:** the 30 s profile writer calls Clang's `__llvm_profile_write_file` / `__llvm_profile_reset_counters` under
   `TLG_PGO_CLANG` (the GCC calls stay for Linux). Needed because the game often ends through `_exit`, which skips the write at exit.
3. **`build.sh` (Windows part):** `TLG_PGO=generate | merge | use | off`:
   - generate: builds an instrumented exe in its own build dir (`out/gpu-pgogen`), copied to `out/tlg-probe-pgogen.exe`;
   - merge: `llvm-profdata merge -o pgo/tlggpu.profdata pgo/*.profraw`;
   - use / off; default use when `pgo/tlggpu.profdata` exists.
4. **`.gitignore` (both):** keep `pgo/tlggpu.profdata` in the repository, ignore the raw files.

Not taken from the fork: their `tlggpu.profdata`. It was recorded on the old renderer; yours must be recorded on your build.

## How to use it (each time the code changed a lot)

1. `TLG_PGO=generate ./build.sh` (instrumented exe; it runs slower).
2. Play `out/tlg-probe-pgogen.exe` with your normal settings (no test switches) for about 10 minutes: the default spot, a cutscene,
   a loading screen, a heavy area. Every 30 s it writes `pgo/*.profraw`.
3. `TLG_PGO=merge ./build.sh` (makes `pgo/tlggpu.profdata`).
4. `./build.sh` (uses the profile from now on).

A profile goes stale as code changes: changed functions are simply compiled without it (no errors). Record a new one before a release
or after big changes.

## How to check it helped

Same spot, same settings, PGO off against on: fps, and Shift+F9's raster thread busy time and the game threads' times. Several runs
each (one run varies by a few fps).

## Checks before relying on it

- `llvm-profdata` is in the MSYS2 CLANG64 toolchain (`mingw-w64-clang-x86_64-llvm`); add it to `packaging/windows/README.md` if missing.
- The instrumented build must start and run (the fork's did); if the game crashes with it, record on a shorter route.
- LTO stays off on Windows as today (PGO works without it).
