# Where the code comes from

This directory started as a copy of **[FelipePrado0/gow3_pc](https://github.com/FelipePrado0/gow3_pc)**
at commit `9a6d17cdb54109b88f9296c0643d6384b62a1aaf` (2026-10-12). That project is a fork of
Supermedo/bloodborne_pc, which in turn derives from [deadinside28/bloodborne_pc](https://github.com/deadinside28/bloodborne_pc).
The video core in `gpu/shadps4/` comes from [shadPS4](https://github.com/shadps4-emu/shadPS4).

License: **GNU GPL v2 or later** (`LICENSE`). Third-party components keep their own licenses (for example SMAA and AMD CAS, MIT: see `gpu/VENDOR.txt`).

The changes for The Last Guardian (CUSA03627 v1.03) are made here and recorded in the git history.
To bring in upstream fixes, compare against the commit above.

**ISA rewrite** (upstream gow3_pc PR #129, via iss_pc `f79c7a3`, `4e4daad`, `c2c14a0`, and iss_pc's SSE4A `3727fde`):
`src/cpu_check.c`, `src/isa_emulate.c`, `src/isa_rewrite.c`, `scripts/isa_rewrite.py` and their tests, for CPUs without
MOVBE/BMI1/LZCNT/SSE4A (every Intel CPU), taken from gr2_pc `5c5d84d`. The same table carries the Windows red zone
protection (iss_pc #45, `scripts/red_zone.py`).
