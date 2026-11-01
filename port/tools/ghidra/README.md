# Ghidra scripts

`ExportNames.java` writes what you added to the program (function and label names, prototypes, comments, structs, applied data types, Note bookmarks) to `ghidra_names.tsv`, a few hundred KB at most. `ImportNames.java` applies such a file to another Ghidra project of the same program (same absolute addresses; the repo notes use image base 0x1000000). Used to hand your analysis to a session that cannot receive the whole project.

Export: Window > Script Manager > the "Script Directories" icon > add `port/tools/ghidra` > select `ExportNames` > run > pick a folder. Zip the `ghidra_names.tsv` and send it.

## Reading the game code in a cloud session (headless Ghidra)

A cloud session can run Ghidra without the owner's project. Recipe (all tested, Ghidra 11.3.2 + Java 21):
1. The owner uploads `eboot.bin` (plaintext PS4 SELF) and a `ghidra_names.tsv` made with `ExportNames.java`.
2. Extract the ELF with `port/scripts/prepare.py` (`parse_self`) and write a flat memory image of the PT_LOAD segments (vaddr = file offset in the image); import it as a raw binary, x86-64, base `0x1000000` (the address convention of the notes: Ghidra address = runtime address - image base + 0x1000000).
3. A raw image has no function list: take the function starts from the ELF's `PT_GNU_EH_FRAME` table (22,478 of them), write them to `func_starts.txt` and run `SetupFuncs.java` (import with `-noanalysis`).
4. `ImportNames.java <tsv>` applies the owner's names and comments. `Decomp.java <list> <out>` decompiles the functions that contain given addresses; `Near.java <list> <out> [context]` prints only the decompiled lines around call sites (a profile's "caller" addresses are return addresses, so the script matches the 8 bytes before). The decompiler's response limit must be raised for the big functions (`setMaxPayloadMBytes`); `RenderDev_RenderScene` (79 KB) takes a few minutes.
