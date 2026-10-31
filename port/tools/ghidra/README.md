# Ghidra scripts

`ExportNames.java` writes what you added to the program (function and label names, prototypes, comments, structs, applied data types, Note bookmarks) to `ghidra_names.tsv`, a few hundred KB at most. `ImportNames.java` applies such a file to another Ghidra project of the same program (same absolute addresses; the repo notes use image base 0x1000000). Used to hand your analysis to a session that cannot receive the whole project.

Export: Window > Script Manager > the "Script Directories" icon > add `port/tools/ghidra` > select `ExportNames` > run > pick a folder. Zip the `ghidra_names.tsv` and send it.
