#!/usr/bin/env python3
"""Where the game calls an import: the PLT stub of each import (from the eboot's SCE dynamic data) and every direct call or jump to it.

    python import_sites.py <eboot.elf> <eboot.mem> <func_starts.txt> [ghidra_names.tsv] [--hrtimer] SYMBOL...

eboot.elf: the decrypted eboot (ELF); eboot.mem: its memory image at the Ghidra base 0x1000000 (port/tools/ghidra/README.md);
func_starts.txt: one function start per line (hex), as the Ghidra recipe writes it; ghidra_names.tsv: optional names (ExportNames.java).
Symbols are the names in port/src/import_names.inc (memcpy, scePthreadRwlockRdlock, sceKernelAddHRTimerEvent, ...).
--hrtimer also lists the timer lengths the sceKernelAddHRTimerEvent call sites pass (needs capstone).
"""
import bisect
import collections
import re
import struct
import sys
from pathlib import Path

BASE = 0x1000000
DT_SCE_JMPREL, DT_SCE_PLTRELSZ, DT_SCE_STRTAB, DT_SCE_SYMTAB = 0x61000029, 0x6100002D, 0x61000035, 0x61000039


def plt_names(elf):
    phoff, phnum = struct.unpack_from('<Q', elf, 0x20)[0], struct.unpack_from('<H', elf, 0x38)[0]
    phs = [struct.unpack_from('<IIQQQQQQ', elf, phoff + i * 56) for i in range(phnum)]
    dyn = next(p for p in phs if p[0] == 2)
    dlib = next(p for p in phs if p[0] == 0x61000000)
    tags = {}
    for i in range(dyn[5] // 16):
        t, v = struct.unpack_from('<qQ', elf, dyn[2] + i * 16)
        tags.setdefault(t, v)
    base = dlib[2]
    jmprel, size = base + tags[DT_SCE_JMPREL], tags[DT_SCE_PLTRELSZ]
    symtab, strtab = base + tags[DT_SCE_SYMTAB], base + tags[DT_SCE_STRTAB]
    out = {}
    for i in range(size // 24):
        r_offset, r_info, _ = struct.unpack_from('<QQq', elf, jmprel + i * 24)
        name_off = struct.unpack_from('<I', elf, symtab + (r_info >> 32) * 24)[0]
        out[r_offset + BASE] = elf[strtab + name_off:elf.index(b'\0', strtab + name_off)].decode()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    hrtimer = '--hrtimer' in sys.argv
    elf, mem = Path(args[0]).read_bytes(), Path(args[1]).read_bytes()
    starts = sorted(int(x.split()[0], 16) for x in Path(args[2]).read_text().split('\n') if x.strip())
    names = {}
    rest = args[3:]
    if rest and rest[0].endswith('.tsv'):
        for line in Path(rest[0]).read_text(errors='replace').split('\n'):
            p = line.split('\t')
            if len(p) > 3 and p[0] == 'FUNC':
                try:
                    names[int(p[1], 16)] = p[2]
                except ValueError:
                    pass
        rest = rest[1:]
    inc = (Path(__file__).resolve().parents[2] / 'src' / 'import_names.inc').read_text()
    nid_symbol = {a.split('#')[0]: b for a, b in re.findall(r'\{"([^"]+)","([^"]+)"\}', inc)}
    slots = plt_names(elf)
    stubs = {}
    code_end = max(slots) - BASE if slots else len(mem)
    for i in range(0, min(len(mem), 0x24C0000 - BASE) - 6):
        if mem[i] == 0xFF and mem[i + 1] == 0x25:
            target = BASE + i + 6 + struct.unpack_from('<i', mem, i + 2)[0]
            if target in slots:
                stubs[BASE + i] = nid_symbol.get(slots[target].split('#')[0], slots[target])
    wanted = {s: a for a, s in stubs.items() if s in rest}
    sites = collections.defaultdict(list)
    limit = min(stubs) - BASE if stubs else 0
    for i in range(limit - 5):
        op = mem[i]
        if op in (0xE8, 0xE9):
            target = BASE + i + 5 + struct.unpack_from('<i', mem, i + 1)[0]
            if target in stubs and stubs[target] in rest:
                sites[stubs[target]].append(BASE + i)
    for symbol in rest:
        found = sites.get(symbol, [])
        functions = collections.Counter(starts[bisect.bisect_right(starts, a) - 1] for a in found)
        print('%-28s stub %-10s %5d call sites in %4d functions' % (symbol, hex(wanted[symbol]) if symbol in wanted else '-', len(found), len(functions)))
        for f, n in functions.most_common(10):
            print('    %#x %-40s %d' % (f, names.get(f, ''), n))
    if hrtimer and 'sceKernelAddHRTimerEvent' in sites:
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        lengths = collections.Counter()
        for site in sites['sceKernelAddHRTimerEvent']:
            values = []
            for ins in md.disasm(mem[site - 0x80 - BASE:site + 5 - BASE], site - 0x80):
                ops = ins.op_str.split(',')
                if ins.mnemonic == 'mov' and len(ops) == 2 and ops[1].strip().startswith('0x'):
                    try:
                        values.append(int(ops[1].strip(), 16))
                    except ValueError:
                        pass
            plausible = [v for v in values if 1000 <= v <= 2_000_000_000]
            lengths[plausible[-1] if plausible else None] += 1
        print('\nHR timer lengths at the call sites (the nanoseconds loaded before the call):')
        for value, n in lengths.most_common(12):
            print('    %5d  %s' % (n, 'not found' if value is None else '%d ns (%.3f ms)' % (value, value / 1e6)))


if __name__ == '__main__':
    main()
