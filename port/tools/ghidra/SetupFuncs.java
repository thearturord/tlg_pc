// Cloud-session helper: paths under /tmp/claude-0/game are the container's defaults (see README.md, "Reading the game code without the binary on the owner's machine").
import ghidra.app.script.GhidraScript;
import ghidra.app.cmd.disassemble.DisassembleCommand;
import ghidra.app.cmd.function.CreateFunctionCmd;
import ghidra.program.model.address.*;
import ghidra.program.model.mem.*;
import java.nio.file.*;
import java.util.*;

public class SetupFuncs extends GhidraScript {
    @Override
    public void run() throws Exception {
        Memory mem = currentProgram.getMemory();
        for (MemoryBlock b : mem.getBlocks()) {
            long start = b.getStart().getOffset();
            // code is vaddr 0 .. 0x16b2d5c, the rest is data
            b.setRead(true); b.setWrite(true); b.setExecute(true);
            println("block " + b.getName() + " " + b.getStart() + " - " + b.getEnd());
        }
        List<String> lines = Files.readAllLines(Paths.get("/tmp/claude-0/game/func_starts.txt"));
        int made = 0, failed = 0;
        for (String l : lines) {
            l = l.trim();
            if (l.isEmpty()) continue;
            Address a = toAddr(Long.decode(l));
            if (getFunctionAt(a) != null) continue;
            DisassembleCommand d = new DisassembleCommand(a, null, true);
            d.applyTo(currentProgram, monitor);
            CreateFunctionCmd c = new CreateFunctionCmd(a);
            if (c.applyTo(currentProgram, monitor)) made++; else failed++;
            if (((made + failed) % 2000) == 0) println("progress " + (made + failed));
        }
        println("functions created: " + made + " failed: " + failed);
    }
}
