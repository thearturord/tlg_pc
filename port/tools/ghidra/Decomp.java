// Cloud-session helper: paths under /tmp/claude-0/game are the container's defaults (see README.md, "Reading the game code without the binary on the owner's machine").
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Decomp extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String list = args.length > 0 ? args[0] : "/tmp/claude-0/game/want.txt";
        String out = args.length > 1 ? args[1] : "/tmp/claude-0/game/decomp.txt";
        DecompInterface di = new DecompInterface();
        di.openProgram(currentProgram);
        StringBuilder sb = new StringBuilder();
        for (String l : Files.readAllLines(Paths.get(list))) {
            l = l.trim();
            if (l.isEmpty() || l.startsWith("#")) continue;
            Address a = toAddr(Long.decode(l));
            Function f = getFunctionContaining(a);
            if (f == null) { sb.append("// ").append(l).append(": no function contains it\n"); continue; }
            DecompileResults r = di.decompileFunction(f, 120, monitor);
            sb.append("// ===== ").append(f.getName()).append(" @ ").append(f.getEntryPoint()).append(" (asked ").append(l).append(")\n");
            if (r.decompileCompleted()) sb.append(r.getDecompiledFunction().getC()).append("\n");
            else sb.append("// decompile failed: ").append(r.getErrorMessage()).append("\n");
        }
        Files.write(Paths.get(out), sb.toString().getBytes());
        println("wrote " + out);
    }
}
