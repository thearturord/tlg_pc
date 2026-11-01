import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import java.nio.file.*;
import java.util.*;

/** Prints the decompiled lines around given addresses: Near.java <list file> <out file> [context lines]. */
public class Near extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        int ctx = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        DecompInterface di = new DecompInterface();
        ghidra.app.decompiler.DecompileOptions opts = new ghidra.app.decompiler.DecompileOptions();
        opts.setMaxPayloadMBytes(800);
        di.setOptions(opts);
        di.openProgram(currentProgram);
        StringBuilder sb = new StringBuilder();
        Map<Address, DecompileResults> cache = new HashMap<>();
        for (String l : Files.readAllLines(Paths.get(args[0]))) {
            l = l.trim();
            if (l.isEmpty() || l.startsWith("#")) continue;
            Address a = toAddr(Long.decode(l));
            Function f = getFunctionContaining(a);
            if (f == null) { sb.append("// ").append(l).append(": no function\n"); continue; }
            DecompileResults r = cache.get(f.getEntryPoint());
            if (r == null) { r = di.decompileFunction(f, 900, monitor); cache.put(f.getEntryPoint(), r); }
            if (!r.decompileCompleted()) { sb.append("// ").append(l).append(": decompile failed: ").append(r.getErrorMessage()).append("\n"); continue; }
            ClangTokenGroup root = r.getCCodeMarkup();
            // collect lines
            List<ClangNode> toks = new ArrayList<>();
            root.flatten(toks);
            // build text lines with address ranges
            List<String> text = new ArrayList<>();
            List<Boolean> hit = new ArrayList<>();
            StringBuilder cur = new StringBuilder(); boolean h = false;
            for (ClangNode node : toks) {
                if (!(node instanceof ClangToken)) continue;
                ClangToken t = (ClangToken) node;
                if (t instanceof ClangBreak) { text.add(cur.toString()); hit.add(h); cur = new StringBuilder(); h = false; cur.append(" ".repeat(Math.max(0, ((ClangBreak) t).getIndent()))); continue; }
                cur.append(t.getText());
                Address mn = t.getMinAddress(), mx = t.getMaxAddress();
                if (mn != null && mx != null && mn.compareTo(a) <= 0 && mx.compareTo(a.subtract(8)) >= 0) h = true;
            }
            text.add(cur.toString()); hit.add(h);
            int first = -1, last = -1;
            for (int i = 0; i < text.size(); i++) if (hit.get(i)) { if (first < 0) first = i; last = i; }
            sb.append("// ===== ").append(f.getName()).append(" @ ").append(f.getEntryPoint()).append(" near ").append(l).append("\n");
            if (first < 0) { sb.append("// (no line carries that address)\n"); continue; }
            for (int i = Math.max(0, first - ctx); i <= Math.min(text.size() - 1, last + ctx); i++) sb.append(i == first ? ">>> " : "    ").append(text.get(i)).append("\n");
        }
        Files.write(Paths.get(args[1]), sb.toString().getBytes());
        println("wrote " + args[1]);
    }
}
