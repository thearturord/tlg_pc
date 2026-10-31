// Applies a ghidra_names.tsv written by ExportNames.java to the program: names, function prototypes (name and comment only), comments. Addresses are absolute (same image base).
//@category TLG
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ImportNames extends GhidraScript {
    private static String unesc(String s) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(++i);
                b.append(n == 'n' ? '\n' : n == 't' ? '\t' : n);
            } else b.append(c);
        }
        return b.toString();
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String path = args.length > 0 ? args[0] : "/tmp/claude-0/game/ghidra_names.tsv";
        int named = 0, comments = 0, skipped = 0;
        for (String line : Files.readAllLines(Paths.get(path))) {
            if (line.startsWith("#") || line.isEmpty()) continue;
            String[] p = line.split("\t", -1);
            try {
                Address a = toAddr(Long.decode(p[1]));
                if (p[0].equals("FUNC")) {
                    Function f = getFunctionAt(a);
                    if (f == null) { createFunction(a, null); f = getFunctionAt(a); }
                    if (f != null) { String n = unesc(p[2]); n = n.substring(n.lastIndexOf("::") + 2 < 2 ? 0 : n.lastIndexOf("::") + 2); f.setName(n, SourceType.USER_DEFINED); named++; }
                    else skipped++;
                } else if (p[0].equals("SYM")) {
                    String n = unesc(p[3]); n = n.contains("::") ? n.substring(n.lastIndexOf("::") + 2) : n;
                    if (p[2].equals("L")) { createLabel(a, n, true); named++; }
                } else if (p[0].equals("COMMENT")) {
                    int kind = p[2].equals("PLATE") ? CodeUnit.PLATE_COMMENT : p[2].equals("PRE") ? CodeUnit.PRE_COMMENT : p[2].equals("EOL") ? CodeUnit.EOL_COMMENT
                             : p[2].equals("POST") ? CodeUnit.POST_COMMENT : CodeUnit.REPEATABLE_COMMENT;
                    currentProgram.getListing().setComment(a, kind, unesc(p[3]));
                    comments++;
                }
            } catch (Exception e) { skipped++; }
        }
        println("applied: " + named + " names, " + comments + " comments, skipped " + skipped);
    }
}
