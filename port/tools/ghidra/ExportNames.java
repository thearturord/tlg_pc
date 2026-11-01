// Exports what you added to a Ghidra program as one small text file: function and label names, comments, function prototypes, data types applied to data,
// bookmarks. Use: Window > Script Manager > "Script Directories" (the list icon) > add this folder > run ExportNames.java. The file is written to
// the folder you pick in the dialog (name: ghidra_names.tsv; zip it, it is usually well under a megabyte). Read by ImportNames.java.
//@category TLG
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.data.*;
import java.io.*;
import java.util.*;

public class ExportNames extends GhidraScript {
    private static String esc(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "").replace("\n", "\\n");
    }

    @Override
    public void run() throws Exception {
        File dir = askDirectory("Folder for ghidra_names.tsv", "Select");
        File out = new File(dir, "ghidra_names.tsv");
        long base = currentProgram.getImageBase().getOffset();
        try (PrintWriter w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(out), "UTF-8"))) {
            w.println("#TLG_NAMES\t1\timage_base=0x" + Long.toHexString(base));
            int syms = 0, funcs = 0, comments = 0, datas = 0, marks = 0;
            // symbols the user (or an analyzer other than the default naming) gave a name
            SymbolIterator it = currentProgram.getSymbolTable().getAllSymbols(true);
            while (it.hasNext() && !monitor.isCancelled()) {
                Symbol s = it.next();
                if (s.getSource() == SourceType.DEFAULT) continue;
                SymbolType t = s.getSymbolType();
                if (t != SymbolType.FUNCTION && t != SymbolType.LABEL) continue;
                if (!s.getAddress().isMemoryAddress()) continue;
                w.println("SYM\t0x" + Long.toHexString(s.getAddress().getOffset()) + "\t" + (t == SymbolType.FUNCTION ? "F" : "L") + "\t" + esc(s.getName(true)) + "\t" + s.getSource());
                syms++;
            }
            // functions: prototype, calling convention, plate comment, tags
            FunctionIterator fi = currentProgram.getFunctionManager().getFunctions(true);
            while (fi.hasNext() && !monitor.isCancelled()) {
                Function f = fi.next();
                boolean named = f.getSymbol().getSource() != SourceType.DEFAULT;
                boolean typed = f.getSignatureSource() != SourceType.DEFAULT;
                if (!named && !typed) continue;
                w.println("FUNC\t0x" + Long.toHexString(f.getEntryPoint().getOffset()) + "\t" + esc(f.getName(true)) + "\t" + esc(f.getPrototypeString(true, true)) + "\t" + esc(f.getCallingConventionName())
                          + "\t" + (f.hasCustomVariableStorage() ? "custom" : "auto") + "\t" + f.getBody().getNumAddresses());
                funcs++;
            }
            // comments
            int[] kinds = {CodeUnit.PLATE_COMMENT, CodeUnit.PRE_COMMENT, CodeUnit.EOL_COMMENT, CodeUnit.POST_COMMENT, CodeUnit.REPEATABLE_COMMENT};
            String[] names = {"PLATE", "PRE", "EOL", "POST", "REPEAT"};
            AddressIterator ai = currentProgram.getListing().getCommentAddressIterator(currentProgram.getMemory(), true);
            while (ai.hasNext() && !monitor.isCancelled()) {
                Address a = ai.next();
                for (int k = 0; k < kinds.length; k++) {
                    String c = currentProgram.getListing().getComment(kinds[k], a);
                    if (c != null && !c.isEmpty()) {
                        w.println("COMMENT\t0x" + Long.toHexString(a.getOffset()) + "\t" + names[k] + "\t" + esc(c));
                        comments++;
                    }
                }
            }
            // data with a type the user applied (not the undefined bytes)
            DataIterator di = currentProgram.getListing().getDefinedData(true);
            while (di.hasNext() && !monitor.isCancelled()) {
                Data d = di.next();
                DataType dt = d.getDataType();
                if (dt instanceof Undefined || dt instanceof DefaultDataType) continue;
                if (dt instanceof Pointer || dt instanceof BuiltInDataType) {
                    if (d.getSymbols().length == 0) continue; // skip the thousands of automatic pointers and integers that have no name
                }
                w.println("DATA\t0x" + Long.toHexString(d.getAddress().getOffset()) + "\t" + esc(dt.getName()) + "\t" + d.getLength() + "\t" + esc(dt.getCategoryPath().toString()));
                datas++;
            }
            // struct and enum definitions the user made (name, size, fields), so the data lines make sense
            Iterator<DataType> dts = currentProgram.getDataTypeManager().getAllDataTypes();
            int types = 0;
            while (dts.hasNext() && !monitor.isCancelled()) {
                DataType dt = dts.next();
                if (!(dt instanceof Structure) || dt.getCategoryPath().toString().startsWith("/Demangler")) continue;
                Structure st = (Structure) dt;
                StringBuilder sb = new StringBuilder();
                for (DataTypeComponent c : st.getDefinedComponents()) {
                    sb.append(c.getOffset()).append(':').append(c.getDataType().getName()).append(':').append(c.getFieldName() == null ? "" : c.getFieldName()).append(';');
                }
                w.println("STRUCT\t" + esc(st.getPathName()) + "\t" + st.getLength() + "\t" + esc(sb.toString()));
                types++;
            }
            // bookmarks
            Iterator<Bookmark> bi = currentProgram.getBookmarkManager().getBookmarksIterator();
            while (bi.hasNext()) {
                Bookmark b = bi.next();
                if (!"Note".equals(b.getTypeString())) continue; // analysis errors and warnings are not wanted
                w.println("BOOKMARK\t0x" + Long.toHexString(b.getAddress().getOffset()) + "\t" + esc(b.getTypeString()) + "\t" + esc(b.getCategory()) + "\t" + esc(b.getComment()));
                marks++;
            }
            println("exported: " + syms + " symbols, " + funcs + " functions, " + comments + " comments, " + datas + " data items, " + types + " structs, " + marks + " bookmarks -> " + out);
        }
    }
}
