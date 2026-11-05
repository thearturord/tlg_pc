/* Function coverage of the guest code (measurement only, nothing else changes). TLG_COVERAGE=<file>: every line starts with the Ghidra
 * address (8 hex digits) of a function entry (functions.tsv of documents/tlg/tools/ghidra/FrameMapExport.java works as it is). The first byte
 * of each such function becomes int3; the first call per window restores it (one trap, a few microseconds) and is recorded; a thread
 * re-arms the functions hit in the last window every TLG_COVERAGE_WINDOW_MS (default 250). So the table says, per function, in how many
 * windows it ran and on which threads. Documents: engine/frame-map-100-plan.md, check 4 (dynamic coverage).
 *   TLG_COVERAGE_TAG            name part of the output file (default "run")
 *   TLG_COVERAGE_START_S        seconds after start before arming (default 0)
 *   TLG_COVERAGE_OUT            output directory (default "user/log")
 * Output coverage_<tag>.tsv, rewritten every 5 s: header lines start with '#', then "ghidra_addr windows_hit first_window tid tid tid tid"
 * (tids in hex, 0 = none), then "T tid name" lines. Windows only; elsewhere init does nothing. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

#define COV_GHIDRA_BASE 0x1000000ull
#define COV_TIDS 4

typedef struct {
    uintptr_t address;
    uint32_t ghidra;
    uint8_t original;
    volatile LONG hit_now;   /* hit in the current window, to be re-armed */
    volatile LONG windows;   /* windows in which it ran */
    volatile LONG first_window;
    volatile LONG tids[COV_TIDS];
} CovEntry;

static CovEntry *entries;
static size_t entry_count;
static volatile LONG window_index;
static char out_path[600];
static unsigned window_ms = 250;
static unsigned start_s;

static int compare_entry(const void *a, const void *b) {
    uintptr_t x = ((const CovEntry *)a)->address, y = ((const CovEntry *)b)->address;
    return x < y ? -1 : x > y;
}

int runtime_cov_trap(uintptr_t address, uintptr_t *resume) {
    if (!entry_count) return 0;
    size_t lo = 0, hi = entry_count;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (entries[mid].address < address) lo = mid + 1; else hi = mid;
    }
    if (lo >= entry_count || entries[lo].address != address) return 0;
    CovEntry *e = &entries[lo];
    /* restore first: another thread arriving at the same moment also traps and lands here */
    *(volatile uint8_t *)address = e->original;
    if (InterlockedExchange(&e->hit_now, 1) == 0) {
        InterlockedIncrement(&e->windows);
        InterlockedCompareExchange(&e->first_window, window_index + 1, 0);
    }
    LONG tid = (LONG)GetCurrentThreadId();
    for (int i = 0; i < COV_TIDS; ++i) {
        LONG seen = e->tids[i];
        if (seen == tid) break;
        if (seen == 0 && InterlockedCompareExchange(&e->tids[i], tid, 0) == 0) break;
        if (e->tids[i] == tid) break;
    }
    *resume = address;
    return 1;
}

static void dump(void) {
    FILE *f = fopen(out_path, "wb");
    if (!f) return;
    size_t hit = 0;
    for (size_t i = 0; i < entry_count; ++i) if (entries[i].windows) ++hit;
    fprintf(f, "# windows %ld window_ms %u functions %zu hit %zu\n", (long)window_index, window_ms, entry_count, hit);
    LONG seen_tids[256];
    int seen_count = 0;
    for (size_t i = 0; i < entry_count; ++i) {
        CovEntry *e = &entries[i];
        if (!e->windows) continue;
        fprintf(f, "%08x\t%ld\t%ld", e->ghidra, (long)e->windows, (long)e->first_window);
        for (int k = 0; k < COV_TIDS; ++k) {
            LONG t = e->tids[k];
            fprintf(f, "\t%lx", (unsigned long)t);
            if (t) {
                int known = 0;
                for (int s = 0; s < seen_count; ++s) if (seen_tids[s] == t) known = 1;
                if (!known && seen_count < 256) seen_tids[seen_count++] = t;
            }
        }
        fputc('\n', f);
    }
    typedef HRESULT (WINAPI *GetDesc)(HANDLE, PWSTR *);
    GetDesc get_desc = (GetDesc)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
    for (int s = 0; s < seen_count; ++s) {
        char name[128] = "";
        HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)seen_tids[s]);
        if (h && get_desc) {
            PWSTR w = NULL;
            if (SUCCEEDED(get_desc(h, &w)) && w) {
                WideCharToMultiByte(CP_UTF8, 0, w, -1, name, sizeof(name) - 1, NULL, NULL);
                LocalFree(w);
            }
        }
        if (h) CloseHandle(h);
        fprintf(f, "T\t%lx\t%s\n", (unsigned long)seen_tids[s], name);
    }
    fclose(f);
}

static void arm_all(int only_hit) {
    for (size_t i = 0; i < entry_count; ++i) {
        CovEntry *e = &entries[i];
        if (only_hit && !e->hit_now) continue;
        e->hit_now = 0;
        *(volatile uint8_t *)e->address = 0xCC;
    }
}

static DWORD WINAPI cov_thread(LPVOID unused) {
    (void)unused;
    unsigned long long begin = GetTickCount64();
    while (GetTickCount64() - begin < (unsigned long long)start_s * 1000ull) Sleep(100);
    arm_all(0);
    unsigned long long last_dump = GetTickCount64();
    for (;;) {
        Sleep(window_ms);
        InterlockedIncrement(&window_index);
        arm_all(1);
        if (GetTickCount64() - last_dump > 5000) {
            dump();
            last_dump = GetTickCount64();
        }
    }
    return 0;
}

void runtime_cov_init(unsigned char *image, size_t image_size) {
    const char *list = getenv("TLG_COVERAGE");
    if (!list || !*list) return;
    FILE *f = fopen(list, "rb");
    if (!f) { fprintf(stderr, "TLG_COVERAGE: cannot open %s\n", list); return; }
    size_t capacity = 32768;
    entries = calloc(capacity, sizeof(*entries));
    char line[512];
    size_t skipped = 0;
    while (entries && fgets(line, sizeof(line), f)) {
        unsigned long long g = strtoull(line, NULL, 16);
        if (g < COV_GHIDRA_BASE || g - COV_GHIDRA_BASE >= image_size) continue;
        uintptr_t a = (uintptr_t)image + (size_t)(g - COV_GHIDRA_BASE);
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((void *)a, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            !(mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY))) { ++skipped; continue; }
        uint8_t first = *(uint8_t *)a;
        if (first == 0xCC) continue; /* already an emulation trap */
        if (entry_count == capacity) { capacity *= 2; entries = realloc(entries, capacity * sizeof(*entries)); if (!entries) break; }
        entries[entry_count].address = a;
        entries[entry_count].ghidra = (uint32_t)g;
        entries[entry_count].original = first;
        ++entry_count;
    }
    fclose(f);
    if (!entries || !entry_count) { entry_count = 0; return; }
    qsort(entries, entry_count, sizeof(*entries), compare_entry);
    /* the code pages must be writable for the one byte patch */
    uintptr_t page = 0;
    for (size_t i = 0; i < entry_count; ++i) {
        uintptr_t p = entries[i].address & ~(uintptr_t)0xFFF;
        if (p == page) continue;
        page = p;
        DWORD old;
        VirtualProtect((void *)p, 0x1000, PAGE_EXECUTE_READWRITE, &old);
    }
    const char *tag = getenv("TLG_COVERAGE_TAG");
    const char *dir = getenv("TLG_COVERAGE_OUT");
    snprintf(out_path, sizeof(out_path), "%s/coverage_%s.tsv", dir && *dir ? dir : "user/log", tag && *tag ? tag : "run");
    if (getenv("TLG_COVERAGE_WINDOW_MS") && atoi(getenv("TLG_COVERAGE_WINDOW_MS")) > 0) window_ms = (unsigned)atoi(getenv("TLG_COVERAGE_WINDOW_MS"));
    if (getenv("TLG_COVERAGE_START_S")) start_s = (unsigned)atoi(getenv("TLG_COVERAGE_START_S"));
    printf("Coverage: %zu function entries (%zu not in executable memory), window %u ms, start after %u s -> %s\n", entry_count, skipped, window_ms, start_s, out_path);
    CreateThread(NULL, 0, cov_thread, NULL, 0, NULL);
}

#else
#include <stdint.h>
void runtime_cov_init(unsigned char *image, size_t image_size) { (void)image; (void)image_size; }
int runtime_cov_trap(uintptr_t address, uintptr_t *resume) { (void)address; (void)resume; return 0; }
#endif
