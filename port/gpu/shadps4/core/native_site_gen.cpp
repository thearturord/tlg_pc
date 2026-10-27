// SPDX-License-Identifier: GPL-2.0-or-later
// B7: see native_site_gen.h.

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "core/guest_hooks.h"
#include "core/native_draw.h"
#include "core/native_site_gen.h"

namespace Core::NativeSiteGen {

namespace {

using GuestHooks::MidContext;
using Virtual = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);

template <class T>
T R(u64 address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}
u64 Global(u64 ghidra) {
    return NativeDraw::ImageBase() + (ghidra - 0x1000000);
}
bool Plausible(u64 pointer) {
    return pointer > 0x100000000ull && pointer < (1ull << 47);
}

struct Config {
    bool on{true};
    u32 verify_every{256};
};
const Config& Cfg() {
    static const Config config = [] {
        Config c;
        const char* e = std::getenv("TLG_SITE_GEN");
        c.on = e == nullptr || *e != '0'; // default on since 2026-10-27 (all native switches); TLG_SITE_GEN=0 turns the site generators off
        if (const char* n = std::getenv("TLG_SITE_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

enum Reg : int { RAX, RCX, RDX, RBX, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
u64& RegRef(MidContext* c, int reg) {
    switch (reg) {
    case RAX: return c->rax;
    case RCX: return c->rcx;
    case RDX: return c->rdx;
    case RBX: return c->rbx;
    case RSI: return c->rsi;
    case RDI: return c->rdi;
    case R8: return c->r8;
    case R9: return c->r9;
    case R10: return c->r10;
    case R11: return c->r11;
    case R12: return c->r12;
    case R13: return c->r13;
    case R14: return c->r14;
    default: return c->r15;
    }
}

/// A write the piece makes into memory: recorded (verify) or done (generation).
struct Patch {
    u64 address;
    std::vector<u8> bytes;
};
class Writer {
public:
    explicit Writer(bool live) : live_{live} {}
    void Put(u64 address, const void* data, size_t size) {
        if (live_) {
            std::memcpy(reinterpret_cast<void*>(address), data, size);
            return;
        }
        for (auto& patch : patches) { // a later write to the same range replaces the earlier one
            if (patch.address == address && patch.bytes.size() == size) {
                std::memcpy(patch.bytes.data(), data, size);
                return;
            }
        }
        patches.push_back({address, std::vector<u8>(static_cast<const u8*>(data), static_cast<const u8*>(data) + size)});
    }
    template <class T>
    void Put(u64 address, T value) {
        Put(address, &value, sizeof(T));
    }
    [[nodiscard]] bool Live() const {
        return live_;
    }
    std::vector<Patch> patches;

private:
    bool live_;
};

/// What one execution of a site would do.
struct Plan {
    u64 rsp{}, context{}, site_ghidra{};
    u32 groups{}, groups_y{1}, groups_z{1};
    bool async{true};
    std::vector<std::pair<u64, u64>> context_words; ///< (offset in the context, value): compared at the engine's dispatch
    std::vector<std::pair<int, u64>> registers;     ///< what the registers hold after the piece (set by the generator, compared by the verify)
    u64 v[24]{};                                    ///< values of the scan the apply needs
};

struct SiteStats {
    std::atomic<u64> executions{0}, generated{0}, declined{0}, verify_executions{0}, verify_clean{0}, verify_differences{0}, verify_raced{0}, scan_faults{0};
    std::atomic<bool> switched_off{false};
    std::mutex mutex;
    std::map<std::string, u64> decline_reasons, difference_kinds;
    std::vector<std::string> samples;
};

struct Site {
    const char* name;
    u64 resume_ghidra;
    bool (*scan)(MidContext*, Plan&, const char*&, bool&);
    void (*apply)(const Plan&, Writer&);
    SiteStats stats;
};

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 0: the first system loop of RenderDev_vfn18_BeginRecording, the tail of its body (ghidra 0x01c49d20 .. 0x01c49e40, 26 dispatches per frame). Top hook at 0x01c49d20
// (`mov eax,[r15 + 0x50]`, after the sum of the system's part sizes was added up into ebx), bottom hook at 0x01c49e40 (`mov dword [r15 + 0x2710],1`, after the dispatch call), which
// is also where the generator continues. Read from the disassembly: the holder `[rsp + 0xa8]` gets seven words from the system (`[r15 + 0x50]`, `[r12 + 0x54]` with r12 the class
// chain result or the system itself, the sum, `[rsp + 0x98]`, `[rsp + 0x70]`, the index r14d, `[rsp + 0x88]`), 64 bytes from `[rsp + 0xa0] + 0x70` go to the staging block
// `[rsp + 0x40]`, the constant buffer `[holder + 0x26d0]` is updated from `[rsp + 0x48]` (UpdateSubresource, context virtual 0xd0, context `[rsp + 0x50]`), three views are copied
// to the context's table at `+ 0x1d18` (`[r13 + 0x28]`, `[[rsp + 0x90] + 0x28]`, `[holder + 0x2700]`) and one dispatch of (sum + 63) / 64 groups is issued with a direct call.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan0(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    const u64 system = c->r15;
    const u64 holder = R<u64>(p.rsp + 0xa8);
    p.context = R<u64>(p.rsp + 0x50);
    const u64 view_source = c->r13;
    const u64 view_b_owner = R<u64>(p.rsp + 0x90);
    if (!Plausible(system) || !Plausible(holder) || !Plausible(p.context) || !Plausible(view_source) || !Plausible(view_b_owner)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 chain = c->r12 != 0 ? c->r12 : system;
    const u64 staging = R<u64>(p.rsp + 0x40);
    const u64 update_source = R<u64>(p.rsp + 0x48);
    const u64 copy_source = R<u64>(p.rsp + 0xa0);
    if (!Plausible(chain) || !Plausible(staging) || !Plausible(update_source) || !Plausible(copy_source)) {
        reason = "the staging pointers of the loop are not set";
        return false;
    }
    const u64 vtable = R<u64>(p.context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0))) {
        reason = "the context has no virtual table";
        return false;
    }
    p.v[0] = holder;
    p.v[1] = staging;
    p.v[2] = update_source;
    p.v[3] = copy_source;
    p.v[4] = R<u32>(system + 0x50);
    p.v[5] = R<u32>(chain + 0x54);
    p.v[6] = static_cast<u32>(c->rbx); // the sum of the part sizes
    p.v[7] = R<u32>(p.rsp + 0x98);
    p.v[8] = R<u32>(p.rsp + 0x70);
    p.v[9] = static_cast<u32>(c->r14);
    p.v[10] = R<u32>(p.rsp + 0x88);
    p.v[11] = R<u64>(holder + 0x26d0); // the constant buffer
    p.v[12] = R<u64>(view_source + 0x28);
    p.v[13] = R<u64>(view_b_owner + 0x28);
    p.v[14] = R<u64>(holder + 0x2700);
    p.groups = (static_cast<u32>(p.v[6]) + 63u) >> 6;
    p.site_ghidra = 0x01c49e40;
    p.async = true;
    for (int i = 0; i < 3; ++i) {
        p.context_words.emplace_back(0x1d18 + 8 * i, p.v[12 + i]);
    }
    p.registers = {{RBX, p.groups}, {R12, p.context}, {R15, holder}};
    return true;
}
void Apply0(const Plan& p, Writer& w) {
    const u64 holder = p.v[0];
    w.Put<u32>(holder + 0x2670, static_cast<u32>(p.v[4]));
    w.Put<u32>(holder + 0x2674, static_cast<u32>(p.v[5]));
    w.Put<u32>(holder + 0x2678, static_cast<u32>(p.v[6]));
    w.Put<u32>(holder + 0x2680, static_cast<u32>(p.v[7]));
    w.Put<u32>(holder + 0x2684, static_cast<u32>(p.v[8]));
    w.Put<u32>(holder + 0x2688, static_cast<u32>(p.v[9]));
    w.Put<u32>(holder + 0x268c, static_cast<u32>(p.v[10]));
    u8 block[64];
    std::memcpy(block, reinterpret_cast<const void*>(p.v[3] + 0x70), 32);
    std::memcpy(block + 32, reinterpret_cast<const void*>(p.v[3] + 0x90), 32);
    w.Put(p.v[1], block, 32);
    w.Put(p.v[1] + 0x20, block + 32, 32);
    w.Put<u32>(p.rsp, 0); // the seventh argument slot
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0xd0))(p.context, p.v[11], 0, 0, p.v[2], 0, 0); // UpdateSubresource
    }
    u64 views[3] = {p.v[12], p.v[13], p.v[14]};
    w.Put(p.rsp + 0xe0, views, 24); // the three views on the stack (the 16 bytes at +0xe0 were cleared first, +0xf0 is the third)
    w.Put(p.context + 0x1d18, views, 24);
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, true);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 1: the second system loop of RenderDev_vfn18_BeginRecording (ghidra 0x01c4a8a0 .. 0x01c4a9e3, 8 dispatches per frame). Same shape as site 0 with eight words at `holder + 0x2720 ..`
// (the holder is `[rsp + 0xa8]`, loaded into r13), r8b and r9 live in (r9 + 0x70 / 0x90 are the 64 bytes copied to `[rsp + 0x40]`), the constant buffer `[holder + 0x2780]`, and four views
// `[[rsp + 0x98] + 0x28]`, `[[rsp + 0x70] + 0x28]` (the first view again when `[rsp + 0x70]` is null), `[[rsp + 0x60] + 0x28]`, `[holder + 0x27b0]` copied through the stack at
// `[rsp + 0x140]` to the context's table at + 0x1d18. The dispatch call returns to 0x01c4a9e3.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan1(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    const u64 system = c->r15;
    const u64 holder = R<u64>(p.rsp + 0xa8);
    p.context = R<u64>(p.rsp + 0x50);
    const u64 view0_owner = R<u64>(p.rsp + 0x98);
    const u64 view1_owner = R<u64>(p.rsp + 0x70);
    const u64 view2_owner = R<u64>(p.rsp + 0x60);
    if (!Plausible(system) || !Plausible(holder) || !Plausible(p.context) || !Plausible(view0_owner) || !Plausible(view2_owner) || (view1_owner != 0 && !Plausible(view1_owner))) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 chain = c->r12 != 0 ? c->r12 : system;
    const u64 staging = R<u64>(p.rsp + 0x40);
    const u64 update_source = R<u64>(p.rsp + 0x48);
    const u64 copy_source = c->r9;
    if (!Plausible(chain) || !Plausible(staging) || !Plausible(update_source) || !Plausible(copy_source)) {
        reason = "the staging pointers of the loop are not set";
        return false;
    }
    const u64 vtable = R<u64>(p.context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0))) {
        reason = "the context has no virtual table";
        return false;
    }
    p.v[0] = holder;
    p.v[1] = staging;
    p.v[2] = update_source;
    p.v[3] = copy_source;
    p.v[4] = R<u32>(system + 0x50);
    p.v[5] = R<u32>(chain + 0x54);
    p.v[6] = static_cast<u32>(c->rbx);
    p.v[7] = c->r8 & 0xff;
    p.v[8] = R<u32>(p.rsp + 0xb8);
    p.v[9] = R<u32>(p.rsp + 0x58);
    p.v[10] = static_cast<u32>(c->r14);
    p.v[11] = R<u32>(p.rsp + 0xa0);
    p.v[12] = R<u64>(holder + 0x2780); // the constant buffer
    p.v[13] = R<u64>(view0_owner + 0x28);
    p.v[14] = view1_owner != 0 ? R<u64>(view1_owner + 0x28) : p.v[13];
    p.v[15] = R<u64>(view2_owner + 0x28);
    p.v[16] = R<u64>(holder + 0x27b0);
    p.v[17] = chain;
    p.groups = (static_cast<u32>(p.v[6]) + 63u) >> 6;
    p.site_ghidra = 0x01c4a9e3;
    p.async = true;
    for (int i = 0; i < 4; ++i) {
        p.context_words.emplace_back(0x1d18 + 8 * i, p.v[13 + i]);
    }
    p.registers = {{RBX, p.groups}, {R12, chain}, {R15, p.context}};
    return true;
}
void Apply1(const Plan& p, Writer& w) {
    const u64 holder = p.v[0];
    w.Put<u32>(holder + 0x2720, static_cast<u32>(p.v[4]));
    w.Put<u32>(holder + 0x2724, static_cast<u32>(p.v[5]));
    w.Put<u32>(holder + 0x2728, static_cast<u32>(p.v[6]));
    w.Put<u32>(holder + 0x272c, static_cast<u32>(p.v[7]));
    w.Put<u32>(holder + 0x2730, static_cast<u32>(p.v[8]));
    w.Put<u32>(holder + 0x2734, static_cast<u32>(p.v[9]));
    w.Put<u32>(holder + 0x2738, static_cast<u32>(p.v[10]));
    w.Put<u32>(holder + 0x273c, static_cast<u32>(p.v[11]));
    u8 block[64];
    std::memcpy(block, reinterpret_cast<const void*>(p.v[3] + 0x70), 32);
    std::memcpy(block + 32, reinterpret_cast<const void*>(p.v[3] + 0x90), 32);
    w.Put(p.v[1], block, 32);
    w.Put(p.v[1] + 0x20, block + 32, 32);
    w.Put<u32>(p.rsp, 0); // the seventh argument slot
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0xd0))(p.context, p.v[12], 0, 0, p.v[2], 0, 0); // UpdateSubresource
    }
    u64 views[4] = {p.v[13], p.v[14], p.v[15], p.v[16]};
    w.Put(p.rsp + 0x140, views, 32);
    w.Put(p.context + 0x1d18, views, 32);
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, true);
    }
}

std::array<Site, 2>& Sites() {
    static std::array<Site, 2> sites{{
        {"begin_recording_loop_1", 0x01c49e40, &Scan0, &Apply0, {}},
        {"begin_recording_loop_2", 0x01c4a9e3, &Scan1, &Apply1, {}},
    }};
    return sites;
}

void Decline(Site& site, const char* reason) {
    site.stats.declined.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{site.stats.mutex};
    ++site.stats.decline_reasons[reason];
}
void NoteDifference(Site& site, const std::string& kind, const std::string& text) {
    site.stats.verify_differences.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{site.stats.mutex};
    ++site.stats.difference_kinds[kind];
    if (site.stats.samples.size() < 12) {
        site.stats.samples.push_back(kind + ": " + text);
    }
}

struct ScanArgs {
    Site* site;
    MidContext* c;
    Plan* plan;
    const char* reason;
    bool no_work;
    bool ok;
};
void ScanThunk(void* p) {
    auto& a = *static_cast<ScanArgs*>(p);
    a.ok = a.site->scan(a.c, *a.plan, a.reason, a.no_work);
}
struct ApplyArgs {
    Site* site;
    const Plan* plan;
    Writer* writer;
};
void ApplyThunk(void* p) {
    auto& a = *static_cast<ApplyArgs*>(p);
    a.site->apply(*a.plan, *a.writer);
}
bool Scan(Site& site, MidContext* c, Plan& plan, const char*& reason, bool& no_work) {
    ScanArgs args{&site, c, &plan, "scan faulted", false, false};
    if (!NativeDraw::GuardedCall(&ScanThunk, &args)) {
        site.stats.scan_faults.fetch_add(1, std::memory_order_relaxed);
        reason = "scan faulted";
        return false;
    }
    reason = args.reason;
    no_work = args.no_work;
    return args.ok;
}

/// A verify execution: what the generator predicts, kept until the engine's code has run through the piece.
struct Pending {
    bool armed{false}, dispatch_seen{false};
    u32 site{0};
    MidContext saved{};
    Plan plan;
    std::vector<Patch> patches;
    std::vector<std::string> notes;
};
thread_local Pending pending;

template <u32 N>
u64 PS4_SYSV_ABI Top(MidContext* c) {
    const Config& cfg = Cfg();
    Site& site = Sites()[N];
    SiteStats& s = site.stats;
    if (!cfg.on || s.switched_off.load(std::memory_order_relaxed) || pending.armed) {
        return 0;
    }
    s.executions.fetch_add(1, std::memory_order_relaxed);
    Plan plan;
    const char* reason = "";
    bool no_work = false;
    if (!Scan(site, c, plan, reason, no_work)) {
        if (!no_work) {
            Decline(site, reason);
        }
        return 0;
    }
    const u64 eligible = s.generated.load(std::memory_order_relaxed) + s.verify_executions.load(std::memory_order_relaxed);
    if (cfg.verify_every != 0 && eligible % cfg.verify_every == cfg.verify_every - 1) {
        s.verify_executions.fetch_add(1, std::memory_order_relaxed);
        Writer shadow{false};
        ApplyArgs args{&site, &plan, &shadow};
        if (!NativeDraw::GuardedCall(&ApplyThunk, &args)) {
            s.scan_faults.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        pending.armed = true;
        pending.dispatch_seen = false;
        pending.site = N;
        pending.saved = *c;
        pending.plan = plan;
        pending.patches = std::move(shadow.patches);
        pending.notes.clear();
        return 0; // the engine runs this one
    }
    Writer live{true};
    ApplyArgs args{&site, &plan, &live};
    if (!NativeDraw::GuardedCall(&ApplyThunk, &args)) {
        s.switched_off.store(true); // the writes and calls before the fault are done and the engine cannot run the piece again: switch off
        NoteDifference(site, "fault while generating", site.name);
        return 0;
    }
    s.generated.fetch_add(1, std::memory_order_relaxed);
    for (const auto& [reg, value] : plan.registers) {
        RegRef(c, reg) = value;
    }
    c->rdi = plan.context;
    c->rsi = plan.groups;
    c->rdx = plan.groups_y;
    c->rcx = plan.groups_z;
    return Global(site.resume_ghidra);
}

template <u32 N>
u64 PS4_SYSV_ABI Bottom(MidContext* c) {
    if (!pending.armed || pending.site != N) {
        return 0;
    }
    Site& site = Sites()[N];
    SiteStats& s = site.stats;
    pending.armed = false;
    bool clean = true;
    const auto fail = [&](const char* kind, const std::string& text) {
        clean = false;
        NoteDifference(site, kind, text);
    };
    if (!pending.dispatch_seen) {
        fail("dispatch missing", "the engine issued no dispatch through the piece");
    }
    for (const auto& note : pending.notes) {
        fail("dispatch", note);
    }
    const auto count_differences = [](const std::vector<Patch>& patches, std::string* first) {
        u32 differences = 0;
        for (const auto& patch : patches) {
            if (std::memcmp(reinterpret_cast<const void*>(patch.address), patch.bytes.data(), patch.bytes.size()) != 0) {
                if (differences++ == 0 && first != nullptr) {
                    std::string text = fmt::format("{:#x} ({} bytes: expected", patch.address, patch.bytes.size());
                    for (size_t i = 0; i < patch.bytes.size() && i < 8; ++i) {
                        text += fmt::format(" {:02x}", patch.bytes[i]);
                    }
                    text += ", found";
                    for (size_t i = 0; i < patch.bytes.size() && i < 8; ++i) {
                        text += fmt::format(" {:02x}", R<u8>(patch.address + i));
                    }
                    *first = text + ")";
                }
            }
        }
        return differences;
    };
    std::string first;
    if (const u32 differences = count_differences(pending.patches, &first); differences != 0) {
        // read again with a fresh prediction: a field the engine's other threads moved between the two reads is a race, not a difference
        bool equal = false;
        Plan again;
        const char* reason = "";
        bool no_work = false;
        MidContext copy = pending.saved;
        if (Scan(site, &copy, again, reason, no_work)) {
            Writer shadow{false};
            ApplyArgs args{&site, &again, &shadow};
            if (NativeDraw::GuardedCall(&ApplyThunk, &args)) {
                equal = count_differences(shadow.patches, nullptr) == 0;
            }
        }
        if (equal) {
            s.verify_raced.fetch_add(1, std::memory_order_relaxed);
        } else {
            fail("memory", fmt::format("{} of {} written ranges differ: first {}", differences, pending.patches.size(), first));
        }
    }
    for (const auto& [reg, value] : pending.plan.registers) {
        if (RegRef(c, reg) != value) {
            fail("registers", fmt::format("register {} holds {:#x}, expected {:#x}", reg, RegRef(c, reg), value));
        }
    }
    if (clean) {
        s.verify_clean.fetch_add(1, std::memory_order_relaxed);
    } else if (s.verify_differences.load() >= 8) {
        s.switched_off.store(true);
    }
    pending.patches.clear();
    return 0;
}

} // namespace

void NoteDispatch(u64 context, u64 x, u64 y, u64 z, u64 site_runtime) {
    if (!pending.armed || pending.dispatch_seen || context != pending.plan.context) {
        return;
    }
    const Plan& plan = pending.plan;
    pending.dispatch_seen = true;
    if (x != plan.groups || y != plan.groups_y || z != plan.groups_z) {
        pending.notes.push_back(fmt::format("groups {} x {} x {}, expected {} x {} x {}", x, y, z, plan.groups, plan.groups_y, plan.groups_z));
    }
    if (site_runtime != Global(plan.site_ghidra)) {
        pending.notes.push_back(fmt::format("call site {:#x}, expected {:#x}", site_runtime, Global(plan.site_ghidra)));
    }
    for (const auto& [offset, value] : plan.context_words) {
        if (R<u64>(context + offset) != value) {
            pending.notes.push_back(fmt::format("context word {:#x}: {:#x}, expected {:#x}", offset, R<u64>(context + offset), value));
        }
    }
}

u64 FindHandler(const char* name) {
    const std::string n{name};
    if (n == "site_top_0") {
        return reinterpret_cast<u64>(&Top<0>);
    }
    if (n == "site_bottom_0") {
        return reinterpret_cast<u64>(&Bottom<0>);
    }
    if (n == "site_top_1") {
        return reinterpret_cast<u64>(&Top<1>);
    }
    if (n == "site_bottom_1") {
        return reinterpret_cast<u64>(&Bottom<1>);
    }
    return 0;
}

nlohmann::json Report() {
    nlohmann::json out;
    out["enabled"] = Cfg().on;
    out["verify_every"] = Cfg().verify_every;
    nlohmann::json list = nlohmann::json::array();
    for (auto& site : Sites()) {
        SiteStats& s = site.stats;
        nlohmann::json e;
        e["site"] = site.name;
        e["switched_off"] = s.switched_off.load();
        e["executions_seen"] = s.executions.load();
        e["generated"] = s.generated.load();
        e["declined"] = s.declined.load();
        e["verify_executions"] = s.verify_executions.load();
        e["verify_clean"] = s.verify_clean.load();
        e["verify_with_differences"] = s.verify_differences.load();
        e["verify_raced_memory"] = s.verify_raced.load();
        e["scan_faults"] = s.scan_faults.load();
        std::scoped_lock lock{s.mutex};
        e["decline_reasons"] = s.decline_reasons;
        e["difference_kinds"] = s.difference_kinds;
        e["difference_samples"] = s.samples;
        list.push_back(std::move(e));
    }
    out["sites"] = std::move(list);
    return out;
}

} // namespace Core::NativeSiteGen
