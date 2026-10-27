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

/// One thing a straight line piece does (the step based sites build a list of them in the scan and `ApplySteps` does them in the engine's order: stack slot writes, the virtuals of
/// the context, the dispatch).
struct Step {
    enum Kind : u8 { PutStack64, PutStack32, PutAbs64, PutAbs32, PutAbs8, PutAbsZero, Update, SetCb, SetUav, SetSrv, SetShader, Dispatch } kind;
    u64 a{}, b{}, c{}, d{}, e{};
    // PutStack*: a = offset from rsp, b = value. PutAbs*: a = address, b = value. Update: a = constant buffer, b = source (the seventh argument slot at [rsp] is zeroed).
    // SetCb / SetUav / SetSrv: a = first slot, b = count, c = offset from rsp of the array. SetShader: a = shader. Dispatch: a, b, c = groups, d = call site (ghidra), e = fenced.
};

/// One of several dispatches a site issues: its groups, the call site (the return address in ghidra terms) and the words of the context's tables at that dispatch.
struct Expect {
    u32 x{}, y{1}, z{1};
    u64 site{};
    std::vector<std::pair<u64, u64>> words; ///< (offset in the context, value)
    std::vector<std::pair<u64, u64>> stack; ///< (offset from the function's rsp, value): what the engine's virtuals copy into the context's tables
};

/// What one execution of a site would do.
struct Plan {
    u64 rsp{}, context{}, site_ghidra{};
    u32 groups{}, groups_y{1}, groups_z{1};
    bool async{true};
    std::vector<std::pair<u64, u64>> context_words; ///< (offset in the context, value): compared at the engine's dispatch
    std::vector<std::pair<int, u64>> registers;     ///< what the registers hold after the piece (set by the generator, compared by the verify)
    std::vector<struct Step> steps;                 ///< step based sites: what the piece does, in order (see ApplySteps)
    std::vector<Expect> dispatches;                 ///< sites that issue several dispatches: each one in order (replaces groups / site_ghidra / context_words for the comparison)
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

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Sites 2 and 3: the vertex cluster `FUN_01d7d260` (B6), its two loops over the objects of the system `[holder + 0x110]` (count `[holder + 0x118]`), graphics context. Both build the
// views on the stack and call SetConstantBuffers (0x28: slot 0, two entries from `[rsp + 0x90]`), SetUnorderedAccessViews (0x48: one entry from `[rsp + 0x58]`),
// SetShaderResources (0x40: six entries from `[rsp + 0x60]`) and dispatch (count + 63) / 64 groups with context virtual 0x58. Loop 1 (0x01d7e240 .. 0x01d7e334, objects with `[obj + 0xa8]`
// elements, the shader set before the loop) and loop 2 (0x01d7e3b0 .. 0x01d7e5a9, `[obj + 0xc8]` elements, the shader chosen per object by the system's flag bits 0x20 / 0x10 from the
// device's table at `+ 0x1008 + 0x10 * index`, 0x66 .. 0x68, and set only when it differs from the one cached at `[rsp + 0x30]`).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan2(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 holder = c->r12;
    p.context = c->r15;
    const u32 index = static_cast<u32>(c->rbx);
    if (!Plausible(holder) || !Plausible(p.context)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 array = R<u64>(holder + 0x110);
    const u32 count_objects = R<u32>(holder + 0x118);
    if (!Plausible(array) || index >= count_objects) {
        reason = "no object array";
        return false;
    }
    const u64 element = R<u64>(array + u64{index} * 8);
    if (!Plausible(element)) {
        reason = "the object is not set";
        return false;
    }
    const u32 n = R<u32>(element + 0xa8);
    if (n == 0 || (R<u8>(element + 0x54) & 4) != 0) {
        no_work = true; // the loop skips the object
        return false;
    }
    const u64 inner = R<u64>(element + 0x100);
    if (!Plausible(inner)) {
        reason = "the object has no render data";
        return false;
    }
    const u64 chain_holder = R<u64>(holder + 0xb0);
    const u64 chain = Plausible(chain_holder) ? R<u64>(chain_holder) : 0;
    const u64 srv0 = chain != 0 ? R<u64>(chain + 0x28) : 0;
    const u64 cb_view = R<u64>(inner + 0x90);
    const u64 uav_owner = R<u64>(inner + 0x10);
    if (uav_owner == 0) {
        no_work = true; // the loop skips the object (no dispatch, no table write)
        return false;
    }
    const u64 uav = R<u64>(uav_owner + 0x30);
    if (uav == 0 || cb_view == 0) {
        no_work = true;
        return false;
    }
    if (!Plausible(R<u64>(p.context)) || !Plausible(R<u64>(R<u64>(p.context) + 0x58))) {
        reason = "the context has no virtual table";
        return false;
    }
    p.v[0] = cb_view;
    p.v[1] = srv0;
    p.v[2] = uav;
    p.v[3] = element;
    for (int i = 0; i < 5; ++i) {
        p.v[4 + i] = R<u64>(p.rsp + 0x68 + 8 * i); // the other five view slots, as the loop left them before this object
    }
    p.groups = (n + 63u) >> 6;
    p.site_ghidra = 0x01d7e334;
    p.async = false;
    p.context_words.emplace_back(0x500, 0);
    p.context_words.emplace_back(0x508, cb_view);
    p.context_words.emplace_back(0x1980, srv0);
    for (int i = 0; i < 5; ++i) {
        p.context_words.emplace_back(0x1980 + 8 + 8 * i, p.v[4 + i]);
    }
    p.context_words.emplace_back(0x2098, uav);
    p.registers = {{R13, element}};
    return true;
}
void Apply2(const Plan& p, Writer& w) {
    w.Put<u64>(p.rsp + 0x90, 0);
    w.Put<u64>(p.rsp + 0x98, p.v[0]);
    w.Put<u64>(p.rsp + 0x60, p.v[1]);
    w.Put<u64>(p.rsp + 0x58, p.v[2]);
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 0, 2, p.rsp + 0x90, 0, 0, 0); // SetConstantBuffers
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 1, p.rsp + 0x58, 0, 0, 0); // SetUnorderedAccessViews
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 6, p.rsp + 0x60, 0, 0, 0); // SetShaderResources
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, false);
    }
}

bool Scan3(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 holder = c->r15; // the system (loop 2 keeps it in r15 and the context in r14)
    p.context = c->r14;
    const u32 index = static_cast<u32>(c->rbx);
    if (!Plausible(holder) || !Plausible(p.context)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 array = R<u64>(holder + 0x110);
    const u32 count_objects = R<u32>(holder + 0x118);
    if (!Plausible(array) || index >= count_objects) {
        reason = "no object array";
        return false;
    }
    const u64 element = R<u64>(array + u64{index} * 8);
    if (!Plausible(element)) {
        reason = "the object is not set";
        return false;
    }
    const u32 n = R<u32>(element + 0xc8);
    if (n == 0 || (R<u8>(element + 0x54) & 4) != 0) {
        no_work = true;
        return false;
    }
    const u64 inner = R<u64>(element + 0x100);
    if (!Plausible(inner)) {
        reason = "the object has no render data";
        return false;
    }
    const auto field = [&](u64 owner_pointer, u64 offset) -> u64 { return owner_pointer != 0 ? R<u64>(owner_pointer + offset) : 0; };
    const u64 chain_holder = R<u64>(holder + 0xb0);
    const u64 chain = Plausible(chain_holder) ? R<u64>(chain_holder) : 0;
    p.v[0] = R<u64>(inner + 0xa8);              // the constant buffer view ([rsp + 0x98])
    p.v[1] = field(chain, 0x28);                // srv 0
    p.v[2] = field(R<u64>(inner + 0x10), 0x28); // srv 1
    p.v[3] = field(R<u64>(inner + 0x08), 0x28); // srv 2
    p.v[4] = field(R<u64>(inner + 0x20), 0x28); // srv 3
    const u64 device = R<u64>(Global(0x0689eed8));
    const u64 device_entry = Plausible(device) ? R<u64>(element + 0x108) : 0; // (in loop 2 r12 is the object, r15 the system)
    u64 srv4 = 0;
    if (Plausible(device_entry)) {
        const u64 key = R<u64>(device_entry + 0x50);
        const u64 found = reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64)>(Global(0x01c59060))(device, key);
        srv4 = Plausible(found) ? R<u64>(found + 0x48) : 0;
    }
    p.v[5] = srv4; // srv 4 ([rsp + 0x80])
    const u64 uav_owner = R<u64>(inner + 0x18);
    const u64 uav = uav_owner != 0 ? R<u64>(uav_owner + 0x30) : 0;
    p.v[6] = uav;
    if (uav == 0 || p.v[0] == 0 || p.v[2] == 0 || p.v[3] == 0 || p.v[4] == 0 || srv4 == 0) {
        no_work = true; // the loop skips the object
        return false;
    }
    u32 flags = R<u32>(element + 0x54);
    u32 shader_index = 0x66;
    u64 srv1 = p.v[2];
    if ((flags & 0x20) != 0) {
        srv1 = 0;
        shader_index = 0x67;
        flags = R<u32>(element + 0x54);
    }
    if ((flags & 0x10) != 0) {
        shader_index = 0x68;
    }
    p.v[2] = srv1;
    p.v[7] = shader_index;
    p.v[8] = R<u32>(p.rsp + 0x30); // the shader index cached by the loop
    p.v[9] = Plausible(device) ? R<u64>(device + 0x1008 + (u64{shader_index} << 4)) : 0;
    p.v[10] = element;
    p.v[11] = R<u64>(p.rsp + 0x88); // the sixth view slot, not touched by the loop
    if (p.v[9] == 0 || !Plausible(R<u64>(p.context)) || !Plausible(R<u64>(R<u64>(p.context) + 0x58))) {
        reason = "the shader table or the context is not set";
        return false;
    }
    p.groups = (n + 63u) >> 6;
    p.site_ghidra = 0x01d7e5a9;
    p.async = false;
    p.context_words.emplace_back(0x500, 0);
    p.context_words.emplace_back(0x508, p.v[0]);
    for (int i = 0; i < 5; ++i) {
        p.context_words.emplace_back(0x1980 + 8 * i, p.v[1 + i]);
    }
    p.context_words.emplace_back(0x1980 + 40, p.v[11]);
    p.context_words.emplace_back(0x2098, uav);
    p.context_words.emplace_back(0x4670, p.v[9]);
    p.registers = {{R12, element}, {R13, shader_index}};
    return true;
}
void Apply3(const Plan& p, Writer& w) {
    w.Put<u64>(p.rsp + 0x90, 0);
    w.Put<u64>(p.rsp + 0x98, p.v[0]);
    for (int i = 0; i < 5; ++i) {
        w.Put<u64>(p.rsp + 0x60 + 8 * i, p.v[1 + i]);
    }
    w.Put<u64>(p.rsp + 0x58, p.v[6]);
    const bool set_shader = static_cast<u32>(p.v[8]) != static_cast<u32>(p.v[7]);
    if (w.Live() && set_shader) {
        reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0x38))(p.context, p.v[9], 0, 0, 0, 0, 0); // SetShader
    }
    if (set_shader) {
        w.Put<u32>(p.rsp + 0x30, static_cast<u32>(p.v[7]));
    }
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 0, 2, p.rsp + 0x90, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 1, p.rsp + 0x58, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 6, p.rsp + 0x60, 0, 0, 0);
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, false);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 4: the particle emitter system `FUN_01bcac40(system, context, scene)` (ghidra 0x01bcac40), its last dispatch loop over the linked list of emitters (`node + 0x38`), 0x01bcc264 ..
// 0x01bcc3fc, graphics context in r14, the system in `[rsp + 0x80]`, the emitter data in r12 (`[node + 0x108]`): five stack slots are set from it (`[rsp + 0xb8] = [r12 + 0x190]`,
// `[rsp + 0xc0]` and `[rsp + 0xc8]`, `[rsp + 0xf0]`, `[rsp + 0xf8]`), the shader (0x1e, 0x1f or 0x20 by `[r12 + 0x248]` and `[[r12 + 0x218] + 0x48]`) is set when it differs from the one cached
// at `[system + 0x78]`, the tables are written through SetConstantBuffers (slot 10, one entry from `[rsp + 0x88]`), SetUnorderedAccessViews (two entries from `[rsp + 0x90]`),
// SetShaderResources (four entries from `[rsp + 0xa8]`), and one dispatch of ((w + 7) / 8, (h + 7) / 8, 1) groups with `[r12]` = the size. The loop continues at `0x01bcc400`.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan4(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 node = c->rbx;
    p.context = c->r14;
    const u64 system = R<u64>(p.rsp + 0x80);
    if (!Plausible(node) || !Plausible(p.context) || !Plausible(system)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 data = R<u64>(node + 0x108);
    if (!Plausible(data)) {
        no_work = true;
        return false;
    }
    const float threshold = R<float>(Global(0x024f108c));
    if (threshold > R<float>(data + 0x290)) {
        no_work = true; // the loop skips an emitter that is not alive
        return false;
    }
    const u64 table_pointer_a = R<u64>(p.rsp + 0x88);
    const u64 table_pointer_b = R<u64>(p.rsp + 0x90);
    const u64 table_pointer_c = R<u64>(p.rsp + 0xa8);
    if (!Plausible(table_pointer_a) || !Plausible(table_pointer_b) || !Plausible(table_pointer_c) || !Plausible(R<u64>(p.context)) || !Plausible(R<u64>(R<u64>(p.context) + 0x58))) {
        reason = "the table pointers of the function are not set";
        return false;
    }
    const auto field = [&](u64 owner, u64 offset) -> u64 { return Plausible(owner) ? R<u64>(owner + offset) : 0; };
    const u32 pair = R<u32>(data + 0x250) * 2 + R<u32>(data + 0x24c);
    p.v[0] = R<u64>(data + 0x190);                                   // [rsp + 0xb8]
    p.v[1] = field(R<u64>(data + 0x1c8 + u64{pair} * 8), 0x48);      // [rsp + 0xc0]
    p.v[2] = field(R<u64>(data + 0x1f0), 0x50);                      // [rsp + 0xf0]
    p.v[3] = field(R<u64>(data + 0x1e8), 0x50);                      // [rsp + 0xf8]
    u32 shader_index = 0x20;
    u64 view = 0; // [rsp + 0xc8]
    if (R<u32>(data + 0x248) != 4) {
        shader_index = 0x1e;
        const u64 holder = R<u64>(data + 0x218);
        if (Plausible(holder)) {
            const u64 found = R<u64>(holder + 0x48);
            if (found != 0) {
                shader_index = 0x1f;
                view = found;
            }
        }
    }
    p.v[4] = view;
    p.v[5] = shader_index;
    p.v[6] = system;
    p.v[7] = node;
    p.v[8] = data;
    p.v[9] = R<u32>(system + 0x78); // the shader index cached by the function
    const u64 device = R<u64>(Global(0x0689eed8));
    p.v[10] = Plausible(device) ? R<u64>(device + 0x1008 + (u64{shader_index} << 4)) : 0;
    p.v[11] = R<u64>(p.rsp + 0xd0); // the entries of the shader resource table the loop does not set
    p.v[12] = R<u64>(p.rsp + 0xd8);
    if (p.v[10] == 0) {
        reason = "the shader table is not set";
        return false;
    }
    const u64 size = R<u64>(data);
    const u32 width = static_cast<u32>(size);
    const u32 height = static_cast<u32>(size >> 32);
    p.groups = (width + 7u) >> 3;
    p.groups_y = (height + 7u) >> 3;
    p.site_ghidra = 0x01bcc3fc;
    p.async = false;
    p.context_words.emplace_back(0x500 + 10 * 8, p.v[0]);
    p.context_words.emplace_back(0x1980, p.v[1]);
    p.context_words.emplace_back(0x1980 + 8, p.v[4]);
    p.context_words.emplace_back(0x1980 + 16, p.v[11]);
    p.context_words.emplace_back(0x1980 + 24, p.v[12]);
    p.context_words.emplace_back(0x2098, p.v[2]);
    p.context_words.emplace_back(0x2098 + 8, p.v[3]);
    p.context_words.emplace_back(0x4670, p.v[10]);
    p.registers = {{R12, data}, {R15, shader_index}};
    return true;
}
void Apply4(const Plan& p, Writer& w) {
    w.Put<u64>(p.rsp + 0xb8, p.v[0]);
    w.Put<u64>(p.rsp + 0xc0, p.v[1]);
    w.Put<u64>(p.rsp + 0xf0, p.v[2]);
    w.Put<u64>(p.rsp + 0xf8, p.v[3]);
    w.Put<u64>(p.rsp + 0xc8, p.v[4]);
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 10, 1, R<u64>(p.rsp + 0x88), 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 2, R<u64>(p.rsp + 0x90), 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 4, R<u64>(p.rsp + 0xa8), 0, 0, 0);
    }
    if (static_cast<u32>(p.v[9]) != static_cast<u32>(p.v[5])) {
        w.Put<u32>(p.v[6] + 0x78, static_cast<u32>(p.v[5]));
        if (w.Live()) {
            reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0x38))(p.context, p.v[10], 0, 0, 0, 0, 0); // SetShader
        }
    }
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, p.groups_y, 1, false, p.site_ghidra, false);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Sites 5 and 6: two more dispatch loops of the emitter system `FUN_01bcac40`, over the same list of emitters (`node + 0x38`), the emitter data `[node + 0x108]`, the graphics context in
// r14. Site 5 (0x01bcbdb8 .. 0x01bcbeff, node r12, data r13, the shader was set before the loop, resumes at 0x01bcbf02 where the engine counts the emitter's frames down): the loop
// only runs for `[data + 0x258] > 0`. Site 6 (0x01bcba34 .. 0x01bcbb5b, node rbx, data r15, system r13 with the cached shader index at `+ 0x78`, shader 0x1b from `[device + 0x11b8]`,
// resumes at 0x01bcbb60, the loop's next emitter). Both fill the same stack slots (`[rsp + 0xb8]`, `[rsp + 0xc0]`, `[rsp + 0xc8]`, `[rsp + 0xf0]`), write the tables with
// SetConstantBuffers (slot 10), SetUnorderedAccessViews (two entries) and SetShaderResources (four entries), and dispatch ((w + 7) / 8, (h + 7) / 8, 1) groups of the size `[data]`
// (site 6: `[data + 0x30]`).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool EmitterPreamble(MidContext* c, Plan& p, u64 node, u64 data, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    p.context = c->r14;
    if (!Plausible(node) || !Plausible(p.context)) {
        reason = "an object of the loop is not set";
        return false;
    }
    if (!Plausible(data)) {
        no_work = true;
        return false;
    }
    const float threshold = R<float>(Global(0x024f108c));
    if (threshold > R<float>(data + 0x290)) {
        no_work = true; // the loop skips an emitter that is not alive
        return false;
    }
    if (!Plausible(R<u64>(p.context)) || !Plausible(R<u64>(R<u64>(p.context) + 0x58))) {
        reason = "the context has no virtual table";
        return false;
    }
    return true;
}
bool Scan5(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    const u64 node = c->r12;
    const u64 data = Plausible(node) ? R<u64>(node + 0x108) : 0;
    if (!EmitterPreamble(c, p, node, data, reason, no_work)) {
        return false;
    }
    if (static_cast<s32>(R<u32>(data + 0x258)) <= 0) {
        no_work = true;
        return false;
    }
    const auto field = [&](u64 owner, u64 offset) -> u64 { return Plausible(owner) ? R<u64>(owner + offset) : 0; };
    const u32 pair = R<u32>(data + 0x250) * 2 + R<u32>(data + 0x24c);
    p.v[0] = R<u64>(data + 0x190);                              // [rsp + 0xb8]
    p.v[1] = field(R<u64>(data + 0x1c8 + u64{pair} * 8), 0x48); // [rsp + 0xc0]
    const u64 second = field(R<u64>(data + 0x220), 0x48);
    p.v[2] = second != 0 ? second : p.v[1];                      // [rsp + 0xc8]
    const u32 other = (R<u32>(data + 0x24c) ^ 1u) + R<u32>(data + 0x250) * 2;
    p.v[3] = field(R<u64>(data + 0x1c8 + u64{other} * 8), 0x50); // [rsp + 0xf0]
    p.v[4] = R<u64>(p.rsp + 0xd0);
    p.v[5] = R<u64>(p.rsp + 0xd8);
    p.v[6] = R<u64>(p.rsp + 0xf8);
    p.v[7] = data;
    const u64 size = R<u64>(data);
    p.groups = (static_cast<u32>(size) + 7u) >> 3;
    p.groups_y = (static_cast<u32>(size >> 32) + 7u) >> 3;
    p.site_ghidra = 0x01bcbf02;
    p.async = false;
    p.context_words.emplace_back(0x500 + 10 * 8, p.v[0]);
    p.context_words.emplace_back(0x1980, p.v[1]);
    p.context_words.emplace_back(0x1980 + 8, p.v[2]);
    p.context_words.emplace_back(0x1980 + 16, p.v[4]);
    p.context_words.emplace_back(0x1980 + 24, p.v[5]);
    p.context_words.emplace_back(0x2098, p.v[3]);
    p.context_words.emplace_back(0x2098 + 8, p.v[6]);
    p.registers = {{R13, data}, {RBX, 0}};
    return true;
}
void ApplyEmitterTables(const Plan& p, Writer& w) {
    w.Put<u64>(p.rsp + 0xb8, p.v[0]);
    w.Put<u64>(p.rsp + 0xc0, p.v[1]);
    w.Put<u64>(p.rsp + 0xf0, p.v[3]);
}
void Apply5(const Plan& p, Writer& w) {
    ApplyEmitterTables(p, w);
    w.Put<u64>(p.rsp + 0xc8, p.v[2]);
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 10, 1, R<u64>(p.rsp + 0x88), 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 2, R<u64>(p.rsp + 0x90), 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 4, R<u64>(p.rsp + 0xa8), 0, 0, 0);
        NativeDraw::IssueDispatch(p.context, p.groups, p.groups_y, 1, false, p.site_ghidra, false);
    }
}

bool Scan6(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    const u64 node = c->rbx;
    const u64 data = Plausible(node) ? R<u64>(node + 0x108) : 0;
    if (!EmitterPreamble(c, p, node, data, reason, no_work)) {
        return false;
    }
    if (R<u32>(data + 0x38) == 0) {
        no_work = true;
        return false;
    }
    const u64 system = c->r13;
    if (!Plausible(system)) {
        reason = "the system is not set";
        return false;
    }
    const auto field = [&](u64 owner, u64 offset) -> u64 { return Plausible(owner) ? R<u64>(owner + offset) : 0; };
    p.v[0] = R<u64>(data + 0x160);                              // [rsp + 0xb8]
    p.v[1] = field(R<u64>(data + 0x148), 0x28);                 // [rsp + 0xc0]
    const u32 other = (R<u32>(data + 0x24c) ^ 1u) + R<u32>(data + 0x250) * 2;
    p.v[3] = field(R<u64>(data + 0x1c8 + u64{other} * 8), 0x50); // [rsp + 0xf0]
    p.v[2] = R<u64>(p.rsp + 0xc8);
    p.v[4] = R<u64>(p.rsp + 0xd0);
    p.v[5] = R<u64>(p.rsp + 0xd8);
    p.v[6] = R<u64>(p.rsp + 0xf8);
    p.v[7] = data;
    p.v[8] = system;
    p.v[9] = R<u32>(system + 0x78); // the cached shader index
    const u64 device = R<u64>(Global(0x0689eed8));
    p.v[10] = Plausible(device) ? R<u64>(device + 0x11b8) : 0;
    if (p.v[10] == 0) {
        reason = "the shader table is not set";
        return false;
    }
    const u64 size = R<u64>(data + 0x30);
    p.groups = (static_cast<u32>(size) + 7u) >> 3;
    p.groups_y = (static_cast<u32>(size >> 32) + 7u) >> 3;
    p.site_ghidra = 0x01bcbb5e;
    p.async = false;
    p.context_words.emplace_back(0x500 + 10 * 8, p.v[0]);
    p.context_words.emplace_back(0x1980, p.v[1]);
    p.context_words.emplace_back(0x1980 + 8, p.v[2]);
    p.context_words.emplace_back(0x1980 + 16, p.v[4]);
    p.context_words.emplace_back(0x1980 + 24, p.v[5]);
    p.context_words.emplace_back(0x2098, p.v[3]);
    p.context_words.emplace_back(0x2098 + 8, p.v[6]);
    p.context_words.emplace_back(0x4670, p.v[10]);
    p.registers = {{R15, data}};
    return true;
}
void Apply6(const Plan& p, Writer& w) {
    ApplyEmitterTables(p, w);
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 10, 1, p.rsp + 0xb8, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 2, p.rsp + 0xf0, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 4, p.rsp + 0xc0, 0, 0, 0);
    }
    if (static_cast<u32>(p.v[9]) != 0x1b) {
        w.Put<u32>(p.v[8] + 0x78, 0x1b);
        if (w.Live()) {
            reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0x38))(p.context, p.v[10], 0, 0, 0, 0, 0); // SetShader
        }
    }
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, p.groups_y, 1, false, p.site_ghidra, false);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 7: the piece of `FUN_01d7d260` (the vertex cluster function) before its object loops, 0x01d7e0a1 .. 0x01d7e1bd, resumes at 0x01d7e1c0 (one dispatch per frame). The system is in r12
// (copied from r15), the graphics context in r14: the holder `[system + 0xb0]` gives two constant buffers (`[holder + 0x78]`, `[holder + 0x60]`) to the stack slots 0x90 / 0x98, five
// shader resources (`[[holder + 0x30 / 0x38 / 0x40 / 0x48 / 0x28] + 0x28]`, a null holder pointer gives 0) to 0x60 .. 0x80, one unordered access view (`[[holder] + 0x30]`) to 0x58;
// SetConstantBuffers (0, 2), SetShaderResources (0, 6), SetUnorderedAccessViews (0, 1), SetShader (`[device + 0x1648]`), dispatch of ([system + 0xc0] + 63) / 64 groups.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan7(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 system = c->r12;
    p.context = c->r14;
    if (!Plausible(system) || !Plausible(p.context) || !Plausible(R<u64>(p.context)) || !Plausible(R<u64>(R<u64>(p.context) + 0x58))) {
        reason = "an object is not set";
        return false;
    }
    const u64 holder = R<u64>(system + 0xb0);
    if (!Plausible(holder)) {
        reason = "no holder";
        return false;
    }
    const u32 n = R<u32>(system + 0xc0);
    if (n == 0) {
        no_work = true;
        return false;
    }
    const auto field = [&](u64 owner, u64 offset) -> u64 { return owner != 0 ? R<u64>(owner + offset) : 0; };
    p.v[0] = R<u64>(holder + 0x78);
    p.v[1] = R<u64>(holder + 0x60);
    p.v[2] = field(R<u64>(holder + 0x30), 0x28); // [rsp + 0x60]
    p.v[3] = field(R<u64>(holder + 0x38), 0x28); // [rsp + 0x68]
    p.v[4] = field(R<u64>(holder + 0x40), 0x28); // [rsp + 0x70]
    p.v[5] = field(R<u64>(holder + 0x48), 0x28); // [rsp + 0x78]
    p.v[6] = field(R<u64>(holder + 0x28), 0x28); // [rsp + 0x80]
    p.v[7] = field(R<u64>(holder), 0x30);        // [rsp + 0x58]
    p.v[8] = R<u64>(p.rsp + 0x88);               // the sixth slot, not touched by the piece
    const u64 device = R<u64>(Global(0x0689eed8));
    p.v[9] = Plausible(device) ? R<u64>(device + 0x1648) : 0;
    if (p.v[9] == 0) {
        reason = "the shader is not set";
        return false;
    }
    p.groups = (n + 63u) >> 6;
    p.site_ghidra = 0x01d7e1c0;
    p.async = false;
    p.context_words.emplace_back(0x500, p.v[0]);
    p.context_words.emplace_back(0x508, p.v[1]);
    for (int i = 0; i < 5; ++i) {
        p.context_words.emplace_back(0x1980 + 8 * i, p.v[2 + i]);
    }
    p.context_words.emplace_back(0x1980 + 40, p.v[8]);
    p.context_words.emplace_back(0x2098, p.v[7]);
    p.context_words.emplace_back(0x4670, p.v[9]);
    p.registers = {{RBX, p.context}, {R15, p.context}};
    return true;
}
void Apply7(const Plan& p, Writer& w) {
    w.Put<u64>(p.rsp + 0x90, p.v[0]);
    w.Put<u64>(p.rsp + 0x98, p.v[1]);
    for (int i = 0; i < 5; ++i) {
        w.Put<u64>(p.rsp + 0x60 + 8 * i, p.v[2 + i]);
    }
    w.Put<u64>(p.rsp + 0x58, p.v[7]);
    if (w.Live()) {
        const u64 vtable = R<u64>(p.context);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(p.context, 0, 2, p.rsp + 0x90, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(p.context, 0, 6, p.rsp + 0x60, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(p.context, 0, 1, p.rsp + 0x58, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x38))(p.context, p.v[9], 0, 0, 0, 0, 0); // SetShader
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, false);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Sites 8 and 9: the third and fourth system loop of RenderDev_vfn18_BeginRecording, async context in `[rsp + 0x88]` (loop 3, rbx) or r14 (loop 4), the holder `[rsp + 0xa8]` (r14 /
// r15). Same shape as sites 0 and 1: seven words at `holder + 0x27c0 ..` from the system, the sum of the part sizes and the entry of the array `[rsp + 0xa0]` (loop 3) or the loop
// index and three counters (loop 4), the 64 bytes at `[rsp + 0x140]` copied to the staging block `[rsp + 0x18]`, UpdateSubresource of the constant buffer `[holder + 0x2820]`, the
// views `[[rsp + 0x98] + 0x28]`, `[[rsp + 0x80] + 0x28]`, `[holder + 0x2848]` (loop 4 also `[holder + 0x2858]`) written through the stack at `[rsp + 0x100]` to the context's table at
// `+ 0x1d18` (five entries), the shader (`[device + 0x1778]` / `0x1788`) at `+ 0x1d10` and a direct call of the async dispatch `0x024711c0`.
// Loop 3: top at 0x01c4b590 (eax = the sum), resumes at 0x01c4b6ea. Loop 4 (first piece): top at 0x01c4bd40 (ebx = the sum), resumes at 0x01c4bea6; the piece only dispatches when the
// counter `[[rsp + 0x40]]` is not zero (otherwise the engine runs it). The loop's second piece (0x01c4bea6 .. 0x01c4bfa8, shader 0x1798) is not reached in the default scene (0 per frame).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan8(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 system = c->r13;
    const u64 holder = R<u64>(p.rsp + 0xa8);
    p.context = R<u64>(p.rsp + 0x88);
    const u64 array = R<u64>(p.rsp + 0xa0);
    const u64 entry = array + c->r15;
    const u64 view0_owner = R<u64>(p.rsp + 0x98);
    const u64 view1_owner = R<u64>(p.rsp + 0x80);
    const u64 staging = R<u64>(p.rsp + 0x18);
    const u64 update_source = R<u64>(p.rsp + 0x28);
    if (!Plausible(system) || !Plausible(holder) || !Plausible(p.context) || !Plausible(array) || !Plausible(view0_owner) || !Plausible(view1_owner) || !Plausible(staging) ||
        !Plausible(update_source)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 vtable = R<u64>(p.context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0))) {
        reason = "the context has no virtual table";
        return false;
    }
    const u32 size = R<u32>(entry + 0x10);
    if (size == 0) {
        no_work = true;
        return false;
    }
    const u64 chain_source = R<u64>(p.rsp + 0xc0);
    const u64 chain = chain_source != 0 ? chain_source : system;
    const u64 device = R<u64>(Global(0x0689eed8));
    p.v[0] = holder;
    p.v[1] = staging;
    p.v[2] = update_source;
    p.v[3] = R<u32>(system + 0x50);
    p.v[4] = R<u32>(chain + 0x54);
    p.v[5] = static_cast<u32>(c->rax); // the sum of the part sizes
    p.v[6] = R<u32>(entry + 0xc);
    p.v[7] = R<u32>(entry + 0x10);
    p.v[8] = R<u32>(entry + 0x8);
    p.v[9] = R<u64>(holder + 0x2820); // the constant buffer
    p.v[10] = R<u64>(view0_owner + 0x28);
    p.v[11] = R<u64>(view1_owner + 0x28);
    p.v[12] = R<u64>(holder + 0x2848);
    p.v[13] = Plausible(device) ? R<u64>(device + 0x1778) : 0;
    p.v[14] = entry;
    if (p.v[13] == 0) {
        reason = "the shader is not set";
        return false;
    }
    p.groups = (size + 63u) >> 6;
    p.site_ghidra = 0x01c4b6ea;
    p.async = true;
    p.context_words.emplace_back(0x1d10, p.v[13]);
    p.context_words.emplace_back(0x1d18, p.v[10]);
    p.context_words.emplace_back(0x1d20, p.v[11]);
    p.context_words.emplace_back(0x1d28, p.v[12]);
    p.context_words.emplace_back(0x1d30, 0);
    p.context_words.emplace_back(0x1d38, 0);
    p.registers = {{RBX, p.context}, {R13, entry + 0x10}, {R14, holder}};
    return true;
}
void Apply8(const Plan& p, Writer& w) {
    const u64 holder = p.v[0];
    w.Put<u32>(holder + 0x27c0, static_cast<u32>(p.v[3]));
    w.Put<u32>(holder + 0x27c4, static_cast<u32>(p.v[4]));
    w.Put<u32>(holder + 0x27c8, static_cast<u32>(p.v[5]));
    w.Put<u32>(holder + 0x27d0, static_cast<u32>(p.v[6]));
    w.Put<u32>(holder + 0x27d4, static_cast<u32>(p.v[7]));
    w.Put<u32>(holder + 0x27d8, 0);
    w.Put<u32>(holder + 0x27dc, static_cast<u32>(p.v[8]));
    u8 block[64];
    std::memcpy(block, reinterpret_cast<const void*>(p.rsp + 0x140), 64);
    w.Put(p.v[1], block, 32);
    w.Put(p.v[1] + 0x20, block + 32, 32);
    w.Put<u32>(p.rsp, 0); // the seventh argument slot
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0xd0))(p.context, p.v[9], 0, 0, p.v[2], 0, 0); // UpdateSubresource
    }
    u64 views[5] = {p.v[10], p.v[11], p.v[12], 0, 0};
    w.Put(p.rsp + 0x100, views, 40);
    w.Put(p.context + 0x1d18, views, 40);
    w.Put<u64>(p.context + 0x1d10, p.v[13]);
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, true);
    }
}

bool Scan9(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    p.rsp = c->rsp;
    const u64 system = c->r12;
    const u64 holder = c->r15;
    p.context = c->r14;
    const u64 counter = R<u64>(p.rsp + 0x40);
    const u64 view0_owner = R<u64>(p.rsp + 0x98);
    const u64 view1_owner = R<u64>(p.rsp + 0x80);
    const u64 staging = R<u64>(p.rsp + 0x18);
    const u64 update_source = R<u64>(p.rsp + 0x28);
    const u64 count_a = R<u64>(p.rsp + 0x58);
    const u64 count_b = R<u64>(p.rsp + 0x50);
    const u64 count_c = R<u64>(p.rsp + 0x48);
    if (!Plausible(system) || !Plausible(holder) || !Plausible(p.context) || !Plausible(counter) || !Plausible(view0_owner) || !Plausible(view1_owner) || !Plausible(staging) ||
        !Plausible(update_source) || !Plausible(count_a) || !Plausible(count_b) || !Plausible(count_c)) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 vtable = R<u64>(p.context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0))) {
        reason = "the context has no virtual table";
        return false;
    }
    const u32 sum = static_cast<u32>(c->rbx);
    if (R<u32>(counter) == 0 || sum == 0) {
        no_work = true; // the piece writes the holder and does not dispatch
        return false;
    }
    const u64 chain_source = R<u64>(p.rsp + 0xc0);
    const u64 chain = chain_source != 0 ? chain_source : system;
    const u64 device = R<u64>(Global(0x0689eed8));
    p.v[0] = holder;
    p.v[1] = staging;
    p.v[2] = update_source;
    p.v[3] = R<u32>(system + 0x50);
    p.v[4] = R<u32>(chain + 0x54);
    p.v[5] = sum;
    p.v[6] = R<u32>(count_a);
    p.v[7] = R<u32>(count_b);
    p.v[8] = R<u32>(count_c);
    p.v[9] = R<u64>(holder + 0x2820);
    p.v[10] = R<u64>(view0_owner + 0x28);
    p.v[11] = R<u64>(view1_owner + 0x28);
    p.v[12] = R<u64>(holder + 0x2848);
    p.v[13] = Plausible(device) ? R<u64>(device + 0x1788) : 0;
    p.v[14] = R<u64>(holder + 0x2858);
    p.v[15] = static_cast<u32>(c->r13); // the loop index
    if (p.v[13] == 0) {
        reason = "the shader is not set";
        return false;
    }
    p.groups = (sum + 63u) >> 6;
    p.site_ghidra = 0x01c4bea6;
    p.async = true;
    p.context_words.emplace_back(0x1d10, p.v[13]);
    p.context_words.emplace_back(0x1d18, p.v[10]);
    p.context_words.emplace_back(0x1d20, p.v[11]);
    p.context_words.emplace_back(0x1d28, p.v[12]);
    p.context_words.emplace_back(0x1d30, p.v[14]);
    p.context_words.emplace_back(0x1d38, 0);
    p.registers = {{R14, p.context}, {RBX, sum}};
    return true;
}
void Apply9(const Plan& p, Writer& w) {
    const u64 holder = p.v[0];
    w.Put<u32>(holder + 0x27c0, static_cast<u32>(p.v[3]));
    w.Put<u32>(holder + 0x27c4, static_cast<u32>(p.v[4]));
    w.Put<u32>(holder + 0x27c8, static_cast<u32>(p.v[5]));
    w.Put<u32>(holder + 0x27d0, static_cast<u32>(p.v[6]));
    w.Put<u32>(holder + 0x27d4, static_cast<u32>(p.v[7]));
    w.Put<u32>(holder + 0x27d8, static_cast<u32>(p.v[15]));
    w.Put<u32>(holder + 0x27dc, static_cast<u32>(p.v[8]));
    u8 block[64];
    std::memcpy(block, reinterpret_cast<const void*>(p.rsp + 0x140), 64);
    w.Put(p.v[1], block, 32);
    w.Put(p.v[1] + 0x20, block + 32, 32);
    w.Put<u32>(p.rsp, 0); // the seventh argument slot
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(R<u64>(p.context) + 0xd0))(p.context, p.v[9], 0, 0, p.v[2], 0, 0); // UpdateSubresource
    }
    u64 views[5] = {p.v[10], p.v[11], p.v[12], p.v[14], 0};
    w.Put(p.rsp + 0x100, views, 40);
    w.Put(p.context + 0x1d18, views, 40);
    w.Put<u64>(p.context + 0x1d10, p.v[13]);
    if (w.Live()) {
        NativeDraw::IssueDispatch(p.context, p.groups, 1, 1, false, p.site_ghidra, true);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 10: the chain of screen passes in `RenderDev_RenderScene` (ghidra 0x01c593d0), the loop over the levels `0x01c6d360 .. 0x01c6d7ea`, graphics context in r15 (the fenced dispatch
// is its virtual 0x50), the holder `[rsp + 0xd8]`, the table of views `[rsp + 0x110]`, the level count in rdx (counted down), the offset r12 (0 on the first level, then minus 8 per
// level). One level: four words into the holder (the level size and the level), UpdateSubresource of `[holder + 0x2d0]`, then (first level) one pass with the shader 0x12c8 or (other
// levels) two passes with 0x1308 and 0x12c8, then the pass with 0x12d8 that both have; every pass is SetUnorderedAccessViews (one view from the table or zero), SetShaderResources
// (two entries from the holder or zero), SetShader, and the fenced dispatch of (ceil(w / 8), ceil(h / 8), 1) groups, the views cleared after it. Resumes at 0x01c6d7ea.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan10(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r15;
    const u64 holder = R<u64>(p.rsp + 0xd8);
    const u64 table = R<u64>(p.rsp + 0x110);
    const u64 update_source = R<u64>(p.rsp + 0xb0);
    const u32 level_count = static_cast<u32>(c->rdx);
    if (!Plausible(p.context) || !Plausible(holder) || !Plausible(table) || !Plausible(update_source) || level_count == 0) {
        reason = "an object of the loop is not set";
        return false;
    }
    const u64 vtable = R<u64>(p.context);
    const u64 fenced_entry = Plausible(vtable) ? R<u64>(vtable + 0x50) : 0;
    const bool async = fenced_entry == Global(0x0246f5f0);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0)) || (!async && fenced_entry != Global(0x02479810))) {
        reason = "the context is not a known context";
        return false;
    }
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(device)) {
        reason = "the device is not set";
        return false;
    }
    const s64 offset = static_cast<s64>(c->r12);
    const u64 size = R<u64>(holder + 0x170);
    const u32 shift = (level_count - 1) & 31;
    const u32 w = static_cast<u32>(size) >> shift;
    const u32 h = static_cast<u32>(size >> 32) >> shift;
    const u32 gx = (w >> 3) + ((w & 7) != 0 ? 1 : 0);
    const u32 gy = (h >> 3) + ((h & 7) != 0 ? 1 : 0);
    p.v[0] = holder;
    p.v[1] = table;
    p.v[2] = update_source;
    p.v[3] = R<u64>(holder + 0x2d0); // the constant buffer
    p.v[4] = w;
    p.v[5] = h;
    p.v[6] = level_count;
    p.v[7] = c->rdx;
    p.v[8] = static_cast<u64>(offset);
    p.v[9] = gx;
    p.v[10] = gy;
    p.v[11] = R<u64>(device + 0x1308);
    p.v[12] = R<u64>(device + 0x12c8);
    p.v[13] = R<u64>(device + 0x12d8);
    p.v[14] = R<u64>(holder + 0x118);
    p.v[15] = R<u64>(holder + 0x180);
    p.v[16] = R<u64>(holder + 0x250);
    p.v[17] = R<u64>(holder + 0x1e8);
    // the views: first pass of the other levels, second pass (or the only pass of the first level), the last pass
    p.v[18] = offset != 0 ? R<u64>(table + static_cast<u64>(offset)) : 0;
    p.v[19] = R<u64>(table + static_cast<u64>(offset) - 0x68);
    p.v[20] = R<u64>(table + static_cast<u64>(offset) - 0xd0);
    if (p.v[11] == 0 || p.v[12] == 0 || p.v[13] == 0) {
        reason = "a shader is not set";
        return false;
    }
    const auto expect = [&](u64 site, u64 srv0, u64 srv1, u64 uav, u64 shader) {
        Expect e;
        e.x = gx;
        e.y = gy;
        e.site = site;
        e.words = {{async ? 0x1d10 : 0x4670, shader}}; // the tables are written by the engine's virtuals: their input is compared (the stack slots)
        e.stack = {{0x140, uav}, {0x180, srv0}, {0x188, srv1}};
        p.dispatches.push_back(std::move(e));
    };
    if (offset != 0) {
        expect(0x01c6d4b5, p.v[14], p.v[15], p.v[18], p.v[11]);
        expect(0x01c6d59c, p.v[16], 0, p.v[19], p.v[12]);
    } else {
        expect(0x01c6d6b7, p.v[14], 0, p.v[19], p.v[12]);
    }
    expect(0x01c6d7a9, p.v[17], 0, p.v[20], p.v[13]);
    p.site_ghidra = p.dispatches[0].site;
    p.async = async;
    p.v[21] = async ? 1 : 0;
    p.registers = {{RBX, p.rsp + 0x180}, {R13, p.rsp + 0x140}, {R14, p.rsp + 0x180}, {R12, c->r12}};
    return true;
}
void Apply10(const Plan& p, Writer& w) {
    const u64 holder = p.v[0];
    const u64 context = p.context;
    const u64 vtable = R<u64>(context);
    const u64 view_slot = p.rsp + 0x140;
    const u64 srv_slot = p.rsp + 0x180;
    const auto set_view = [&](u64 view) {
        w.Put<u64>(view_slot, view);
        if (w.Live()) {
            reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(context, 0, 1, view_slot, 0, 0, 0);
        }
    };
    const auto set_srvs = [&](u64 a, u64 b) {
        w.Put<u64>(srv_slot, a);
        w.Put<u64>(srv_slot + 8, b);
        if (w.Live()) {
            reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(context, 0, 2, srv_slot, 0, 0, 0);
        }
    };
    const auto set_shader = [&](u64 shader) {
        if (w.Live()) {
            reinterpret_cast<Virtual>(R<u64>(vtable + 0x38))(context, shader, 0, 0, 0, 0, 0);
        }
    };
    size_t dispatch_index = 0;
    const auto dispatch = [&] {
        const Expect& e = p.dispatches[dispatch_index++];
        if (w.Live()) {
            NativeDraw::IssueDispatch(context, e.x, e.y, e.z, true, e.site, p.v[21] != 0);
        }
    };
    w.Put<u32>(holder + 0x2c0, static_cast<u32>(p.v[4]));
    w.Put<u32>(holder + 0x2c4, static_cast<u32>(p.v[5]));
    w.Put<u32>(holder + 0x2c8, static_cast<u32>(p.v[6]) - 1);
    w.Put<u32>(holder + 0x2cc, static_cast<u32>(p.v[6]));
    w.Put<u64>(p.rsp + 0xd0, p.v[7]);
    w.Put<u32>(p.rsp, 0); // the seventh argument slot
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(vtable + 0xd0))(context, p.v[3], 0, 0, p.v[2], 0, 0); // UpdateSubresource
    }
    w.Put<u64>(p.rsp + 0xe0, p.v[8]);
    w.Put<u32>(p.rsp + 0xc8, static_cast<u32>(p.v[9]));
    w.Put<u32>(p.rsp + 0xc0, static_cast<u32>(p.v[10]));
    if (static_cast<s64>(p.v[8]) != 0) {
        set_view(p.v[18]);
        set_srvs(p.v[14], p.v[15]);
        set_shader(p.v[11]);
        dispatch();
        set_view(0);
        set_srvs(0, 0);
        set_view(p.v[19]);
        set_srvs(p.v[16], 0);
        set_shader(p.v[12]);
        dispatch();
        set_view(0);
        set_srvs(0, 0);
    } else {
        set_view(p.v[19]);
        set_srvs(p.v[14], 0);
        set_shader(p.v[12]);
        dispatch();
        set_view(0);
        set_srvs(0, 0);
    }
    set_view(p.v[20]);
    set_srvs(p.v[17], 0);
    set_shader(p.v[13]);
    dispatch();
    set_view(0);
    set_srvs(0, 0);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// The step based sites: the scan lists the steps, `Finish` derives what the verify compares at every dispatch (the shader word of the context and the stack slots the engine's virtuals
// copy into the tables), `ApplySteps` does them. `ContextKind`: 1 graphics context, 2 async context (told by the fenced dispatch entry of the virtual table).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int ContextKind(u64 context) {
    if (!Plausible(context)) {
        return 0;
    }
    const u64 vtable = R<u64>(context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0))) {
        return 0;
    }
    const u64 fenced = R<u64>(vtable + 0x50);
    return fenced == Global(0x0246f5f0) ? 2 : fenced == Global(0x02479810) ? 1 : 0;
}
bool Finish(Plan& p, int kind) {
    if (kind == 0) {
        return false;
    }
    p.async = kind == 2;
    std::map<u64, u64> stack;
    std::vector<std::pair<u64, u64>> uav, srv;
    u64 shader = 0;
    bool shader_set = false;
    std::map<u64, u64> direct; // writes straight into the context's memory (offset -> value)
    const auto slots = [&](const Step& s, std::vector<std::pair<u64, u64>>& out) {
        out.clear();
        const u64 base = s.d != 0 ? s.d - p.rsp : s.c; // an array behind a pointer: its offset from rsp (wraps for an address outside the frame)
        for (u64 i = 0; i < s.b; ++i) {
            const u64 offset = base + 8 * i;
            const auto found = stack.find(offset);
            out.emplace_back(offset, found != stack.end() ? found->second : R<u64>(p.rsp + offset));
        }
    };
    for (const Step& s : p.steps) {
        switch (s.kind) {
        case Step::PutStack64: stack[s.a] = s.b; break;
        case Step::SetUav: slots(s, uav); break;
        case Step::SetSrv: slots(s, srv); break;
        case Step::SetShader: shader = s.a; shader_set = true; break;
        case Step::PutAbs64:
            if (s.a >= p.context && s.a < p.context + 0x10000) {
                direct[s.a - p.context] = s.b;
            }
            break;
        case Step::Dispatch: {
            Expect e;
            e.x = static_cast<u32>(s.a);
            e.y = static_cast<u32>(s.b);
            e.z = static_cast<u32>(s.c);
            e.site = s.d;
            if (shader_set) {
                e.words = {{p.async ? 0x1d10 : 0x4670, shader}};
            }
            e.words.insert(e.words.end(), direct.begin(), direct.end());
            e.stack = uav;
            e.stack.insert(e.stack.end(), srv.begin(), srv.end());
            p.dispatches.push_back(std::move(e));
            break;
        }
        default: break;
        }
    }
    if (p.dispatches.empty()) {
        return false;
    }
    p.site_ghidra = p.dispatches[0].site;
    return true;
}
void ApplySteps(const Plan& p, Writer& w) {
    const u64 context = p.context;
    const u64 vtable = R<u64>(context);
    for (const Step& s : p.steps) {
        switch (s.kind) {
        case Step::PutStack64: w.Put<u64>(p.rsp + s.a, s.b); break;
        case Step::PutStack32: w.Put<u32>(p.rsp + s.a, static_cast<u32>(s.b)); break;
        case Step::PutAbs64: w.Put<u64>(s.a, s.b); break;
        case Step::PutAbs32: w.Put<u32>(s.a, static_cast<u32>(s.b)); break;
        case Step::PutAbs8: w.Put<u8>(s.a, static_cast<u8>(s.b)); break;
        case Step::PutAbsZero: {
            u8 zeros[64]{};
            w.Put(s.a, zeros, s.b); // s.b <= 64 bytes
            break;
        }
        case Step::Update:
            w.Put<u32>(p.rsp, 0); // the seventh argument slot
            if (w.Live()) {
                reinterpret_cast<Virtual>(R<u64>(vtable + 0xd0))(context, s.a, 0, 0, s.b, 0, 0);
            }
            break;
        case Step::SetCb:
            if (w.Live()) {
                reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(context, s.a, s.b, s.d != 0 ? s.d : p.rsp + s.c, 0, 0, 0);
            }
            break;
        case Step::SetSrv:
            if (w.Live()) {
                reinterpret_cast<Virtual>(R<u64>(vtable + 0x40))(context, s.a, s.b, s.d != 0 ? s.d : p.rsp + s.c, 0, 0, 0);
            }
            break;
        case Step::SetUav:
            if (w.Live()) {
                reinterpret_cast<Virtual>(R<u64>(vtable + 0x48))(context, s.a, s.b, s.d != 0 ? s.d : p.rsp + s.c, 0, 0, 0);
            }
            break;
        case Step::SetShader:
            if (w.Live()) {
                reinterpret_cast<Virtual>(R<u64>(vtable + 0x38))(context, s.a, 0, 0, 0, 0, 0);
            }
            break;
        case Step::Dispatch:
            if (w.Live()) {
                NativeDraw::IssueDispatch(context, s.a, s.b, s.c, s.e != 0, s.d, p.async);
            }
            break;
        }
    }
}
/// The common pass: one view, srv slots, a shader, a fixed dispatch, then the views cleared again (the engine's pattern around most of the screen sized passes).
void AddPass(Plan& p, u64 uav_slot, u64 uav, u64 srv_slot, u32 srv_count, const u64* srvs, u64 shader, u64 x, u64 y, u64 z, u64 site, bool fenced, bool clear) {
    p.steps.push_back({Step::PutStack64, uav_slot, uav});
    p.steps.push_back({Step::SetUav, 0, 1, uav_slot});
    for (u32 i = 0; i < srv_count; ++i) {
        p.steps.push_back({Step::PutStack64, srv_slot + 8 * i, srvs[i]});
    }
    p.steps.push_back({Step::SetSrv, 0, srv_count, srv_slot});
    p.steps.push_back({Step::SetShader, shader});
    p.steps.push_back({Step::Dispatch, x, y, z, site, fenced ? 1u : 0u});
    if (clear) {
        p.steps.push_back({Step::PutStack64, uav_slot, 0});
        p.steps.push_back({Step::SetUav, 0, 1, uav_slot});
        for (u32 i = 0; i < srv_count; ++i) {
            p.steps.push_back({Step::PutStack64, srv_slot + 8 * i, 0});
        }
        p.steps.push_back({Step::SetSrv, 0, srv_count, srv_slot});
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 11: a fixed pass of `RenderDev_RenderScene`, 0x01c6cefc .. 0x01c6cfc9: context r14, the view `[[[rsp + 0x80] + 0x48] + 0x10]`, the shader resource in r12, the shader
// `[device + 0x1328]`, a fenced dispatch of (16, 16, 1) groups, then both cleared. Leaves r13 = rsp + 0x140 and r12 = rsp + 0x180.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan11(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    const u64 owner = R<u64>(p.rsp + 0x80);
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(owner) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 inner = R<u64>(owner + 0x48);
    if (!Plausible(inner)) {
        reason = "the view owner is not set";
        return false;
    }
    const u64 uav = R<u64>(inner + 0x10);
    const u64 shader = R<u64>(device + 0x1328);
    const u64 srv = c->r12;
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    AddPass(p, 0x140, uav, 0x180, 1, &srv, shader, 16, 16, 1, 0x01c6cf88, true, true);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R13, p.rsp + 0x140}, {R12, p.rsp + 0x180}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 12: two consecutive passes of `RenderDev_RenderScene` with eight shader resources, 0x01c6d944 .. 0x01c6db15: context r14, the object r12 (`+ 0x180`, `+ 0x1540`, `+ 0x1538`,
// `+ 0x1568`), the sixteen byte pair `[rsp + 0x90]`, the 32 bytes `[rsp + 0x40]` (a b c d: the resources are s0 s1 0 [r12 + 0x180] 0 b c d), shaders `[device + 0x1228]` and `0x1238`,
// fenced dispatches of ((r13d + 7) / 8, (r15d + 7) / 8, 1). Leaves r13d and r15d divided.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan12(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    const u64 object = c->r12;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(object) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 s0 = R<u64>(p.rsp + 0x90), s1 = R<u64>(p.rsp + 0x98);
    const u64 a = R<u64>(p.rsp + 0x40), b = R<u64>(p.rsp + 0x48), cc = R<u64>(p.rsp + 0x50), dd = R<u64>(p.rsp + 0x58);
    (void)a;
    const u64 x180 = R<u64>(object + 0x180);
    const u64 shader1 = R<u64>(device + 0x1228), shader2 = R<u64>(device + 0x1238);
    if (shader1 == 0 || shader2 == 0) {
        reason = "a shader is not set";
        return false;
    }
    const u32 gx = (static_cast<u32>(c->r13) + 7u) >> 3;
    const u32 gy = (static_cast<u32>(c->r15) + 7u) >> 3;
    const auto put = [&](u64 offset, u64 value) { p.steps.push_back({Step::PutStack64, offset, value}); };
    // pass one
    put(0x180, s0);
    put(0x188, s1);
    put(0x190, 0);
    put(0x198, x180);
    put(0x1a0, 0);
    put(0x110, cc);
    put(0x118, dd);
    put(0x1a8, b);
    put(0x1b0, cc);
    put(0x1b8, dd);
    put(0xe0, b);
    put(0xe8, cc);
    put(0x140, R<u64>(object + 0x1540));
    p.steps.push_back({Step::SetUav, 0, 1, 0x140});
    p.steps.push_back({Step::SetSrv, 0, 8, 0x180});
    p.steps.push_back({Step::SetShader, shader1});
    p.steps.push_back({Step::Dispatch, gx, gy, 1, 0x01c6da38, 1});
    // pass two
    put(0x180, R<u64>(object + 0x1538));
    put(0x188, R<u64>(p.rsp + 0x38));
    put(0x190, 0);
    put(0x198, x180);
    put(0x1a0, 0);
    put(0x1a8, b);
    put(0x1b0, cc);
    put(0x1b8, dd);
    put(0x140, R<u64>(object + 0x1568));
    p.steps.push_back({Step::SetUav, 0, 1, 0x140});
    p.steps.push_back({Step::SetSrv, 0, 8, 0x180});
    p.steps.push_back({Step::SetShader, shader2});
    p.steps.push_back({Step::Dispatch, gx, gy, 1, 0x01c6db15, 1});
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R13, gx}, {R15, gy}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 13: the pass before the level chain of `RenderDev_RenderScene`, 0x01c6d199 .. 0x01c6d33e (resumes where the engine counts the levels): context rbx, the holder r14 (five views at
// `+ 0x148`, the size at `+ 0x170`), five shader resources (r12, 0, `[rsp + 0xc8]`, `[rsp + 0xe0]`, `[rsp + 0xd0]` or r12), the shader 0x31 / 0x2f / 0x2e by two flags, a fenced
// dispatch of ((w >> 4) + ((w & 0xe) != 0), (h >> 4) + ..., 1) groups, both cleared. Leaves rbx = holder, r14 = virtual table, r15 = context, r12 = 0, r13b = the flag `[rsp + 0x110] != 0`.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan13(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->rbx;
    const u64 holder = c->r14;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(holder) || !Plausible(device) || !Plausible(p.context)) {
        reason = "an object is not set";
        return false;
    }
    const u64 r12_old = c->r12;
    const u64 chosen = R<u64>(p.rsp + 0xd0);
    const u64 r12_new = chosen != 0 ? chosen : r12_old;
    const u32 index = (R<u8>(Global(0x06af6af4)) & 1) != 0 ? 0x31u : R<u8>(device + 0x18a51) != 0 ? 0x2fu : 0x2eu;
    const u64 shader = R<u64>(device + 0x1008 + (u64{index} << 4));
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u64 size = R<u64>(holder + 0x170);
    const u32 w = static_cast<u32>(size);
    const u32 gx = (w >> 4) + ((w & 0xe) != 0 ? 1u : 0u);
    const u32 gy = static_cast<u32>(size >> 36) + ((size & 0xe00000000ull) != 0 ? 1u : 0u);
    const auto put = [&](u64 offset, u64 value) { p.steps.push_back({Step::PutStack64, offset, value}); };
    put(0x180, r12_old);
    put(0x188, 0);
    put(0x190, R<u64>(p.rsp + 0xc8));
    put(0x198, R<u64>(p.rsp + 0xe0));
    put(0x1a0, r12_new);
    p.steps.push_back({Step::SetSrv, 0, 5, 0x180});
    for (u64 i = 0; i < 5; ++i) {
        put(0x140 + 8 * i, R<u64>(holder + 0x148 + 8 * i));
    }
    p.steps.push_back({Step::SetUav, 0, 5, 0x140});
    p.steps.push_back({Step::SetShader, shader});
    put(0xd8, holder);
    p.steps.push_back({Step::Dispatch, gx, gy, 1, 0x01c6d2d9, 1});
    for (u64 i = 0; i < 5; ++i) {
        put(0x180 + 8 * i, 0);
    }
    p.steps.push_back({Step::SetSrv, 0, 5, 0x180});
    for (u64 i = 0; i < 5; ++i) {
        put(0x140 + 8 * i, 0);
    }
    p.steps.push_back({Step::SetUav, 0, 5, 0x140});
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{RBX, holder}, {R14, R<u64>(p.context)}, {R15, p.context}, {R12, 0}, {R13, (c->r13 & ~0xffull) | (R<u32>(p.rsp + 0x110) != 0 ? 1u : 0u)}};
    return true;
}

/// Writes a step list in the engine's words.
struct StepList {
    Plan& p;
    void put(u64 offset, u64 value) { p.steps.push_back({Step::PutStack64, offset, value}); }
    void put32(u64 offset, u64 value) { p.steps.push_back({Step::PutStack32, offset, value}); }
    void cb(u64 first, u64 count, u64 offset) { p.steps.push_back({Step::SetCb, first, count, offset}); }
    void srv(u64 count, u64 offset, u64 first = 0) { p.steps.push_back({Step::SetSrv, first, count, offset}); }
    void uav(u64 count, u64 offset) { p.steps.push_back({Step::SetUav, 0, count, offset}); }
    void shader(u64 value) { p.steps.push_back({Step::SetShader, value}); }
    void dispatch(u64 x, u64 y, u64 site, bool fenced = true) { p.steps.push_back({Step::Dispatch, x, y, 1, site, fenced ? 1u : 0u}); }
    void cbp(u64 first, u64 count, u64 pointer) { p.steps.push_back({Step::SetCb, first, count, 0, pointer}); }
    void srvp(u64 count, u64 pointer) { p.steps.push_back({Step::SetSrv, 0, count, 0, pointer}); }
    void uavp(u64 count, u64 pointer) { p.steps.push_back({Step::SetUav, 0, count, 0, pointer}); }
    void abs(u64 address, u64 value) { p.steps.push_back({Step::PutAbs64, address, value}); }
    void abs32(u64 address, u64 value) { p.steps.push_back({Step::PutAbs32, address, value}); }
    void abs8(u64 address, u64 value) { p.steps.push_back({Step::PutAbs8, address, value}); }
    void zero(u64 address, u64 size) { p.steps.push_back({Step::PutAbsZero, address, size}); }
    void update(u64 buffer, u64 source) { p.steps.push_back({Step::Update, buffer, source}); }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 14: two passes of `RenderDev_RenderScene` in a row, 0x01c6549f .. 0x01c65693 (context r13, the object r14, r15d and ebx the sizes): SetConstantBuffers (slot 2, `[r14 + 0x1450]`),
// five shader resources and two views, shaders `[device + 0x1818]` and `0x1808`, the first dispatch of (`[rsp + 0x230]`, `[rsp + 0x220]`, 1) groups, the second of (((r15d + 1) / 2 +
// 7) / 8, ((ebx + 1) / 2 + 7) / 8, 1), both cleared at the end. Leaves r15d, ebx the group counts.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan14(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r13;
    const u64 object = c->r14;
    const u64 device = R<u64>(Global(0x0689eed8));
    const u64 owner = R<u64>(p.rsp + 0x240);
    const u64 pointer_290 = R<u64>(p.rsp + 0x290);
    const u64 inner = R<u64>(object + 0x1478);
    if (!Plausible(object) || !Plausible(device) || !Plausible(owner) || !Plausible(pointer_290) || !Plausible(inner)) {
        reason = "an object is not set";
        return false;
    }
    const u64 shader1 = R<u64>(device + 0x1818), shader2 = R<u64>(device + 0x1808);
    if (shader1 == 0 || shader2 == 0) {
        reason = "a shader is not set";
        return false;
    }
    StepList s{p};
    s.put(0x820, R<u64>(object + 0x1450));
    s.cb(2, 1, 0x820);
    s.put(0x860, R<u64>(p.rsp + 0x380));
    s.put(0x868, R<u64>(p.rsp + 0x340));
    s.put(0x870, R<u64>(object + 0x1498));
    s.put(0x878, R<u64>(object + 0x14a0));
    s.put(0x880, 0);
    s.srv(5, 0x860);
    s.put(0x7c0, R<u64>(inner + 0x50));
    s.put(0x7c8, R<u64>(owner));
    s.uav(2, 0x7c0);
    s.shader(shader1);
    s.dispatch(R<u32>(p.rsp + 0x230), R<u32>(p.rsp + 0x220), 0x01c6559c);
    s.put(0x7c8, 0);
    s.put(0x880, R<u64>(pointer_290));
    s.uav(2, 0x7c0);
    s.srv(5, 0x860);
    s.shader(shader2);
    const u32 gx = (((static_cast<u32>(c->r15) + 1u) >> 1) + 7u) >> 3;
    const u32 gy = (((static_cast<u32>(c->rbx) + 1u) >> 1) + 7u) >> 3;
    s.dispatch(gx, gy, 0x01c65638);
    for (u64 i = 0; i < 5; ++i) {
        s.put(0x860 + 8 * i, 0);
    }
    s.srv(5, 0x860);
    s.put(0x7c0, 0);
    s.put(0x7c8, 0);
    s.uav(2, 0x7c0);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R15, gx}, {RBX, gy}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 15: two passes between the targets `[r14 + 0x1460]` and `[r14 + 0x1468]` (context r13), 0x01c65997 .. 0x01c65b34: the first reads the first target and writes the second, the
// second the other way round; two shader resources, one view, shaders `[device + 0x13e8]` and `0x13f8`, groups ((w + 7) / 8, (h + 7) / 8, 1) of the first target's size (`+ 0x2c`, `+ 0x30`).
// Leaves r12 = rsp + 0x840, rbx and r15 the group counts.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan15(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r13;
    const u64 object = c->r14;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(object) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 first = R<u64>(object + 0x1460), second = R<u64>(object + 0x1468);
    if (!Plausible(first) || !Plausible(second)) {
        reason = "a target is not set";
        return false;
    }
    const u64 shader1 = R<u64>(device + 0x13e8), shader2 = R<u64>(device + 0x13f8);
    if (shader1 == 0 || shader2 == 0) {
        reason = "a shader is not set";
        return false;
    }
    const u32 width = R<u32>(first + 0x2c), height = R<u32>(first + 0x30);
    const u32 gx = (width + 7u) >> 3, gy = (height + 7u) >> 3;
    StepList s{p};
    s.put(0x840, 0);
    s.put(0x230, first);
    s.put(0x240, second);
    s.put32(0x250, width);
    s.uav(1, 0x840);
    s.put(0x860, R<u64>(first + 0x48));
    s.put(0x868, R<u64>(first + 0x48));
    s.srv(2, 0x860);
    s.put(0x840, R<u64>(second + 0x50));
    s.uav(1, 0x840);
    s.shader(shader1);
    s.dispatch(gx, gy, 0x01c65a79);
    s.put(0x840, 0);
    s.uav(1, 0x840);
    s.put(0x860, R<u64>(second + 0x48));
    s.put(0x868, R<u64>(second + 0x48));
    s.srv(2, 0x860);
    s.put(0x840, R<u64>(first + 0x50));
    s.uav(1, 0x840);
    s.shader(shader2);
    s.dispatch(gx, gy, 0x01c65b34);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R12, p.rsp + 0x840}, {RBX, gx}, {R15, gy}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 16: two passes of `RenderDev_RenderScene`, 0x01c67e70 .. 0x01c68036 (context r14, rbx the target set `[system + 0x50]`, r13 the object with `+ 0x50`, r15d / r12d the sizes):
// SetConstantBuffers (slot 12), views from `[[rbx + 0x28] + 0x20 / 0x28] + 0x50`, four shader resources, shaders `[device + 0x1148]` and `0x1158`, dispatches of ((w + 7) / 8,
// (h + 7) / 8, 1) groups. Leaves r13 the object's `+ 0x48`, r15 and r12 the group counts.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan16(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    const u64 targets = c->rbx;
    const u64 object = c->r13;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(targets) || !Plausible(object) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 inner = R<u64>(targets + 0x28);
    const u64 view_owner_a = Plausible(inner) ? R<u64>(inner + 0x20) : 0;
    const u64 view_owner_b = Plausible(inner) ? R<u64>(inner + 0x28) : 0;
    const u64 sub = R<u64>(object + 0x50);
    if (!Plausible(inner) || !Plausible(view_owner_a) || !Plausible(view_owner_b) || !Plausible(sub)) {
        reason = "an object is not set";
        return false;
    }
    const u64 new_r13 = R<u64>(sub + 0x48);
    const u64 shader1 = R<u64>(device + 0x1148), shader2 = R<u64>(device + 0x1158);
    if (shader1 == 0 || shader2 == 0) {
        reason = "a shader is not set";
        return false;
    }
    const u64 index = R<u32>(targets + 0x14);
    const u64 tail_owner = R<u64>(targets + index * 8 + 0x18);
    const u64 tail = Plausible(tail_owner) ? R<u64>(tail_owner + 0x48) : 0;
    const u32 gx = (static_cast<u32>(c->r15) + 7u) >> 3;
    const u32 gy = (static_cast<u32>(c->r12) + 7u) >> 3;
    StepList s{p};
    s.put(0x7c0, R<u64>(targets + 0x40));
    s.cb(12, 1, 0x7c0);
    s.put(0x840, R<u64>(view_owner_a + 0x50));
    s.uav(1, 0x840);
    s.put(0x860, 0);
    s.put(0x868, new_r13);
    s.put(0x870, 0);
    s.put(0x878, tail);
    s.srv(4, 0x860);
    s.shader(shader1);
    s.dispatch(gx, gy, 0x01c67f59);
    s.put(0x840, 0);
    s.uav(1, 0x840);
    s.put(0x820, R<u64>(view_owner_b + 0x50));
    s.uav(1, 0x820);
    s.put(0x860, R<u64>(view_owner_a + 0x48));
    s.put(0x868, new_r13);
    s.put(0x870, R<u64>(p.rsp + 0x380));
    s.put(0x878, tail);
    s.srv(4, 0x860);
    s.shader(shader2);
    s.dispatch(gx, gy, 0x01c68036);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R13, new_r13}, {R15, gx}, {R12, gy}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 17: three passes of a system near `0x01c98ee3` (context r14, the object rbx, r15d the width of the second pass, `[rsp + 0x38]` its height): SetConstantBuffers (slot 0,
// `[rbx + 0x150]`), three views, seven shader resources (`[rsp + 0x58 / 0x48 / 0x50]` and four words of the device), the shader `[device + 0x1298]`, a fenced dispatch of (1, 1, 1);
// the shader `0x12a8` and a fenced dispatch of (r15d, `[rsp + 0x38]`, 1); three views again, eight shader resources, the shader `0x12b8` and a plain dispatch of (`[rbx + 0x58]`,
// `[rsp + 0x38]`, 1). Resumes at 0x01c99120. Leaves r13 = `[rsp + 0x48]`, r12 = `[rsp + 0x50]`, r15 = `[rsp + 0x38]`.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan17(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    const u64 object = c->rbx;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(object) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 shader1 = R<u64>(device + 0x1298), shader2 = R<u64>(device + 0x12a8), shader3 = R<u64>(device + 0x12b8);
    if (shader1 == 0 || shader2 == 0 || shader3 == 0) {
        reason = "a shader is not set";
        return false;
    }
    const u64 a58 = R<u64>(p.rsp + 0x58), a48 = R<u64>(p.rsp + 0x48), a50 = R<u64>(p.rsp + 0x50), a38 = R<u64>(p.rsp + 0x38);
    const u64 d4458 = R<u64>(device + 0x4458), d4448 = R<u64>(device + 0x4448), d4460 = R<u64>(device + 0x4460), d4468 = R<u64>(device + 0x4468);
    StepList s{p};
    s.put(0x198, R<u64>(object + 0x150));
    s.cb(0, 1, 0x198);
    s.put(0x180, R<u64>(object + 0x70));
    s.put(0x188, R<u64>(object + 0x98));
    s.put(0x190, R<u64>(object + 0x80));
    s.uav(3, 0x180);
    s.put(0x140, a58);
    s.put(0x148, a48);
    s.put(0x150, a50);
    s.put(0x158, d4458);
    s.put(0x160, d4448);
    s.put(0x168, d4460);
    s.put(0x170, d4468);
    s.srv(7, 0x140);
    s.shader(shader1);
    s.dispatch(1, 1, 0x01c99004);
    s.shader(shader2);
    s.dispatch(static_cast<u32>(c->r15), static_cast<u32>(a38), 0x01c99038);
    s.put(0x120, R<u64>(object + 0x20));
    s.put(0x128, R<u64>(object + 0x48));
    s.put(0x130, R<u64>(object + 0x50));
    s.uav(3, 0x120);
    s.put(0xe0, a58);
    s.put(0xe8, a48);
    s.put(0xf0, a50);
    s.put(0xf8, d4458);
    s.put(0x100, d4448);
    s.put(0x108, d4460);
    s.put(0x110, d4468);
    s.put(0x118, R<u64>(object + 0x68));
    s.srv(8, 0xe0);
    s.shader(shader3);
    s.dispatch(R<u32>(object + 0x58), static_cast<u32>(a38), 0x01c99120, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R13, a48}, {R12, a50}, {R15, a38}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 18: a pass of the system `[r13 + 0x4000 ..]` (context r15, the object r13), 0x01c44e56 .. 0x01c44f64: SetConstantBuffers (slot 11, `[r13 + 0x4170]`), six shader resources from
// slot 100 (five words of the device and `[r13 + 0x4188]`), the shader `[device + 0x10b8]`, one view `[r13 + 0x4198]`, a fenced dispatch of (([r13 + 0x4000] + 63) / 64, 1, 1). Leaves r14 = rsp + 0x118.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan18(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r15;
    const u64 object = c->r13;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(object) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 shader = R<u64>(device + 0x10b8);
    const u32 count = R<u32>(object + 0x4000);
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    if (count == 0) {
        no_work = true;
        return false;
    }
    StepList s{p};
    s.put(0xf8, R<u64>(object + 0x4170));
    s.cb(11, 1, 0xf8);
    s.put(0x120, R<u64>(device + 0x41b0));
    s.put(0x128, R<u64>(device + 0x4188));
    s.put(0x130, R<u64>(device + 0x4190));
    s.put(0x138, R<u64>(device + 0x4198));
    s.put(0x140, R<u64>(device + 0x41a0));
    s.put(0x148, R<u64>(object + 0x4188));
    s.srv(6, 0x120, 100);
    s.shader(shader);
    s.put(0x118, R<u64>(object + 0x4198));
    s.uav(1, 0x118);
    s.dispatch((count + 63u) >> 6, 1, 0x01c44f64);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R14, p.rsp + 0x118}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 19: a pass that writes straight into the graphics context `[device + 0x2d8]` (rbx) and dispatches with the plain function, 0x01cf1e8a .. 0x01cf1f41: the system r15, its job
// `[r15 + 0x360]` (r14). The constant buffer slot 1 and the shader resources 1 .. 3, the views 0 and 1 and the shader are words of the context; the groups are
// (`[[r15 + 0x248] + 0x48]` + 255) / 256. Afterwards the engine's words are zero again.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan19(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->rbx;
    const u64 system = c->r15;
    const u64 job = c->r14;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(system) || !Plausible(job) || !Plausible(device) || !Plausible(p.context)) {
        reason = "an object is not set";
        return false;
    }
    const u64 srv_a = R<u64>(job + 0x10), srv_b = R<u64>(job + 0x18), uav_a = R<u64>(job), uav_b = R<u64>(job + 8);
    const u64 owner = R<u64>(system + 0x248);
    if (!Plausible(srv_a) || !Plausible(srv_b) || !Plausible(uav_a) || !Plausible(uav_b) || !Plausible(owner)) {
        reason = "the job is not set up";
        return false;
    }
    const u64 shader = R<u64>(device + 0x1738);
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u32 groups = (R<u32>(owner + 0x48) + 0xffu) >> 8;
    if (groups == 0) {
        no_work = true;
        return false;
    }
    const u64 ctx = p.context;
    StepList s{p};
    s.abs(ctx + 0x508, R<u64>(job + 0x30));
    s.abs(ctx + 0x1988, R<u64>(srv_a + 0x28));
    s.abs(ctx + 0x1990, R<u64>(srv_b + 0x28));
    s.abs(ctx + 0x1998, R<u64>(device + 0x41b0));
    s.abs(ctx + 0x2098, R<u64>(R<u64>(job) + 0x30));
    s.abs(ctx + 0x20a0, R<u64>(uav_b + 0x30));
    s.abs(ctx + 0x4670, shader);
    s.dispatch(groups, 1, 0x01cf1f17, false);
    s.abs(ctx + 0x508, 0);
    s.abs(ctx + 0x1988, 0);
    s.abs(ctx + 0x1990, 0);
    s.abs(ctx + 0x1998, 0);
    s.abs(ctx + 0x2098, 0);
    s.abs(ctx + 0x20a0, 0);
    if (!Finish(p, ContextKind(ctx))) {
        reason = "the context is not a known context";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 20: a pass of `RenderDev_RenderScene` on the target set `[[r15 + 0x50]]` (context r14), 0x01c640ca .. 0x01c6413d: one view of the target (`[set + [set + 0x14] * 8 + 0x18] + 0x50`),
// the shader `[device + 0x10e8]`, a plain dispatch of ((w + 7) / 8, (h + 7) / 8, 1) groups of the set's size `[set]`; the engine clears the view afterwards. Leaves r15 = the set and
// r12 = rsp + 0x840.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan20(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    const u64 holder = c->r15;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(holder) || !Plausible(device)) {
        reason = "an object is not set";
        return false;
    }
    const u64 set = R<u64>(holder + 0x50);
    if (!Plausible(set)) {
        reason = "the target set is not set";
        return false;
    }
    const u64 target = R<u64>(set + u64{R<u32>(set + 0x14)} * 8 + 0x18);
    if (!Plausible(target)) {
        reason = "the target is not set";
        return false;
    }
    const u64 shader = R<u64>(device + 0x10e8);
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u64 size = R<u64>(set);
    const u32 gx = (static_cast<u32>(size) + 7u) >> 3;
    const u32 gy = (static_cast<u32>(size >> 32) + 7u) >> 3;
    StepList s{p};
    s.put(0x840, R<u64>(target + 0x50));
    s.uav(1, 0x840);
    s.shader(shader);
    s.dispatch(gx, gy, 0x01c6413d, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R15, set}, {R12, p.rsp + 0x840}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 21: the pass after it, 0x01c64484 .. 0x01c6456d (context rbx, the target set r15, the update source r12, r14 the case value of the shader): UpdateSubresource of `[r15 + 0x40]`,
// one view, twelve shader resources of which only the last (`[rsp + 0x8b8]`, the other target of the set) is written here, the shader by the case value (0x13 or r14d + 14 for 1 .. 4),
// a plain dispatch of ((w + 7) / 8, (h + 7) / 8, 1). Leaves r14 = the context.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan21(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->rbx;
    const u64 set = c->r15;
    const u64 device = R<u64>(Global(0x0689eed8));
    if (!Plausible(set) || !Plausible(device) || !Plausible(c->r12)) {
        reason = "an object is not set";
        return false;
    }
    const u32 index = R<u32>(set + 0x14);
    const u64 target = R<u64>(set + u64{index} * 8 + 0x18);
    const u64 other = R<u64>(set + u64{index ^ 1u} * 8 + 0x18);
    if (!Plausible(target) || !Plausible(other)) {
        reason = "a target is not set";
        return false;
    }
    u64 shader_index = 0x13;
    if ((c->r14 >> 32) == 0 && static_cast<u32>(c->r14) - 1u < 4u) {
        shader_index = static_cast<u32>(static_cast<u32>(c->r14) + 0xeu);
    }
    const u64 shader = R<u64>(device + 0x1008 + (shader_index << 4));
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u64 size = R<u64>(set);
    const u32 gx = (static_cast<u32>(size) + 7u) >> 3;
    const u32 gy = (static_cast<u32>(size >> 32) + 7u) >> 3;
    StepList s{p};
    s.update(R<u64>(set + 0x40), c->r12);
    s.put(0x840, R<u64>(target + 0x50));
    s.uav(1, 0x840);
    s.put(0x8b8, R<u64>(other + 0x48));
    s.srv(12, 0x860);
    s.shader(shader);
    s.dispatch(gx, gy, 0x01c6456d, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R14, p.context}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 22: the last dispatch of a RenderScene pass whose bindings the engine's branches set up (several paths join at 0x01c6e0f0): a fenced dispatch of (the first count rounded up by
// 16 when `[rsp + 0xbc]` is 3, by 8 otherwise, (ebx + 7) / 8, 1) groups. Resumes at 0x01c6e128. Leaves r12 and ebx the group counts.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan22(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r14;
    if (!Plausible(p.context)) {
        reason = "the context is not set";
        return false;
    }
    const bool wide = R<u32>(p.rsp + 0xbc) == 3;
    const u32 gx = wide ? (static_cast<u32>(c->r12) + 15u) >> 4 : (static_cast<u32>(c->r12) + 7u) >> 3;
    const u32 gy = (static_cast<u32>(c->rbx) + 7u) >> 3;
    StepList s{p};
    s.dispatch(gx, gy, 0x01c6e128);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R12, gx}, {RBX, gy}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 23: the scene job `0x01c6018a .. 0x01c6024c` of `RenderDev_RenderScene` (graphics context r13, the system `[rsp + 0x3c8]` in rbx, r15d the element count, nonzero): two
// constant buffers from slot 12 (`[rbx + 0x3980]`, `[rbx + 0x3b80]`), three words of the system (count, 0, 0), UpdateSubresource of `[rbx + 0x3b80]` from `rbx + 0x3990`, the shader
// `[device + 0x1838]` and a plain dispatch of ((count + 63) / 64, 1, 1). Leaves r15 = the groups, r12 = rbx + 0x3990.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan23(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    p.context = c->r13;
    const u64 system = R<u64>(p.rsp + 0x3c8);
    const u64 device = R<u64>(Global(0x0689eed8));
    const u32 count = static_cast<u32>(c->r15);
    if (!Plausible(system) || !Plausible(device) || count == 0) {
        reason = "an object is not set";
        return false;
    }
    const u64 shader = R<u64>(device + 0x1838);
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u32 gx = (count + 63u) >> 6;
    StepList s{p};
    s.put(0x820, R<u64>(system + 0x3980));
    s.put(0x828, R<u64>(system + 0x3b80));
    s.cb(12, 2, 0x820);
    s.abs32(system + 0x3b30, count);
    s.abs(system + 0x3b34, 0);
    s.abs32(system + 0x3b3c, 0);
    s.update(R<u64>(system + 0x3b80), system + 0x3990);
    s.shader(shader);
    s.dispatch(gx, 1, 0x01c6024f, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R15, gx}, {R12, system + 0x3990}, {RBX, system}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Site 24: a pass that writes straight into the graphics context `[rbx + 0x2d8]` (rbx the device, r12 `[rsp + 0xb8]` the target set, 0x01c636dc .. 0x01c63811, fenced dispatch with the
// plain call `0x02479810`): the context's bookkeeping words at `+ 0x4484 ..` are cleared and set, the view `[[r12 + 0x20] + 0x50]`, the shader resources from slot 0 (`[rsp + 0x320]`,
// r13, `[rsp + 0x2e0]`, `[rsp + 0x2d0]`, `[rsp + 0x2c0]`, 0, rsi, zeros, r11, r9, r10, r8), the shader `[rbx + (rax << 4) + 0x1008]`, groups ((w + 7) / 8, (h + 7) / 8, 1).
// Leaves r12 = the set, r13 = rbx, r15 = rdx.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool Scan24(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    (void)no_work;
    p.rsp = c->rsp;
    const u64 device = c->rbx;
    const u64 set = R<u64>(p.rsp + 0xb8);
    if (!Plausible(device) || !Plausible(set)) {
        reason = "an object is not set";
        return false;
    }
    const u64 ctx = R<u64>(device + 0x2d8);
    p.context = ctx;
    const u64 view_owner = R<u64>(set + 0x20);
    if (!Plausible(ctx) || !Plausible(view_owner)) {
        reason = "the context or the target is not set";
        return false;
    }
    const u64 shader = R<u64>(device + (c->rax << 4) + 0x1008);
    if (shader == 0) {
        reason = "the shader is not set";
        return false;
    }
    const u32 gx = (R<u32>(set + 0x84) + 7u) >> 3;
    const u32 gy = (R<u32>(set + 0x88) + 7u) >> 3;
    StepList s{p};
    s.zero(ctx + 0x44a4, 32);
    s.zero(ctx + 0x4484, 32);
    s.abs32(ctx + 0x44c4, 0);
    s.abs8(ctx + 0x44c8, 1);
    s.abs(ctx + 0x44d0, 0);
    s.abs8(ctx + 0x44d8, 1);
    s.abs(ctx + 0x2098, R<u64>(view_owner + 0x50));
    s.abs(ctx + 0x1980, R<u64>(p.rsp + 0x320));
    s.abs(ctx + 0x1988, c->r13);
    s.abs(ctx + 0x1990, R<u64>(p.rsp + 0x2e0));
    s.abs(ctx + 0x1998, R<u64>(p.rsp + 0x2d0));
    s.abs(ctx + 0x19a0, R<u64>(p.rsp + 0x2c0));
    s.abs(ctx + 0x19a8, 0);
    s.abs(ctx + 0x19b0, c->rsi);
    s.zero(ctx + 0x19e8, 32);
    s.zero(ctx + 0x19d8, 32);
    s.zero(ctx + 0x19b8, 32);
    s.abs(ctx + 0x1a08, c->r11);
    s.abs(ctx + 0x1a10, c->r9);
    s.abs(ctx + 0x1a18, c->r10);
    s.abs(ctx + 0x1a20, c->r8);
    s.abs(ctx + 0x4670, shader);
    s.dispatch(gx, gy, 0x01c63816);
    if (!Finish(p, ContextKind(ctx))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R12, set}, {R13, device}, {R15, c->rdx}};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Sites 25 and 26: the other two dispatch loops of the emitter system `FUN_01bcac40` (graphics context r14, the system `[rsp + 0x80]`, the tables behind the pointers `[rsp + 0x88]`
// (constant buffer, slot 10), `[rsp + 0x90]` (two views), `[rsp + 0xa8]` (four shader resources)). Loop B (0x01bcbc29 .. 0x01bcbd38, node rbx, only for `[data + 0x58] != 0`, shader
// 0x1c at `[device + 0x11c8]`, size `[data + 0x50]`) and loop D (0x01bcc075 .. 0x01bcc200, node rbx, data r12, shader 0x1a at `[device + 0x11a8]`, size `[data + 0x80]`, the stack slots
// 0xb8 / 0xc0 / 0xc8 / 0xf0 / 0xf8 from the pair of targets), both plain dispatches of ((w + 7) / 8, (h + 7) / 8, 1).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
u64 TargetWord(u64 data, u32 index, u64 offset) {
    const u64 target = R<u64>(data + u64{index} * 8 + 0x1c8);
    return target != 0 ? R<u64>(target + offset) : 0;
}
bool Scan25(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    const u64 node = c->rbx;
    const u64 data = Plausible(node) ? R<u64>(node + 0x108) : 0;
    if (!EmitterPreamble(c, p, node, data, reason, no_work)) {
        return false;
    }
    if (R<u32>(data + 0x58) == 0) {
        no_work = true;
        return false;
    }
    const u64 system = R<u64>(p.rsp + 0x80);
    const u64 device = R<u64>(Global(0x0689eed8));
    const u64 pointer_a = R<u64>(p.rsp + 0x88), pointer_b = R<u64>(p.rsp + 0x90), pointer_c = R<u64>(p.rsp + 0xa8);
    if (!Plausible(system) || !Plausible(device) || !Plausible(pointer_a) || !Plausible(pointer_b) || !Plausible(pointer_c)) {
        reason = "the table pointers of the function are not set";
        return false;
    }
    const u32 a = R<u32>(data + 0x24c), b = R<u32>(data + 0x250);
    StepList s{p};
    s.put(0xb8, R<u64>(data + 0x178));
    s.put(0xf0, TargetWord(data, (a ^ 1u) + b * 2u, 0x50));
    s.cbp(10, 1, pointer_a);
    s.uavp(2, pointer_b);
    s.srvp(4, pointer_c);
    if (R<u32>(system + 0x78) != 0x1c) {
        const u64 shader = R<u64>(device + 0x11c8);
        if (shader == 0) {
            reason = "the shader is not set";
            return false;
        }
        s.abs32(system + 0x78, 0x1c);
        s.shader(shader);
    }
    const u64 size = R<u64>(data + 0x50);
    s.dispatch((static_cast<u32>(size) + 7u) >> 3, (static_cast<u32>(size >> 32) + 7u) >> 3, 0x01bcbd38, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    p.registers = {{R15, data}};
    return true;
}
bool Scan26(MidContext* c, Plan& p, const char*& reason, bool& no_work) {
    const u64 node = c->rbx;
    const u64 data = c->r12;
    if (!EmitterPreamble(c, p, node, data, reason, no_work)) {
        return false;
    }
    const u64 system = R<u64>(p.rsp + 0x80);
    const u64 device = R<u64>(Global(0x0689eed8));
    const u64 pointer_a = R<u64>(p.rsp + 0x88), pointer_b = R<u64>(p.rsp + 0x90), pointer_c = R<u64>(p.rsp + 0xa8);
    if (!Plausible(system) || !Plausible(device) || !Plausible(pointer_a) || !Plausible(pointer_b) || !Plausible(pointer_c)) {
        reason = "the table pointers of the function are not set";
        return false;
    }
    const u32 a = R<u32>(data + 0x24c), b = R<u32>(data + 0x250);
    StepList s{p};
    s.put(0xb8, R<u64>(data + 0x1a8));
    s.put(0xc0, TargetWord(data, b * 2u + a, 0x48));
    s.put(0xc8, TargetWord(data, (a ^ 1u) + b * 2u, 0x48));
    s.put(0xf0, TargetWord(data, ((b * 2u) ^ 2u) + a, 0x50));
    s.put(0xf8, TargetWord(data, ((b * 2u) ^ 2u) + (a ^ 1u), 0x50));
    s.cbp(10, 1, pointer_a);
    s.uavp(2, pointer_b);
    s.srvp(4, pointer_c);
    if (R<u32>(system + 0x78) != 0x1a) {
        const u64 shader = R<u64>(device + 0x11a8);
        if (shader == 0) {
            reason = "the shader is not set";
            return false;
        }
        s.abs32(system + 0x78, 0x1a);
        s.shader(shader);
    }
    const u64 size = R<u64>(data + 0x80);
    s.dispatch((static_cast<u32>(size) + 7u) >> 3, (static_cast<u32>(size >> 32) + 7u) >> 3, 0x01bcc200, false);
    if (!Finish(p, ContextKind(p.context))) {
        reason = "the context is not a known context";
        return false;
    }
    return true;
}

std::array<Site, 27>& Sites() {
    static std::array<Site, 27> sites{{
        {"begin_recording_loop_1", 0x01c49e40, &Scan0, &Apply0, {}},
        {"begin_recording_loop_2", 0x01c4a9e3, &Scan1, &Apply1, {}},
        {"vertex_cluster_loop_1", 0x01d7e334, &Scan2, &Apply2, {}},
        {"vertex_cluster_loop_2", 0x01d7e5b0, &Scan3, &Apply3, {}},
        {"emitter_system_loop_e", 0x01bcc400, &Scan4, &Apply4, {}},
        {"emitter_system_loop_c", 0x01bcbf02, &Scan5, &Apply5, {}},
        {"emitter_system_loop_a", 0x01bcbb60, &Scan6, &Apply6, {}},
        {"vertex_cluster_pre", 0x01d7e1c0, &Scan7, &Apply7, {}},
        {"begin_recording_loop_3", 0x01c4b6ea, &Scan8, &Apply8, {}},
        {"begin_recording_loop_4", 0x01c4bea6, &Scan9, &Apply9, {}},
        {"render_scene_chain", 0x01c6d7ea, &Scan10, &Apply10, {}},
        {"render_scene_pass_a", 0x01c6cfc9, &Scan11, &ApplySteps, {}},
        {"render_scene_pass_b", 0x01c6db15, &Scan12, &ApplySteps, {}},
        {"render_scene_pre_chain", 0x01c6d33e, &Scan13, &ApplySteps, {}},
        {"render_scene_pass_d", 0x01c65693, &Scan14, &ApplySteps, {}},
        {"render_scene_pass_e", 0x01c65b34, &Scan15, &ApplySteps, {}},
        {"render_scene_pass_f", 0x01c68036, &Scan16, &ApplySteps, {}},
        {"system_passes_99", 0x01c99120, &Scan17, &ApplySteps, {}},
        {"system_pass_44", 0x01c44f64, &Scan18, &ApplySteps, {}},
        {"system_pass_cf", 0x01cf1f41, &Scan19, &ApplySteps, {}},
        {"render_scene_pass_g", 0x01c6413d, &Scan20, &ApplySteps, {}},
        {"render_scene_pass_h", 0x01c6456d, &Scan21, &ApplySteps, {}},
        {"render_scene_dispatch_tail", 0x01c6e128, &Scan22, &ApplySteps, {}},
        {"scene_job_1c60", 0x01c6024f, &Scan23, &ApplySteps, {}},
        {"render_scene_pass_i", 0x01c63816, &Scan24, &ApplySteps, {}},
        {"emitter_system_loop_b", 0x01bcbc10, &Scan25, &ApplySteps, {}},
        {"emitter_system_loop_d", 0x01bcc200, &Scan26, &ApplySteps, {}},
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
    u32 site{0}, dispatch_count{0};
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
        pending.dispatch_count = 0;
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
    } else if (!pending.plan.dispatches.empty() && pending.dispatch_count != pending.plan.dispatches.size()) {
        fail("dispatch count", fmt::format("{} dispatches, expected {}", pending.dispatch_count, pending.plan.dispatches.size()));
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

namespace {
struct Census {
    bool on{std::getenv("TLG_SITE_CENSUS") != nullptr && *std::getenv("TLG_SITE_CENSUS") == '1'};
    std::mutex mutex;
    std::map<u64, std::pair<u64, u64>> sites; ///< call site -> (issued by the engine's code, issued by a generator)
};
Census& GetCensus() {
    static Census census;
    return census;
}
} // namespace

void CensusDispatch(u64 site_runtime, bool generated) {
    Census& census = GetCensus();
    if (!census.on) {
        return;
    }
    std::scoped_lock lock{census.mutex};
    auto& entry = census.sites[site_runtime];
    ++(generated ? entry.second : entry.first);
}

void NoteDispatch(u64 context, u64 x, u64 y, u64 z, u64 site_runtime) {
    if (!pending.armed || context != pending.plan.context) {
        return;
    }
    const Plan& plan = pending.plan;
    if (!plan.dispatches.empty()) {
        const u32 index = pending.dispatch_count++;
        pending.dispatch_seen = true;
        if (index >= plan.dispatches.size()) {
            pending.notes.push_back(fmt::format("dispatch {} is more than the {} expected", index, plan.dispatches.size()));
            return;
        }
        const Expect& e = plan.dispatches[index];
        if (x != e.x || y != e.y || z != e.z) {
            pending.notes.push_back(fmt::format("dispatch {}: groups {} x {} x {}, expected {} x {} x {}", index, x, y, z, e.x, e.y, e.z));
        }
        if (site_runtime != Global(e.site)) {
            pending.notes.push_back(fmt::format("dispatch {}: call site {:#x}, expected {:#x}", index, site_runtime, Global(e.site)));
        }
        for (const auto& [offset, value] : e.words) {
            if (R<u64>(context + offset) != value) {
                pending.notes.push_back(fmt::format("dispatch {}: context word {:#x}: {:#x}, expected {:#x}", index, offset, R<u64>(context + offset), value));
            }
        }
        for (const auto& [offset, value] : e.stack) {
            if (R<u64>(plan.rsp + offset) != value) {
                pending.notes.push_back(fmt::format("dispatch {}: stack word {:#x}: {:#x}, expected {:#x}", index, offset, R<u64>(plan.rsp + offset), value));
            }
        }
        return;
    }
    if (pending.dispatch_seen) {
        return;
    }
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
    if (n == "site_top_2") {
        return reinterpret_cast<u64>(&Top<2>);
    }
    if (n == "site_bottom_2") {
        return reinterpret_cast<u64>(&Bottom<2>);
    }
    if (n == "site_top_3") {
        return reinterpret_cast<u64>(&Top<3>);
    }
    if (n == "site_bottom_3") {
        return reinterpret_cast<u64>(&Bottom<3>);
    }
    if (n == "site_top_4") {
        return reinterpret_cast<u64>(&Top<4>);
    }
    if (n == "site_bottom_4") {
        return reinterpret_cast<u64>(&Bottom<4>);
    }
    if (n == "site_top_5") {
        return reinterpret_cast<u64>(&Top<5>);
    }
    if (n == "site_bottom_5") {
        return reinterpret_cast<u64>(&Bottom<5>);
    }
    if (n == "site_top_6") {
        return reinterpret_cast<u64>(&Top<6>);
    }
    if (n == "site_bottom_6") {
        return reinterpret_cast<u64>(&Bottom<6>);
    }
    if (n == "site_top_7") {
        return reinterpret_cast<u64>(&Top<7>);
    }
    if (n == "site_bottom_7") {
        return reinterpret_cast<u64>(&Bottom<7>);
    }
    if (n == "site_top_8") {
        return reinterpret_cast<u64>(&Top<8>);
    }
    if (n == "site_bottom_8") {
        return reinterpret_cast<u64>(&Bottom<8>);
    }
    if (n == "site_top_9") {
        return reinterpret_cast<u64>(&Top<9>);
    }
    if (n == "site_bottom_9") {
        return reinterpret_cast<u64>(&Bottom<9>);
    }
    if (n == "site_top_10") {
        return reinterpret_cast<u64>(&Top<10>);
    }
    if (n == "site_bottom_10") {
        return reinterpret_cast<u64>(&Bottom<10>);
    }
    if (n == "site_top_11") {
        return reinterpret_cast<u64>(&Top<11>);
    }
    if (n == "site_bottom_11") {
        return reinterpret_cast<u64>(&Bottom<11>);
    }
    if (n == "site_top_12") {
        return reinterpret_cast<u64>(&Top<12>);
    }
    if (n == "site_bottom_12") {
        return reinterpret_cast<u64>(&Bottom<12>);
    }
    if (n == "site_top_13") {
        return reinterpret_cast<u64>(&Top<13>);
    }
    if (n == "site_bottom_13") {
        return reinterpret_cast<u64>(&Bottom<13>);
    }
    if (n == "site_top_14") {
        return reinterpret_cast<u64>(&Top<14>);
    }
    if (n == "site_bottom_14") {
        return reinterpret_cast<u64>(&Bottom<14>);
    }
    if (n == "site_top_15") {
        return reinterpret_cast<u64>(&Top<15>);
    }
    if (n == "site_bottom_15") {
        return reinterpret_cast<u64>(&Bottom<15>);
    }
    if (n == "site_top_16") {
        return reinterpret_cast<u64>(&Top<16>);
    }
    if (n == "site_bottom_16") {
        return reinterpret_cast<u64>(&Bottom<16>);
    }
    if (n == "site_top_17") {
        return reinterpret_cast<u64>(&Top<17>);
    }
    if (n == "site_bottom_17") {
        return reinterpret_cast<u64>(&Bottom<17>);
    }
    if (n == "site_top_18") {
        return reinterpret_cast<u64>(&Top<18>);
    }
    if (n == "site_bottom_18") {
        return reinterpret_cast<u64>(&Bottom<18>);
    }
    if (n == "site_top_19") {
        return reinterpret_cast<u64>(&Top<19>);
    }
    if (n == "site_bottom_19") {
        return reinterpret_cast<u64>(&Bottom<19>);
    }
    if (n == "site_top_20") {
        return reinterpret_cast<u64>(&Top<20>);
    }
    if (n == "site_bottom_20") {
        return reinterpret_cast<u64>(&Bottom<20>);
    }
    if (n == "site_top_21") {
        return reinterpret_cast<u64>(&Top<21>);
    }
    if (n == "site_bottom_21") {
        return reinterpret_cast<u64>(&Bottom<21>);
    }
    if (n == "site_top_22") {
        return reinterpret_cast<u64>(&Top<22>);
    }
    if (n == "site_bottom_22") {
        return reinterpret_cast<u64>(&Bottom<22>);
    }
    if (n == "site_top_23") {
        return reinterpret_cast<u64>(&Top<23>);
    }
    if (n == "site_bottom_23") {
        return reinterpret_cast<u64>(&Bottom<23>);
    }
    if (n == "site_top_24") {
        return reinterpret_cast<u64>(&Top<24>);
    }
    if (n == "site_bottom_24") {
        return reinterpret_cast<u64>(&Bottom<24>);
    }
    if (n == "site_top_25") {
        return reinterpret_cast<u64>(&Top<25>);
    }
    if (n == "site_bottom_25") {
        return reinterpret_cast<u64>(&Bottom<25>);
    }
    if (n == "site_top_26") {
        return reinterpret_cast<u64>(&Top<26>);
    }
    if (n == "site_bottom_26") {
        return reinterpret_cast<u64>(&Bottom<26>);
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
    if (Census& census = GetCensus(); census.on) {
        nlohmann::json rows = nlohmann::json::array();
        std::scoped_lock lock{census.mutex};
        for (const auto& [site, counts] : census.sites) {
            nlohmann::json r;
            r["call_site"] = fmt::format("{:#x}", site - NativeDraw::ImageBase() + 0x1000000);
            r["engine"] = counts.first;
            r["generated"] = counts.second;
            rows.push_back(std::move(r));
        }
        out["dispatch_census"] = std::move(rows);
    }
    return out;
}

} // namespace Core::NativeSiteGen
