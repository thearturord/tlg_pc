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

std::array<Site, 5>& Sites() {
    static std::array<Site, 5> sites{{
        {"begin_recording_loop_1", 0x01c49e40, &Scan0, &Apply0, {}},
        {"begin_recording_loop_2", 0x01c4a9e3, &Scan1, &Apply1, {}},
        {"vertex_cluster_loop_1", 0x01d7e334, &Scan2, &Apply2, {}},
        {"vertex_cluster_loop_2", 0x01d7e5b0, &Scan3, &Apply3, {}},
        {"emitter_system_loop_e", 0x01bcc400, &Scan4, &Apply4, {}},
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
