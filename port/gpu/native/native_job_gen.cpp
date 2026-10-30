// SPDX-License-Identifier: GPL-2.0-or-later
// B7: see native_job_gen.h. Ghidra: RenderScene 0x01c606f0 .. 0x01c60bf6 (the loop over the scene's small async jobs), read from the disassembly (stages/b7-scene-jobs.md).

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "native/native_hooks.h"
#include "native/native_draw.h"
#include "native/native_job_gen.h"

namespace Core::NativeJobGen {

namespace {

constexpr u64 LoopTop = 0x01c606f0, LoopAfterBody = 0x01c60beb, LoopResume = 0x01c60bf3, DispatchSite = 0x01c60bb2;
constexpr u64 DeviceGlobal = 0x0689eed8; // the render device (shader list at +0x1990 / +0x1998, default shader table at +0x1008, index at +0x19a8)
constexpr u64 ShaderKeyFunction = 0x01c90b70; // stored at [rsp + 0x840] for the creation path (the generator keeps the stack the way the loop leaves it)

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
    u32 verify_every{0}; // off by default since 2026-10-30 (TLG_JOB_GEN_VERIFY=N)
};
const Config& Cfg() {
    static const Config config = [] {
        Config c;
        const char* e = std::getenv("TLG_JOB_GEN");
        c.on = e == nullptr || *e != '0'; // default on since 2026-10-27 (all native switches); TLG_JOB_GEN=0 turns it off
        if (const char* n = std::getenv("TLG_JOB_GEN_VERIFY"); n != nullptr && *n != 0) {
            c.verify_every = static_cast<u32>(std::atoi(n));
        }
        return c;
    }();
    return config;
}

struct Stats {
    std::atomic<u64> iterations{0}, no_work{0}, generated{0}, declined{0}, dispatches{0};
    std::atomic<u64> verify_iterations{0}, verify_clean{0}, verify_differences{0}, verify_raced{0}, scan_faults{0};
    std::atomic<bool> switched_off{false};
    std::mutex mutex;
    std::map<std::string, u64> decline_reasons;
    std::map<std::string, u64> difference_kinds;
    std::vector<std::string> samples;
};
Stats& S() {
    static Stats stats;
    return stats;
}

/// What one iteration of the loop would do, read from the engine's objects.
struct Iteration {
    u64 rsp{}, holder{}, context{}, entry{}, object{}, block{}, device{}, shader_entry{}, shader{};
    u64 frame{}, staged_368{}, staged_370{}, staged_380{};
    u32 index{}, count{}, dword_c{}, id{}, groups{};
    u64 views[4]{};
    u64 constant_buffer{}; ///< the resource the staged block is sent to ([frame + 0x3b80])
};

/// A write the loop makes into memory: recorded (verify) or done (generation).
struct Patch {
    u64 address;
    std::vector<u8> bytes;
};

/// Where the iteration's writes go.
class Writer {
public:
    explicit Writer(bool live) : live_{live} {}
    void Put(u64 address, const void* data, size_t size) {
        if (live_) {
            std::memcpy(reinterpret_cast<void*>(address), data, size);
        } else {
            // a later write to the same range replaces the earlier one (the table on the stack is set and reset): only the last value is what the memory holds afterwards
            for (auto& patch : patches) {
                if (patch.address == address && patch.bytes.size() == size) {
                    std::memcpy(patch.bytes.data(), data, size);
                    return;
                }
            }
            patches.push_back({address, std::vector<u8>(static_cast<const u8*>(data), static_cast<const u8*>(data) + size)});
        }
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

void Decline(const char* reason) {
    S().declined.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{S().mutex};
    ++S().decline_reasons[reason];
}

/// Reads everything the iteration needs. False: the engine runs the iteration (the reason is counted); `no_work` for the iterations that do nothing.
struct ScanArgs {
    GuestHooks::MidContext* c;
    Iteration* it;
    const char* reason;
    bool no_work;
    bool ok;
};
void ScanBody(void* p) {
    auto& a = *static_cast<ScanArgs*>(p);
    Iteration& it = *a.it;
    it.rsp = a.c->rsp;
    it.holder = a.c->rdi;
    it.context = a.c->r13;
    it.index = static_cast<u32>(a.c->r14);
    if (!Plausible(it.holder) || !Plausible(it.context)) {
        a.reason = "the holder or the context is not a heap pointer";
        return;
    }
    const u64 list = R<u64>(it.holder);
    if (!Plausible(list)) {
        a.reason = "no job list";
        return;
    }
    it.entry = list + static_cast<u64>(static_cast<s64>(static_cast<s32>(it.index))) * 24;
    it.count = R<u32>(it.entry + 8);
    if (it.count + 1u < 2u) { // the loop skips an entry whose count is 0 or -1
        a.no_work = true;
        return;
    }
    it.dword_c = R<u32>(it.entry + 12);
    it.object = R<u64>(it.entry);
    if (!Plausible(it.object)) {
        a.reason = "the entry has no object";
        return;
    }
    it.block = R<u64>(it.object + 0x5b0);
    if (!Plausible(it.block)) {
        a.reason = "the object has no system block";
        return;
    }
    it.id = R<u32>(it.block + 8);
    it.frame = R<u64>(it.rsp + 0x3c8);
    it.staged_368 = R<u64>(it.rsp + 0x368);
    it.staged_370 = R<u64>(it.rsp + 0x370);
    it.staged_380 = R<u64>(it.rsp + 0x380);
    if (!Plausible(it.frame) || !Plausible(it.staged_368) || !Plausible(it.staged_370) || !Plausible(it.staged_380)) {
        a.reason = "the staging pointers of the loop are not set";
        return;
    }
    it.constant_buffer = R<u64>(it.frame + 0x3b80);
    // the vtable of the context: UpdateSubresource, SetConstantBuffers, SetShader, Dispatch
    const u64 vtable = R<u64>(it.context);
    if (!Plausible(vtable) || !Plausible(R<u64>(vtable + 0xd0)) || !Plausible(R<u64>(vtable + 0x28)) || !Plausible(R<u64>(vtable + 0x38))) {
        a.reason = "the context has no virtual table";
        return;
    }
    it.device = R<u64>(Global(DeviceGlobal));
    if (!Plausible(it.device)) {
        a.reason = "no render device";
        return;
    }
    // the shader of the job's id in the device's list; a miss is the first use (the loop creates the shader: the engine's)
    const u32 listed = R<u32>(it.device + 0x1998);
    const u64 shaders = R<u64>(it.device + 0x1990);
    if (listed == 0 || !Plausible(shaders)) {
        a.reason = "first use: the device has no shader list yet";
        return;
    }
    for (u32 i = 0; i < listed; ++i) {
        const u64 entry = R<u64>(shaders + u64{i} * 8);
        if (Plausible(entry) && R<u32>(entry + 0x40) == it.id) {
            it.shader_entry = entry;
            break;
        }
    }
    if (it.shader_entry == 0) {
        a.reason = "first use: the shader of the job is not in the list";
        return;
    }
    if (R<u32>(it.shader_entry + 0x30) == 0) {
        it.shader = R<u64>(it.shader_entry + 0x28);
    } else {
        const u64 slot = u64{R<u32>(it.device + 0x19a8)} << 4;
        it.shader = R<u64>(it.device + slot + 0x1008);
    }
    // the four resource views set as the constant buffers (the fourth by the flag bits of the block)
    it.views[0] = R<u64>(it.block + 0x70);
    it.views[1] = R<u64>(it.block + 0x110);
    it.views[2] = R<u64>(it.block + 0x150);
    const u32 flags = R<u32>(it.block + 8);
    it.views[3] = (flags & 8) != 0 ? R<u64>(it.block + 0x1b0) : (flags & 0x10) != 0 ? R<u64>(it.block + 0x1f0) : 0;
    it.groups = (it.count + 63u) >> 6;
    a.ok = true;
}

bool Scan(GuestHooks::MidContext* c, Iteration& it, const char*& reason, bool& no_work) {
    ScanArgs args{c, &it, "scan faulted", false, false};
    if (!NativeDraw::GuardedCall(&ScanBody, &args)) {
        S().scan_faults.fetch_add(1, std::memory_order_relaxed);
        reason = "scan faulted";
        return false;
    }
    reason = args.reason;
    no_work = args.no_work;
    return args.ok;
}

/// The writes of the iteration into the staged block, the stack slots the loop uses and the frame; with `writer.Live()` also the engine calls and the dispatch.
void Apply(const Iteration& it, Writer& w) {
    const u64 object = it.object;
    const u64 frame = it.frame;
    const u32 counted = R<u32>(object + 0x2c4) == 0 ? R<u32>(object + 0x2b4) : R<u32>(object + 0x2c0);
    const float converted = static_cast<float>(static_cast<s64>(u64{counted}));
    const float radius = R<float>(object + 0xa4);
    const float squared = radius * radius;
    w.Put<u32>(frame + 0x3a40, R<u32>(object + 0x2c8));
    w.Put<float>(frame + 0x3a44, converted);
    w.Put<float>(frame + 0x3a48, squared);
    w.Put<u32>(frame + 0x3a4c, R<u32>(object + 0x5bc));
    u8 sixteen[16], thirty_two_a[32], thirty_two_b[32];
    std::memcpy(sixteen, reinterpret_cast<const void*>(object + 0x540), 16);
    w.Put(it.staged_380, sixteen, 16);
    w.Put<u32>(frame + 0x3a60, R<u32>(object + 0x5c0));
    w.Put<u32>(frame + 0x3a64, R<u32>(object + 0x5c4));
    w.Put<u32>(frame + 0x3a68, 0);
    w.Put<u32>(frame + 0x3a6c, 0);
    std::memcpy(thirty_two_a, reinterpret_cast<const void*>(object + 0x2d0), 32);
    std::memcpy(thirty_two_b, reinterpret_cast<const void*>(object + 0x2f0), 32);
    w.Put(it.staged_368 + 0x20, thirty_two_b, 32);
    w.Put(it.staged_368, thirty_two_a, 32);
    w.Put<u32>(frame + 0x3b30, it.count);
    w.Put<u32>(frame + 0x3b34, counted);
    w.Put<u32>(frame + 0x3b38, it.dword_c);
    w.Put<u32>(frame + 0x3b3c, 0);
    w.Put<u32>(it.rsp, 0); // the seventh argument slot of the UpdateSubresource call
    using Virtual = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
    const u64 vtable = R<u64>(it.context);
    if (w.Live()) {
        // UpdateSubresource(context, constant buffer, 0, 0, staged block, 0, 0)
        reinterpret_cast<Virtual>(R<u64>(vtable + 0xd0))(it.context, it.constant_buffer, 0, 0, it.staged_370, 0, 0);
    }
    // the table of the four views on the stack, then SetConstantBuffers(context, 1, 4, table)
    alignas(32) u64 table[4] = {it.views[0], it.views[1], it.views[2], it.views[3]};
    w.Put(it.rsp + 0x820, table, 32);
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(it.context, 1, 4, it.rsp + 0x820, 0, 0, 0);
    }
    // the key of the shader search (kept on the stack as the loop leaves it)
    w.Put<u64>(it.rsp + 0x840, Global(ShaderKeyFunction));
    w.Put<u32>(it.rsp + 0x848, it.id);
    w.Put<u32>(it.rsp + 0x84c, 0);
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x38))(it.context, it.shader, 0, 0, 0, 0, 0); // SetShader
        NativeDraw::IssueDispatch(it.context, it.groups, 1, 1, false, DispatchSite, true);
    }
    // the resources and the shader are reset
    alignas(32) u64 zeros[4] = {0, 0, 0, 0};
    w.Put(it.rsp + 0x820, zeros, 32);
    if (w.Live()) {
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x28))(it.context, 1, 4, it.rsp + 0x820, 0, 0, 0);
        reinterpret_cast<Virtual>(R<u64>(vtable + 0x38))(it.context, 0, 0, 0, 0, 0, 0);
    }
}

using ApplyArgs = std::pair<const Iteration*, Writer*>;
void ApplyThunk(void* p) {
    auto* a = static_cast<ApplyArgs*>(p);
    Apply(*a->first, *a->second);
}

/// A verify iteration: what the generator predicts, kept until the engine's iteration is over.
struct Pending {
    bool armed{false};
    bool dispatch_seen{false};
    Iteration it;
    std::vector<Patch> patches;
    u64 table[4]{};
    std::vector<std::string> notes; // differences found at the dispatch
};
thread_local Pending pending;

void NoteDifference(const std::string& kind, const std::string& text) {
    S().verify_differences.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{S().mutex};
    ++S().difference_kinds[kind];
    if (S().samples.size() < 16) {
        S().samples.push_back(kind + ": " + text);
    }
}

} // namespace

bool Enabled() {
    return Cfg().on && !S().switched_off.load(std::memory_order_relaxed);
}

u64 PS4_SYSV_ABI JobLoopTop(GuestHooks::MidContext* c) {
    const Config& cfg = Cfg();
    Stats& s = S();
    if (!cfg.on || s.switched_off.load(std::memory_order_relaxed) || pending.armed) {
        return 0;
    }
    s.iterations.fetch_add(1, std::memory_order_relaxed);
    Iteration it;
    const char* reason = "";
    bool no_work = false;
    if (!Scan(c, it, reason, no_work)) {
        if (no_work) {
            s.no_work.fetch_add(1, std::memory_order_relaxed);
        } else {
            Decline(reason);
        }
        return 0;
    }
    const u64 eligible = s.generated.load(std::memory_order_relaxed) + s.verify_iterations.load(std::memory_order_relaxed);
    const bool verify = cfg.verify_every != 0 && eligible % cfg.verify_every == cfg.verify_every - 1;
    if (verify) {
        // the engine runs this iteration; the prediction is compared with what it did (the dispatch as it is issued, the memory and the registers after the body)
        s.verify_iterations.fetch_add(1, std::memory_order_relaxed);
        Writer shadow{false};
        ApplyArgs args{&it, &shadow};
        if (!NativeDraw::GuardedCall(&ApplyThunk, &args)) {
            s.scan_faults.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        pending.armed = true;
        pending.dispatch_seen = false;
        pending.it = it;
        pending.patches = std::move(shadow.patches);
        pending.notes.clear();
        for (int i = 0; i < 4; ++i) {
            pending.table[i] = it.views[i];
        }
        return 0;
    }
    // generation: the engine's iteration is replaced
    bool faulted = false;
    {
        Writer live{true};
        ApplyArgs args{&it, &live};
        faulted = !NativeDraw::GuardedCall(&ApplyThunk, &args);
    }
    if (faulted) {
        // the writes and calls before the fault are done: the engine cannot run the iteration again, so this is counted loudly and the generator switches itself off
        s.switched_off.store(true);
        NoteDifference("fault while generating", fmt::format("iteration {} of the job loop", it.index));
        return 0;
    }
    s.generated.fetch_add(1, std::memory_order_relaxed);
    s.dispatches.fetch_add(1, std::memory_order_relaxed);
    c->rbx = it.context;
    c->r12 = it.groups;
    c->r13 = it.context;
    c->r15 = it.context;
    c->rdi = R<u64>(it.rsp + 0x3b0);
    return Global(LoopResume);
}

u64 PS4_SYSV_ABI JobLoopBottom(GuestHooks::MidContext* c) {
    if (!pending.armed) {
        return 0;
    }
    Stats& s = S();
    pending.armed = false;
    bool clean = true;
    std::vector<std::string> kinds;
    const auto fail = [&](const char* kind, const std::string& text) {
        clean = false;
        kinds.push_back(kind);
        NoteDifference(kind, text);
    };
    if (!pending.dispatch_seen) {
        fail("dispatch missing", fmt::format("the engine's iteration {} issued no async dispatch", pending.it.index));
    }
    for (const auto& note : pending.notes) {
        fail("dispatch", note);
    }
    // the memory the iteration wrote: the generator's bytes against what is there now; a mismatch is read again with a fresh prediction (the engine's other threads move the object's fields)
    u32 memory_differences = 0;
    std::string first_difference;
    const auto describe = [&](const Patch& patch) {
        const u64 from_rsp = patch.address - pending.it.rsp;
        std::string text = from_rsp < 0x1000 ? fmt::format("rsp+{:#x}", from_rsp) : fmt::format("frame+{:#x}", patch.address - pending.it.frame);
        text += fmt::format(" ({} bytes: expected", patch.bytes.size());
        for (size_t i = 0; i < patch.bytes.size() && i < 8; ++i) {
            text += fmt::format(" {:02x}", patch.bytes[i]);
        }
        text += ", found";
        for (size_t i = 0; i < patch.bytes.size() && i < 8; ++i) {
            text += fmt::format(" {:02x}", R<u8>(patch.address + i));
        }
        return text + ")";
    };
    for (const auto& patch : pending.patches) {
        if (std::memcmp(reinterpret_cast<const void*>(patch.address), patch.bytes.data(), patch.bytes.size()) != 0) {
            if (memory_differences++ == 0) {
                first_difference = describe(patch);
            }
        }
    }
    if (memory_differences != 0) {
        Iteration again;
        const char* reason = "";
        bool no_work = false;
        GuestHooks::MidContext probe = *c;
        probe.r13 = pending.it.context;
        probe.rdi = pending.it.holder;
        probe.r14 = pending.it.index;
        probe.rsp = pending.it.rsp;
        bool equal = false;
        if (Scan(&probe, again, reason, no_work)) {
            Writer shadow{false};
            ApplyArgs args{&again, &shadow};
            if (NativeDraw::GuardedCall(&ApplyThunk, &args)) {
                equal = true;
                for (const auto& patch : shadow.patches) {
                    if (std::memcmp(reinterpret_cast<const void*>(patch.address), patch.bytes.data(), patch.bytes.size()) != 0) {
                        equal = false;
                    }
                }
            }
        }
        if (equal) {
            s.verify_raced.fetch_add(1, std::memory_order_relaxed);
        } else {
            fail("memory", fmt::format("{} of {} written ranges differ after iteration {}: first {}", memory_differences, pending.patches.size(), pending.it.index, first_difference));
        }
    }
    // the registers the code after the body reads
    if (c->r12 != pending.it.groups || c->r13 != pending.it.context || c->rbx != pending.it.context || c->r15 != pending.it.context) {
        fail("registers", fmt::format("r12 {:#x} (expected {:#x}), r13 {:#x}, rbx {:#x}, r15 {:#x} (context {:#x})", c->r12, u64{pending.it.groups}, c->r13, c->rbx, c->r15, pending.it.context));
    }
    if (clean) {
        s.verify_clean.fetch_add(1, std::memory_order_relaxed);
    } else if (s.verify_differences.load() >= 8) {
        s.switched_off.store(true); // a generator that differs is switched off, the engine's loop takes over
    }
    pending.patches.clear();
    return 0;
}

void NoteAsyncDispatch(u64 context, u64 x, u64 y, u64 z, u64 site) {
    if (!pending.armed || pending.dispatch_seen || context != pending.it.context) {
        return;
    }
    pending.dispatch_seen = true;
    if (x != pending.it.groups || y != 1 || z != 1) {
        pending.notes.push_back(fmt::format("groups {} x {} x {}, expected {} x 1 x 1", x, y, z, pending.it.groups));
    }
    if (site != Global(DispatchSite)) {
        pending.notes.push_back(fmt::format("call site {:#x}, expected {:#x}", site, Global(DispatchSite)));
    }
    // the async context's tables at the moment of the dispatch: the shader (+0x1d10) and the four constant buffers (+0x21f8 + 8)
    if (R<u64>(context + 0x1d10) != pending.it.shader) {
        pending.notes.push_back(fmt::format("shader {:#x}, expected {:#x}", R<u64>(context + 0x1d10), pending.it.shader));
    }
    for (u32 i = 0; i < 4; ++i) {
        if (R<u64>(context + 0x21f8 + 8 + u64{i} * 8) != pending.table[i]) {
            pending.notes.push_back(fmt::format("constant buffer {}: {:#x}, expected {:#x}", i, R<u64>(context + 0x21f8 + 8 + u64{i} * 8), pending.table[i]));
        }
    }
}

nlohmann::json Report() {
    Stats& s = S();
    nlohmann::json out;
    out["enabled"] = Cfg().on;
    out["switched_off"] = s.switched_off.load();
    out["verify_every"] = Cfg().verify_every;
    out["iterations_seen"] = s.iterations.load();
    out["iterations_without_work"] = s.no_work.load();
    out["generated"] = s.generated.load();
    out["dispatches_generated"] = s.dispatches.load();
    out["declined"] = s.declined.load();
    out["verify_iterations"] = s.verify_iterations.load();
    out["verify_clean"] = s.verify_clean.load();
    out["verify_with_differences"] = s.verify_differences.load();
    out["verify_raced_memory"] = s.verify_raced.load();
    out["scan_faults"] = s.scan_faults.load();
    std::scoped_lock lock{s.mutex};
    out["decline_reasons"] = s.decline_reasons;
    out["difference_kinds"] = s.difference_kinds;
    out["difference_samples"] = s.samples;
    return out;
}

} // namespace Core::NativeJobGen
