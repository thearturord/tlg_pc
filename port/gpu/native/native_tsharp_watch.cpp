// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "common/logging/log.h"
#include "native/native_draw.h"
#include "native/native_log.h"
#include "native/native_timeline.h"
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "native/native_tsharp_watch.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Core::NativeTsharpWatch {

#ifdef _WIN32
namespace {

constexpr u32 MaxWatches = 4;
struct Watch {
    std::atomic<u64> address{};
    std::atomic<u32> last{};
    u64 view_object{};
    std::atomic<u32> hits{};
};
std::array<Watch, MaxWatches> g_watch;
std::atomic<u32> g_armed{};
std::atomic<bool> g_handler{};

/// Debugger mode: every write the watches catch, kept for the report (the log lines stop after 24 per watch)
struct Hit {
    u64 ns;
    u64 address;
    u64 rip;        ///< raw instruction address
    u64 rva;        ///< Ghidra address when the instruction is in the game image, else 0
    u64 returns[6]; ///< Ghidra addresses of the return addresses found on the stack
    u64 stack[32];  ///< the first stack words (raw), resolved to modules in the report
    u32 thread;
    u32 slot;
};
constexpr u32 HitCapacity = 256;
std::array<Hit, HitCapacity> g_hits{};
std::atomic<u64> g_hit_count{};
std::array<std::string, MaxWatches> g_watch_tag;

LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    CONTEXT* ctx = info->ContextRecord;
    const u64 dr6 = ctx->Dr6;
    if ((dr6 & 0xF) == 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    for (u32 i = 0; i < MaxWatches; ++i) {
        if ((dr6 & (1ull << i)) == 0) {
            continue;
        }
        Watch& w = g_watch[i];
        const u64 address = w.address.load();
        {
            const u64 base = Core::NativeDraw::ImageBase();
            Hit& hit = g_hits[g_hit_count.fetch_add(1) % HitCapacity];
            hit = {};
            hit.ns = Core::FrameTimeline::Now();
            hit.address = address;
            hit.rip = ctx->Rip;
            hit.rva = ctx->Rip > base && ctx->Rip < base + 0x6000000 ? ctx->Rip - base + 0x1000000 : 0;
            hit.thread = GetCurrentThreadId();
            hit.slot = i;
            const u64* sp = reinterpret_cast<const u64*>(ctx->Rsp);
            u32 found = 0;
            for (u32 k = 0; k < 160 && found < 6; ++k) {
                u64 v = 0;
                __try {
                    v = sp[k];
                } __except (EXCEPTION_EXECUTE_HANDLER) {
                    break;
                }
                if (k < 32) {
                    hit.stack[k] = v;
                }
                if (v > base && v < base + 0x6000000) {
                    hit.returns[found++] = v - base + 0x1000000;
                }
            }
            Core::FrameTimeline::Record(Core::FrameTimeline::WriterHit, address, hit.ns, hit.rva | (hit.returns[0] << 32));
        }
        u32 now = 0;
        std::memcpy(&now, reinterpret_cast<const void*>(address), sizeof(now));
        const u32 before = w.last.exchange(now);
        const u32 count = w.hits.fetch_add(1);
        if (count < 24) {
            const u64 base = Core::NativeDraw::ImageBase();
            char text[1024];
            int n = std::snprintf(text, sizeof(text), "tsharp watch %u: view %#llx dword %#x -> %#x (height %u -> %u) rip %#llx thread %lu returns:", i,
                                  static_cast<unsigned long long>(w.view_object), before, now, ((before >> 14) & 0x3fff) + 1, ((now >> 14) & 0x3fff) + 1,
                                  static_cast<unsigned long long>(ctx->Rip - base + 0x1000000), GetCurrentThreadId());
            // return addresses: every stack word that points into the game image
            const u64* sp = reinterpret_cast<const u64*>(ctx->Rsp);
            u32 found = 0;
            for (u32 k = 0; k < 160 && found < 8 && n < static_cast<int>(sizeof(text)) - 24; ++k) {
                u64 v = 0;
                __try {
                    v = sp[k];
                } __except (EXCEPTION_EXECUTE_HANDLER) {
                    break;
                }
                if (v > base && v < base + 0x6000000) {
                    n += std::snprintf(text + n, sizeof(text) - n, " %#llx", static_cast<unsigned long long>(v - base + 0x1000000));
                    ++found;
                }
            }
            TLG_LOG(IMAGE, Error, "{}", text);
        }
    }
    ctx->Dr6 = 0;
    return EXCEPTION_CONTINUE_EXECUTION; // a data breakpoint traps after the write, nothing to skip
}

/// Sets the debug register `slot` of every thread of the process except the helper (done from a helper thread that suspends the others, the caller included)
void SetDebugRegisters(u32 slot, u64 address) {
    const DWORD pid = GetCurrentProcessId();
    std::thread([=] {
        const DWORD self = GetCurrentThreadId();
        std::vector<DWORD> ids;
        ids.reserve(512);
        const HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            THREADENTRY32 e{};
            e.dwSize = sizeof(e);
            for (bool ok = Thread32First(snap, &e); ok; ok = Thread32Next(snap, &e)) {
                if (e.th32OwnerProcessID == pid && e.th32ThreadID != self) {
                    ids.push_back(e.th32ThreadID);
                }
            }
            CloseHandle(snap);
        }
        std::vector<HANDLE> handles;
        handles.reserve(ids.size());
        for (const DWORD id : ids) {
            if (const HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, id)) {
                handles.push_back(h);
            }
        }
        for (const HANDLE h : handles) {
            SuspendThread(h);
        }
        for (const HANDLE h : handles) {
            CONTEXT c{};
            c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (!GetThreadContext(h, &c)) {
                continue;
            }
            (&c.Dr0)[slot] = address;
            // local enable bit of the slot, condition 01 (write), length 11 (4 bytes)
            c.Dr7 &= ~((3ull << (slot * 2)) | (0xFull << (16 + slot * 4)));
            c.Dr7 |= (1ull << (slot * 2)) | (0x1ull << (16 + slot * 4)) | (0x3ull << (18 + slot * 4));
            SetThreadContext(h, &c);
        }
        for (const HANDLE h : handles) {
            ResumeThread(h);
            CloseHandle(h);
        }
    }).join();
}

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_TSHARP_WATCH");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void Arm(u64 view_object, u64 address, u32 width, u32 height) {
    if (!On() || width != 1920 || height != 1088) {
        return;
    }
    const u32 slot = g_armed.fetch_add(1);
    if (slot >= MaxWatches) {
        return;
    }
    if (!g_handler.exchange(true)) {
        AddVectoredExceptionHandler(1, Handler);
    }
    Watch& w = g_watch[slot];
    w.view_object = view_object;
    u32 value = 0;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    w.last.store(value);
    w.address.store(address);
    SetDebugRegisters(slot, address);
    TLG_LOG(IMAGE, Error, "tsharp watch {} armed: view {:#x} dword at {:#x} = {:#x} (height {})", slot, view_object, address, value, ((value >> 14) & 0x3fff) + 1);
}

void ArmAddress(u64 address, const char* tag) {
    // W1.5 census: a data breakpoint on any dword (logs the writing instruction and the game return addresses, Ghidra addresses)
    address &= ~u64{3};
    const u32 slot = g_armed.fetch_add(1);
    if (slot >= MaxWatches) {
        return;
    }
    if (!g_handler.exchange(true)) {
        AddVectoredExceptionHandler(1, Handler);
    }
    Watch& w = g_watch[slot];
    w.view_object = 0;
    u32 value = 0;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    w.last.store(value);
    w.address.store(address);
    SetDebugRegisters(slot, address);
    TLG_LOG(IMAGE, Error, "write watch {} armed ({}): dword at {:#x} = {:#x}", slot, tag, address, value);
}

void ArmRotating(u64 address, const char* tag) {
    // debugger mode: the watches are reused in turn, armed from a thread of its own (arming suspends every thread of the process for a moment)
    address &= ~u64{3};
    const u32 slot = g_armed.fetch_add(1) % MaxWatches;
    if (!g_handler.exchange(true)) {
        AddVectoredExceptionHandler(1, Handler);
    }
    Watch& w = g_watch[slot];
    w.view_object = 0;
    w.hits.store(0);
    u32 value = 0;
    __try {
        std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return;
    }
    w.last.store(value);
    w.address.store(address);
    g_watch_tag[slot] = tag;
    std::thread([slot, address] { SetDebugRegisters(slot, address); }).detach();
}

/// "module.dll+0xoffset" when the address is inside a loaded module's image, else empty
std::string ModuleOf(u64 address) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) || module == nullptr) {
        return {};
    }
    char path[MAX_PATH]{};
    GetModuleFileNameA(module, path, MAX_PATH);
    const char* name = std::strrchr(path, '\\');
    return fmt::format("{}+{:#x}", name != nullptr ? name + 1 : path, address - reinterpret_cast<u64>(module));
}

std::string HitsJson() {
    nlohmann::json j = nlohmann::json::array();
    const u64 total = g_hit_count.load();
    const u64 first = total > HitCapacity ? total - HitCapacity : 0;
    for (u64 n = first; n < total; ++n) {
        const Hit& h = g_hits[n % HitCapacity];
        nlohmann::json e;
        e["ns"] = h.ns;
        e["address"] = fmt::format("{:#x}", h.address);
        e["rip"] = fmt::format("{:#x}", h.rip);
        e["rip_module"] = ModuleOf(h.rip);
        nlohmann::json chain = nlohmann::json::array();
        for (const u64 v : h.stack) {
            const std::string m = v > 0x10000 ? ModuleOf(v) : std::string();
            if (!m.empty() && chain.size() < 10) {
                chain.push_back(m);
            }
        }
        e["stack_code_addresses"] = chain;
        e["game_instruction"] = h.rva != 0 ? fmt::format("{:#x}", h.rva) : std::string("(host code, not the game image)");
        nlohmann::json r = nlohmann::json::array();
        for (const u64 v : h.returns) {
            if (v != 0) {
                r.push_back(fmt::format("{:#x}", v));
            }
        }
        e["returns"] = r;
        e["thread"] = h.thread;
        e["watch_tag"] = g_watch_tag[h.slot % MaxWatches];
        j.push_back(e);
    }
    return j.dump();
}

#else
bool On() noexcept {
    return false;
}
void ArmRotating(u64, const char*) {}
std::string HitsJson() {
    return "[]";
}
void Arm(u64, u64, u32, u32) {}
void ArmAddress(u64, const char*) {}
#endif

} // namespace Core::NativeTsharpWatch
