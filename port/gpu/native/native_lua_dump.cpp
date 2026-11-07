// SPDX-License-Identifier: GPL-2.0-or-later
// L0 of the Lua timer design: a mid hook at the entry of hksL_loadbuffer (ghidra 0x01e7e430, (L, settings, buff, size, name) = rdi, rsi, rdx, rcx, r8) writes each distinct chunk
// to user/log/lua_dump/<number>_<name>.lua (source text) or .luac (starts with the 0x1b signature) and a line to index.csv (number, kind, size, hash, caller, name).
// Only when TLG_LUA_DUMP=1; the hook is installed through the hooks file either way and does nothing otherwise.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "common/path_util.h"
#include "common/types.h"
#include "native/native_debug_server.h"
#include "native/native_hooks.h"
#include "native/native_lua_dump.h"
#include "native/native_lua_time.h"

namespace Core::NativeLuaDump {

namespace {
std::atomic<u64> chunks{0}, written{0}, duplicates{0}, bytecode{0}, source{0}, too_big{0};
std::mutex mutex;
std::unordered_set<u64> seen;
FILE* index_file = nullptr;

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_LUA_DUMP");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

u64 Hash(const u8* data, u64 size, u64 h) {
    for (u64 i = 0; i < size; ++i) {
        h = (h ^ data[i]) * 1099511628211ull;
    }
    return h;
}

std::string Clean(const char* name) {
    std::string s = name != nullptr ? std::string(name, strnlen(name, 120)) : std::string("null");
    for (char& c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '.')) {
            c = '_';
        }
    }
    return s;
}

u64 PS4_SYSV_ABI DumpImpl(GuestHooks::MidContext* context) {
    if (!Enabled()) {
        return 0;
    }
    const u8* data = reinterpret_cast<const u8*>(context->rdx);
    const u64 size = context->rcx;
    const char* name = reinterpret_cast<const char*>(context->r8);
    chunks.fetch_add(1, std::memory_order_relaxed);
    if (data == nullptr || size == 0) {
        return 0;
    }
    if (size > (64u << 20)) {
        too_big.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u64 hash = Hash(data, size, Hash(reinterpret_cast<const u8*>(name != nullptr ? name : ""), name != nullptr ? strnlen(name, 120) : 0, 14695981039346656037ull));
    const bool is_bytecode = data[0] == 0x1b;
    std::lock_guard lock{mutex};
    if (!seen.insert(hash).second) {
        duplicates.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "lua_dump";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const u64 number = written.fetch_add(1, std::memory_order_relaxed);
    char prefix[16];
    std::snprintf(prefix, sizeof(prefix), "%05llu_", static_cast<unsigned long long>(number));
    const auto path = dir / (std::string(prefix) + Clean(name) + (is_bytecode ? ".luac" : ".lua"));
    if (FILE* f = std::fopen(path.string().c_str(), "wb")) {
        std::fwrite(data, 1, size, f);
        std::fclose(f);
    }
    if (index_file == nullptr) {
        index_file = std::fopen((dir / "index.csv").string().c_str(), "ab");
    }
    if (index_file != nullptr) {
        const u64 caller = *reinterpret_cast<const u64*>(context->rsp);
        const u64 base = GuestHooks::EbootBase();
        std::fprintf(index_file, "%llu,%s,%llu,%016llx,0x%llx,%s\n", static_cast<unsigned long long>(number), is_bytecode ? "bytecode" : "source", static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(hash), static_cast<unsigned long long>(caller - base + 0x1000000), name != nullptr ? Clean(name).c_str() : "null");
        std::fflush(index_file);
    }
    (is_bytecode ? bytecode : source).fetch_add(1, std::memory_order_relaxed);
    if (number % 50 == 0) {
        Report();
    }
    return 0;
}

u64 PS4_SYSV_ABI Dump(GuestHooks::MidContext* context) {
    DumpImpl(context); // the raw text first
    NativeLuaTime::Patch(context);
    return 0;
}
} // namespace

u64 FindHandler(const char* name) {
    return std::string_view{name} == "lua_dump" ? reinterpret_cast<u64>(&Dump) : 0;
}

void Report() {
    DebugServer::Publish("lua_dump", {{"enabled", Enabled()},
                                      {"chunks_loaded", chunks.load()},
                                      {"files_written", written.load()},
                                      {"duplicates_skipped", duplicates.load()},
                                      {"source", source.load()},
                                      {"bytecode", bytecode.load()},
                                      {"too_big", too_big.load()}});
}

} // namespace Core::NativeLuaDump
