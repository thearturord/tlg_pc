// SPDX-License-Identifier: GPL-2.0-or-later
// Load-time text rules for the game's Lua scripts: <serial>.lua_patches.json in the custom configs holds {"rules":[{"chunk","find","replace","count"}]}. A rule applies to the chunk whose
// cleaned name (path separators as '_', no case) ends with `chunk`; `find` must occur exactly `count` times or the rule is skipped and counted as a mismatch. The patched text lives in
// host memory for the rest of the run; the hook points the loader at it. Switch: TLG_LUA_TIME=1 (default off).

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_lua_time.h"

namespace Core::NativeLuaTime {

namespace {
struct Rule {
    std::string chunk, find, replace;
    u64 count = 0;
    u64 applied = 0, mismatched = 0;
};

std::mutex mutex;
bool loaded = false;
std::vector<Rule> rules;
std::deque<std::string> patched_text;
std::map<std::string, std::string> chunk_results; // cleaned chunk name -> "applied/total rules"
u64 chunks_seen = 0, chunks_patched = 0, rules_applied = 0, rules_mismatched = 0, rule_file_errors = 0;

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_LUA_TIME");
        return e == nullptr || e[0] != '0'; // on by default; TLG_LUA_TIME=0 turns it off
    }();
    return on;
}

std::string Lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string Clean(const char* name) {
    std::string s = name != nullptr ? std::string(name, strnlen(name, 250)) : std::string();
    for (char& c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '.')) {
            c = '_';
        }
    }
    return Lower(s);
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void LoadRules() {
    loaded = true;
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (MemoryPatcher::g_game_serial + ".lua_patches.json");
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        LOG_WARNING(Core, "lua patches: {} not found, nothing is patched", path.string());
        return;
    }
    try {
        const auto json = nlohmann::json::parse(file);
        for (const auto& r : json.at("rules")) {
            Rule rule;
            rule.chunk = Lower(r.at("chunk").get<std::string>());
            rule.find = r.at("find").get<std::string>();
            rule.replace = r.at("replace").get<std::string>();
            rule.count = r.at("count").get<u64>();
            if (!rule.find.empty()) {
                rules.push_back(std::move(rule));
            }
        }
        LOG_INFO(Core, "lua patches: {} rules from {}", rules.size(), path.string());
    } catch (const std::exception& e) {
        ++rule_file_errors;
        rules.clear();
        LOG_ERROR(Core, "lua patches: {} is not valid: {}", path.string(), e.what());
    }
}
} // namespace

namespace {
void PatchLocked(GuestHooks::MidContext* context);
u64 seen_for_report = 0;
} // namespace

void Patch(GuestHooks::MidContext* context) {
    if (!Enabled()) {
        return;
    }
    PatchLocked(context);
    if (++seen_for_report % 20 == 0) {
        Report();
    }
}

namespace {
void PatchLocked(GuestHooks::MidContext* context) {
    const char* raw = reinterpret_cast<const char*>(context->rdx);
    const u64 size = context->rcx;
    if (raw == nullptr || size == 0 || size > (64u << 20)) {
        return;
    }
    std::lock_guard lock{mutex};
    if (!loaded) {
        LoadRules();
    }
    ++chunks_seen;
    const std::string name = Clean(reinterpret_cast<const char*>(context->r8));
    std::vector<Rule*> mine;
    for (Rule& rule : rules) {
        if (EndsWith(name, rule.chunk)) {
            mine.push_back(&rule);
        }
    }
    if (mine.empty()) {
        return;
    }
    std::string text(raw, size);
    u64 applied_here = 0;
    for (Rule* rule : mine) {
        u64 found = 0;
        for (size_t at = text.find(rule->find); at != std::string::npos; at = text.find(rule->find, at + rule->find.size())) {
            ++found;
        }
        if (found != rule->count) {
            ++rule->mismatched;
            ++rules_mismatched;
            LOG_ERROR(Core, "lua patches: chunk '{}' rule '{}' found {} times, expected {}: skipped", name, rule->find.substr(0, 60), found, rule->count);
            continue;
        }
        std::string out;
        out.reserve(text.size() + 64 * found);
        size_t from = 0;
        for (size_t at = text.find(rule->find); at != std::string::npos; at = text.find(rule->find, from)) {
            out.append(text, from, at - from);
            out += rule->replace;
            from = at + rule->find.size();
        }
        out.append(text, from, std::string::npos);
        text = std::move(out);
        ++rule->applied;
        ++rules_applied;
        ++applied_here;
    }
    chunk_results[name] = std::to_string(applied_here) + "/" + std::to_string(mine.size());
    if (applied_here == 0) {
        return;
    }
    ++chunks_patched;
    patched_text.push_back(std::move(text));
    context->rdx = reinterpret_cast<u64>(patched_text.back().data());
    context->rcx = patched_text.back().size();
}
} // namespace

void Report() {
    std::lock_guard lock{mutex};
    nlohmann::json results = nlohmann::json::object();
    for (const auto& [name, result] : chunk_results) {
        results[name] = result;
    }
    DebugServer::Publish("lua_time", {{"enabled", Enabled()},
                                      {"rules", rules.size()},
                                      {"chunks_seen", chunks_seen},
                                      {"chunks_patched", chunks_patched},
                                      {"rules_applied", rules_applied},
                                      {"rules_mismatched", rules_mismatched},
                                      {"rule_file_errors", rule_file_errors},
                                      {"per_chunk_applied_of_rules", results}});
}

} // namespace Core::NativeLuaTime
