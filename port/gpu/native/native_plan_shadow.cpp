// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_plan_shadow.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace Core::NativePlanShadow {

namespace {

struct Entry {
    u64 evaluated{0};
    u64 differs{0};
    u64 nonzero{0}; ///< register only fields
    bool register_only{false};
    std::array<std::string, 3> examples{};
    u32 example_count{0};
};

std::mutex g_mutex;
std::map<std::string, Entry> g_entries;

Entry& EntryOf(const char* group, const char* field) {
    return g_entries[std::string(group) + "/" + field];
}

u64 First8(const void* p, size_t size) {
    u64 v = 0;
    std::memcpy(&v, p, size < sizeof(v) ? size : sizeof(v));
    return v;
}

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_PLAN_SHADOW");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void Compare(const char* group, const char* field, u64 native, u64 registers) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    Entry& e = EntryOf(group, field);
    ++e.evaluated;
    if (native != registers) {
        ++e.differs;
        if (e.example_count < e.examples.size()) {
            e.examples[e.example_count++] = fmt::format("native {:#x}, registers {:#x}", native, registers);
        }
    }
}

void CompareBytes(const char* group, const char* field, const void* native, const void* registers, size_t size) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    Entry& e = EntryOf(group, field);
    ++e.evaluated;
    if (std::memcmp(native, registers, size) != 0) {
        ++e.differs;
        if (e.example_count < e.examples.size()) {
            e.examples[e.example_count++] = fmt::format("native {:#x}, registers {:#x} (first 8 bytes of {})", First8(native, size), First8(registers, size), size);
        }
    }
}

void RegisterOnly(const char* group, const char* field, u64 registers) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    Entry& e = EntryOf(group, field);
    e.register_only = true;
    ++e.evaluated;
    if (registers != 0) {
        ++e.nonzero;
        if (e.example_count < e.examples.size()) {
            e.examples[e.example_count++] = fmt::format("registers {:#x}", registers);
        }
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["enabled"] = On();
    j["note"] = "TLG_PLAN_SHADOW=1: the register fallback of the draw path against the native side, per field; 'differs' counts evaluations where the register image's value is not the native one; "
                "register_only fields have no native value (nonzero counts evaluations where the register value is not 0)";
    auto& fields = j["fields"] = nlohmann::json::object();
    std::scoped_lock lock{g_mutex};
    for (const auto& [name, e] : g_entries) {
        nlohmann::json row = {{"evaluated", e.evaluated}};
        if (e.register_only) {
            row["register_only"] = true;
            row["nonzero"] = e.nonzero;
        } else {
            row["differs"] = e.differs;
        }
        nlohmann::json examples = nlohmann::json::array();
        for (u32 i = 0; i < e.example_count; ++i) {
            examples.push_back(e.examples[i]);
        }
        if (!examples.empty()) {
            row["examples"] = std::move(examples);
        }
        fields[name] = std::move(row);
    }
    return j;
}

} // namespace Core::NativePlanShadow
