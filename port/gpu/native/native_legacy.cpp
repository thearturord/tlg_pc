// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_legacy.h"
#include <cstdlib>
#include <string>
#include <nlohmann/json.hpp>

namespace Core::Legacy {

bool EngineRows() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_LEGACY_ENGINE_ROWS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

nlohmann::json Report() {
    static constexpr const char* Names[] = {"TLG_LEGACY_ENGINE_ROWS", "TLG_EPOCH_PLANNER", "TLG_BARRIER_GUARD", "TLG_GEOMETRY_ROWS", "TLG_DROP_DIRTY_APPLIES", "TLG_HOOK_OWNED", "TLG_VIEW_TABLE",
                                            "TLG_OWN_IMMUTABLE", "TLG_OWN_MUTABLE", "TLG_OWN_CLOSE", "TLG_ID_MARKS", "TLG_IMAGE_BY_ID", "TLG_ROW_VERSION",
                                            "TLG_STATIC_ROWS", "TLG_WRITTEN_ROWS", "TLG_WIDE_ROWS", "TLG_TEXSET_BY_ID", "TLG_HOOK_RING"};
    nlohmann::json j = nlohmann::json::object();
    for (const char* name : Names) {
        const char* e = std::getenv(name);
        j["switches"][name] = e != nullptr && e[0] != 0 ? std::string(e) : std::string("unset");
    }
    j["deleted_paths_back"]["engine_rows"] = EngineRows();
    return j;
}

} // namespace Core::Legacy
