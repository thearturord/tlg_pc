// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_legacy.h"
#include <cstdlib>
#include <string>
#include <nlohmann/json.hpp>

namespace Core::Legacy {

nlohmann::json Report() {
    static constexpr const char* Names[] = {"TLG_EPOCH_PLANNER", "TLG_BARRIER_GUARD", "TLG_DROP_DIRTY_APPLIES", "TLG_HOOK_OWNED", "TLG_HOOK_OWNED_RECORDING", "TLG_INSTANCE_SCRATCH", "TLG_VIEW_TABLE", "TLG_TARGET_TABLE", "TLG_WRITTEN_NO_IMAGE",
                                            "TLG_OWN_IMMUTABLE", "TLG_OWN_MUTABLE", "TLG_OWN_CLOSE", "TLG_OWN_LARGE", "TLG_OWN_GEOMETRY", "TLG_OWN_RANGE", "TLG_OWN_PLACE", "TLG_OWN_EXACT_END", "TLG_ID_MARKS", "TLG_IMAGE_BY_ID", "TLG_ROW_VERSION",
                                            "TLG_STATIC_ROWS", "TLG_WRITTEN_ROWS", "TLG_WIDE_ROWS", "TLG_TEXSET_BY_ID", "TLG_HOOK_RING",
                                            "TLG_MAP_RING", "TLG_LIGHT_RING", "TLG_SLOT_RING", "TLG_SKIN_RING", "TLG_RECORDING_RING", "TLG_FREE_HOLD", "TLG_FREE_HOLD_FRAMES", "TLG_ORDERED_VIEW",
                                            "TLG_COMPUTE_CONST_RING", "TLG_SKIN_RING_OUT", "TLG_RING_OUT_UNSAMPLED", "TLG_OWN_GEOMETRY_RANGE", "TLG_OWN_VERTEX_SPLIT", "TLG_OWN_VERTEX_RANGE", "TLG_OWN_VERTEX_SLACK", "TLG_PROGRAM_CONSTS", "TLG_OWN_INDEX_BY_ADDRESS", "TLG_TEXTURE_AT_CREATION", "TLG_STORAGE_TABLE", "TLG_HOOK_DIFF", "TLG_DISPATCH_EXECUTOR", "TLG_TARGET_TAIL", "TLG_PLAN_POINTERS", "TLG_OWN_SYNC", "TLG_DYNAMIC_TEXTURES", "TLG_DRAW_EXECUTOR", "TLG_RESOLVED_ANSWERS", "TLG_WIDE_PAST_END", "TLG_OPEN_MAP_GUARD"};
    nlohmann::json j = nlohmann::json::object();
    for (const char* name : Names) {
        const char* e = std::getenv(name);
        j["switches"][name] = e != nullptr && e[0] != 0 ? std::string(e) : std::string("unset");
    }
    j["deleted_paths_back"] = nlohmann::json::object(); // (none since note 141)
    return j;
}

} // namespace Core::Legacy
