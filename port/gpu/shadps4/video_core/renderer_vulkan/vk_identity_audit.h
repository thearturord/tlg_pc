// tlg C1 (images and compute): audit of how stable the engine's identities are on the consumer side (read only, always on, cheap). Raster thread writes, any thread reads.
//  * images: for every image binding of a record that names an engine id, does the description (the T# words) of that id stay the same between draws, and does its address?
//  * compute: for every program address, is the pipeline the consumer resolves always the same one?
// What it answers: whether a table keyed by the engine's id could replace the lookup by T# words (images) or by program and user data (compute pipelines).

#pragma once

#include <atomic>
#include <unordered_map>

#include "common/types.h"

namespace Vulkan {

struct IdentityAudit {
    // images (record bindings with an engine id)
    std::atomic<u64> image_bindings{0}, image_with_id{0}, image_ids_seen{0}, image_same{0}, image_address_changed{0}, image_other_words_changed{0};
    // compute pipelines by program address
    std::atomic<u64> dispatches{0}, programs_seen{0}, pipeline_same{0}, pipeline_switched{0};
};
inline IdentityAudit g_identity_audit;

/// Raster thread only.
class IdentityTables {
public:
    struct ImageEntry {
        u64 words_hash{}, address{};
    };
    std::unordered_map<u64, ImageEntry> images;
    std::unordered_map<u64, const void*> programs;
};

} // namespace Vulkan
