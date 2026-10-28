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
    // L1(b): which of the 8 T# dwords differ when the same id shows other words (per dword index), and how many ids are views (compute) vs draws
    std::array<std::atomic<u64>, 8> image_word_changed{};
    // L1(b), images by engine id: lookups served by the id table after the generation moved (no registration touched the range: proven by the texture cache's event ring), refused (an event
    // touched the range, or the ring could not tell), verified against FindImage, and the verify's differences (the image differs, the view range differs)
    std::atomic<u64> texture_gen_verified{0}, texture_gen_diff_image{0}, texture_gen_diff_range{0}; // the same, for hits at an unchanged generation
    std::atomic<u64> texture_miss_cold{0}, texture_miss_other_key{0}, texture_miss_native_reuse_off{0};
    std::atomic<u64> texture_id_served{0}, texture_id_refused{0}, texture_id_verified{0}, texture_id_diff_image{0}, texture_id_diff_range{0};
    // compute pipelines by program address
    std::atomic<u64> dispatches{0}, programs_seen{0}, pipeline_same{0}, pipeline_switched{0};
    // B9: the typed bindings the dispatch records carry (dispatches planned from a record, with a complete list, entries in them)
    std::atomic<u64> dispatch_with_record{0}, dispatch_bindings_complete{0}, dispatch_binding_entries{0};
    // B9 compare (TLG_DISPATCH_AUDIT=1): buffer loads of dispatches, served by the record, equal to the guest's descriptor, different
    std::atomic<u64> dispatch_buffer_loads{0}, dispatch_buffer_record{0}, dispatch_buffer_equal{0}, dispatch_buffer_different{0};
    // L2: buffer loads the record has no entry for, and of those the ones where the guest's descriptor is a null buffer too (base or size 0)
    std::atomic<u64> dispatch_buffer_missing{0}, dispatch_buffer_missing_null{0};
    // B9: classes of the differing buffer descriptors: 0 base differs, 1 size differs, 2 other words differ, 3 the id has an immutable buffer, 4 the record's range is inside it, 5 the guest's range is inside it, 6 neither, 7 the record's buffer is not alive
    std::array<std::atomic<u64>, 8> dispatch_diff{};
    // B9: of the differing ones: 0 the record's base is the previous dispatch's guest base (the record lags one update), 1 the guest's base is the previous record's base, 2 neither
    std::array<std::atomic<u64>, 3> dispatch_lag{};
};
inline IdentityAudit g_identity_audit;

/// Raster thread only.
class IdentityTables {
public:
    struct ImageEntry {
        u64 words_hash{}, address{};
        u32 words[8]{};
    };
    std::unordered_map<u64, ImageEntry> images;
    std::unordered_map<u64, const void*> programs;
};

} // namespace Vulkan
