// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_e3_census.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <xxhash.h>

#include "native/native_device.h"

namespace Core::NativeE3Census {

namespace {

constexpr u32 NumFns = static_cast<u32>(Fn::Count);
constexpr u32 NumCallers = static_cast<u32>(Caller::Count);

constexpr std::array<const char*, NumFns> FnNames{"RefreshImage", "ExpandImage", "ResolveDepthOverlap", "ResolveOverlap", "exact_match"};
constexpr std::array<const char*, NumCallers> CallerNames{
    "unknown", "Presenter::PrepareFrame", "Rasterizer::BindTexturesFromSet", "Rasterizer::BeginRenderingFull", "Rasterizer::TryCopyImages", "Rasterizer::NativeClear",
    "TextureCache::FindTexture", "TextureCache::FindRenderTarget", "TextureCache::FindDepthTarget", "TextureCache::FindImage (rest of a texture before its engine clear)",
    "TextureCache::ExpandImage", "TextureCache::ResolveDepthOverlap", "ResolveOverlap equal address: block view of an uncompressed image (2)",
    "ResolveOverlap equal address: 3D view of equal size (3)", "ResolveOverlap equal address: more resources, grow (6)", "ResolveOverlap: depth overlap", "TextureCache::FindImage (calls ResolveOverlap)",
    "TextureCache::FindImage (exact match, engine texture's list)", "TextureCache::FindImage (exact match, remembered answer)",
};

std::array<std::array<std::atomic<u64>, NumCallers>, NumFns> g_calls{};

struct UploadCounters {
    std::atomic<u64> uploads{0}, source_gpu{0}, first_upload{0}, same_bytes_as_last_upload{0}, new_bytes{0}, untracked{0}, bytes_hashed{0}, bytes_same{0};
};
std::array<UploadCounters, NumCallers> g_uploads{};

std::mutex g_mutex;
std::unordered_map<u64, u64> g_last_hash; ///< image uid -> hash of the bytes of its last upload
std::atomic<u64> g_first_ns{0};

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// E7c Task 3 (TLG_E3_ANNOUNCED=1)
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

constexpr u32 NumOutcomes = static_cast<u32>(Outcome::Count);
constexpr std::array<const char*, NumOutcomes> OutcomeNames{"not_dirty (returned at once: no CpuDirty / GpuDirty mark)", "multisample (dirty, but more than one sample)", "maybe_cpu_dirty_same_hash (first 8x8 pixels unchanged)",
                                                            "no_copies (no mip to copy)", "uploaded"};
std::array<std::array<std::atomic<u64>, NumOutcomes>, NumCallers> g_outcomes{};

constexpr u32 NumSites = static_cast<u32>(WriteSite::Count);
constexpr std::array<const char*, NumSites + 1> SiteNames{"none (the call site did not name itself)", "shader_graphics (a draw's storage buffer that the shader writes)", "shader_compute (a dispatch's storage buffer that the shader writes)",
                                                          "fill_buffer (a GPU fill: the engine's clear or CP fill)", "copy_buffer (a GPU copy, DMA)", "hle_copy (a recompiler HLE copy shader)", "no_range_recorded (the 8192 entry ring no longer holds a write over it)"};
thread_local WriteSite t_site = WriteSite::None;

struct PageRec {
    u64 hash{0}; ///< XXH3 of the page when announced (0: not hashed, the range was over 2 MB)
    u64 seq{0};
};
struct WriteRec {
    u64 begin{0}, end{0};
    u64 seq{0};
    WriteSite site{WriteSite::None};
};
constexpr u64 MaxPages = 1ull << 21;
constexpr u64 MaxHashedRange = 2ull << 20;
constexpr u32 WriteRing = 8192;
constexpr u32 NumGroups = 3; // first upload of the image, GPU source, repeat upload
constexpr std::array<const char*, NumGroups> GroupNames{"first_upload_of_the_image", "source_gpu_written_not_hashed", "repeat_upload"};
enum Class : u32 { Equal, Differs, None, NumClasses };
constexpr std::array<const char*, NumClasses> ClassNames{"announced_equal", "announced_differs", "no_announcement"};

std::mutex g_ann_mutex;
std::unordered_map<u64, PageRec> g_pages;     ///< page number -> what the last first-time announcement of it said
std::unordered_set<u64> g_seen_images;
std::vector<WriteRec> g_writes;
u64 g_write_count = 0;
u64 g_seq = 0;
std::atomic<u64> g_ann_first{0}, g_ann_again{0}, g_ann_bytes{0}, g_ann_hashed_pages{0}, g_ann_unhashed_pages{0}, g_ann_table_full{0};
std::array<std::array<std::array<std::atomic<u64>, NumClasses>, NumCallers>, NumGroups> g_class{};
std::array<std::array<std::atomic<u64>, 6>, NumGroups> g_detail{}; ///< per group: fully covered, partly covered, announced pages, equal pages, differing pages, unannounced pages
std::array<std::atomic<u64>, NumSites + 1> g_gpu_writer_site{};
std::array<std::atomic<u64>, 6> g_gpu_image{}; ///< uploads with a GPU source: image has GpuModified, colour attachment usage, depth attachment usage, storage usage, engine texture owns it, no engine texture owns it
std::array<std::array<nlohmann::json, NumClasses>, NumGroups> g_examples;
constexpr u32 MaxExamples = 8;

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void StartClock() {
    u64 expected = 0;
    g_first_ns.compare_exchange_strong(expected, NowNs() | 1, std::memory_order_relaxed);
}

} // namespace

bool On() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_E3_CENSUS");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void NoteCall(Fn fn, Caller caller) {
    if (!On()) {
        return;
    }
    StartClock();
    g_calls[static_cast<u32>(fn)][static_cast<u32>(caller)].fetch_add(1, std::memory_order_relaxed);
}


bool AnnouncedOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_E3_ANNOUNCED");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void NoteAnnounced(u64 address, u64 size, bool first) {
    if (!AnnouncedOn()) {
        return;
    }
    if (!first) {
        g_ann_again.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_ann_first.fetch_add(1, std::memory_order_relaxed);
    g_ann_bytes.fetch_add(size, std::memory_order_relaxed);
    if (size == 0 || address < 4096) {
        return;
    }
    const u64 p0 = address >> 12;
    const u64 p1 = (address + size - 1) >> 12;
    const bool hashed = size <= MaxHashedRange;
    std::vector<u64> hashes;
    if (hashed) {
        hashes.reserve(p1 - p0 + 1);
        for (u64 p = p0; p <= p1; ++p) {
            hashes.push_back(XXH3_64bits(reinterpret_cast<const void*>(p << 12), 4096) | 1);
        }
        g_ann_hashed_pages.fetch_add(p1 - p0 + 1, std::memory_order_relaxed);
    } else {
        g_ann_unhashed_pages.fetch_add(p1 - p0 + 1, std::memory_order_relaxed);
    }
    std::scoped_lock lock{g_ann_mutex};
    const u64 seq = ++g_seq;
    for (u64 p = p0; p <= p1; ++p) {
        if (g_pages.size() >= MaxPages && g_pages.find(p) == g_pages.end()) {
            g_ann_table_full.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        g_pages[p] = {hashed ? hashes[p - p0] : 0, seq};
    }
}

WriteSiteScope::WriteSiteScope(WriteSite site) noexcept : previous{t_site} {
    t_site = site;
}

WriteSiteScope::~WriteSiteScope() {
    t_site = previous;
}

void NoteGpuWritten(u64 address, u64 size) {
    if (!AnnouncedOn()) {
        return;
    }
    std::scoped_lock lock{g_ann_mutex};
    if (g_writes.size() < WriteRing) {
        g_writes.resize(WriteRing);
    }
    g_writes[g_write_count++ % WriteRing] = {address, address + size, ++g_seq, t_site};
}

void NoteOutcome(Caller caller, Outcome outcome) {
    if (!On()) {
        return;
    }
    g_outcomes[static_cast<u32>(caller)][static_cast<u32>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

void NoteAnnouncedUpload(const UploadInfo& info) {
    if (!AnnouncedOn() || info.size == 0 || info.address < 4096) {
        return;
    }
    const u64 p0 = info.address >> 12;
    const u64 p1 = (info.address + info.size - 1) >> 12;
    const u64 pages = p1 - p0 + 1;
    std::vector<PageRec> recs(pages);
    std::vector<u8> known(pages, 0);
    u32 group = 2;
    u32 site = NumSites; // no_range_recorded
    {
        std::scoped_lock lock{g_ann_mutex};
        group = g_seen_images.insert(info.image_uid).second ? 0u : 2u;
        if (info.gpu_source) {
            group = 1;
            const u64 count = std::min<u64>(g_write_count, WriteRing);
            for (u64 i = 0; i < count; ++i) {
                const WriteRec& w = g_writes[(g_write_count - 1 - i) % WriteRing];
                if (w.begin < info.address + info.size && info.address < w.end) {
                    site = static_cast<u32>(w.site);
                    break;
                }
            }
        }
        for (u64 p = p0; p <= p1; ++p) {
            if (const auto it = g_pages.find(p); it != g_pages.end()) {
                recs[p - p0] = it->second;
                known[p - p0] = 1;
            }
        }
    }
    u64 announced = 0, unhashed = 0, equal = 0, differs = 0, newest_seq = 0;
    u64 first_diff = ~0ull;
    for (u64 i = 0; i < pages; ++i) {
        if (known[i] == 0) {
            continue;
        }
        ++announced;
        newest_seq = std::max(newest_seq, recs[i].seq);
        if (recs[i].hash == 0) {
            ++unhashed;
            continue;
        }
        const u64 now = XXH3_64bits(reinterpret_cast<const void*>((p0 + i) << 12), 4096) | 1;
        if (now == recs[i].hash) {
            ++equal;
        } else {
            ++differs;
            if (first_diff == ~0ull) {
                first_diff = std::max<u64>((p0 + i) << 12, info.address) - info.address;
            }
        }
    }
    const u32 klass = announced == 0 ? None : differs != 0 ? Differs : Equal;
    const u32 caller = static_cast<u32>(info.caller);
    g_class[group][caller][klass].fetch_add(1, std::memory_order_relaxed);
    g_detail[group][announced == pages && unhashed == 0 ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
    g_detail[group][2].fetch_add(announced, std::memory_order_relaxed);
    g_detail[group][3].fetch_add(equal, std::memory_order_relaxed);
    g_detail[group][4].fetch_add(differs, std::memory_order_relaxed);
    g_detail[group][5].fetch_add(pages - announced, std::memory_order_relaxed);

    Core::NativeDevice::Object owner;
    const bool owned = Core::NativeDevice::ResourceAt(info.address, owner);
    if (info.gpu_source) {
        g_gpu_writer_site[site].fetch_add(1, std::memory_order_relaxed);
        g_gpu_image[0].fetch_add(info.gpu_modified_flag ? 1 : 0, std::memory_order_relaxed);
        g_gpu_image[1].fetch_add((info.usage_bits & 0x10u) != 0 ? 1 : 0, std::memory_order_relaxed);
        g_gpu_image[2].fetch_add((info.usage_bits & 0x20u) != 0 ? 1 : 0, std::memory_order_relaxed);
        g_gpu_image[3].fetch_add((info.usage_bits & 0x8u) != 0 ? 1 : 0, std::memory_order_relaxed);
        g_gpu_image[owned ? 4 : 5].fetch_add(1, std::memory_order_relaxed);
    }

    std::scoped_lock lock{g_ann_mutex};
    nlohmann::json& list = g_examples[group][klass];
    if (list.is_null()) {
        list = nlohmann::json::array();
    }
    if (list.size() >= MaxExamples) {
        return;
    }
    nlohmann::json e = {{"caller", CallerNames[caller]},
                        {"address", fmt::format("{:#x}", info.address)},
                        {"size", info.size},
                        {"width", info.width},
                        {"height", info.height},
                        {"host_format", info.format},
                        {"pages", pages},
                        {"announced_pages", announced},
                        {"announced_pages_not_hashed", unhashed},
                        {"equal_pages", equal},
                        {"differing_pages", differs},
                        {"unannounced_pages", pages - announced},
                        {"announcements_ago", announced == 0 ? 0 : g_seq - newest_seq},
                        {"image_gpu_modified_flag", info.gpu_modified_flag},
                        {"image_usage_bits", fmt::format("{:#x}", info.usage_bits)},
                        {"engine_texture", owned ? fmt::format("kind {} id {} {:#x}+{:#x} {}x{} dxgi {}", static_cast<u32>(owner.kind), owner.id, owner.address, owner.bytes, owner.width, owner.height, owner.format) : std::string{"none"}}};
    if (first_diff != ~0ull) {
        e["first_differing_offset"] = first_diff;
    }
    if (info.gpu_source) {
        e["gpu_writer"] = SiteNames[site];
    }
    list.push_back(std::move(e));
}

void NoteUpload(Caller caller, u64 image_uid, const u8* bytes, u64 size) {
    if (!On()) {
        return;
    }
    UploadCounters& c = g_uploads[static_cast<u32>(caller)];
    c.uploads.fetch_add(1, std::memory_order_relaxed);
    if (bytes == nullptr || size == 0) {
        c.source_gpu.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 hash = XXH3_64bits(bytes, size) | 1;
    c.bytes_hashed.fetch_add(size, std::memory_order_relaxed);
    std::scoped_lock lock{g_mutex};
    const auto it = g_last_hash.find(image_uid);
    if (it == g_last_hash.end()) {
        if (g_last_hash.size() >= (1u << 17)) {
            c.untracked.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_last_hash.emplace(image_uid, hash);
        c.first_upload.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (it->second == hash) {
        c.same_bytes_as_last_upload.fetch_add(1, std::memory_order_relaxed);
        c.bytes_same.fetch_add(size, std::memory_order_relaxed);
    } else {
        c.new_bytes.fetch_add(1, std::memory_order_relaxed);
        it->second = hash;
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    const u64 first = g_first_ns.load(std::memory_order_relaxed);
    const double seconds = first == 0 ? 0.0 : static_cast<double>(NowNs() - first) / 1e9;
    j["enabled"] = On();
    j["observed_seconds"] = seconds;
    j["note"] = "TLG_E3_CENSUS=1: per caller of the address keyed image code; per_25s = count scaled to 25 s of observed time; uploads: RefreshImage uploads from guest memory compared with the image's previous upload";
    const auto scaled = [&](u64 n) { return seconds > 0.0 ? static_cast<double>(n) * 25.0 / seconds : 0.0; };
    auto& functions = j["functions"] = nlohmann::json::object();
    for (u32 f = 0; f < NumFns; ++f) {
        nlohmann::json list = nlohmann::json::array();
        u64 total = 0;
        for (u32 c = 0; c < NumCallers; ++c) {
            const u64 n = g_calls[f][c].load();
            if (n == 0) {
                continue;
            }
            total += n;
            list.push_back({{"caller", CallerNames[c]}, {"count", n}, {"per_25s", scaled(n)}});
        }
        functions[FnNames[f]] = {{"total", total}, {"per_25s", scaled(total)}, {"callers", std::move(list)}};
    }
    nlohmann::json uploads = nlohmann::json::array();
    u64 all = 0, same = 0;
    for (u32 c = 0; c < NumCallers; ++c) {
        const UploadCounters& u = g_uploads[c];
        const u64 n = u.uploads.load();
        if (n == 0) {
            continue;
        }
        all += n;
        same += u.same_bytes_as_last_upload.load();
        uploads.push_back({{"caller", CallerNames[c]}, {"uploads", n}, {"per_25s", scaled(n)}, {"source_gpu_written_not_hashed", u.source_gpu.load()}, {"first_upload_of_the_image", u.first_upload.load()},
                           {"bytes_differ_from_last_upload", u.new_bytes.load()}, {"bytes_same_as_last_upload", u.same_bytes_as_last_upload.load()}, {"untracked_table_full", u.untracked.load()},
                           {"megabytes_hashed", static_cast<double>(u.bytes_hashed.load()) / 1048576.0}, {"megabytes_same", static_cast<double>(u.bytes_same.load()) / 1048576.0}});
    }
    j["refresh_uploads"] = {{"total", all}, {"same_bytes_as_last_upload", same}, {"by_caller", std::move(uploads)}};
    {
        nlohmann::json o = nlohmann::json::object();
        o["note"] = "RefreshImage exits per caller: every call is counted under functions.RefreshImage; these are what the call did";
        nlohmann::json by_reason = nlohmann::json::object();
        for (u32 r = 0; r < NumOutcomes; ++r) {
            u64 total = 0;
            nlohmann::json callers = nlohmann::json::array();
            for (u32 c = 0; c < NumCallers; ++c) {
                const u64 n = g_outcomes[c][r].load();
                if (n == 0) {
                    continue;
                }
                total += n;
                callers.push_back({{"caller", CallerNames[c]}, {"count", n}, {"per_25s", scaled(n)}});
            }
            by_reason[OutcomeNames[r]] = {{"total", total}, {"per_25s", scaled(total)}, {"callers", std::move(callers)}};
        }
        o["by_reason"] = std::move(by_reason);
        j["refresh_outcomes"] = std::move(o);
    }
    if (AnnouncedOn()) {
        nlohmann::json a = nlohmann::json::object();
        a["switch"] = "TLG_E3_ANNOUNCED=1";
        a["note"] = "an announcement is a memory range (NativeUpdates), not stored bytes: the bytes of each 4 KB page are hashed when the announcement is applied and again at the upload; equal = the page still has the bytes it had when announced; the upload reads guest memory before the detile, both sides are the same raw bytes. gpu-source uploads read the GPU written buffer, not guest memory: their class says whether the guest pages were ever announced and still unchanged, nothing more";
        a["announcements"] = {{"first_applications", g_ann_first.load()}, {"second_applications_not_hashed", g_ann_again.load()}, {"bytes", g_ann_bytes.load()}, {"pages_hashed", g_ann_hashed_pages.load()},
                              {"pages_not_hashed_range_over_2mb", g_ann_unhashed_pages.load()}, {"page_table_full_skipped", g_ann_table_full.load()}};
        nlohmann::json groups = nlohmann::json::object();
        for (u32 g = 0; g < NumGroups; ++g) {
            nlohmann::json grp = nlohmann::json::object();
            u64 totals[NumClasses] = {};
            nlohmann::json callers = nlohmann::json::array();
            for (u32 c = 0; c < NumCallers; ++c) {
                u64 n[NumClasses];
                u64 sum = 0;
                for (u32 k = 0; k < NumClasses; ++k) {
                    n[k] = g_class[g][c][k].load();
                    sum += n[k];
                    totals[k] += n[k];
                }
                if (sum == 0) {
                    continue;
                }
                callers.push_back({{"caller", CallerNames[c]}, {"uploads", sum}, {ClassNames[Equal], n[Equal]}, {ClassNames[Differs], n[Differs]}, {ClassNames[None], n[None]}});
            }
            for (u32 k = 0; k < NumClasses; ++k) {
                grp[ClassNames[k]] = totals[k];
            }
            grp["uploads"] = totals[Equal] + totals[Differs] + totals[None];
            grp["every_page_announced_and_hashed"] = g_detail[g][0].load();
            grp["some_page_not_announced"] = g_detail[g][1].load();
            grp["pages"] = {{"announced", g_detail[g][2].load()}, {"equal", g_detail[g][3].load()}, {"differing", g_detail[g][4].load()}, {"never_announced", g_detail[g][5].load()}};
            grp["by_caller"] = std::move(callers);
            nlohmann::json ex = nlohmann::json::object();
            {
                std::scoped_lock lock{g_ann_mutex};
                for (u32 k = 0; k < NumClasses; ++k) {
                    if (!g_examples[g][k].is_null()) {
                        ex[ClassNames[k]] = g_examples[g][k];
                    }
                }
            }
            grp["examples"] = std::move(ex);
            groups[GroupNames[g]] = std::move(grp);
        }
        a["groups"] = std::move(groups);
        nlohmann::json writers = nlohmann::json::object();
        for (u32 s = 0; s <= NumSites; ++s) {
            writers[SiteNames[s]] = g_gpu_writer_site[s].load();
        }
        a["gpu_source_uploads_by_last_writer"] = std::move(writers);
        a["gpu_source_uploads_image"] = {{"image_gpu_modified_flag", g_gpu_image[0].load()}, {"colour_attachment_usage", g_gpu_image[1].load()}, {"depth_attachment_usage", g_gpu_image[2].load()},
                                         {"storage_usage", g_gpu_image[3].load()}, {"engine_texture_owns_the_memory", g_gpu_image[4].load()}, {"no_engine_texture_owns_the_memory", g_gpu_image[5].load()}};
        j["announced_uploads"] = std::move(a);
    }
    return j;
}

} // namespace Core::NativeE3Census
