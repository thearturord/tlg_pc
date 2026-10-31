// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_e3_census.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <xxhash.h>

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
    return j;
}

} // namespace Core::NativeE3Census
