// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <nlohmann/json.hpp>
#include "native/native_pass.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace Core::NativePass {

std::atomic<bool> enabled{[] {
    const char* e = std::getenv("TLG_PASS_AUDIT");
    return e != nullptr && *e != 0 && *e != '0';
}()};

namespace {

constexpr u32 MaxPipelines = 8;

/// The pass being recorded (written by the thread that records it).
struct Working {
    bool active{};
    u32 colors{}, width{}, height{};
    u32 draws{}, indirect{};
    std::array<u32, 16> classes{};
    u32 barrier_draws{};
    u32 wrote_buffer_draws{}, wrote_image_draws{};
    std::array<const void*, MaxPipelines> pipelines{};
    u32 pipeline_count{}, pipeline_overflow{};
    bool pending_write_buffer{}, pending_write_image{};
    EndCause cause{EndCause::Other};
};
thread_local Working work;

constexpr std::array<u32, 7> BucketLimits{1, 4, 16, 64, 256, 1024, ~0u};

struct Kind {
    std::atomic<u64> passes{}, draws{};
    std::atomic<u64> pure_passes{}, pure_draws{}; ///< only draws of one class, no barrier flush before any draw, no shader write
    std::atomic<u64> barrier_passes{}, write_passes{}, mixed_class_passes{};
    std::atomic<u64> indirect_draws{}, barrier_draws{};
    std::array<std::atomic<u64>, 7> size_buckets{}; ///< passes by draw count: 1, 2-4, 5-16, 17-64, 65-256, 257-1024, more
    std::array<std::atomic<u64>, 7> size_bucket_draws{};
    std::array<std::atomic<u64>, 4> ended_by{};
    std::atomic<u64> pipelines_sum{}, pipeline_overflow_passes{};
};
std::array<Kind, 3> kinds; ///< 0 depth only, 1 one to three colour attachments, 2 four or more

struct Sample {
    u32 draws, indirect, colors, width, height, barrier_draws, wrote, pipelines;
    u8 cause;
    u8 main_class;
};
struct Site {
    u64 address;
    u64 count;
    u64 draws;
};
std::mutex site_mutex;
std::array<Site, 128> sites{};
u32 site_count = 0;
std::mutex sample_mutex;
std::array<Sample, 96> samples{};
u64 sample_count = 0;

constexpr const char* KindNames[3] = {"depth_only", "one_to_three_colour", "gbuffer_four_or_more"};
constexpr const char* CauseNames[4] = {"other_operation", "barrier_flush", "other_attachments", "submit"};

} // namespace

thread_local const void* end_site_hint = nullptr;

void Start() {
    enabled.store(true, std::memory_order_relaxed);
}

void PassBegin(u32 colors, u32 width, u32 height) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    work = {};
    end_site_hint = nullptr; // a hint left by an operation that found no open pass belongs to no pass
    work.active = true;
    work.colors = colors;
    work.width = width;
    work.height = height;
}

void SetEndCause(EndCause cause) {
    if (enabled.load(std::memory_order_relaxed)) {
        work.cause = cause;
    }
}

void SetEndSite(const void* site) {
    end_site_hint = site;
}

void NoteWrite(bool image) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    (image ? work.pending_write_image : work.pending_write_buffer) = true;
}

void NoteDraw(u32 draw_class, bool indirect, bool barrier, const void* pipeline) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    if (!work.active) {
        work.pending_write_buffer = work.pending_write_image = false;
        return;
    }
    ++work.draws;
    work.indirect += indirect ? 1 : 0;
    ++work.classes[draw_class & 15];
    work.barrier_draws += barrier ? 1 : 0;
    work.wrote_buffer_draws += work.pending_write_buffer ? 1 : 0;
    work.wrote_image_draws += work.pending_write_image ? 1 : 0;
    work.pending_write_buffer = work.pending_write_image = false;
    bool known = false;
    for (u32 i = 0; i < work.pipeline_count; ++i) {
        known |= work.pipelines[i] == pipeline;
    }
    if (!known) {
        if (work.pipeline_count < MaxPipelines) {
            work.pipelines[work.pipeline_count++] = pipeline;
        } else {
            ++work.pipeline_overflow;
        }
    }
}

void PassEnd(const void* site) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    Working done = work;
    work = {};
    if (end_site_hint != nullptr) {
        site = end_site_hint;
    }
    end_site_hint = nullptr;
    if (!done.active || done.draws == 0) {
        return; // no pass, or a pass without draws (a clear, a load / store): not interesting
    }
    Kind& k = kinds[done.colors == 0 ? 0 : done.colors >= 4 ? 2 : 1];
    u32 distinct_classes = 0, main_class = 0, main_count = 0;
    for (u32 i = 0; i < done.classes.size(); ++i) {
        distinct_classes += done.classes[i] != 0;
        if (done.classes[i] > main_count) {
            main_count = done.classes[i];
            main_class = i;
        }
    }
    const bool wrote = done.wrote_buffer_draws != 0 || done.wrote_image_draws != 0;
    const bool pure = !wrote && distinct_classes <= 1;
    k.passes.fetch_add(1, std::memory_order_relaxed);
    k.draws.fetch_add(done.draws, std::memory_order_relaxed);
    k.indirect_draws.fetch_add(done.indirect, std::memory_order_relaxed);
    k.barrier_draws.fetch_add(done.barrier_draws, std::memory_order_relaxed);
    if (pure) {
        k.pure_passes.fetch_add(1, std::memory_order_relaxed);
        k.pure_draws.fetch_add(done.draws, std::memory_order_relaxed);
    }
    k.barrier_passes.fetch_add(done.barrier_draws != 0, std::memory_order_relaxed);
    k.write_passes.fetch_add(wrote, std::memory_order_relaxed);
    k.mixed_class_passes.fetch_add(distinct_classes > 1, std::memory_order_relaxed);
    u32 bucket = 0;
    while (done.draws > BucketLimits[bucket]) {
        ++bucket;
    }
    k.size_buckets[bucket].fetch_add(1, std::memory_order_relaxed);
    k.size_bucket_draws[bucket].fetch_add(done.draws, std::memory_order_relaxed);
    k.ended_by[static_cast<u32>(done.cause) & 3].fetch_add(1, std::memory_order_relaxed);
    if (done.cause == EndCause::Other && site != nullptr) {
        const u64 address = reinterpret_cast<u64>(site);
        std::scoped_lock lock{site_mutex};
        bool found = false;
        for (u32 i = 0; i < site_count && !found; ++i) {
            if (sites[i].address == address) {
                sites[i].count += 1;
                sites[i].draws += done.draws;
                found = true;
            }
        }
        if (!found && site_count < sites.size()) {
            sites[site_count++] = {address, 1, done.draws};
        }
    }
    k.pipelines_sum.fetch_add(done.pipeline_count + done.pipeline_overflow, std::memory_order_relaxed);
    k.pipeline_overflow_passes.fetch_add(done.pipeline_overflow != 0, std::memory_order_relaxed);
    if (done.draws >= 32) {
        std::scoped_lock lock{sample_mutex};
        samples[sample_count++ % samples.size()] = {done.draws, done.indirect, done.colors, done.width, done.height, done.barrier_draws,
                                                    done.wrote_buffer_draws + done.wrote_image_draws, done.pipeline_count + done.pipeline_overflow,
                                                    static_cast<u8>(done.cause), static_cast<u8>(main_class)};
    }
}

nlohmann::json Report() {
    nlohmann::json j = nlohmann::json::object();
    j["enabled"] = enabled.load();
    for (u32 n = 0; n < kinds.size(); ++n) {
        const Kind& k = kinds[n];
        auto& o = j[KindNames[n]] = nlohmann::json::object();
        o["passes"] = k.passes.load();
        o["draws"] = k.draws.load();
        o["indirect_draws"] = k.indirect_draws.load();
        o["pure_passes_one_class_no_shader_write"] = k.pure_passes.load();
        o["draws_in_pure_passes"] = k.pure_draws.load();
        o["passes_started_by_a_draw_that_needed_a_barrier_flush"] = k.barrier_passes.load();
        o["draws_that_needed_a_barrier_flush_first"] = k.barrier_draws.load();
        o["passes_where_a_shader_wrote_a_resource"] = k.write_passes.load();
        o["passes_with_more_than_one_draw_class"] = k.mixed_class_passes.load();
        o["pipelines_per_pass_sum"] = k.pipelines_sum.load();
        o["passes_with_more_than_8_pipelines"] = k.pipeline_overflow_passes.load();
        auto& sizes = o["passes_by_draw_count"] = nlohmann::json::object();
        constexpr const char* names[7] = {"1", "2_to_4", "5_to_16", "17_to_64", "65_to_256", "257_to_1024", "more"};
        for (u32 i = 0; i < 7; ++i) {
            sizes[names[i]] = {{"passes", k.size_buckets[i].load()}, {"draws", k.size_bucket_draws[i].load()}};
        }
        auto& ended = o["ended_by"] = nlohmann::json::object();
        for (u32 i = 0; i < 4; ++i) {
            ended[CauseNames[i]] = k.ended_by[i].load();
        }
    }
    {
        std::scoped_lock site_lock{site_mutex};
        auto& cuts = j["passes_cut_by_other_operations_by_call_site"] = nlohmann::json::array();
        for (u32 i = 0; i < site_count; ++i) {
#ifdef _WIN32
            const u64 base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
#else
            const u64 base = 0;
#endif
            cuts.push_back({{"exe_rva", sites[i].address - base}, {"passes", sites[i].count}, {"draws", sites[i].draws}});
        }
    }
    auto& list = j["latest_passes_of_32_draws_or_more"] = nlohmann::json::array();
    std::scoped_lock lock{sample_mutex};
    const u64 count = std::min<u64>(sample_count, samples.size());
    for (u64 i = 0; i < count; ++i) {
        const Sample& s = samples[(sample_count - count + i) % samples.size()];
        list.push_back({{"draws", s.draws}, {"indirect", s.indirect}, {"colour_attachments", s.colors}, {"width", s.width}, {"height", s.height},
                        {"started_after_a_barrier_flush", s.barrier_draws}, {"draws_where_a_shader_wrote", s.wrote}, {"pipelines", s.pipelines},
                        {"ended_by", CauseNames[s.cause & 3]}, {"main_class", s.main_class}});
    }
    return j;
}

} // namespace Core::NativePass
