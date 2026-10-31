// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_flip.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "native/native_debug_server.h"

namespace Core::NativeFlip {

namespace {
constexpr u64 RingSize = 512;
struct Slot {
    u64 id{};
    Request request{};
};
std::mutex g_mutex;
std::array<Slot, RingSize> g_ring{};
u64 g_next_id = 1;
std::atomic<u64> g_published{}, g_taken{}, g_gone{};
std::mutex g_variant_mutex;
std::map<u32, u64> g_variants;
} // namespace

u64 Publish(const Request& request) {
    std::scoped_lock lock{g_mutex};
    const u64 id = g_next_id++;
    g_ring[id % RingSize] = {id, request};
    g_published.fetch_add(1, std::memory_order_relaxed);
    return id;
}

bool Take(u64 id, Request& out) {
    std::scoped_lock lock{g_mutex};
    const Slot& slot = g_ring[id % RingSize];
    if (slot.id != id) {
        g_gone.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    out = slot.request;
    g_taken.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void NoteVariant(u32 payload_type) {
    {
        std::scoped_lock lock{g_variant_mutex};
        ++g_variants[payload_type];
    }
    if (payload_type != 0x68750777u) {
        // the game was only seen to use the plain prepareFlip; a variant with a label or an interrupt is a rare event (its typed handling was checked against the old rewrite only)
        Core::RareEvents::Note("flip_variant_" + std::to_string(payload_type), "a prepareFlip packet that is not the plain one");
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["requests_published"] = g_published.load();
    j["requests_taken_by_the_command_processor"] = g_taken.load();
    j["requests_gone_when_taken"] = g_gone.load();
    {
        std::scoped_lock lock{g_variant_mutex};
        auto& v = j["prepare_flip_variants"] = nlohmann::json::object();
        for (const auto& [type, n] : g_variants) {
            v[type == 0x68750777u ? "plain" : type == 0x68750778u ? "label" : type == 0x68750780u ? "interrupt" : type == 0x68750781u ? "interrupt_and_label" : "other_" + std::to_string(type)] = n;
        }
    }
    return j;
}

} // namespace Core::NativeFlip
