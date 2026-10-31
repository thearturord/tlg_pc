// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_raycast_cpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "native/native_debug_server.h"

// The arithmetic below is the shader's, statement by statement (documents/tlg/shaders/cs_e15f3156.glsl): the GPU compiled every `precise` operation without fusing, so
// the CPU must not fuse either.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace Core::NativeRaycast {

namespace {

constexpr u32 ClosedLists = 6; ///< lists kept for the comparison (the game reads one frame late, a second copy may already be closed)

struct Record {
    std::array<u32, 28> w{};
};

float F(u32 bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

u32 B(float value) {
    u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct Closed {
    u64 seq{};
    u64 list_destination{};
    u64 counter_destination{};
    u32 count{};     ///< hits the shader found (the atomic counter)
    u32 stored{};    ///< records inside the output buffer's capacity
    bool complete{};
    std::vector<Record> records;
};

struct Pending {
    std::vector<Record> records;
    u32 count{};       ///< hits found (the atomic counter)
    u32 capacity{};    ///< records the output buffer holds
    u64 dispatches{};
    bool complete{true}; ///< every dispatch of the list had CPU skinned vertices and a usable index buffer
    u64 first_params[8]{};
    u32 params_dwords{}, indices_dwords_seen{}, vertices_dwords_seen{}, rays_dwords_seen{};
};

struct Writer {
    u64 count{};
    u64 example_base{}, example_size{}, example_vertex_base{}, example_vertex_size{};
    u32 example_groups{};
};
std::unordered_map<u32, Writer> g_writers;                     ///< program hash -> overlaps with a ray dispatch's vertex buffer
std::vector<std::pair<u64, u64>> g_vertex_ranges;              ///< distinct vertex buffers (base, size), capped
std::mutex g_mutex;
std::unordered_map<u64, Pending> g_pending; ///< by output buffer: the shader runs for several systems, only the lists the game copies out matter
std::unordered_map<u64, u64> g_copy_sources; ///< memory to memory copies seen (source -> count), capped, to see whether an output buffer is ever copied
u64 g_last_counter_destination{};
u64 g_close_seq{};
std::deque<Closed> g_closed;

std::atomic<u64> g_dispatches{}, g_threads_run{}, g_hits{}, g_overflow{}, g_bad_input{};
std::atomic<u64> g_bad_params{}, g_bad_index{}, g_bad_vertex{}, g_bad_ray{};
std::atomic<u64> g_from_skin{}, g_from_guest{}, g_index_cache_hits{}, g_index_zero{}, g_complete_compares{}, g_complete_matched_nonempty{}, g_complete_matched_empty{}, g_complete_mismatch{};
std::unordered_map<u64, std::vector<u32>> g_index_cache; ///< static index buffers: the engine frees their guest memory after the GPU copy exists
std::atomic<u64> g_tris{}, g_ray_tests{}, g_pass_det{}, g_pass_u{}, g_pass_v{};
std::string g_first_bad;
std::string g_dump_first;
std::string g_dump_third;
std::string g_dump_skin;
std::atomic<u64> g_closes{}, g_compares{}, g_matched{}, g_matched_empty{}, g_matched_nonempty{}, g_mismatch{}, g_mismatch_count{}, g_mismatch_data{};
std::atomic<u64> g_unreadable{}, g_no_list{};
std::array<std::atomic<u64>, ClosedLists> g_matched_age{};
std::atomic<u32> g_max_ulp{};
std::string g_last_mismatch;

/// Distance of two floats in units of the last place (sign magnitude order); NaN equals NaN.
u32 UlpDistance(u32 a, u32 b) {
    if (a == b) {
        return 0;
    }
    const float fa = F(a), fb = F(b);
    if (std::isnan(fa) && std::isnan(fb)) {
        return 0;
    }
    if (std::isnan(fa) || std::isnan(fb)) {
        return 0xffffffffu;
    }
    if (std::fabs(fa - fb) < 1e-6f) {
        return 1; // near zero the unit of the last place is meaningless
    }
    const auto order = [](u32 v) -> s64 {
        return (v & 0x80000000u) ? -static_cast<s64>(v & 0x7fffffffu) : static_cast<s64>(v);
    };
    const s64 d = order(a) - order(b);
    return static_cast<u32>(d < 0 ? -d : d);
}

bool Equal(const Record& cpu, const Record& gpu, u32& worst) {
    for (u32 i = 0; i < 28; ++i) {
        if (i == 26 || i == 27) {
            if (cpu.w[i] != gpu.w[i]) {
                return false;
            }
            continue;
        }
        const u32 d = UlpDistance(cpu.w[i], gpu.w[i]);
        worst = std::max(worst, d);
        if (d > 16) {
            return false;
        }
    }
    return true;
}

u64 Key(const Record& r) {
    return (static_cast<u64>(r.w[26]) << 32) | r.w[27];
}

/// The shader, one invocation per triangle (`gid`), the loop over the rays inside.
void RunThread(const Buffers& b, u32 gid, std::vector<Record>& out, u32& count) {
    const u32* p = b.params;
    const auto P = [&](u32 n) { return F(p[n]); };
    if (!(p[2] > gid)) {
        return;
    }
    const u32 base = p[1] + gid * 3u;
    const u32 d173 = base >> 1;
    if (static_cast<u64>(d173) + 1 >= b.indices_dwords) {
        g_bad_index.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 w183 = b.indices[d173 + 1];
    const u32 w190 = b.indices[d173];
    const u32 odd = base & 1u;
    const u32 e[4] = {w190 & 0xffffu, w190 >> 16, w183 & 0xffffu, w183 >> 16};
    const u32 ia = e[odd + 2], ib = e[odd + 1], ic = e[odd];
    const auto vertex = [&](u32 index, u32 out3[3]) {
        const u64 at = (static_cast<u64>(p[0] + index) << 3);
        if (at + 2 >= b.vertices_dwords) {
            if (g_bad_vertex.fetch_add(1, std::memory_order_relaxed) == 0) {
                g_first_bad = fmt::format("gid {} base {} odd {} w190 {:#x} w183 {:#x} index {} p0 {} p1 {} p2 {} vertices_dwords {} indices_dwords {}", gid, base, odd, w190, w183, index,
                                          p[0], p[1], p[2], b.vertices_dwords, b.indices_dwords);
            }
            out3[0] = out3[1] = out3[2] = 0;
            return;
        }
        out3[0] = b.vertices[at];
        out3[1] = b.vertices[at + 1];
        out3[2] = b.vertices[at + 2];
    };
    u32 va[3], vb[3], vc[3];
    vertex(ia, va);
    vertex(ib, vb);
    vertex(ic, vc);

    const u32 w431 = (p[6] << 16) | gid;
    const u32 w432 = p[7] << 16;
    const float f433 = F(vc[0]);
    const float f434 = P(10);
    const float f435 = P(22);
    const float f436 = f433 * f434;
    const float f437 = f436 + f435;
    const float f438 = F(va[0]);
    const float f439 = P(8);
    const float f440 = P(20);
    const float f441 = f438 * f439;
    const float f442 = f441 + f440;
    const float f443 = F(vb[0]);
    const float f444 = f443 * f439;
    const float f445 = f444 + f440;
    const float f446 = P(14);
    const float f447 = F(vc[1]);
    const float f448 = f446 * f447;
    const float f449 = f448 + f437;
    const float f450 = f438 * f434;
    const float f451 = f450 + f435;
    const float f452 = f443 * f434;
    const float f453 = f452 + f435;
    const float f454 = P(9);
    const float f455 = P(21);
    const float f456 = f438 * f454;
    const float f457 = f456 + f455;
    const float f458 = f433 * f454;
    const float f459 = f458 + f455;
    const float f460 = f454 * f443;
    const float f461 = f460 + f455;
    const float f462 = P(13);
    const float f463 = F(va[1]);
    const float f464 = f462 * f463;
    const float f465 = f464 + f457;
    const float f466 = F(vb[1]);
    const float f467 = f462 * f466;
    const float f468 = f467 + f461;
    const float f469 = P(17);
    const float f470 = F(vb[2]);
    const float f471 = f469 * f470;
    const float f472 = f471 + f468;
    const float f474 = F(va[2]);
    const float f475 = f469 * f474;
    const float f476 = f475 + f465;
    const float f478 = f462 * f447;
    const float f479 = f478 + f459;
    const float f480 = P(12);
    const float f481 = f480 * f463;
    const float f482 = f481 + f442;
    const float f483 = f480 * f466;
    const float f484 = f483 + f445;
    const float f485 = f439 * f433;
    const float f486 = f485 + f440;
    const float f487 = P(16);
    const float f488 = f487 * f474;
    const float f489 = f488 + f482;
    const float f491 = f487 * f470;
    const float f492 = f491 + f484;
    const float f494 = f480 * f447;
    const float f495 = f494 + f486;
    const float f496 = f446 * f463;
    const float f497 = f496 + f451;
    const float f498 = f446 * f466;
    const float f499 = f498 + f453;
    const float f500 = F(vc[2]);
    const float f501 = f487 * f500;
    const float f502 = f501 + f495;
    const float f504 = P(18);
    const float f505 = f504 * f474;
    const float f506 = f505 + f497;
    const float f508 = f504 * f470;
    const float f509 = f508 + f499;
    const float f511 = f506 + f509;
    const float f512 = f504 * f500;
    const float f513 = f512 + f449;
    const float f515 = f469 * f500;
    const float f516 = f515 + f479;
    const float f518 = f476 + f472;
    const float f519 = f518 + f516;
    const float f520 = f511 + f513;
    const float f523 = 0.3333333432674407958984375f * (-f520);
    const float f524 = f523 + f506;
    const float f526 = 0.3333333432674407958984375f * (-f519);
    const float f527 = f526 + f476;
    const float f528 = f489 + f492;
    const float f529 = f528 + f502;
    const float f531 = 0.3333333432674407958984375f * (-f529);
    const float f532 = f531 + f489;
    const float f533 = f532 * f532;
    const float f534 = f472 - f476;
    const float f535 = f513 - f506;
    const float f536 = f527 * f527;
    const float f537 = f536 + f533;
    const float f538 = f524 * f524;
    const float f539 = f538 + f537;
    const float f540 = f509 - f506;
    const float f541 = f516 - f476;
    const float f542 = f502 - f489;
    const float f544 = 1.0f / std::sqrt(f539);
    const float f545 = f540 * f542;
    const float f546 = f492 - f489;
    const float f548 = f535 * (-f546);
    const float f549 = f548 + f545;
    const float f550 = f544 * f532;
    const float f552 = f544 * f524;
    const float f554 = f544 * f527;
    const float f556 = f534 * f535;
    const float f558 = f541 * (-f540);
    const float f559 = f558 + f556;
    const float f560 = f559 * f559;
    const float f561 = f546 * f541;
    const float f562 = f549 * f549;
    const float f563 = f562 + f560;
    const float f565 = f542 * (-f534);
    const float f566 = f565 + f561;
    const float f567 = f566 * f566;
    const float f568 = f567 + f563;
    const float f569 = 1.0f / std::sqrt(f568);
    const float f570 = f569 * f559;
    const float f572 = f569 * f566;
    const float f574 = f542 * f542;
    const float f575 = f541 * f541;
    const float f576 = f575 + f574;
    const float f577 = f535 * f535;
    const float f578 = f577 + f576;
    const float f579 = f569 * f549;
    const float f581 = f546 * f546;
    const float f582 = f546 * f542;
    const float f583 = f534 * f534;
    const float f584 = f583 + f581;
    const float f585 = f534 * f541;
    const float f586 = f585 + f582;
    const float f587 = f540 * f540;
    const float f588 = f587 + f584;
    const float f589 = f588 * f578;
    const float f590 = f535 * f540;
    const float f591 = f590 + f586;
    const float f593 = f591 * (-f591);
    const float f594 = f593 + f589;
    const float f595 = 1.0f / f594;

    g_tris.fetch_add(1, std::memory_order_relaxed);
    const u32 ray_start = p[4];
    const u32 ray_count = p[5];
    for (u32 r = 0; r < ray_count; ++r) {
        const u64 ray = static_cast<u64>(ray_start) + r;
        const u64 dir = ray * 8 + 4;
        const u64 org = ray * 8;
        if (dir + 3 >= b.rays_dwords) {
            g_bad_ray.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        const u32 u606 = b.rays[dir], u610 = b.rays[dir + 1], u614 = b.rays[dir + 2], u618 = b.rays[dir + 3];
        const float f619 = F(u610);
        const float f620 = f619 * f535;
        const float f621 = F(u614);
        const float f622 = f621 * f542;
        const float f624 = (-f621) * f541;
        const float f625 = f624 + f620;
        const float f626 = f625 * f546;
        const float f627 = F(u606);
        const float f628 = f627 * f541;
        const float f630 = (-f627) * f535;
        const float f631 = f630 + f622;
        const float f632 = f534 * f631;
        const float f633 = f632 + f626;
        const float f635 = (-f619) * f542;
        const float f636 = f635 + f628;
        const float f637 = f540 * f636;
        const float f638 = f637 + f633;
        g_ray_tests.fetch_add(1, std::memory_order_relaxed);
        if (!(f638 > 0.0f)) {
            continue;
        }
        g_pass_det.fetch_add(1, std::memory_order_relaxed);
        const float f641 = 1.0f / f638;
        const u32 u646 = b.rays[org], u650 = b.rays[org + 1], u654 = b.rays[org + 2];
        const float f656 = F(u646) - f489;
        const float f657 = f625 * f656;
        const float f659 = F(u650) - f476;
        const float f661 = F(u654) - f506;
        const float f662 = f659 * f631;
        const float f663 = f662 + f657;
        const float f664 = f661 * f636;
        const float f665 = f664 + f663;
        const float f666 = f641 * f665;
        if ((0.0f > f666) || (1.0f < f666)) {
            continue;
        }
        g_pass_u.fetch_add(1, std::memory_order_relaxed);
        const float f672 = f540 * f659;
        const float f674 = f661 * (-f534);
        const float f675 = f674 + f672;
        const float f676 = f546 * f661;
        const float f678 = f656 * (-f540);
        const float f679 = f678 + f676;
        const float f680 = f534 * f656;
        const float f682 = f659 * (-f546);
        const float f683 = f682 + f680;
        const float f685 = F(u606) * f675;
        const float f687 = F(u610) * f679;
        const float f688 = f687 + f685;
        const float f690 = F(u614) * f683;
        const float f691 = f690 + f688;
        const float f692 = f691 * f641;
        const float f693 = f692 + f666;
        const float f694 = f641 * f691;
        if ((0.0f > f694) || (1.0f < f693)) {
            continue;
        }
        g_pass_v.fetch_add(1, std::memory_order_relaxed);
        const float f700 = f675 * f542;
        const float f701 = f541 * f679;
        const float f702 = f701 + f700;
        const float f703 = f535 * f683;
        const float f704 = f703 + f702;
        const float f705 = f641 * f704;
        if (!((1.00000001168609742e-07f < f705) && (F(u618) > f705))) {
            continue;
        }
        const u32 slot = count++;
        const float f719 = F(u606) * f705;
        const float f720 = f719 + F(u646);
        const float f721 = f720 - f489;
        const float f724 = F(u610) * f705;
        const float f725 = f724 + F(u650);
        const float f727 = F(u614) * f705;
        const float f728 = f727 + F(u654);
        const float f729 = f728 - f506;
        const float f731 = f725 - f476;
        const float f732 = f546 * f721;
        const float f733 = f542 * f721;
        const float f734 = f534 * f731;
        const float f735 = f734 + f732;
        const float f736 = f541 * f731;
        const float f737 = f736 + f733;
        const float f738 = f729 * f540;
        const float f739 = f738 + f735;
        const float f740 = f729 * f535;
        const float f741 = f740 + f737;
        const float f742 = f578 * f739;
        const float f743 = f588 * f741;
        const float f747 = f741 * (-f591);
        const float f748 = f747 + f742;
        const float f750 = f739 * (-f591);
        const float f751 = f750 + f743;
        const float f752 = f748 * f595;
        const float f754 = f751 * f595;
        const float f756 = (1.0f / F(u618)) * f705;

        Record rec;
        auto& w = rec.w;
        w[0] = B(f570);
        w[1] = B(f579);
        w[2] = B(f572);
        w[3] = B(f756);
        w[4] = B(f550);
        w[5] = B(f554);
        w[6] = B(f552);
        w[7] = 1065353216u;
        w[8] = B(f570);
        w[9] = B(f579);
        w[10] = B(f572);
        w[11] = 1065353216u;
        w[12] = B(f489);
        w[13] = B(f476);
        w[14] = B(f506);
        w[15] = 1065353216u;
        w[16] = B(f492);
        w[17] = B(f472);
        w[18] = B(f509);
        w[19] = 1065353216u;
        w[20] = B(f502);
        w[21] = B(f516);
        w[22] = B(f513);
        w[23] = 1065353216u;
        w[24] = B(f752);
        w[25] = B(f754);
        w[26] = r | w432;
        w[27] = w431;
        (void)slot;
        out.push_back(rec);
    }
}

} // namespace

bool ShadowOn() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("TLG_RAYCAST_SHADOW");
        return e != nullptr && *e == '1';
    }();
    return on;
}

void ShadowDispatch(const Buffers& buffers_in) {
    Buffers buffers = buffers_in;
    bool complete = buffers.vertices_from_skin;
    if (buffers.indices != nullptr && buffers.indices_dwords != 0) {
        // The index buffers are static mesh data, uploaded once; their guest memory is reused later (zeros). Keep the first nonzero copy per buffer (shadow only; the real
        // fix is to copy the bytes when the engine announces the buffer's creation).
        const u64 key = reinterpret_cast<u64>(buffers.indices) ^ (buffers.indices_dwords << 48);
        bool nonzero = false;
        for (u64 i = 0; i < buffers.indices_dwords && !nonzero; ++i) {
            nonzero = buffers.indices[i] != 0;
        }
        std::lock_guard lock(g_mutex);
        if (nonzero) {
            auto& cached = g_index_cache[key];
            if (cached.empty()) {
                cached.assign(buffers.indices, buffers.indices + buffers.indices_dwords);
            }
        } else {
            g_index_zero.fetch_add(1, std::memory_order_relaxed);
            if (const auto it = g_index_cache.find(key); it != g_index_cache.end()) {
                buffers.indices = it->second.data();
                g_index_cache_hits.fetch_add(1, std::memory_order_relaxed);
            } else {
                complete = false;
            }
        }
    }
    if (buffers.params == nullptr || buffers.indices == nullptr || buffers.vertices == nullptr || buffers.rays == nullptr || buffers.params_dwords < 23) {
        g_bad_params.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_dispatches.fetch_add(1, std::memory_order_relaxed);
    (buffers.vertices_from_skin ? g_from_skin : g_from_guest).fetch_add(1, std::memory_order_relaxed);
    std::vector<Record> hits;
    u32 count = 0;
    const u32 threads = buffers.groups_x * 64u;
    for (u32 gid = 0; gid < threads; ++gid) {
        RunThread(buffers, gid, hits, count);
    }
    g_threads_run.fetch_add(threads, std::memory_order_relaxed);
    g_hits.fetch_add(hits.size(), std::memory_order_relaxed);
    std::lock_guard lock(g_mutex);
    {
        const std::pair<u64, u64> range{reinterpret_cast<u64>(buffers.vertices), buffers.vertices_dwords * 4};
        if (g_vertex_ranges.size() < 256 && std::find(g_vertex_ranges.begin(), g_vertex_ranges.end(), range) == g_vertex_ranges.end()) {
            g_vertex_ranges.push_back(range);
        }
    }
    auto& pending = g_pending[buffers.out_base];
    if (!complete && buffers.params[2] != 0) {
        pending.complete = false;
    }
    if (pending.dispatches == 0 || pending.dispatches == 3000) {
        auto& dump = pending.dispatches == 0 ? g_dump_first : g_dump_third;
        const auto words = [](const u32* a, u64 from, u64 n, u64 limit) {
            std::string out;
            for (u64 i = 0; i < n && from + i < limit; ++i) {
                out += fmt::format("{:08x} ", a[from + i]);
            }
            return out;
        };
        const u64 istart = buffers.params[1] / 2;
        dump = fmt::format("params: {} | indices at dword {} (p1/2): {} | indices dword 0: {} | indices last: {} | vertices 0: {} | rays 0: {} | rays 8: {}",
                           words(buffers.params, 0, 24, buffers.params_dwords), istart, words(buffers.indices, istart, 24, buffers.indices_dwords),
                           words(buffers.indices, 0, 16, buffers.indices_dwords), words(buffers.indices, buffers.indices_dwords > 16 ? buffers.indices_dwords - 16 : 0, 16, buffers.indices_dwords),
                           words(buffers.vertices, 0, 24, buffers.vertices_dwords), words(buffers.rays, 0, 8, buffers.rays_dwords), words(buffers.rays, 8, 8, buffers.rays_dwords));
    }
    if (buffers.vertices_from_skin && g_dump_skin.empty() && buffers.params[2] != 0) {
        const auto words = [](const u32* a, u64 from, u64 n, u64 limit) {
            std::string out;
            for (u64 i = 0; i < n && from + i < limit; ++i) {
                out += fmt::format("{:08x} ", a[from + i]);
            }
            return out;
        };
        g_dump_skin = fmt::format("params: {} | vertices 0: {} | vertices 100: {} | indices at p1/2: {} | rays 0: {}", words(buffers.params, 0, 24, buffers.params_dwords),
                                  words(buffers.vertices, 0, 24, buffers.vertices_dwords), words(buffers.vertices, 800, 16, buffers.vertices_dwords),
                                  words(buffers.indices, buffers.params[1] / 2, 12, buffers.indices_dwords), words(buffers.rays, 0, 8, buffers.rays_dwords));
    }
    if (pending.dispatches++ == 0) {
        pending.capacity = buffers.out_capacity;
        pending.params_dwords = static_cast<u32>(buffers.params_dwords);
        pending.indices_dwords_seen = static_cast<u32>(buffers.indices_dwords);
        pending.vertices_dwords_seen = static_cast<u32>(buffers.vertices_dwords);
        pending.rays_dwords_seen = static_cast<u32>(buffers.rays_dwords);
        for (u32 i = 0; i < 8; ++i) {
            pending.first_params[i] = buffers.params[i];
        }
    }
    for (auto& rec : hits) {
        if (pending.count < buffers.out_capacity) {
            pending.records.push_back(rec);
        } else {
            g_overflow.fetch_add(1, std::memory_order_relaxed);
        }
        ++pending.count;
    }
}

void NoteComputeWrite(u32 hash, u64 base, u64 size, u32 groups_x) {
    std::lock_guard lock(g_mutex);
    for (const auto& [vb, vs] : g_vertex_ranges) {
        if (base < vb + vs && vb < base + size) {
            auto& w = g_writers[hash];
            if (w.count++ == 0) {
                w.example_base = base;
                w.example_size = size;
                w.example_vertex_base = vb;
                w.example_vertex_size = vs;
                w.example_groups = groups_x;
            }
            return;
        }
    }
}

void NoteCounterCopy(u64, u64 destination) {
    std::lock_guard lock(g_mutex);
    g_last_counter_destination = destination;
}

void NoteMemoryCopy(u64 source, u64 destination, u64) {
    std::lock_guard lock(g_mutex);
    if (g_copy_sources.size() < 64 || g_copy_sources.contains(source)) {
        ++g_copy_sources[source];
    }
    const auto it = g_pending.find(source);
    if (it == g_pending.end()) {
        return;
    }
    auto& pending = it->second;
    Closed closed;
    closed.seq = ++g_close_seq;
    closed.list_destination = destination;
    closed.counter_destination = g_last_counter_destination;
    closed.count = pending.count;
    closed.stored = static_cast<u32>(pending.records.size());
    closed.complete = pending.complete;
    pending.complete = true;
    closed.records = std::move(pending.records);
    pending.records.clear();
    pending.count = 0;
    g_closes.fetch_add(1, std::memory_order_relaxed);
    g_closed.push_back(std::move(closed));
    while (g_closed.size() > ClosedLists) {
        g_closed.pop_front();
    }
}

void CompareAfterRead() {
    std::vector<Closed> lists;
    {
        std::lock_guard lock(g_mutex);
        if (g_closed.empty()) {
            g_no_list.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        lists.assign(g_closed.begin(), g_closed.end());
    }
    const Closed& newest = lists.back();
    if (newest.list_destination == 0 || newest.counter_destination == 0) {
        g_unreadable.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_compares.fetch_add(1, std::memory_order_relaxed);
    const u32 landed_count = *reinterpret_cast<const volatile u32*>(newest.counter_destination);

    std::string first_difference;
    bool any_count_equal = false;
    for (size_t age = 0; age < lists.size(); ++age) {
        const Closed& c = lists[lists.size() - 1 - age];
        if (c.count != landed_count) {
            continue;
        }
        any_count_equal = true;
        if (c.count > c.stored || c.count > 256) {
            continue; // the output buffer overflowed: the GPU kept an unknown subset
        }
        if (c.count == 0) {
            if (c.complete) {
                g_complete_matched_empty.fetch_add(1, std::memory_order_relaxed);
            }
            g_matched.fetch_add(1, std::memory_order_relaxed);
            g_matched_empty.fetch_add(1, std::memory_order_relaxed);
            g_matched_age[std::min<size_t>(age, ClosedLists - 1)].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // the landed list: copy it first, the game's worker thread may be about to read the next frame
        std::vector<Record> landed(c.count);
        std::memcpy(landed.data(), reinterpret_cast<const void*>(c.list_destination), sizeof(Record) * c.count);
        std::unordered_map<u64, const Record*> cpu_by_key;
        cpu_by_key.reserve(c.records.size() * 2);
        for (const auto& r : c.records) {
            cpu_by_key.emplace(Key(r), &r);
        }
        u32 worst = 0;
        bool ok = true;
        for (const auto& gpu : landed) {
            const auto it = cpu_by_key.find(Key(gpu));
            if (it == cpu_by_key.end()) {
                ok = false;
                if (first_difference.empty()) {
                    first_difference = fmt::format("age {} count {}: GPU hit ray {:#x} tri {:#x} has no CPU hit", age, c.count, gpu.w[26], gpu.w[27]);
                }
                break;
            }
            if (!Equal(*it->second, gpu, worst)) {
                ok = false;
                if (first_difference.empty()) {
                    u32 field = 0;
                    for (; field < 28; ++field) {
                        if (field >= 26 ? it->second->w[field] != gpu.w[field] : UlpDistance(it->second->w[field], gpu.w[field]) > 16) {
                            break;
                        }
                    }
                    first_difference = fmt::format("age {} count {}: field {} CPU {:#010x} GPU {:#010x}", age, c.count, field,
                                                    field < 28 ? it->second->w[field] : 0, field < 28 ? gpu.w[field] : 0);
                }
                break;
            }
        }
        if (ok) {
            u32 previous = g_max_ulp.load(std::memory_order_relaxed);
            while (worst > previous && !g_max_ulp.compare_exchange_weak(previous, worst, std::memory_order_relaxed)) {
            }
            g_matched.fetch_add(1, std::memory_order_relaxed);
            g_matched_nonempty.fetch_add(1, std::memory_order_relaxed);
            if (c.complete) {
                g_complete_matched_nonempty.fetch_add(1, std::memory_order_relaxed);
            }
            g_matched_age[std::min<size_t>(age, ClosedLists - 1)].fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    g_mismatch.fetch_add(1, std::memory_order_relaxed);
    if (newest.complete) {
        g_complete_mismatch.fetch_add(1, std::memory_order_relaxed);
    }
    if (!any_count_equal) {
        g_mismatch_count.fetch_add(1, std::memory_order_relaxed);
        first_difference = fmt::format("landed counter {} equals no closed list (counts {}, {}, {}...)", landed_count, lists.back().count,
                                       lists.size() > 1 ? lists[lists.size() - 2].count : 0, lists.size() > 2 ? lists[lists.size() - 3].count : 0);
    } else {
        g_mismatch_data.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(g_mutex);
        g_last_mismatch = first_difference;
    }
    if (g_mismatch.load(std::memory_order_relaxed) <= 5) {
        Core::RareEvents::Note("raycast_shadow_mismatch", first_difference);
    }
}

nlohmann::json Report() {
    nlohmann::json j;
    j["shadow_on"] = ShadowOn();
    j["dispatches"] = g_dispatches.load();
    j["threads_run"] = g_threads_run.load();
    j["hits"] = g_hits.load();
    j["lists_closed"] = g_closes.load();
    j["compares"] = g_compares.load();
    j["matched"] = g_matched.load();
    j["matched_empty"] = g_matched_empty.load();
    j["matched_nonempty"] = g_matched_nonempty.load();
    j["matched_by_age"] = nlohmann::json::array();
    for (auto& a : g_matched_age) {
        j["matched_by_age"].push_back(a.load());
    }
    j["complete_lists_matched_empty"] = g_complete_matched_empty.load();
    j["complete_lists_matched_nonempty"] = g_complete_matched_nonempty.load();
    j["complete_lists_mismatch"] = g_complete_mismatch.load();
    j["index_buffers_zero_in_guest_memory"] = g_index_zero.load();
    j["index_cache_hits"] = g_index_cache_hits.load();
    j["mismatch"] = g_mismatch.load();
    j["mismatch_counter_equals_no_list"] = g_mismatch_count.load();
    j["mismatch_data"] = g_mismatch_data.load();
    j["overflow_records"] = g_overflow.load();
    j["dispatches_vertices_from_cpu_skin"] = g_from_skin.load();
    j["dispatches_vertices_from_guest_memory"] = g_from_guest.load();
    j["bad_params_buffer"] = g_bad_params.load();
    j["bad_index"] = g_bad_index.load();
    j["bad_vertex"] = g_bad_vertex.load();
    j["triangles_processed"] = g_tris.load();
    j["ray_triangle_tests"] = g_ray_tests.load();
    j["pass_determinant"] = g_pass_det.load();
    j["pass_u"] = g_pass_u.load();
    j["pass_v"] = g_pass_v.load();
    j["bad_ray"] = g_bad_ray.load();
    j["no_list_yet"] = g_no_list.load();
    j["unreadable"] = g_unreadable.load();
    j["worst_ulp_seen_in_matches"] = g_max_ulp.load();
    std::lock_guard lock(g_mutex);
    j["last_mismatch"] = g_last_mismatch;
    j["first_bad_vertex"] = g_first_bad;
    j["dump_first_dispatch"] = g_dump_first;
    j["dump_dispatch_3000"] = g_dump_third;
    j["dump_first_skinned"] = g_dump_skin;
    j["output_buffers"] = nlohmann::json::array();
    for (const auto& [base, pending] : g_pending) {
        j["output_buffers"].push_back({{"base", fmt::format("{:#x}", base)}, {"dispatches", pending.dispatches}, {"capacity", pending.capacity},
                                       {"params_dwords", pending.params_dwords}, {"indices_dwords", pending.indices_dwords_seen},
                                       {"vertices_dwords", pending.vertices_dwords_seen}, {"rays_dwords", pending.rays_dwords_seen},
                                       {"first_params", fmt::format("{:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}", pending.first_params[0], pending.first_params[1],
                                                                    pending.first_params[2], pending.first_params[3], pending.first_params[4], pending.first_params[5],
                                                                    pending.first_params[6], pending.first_params[7])}});
    }
    j["vertex_ranges_seen"] = g_vertex_ranges.size();
    j["vertex_writers"] = nlohmann::json::array();
    for (const auto& [hash, w] : g_writers) {
        j["vertex_writers"].push_back({{"program_hash", fmt::format("{:#x}", hash)}, {"overlapping_dispatches", w.count}, {"example_written_base", fmt::format("{:#x}", w.example_base)},
                                       {"example_written_size", w.example_size}, {"example_vertex_base", fmt::format("{:#x}", w.example_vertex_base)},
                                       {"example_vertex_size", w.example_vertex_size}, {"example_groups", w.example_groups}});
    }
    j["copy_sources"] = nlohmann::json::object();
    for (const auto& [source, n] : g_copy_sources) {
        j["copy_sources"][fmt::format("{:#x}", source)] = n;
    }
    return j;
}

} // namespace Core::NativeRaycast
