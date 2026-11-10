// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
#include "native/native_texture_version.h"
#include <array>
#include <cstdlib>
#include <memory>
#include <nlohmann/json.hpp>

namespace Core::TextureVersion {

u32 Mode() noexcept {
    static const u32 mode = [] {
        const char* e = std::getenv("TLG_TEXTURE_VERSION");
        return e != nullptr && e[0] == '1' ? 1u : 0u; // (2, real, is not built yet)
    }();
    return mode;
}

namespace {
// open addressing by texture id; entries are never removed (a slot outlives its texture: an image may still point at it)
constexpr u32 Bits = 16;
constexpr u32 Probe = 64;
struct Cell {
    std::atomic<u64> id{0};
    Slot slot;
};
Cell* Table() noexcept {
    static const std::unique_ptr<Cell[]> table{new Cell[std::size_t{1} << Bits]};
    return table.get();
}
u32 Home(u64 id) noexcept {
    return static_cast<u32>((id * 0x9E3779B97F4A7C15ull) >> (64 - Bits));
}
Slot* FindOrInsert(u64 id, bool insert) noexcept {
    Cell* t = Table();
    const u32 h = Home(id);
    for (u32 i = 0; i < Probe; ++i) {
        Cell& c = t[(h + i) & ((1u << Bits) - 1)];
        u64 k = c.id.load(std::memory_order_acquire);
        if (k == id) {
            return &c.slot;
        }
        if (k == 0) {
            if (!insert) {
                return nullptr;
            }
            u64 expected = 0;
            if (c.id.compare_exchange_strong(expected, id, std::memory_order_acq_rel)) {
                return &c.slot;
            }
            if (expected == id) {
                return &c.slot;
            }
        }
    }
    return nullptr;
}
std::array<std::atomic<u64>, Count_> counts{};
} // namespace

void Note(Count c) noexcept {
    counts[c].fetch_add(1, std::memory_order_relaxed);
}

void Created(u64 id, bool in_class) noexcept {
    if (Mode() == 0 || id == 0) {
        return;
    }
    if (!in_class) {
        Note(TexturesNotInClass);
        if (Slot* s = FindOrInsert(id, false)) {
            s->state.store(2, std::memory_order_release);
        }
        return;
    }
    Slot* s = FindOrInsert(id, true);
    if (s == nullptr) {
        Note(TexturesNotInClass); // (the probe is full: the texture stays on the address route)
        return;
    }
    u32 none = 0;
    if (s->state.compare_exchange_strong(none, 1, std::memory_order_acq_rel)) {
        Note(TexturesInClass);
    }
}

void Exclude(u64 id) noexcept {
    if (Mode() == 0) {
        return;
    }
    if (Slot* s = FindOrInsert(id, false)) {
        if (s->state.exchange(2, std::memory_order_acq_rel) == 1) {
            Note(TexturesExcluded);
        }
    }
}

void Bump(u64 id) noexcept {
    if (Mode() == 0 || id == 0) {
        return;
    }
    Slot* s = FindOrInsert(id, false);
    if (s == nullptr || !s->InClass()) {
        Note(BumpsNotInClass);
        return;
    }
    s->version.fetch_add(1, std::memory_order_acq_rel);
    Note(Bumps);
}

Slot* Find(u64 id) noexcept {
    return id != 0 ? FindOrInsert(id, false) : nullptr;
}

nlohmann::json Report() {
    const auto c = [](Count k) { return counts[k].load(std::memory_order_relaxed); };
    nlohmann::json j;
    j["mode"] = Mode();
    j["textures"] = {{"in_class", c(TexturesInClass)}, {"not_in_class", c(TexturesNotInClass)}, {"excluded_later", c(TexturesExcluded)}};
    j["writes"] = {{"writer6_class_bumps", c(Bumps)}, {"writer6_other_textures", c(BumpsNotInClass)},
                   {"ADDRESS_WRITE_ON_CLASS_TEXTURE_MUST_BE_0", c(AddressWriteOnClass)}};
    j["refresh"] = {{"images_linked", c(ImagesLinked)}, {"checked", c(RefreshChecked)}, {"first_upload_not_compared", c(RefreshFirstUpload)},
                    {"gpu_written_not_compared", c(RefreshGpuSkipped)}, {"both_clean", c(BothClean)}, {"both_dirty", c(BothDirty)},
                    {"version_dirty_flags_clean", c(VersionOnly)}, {"VERSION_CLEAN_FLAGS_DIRTY_MUST_BE_0", c(FlagsOnly)}};
    return j;
}

} // namespace Core::TextureVersion
