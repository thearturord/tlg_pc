// SPDX-License-Identifier: GPL-2.0-or-later
// Plan V (documents/tlg/native plan/native-data-path-notes.md, "Plan V: exact versions"), step V0: the identity events of the renderer, with their memory ranges.
//
// A draw's resolution (which Vulkan buffer, offset, image view a binding names) stays right until something changes WHAT the binding resolves to: an engine buffer is created or
// destroyed, a buffer is placed or moved back to the arena, an image is registered or unregistered. Those are identity events. Writes of bytes (the game's rings, announcements,
// command processor writes) are content events and are NOT logged here: they only make an object upload its changed bytes.
//
// Today five global counters stand for this and one change anywhere voids every recipe. This log keeps the memory range of each identity event, so a recipe is checked against the
// events that overlap ITS ranges only. V0 uses it in a shadow census (`TLG_VERSION_CENSUS=1`); nothing reads it otherwise.

#pragma once

#include <atomic>

#include "common/types.h"

namespace Core::NativeIdentity {

enum Kind : u32 {
    BufferPlaced = 1,    ///< an engine buffer (or a block of them) got a Vulkan buffer of its own
    BufferDropped = 2,   ///< a placed buffer was moved back to the arena or freed
    ImmutableAdded = 3,  ///< an immutable engine buffer was registered (a replaced one counts as removed first)
    ImmutableRemoved = 4,
    ImageRegistered = 5,
    ImageUnregistered = 6,
    GpuWrittenState = 7, ///< memory became GPU written (a first write) or stopped being (superseded, dropped): a read of it is served from another place
    KindCount,
};

/// An identity event over [begin, end). Any thread.
void Note(u64 begin, u64 end, Kind kind) noexcept;

/// The number of events so far (the clock a recipe stores).
[[nodiscard]] u64 Clock() noexcept;

enum class Change : u32 { None = 0, Overlaps = 1, Unknown = 2 };

/// Did an event overlap [begin, end) after `since`? `Unknown` when the log has been overrun or an entry is not published yet (counted as changed by the caller).
[[nodiscard]] Change ChangedSince(u64 since, u64 begin, u64 end, Kind* kind_out = nullptr) noexcept;

/// How many events of each kind were logged (diagnostics).
[[nodiscard]] u64 Count(Kind kind) noexcept;

} // namespace Core::NativeIdentity
