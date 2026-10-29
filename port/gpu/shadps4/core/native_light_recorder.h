// SPDX-License-Identifier: GPL-2.0-or-later
// Dark frame flight recorder (`TLG_LIGHT_RECORDER=1`). For the compute dispatches of the lighting passes (default call sites: 0x1c6456d LightVolume, 0x1c67f59 LightVolumeRasterize, 0x1c68036
// LightVolumeChromaticAberration, 0x1c44f64 AmbientLighting; `TLG_LIGHT_RECORDER_CALLERS=hex,hex` replaces the list) it keeps, per dispatch, what every buffer and image of the dispatch was
// when it was bound: the engine id, the address, the size, a hash of the first bytes the buffer holds in guest memory at that moment, how a buffer request was answered, and for an
// image the cache image (slot and unique id) and how the lookup was served. The last 800 dispatches are written to `user/log/light_recorder_<tag>.txt` when the dark frame monitor
// (`TLG_DARK_FRAMES=1`) logs a dark frame: the dispatches of the dark frame against those of its neighbours show which input of the sun / fog pass changed (a constant buffer served from
// before an update, an image of another frame) or that none did (then the data inside was wrong).

#pragma once

#include <string>

#include "common/types.h"

namespace Core::LightRecorder {

[[nodiscard]] bool On();
[[nodiscard]] bool IsLightCaller(u64 ghidra_caller);

struct Entry {
    u8 kind{};      ///< 1 buffer, 2 image
    u8 via{};       ///< buffer: 0 normal request, 1 answered by the resident buffer without a synchronization; image: 0 FindImage, 1 same generation, 2 kept across generations
    u8 written{};   ///< the shader writes it
    u64 id{};       ///< engine id of the object (0: none)
    u64 address{};  ///< guest address of the buffer / image
    u64 size{};     ///< buffer size
    u64 uid{};      ///< image: the cache image's unique id (never reused), 0 for a buffer
    u64 hash{};     ///< buffer: hash of the first 256 bytes in guest memory now; image: the T# words hash
    u32 slot{};     ///< image: the cache image slot
    u32 words[64]{}; ///< buffer: the first 64 dwords of the guest memory now (the cascade matrices of the lighting passes are among them)
};

/// Per frame signature (raster thread): every draw and dispatch adds itself (`kind`: 0 dispatch, 1 to 5 draw of class z / shadow, g buffer, forward, other, indirect, 6 untagged draw; `key`: the
/// pipeline or program), so a frame with a pass missing, doubled or a different pipeline shows in its counts and its order independent signature.
void FrameOp(u64 present, u32 kind, u64 key, u64 image_generation);

/// The image a recorded copy dispatch (the CopyResource upload kernel at 0x247ea43 / 0x1c98a17: the average luminance copy among them) writes: its guest memory is what the CPU reads back
/// (the auto exposure takes its luminance from it). Up to 8 blocks are watched; at every present change the float contents of each (mean, non zero count) are sampled into the frame census,
/// so a frame whose exposure input was zero, stale or wrong shows as an outlier in the dump. Raster thread.
void Watch(u64 address, u64 bytes);

/// A recorded (lighting) dispatch is being bound on this thread: only then is a buffer's content worth reading.
[[nodiscard]] bool Recording();

/// A buffer of at most 1 KB bound by a z / shadow draw (raster thread): the buffers many draws of the frame share are the passes' view constants (the cascade matrices the
/// shadow maps are rendered with); their first 64 dwords are kept per frame, so the dump can tell whether the lighting reads the cascades of the same frame.
void ShadowBuffer(u64 present, u64 address, u64 size);

/// A lighting dispatch is being bound (raster thread).
void Begin(u64 present, u64 caller, u64 program);
void Note(const Entry& entry);
/// The dispatch is bound; keep the record.
void End();
/// Asks for the ring to be written to user/log/light_recorder_<tag>.txt once four more frames have been recorded (the monitor sees a dark frame a frame or two after it was presented, and
/// the frames after it are needed to tell a one frame change from a ping pong); any thread.
void Dump(const char* tag, u64 detail);

/// The texture cache's description of its image events between two generations (set once by the rasterizer; the dump prints the events of the frames around the dark one).
void SetEventDescriber(std::string (*describe)(void* context, u64 first, u64 last), void* context);

} // namespace Core::LightRecorder
