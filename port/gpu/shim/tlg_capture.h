// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: F12 saves a screenshot of the game frame (PNG, full resolution), Shift+F12 records a 5 second video (MP4, H.264, up to 1920x1080) into
// user/screenshots. The frame is copied before the overlay is drawn (no menu or FPS counter in the files); the copies are read back when the GPU is done
// with them and written by a thread of their own, so the game does not wait.

#pragma once

#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace TlgCapture {

/// Window thread: the next presented frame is saved as a PNG.
void RequestScreenshot();
/// Any thread: like RequestScreenshot, the PNG goes to `path` (the glitch marker's folder) and no "saved" toast is shown.
void RequestScreenshotTo(const std::string& path);
/// Window thread: the next `seconds` of presented frames are recorded (ignored while a recording runs).
void RequestVideo(double seconds);

/// Debugger mode (Shift+F5): keep a 256x144 thumbnail of every presented frame (the newest 480); any thread.
void SetThumbs(bool on);
/// Writes the kept thumbnails to `path` (thumbs.bin: "TLTH0001", u32 count, width, height, then per frame u64 id, u64 FrameTimeline ns, BGRA pixels); returns the count.
unsigned DumpThumbs(const std::string& path);

/// Present thread: `image` (layout TransferSrcOptimal, `format`) is the game frame of the command buffer `cmdbuf`, submitted as `tick`.
void Record(const Vulkan::Instance& instance, vk::CommandBuffer cmdbuf, vk::Image image, u32 width, u32 height, vk::Format format, u64 tick);
/// Present thread, every frame: copies the GPU finished are handed to the writer.
void Poll(Vulkan::Scheduler& scheduler);

} // namespace TlgCapture
