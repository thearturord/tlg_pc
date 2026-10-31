// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <map>
#include <string>

#include "common/types.h"
#include "native/native_timeline.h"

/// The stall profiler (key F8, endpoint /stall_capture): a capture of the frame timeline that is not written to disk as events but analysed on the spot, to answer
/// "what makes my frames slow": the frames are the game's submits (SubmitFlip), a slow frame is one that is clearly longer than the median, and for every wait the
/// game's threads make (thread, object, **guest call site**), every wait of our own threads (command processor, raster, encoder) and the GPU's busy time the report
/// says how much more of it the slow frames contain than the normal ones. The ranked list of that excess is the answer; the worst frames are listed one by one.
/// F9 (the full frame profiler) is not touched.
namespace Core::Stall {

/// Starts a stall capture of `seconds` seconds (false when a capture is running). With `open_in_browser` the finished report (an HTML page with charts, user/log/stall_<time>.html)
/// is opened in the default browser.
bool Start(u32 seconds, bool open_in_browser = false);
/// Stops a running stall capture early; it is analysed as if its time had run out.
bool Stop();
/// True while a stall capture runs.
[[nodiscard]] bool Recording();
/// True when the capture that just ran (or runs) is a stall capture; read by the timeline writer.
[[nodiscard]] bool ThisIsStallCapture();

/// Called by the timeline writer when the stall capture ended: analyses the events and keeps the report.
void Analyze(const FrameTimeline::CapEvent* events, u64 count, u64 start_ns, u64 end_ns, const std::map<u32, std::string>& threads, const std::map<u64, std::string>& object_names);

/// The report of the last stall capture: a short text for the screen (first lines), the full text, and JSON for the debug server.
[[nodiscard]] std::string ScreenText();
[[nodiscard]] std::string FullText();
[[nodiscard]] std::string ReportJson();
/// Seconds since the last report was made (a large value when there is none); the overlay shows the report for a while.
[[nodiscard]] double ReportAgeSeconds();

} // namespace Core::Stall
