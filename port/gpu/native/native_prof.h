// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "common/types.h"
#include "native/native_timeline.h"

/// The profiler (key Shift+F9, endpoint /prof). One tool for any thread and any question: the code carries named probes, a capture chooses which of them are recorded, and the report
/// is made from whatever was recorded without knowing what the probes mean. Adding a probe is one line at the place; nothing else changes.
///
/// PROBES (put them anywhere, in any thread's code; a probe costs one relaxed load while no profile runs):
///
///     TLG_PROF("raster.bind_buffers");              // times the rest of the enclosing block (level 2)
///     TLG_PROF_LEVEL("texture.find_view", 3);       // the same with a level: 1 coarse (an operation), 2 normal, 3 fine (a call made thousands of times per frame)
///     TLG_PROF_NAMED(scope, "raster.begin");        // a scope object `scope` that can be ended early with scope.End()
///     TLG_PROF_COUNT("buffer.memo_hit");            // counts an event (level 2)
///     TLG_PROF_ADD("buffer.bytes_streamed", size);  // counts an event with a value: the report shows the count, the sum and the maximum
///     TLG_PROF_TAG("caller", address, nullptr);     // names the operation being timed (the outermost scope that is open on this thread): the report ranks the operations by that name
///     TLG_PROF_TAG_HEX("caller", address);          // the same, shown in hexadecimal
///
/// The name before the first '.' is the probe's GROUP ("raster", "buffer", "texture", ...). A capture can record chosen groups and chosen levels, so the cost can be kept low and the
/// question narrowed: `/prof?groups=raster,buffer&level=3&threads=GpuRaster,GpuCommandProcessor&seconds=10`. Tags only attach to an outermost scope that was recorded.
///
/// WHAT A CAPTURE GIVES (user/log/prof_<time>.html / .txt / .json, also prof_latest.* and raster_latest.*): per thread the busy time per frame, the probes ranked by their own time and the tree of how they nest,
/// the operations ranked by every tag, the counters, and the slowest single operations with what took the time in each. The measured cost of a probe is subtracted (the report says how much).
/// `/prof_baseline` keeps a report; the next reports show the change against it per probe, so an optimisation can be measured by running the same capture again.
///
/// Probes are recorded on the threads the capture selects (default: the raster thread, name part "GpuRaster"); time is read from the timestamp counter, accumulated per thread (no event per scope).
/// `/prof_sites` lists every probe that has run so far.
namespace Core::Prof {

enum class Kind : u8 { Scope, Counter };

/// A probe. Declared once per place as a function-local static (see the macros); registers itself the first time the place runs.
struct Site {
    Site(const char* name, u8 level, Kind kind = Kind::Scope);
    Site(const Site&) = delete;
    Site& operator=(const Site&) = delete;

    const char* name;
    u8 level;
    Kind kind;
    u16 id{0};
    std::atomic<bool> enabled{false}; ///< the running capture records this probe (level and group are inside the capture's choice)
};

/// The name of a tag (an attribute of an operation, for example the function that issued it).
struct TagKind {
    explicit TagKind(const char* name, bool hex = false);
    const char* name;
    bool hex;
    u16 id{0};
};

struct ThreadData;
[[nodiscard]] ThreadData* EnterSlow(Site& site) noexcept;
void Leave(ThreadData* data) noexcept;
void CountSlow(Site& site, u64 value) noexcept;
void TagSlow(const TagKind& kind, u64 value, const char* label) noexcept;

inline std::atomic<bool> g_running{false};

/// True while a capture runs (use it to skip the work of computing a tag value).
[[nodiscard]] inline bool Running() noexcept {
    return g_running.load(std::memory_order_relaxed);
}

/// A timed scope.
class Scope {
public:
    explicit Scope(Site& site) noexcept : data{site.enabled.load(std::memory_order_relaxed) ? EnterSlow(site) : nullptr} {}
    ~Scope() {
        End();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    /// Ends the scope before the end of the block (for a scope whose results are declared inside it).
    void End() noexcept {
        if (data != nullptr) [[unlikely]] {
            Leave(data);
            data = nullptr;
        }
    }

private:
    ThreadData* data;
};

inline void Count(Site& site, u64 value = 1) noexcept {
    if (site.enabled.load(std::memory_order_relaxed)) [[unlikely]] {
        CountSlow(site, value);
    }
}

inline void Tag(const TagKind& kind, u64 value, const char* label = nullptr) noexcept {
    if (Running()) [[unlikely]] {
        TagSlow(kind, value, label);
    }
}

/// What a capture records.
struct Config {
    u32 seconds{10};
    u32 level{2};         ///< probes up to this level
    std::string groups;   ///< comma separated group names; empty: every group
    std::string threads;  ///< comma separated parts of thread names; empty: the raster thread ("GpuRaster"); "all": every thread
    bool open{false};     ///< open the report in the browser when it is done
};

/// The configuration of Shift+F9: the environment variables TLG_PROF_SECONDS / _LEVEL / _GROUPS / _THREADS, else the defaults above (opens the report).
[[nodiscard]] Config DefaultConfig();

/// Starts a capture (false when a capture of any kind is running).
bool Start(const Config& config);
/// Stops a running capture early; it is analysed as if its time had run out.
bool Stop();
[[nodiscard]] bool Recording();
/// True when the capture that just ran (or runs) is a profile; read by the timeline writer.
[[nodiscard]] bool ThisIsProfCapture();
/// Called by the timeline writer when the capture ended.
void Analyze(const FrameTimeline::CapEvent* events, u64 count, u64 start_ns, u64 end_ns);

[[nodiscard]] std::string FullText();
[[nodiscard]] std::string ReportJson();
[[nodiscard]] double ReportAgeSeconds();
/// The probes that exist (have run at least once): name, group, level, kind.
[[nodiscard]] nlohmann::json SitesJson();
/// Keeps the last report as the baseline later reports are compared with (`clear`: forgets it). False when there is no report to keep.
bool SetBaseline(bool clear);

} // namespace Core::Prof

#define TLG_PROF_CAT2(a, b) a##b
#define TLG_PROF_CAT(a, b) TLG_PROF_CAT2(a, b)
#define TLG_PROF_NAMED_LEVEL(var, name, level)                                                      \
    static ::Core::Prof::Site TLG_PROF_CAT(var, _site){name, level};                                \
    ::Core::Prof::Scope var { TLG_PROF_CAT(var, _site) }
#define TLG_PROF_NAMED(var, name) TLG_PROF_NAMED_LEVEL(var, name, 2)
#define TLG_PROF_LEVEL(name, level) TLG_PROF_NAMED_LEVEL(TLG_PROF_CAT(tlg_prof_scope_, __LINE__), name, level)
#define TLG_PROF(name) TLG_PROF_LEVEL(name, 2)
#define TLG_PROF_ADD(name, value)                                                                   \
    do {                                                                                            \
        static ::Core::Prof::Site tlg_prof_counter{name, 2, ::Core::Prof::Kind::Counter};           \
        ::Core::Prof::Count(tlg_prof_counter, static_cast<u64>(value));                             \
    } while (false)
#define TLG_PROF_COUNT(name) TLG_PROF_ADD(name, 1)
#define TLG_PROF_TAG(kind_name, value, label)                                                       \
    do {                                                                                            \
        static const ::Core::Prof::TagKind tlg_prof_tag{kind_name};                                 \
        ::Core::Prof::Tag(tlg_prof_tag, static_cast<u64>(value), label);                            \
    } while (false)
#define TLG_PROF_TAG_HEX(kind_name, value)                                                          \
    do {                                                                                            \
        static const ::Core::Prof::TagKind tlg_prof_tag{kind_name, true};                           \
        ::Core::Prof::Tag(tlg_prof_tag, static_cast<u64>(value), nullptr);                          \
    } while (false)
