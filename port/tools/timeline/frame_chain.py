#!/usr/bin/env python3
"""Frame chain of a timeline capture (`/timeline_capture?seconds=2&tag=wf` -> user/log/timeline_wf.bin + .json).

Answers "what sets the frame time" from the game's own task spans: for every game frame (Task:EventBeginFrame to the next one)
it prints when the main tasks start, how long the render thread waits for the game, where no game task or job runs, and who
woke the game's task scheduler after such a hole. No numpy needed.

    python frame_chain.py path/to/timeline_wf        (without .bin / .json)
"""
import bisect
import collections
import json
import statistics
import struct
import sys

# the game's tasks the summary follows, in frame order (names as the spans file gives them)
MILESTONES = [
    ("Task:AnimationTick", "end"),
    ("Task:DelayAnimPostPhysicsTick", "start"),
    ("Task:SceneCull", "start"),
    ("Task:SceneDraw", "start"),
    ("Task:PhysicsBegin", "start"),
    ("Task:RenderFrame", "start"),
    ("Task:PhysicsEnd", "start"),
]
JOB_SPAN = 1000  # "Job:Run"


def load(base):
    meta = json.load(open(base + ".json"))
    raw = open(base + ".bin", "rb").read()
    count = struct.unpack_from("<Q", raw, 8)[0]
    events = list(struct.iter_unpack("<QQQII", raw[32:32 + count * 32]))
    events.sort(key=lambda e: e[0])
    return meta, events


def main():
    base = sys.argv[1]
    meta, events = load(base)
    threads = {int(k): v for k, v in meta["threads"].items()}
    names = {int(k): v for k, v in meta["names"].items()}
    span_names = {int(k): v for k, v in meta["spans"].items()}
    span_ids = {v: k for k, v in span_names.items()}
    types = {v: int(k) for k, v in meta["types"].items()}
    tname = lambda t: threads.get(t, "tid%d" % t)
    mask56 = (1 << 56) - 1

    spans = []  # (start, end, span id, thread)
    for ns, arg, arg2, ty, th in events:
        if ty == types["GuestSpan"]:
            spans.append((arg2, ns, arg >> 48, th))
    spans.sort()
    by_id = collections.defaultdict(list)
    for s, e, k, th in spans:
        by_id[k].append((s, e))
    begins = [s for s, e in by_id[span_ids["Task:EventBeginFrame"]]]
    if len(begins) < 3:
        print("no game frames (Task:EventBeginFrame spans) in the capture")
        return

    def first(name, lo, hi):
        for s, e in by_id.get(span_ids.get(name, -1), []):
            if lo <= s < hi:
                return s, e
        return None

    rows = collections.defaultdict(list)
    holes = []
    for i in range(1, len(begins) - 1):
        b, nb = begins[i], begins[i + 1]
        rows["frame"].append((nb - b) / 1e6)
        for name, edge in MILESTONES:
            x = first(name, b - 1_000_000, nb)
            if x:
                rows["%s %s" % (name, edge)].append(((x[1] if edge == "end" else x[0]) - b) / 1e6)
        last = b
        hole = 0
        for s, e, k, th in spans:
            if e <= b or s >= nb:
                continue
            s = max(s, b)
            if s > last:
                hole += s - last
            last = max(last, min(e, nb))
        hole += max(0, nb - last)
        holes.append(hole / 1e6)

    print("game frames: %d, frame %.2f ms mean (p10 %.2f, p90 %.2f)" % (len(rows["frame"]), statistics.mean(rows["frame"]),
                                                                        *statistics.quantiles(rows["frame"], n=10)[0:9:8]))
    print("\nwhen the game's main tasks run (ms from the start of the game frame, mean):")
    for name, edge in MILESTONES:
        v = rows.get("%s %s" % (name, edge))
        if v:
            print("  %-36s %-5s %6.2f" % (name, edge, statistics.mean(v)))
    print("  %-36s       %6.2f" % ("next frame starts", statistics.mean(rows["frame"])))
    print("\nper frame, no game task or recorded job runs anywhere: %.2f ms mean" % statistics.mean(holes))

    # the render thread and the submit-done thread: what they wait on
    waits = collections.defaultdict(float)
    for ns, arg, arg2, ty, th in events:
        if ty == types["GameWait"] and tname(th) in ("ps4render", "ps4render_submi"):
            waits[(tname(th), names.get(arg2, hex(arg2)))] += (arg & mask56) / 1e6
    print("\nrender side waits (ms per game frame):")
    nframes = len(rows["frame"]) + 2
    for (th, obj), total in sorted(waits.items(), key=lambda x: -x[1]):
        print("  %-16s %-28s %6.2f" % (th, obj, total / nframes))

    # the scheduler's wake after the largest hole kind: who posted it, how long after the last recorded job ended
    job_ends = sorted(e for s, e, k, th in spans if k == JOB_SPAN)
    sched_waits = [(ns - (arg & mask56), ns, arg2, th) for ns, arg, arg2, ty, th in events if ty == types["GameWait"] and tname(th).startswith("tid")]
    if sched_waits:
        sched_thread = collections.Counter(th for _, _, _, th in sched_waits).most_common(1)[0][0]
        late = []
        for s, wake, obj, th in sched_waits:
            if th != sched_thread:
                continue
            i = bisect.bisect_right(job_ends, wake) - 1
            if i >= 0 and wake - job_ends[i] > 500_000:
                late.append((wake - job_ends[i]) / 1e6)
        if late:
            print("\nthe task scheduler (%s) woke %d times more than 0.5 ms after the last recorded job ended (mean %.2f ms): small jobs below TLG_JOB_MIN_US run in that time"
                  % (tname(sched_thread), len(late), statistics.mean(late)))


if __name__ == "__main__":
    main()
