# What sets the frame time (2026-11-01, cloud round)

Scene: the default in-game spot, `cloud-e7c` build `3752058`, frames in flight 2, 120 fps patch mode. All numbers are per frame.

## Answer

1. **In the real game (about 25 fps) the raster thread sets the frame time.** It is busy about 37 ms per frame: draws and dispatches about 27 ms
   (binding and recording on the CPU; no `wait.gpu_tick`), the announcement apply 6 to 8.5 ms. Proof: leaving the draws and dispatches out
   (`TLG_ABLATE=draws,dispatches`) took the game from 25 to 68-74 fps.
2. **Behind it, the game's own CPU frame sets about 13.6 ms (about 73 fps) at this spot.** With the draws out, no host thread is saturated
   and the GPU is idle; the game's task graph runs one frame after the other and the next frame starts as soon as the previous one's
   post-physics work ends. The render side (the game's render thread, our command processor and raster thread) runs alongside and is not
   in that chain.
3. Not the limit (measured): the vblank tick, the depth of the raster queue, the frame gate, the GPU, latency between our threads.

Open: before the native renderer (about 10 days earlier) the same spot reached about 100 fps, so the game's CPU frame was 10 ms or less
then. What made it 3 to 4 ms longer is not known yet (candidates: the work our hooks do inside the game's threads, our threads competing
with the game's workers for the CPU).

## The frame loop (read from the game code and ours)

Game side (Ghidra, image base 0x1000000):

- The game's task graph runs a frame of tasks and jobs on `workerThread0..6`, scheduled by an unnamed manager thread (task spans
  `Task:*`, jobs `Job:Run`).
- `RenderFrameTask_PostAndWaitBegin` (0x1d0afa0) posts the frame to the render thread (`RenderDev_PostFrameRequest`, 0x1c551a0:
  waits on `RenderDev:m_condRequest` until the previous request was taken) and then spins until the render thread has started the
  frame (context state at +0x31ef8 >= 2).
- `ps4render_ThreadMain` (0x1c53fe0): waits on `RenderDev:m_condRequest` for a request, sets the rendering flag (0x689eed5, under
  `RenderDev:s_mutexWaitRender`), calls the render device's `RotateFrameResources` (vtable +0x10, 0x1c490e0), `BeginRecording` (+0x18,
  0x1c49540, then state = 2), `RenderFrame` (+0x28, 0x1c55690), clears the flag and broadcasts `RenderDev:s_condWaitRender`.
  `FUN_01c4efb0` (vtable +0x48) waits on that condition while the flag is set. Render device vtable: 0x26e6e68.
- `JsSwapChain_SubmitAndFlip` (0x1c4dfa0, vtable +0x30): from the second call on it waits on the condition at 0x6b03780 until the done
  count 0x6b03798 is above 0 and takes one; calls `sceGnmSubmitAndFlipCommandBuffers`; adds one to the request count 0x6b03790 and
  signals `submitWait:m_condRequest` (0x6b03770).
- `ps4render_submitDone_ThreadMain` (0x2494d40, the thread named `ps4render_submi`): waits for a request, then for the end-of-pipe event
  (`EOP QUEUE`), then adds one to the done count. Its waits are its idle time, not the game's.
- The game registers `FlipScreenTask` (vtable 0x26e8b28, run = 0x1d0b3b0: the object at render + 0x48, vtable +0x38) in the task
  registration function 0x1cff010.

Host side:

- Command processor: plans each operation (`front.*` probes: about 7 ms per frame in game) and pushes it to the raster queue
  (`TLG_RASTER_QUEUE`, default 256; the raster thread takes the whole queue in one swap, `liverpool.cpp:981`).
- Raster thread: applies the announced writes (closure `native_liverpool.cpp:634`, about 6 ms), records the draws, signals the end of
  pipe (`SignalGraphicsEop`), which credits the game's done count.
- Present thread: in the 120 mode flips only on a 120 Hz vblank tick (`driver.cpp:713`); a flip sends the game's flip event and frees
  the previous display buffer (`driver.cpp:584`), which the command processor waits for before it writes the next frame into it
  (`DirectWaitDisplay`).

## The tests

| run | fps | what it showed |
|---|---|---|
| normal, depth 2 | 25-29 | raster thread 93 % busy: draws + dispatches about 27 ms, apply 6-8.5 ms |
| `TLG_ABLATE=draws,dispatches` | 69 | raster idle half the time, command processor blocked 5 ms on a full raster queue |
| + `TLG_PROF_THREADS=all` | 65-69 | no thread saturated (raster 52-61 %, command processor 37-55 %, workers 41-44 %) |
| + `TLG_RASTER_QUEUE=2048` | 68 | the queue wait is gone, fps unchanged: not the limit |
| F9 of the ablated run | 72 | frame times only 8.4 / 16.7 / 24.8 ms (the 120 Hz tick) |
| + `TLG_VBLANK_HZ=0` (480 Hz vblank, 120 cap) | 69 | the tick pattern is gone (frames 9-21 ms), fps unchanged: not the limit |
| `/timeline_capture?seconds=2` of that run | 74 | the game's frame chain below |

The frame chain (148 frames; `port/tools/timeline/frame_chain.py`), ms from the start of the game frame:

| ms | the game's tasks |
|---|---|
| 0 - 1.7 | early update (AI, animation tick) |
| 1.7 - 5.4 | animation jobs on the workers (about 22 recorded jobs, plus smaller ones) |
| 5.4 - 8.1 | scene cull, prepare material, scene draw |
| 8.1 - 11.1 | physics (begin, update phase 0, end phase 0) |
| 11.1 - 13.6 | after physics; the next frame starts at once (`Task:EventBeginFrame` right after `Task:WorldPostAnim`) |

- The render frame is handed over at 9.0 ms; the render thread waits 8.2 ms per frame for the next request (`RenderDev:m_condRequest`).
- About 3.2 ms per frame no game task or recorded job runs; there the worker that later wakes the scheduler is on the CPU (1.4 of the
  1.8 ms before the wake): jobs below `TLG_JOB_MIN_US` (8 us) are not recorded. The scheduler wakes 15 us after its semaphore is posted.
- The F9 report's "critical path 3.6 ms" adds only the named task durations and misses the jobs and the small work; the real chain is
  13.6 ms.

## Next (to answer the open point)

1. The same spot on the build from before the native renderer: the frame time there (and the frame chain, if that build has the timeline
   capture) against 13.6 ms.
2. In this build: the time our hooks take inside the game's threads. `TLG_HOOK_PROF=1` (default 0) runs every call and mid hook's handler
   inside a profiler probe `hook.<name>` (group `hook`); capture with Shift+F9 and `TLG_PROF_THREADS=all`. A handler that runs the game's
   function itself through its original entry has that game code inside its probe too. The generated stub was tested on Linux with xbyak
   (arguments, return address, caller stack pointer, both return paths) with the switch on and off.
3. Our threads' share of the CPU next to the game's workers.
