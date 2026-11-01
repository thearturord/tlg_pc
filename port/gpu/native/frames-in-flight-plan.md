# Frames in flight: from the workaround to a proper fix (plan, 2026-11-01)

Today the recommended setting is `TLG_FRAMES_IN_FLIGHT=0`: the game waits for its own frame and the presenter waits for the frame it just
made. With 1 or 2 the game runs ahead and one frame now and then is dark or bright while walking (sunlight and fog missing for that frame;
README, known issues). The goal: find what that frame reads wrong, fix that one cause, then let the game run two frames ahead with a gate
that waits only for what really has to be finished, and remove the depth 0 waits.

Rules as always: measuring first, every change behind a switch and off by default, the owner plays it, then the default.

## How the gate works today (code)

Game side (Ghidra, base 0x1000000):
- `JsSwapChain_SubmitAndFlip` (0x1c4dfa0) on the render thread: from its second call it waits until the done count `0x6b03798` is above 0
  and takes one, then calls `sceGnmSubmitAndFlipCommandBuffers` and adds one to the request count `0x6b03790`.
- `ps4render_submitDone_ThreadMain` (0x2494d40): waits for a request, then for the end-of-pipe event (`EOP QUEUE`), then adds one to the
  done count. So the game itself keeps one frame in flight: frame N is submitted after the EOP of frame N-1.
- The engine has 3 frame contexts and 3 display buffers (`RenderDev_vfn10_RotateFrameResources`, modulo 3 in at least eight functions),
  so two frames ahead is the most it supports (`FrameGate::MaxDepth = 2`).

Host side (`native/native_frame_gate.{h,cpp}`, one setting read by every place below):

| place | depth 0 | depth 1 (default in code) | depth 2 |
|---|---|---|---|
| game's done count (`CreditGuestGate`) | as the game does it | as the game does it | one credit at the first submit: two frames ahead |
| Gnm submit (`gnmdriver.cpp`, `WaitSubmitAllowed`) | waits for an idle GPU | waits for an idle GPU | no idle wait; `WaitForSlot`: fewer than 2 unfinished frames |
| after the flip submit (`WaitOwnFrame`, `TLG_FIF_GAME_WAIT`) | 1: until its own frame's EOP | 0 | 0 |
| presenter (`vk_presenter.cpp`, `TLG_FIF_PRESENTER_WAIT`) | waits for the frame it just made | `frames_queued` flips back | 2 flips back |
| fences, occlusion answers, readbacks | ordered on the raster thread; the CPU's maps of GPU-written memory wait for their downloads (`wait.readback_gate_map`) | same | same |

`TLG_FIF_GAME_WAIT=2` waits instead until the raster thread passed the frame's end (`NoteRasterReached`, queued before the EOP signal):
everything the frame reads from guest memory has been read, the GPU may still run.

The end of pipe: `SignalGraphicsEop` -> `SignalFenceAfterReadbacks` (waits for the frame's readbacks) -> `RunFenceSignal` ->
`FrameGate::NoteEndOfPipe` and the game's EOP event.

Waits between the queues: a command stream wait on a label passes on the recorded fence timeline when `fence_shortcut` allows it
(`Liverpool::FenceShortcutEnabled`, `TLG_READBACK_FENCE_WAIT_SHORTCUT`, default 1 = graphics queue waits; live: `/set?name=fence_shortcut&value=N`).

## Phase 0: bisect with the switches that exist (no code; skip what the open bisect already answered)

Default spot, walking the same path for 2 minutes per run, counting dark/bright frames:

| run | settings | if the bad frame goes away, the cause is |
|---|---|---|
| A | `TLG_FRAMES_IN_FLIGHT=1` | (the bug, as today) |
| B | A + `TLG_FIF_GAME_WAIT=2` | the raster thread reads guest memory the game has already rewritten for the next frame (read-ahead) |
| C | A + `TLG_FIF_GAME_WAIT=1` (and B did not help) | a GPU result the game reads before the GPU wrote it |
| D | A + `TLG_FIF_PRESENTER_WAIT=1` | presentation: a frame shown before its own work finished |
| E | A + `/set?name=fence_shortcut&value=0` | a wait between the queues passed on the recorded fence too early (fog or sun from async compute) |

One run answers one question; the table decides which detector of phase 1 is needed first.

## Phase 1: catch the bad frame and see what it read (measuring only)

1. **Bad frame catcher** (`TLG_FLASH_DETECT=1`): the presenter reduces each final image to its mean luminance on the GPU (one small
   compute pass, read back a few frames late so nothing waits). A frame whose value jumps away from both neighbours by more than a
   threshold is flagged: a log line, a timeline marker, and the detectors below write their ring of the last 8 frames.
2. **Read-ahead detector** (for B): at the frame's submit (`NoteSubmitFlip`, on the game thread) hash the announced ranges of that frame
   (`TakeBatch`); when the raster thread uploads or copies guest bytes of those ranges for that frame (`SynchronizeMemory`, stream and
   object copies, texture uploads) hash them again. A difference means the game changed the bytes between the submit and the read. Report
   by owner (`NativeDevice::OwnerName`) and announcer (`WhoWrote`: Map, skin, **light**, stream copy, recording / scene slots, texture,
   instance group). Sampled (every Nth range) outside a flagged frame, all ranges in the 8 frames around one.
3. **Early fence detector** (for E): every wait that passes on a recorded fence (`MatchRecordedFence` satisfied, `OnWaitPassedOnFence`)
   records the label, the queue, the recorded value and whether the real label value had arrived; around a flagged frame, which draws
   read what the async queue wrote after that wait.
4. **GPU result detector** (for C): the game's reads of GPU-written memory that did not go through the readback gate (a write fault on a
   page the GPU wrote while the frame that wrote it is unfinished).

Exit: the flagged frames name one range or one wait every time.

## Phase 2: fix that cause (behind a switch, verify first)

By the cause found:
- **Read-ahead:** keep the frame's copy. The announced bytes of a frame are copied when the game submits it (the host already copies them
  later; this moves the copy to the submit for the ranges the detector names, or for all ranges of the owners it names). The raster
  thread then reads the frame's copy, never the live guest memory.
- **Early fence:** the waits on the labels the detector names stop passing on the recorded fence; they wait for the real value (or the
  GPU barrier of shortcut mode 3 is added for them).
- **GPU result:** that result gets its own gate: the game's map or read of it waits for the GPU (as `wait.readback_gate_map` does for
  explicit downloads), not the whole frame.
- **Presentation:** the presenter waits for that frame's own ready tick only when it would show an unfinished frame.

Exit: phase 0's run A shows no bad frame in a long walk, a cutscene and a loading screen.

## Phase 3: the proper gate

With the cause fixed:
1. **Depth 2 by default, gated on consumption.** The game's done count is credited when the raster thread has read the frame
   (`NoteRasterReached`, mode 2), not when the GPU finished it: the host copies guest memory when it reads it, so the game may rewrite
   it as soon as it was read. GPU results the CPU reads keep their own gates (readbacks, and whatever phase 2 added).
2. **No GPU idle wait on submit** at any depth (`WaitSubmitAllowed` keeps only the slot wait).
3. **Presenter:** waits only to keep pacing even (`frames_queued`), never for its own frame.
4. Remove depth 0's waits (`WaitOwnFrame` mode 1, `PresenterWaitsOwnFrame`), `TLG_FIF_GAME_WAIT` and `TLG_FIF_PRESENTER_WAIT`;
   `TLG_FRAMES_IN_FLIGHT` keeps 1 and 2 for comparison, then goes; the README stops recommending 0.

Measure each step with Shift+F9 (`wait.frame_gate_*`, `wait.gnm_submit_lock`, `present.wait_*`) and the frame chain of a timeline
capture (`port/tools/timeline/frame_chain.py`): the render thread's wait for the GPU and the game's wait in `SubmitAndFlip` should be 0.

## Order with the raster plan

The fps gain of this plan shows once the raster thread is no longer the limit (`raster-optimization-plan.md`); its latency gain shows at
once. Phases 0 and 1 can run now; phase 3 is worth most after the raster plan's phase 2.
