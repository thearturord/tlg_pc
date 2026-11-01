# Memory ownership: removing the frame gates (plan, 2026-11-01)

## In short

The game and our renderer share the game's memory. The game writes a frame's data, submits the frame, and moves on; our renderer reads
that data later (the command processor when it plans the frame, the raster thread when it binds and uploads). If the game is allowed to
run ahead, it can rewrite the memory before we have read it, and we draw with the next frame's data. That is why there are gates: the
game is held back until we are done with its memory (frames in flight 0 and 1, the GPU idle wait on submit, the game's own end-of-pipe
gate). They cost latency and throughput, and when they are relaxed (frames in flight 1 or 2) a frame can come out wrong (the dark or
bright frame while walking is one such case).

The proper fix is **ownership**: the renderer keeps its own copy of every byte a frame needs, taken when the game wrote it (at the latest
when it submitted the frame), and never reads the game's memory for that frame afterwards. Then the game may reuse its memory at once,
and the gates are no longer needed for correctness. Only one kind of gate stays: the CPU waiting for a result the GPU computes (readbacks),
and that becomes a gate on that one result instead of on the whole frame.

Goal: the game runs up to two frames ahead (the most its engine supports), no frame reads data from another frame, and no thread waits
for the GPU except for a result it actually reads.

## 1. When the renderer reads the game's memory today

A frame passes these points: **T0** the game writes data; **T1** the game's render thread records the frame (our hooks build native records:
the draw state and the descriptor words are copied here); **T2** the game submits (`sceGnmSubmitAndFlipCommandBuffers`); **T3** our command
processor plans each operation; **T4** our raster thread applies the announced writes, binds and uploads; **T5** the GPU runs (on our
copies); **T6** GPU results go back to the game's memory (labels, readbacks).

| data | where it lives | read by us at | already ours? |
|---|---|---|---|
| draw state, descriptor words (V#, T#, S#) of native draws | native records | T1 | yes (copied when the record is made) |
| command buffer of a pure graphics submit | `NativeDirect::Build` entries | T2 | yes |
| command buffer of other submits (packet path) | game memory (the frame context's command memory) | T3 | no; reused only after 3 frames (safe at depth 2), `TLG_COPY_GPU_BUFFERS` can copy it at T2 |
| descriptor tables / user data the walker or fetch shader reads (fallbacks) | game memory | T3, T4 | no |
| shader binaries | game memory | T3 (pipeline lookup, by hash) | effectively static |
| small constant buffers (constant ring, N5) | game memory | T3 (front copies into the ring) | copied at T3; Map writes up to 64 KB can come from the writer snapshots (N5d, `TLG_SNAPSHOTS`, off by default) |
| vertex and index ranges (geometry ring, N5g, `TLG_RING_GEOMETRY`) | game memory | T3 | copied at T3 when on |
| every other buffer read (`ObtainBuffer`: stream copy, object copy, arena `SynchronizeMemory`) | game memory | T4 | no: read at bind time |
| textures the CPU updates (`TextureUpdateSubresource`, about 680 calls per frame) | game memory | T4 (`RefreshImage`, `texture.refresh.obtain_bytes`) | no |
| indirect draw and dispatch arguments | game memory or GPU written | T4 / GPU | GPU-written ones stay on the GPU |
| labels and fences (end of pipe, release mem) | written by us after the work | T6 | the game reads them: GPU to CPU |
| readbacks: copies into STAGING resources, counter (occlusion) results, GPU-written bytes the CPU reads by fault | written by us at T6 | T6 | GPU to CPU |

Per frame the game announces about 3 MB of CPU writes (`updates.bytes_first`, about 1,500 ranges, about 2 KB each); the raster thread
uploads about 2.6 MB from game memory at bind time (`buffer.uncached.arena_synchronized`). That is the memory read late.

Which of it is dangerous when the game runs ahead: memory the game **rewrites in place** every frame (scene slots, lights, skinning outputs,
instance groups; the writers `SceneSlotWrites`, `LightWrites`, `SkinWrites`, `InstanceGroupBind`, `Map`). Memory the game takes from its
frame allocator (`AllocFrameGarlic`, a ring) and its command memory is reused only after three frames and is safe up to depth 2.

## 2. The gates and why each one exists

| gate (`native_frame_gate.h` lists six places) | protects | after ownership |
|---|---|---|
| 1. the game's done count (`SubmitAndFlip` waits for one end of pipe per frame; depth 2 credits one more) | the game's frame memory and command memory, and the GPU results the game reads after the end of pipe | credited when we own the frame's data; GPU results get their own gates (step 6) |
| 2. Gnm submit: GPU idle wait at depth 0 and 1 (`WaitSubmitAllowed`), slot wait at depth 2 (`WaitForSlot`) | the same, from our side | the idle wait goes; the slot wait becomes a capacity check of our copies |
| 3. the presenter's wait for frames ahead (`vk_presenter.cpp`, own frame at depth 0) | even pacing; at depth 0 shows only finished frames | pacing only |
| 4. flip queue and display buffer labels (`DirectWaitDisplay`) | the game's display buffers (3) | unchanged (presentation order; our images are host images) |
| 5. fences, occlusion answers, readbacks | **the CPU reading what the GPU computed** | stays, per result |
| 6. double-buffered host resources (GPU side) | GPU-side reuse | unchanged |

Only gate 5 is about data the CPU needs from the GPU. Gates 1 to 3 are about memory the game and we share.

## 3. The design

### 3.1 The frame data store (the core)

A store of versions of the game's memory, filled at write time:
- **Fill (T0):** every announced write (the writer hooks already report each write with its address and size, `NativeUpdates`) also
  copies its bytes into the store, stamped with the number the next native record gets (`NativeDraw::NextSeq()`, as the N5d snapshots do).
  Per-thread append buffers (no global lock; the N5d store has one mutex, too slow for 1,500 writes per frame from several threads).
  No size cap (N5d drops writes over 64 KB).
- **Read (T3, T4):** `Read(address, size, seq)` gives the bytes as the writes before record `seq` left them: the newest version at or before
  `seq` for each part of the range, composed from several versions when writes overlap. A part with no version was never written since
  it was last read for a frame, so the game's memory holds it unchanged and is read directly.
- **Free:** versions older than the oldest frame still in flight (the raster thread's done position) are dropped.
- **Size:** about 3 MB per frame announced, three frames in flight, plus textures: a few tens of MB. If the store is full the submit waits
  (the only wait left on this path, sized never to happen; counted).

### 3.2 Every reader goes through it

The command processor's constant and geometry rings (T3), the buffer cache's stream copies, object copies and `SynchronizeMemory` (T4),
the texture cache's `RefreshImage` (T4), the walker and fetch reads of the fallbacks, and the packet path's command buffer read. Each one
asks the store with the record number of the operation it serves instead of reading the game's memory.

### 3.3 Writes nobody announces

A byte the game writes without a writer hook is not in the store and is read from the game's memory late. Close the gap:
- census of such writes (the page tracker's write faults and the B8/B9 verify already detect "changed by the CPU with no announcement"),
- a hook for every writer that shows up, or, for owners that cannot be hooked, write protection of their pages while a frame that reads
  them is in flight (copy the old bytes into the store on the first write fault, then let the write go on).
Target: no read of memory that changed without a version.

### 3.4 GPU results the CPU reads (gate 5, per result)

Inventory and give each its own completion: explicit readbacks into STAGING resources (`AsyncMap` already waits for its own download,
`wait.readback_gate_map`), counter (occlusion) results, GPU-written bytes the CPU reads by fault (FaceGen; downloaded on the read fault),
and labels the game polls for GPU work. The game's read of one of them waits for that GPU work only.

### 3.5 The new gate

The game's done count is credited when the frame's data is ours (the command processor has planned the frame and every read it will need
is in the store or copied), not when the GPU finished it. Frames in flight becomes "how far ahead may the game be" (2) and nothing else;
the GPU idle wait and the depth 0 waits are removed.

## 4. Phases

Each step: measuring first, behind a switch and off by default where it changes behaviour, shadow or verify mode where a wrong answer is
possible, the owner plays, then the default and the old path go.

**Phase 0: census of late reads (measuring only).**
For every read of the game's memory after the frame's submit: who reads (command processor or raster thread), what (the table in 1),
bytes, the writer that last wrote it (`WhoWrote`), and, sampled, whether the bytes changed between the submit and the read (a hash at the
submit of the announced ranges, again at the read). Same for writes without an announcement. Output in `/stats` and Shift+F9 counters.
Exit: a table per frame: late bytes by reader and writer, and the share that would be wrong at depth 2. This also shows whether the dark
or bright frame's data is among them (`LightWrites`, `SceneSlotWrites`).

**Phase 1: the frame data store, filled, read in shadow.**
`TLG_FRAME_STORE=1`: every announced write is copied with its stamp; readers still read the game's memory but also ask the store and
compare (as `TLG_SNAPSHOT_GUARD` does today for N5d), counting differences by writer. Measure the cost on the game's threads (Shift+F9,
group `os` and the writer hooks' `hook.*` probes) and the store's size.
Exit: at depth 0 the store and the game's memory agree (differences 0 outside writes without an announcement); at depth 2 the differences
appear exactly where phase 0 predicted.

**Phase 2: readers use the store.**
`TLG_FRAME_STORE=2`, then one reader at a time (constant ring, geometry ring, buffer uploads, texture uploads, fallbacks, packet path),
each with its own counter of reads served from the store. At depth 0 nothing may change; at depth 2 the wrong frames must go.
Exit: at depth 2 a long walk, a cutscene and a loading screen show no wrong frame; the regression script passes.

**Phase 3: writes without an announcement.**
From the phase 0 census: a hook per writer, or write protection for owners without one.
Exit: the census shows no late read of memory changed without a version.

**Phase 4: GPU results per result.**
Inventory (phase 0 lists them), then each gets its own gate; the end-of-pipe credit no longer stands for them.
Exit: with the credit moved (phase 5) no CPU read of a GPU result finds old bytes (the readback counters and the FaceGen case show it).

**Phase 5: the new gate.**
`TLG_OWNERSHIP_GATE=1`: the game's done count is credited when the frame's data is ours; the GPU idle wait goes; the presenter only paces.
Measure: `wait.frame_gate_*`, `wait.gnm_submit_lock`, `present.wait_*` in Shift+F9, and the frame chain of a timeline capture
(`port/tools/timeline/frame_chain.py`): the render thread's wait for the end of pipe and the game's wait in `SubmitAndFlip` should be about 0.
Exit: no wrong frame in long play; latency from submit to present lower than at depth 0.

**Phase 6: remove the old gates.**
Depth 0's waits (`WaitOwnFrame` mode 1, `PresenterWaitsOwnFrame`), `TLG_FIF_GAME_WAIT`, `TLG_FIF_PRESENTER_WAIT`, the GPU idle wait on submit;
`TLG_FRAMES_IN_FLIGHT` stays only as "how far ahead" until it is not needed; the README stops recommending 0.

## 5. Costs and risks

- **Copies on the game's threads:** about 3 MB per frame at write time (memory speed: well under 1 ms per frame spread over the writers).
  Measured in phase 1 before anything depends on it.
- **Memory:** tens of MB for the store; counted, with a hard limit and a wait counter.
- **Overlapping writes and partial reads:** composing versions has to be exact; the shadow compare of phase 1 is the check.
- **GPU-written memory the CPU then writes:** the store must not hold a stale copy of bytes the GPU wrote (the existing GPU-written range
  tracking, `IsRangeGpuWritten`, and `PendingGpuWriteOverlaps` decide which bytes are the GPU's).
- **Writes without an announcement:** the main risk; phase 3 exists for it.
- **Textures:** the largest copies; textures the game writes once stay in the game's memory (no version, read directly).

## 6. How we know it worked

1. At frames in flight 2: no dark or bright frame (the known case) and no other frame with another frame's data in long play.
2. The game never waits for the GPU except in a read of a GPU result (Shift+F9 wait probes, timeline frame chain).
3. Latency from the game's submit to the present is lower than at depth 0; fps is not lower.
4. The regression script passes at every phase.

## 7. Order with the raster plan

Phases 0 and 1 can run now: they only measure and copy. The fps gain shows once the raster thread is no longer the limit
(`raster-optimization-plan.md`); the latency gain and the end of the wrong frames show at once.
