# Raster thread optimization plan (2026-11-01)

Goal: the raster thread (`shadPS4:GpuRaster`) under 16.6 ms per frame at the default spot (60 fps with the draws on), then under 8.3 ms
(120 fps). Today it is the only limit in the real game: the game's own CPU frame is 9.6 ms since the semaphore fix
(`frame-loop-findings.md`), the command processor waits on a full raster queue, the GPU is not the limit.

Rules (as for every native step): each change behind a switch, off by default; a shadow or verify mode first where a wrong answer is
possible; the owner plays it; then it becomes the default and the old path goes.

## Where the time goes (Shift+F9, real game with the semaphore fix, 19.9 fps at a heavy spot)

| raster thread, own time | ms per frame | calls per frame | what it is |
|---|---|---|---|
| `buffer.obtain` | 10.5 | 14,178 | buffer lookup per binding (`BufferCache::ObtainBuffer`): bind memo, then the uncached paths |
| `updates.apply_first` + `apply_again` | 9.1 | 3 batches | the announced CPU writes applied to the caches, each range twice (this batch, and again in the next) |
| `raster.bind_textures` | 8.7 | 4,181 | image lookup and descriptors per stage (`Rasterizer::BindTextures`) |
| `raster.bind_buffers` | 4.7 | 4,181 | buffer descriptors per stage (`Rasterizer::BindBuffers`) |
| `buffer.synchronize` | 4.0 | 3,073 | uploads of CPU-changed bytes (`SynchronizeMemory`) |
| `raster.op.draw` / `op.dispatch` (own) | 2.9 | 1,716 / 639 | the draw and dispatch calls themselves |
| `raster.begin_rendering`, `reset_bindings`, `bind_resources`, `pipeline_lookup`, `bind_index_buffer` | 3.4 | | per draw setup |

About 37 of the 46 ms are bookkeeping around the draws, not recording. At the default spot (fewer draws) the thread was at 37 ms with the
same shape.

What exists already and is built on below: the bind memo (`BufferCache::ObtainBuffer`, read-only requests up to 64 KB, epochs for
staleness), the texture set memo (`texture_sets`, a stage's resolved images by the hash of its T#s), the per-T# lookup cache
(`texture_lookup_cache`, 8192 slots), the command processor's resolution ahead (`Rasterizer::ResolveAhead`, R2 buffers / R3 image
words), the binder plan (`binder_plan`: the descriptor words of each draw from the engine's record).

## Phase 0: measure inside the two blind blocks (probes only, 1 step)

`bind_textures` (8.7 ms) and `buffer.obtain` (10.5 ms) have almost no probes inside, so 19 ms is not split. Add level 3 probes and
counters (no behaviour change):

- `BindTextures`: `texture.set_memo_key` (hashing the T#s), `texture.set_memo_hit` / `miss` (counters), `texture.lookup_cache` (the
  per-T# path), `texture.find_image` (exists), `texture.find_view` (exists), `texture.descriptor_write`, the native registry audit block
  (`audit.image_*` counters run on every image when the registry is on).
- `ObtainBufferUncached`: one probe per path (immutable resident, object copy reused / made, stream copy / reuse, obtain memo hit, arena
  with `SynchronizeMemory`, written), so the 14,178 calls are split by path and cost.
- `ApplyCpuWrite` (the apply): `BindMemoErase`, `SupersedeGpuWritten`, `NativeResources::NoteCpuWrite`, `MarkBuffersChanged` (with its
  mutex and the audit set `all_object_dirty`), `texture_cache.NoteAnnouncedWrite`.

Capture: Shift+F9, level 3, default spot, `/prof_baseline?set=1`, and keep the same spot for every later phase.
Exit: every block over 1 ms has a name. The order of phases 1 to 3 below is then checked against the numbers.

## Phase 1: the apply (9.1 ms; small steps, low risk)

1. **The re-apply (`again`, about 4 ms).** `TakeBatch` hands every range a second time with the next batch (`batch.again = applied_last`).
   Shadow first (`TLG_REAPPLY_AUDIT=1`): for each re-applied range, record whether the second apply changed anything the caches read (the
   bind memo had an entry over it, a resident buffer's generation or a dirty set changed, the texture cache marked an image). If it never
   does at the default spot, a cutscene and a loading screen: `TLG_REAPPLY=0` drops it; then the default.
2. **Audit work inside the apply.** `MarkBuffersChanged` adds every range to `all_object_dirty` ("audit: every changed byte") under
   `object_dirty_mutex` on every apply. Anything that only feeds an audit or a report moves behind that audit's switch (the same pass for
   the always-on counters on the bind path, which the profile shows as part of `bind_*`).
3. **One pass per batch instead of one per range.** The ranges of a batch are sorted once; each cache is updated with the sorted list
   (one `BuffersIn` walk, one lock of `object_dirty_mutex`, one bind memo walk) instead of per range.

Exit: apply under 3 ms per frame.

## Phase 2: reuse a draw's whole binding (the biggest expected win)

Most draws bind the same buffers and textures as the last time the same draw ran. Today each one still goes through `BindBuffers`
(per buffer `ObtainBuffer`) and `BindTextures` (set memo or per-image lookup), then writes descriptors.

**Draw binding cache** (`TLG_DRAW_BIND_CACHE`, 0 off, 1 verify, 2 use):
- Key: the pipeline, the binder plan's descriptor words of the draw (buffer V#s, image T#s, sampler S#s, already in the record), the
  stage's user data the shader reads.
- Value: what `BindResources` produced: `buffer_infos`, `image_infos`, the descriptor writes / push data, the vertex and index buffer
  bindings, and the image ids for the barriers.
- Staleness: the same epochs the existing memos use (bind memo epoch, range epoch, placement epoch, stream buffer wraps, the texture
  cache's image generation), plus any announced write or GPU write over a buffer the entry names (`BindMemoErase` already finds these).
  Entries naming stream copies drop at a new command buffer, as in the bind memo.
- Mode 1: compute both, compare, count `compared` / `different`, log the first differences. Mode 2 uses the entry.
- Written resources (storage buffers and images the shader writes) are excluded at first: each one must mark its bytes GPU written, which
  is state, not a lookup. They can join later with that marking replayed from the entry.

Exit: in mode 1 `different` stays 0 over a long play; in mode 2 `buffer.obtain` + `bind_buffers` + `bind_textures` fall by the hit rate.

## Phase 3: move the remaining apply off the raster thread

After phase 1 the apply is the ordered part only. Split it:
- Thread-safe bookkeeping (dirty sets with their own lock, the B9 tracker, the resources' CPU write notes) runs where the batch is taken
  (`TakeBatch`, on the submitting thread), before the batch is queued.
- The parts that must stay in raster order (`BindMemoErase` for the raster thread's own memo, `SupersedeGpuWritten`, and the draw binding
  cache's invalidation) stay in the raster closure.

Shadow: run both and compare the caches' answers for a frame (`TLG_APPLY_SPLIT=1` verify). Risk: ordering against GPU writes of earlier
operations; the raster closure keeps everything that depends on that order.

## Phase 4: prepare draws on worker threads (largest change, largest ceiling)

The command processor already resolves buffers ahead (R2). Extend that to the whole binding: workers resolve buffers, image views and
descriptors for queued draws in parallel; the raster thread records in order and only re-checks epochs. Needs first:
- a thread-safety audit of `BufferCache` and `TextureCache` (today single-threaded on the raster thread except documented paths),
- per-thread descriptor staging,
- a rule for what a worker may resolve before the operations queued ahead of it have run (writes queued earlier make an answer stale:
  the epoch check on the raster thread catches it, the worker's answer is then redone there).

Do this only if phases 1 to 3 do not reach 16.6 ms.

## Not in this plan

- Recording on several threads (secondary command buffers): only worth it after phase 4, the recording itself is 2.9 ms.
- Fewer draws (game-side batching): the engine's draw count is the engine's.

## How each phase is measured

1. Default spot, same camera, frames in flight as today.
2. Shift+F9 baseline before the change, then with the switch on; read the raster thread's busy time and the probes the phase targets.
3. A cutscene and a loading screen with the switch on (verify modes must report no differences).
4. The owner plays; then the switch becomes the default and the old path is removed.

## Result 2026-11-07: phase 2 (the draw binding cache) would not hit; do phase 1 first

Shadow census (`TLG_BIND_CENSUS=1`, `Rasterizer::BindCensus`, logged as "bind census" every 20000 binds; the Nth use of a pipeline in a frame is taken as the same draw as the Nth use in the previous frame; the descriptors it got are hashed and compared with the last time). Headless default scene, depth 1, 3.13 M binds:

* **97% of the binds (3.04 M) include a copy in the stream ring, and their descriptors differ from the previous frame's in every case (3,038,688 different, 0 same).** The ring offset moves with every command buffer by design, so a cache of "what the bind produced" cannot be reused; the work it would skip (the copy and the offset) is the work that has to be done.
* Only 2.4% of the binds (74 k) are plain bindings with the same descriptors as last time, 0.4% differ, 0.4% have no buffers.
* So the 12,800 lookups per frame are not repeats of an answer; they are small per-draw constants (2,000 stream copies per frame, 0.76 MB) and arena synchronizations (2,400 per frame, 10 MB). What can still be cut is the cost per request (done in part: sorted buffer index, no walk of the memo at each command buffer, no empty subtractions) and the bookkeeping around the apply (phase 1: about 7 ms per frame in `cpuwrite.*`, 21% of the busy time).
* Profile of the raster thread, 2026-11-07 (headless, real depth-1 setup, 33 ms busy per frame): `buffer.obtain` tree about 12 ms, `cpuwrite.*` about 7 ms, `bind_textures` 3.8 to 4.6 ms. Single captures vary by up to 3x per probe; compare only A/B in the same session.

## Result 2026-11-07: phase 1 step 1, the re-apply is gated by a hash (`TLG_REAPPLY`, default on, =0 old behaviour)

Audit (`TLG_REAPPLY_AUDIT=1`, headless default scene, depth 1): of 4.35 M ranges that came back for their second application, **99.90% had the same memory as at the first application**; 0.02% changed and the newer batch announced the same bytes again; **0.078% (3,401 ranges, 20 MB) changed with nobody announcing them: those are what the second application exists for (an announcer that fires before its write), so it cannot simply be dropped.** Step taken instead: the memory of every range is hashed when applied first and again when it comes back; only a changed range (or one that cannot be hashed, over 1 MB, or a batch whose lists do not line up) is applied the second time. Result in the same run: 4,346,213 second applications skipped, 7,005 applied. Raster thread busy time per frame 33.15 to 26.54 ms (one capture each, headless, 26.0 to 30.3 fps); `cpuwrite.*` calls per frame 3,100 to 1,548. Counters: `/stats native_backend.b2_applied.reapply_gate`. Not play-tested yet.

## Result 2026-11-07: state of the raster thread after the first optimization round (branch `tlg-optimize`)

Headless, real depth-1 setup (`Play TLG writers depth 1.cmd` switches), default scene. Raster thread busy time per frame: **33.15 ms at the start, 24.5 to 25.4 ms now** (frame 31 to 32 ms, 31 to 33 fps; the game's own frame is not the limit). Changes, newest last:

1. Buffer index as a sorted array (binary search) instead of `std::map`; empty `Subtract` calls skipped.
2. Bind memo: per-entry command buffer tick instead of a walk of the whole memo at each command buffer.
3. B-tree maps for the bind memo and the live resource table.
4. Re-apply of announced writes gated by a hash (4.35 M of 4.35 M second applications skipped, 7,005 applied). `TLG_REAPPLY=0` turns it off.
5. Texture set memo on by default (`TLG_TEXTURE_SET_MEMO=0` off).
6. The census walk in `NoteCpuWrite` runs only when the census is on; `PlacedAt` keeps its last answer per thread (valid until the placed table changes).
7. `MarkBuffersChanged`: the unattributed set is updated as set algebra (parts leave, gaps join) without adding the whole range and removing it again (1.17 to 0.96 us per call).

Dead end: a draw binding cache (97% of binds copy into the stream ring, the descriptors differ from the previous frame in every case).

What is left is flat: 12.8 k buffer lookups per frame (`buffer.obtain` 3.0 ms own, `bind_buffers` 2.7, `part.stream` 1.9 for 5.8 k small copies, `mark_buffers` 1.5, `texture.slow_path` 1.6, `sync.upload` 1.3, the rest each under 1 ms). Sampling profile: the boost interval sets (`gpu_modified_ranges`, `object_dirty`, `b9_unattributed`, `all_object_dirty`) are about 6% inclusive, xxhash 2.6%, memcpy 2.7%. A further large cut needs a structural change (fewer lookups per draw, or the apply of announced writes off the raster thread), not more micro-steps. Single captures vary by up to 3x per probe: compare only A/B in the same session.
