# Blind spots in the runtime (the OS layer the game calls): plan, 2026-11-01

## Why

The kernel semaphores were a blind spot: one lock for all semaphores in `port/src/runtime_sema.c` cost the game's threads about 25 ms
per frame (summed), and nothing pointed at it until probes were added. Fixing it took the game's own frame from 13.6 to 9.6 ms with the
draws left out (`frame-loop-findings.md`). Upstream shadPS4 never had that problem; it came from this port's own runtime.

This plan looks for the next ones in the same layer: the code in `port/src` (and the few shadPS4 kernel files the port uses) that the
game's threads call every frame. The patterns to look for are the ones the semaphores had: one lock or one shared counter for everything,
searching instead of looking up, setup work on every call, and waiting by spinning where the console sleeps.

Method, as for the semaphores: measure first (probes, Shift+F9 with `TLG_PROF_THREADS=all`), fix only what the numbers show, each fix
behind a switch and off by default, tests for correctness, the owner plays, then the default.

## What the code review and the eboot found (evidence)

The eboot's imports were mapped to their PLT stubs and every call site counted (`port/tools/eboot/import_sites.py`, on the decrypted eboot,
its memory image at 0x1000000 and the Ghidra function list; see `port/tools/ghidra/README.md`).

| # | candidate | evidence | how often (static) | confidence |
|---|---|---|---|---|
| 1 | short HR timer sleeps are busy spins | the engine's "sleep N" is an inlined `sceKernelAddHRTimerEvent(eq, 0, N)` + `sceKernelWaitEqueue(eq, ..., timeout)` (fallback `sceKernelUsleep`, cleanup `sceKernelDeleteHRTimerEvent`), e.g. `Engine_WorkerThreadMain` 0x1e4c4b5; shadPS4's event queue spins with `yield` for any HR timer under 1.2 ms (`equeue.cpp`, `HrTimerSpinlockThresholdNs`, `WaitForSmallTimer`) | 1,647 call sites in 1,517 functions; 1,280 to 1,570 of them sleep **100 us** (the engine lock's backoff: a failed compare-and-swap, then sleep 100 us, retry; e.g. `FlipScreenTask`'s constructor 0x1d0b200) | high that it spins; the rate in play is unknown |
| 2 | one shared counter on every `memcpy` / `memset` / `memmove` / `memcmp` | `runtime.c` `guest_mem*`: `atomic_fetch_add(&memory_calls)` on each call, a report-only number; every game thread writes the same cache line | 508 / 546 / 372 / 2 call sites; `RenderDev_RenderScene`, `RenderDev_InitComputeSystems`, `ComputeInterface_MapReadThreeBuffers`, jobs | high that it costs per call; the call rate is unknown (`Runtime: memory operations=` in the log at exit gives the total) |
| 3 | the reader-writer locks repeat the old semaphore design | `runtime_rwlock.c`: every lock takes a global `registry_lock` twice and searches the list of all rwlocks (`find_locked`), walks the holder list, and allocates / frees a `Holder` on most operations (`calloc`, `prune_locked`); unlock takes the registry lock and searches too | the 7 rwlock imports are called from one wrapper each (0x1bd7e50 read, 0x1bd7e70 write, 0x1bd7e60 / 0x1bd7e80 unlock), virtual methods of two classes (vtables 0x26e3ce8 and 0x26e3de8, code at 0x1bd7000 to 0x1bde000, next to the particle system) | the design is certain; use in play unknown (`Runtime: rwlocks created=..., reads=..., writes=...` at exit) |
| 4 | the audio mixer holds one lock for the whole mix | `runtime_ngs2.c` (the default mixer; `TLG_NGS2=native` uses the firmware module instead): `ngs2_render` holds the global recursive `lock` while it mixes every rack and voice; `ngs2_voice_control` and `ngs2_voice_state` take the same lock, so a game thread changing a voice waits for the whole mix (user effects already run unlocked) | `sceNgs2VoiceControl`: 111 call sites in 15 functions (0x11a9690, 0x11adc10, ... the game's sound code); `Task:SoundMgr` runs every frame | medium: depends on the mix time and on which thread calls |
| 5 | shared plain counters in the mutex and condition paths | `runtime_mutex.c`: `++locks`, `++unlocks`, `++waits`, `++wakeups` (not atomic, report-only) from every thread | about 18,000 mutex locks per frame (`os.mutex_lock` probe) | low (same pattern as 2, smaller) |

Checked and fine (no change planned):
- thread-local keys (`scePthreadGetspecific`: a thread-local array, no lock), module TLS (`__tls_get_addr`: per-thread blocks),
  thread identity (`runtime_thread_self`: thread-local);
- clocks (`clock_gettime`, `sceKernelReadTsc`) and sleeps (`compat_sleep_ns`: a high-resolution waitable timer per thread);
- the memory map lock (`runtime_memory.c`: a reader-writer lock; readers block only while the map changes; region lookups are cached per
  thread);
- mutexes and condition variables (one host object each; measured: 0.1 ms per frame for 18,343 locks);
- event queues (one lock per queue); `malloc` / `free` (the console's own libc module).

## Phase 0: measure everything at once (probes and counters only, no behaviour change)

0.1 **A counter entry point for the C runtime:** `tlggpu_prof_count(site, value)` next to `tlggpu_prof_site/enter/leave`
(`shim/tlggpu.cpp`, `src/runtime_prof.h`), counted per thread like `TLG_PROF_COUNT`.

0.2 **Probes per candidate** (group `os`, level 1 or 2, Shift+F9 with `TLG_PROF_THREADS=all`):
- HR timers (`equeue.cpp`): `os.hrtimer_add` (count, and a counter per length class: under 200 us, under 1.2 ms, longer), `os.hrtimer_spin`
  around `WaitForSmallTimer` (time spent spinning per thread), and the caller's return address for the top callers (as the game wait sites
  are recorded today) to name the engine functions that sleep most.
- memory functions (`runtime.c`): a per-thread call counter (thread-local, summed at the report) instead of the shared atomic, plus a
  `os.memcpy_bytes` counter; the call rate per thread and frame is what decides.
- reader-writer locks (`runtime_rwlock.c`): `os.rwlock_read`, `os.rwlock_write`, `os.rwlock_unlock`, `os.rwlock_registry` (the wait for the
  global registry lock), `wait.os_rwlock_blocked`, and a counter of `Holder` allocations.
- audio (`runtime_ngs2.c`): `os.ngs2_render` (hold time of the mixer lock), `os.ngs2_lock` in `voice_control` / `voice_state` (time a caller
  waits for it) and the calling thread.
- mutex counters: none needed; candidate 5 is judged from 2's result.

0.3 **Captures:** the default spot, a heavy spot and a cutscene, in the real game; the game threads' sections of Shift+F9 show each
candidate's ms per frame per thread. Also the exit log lines `Runtime: memory operations=` and `Runtime: rwlocks created=` over a known
play time.

Exit: each candidate has a number (ms per frame on the game threads, calls per frame). Only candidates over about 0.3 ms per frame on a
thread in the game's frame chain, or over 1 ms summed, go to the next phases.

## Phase 1: candidate 1, HR timer sleeps (likely the biggest)

The engine sleeps 100 us when one of its own locks is taken (the console's scheduler gives the core to the lock's holder meanwhile). The
port spins with `yield` instead: the waiting thread keeps a core, competes with the holder (on SMT siblings it slows it down), and the CPU
use of the game's workers looks higher than their work.

Fix (`TLG_HRTIMER_SLEEP=1`, off by default):
- HR timers under the spin threshold sleep instead of spinning: the thread's high-resolution waitable timer (`compat_sleep_ns` on Windows)
  for the timer's length, a short spin only for the last few tens of microseconds when the length is over a few hundred microseconds (a
  lock backoff does not need precision; a real short timer does);
- the wait still returns the event as now (the semantics of `sceKernelWaitEqueue` and the event's data do not change).

Check: a counter of wake-up lateness (requested length against actual) for each length class; fps and the game threads' CPU use; Shift+F9
`os.hrtimer_spin` gone. Tests: a unit test of `WaitForSmallTimer` lengths (100 us, 1 ms) in both modes.

## Phase 2: candidate 2, the memory function counter (trivial)

Remove the shared counter from `guest_memset/memcpy/memmove/memcmp`, or keep it per thread (thread-local, summed in `runtime_report`).
No switch needed for a report-only number, but measure before and after (a micro benchmark: 8 threads copying small blocks through the
wrapper, shared counter against none; and the game's fps at the same spot).

## Phase 3: candidate 3, reader-writer locks (only if phase 0 shows use)

As for the semaphores (`TLG_RWLOCK_FAST=1` while tested, then the only path):
- the handle points straight at its `Rwlock`, validated by a magic word instead of the registry search; the registry lock is taken only to
  create and destroy;
- the holder bookkeeping (needed for EDEADLK / EPERM and the freeze report) moves to a small per-thread table (thread-local, no `calloc`);
- the native `pthread_rwlock_t` keeps the waiting.
Tests: a new `tests/test_rwlock.c` (read sharing, write exclusion, try and timed forms, EDEADLK on a second write, EPERM on a foreign unlock,
destroy while held), in both modes, plus a stress test like the semaphore one.

## Phase 4: candidate 4, the audio mixer lock (only if phase 0 shows waits on game threads)

`ngs2_voice_control` and `ngs2_voice_state` stop waiting for a whole mix:
- voice parameter blocks go into a per-voice pending list under a short lock and are applied at the start of the next mix (the mix runs
  every grain, so the delay is one grain, as on the console where control applies at the next render);
- the state a game thread reads is a copy the mixer publishes after each grain.
Switch `TLG_NGS2_QUEUE=1`. Check by ear (no clicks, no late starts) and with the NGS2 trace (`TLG_NGS2_TRACE`).

## Phase 5: a sweep for what this review did not think of

An import counter: with `TLG_IMPORT_COUNT=1`, `runtime_resolve` hands out a small generated thunk per import that counts the call (per import)
and jumps to the real function (no return interception, so the cost is one counter and one jump). The report lists the imports by calls per
frame; the top ones that are not yet probed are reviewed like candidates 1 to 5. Timing per import stays with the probes of phase 0.

## Order and expected value

1. Phase 0 (one build, a few captures): decides everything below.
2. Phase 1 if the HR timer spins show up (expected: they show the engine's lock contention as CPU time; the value is CPU for the game's
   workers and the raster thread on a 16 logical core CPU).
3. Phase 2 regardless (trivial), measured.
4. Phases 3 and 4 only on evidence.
5. Phase 5 once, after the others, to look for the next one.

## Notes

- The 100 us sleeps are the engine's own lock backoff (1,500 functions); how often they fire tells how contended the engine's locks are on
  this machine, which is useful on its own (it was invisible: a spin is not a wait in the stall report).
- `TLG_ABLATE` was removed in the cleanup; phase 0 runs on the real game. If a capture without the raster thread's load is needed, the
  ablation switch can come back for that run only.
- Evidence can be reproduced with `port/tools/eboot/import_sites.py <eboot.elf> <eboot.mem> <func_starts.txt> <ghidra_names.tsv> --hrtimer
  sceKernelAddHRTimerEvent memcpy memset memmove scePthreadRwlockRdlock sceNgs2VoiceControl`.
