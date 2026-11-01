/* Probes of the GPU library's profiler (Shift+F9, native/native_prof.h) in runtime code. A probe costs one load and a branch while no
 * capture records it. Usage: OS_PROF(var, "os.name"); ... tlggpu_prof_leave(var);  (probes end in the reverse order they began) */
#pragma once
void *tlggpu_prof_site(const char *name, int level);
void *tlggpu_prof_enter(void *site);
void tlggpu_prof_leave(void *token);
#define OS_PROF(var, name)                                                                                                               \
    static void *var##_site;                                                                                                            \
    if (!var##_site) var##_site = tlggpu_prof_site(name, 1);                                                                            \
    void *var = tlggpu_prof_enter(var##_site)
