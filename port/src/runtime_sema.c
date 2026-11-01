/* Kernel semaphores: 32-bit IDs, counted tokens and FIFO waiters.
 * Priority-ordered semaphores are approximated by FIFO order.
 *
 * Each semaphore has its own lock and is found by its ID in a table read without a lock; a blocking wait uses its thread's condition
 * variable (made once per thread). One lock for all semaphores, with a search of every semaphore per call, cost the game's threads about
 * 25 ms per frame (summed) waiting for that lock (port/gpu/native/frame-loop-findings.md). A deleted semaphore's memory is kept: a thread
 * that found it just before the delete still locks it and sees it deleted. */
#define _GNU_SOURCE
#include "runtime.h"
#include "runtime_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#define ERR(n) ((int32_t)(UINT32_C(0x80020000)|(n)))
typedef struct Waiter {
    int32_t need, result;
    int done;
    pthread_cond_t *wake; /* the waiting thread's condition variable */
    struct Waiter *next;
} Waiter;
typedef struct Sema {
    uint32_t id, attr;
    int32_t count, initial, maximum;
    int deleted;
    char name[32];
    Waiter *first;
    struct Sema *next;
    pthread_mutex_t mu; /* guards count, the waiters and deleted */
} Sema;
/* `lock` guards the list of all semaphores, next_id and the table's pages (creation and deletion only) */
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static Sema *registry;
static uint32_t next_id=1;
static size_t created, deleted, acquired, signaled, timed_out;
static void host_check(int e) {
    if (e) { fprintf(stderr,"STOP: semaphore host error %d\n",e); exit(21); }
}
/* Waits use this clock for their deadlines. winpthreads condition variables accept only
 * CLOCK_REALTIME (pthread_condattr_setclock(CLOCK_MONOTONIC) fails with EINVAL). */
#ifdef _WIN32
#define WAIT_CLOCK CLOCK_REALTIME
#else
#define WAIT_CLOCK CLOCK_MONOTONIC
#endif
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(WAIT_CLOCK,&t)) { perror("clock_gettime"); exit(1); }
    return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
static int trace_waits(void) {
    static int trace=-1;
    int t=__atomic_load_n(&trace,__ATOMIC_RELAXED);
    if (t<0) { t=getenv("TLG_TRACE_SEMA")!=NULL; __atomic_store_n(&trace,t,__ATOMIC_RELAXED); } /* deadlock diagnosis */
    return t;
}
/* IDs 1 to TABLE_PAGE*TABLE_PAGES-1 are found in pages made on demand under `lock` and never freed; a slot holds the semaphore while it
 * exists. Higher IDs are looked up in the list under `lock`. */
#define TABLE_PAGE 1024u
#define TABLE_PAGES 1024u
static Sema **table[TABLE_PAGES];
static Sema *find(uint32_t id) { /* under `lock` */
    for (Sema *s=registry;s;s=s->next) if (s->id==id) return s;
    return NULL;
}
static Sema *lookup(uint32_t id) {
    if (id<TABLE_PAGE*TABLE_PAGES) {
        Sema **page=__atomic_load_n(&table[id/TABLE_PAGE],__ATOMIC_ACQUIRE);
        return page ? __atomic_load_n(&page[id%TABLE_PAGE],__ATOMIC_ACQUIRE) : NULL;
    }
    pthread_mutex_lock(&lock);
    Sema *s=find(id);
    pthread_mutex_unlock(&lock);
    return s;
}
static void publish(uint32_t id,Sema *s) { /* under `lock` */
    if (id>=TABLE_PAGE*TABLE_PAGES) return;
    Sema **page=table[id/TABLE_PAGE];
    if (!page) {
        page=calloc(TABLE_PAGE,sizeof(*page));
        if (!page) { fputs("STOP: semaphore table: out of memory\n",stderr); exit(21); }
        __atomic_store_n(&table[id/TABLE_PAGE],page,__ATOMIC_RELEASE);
    }
    __atomic_store_n(&page[id%TABLE_PAGE],s,__ATOMIC_RELEASE);
}
static pthread_cond_t *thread_wake(void) {
    static _Thread_local pthread_cond_t wake;
    static _Thread_local int made;
    if (!made) {
        pthread_condattr_t attr;
        host_check(pthread_condattr_init(&attr));
        host_check(pthread_condattr_setclock(&attr,WAIT_CLOCK));
        host_check(pthread_cond_init(&wake,&attr));
        host_check(pthread_condattr_destroy(&attr));
        made=1;
    }
    return &wake;
}
/* Locks the semaphore; the profiler (Shift+F9, group os) sees the wait for the lock. */
static void lock_sema(Sema *s) {
    OS_PROF(p_lock,"os.sem_lock (the semaphore's lock)");
    pthread_mutex_lock(&s->mu);
    tlggpu_prof_leave(p_lock);
}
static ABI int32_t sem_create(uint32_t *out,const char *name,uint32_t attr,
                              int32_t initial,int32_t maximum,const void *options) {
    if (!out || !name || attr>2 || initial<0 || maximum<=0 || initial>maximum) return ERR(22);
    if (options) { fputs("STOP: semaphore options are not implemented\n",stderr); exit(21); }
    Sema *s=calloc(1,sizeof(*s));
    if (!s) return ERR(12);
    host_check(pthread_mutex_init(&s->mu,NULL));
    pthread_mutex_lock(&lock);
    if (next_id==UINT32_MAX) { pthread_mutex_unlock(&lock); pthread_mutex_destroy(&s->mu); free(s); return ERR(28); }
    snprintf(s->name,sizeof(s->name),"%s",name);
    s->id=next_id++; s->attr=attr; s->count=s->initial=initial; s->maximum=maximum;
    s->next=registry; registry=s; *out=s->id; __atomic_add_fetch(&created,1,__ATOMIC_RELAXED);
    publish(s->id,s);
    printf("Runtime: semaphore id=%u, attr=%u, initial=%d, max=%d\n",s->id,attr,initial,maximum);
    pthread_mutex_unlock(&lock);
    return 0;
}
static int32_t wait_count(uint32_t id,int32_t need,uint32_t *timeout,int block,uintptr_t frame) {
    /* profiler probes (Shift+F9, group os): the call and the wait for the semaphore's lock; the blocked time is a wait probe */
    static void *wait_site,*poll_site;
    if (!wait_site) { poll_site=tlggpu_prof_site("os.sem_poll",1); wait_site=tlggpu_prof_site("os.sem_wait",1); }
    void *p_call=tlggpu_prof_enter(block ? wait_site : poll_site);
    Sema *s=lookup(id);
    if (!s) { tlggpu_prof_leave(p_call); return ERR(3); }
    lock_sema(s);
    int32_t result=0;
    if (s->deleted) result=ERR(3);
    else if (need<=0 || need>s->maximum) result=ERR(22);
    else if (s->count>=need) { s->count-=need; __atomic_add_fetch(&acquired,1,__ATOMIC_RELAXED); }
    else if (!block) result=ERR(16);
    else if (timeout && !*timeout) { result=ERR(60); __atomic_add_fetch(&timed_out,1,__ATOMIC_RELAXED); }
    else {
        /* Priority-ordered semaphores (attr 2) are served FIFO: there is no
           PS4 priority scheduler, host threads run at equal priority. */
        if (trace_waits()) fprintf(stderr,"Runtime: blocking wait on semaphore %u (need %d, count %d)\n",id,need,s->count);
        Waiter w={.need=need};
        w.wake=thread_wake();
        Waiter **tail=&s->first;
        while (*tail) tail=&(*tail)->next;
        *tail=&w;
        uint64_t deadline=timeout ? now_ns()+(uint64_t)*timeout*1000 : 0;
        struct timespec end={.tv_sec=(time_t)(deadline/1000000000),.tv_nsec=(long)(deadline%1000000000)};
        runtime_wait_begin(timeout ? "semaphore (timed)" : "semaphore",(const void *)(uintptr_t)id,s->name,NULL,frame);
        OS_PROF(p_blocked,"wait.os_sem_blocked (game thread: a kernel semaphore has no count)");
        while (!w.done) {
            int e=timeout ? pthread_cond_timedwait(w.wake,&s->mu,&end) : pthread_cond_wait(w.wake,&s->mu);
            if (e==ETIMEDOUT && !w.done) {
                Waiter **p=&s->first;
                while (*p!=&w) p=&(*p)->next;
                *p=w.next; w.result=ERR(60); w.done=1; __atomic_add_fetch(&timed_out,1,__ATOMIC_RELAXED);
            } else if (e!=ETIMEDOUT) host_check(e);
        }
        tlggpu_prof_leave(p_blocked);
        runtime_wait_end();
        result=w.result;
        if (!result) __atomic_add_fetch(&acquired,1,__ATOMIC_RELAXED);
        if (timeout) {
            uint64_t now=now_ns();
            *timeout=result ? 0 : (uint32_t)(now>=deadline ? 0 : (deadline-now)/1000);
        }
    }
    pthread_mutex_unlock(&s->mu);
    tlggpu_prof_leave(p_call);
    return result;
}
static ABI int32_t sem_wait(uint32_t id,int32_t need,uint32_t *timeout) { return wait_count(id,need,timeout,1,RUNTIME_GUEST_FRAME()); }
static ABI int32_t sem_poll(uint32_t id,int32_t need) { return wait_count(id,need,NULL,0,0); }
extern void tlggpu_timeline_signal(int kind,const void *object);
static ABI int32_t sem_signal(uint32_t id,int32_t count) {
    OS_PROF(p_call,"os.sem_signal");
    tlggpu_timeline_signal(7,(const void *)(uintptr_t)id);
    Sema *s=lookup(id);
    if (!s) { tlggpu_prof_leave(p_call); return ERR(3); }
    lock_sema(s);
    int32_t result=0;
    if (s->deleted) result=ERR(3);
    else if (count<=0 || count>s->maximum-s->count) result=ERR(22);
    else {
        s->count+=count; __atomic_add_fetch(&signaled,1,__ATOMIC_RELAXED);
        Waiter **p=&s->first;
        while (*p) {
            Waiter *w=*p;
            if (w->need>s->count) { p=&w->next; continue; }
            s->count-=w->need; *p=w->next; w->done=1;
            host_check(pthread_cond_signal(w->wake));
        }
    }
    pthread_mutex_unlock(&s->mu);
    tlggpu_prof_leave(p_call);
    return result;
}
static int32_t wake_all(Sema *s,int32_t result) { /* under s->mu */
    int32_t count=0;
    while (s->first) {
        Waiter *w=s->first; s->first=w->next;
        w->result=result; w->done=1; ++count;
        host_check(pthread_cond_signal(w->wake));
    }
    return count;
}
static ABI int32_t sem_cancel(uint32_t id,int32_t count,int32_t *waiters) {
    Sema *s=lookup(id);
    if (!s) return ERR(3);
    pthread_mutex_lock(&s->mu);
    int32_t result=0;
    if (s->deleted) result=ERR(3);
    else if (count>s->maximum) result=ERR(22);
    else {
        int32_t n=wake_all(s,ERR(85));
        if (waiters) *waiters=n;
        s->count=count<0 ? s->initial : count;
    }
    pthread_mutex_unlock(&s->mu);
    return result;
}
static ABI int32_t sem_delete(uint32_t id) {
    pthread_mutex_lock(&lock);
    Sema **p=&registry;
    while (*p && (*p)->id!=id) p=&(*p)->next;
    if (!*p) { pthread_mutex_unlock(&lock); return ERR(3); }
    /* out of the list and the table first; a thread that found it before sees `deleted` under its lock; the memory is kept */
    Sema *s=*p; *p=s->next;
    publish(id,NULL);
    __atomic_add_fetch(&deleted,1,__ATOMIC_RELAXED);
    pthread_mutex_unlock(&lock);
    pthread_mutex_lock(&s->mu);
    s->deleted=1;
    wake_all(s,ERR(13));
    pthread_mutex_unlock(&s->mu);
    return 0;
}
/* Read-only test/diagnostic snapshot, not exposed to guest code. */
unsigned runtime_sema_waiters(uint32_t id) {
    Sema *s=lookup(id); unsigned n=0;
    if (!s) return 0;
    pthread_mutex_lock(&s->mu);
    if (!s->deleted) for (Waiter *w=s->first;w;w=w->next) ++n;
    pthread_mutex_unlock(&s->mu);
    return n;
}
uintptr_t runtime_sema_resolve(const char *name) {
    if (!strcmp(name,"188x57JYp0g#libkernel")) return (uintptr_t)sem_create;
    if (!strcmp(name,"Zxa0VhQVTsk#libkernel")) return (uintptr_t)sem_wait;
    if (!strcmp(name,"4czppHBiriw#libkernel")) return (uintptr_t)sem_signal;
    if (!strcmp(name,"12wOHk8ywb0#libkernel")) return (uintptr_t)sem_poll;
    if (!strcmp(name,"4DM06U2BNEY#libkernel")) return (uintptr_t)sem_cancel;
    if (!strcmp(name,"R1Jvn8bSCW8#libkernel")) return (uintptr_t)sem_delete;
    return 0;
}
void runtime_sema_report(void) {
    printf("Runtime: semaphores created=%zu, deleted=%zu, acquired=%zu, signals=%zu, timeouts=%zu\n",
           __atomic_load_n(&created,__ATOMIC_RELAXED),__atomic_load_n(&deleted,__ATOMIC_RELAXED),__atomic_load_n(&acquired,__ATOMIC_RELAXED),
           __atomic_load_n(&signaled,__ATOMIC_RELAXED),__atomic_load_n(&timed_out,__ATOMIC_RELAXED));
}
