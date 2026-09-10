// Allocation tracer for the PS Vita build.
//
// It watches the allocator that can actually run out. "Cannot allocate physical page ...
// PhyMemPart->name [ScePhyMemPartGame]" comes from the kernel refusing a *memblock*, so the
// interesting calls are sceKernelAllocMemBlock and the thread stacks the kernel carves out of
// the same partition - not newlib's malloc, which only ever hands out pieces of the one heap
// block crt0 already took. Wrapping malloc could never see that failure; it is wrapped too, but
// only to account for what is used inside that heap.
//
// NOTHING IS PRINTED FROM THE ALLOCATION PATH. The first thing that runs through here is crt0's
// own heap block, before the C runtime is up, and printing there hangs the process. The wrappers
// only fill in a table; VitaAllocTraceDump() prints it from an ordinary thread, and the frontend
// calls that as soon as there is a console to print to.
//
// The table lives in .bss, so a run that dies before any dump still leaves it readable in a crash
// dump: g_records, g_record_count, and the four totals.
//
// addr2line the return addresses against citra_vita to name the call sites.
//
// Enabled by -DVITA_ALLOC_TRACE at configure time; when off, this file is not compiled and the
// --wrap flags are not passed, so a normal build carries none of it.

#include <stdint.h>
#include <stdlib.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>

extern void* __real_malloc(size_t size);
extern void* __real_calloc(size_t n, size_t size);
extern void* __real_realloc(void* p, size_t size);
extern void* __real_aligned_alloc(size_t align, size_t size);
extern void* __real_memalign(size_t align, size_t size);
extern void __real_free(void* p);
extern size_t malloc_usable_size(void* p);

extern SceUID __real_sceKernelAllocMemBlock(const char* name, SceKernelMemBlockType type,
                                            SceSize size, SceKernelAllocMemBlockOpt* opt);
extern int __real_sceKernelFreeMemBlock(SceUID uid);
extern SceUID __real_sceKernelCreateThread(const char* name, SceKernelThreadEntry entry,
                                           int initPriority, SceSize stackSize, SceUInt attr,
                                           int cpuAffinityMask,
                                           const SceKernelThreadOptParam* option);

// Record heap allocations at or above this many KiB. Kernel memblocks and threads are always
// recorded whatever their size: there are few of them and each one is a bite out of the
// partition. Both knobs are set from CMake (VITA_ALLOC_TRACE_THRESHOLD_KIB, VITA_ALLOC_TRACE_RECORDS)
// so the trace can be widened without editing this file.
#ifndef ALLOC_TRACE_THRESHOLD_KIB
#define ALLOC_TRACE_THRESHOLD_KIB 64
#endif
#ifndef ALLOC_TRACE_RECORDS
#define ALLOC_TRACE_RECORDS 4096
#endif
#define ALLOC_TRACE_THRESHOLD ((size_t)ALLOC_TRACE_THRESHOLD_KIB * 1024u)
// Live memblocks tracked for the free path. More than the console will ever have at once.
#define ALLOC_TRACE_BLOCKS 512
#define ALLOC_TRACE_NAME 20

enum AllocKind { ALLOC_MEMBLOCK = 0, ALLOC_THREAD = 1, ALLOC_HEAP = 2 };

struct AllocRecord {
    void* ra;                     // caller (addr2line this against citra_vita)
    uint32_t bytes;               // what was asked for
    int32_t result;               // uid, or the error, or 0/1 for the heap
    uint32_t kernel_mib;          // partition total when the request was made
    uint32_t heap_mib;            // heap total when the request was made
    uint8_t kind;
    uint8_t failed;
    char name[ALLOC_TRACE_NAME];  // copied, never a borrowed pointer
};

// Everything below is plain .bss, readable from a crash dump.
struct AllocRecord g_records[ALLOC_TRACE_RECORDS];
volatile unsigned int g_record_count;  // total seen; index is count % ALLOC_TRACE_RECORDS
volatile long g_heap_live;             // bytes handed out inside the newlib heap
volatile long g_heap_peak;
volatile long g_kernel_live;           // bytes of kernel memblocks and thread stacks
volatile long g_kernel_peak;

static unsigned int g_dumped; // how many records have already been printed
// Set by the first dump, which proves printing works from an ordinary thread. After that a
// refusal is printed where it happens: it is rare, it is the thing being hunted, and waiting for
// the next dump is no use when the failure is what stops the run reaching one.
static volatile int g_armed;
// sceClibPrintf can itself allocate, and a tracer that traces itself never returns.
static __thread int g_in_trace;

/// What each live memblock cost, so a free can be subtracted. Racy by design: a diagnostic must
/// not take a lock on the allocation path.
static struct {
    SceUID uid;
    SceSize size;
} g_blocks[ALLOC_TRACE_BLOCKS];

static void remember_block(SceUID uid, SceSize size) {
    for (unsigned int i = 0; i < ALLOC_TRACE_BLOCKS; i++) {
        if (g_blocks[i].uid == 0) {
            g_blocks[i].uid = uid;
            g_blocks[i].size = size;
            return;
        }
    }
}

static SceSize forget_block(SceUID uid) {
    for (unsigned int i = 0; i < ALLOC_TRACE_BLOCKS; i++) {
        if (g_blocks[i].uid == uid) {
            const SceSize size = g_blocks[i].size;
            g_blocks[i].uid = 0;
            return size;
        }
    }
    return 0;
}

/// Fills in one record. No allocation, no locks, no output: safe from crt0 and from inside the
/// allocator itself. Returns the slot, so the result can be patched in once it is known.
static struct AllocRecord* record(uint8_t kind, const char* name, size_t size, void* ra) {
    const unsigned int i =
        __atomic_fetch_add(&g_record_count, 1, __ATOMIC_RELAXED) % ALLOC_TRACE_RECORDS;
    struct AllocRecord* r = &g_records[i];
    r->ra = ra;
    r->bytes = (uint32_t)size;
    r->result = 0;
    r->kernel_mib = (uint32_t)(g_kernel_live / (1024 * 1024));
    r->heap_mib = (uint32_t)(g_heap_live / (1024 * 1024));
    r->kind = kind;
    r->failed = 1; // cleared when the allocator comes back with something
    unsigned int n = 0;
    if (name != NULL) {
        for (; n < ALLOC_TRACE_NAME - 1 && name[n] != '\0'; n++) {
            r->name[n] = name[n];
        }
    }
    r->name[n] = '\0';
    return r;
}

// ---------------------------------------------------------------- the kernel's own allocator

SceUID __wrap_sceKernelAllocMemBlock(const char* name, SceKernelMemBlockType type, SceSize size,
                                     SceKernelAllocMemBlockOpt* opt) {
    struct AllocRecord* r = record(ALLOC_MEMBLOCK, name, size, __builtin_return_address(0));
    const SceUID uid = __real_sceKernelAllocMemBlock(name, type, size, opt);
    r->result = uid;
    if (uid < 0) {
        if (g_armed && !g_in_trace) {
            g_in_trace = 1;
            sceClibPrintf("[alloc] memblock REFUSED %u B \"%s\": 0x%08X  kernel %ld MiB  ra %p\n",
                          (unsigned int)size, r->name, (unsigned int)uid,
                          g_kernel_live / (1024 * 1024), r->ra);
            g_in_trace = 0;
        }
        return uid;
    }
    r->failed = 0;
    const long live = __atomic_add_fetch(&g_kernel_live, (long)size, __ATOMIC_RELAXED);
    if (live > g_kernel_peak) {
        g_kernel_peak = live;
    }
    remember_block(uid, size);
    return uid;
}

int __wrap_sceKernelFreeMemBlock(SceUID uid) {
    const int ret = __real_sceKernelFreeMemBlock(uid);
    if (ret >= 0) {
        __atomic_sub_fetch(&g_kernel_live, (long)forget_block(uid), __ATOMIC_RELAXED);
    }
    return ret;
}

// A thread's stack comes out of the same partition, and nothing in the process sees it as an
// allocation - which is how a large stack goes unnoticed until pte reports PTE_OS_NO_RESOURCES.
SceUID __wrap_sceKernelCreateThread(const char* name, SceKernelThreadEntry entry, int initPriority,
                                    SceSize stackSize, SceUInt attr, int cpuAffinityMask,
                                    const SceKernelThreadOptParam* option) {
    struct AllocRecord* r = record(ALLOC_THREAD, name, stackSize, __builtin_return_address(0));
    const SceUID uid = __real_sceKernelCreateThread(name, entry, initPriority, stackSize, attr,
                                                    cpuAffinityMask, option);
    r->result = uid;
    if (uid < 0) {
        if (g_armed && !g_in_trace) {
            g_in_trace = 1;
            sceClibPrintf("[alloc] thread REFUSED %u B \"%s\": 0x%08X  kernel %ld MiB  ra %p\n",
                          (unsigned int)stackSize, r->name, (unsigned int)uid,
                          g_kernel_live / (1024 * 1024), r->ra);
            g_in_trace = 0;
        }
        return uid;
    }
    r->failed = 0;
    const long live = __atomic_add_fetch(&g_kernel_live, (long)stackSize, __ATOMIC_RELAXED);
    if (live > g_kernel_peak) {
        g_kernel_peak = live;
    }
    return uid;
}

// ---------------------------------------------------------------- the stack watch
//
// Two crashes now have ended the same way: a thread's whole stack consumed, and whichever
// function happened to need a few hundred bytes at that moment blamed for it. {fmt} was the leaf
// both times because it takes a decent frame, not because it recurses - the chain above it was
// six frames deep with eight megabytes already gone underneath.
//
// So: watch the stack from somewhere that is called constantly and cheaply. malloc is already
// wrapped, so it costs one register read and a compare. When what is left drops below the
// threshold, scan the stack for words that look like code addresses and print the ones that
// repeat: a function that appears hundreds of times is the one recursing, and that is the
// question worth answering while there is still stack to answer it in.

#define STACK_WATCH_MARGIN (96u * 1024u)   // report while this much is still left
#define STACK_WATCH_SAMPLES 4096           // how far up the stack to look
#define STACK_WATCH_TOP 24                 // how many distinct addresses to print

// The text range, for deciding whether a stack word could be a return address. The linker script
// puts the image at 0x81000000; _etext is the end of it.
extern char __executable_start[];
extern char __etext[];

// Plain globals, deliberately. Anything __thread here goes through emulated TLS, and
// __emutls_get_address allocates a thread's block on its first access - a malloc from inside
// the malloc wrapper, under the emutls lock, which is a deadlock the moment the watch arms.
// So the watch covers one thread, registered explicitly, instead of whichever thread happens
// to be allocating: the emulation thread is where every stack overflow so far has happened.
static uintptr_t g_watch_low;  // low end of the watched thread's stack; zero until registered
static uintptr_t g_watch_high; // one past its high end
static volatile int g_watch_reported;

/// Called by the thread that wants its stack watched (the emulation thread, from EmulatorMain).
void VitaAllocTraceWatchThread(void) {
    SceKernelThreadInfo info;
    info.size = sizeof(info);
    if (sceKernelGetThreadInfo(sceKernelGetThreadId(), &info) < 0 || info.stack == NULL) {
        sceClibPrintf("[stack] cannot read this thread's stack bounds; the watch is off\n");
        return;
    }
    g_watch_high = (uintptr_t)info.stack + info.stackSize;
    g_watch_low = (uintptr_t)info.stack; // written last: this is what arms the checks
    sceClibPrintf("[stack] watching thread %08X, stack %08X + %u B\n",
                  (unsigned int)sceKernelGetThreadId(), (unsigned int)g_watch_low,
                  (unsigned int)info.stackSize);
}

static void stack_watch(void) {
    // Two compares against plain globals on the usual path: not armed, not the watched thread's
    // stack, or not yet near the edge - out. Reading sp identifies the thread by where its stack
    // lives, so this costs no syscall and touches no TLS.
    const uintptr_t low = g_watch_low;
    if (low == 0 || g_watch_reported) {
        return;
    }
    uintptr_t sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    if (sp - low >= STACK_WATCH_MARGIN) { // also rejects every other thread: their sp wraps huge
        return;
    }
    if (sp >= g_watch_high) {
        return;
    }

    // Closed before the first print, so nothing this does can come back in - not even if a
    // print allocates. One report is the useful one anyway; after it the crash may proceed.
    g_watch_reported = 1;
    const uintptr_t text_lo = (uintptr_t)__executable_start;
    const uintptr_t text_hi = (uintptr_t)__etext;
    sceClibPrintf("[stack] watched thread within %u B of its limit (sp %08X, low %08X)\n",
                  (unsigned int)(sp - low), (unsigned int)sp, (unsigned int)low);

    // Count how often each code address appears in the frames just above the stack pointer. The
    // repeating one is the recursion; a linear chain shows every address once.
    struct { uintptr_t addr; unsigned int count; } seen[STACK_WATCH_TOP];
    unsigned int seen_n = 0;
    const uintptr_t* word = (const uintptr_t*)sp;
    for (unsigned int i = 0; i < STACK_WATCH_SAMPLES; i++, word++) {
        const uintptr_t v = word[0];
        if (v < text_lo || v >= text_hi) {
            continue;
        }
        unsigned int j = 0;
        for (; j < seen_n; j++) {
            if (seen[j].addr == v) {
                seen[j].count++;
                break;
            }
        }
        if (j == seen_n && seen_n < STACK_WATCH_TOP) {
            seen[seen_n].addr = v;
            seen[seen_n].count = 1;
            seen_n++;
        }
    }
    for (unsigned int i = 0; i < seen_n; i++) {
        sceClibPrintf("[stack]   %08X x%u\n", (unsigned int)seen[i].addr, seen[i].count);
    }
    sceClibPrintf("[stack] addr2line these against citra_vita; the one repeating is the recursion\n");
}

// ---------------------------------------------------------------- inside the newlib heap

static void heap_refused(size_t size, void* ra) {
    if (!g_armed || g_in_trace) {
        return;
    }
    g_in_trace = 1;
    sceClibPrintf("[alloc] heap REFUSED %u B  heap %ld MiB  kernel %ld MiB  ra %p\n",
                  (unsigned int)size, g_heap_live / (1024 * 1024),
                  g_kernel_live / (1024 * 1024), ra);
    g_in_trace = 0;
}

#define WRAP_BODY(call, req_size)                                                                  \
    stack_watch();                                                                                 \
    void* const ra = __builtin_return_address(0);                                                  \
    const size_t sz = (req_size);                                                                  \
    struct AllocRecord* r = sz >= ALLOC_TRACE_THRESHOLD ? record(ALLOC_HEAP, NULL, sz, ra) : NULL; \
    void* p = (call);                                                                              \
    if (!p && sz != 0) {                                                                           \
        heap_refused(sz, ra);                                                                      \
    }                                                                                              \
    if (p) {                                                                                       \
        if (r) {                                                                                   \
            r->failed = 0;                                                                         \
            r->result = 1;                                                                         \
        }                                                                                          \
        const long live = __atomic_add_fetch(&g_heap_live, (long)sz, __ATOMIC_RELAXED);            \
        if (live > g_heap_peak) {                                                                  \
            g_heap_peak = live;                                                                    \
        }                                                                                          \
    }                                                                                              \
    return p;

void* __wrap_malloc(size_t size) {
    WRAP_BODY(__real_malloc(size), size)
}

void* __wrap_calloc(size_t n, size_t size) {
    WRAP_BODY(__real_calloc(n, size), n * size)
}

void* __wrap_aligned_alloc(size_t align, size_t size) {
    WRAP_BODY(__real_aligned_alloc(align, size), size)
}

void* __wrap_memalign(size_t align, size_t size) {
    WRAP_BODY(__real_memalign(align, size), size)
}

void* __wrap_realloc(void* old, size_t size) {
    void* const ra = __builtin_return_address(0);
    const size_t old_size = old ? malloc_usable_size(old) : 0;
    struct AllocRecord* r = size >= ALLOC_TRACE_THRESHOLD ? record(ALLOC_HEAP, NULL, size, ra) : NULL;
    void* p = __real_realloc(old, size);
    if (p) {
        if (r) {
            r->failed = 0;
            r->result = 1;
        }
        __atomic_sub_fetch(&g_heap_live, (long)old_size, __ATOMIC_RELAXED);
        const long live = __atomic_add_fetch(&g_heap_live, (long)size, __ATOMIC_RELAXED);
        if (live > g_heap_peak) {
            g_heap_peak = live;
        }
    }
    return p;
}

void __wrap_free(void* p) {
    if (p) {
        __atomic_sub_fetch(&g_heap_live, (long)malloc_usable_size(p), __ATOMIC_RELAXED);
    }
    __real_free(p);
}

/// Prints the records made since the last call, then the totals. Call from an ordinary thread -
/// never from the allocation path, which is the whole reason the wrappers only record.
void VitaAllocTraceDump(void) {
    static const char* const kind_name[] = {"memblock", "thread  ", "heap    "};
    const unsigned int seen = g_record_count;
    unsigned int first = g_dumped;
    if (seen - first > ALLOC_TRACE_RECORDS) {
        first = seen - ALLOC_TRACE_RECORDS; // the ring wrapped; show what survives
    }
    for (unsigned int i = first; i < seen; i++) {
        const struct AllocRecord* r = &g_records[i % ALLOC_TRACE_RECORDS];
        if (r->failed) {
            sceClibPrintf("[alloc] %s %9u B  kernel %4u MiB  heap %4u MiB  ra %p  %s  "
                          "REFUSED 0x%08X\n",
                          kind_name[r->kind < 3 ? r->kind : 2], r->bytes, r->kernel_mib,
                          r->heap_mib, r->ra, r->name, (unsigned int)r->result);
        } else {
            sceClibPrintf("[alloc] %s %9u B  kernel %4u MiB  heap %4u MiB  ra %p  %s\n",
                          kind_name[r->kind < 3 ? r->kind : 2], r->bytes, r->kernel_mib,
                          r->heap_mib, r->ra, r->name);
        }
    }
    g_dumped = seen;
    g_armed = 1;
    // What the kernel says is left, which is the only figure that decides whether the next
    // request succeeds. The totals above are what this process asked for; this is the ceiling
    // it is asking against.
    SceKernelFreeMemorySizeInfo info;
    info.size = sizeof(info);
    const int ret = sceKernelGetFreeMemorySize(&info);
    sceClibPrintf("[alloc] totals: kernel %ld MiB (peak %ld)  heap %ld MiB (peak %ld)  %u requests\n",
                  g_kernel_live / (1024 * 1024), g_kernel_peak / (1024 * 1024),
                  g_heap_live / (1024 * 1024), g_heap_peak / (1024 * 1024), seen);
    if (ret >= 0) {
        sceClibPrintf("[alloc] free:   user %ld MiB  cdram %ld MiB  phycont %ld MiB\n",
                      (long)info.size_user / (1024 * 1024), (long)info.size_cdram / (1024 * 1024),
                      (long)info.size_phycont / (1024 * 1024));
    }
}
