#!/usr/bin/env python3
"""v3k_profile.py <elf> [profile dir] [top]

Folds Vita3K's per-block instruction counts (VITA3K_PROFILE=1, one <thread id>.txt per guest
thread under ~/.local/share/Vita3K/profile/, lines "pc insts hits") onto the functions of the
unstripped Vita ELF. The module loads at its link address under the local Vita3K, so a block's
pc is the ELF's address. A "block" is where a profiled run resumed; runs end every 32
instructions, so a run's count lands on the block it started in. Counts instructions retired, not time: memcpy and every sce call are
host-side there and cost nothing, so this ranks the guest's own code.
"""
import bisect
import glob
import os
import subprocess
import sys

elf = sys.argv[1]
prof_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.local/share/Vita3K/profile")
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40

# Function symbols, sorted by address. nm -n on an ARM ELF prints thumb functions at their
# even address; block pcs are even too.
nm = subprocess.run(["arm-vita-eabi-nm", "-n", "--defined-only", "-C", elf],
                    capture_output=True, text=True, check=True).stdout
addrs, names = [], []
for line in nm.splitlines():
    parts = line.split(" ", 2)
    if len(parts) < 3 or parts[1] not in ("t", "T", "w", "W"):
        continue
    a = int(parts[0], 16)
    if a == 0 or parts[2].startswith("$"):
        continue
    addrs.append(a)
    names.append(parts[2])


def symbol(pc):
    i = bisect.bisect_right(addrs, pc) - 1
    return names[i] if i >= 0 else "?"


threads = {}
for path in sorted(glob.glob(os.path.join(prof_dir, "*.txt"))):
    tid = os.path.basename(path)[:-4]
    per_fn = {}
    total = 0
    with open(path) as f:
        for line in f:
            pc, insts, hits = line.split()
            n = int(insts) # instructions retired in this block over all its runs
            total += n
            fn = symbol(int(pc, 16) & ~1)
            per_fn[fn] = per_fn.get(fn, 0) + n
    threads[tid] = (total, per_fn)

grand = sum(t for t, _ in threads.values())
print("threads by instructions retired (total %.1f G):" % (grand / 1e9))
for tid, (total, _) in sorted(threads.items(), key=lambda kv: -kv[1][0]):
    print("  thread %-6s %8.2f G  %5.1f%%" % (tid, total / 1e9, 100.0 * total / max(grand, 1)))

for tid, (total, per_fn) in sorted(threads.items(), key=lambda kv: -kv[1][0]):
    print("\nthread %s: top %d functions of %.2f G instructions" % (tid, top, total / 1e9))
    for fn, n in sorted(per_fn.items(), key=lambda kv: -kv[1])[:top]:
        print("  %6.2f%%  %9.1f M  %s" % (100.0 * n / max(total, 1), n / 1e6, fn[:110]))
