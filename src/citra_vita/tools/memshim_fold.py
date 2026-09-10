#!/usr/bin/env python3
"""memshim_fold.py <binary> <dump> [top]

Folds an LD_PRELOAD interposer dump (rows "what ra tid calls bytes", from memshim.c on the
pi5) onto the functions of the binary it ran against, per thread and per interposed call.
The binary is linked -no-pie at a fixed text address, so a return address resolves directly.
"""
import collections
import subprocess
import sys

binary, dump = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25

rows = []
main_tid = None
for line in open(dump):
    if line.startswith("#"):
        main_tid = int(line.split()[-1])
        continue
    what, ra, tid, calls, nbytes = line.split()
    rows.append((what, int(ra, 16), int(tid), int(calls), int(nbytes)))

addrs = sorted({ra for _, ra, _, _, _ in rows})
out = subprocess.run(["arm-linux-gnueabihf-addr2line", "-f", "-C", "-e", binary] +
                     ["0x%x" % a for a in addrs], capture_output=True, text=True).stdout.splitlines()
name_of = {}
for i, a in enumerate(addrs):
    fn = out[2 * i] if 2 * i < len(out) else "?"
    loc = out[2 * i + 1] if 2 * i + 1 < len(out) else "?"
    name_of[a] = (fn, loc.split("/")[-1])

by_thread = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for what, ra, tid, calls, nbytes in rows:
    fn, loc = name_of[ra]
    key = (what, fn[:80], loc)
    by_thread[tid][key][0] += calls
    by_thread[tid][key][1] += nbytes

for tid, table in sorted(by_thread.items(), key=lambda kv: -sum(v[1] for v in kv[1].values())):
    label = "emulation (main)" if tid == main_tid else "tid %d" % tid
    total_bytes = sum(v[1] for v in table.values())
    total_calls = sum(v[0] for v in table.values())
    print("\n== %s: %.1f MB in %d calls" % (label, total_bytes / 1e6, total_calls))
    for (what, fn, loc), (calls, nbytes) in sorted(table.items(), key=lambda kv: -kv[1][1])[:top]:
        print("  %-10s %8.1f MB %9d calls %6.0f B/call  %s  (%s)" %
              (what, nbytes / 1e6, calls, nbytes / max(calls, 1), fn, loc))
    # Division and clock rows carry no bytes: rank those by calls.
    counted = [(k, v) for k, v in table.items() if k[0] == "udivmoddi4"]
    if counted:
        print("  -- udivmoddi4 by calls:")
        for (what, fn, loc), (calls, nbytes) in sorted(counted, key=lambda kv: -kv[1][0])[:10]:
            print("  %-10s %9d calls  %s  (%s)" % (what, calls, fn, loc))
