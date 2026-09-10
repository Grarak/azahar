#!/usr/bin/env python3
"""Interpret ux0:data/azahar/perf.bin (the Vita build's per-second performance blob).

Usage: read_perf.py perf.bin [--summary]

One record per second: speed/fps, native slice counts, GPU-thread health, the emulation
thread's time split, and all four cores' PMU counters (core 2 exact over guest slices, the
others 10 ms whole-core samples). Events, in order: instructions passing rename, icache-stall
cycles, dcache-stall cycles, L1D refills, L1I refills, branch mispredicts.
"""
import struct
import sys

try:  # head/less closing the pipe is normal use, not an error
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (ImportError, AttributeError, ValueError):
    pass

HDR = struct.Struct("<4sII")
FIXED = struct.Struct("<QffIIQIIQQQQ")
PMU = struct.Struct("<7QII")  # cycles, 6 events, runs, pad
# v2/v3: svc_us, drain_us, drain_count, barrier_count (pad in v2), gpu_swap_ns, barrier_ns
EXTRA = struct.Struct("<QQIIQQ")
IRQ = struct.Struct("<QII")  # v4: irq_lag_us, irq_count, pad
CORE_NAMES = ["c0", "c1", "c2(guest)", "c3"]


def read_records(path):
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, rec_size = HDR.unpack_from(blob, 0)
    if magic != b"AZPF":
        sys.exit(f"not a perf blob (magic {magic!r})")
    if version not in (1, 2, 3, 4, 5):
        sys.exit(f"unknown version {version}")
    off = HDR.size
    while off + rec_size <= len(blob):
        fixed = FIXED.unpack_from(blob, off)
        pmus = [PMU.unpack_from(blob, off + FIXED.size + i * PMU.size) for i in range(4)]
        extra = None
        irq = None
        if version >= 2:
            extra = EXTRA.unpack_from(blob, off + FIXED.size + 4 * PMU.size)
        if version >= 4:
            irq = IRQ.unpack_from(blob, off + FIXED.size + 4 * PMU.size + EXTRA.size)
        brake_us = None
        if version >= 5:
            brake_us = struct.unpack_from(
                "<Q", blob, off + FIXED.size + 4 * PMU.size + EXTRA.size + IRQ.size)[0]
        yield fixed, pmus, extra, irq, brake_us
        off += rec_size


def pmu_line(name, p):
    cycles, e0, e1, e2, e3, e4, e5, runs, _ = p
    if cycles == 0:
        return None
    return (f"  pmu {name:<9} {cycles / 1e6:7.1f} Mcyc  ipc {e0 / cycles:4.2f}  "
            f"stall i {100 * e1 // cycles:2d}% d {100 * e2 // cycles:2d}%  "
            f"refill d {e3 / 1e3:.0f}k i {e4 / 1e3:.0f}k  bmiss {e5 / 1e3:.0f}k  ({runs})")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    summary_only = "--summary" in sys.argv
    records = list(read_records(args[0]))
    t0 = records[0][0][0] if records else 0

    if not summary_only:
        for fixed, pmus, extra, irq, brake_us in records:
            (t, speed, fps, slices, svcs, guest_ns, gpu_ops, gpu_brakes, gpu_busy_ns,
             runloop_us, vanrun_us, present_us) = fixed
            print(f"t={(t - t0) / 1e6:7.1f}s  speed {speed * 100:5.1f}%  game {fps:5.1f} fps  "
                  f"slices {slices} (svc {svcs})  guest {guest_ns // 1000} us")
            brake = f" ({brake_us // 1000} ms)" if brake_us else ""
            print(f"  gpu {gpu_ops} ops  busy {gpu_busy_ns // 1_000_000} ms  "
                  f"brake {gpu_brakes}{brake}")
            print(f"  emu runloop {runloop_us // 1000} ms (vanrun {vanrun_us // 1000} ms)  "
                  f"present {present_us // 1000} ms")
            if extra:
                svc_us, drain_us, drain_count, barriers, swap_ns, barrier_ns = extra
                per = f" {barrier_ns / barriers / 1000:.2f} ms each" if barriers else ""
                print(f"  emu svc {svc_us // 1000} ms  drain {drain_us // 1000} ms "
                      f"({drain_count})  |  gpu swap {swap_ns // 1_000_000} ms  "
                      f"barrier {barrier_ns // 1_000_000} ms ({barriers}{per})")
            if irq and irq[1]:
                print(f"  irq {irq[1]}/s  queued {irq[0] / irq[1] / 1000:.2f} ms each "
                      f"before the render thread posted it")
            for name, p in zip(CORE_NAMES, pmus):
                line = pmu_line(name, p)
                if line:
                    print(line)

    if records:
        n = len(records)
        print(f"\n== {n} s  speed avg {sum(r[0][1] for r in records) / n * 100:.1f}%  "
              f"fps avg {sum(r[0][2] for r in records) / n:.1f}  "
              f"gpu busy avg {sum(r[0][8] for r in records) / n / 1e6:.0f} ms  "
              f"runloop avg {sum(r[0][9] for r in records) / n / 1e3:.0f} ms "
              f"(vanrun {sum(r[0][10] for r in records) / n / 1e3:.0f} ms)  "
              f"present avg {sum(r[0][11] for r in records) / n / 1e3:.0f} ms")
        irqs = [r[3] for r in records if r[3] is not None and r[3][1]]
        if irqs:
            n_irq = sum(i[1] for i in irqs)
            print(f"   irq avg {n_irq / len(irqs):.0f}/s  lag "
                  f"{sum(i[0] for i in irqs) / n_irq / 1000:.2f} ms avg")
        extras = [r[2] for r in records if r[2] is not None]
        if extras:
            m = len(extras)
            print(f"   emu svc avg {sum(e[0] for e in extras) / m / 1e3:.0f} ms  "
                  f"drain avg {sum(e[1] for e in extras) / m / 1e3:.0f} ms "
                  f"({sum(e[2] for e in extras) / m:.0f}/s)  |  "
                  f"gpu swap avg {sum(e[4] for e in extras) / m / 1e6:.0f} ms  "
                  f"barrier avg {sum(e[5] for e in extras) / m / 1e6:.0f} ms "
                  f"({sum(e[3] for e in extras) / m:.0f}/s)")
        for c, name in enumerate(CORE_NAMES):
            cyc = sum(r[1][c][0] for r in records)
            if cyc == 0:
                continue
            ev = [sum(r[1][c][1 + i] for r in records) for i in range(6)]
            print(f"   {name:<9} {cyc / n / 1e6:7.1f} Mcyc/s  ipc {ev[0] / cyc:4.2f}  "
                  f"stall i {100 * ev[1] // cyc:2d}% d {100 * ev[2] // cyc:2d}%")


if __name__ == "__main__":
    main()
