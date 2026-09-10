#!/usr/bin/env python3
# Pixel-diff two binary PPMs (P6) from the emulator's screenshot command.
# Usage: sw_pixel_diff.py a.ppm b.ppm [tolerance]
# Exit 0 when every channel differs by at most `tolerance` (default 0);
# exit 1 otherwise, printing max delta, offending-pixel count, and the first mismatch.
#
# This is the arbiter for the NEON software-renderer rewrite (SW_RENDERER_NEON_PLAN.md):
# old and new paths render the same checkpoint scenes headlessly, and no phase merges with
# diffs it cannot explain against the documented tolerance list.

import sys


def load_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"P6"):
        raise SystemExit(f"{path}: not a binary PPM")
    # Header: P6 <ws> width <ws> height <ws> maxval <single ws> pixels
    fields = []
    pos = 2
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            while data[pos : pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1  # single whitespace after maxval
    w, h, maxval = fields
    if maxval != 255:
        raise SystemExit(f"{path}: unsupported maxval {maxval}")
    px = data[pos : pos + w * h * 3]
    if len(px) != w * h * 3:
        raise SystemExit(f"{path}: truncated pixel data")
    return w, h, px


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    wa, ha, a = load_ppm(sys.argv[1])
    wb, hb, b = load_ppm(sys.argv[2])
    if (wa, ha) != (wb, hb):
        print(f"size mismatch: {wa}x{ha} vs {wb}x{hb}")
        raise SystemExit(1)

    max_delta = 0
    bad = 0
    first = None
    for i in range(0, len(a)):
        d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > max_delta:
            max_delta = d
        if d > tol:
            bad += 1
            if first is None:
                p = i // 3
                first = (p % wa, p // wa, i % 3, a[i], b[i])

    total = len(a)
    if bad:
        x, y, c, va, vb = first
        print(
            f"FAIL max_delta={max_delta} channels_over_tol={bad}/{total} "
            f"first=({x},{y}) ch{c} {va} vs {vb} tol={tol}"
        )
        raise SystemExit(1)
    print(f"OK max_delta={max_delta} tol={tol} ({wa}x{ha})")


if __name__ == "__main__":
    main()
