#!/usr/bin/env bash
# Regenerates the vendored .gxp next to each .cg in this directory.
#
# psp2cgc is the SDK's own Cg compiler: a 32-bit Windows tool, so it runs under wine in a
# container. Its output is committed, which is why the console needs no runtime compiler
# (libshacccg.suprx) to present - see the note at the top of gxm_present.cpp.
#
# Usage: shaders/compile.sh [directory]   (needs docker and ~/psvita_sdk)
# The directory defaults to this one; src/video_core/renderer_gxm/shaders holds the
# fragment ubershader, compiled the same way.
set -euo pipefail
cd "${1:-$(dirname "$0")}"
IMAGE=psp2cgc
if ! docker image inspect $IMAGE >/dev/null 2>&1; then
    echo "building $IMAGE image (wine + i386 runtime)" >&2
    docker build -q -t $IMAGE - <<'DOCKEREOF'
FROM ubuntu:24.04
RUN dpkg --add-architecture i386 && apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wine wine32:i386 && \
    rm -rf /var/lib/apt/lists/*
ENV WINEPREFIX=/wpfx WINEDEBUG=-all
RUN wineboot -u 2>/dev/null; true
DOCKEREOF
fi
# wine refuses to run as a uid that owns no home, so the container runs as root and hands
# the outputs back to the invoking user at the end.
docker run --rm -v "$HOME/psvita_sdk:/sdk:ro" -v "$PWD:/shaders" \
    -e OWNER="$(id -u):$(id -g)" $IMAGE bash -c '
    fail=0
    for f in /shaders/*.cg; do
        b=$(basename "$f" .cg)
        # Vertex programs are the *_v sources; everything else is a fragment program.
        case $b in *_v) p=sce_vp_psp2;; *) p=sce_fp_psp2;; esac
        # The ubershader needs -O2: at -O3 the compiler dies with a "fatal internal
        # compiler error" on it (measured 2026-09-03; the register allocator, not the
        # source - it compiles once lighting or the TEV stages are taken out).
        case $b in uber*) o=-O2;; *) o=;; esac
        if wine /sdk/PSVITA/sdk/host_tools/bin/psp2cgc.exe -profile $p $o "$f" \
                -o "/shaders/$b.gxp" 2>/tmp/err && [ -s "/shaders/$b.gxp" ]; then
            echo "  $b.gxp ($p, $(stat -c%s /shaders/$b.gxp) bytes)"
        else
            fail=1; echo "FAILED $b ($p)"; head -8 /tmp/err
        fi
    done
    chown "$OWNER" /shaders/*.gxp 2>/dev/null || true
    exit $fail' 2>&1 | grep -v "^002\|^wine:"
