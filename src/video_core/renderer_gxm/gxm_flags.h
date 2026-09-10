// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

namespace GxmRenderer {

/**
 * Bring-up switches read once from ux0:data/azahar/gxm_flags.txt: whitespace-separated
 * words, one per switch, so a console round trip needs no rebuild. Absent file: nothing set.
 *
 *   logdraw   log one draw's state and first vertex per report, and the first few
 *             format-converting GPU blits with their rectangles
 *   gpuhud    the Razor GPU Live counters drawn over the game every frame (imgui, as
 *             vitaGL's debugger window). Selects the USSE overview group unless one of the
 *             gpulive switches below asked for another one
 *   gpulive, gpulive0, gpulive2, gpulive3
 *             Razor GPU Live without the on-screen counters: a [gpu] line once a second with
 *             the GPU's own job timings, the idle gaps between jobs, and one counter group.
 *             The hardware reports a single group at a time, so a run picks one and the
 *             fields belonging to the others read zero:
 *               gpulive   the USSE overview - fragment and vertex processing, and the
 *                         dependent and non-dependent texture read requests
 *               gpulive0  the parameter buffer - peak usage, partial renders, paused vertex
 *                         jobs, which together say whether a scene is being split
 *               gpulive2  pixels before and after hidden surface removal, and the vertex and
 *                         primitive counts: overdraw, and how much of it HSR removes
 *               gpulive3  tiling accelerator writes and ISP parameter fetch reads, in bytes
 *   notrim    keep every surface until the video memory pool is exhausted. The default gives
 *             up the least recently used surfaces once the pool is within 12 MB of its end
 *             (RasterizerCache::TrimSurfaces via RasterizerGxm::TrimSurfaceCache); a Smash
 *             fight on the console froze at 60954 of 65280 KiB with no trim (2026-09-07).
 *             Under Vita3K a trimmed stage can come back blank: its readback limit, not the
 *             trim.
 *   smallcache behave as if the pool ended at 16 MB, so the trim runs off the
 *             console too - under Vita3K the device has memory to spare and the real ceiling
 *             is never met
 *   nogpublit format conversions go back to the CPU path, so a picture that is wrong only
 *             with the GPU blit can be told apart from one that is wrong without it
 *   timedraw  accumulate microseconds per phase of a draw and per frame end (off by default
 *             since 2026-09-07: nine clock reads a draw, a syscall each on the console) - the
 *             old notimedraw switched the same thing the other way (was on by
 *             default, reported as the "gxm time" line every report window: where the
 *             render thread's time goes, and how much of it is the GPU wait)
 *   dumpframe every five seconds, write the presented frame to ux0:data/azahar/frame_NN.ppm
 *             (twenty of them, then round again), so the picture can be looked at off the
 *             console without a camera. The pause menu's "Dump frame" button writes one
 *             such frame and, with it, every surface the cache holds (textures, render
 *             targets, depth) under ux0:data/azahar/dump/ - see RequestSurfaceDump
 *   lockstep  the emulation thread waits for each queued operation to finish
 *   alwayssync every cache operation waits for the GPU, whatever the serials say
 *   nohwvs    every draw shaded in software on the render thread, none through a generated
 *             vertex program, so a picture wrong only with the hardware path can be told
 *             from one wrong without it
 *   noskip    frameskip off: every top-screen draw is made however far behind the render
 *             thread is, so a missing picture can be told from a skipped one
 *   dumpsurfaces
 *             the surface dump the pause menu's "Dump frame" makes, once, 300 frames in
 *   vsdump    the first 16 hardware draws of the vertex program whose name contains the
 *             word in ux0:data/azahar/vsdump.txt, each to ux0:data/azahar/vsdump_NN.bin
 *             (vertex count, stride, stream bytes, uniform bytes; the default uniform
 *             buffer; stream 0), so the program's translation can be run off the console on
 *             the same data, against a host harness that runs the same program another
 *             way and compares
 *   notier2   no second tier: every program stays the first emission (usse_ir level 1)
 *   tier2now  the second tier for every program at its first draw instead of after 100
 *             (parity checks of the optimising passes)
 *   usse_dump every emitted program to ux0:data/azahar/usse/us_<hash>.gxp with a .txt of
 *             its configuration (fragment) and uv_<key>.gxp (vertex), for the SDK tools
 *             off the console
 *   usse_nohoist, usse_nocse, usse_nomadfold
 *       Tier 2 bisecting: no load hoisting (and so no shared chains or reuse), no reuse of
 *       an earlier read of the same uniform, no direct two-wide mads.
 *       Console bisecting of the vertex emitter: programs with control flow (vs_noflow),
 *       or reading uniforms through the address registers or past the prefetched ones
 *       (vs_nodyn), are refused and drawn by the software renderer.
 *             the emitter refuses that feature (those draws are skipped), to bisect a
 *             wrong picture by feature
 *   razor     load the Razor GPU capture module at init, so the Razor host tool on a PC can
 *             capture a frame over the target connection, and name each draw with the
 *             fragment program that made it. Also arms crash salvage to
 *             ux0:data/azahar/gpucrash.sgx. On a retail console the module must be installed
 *             at ur0:data/librazorcapture_es4.suprx; the load is logged either way
 */
[[nodiscard]] bool GxmFlag(const char* name);

/**
 * The pause menu's "Dump frame", the cache's half: on its next frame end the render thread
 * writes every surface the cache holds to ux0:data/azahar/dump/fNN_*.ppm (level 0, as the
 * GPU sees it: what a draw sampled and what it drew into, before the presentation quad),
 * with fNN_index.txt saying what each file is. The request is consumed by the rasterizer.
 */
void RequestSurfaceDump();
[[nodiscard]] bool SurfaceDumpPending();
[[nodiscard]] bool TakeSurfaceDumpRequest();
/// The rasterizer registers what a dump does; the presentation layer runs a pending dump
/// right after it writes the frame, since with the emulation paused in the menu the
/// rasterizer's own frame end never comes. GXM is idle at that point: no scene is open.
void SetSurfaceDumpHook(void (*hook)(void*), void* user);
void RunSurfaceDumpIfRequested();

} // namespace GxmRenderer
