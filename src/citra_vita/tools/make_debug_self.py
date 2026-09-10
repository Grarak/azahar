#!/usr/bin/env python3
"""Produce a SELF that still carries its section headers and symbol tables.

`vita-make-fself` copies only the loadable segments of the velf into the SELF and writes a
zero into the SCE header's `shdr_offset`, so the eboot the console runs has no section
headers, no `.symtab` and no DWARF - which is why a host profiler shows addresses instead of
function names. Everything it drops is still present in the `.velf` the same build produced
(ours is 46 MB against an 8 MB eboot).

This tool takes both files and writes a third: the finished SELF verbatim, with the sections
appended after it and the two places that point at section headers filled in - the SCE
header's `shdr_offset`, and `e_shoff`/`e_shnum`/`e_shentsize`/`e_shstrndx` in the ELF header
embedded in the SELF. Nothing the loader reads to map segments is touched, because those
bytes are copied from the official tool's output rather than regenerated.

    make_debug_self.py citra_vita.self citra_vita.velf tracer.self

The result is large and is not meant to be shipped; it is what you hand a profiler that
wants symbols. Whether a given host tool reads them from here is its business - this only
makes sure they are present and reachable.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# SCE_header field offsets (src/self.h in vita-toolchain). Only the two we need.
SCE_MAGIC = 0x00454353  # "SCE\0"
SCE_ELF_OFFSET = 0x40  # uint64: where the embedded Elf32_Ehdr sits
SCE_SHDR_OFFSET = 0x50  # uint64: where the section headers sit, zero as written today

# Elf32_Ehdr field offsets.
E_SHOFF = 0x20
E_SHENTSIZE = 0x2E
E_SHNUM = 0x30
E_SHSTRNDX = 0x32

SHT_NULL = 0
SHT_NOBITS = 8


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def parse_sections(velf: bytes) -> tuple[list[dict], int, int]:
    """The velf's section table, as a list of field dicts, plus shentsize and shstrndx."""
    if velf[:4] != b"\x7fELF":
        raise SystemExit("velf: not an ELF")
    e_shoff, = struct.unpack_from("<I", velf, E_SHOFF)
    e_shentsize, = struct.unpack_from("<H", velf, E_SHENTSIZE)
    e_shnum, = struct.unpack_from("<H", velf, E_SHNUM)
    e_shstrndx, = struct.unpack_from("<H", velf, E_SHSTRNDX)
    if e_shoff == 0 or e_shnum == 0:
        raise SystemExit("velf: no section headers - the build stripped them")
    if e_shentsize != 40:
        raise SystemExit(f"velf: unexpected section header size {e_shentsize}")

    fields = ("name", "type", "flags", "addr", "offset", "size", "link", "info", "addralign",
              "entsize")
    sections = []
    for i in range(e_shnum):
        values = struct.unpack_from("<10I", velf, e_shoff + i * e_shentsize)
        sections.append(dict(zip(fields, values)))
    return sections, e_shentsize, e_shstrndx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("self_in", type=Path, help="the SELF vita-make-fself produced")
    parser.add_argument("velf_in", type=Path, help="the velf it was made from, unstripped")
    parser.add_argument("self_out", type=Path, help="where to write the debuggable SELF")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    blob = bytearray(args.self_in.read_bytes())
    velf = args.velf_in.read_bytes()

    magic, = struct.unpack_from("<I", blob, 0)
    if magic != SCE_MAGIC:
        raise SystemExit(f"{args.self_in}: not a SELF (magic {magic:#x})")
    elf_offset, = struct.unpack_from("<Q", blob, SCE_ELF_OFFSET)
    if blob[elf_offset:elf_offset + 4] != b"\x7fELF":
        raise SystemExit("SELF: the embedded ELF header is not where the SCE header says")

    sections, shentsize, shstrndx = parse_sections(velf)

    # Append every section that has bytes, then the fixed-up table. Sections keep their
    # addresses, which is what makes an address resolvable; only their file offsets move.
    out = bytearray(blob)
    copied = 0
    for section in sections:
        if section["type"] in (SHT_NULL, SHT_NOBITS) or section["size"] == 0:
            continue
        alignment = max(section["addralign"], 4)
        pad = align_up(len(out), alignment) - len(out)
        out.extend(b"\0" * pad)
        section["offset"], start = len(out), section["offset"]
        out.extend(velf[start:start + section["size"]])
        copied += section["size"]

    pad = align_up(len(out), 16) - len(out)
    out.extend(b"\0" * pad)
    shdr_offset = len(out)
    for section in sections:
        out.extend(struct.pack("<10I", section["name"], section["type"], section["flags"],
                               section["addr"], section["offset"], section["size"],
                               section["link"], section["info"], section["addralign"],
                               section["entsize"]))

    # The two pointers that make the above findable.
    struct.pack_into("<Q", out, SCE_SHDR_OFFSET, shdr_offset)
    struct.pack_into("<I", out, elf_offset + E_SHOFF, shdr_offset)
    struct.pack_into("<H", out, elf_offset + E_SHENTSIZE, shentsize)
    struct.pack_into("<H", out, elf_offset + E_SHNUM, len(sections))
    struct.pack_into("<H", out, elf_offset + E_SHSTRNDX, shstrndx)

    args.self_out.write_bytes(bytes(out))
    if not args.quiet:
        names = {s["type"] for s in sections}
        print(f"{args.self_out}: {len(blob)} bytes of SELF + {copied} bytes of sections "
              f"({len(sections)} headers at {shdr_offset:#x}, {len(names)} section types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
