# azahar-native

A PS Vita kernel plugin that lets [azahar](https://github.com/azahar-emu/azahar), a Nintendo 3DS
emulator, run the guest's ARM11 code on the Vita's own Cortex-A9 rather than interpreting or
recompiling it.

3DS' ARMv6 ISA is backwards compatible with the Vita's ARMv7 ISA, so most 3DS user-mode code is already valid Vita instructions. What
stops it running directly is that a 3DS binary expects its own virtual addresses — `.text` at
`0x00100000`, its heap and stack where the 3DS kernel put them — and those addresses belong to
the Vita kernel in any ordinary process. This plugin takes one CPU core away from Sony's
scheduler, installs a translation table and exception vectors of its own on it, and enters the
guest in user mode at the addresses it expects. Execution leaves on the first exception — a
system call, an undefined instruction, a fault, or the preemption timer — and hands the register
file back to the emulator, which services it and asks for the next slice.

The emulator keeps the guest's memory as ordinary user memblocks and describes them to the
plugin, so there is no second copy: the guest sees the emulator's own pages at guest addresses.

## Scope

This runs on a console its owner has opened themselves, through
[taiHEN](https://github.com/TheOfficialFloW/taiHEN), the plugin framework every Vita homebrew
uses. It has no networking. It reads and writes CPU system registers on the core it takes and
restores them when it releases it. It touches no system but the one it is running on.

## Building

Needs [VitaSDK](https://vitasdk.org).

```sh
export VITASDK=/usr/local/vitasdk
cmake -S . -B build
cmake --build build
```

That produces `build/azaharnative.skprx`, the plugin, and `build/libazaharnative_stub.a`, the
stub the emulator links against.

## Installing

Copy the plugin somewhere on the memory card and add it to taiHEN's kernel section:

```sh
# on the Vita, ux0:tai/config.txt
*KERNEL
ux0:tai/azaharnative.skprx
```

Reboot, or reload the config with a taiHEN plugin manager. The plugin does nothing until an
application calls it, so a console with it installed and no emulator running behaves normally.

To build against it, put `libazaharnative_stub.a` on the linker line and include
`src/azahar_native.h`; the nine calls it declares are the whole interface.

## Where it writes

Nothing, unless asked. Create `ux0:data/azahar/log` to turn on the plugin's own log, which then
goes to `ux0:data/azahar/native.txt`. A title makes thousands of calls a second and each logged
line costs a file write, so this is off by default and should stay off outside diagnosis.

## How it works

`DEVELOPMENT.md`.
