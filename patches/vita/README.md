# Vita patches to vendored submodules

`externals/libressl` is a git submodule, so changes to it cannot be committed to this
repository. The two it needs for the PS Vita build are kept here instead, and have to be applied
to a fresh checkout before building for that target:

```sh
git -C externals/libressl apply ../../patches/vita/libressl-vita.patch
```

Both are places where the library assumes a desktop kernel:

- `crypto/armcap.c` — the ARM capability probe identifies instructions by executing them and
  catching `SIGILL`, which needs `sigsetjmp` (absent from the console's newlib) and signal
  delivery this platform does not provide. The answer is known at build time — Cortex-A9, NEON,
  no ARMv8 crypto extensions — so the patch states it instead of probing for it.
- `crypto/cryptlib.c` — `OPENSSL_showfatal` wrote to syslog, which does not exist here. It
  writes to stderr instead.

Everything else the port needs from third-party code is done from CMake or from our own headers,
which is why this directory has only the one patch in it.
