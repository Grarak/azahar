![Azahar Emulator](https://azahar-emu.org/resources/images/logo/azahar-name-and-logo.svg)

# Azahar Emulator for the PSVita

This is an early, almost entirely vibecoded port of Azahar for the PSVita, so do not expect playability or support from my side.

## Install

- Grab the latest vpk and skprx from [releases](https://github.com/Grarak/azahar/releases)
- Grab latest release of [CapUnlocker](https://github.com/GrapheneCt/CapUnlocker/releases)
- Install the vpk
- Install both skprx files (azaharnative and CapUnlocker) to the KERNEL section in your config.txt
- Put your **decrypted** ROMs in ux0:/data/azahar/roms
- Set your CPU to 500MHz before launching a game

## Known Issues
- Pausing the game and resuming might freeze your console, you need a hardreset
- Savestates are untested and probably don't work
- Expect crashes

## Build
See [DEVELOPMENT.md](./DEVELOPMENT.md)

## Credits
- The Azahar Team
- vitaGL
- Vita3K
- @TheIronUniverse for livearea assets
