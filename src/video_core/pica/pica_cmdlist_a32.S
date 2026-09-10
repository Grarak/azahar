// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// The PICA command-list fast path, hand-written for Thumb-2.
//
// The C++ version batches four commands through stack arrays so the compiler can try to
// vectorize; on ARM32 it never does, and the staging becomes pure stack traffic around every
// LUT access. This loop applies each plain register write immediately instead: one LDRD for
// the {value, header} pair walked by pointer (with a PLD running ahead, since the guest
// rewrote these lines this frame and they arrive cold), UBFX/TST field checks, one flags-LUT
// halfword, the mask-expansion word, an EOR-blended read-modify-write of the register file,
// and a byte-wise dirty-bitmap update. It exits at the first command needing special handling
// (extra data, out-of-range id, or a flags-LUT hit) and the C++ caller runs that one through
// the slow path.
//
//   extern "C" u32 PicaCmdListFastLoop(const u32* head, u32 index, u32 length,
//                                      const u16* flags_lut, const u32* expand_lut,
//                                      u32* reg_array, u8* dirty_bytes, u32* out_count);
//
// Returns the new command index. *out_count is the number of commands applied. The caller
// re-checks index+1 < length to distinguish list-end from a special command.

#if defined(__arm__)

.syntax unified
.thumb
.text

.global PicaCmdListFastLoop
.hidden PicaCmdListFastLoop
.type PicaCmdListFastLoop, %function
.thumb_func
.p2align 2
PicaCmdListFastLoop:
    // r0 = head, r1 = index, r2 = length, r3 = flags_lut
    push    {r4-r11, lr}
    ldr     r4, [sp, #36]           // expand_lut
    ldr     r5, [sp, #40]           // reg_array
    ldr     r6, [sp, #44]           // dirty_bytes
    movs    r7, #0                  // applied-command count
    mov     lr, #1                  // dirty-bit seed, hoisted out of the loop
    // Walk by pointer: r1 = &head[index], r2 = &head[length - 1]. A pair read needs
    // index + 1 < length, i.e. r1 < r2.
    add     r2, r0, r2, lsl #2
    add     r1, r0, r1, lsl #2
    subs    r2, r2, #4

.p2align 2
1:  // Per-command loop. Advance (adds r1, #8) only after all checks pass, so on a
    // special-command exit r1 still points at that pair for the slow path.
    cmp     r1, r2
    bhs     3f
    pld     [r1, #64]               // two iterations ahead; the list is cold DRAM

    ldrd    r9, r10, [r1]           // value, header

    ubfx    r11, r10, #0, #16       // cmd_id
    cmp     r11, #0x300             // RegsInternal::NUM_REGS
    bhs     3f
    tst     r10, #0x0ff00000        // extra_data_length (header bits 20-27)
    bne     3f
    ldrh    r12, [r3, r11, lsl #1]  // RegImplInfo flags
    tst     r12, #1                 // NeedsSpecialHandling
    bne     3f

    ubfx    r12, r10, #16, #4       // parameter mask nibble
    ldr     r12, [r4, r12, lsl #2]  // ExpandBitsToBytes[mask]
    ldr     r10, [r5, r11, lsl #2]  // old register value (header dead)
    eor     r9, r9, r10             // EOR-blend: old ^ ((old ^ new) & mask)
    and     r9, r9, r12
    eor     r10, r10, r9
    str     r10, [r5, r11, lsl #2]

    lsr     r8, r11, #3             // dirty byte index
    and     r12, r11, #7
    ldrb    r10, [r6, r8]
    lsl     r9, lr, r12
    orr     r10, r10, r9
    strb    r10, [r6, r8]

    adds    r1, #8
    adds    r7, #1
    b       1b

3:  // Done: list end or a command for the slow path.
    ldr     r12, [sp, #48]          // out_count
    str     r7, [r12]
    sub     r0, r1, r0              // back to a command index
    lsrs    r0, r0, #2
    pop     {r4-r11, pc}
.size PicaCmdListFastLoop, . - PicaCmdListFastLoop

.section .note.GNU-stack, "", %progbits

#endif
