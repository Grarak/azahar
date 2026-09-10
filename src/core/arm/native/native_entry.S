// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// Entry to and exit from natively executed guest code.
//
// Entry loads the guest's register file and branches to the guest's PC, so from that instruction on
// the host CPU is the guest CPU. Nothing returns here by falling through: the guest is left only by
// a signal — a trapped svc, a fault, or the preemption timer — whose handler rewrites the interrupted
// context to resume at NativeExitGuest instead of in the guest. The kernel's sigreturn then restores
// that rewritten context, which is what makes the exit look like an ordinary function return.
//
// Neither stub names a global symbol. Everything they need travels in registers or in the context
// structure, so the code needs no absolute relocations and works unchanged in a position
// independent executable.

#if defined(__arm__)

.syntax unified
.arm
.fpu neon

// Field offsets within Core::NativeGuestContext, asserted against the C++ definition.
#define CTX_REGS    0
#define CTX_CPSR    64
#define CTX_FPSCR   68
#define CTX_EXIT    72
#define CTX_HOST_SP 76
#define CTX_FPREGS  80

.section .text.NativeEnterGuest, "ax", %progbits
.global NativeEnterGuest
.hidden NativeEnterGuest
.type NativeEnterGuest, %function
.align 2
// u32 NativeEnterGuest(NativeGuestContext* ctx)
NativeEnterGuest:
    // r3 is pushed only to keep sp 8-byte aligned across the d-register push.
    push    {r3, r4-r11, lr}

    // The guest register load below clobbers every d register, but d8-d15 are callee-saved in
    // the host ABI — the caller's compiler keeps live doubles in them across this call. The same
    // goes for the host's FPSCR mode bits (rounding, flush-to-zero), which the guest sets to its
    // own values. Both must survive the round trip; missing this corrupted unrelated host
    // floating point on the emulation thread, in ways that came and went with code generation.
    vpush   {d8-d15}
    vmrs    r1, fpscr
    push    {r1, r2}

    // Record the host stack pointer in the context. The exit stub cannot recover it from the
    // guest's own sp, which by then holds a guest address.
    str     sp, [r0, #CTX_HOST_SP]

    // Restore the guest's floating point state.
    add     r1, r0, #CTX_FPREGS
    vldmia  r1, {d0-d15}
    add     r1, r0, #(CTX_FPREGS + 128)
    vldmia  r1, {d16-d31}
    ldr     r1, [r0, #CTX_FPSCR]
    vmsr    fpscr, r1

    // Restore the guest's condition flags.
    ldr     r1, [r0, #CTX_CPSR]
    msr     APSR_nzcvq, r1

    // Load r0-r14 and branch to the guest's PC. Bit 0 of the loaded PC selects ARM or Thumb, which
    // is why the guest PC is stored with the T bit folded in.
    ldm     r0, {r0-pc}
.size NativeEnterGuest, . - NativeEnterGuest

.section .text.NativeExitGuest, "ax", %progbits
.global NativeExitGuest
.hidden NativeExitGuest
.type NativeExitGuest, %function
.align 2
// Resumed into by sigreturn, with r0 holding the exit reason, r1 the context, and sp already
// restored to the host stack. Never called directly.
NativeExitGuest:
    // The integer registers were copied out of the signal context by the handler, but floating
    // point state was not: sigreturn has just restored it to the guest's values, so it is live in
    // the hardware right now and has to be saved before any C++ runs.
    add     r2, r1, #CTX_FPREGS
    vstmia  r2, {d0-d15}
    add     r2, r1, #(CTX_FPREGS + 128)
    vstmia  r2, {d16-d31}
    vmrs    r2, fpscr
    str     r2, [r1, #CTX_FPSCR]

    // The guest may have left an exclusive monitor reservation open. Dropping it here makes any
    // later strex fail, which every ldrex/strex loop already copes with.
    clrex

    // Restore the host's floating point state saved at entry: FPSCR mode bits first, then the
    // callee-saved d registers. r0 still holds the exit reason and must survive.
    pop     {r2, r3}
    vmsr    fpscr, r2
    vpop    {d8-d15}

    pop     {r3, r4-r11, lr}
    bx      lr
.size NativeExitGuest, . - NativeExitGuest

.section .note.GNU-stack, "", %progbits

#endif
