@ vanative — entering the guest and leaving it.
@
@ ARM state throughout: exception entry is ARM, and `ldm ^` / `rfe` / `movs pc` have no useful
@ Thumb forms. Called from Thumb C on the taken core (core 2), returned to with interworking.
@
@ azahar_enter(ctx, host): loads the guest's user-mode register file from ctx and returns from
@ exception into user mode at ctx.r15 with ctx.cpsr. It "returns" only when the guest takes an
@ exception — svc, undefined instruction, prefetch abort or data abort — through one of the four
@ vector stubs below, which save the user registers back into ctx, record why, and unwind to the
@ C caller as if azahar_enter had returned. The return value is the reason.
@
@ Layout of AzaharContext: r0..r15 at 0x00..0x3c, cpsr at 0x40, tpidruro at 0x44.
@ Layout of the host block: 0x00 host cpsr, 0x04 host sp, 0x08 original SP_svc, 0x0c ctx address.
@
@ Modes. The core-2 thread runs in whatever PL1 mode Sony's kernel threads use — System or
@ Supervisor; the entry code saves the CPSR and switches to Supervisor, because `ldm {..}^` from
@ System mode is unpredictable (System already uses the user registers). SP_svc is borrowed as
@ the pointer to ctx for the `ldm ^` + `rfe` pair and put back on exit. The exception stubs load
@ their mode's SP from a literal, so SP_und and SP_abt are clobbered — the C side saves Sony's
@ values before installing the vectors and restores them before handing the core back.

    .syntax unified
    .arm
    .fpu neon
    .text

    .global azahar_enter
    .type azahar_enter, %function
azahar_enter:
    @ r0 = ctx, r1 = host block
    push    {r4-r11, lr}
    ldr     r2, =azahar_progress
    mov     r3, #10
    str     r3, [r2]
    mrs     r2, cpsr
    str     r2, [r1, #0x00]         @ host cpsr: mode and interrupt bits to come back to
    str     sp, [r1, #0x04]         @ host sp, in the host's mode
    str     r0, [r1, #0x0c]         @ ctx, for the exception stubs
    ldr     r3, =azahar_host_ptr
    str     r1, [r3]
    ldr     r3, =azahar_ctx_ptr
    str     r0, [r3]

    ldr     r2, =azahar_guest_active
    mov     r3, #1
    str     r3, [r2]
    @ The guest's VFP state. Core 2 is ours alone, so the register file simply is the guest's
    @ between runs; loading it here and storing it at exit is what makes a guest thread switch
    @ (SaveContext/LoadContext on the app side) carry it.
    add     r2, r0, #0x50
    vldmia  r2!, {d0-d15}
    vldmia  r2, {d16-d31}
    ldr     r3, [r0, #0x4c]
    vmsr    fpscr, r3
    @ Drop any exclusive reservation before the guest resumes. Whether an exception return
    @ clears the local monitor is IMPLEMENTATION DEFINED on ARMv7-A, so it cannot be assumed:
    @ guest thread A can take its exit between an ldrex and its strex, and if the reservation
    @ survives into thread B then B's strex to the same word succeeds when the architecture
    @ says it must fail. That is a 3DS LightLock handed to two owners at once, and it is how
    @ eight of thirteen guest threads ended up parked on address arbiters nobody would signal
    @ (2026-09-01). One instruction per entry, and it also makes ClearExclusiveState's
    @ no-op honest. A spurious strex failure is architecturally permitted and every lock loop
    @ retries, so clearing on every entry costs nothing but the instruction.
    clrex
    cpsid   if                      @ nothing may interrupt the banked-register dance
    cps     #0x13                   @ Supervisor
    str     sp, [r1, #0x08]         @ Sony's SP_svc, put back on exit
    ldr     r2, =azahar_progress
    mov     r3, #11
    str     r3, [r2]
    mov     sp, r0                  @ SP_svc = ctx
    ldmia   sp, {r0-r14}^           @ the guest's r0-r12 and its user-mode sp and lr
    nop                             @ no banked access in the instruction after ldm ^
    add     sp, sp, #0x3c           @ -> ctx.r15, ctx.cpsr
    rfeia   sp                      @ pc = ctx.r15, cpsr = ctx.cpsr: user mode, guest runs

@ ---------------------------------------------------------------------------- vector stubs
@ Each runs in its own exception mode with that mode's LR and SPSR. They save the user register
@ file first — before any register is touched — then set the reason and fall into the common tail.

@ One stub per exception mode. With a guest active: save the user register file into ctx, set
@ the reason, fall into the common tail. With no guest active — an exception taken by whatever
@ else ran on this core while our vectors were installed — record it once, put this mode's SP
@ back, and chain to Sony's handler for that vector through a {sp, pc} pair C filled in at
@ install. The 2026-08-29 run without this took a prefetch abort between install and the first
@ run and looped through the stubs storing to a null ctx pointer.
    .macro AZAHAR_STUB name, reason
    .global azahar_vec_\name
    .type azahar_vec_\name, %function
azahar_vec_\name:
    ldr     sp, =azahar_prog_\name    @ the stub was reached: the word becomes its own address
    str     sp, [sp]
    ldr     sp, =azahar_guest_active
    ldr     sp, [sp]
    cmp     sp, #0
    beq     9f
    ldr     sp, =azahar_ctx_ptr
    ldr     sp, [sp]
    stmia   sp, {r0-r14}^
    nop
    mov     r0, #\reason
    b       azahar_exit_common
9:  ldr     sp, =azahar_stray_stack_top
    push    {r0-r3}
    ldr     r0, =azahar_stray
    ldr     r1, [r0]                @ count: only the first is recorded in full
    cmp     r1, #0
    bne     8f
    mov     r2, #\reason
    str     r2, [r0, #4]
    str     lr, [r0, #8]
    mrs     r2, spsr
    str     r2, [r0, #12]
    mrc     p15, 0, r2, c5, c0, 0   @ DFSR
    str     r2, [r0, #16]
    mrc     p15, 0, r2, c6, c0, 0   @ DFAR
    str     r2, [r0, #20]
    mrc     p15, 0, r2, c5, c0, 1   @ IFSR
    str     r2, [r0, #24]
    mrc     p15, 0, r2, c6, c0, 2   @ IFAR
    str     r2, [r0, #28]
8:  add     r1, r1, #1
    str     r1, [r0]
    pop     {r0-r3}
    ldr     sp, =azahar_seal_flag
    ldr     sp, [sp]
    cmp     sp, #0
    beq     7f
    .if \reason == 4
    @ Sealed data aborts are retried a few times before parking. Kept as a generic guard:
    @ re-executing the aborted store costs nothing, and a recoverable one-off (spurious
    @ abort, transient walk hiccup) then survives instead of wedging the core. The counter
    @ is cumulative per install and shown in the stray report; past the cap the stub parks
    @ as before. (The 2026-08-30 wedges this was written for turned out to be a dropped
    @ relocation - see azahar_irq_stack_top below - not a transient fault.)
    ldr     sp, =azahar_stray_stack_top
    push    {r0-r1}
    ldr     r0, =azahar_stray_retry
    ldr     r1, [r0]
    add     r1, r1, #1
    str     r1, [r0]
    cmp     r1, #16
    pop     {r0-r1}
    bgt     6f
    subs    pc, lr, #8              @ back to the aborted store, mode and cpsr restored
    .endif
6:  b       6b                      @ sealed: nothing of Sony's is mapped to chain to; the
                                    @ record above is the report, the syscall side times out
7:  ldr     sp, =azahar_chain_\name
    ldm     sp, {sp, pc}            @ this mode's SP as Sony left it, then Sony's handler
    .endm

    AZAHAR_STUB und, 2
    AZAHAR_STUB svc, 1
    AZAHAR_STUB pabt, 3
    AZAHAR_STUB dabt, 4

@ IRQ. With ICCPMR at 0x10 the only interrupt that reaches this core is our private timer (PPI
@ 29, banked priority 00), so this is the preemption tick. With a guest active: save the user
@ file, acknowledge at the GIC, stop the timer, exit with reason 5. Without one — the timer
@ fired while the host was between runs — acknowledge, stop it, and return to the interrupted
@ instruction with every register intact. Uses its own stack; Sony's SP_irq is put back before
@ the return, and at uninstall.
    .global azahar_vec_irq
    .type azahar_vec_irq, %function
azahar_vec_irq:
    ldr     sp, =azahar_prog_irq
    str     sp, [sp]
    ldr     sp, =azahar_guest_active
    ldr     sp, [sp]
    cmp     sp, #0
    beq     9f
    ldr     sp, =azahar_ctx_ptr
    ldr     sp, [sp]
    stmia   sp, {r0-r14}^
    nop
    mov     r3, lr                  @ LR_irq: exit_common needs it after the call
    bl      azahar_irq_ack             @ r0-r2 free now; r3 preserved by it
    mov     lr, r3
    mov     r0, #5
    b       azahar_exit_common
9:  ldr     sp, =azahar_irq_stack_top
    push    {r0-r3, lr}
    bl      azahar_irq_ack
    ldr     r0, =azahar_irq_stray
    ldr     r1, [r0]
    add     r1, r1, #1
    str     r1, [r0]
    pop     {r0-r3, lr}
    ldr     sp, =azahar_chain_irq
    ldr     sp, [sp]                @ Sony's SP_irq
    subs    pc, lr, #4

@ Acknowledge whatever is at the CPU interface, stop the private timer, record the ID. Clobbers
@ r0-r2 only.
azahar_irq_ack:
    ldr     r0, =azahar_gic_va
    ldr     r0, [r0]
    cmp     r0, #0
    bxeq    lr
    ldr     r1, [r0, #0x10c]        @ GICC_IAR
    lsl     r2, r1, #22
    lsr     r2, r2, #22             @ the ID field, bits [9:0]
    add     r2, r2, #1
    cmp     r2, #0x400              @ ID 0x3ff is spurious: nothing to end
    strne   r1, [r0, #0x110]        @ GICC_EOIR
    ldr     r2, =azahar_irq_last
    str     r1, [r2]
    mov     r2, #0
    str     r2, [r0, #0x608]        @ private timer control: off
    mov     r2, #1
    str     r2, [r0, #0x60c]        @ interrupt status: clear
    dsb     sy
    bx      lr
@ sp = ctx (this mode's SP), r0 = reason, guest r0-r14 already in ctx.
azahar_exit_common:
    ldr     r1, =azahar_progress
    add     r2, r0, #30             @ 31..34: the user file is saved
    str     r2, [r1]
    ldr     r1, =azahar_guest_active
    mov     r2, #0
    str     r2, [r1]
    str     lr, [sp, #0x3c]         @ raw exception LR; C adjusts it per reason
    mrs     r1, spsr
    str     r1, [sp, #0x40]         @ the guest CPSR at the exception
    add     r2, sp, #0x50
    vstmia  r2!, {d0-d15}
    vstmia  r2, {d16-d31}
    vmrs    r3, fpscr
    str     r3, [sp, #0x4c]
    ldr     r1, =azahar_exit_info
    str     r0, [r1, #0x00]         @ reason
    str     lr, [r1, #0x04]         @ raw lr
    mrs     r2, spsr
    str     r2, [r1, #0x08]         @ spsr
    mrc     p15, 0, r2, c5, c0, 0   @ DFSR
    str     r2, [r1, #0x0c]
    mrc     p15, 0, r2, c6, c0, 0   @ DFAR
    str     r2, [r1, #0x10]
    mrc     p15, 0, r2, c5, c0, 1   @ IFSR
    str     r2, [r1, #0x14]
    mrc     p15, 0, r2, c6, c0, 2   @ IFAR
    str     r2, [r1, #0x18]

    @ Back to the host: Sony's SP_svc first, then the host's own mode, sp and callee-saved
    @ registers, then return to the C caller of azahar_enter with the reason in r0.
    ldr     r1, =azahar_host_ptr
    ldr     r1, [r1]
    cps     #0x13
    ldr     sp, [r1, #0x08]         @ SP_svc as it was
    ldr     r2, =azahar_progress
    mov     r3, #40
    str     r3, [r2]
    ldr     r2, [r1, #0x00]
    bic     r2, r2, #0x20           @ MSR ignores T anyway; keep the write honest
    msr     cpsr_c, r2              @ host mode, host I/F
    ldr     sp, [r1, #0x04]         @ host sp
    ldr     r2, =azahar_progress
    mov     r3, #41
    str     r3, [r2]
    pop     {r4-r11, pc}            @ r0 = reason; interworks back to Thumb

@ ---------------------------------------------------------------------------- banked SPs
@ uint32_t azahar_get_banked_sp(uint32_t mode); void azahar_set_banked_sp(uint32_t mode, uint32_t sp).
@ Switches to the named mode with interrupts masked, reads or writes its SP, and comes back.
@ Used to save Sony's SP_und and SP_abt before our vectors clobber them and restore them after.

    .global azahar_get_banked_sp
    .type azahar_get_banked_sp, %function
azahar_get_banked_sp:
    mrs     r1, cpsr
    cpsid   if
    mrs     r2, cpsr
    bic     r2, r2, #0x1f
    orr     r2, r2, r0
    msr     cpsr_c, r2
    mov     r0, sp
    msr     cpsr_c, r1
    bx      lr

    .global azahar_set_banked_sp
    .type azahar_set_banked_sp, %function
azahar_set_banked_sp:
    mrs     r2, cpsr
    cpsid   if
    mrs     r3, cpsr
    bic     r3, r3, #0x1f
    orr     r3, r3, r0
    msr     cpsr_c, r3
    mov     sp, r1
    msr     cpsr_c, r2
    bx      lr

    .ltorg

@ ---------------------------------------------------------------------------- data
    .data
    .align 2
    .global azahar_host_ptr
azahar_host_ptr:   .word 0
    .global azahar_ctx_ptr
azahar_ctx_ptr:    .word 0
    @ reason, raw lr, spsr, dfsr, dfar, ifsr, ifar
    .global azahar_exit_info
azahar_exit_info:  .word 0, 0, 0, 0, 0, 0, 0
    .global azahar_progress
azahar_progress:   .word 0
    .global azahar_prog_und
azahar_prog_und:   .word 0
    .global azahar_prog_svc
azahar_prog_svc:   .word 0
    .global azahar_prog_pabt
azahar_prog_pabt:  .word 0
    .global azahar_prog_dabt
azahar_prog_dabt:  .word 0
    .global azahar_guest_active
azahar_guest_active: .word 0
    @ count, reason, lr, spsr, dfsr, dfar, ifsr, ifar
    .global azahar_stray
azahar_stray:      .word 0, 0, 0, 0, 0, 0, 0, 0
    @ {saved SP for the mode, Sony's handler} per vector, filled by C at install
    .global azahar_chain_und
azahar_chain_und:  .word 0, 0
    .global azahar_chain_svc
azahar_chain_svc:  .word 0, 0
    .global azahar_chain_pabt
azahar_chain_pabt: .word 0, 0
    .global azahar_chain_dabt
azahar_chain_dabt: .word 0, 0
    .global azahar_chain_irq
azahar_chain_irq:  .word 0, 0
    .global azahar_prog_irq
azahar_prog_irq:   .word 0
    .global azahar_gic_va
azahar_gic_va:     .word 0
    .global azahar_seal_flag
azahar_seal_flag:  .word 0
    .global azahar_irq_last
azahar_irq_last:   .word 0
    .global azahar_irq_stray
azahar_irq_stray:  .word 0
    .global azahar_stray_retry
azahar_stray_retry: .word 0
    .bss
    .align 3
azahar_stray_stack: .space 64
    .global azahar_stray_stack_top
azahar_stray_stack_top:
azahar_irq_stack:  .space 64
    .global azahar_irq_stack_top
azahar_irq_stack_top:
    .space 8                        @ keeps azahar_irq_stack_top strictly inside the segment:
                                    @ a symbol at the exact end (one-past-the-end address)
                                    @ gets its relocations silently DROPPED by
                                    @ vita-elf-create - it cannot attribute the address to
                                    @ a segment - so every reference to it stayed at the
                                    @ link-time address after load (2026-08-30: the stray
                                    @ pushes aborted on the unmapped link VA, the seal
                                    @ ranges stretched toward it, and a C dereference of it
                                    @ crashed the resident thread outright)

    .section .note.GNU-stack,"",%progbits
