// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include "common/assert.h"
#include "common/common_types.h"

namespace Pica::Shader::A32 {

/**
 * A minimal AArch32 (ARM encoding, little-endian) assembler, covering exactly the instructions
 * the PICA shader JIT emits. There is no A32 backend in any assembler library this project
 * already ships, and the JIT needs perhaps forty encodings, so they are spelled out here rather
 * than pulling in a new external. Every encoding is exercised against GNU as by
 * `tools/check_a32_emitter.sh` — extend that check when adding instructions.
 *
 * Emitted words go into a caller-owned vector; nothing here touches executable memory.
 */

enum class Reg : u8 {
    R0 = 0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, SP, LR, PC,
};

/// NEON quad registers. Q(n) aliases D(2n), D(2n+1); S-register views exist only for Q0-Q7.
enum class QReg : u8 {
    Q0 = 0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15,
};

enum class Cond : u8 {
    EQ = 0, NE, CS, CC, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE, AL,
};

class Label {
public:
    Label() = default;
    bool IsBound() const {
        return bound.has_value();
    }
    /// Byte offset of the bound target within the emitted code.
    std::size_t BoundOffset() const {
        ASSERT(bound);
        return *bound * sizeof(u32);
    }

private:
    friend class Emitter;
    std::optional<std::size_t> bound; ///< Word index of the target
    std::vector<std::size_t> uses;    ///< Word indices of branches awaiting the target
};

class Emitter {
public:
    explicit Emitter(std::vector<u32>& code_) : code{code_} {}

    std::size_t Offset() const {
        return code.size() * sizeof(u32);
    }

    /// Encodes a value in an ARM modified-immediate (8 bits rotated right by an even amount).
    static std::optional<u32> EncodeImm(u32 value) {
        for (u32 rot = 0; rot < 32; rot += 2) {
            const u32 v = (value << rot) | (rot ? (value >> (32 - rot)) : 0);
            if (v <= 0xFF) {
                return (rot / 2) << 8 | v;
            }
        }
        return std::nullopt;
    }

    // --- Labels -------------------------------------------------------------------------------

    void Bind(Label& label) {
        ASSERT(!label.bound);
        label.bound = code.size();
        for (const std::size_t use : label.uses) {
            Patch(use, *label.bound);
        }
        label.uses.clear();
    }

    // --- Core: moves and arithmetic -----------------------------------------------------------

    void MOVW(Reg rd, u32 imm16) {
        ASSERT(imm16 <= 0xFFFF);
        Emit(0xE3000000 | ((imm16 >> 12) << 16) | (R(rd) << 12) | (imm16 & 0xFFF));
    }

    void MOVT(Reg rd, u32 imm16) {
        ASSERT(imm16 <= 0xFFFF);
        Emit(0xE3400000 | ((imm16 >> 12) << 16) | (R(rd) << 12) | (imm16 & 0xFFF));
    }

    /// rd = any 32-bit constant, in one or two instructions.
    void MovImm32(Reg rd, u32 value) {
        MOVW(rd, value & 0xFFFF);
        if ((value >> 16) != 0) {
            MOVT(rd, value >> 16);
        }
    }

    void MOV(Reg rd, Reg rm) {
        Emit(0xE1A00000 | (R(rd) << 12) | R(rm));
    }

    void MOV(Cond cond, Reg rd, Reg rm) {
        Emit((C(cond) << 28) | 0x01A00000 | (R(rd) << 12) | R(rm));
    }

    void MOV(Reg rd, u32 imm) {
        MOV(Cond::AL, rd, imm);
    }

    void MOV(Cond cond, Reg rd, u32 imm) {
        const auto enc = EncodeImm(imm);
        ASSERT_MSG(enc, "unencodable MOV immediate {:#x}", imm);
        Emit((C(cond) << 28) | 0x03A00000 | (R(rd) << 12) | *enc);
    }

    void ADD(Reg rd, Reg rn, u32 imm) {
        DataImm(0x4, rd, rn, imm);
    }

    void SUB(Reg rd, Reg rn, u32 imm) {
        DataImm(0x2, rd, rn, imm);
    }

    void SUBS(Reg rd, Reg rn, u32 imm) {
        const auto enc = EncodeImm(imm);
        ASSERT_MSG(enc, "unencodable SUBS immediate {:#x}", imm);
        Emit(0xE2500000 | (R(rn) << 16) | (R(rd) << 12) | *enc);
    }

    void AND(Reg rd, Reg rn, u32 imm) {
        DataImm(0x0, rd, rn, imm);
    }

    void EOR(Reg rd, Reg rn, u32 imm) {
        DataImm(0x1, rd, rn, imm);
    }

    void ADD(Reg rd, Reg rn, Reg rm) {
        Emit(0xE0800000 | (R(rn) << 16) | (R(rd) << 12) | R(rm));
    }

    void ADD_LSL(Reg rd, Reg rn, Reg rm, u32 shift) {
        ASSERT(shift < 32);
        Emit(0xE0800000 | (R(rn) << 16) | (R(rd) << 12) | (shift << 7) | R(rm));
    }

    void ORR(Reg rd, Reg rn, Reg rm) {
        Emit(0xE1800000 | (R(rn) << 16) | (R(rd) << 12) | R(rm));
    }

    void CMP(Reg rn, u32 imm) {
        const auto enc = EncodeImm(imm);
        ASSERT_MSG(enc, "unencodable CMP immediate {:#x}", imm);
        Emit(0xE3500000 | (R(rn) << 16) | *enc);
    }

    void CMP(Reg rn, Reg rm) {
        Emit(0xE1500000 | (R(rn) << 16) | R(rm));
    }

    void TST(Reg rn, Reg rm) {
        Emit(0xE1100000 | (R(rn) << 16) | R(rm));
    }

    void UBFX(Reg rd, Reg rn, u32 lsb, u32 width) {
        ASSERT(lsb < 32 && width >= 1 && lsb + width <= 32);
        Emit(0xE7E00050 | ((width - 1) << 16) | (R(rd) << 12) | (lsb << 7) | R(rn));
    }

    void UXTB(Reg rd, Reg rm) {
        Emit(0xE6EF0070 | (R(rd) << 12) | R(rm));
    }

    // --- Core: memory -------------------------------------------------------------------------

    void LDR(Reg rt, Reg rn, u32 imm12 = 0) {
        ASSERT(imm12 <= 0xFFF);
        Emit(0xE5900000 | (R(rn) << 16) | (R(rt) << 12) | imm12);
    }

    void STR(Reg rt, Reg rn, u32 imm12 = 0) {
        ASSERT(imm12 <= 0xFFF);
        Emit(0xE5800000 | (R(rn) << 16) | (R(rt) << 12) | imm12);
    }

    void LDRB(Reg rt, Reg rn, u32 imm12 = 0) {
        ASSERT(imm12 <= 0xFFF);
        Emit(0xE5D00000 | (R(rn) << 16) | (R(rt) << 12) | imm12);
    }

    void STRB(Reg rt, Reg rn, u32 imm12 = 0) {
        ASSERT(imm12 <= 0xFFF);
        Emit(0xE5C00000 | (R(rn) << 16) | (R(rt) << 12) | imm12);
    }

    /// PUSH {mask}; bit n = Rn.
    void PUSH(u16 mask) {
        Emit(0xE92D0000 | mask);
    }

    /// POP {mask}; bit n = Rn.
    void POP(u16 mask) {
        Emit(0xE8BD0000 | mask);
    }

    static constexpr u16 RegMask(std::initializer_list<Reg> regs) {
        u16 mask = 0;
        for (const Reg r : regs) {
            mask |= u16(1) << static_cast<u8>(r);
        }
        return mask;
    }

    // --- Core: control flow -------------------------------------------------------------------

    void B(Label& label) {
        Branch(Cond::AL, 0xA, label);
    }

    void B(Cond cond, Label& label) {
        Branch(cond, 0xA, label);
    }

    void BL(Label& label) {
        Branch(Cond::AL, 0xB, label);
    }

    void BX(Reg rm) {
        Emit(0xE12FFF10 | R(rm));
    }

    void BLX(Reg rm) {
        Emit(0xE12FFF30 | R(rm));
    }

    // --- NEON: loads/stores (two D registers = one Q) ------------------------------------------

    void VLD1(QReg qd, Reg rn) {
        Emit(0xF4200A8F | VdBits(qd) | (R(rn) << 16));
    }

    void VST1(QReg qd, Reg rn) {
        Emit(0xF4000A8F | VdBits(qd) | (R(rn) << 16));
    }

    // --- NEON: bitwise -------------------------------------------------------------------------

    void VMOV(QReg qd, QReg qm) {
        // VORR qd, qm, qm
        Emit(0xF2200150 | VdBits(qd) | VnBits(qm) | VmBits(qm));
    }

    void VAND(QReg qd, QReg qn, QReg qm) {
        Emit(0xF2000150 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VBIC(QReg qd, QReg qn, QReg qm) {
        Emit(0xF2100150 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    /// qd = (qd & qn) | (~qd & qm) — select by the mask already in qd.
    void VBSL(QReg qd, QReg qn, QReg qm) {
        Emit(0xF3100150 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    /// Inserts qn bits into qd where qm is false.
    void VBIF(QReg qd, QReg qn, QReg qm) {
        Emit(0xF3300150 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    // --- NEON: float arithmetic ----------------------------------------------------------------

    void VADD_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF2000D40 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VSUB_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF2200D40 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VMUL_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF3000D50 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VCEQ_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF2000E40 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VCGE_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF3000E40 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VCGT_F32(QReg qd, QReg qn, QReg qm) {
        Emit(0xF3200E40 | VdBits(qd) | VnBits(qn) | VmBits(qm));
    }

    void VNEG_F32(QReg qd, QReg qm) {
        Emit(0xF3B907C0 | VdBits(qd) | VmBits(qm));
    }

    /// Round toward minus infinity (ARMv8 AArch32).
    void VRINTM_F32(QReg qd, QReg qm) {
        Emit(0xF3BA06C0 | VdBits(qd) | VmBits(qm));
    }

    /// Float to signed int, truncating.
    void VCVT_S32_F32(QReg qd, QReg qm) {
        Emit(0xF3BB0740 | 0x40 | VdBits(qd) | VmBits(qm));
    }

    /// Pairwise add on D registers: dd = {dn[0]+dn[1], dm[0]+dm[1]}.
    void VPADD_F32(u8 dd, u8 dn, u8 dm) {
        Emit(0xF3000D00 | DdBits(dd) | DnBits(dn) | DmBits(dm));
    }

    void VADD_F32_D(u8 dd, u8 dn, u8 dm) {
        Emit(0xF2000D00 | DdBits(dd) | DnBits(dn) | DmBits(dm));
    }

    // --- NEON: lane traffic --------------------------------------------------------------------

    /// VDUP.32 qd, dm[lane]
    void VDUP_32(QReg qd, u8 dm, u8 lane) {
        ASSERT(lane < 2);
        const u32 imm4 = (u32(lane) << 3) | 0b0100;
        Emit(0xF3B00C40 | (imm4 << 16) | VdBits(qd) | DmBits(dm));
    }

    /// VMOV.32 dd[lane], rt
    void VMOV_ToLane(u8 dd, u8 lane, Reg rt) {
        ASSERT(lane < 2);
        Emit(0xEE000B10 | (u32(lane) << 21) | ((dd & 0xF) << 16) | (R(rt) << 12) |
             ((dd >> 4) << 7));
    }

    /// VMOV.32 rt, dn[lane]
    void VMOV_FromLane(Reg rt, u8 dn, u8 lane) {
        ASSERT(lane < 2);
        Emit(0xEE100B10 | (u32(lane) << 21) | ((dn & 0xF) << 16) | (R(rt) << 12) |
             ((dn >> 4) << 7));
    }

    /// VMOV.F32 qd, #1.0
    void VMOV_F32_One(QReg qd) {
        // Modified immediate: abcdefgh = 0b01110000, cmode = 1111, op = 0.
        Emit(0xF2800F50 | (0b111 << 16) | VdBits(qd));
    }

    // --- VFP scalar (S registers view D0-D15 only) ---------------------------------------------

    void VMOV_F32_S(u8 sd, u8 sm) {
        Emit(0xEEB00A40 | SdBits(sd) | SmBits(sm));
    }

    void VCMP_F32(u8 sd, u8 sm) {
        Emit(0xEEB40A40 | SdBits(sd) | SmBits(sm));
    }

    /// Transfers the VFP compare flags to APSR.
    void VMRS_APSR() {
        Emit(0xEEF1FA10);
    }

    void VDIV_F32(u8 sd, u8 sn, u8 sm) {
        Emit(0xEE800A00 | SdBits(sd) | SnBits(sn) | SmBits(sm));
    }

    void VSQRT_F32(u8 sd, u8 sm) {
        Emit(0xEEB10AC0 | SdBits(sd) | SmBits(sm));
    }

    // --- VFP: callee-saved spills --------------------------------------------------------------

    /// VPUSH {d8, d9}
    void VPUSH_D8D9() {
        Emit(0xED2D8B04);
    }

    /// VPOP {d8, d9}
    void VPOP_D8D9() {
        Emit(0xECBD8B04);
    }

private:
    static u32 R(Reg r) {
        return static_cast<u32>(r);
    }
    static u32 C(Cond c) {
        return static_cast<u32>(c);
    }
    static u32 Q2D(QReg q) {
        return static_cast<u32>(q) * 2;
    }
    static u32 VdBits(QReg q) {
        const u32 d = Q2D(q);
        return ((d & 0xF) << 12) | ((d >> 4) << 22);
    }
    static u32 VnBits(QReg q) {
        const u32 d = Q2D(q);
        return ((d & 0xF) << 16) | ((d >> 4) << 7);
    }
    static u32 VmBits(QReg q) {
        const u32 d = Q2D(q);
        return (d & 0xF) | ((d >> 4) << 5);
    }
    static u32 DdBits(u32 d) {
        return ((d & 0xF) << 12) | ((d >> 4) << 22);
    }
    static u32 DnBits(u32 d) {
        return ((d & 0xF) << 16) | ((d >> 4) << 7);
    }
    static u32 DmBits(u32 d) {
        return (d & 0xF) | ((d >> 4) << 5);
    }
    static u32 SdBits(u32 s) {
        return ((s >> 1) << 12) | ((s & 1) << 22);
    }
    static u32 SnBits(u32 s) {
        return ((s >> 1) << 16) | ((s & 1) << 7);
    }
    static u32 SmBits(u32 s) {
        return (s >> 1) | ((s & 1) << 5);
    }

    void DataImm(u32 opcode, Reg rd, Reg rn, u32 imm) {
        const auto enc = EncodeImm(imm);
        ASSERT_MSG(enc, "unencodable data-processing immediate {:#x}", imm);
        Emit(0xE2000000 | (opcode << 21) | (R(rn) << 16) | (R(rd) << 12) | *enc);
    }

    void Branch(Cond cond, u32 op, Label& label) {
        if (label.bound) {
            const std::size_t use = code.size();
            Emit((C(cond) << 28) | (op << 24));
            Patch(use, *label.bound);
        } else {
            label.uses.push_back(code.size());
            Emit((C(cond) << 28) | (op << 24));
        }
    }

    void Patch(std::size_t use, std::size_t target) {
        const s32 offset = static_cast<s32>(target) - static_cast<s32>(use) - 2;
        ASSERT_MSG(offset >= -(1 << 23) && offset < (1 << 23), "branch out of range");
        code[use] = (code[use] & 0xFF000000) | (static_cast<u32>(offset) & 0x00FFFFFF);
    }

    void Emit(u32 word) {
        code.push_back(word);
    }

    std::vector<u32>& code;
};

} // namespace Pica::Shader::A32
