// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/arch.h"
#if CITRA_ARCH(arm32) && CITRA_HAS_SHADER_JIT

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/shader/shader_jit_a32_compiler.h"

#ifdef __linux__
#include <sys/mman.h>
#endif

using namespace Pica::Shader::A32;
using nihstro::DestRegister;
using nihstro::RegisterType;

namespace Pica::Shader {

typedef void (JitShader::*JitFunction)(Instruction instr);

const std::array<JitFunction, 64> instr_table = {
    &JitShader::Compile_ADD,    // add
    &JitShader::Compile_DP3,    // dp3
    &JitShader::Compile_DP4,    // dp4
    &JitShader::Compile_DPH,    // dph
    nullptr,                    // unknown
    &JitShader::Compile_EX2,    // ex2
    &JitShader::Compile_LG2,    // lg2
    nullptr,                    // unknown
    &JitShader::Compile_MUL,    // mul
    &JitShader::Compile_SGE,    // sge
    &JitShader::Compile_SLT,    // slt
    &JitShader::Compile_FLR,    // flr
    &JitShader::Compile_MAX,    // max
    &JitShader::Compile_MIN,    // min
    &JitShader::Compile_RCP,    // rcp
    &JitShader::Compile_RSQ,    // rsq
    nullptr,                    // unknown
    nullptr,                    // unknown
    &JitShader::Compile_MOVA,   // mova
    &JitShader::Compile_MOV,    // mov
    nullptr,                    // unknown
    nullptr,                    // unknown
    nullptr,                    // unknown
    nullptr,                    // unknown
    &JitShader::Compile_DPH,    // dphi
    nullptr,                    // unknown
    &JitShader::Compile_SGE,    // sgei
    &JitShader::Compile_SLT,    // slti
    nullptr,                    // unknown
    nullptr,                    // unknown
    nullptr,                    // unknown
    nullptr,                    // unknown
    nullptr,                    // unknown
    &JitShader::Compile_NOP,    // nop
    &JitShader::Compile_END,    // end
    &JitShader::Compile_BREAKC, // breakc
    &JitShader::Compile_CALL,   // call
    &JitShader::Compile_CALLC,  // callc
    &JitShader::Compile_CALLU,  // callu
    &JitShader::Compile_IF,     // ifu
    &JitShader::Compile_IF,     // ifc
    &JitShader::Compile_LOOP,   // loop
    &JitShader::Compile_EMIT,   // emit
    &JitShader::Compile_SETE,   // sete
    &JitShader::Compile_JMP,    // jmpc
    &JitShader::Compile_JMP,    // jmpu
    &JitShader::Compile_CMP,    // cmp
    &JitShader::Compile_CMP,    // cmp
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // madi
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
    &JitShader::Compile_MAD,    // mad
};

// Register allocation. AAPCS32: r0-r3 and r12 are caller-saved, r4-r11 callee-saved,
// d8-d15 (Q4-Q7) are the only callee-saved NEON registers.
/// Pointer to the uniform memory (first argument, kept live)
constexpr Reg UNIFORMS = Reg::R0;
/// Pointer to the ShaderUnit instance (second argument, kept live)
constexpr Reg STATE = Reg::R1;
/// Scratch registers
constexpr Reg SCRATCH0 = Reg::R2;
constexpr Reg SCRATCH1 = Reg::R3;
constexpr Reg SCRATCH2 = Reg::R12;
/// The two 32-bit VS address offset registers set by the MOVA instruction
constexpr Reg ADDROFFS_REG_0 = Reg::R4;
constexpr Reg ADDROFFS_REG_1 = Reg::R5;
/// VS loop count register (the aL source for relative addressing)
constexpr Reg LOOPCOUNT_REG = Reg::R6;
/// Current LOOP iteration counter
constexpr Reg LOOPCOUNT = Reg::R7;
/// LOOPCOUNT_REG increment per iteration
constexpr Reg LOOPINC = Reg::R8;
/// Results of the previous CMP instruction, one register per component
constexpr Reg COND0 = Reg::R10;
constexpr Reg COND1 = Reg::R11;

/// Loaded with the swizzled source registers; scratch otherwise
constexpr QReg SRC1 = QReg::Q1; // lanes are s4-s7
constexpr QReg SRC2 = QReg::Q2; // lanes are s8-s11
constexpr QReg SRC3 = QReg::Q3;
constexpr QReg VSCRATCH0 = QReg::Q0; // lanes are s0-s3
constexpr QReg VSCRATCH1 = QReg::Q8;
constexpr QReg VSCRATCH2 = QReg::Q9;
/// [1.0f x4]. Q4 is callee-saved, so it survives calls out to helpers; lane 0 is s16.
constexpr QReg ONE = QReg::Q4;

/// First D register of a Q register
static constexpr u8 D(QReg q) {
    return static_cast<u8>(q) * 2;
}

/// Raw constant for the source register selector that indicates no swizzling is performed
static const u8 NO_SRC_REG_SWIZZLE = 0x1b;
/// Raw constant for the destination register enable mask that indicates all components are enabled
static const u8 NO_DEST_REG_MASK = 0xf;

/// Per-lane write masks for every dest_mask value, selected by absolute address at runtime.
alignas(16) static const std::array<std::array<u32, 4>, 16> dest_mask_table = [] {
    std::array<std::array<u32, 4>, 16> table{};
    for (u32 mask = 0; mask < 16; mask++) {
        for (u32 comp = 0; comp < 4; comp++) {
            table[mask][comp] = (mask & (8u >> comp)) ? 0xFFFFFFFF : 0;
        }
    }
    return table;
}();

static void LogCritical(const char* msg) {
    LOG_CRITICAL(HW_GPU, "{}", msg);
}

// The interpreter computes EX2/LG2 with libm on the first component and broadcasts; doing the
// same keeps the JIT bit-identical to it, which the x64/a64 polynomial approximations are not.
static void Exp2Vec(f32* v) {
    const f32 r = std::exp2(v[0]);
    v[0] = v[1] = v[2] = v[3] = r;
}

static void Log2Vec(f32* v) {
    const f32 r = std::log2(v[0]);
    v[0] = v[1] = v[2] = v[3] = r;
}

static void EmitThunk(GeometryEmitter* emitter, ShaderUnit* unit) {
    emitter->Emit(unit->output[unit->output_bank]);
    unit->output_bank = !unit->output_bank;
}

/// The caller-saved state that is live across an external call: the two pointer arguments and
/// the link register, plus R2 to keep the stack 8-aligned at the call.
static constexpr u16 FAR_CALL_SAVE =
    Emitter::RegMask({Reg::R0, Reg::R1, Reg::R2, Reg::LR});

void JitShader::Compile_SwizzleSrc(Instruction instr, u32 src_num, SourceRegister src_reg,
                                   QReg dest) {
    Reg src_ptr = Reg::PC; // poison
    std::size_t src_offset = 0;
    switch (src_reg.GetRegisterType()) {
    case RegisterType::FloatUniform:
        src_ptr = UNIFORMS;
        src_offset = Uniforms::GetFloatUniformOffset(src_reg.GetIndex());
        break;
    case RegisterType::Input:
        src_ptr = STATE;
        src_offset = ShaderUnit::InputOffset(src_reg.GetIndex());
        break;
    case RegisterType::Temporary:
        src_ptr = STATE;
        src_offset = ShaderUnit::TemporaryOffset(src_reg.GetIndex());
        break;
    default:
        UNREACHABLE_MSG("Encountered unknown source register type: {}", src_reg.GetRegisterType());
        break;
    }

    u32 operand_desc_id;
    const bool is_inverted =
        (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));
    u32 address_register_index;
    u32 offset_src;

    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD ||
        instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI) {
        operand_desc_id = instr.mad.operand_desc_id;
        offset_src = is_inverted ? 3 : 2;
        address_register_index = instr.mad.address_register_index;
    } else {
        operand_desc_id = instr.common.operand_desc_id;
        offset_src = is_inverted ? 2 : 1;
        address_register_index = instr.common.address_register_index;
    }

    auto& e = *emitter;
    if (src_reg.GetRegisterType() == RegisterType::FloatUniform && src_num == offset_src &&
        address_register_index != 0) {
        Reg address_reg = Reg::PC;
        switch (address_register_index) {
        case 1:
            address_reg = ADDROFFS_REG_0;
            break;
        case 2:
            address_reg = ADDROFFS_REG_1;
            break;
        case 3:
            address_reg = LOOPCOUNT_REG;
            break;
        default:
            UNREACHABLE();
            break;
        }

        // offset = (address_reg >= -128 && address_reg <= 127) ? address_reg : 0
        e.ADD(SCRATCH0, address_reg, 128);
        e.CMP(SCRATCH0, 256);
        e.MOV(Cond::CC, SCRATCH0, address_reg);
        e.MOV(Cond::CS, SCRATCH0, u32(0));

        // index = (src_reg.GetIndex() + offset) & 0x7f
        if (src_reg.GetIndex() != 0) {
            e.ADD(SCRATCH0, SCRATCH0, src_reg.GetIndex());
        }
        e.AND(SCRATCH0, SCRATCH0, 0x7f);

        // index > 95 ? vec4(1.0) : uniforms.f[index]
        e.VMOV(dest, ONE);
        e.CMP(SCRATCH0, 95);
        Label load_end;
        e.B(Cond::GT, load_end);
        e.ADD_LSL(SCRATCH1, src_ptr, SCRATCH0, 4);
        e.VLD1(dest, SCRATCH1);
        e.Bind(load_end);
    } else {
        if (src_offset != 0) {
            e.ADD(SCRATCH1, src_ptr, static_cast<u32>(src_offset));
            e.VLD1(dest, SCRATCH1);
        } else {
            e.VLD1(dest, src_ptr);
        }
    }

    const SwizzlePattern swiz = {(*swizzle_data)[operand_desc_id]};

    const u8 sel = swiz.GetRawSelector(src_num);
    switch (sel) {
    case NO_SRC_REG_SWIZZLE:
        break;
    case 0b00'00'00'00:
    case 0b01'01'01'01:
    case 0b10'10'10'10:
    case 0b11'11'11'11: {
        const u8 lane = sel & 0b11;
        e.VDUP_32(dest, D(dest) + (lane >> 1), lane & 1);
        break;
    }
    default: {
        const u8 table[] = {
            u8((sel & 0b11'00'00'00) >> 6),
            u8((sel & 0b00'11'00'00) >> 4),
            u8((sel & 0b00'00'11'00) >> 2),
            u8((sel & 0b00'00'00'11) >> 0),
        };
        e.VMOV(VSCRATCH0, dest);
        for (u8 comp = 0; comp < 4; comp++) {
            if (table[comp] != comp) {
                e.VMOV_FromLane(SCRATCH0, D(VSCRATCH0) + (table[comp] >> 1), table[comp] & 1);
                e.VMOV_ToLane(D(dest) + (comp >> 1), comp & 1, SCRATCH0);
            }
        }
        break;
    }
    }

    const bool negate[] = {swiz.negate_src1, swiz.negate_src2, swiz.negate_src3};
    if (negate[src_num - 1]) {
        e.VNEG_F32(dest, dest);
    }
}

void JitShader::Compile_DestEnable(Instruction instr, QReg src) {
    DestRegister dest;
    u32 operand_desc_id;
    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD ||
        instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI) {
        operand_desc_id = instr.mad.operand_desc_id;
        dest = instr.mad.dest.Value();
    } else {
        operand_desc_id = instr.common.operand_desc_id;
        dest = instr.common.dest.Value();
    }

    const SwizzlePattern swiz = {(*swizzle_data)[operand_desc_id]};

    std::size_t dest_offset = 0;
    switch (dest.GetRegisterType()) {
    case RegisterType::Output:
        dest_offset = ShaderUnit::OutputOffset(dest.GetIndex());
        break;
    case RegisterType::Temporary:
        dest_offset = ShaderUnit::TemporaryOffset(dest.GetIndex());
        break;
    default:
        UNREACHABLE_MSG("Encountered unknown destination register type: {}",
                        dest.GetRegisterType());
        break;
    }

    constexpr u32 OutputBankShift = std::countr_zero(ShaderUnit::OutputBankSize);

    auto& e = *emitter;
    // Compute the destination address into SCRATCH1.
    e.ADD(SCRATCH1, STATE, static_cast<u32>(dest_offset));
    if (dest.GetRegisterType() == RegisterType::Output) {
        e.LDRB(SCRATCH0, STATE, static_cast<u32>(ShaderUnit::OutputBankOffset()));
        e.ADD_LSL(SCRATCH1, SCRATCH1, SCRATCH0, OutputBankShift);
    }

    if (swiz.dest_mask == NO_DEST_REG_MASK) {
        e.VST1(src, SCRATCH1);
    } else {
        // Blend the enabled components into the old value through the precomputed lane mask.
        e.VLD1(VSCRATCH0, SCRATCH1);
        e.MovImm32(SCRATCH0, reinterpret_cast<u32>(dest_mask_table[swiz.dest_mask].data()));
        e.VLD1(VSCRATCH2, SCRATCH0);
        e.VBSL(VSCRATCH2, src, VSCRATCH0);
        e.VST1(VSCRATCH2, SCRATCH1);
    }
}

void JitShader::Compile_SanitizedMul(QReg src1, QReg src2) {
    // 0 * inf must give 0 rather than NaN, while NaN inputs still propagate. Compute masks of
    // "both inputs ordered" and "result ordered": a result that went NaN from ordered inputs can
    // only be a 0 * inf, and gets cleared.
    auto& e = *emitter;
    e.VCEQ_F32(VSCRATCH0, src1, src1);
    e.VCEQ_F32(VSCRATCH1, src2, src2);
    e.VAND(VSCRATCH0, VSCRATCH0, VSCRATCH1);
    e.VMUL_F32(src1, src1, src2);
    e.VCEQ_F32(VSCRATCH1, src1, src1);
    e.VBIC(VSCRATCH0, VSCRATCH0, VSCRATCH1);
    e.VBIC(src1, src1, VSCRATCH0);
}

void JitShader::Compile_HorizontalSum(QReg src) {
    // {a,b,c,d} -> {a+c, b+d} -> {a+b+c+d} broadcast
    auto& e = *emitter;
    e.VADD_F32_D(D(VSCRATCH0), D(src), D(src) + 1);
    e.VPADD_F32(D(VSCRATCH0), D(VSCRATCH0), D(VSCRATCH0));
    e.VDUP_32(src, D(VSCRATCH0), 0);
}

void JitShader::Compile_EvaluateCondition(Instruction instr) {
    // Leaves the truth of the condition in the NE flag.
    const bool refx = instr.flow_control.refx.Value();
    const bool refy = instr.flow_control.refy.Value();

    auto& e = *emitter;
    switch (instr.flow_control.op) {
    case Instruction::FlowControlType::Or: {
        Reg op_x = COND0;
        if (!refx) {
            e.EOR(SCRATCH0, COND0, 1);
            op_x = SCRATCH0;
        }
        Reg op_y = COND1;
        if (!refy) {
            e.EOR(SCRATCH1, COND1, 1);
            op_y = SCRATCH1;
        }
        e.ORR(SCRATCH0, op_x, op_y);
        e.CMP(SCRATCH0, u32(0));
        break;
    }
    case Instruction::FlowControlType::And: {
        Reg op_x = COND0;
        if (!refx) {
            e.EOR(SCRATCH0, COND0, 1);
            op_x = SCRATCH0;
        }
        Reg op_y = COND1;
        if (!refy) {
            e.EOR(SCRATCH1, COND1, 1);
            op_y = SCRATCH1;
        }
        e.TST(op_x, op_y);
        break;
    }
    case Instruction::FlowControlType::JustX:
        e.CMP(COND0, u32(refx ? 0 : 1));
        break;
    case Instruction::FlowControlType::JustY:
        e.CMP(COND1, u32(refy ? 0 : 1));
        break;
    default:
        UNREACHABLE();
        break;
    }
}

void JitShader::Compile_UniformCondition(Instruction instr) {
    const std::size_t offset = Uniforms::GetBoolUniformOffset(instr.flow_control.bool_uniform_id);
    emitter->LDRB(SCRATCH0, UNIFORMS, static_cast<u32>(offset));
    emitter->CMP(SCRATCH0, u32(0));
}

void JitShader::Compile_ADD(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    emitter->VADD_F32(SRC1, SRC1, SRC2);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_DP3(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);

    Compile_SanitizedMul(SRC1, SRC2);

    // Zero the last element
    emitter->MOV(SCRATCH0, u32(0));
    emitter->VMOV_ToLane(D(SRC1) + 1, 1, SCRATCH0);

    Compile_HorizontalSum(SRC1);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_DP4(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);

    Compile_SanitizedMul(SRC1, SRC2);
    Compile_HorizontalSum(SRC1);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_DPH(Instruction instr) {
    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::DPHI) {
        Compile_SwizzleSrc(instr, 1, instr.common.src1i, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2i, SRC2);
    } else {
        Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    }

    // Set the fourth component of src1 to 1.0. SRC1's lanes are s4-s7, ONE's lane 0 is s16.
    emitter->VMOV_F32_S(7, 16);

    Compile_SanitizedMul(SRC1, SRC2);
    Compile_HorizontalSum(SRC1);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_VectorHelperCall(void (*fn)(f32*)) {
    auto& e = *emitter;
    e.PUSH(FAR_CALL_SAVE);
    e.SUB(Reg::SP, Reg::SP, 16);
    e.VST1(SRC1, Reg::SP);
    e.MOV(Reg::R0, Reg::SP);
    e.MovImm32(SCRATCH1, reinterpret_cast<u32>(fn));
    e.BLX(SCRATCH1);
    e.VLD1(SRC1, Reg::SP);
    e.ADD(Reg::SP, Reg::SP, 16);
    e.POP(FAR_CALL_SAVE);
}

void JitShader::Compile_EX2(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_VectorHelperCall(&Exp2Vec);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_LG2(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_VectorHelperCall(&Log2Vec);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_MUL(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    Compile_SanitizedMul(SRC1, SRC2);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_SGE(Instruction instr) {
    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::SGEI) {
        Compile_SwizzleSrc(instr, 1, instr.common.src1i, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2i, SRC2);
    } else {
        Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    }

    emitter->VCGE_F32(SRC2, SRC1, SRC2);
    emitter->VAND(SRC2, SRC2, ONE);
    Compile_DestEnable(instr, SRC2);
}

void JitShader::Compile_SLT(Instruction instr) {
    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::SLTI) {
        Compile_SwizzleSrc(instr, 1, instr.common.src1i, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2i, SRC2);
    } else {
        Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
        Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    }

    emitter->VCGT_F32(SRC1, SRC2, SRC1);
    emitter->VAND(SRC1, SRC1, ONE);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_FLR(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    emitter->VRINTM_F32(SRC1, SRC1);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_MAX(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    // (b < a) ? a : b, with the PICA's NaN behaviour: NaN in src2 propagates.
    emitter->VCGT_F32(VSCRATCH0, SRC1, SRC2);
    emitter->VBIF(SRC1, SRC2, VSCRATCH0);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_MIN(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);
    // (a < b) ? a : b, with the PICA's NaN behaviour: NaN in src2 propagates.
    emitter->VCGT_F32(VSCRATCH0, SRC2, SRC1);
    emitter->VBIF(SRC1, SRC2, VSCRATCH0);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_MOVA(Instruction instr) {
    const SwizzlePattern swiz = {(*swizzle_data)[instr.common.operand_desc_id]};

    if (!swiz.DestComponentEnabled(0) && !swiz.DestComponentEnabled(1)) {
        return;
    }

    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);

    // Convert floats to integers using truncation
    emitter->VCVT_S32_F32(SRC1, SRC1);

    if (swiz.DestComponentEnabled(0)) {
        emitter->VMOV_FromLane(ADDROFFS_REG_0, D(SRC1), 0);
    }
    if (swiz.DestComponentEnabled(1)) {
        emitter->VMOV_FromLane(ADDROFFS_REG_1, D(SRC1), 1);
    }
}

void JitShader::Compile_MOV(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_RCP(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);

    // Exact 1.0f / N on the first lane, then broadcast. s4 is SRC1 lane 0, s16 is 1.0.
    emitter->VDIV_F32(4, 16, 4);
    emitter->VDUP_32(SRC1, D(SRC1), 0);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_RSQ(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);

    // Exact 1.0f / sqrt(N) on the first lane, then broadcast.
    emitter->VSQRT_F32(0, 4); // s0 = sqrt(s4)
    emitter->VDIV_F32(4, 16, 0);
    emitter->VDUP_32(SRC1, D(SRC1), 0);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_NOP(Instruction instr) {}

void JitShader::Compile_END(Instruction instr) {
    auto& e = *emitter;
    // Save conditional code
    e.STRB(COND0, STATE, u32(offsetof(ShaderUnit, conditional_code[0])));
    e.STRB(COND1, STATE, u32(offsetof(ShaderUnit, conditional_code[1])));

    // Save address/loop registers
    e.STR(ADDROFFS_REG_0, STATE, u32(offsetof(ShaderUnit, address_registers[0])));
    e.STR(ADDROFFS_REG_1, STATE, u32(offsetof(ShaderUnit, address_registers[1])));
    e.STR(LOOPCOUNT_REG, STATE, u32(offsetof(ShaderUnit, address_registers[2])));

    // Drop the return-guard frame, restore and return
    e.ADD(Reg::SP, Reg::SP, 8);
    e.VPOP_D8D9();
    e.POP(Emitter::RegMask({Reg::R3, Reg::R4, Reg::R5, Reg::R6, Reg::R7, Reg::R8, Reg::R9,
                            Reg::R10, Reg::R11, Reg::PC}));
}

void JitShader::Compile_BREAKC(Instruction instr) {
    if (loop_depth) {
        Compile_EvaluateCondition(instr);
        ASSERT(!loop_break_labels.empty());
        emitter->B(Cond::NE, loop_break_labels.back());
    }
}

void JitShader::Compile_CALL(Instruction instr) {
    auto& e = *emitter;
    // Push the return offset and the link register; both keep the stack 8-aligned.
    e.MOVW(SCRATCH0, instr.flow_control.dest_offset + instr.flow_control.num_instructions);
    e.PUSH(Emitter::RegMask({Reg::R2, Reg::LR}));

    e.BL(instruction_labels[instr.flow_control.dest_offset]);

    // Discard the return offset, restore the link register
    e.POP(Emitter::RegMask({Reg::R2, Reg::LR}));
}

void JitShader::Compile_CALLC(Instruction instr) {
    Compile_EvaluateCondition(instr);
    Label skip;
    emitter->B(Cond::EQ, skip);
    Compile_CALL(instr);
    emitter->Bind(skip);
}

void JitShader::Compile_CALLU(Instruction instr) {
    Compile_UniformCondition(instr);
    Label skip;
    emitter->B(Cond::EQ, skip);
    Compile_CALL(instr);
    emitter->Bind(skip);
}

void JitShader::Compile_CMP(Instruction instr) {
    using Op = Instruction::Common::CompareOpType::Op;
    const Op op_x = instr.common.compare_op.x;
    const Op op_y = instr.common.compare_op.y;

    Compile_SwizzleSrc(instr, 1, instr.common.src1, SRC1);
    Compile_SwizzleSrc(instr, 2, instr.common.src2, SRC2);

    // Conditions after a VFP compare (VMRS): unordered results compare false except for NE.
    static constexpr Cond cmp[] = {Cond::EQ, Cond::NE, Cond::MI, Cond::LS, Cond::GT, Cond::GE};

    auto& e = *emitter;
    // X component: SRC1 lane 0 is s4, SRC2 lane 0 is s8.
    e.VCMP_F32(4, 8);
    e.VMRS_APSR();
    e.MOV(COND0, u32(0));
    e.MOV(cmp[op_x], COND0, u32(1));

    // Y component: lanes 1 are s5 and s9.
    e.VCMP_F32(5, 9);
    e.VMRS_APSR();
    e.MOV(COND1, u32(0));
    e.MOV(cmp[op_y], COND1, u32(1));
}

void JitShader::Compile_MAD(Instruction instr) {
    Compile_SwizzleSrc(instr, 1, instr.mad.src1, SRC1);

    if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI) {
        Compile_SwizzleSrc(instr, 2, instr.mad.src2i, SRC2);
        Compile_SwizzleSrc(instr, 3, instr.mad.src3i, SRC3);
    } else {
        Compile_SwizzleSrc(instr, 2, instr.mad.src2, SRC2);
        Compile_SwizzleSrc(instr, 3, instr.mad.src3, SRC3);
    }

    Compile_SanitizedMul(SRC1, SRC2);
    emitter->VADD_F32(SRC1, SRC1, SRC3);
    Compile_DestEnable(instr, SRC1);
}

void JitShader::Compile_IF(Instruction instr) {
    ASSERT_MSG(instr.flow_control.dest_offset >= program_counter,
               "Backwards if-statements not supported");
    Label l_else, l_endif;

    if (instr.opcode.Value() == OpCode::Id::IFU) {
        Compile_UniformCondition(instr);
    } else if (instr.opcode.Value() == OpCode::Id::IFC) {
        Compile_EvaluateCondition(instr);
    }
    emitter->B(Cond::EQ, l_else);

    Compile_Block(instr.flow_control.dest_offset);

    if (instr.flow_control.num_instructions == 0) {
        emitter->Bind(l_else);
        return;
    }

    emitter->B(l_endif);

    emitter->Bind(l_else);
    Compile_Block(instr.flow_control.dest_offset + instr.flow_control.num_instructions);

    emitter->Bind(l_endif);
}

void JitShader::Compile_LOOP(Instruction instr) {
    ASSERT_MSG(instr.flow_control.dest_offset >= program_counter,
               "Backwards loops not supported");

    auto& e = *emitter;
    if (loop_depth++) {
        // R9 pads the frame to 8 bytes
        e.PUSH(Emitter::RegMask({LOOPCOUNT_REG, LOOPCOUNT, LOOPINC, Reg::R9}));
    }

    // The integer uniform at index instr.flow_control.int_uniform_id:
    // X-component is the iteration count, Y the aL start, Z the aL increment.
    const std::size_t offset = Uniforms::GetIntUniformOffset(instr.flow_control.int_uniform_id);
    e.LDR(LOOPCOUNT, UNIFORMS, static_cast<u32>(offset));
    e.UBFX(LOOPCOUNT_REG, LOOPCOUNT, 8, 8);
    e.UBFX(LOOPINC, LOOPCOUNT, 16, 8);
    e.UXTB(LOOPCOUNT, LOOPCOUNT);
    e.ADD(LOOPCOUNT, LOOPCOUNT, 1);

    Label l_loop_start;
    e.Bind(l_loop_start);

    loop_break_labels.emplace_back(Label());
    Compile_Block(instr.flow_control.dest_offset + 1);

    e.ADD(LOOPCOUNT_REG, LOOPCOUNT_REG, LOOPINC);
    e.SUBS(LOOPCOUNT, LOOPCOUNT, 1);
    e.B(Cond::NE, l_loop_start);

    e.Bind(loop_break_labels.back());
    loop_break_labels.pop_back();

    if (--loop_depth) {
        e.POP(Emitter::RegMask({LOOPCOUNT_REG, LOOPCOUNT, LOOPINC, Reg::R9}));
    }
}

void JitShader::Compile_JMP(Instruction instr) {
    if (instr.opcode.Value() == OpCode::Id::JMPC) {
        Compile_EvaluateCondition(instr);
    } else if (instr.opcode.Value() == OpCode::Id::JMPU) {
        Compile_UniformCondition(instr);
    } else {
        UNREACHABLE();
    }

    const bool inverted_condition =
        (instr.opcode.Value() == OpCode::Id::JMPU) && (instr.flow_control.num_instructions & 1);

    Label& b = instruction_labels[instr.flow_control.dest_offset];
    emitter->B(inverted_condition ? Cond::EQ : Cond::NE, b);
}

void JitShader::Compile_EMIT(Instruction instr) {
    auto& e = *emitter;
    Label have_emitter, end;

    e.LDR(SCRATCH0, STATE, u32(offsetof(ShaderUnit, emitter_ptr)));
    e.CMP(SCRATCH0, u32(0));
    e.B(Cond::NE, have_emitter);

    e.PUSH(FAR_CALL_SAVE);
    e.MovImm32(Reg::R0, reinterpret_cast<u32>("Execute EMIT on VS"));
    e.MovImm32(SCRATCH1, reinterpret_cast<u32>(&LogCritical));
    e.BLX(SCRATCH1);
    e.POP(FAR_CALL_SAVE);
    e.B(end);

    e.Bind(have_emitter);
    e.PUSH(FAR_CALL_SAVE);
    e.MOV(Reg::R0, SCRATCH0);
    // R1 already holds the ShaderUnit pointer
    e.MovImm32(SCRATCH1, reinterpret_cast<u32>(&EmitThunk));
    e.BLX(SCRATCH1);
    e.POP(FAR_CALL_SAVE);
    e.Bind(end);
}

void JitShader::Compile_SETE(Instruction instr) {
    auto& e = *emitter;
    Label have_emitter, end;

    e.LDR(SCRATCH0, STATE, u32(offsetof(ShaderUnit, emitter_ptr)));
    e.CMP(SCRATCH0, u32(0));
    e.B(Cond::NE, have_emitter);

    e.PUSH(FAR_CALL_SAVE);
    e.MovImm32(Reg::R0, reinterpret_cast<u32>("Execute SETEMIT on VS"));
    e.MovImm32(SCRATCH1, reinterpret_cast<u32>(&LogCritical));
    e.BLX(SCRATCH1);
    e.POP(FAR_CALL_SAVE);
    e.B(end);

    e.Bind(have_emitter);
    const GeometryEmitter::EmitState new_state{
        .winding = instr.setemit.winding != 0,
        .prim_emit = instr.setemit.prim_emit != 0,
        .vertex_id = static_cast<uint8_t>(instr.setemit.vertex_id),
    };
    e.MOV(SCRATCH1, u32(new_state.raw));
    e.STRB(SCRATCH1, SCRATCH0, u32(offsetof(GeometryEmitter, emit_state)));
    e.Bind(end);
}

void JitShader::Compile_Block(u32 end) {
    while (program_counter < end) {
        Compile_NextInstr();
    }
}

void JitShader::Compile_Return() {
    auto& e = *emitter;
    // Peek the return offset pushed by the innermost CALL (or the entry sentinel) and return to
    // the call site if we've reached it.
    e.LDR(SCRATCH0, Reg::SP, 0);
    e.MOVW(SCRATCH1, program_counter);
    e.CMP(SCRATCH0, SCRATCH1);
    Label skip;
    e.B(Cond::NE, skip);
    e.BX(Reg::LR);
    e.Bind(skip);
}

void JitShader::Compile_NextInstr() {
    if (std::binary_search(return_offsets.begin(), return_offsets.end(), program_counter)) {
        Compile_Return();
    }

    emitter->Bind(instruction_labels[program_counter]);

    // Treat the last possible instruction as an implicit END, like the other backends.
    Instruction instr{};
    if (program_counter < MAX_PROGRAM_CODE_LENGTH - 1) {
        instr.hex = (*program_code)[program_counter];
    } else {
        instr.opcode.Assign(OpCode::Id::END);
    }
    ++program_counter;

    const OpCode::Id opcode = instr.opcode.Value();
    const auto instr_func = instr_table[static_cast<std::size_t>(opcode)];

    if (instr_func) {
        ((*this).*instr_func)(instr);
    } else {
        LOG_CRITICAL(HW_GPU, "Unhandled instruction: 0x{:02x} (0x{:08x})",
                     static_cast<u32>(instr.opcode.Value().EffectiveOpCode()), instr.hex);
    }
}

void JitShader::FindReturnOffsets() {
    return_offsets.clear();

    for (std::size_t offset = 0; offset < program_code->size(); ++offset) {
        const Instruction instr = {(*program_code)[offset]};

        switch (instr.opcode.Value()) {
        case OpCode::Id::CALL:
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU:
            return_offsets.push_back(instr.flow_control.dest_offset +
                                     instr.flow_control.num_instructions);
            break;
        default:
            break;
        }
    }

    std::sort(return_offsets.begin(), return_offsets.end());
}

void JitShader::Compile(const std::array<u32, MAX_PROGRAM_CODE_LENGTH>* program_code_,
                        const std::array<u32, MAX_SWIZZLE_DATA_LENGTH>* swizzle_data_) {
    program_code = program_code_;
    swizzle_data = swizzle_data_;

    code_vec.clear();
    emitter = std::make_unique<Emitter>(code_vec);
    program_counter = 0;
    loop_depth = 0;
    instruction_labels.fill(Label());

    FindReturnOffsets();

    auto& e = *emitter;

    // Prologue. Arguments: R0 = uniforms, R1 = state, R2 = entry address inside this block.
    // Ten registers, not the nine that need saving: the AAPCS wants the stack 8-aligned at
    // every call, and nine words left it off by four for everything the program calls out to
    // (the geometry emitter faulted on an aligned NEON stack load in OutputVertex). R3 is the
    // padding word.
    e.PUSH(Emitter::RegMask({Reg::R3, Reg::R4, Reg::R5, Reg::R6, Reg::R7, Reg::R8, Reg::R9,
                             Reg::R10, Reg::R11, Reg::LR}));
    e.VPUSH_D8D9();

    // Return-guard frame: a sentinel that never equals a program counter, so a Compile_Return
    // reached from the main routine falls through instead of returning.
    e.MovImm32(SCRATCH1, 0xFFFFFFFF);
    e.PUSH(Emitter::RegMask({Reg::R3, Reg::LR}));

    // Load address/loop registers and conditional code
    e.LDR(ADDROFFS_REG_0, STATE, u32(offsetof(ShaderUnit, address_registers[0])));
    e.LDR(ADDROFFS_REG_1, STATE, u32(offsetof(ShaderUnit, address_registers[1])));
    e.LDR(LOOPCOUNT_REG, STATE, u32(offsetof(ShaderUnit, address_registers[2])));
    e.LDRB(COND0, STATE, u32(offsetof(ShaderUnit, conditional_code[0])));
    e.LDRB(COND1, STATE, u32(offsetof(ShaderUnit, conditional_code[1])));

    e.VMOV_F32_One(ONE);

    // Jump to the requested entry point
    e.BX(SCRATCH0);

    // Compile entire program
    Compile_Block(static_cast<u32>(program_code->size()));

    program_code = nullptr;
    swizzle_data = nullptr;
    return_offsets.clear();
    return_offsets.shrink_to_fit();

    // Copy to executable memory
    const std::size_t code_size = code_vec.size() * sizeof(u32);
#ifdef __linux__
    void* mem = mmap(nullptr, code_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_MSG(mem != MAP_FAILED, "could not allocate shader JIT code memory");
    code_mem = static_cast<u8*>(mem);
    code_mem_size = code_size;
    std::memcpy(code_mem, code_vec.data(), code_size);
    __builtin___clear_cache(reinterpret_cast<char*>(code_mem),
                            reinterpret_cast<char*>(code_mem) + code_size);
#else
#error "The A32 shader JIT only knows how to map code on Linux"
#endif

    program = reinterpret_cast<CompiledShader*>(code_mem);

    emitter.reset();
    code_vec.clear();
    code_vec.shrink_to_fit();
}

void JitShader::Run(const ShaderSetup& setup, ShaderUnit& state, u32 offset) const {
    ASSERT(program != nullptr);
    ASSERT(instruction_labels[offset].IsBound());
    const void* entry = code_mem + instruction_labels[offset].BoundOffset();
    program(&setup.uniforms, &state, entry);
}

JitShader::JitShader() = default;

JitShader::~JitShader() {
#ifdef __linux__
    if (code_mem != nullptr) {
        munmap(code_mem, code_mem_size);
    }
#endif
}

} // namespace Pica::Shader

#endif // CITRA_ARCH(arm32)
