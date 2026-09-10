// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/arch.h"
#if CITRA_ARCH(arm32) && CITRA_HAS_SHADER_JIT

#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "common/common_types.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/shader/a32_emitter.h"

using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

namespace Pica {
struct ShaderUnit;
}

namespace Pica::Shader {

/**
 * The PICA vertex shader JIT for 32-bit ARM hosts, assuming an ARMv8 AArch32 CPU (it emits
 * VRINTM and expects NEON). Same shape as the AArch64 compiler; the emitter is a32_emitter.h.
 *
 * Two deliberate semantic notes. EX2 and LG2 call out to libm's exp2/log2, which is exactly what
 * the interpreter computes — closer to it than the polynomial approximations the x64/a64 JITs
 * carry. And A32 NEON always flushes denormals to zero, where the interpreter's scalar VFP math
 * preserves them; PICA f24 has no subnormals of its own, so nothing observable depends on it.
 */
class JitShader {
public:
    JitShader();
    ~JitShader();

    void Run(const ShaderSetup& setup, ShaderUnit& state, u32 offset) const;

    void Compile(const std::array<u32, MAX_PROGRAM_CODE_LENGTH>* program_code,
                 const std::array<u32, MAX_SWIZZLE_DATA_LENGTH>* swizzle_data);

    void Compile_ADD(Instruction instr);
    void Compile_DP3(Instruction instr);
    void Compile_DP4(Instruction instr);
    void Compile_DPH(Instruction instr);
    void Compile_EX2(Instruction instr);
    void Compile_LG2(Instruction instr);
    void Compile_MUL(Instruction instr);
    void Compile_SGE(Instruction instr);
    void Compile_SLT(Instruction instr);
    void Compile_FLR(Instruction instr);
    void Compile_MAX(Instruction instr);
    void Compile_MIN(Instruction instr);
    void Compile_RCP(Instruction instr);
    void Compile_RSQ(Instruction instr);
    void Compile_MOVA(Instruction instr);
    void Compile_MOV(Instruction instr);
    void Compile_NOP(Instruction instr);
    void Compile_END(Instruction instr);
    void Compile_BREAKC(Instruction instr);
    void Compile_CALL(Instruction instr);
    void Compile_CALLC(Instruction instr);
    void Compile_CALLU(Instruction instr);
    void Compile_IF(Instruction instr);
    void Compile_LOOP(Instruction instr);
    void Compile_JMP(Instruction instr);
    void Compile_CMP(Instruction instr);
    void Compile_MAD(Instruction instr);
    void Compile_EMIT(Instruction instr);
    void Compile_SETE(Instruction instr);

private:
    void Compile_Block(u32 end);
    void Compile_NextInstr();

    void Compile_SwizzleSrc(Instruction instr, u32 src_num, SourceRegister src_reg, A32::QReg dest);
    void Compile_DestEnable(Instruction instr, A32::QReg src);

    /// PICA-correct multiply: 0 * inf gives 0, NaN inputs still propagate.
    void Compile_SanitizedMul(A32::QReg src1, A32::QReg src2);

    /// Reduces the four lanes of src to their sum, broadcast across all lanes.
    void Compile_HorizontalSum(A32::QReg src);

    void Compile_EvaluateCondition(Instruction instr);
    void Compile_UniformCondition(Instruction instr);
    void Compile_Return();

    /// Calls a `void fn(float*)` helper on SRC1 through a stack buffer.
    void Compile_VectorHelperCall(void (*fn)(f32*));

    void FindReturnOffsets();

    const std::array<u32, MAX_PROGRAM_CODE_LENGTH>* program_code = nullptr;
    const std::array<u32, MAX_SWIZZLE_DATA_LENGTH>* swizzle_data = nullptr;

    std::vector<u32> code_vec;
    std::unique_ptr<A32::Emitter> emitter;

    /// Mapping of PICA VS instructions to offsets in the emitted code
    std::array<A32::Label, MAX_PROGRAM_CODE_LENGTH> instruction_labels;

    /// Labels pointing to the end of each nested LOOP block, for BREAKC.
    std::vector<A32::Label> loop_break_labels;

    /// Offsets in code where a return needs to be inserted
    std::vector<u32> return_offsets;

    u32 program_counter = 0; ///< Offset of the next instruction to decode
    u8 loop_depth = 0;       ///< Depth of the (nested) loops currently compiled

    /// Executable copy of the finished code
    u8* code_mem = nullptr;
    std::size_t code_mem_size = 0;

    using CompiledShader = void(const void* setup, void* state, const void* start_addr);
    CompiledShader* program = nullptr;
};

} // namespace Pica::Shader

#endif // CITRA_ARCH(arm32)
