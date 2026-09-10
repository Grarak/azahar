// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// Compares the shader JIT against the interpreter on this machine. Exists because the A32 JIT is
// new and has no other test coverage; run it on the target after any change to the compiler:
//
//   ./shader_jit_selftest        exits 0 and prints a summary when the backends agree
//
// The interpreter quantizes through f24 after every operation while the JITs (all of them, not
// just A32) compute in f32, so inputs are chosen to be exactly representable in f24 and the
// arithmetic checks demand bit-equality only where the operation is exact in both. RCP/RSQ/EX2/
// LG2 use a relative tolerance plus NaN/inf class matching.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/shader/shader.h"

using namespace Pica;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::SwizzlePattern;

namespace {

int failures = 0;
int checks = 0;

struct ProgramBuilder {
    std::vector<u32> code;
    std::vector<u32> swizzles;

    u32 AddSwizzle(u32 dest_mask, u32 sel1 = 0x1b, bool neg1 = false, u32 sel2 = 0x1b,
                   bool neg2 = false, u32 sel3 = 0x1b, bool neg3 = false) {
        SwizzlePattern p{};
        p.dest_mask.Assign(dest_mask);
        p.negate_src1.Assign(neg1 ? 1 : 0);
        p.negate_src2.Assign(neg2 ? 1 : 0);
        p.negate_src3.Assign(neg3 ? 1 : 0);
        p.src1_selector.Assign(sel1);
        p.src2_selector.Assign(sel2);
        p.src3_selector.Assign(sel3);
        swizzles.push_back(p.hex);
        return static_cast<u32>(swizzles.size() - 1);
    }

    void Common(OpCode::Id op, u32 dest, u32 src1, u32 src2, u32 desc, u32 addr_reg = 0) {
        Instruction i{};
        i.opcode.Assign(op);
        i.common.dest.Assign(dest);
        i.common.src1.Assign(src1);
        i.common.src2.Assign(src2);
        i.common.operand_desc_id.Assign(desc);
        i.common.address_register_index.Assign(addr_reg);
        code.push_back(i.hex);
    }

    void Mad(u32 dest, u32 src1, u32 src2, u32 src3, u32 desc) {
        Instruction i{};
        i.opcode.Assign(OpCode::Id::MAD);
        i.mad.dest.Assign(dest);
        i.mad.src1.Assign(src1);
        i.mad.src2.Assign(src2);
        i.mad.src3.Assign(src3);
        i.mad.operand_desc_id.Assign(desc);
        code.push_back(i.hex);
    }

    void Flow(OpCode::Id op, u32 dest_offset, u32 num_instructions, u32 bool_id = 0,
              u32 int_id = 0, Instruction::FlowControlType::Op cond_op =
                                 Instruction::FlowControlType::JustX,
              bool refx = true, bool refy = true) {
        Instruction i{};
        i.opcode.Assign(op);
        i.flow_control.dest_offset.Assign(dest_offset);
        i.flow_control.num_instructions.Assign(num_instructions);
        i.flow_control.bool_uniform_id.Assign(bool_id);
        i.flow_control.int_uniform_id.Assign(int_id);
        i.flow_control.op.Assign(cond_op);
        i.flow_control.refx.Assign(refx ? 1 : 0);
        i.flow_control.refy.Assign(refy ? 1 : 0);
        code.push_back(i.hex);
    }

    void Cmp(u32 src1, u32 src2, Instruction::Common::CompareOpType::Op opx,
             Instruction::Common::CompareOpType::Op opy, u32 desc) {
        Instruction i{};
        i.opcode.Assign(OpCode::Id::CMP);
        i.common.src1.Assign(src1);
        i.common.src2.Assign(src2);
        i.common.compare_op.x.Assign(opx);
        i.common.compare_op.y.Assign(opy);
        i.common.operand_desc_id.Assign(desc);
        code.push_back(i.hex);
    }

    void End() {
        Instruction i{};
        i.opcode.Assign(OpCode::Id::END);
        code.push_back(i.hex);
    }
};

/// Source register identifiers
constexpr u32 In(u32 i) {
    return i; // v0-v15
}
constexpr u32 Uni(u32 i) {
    return 0x20 + i; // c0-c95
}
/// Destination register identifiers
constexpr u32 Out(u32 i) {
    return i; // o0-o15
}
constexpr u32 Temp(u32 i) {
    return 0x10 + i; // r0-r15
}

struct RunResult {
    std::array<Common::Vec4<f24>, 16> output;
    std::array<s32, 3> address_registers;
    std::array<bool, 2> conditional_code;
};

RunResult RunOn(ShaderEngine& engine, const ProgramBuilder& prog,
                const std::array<Common::Vec4<f24>, 16>& input, ShaderSetup& setup) {
    ProgramCode code_array{};
    SwizzleData swizzle_array{};
    std::copy(prog.code.begin(), prog.code.end(), code_array.begin());
    std::copy(prog.swizzles.begin(), prog.swizzles.end(), swizzle_array.begin());
    setup.UpdateProgramCode(code_array);
    setup.UpdateSwizzleData(swizzle_array);

    engine.SetupBatch(setup, 0);

    ShaderUnit unit{};
    unit.input = input;
    engine.Run(setup, unit);

    RunResult r{};
    r.output = unit.output[0];
    r.address_registers = {unit.address_registers[0], unit.address_registers[1],
                           unit.address_registers[2]};
    r.conditional_code = {unit.conditional_code[0], unit.conditional_code[1]};
    return r;
}

bool FloatsMatch(f32 a, f32 b, bool exact) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) == std::isnan(b);
    }
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    if (exact) {
        u32 ua, ub;
        std::memcpy(&ua, &a, 4);
        std::memcpy(&ub, &b, 4);
        // The interpreter's f24 round-trip can flush tiny values the JIT keeps; both are zero
        // in f24 terms.
        if (std::fabs(a) < 1e-30f && std::fabs(b) < 1e-30f) {
            return true;
        }
        return ua == ub;
    }
    const f32 mag = std::max(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= mag * 1e-4f + 1e-6f;
}

void Compare(const char* what, const RunResult& ref, const RunResult& jit, bool exact,
             u32 out_mask = 1) {
    checks++;
    for (u32 reg = 0; reg < 16; reg++) {
        if (!(out_mask & (1u << reg))) {
            continue;
        }
        for (u32 c = 0; c < 4; c++) {
            const f32 a = ref.output[reg][c].ToFloat32();
            const f32 b = jit.output[reg][c].ToFloat32();
            if (!FloatsMatch(a, b, exact)) {
                std::printf("FAIL %s: o%u.%c interpreter=%g jit=%g\n", what, reg, "xyzw"[c], a, b);
                failures++;
                return;
            }
        }
    }
    for (u32 i = 0; i < 3; i++) {
        if (ref.address_registers[i] != jit.address_registers[i]) {
            std::printf("FAIL %s: a%u interpreter=%d jit=%d\n", what, i, ref.address_registers[i],
                        jit.address_registers[i]);
            failures++;
            return;
        }
    }
    for (u32 i = 0; i < 2; i++) {
        if (ref.conditional_code[i] != jit.conditional_code[i]) {
            std::printf("FAIL %s: cc%u interpreter=%d jit=%d\n", what, i,
                        static_cast<int>(ref.conditional_code[i]),
                        static_cast<int>(jit.conditional_code[i]));
            failures++;
            return;
        }
    }
}

Common::Vec4<f24> V4(f32 x, f32 y, f32 z, f32 w) {
    return {f24::FromFloat32(x), f24::FromFloat32(y), f24::FromFloat32(z), f24::FromFloat32(w)};
}

// f24-exact values plus the special cases the PICA rules care about.
const std::vector<Common::Vec4<f24>> test_vectors = {
    V4(0.0f, 1.0f, -1.0f, 0.5f),
    V4(2.0f, -2.0f, 4.0f, -0.25f),
    V4(123.5f, -77.25f, 1024.0f, 3.0f),
    V4(INFINITY, -INFINITY, 0.0f, 1.0f),
    V4(NAN, 0.0f, INFINITY, -1.0f),
    V4(-0.0f, 0.125f, 8.0f, -128.0f),
    V4(96.0f, 0.0625f, -3.5f, 7.0f),
};

void RunCase(const char* what, ShaderEngine& interp, ShaderEngine& jit,
             const ProgramBuilder& prog, bool exact, u32 out_mask = 1,
             ShaderSetup* shared_setup = nullptr) {
    for (std::size_t a = 0; a < test_vectors.size(); a++) {
        for (std::size_t b = 0; b < test_vectors.size(); b++) {
            std::array<Common::Vec4<f24>, 16> input{};
            input[0] = test_vectors[a];
            input[1] = test_vectors[b];
            input[2] = V4(0.25f, 0.75f, -6.0f, 9.0f);

            ShaderSetup setup_i{}, setup_j{};
            if (shared_setup) {
                setup_i = *shared_setup;
                setup_j = *shared_setup;
            }
            const RunResult r0 = RunOn(interp, prog, input, setup_i);
            const RunResult r1 = RunOn(jit, prog, input, setup_j);
            Compare(what, r0, r1, exact, out_mask);
        }
    }
}

} // namespace

int main() {
    auto interp = CreateEngine(false);
    auto jit = CreateEngine(true);

    const auto simple_binary = [&](const char* name, OpCode::Id op, bool exact) {
        ProgramBuilder p;
        const u32 d = p.AddSwizzle(0xF);
        p.Common(op, Out(0), In(0), In(1), d);
        p.End();
        RunCase(name, *interp, *jit, p, exact);
    };

    simple_binary("ADD", OpCode::Id::ADD, true);
    simple_binary("MUL", OpCode::Id::MUL, true);
    simple_binary("MAX", OpCode::Id::MAX, true);
    simple_binary("MIN", OpCode::Id::MIN, true);
    simple_binary("SGE", OpCode::Id::SGE, true);
    simple_binary("SLT", OpCode::Id::SLT, true);
    simple_binary("DP3", OpCode::Id::DP3, true);
    simple_binary("DP4", OpCode::Id::DP4, true);
    simple_binary("DPH", OpCode::Id::DPH, true);

    { // MOV with a pile of swizzles and negation
        const u32 selectors[] = {0x1b, 0x00, 0x55, 0xaa, 0xff, 0xe4, 0x39, 0x93, 0x6c};
        for (const u32 sel : selectors) {
            for (int neg = 0; neg < 2; neg++) {
                ProgramBuilder p;
                const u32 d = p.AddSwizzle(0xF, sel, neg != 0);
                p.Common(OpCode::Id::MOV, Out(0), In(0), 0, d);
                p.End();
                RunCase("MOV/swizzle", *interp, *jit, p, true);
            }
        }
    }

    { // Every destination mask, applied over a pre-written output
        for (u32 mask = 1; mask < 16; mask++) {
            ProgramBuilder p;
            const u32 dfull = p.AddSwizzle(0xF);
            const u32 dmask = p.AddSwizzle(mask);
            p.Common(OpCode::Id::MOV, Out(0), In(1), 0, dfull);
            p.Common(OpCode::Id::MOV, Out(0), In(0), 0, dmask);
            p.End();
            RunCase("dest-mask", *interp, *jit, p, true);
        }
    }

    { // MAD, both temporaries and outputs
        ProgramBuilder p;
        const u32 d = p.AddSwizzle(0xF);
        p.Mad(Temp(0), In(0), In(1), In(2), d);
        p.Common(OpCode::Id::MOV, Out(0), Temp(0) - 0x10 + 0x10, 0, d); // r0 as src = 0x10
        p.End();
        // Fix the source encoding: temporary as source is 0x10 | index.
        Instruction fix{};
        fix.hex = p.code[1];
        fix.common.src1.Assign(0x10);
        p.code[1] = fix.hex;
        RunCase("MAD", *interp, *jit, p, true);
    }

    { // FLR / RCP / RSQ / EX2 / LG2
        const struct {
            const char* name;
            OpCode::Id op;
            bool exact;
        } unaries[] = {
            {"FLR", OpCode::Id::FLR, true},
            {"RCP", OpCode::Id::RCP, false},
            {"RSQ", OpCode::Id::RSQ, false},
            {"EX2", OpCode::Id::EX2, false},
            {"LG2", OpCode::Id::LG2, false},
        };
        for (const auto& u : unaries) {
            ProgramBuilder p;
            const u32 d = p.AddSwizzle(0xF);
            p.Common(u.op, Out(0), In(0), 0, d);
            p.End();
            RunCase(u.name, *interp, *jit, p, u.exact);
        }
    }

    { // CMP over every comparison pair, plus a dependent IFC on both branches
        using CmpOp = Instruction::Common::CompareOpType::Op;
        const CmpOp ops[] = {CmpOp::Equal,       CmpOp::NotEqual,    CmpOp::LessThan,
                             CmpOp::LessEqual,   CmpOp::GreaterThan, CmpOp::GreaterEqual};
        for (const CmpOp ox : ops) {
            for (const CmpOp oy : ops) {
                ProgramBuilder p;
                const u32 d = p.AddSwizzle(0xF);
                p.Cmp(In(0), In(1), ox, oy, d);
                // IFC cc.x: o0 = i0 on true else i1; then always o0.w-add via second dest reg
                p.Flow(OpCode::Id::IFC, 4, 1); // else-block of 1 instruction starts at 4
                p.Common(OpCode::Id::MOV, Out(0), In(0), 0, d);
                p.Flow(OpCode::Id::JMPU, 5, 0, 15); // never taken (b15 false): plain filler jump
                p.Common(OpCode::Id::MOV, Out(0), In(1), 0, d);
                p.End();
                RunCase("CMP/IFC", *interp, *jit, p, true);
            }
        }
    }

    { // CALL / CALLU / JMPU / IFU driven by bool uniforms
        for (int b0 = 0; b0 < 2; b0++) {
            ProgramBuilder p;
            const u32 d = p.AddSwizzle(0xF);
            // 0: mov o0, i1
            p.Common(OpCode::Id::MOV, Out(0), In(1), 0, d);
            // 1: callu b0 -> 4 (1 instruction)
            p.Flow(OpCode::Id::CALLU, 4, 1, 0);
            // 2: ifu b0 { mov o1, i0 } else { }
            p.Flow(OpCode::Id::IFU, 4, 0, 0);
            // 3: mov o1, i0  (the if body)
            p.Common(OpCode::Id::MOV, Out(1), In(0), 0, d);
            // 4: mov o0, i0  (the subroutine)
            p.Common(OpCode::Id::MOV, Out(0), In(0), 0, d);
            p.End();

            ShaderSetup shared{};
            shared.uniforms.b[0] = b0 != 0;
            RunCase(b0 ? "flow/b0=1" : "flow/b0=0", *interp, *jit, p, true, 0b11, &shared);
        }
    }

    { // LOOP with aL-relative uniform addressing, plus MOVA-relative reads
        ProgramBuilder p;
        const u32 d = p.AddSwizzle(0xF);
        const u32 dx = p.AddSwizzle(0x8); // dest mask x only for MOVA
        // 0: mov r0, c0 (accumulator zero source; c0 = 0)
        Instruction i{};
        p.Common(OpCode::Id::MOV, Temp(0), Uni(0), 0, d);
        // 1: loop i0 { 2: add r0, r0, c[aL + 1] }
        p.Flow(OpCode::Id::LOOP, 2, 0, 0, 0);
        i.hex = 0;
        i.opcode.Assign(OpCode::Id::ADD);
        i.common.dest.Assign(Temp(0));
        i.common.src1.Assign(0x10); // r0
        i.common.src2.Assign(Uni(1));
        i.common.operand_desc_id.Assign(d);
        i.common.address_register_index.Assign(3); // aL on src2
        p.code.push_back(i.hex);
        // 3: mova a0.x, i0
        p.Common(OpCode::Id::MOVA, 0, In(0), 0, dx);
        // 4: mov o0, c[a0 + 2]
        i.hex = 0;
        i.opcode.Assign(OpCode::Id::MOV);
        i.common.dest.Assign(Out(0));
        i.common.src1.Assign(Uni(2));
        i.common.operand_desc_id.Assign(d);
        i.common.address_register_index.Assign(1);
        p.code.push_back(i.hex);
        // 5: mov o1, r0
        i.hex = 0;
        i.opcode.Assign(OpCode::Id::MOV);
        i.common.dest.Assign(Out(1));
        i.common.src1.Assign(0x10);
        i.common.operand_desc_id.Assign(d);
        p.code.push_back(i.hex);
        p.End();

        ShaderSetup shared{};
        for (u32 u = 0; u < 96; u++) {
            shared.uniforms.f[u] = V4(f32(u), f32(u) * 2, -f32(u), 1.0f);
        }
        shared.uniforms.i[0] = {4, 1, 2, 0}; // 5 iterations, aL start 1, step 2
        RunCase("LOOP/aL", *interp, *jit, p, true, 0b11, &shared);
    }

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
