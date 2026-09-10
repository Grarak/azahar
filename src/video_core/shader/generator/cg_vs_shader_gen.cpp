// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/shader/generator/cg_fs_shader_gen.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"

/**
 * A port of glsl_shader_decompiler.cpp and the vertex half of glsl_shader_gen.cpp to the Cg
 * dialect SceShaccCg and psp2cgc accept for sce_vp_psp2. The control-flow analysis is the
 * same algorithm; the emitter differs where the languages do:
 *
 *  - float4/bool2/int3 for vec4/bvec2/ivec3; rsqrt for inversesqrt; a component-wise `?:`
 *    for mix(a, b, bvec) and for the 0/1 vectors of SGE and SLT.
 *  - Cg has no switch, so a subroutine with jump labels is an ascending
 *    `if (jmp_to <= label)` chain. A jump ends its block and moves the target forward, so
 *    the blocks it skips are skipped by their guards and no loop is needed; falling out of
 *    the last block returns. Only a subroutine with a backward jump wraps the chain in a
 *    `while (true)` that the jump restarts - the form the compiler is slowest on.
 *  - Program-scope registers are `static` globals, as Cg requires for non-uniform globals.
 *  - The PICA uniforms are three flat arrays in the default vertex uniform buffer. Bools and
 *    ints are float, for the reason the fragment emitter gives: they are only ever compared
 *    or counted, and an integer that never exists costs nothing to convert.
 *  - Input registers without a vertex loader are uniforms (see VSExtra::default_regs).
 *  - sanitize_mul is not offered. It exists for the NaN semantics of PICA's multiplier and
 *    needs isnan, which this compiler's profile does not promise; the Vita build runs with
 *    accurate multiplication off.
 */
namespace Pica::Shader::Generator::Cg {

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;
using VSOutputAttributes = Pica::RasterizerRegs::VSOutputAttributes;

namespace {

constexpr u32 PROGRAM_END = MAX_PROGRAM_CODE_LENGTH;

// Bisect switches (hwvs_nomova, hwvs_noloop, hwvs_nobranch): a program using the named
// construct is refused, so the software shader takes its draws and a wrong picture can be
// pinned on the construct.
u32 vs_refuse_mask = 0;

class DecompileFail : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void RefuseIf(u32 bit, const char* what) {
    if ((vs_refuse_mask & bit) != 0) {
        throw DecompileFail(std::string("refused by bisect switch: ") + what);
    }
}

enum class ExitMethod {
    Undetermined,
    AlwaysReturn,
    Conditional,
    AlwaysEnd,
};

struct Subroutine {
    std::string GetName() const {
        return "sub_" + std::to_string(begin) + "_" + std::to_string(end);
    }

    u32 begin;
    u32 end;
    ExitMethod exit_method;
    std::set<u32> labels;

    bool operator<(const Subroutine& rhs) const {
        return std::tie(begin, end) < std::tie(rhs.begin, rhs.end);
    }
};

/// A bool uniform's value when it is baked into the program, or nothing when it stays a
/// uniform.
struct KnownBools {
    u16 values = 0;
    u16 mask = 0;
    std::optional<bool> Get(u32 id) const {
        if (id < 16 && (mask & (1u << id)) != 0) {
            return (values & (1u << id)) != 0;
        }
        return std::nullopt;
    }
};

class ControlFlowAnalyzer {
public:
    ControlFlowAnalyzer(const ProgramCode& program_code, u32 main_offset, KnownBools bools)
        : program_code(program_code), bools(bools) {
        const Subroutine& program_main = AddSubroutine(main_offset, PROGRAM_END);
        if (program_main.exit_method != ExitMethod::AlwaysEnd) {
            throw DecompileFail("Program does not always end");
        }
    }

    std::set<Subroutine> MoveSubroutines() {
        return std::move(subroutines);
    }

    /// Ranges in which a jump goes backwards, which the linear emission cannot express.
    std::set<std::pair<u32, u32>> MoveBackwardJumps() {
        return std::move(backward_jumps);
    }

    /// Every instruction the scan visited. A block that starts at an offset the scan never
    /// reached is dead code - what follows a jump resolved as always taken - and the emitter
    /// leaves it out, which it must, since the subroutines it would call were never made.
    std::vector<bool> MoveReachable() {
        return std::move(reachable);
    }

private:
    const ProgramCode& program_code;
    const KnownBools bools;
    std::vector<bool> reachable = std::vector<bool>(PROGRAM_END, false);
    std::set<Subroutine> subroutines;
    std::map<std::pair<u32, u32>, ExitMethod> exit_method_map;
    std::set<std::pair<u32, u32>> backward_jumps;

    const Subroutine& AddSubroutine(u32 begin, u32 end) {
        auto iter = subroutines.find(Subroutine{begin, end});
        if (iter != subroutines.end()) {
            return *iter;
        }
        Subroutine subroutine{begin, end};
        subroutine.exit_method = Scan(begin, end, subroutine.labels);
        if (subroutine.exit_method == ExitMethod::Undetermined) {
            throw DecompileFail("Recursive function detected");
        }
        return *subroutines.insert(std::move(subroutine)).first;
    }

    static ExitMethod ParallelExit(ExitMethod a, ExitMethod b) {
        if (a == ExitMethod::Undetermined) {
            return b;
        }
        if (b == ExitMethod::Undetermined) {
            return a;
        }
        if (a == b) {
            return a;
        }
        return ExitMethod::Conditional;
    }

    static ExitMethod SeriesExit(ExitMethod a, ExitMethod b) {
        DEBUG_ASSERT(a != ExitMethod::AlwaysEnd);
        if (a == ExitMethod::Undetermined) {
            return ExitMethod::Undetermined;
        }
        if (a == ExitMethod::AlwaysReturn) {
            return b;
        }
        if (b == ExitMethod::Undetermined || b == ExitMethod::AlwaysEnd) {
            return ExitMethod::AlwaysEnd;
        }
        return ExitMethod::Conditional;
    }

    ExitMethod Scan(u32 begin, u32 end, std::set<u32>& labels) {
        auto [iter, inserted] =
            exit_method_map.emplace(std::make_pair(begin, end), ExitMethod::Undetermined);
        ExitMethod& exit_method = iter->second;
        if (!inserted) {
            return exit_method;
        }
        for (u32 offset = begin; offset != end && offset != PROGRAM_END; ++offset) {
            reachable[offset] = true;
            const Instruction instr = {program_code[offset]};
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                return exit_method = ExitMethod::AlwaysEnd;
            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                std::optional<bool> taken;
                if (instr.opcode.Value() == OpCode::Id::JMPU) {
                    if (const auto known = bools.Get(instr.flow_control.bool_uniform_id)) {
                        taken = *known != ((instr.flow_control.num_instructions & 1) != 0);
                    }
                }
                if (taken.has_value() && !*taken) {
                    break; // a NOP
                }
                labels.insert(instr.flow_control.dest_offset);
                // The jump ends its block, so that taking it skips nothing but whole blocks.
                if (offset + 1 < end) {
                    labels.insert(offset + 1);
                }
                if (instr.flow_control.dest_offset <= offset) {
                    backward_jumps.insert(std::make_pair(begin, end));
                }
                const ExitMethod jmp = Scan(instr.flow_control.dest_offset, end, labels);
                if (taken.has_value()) {
                    return exit_method = jmp;
                }
                const ExitMethod no_jmp = Scan(offset + 1, end, labels);
                return exit_method = ParallelExit(no_jmp, jmp);
            }
            case OpCode::Id::CALLU: {
                const auto known = bools.Get(instr.flow_control.bool_uniform_id);
                if (known.has_value() && !*known) {
                    break; // a NOP
                }
                if (!known.has_value()) {
                    auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                               instr.flow_control.dest_offset +
                                                   instr.flow_control.num_instructions);
                    const ExitMethod after_call = Scan(offset + 1, end, labels);
                    return exit_method = SeriesExit(
                               ParallelExit(call.exit_method, ExitMethod::AlwaysReturn),
                               after_call);
                }
                [[fallthrough]]; // known taken: a CALL
            }
            case OpCode::Id::CALL: {
                auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                           instr.flow_control.dest_offset +
                                               instr.flow_control.num_instructions);
                if (call.exit_method == ExitMethod::AlwaysEnd) {
                    return exit_method = ExitMethod::AlwaysEnd;
                }
                const ExitMethod after_call = Scan(offset + 1, end, labels);
                return exit_method = SeriesExit(call.exit_method, after_call);
            }
            case OpCode::Id::LOOP: {
                auto& loop = AddSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
                if (loop.exit_method == ExitMethod::AlwaysEnd) {
                    return exit_method = ExitMethod::AlwaysEnd;
                }
                const ExitMethod after_loop =
                    Scan(instr.flow_control.dest_offset + 1, end, labels);
                return exit_method = SeriesExit(loop.exit_method, after_loop);
            }
            case OpCode::Id::CALLC: {
                auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                           instr.flow_control.dest_offset +
                                               instr.flow_control.num_instructions);
                const ExitMethod after_call = Scan(offset + 1, end, labels);
                return exit_method = SeriesExit(
                           ParallelExit(call.exit_method, ExitMethod::AlwaysReturn), after_call);
            }
            case OpCode::Id::IFU:
            case OpCode::Id::IFC: {
                const u32 endif_offset =
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions;
                std::optional<bool> known;
                if (instr.opcode.Value() == OpCode::Id::IFU) {
                    known = bools.Get(instr.flow_control.bool_uniform_id);
                }
                if (known.has_value()) {
                    // One side only: the other is never analysed, never emitted.
                    ExitMethod side = ExitMethod::AlwaysReturn;
                    if (*known) {
                        side = AddSubroutine(offset + 1, instr.flow_control.dest_offset)
                                   .exit_method;
                    } else if (instr.flow_control.num_instructions != 0) {
                        side = AddSubroutine(instr.flow_control.dest_offset, endif_offset)
                                   .exit_method;
                    }
                    if (side == ExitMethod::AlwaysEnd) {
                        return exit_method = ExitMethod::AlwaysEnd;
                    }
                    const ExitMethod after = Scan(endif_offset, end, labels);
                    return exit_method = SeriesExit(side, after);
                }
                auto& if_sub = AddSubroutine(offset + 1, instr.flow_control.dest_offset);
                ExitMethod else_method;
                if (instr.flow_control.num_instructions != 0) {
                    auto& else_sub = AddSubroutine(instr.flow_control.dest_offset,
                                                   instr.flow_control.dest_offset +
                                                       instr.flow_control.num_instructions);
                    else_method = else_sub.exit_method;
                } else {
                    else_method = ExitMethod::AlwaysReturn;
                }
                const ExitMethod both = ParallelExit(if_sub.exit_method, else_method);
                if (both == ExitMethod::AlwaysEnd) {
                    return exit_method = ExitMethod::AlwaysEnd;
                }
                const ExitMethod after_call =
                    Scan(instr.flow_control.dest_offset + instr.flow_control.num_instructions, end,
                         labels);
                return exit_method = SeriesExit(both, after_call);
            }
            default:
                break;
            }
        }
        return exit_method = ExitMethod::AlwaysReturn;
    }
};

class ShaderWriter {
public:
    template <typename... Args>
    void AddLine(fmt::format_string<Args...> text, Args&&... args) {
        AddExpression(fmt::format(text, std::forward<Args>(args)...));
        AddNewLine();
    }

    void AddNewLine() {
        DEBUG_ASSERT(scope >= 0);
        shader_source += '\n';
    }

    std::string MoveResult() {
        return std::move(shader_source);
    }

    int scope = 0;

private:
    void AddExpression(std::string_view text) {
        if (!text.empty()) {
            shader_source.append(static_cast<std::size_t>(scope) * 4, ' ');
        }
        shader_source += text;
    }

    std::string shader_source;
};

template <SwizzlePattern::Selector (SwizzlePattern::*getter)(int) const>
std::string GetSelectorSrc(const SwizzlePattern& pattern) {
    std::string out;
    for (int i = 0; i < 4; ++i) {
        switch ((pattern.*getter)(i)) {
        case SwizzlePattern::Selector::x:
            out += 'x';
            break;
        case SwizzlePattern::Selector::y:
            out += 'y';
            break;
        case SwizzlePattern::Selector::z:
            out += 'z';
            break;
        case SwizzlePattern::Selector::w:
            out += 'w';
            break;
        default:
            UNREACHABLE();
            return "";
        }
    }
    return out;
}

constexpr auto GetSelectorSrc1 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc1>;
constexpr auto GetSelectorSrc2 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc2>;
constexpr auto GetSelectorSrc3 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc3>;

/// Broadcasts a scalar expression to `n` components the way Cg allows: a scalar swizzle.
std::string Broadcast(std::string_view value, u32 n) {
    return fmt::format("({}).{}", value, std::string_view{"xxxx"}.substr(0, n));
}

class CgGenerator {
public:
    CgGenerator(const std::set<Subroutine>& subroutines,
                const std::set<std::pair<u32, u32>>& backward_jumps,
                const std::vector<bool>& reachable, const ProgramCode& program_code,
                const SwizzleData& swizzle_data, u32 main_offset,
                const std::array<u32, 16>& output_map, u32 num_outputs, KnownBools bools,
                u32& input_regs)
        : subroutines(subroutines), backward_jumps(backward_jumps), reachable(reachable),
          program_code(program_code), swizzle_data(swizzle_data), main_offset(main_offset),
          output_map(output_map), num_outputs(num_outputs), bools(bools),
          input_regs(input_regs) {
        Generate();
    }

    std::string MoveShaderCode() {
        return shader.MoveResult();
    }

private:
    const Subroutine& GetSubroutine(u32 begin, u32 end) const {
        auto iter = subroutines.find(Subroutine{begin, end});
        ASSERT(iter != subroutines.end());
        return *iter;
    }

    static std::string EvaluateCondition(Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;
        const std::string_view result_x =
            flow_control.refx.Value() ? "conditional_code.x" : "!conditional_code.x";
        const std::string_view result_y =
            flow_control.refy.Value() ? "conditional_code.y" : "!conditional_code.y";
        switch (flow_control.op) {
        case Op::JustX:
            return std::string(result_x);
        case Op::JustY:
            return std::string(result_y);
        case Op::Or:
            return fmt::format("({} || {})", result_x, result_y);
        case Op::And:
            return fmt::format("({} && {})", result_x, result_y);
        default:
            UNREACHABLE();
            return "";
        }
    }

    std::string InputRegister(u32 index) {
        ASSERT(index < 16);
        input_regs |= 1u << index;
        return fmt::format("vs_in_reg{}", index);
    }

    std::string OutputRegister(u32 index) const {
        ASSERT(index < 16);
        if (output_map[index] < num_outputs) {
            return fmt::format("vs_out_attr{}", output_map[index]);
        }
        return "";
    }

    std::string GetSourceRegister(const SourceRegister& source_reg, u32 address_register_index) {
        const u32 index = static_cast<u32>(source_reg.GetIndex());
        switch (source_reg.GetRegisterType()) {
        case RegisterType::Input:
            return InputRegister(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        case RegisterType::FloatUniform:
            if (address_register_index != 0) {
                // The PICA rule, (base + offset) & 0x7F, with entries 96..127 of the bank
                // holding ones so the "past the real uniforms reads as one" case is data,
                // not code. Written as the mask because it is the one form the compiler
                // handles cheaply: a helper with conditionals tripled the compile time of
                // every program that uses the address register, once per use.
                return fmt::format("uniforms_f[({} + address_registers.{}) & 127]", index,
                                   "xyz"[address_register_index - 1]);
            }
            return fmt::format("uniforms_f[{}]", index);
        default:
            UNREACHABLE();
            return "";
        }
    }

    std::string GetDestRegister(const DestRegister& dest_reg) const {
        const u32 index = static_cast<u32>(dest_reg.GetIndex());
        switch (dest_reg.GetRegisterType()) {
        case RegisterType::Output:
            return OutputRegister(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        default:
            UNREACHABLE();
            return "";
        }
    }

    static std::string GetUniformBool(u32 index, bool invert_test = false) {
        return fmt::format("(uniforms_b[{}] {} 0.5)", index, invert_test ? "<" : ">");
    }

    void CallSubroutine(const Subroutine& subroutine) {
        if (subroutine.exit_method == ExitMethod::AlwaysEnd) {
            shader.AddLine("{}();", subroutine.GetName());
            shader.AddLine("return true;");
        } else if (subroutine.exit_method == ExitMethod::Conditional) {
            shader.AddLine("if ({}()) {{ return true; }}", subroutine.GetName());
        } else {
            shader.AddLine("{}();", subroutine.GetName());
        }
    }

    void SetDest(const SwizzlePattern& swizzle, std::string_view reg, std::string_view value,
                 u32 dest_num_components, u32 value_num_components) {
        u32 dest_mask_num_components = 0;
        std::string dest_mask_swizzle = ".";
        for (u32 i = 0; i < dest_num_components; ++i) {
            if (swizzle.DestComponentEnabled(static_cast<int>(i))) {
                dest_mask_swizzle += "xyzw"[i];
                ++dest_mask_num_components;
            }
        }
        if (reg.empty() || dest_mask_num_components == 0) {
            return;
        }
        DEBUG_ASSERT(value_num_components >= dest_num_components || value_num_components == 1);

        const std::string dest =
            fmt::format("{}{}", reg, dest_num_components != 1 ? dest_mask_swizzle : "");
        std::string src{value};
        if (value_num_components == 1) {
            if (dest_mask_num_components != 1) {
                src = Broadcast(value, dest_mask_num_components);
            }
        } else if (value_num_components != dest_mask_num_components) {
            src = fmt::format("({}){}", value, dest_mask_swizzle);
        }
        shader.AddLine("{} = {};", dest, src);
    }

    /// A per-component 0/1 vector from a per-component comparison.
    static std::string Select01(std::string_view condition) {
        return fmt::format("(({}) ? float4(1.0, 1.0, 1.0, 1.0) : float4(0.0, 0.0, 0.0, 0.0))",
                           condition);
    }

    u32 CompileInstr(u32 offset) {
        const Instruction instr = {program_code[offset]};
        const std::size_t swizzle_offset =
            instr.opcode.Value().GetInfo().type == OpCode::Type::MultiplyAdd
                ? instr.mad.operand_desc_id
                : instr.common.operand_desc_id;
        const SwizzlePattern swizzle = {swizzle_data[swizzle_offset]};


        switch (instr.opcode.Value().GetInfo().type) {
        case OpCode::Type::Arithmetic: {
            const bool is_inverted =
                (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

            std::string src1 = swizzle.negate_src1 ? "-" : "";
            src1 += GetSourceRegister(instr.common.GetSrc1(is_inverted),
                                      !is_inverted * instr.common.address_register_index);
            src1 += "." + GetSelectorSrc1(swizzle);

            std::string src2 = swizzle.negate_src2 ? "-" : "";
            src2 += GetSourceRegister(instr.common.GetSrc2(is_inverted),
                                      is_inverted * instr.common.address_register_index);
            src2 += "." + GetSelectorSrc2(swizzle);

            const std::string dest_reg = GetDestRegister(instr.common.dest.Value());

            switch (instr.opcode.Value().EffectiveOpCode()) {
            case OpCode::Id::ADD:
                SetDest(swizzle, dest_reg, fmt::format("{} + {}", src1, src2), 4, 4);
                break;
            case OpCode::Id::MUL:
                SetDest(swizzle, dest_reg, fmt::format("{} * {}", src1, src2), 4, 4);
                break;
            case OpCode::Id::FLR:
                SetDest(swizzle, dest_reg, fmt::format("floor({})", src1), 4, 4);
                break;
            case OpCode::Id::MAX:
                SetDest(swizzle, dest_reg, fmt::format("max({}, {})", src1, src2), 4, 4);
                break;
            case OpCode::Id::MIN:
                SetDest(swizzle, dest_reg, fmt::format("min({}, {})", src1, src2), 4, 4);
                break;
            case OpCode::Id::DP3:
                SetDest(swizzle, dest_reg, fmt::format("dot(({}).xyz, ({}).xyz)", src1, src2), 4,
                        1);
                break;
            case OpCode::Id::DP4:
                SetDest(swizzle, dest_reg, fmt::format("dot({}, {})", src1, src2), 4, 1);
                break;
            case OpCode::Id::DPH:
            case OpCode::Id::DPHI:
                SetDest(swizzle, dest_reg,
                        fmt::format("dot(float4(({}).xyz, 1.0), {})", src1, src2), 4, 1);
                break;
            case OpCode::Id::RCP:
                // The NaN guard the GLSL emitter uses without sanitize_mul (Ocarina of Time).
                shader.AddLine("if (({}).x != 0.0)", src1);
                SetDest(swizzle, dest_reg, fmt::format("(1.0 / ({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::RSQ:
                shader.AddLine("if (({}).x > 0.0)", src1);
                SetDest(swizzle, dest_reg, fmt::format("rsqrt(({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::MOVA: {
                RefuseIf(VsRefuseMova, "mova");
                // The offset range check the GLSL helper applies per use, applied here once:
                // a value outside the 8-bit offset range addresses as zero.
                // short throughout the integer side: the values are uniform indices and loop
                // counts, all below 256, and the compiler warns that a full float-to-int
                // conversion is expensive where a 16-bit one would do.
                const std::string mova = fmt::format("mova{}", offset);
                shader.AddLine("short2 {} = short2(({}).xy);", mova, src1);
                SetDest(swizzle, "address_registers",
                        fmt::format("(({0} >= short2(-128, -128) && {0} <= short2(127, 127)) ? "
                                    "{0} : short2(0, 0))",
                                    mova),
                        2, 2);
                break;
            }
            case OpCode::Id::MOV:
                SetDest(swizzle, dest_reg, src1, 4, 4);
                break;
            case OpCode::Id::SGE:
            case OpCode::Id::SGEI:
                SetDest(swizzle, dest_reg, Select01(fmt::format("{} >= {}", src1, src2)), 4, 4);
                break;
            case OpCode::Id::SLT:
            case OpCode::Id::SLTI:
                SetDest(swizzle, dest_reg, Select01(fmt::format("{} < {}", src1, src2)), 4, 4);
                break;
            case OpCode::Id::CMP: {
                using CompareOp = Instruction::Common::CompareOpType::Op;
                const std::map<CompareOp, std::string_view> cmp_ops{
                    {CompareOp::Equal, "=="},        {CompareOp::NotEqual, "!="},
                    {CompareOp::LessThan, "<"},      {CompareOp::LessEqual, "<="},
                    {CompareOp::GreaterThan, ">"},   {CompareOp::GreaterEqual, ">="},
                };
                const CompareOp op_x = instr.common.compare_op.x.Value();
                const CompareOp op_y = instr.common.compare_op.y.Value();
                if (cmp_ops.find(op_x) == cmp_ops.end()) {
                    LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", op_x);
                } else if (cmp_ops.find(op_y) == cmp_ops.end()) {
                    LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", op_y);
                } else {
                    shader.AddLine("conditional_code.x = ({}).x {} ({}).x;", src1,
                                   cmp_ops.find(op_x)->second, src2);
                    shader.AddLine("conditional_code.y = ({}).y {} ({}).y;", src1,
                                   cmp_ops.find(op_y)->second, src2);
                }
                break;
            }
            case OpCode::Id::EX2:
                SetDest(swizzle, dest_reg, fmt::format("exp2(({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::LG2:
                SetDest(swizzle, dest_reg, fmt::format("log2(({}).x)", src1), 4, 1);
                break;
            default:
                LOG_ERROR(HW_GPU, "Unhandled arithmetic instruction: 0x{:02x} ({}): 0x{:08x}",
                          static_cast<int>(instr.opcode.Value().EffectiveOpCode()),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
            }
            break;
        }

        case OpCode::Type::MultiplyAdd: {
            if (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD ||
                instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI) {
                const bool is_inverted =
                    instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI;

                std::string src1 = swizzle.negate_src1 ? "-" : "";
                src1 += GetSourceRegister(instr.mad.GetSrc1(is_inverted), 0);
                src1 += "." + GetSelectorSrc1(swizzle);

                std::string src2 = swizzle.negate_src2 ? "-" : "";
                src2 += GetSourceRegister(instr.mad.GetSrc2(is_inverted),
                                          !is_inverted * instr.mad.address_register_index);
                src2 += "." + GetSelectorSrc2(swizzle);

                std::string src3 = swizzle.negate_src3 ? "-" : "";
                src3 += GetSourceRegister(instr.mad.GetSrc3(is_inverted),
                                          is_inverted * instr.mad.address_register_index);
                src3 += "." + GetSelectorSrc3(swizzle);

                const std::string dest_reg =
                    (instr.mad.dest.Value() < 0x10)
                        ? OutputRegister(static_cast<u32>(instr.mad.dest.Value().GetIndex()))
                    : (instr.mad.dest.Value() < 0x20)
                        ? "reg_tmp" + std::to_string(instr.mad.dest.Value().GetIndex())
                        : "";
                SetDest(swizzle, dest_reg, fmt::format("{} * {} + {}", src1, src2, src3), 4, 4);
            } else {
                LOG_ERROR(HW_GPU, "Unhandled multiply-add instruction: 0x{:02x} ({}): 0x{:08x}",
                          static_cast<int>(instr.opcode.Value().EffectiveOpCode()),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
            }
            break;
        }

        default: {
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                shader.AddLine("return true;");
                offset = PROGRAM_END - 1;
                break;

            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                RefuseIf(VsRefuseBranch, "jump");
                std::string condition;
                if (instr.opcode.Value() == OpCode::Id::JMPC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else {
                    const bool invert_test = instr.flow_control.num_instructions & 1;
                    if (const auto known = bools.Get(instr.flow_control.bool_uniform_id)) {
                        if (*known == invert_test) {
                            break; // never taken
                        }
                        condition = "true";
                    } else {
                        condition =
                            GetUniformBool(instr.flow_control.bool_uniform_id, invert_test);
                    }
                }
                // The jump is the last instruction of its block (Scan made the next one a
                // label), so setting the target is enough: every block up to it is skipped by
                // its own guard. Only a backward jump needs the chain restarted.
                shader.AddLine("if ({}) {{ jmp_to = {};{} }}", condition,
                               instr.flow_control.dest_offset.Value(),
                               loop_form ? " continue;" : "");
                break;
            }

            case OpCode::Id::CALL:
            case OpCode::Id::CALLC:
            case OpCode::Id::CALLU: {
                if (instr.opcode.Value() != OpCode::Id::CALL) {
                    RefuseIf(VsRefuseBranch, "conditional call");
                }
                std::string condition;
                bool always = instr.opcode.Value() == OpCode::Id::CALL;
                if (instr.opcode.Value() == OpCode::Id::CALLC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else if (instr.opcode.Value() == OpCode::Id::CALLU) {
                    if (const auto known = bools.Get(instr.flow_control.bool_uniform_id)) {
                        if (!*known) {
                            break; // never called
                        }
                        always = true;
                    } else {
                        condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                    }
                }
                if (condition.empty()) {
                    shader.AddLine("{{");
                } else {
                    shader.AddLine("if ({}) {{", condition);
                }
                ++shader.scope;
                auto& call_sub = GetSubroutine(instr.flow_control.dest_offset,
                                               instr.flow_control.dest_offset +
                                                   instr.flow_control.num_instructions);
                CallSubroutine(call_sub);
                if (always && call_sub.exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }
                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::NOP:
                break;

            case OpCode::Id::IFC:
            case OpCode::Id::IFU: {
                RefuseIf(VsRefuseBranch, "if");
                std::string condition;
                if (instr.opcode.Value() == OpCode::Id::IFC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else {
                    condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                }
                const u32 if_offset = offset + 1;
                const u32 else_offset = instr.flow_control.dest_offset;
                const u32 endif_offset =
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions;

                if (instr.opcode.Value() == OpCode::Id::IFU) {
                    if (const auto known = bools.Get(instr.flow_control.bool_uniform_id)) {
                        // Resolved at generation: the taken side inline, the other gone.
                        const Subroutine* side = nullptr;
                        if (*known) {
                            side = &GetSubroutine(if_offset, else_offset);
                        } else if (instr.flow_control.num_instructions != 0) {
                            side = &GetSubroutine(else_offset, endif_offset);
                        }
                        if (side != nullptr) {
                            shader.AddLine("{{");
                            ++shader.scope;
                            CallSubroutine(*side);
                            --shader.scope;
                            shader.AddLine("}}");
                        }
                        offset = (side != nullptr && side->exit_method == ExitMethod::AlwaysEnd)
                                     ? PROGRAM_END - 1
                                     : endif_offset - 1;
                        break;
                    }
                }

                shader.AddLine("if ({}) {{", condition);
                ++shader.scope;
                auto& if_sub = GetSubroutine(if_offset, else_offset);
                CallSubroutine(if_sub);
                offset = else_offset - 1;

                if (instr.flow_control.num_instructions != 0) {
                    --shader.scope;
                    shader.AddLine("}} else {{");
                    ++shader.scope;
                    auto& else_sub = GetSubroutine(else_offset, endif_offset);
                    CallSubroutine(else_sub);
                    offset = endif_offset - 1;
                    if (if_sub.exit_method == ExitMethod::AlwaysEnd &&
                        else_sub.exit_method == ExitMethod::AlwaysEnd) {
                        offset = PROGRAM_END - 1;
                    }
                }
                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::LOOP: {
                RefuseIf(VsRefuseLoop, "loop");
                const std::string int_uniform =
                    fmt::format("uniforms_i[{}]", instr.flow_control.int_uniform_id.Value());
                shader.AddLine("address_registers.z = short({}.y);", int_uniform);
                const std::string loop_var = fmt::format("loop{}", offset);
                shader.AddLine("for (short {0} = 0; {0} <= short({1}.x); address_registers.z += "
                               "short({1}.z), ++{0}) {{",
                               loop_var, int_uniform);
                ++shader.scope;
                auto& loop_sub = GetSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
                CallSubroutine(loop_sub);
                offset = instr.flow_control.dest_offset;
                --shader.scope;
                shader.AddLine("}}");
                if (loop_sub.exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }
                break;
            }

            case OpCode::Id::EMIT:
            case OpCode::Id::SETEMIT:
                LOG_ERROR(HW_GPU, "Geometry shader operation detected in vertex shader");
                break;

            default:
                LOG_ERROR(HW_GPU, "Unhandled instruction: 0x{:02x} ({}): 0x{:08x}",
                          static_cast<int>(instr.opcode.Value().EffectiveOpCode()),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
            }
            break;
        }
        }
        return offset + 1;
    }

    u32 CompileRange(u32 begin, u32 end) {
        u32 program_counter;
        for (program_counter = begin; program_counter < (begin > end ? PROGRAM_END : end);) {
            program_counter = CompileInstr(program_counter);
        }
        return program_counter;
    }

    void Generate() {
        shader.AddLine("static bool2 conditional_code = bool2(false, false);");
        shader.AddLine("static short3 address_registers = short3(0, 0, 0);");
        for (int i = 0; i < 16; ++i) {
            shader.AddLine("static float4 reg_tmp{} = float4(0.0, 0.0, 0.0, 1.0);", i);
        }
        shader.AddNewLine();

        for (const auto& subroutine : subroutines) {
            shader.AddLine("bool {}();", subroutine.GetName());
        }
        shader.AddNewLine();

        shader.AddLine("bool exec_shader() {{");
        ++shader.scope;
        CallSubroutine(GetSubroutine(main_offset, PROGRAM_END));
        --shader.scope;
        shader.AddLine("}}\n");

        for (const auto& subroutine : subroutines) {
            std::set<u32> labels = subroutine.labels;

            shader.AddLine("bool {}() {{", subroutine.GetName());
            ++shader.scope;

            if (labels.empty()) {
                if (CompileRange(subroutine.begin, subroutine.end) != PROGRAM_END) {
                    shader.AddLine("return false;");
                }
            } else {
                labels.insert(subroutine.begin);
                loop_form = HasBackwardJump(subroutine);
                shader.AddLine("short jmp_to = {};", subroutine.begin);
                if (loop_form) {
                    shader.AddLine("while (true) {{");
                    ++shader.scope;
                }

                // Blocks in address order, each guarded by "the target is at or before me",
                // which is the switch's fall-through: a pass that enters at label L runs L and
                // everything after it until the block returns. A forward jump ends its block
                // and moves the target ahead, so the blocks between are skipped by their
                // guards and nothing loops; only a backward jump restarts the chain, which is
                // the form the compiler is slowest on and the form PICA programs rarely need.
                for (auto it = labels.begin(); it != labels.end(); ++it) {
                    const u32 label = *it;
                    if (label < PROGRAM_END && !reachable[label]) {
                        continue; // dead: behind a jump resolved as always taken
                    }
                    shader.AddLine("if (jmp_to <= {}) {{", label);
                    ++shader.scope;
                    auto next_it = labels.lower_bound(label + 1);
                    const u32 next_label = next_it == labels.end() ? subroutine.end : *next_it;
                    const u32 compile_end = CompileRange(label, next_label);
                    if (compile_end > next_label && compile_end != PROGRAM_END) {
                        // A label inside an IF/LOOP block: resume after it.
                        shader.AddLine("jmp_to = {};{}", compile_end, loop_form ? " continue;" : "");
                        labels.emplace(compile_end);
                    }
                    --shader.scope;
                    shader.AddLine("}}");
                }
                if (loop_form) {
                    // The chain ran out: the subroutine reached its end.
                    shader.AddLine("return false;");
                    --shader.scope;
                    shader.AddLine("}}");
                }
                shader.AddLine("return false;");
                loop_form = false;
            }

            --shader.scope;
            shader.AddLine("}}\n");
            DEBUG_ASSERT(shader.scope == 0);
        }
    }

    /// Whether any jump inside the subroutine goes backwards. Scan records the range it
    /// was walking when it met one; that range ends where the subroutine ends and starts
    /// at or after the subroutine's own start.
    bool HasBackwardJump(const Subroutine& subroutine) const {
        for (const auto& [begin, end] : backward_jumps) {
            if (end == subroutine.end && begin >= subroutine.begin) {
                return true;
            }
        }
        return false;
    }

    const std::set<Subroutine>& subroutines;
    const std::set<std::pair<u32, u32>>& backward_jumps;
    const std::vector<bool>& reachable;
    const ProgramCode& program_code;
    const SwizzleData& swizzle_data;
    const u32 main_offset;
    const std::array<u32, 16>& output_map;
    const u32 num_outputs;
    const KnownBools bools;
    u32& input_regs;
    /// Set while a subroutine is emitted in the loop form, where a jump restarts the chain.
    bool loop_form = false;
    ShaderWriter shader;
};

std::string DecompileProgram(const ProgramCode& program_code, const SwizzleData& swizzle_data,
                             u32 main_offset, const std::array<u32, 16>& output_map,
                             u32 num_outputs, KnownBools bools, u32& input_regs) {
    try {
        ControlFlowAnalyzer analyzer(program_code, main_offset, bools);
        const auto subroutines = analyzer.MoveSubroutines();
        const auto backward_jumps = analyzer.MoveBackwardJumps();
        const auto reachable = analyzer.MoveReachable();
        CgGenerator generator(subroutines, backward_jumps, reachable, program_code,
                              swizzle_data, main_offset, output_map, num_outputs, bools,
                              input_regs);
        return generator.MoveShaderCode();
    } catch (const DecompileFail& exception) {
        LOG_INFO(HW_GPU, "Cg vertex shader decompilation failed: {}", exception.what());
        return "";
    }
}

} // Anonymous namespace

u16 WrittenOutputRegisters(const ShaderSetup& setup);

void SetVsRefuse(u32 mask) {
    vs_refuse_mask = mask;
}

std::string GenerateVertexShader(const ShaderSetup& setup, const PicaVSConfig& config,
                                 const VSExtra& extra, u32& input_regs) {
    const auto& state = config.state;
    input_regs = 0;
    const KnownBools bools{extra.bools, extra.bool_mask};
    const std::string program_source =
        DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(), state.main_offset,
                         state.output_map, state.num_outputs, bools, input_regs);
    if (program_source.empty()) {
        return "";
    }

    std::string out;
    out.reserve(16 * 1024 + program_source.size());
    // What this variant was built for, so a dumped program can be reproduced by hand.
    out += fmt::format("// bools {:#06x} of {:#06x}, default regs {:#06x}, components", extra.bools & extra.bool_mask,
                       extra.bool_mask, extra.default_regs);
    for (u32 i = 0; i < 16; i++) {
        out += fmt::format(" {}", extra.reg_components[i]);
    }
    out += '\n';

    // The PICA uniforms. Bools and ints are float: see the file comment.
    out += "uniform float4 uniforms_f[128];\n"
           "uniform float4 uniforms_i[4];\n";
    if (program_source.find("uniforms_b[") != std::string::npos) {
        out += "uniform float uniforms_b[16];\n";
    }
    out += "uniform float flip_viewport;\n"
           "uniform float depth_scale;\n"
           "uniform float depth_offset;\n";
    for (u32 i = 0; i < 16; ++i) {
        if ((input_regs & (1u << i)) != 0 && (extra.default_regs & (1u << i)) != 0) {
            out += fmt::format("uniform float4 vs_in_reg{};\n", i);
        }
    }
    out += '\n';
    for (u32 i = 0; i < state.num_outputs; ++i) {
        out += fmt::format("static float4 vs_out_attr{};\n", i);
    }
    // The input registers a loader feeds arrive as main's parameters; the program reads them
    // through these, which the entry point copies in. Registers with no loader are the
    // uniforms declared above.
    for (u32 i = 0; i < 16; ++i) {
        if ((input_regs & (1u << i)) != 0 && (extra.default_regs & (1u << i)) == 0) {
            out += fmt::format("static float4 vs_in_reg{};\n", i);
        }
    }
    out += '\n';
    out += program_source;

    // The fragment interface, by semantic, exactly as the fixed pass-through program emits
    // it, and the same depth mapping: PICA depth into clip z so the hardware's interpolated
    // depth is the PICA depth.
    const auto semantic_maps = state.gs_state.GetSemanticMaps();
    const auto semantic = [&](VSOutputAttributes::Semantic slot_semantic) -> std::string {
        const u32 slot = static_cast<u32>(slot_semantic);
        const u32 attrib = semantic_maps[slot].attribute_index;
        const u32 comp = semantic_maps[slot].component_index;
        if (attrib < state.gs_state.gs_output_attributes_count) {
            return fmt::format("vs_out_attr{}.{}", attrib, "xyzw"[comp]);
        }
        return "1.0";
    };

    const auto components = [&](u32 i) -> u32 {
        const u32 c = extra.reg_components[i];
        return c == 0 || c > 4 ? 4 : c;
    };
    out += "void main(\n";
    for (u32 i = 0; i < 16; ++i) {
        if ((input_regs & (1u << i)) != 0 && (extra.default_regs & (1u << i)) == 0) {
            out += fmt::format("    float{} vs_in_attr{},\n", components(i), i);
        }
    }
    out += "    float4 out vPosition : POSITION,\n"
           "    float4 out primary_color : COLOR0,\n"
           "    float2 out texcoord0 : TEXCOORD0,\n"
           "    float2 out texcoord1 : TEXCOORD1,\n"
           "    float2 out texcoord2 : TEXCOORD2,\n"
           "    float out texcoord0_w : TEXCOORD3,\n"
           "    float4 out normquat : TEXCOORD4,\n"
           "    float4 out normquat_flat : TEXCOORD5,\n"
           "    float3 out view : TEXCOORD6)\n"
           "{\n";
    for (u32 i = 0; i < 16; ++i) {
        if ((input_regs & (1u << i)) != 0 && (extra.default_regs & (1u << i)) == 0) {
            switch (components(i)) {
            case 1:
                out += fmt::format("    vs_in_reg{0} = float4(vs_in_attr{0}, 0.0, 0.0, 1.0);\n", i);
                break;
            case 2:
                out += fmt::format("    vs_in_reg{0} = float4(vs_in_attr{0}, 0.0, 1.0);\n", i);
                break;
            case 3:
                out += fmt::format("    vs_in_reg{0} = float4(vs_in_attr{0}, 1.0);\n", i);
                break;
            default:
                out += fmt::format("    vs_in_reg{0} = vs_in_attr{0};\n", i);
                break;
            }
        }
    }
    // The default an output register holds when the program never assigns it, as the GLSL
    // wrapper writes it. It has to be an executable statement: an initialiser on a static is
    // not one here, so a semantic the program never writes was read undefined. Only the
    // registers the program leaves alone are written - assigning all sixteen cost a vec4 store
    // per vertex each, and two thirds of the frame rate on a title that writes three.
    const u16 assigned = WrittenOutputRegisters(setup);
    for (u32 i = 0; i < state.num_outputs; ++i) {
        if ((assigned & (1u << i)) == 0) {
            out += fmt::format("    vs_out_attr{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
        }
    }
    out += "    exec_shader();\n";
    out += "    float4 vtx_pos = float4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
           semantic(VSOutputAttributes::POSITION_Y) + ", " +
           semantic(VSOutputAttributes::POSITION_Z) + ", " +
           semantic(VSOutputAttributes::POSITION_W) + ");\n";
    // SanitizeVertex from the GLSL wrapper: nudge z that is barely past a clip boundary.
    out += "    float ndc_z = vtx_pos.z / vtx_pos.w;\n"
           "    if (ndc_z > 0.0 && ndc_z < 0.000001) { vtx_pos.z = 0.0; }\n"
           "    if (ndc_z < -1.0 && ndc_z > -1.00001) { vtx_pos.z = -vtx_pos.w; }\n"
           "    if (flip_viewport > 0.5) { vtx_pos.y = -vtx_pos.y; }\n"
           "    float z_over_w = vtx_pos.z / vtx_pos.w;\n"
           "    float depth = z_over_w * depth_scale + depth_offset;\n"
           "    vPosition = float4(vtx_pos.x, vtx_pos.y, depth * vtx_pos.w, vtx_pos.w);\n";
    out += "    normquat = float4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
           semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
    out += "    normquat_flat = normquat;\n";
    out += "    float4 vtx_color = float4(" + semantic(VSOutputAttributes::COLOR_R) + ", " +
           semantic(VSOutputAttributes::COLOR_G) + ", " + semantic(VSOutputAttributes::COLOR_B) +
           ", " + semantic(VSOutputAttributes::COLOR_A) + ");\n";
    out += "    primary_color = min(abs(vtx_color), float4(1.0, 1.0, 1.0, 1.0));\n";
    out += "    texcoord0 = float2(" + semantic(VSOutputAttributes::TEXCOORD0_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD0_V) + ");\n";
    out += "    texcoord1 = float2(" + semantic(VSOutputAttributes::TEXCOORD1_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD1_V) + ");\n";
    out += "    texcoord0_w = " + semantic(VSOutputAttributes::TEXCOORD0_W) + ";\n";
    out += "    view = float3(" + semantic(VSOutputAttributes::VIEW_X) + ", " +
           semantic(VSOutputAttributes::VIEW_Y) + ", " + semantic(VSOutputAttributes::VIEW_Z) +
           ");\n";
    out += "    texcoord2 = float2(" + semantic(VSOutputAttributes::TEXCOORD2_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD2_V) + ");\n";
    out += "}\n";
    return out;
}

u64 VertexShaderKey(const PicaVSConfig& config, const VSExtra& extra) {
    u64 layout = 0;
    for (u32 i = 0; i < 16; i++) {
        layout |= static_cast<u64>(extra.reg_components[i] & 7) << (i * 4);
    }
    return Common::HashCombine(config.Hash(), static_cast<u64>(extra.default_regs) |
                                                  (static_cast<u64>(extra.bool_mask) << 16) |
                                                  (static_cast<u64>(extra.bools & extra.bool_mask)
                                                   << 32),
                               layout);
}

/// The output registers the program assigns somewhere in its written extent. Over-approximate
/// for code it never reaches, which only costs an assignment that was not needed.
u16 WrittenOutputRegisters(const ShaderSetup& setup) {
    const ProgramCode& code = setup.GetProgramCode();
    const u32 extent = std::min<u32>(setup.GetBiggestProgramSize(), MAX_PROGRAM_CODE_LENGTH);
    u16 written = 0;
    for (u32 offset = 0; offset < extent; offset++) {
        const Instruction instr = {code[offset]};
        const OpCode::Info info = instr.opcode.Value().GetInfo();
        if (info.type == OpCode::Type::MultiplyAdd) {
            if (instr.mad.dest.Value() < 0x10) {
                written |= static_cast<u16>(1u << instr.mad.dest.Value().GetIndex());
            }
        } else if (info.type == OpCode::Type::Arithmetic) {
            if (instr.common.dest.Value() < 0x10) {
                written |= static_cast<u16>(1u << instr.common.dest.Value().GetIndex());
            }
        }
    }
    return written;
}

u16 UsedBoolUniforms(const ShaderSetup& setup) {
    const ProgramCode& code = setup.GetProgramCode();
    const u32 extent = std::min<u32>(setup.GetBiggestProgramSize(), MAX_PROGRAM_CODE_LENGTH);
    u16 used = 0;
    for (u32 offset = 0; offset < extent; offset++) {
        const Instruction instr = {code[offset]};
        switch (instr.opcode.Value()) {
        case OpCode::Id::IFU:
        case OpCode::Id::CALLU:
        case OpCode::Id::JMPU:
            used |= static_cast<u16>(1u << (instr.flow_control.bool_uniform_id & 15));
            break;
        default:
            break;
        }
    }
    return used;
}

void MaybeDumpVertex(const ShaderSetup& setup, const PicaVSConfig& config, u64 hash) {
    const std::string& dir = DumpDir();
    if (dir.empty()) {
        return;
    }
    u32 input_regs = 0;
    // As the console would build it: the bools the program branches on, at their values.
    VSExtra extra{};
    extra.bool_mask = UsedBoolUniforms(setup);
    for (u32 n = 0; n < 16; n++) {
        extra.bools |= static_cast<u16>((setup.uniforms.b[n] ? 1u : 0u) << n);
    }
    const std::string source = GenerateVertexShader(setup, config, extra, input_regs);
    const std::string path = fmt::format("{}/vs_{:016x}.cg", dir, hash);
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        LOG_ERROR(Render, "could not write {}", path);
        return;
    }
    std::fwrite(source.data(), 1, source.size(), file);
    std::fclose(file);
    LOG_INFO(Render, "dumped Cg vertex shader {:016x} ({} bytes{})", hash, source.size(),
             source.empty() ? ", decompilation refused" : "");
}

} // namespace Pica::Shader::Generator::Cg
