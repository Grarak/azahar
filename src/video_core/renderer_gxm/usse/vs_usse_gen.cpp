// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <nihstro/shader_bytecode.h>
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/usse/gxp_writer.h"
#include "video_core/renderer_gxm/usse/usse_encoder.h"
#include "video_core/renderer_gxm/usse/usse_ir.h"
#include "video_core/renderer_gxm/usse/vs_usse_gen.h"

/**
 * How a PICA vertex program becomes a USSE one (the forms are psp2cgc's for the same
 * operations, probes p_a-p_i of 2026-09-05, see vs_encoder_test):
 *
 * Registers. A PICA float4 lives in a quad of four consecutive registers of the unified
 * store: inputs a loader feeds in primary attributes (four per attribute, as the compiler
 * lays them out), inputs with no loader and the register-resident uniforms in secondary
 * attributes, everything else (temporaries, output attributes, the load quad, the
 * address registers as 16-bit integers, the loop state) in virtual quads that the IR
 * (usse_ir.h) places by liveness: temporaries first, then primary attributes past the
 * inputs, then the output registers the epilogue writes last. An operand with an
 * arbitrary swizzle is read four wide into an internal register (a four-wide op needs an
 * internal destination); the result lands in i2 and is moved to its quad two lanes at a
 * time.
 *
 * Uniforms. The 128 float4 uniforms cannot be register-resident (512 of 128 SAs), so
 * they sit in memory behind the resident part of the default uniform buffer: the ones a
 * program reads with a constant index are prefetched into SAs by the secondary program
 * (per draw), the others are loaded at their use with lda32 through the buffer pointer
 * libgxm plants in the data container: base + address register, wrapped to 128 entries
 * and scaled to bytes with the compiler's integer sequence.
 *
 * Control flow. The condition codes are the predicates p0 and p1 (CMP writes them, the
 * conditional forms branch on them), p2 is the scratch predicate, always used positive
 * since the vector ops have no !p2. Subroutines are inlined at each call, IF/ELSE and LOOP
 * are branches, END branches to the epilogue. A branch on per-vertex data puts the
 * program in per-instance mode (flag 0x2), as the compiler does.
 */
namespace GxmRenderer::Usse {

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

namespace {

[[noreturn]] void Refuse(const std::string& what) {
    throw std::runtime_error(what);
}

// Scratch, in the virtual store. One load quad: a PICA instruction reads at most one
// uniform (the inverted forms exist so that either operand can be it). The floats sit at
// even registers, where a compare reads them.
constexpr uint8_t TmpLoad = 0;    ///< uniform loaded at use (4)
constexpr uint8_t TmpA0X = 4;     ///< a0.x as int16
constexpr uint8_t TmpA0Y = 5;     ///< a0.y as int16
constexpr uint8_t TmpAlS16 = 6;   ///< aL as int16
constexpr uint8_t TmpAddr = 7;    ///< the byte offset of a dynamically indexed uniform
constexpr uint8_t TmpAlF = 8;     ///< aL as float
constexpr uint8_t TmpCount0 = 10; ///< loop counters by depth, float
constexpr uint8_t TmpCount1 = 12;
constexpr uint8_t TmpQuads = 16; ///< the first PICA quad
constexpr uint32_t VirtualLimit = 256;
constexpr uint32_t PaLimit = 64;
constexpr uint32_t SaLimit = 128;
/// Output registers free as scratch until the epilogue writes them.
constexpr uint8_t OutScratchEnd = 24;

// The resident secondary attributes: the depth mapping, then the input defaults and the
// integer uniforms (container 14, in the order the parameters name them).
constexpr uint32_t SaFlipViewport = 0;
constexpr uint32_t SaDepthScale = 1;
constexpr uint32_t SaDepthOffset = 2;
constexpr uint32_t SaResidentHead = 4;

// The data container (19): the buffer pointer, then the literals. The ones a compare reads
// as a scalar sit at even offsets (a test reads the .x of a pair); the odd slots between
// them are the secondary program's scratch for computed load offsets.
enum DataSlot : uint32_t {
    DataPtr = 0,
    DataPad1 = 1,    ///< zero (the compiler keeps integer operands on even SAs too)
    DataNeg128 = 2,  ///< -128.0f
    Data0x10000 = 3, ///< the base of a register load offset
    Data127 = 4,     ///< 127.0f
    DataScratch0 = 5,
    DataEpsilon = 6, ///< 1e-6f
    DataScratch1 = 7,
    DataNeg1Eps = 8, ///< -1.00001f
    DataScratch2 = 9,
    DataNeg1 = 10, ///< -1.0f
    DataScratch3 = 11,
    DataOnes16 = 12, ///< 0x00010001: the mad.i16 multiplier for an add (even, like psp2cgc's)
    DataPad13 = 13,
    DataSize = 14,
};
constexpr uint32_t DataLiteralCount = DataSize - 1;
constexpr uint32_t ScratchSlots[4] = {DataScratch0, DataScratch1, DataScratch2, DataScratch3};

// Output registers of the fragment interface, as the compiled programs with the same
// interface lay them out: POSITION o0-o3, COLOR0 o4-o7, TEXCOORD0-2 two each from o8,
// TEXCOORD3 o14, TEXCOORD4 o16-19, TEXCOORD5 o20-23, TEXCOORD6 o24-26.
constexpr uint8_t OutPosition = 0;
constexpr uint8_t OutColor = 4;
constexpr uint8_t OutTexcoord0 = 8;
constexpr uint8_t OutTexcoord1 = 10;
constexpr uint8_t OutTexcoord2 = 12;
constexpr uint8_t OutTexcoord0W = 14;
constexpr uint8_t OutNormquat = 16;
constexpr uint8_t OutNormquatFlat = 20;
constexpr uint8_t OutView = 24;
constexpr uint32_t VertexOutputs1 = 0x1b001800; ///< 27 registers, position, colour
constexpr uint32_t VertexOutputs2 = 0xff249;    ///< the texcoord widths

// VSOutputAttributes::Semantic, as regs_rasterizer.h numbers them.
enum Semantic : uint8_t {
    SemPositionX = 0,
    SemQuaternionX = 4,
    SemColorR = 8,
    SemTexcoord0U = 12,
    SemTexcoord0V = 13,
    SemTexcoord1U = 14,
    SemTexcoord1V = 15,
    SemTexcoord0W = 16,
    SemViewX = 18,
    SemTexcoord2U = 22,
    SemTexcoord2V = 23,
};

const Src ConstZero{Reg::C(0), XXXX};
const Src ConstOne{Reg::C(2), XXXX};
const Src ConstZeroOne{Reg::C(1), XYZW}; ///< (0, 1) as a pair
const Src ConstHalf{Reg::C(12), XXXX};
const Reg I0 = Reg::I(0);
const Reg I1 = Reg::I(1);
const Reg I2 = Reg::I(2);

/// A float4 in the unified store: four consecutive registers from a pair-aligned base.
struct Quad {
    Bank bank;
    uint8_t base;
    Reg Base() const {
        return {bank, base};
    }
    Reg Pair(int which) const {
        return {bank, static_cast<uint8_t>(base + which * 2)};
    }
};

/// A source operand: a quad with the instruction's swizzle and negation applied.
struct Operand {
    Quad quad;
    Swizzle swz;
    Mod mod;
};

/// An inlining of a code range: main, or a subroutine at one call site. Jumps resolve
/// within it.
struct Context {
    uint32_t begin;
    uint32_t end;
    std::map<uint32_t, int> labels; ///< PICA offset -> IR label
};

Ch ToCh(SwizzlePattern::Selector s) {
    switch (s) {
    case SwizzlePattern::Selector::x:
        return Ch::X;
    case SwizzlePattern::Selector::y:
        return Ch::Y;
    case SwizzlePattern::Selector::z:
        return Ch::Z;
    default:
        return Ch::W;
    }
}

Swizzle SelectorSwizzle(const SwizzlePattern& p, int source) {
    switch (source) {
    case 1:
        return {ToCh(p.src1_selector_0), ToCh(p.src1_selector_1), ToCh(p.src1_selector_2),
                ToCh(p.src1_selector_3)};
    case 2:
        return {ToCh(p.src2_selector_0), ToCh(p.src2_selector_1), ToCh(p.src2_selector_2),
                ToCh(p.src2_selector_3)};
    default:
        return {ToCh(p.src3_selector_0), ToCh(p.src3_selector_1), ToCh(p.src3_selector_2),
                ToCh(p.src3_selector_3)};
    }
}

uint8_t DestMask(const SwizzlePattern& p) {
    uint8_t mask = 0;
    for (int i = 0; i < 4; i++) {
        if (p.DestComponentEnabled(i)) {
            mask |= static_cast<uint8_t>(1u << i);
        }
    }
    return mask;
}

/// Whether a four-wide swizzle is one the V32NMAD/VMAD source tables hold.
bool InVec4Table(const Swizzle& s) {
    static constexpr Swizzle table[] = {
        {Ch::X, Ch::X, Ch::X, Ch::X},   {Ch::Y, Ch::Y, Ch::Y, Ch::Y}, {Ch::Z, Ch::Z, Ch::Z, Ch::Z},
        {Ch::W, Ch::W, Ch::W, Ch::W},   {Ch::X, Ch::Y, Ch::Z, Ch::W}, {Ch::Y, Ch::Z, Ch::W, Ch::W},
        {Ch::X, Ch::Y, Ch::Z, Ch::Z},   {Ch::X, Ch::X, Ch::Y, Ch::Z}, {Ch::X, Ch::Y, Ch::X, Ch::Y},
        {Ch::X, Ch::Y, Ch::W, Ch::Z},   {Ch::Z, Ch::X, Ch::Y, Ch::W}, {Ch::Z, Ch::W, Ch::Z, Ch::W},
        {Ch::Y, Ch::Z, Ch::X, Ch::Z},   {Ch::X, Ch::X, Ch::Y, Ch::Y}, {Ch::X, Ch::Z, Ch::W, Ch::W},
        {Ch::X, Ch::Y, Ch::Z, Ch::One}, {Ch::Y, Ch::Z, Ch::X, Ch::W}, {Ch::Z, Ch::W, Ch::X, Ch::Y},
        {Ch::X, Ch::Z, Ch::W, Ch::Y},   {Ch::Y, Ch::Y, Ch::W, Ch::W}, {Ch::W, Ch::Y, Ch::Z, Ch::W},
        {Ch::W, Ch::Z, Ch::W, Ch::Z},   {Ch::X, Ch::Y, Ch::Z, Ch::X}, {Ch::Z, Ch::Z, Ch::W, Ch::W},
        {Ch::X, Ch::W, Ch::Z, Ch::X},   {Ch::Y, Ch::Y, Ch::Y, Ch::X}, {Ch::Y, Ch::Y, Ch::Y, Ch::Z},
        {Ch::X, Ch::Z, Ch::Y, Ch::W},   {Ch::X, Ch::X, Ch::X, Ch::Y}, {Ch::Z, Ch::Y, Ch::X, Ch::W},
        {Ch::Y, Ch::Y, Ch::Z, Ch::Z},   {Ch::Z, Ch::Z, Ch::Z, Ch::Y},
    };
    for (const auto& t : table) {
        if (t == s) {
            return true;
        }
    }
    return false;
}

class VertexEmitter {
public:
    VertexEmitter(const VsRequest& request_, int level_) : req{request_}, level{level_} {}

    std::vector<uint8_t> Emit(uint32_t& input_regs);
    int level = 1;

private:
    // --- analysis ---
    struct Scan {
        uint16_t inputs = 0;                 ///< input registers read
        uint16_t temps = 0;                  ///< temporaries touched
        std::array<uint8_t, 16> temp_init{}; ///< lanes read before (linearly) written
        std::array<uint8_t, 16> temp_written{};
        uint16_t attrs_written = 0;
        std::array<uint16_t, 128> uniform_uses{};
        bool any_uniform = false;
        bool a0x = false, a0y = false, al = false;
        uint8_t ints = 0; ///< integer uniforms LOOP reads
        bool cc = false;  ///< a conditional form reads the condition codes
    };
    void ScanProgram();
    void MarkReachable(uint32_t begin, uint32_t end);
    void NoteSource(const SourceRegister& reg, uint32_t addr, const Swizzle& swz);
    void NoteDest(const DestRegister& reg, uint8_t mask);

    // --- layout ---
    void Layout();
    Quad AllocateQuad();

    // --- emission ---
    void EmitSecondary();
    void PlanHoist();
    void CopyHoisted(uint32_t temp, uint8_t lanes);
    void HoistCopy(Ir::Builder& b, const Quad& dst, uint8_t lanes, const Quad& src,
                   const Swizzle& swz);
    int AliasUniform(const Instruction& instr, const SwizzlePattern& swz) const;
    void NotePrimaryWrite(const DestRegister& dest, uint8_t mask, OpCode::Id op);
    void Prologue();
    void Epilogue();
    uint32_t EmitRange(uint32_t begin, uint32_t end);
    uint32_t EmitInstruction(uint32_t offset);
    void EmitArithmetic(const Instruction& instr, const SwizzlePattern& swz);
    void EmitMad(const Instruction& instr, const SwizzlePattern& swz);

    Operand Source(const SourceRegister& reg, uint32_t addr, const Swizzle& swz, bool negate);
    std::optional<Quad> Dest(const DestRegister& reg);
    /// The operand read four wide into `internal` with its swizzle and sign applied.
    Src Materialize(Reg internal, const Operand& op);
    /// Console bisecting (gxm_flags `vs_noflow`, `vs_nodyn`): programs with control flow,
    /// or with uniform reads through the address registers, go to the software renderer.
    /// The operand as a four-wide source straight from the unified store (V32NMAD's and
    /// VDP's first source take any swizzle).
    static Src Direct(const Operand& op) {
        return {op.quad.Base(), op.swz, op.mod};
    }
    /// A source for one lane of a two-wide move: the pair holding the lane's channel.
    static Src LaneSrc(const Operand& op, int lane) {
        const Ch ch = op.swz[lane];
        return {op.quad.Pair(static_cast<int>(ch) / 2), (static_cast<int>(ch) & 1) ? YYYY : XXXX,
                op.mod};
    }
    /// i2's lanes into the destination quad under the mask.
    void Store(const Quad& dst, uint8_t mask, Reg internal);
    Src Data(DataSlot slot) const {
        return {Reg::Sa(static_cast<uint8_t>(data_base + slot)), XXXX};
    }
    Reg DataReg(DataSlot slot) const {
        return Reg::Sa(static_cast<uint8_t>(data_base + slot));
    }

    // Control flow.
    int NewLabel();
    int LabelAt(uint32_t offset);
    void Bind(int label);
    bool Bound(int label) const {
        return label_bound[static_cast<std::size_t>(label)];
    }
    void Branch(Pred pred, int target);
    void CondBranch(Instruction::FlowControlType flow, int target, bool when_true);
    std::optional<bool> KnownBool(uint32_t id) const {
        if (id < 16 && (req.bool_mask & (1u << id)) != 0) {
            return (req.bools & (1u << id)) != 0;
        }
        return std::nullopt;
    }

    // Epilogue helpers.
    /// The pair and lane a fragment-interface semantic comes from, or nothing for 1.0.
    std::optional<Src> SemanticSrc(Semantic s) const;
    void GatherFour(Reg internal, Semantic first);
    void OutputPair(uint8_t out, Semantic s0, Semantic s1);
    void OutputLane(uint8_t out, int lane, Semantic s);

    const VsRequest& req;
    Scan scan;
    std::vector<bool> reachable;
    Ir::Builder e;   ///< the primary program
    Ir::Builder sec; ///< the secondary program
    /// Where the arithmetic emitters write: the primary program, or the secondary while an
    /// instruction is being hoisted there (see PlanHoist).
    Ir::Builder* out = &e;
    /// The uniform-only hoist: instructions of main's straight-line prefix whose every
    /// source is a prefetched uniform, a constant, or a temporary lane an earlier hoisted
    /// instruction wrote, and whose destination is a temporary, run in the secondary
    /// program - once per draw instead of once per vertex - and leave their result in an
    /// SA quad. psp2cgc puts a skinning program's whole bone blend there.
    /// Per instruction offset: 0 not hoisted, 1 hoisted, 2 hoisted into a fresh SA home for
    /// its temporary (a lane it writes was hoisted before, and the earlier value may still
    /// have primary readers).
    std::vector<uint8_t> hoist_at;
    /// Before the conditional region starting at this offset: the lanes, per temporary, whose
    /// hoisted value the primary's quad must hold going in (PlanHoist).
    std::map<uint32_t, std::array<uint8_t, 16>> hoist_flush;
    int region_depth = 0; ///< conditional regions the emitter is inside (IFC arms, CALLC bodies)
    std::array<uint8_t, 16> hoist_lanes{};    ///< temp -> lanes whose value is in its SA home
    std::array<uint8_t, 16> hoist_uncopied{}; ///< of those, lanes the primary's quad lacks
    std::array<int, 16> hoist_sa{};           ///< temp -> SA home, or -1
    /// A hoisted whole-register MOV from a prefetched uniform makes that uniform's SA quad
    /// the temporary's home, read through the MOV's swizzle: no copy, no SA quad of its own.
    std::array<Swizzle, 16> hoist_swz{};
    std::array<bool, 16> hoist_alias{};
    bool hoisting = false; ///< the instruction being emitted is hoisted
    uint32_t hoisted_count = 0;
    GxpProgram gxp;

    // Where things landed.
    std::array<int, 16> input_pa{};   ///< input register -> PA base, or -1
    std::array<int, 16> default_sa{}; ///< input register -> SA base, or -1
    std::array<std::optional<Quad>, 16> temp_quad{};
    std::array<std::optional<Quad>, 16> attr_quad{};
    std::array<int, 128> uniform_sa{}; ///< uniform -> SA base when prefetched, or -1
    uint32_t resident = 0;             ///< container 14 size (floats)
    uint32_t data_base = 0;            ///< container 19's first SA
    uint32_t sa_top = 0;               ///< SAs in use
    int int_sa = -1;                   ///< uniforms_i's SA base
    uint8_t pa_inputs = 0;             ///< PA registers the attributes take
    uint32_t next_virtual = TmpQuads;  ///< the next free virtual quad's first register
    std::vector<uint8_t> prefetch;     ///< uniforms the secondary program loads, by index

    std::vector<bool> label_bound;
    int end_label = -1;
    std::vector<Context> contexts;
    int loop_depth = 0;
    bool per_instance = false;
};

// ---------------------------------------------------------------------------------------
// Analysis

void VertexEmitter::NoteSource(const SourceRegister& reg, uint32_t addr, const Swizzle& swz) {
    const uint32_t index = static_cast<uint32_t>(reg.GetIndex());
    uint8_t lanes = 0;
    for (const Ch ch : swz) {
        if (ch <= Ch::W) {
            lanes |= static_cast<uint8_t>(1u << static_cast<int>(ch));
        }
    }
    switch (reg.GetRegisterType()) {
    case RegisterType::Input:
        scan.inputs |= static_cast<uint16_t>(1u << (index & 15));
        break;
    case RegisterType::Temporary:
        scan.temps |= static_cast<uint16_t>(1u << (index & 15));
        scan.temp_init[index & 15] |= static_cast<uint8_t>(lanes & ~scan.temp_written[index & 15]);
        break;
    case RegisterType::FloatUniform:
        scan.any_uniform = true;
        if (addr == 0) {
            scan.uniform_uses[index & 127]++;
        } else if (addr == 1) {
            scan.a0x = true;
        } else if (addr == 2) {
            scan.a0y = true;
        } else {
            scan.al = true;
        }
        break;
    default:
        break;
    }
}

void VertexEmitter::NoteDest(const DestRegister& reg, uint8_t mask) {
    const uint32_t index = static_cast<uint32_t>(reg.GetIndex());
    if (reg.GetRegisterType() == RegisterType::Temporary) {
        scan.temps |= static_cast<uint16_t>(1u << (index & 15));
        scan.temp_written[index & 15] |= mask;
    } else if (reg.GetRegisterType() == RegisterType::Output) {
        const uint32_t attr = req.output_map[index & 15];
        if (attr < req.num_outputs) {
            scan.attrs_written |= static_cast<uint16_t>(1u << attr);
        }
    }
}

/// Marks the instructions main can reach: the code buffer holds every program the title
/// uploaded, and scanning them all would charge this program for their registers.
void VertexEmitter::MarkReachable(uint32_t begin, uint32_t end) {
    for (uint32_t offset = begin; offset < end && offset < reachable.size(); offset++) {
        if (reachable[offset]) {
            return; // joined a range already walked
        }
        reachable[offset] = true;
        const Instruction instr = {req.code[offset]};
        const auto& flow = instr.flow_control;
        switch (instr.opcode.Value()) {
        case OpCode::Id::END:
            return;
        case OpCode::Id::JMPC:
        case OpCode::Id::JMPU:
            MarkReachable(flow.dest_offset, end);
            break;
        case OpCode::Id::CALL:
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU:
            MarkReachable(flow.dest_offset, flow.dest_offset + flow.num_instructions);
            break;
        case OpCode::Id::IFC:
        case OpCode::Id::IFU:
            MarkReachable(offset + 1, flow.dest_offset);
            MarkReachable(flow.dest_offset, flow.dest_offset + flow.num_instructions);
            offset = flow.dest_offset + flow.num_instructions - 1;
            break;
        case OpCode::Id::LOOP:
            MarkReachable(offset + 1, flow.dest_offset + 1);
            offset = flow.dest_offset;
            break;
        default:
            break;
        }
    }
}

void VertexEmitter::ScanProgram() {
    const uint32_t extent =
        std::min<uint32_t>(req.code_size, static_cast<uint32_t>(req.code.size()));
    reachable.assign(extent, false);
    MarkReachable(req.main_offset, extent);
    for (uint32_t offset = 0; offset < extent; offset++) {
        if (!reachable[offset]) {
            continue;
        }
        const Instruction instr = {req.code[offset]};
        const OpCode::Info info = instr.opcode.Value().GetInfo();
        if (info.type == OpCode::Type::Arithmetic) {
            const SwizzlePattern swz = {req.swizzle[instr.common.operand_desc_id]};
            const bool inverted = (info.subtype & OpCode::Info::SrcInversed) != 0;
            NoteSource(instr.common.GetSrc1(inverted),
                       !inverted * instr.common.address_register_index, SelectorSwizzle(swz, 1));
            NoteSource(instr.common.GetSrc2(inverted),
                       inverted * instr.common.address_register_index, SelectorSwizzle(swz, 2));
            const auto op = instr.opcode.Value().EffectiveOpCode();
            if (op == OpCode::Id::MOVA) {
                if (swz.DestComponentEnabled(0)) {
                    scan.a0x = true;
                }
                if (swz.DestComponentEnabled(1)) {
                    scan.a0y = true;
                }
            } else if (op != OpCode::Id::CMP) {
                NoteDest(instr.common.dest.Value(), DestMask(swz));
            }
        } else if (info.type == OpCode::Type::MultiplyAdd) {
            const SwizzlePattern swz = {req.swizzle[instr.mad.operand_desc_id]};
            const bool inverted = instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI;
            NoteSource(instr.mad.GetSrc1(inverted), 0, SelectorSwizzle(swz, 1));
            NoteSource(instr.mad.GetSrc2(inverted), !inverted * instr.mad.address_register_index,
                       SelectorSwizzle(swz, 2));
            NoteSource(instr.mad.GetSrc3(inverted), inverted * instr.mad.address_register_index,
                       SelectorSwizzle(swz, 3));
            NoteDest(instr.mad.dest.Value(), DestMask(swz));
        } else {
            switch (instr.opcode.Value()) {
            case OpCode::Id::LOOP:
                scan.ints |= static_cast<uint8_t>(1u << (instr.flow_control.int_uniform_id & 3));
                scan.al = true;
                break;
            case OpCode::Id::IFC:
            case OpCode::Id::JMPC:
            case OpCode::Id::CALLC:
                scan.cc = true;
                break;
            default:
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Layout

Quad VertexEmitter::AllocateQuad() {
    if (next_virtual + 4 > VirtualLimit) {
        Refuse("virtual register budget");
    }
    const Quad q{Bank::Virtual, static_cast<uint8_t>(next_virtual)};
    next_virtual += 4;
    return q;
}

void VertexEmitter::Layout() {
    input_pa.fill(-1);
    default_sa.fill(-1);
    uniform_sa.fill(-1);

    // Attributes: four PAs each, in register order; the components a loader does not
    // supply are filled by the prologue.
    for (uint32_t i = 0; i < 16; i++) {
        if ((scan.inputs & (1u << i)) == 0 || (req.default_regs & (1u << i)) != 0) {
            continue;
        }
        input_pa[i] = pa_inputs;
        gxp.attributes.push_back({"vs_in_attr" + std::to_string(i), pa_inputs});
        uint32_t comps = req.reg_components[i];
        if (comps == 0 || comps > 4) {
            comps = 4;
        }
        gxp.attrib_pa_regs |= ((1ull << comps) - 1) << pa_inputs;
        pa_inputs = static_cast<uint8_t>(pa_inputs + 4);
    }

    // Resident uniforms: the depth mapping, the input defaults, the integer uniforms.
    gxp.uniforms.push_back({"flip_viewport", SaFlipViewport, 1, 1});
    gxp.uniforms.push_back({"depth_scale", SaDepthScale, 1, 1});
    gxp.uniforms.push_back({"depth_offset", SaDepthOffset, 1, 1});
    resident = SaResidentHead;
    for (uint32_t i = 0; i < 16; i++) {
        if ((scan.inputs & (1u << i)) != 0 && (req.default_regs & (1u << i)) != 0) {
            default_sa[i] = static_cast<int>(resident);
            gxp.uniforms.push_back({"vs_in_reg" + std::to_string(i), resident, 4, 1});
            resident += 4;
        }
    }
    if (scan.ints != 0) {
        int_sa = static_cast<int>(resident);
        gxp.uniforms.push_back({"uniforms_i", resident, 4, 4});
        resident += 16;
    }
    gxp.uniform_floats = resident;

    // The data container, then the prefetched uniforms.
    data_base = resident;
    sa_top = data_base + DataSize;
    if (scan.any_uniform) {
        // 128 entries plus a mirror of the first four: an indexed read whose index wraps
        // at 128 (Smash reads with a0 = -2) may be issued at an immediate offset from a
        // neighbouring row's address, and the offset runs into the mirror.
        gxp.uniforms.push_back({"uniforms_f", resident, 4, 128 + Ir::UniformMirrorRows});
        gxp.buffer_pointer = true;
        gxp.ldst_base_value = static_cast<int32_t>(resident * 4) - 4;
        gxp.buffer_floats = resident + 512 + Ir::UniformMirrorRows * 4;
    }
    // Most used first, as many as fit.
    std::vector<uint8_t> order;
    for (uint32_t n = 0; n < 128; n++) {
        if (scan.uniform_uses[n] != 0) {
            order.push_back(static_cast<uint8_t>(n));
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](uint8_t a, uint8_t b) {
        return scan.uniform_uses[a] > scan.uniform_uses[b];
    });
    for (const uint8_t n : order) {
        if (sa_top + 4 > SaLimit) {
            break;
        }
        uniform_sa[n] = static_cast<int>(sa_top);
        prefetch.push_back(n);
        sa_top += 4;
    }
    std::sort(prefetch.begin(), prefetch.end());
    gxp.secondary_reg_count = sa_top;

    // Output attributes, then the temporaries.
    for (uint32_t k = 0; k < req.num_outputs && k < 16; k++) {
        attr_quad[k] = AllocateQuad();
    }
    for (uint32_t t = 0; t < 16; t++) {
        if ((scan.temps & (1u << t)) != 0) {
            temp_quad[t] = AllocateQuad();
        }
    }
}

// ---------------------------------------------------------------------------------------
// The secondary program: prefetch the constant-index uniforms into their SAs.

void VertexEmitter::EmitSecondary() {
    sec.secondary = true;
    sec.skip_invalid = true;
    const bool any_hoist =
        std::any_of(hoist_at.begin(), hoist_at.end(), [](uint8_t h) { return h != 0; });
    if (prefetch.empty() && !any_hoist) {
        // Nothing to do: no words at all, as the compiler's programs without one.
        return;
    }
    const Reg ptr = DataReg(DataPtr);
    uint32_t outstanding = 0;
    uint32_t scratch = 0;
    const auto wait = [&] {
        if (outstanding != 0) {
            sec.WdfVertex(0);
            outstanding = 0;
            scratch = 0;
        }
    };
    for (const uint8_t n : prefetch) {
        const Reg dst = Reg::Sa(static_cast<uint8_t>(uniform_sa[n]));
        if (n <= 31) {
            // The immediate offset is in words, seven bits: uniform n at word 4n.
            sec.Lda32(dst, ptr, nullptr, static_cast<uint8_t>(n * 4), 4, 0);
        } else {
            // Past the immediate's reach: the offset in a scratch SA, one per outstanding
            // load so a load never has its offset overwritten under it.
            if (scratch == 4) {
                wait();
            }
            const Reg off = DataReg(static_cast<DataSlot>(ScratchSlots[scratch++]));
            sec.Bitwise(BwOp::Or, off, IOp::Imm(n), IOp::Imm(0));
            sec.MadI32(off, off, 16, DataReg(Data0x10000));
            sec.Lda32(dst, ptr, &off, 0, 4, 0);
        }
        if (++outstanding == 8) {
            wait();
        }
    }
    wait();
    // The hoisted instructions follow, appended as the primary walk reaches them; the
    // program is ended once that walk is done (Emit).
}

// ---------------------------------------------------------------------------------------
// Operands

Operand VertexEmitter::Source(const SourceRegister& reg, uint32_t addr, const Swizzle& swz,
                              bool negate) {
    constexpr uint8_t load_quad = TmpLoad;
    const uint32_t index = static_cast<uint32_t>(reg.GetIndex());
    const Mod mod = negate ? Mod::Neg : Mod::None;
    switch (reg.GetRegisterType()) {
    case RegisterType::Input:
        if (input_pa[index & 15] >= 0) {
            return {{Bank::PrimAttr, static_cast<uint8_t>(input_pa[index & 15])}, swz, mod};
        }
        if (default_sa[index & 15] >= 0) {
            return {{Bank::SecAttr, static_cast<uint8_t>(default_sa[index & 15])}, swz, mod};
        }
        Refuse("input register without a place");
    case RegisterType::Temporary: {
        const uint32_t t = index & 15;
        uint8_t lanes = 0;
        for (int l = 0; l < 4; l++) {
            if (swz[l] <= Ch::W) {
                lanes |= static_cast<uint8_t>(1u << static_cast<int>(swz[l]));
            }
        }
        const auto home_read = [&]() -> Operand {
            Swizzle composed = swz;
            for (int l = 0; l < 4; l++) {
                if (swz[l] <= Ch::W) {
                    composed[l] = hoist_swz[t][static_cast<int>(swz[l])];
                }
            }
            return {Quad{Bank::SecAttr, static_cast<uint8_t>(hoist_sa[t])}, composed, mod};
        };
        if (hoisting) {
            // Planned only where every lane read sits in the SA home.
            if ((hoist_lanes[t] & lanes) != lanes || hoist_sa[t] < 0) {
                Refuse("hoisted read of a lane the secondary program did not write");
            }
            return home_read();
        }
        // Every lane read holds the hoisted value: read the SA home itself, like a uniform.
        // Otherwise the quad, after the hoisted lanes it lacks are copied in.
        if (hoist_sa[t] >= 0 && (hoist_lanes[t] & lanes) == lanes) {
            return home_read();
        }
        if (!temp_quad[t]) {
            Refuse("temporary without a place");
        }
        if ((hoist_uncopied[t] & lanes) != 0) {
            CopyHoisted(t, static_cast<uint8_t>(hoist_uncopied[t] & lanes));
        }
        return {*temp_quad[t], swz, mod};
    }
    case RegisterType::FloatUniform: {
        const Reg ptr = DataReg(DataPtr);
        const Reg dst = Reg::V(load_quad);
        if (addr == 0) {
            if (uniform_sa[index & 127] >= 0) {
                return {{Bank::SecAttr, static_cast<uint8_t>(uniform_sa[index & 127])}, swz, mod};
            }
            if (hoisting) {
                Refuse("hoisted read of a uniform that is not prefetched");
            }
            // Not prefetched: loaded here.
            const uint32_t n = index & 127;
            if (n <= 31) {
                out->Lda32(dst, ptr, nullptr, static_cast<uint8_t>(n * 4), 4, 0);
            } else {
                const Reg address = Reg::V(TmpAddr);
                out->Bitwise(BwOp::Or, address, IOp::Imm(static_cast<uint8_t>(n)), IOp::Imm(0));
                out->MadI32(address, address, 16, ptr);
                out->Lda32(dst, address, nullptr, 0, 4, 0);
            }
            out->WdfVertex(0);
            return {{Bank::Virtual, load_quad}, swz, mod};
        }
        // Indexed by an address register: (index + a) & 127, times 16, plus the buffer
        // pointer, into a per-instance register that is the load's base with offset 0.
        // That is psp2cgc's form in every per-instance program (`mad.i32 o0.x, r0.lo16,
        // 0x10.lo16, sa16.x; lda32.fetch4 r0, [o0, 0x0]`); the `[sa16, r3]` form it uses
        // in programs without branches broke skinned vertices on the console (2026-09-06).
        const Reg areg = addr == 1 ? Reg::V(TmpA0X) : addr == 2 ? Reg::V(TmpA0Y) : Reg::V(TmpAlS16);
        const Reg address = Reg::V(TmpAddr);
        out->MadI16(address, areg, DataReg(DataOnes16),
                    IOp::Imm(static_cast<uint8_t>(index & 127)));
        out->Bitwise(BwOp::And, address, IOp::Imm(0x7f), IOp::R(address));
        out->MadI32(address, address, 16, ptr);
        out->Lda32(dst, address, nullptr, 0, 4, 0);
        out->WdfVertex(0);
        return {{Bank::Virtual, load_quad}, swz, mod};
    }
    default:
        Refuse("unknown source register type");
    }
}

std::optional<Quad> VertexEmitter::Dest(const DestRegister& reg) {
    const uint32_t index = static_cast<uint32_t>(reg.GetIndex());
    switch (reg.GetRegisterType()) {
    case RegisterType::Output: {
        const uint32_t attr = req.output_map[index & 15];
        if (attr < req.num_outputs && attr_quad[attr]) {
            return attr_quad[attr];
        }
        return std::nullopt; // an output nothing reads
    }
    case RegisterType::Temporary: {
        const uint32_t t = index & 15;
        if (hoisting) {
            if (hoist_sa[t] < 0) {
                if (sa_top + 4 > SaLimit) {
                    Refuse("no secondary attributes left for a hoisted temporary");
                }
                hoist_sa[t] = static_cast<int>(sa_top);
                sa_top += 4;
            }
            if (hoist_alias[t]) {
                Refuse("hoisted write into a uniform's own quad");
            }
            return Quad{Bank::SecAttr, static_cast<uint8_t>(hoist_sa[t])};
        }
        if (!temp_quad[t]) {
            Refuse("temporary without a place");
        }
        return temp_quad[t];
    }
    default:
        Refuse("unknown destination register type");
    }
}

Src VertexEmitter::Materialize(Reg internal, const Operand& op) {
    out->Vec(VecOp::Mul, {internal, 0xf}, Direct(op), ConstOne);
    return {internal, XYZW};
}

void VertexEmitter::Store(const Quad& dst, uint8_t mask, Reg internal) {
    if (mask & 0x3) {
        out->MoveInternal({dst.Pair(0), static_cast<uint8_t>(mask & 0x3)}, {internal, XYZW});
    }
    if (mask & 0xc) {
        out->MoveInternal({dst.Pair(1), static_cast<uint8_t>(mask >> 2)}, {internal, ZWZW});
    }
}

// ---------------------------------------------------------------------------------------
// Control flow

int VertexEmitter::NewLabel() {
    const int label = e.NewLabel();
    label_bound.push_back(false);
    return label;
}

int VertexEmitter::LabelAt(uint32_t offset) {
    Context& ctx = contexts.back();
    if (offset < ctx.begin || offset >= ctx.end) {
        Refuse("jump out of its subroutine");
    }
    auto it = ctx.labels.find(offset);
    if (it == ctx.labels.end()) {
        it = ctx.labels.emplace(offset, NewLabel()).first;
    }
    return it->second;
}

void VertexEmitter::Bind(int label) {
    label_bound[static_cast<std::size_t>(label)] = true;
    e.Bind(label);
}

void VertexEmitter::Branch(Pred pred, int target) {
    per_instance = true;
    e.Br(pred, target);
}

void VertexEmitter::CondBranch(Instruction::FlowControlType flow, int target, bool when_true) {
    using Op = Instruction::FlowControlType::Op;
    const Pred a = flow.refx.Value() ? Pred::P0 : Pred::NotP0;
    const Pred not_a = flow.refx.Value() ? Pred::NotP0 : Pred::P0;
    const Pred b = flow.refy.Value() ? Pred::P1 : Pred::NotP1;
    const Pred not_b = flow.refy.Value() ? Pred::NotP1 : Pred::P1;
    switch (flow.op) {
    case Op::JustX:
        Branch(when_true ? a : not_a, target);
        break;
    case Op::JustY:
        Branch(when_true ? b : not_b, target);
        break;
    case Op::Or:
        if (when_true) {
            Branch(a, target);
            Branch(b, target);
        } else {
            const int skip = NewLabel();
            Branch(a, skip);
            Branch(not_b, target);
            Bind(skip);
        }
        break;
    case Op::And:
        if (when_true) {
            const int skip = NewLabel();
            Branch(not_a, skip);
            Branch(b, target);
            Bind(skip);
        } else {
            Branch(not_a, target);
            Branch(not_b, target);
        }
        break;
    default:
        Refuse("unknown condition");
    }
}

// ---------------------------------------------------------------------------------------
// Instructions

uint32_t VertexEmitter::EmitRange(uint32_t begin, uint32_t end) {
    uint32_t pc = begin;
    while (pc < end) {
        const Context& ctx = contexts.back();
        const auto it = ctx.labels.find(pc);
        if (it != ctx.labels.end() && !Bound(it->second)) {
            Bind(it->second);
        }
        pc = EmitInstruction(pc);
    }
    return pc;
}

void VertexEmitter::EmitArithmetic(const Instruction& instr, const SwizzlePattern& swz) {
    const OpCode::Info info = instr.opcode.Value().GetInfo();
    const bool inverted = (info.subtype & OpCode::Info::SrcInversed) != 0;
    const auto op = instr.opcode.Value().EffectiveOpCode();
    const uint8_t mask = DestMask(swz);
    const auto src1 = [&] {
        return Source(instr.common.GetSrc1(inverted),
                      !inverted * instr.common.address_register_index, SelectorSwizzle(swz, 1),
                      swz.negate_src1);
    };
    const auto src2 = [&] {
        return Source(instr.common.GetSrc2(inverted),
                      inverted * instr.common.address_register_index, SelectorSwizzle(swz, 2),
                      swz.negate_src2);
    };

    switch (op) {
    case OpCode::Id::ADD:
    case OpCode::Id::MUL:
    case OpCode::Id::MAX:
    case OpCode::Id::MIN: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        const Operand a = src1();
        const Operand b = src2();
        const VecOp vop = op == OpCode::Id::ADD   ? VecOp::Add
                          : op == OpCode::Id::MUL ? VecOp::Mul
                          : op == OpCode::Id::MAX ? VecOp::Max
                                                  : VecOp::Min;
        // The second source goes through i1: the op's second source takes neither a free
        // swizzle nor a negation, and a four-wide read of the unified store is only proven
        // for the first.
        const Src sb = Materialize(I1, b);
        out->Vec(vop, {I2, 0xf}, Direct(a), sb);
        Store(*dst, mask, I2);
        return;
    }
    case OpCode::Id::DP3:
    case OpCode::Id::DP4:
    case OpCode::Id::DPH:
    case OpCode::Id::DPHI: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        Operand a = src1();
        const Operand b = src2();
        const bool vec4 = op != OpCode::Id::DP3;
        if (op == OpCode::Id::DPH || op == OpCode::Id::DPHI) {
            a.swz[3] = Ch::One;
        }
        const Src sb = Materialize(I1, b);
        if (std::popcount(mask) == 1) {
            const int lane = std::countr_zero(mask);
            out->Dot({dst->Pair(lane / 2), static_cast<uint8_t>(1u << (lane & 1))}, Direct(a), sb,
                     vec4);
        } else {
            out->Dot({I2, 0x1}, Direct(a), sb, vec4);
            if (mask & 0x3) {
                out->MoveInternal({dst->Pair(0), static_cast<uint8_t>(mask & 0x3)}, {I2, XXXX});
            }
            if (mask & 0xc) {
                out->MoveInternal({dst->Pair(1), static_cast<uint8_t>(mask >> 2)}, {I2, XXXX});
            }
        }
        return;
    }
    case OpCode::Id::MOV: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        const Operand a = src1();
        // Lane pairs whose channels sit in one source pair move straight across.
        bool direct = true;
        for (int h = 0; h < 2; h++) {
            const uint8_t m = static_cast<uint8_t>((mask >> (2 * h)) & 3);
            if (m == 0) {
                continue;
            }
            const Ch c0 = a.swz[2 * h];
            const Ch c1 = a.swz[2 * h + 1];
            const bool need0 = (m & 1) != 0, need1 = (m & 2) != 0;
            if ((need0 && c0 > Ch::W) || (need1 && c1 > Ch::W)) {
                direct = false;
                break;
            }
            if (need0 && need1) {
                if (static_cast<int>(c0) / 2 != static_cast<int>(c1) / 2 ||
                    (c0 != c1 && (static_cast<int>(c0) & 1) != 0)) {
                    direct = false; // different pairs, or the yx order the table lacks
                    break;
                }
            }
        }
        if (direct) {
            for (int h = 0; h < 2; h++) {
                const uint8_t m = static_cast<uint8_t>((mask >> (2 * h)) & 3);
                if (m == 0) {
                    continue;
                }
                const Ch c0 = a.swz[2 * h];
                const Ch c1 = a.swz[2 * h + 1];
                const Ch lead = (m & 1) ? c0 : c1;
                const Reg pair = a.quad.Pair(static_cast<int>(lead) / 2);
                Swizzle s = XXXX;
                if (m == 3 && c0 != c1) {
                    s = XYZW;
                } else if ((static_cast<int>(lead) & 1) != 0) {
                    s = YYYY;
                }
                if (a.mod == Mod::None) {
                    out->MoveInternal({dst->Pair(h), m}, {pair, s});
                } else {
                    out->Vec(VecOp::Mul, {dst->Pair(h), m}, {pair, s, a.mod}, ConstOne);
                }
            }
            return;
        }
        Materialize(I2, a);
        Store(*dst, mask, I2);
        return;
    }
    case OpCode::Id::FLR: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        const Src sa = Materialize(I0, src1());
        out->Vec(VecOp::Frc, {I1, 0xf}, sa, sa);                   // a - floor(a)
        out->Vec(VecOp::Add, {I2, 0xf}, {I1, XYZW, Mod::Neg}, sa); // a - frac
        Store(*dst, mask, I2);
        return;
    }
    case OpCode::Id::RCP:
    case OpCode::Id::RSQ:
    case OpCode::Id::EX2:
    case OpCode::Id::LG2: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        const Operand a = src1();
        const Ch ch = a.swz[0];
        if (ch > Ch::W) {
            Refuse("complex op on a constant channel");
        }
        const Reg pair = a.quad.Pair(static_cast<int>(ch) / 2);
        const uint8_t comp = static_cast<uint8_t>(static_cast<int>(ch) & 1);
        Pred pred = Pred::None;
        if (op == OpCode::Id::RCP) {
            // The GLSL emitter's guard: 1/0 leaves the register alone.
            out->Test(Cond::Ne, 2, {pair}, comp, ConstZero, true);
            pred = Pred::P2;
        } else if (op == OpCode::Id::RSQ) {
            Cond cond = Cond::Gt;
            if (a.mod == Mod::Neg) {
                cond = Cond::Lt;
            } else if (a.mod == Mod::Abs) {
                cond = Cond::Ne;
            } else if (a.mod == Mod::NegAbs) {
                return; // never positive
            }
            out->Test(cond, 2, {pair}, comp, ConstZero, true);
            pred = Pred::P2;
        }
        const CompOp cop = op == OpCode::Id::RCP   ? CompOp::Rcp
                           : op == OpCode::Id::RSQ ? CompOp::Rsq
                           : op == OpCode::Id::EX2 ? CompOp::Exp
                                                   : CompOp::Log;
        for (int h = 0; h < 2; h++) {
            const uint8_t m = static_cast<uint8_t>((mask >> (2 * h)) & 3);
            if (m != 0) {
                out->Comp(cop, {dst->Pair(h), m}, {pair, XYZW, a.mod}, comp, pred);
            }
        }
        return;
    }
    case OpCode::Id::SGE:
    case OpCode::Id::SGEI:
    case OpCode::Id::SLT:
    case OpCode::Id::SLTI: {
        const auto dst = Dest(instr.common.dest.Value());
        if (!dst || mask == 0) {
            return;
        }
        // Both operands first: one may be a uniform read (a load and a wait), and no
        // internal register may be live across that wait.
        const Operand a = src1();
        const Operand b = src2();
        Materialize(I0, a);
        Materialize(I1, b);
        const Cond cond = (op == OpCode::Id::SGE || op == OpCode::Id::SGEI) ? Cond::Ge : Cond::Lt;
        for (int lane = 0; lane < 4; lane++) {
            if ((mask & (1u << lane)) == 0) {
                continue;
            }
            const Dst d{dst->Pair(lane / 2), static_cast<uint8_t>(1u << (lane & 1))};
            out->Test(cond, 2, {I0}, static_cast<uint8_t>(lane), {I1}, false);
            out->MoveInternal(d, ConstZero);
            out->MoveInternal(d, ConstOne, Pred::P2);
        }
        return;
    }
    case OpCode::Id::CMP: {
        using CompareOp = Instruction::Common::CompareOpType::Op;
        const auto cond = [](CompareOp c) -> Cond {
            switch (c) {
            case CompareOp::Equal:
                return Cond::Eq;
            case CompareOp::NotEqual:
                return Cond::Ne;
            case CompareOp::LessThan:
                return Cond::Lt;
            case CompareOp::LessEqual:
                return Cond::Le;
            case CompareOp::GreaterThan:
                return Cond::Gt;
            case CompareOp::GreaterEqual:
                return Cond::Ge;
            default:
                Refuse("unknown compare");
            }
        };
        const Cond cx = cond(instr.common.compare_op.x.Value());
        const Cond cy = cond(instr.common.compare_op.y.Value());
        // Both operands first: one may be a uniform read (a load and a wait), and no
        // internal register may be live across that wait.
        const Operand a = src1();
        const Operand b = src2();
        Materialize(I0, a);
        Materialize(I1, b);
        out->Test(cx, 0, {I0}, 0, {I1}, false);
        out->Test(cy, 1, {I0}, 1, {I1}, false);
        return;
    }
    case OpCode::Id::MOVA: {
        // a0.x/y = (int) src.x/y when within [-128, 127], else 0 (the GLSL helper's rule),
        // as int16 words the address arithmetic reads. In psp2cgc's shape (2026-09-06):
        // the pack reads the unified register directly and is never predicated (the
        // compiler never packs from an internal register nor under a predicate; the
        // console showed wrong glyph indices with both), the out-of-range case zeroes
        // the result through a predicated `or.u32`, the compiler's own idiom.
        const Operand a = src1();
        for (int lane = 0; lane < 2; lane++) {
            if (!swz.DestComponentEnabled(lane)) {
                continue;
            }
            const Reg areg = Reg::V(lane == 0 ? TmpA0X : TmpA0Y);
            const Ch ch = a.swz[lane];
            if (ch > Ch::W || a.mod != Mod::None) {
                // A constant channel or a modifier: through an internal register.
                const Src sa = Materialize(I0, a);
                out->Bitwise(BwOp::Or, areg, IOp::Imm(0), IOp::Imm(0));
                out->Test(Cond::Ge, 2, sa, static_cast<uint8_t>(lane), Data(DataNeg128), true);
                out->Test(Cond::Le, 2, sa, static_cast<uint8_t>(lane), Data(Data127), true,
                          Pred::P2);
                out->PackS16(areg, I0, static_cast<uint8_t>(lane), Pred::P2);
                continue;
            }
            const Reg pair = a.quad.Pair(static_cast<int>(ch) / 2);
            const uint8_t comp = static_cast<uint8_t>(static_cast<int>(ch) & 1);
            out->PackS16(areg, pair, comp);
            out->Test(Cond::Lt, 2, {pair}, comp, Data(DataNeg128), true);
            out->Bitwise(BwOp::Or, areg, IOp::Imm(0), IOp::Imm(0), Pred::P2);
            out->Test(Cond::Gt, 2, {pair}, comp, Data(Data127), true);
            out->Bitwise(BwOp::Or, areg, IOp::Imm(0), IOp::Imm(0), Pred::P2);
        }
        return;
    }
    default:
        Refuse(std::string("unhandled arithmetic instruction ") + info.name);
    }
}

void VertexEmitter::EmitMad(const Instruction& instr, const SwizzlePattern& swz) {
    const bool inverted = instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI;
    const uint8_t mask = DestMask(swz);
    const auto dst = Dest(instr.mad.dest.Value());
    if (!dst || mask == 0) {
        return;
    }
    const Operand a =
        Source(instr.mad.GetSrc1(inverted), 0, SelectorSwizzle(swz, 1), swz.negate_src1);
    const Operand b =
        Source(instr.mad.GetSrc2(inverted), !inverted * instr.mad.address_register_index,
               SelectorSwizzle(swz, 2), swz.negate_src2);
    const Operand c =
        Source(instr.mad.GetSrc3(inverted), inverted * instr.mad.address_register_index,
               SelectorSwizzle(swz, 3), swz.negate_src3);
    const Src sb = Materialize(I0, b);
    const Src sc = Materialize(I1, c);
    // VMAD's first source only has the table swizzles.
    const Src sa = InVec4Table(a.swz) ? Direct(a) : Materialize(I2, a);
    out->Mad({I2, 0xf}, sa, sb, sc, true);
    Store(*dst, mask, I2);
}

uint32_t VertexEmitter::EmitInstruction(uint32_t offset) {
    const Instruction instr = {req.code[offset]};
    const OpCode::Info info = instr.opcode.Value().GetInfo();
    // A hoisted instruction is emitted into the secondary program, its temporary
    // destination an SA home; the first primary read of that temporary copies the home
    // into the quad (Source), the first primary write ends the hoisted lanes (Dest).
    const bool hoist = offset < hoist_at.size() && hoist_at[offset] != 0;
    if (const auto flush = hoist_flush.find(offset); flush != hoist_flush.end()) {
        // Hoisted lanes this region's primary code writes: into the quad before the branch,
        // so every path after it finds them where NotePrimaryWrite will say they are.
        for (uint32_t t = 0; t < 16; t++) {
            if (const uint8_t lanes = static_cast<uint8_t>(flush->second[t] & hoist_uncopied[t])) {
                CopyHoisted(t, lanes);
            }
        }
    }
    const auto begin_hoist = [&](const DestRegister& dest) {
        hoisting = true;
        out = &sec;
        const uint32_t t = static_cast<uint32_t>(dest.GetIndex()) & 15;
        if (hoist_at[offset] == 2 && hoist_sa[t] >= 0) {
            // A second value for a lane of this temporary: a fresh home, so readers of the
            // first, emitted earlier into the primary, keep the home they were given. The
            // lanes still current move over first, the instruction itself reads them there.
            if (sa_top + 4 > SaLimit) {
                Refuse("no secondary attributes left for a hoisted temporary");
            }
            const Quad old{Bank::SecAttr, static_cast<uint8_t>(hoist_sa[t])};
            const Quad fresh{Bank::SecAttr, static_cast<uint8_t>(sa_top)};
            sa_top += 4;
            HoistCopy(sec, fresh, hoist_lanes[t], old, hoist_swz[t]);
            hoist_sa[t] = static_cast<int>(fresh.base);
            hoist_swz[t] = XYZW;
            hoist_alias[t] = false;
        }
    };
    const auto end_hoist = [&](const DestRegister& dest, uint8_t mask) {
        hoisting = false;
        out = &e;
        const uint32_t t = static_cast<uint32_t>(dest.GetIndex()) & 15;
        hoist_lanes[t] |= mask;
        hoist_uncopied[t] |= mask;
        hoisted_count++;
    };
    if (info.type == OpCode::Type::Arithmetic) {
        const SwizzlePattern swz{req.swizzle[instr.common.operand_desc_id]};
        if (hoist_at[offset] == 3) {
            // The whole register from a prefetched uniform: the uniform's quad is its home.
            const uint32_t t = static_cast<uint32_t>(instr.common.dest.Value().GetIndex()) & 15;
            const int n = AliasUniform(instr, swz);
            if (n < 0) {
                Refuse("planned alias is not one");
            }
            hoist_sa[t] = uniform_sa[n];
            hoist_swz[t] = SelectorSwizzle(swz, 1);
            hoist_alias[t] = true;
            hoist_lanes[t] = 0xf;
            hoist_uncopied[t] = 0xf;
            hoisted_count++;
            return offset + 1;
        }
        if (hoist) {
            begin_hoist(instr.common.dest.Value());
            EmitArithmetic(instr, swz);
            end_hoist(instr.common.dest.Value(), DestMask(swz));
            return offset + 1;
        }
        EmitArithmetic(instr, swz);
        NotePrimaryWrite(instr.common.dest.Value(), DestMask(swz),
                         instr.opcode.Value().EffectiveOpCode());
        return offset + 1;
    }
    if (info.type == OpCode::Type::MultiplyAdd) {
        const auto op = instr.opcode.Value().EffectiveOpCode();
        if (op != OpCode::Id::MAD && op != OpCode::Id::MADI) {
            Refuse("unhandled multiply-add instruction");
        }
        const SwizzlePattern swz{req.swizzle[instr.mad.operand_desc_id]};
        if (hoist) {
            begin_hoist(instr.mad.dest.Value());
            EmitMad(instr, swz);
            end_hoist(instr.mad.dest.Value(), DestMask(swz));
            return offset + 1;
        }
        EmitMad(instr, swz);
        NotePrimaryWrite(instr.mad.dest.Value(), DestMask(swz), op);
        return offset + 1;
    }

    const auto& flow = instr.flow_control;
    switch (instr.opcode.Value()) {
    case OpCode::Id::NOP:
        return offset + 1;

    case OpCode::Id::END: {
        const Context& ctx = contexts.back();
        Branch(Pred::None, end_label);
        // Main stops here unless a jump still targets something further on; a subroutine
        // range keeps going (its remaining instructions may be reached by a jump too).
        if (contexts.size() == 1) {
            for (const auto& [at, label] : ctx.labels) {
                if (at > offset && !Bound(label)) {
                    return offset + 1;
                }
            }
            return ctx.end;
        }
        return offset + 1;
    }

    case OpCode::Id::JMPC: {
        const int target = LabelAt(flow.dest_offset);
        CondBranch(flow, target, true);
        return offset + 1;
    }
    case OpCode::Id::JMPU: {
        const auto known = KnownBool(flow.bool_uniform_id);
        if (!known) {
            Refuse("bool uniform not baked");
        }
        const bool invert = (flow.num_instructions & 1) != 0;
        if (*known != invert) {
            Branch(Pred::None, LabelAt(flow.dest_offset));
        }
        return offset + 1;
    }

    case OpCode::Id::CALL:
    case OpCode::Id::CALLC:
    case OpCode::Id::CALLU: {
        int skip = -1;
        if (instr.opcode.Value() == OpCode::Id::CALLC) {
            skip = NewLabel();
            CondBranch(flow, skip, false);
        } else if (instr.opcode.Value() == OpCode::Id::CALLU) {
            const auto known = KnownBool(flow.bool_uniform_id);
            if (!known) {
                Refuse("bool uniform not baked");
            }
            if (!*known) {
                return offset + 1;
            }
        }
        if (contexts.size() >= 8) {
            Refuse("call depth");
        }
        const uint32_t begin = flow.dest_offset;
        const uint32_t end = begin + flow.num_instructions;
        contexts.push_back({begin, end, {}});
        region_depth += skip >= 0;
        EmitRange(begin, end);
        region_depth -= skip >= 0;
        for (const auto& [at, label] : contexts.back().labels) {
            if (!Bound(label)) {
                Refuse("jump into code the subroutine never reaches");
            }
        }
        contexts.pop_back();
        if (skip >= 0) {
            Bind(skip);
        }
        return offset + 1;
    }

    case OpCode::Id::IFU:
    case OpCode::Id::IFC: {
        const uint32_t if_begin = offset + 1;
        const uint32_t else_begin = flow.dest_offset;
        const uint32_t endif = else_begin + flow.num_instructions;
        if (instr.opcode.Value() == OpCode::Id::IFU) {
            const auto known = KnownBool(flow.bool_uniform_id);
            if (!known) {
                Refuse("bool uniform not baked");
            }
            if (*known) {
                EmitRange(if_begin, else_begin);
            } else if (flow.num_instructions != 0) {
                EmitRange(else_begin, endif);
            }
            return endif;
        }
        const int else_label = NewLabel();
        CondBranch(flow, else_label, false);
        region_depth++;
        EmitRange(if_begin, else_begin);
        if (flow.num_instructions != 0) {
            const int end = NewLabel();
            Branch(Pred::None, end);
            Bind(else_label);
            EmitRange(else_begin, endif);
            Bind(end);
        } else {
            Bind(else_label);
        }
        region_depth--;
        return endif;
    }

    case OpCode::Id::LOOP: {
        if (loop_depth >= 2 || int_sa < 0) {
            Refuse("loop nesting");
        }
        const uint8_t iq = static_cast<uint8_t>(int_sa + 4 * (flow.int_uniform_id & 3));
        const Reg count = Reg::V(loop_depth == 0 ? TmpCount0 : TmpCount1);
        // aL = i.y; count = 0; while (count <= i.x) { body; aL += i.z; count++ }
        e.MoveInternal({Reg::V(TmpAlF), 0x1}, {Reg::Sa(iq), YYYY});
        e.PackS16(Reg::V(TmpAlS16), Reg::V(TmpAlF), 0);
        e.MoveInternal({count, 0x1}, ConstZero);
        const int top = NewLabel();
        const int exit = NewLabel();
        Bind(top);
        e.Test(Cond::Gt, 2, {count}, 0, {Reg::Sa(iq)}, true);
        Branch(Pred::P2, exit);
        loop_depth++;
        EmitRange(offset + 1, flow.dest_offset + 1);
        loop_depth--;
        e.Vec(VecOp::Add, {Reg::V(TmpAlF), 0x1}, {Reg::V(TmpAlF), XYZW},
              {Reg::Sa(static_cast<uint8_t>(iq + 2)), XXXX});
        e.PackS16(Reg::V(TmpAlS16), Reg::V(TmpAlF), 0);
        e.Vec(VecOp::Add, {count, 0x1}, {count, XYZW}, ConstOne);
        Branch(Pred::None, top);
        Bind(exit);
        return flow.dest_offset + 1;
    }

    default:
        Refuse(std::string("unhandled instruction ") + info.name);
    }
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// The uniform-only hoist

void VertexEmitter::NotePrimaryWrite(const DestRegister& dest, uint8_t mask, OpCode::Id op) {
    if (op == OpCode::Id::MOVA || op == OpCode::Id::CMP ||
        dest.GetRegisterType() != RegisterType::Temporary) {
        return;
    }
    const uint32_t t = static_cast<uint32_t>(dest.GetIndex()) & 15;
    hoist_lanes[t] &= static_cast<uint8_t>(~mask);
    hoist_uncopied[t] &= static_cast<uint8_t>(~mask);
}

void VertexEmitter::CopyHoisted(uint32_t t, uint8_t lanes) {
    // The SA home's lanes into the primary's quad, a pair at a time. Inside a conditional
    // region the copy runs on one path only, so the lanes stay marked uncopied and a later
    // reader copies again; outside, the quad holds them from here on.
    const Quad home{Bank::SecAttr, static_cast<uint8_t>(hoist_sa[t])};
    HoistCopy(e, *temp_quad[t], lanes, home, hoist_swz[t]);
    if (region_depth == 0) {
        hoist_uncopied[t] &= static_cast<uint8_t>(~lanes);
    }
}

// dst's lanes from src read through swz, a pair at a time. Both moves name the source by its
// quad base, the second pair through the swizzle's z and w entries, which is the form Store
// uses and the passes that rebase pair moves expect.
void VertexEmitter::HoistCopy(Ir::Builder& b, const Quad& dst, uint8_t lanes, const Quad& src,
                              const Swizzle& swz) {
    if (lanes & 0x3) {
        b.MoveInternal({dst.Pair(0), static_cast<uint8_t>(lanes & 0x3)}, {src.Pair(0), swz});
    }
    if (lanes & 0xc) {
        b.MoveInternal({dst.Pair(1), static_cast<uint8_t>(lanes >> 2)},
                       {src.Pair(0), Swizzle{swz[2], swz[3], swz[2], swz[3]}});
    }
}

// The uniform a hoisted MOV of a whole register from a prefetched uniform copies, or -1: such
// a MOV needs no instruction, the temporary's home becomes the uniform's quad.
int VertexEmitter::AliasUniform(const Instruction& instr, const SwizzlePattern& swz) const {
    const OpCode::Info info = instr.opcode.Value().GetInfo();
    if (info.type != OpCode::Type::Arithmetic ||
        instr.opcode.Value().EffectiveOpCode() != OpCode::Id::MOV || DestMask(swz) != 0xf ||
        swz.negate_src1 || instr.common.address_register_index != 0) {
        return -1;
    }
    const SourceRegister src = instr.common.GetSrc1(false);
    if (src.GetRegisterType() != RegisterType::FloatUniform) {
        return -1;
    }
    // Only a swizzle whose composition with any later read stays in the hardware's tables:
    // the identity leaves the read's own, a broadcast turns it into a broadcast.
    const Swizzle read = SelectorSwizzle(swz, 1);
    const bool broadcast = read[0] == read[1] && read[1] == read[2] && read[2] == read[3];
    if (read != XYZW && !broadcast) {
        return -1;
    }
    const uint32_t n = static_cast<uint32_t>(src.GetIndex()) & 127;
    return uniform_sa[n] >= 0 ? static_cast<int>(n) : -1;
}

void VertexEmitter::PlanHoist() {
    // Walks the program in the order the emitter emits it, deciding per instruction. The
    // state per temporary lane: whether an earlier hoisted instruction wrote it and nothing
    // in the primary has since (hoisted), whether anything wrote it (defined), whether a
    // hoisted instruction ever wrote it (ever). A lane hoisted a second time gets a fresh SA
    // home for its temporary: the first home holds the first value for the primary readers
    // already emitted against it, the fresh one the second.
    //
    // Conditional regions, an IFC's two arms and a CALLC's subroutine, run on one path only;
    // the secondary runs unconditionally. Inside one, a hoisted write is allowed only to a
    // lane that was undefined at the region's entry and is written once in the whole region
    // (the two arms included), so no path can observe a value the hoist replaced. Lanes
    // hoisted before the region that the region's primary code writes are copied into the
    // quad before the branch, so the paths agree on where the value lives afterwards.
    // Straight-line inlining, a CALL or a bool-known CALLU or IFU, opens no region. Anything
    // else - loops, jumps - ends the walk; what was decided before stays valid.
    hoist_at.assign(req.code_size, 0);
    hoist_flush.clear();
    hoist_sa.fill(-1);
    hoist_swz.fill(XYZW);
    hoist_alias.fill(false);
    hoist_lanes.fill(0);
    hoist_uncopied.fill(0);
    // Opt-in until the console has shown the declared-SA fix holds: `usse_sechoist`.
    static const bool hoist_on = GxmFlag("usse_sechoist");
    if (!hoist_on || level < 2) {
        return;
    }
    using Lanes = std::array<uint8_t, 16>;
    Lanes hoisted{}, defined{}, ever{};
    // Lanes the primary has read since the hoisted write that produced them: a second
    // hoisted write to such a lane needs a fresh home, one to a lane nothing has read since
    // can overwrite in place (the secondary runs in order; only primary readers, emitted
    // against the earlier home, would see the later value).
    Lanes pread{};
    std::array<uint8_t, 16> home{}; // 0 none, 1 an SA quad of its own, 2 a uniform's quad
    uint32_t reserved = 0;          // SA quads the hoisted temporaries would take
    std::vector<Lanes> forbidden;   // per open region: lanes no hoisted write may touch
    std::vector<uint32_t> entered;
    int depth = 0;

    const auto read_lanes = [](const Swizzle& swz) {
        uint8_t read = 0;
        for (int l = 0; l < 4; l++) {
            if (swz[l] <= Ch::W) {
                read |= static_cast<uint8_t>(1u << static_cast<int>(swz[l]));
            }
        }
        return read;
    };
    const auto note_read = [&](const SourceRegister& reg, const Swizzle& swz) {
        if (reg.GetRegisterType() == RegisterType::Temporary) {
            pread[static_cast<uint32_t>(reg.GetIndex()) & 15] |= read_lanes(swz);
        }
    };
    const auto source_ok = [&](const SourceRegister& reg, uint32_t addr, const Swizzle& swz) {
        if (addr != 0) {
            return false;
        }
        const uint32_t index = static_cast<uint32_t>(reg.GetIndex());
        switch (reg.GetRegisterType()) {
        case RegisterType::FloatUniform:
            return uniform_sa[index & 127] >= 0;
        case RegisterType::Temporary: {
            const uint8_t read = read_lanes(swz);
            return (hoisted[index & 15] & read) == read;
        }
        default:
            return false;
        }
    };
    // The destination side of a decision: whether the write may be hoisted, and the state
    // after it either way. The value is what hoist_at records: 1 hoisted into the
    // temporary's home, 2 into a fresh home (a lane written was hoisted before, or the home is
    // a uniform's quad), 3 an alias of a uniform's quad, no instruction at all.
    const auto decide = [&](const DestRegister& dest, uint8_t mask, bool sources_ok,
                            bool alias) -> uint8_t {
        if (dest.GetRegisterType() != RegisterType::Temporary) {
            return 0;
        }
        const uint32_t t = static_cast<uint32_t>(dest.GetIndex()) & 15;
        bool ok =
            sources_ok && mask != 0 && (forbidden.empty() || (forbidden.back()[t] & mask) == 0);
        const bool fresh = home[t] == 2 || (ever[t] & mask & pread[t]) != 0;
        uint8_t kind = 0;
        if (ok && alias) {
            kind = 3;
            home[t] = 2;
        } else if (ok) {
            // A quad of its own: the first, or a fresh one.
            if (home[t] != 1 || fresh) {
                ok = sa_top + (reserved + 1) * 4 <= SaLimit;
                reserved += ok;
            }
            if (ok) {
                kind = fresh ? 2 : 1;
                home[t] = 1;
            }
        }
        if (ok) {
            ever[t] |= mask;
            hoisted[t] |= mask;
            pread[t] &= static_cast<uint8_t>(~mask);
        } else {
            hoisted[t] &= static_cast<uint8_t>(~mask);
        }
        defined[t] |= mask;
        return kind;
    };

    // Every temporary lane written in [begin, end) as the emitter inlines it: once, and more
    // than once. False when the range holds flow control the walk does not model.
    const std::function<bool(uint32_t, uint32_t, Lanes&, Lanes&)> scan_writes =
        [&](uint32_t begin, uint32_t end, Lanes& once, Lanes& multi) {
            const auto note = [&](const DestRegister& dest, uint8_t mask) {
                if (dest.GetRegisterType() != RegisterType::Temporary) {
                    return;
                }
                const uint32_t t = static_cast<uint32_t>(dest.GetIndex()) & 15;
                multi[t] |= static_cast<uint8_t>(once[t] & mask);
                once[t] |= mask;
            };
            for (uint32_t offset = begin; offset < end && offset < req.code_size; offset++) {
                const Instruction instr = {req.code[offset]};
                const OpCode::Info info = instr.opcode.Value().GetInfo();
                const auto op = instr.opcode.Value().EffectiveOpCode();
                if (info.type == OpCode::Type::Arithmetic) {
                    if (op != OpCode::Id::MOVA && op != OpCode::Id::CMP) {
                        note(instr.common.dest.Value(),
                             DestMask(SwizzlePattern{req.swizzle[instr.common.operand_desc_id]}));
                    }
                    continue;
                }
                if (info.type == OpCode::Type::MultiplyAdd) {
                    note(instr.mad.dest.Value(),
                         DestMask(SwizzlePattern{req.swizzle[instr.mad.operand_desc_id]}));
                    continue;
                }
                const auto& flow = instr.flow_control;
                switch (op) {
                case OpCode::Id::NOP:
                    continue;
                case OpCode::Id::END:
                    return true;
                case OpCode::Id::CALL:
                case OpCode::Id::CALLC:
                    if (!scan_writes(flow.dest_offset, flow.dest_offset + flow.num_instructions,
                                     once, multi)) {
                        return false;
                    }
                    continue;
                case OpCode::Id::CALLU: {
                    const auto known = KnownBool(flow.bool_uniform_id);
                    if (!known) {
                        return false;
                    }
                    if (*known &&
                        !scan_writes(flow.dest_offset, flow.dest_offset + flow.num_instructions,
                                     once, multi)) {
                        return false;
                    }
                    continue;
                }
                case OpCode::Id::IFC:
                case OpCode::Id::IFU: {
                    const uint32_t endif = flow.dest_offset + flow.num_instructions;
                    if (op == OpCode::Id::IFU && !KnownBool(flow.bool_uniform_id)) {
                        return false;
                    }
                    if (!scan_writes(offset + 1, endif, once, multi)) {
                        return false;
                    }
                    offset = endif - 1;
                    continue;
                }
                default:
                    return false;
                }
            }
            return true;
        };

    // A conditional region about to be walked: what to flush before it, what its hoisted
    // writes must avoid.
    const auto enter_region = [&](uint32_t at, uint32_t begin, uint32_t end) {
        Lanes once{}, multi{};
        if (!scan_writes(begin, end, once, multi)) {
            return false;
        }
        Lanes flush{}, forbid{};
        bool any_flush = false;
        for (uint32_t t = 0; t < 16; t++) {
            flush[t] = static_cast<uint8_t>(hoisted[t] & (once[t] | multi[t]));
            any_flush = any_flush || flush[t] != 0;
            forbid[t] = static_cast<uint8_t>(defined[t] | multi[t]);
        }
        if (any_flush) {
            hoist_flush[at] = flush;
        }
        forbidden.push_back(forbid);
        return true;
    };

    const std::function<bool(uint32_t, uint32_t)> walk = [&](uint32_t begin, uint32_t end) {
        for (uint32_t offset = begin; offset < end && offset < req.code_size; offset++) {
            const Instruction instr = {req.code[offset]};
            const OpCode::Info info = instr.opcode.Value().GetInfo();
            const auto op = instr.opcode.Value().EffectiveOpCode();
            if (info.type == OpCode::Type::Arithmetic) {
                const SwizzlePattern swz = {req.swizzle[instr.common.operand_desc_id]};
                const bool inverted = (info.subtype & OpCode::Info::SrcInversed) != 0;
                const auto src1_ok = [&] {
                    return source_ok(instr.common.GetSrc1(inverted),
                                     !inverted * instr.common.address_register_index,
                                     SelectorSwizzle(swz, 1));
                };
                const auto src2_ok = [&] {
                    return source_ok(instr.common.GetSrc2(inverted),
                                     inverted * instr.common.address_register_index,
                                     SelectorSwizzle(swz, 2));
                };
                bool ok = false;
                switch (op) {
                case OpCode::Id::ADD:
                case OpCode::Id::MUL:
                case OpCode::Id::MAX:
                case OpCode::Id::MIN:
                case OpCode::Id::DP3:
                case OpCode::Id::DP4:
                case OpCode::Id::DPH:
                case OpCode::Id::DPHI:
                    ok = src1_ok() && src2_ok();
                    break;
                case OpCode::Id::MOV:
                case OpCode::Id::FLR:
                case OpCode::Id::RCP:
                case OpCode::Id::RSQ:
                case OpCode::Id::EX2:
                case OpCode::Id::LG2:
                    ok = src1_ok();
                    break;
                default:
                    break;
                }
                const bool two_sources = op != OpCode::Id::MOV && op != OpCode::Id::FLR &&
                                         op != OpCode::Id::RCP && op != OpCode::Id::RSQ &&
                                         op != OpCode::Id::EX2 && op != OpCode::Id::LG2 &&
                                         op != OpCode::Id::MOVA;
                const auto note_reads = [&] {
                    note_read(instr.common.GetSrc1(inverted), SelectorSwizzle(swz, 1));
                    if (two_sources) {
                        note_read(instr.common.GetSrc2(inverted), SelectorSwizzle(swz, 2));
                    }
                };
                if (op == OpCode::Id::MOVA || op == OpCode::Id::CMP) {
                    note_reads();
                    continue; // no temporary written, nothing to hoist
                }
                hoist_at[offset] = decide(instr.common.dest.Value(), DestMask(swz), ok,
                                          AliasUniform(instr, swz) >= 0);
                if (hoist_at[offset] == 0) {
                    note_reads();
                }
                continue;
            }
            if (info.type == OpCode::Type::MultiplyAdd) {
                const SwizzlePattern swz = {req.swizzle[instr.mad.operand_desc_id]};
                const bool inverted = op == OpCode::Id::MADI;
                const bool ok =
                    (op == OpCode::Id::MAD || op == OpCode::Id::MADI) &&
                    source_ok(instr.mad.GetSrc1(inverted), 0, SelectorSwizzle(swz, 1)) &&
                    source_ok(instr.mad.GetSrc2(inverted),
                              !inverted * instr.mad.address_register_index,
                              SelectorSwizzle(swz, 2)) &&
                    source_ok(instr.mad.GetSrc3(inverted),
                              inverted * instr.mad.address_register_index, SelectorSwizzle(swz, 3));
                hoist_at[offset] = decide(instr.mad.dest.Value(), DestMask(swz), ok, false);
                if (hoist_at[offset] == 0) {
                    note_read(instr.mad.GetSrc1(inverted), SelectorSwizzle(swz, 1));
                    note_read(instr.mad.GetSrc2(inverted), SelectorSwizzle(swz, 2));
                    note_read(instr.mad.GetSrc3(inverted), SelectorSwizzle(swz, 3));
                }
                continue;
            }
            const auto& flow = instr.flow_control;
            switch (op) {
            case OpCode::Id::NOP:
                continue;
            case OpCode::Id::END:
                return true;
            case OpCode::Id::CALL:
            case OpCode::Id::CALLU:
            case OpCode::Id::CALLC: {
                if (op == OpCode::Id::CALLU) {
                    const auto known = KnownBool(flow.bool_uniform_id);
                    if (!known) {
                        return false;
                    }
                    if (!*known) {
                        continue;
                    }
                }
                // A subroutine reached twice would emit its hoisted instructions twice into
                // one secondary program, the second overwriting the homes the first's readers
                // use; the walk ends before the second entry.
                const uint32_t sub = flow.dest_offset;
                const uint32_t sub_end = sub + flow.num_instructions;
                if (depth >= 8 || std::find(entered.begin(), entered.end(), sub) != entered.end()) {
                    return false;
                }
                entered.push_back(sub);
                const bool conditional = op == OpCode::Id::CALLC;
                if (conditional && !enter_region(offset, sub, sub_end)) {
                    return false;
                }
                depth++;
                const bool more = walk(sub, sub_end);
                depth--;
                if (conditional) {
                    forbidden.pop_back();
                }
                if (!more) {
                    return false;
                }
                continue;
            }
            case OpCode::Id::IFU:
            case OpCode::Id::IFC: {
                const uint32_t if_begin = offset + 1;
                const uint32_t else_begin = flow.dest_offset;
                const uint32_t endif = else_begin + flow.num_instructions;
                if (op == OpCode::Id::IFU) {
                    const auto known = KnownBool(flow.bool_uniform_id);
                    if (!known) {
                        return false;
                    }
                    const bool more = *known ? walk(if_begin, else_begin) : walk(else_begin, endif);
                    if (!more) {
                        return false;
                    }
                    offset = endif - 1;
                    continue;
                }
                if (!enter_region(offset, if_begin, endif)) {
                    return false;
                }
                const bool more = walk(if_begin, else_begin) && walk(else_begin, endif);
                forbidden.pop_back();
                if (!more) {
                    return false;
                }
                offset = endif - 1;
                continue;
            }
            default:
                return false; // loops and jumps end the walk
            }
        }
        return true;
    };
    walk(req.main_offset, req.code_size);
}

// Prologue and epilogue

void VertexEmitter::Prologue() {
    // The components a loader does not supply: (0, 0, 0, 1), as the PICA fills them.
    for (uint32_t i = 0; i < 16; i++) {
        if (input_pa[i] < 0) {
            continue;
        }
        uint32_t comps = req.reg_components[i];
        if (comps == 0 || comps > 4) {
            comps = 4;
        }
        const Quad q{Bank::PrimAttr, static_cast<uint8_t>(input_pa[i])};
        switch (comps) {
        case 1:
            e.MoveInternal({q.Pair(0), 0x2}, ConstZero);
            [[fallthrough]];
        case 2:
            e.MoveInternal({q.Pair(1), 0x3}, ConstZeroOne);
            break;
        case 3:
            e.MoveInternal({q.Pair(1), 0x2}, ConstOne);
            break;
        default:
            break;
        }
    }
    // Output attributes the program never assigns: (0, 0, 0, 1), as the wrappers write it.
    for (uint32_t k = 0; k < req.num_outputs && k < 16; k++) {
        if ((scan.attrs_written & (1u << k)) == 0 && attr_quad[k]) {
            e.MoveInternal({attr_quad[k]->Pair(0), 0x3}, {Reg::C(0), XYZW});
            e.MoveInternal({attr_quad[k]->Pair(1), 0x3}, ConstZeroOne);
        }
    }
    // Temporaries read before they are written start at (0, 0, 0, 1) too.
    for (uint32_t t = 0; t < 16; t++) {
        if (!temp_quad[t] || scan.temp_init[t] == 0) {
            continue;
        }
        const uint8_t m = scan.temp_init[t];
        if (m & 0x3) {
            e.MoveInternal({temp_quad[t]->Pair(0), static_cast<uint8_t>(m & 0x3)},
                           {Reg::C(0), XYZW});
        }
        if (m & 0xc) {
            e.MoveInternal({temp_quad[t]->Pair(1), static_cast<uint8_t>(m >> 2)}, ConstZeroOne);
        }
    }
    if (scan.cc) {
        // The condition codes start false.
        e.Test(Cond::Ne, 0, ConstZero, 0, ConstZero, true);
        e.Test(Cond::Ne, 1, ConstZero, 0, ConstZero, true);
    }
    if (scan.a0x) {
        e.Bitwise(BwOp::Or, Reg::V(TmpA0X), IOp::Imm(0), IOp::Imm(0));
    }
    if (scan.a0y) {
        e.Bitwise(BwOp::Or, Reg::V(TmpA0Y), IOp::Imm(0), IOp::Imm(0));
    }
    if (scan.al) {
        e.Bitwise(BwOp::Or, Reg::V(TmpAlS16), IOp::Imm(0), IOp::Imm(0));
    }
}

std::optional<Src> VertexEmitter::SemanticSrc(Semantic s) const {
    const VsSemantic& m = req.semantics[s];
    if (m.attribute >= req.num_outputs || m.attribute >= 16 || !attr_quad[m.attribute]) {
        return std::nullopt;
    }
    const Quad& q = *attr_quad[m.attribute];
    return Src{q.Pair(m.component / 2), (m.component & 1) ? YYYY : XXXX};
}

void VertexEmitter::GatherFour(Reg internal, Semantic first) {
    // One four-wide read when the four come from one attribute in order.
    const VsSemantic& m0 = req.semantics[first];
    bool straight = m0.attribute < req.num_outputs && m0.component == 0;
    for (int i = 1; i < 4 && straight; i++) {
        const VsSemantic& mi = req.semantics[first + i];
        straight = mi.attribute == m0.attribute && mi.component == static_cast<uint8_t>(i);
    }
    if (straight && attr_quad[m0.attribute]) {
        e.Vec(VecOp::Mul, {internal, 0xf}, {attr_quad[m0.attribute]->Base(), XYZW}, ConstOne);
        return;
    }
    for (int i = 0; i < 4; i++) {
        const auto src = SemanticSrc(static_cast<Semantic>(first + i));
        e.MoveInternal({internal, static_cast<uint8_t>(1u << i)}, src ? *src : ConstOne);
    }
}

void VertexEmitter::OutputLane(uint8_t out, int lane, Semantic s) {
    const auto src = SemanticSrc(s);
    e.MoveInternal({Reg::O(out), static_cast<uint8_t>(1u << lane)}, src ? *src : ConstOne);
}

void VertexEmitter::OutputPair(uint8_t out, Semantic s0, Semantic s1) {
    const VsSemantic& m0 = req.semantics[s0];
    const VsSemantic& m1 = req.semantics[s1];
    if (m0.attribute < req.num_outputs && m1.attribute == m0.attribute && attr_quad[m0.attribute] &&
        (m0.component & 1) == 0 && m1.component == m0.component + 1) {
        e.MoveInternal({Reg::O(out), 0x3}, {attr_quad[m0.attribute]->Pair(m0.component / 2), XYZW});
        return;
    }
    // Two lanes no attribute feeds (the emitter's 1.0 fill) go in one pair move: the Smash
    // corpus wrote 477 single output lanes, 385 of them in adjacent pairs (2026-09-07).
    if (!SemanticSrc(s0) && !SemanticSrc(s1)) {
        e.MoveInternal({Reg::O(out), 0x3}, ConstOne);
        return;
    }
    OutputLane(out, 0, s0);
    OutputLane(out, 1, s1);
}

void VertexEmitter::Epilogue() {
    // Position: the PICA depth into clip z so the interpolated depth is the PICA depth,
    // after the wrappers' nudge of a z barely past a clip boundary and the viewport flip.
    GatherFour(I0, SemPositionX);
    e.Comp(CompOp::Rcp, {I1, 0x1}, {I0}, 3);              // 1/w
    e.Vec(VecOp::Mul, {I1, 0x2}, {I0, ZZZZ}, {I1, XXXX}); // ndc z
    e.Test(Cond::Gt, 2, {I1}, 1, ConstZero, true);
    e.Test(Cond::Lt, 2, {I1}, 1, Data(DataEpsilon), true, Pred::P2);
    e.MoveInternal({I0, 0x4}, ConstZero, Pred::P2); // z = 0
    e.Test(Cond::Lt, 2, {I1}, 1, Data(DataNeg1), true);
    e.Test(Cond::Gt, 2, {I1}, 1, Data(DataNeg1Eps), true, Pred::P2);
    e.Vec(VecOp::Mul, {I0, 0x4}, {I0, WWWW, Mod::Neg}, ConstOne, Pred::P2); // z = -w
    e.Test(Cond::Gt, 2, {Reg::Sa(SaFlipViewport)}, 0, ConstHalf, false);
    e.Vec(VecOp::Mul, {I0, 0x2}, {I0, YYYY, Mod::Neg}, ConstOne, Pred::P2); // y = -y
    e.Vec(VecOp::Mul, {I1, 0x2}, {I0, ZZZZ}, {I1, XXXX});                   // z / w again
    e.Vec(VecOp::Mul, {I1, 0x2}, {I1, YYYY}, {Reg::Sa(0), YYYY});           // * depth_scale (sa1)
    e.Vec(VecOp::Add, {I1, 0x2}, {I1, YYYY}, {Reg::Sa(2), XXXX});           // + depth_offset (sa2)
    e.Vec(VecOp::Mul, {I0, 0x4}, {I1, YYYY}, {I0, WWWW});                   // z = depth * w
    e.MoveInternal({Reg::O(OutPosition), 0x3}, {I0, XYZW});
    e.MoveInternal({Reg::O(OutPosition + 2), 0x3}, {I0, ZWZW});

    // Colour: min(abs(c), 1).
    GatherFour(I1, SemColorR);
    e.Vec(VecOp::Min, {Reg::O(OutColor), 0x3}, {I1, XYZW, Mod::Abs}, ConstOne);
    e.Vec(VecOp::Min, {Reg::O(OutColor + 2), 0x3}, {I1, ZWZW, Mod::Abs}, ConstOne);

    // The quaternion, twice (TEXCOORD5 is its flat copy).
    GatherFour(I2, SemQuaternionX);
    e.MoveInternal({Reg::O(OutNormquat), 0x3}, {I2, XYZW});
    e.MoveInternal({Reg::O(OutNormquat + 2), 0x3}, {I2, ZWZW});
    e.MoveInternal({Reg::O(OutNormquatFlat), 0x3}, {I2, XYZW});
    e.MoveInternal({Reg::O(OutNormquatFlat + 2), 0x3}, {I2, ZWZW});

    OutputPair(OutTexcoord0, SemTexcoord0U, SemTexcoord0V);
    OutputPair(OutTexcoord1, SemTexcoord1U, SemTexcoord1V);
    OutputPair(OutTexcoord2, SemTexcoord2U, SemTexcoord2V);
    OutputLane(OutTexcoord0W, 0, SemTexcoord0W);
    OutputPair(OutView, SemViewX, static_cast<Semantic>(SemViewX + 1));
    OutputLane(OutView + 2, 0, static_cast<Semantic>(SemViewX + 2));
}

// ---------------------------------------------------------------------------------------

std::vector<uint8_t> VertexEmitter::Emit(uint32_t& input_regs) {
    if (req.code.size() < req.code_size || req.main_offset >= req.code_size) {
        Refuse("program extent");
    }
    ScanProgram();
    input_regs = scan.inputs;
    Layout();
    PlanHoist();
    EmitSecondary();

    e.skip_invalid = true;
    e.nosched_override = 0;
    end_label = NewLabel();
    e.Phas(true);
    e.Nop();
    Prologue();
    contexts.push_back({req.main_offset, req.code_size, {}});
    EmitRange(req.main_offset, req.code_size);
    for (const auto& [at, label] : contexts.back().labels) {
        if (!Bound(label)) {
            Refuse("jump into code main never reaches");
        }
    }
    contexts.pop_back();
    // The hoist takes SA quads past the prefetch as it emits (fresh homes, section
    // "hoist"), and the header's secondary register count decides what the hardware
    // allocates: Layout's count left them undeclared, which Vita3K did not mind and the
    // console answered with black screens (2026-09-08).
    gxp.secondary_reg_count = sa_top;
    if (sec.Count() != 0) {
        sec.EndSecondary();
        if (hoisted_count != 0 && std::getenv("USSE_IR_DUMP") != nullptr) {
            std::fprintf(stderr, "hoisted %u instructions into the secondary program\n",
                         hoisted_count);
        }
    }
    const std::size_t epilogue_at = e.Count();
    Bind(end_label);
    Epilogue();
    e.EndVertex();

    // Where the virtual quads may live, in spill order: the temporaries, the primary
    // attributes past the inputs, the output registers until the epilogue writes them.
    e.AddClass({Bank::Temp, 0, 64});
    e.AddClass({Bank::PrimAttr, pa_inputs, PaLimit});
    e.AddClass({Bank::Output, 0, OutScratchEnd, epilogue_at});
    Encoder primary;
    const Ir::LowerResult lowered = e.Lower(primary, level);
    primary.FinishNosched();
    Encoder secondary;
    if (sec.Count() != 0) {
        sec.Lower(secondary);
        secondary.FinishNosched();
    }

    gxp.vertex = true;
    gxp.per_instance = per_instance;
    gxp.vertex_outputs1 = VertexOutputs1;
    gxp.vertex_outputs2 = VertexOutputs2;
    gxp.pa_count = std::max<uint32_t>({static_cast<uint32_t>(pa_inputs), lowered.pa_count, 4u});
    gxp.temp_count = lowered.temp_count;
    gxp.literals = {
        0u, std::bit_cast<uint32_t>(-128.0f), 0x00010000u, std::bit_cast<uint32_t>(127.0f),
        0u, std::bit_cast<uint32_t>(1e-6f),   0u,          std::bit_cast<uint32_t>(-1.00001f),
        0u, std::bit_cast<uint32_t>(-1.0f),   0u,          0x00010001u,
        0u};
    static_assert(DataLiteralCount == 13);
    if (!gxp.buffer_pointer) {
        // No uniform read: the pointer slot stays, as a literal, so the layout is one.
        gxp.literals.insert(gxp.literals.begin(), 0u);
    }
    gxp.primary = primary.Words();
    gxp.secondary = secondary.Words();
    return WriteGxp(gxp);
}

} // Anonymous namespace

std::vector<uint8_t> EmitVertexProgram(const VsRequest& request, uint32_t& input_regs,
                                       std::string* refusal, int level) {
    try {
        VertexEmitter emitter{request, level};
        return emitter.Emit(input_regs);
    } catch (const std::exception& ex) {
        if (refusal) {
            *refusal = ex.what();
        }
        return {};
    }
}

} // namespace GxmRenderer::Usse
