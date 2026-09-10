// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/usse/usse_ir.h"

namespace GxmRenderer::Usse::Ir {

namespace {

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("usse ir: " + what);
}

bool IsVirtual(Reg r) {
    return r.bank == Bank::Virtual;
}


uint32_t QuadOf(Reg r) {
    return r.n / 4;
}

/// A register the instruction reads or writes.
struct Ref {
    Reg* reg;
    bool def;
};

/// The registers an instruction touches, in a fixed order; `hi` of a four-wide move is
/// unused when `lo` is internal, and a load or sample defines a whole quad.
void Refs(Inst& i, std::vector<Ref>& out) {
    out.clear();
    switch (i.op) {
    case Op::Vec:
    case Op::Vec16:
        out.push_back({&i.dst.reg, true});
        out.push_back({&i.src[0].reg, false});
        out.push_back({&i.src[1].reg, false});
        break;
    case Op::Mad:
    case Op::Mad2:
    case Op::CondMove:
        out.push_back({&i.dst.reg, true});
        out.push_back({&i.src[0].reg, false});
        out.push_back({&i.src[1].reg, false});
        out.push_back({&i.src[2].reg, false});
        break;
    case Op::Dot:
        out.push_back({&i.dst.reg, true});
        out.push_back({&i.src[0].reg, false});
        out.push_back({&i.src[1].reg, false});
        break;
    case Op::Comp:
    case Op::MoveInternal:
        out.push_back({&i.dst.reg, true});
        out.push_back({&i.src[0].reg, false});
        break;
    case Op::MoveF32x4:
    case Op::PackF16:
        out.push_back({&i.dst.reg, true});
        out.push_back({&i.reg[0], false});
        if (i.reg[0].bank != Bank::Internal) {
            out.push_back({&i.reg[1], false});
        }
        break;
    case Op::Test:
        out.push_back({&i.src[0].reg, false});
        out.push_back({&i.src[1].reg, false});
        break;
    case Op::Sample2D:
        out.push_back({&i.reg[0], true});
        out.push_back({&i.reg[1], false});
        if (i.b[0]) {
            out.push_back({&i.src[0].reg, false});
        }
        break;
    case Op::Lda32:
        out.push_back({&i.reg[0], true});
        out.push_back({&i.reg[1], false});
        if (i.b[0]) {
            out.push_back({&i.src[0].reg, false});
        }
        break;
    case Op::Bitwise:
        out.push_back({&i.reg[0], true});
        if (!i.iop[0].imm) {
            out.push_back({&i.iop[0].reg, false});
        }
        if (!i.iop[1].imm) {
            out.push_back({&i.iop[1].reg, false});
        }
        break;
    case Op::MadI16:
        out.push_back({&i.reg[0], true});
        out.push_back({&i.reg[1], false});
        out.push_back({&i.src[0].reg, false});
        if (!i.iop[0].imm) {
            out.push_back({&i.iop[0].reg, false});
        }
        break;
    case Op::MadI32:
        out.push_back({&i.reg[0], true});
        out.push_back({&i.reg[1], false});
        out.push_back({&i.src[0].reg, false});
        break;
    case Op::PackS16:
        out.push_back({&i.reg[0], true});
        out.push_back({&i.reg[1], false});
        break;
    case Op::DepthF:
        out.push_back({&i.reg[0], false});
        break;
    default:
        break;
    }
}

/// Whether removing the instruction changes nothing but its destination register.
bool PureWrite(const Inst& i) {
    switch (i.op) {
    case Op::Vec:
    case Op::Vec16:
    case Op::Mad:
    case Op::Mad2:
    case Op::Dot:
    case Op::Comp:
    case Op::MoveF32x4:
    case Op::PackF16:
    case Op::MoveInternal:
    case Op::CondMove:
    case Op::Bitwise:
    case Op::MadI16:
    case Op::MadI32:
    case Op::PackS16:
        return true;
    default:
        return false; // loads and samples have waits paired with them; tests write predicates
    }
}

/// A four-wide load of a unified-store operand into an internal register: what the
/// emitters write to read an operand with a free swizzle.
bool IsInternalLoad(const Inst& i) {
    return i.op == Op::Vec && static_cast<VecOp>(i.u[0]) == VecOp::Mul &&
           i.dst.reg.bank == Bank::Internal && i.dst.mask == 0xf && i.pred == Pred::None &&
           i.src[1].reg.bank == Bank::Const && i.src[1].reg.n == 2 && i.src[1].swz == XXXX &&
           i.src[1].mod == Mod::None && i.src[0].reg.bank != Bank::Internal;
}

bool SameReg(Reg a, Reg b) {
    return a.bank == b.bank && a.n == b.n;
}

const char* OpName(Op op) {
    static const char* names[] = {
        "phas",   "nop",   "nop_after_kill", "wdf",     "wdf_vertex", "kill_unless_p1",
        "depthf", "end",   "end_secondary",  "label",   "br",         "phase_start",
        "vec",    "vec16", "mad",            "dot",     "comp",       "mov_f32x4",
        "pack_f16", "mov_internal", "cond_mov", "test", "sample2d",  "lda32",
        "bitwise", "mad_i16", "mad_i32",     "pack_s16", "mad2",
    };
    const auto n = static_cast<std::size_t>(op);
    return n < sizeof(names) / sizeof(names[0]) ? names[n] : "?";
}

std::string RegName(Reg r) {
    static const char* banks[] = {"r", "pa", "sa", "i", "c", "o", "v", "?"};
    const auto b = static_cast<std::size_t>(r.bank);
    return std::string(b < 7 ? banks[b] : "?") + std::to_string(r.n);
}

std::string SrcName(const Src& s) {
    static const char chans[] = "xyzw01th";
    std::string out = RegName(s.reg) + ".";
    for (int l = 0; l < 4; l++) {
        out += chans[static_cast<int>(s.swz[l]) & 7];
    }
    static const char* mods[] = {"", "-", "|", "-|"};
    return mods[static_cast<int>(s.mod) & 3] + out;
}

/// One line per instruction, for refusal messages, tests and the dump: every field the
/// simulator needs.
std::string Describe(Inst& i) {
    std::vector<Ref> refs;
    Refs(i, refs);
    std::string out = OpName(i.op);
    if (i.op == Op::Br || i.op == Op::Label) {
        out += " L" + std::to_string(i.label);
    }
    for (const Ref& r : refs) {
        out += r.def ? " >" : " ";
        out += RegName(*r.reg);
    }
    static const char* preds[] = {"", "p0", "p1", "p2", "!p0", "!p1"};
    out += " |";
    out += std::string(" pred=") + preds[static_cast<int>(i.pred) % 6];
    out += " dst=" + RegName(i.dst.reg) + ":" + std::to_string(i.dst.mask);
    for (int k = 0; k < 3; k++) {
        out += " s" + std::to_string(k) + "=" + SrcName(i.src[k]);
    }
    out += " r0=" + RegName(i.reg[0]) + " r1=" + RegName(i.reg[1]);
    out += " u=" + std::to_string(i.u[0]) + "," + std::to_string(i.u[1]) + "," + std::to_string(i.u[2]);
    out += " b=" + std::to_string(i.b[0]) + "," + std::to_string(i.b[1]);
    for (int k = 0; k < 2; k++) {
        out += " iop" + std::to_string(k) + "=" +
               (i.iop[k].imm ? "#" + std::to_string(i.iop[k].value) : RegName(i.iop[k].reg));
    }
    return out;
}

/// Where no internal register survives: labels and branches (the scheduler's instance
/// switch cannot be held across them), and the waits and memory reads that deschedule
/// the instance (psp2cgc hoists every uniform-buffer load ahead of its internal-register
/// work; the console showed why, 2026-09-05).
bool IsBarrier(const Inst& i) {
    switch (i.op) {
    case Op::Label:
    case Op::Br:
    case Op::PhaseStart:
    case Op::Wdf:
    case Op::WdfVertex:
    case Op::Lda32:
    case Op::Sample2D:
    case Op::Phas:
        return true;
    default:
        return false;
    }
}

/// Modifier `outer` applied to a value already carrying `inner`.
Mod Compose(Mod outer, Mod inner) {
    const bool outer_abs = outer == Mod::Abs || outer == Mod::NegAbs;
    const bool outer_neg = outer == Mod::Neg || outer == Mod::NegAbs;
    const bool inner_abs = inner == Mod::Abs || inner == Mod::NegAbs;
    const bool inner_neg = inner == Mod::Neg || inner == Mod::NegAbs;
    // An outer abs forgets the inner sign; an outer negation flips it.
    const bool abs = outer_abs || inner_abs;
    const bool neg = outer_abs ? outer_neg : (inner_neg != outer_neg);
    return abs ? (neg ? Mod::NegAbs : Mod::Abs) : (neg ? Mod::Neg : Mod::None);
}

} // Anonymous namespace

// ---------------------------------------------------------------------------------------
// Builders

Inst& Builder::Push(Op op) {
    insts.push_back(Inst{op});
    return insts.back();
}

int Builder::NewLabel() {
    return label_count++;
}

void Builder::Bind(int label) {
    Push(Op::Label).label = label;
}

void Builder::Br(Pred pred, int label) {
    Inst& i = Push(Op::Br);
    i.pred = pred;
    i.label = label;
}

void Builder::PhaseStart() {
    Push(Op::PhaseStart);
}

void Builder::Phas(bool wait_all) {
    Push(Op::Phas).b[0] = wait_all;
}
void Builder::Nop() {
    Push(Op::Nop);
}
void Builder::NopAfterKill() {
    Push(Op::NopAfterKill);
}
void Builder::Wdf(uint8_t drc) {
    Push(Op::Wdf).u[0] = drc;
}
void Builder::WdfVertex(uint8_t drc) {
    Push(Op::WdfVertex).u[0] = drc;
}
void Builder::KillUnlessP1() {
    Push(Op::KillUnlessP1);
}
void Builder::DepthF(Reg src, uint8_t control_sa) {
    Inst& i = Push(Op::DepthF);
    i.reg[0] = src;
    i.u[0] = control_sa;
}
void Builder::EndVertex() {
    Push(Op::EndVertex);
}
void Builder::EndSecondary() {
    Push(Op::EndSecondary);
}

void Builder::Vec(VecOp op, Dst dst, Src src1, Src src2, Pred pred, bool) {
    Inst& i = Push(Op::Vec);
    i.u[0] = static_cast<uint8_t>(op);
    i.dst = dst;
    i.src[0] = src1;
    i.src[1] = src2;
    i.pred = pred;
}

void Builder::Vec16(VecOp op, Dst dst, Src src1, Src src2, Pred pred, bool) {
    Inst& i = Push(Op::Vec16);
    i.u[0] = static_cast<uint8_t>(op);
    i.dst = dst;
    i.src[0] = src1;
    i.src[1] = src2;
    i.pred = pred;
}

void Builder::Mad(Dst dst, Src src1, Src gpi0, Src gpi1, bool vec4, Pred pred, bool) {
    Inst& i = Push(Op::Mad);
    i.dst = dst;
    i.src[0] = src1;
    i.src[1] = gpi0;
    i.src[2] = gpi1;
    i.b[0] = vec4;
    i.pred = pred;
}

void Builder::Mad2(Dst dst, Src src0, Src src1, Src src2, Pred pred) {
    Inst& i = Push(Op::Mad2);
    i.dst = dst;
    i.src[0] = src0;
    i.src[1] = src1;
    i.src[2] = src2;
    i.pred = pred;
}

void Builder::Dot(Dst dst, Src src1, Src gpi0, bool vec4, Pred pred, bool) {
    Inst& i = Push(Op::Dot);
    i.dst = dst;
    i.src[0] = src1;
    i.src[1] = gpi0;
    i.b[0] = vec4;
    i.pred = pred;
}

void Builder::Comp(CompOp op, Dst dst, Src src, uint8_t src_comp, Pred pred, bool) {
    Inst& i = Push(Op::Comp);
    i.u[0] = static_cast<uint8_t>(op);
    i.u[1] = src_comp;
    i.dst = dst;
    i.src[0] = src;
    i.pred = pred;
}

void Builder::MoveF32x4(Dst dst, Reg lo, Reg hi, bool) {
    Inst& i = Push(Op::MoveF32x4);
    i.dst = dst;
    i.reg[0] = lo;
    i.reg[1] = hi;
}

void Builder::PackF16(Dst dst, Reg lo, Reg hi, bool) {
    Inst& i = Push(Op::PackF16);
    i.dst = dst;
    i.reg[0] = lo;
    i.reg[1] = hi;
}

void Builder::MoveInternal(Dst dst, Src src, Pred pred, bool) {
    Inst& i = Push(Op::MoveInternal);
    i.dst = dst;
    i.src[0] = src;
    i.pred = pred;
}

void Builder::CondMove(Dst dst, Src src0, Src src1, Src src2, bool) {
    Inst& i = Push(Op::CondMove);
    i.dst = dst;
    i.src[0] = src0;
    i.src[1] = src1;
    i.src[2] = src2;
}

void Builder::Test(Cond cond, uint8_t pred_n, Src src1, uint8_t chan, Src src2, bool src2_scalar,
                   Pred pred) {
    Inst& i = Push(Op::Test);
    i.u[0] = static_cast<uint8_t>(cond);
    i.u[1] = pred_n;
    i.u[2] = chan;
    i.src[0] = src1;
    i.src[1] = src2;
    i.b[0] = src2_scalar;
    i.pred = pred;
}

void Builder::Sample2D(Reg dst, Reg coord, uint8_t sampler_sa, const Reg* lod, uint8_t drc, bool) {
    Inst& i = Push(Op::Sample2D);
    i.reg[0] = dst;
    i.reg[1] = coord;
    i.u[0] = sampler_sa;
    i.u[1] = drc;
    i.b[0] = lod != nullptr;
    if (lod != nullptr) {
        i.src[0].reg = *lod;
    }
}

void Builder::Lda32(Reg dst, Reg base_sa, const Reg* offset_reg, uint8_t offset_words,
                    uint8_t count, uint8_t drc) {
    Inst& i = Push(Op::Lda32);
    i.reg[0] = dst;
    i.reg[1] = base_sa;
    i.u[0] = offset_words;
    i.u[1] = count;
    i.u[2] = drc;
    i.b[0] = offset_reg != nullptr;
    if (offset_reg != nullptr) {
        i.src[0].reg = *offset_reg;
    }
}

void Builder::Bitwise(BwOp op, Reg dst, IOp src1, IOp src2, Pred pred) {
    Inst& i = Push(Op::Bitwise);
    i.u[0] = static_cast<uint8_t>(op);
    i.reg[0] = dst;
    i.iop[0] = src1;
    i.iop[1] = src2;
    i.pred = pred;
}

void Builder::MadI16(Reg dst, Reg src0, Reg src1, IOp src2) {
    Inst& i = Push(Op::MadI16);
    i.reg[0] = dst;
    i.reg[1] = src0;
    i.src[0].reg = src1;
    i.iop[0] = src2;
}

void Builder::MadI32(Reg dst, Reg src0, uint8_t imm, Reg src2) {
    Inst& i = Push(Op::MadI32);
    i.reg[0] = dst;
    i.reg[1] = src0;
    i.u[0] = imm;
    i.src[0].reg = src2;
}

void Builder::PackS16(Reg dst, Reg src, uint8_t comp, Pred pred) {
    Inst& i = Push(Op::PackS16);
    i.reg[0] = dst;
    i.reg[1] = src;
    i.u[0] = comp;
    i.pred = pred;
}

void Builder::AddClass(RegClass cls) {
    classes.push_back(cls);
}

// ---------------------------------------------------------------------------------------
// Lowering

namespace {

class Lowering {
public:
    Lowering(std::vector<Inst>& insts_, const std::vector<RegClass>& classes_)
        : insts{insts_}, classes{classes_} {}

    LowerResult Run(Encoder& e, int level);

private:
    void Liveness();
    void DeadStores();
    void Reloads();
    void HoistLoads(int max_group);
    void HoistLoadsGlobal();
    void CoalesceOutputs();
    void ResultsIntoInternal();
    void TwoWideChains();
    std::optional<Src> LoadedSourceOf(std::size_t at, const Src& src) const;
    std::optional<Src> CopiedSourceOf(std::size_t at, const Src& half) const;
    void DirectTwoWide();
    void LoadSources();
    void InternalRegisters();
    void InternalDeadWrites();
    /// Whether internal register `reg`'s lanes `lanes` are read before being fully
    /// rewritten, from instruction `from` on (a label or branch ends the search: the
    /// internal registers never hold a value across one).
    bool InternalRead(std::size_t from, uint8_t reg, uint8_t lanes);
    void Allocate();
    /// USSE_IR_DUMP=1 in the environment prints the instruction list after each pass
    /// (host tests).
    void Dump(const char* stage) const;
    /// Refuses the program when an internal register is live across a barrier: the
    /// hardware would read stale lanes there, and Vita3K would not notice.
    void CheckInternals() const;
    void Encode(Encoder& e, LowerResult& result);
    Reg Map(Reg r) const;

    std::vector<Inst>& insts;
    const std::vector<RegClass>& classes;
    uint32_t quad_count = 0;
    /// live[i][q]: the lanes of virtual quad q live after instruction i has executed.
    std::vector<std::vector<uint8_t>> live;
    /// The interval each quad occupies, in instruction indices.
    struct Interval {
        std::size_t start = SIZE_MAX;
        std::size_t end = 0;
        bool used = false;
        bool temp_or_pa = false; ///< a load, sample or integer op names it: not an output register
        std::optional<Reg> home;
        /// Bit i set when the quad is live at instruction i (defined, used, or carried
        /// across): the allocator fits quads into the holes, not the hull.
        std::vector<uint64_t> active;
    };
    std::vector<Interval> intervals;
    std::set<uint32_t> coalesced_outputs; ///< output quads written in place: no spills there
    uint32_t spilled = 0;
    uint32_t temp_count = 0;
    uint32_t pa_count = 0;
};

/// The lanes of a virtual quad an operand touches: a write covers its mask from the
/// register named (a pair's two lanes, a load's count); a read is taken to cover every
/// lane from the register named upward, since a four-wide source spans the quad.
uint8_t WriteLanes(const Inst& inst, const Reg& r) {
    const unsigned at = r.n & 3;
    unsigned mask = 0xf;
    switch (inst.op) {
    case Op::Lda32:
        mask = (1u << std::min<unsigned>(inst.u[1], 4)) - 1;
        break;
    case Op::Sample2D:
        mask = 0xf;
        break;
    case Op::Bitwise:
    case Op::MadI16:
    case Op::MadI32:
    case Op::PackS16:
        mask = 1;
        break;
    default:
        mask = inst.dst.mask;
        break;
    }
    return static_cast<uint8_t>((mask << at) & 0xf);
}

uint8_t ReadLanes(const Reg& r) {
    return static_cast<uint8_t>((0xf << (r.n & 3)) & 0xf);
}

/// The lanes an instruction writes of an internal register (its number is not a lane
/// offset: i0-i2 are whole 128-bit registers).
uint8_t InternalWriteLanes(const Inst& inst) {
    switch (inst.op) {
    case Op::Vec:
    case Op::Vec16:
    case Op::Mad:
    case Op::Mad2:
    case Op::Dot:
    case Op::Comp:
    case Op::MoveF32x4:
    case Op::PackF16:
    case Op::MoveInternal:
    case Op::CondMove:
        return inst.dst.mask;
    default:
        return 0xf;
    }
}

void Lowering::Liveness() {
    const std::size_t n = insts.size();
    std::vector<Ref> refs;
    std::map<int, std::size_t> label_at;
    for (std::size_t i = 0; i < n; i++) {
        if (insts[i].op == Op::Label) {
            label_at[insts[i].label] = i;
        }
    }
    // Successors: the fallthrough unless an unconditional branch, plus a branch target.
    const auto successors = [&](std::size_t i, std::size_t out[2]) -> int {
        int count = 0;
        const Inst& inst = insts[i];
        if (inst.op == Op::Br) {
            const auto it = label_at.find(inst.label);
            if (it == label_at.end()) {
                Fail("branch to an unbound label");
            }
            out[count++] = it->second;
            if (inst.pred == Pred::None) {
                return count;
            }
        }
        if (i + 1 < n) {
            out[count++] = i + 1;
        }
        return count;
    };

    // Per instruction and quad: the lanes read and the lanes written outright. A
    // predicated write keeps the old lanes alive, so it reads them and defines nothing.
    std::vector<std::vector<uint8_t>> uses(n, std::vector<uint8_t>(quad_count, 0));
    std::vector<std::vector<uint8_t>> defs(n, std::vector<uint8_t>(quad_count, 0));
    for (std::size_t i = 0; i < n; i++) {
        Inst& inst = insts[i];
        Refs(inst, refs);
        for (const Ref& r : refs) {
            if (!IsVirtual(*r.reg)) {
                continue;
            }
            const uint32_t q = QuadOf(*r.reg);
            if (r.def) {
                const uint8_t lanes = WriteLanes(inst, *r.reg);
                if (inst.pred == Pred::None) {
                    defs[i][q] |= lanes;
                } else {
                    uses[i][q] |= lanes;
                }
            } else {
                uses[i][q] |= ReadLanes(*r.reg);
            }
        }
    }

    live.assign(n, std::vector<uint8_t>(quad_count, 0));
    std::vector<std::vector<uint8_t>> in(n, std::vector<uint8_t>(quad_count, 0));
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = n; i-- > 0;) {
            std::size_t succ[2];
            const int count = successors(i, succ);
            for (uint32_t q = 0; q < quad_count; q++) {
                uint8_t out = 0;
                for (int s = 0; s < count; s++) {
                    out |= in[succ[s]][q];
                }
                if (out != live[i][q]) {
                    live[i][q] = out;
                    changed = true;
                }
                const uint8_t new_in = static_cast<uint8_t>(uses[i][q] | (out & ~defs[i][q]));
                if (new_in != in[i][q]) {
                    in[i][q] = new_in;
                    changed = true;
                }
            }
        }
    }

    intervals.assign(quad_count, {});
    for (uint32_t q = 0; q < quad_count; q++) {
        intervals[q].active.assign((n + 63) / 64, 0);
    }
    for (std::size_t i = 0; i < n; i++) {
        for (uint32_t q = 0; q < quad_count; q++) {
            if (uses[i][q] || defs[i][q] || in[i][q] || live[i][q]) {
                Interval& iv = intervals[q];
                iv.start = std::min(iv.start, i);
                iv.end = std::max(iv.end, i);
                iv.used = iv.used || uses[i][q] != 0;
                iv.active[i / 64] |= 1ull << (i % 64);
            }
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        const Inst& inst = insts[i];
        if (inst.op == Op::Lda32 || inst.op == Op::Sample2D || inst.op == Op::MadI16 ||
            inst.op == Op::MadI32 || inst.op == Op::DepthF) {
            for (const Reg r : {inst.reg[0], inst.reg[1]}) {
                if (IsVirtual(r)) {
                    intervals[QuadOf(r)].temp_or_pa = true;
                }
            }
        }
        if (inst.op == Op::Mad2 && IsVirtual(inst.src[0].reg)) {
            intervals[QuadOf(inst.src[0].reg)].temp_or_pa = true;
        }
    }
}

void Lowering::DeadStores() {
    std::vector<Ref> refs;
    std::vector<Inst> kept;
    kept.reserve(insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        Inst& inst = insts[i];
        bool dead = false;
        if (PureWrite(inst)) {
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (r.def) {
                    dead = IsVirtual(*r.reg) &&
                           (live[i][QuadOf(*r.reg)] & WriteLanes(inst, *r.reg)) == 0;
                }
            }
        }
        if (!dead) {
            kept.push_back(inst);
        }
    }
    insts.swap(kept);
}

void Lowering::Reloads() {
    // What each internal register holds: the load that filled it.
    std::optional<Inst> held[3];
    std::vector<Ref> refs;
    std::vector<Inst> kept;
    kept.reserve(insts.size());
    for (Inst& inst : insts) {
        if (IsBarrier(inst)) {
            for (auto& h : held) {
                h.reset();
            }
            kept.push_back(inst);
            continue;
        }
        if (IsInternalLoad(inst)) {
            const auto& h = held[inst.dst.reg.n];
            if (h && SameReg(h->src[0].reg, inst.src[0].reg) && h->src[0].swz == inst.src[0].swz &&
                h->src[0].mod == inst.src[0].mod) {
                continue; // the register already holds it
            }
        }
        // Anything written invalidates: the internal register itself, or the source of a
        // load an internal register holds.
        Refs(inst, refs);
        for (const Ref& r : refs) {
            if (!r.def) {
                continue;
            }
            for (auto& h : held) {
                if (h && (SameReg(*r.reg, h->dst.reg) ||
                          (QuadOf(*r.reg) == QuadOf(h->src[0].reg) && r.reg->bank == h->src[0].reg.bank))) {
                    h.reset();
                }
            }
        }
        if (IsInternalLoad(inst)) {
            held[inst.dst.reg.n] = inst;
        }
        kept.push_back(inst);
    }
    insts.swap(kept);
}

bool Lowering::InternalRead(std::size_t from, uint8_t reg, uint8_t lanes) {
    std::vector<Ref> refs;
    for (std::size_t i = from; i < insts.size() && lanes != 0; i++) {
        Inst& inst = insts[i];
        if (IsBarrier(inst)) {
            return false;
        }
        Refs(inst, refs);
        for (const Ref& r : refs) {
            if (r.reg->bank != Bank::Internal || r.reg->n != reg) {
                continue;
            }
            if (!r.def) {
                return true; // conservatively any read
            }
        }
        // Writes after the reads: an unpredicated write clears what it covers.
        if (inst.pred == Pred::None) {
            for (const Ref& r : refs) {
                if (r.def && r.reg->bank == Bank::Internal && r.reg->n == reg) {
                    lanes &= static_cast<uint8_t>(~InternalWriteLanes(inst));
                }
            }
        }
    }
    return false;
}

void Lowering::InternalDeadWrites() {
    std::vector<Ref> refs;
    std::vector<Inst> kept;
    kept.reserve(insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        Inst& inst = insts[i];
        bool dead = false;
        if (PureWrite(inst)) {
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (r.def && r.reg->bank == Bank::Internal) {
                    dead = !InternalRead(i + 1, r.reg->n, InternalWriteLanes(inst));
                }
            }
        }
        if (!dead) {
            kept.push_back(inst);
        }
    }
    insts.swap(kept);
}

namespace {

/// The lanes (2h, 2h + 1) of a four-wide source as a two-wide source: the pair holding
/// them (both channels in one pair, or constants) with the channels moved down, or an
/// internal register with the lanes picked. `second` is the op's second source, which
/// only has the standard swizzle table and no negation.
std::optional<Src> HalfSource(const Src& src, int h, bool second) {
    const Ch c0 = src.swz[2 * h];
    const Ch c1 = src.swz[2 * h + 1];
    if (src.reg.bank == Bank::Internal) {
        if (second) {
            Swizzle s;
            if (src.mod == Mod::Neg || src.mod == Mod::NegAbs || !StdSwizzleForPair(c0, c1, s)) {
                return std::nullopt;
            }
            return Src{src.reg, s, src.mod};
        }
        return Src{src.reg, {c0, c1, c0, c1}, src.mod};
    }
    if (src.reg.bank == Bank::Const) {
        // Constants are read as (entry n, entry n + 1) pairs; XXXX/YYYY broadcasts survive
        // any half unchanged, the (0, 1) pair only as lanes x, y.
        if (c0 == c1 && c0 <= Ch::W) {
            return src;
        }
        if (c0 == Ch::X && c1 == Ch::Y) {
            return src;
        }
        return std::nullopt;
    }
    // Unified store: both channels from one pair, moved down to that pair's lanes.
    const bool k0 = c0 <= Ch::W, k1 = c1 <= Ch::W;
    int pair = -1;
    if (k0) {
        pair = static_cast<int>(c0) / 2;
    }
    if (k1) {
        const int p1 = static_cast<int>(c1) / 2;
        if (pair >= 0 && p1 != pair) {
            return std::nullopt;
        }
        pair = p1;
    }
    if (pair < 0) {
        return std::nullopt; // two constants: no register to read
    }
    const Ch d0 = k0 ? static_cast<Ch>(static_cast<int>(c0) & 1) : c0;
    const Ch d1 = k1 ? static_cast<Ch>(static_cast<int>(c1) & 1) : c1;
    const Reg reg{src.reg.bank, static_cast<uint8_t>(src.reg.n + pair * 2)};
    if (second) {
        Swizzle s;
        if (!k0 || !k1 || src.mod == Mod::Neg || src.mod == Mod::NegAbs ||
            !StdSwizzleForPair(d0, d1, s)) {
            return std::nullopt;
        }
        return Src{reg, s, src.mod};
    }
    return Src{reg, {d0, d1, d0, d1}, src.mod};
}

} // Anonymous namespace

namespace {

/// The registers an instruction writes, as (bank, register) pairs at lane granularity.
void WrittenRegs(Inst& inst, std::vector<std::pair<Bank, int>>& out) {
    out.clear();
    std::vector<Ref> refs;
    Refs(inst, refs);
    for (const Ref& r : refs) {
        if (!r.def) {
            continue;
        }
        const Reg& reg = *r.reg;
        int count = 1;
        uint8_t mask = 1;
        switch (inst.op) {
        case Op::Lda32:
            count = std::min<int>(inst.u[1], 4);
            mask = static_cast<uint8_t>((1u << count) - 1);
            break;
        case Op::Sample2D:
            mask = 0xf;
            break;
        case Op::Bitwise:
        case Op::MadI16:
        case Op::MadI32:
        case Op::PackS16:
            mask = 1;
            break;
        default:
            mask = inst.dst.mask;
            break;
        }
        for (int l = 0; l < 4; l++) {
            if (mask & (1u << l)) {
                out.emplace_back(reg.bank, reg.n + l);
            }
        }
    }
}

} // Anonymous namespace

void Lowering::HoistLoadsGlobal() {
    // Indexed uniform reads whose address register is written only before every use
    // (the MOVA at the top of a skinning program) are read once, right after that last
    // write, and every later read of the same index is that quad: the compiler keeps
    // the bone matrix in three temporaries for the whole program.
    std::vector<Ref> refs;
    struct Site {
        std::size_t start, load, wait;
        Reg addr, dst, areg, ones, ptr;
        int index;
    };
    std::vector<Site> sites;
    for (std::size_t w = 4; w < insts.size(); w++) {
        if (insts[w].op != Op::WdfVertex || insts[w - 1].op != Op::Lda32) {
            continue;
        }
        const Inst& load = insts[w - 1];
        const Inst& m16 = insts[w - 4];
        const Inst& bw = insts[w - 3];
        const Inst& m32 = insts[w - 2];
        if (load.b[0] || load.reg[0].bank != Bank::Virtual || load.reg[1].bank != Bank::Virtual ||
            load.u[0] != 0) {
            continue;
        }
        const Reg addr = load.reg[1];
        if (!(m16.op == Op::MadI16 && m16.iop[0].imm && SameReg(m16.reg[0], addr) &&
              bw.op == Op::Bitwise && static_cast<BwOp>(bw.u[0]) == BwOp::And && bw.iop[0].imm &&
              bw.iop[0].value == 0x7f && !bw.iop[1].imm && SameReg(bw.iop[1].reg, addr) &&
              SameReg(bw.reg[0], addr) && m32.op == Op::MadI32 && m32.u[0] == 16 &&
              SameReg(m32.reg[1], addr) && SameReg(m32.reg[0], addr) && m16.pred == Pred::None &&
              bw.pred == Pred::None && m32.pred == Pred::None)) {
            continue;
        }
        sites.push_back({w - 4, w - 1, w, addr, load.reg[0], m16.reg[1], m16.src[0].reg,
                         m32.src[0].reg, m16.iop[0].value});
    }
    if (sites.size() < 2) {
        return;
    }
    // Per address register: the last write, and whether every site comes after it.
    std::map<std::pair<Bank, int>, std::size_t> last_write;
    std::vector<std::pair<Bank, int>> written;
    for (std::size_t i = 0; i < insts.size(); i++) {
        WrittenRegs(insts[i], written);
        for (const auto& wr : written) {
            last_write[wr] = i;
        }
    }
    // The hoisted reads go right after the register's last write, which must be in the
    // entry block so every path runs them.
    std::size_t entry_end = insts.size();
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (insts[i].op == Op::Label || insts[i].op == Op::Br || insts[i].op == Op::PhaseStart) {
            entry_end = i;
            break;
        }
    }
    std::vector<bool> drop(insts.size(), false);
    std::map<std::size_t, std::vector<Inst>> insert_after; // instruction -> hoisted reads
    std::set<std::size_t> handled;
    for (std::size_t a = 0; a < sites.size(); a++) {
        if (handled.count(a)) {
            continue;
        }
        const Site& sa = sites[a];
        const auto lw = last_write.find({sa.areg.bank, sa.areg.n});
        if (lw == last_write.end() || lw->second >= sa.start || lw->second >= entry_end) {
            continue; // written after (or inside) a use, or under control flow
        }
        // Every site with this address register (and pointer): same treatment.
        std::vector<std::size_t> members;
        for (std::size_t b = a; b < sites.size(); b++) {
            const Site& sb = sites[b];
            if (!handled.count(b) && SameReg(sb.areg, sa.areg) && SameReg(sb.ones, sa.ones) &&
                SameReg(sb.ptr, sa.ptr) && lw->second < sb.start) {
                members.push_back(b);
            }
        }
        // Distinct indices, lowest first; at most four quads per register.
        std::vector<int> indices;
        for (const std::size_t b : members) {
            if (std::find(indices.begin(), indices.end(), sites[b].index) == indices.end()) {
                indices.push_back(sites[b].index);
            }
        }
        std::sort(indices.begin(), indices.end());
        if (indices.size() > 4 || quad_count + 1 + indices.size() > 60) {
            continue;
        }
        // The hoisted reads: a chain per index, except that an index within three rows
        // of an earlier one loads at an immediate offset from its chain (the buffer's
        // mirror rows past 127 cover the wrap); the loads, one wait. The addresses share
        // one quad, a lane each.
        const uint8_t addr_quad = static_cast<uint8_t>(quad_count++ * 4);
        std::vector<Inst> hoisted;
        std::map<int, Reg> quad_of_index;
        {
            int lane = 0;
            int base_index = -1000;
            Reg base_addr{};
            for (const int index : indices) {
                Reg addr = base_addr;
                if (index - base_index > static_cast<int>(UniformMirrorRows) - 1) {
                    Inst m16 = insts[sa.start];
                    Inst bw = insts[sa.start + 1];
                    Inst m32 = insts[sa.start + 2];
                    addr = Reg::V(static_cast<uint8_t>(addr_quad + lane));
                    m16.reg[0] = addr;
                    m16.iop[0].value = static_cast<uint8_t>(index);
                    bw.reg[0] = addr;
                    bw.iop[1].reg = addr;
                    m32.reg[0] = addr;
                    m32.reg[1] = addr;
                    hoisted.push_back(m16);
                    hoisted.push_back(bw);
                    hoisted.push_back(m32);
                    base_index = index;
                    base_addr = addr;
                    lane++;
                }
                Inst load = insts[sa.load];
                const Reg dst = Reg::V(static_cast<uint8_t>(quad_count++ * 4));
                load.reg[0] = dst;
                load.reg[1] = addr;
                load.u[0] = static_cast<uint8_t>((index - base_index) * 4);
                hoisted.push_back(load);
                quad_of_index[index] = dst;
            }
            hoisted.push_back(insts[sa.wait]);
        }
        insert_after[lw->second].insert(insert_after[lw->second].end(), hoisted.begin(), hoisted.end());
        // Each site: dropped, its readers renamed to the hoisted quad until the load's
        // quad is written again or the block ends.
        for (const std::size_t b : members) {
            const Site& sb = sites[b];
            handled.insert(b);
            for (std::size_t i = sb.start; i <= sb.wait; i++) {
                drop[i] = true;
            }
            const Reg dst = quad_of_index[sb.index];
            for (std::size_t i = sb.wait + 1; i < insts.size(); i++) {
                Inst& inst = insts[i];
                if (IsBarrier(inst) && inst.op != Op::Wdf && inst.op != Op::WdfVertex &&
                    inst.op != Op::Lda32 && inst.op != Op::Sample2D) {
                    break; // control flow
                }
                WrittenRegs(inst, written);
                bool writes_dst = false;
                for (const auto& wr : written) {
                    writes_dst |= wr.first == Bank::Virtual && wr.second / 4 == static_cast<int>(QuadOf(sb.dst));
                }
                if (writes_dst) {
                    break;
                }
                Refs(inst, refs);
                for (const Ref& r : refs) {
                    if (!r.def && r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == QuadOf(sb.dst)) {
                        r.reg->n = static_cast<uint8_t>(dst.n + (r.reg->n & 3));
                    }
                }
            }
        }
    }
    if (insert_after.empty()) {
        return;
    }
    std::vector<Inst> out;
    out.reserve(insts.size() + 16);
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!drop[i]) {
            out.push_back(insts[i]);
        }
        const auto it = insert_after.find(i);
        if (it != insert_after.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        }
    }
    insts.swap(out);
}

void Lowering::HoistLoads(int max_group) {
    // A uniform read is an address chain (mad.i16, and, mad.i32 into one register), the
    // load, and its wait. Consecutive reads in a block, with nothing between them that
    // writes what a later chain reads, are issued together and wait once, each with its
    // own address register and destination quad, so the values the reads feed stay in
    // the internal registers across the group (the compiler issues up to three loads
    // before a wait).
    const int MaxGroup = max_group;
    struct Sequence {
        std::size_t start, load, wait; // [start, wait] inclusive; load is the Lda32
        Reg addr;                      // the chain's register, or none (immediate form)
        bool has_addr;
        Reg dst;                       // the quad the load fills
        std::vector<std::pair<Bank, int>> inputs; // registers the chain and load read
    };
    std::vector<Sequence> seqs;
    std::vector<Ref> refs;
    // A load whose quad is written by no other instruction is one already hoisted (its
    // readers may sit in other blocks): left alone.
    std::map<uint32_t, int> quad_defs;
    for (const Inst& inst : insts) {
        if (inst.op == Op::Lda32 && inst.reg[0].bank == Bank::Virtual) {
            quad_defs[QuadOf(inst.reg[0])]++;
        }
    }
    for (std::size_t w = 1; w < insts.size(); w++) {
        if (insts[w].op != Op::WdfVertex || insts[w - 1].op != Op::Lda32) {
            continue;
        }
        const Inst& load = insts[w - 1];
        if (load.b[0] || load.reg[0].bank != Bank::Virtual ||
            quad_defs[QuadOf(load.reg[0])] < 2) {
            continue; // the register-offset form, a physical destination, or a hoisted one
        }
        Sequence sq{w - 1, w - 1, w, load.reg[1], false, load.reg[0], {}};
        if (load.reg[1].bank == Bank::Virtual) {
            sq.has_addr = true;
            std::size_t st = w - 1;
            while (st > 0) {
                const Inst& c = insts[st - 1];
                const bool chain = (c.op == Op::MadI16 || c.op == Op::Bitwise || c.op == Op::MadI32) &&
                                   c.pred == Pred::None && SameReg(c.reg[0], sq.addr);
                if (!chain) {
                    break;
                }
                st--;
            }
            sq.start = st;
            // No chain of its own (a load already hoisted at an immediate offset from
            // another's address register): the base stays what it is.
            sq.has_addr = st < w - 1;
        }
        for (std::size_t i = sq.start; i <= sq.load; i++) {
            Refs(insts[i], refs);
            for (const Ref& r : refs) {
                if (!r.def && !(sq.has_addr && SameReg(*r.reg, sq.addr))) {
                    sq.inputs.emplace_back(r.reg->bank, r.reg->n);
                }
            }
        }
        seqs.push_back(sq);
    }
    if (seqs.size() < 2) {
        return;
    }
    // Chains of the emitter's indexed form, for sharing and for reuse.
    struct ChainInfo {
        bool indexed = false;
        Reg areg{}, ones{}, ptr{};
        int index = 0;
    };
    std::vector<ChainInfo> chain_of(seqs.size());
    for (std::size_t k = 0; k < seqs.size(); k++) {
        const Sequence& sq = seqs[k];
        if (!sq.has_addr || sq.load - sq.start != 3) {
            continue;
        }
        const Inst& m16 = insts[sq.start];
        const Inst& bw = insts[sq.start + 1];
        const Inst& m32 = insts[sq.start + 2];
        if (m16.op == Op::MadI16 && m16.iop[0].imm && bw.op == Op::Bitwise &&
            static_cast<BwOp>(bw.u[0]) == BwOp::And && bw.iop[0].imm && bw.iop[0].value == 0x7f &&
            !bw.iop[1].imm && SameReg(bw.iop[1].reg, sq.addr) && m32.op == Op::MadI32 &&
            m32.u[0] == 16 && SameReg(m32.reg[1], sq.addr)) {
            ChainInfo& c = chain_of[k];
            c.indexed = true;
            c.areg = m16.reg[1];
            c.ones = m16.src[0].reg;
            c.ptr = m32.src[0].reg;
            c.index = m16.iop[0].value;
        }
    }
    // A read of the same uniform (same address register value, same index) earlier in
    // the block still holds in the quad it filled: the later read is that quad. Writes
    // to the address register, and barriers, end what is known.
    struct Known {
        ChainInfo chain;
        Reg dst;
    };
    std::vector<Known> known;
    std::vector<bool> done(seqs.size(), false);
    std::vector<Inst> out;
    out.reserve(insts.size() + 8);
    std::vector<std::pair<Bank, int>> written;
    const auto emit = [&](const Inst& inst) {
        if (inst.op == Op::Label || inst.op == Op::Br || inst.op == Op::PhaseStart) {
            known.clear(); // control flow; waits and loads do not change a loaded quad
        } else {
            WrittenRegs(const_cast<Inst&>(inst), written);
            for (const auto& wr : written) {
                for (std::size_t j = 0; j < known.size();) {
                    const Reg& a = known[j].chain.areg;
                    if (wr.first == a.bank && wr.second == a.n) {
                        known.erase(known.begin() + static_cast<long>(j));
                    } else {
                        j++;
                    }
                }
            }
        }
        out.push_back(inst);
    };
    static const bool no_cse = GxmFlag("usse_nocse");
    const auto find_known = [&](const ChainInfo& c) -> const Reg* {
        if (!c.indexed || no_cse) {
            return nullptr;
        }
        for (const Known& kn : known) {
            if (SameReg(kn.chain.areg, c.areg) && SameReg(kn.chain.ones, c.ones) &&
                SameReg(kn.chain.ptr, c.ptr) && kn.chain.index == c.index) {
                return &kn.dst;
            }
        }
        return nullptr;
    };
    std::size_t next = 0; // next instruction of `insts` not yet copied
    for (std::size_t a = 0; a < seqs.size(); a++) {
        if (done[a]) {
            continue;
        }
        // Extend the group while the gap holds no barrier and writes nothing a later
        // chain reads.
        std::size_t b = a;
        std::vector<std::pair<Bank, int>> gap_writes;
        while (b + 1 < seqs.size() && static_cast<int>(b - a + 1) < MaxGroup) {
            const Sequence& cur = seqs[b];
            const Sequence& nxt = seqs[b + 1];
            bool ok = true;
            for (std::size_t i = cur.wait + 1; i < nxt.start && ok; i++) {
                if (IsBarrier(insts[i]) || insts[i].op == Op::Lda32) {
                    ok = false;
                    break;
                }
                WrittenRegs(insts[i], written);
                gap_writes.insert(gap_writes.end(), written.begin(), written.end());
            }
            for (const auto& in : nxt.inputs) {
                for (const auto& wr : gap_writes) {
                    if (in.first == wr.first && in.second == wr.second) {
                        ok = false;
                    }
                }
            }
            // The next chain must not read the previous load's destination either (it
            // would be renamed): a dependent index goes through a MOVA, which writes
            // the address register the chain reads, so the write check covers it.
            if (!ok) {
                break;
            }
            b++;
        }
        // Copy everything before the group unchanged.
        for (; next < seqs[a].start; next++) {
            emit(insts[next]);
        }
        // Virtual register numbers are 8-bit: 64 quads at most, one for the addresses
        // and one per member here.
        if (quad_count + 1 + (b - a + 1) > 64) {
            b = a;
        }
        if (b == a) {
            // A single read: reused when known, else its own quad so a later read can
            // reuse it.
            const Sequence& sq = seqs[a];
            const Reg* have = find_known(chain_of[a]);
            Reg dst = sq.dst;
            if (have != nullptr) {
                dst = *have;
                next = sq.wait + 1;
            } else if (quad_count < 64) {
                dst = Reg::V(static_cast<uint8_t>(quad_count * 4));
                quad_count++;
                for (; next <= sq.wait; next++) {
                    Inst inst = insts[next];
                    Refs(inst, refs);
                    for (const Ref& r : refs) {
                        if (r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == QuadOf(sq.dst)) {
                            r.reg->n = static_cast<uint8_t>(dst.n + (r.reg->n & 3));
                        }
                    }
                    emit(inst);
                }
                if (chain_of[a].indexed) {
                    known.push_back({chain_of[a], dst});
                }
            } else {
                for (; next <= sq.wait; next++) {
                    emit(insts[next]);
                }
            }
            if (!SameReg(dst, sq.dst)) {
                // What follows reads the load's quad until it is written again.
                std::size_t until = a + 1 < seqs.size() ? seqs[a + 1].start : insts.size();
                for (std::size_t i = next; i < until; i++) {
                    WrittenRegs(insts[i], written);
                    bool writes_dst = false;
                    for (const auto& wr : written) {
                        writes_dst |= wr.first == Bank::Virtual && wr.second / 4 == static_cast<int>(QuadOf(sq.dst));
                    }
                    if (writes_dst || IsBarrier(insts[i])) {
                        until = std::min(until, i);
                        break;
                    }
                }
                for (; next < until; next++) {
                    Inst inst = insts[next];
                    Refs(inst, refs);
                    for (const Ref& r : refs) {
                        if (!r.def && r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == QuadOf(sq.dst)) {
                            r.reg->n = static_cast<uint8_t>(dst.n + (r.reg->n & 3));
                        }
                    }
                    emit(inst);
                }
            }
            done[a] = true;
            continue;
        }
        // One fresh quad holds the group's addresses (a lane each), and each member's
        // destination is a fresh quad.
        std::vector<Reg> new_addr(b - a + 1), new_dst(b - a + 1);
        const uint32_t addr_quad = quad_count++;
        for (std::size_t k = a; k <= b; k++) {
            new_addr[k - a] = Reg::V(static_cast<uint8_t>(addr_quad * 4 + (k - a)));
            new_dst[k - a] = Reg::V(static_cast<uint8_t>(quad_count * 4));
            quad_count++;
        }
        const auto rename = [&](Inst& inst, const Sequence& sq, Reg addr, Reg dst) {
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (sq.has_addr && SameReg(*r.reg, sq.addr)) {
                    *r.reg = addr;
                } else if (r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == QuadOf(sq.dst)) {
                    r.reg->n = static_cast<uint8_t>(dst.n + (r.reg->n & 3));
                }
            }
        };
        // Members whose chain is the emitter's indexed form (mad.i16 index + a, and
        // 0x7f, mad.i32 * 16 + pointer) over the same address register and pointer
        // share the lowest index's chain and load at an immediate offset from it (an
        // index that wraps at 128 across the group is a matrix straddling the array
        // end: never seen). A member already known from an earlier read is that quad.
        // Members whose chain is the emitter's indexed form over the same address
        // register and pointer share the lowest index's chain and load at an immediate
        // offset from it, up to three rows away: the PICA index wraps at 128 per read
        // (Smash reads with a0 = -2), and the uniform buffer carries rows 0-3 again past
        // 127 so an offset from row 127 lands on the right data.
        std::vector<int> share_base(b - a + 1, -1);
        std::vector<bool> reused(b - a + 1, false);
        for (std::size_t k = a; k <= b; k++) {
            if (const Reg* have = find_known(chain_of[k])) {
                new_dst[k - a] = *have;
                reused[k - a] = true;
                continue;
            }
            const ChainInfo& c = chain_of[k];
            if (!c.indexed) {
                continue;
            }
            for (std::size_t j = a; j < k; j++) {
                const ChainInfo& d = chain_of[j];
                if (d.indexed && !reused[j - a] && share_base[j - a] < 0 && SameReg(d.areg, c.areg) &&
                    SameReg(d.ones, c.ones) && SameReg(d.ptr, c.ptr) && c.index >= d.index &&
                    c.index - d.index <= UniformMirrorRows - 1) {
                    share_base[k - a] = static_cast<int>(j - a);
                    break;
                }
            }
        }
        // The chains and loads, then one wait.
        bool any_load = false;
        for (std::size_t k = a; k <= b; k++) {
            if (reused[k - a]) {
                continue;
            }
            any_load = true;
            if (share_base[k - a] >= 0) {
                Inst load = insts[seqs[k].load];
                rename(load, seqs[k], new_addr[k - a], new_dst[k - a]);
                load.reg[1] = new_addr[share_base[k - a]];
                load.u[0] = static_cast<uint8_t>(
                    (chain_of[k].index - chain_of[a + share_base[k - a]].index) * 4);
                emit(load);
            } else {
                for (std::size_t i = seqs[k].start; i <= seqs[k].load; i++) {
                    Inst inst = insts[i];
                    rename(inst, seqs[k], new_addr[k - a], new_dst[k - a]);
                    emit(inst);
                }
            }
            if (chain_of[k].indexed) {
                known.push_back({chain_of[k], new_dst[k - a]});
            }
        }
        if (any_load) {
            emit(insts[seqs[b].wait]);
        }
        // The gaps, each reading its own load's quad.
        for (std::size_t k = a; k < b; k++) {
            for (std::size_t i = seqs[k].wait + 1; i < seqs[k + 1].start; i++) {
                Inst inst = insts[i];
                rename(inst, seqs[k], new_addr[k - a], new_dst[k - a]);
                emit(inst);
            }
            done[k] = true;
        }
        done[b] = true;
        next = seqs[b].wait + 1;
        // What follows reads the last load's quad until it is written again (the next
        // sequence's own instructions are the next group's to copy).
        std::size_t until = b + 1 < seqs.size() ? seqs[b + 1].start : insts.size();
        for (std::size_t i = next; i < until; i++) {
            WrittenRegs(insts[i], written);
            bool writes_dst = false;
            for (const auto& wr : written) {
                writes_dst |= wr.first == Bank::Virtual && wr.second / 4 == static_cast<int>(QuadOf(seqs[b].dst));
            }
            if (writes_dst || IsBarrier(insts[i])) {
                until = std::min(until, i);
                break;
            }
        }
        for (; next < until; next++) {
            Inst inst = insts[next];
            rename(inst, seqs[b], new_addr[b - a], new_dst[b - a]);
            emit(inst);
        }
    }
    for (; next < insts.size(); next++) {
        emit(insts[next]);
    }
    insts.swap(out);
}

std::optional<Src> Lowering::LoadedSourceOf(std::size_t at, const Src& src) const {
    // An internal-register read at `at` as a read of the unified quad the register was last
    // loaded from: the nearest earlier write of the register must be a whole-quad load, the
    // quad untouched since, and the read's channels compose through the load's.
    if (src.reg.bank != Bank::Internal) {
        return src;
    }
    std::optional<Src> loaded;
    std::vector<Ref> rr;
    for (std::size_t j = at; j-- > 0;) {
        if (IsBarrier(insts[j])) {
            break;
        }
        Refs(const_cast<Inst&>(insts[j]), rr);
        bool writes_reg = false;
        for (const Ref& r : rr) {
            writes_reg |= r.def && r.reg->bank == Bank::Internal && r.reg->n == src.reg.n;
        }
        if (!writes_reg) {
            continue;
        }
        // Any swizzle of real channels composes (a broadcast `sa42.yyyy` as much as
        // `xyzw`); a constant lane cannot be read through.
        const auto channels_only = [](const Swizzle& w) {
            return w[0] <= Ch::W && w[1] <= Ch::W && w[2] <= Ch::W && w[3] <= Ch::W;
        };
        if (IsInternalLoad(insts[j]) && channels_only(insts[j].src[0].swz) &&
            insts[j].src[0].mod == Mod::None && insts[j].src[0].reg.bank != Bank::Const) {
            loaded = insts[j].src[0];
            for (std::size_t k = j + 1; k < at && loaded; k++) {
                Refs(const_cast<Inst&>(insts[k]), rr);
                for (const Ref& r : rr) {
                    if (r.def && r.reg->bank == loaded->reg.bank &&
                        QuadOf(*r.reg) == QuadOf(loaded->reg)) {
                        loaded.reset();
                    }
                }
            }
        }
        break;
    }
    if (!loaded) {
        return std::nullopt;
    }
    Swizzle swz;
    for (int l = 0; l < 4; l++) {
        const Ch ch = src.swz[l];
        swz[l] = ch <= Ch::W ? loaded->swz[static_cast<int>(ch)] : ch;
    }
    return Src{loaded->reg, swz, Compose(src.mod, loaded->mod)};
}

std::optional<Src> Lowering::CopiedSourceOf(std::size_t at, const Src& half) const {
    // A two-wide read of a virtual pair that a copy filled - `vec.mul v152.xy, sa42.yyyy,
    // c2.x` (the emitter's broadcast of a uniform scalar) or `mov v40.xy, sa26.xy` - as the
    // same read of what was copied: the pair untouched since, the copy unpredicated, the
    // source not virtual or internal (a secondary attribute, a primary attribute, a constant
    // the reader can take). Without this the copies stay alive for the chain's sake.
    if (half.reg.bank != Bank::Virtual) {
        return std::nullopt;
    }
    const uint8_t pair = half.reg.n & ~1u;
    std::vector<Ref> rr;
    for (std::size_t j = at; j-- > 0;) {
        if (IsBarrier(insts[j])) {
            return std::nullopt;
        }
        Inst& d = const_cast<Inst&>(insts[j]);
        Refs(d, rr);
        bool writes_pair = false;
        for (const Ref& r : rr) {
            writes_pair |= r.def && r.reg->bank == Bank::Virtual && (r.reg->n & ~1u) == pair;
        }
        if (!writes_pair) {
            continue;
        }
        const bool by_one = d.op == Op::Vec && static_cast<VecOp>(d.u[0]) == VecOp::Mul &&
                            d.src[1].reg.bank == Bank::Const && d.src[1].reg.n == 2 &&
                            d.src[1].swz == XXXX && d.src[1].mod == Mod::None;
        const bool move = d.op == Op::MoveInternal;
        if (!(by_one || move) || d.pred != Pred::None || d.dst.reg.n != pair ||
            d.dst.mask != 0x3 || d.src[0].reg.bank == Bank::Internal ||
            d.src[0].mod != Mod::None) {
            return std::nullopt;
        }
        if (d.src[0].reg.bank == Bank::Virtual) {
            // A copy of another virtual pair: that pair must not have been written since.
            const uint8_t src_pair = d.src[0].reg.n & ~1u;
            for (std::size_t k = j + 1; k < at; k++) {
                Refs(const_cast<Inst&>(insts[k]), rr);
                for (const Ref& r : rr) {
                    if (r.def && r.reg->bank == Bank::Virtual && (r.reg->n & ~1u) == src_pair) {
                        return std::nullopt;
                    }
                }
            }
        }
        // The half reads lanes x/y of the pair; the copy's lane l came from src.swz[l].
        Swizzle swz;
        for (int l = 0; l < 4; l++) {
            const Ch ch = half.swz[l];
            swz[l] = ch <= Ch::Y ? d.src[0].swz[static_cast<int>(ch)] : ch;
        }
        return Src{d.src[0].reg, swz, half.mod};
    }
    return std::nullopt;
}

void Lowering::TwoWideChains() {
    // A chain of four-wide ops accumulating in the internal registers - `mad i2, v44, i0,
    // i1; mad i1, v48, i0, i2; mad i2, v52, i0, i1`, the compiler's skinning idiom, each
    // link reading the previous result as a gpi operand and the last moved out to unified
    // pairs - becomes two-wide ops on the unified store: each link two vmad2/vec ops over
    // fresh virtual quads, reading the loaded operands' quads directly, the last link
    // writing the pairs the moves went to. Smash's skinning program: 11 instructions per
    // bone group to 8, and the loads that fed the internal registers die. A single op
    // followed by its moves is DirectTwoWide's case and is left to it.
    static const bool no_chains = GxmFlag("usse_nochains");
    if (no_chains) {
        return;
    }
    const auto candidate = [](const Inst& inst) {
        const bool is_mad = inst.op == Op::Mad && inst.b[0];
        const bool is_vec = inst.op == Op::Vec &&
                            (static_cast<VecOp>(inst.u[0]) == VecOp::Add ||
                             static_cast<VecOp>(inst.u[0]) == VecOp::Mul ||
                             static_cast<VecOp>(inst.u[0]) == VecOp::Max ||
                             static_cast<VecOp>(inst.u[0]) == VecOp::Min);
        return (is_mad || is_vec) && !IsInternalLoad(inst) &&
               inst.dst.reg.bank == Bank::Internal && inst.dst.mask == 0xf &&
               inst.pred == Pred::None;
    };
    const auto reads_reg = [](Inst& inst, uint8_t k, std::vector<Ref>& rr) {
        Refs(inst, rr);
        for (const Ref& r : rr) {
            if (!r.def && r.reg->bank == Bank::Internal && r.reg->n == k) {
                return true;
            }
        }
        return false;
    };
    const auto writes_reg = [](Inst& inst, uint8_t k, std::vector<Ref>& rr) {
        Refs(inst, rr);
        for (const Ref& r : rr) {
            if (r.def && r.reg->bank == Bank::Internal && r.reg->n == k) {
                return true;
            }
        }
        return false;
    };
    std::vector<Ref> rr;
    std::vector<bool> drop(insts.size(), false);
    // Replacement ops keyed by the original instruction they stand at.
    std::vector<std::vector<Inst>> replace(insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (drop[i] || !candidate(insts[i])) {
            continue;
        }
        // Follow the result.
        std::vector<std::size_t> links{i};
        std::vector<std::size_t> moves;
        uint8_t cur = insts[i].dst.reg.n;
        std::size_t pos = i;
        bool ok = true;
        while (ok) {
            std::size_t reader = SIZE_MAX;
            for (std::size_t j = pos + 1; j < insts.size(); j++) {
                if (IsBarrier(insts[j]) || drop[j]) {
                    break;
                }
                if (reads_reg(insts[j], cur, rr)) {
                    reader = j;
                    break;
                }
                if (writes_reg(insts[j], cur, rr)) {
                    break;
                }
            }
            if (reader == SIZE_MAX) {
                ok = false;
                break;
            }
            Inst& r = insts[reader];
            if (candidate(r) && r.src[0].reg.bank != Bank::Internal &&
                ((r.src[1].reg.bank == Bank::Internal && r.src[1].reg.n == cur) ||
                 (r.op == Op::Mad && r.src[2].reg.bank == Bank::Internal &&
                  r.src[2].reg.n == cur)) &&
                r.dst.reg.n != cur) {
                // The link must be the only reader before the register is written again.
                bool sole = true;
                for (std::size_t j = reader + 1; j < insts.size() && sole; j++) {
                    if (IsBarrier(insts[j]) || writes_reg(insts[j], cur, rr)) {
                        break;
                    }
                    sole = !reads_reg(insts[j], cur, rr);
                }
                if (!sole) {
                    ok = false;
                    break;
                }
                links.push_back(reader);
                cur = r.dst.reg.n;
                pos = reader;
                continue;
            }
            // Terminal: the moves out, and nothing else reading the register after.
            std::size_t j = reader;
            while (j < insts.size() && insts[j].op == Op::MoveInternal &&
                   insts[j].src[0].reg.bank == Bank::Internal && insts[j].src[0].reg.n == cur &&
                   insts[j].pred == Pred::None && insts[j].dst.reg.bank != Bank::Internal &&
                   (insts[j].dst.mask & ~3u) == 0 &&
                   (insts[j].src[0].swz == XYZW || insts[j].src[0].swz == ZWZW)) {
                moves.push_back(j);
                j++;
            }
            if (moves.empty() || InternalRead(j, cur, 0xf)) {
                ok = false;
            }
            break;
        }
        if (!ok || links.size() < 2 || moves.size() > 2) {
            continue;
        }
        // Nothing between the last link and its moves may touch the pairs the moves write.
        for (std::size_t j = links.back() + 1; j < moves.front() && ok; j++) {
            if (drop[j]) {
                continue;
            }
            Refs(insts[j], rr);
            for (const Ref& r : rr) {
                for (std::size_t m : moves) {
                    const Reg& d = insts[m].dst.reg;
                    if (r.reg->bank == d.bank &&
                        (IsVirtual(d) ? QuadOf(*r.reg) == QuadOf(d) : r.reg->n / 2 == d.n / 2)) {
                        ok = false;
                    }
                }
            }
        }
        if (!ok) {
            continue;
        }
        // Convert link by link. An internal operand is either the previous link's result,
        // now in a virtual quad, or a loaded quad read through the load.
        std::optional<std::pair<uint8_t, uint32_t>> carried; // internal register -> quad
        std::vector<std::vector<Inst>> produced(links.size());
        const uint32_t first_new_quad = quad_count;
        uint32_t new_quads = 0;
        for (std::size_t li = 0; li < links.size() && ok; li++) {
            const Inst& inst = insts[links[li]];
            const bool is_mad = inst.op == Op::Mad;
            const bool last = li + 1 == links.size();
            Src eff[3];
            const int nsrc = is_mad ? 3 : 2;
            for (int n = 0; n < nsrc && ok; n++) {
                const Src& src = inst.src[n];
                if (src.reg.bank == Bank::Internal && carried && src.reg.n == carried->first) {
                    Swizzle swz = src.swz;
                    eff[n] = Src{Reg::V(static_cast<uint8_t>(carried->second * 4)), swz, src.mod};
                } else if (src.reg.bank == Bank::Internal) {
                    const auto through = LoadedSourceOf(links[li], src);
                    if (!through) {
                        ok = false;
                        break;
                    }
                    eff[n] = *through;
                } else {
                    eff[n] = src;
                }
            }
            if (!ok) {
                break;
            }
            uint32_t dst_quad = 0;
            if (!last) {
                dst_quad = first_new_quad + new_quads++;
            }
            for (int h = 0; h < 2 && ok; h++) {
                Dst dst;
                if (last) {
                    // The pair the move of this half went to; a half nobody moved out is dead.
                    const Inst* move = nullptr;
                    for (std::size_t m : moves) {
                        if ((insts[m].src[0].swz == XYZW) == (h == 0)) {
                            move = &insts[m];
                        }
                    }
                    if (move == nullptr) {
                        continue;
                    }
                    dst = move->dst;
                } else {
                    dst = Dst{Reg::V(static_cast<uint8_t>(dst_quad * 4 + 2 * h)), 0x3};
                }
                Inst d;
                d.pred = Pred::None;
                d.dst = dst;
                if (is_mad) {
                    const auto src0_ok = [](const Src& x) {
                        return (x.reg.bank == Bank::Temp || x.reg.bank == Bank::PrimAttr ||
                                x.reg.bank == Bank::Virtual) &&
                               x.mod != Mod::Neg && x.mod != Mod::NegAbs;
                    };
                    auto s0 = HalfSource(eff[0], h, false);
                    auto s1 = HalfSource(eff[1], h, false);
                    auto s2 = HalfSource(eff[2], h, false);
                    for (auto* x : {&s0, &s1, &s2}) {
                        if (*x) {
                            if (const auto copied = CopiedSourceOf(links[li], **x)) {
                                *x = copied;
                            }
                        }
                    }
                    uint64_t code;
                    const auto legal = [&](const std::optional<Src>& x) {
                        return x && x->reg.bank != Bank::Const && x->reg.bank != Bank::Internal &&
                               Encoder::Mad2Swizzle(x->swz, code);
                    };
                    if (!legal(s0) || !legal(s1) || !legal(s2)) {
                        ok = false;
                        break;
                    }
                    if (!src0_ok(*s0) && src0_ok(*s1) && s0->mod == Mod::None) {
                        std::swap(s0, s1);
                    }
                    if (!src0_ok(*s0) || s1->mod != Mod::None || s2->mod != Mod::None) {
                        ok = false;
                        break;
                    }
                    d.op = Op::Mad2;
                    d.src[0] = *s0;
                    d.src[1] = *s1;
                    d.src[2] = *s2;
                } else {
                    auto a = HalfSource(eff[0], h, false);
                    auto b = HalfSource(eff[1], h, true);
                    for (auto* x : {&a, &b}) {
                        if (*x) {
                            if (const auto copied = CopiedSourceOf(links[li], **x)) {
                                *x = copied;
                            }
                        }
                    }
                    if (!a || !b || a->reg.bank == Bank::Internal || b->reg.bank == Bank::Internal) {
                        ok = false;
                        break;
                    }
                    d = inst;
                    d.dst = dst;
                    d.src[0] = *a;
                    d.src[1] = *b;
                    d.pred = Pred::None;
                }
                produced[li].push_back(d);
            }
            if (ok && !last) {
                carried = std::make_pair(inst.dst.reg.n, dst_quad);
            }
        }
        if (!ok) {
            continue;
        }
        quad_count = first_new_quad + new_quads;
        if (std::getenv("USSE_IR_DUMP") != nullptr) {
            std::fprintf(stderr, "chain: %zu links at %zu, %zu moves\n", links.size(), i,
                         moves.size());
        }
        for (std::size_t li = 0; li < links.size(); li++) {
            replace[links[li]] = produced[li];
            drop[links[li]] = true;
        }
        for (std::size_t m : moves) {
            drop[m] = true;
        }
    }
    std::vector<Inst> out;
    out.reserve(insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!drop[i]) {
            out.push_back(insts[i]);
            continue;
        }
        for (const Inst& d : replace[i]) {
            out.push_back(d);
        }
    }
    insts.swap(out);
}

void Lowering::DirectTwoWide() {
    std::vector<Inst> out;
    out.reserve(insts.size());
    std::size_t i = 0;
    while (i < insts.size()) {
        const Inst& inst = insts[i];
        static const bool no_mad_fold = GxmFlag("usse_nomadfold");
        const bool is_mad = !no_mad_fold && inst.op == Op::Mad && inst.b[0];
        const bool candidate =
            (is_mad || (inst.op == Op::Vec &&
                        (static_cast<VecOp>(inst.u[0]) == VecOp::Add ||
                         static_cast<VecOp>(inst.u[0]) == VecOp::Mul ||
                         static_cast<VecOp>(inst.u[0]) == VecOp::Max ||
                         static_cast<VecOp>(inst.u[0]) == VecOp::Min))) &&
            inst.dst.reg.bank == Bank::Internal && inst.dst.mask == 0xf &&
            inst.pred == Pred::None;
        if (!candidate) {
            out.push_back(inst);
            i++;
            continue;
        }
        // The moves out of the result: lanes xy to one pair, zw to another, nothing else
        // reading the register afterwards.
        const uint8_t k = inst.dst.reg.n;
        std::size_t j = i + 1;
        std::vector<Inst> direct;
        bool ok = true;
        uint8_t lanes_read = 0;
        while (j < insts.size() && insts[j].op == Op::MoveInternal &&
               insts[j].src[0].reg.bank == Bank::Internal && insts[j].src[0].reg.n == k &&
               insts[j].pred == Pred::None && insts[j].dst.reg.bank != Bank::Internal &&
               (insts[j].dst.mask & ~3u) == 0) {
            const Inst& move = insts[j];
            int h;
            if (move.src[0].swz == XYZW) {
                h = 0;
            } else if (move.src[0].swz == ZWZW) {
                h = 1;
            } else {
                ok = false;
                break;
            }
            // A mad's second and third sources are internal registers with the vec3
            // tables' lane picks; a vec op's second source has the standard table only.
            // An internal register a broadcast load filled (`vec i0, sa42.wwww, 1`) holds
            // the same value in every lane: read it as xx for either half, which is in
            // every table, the way the compiler reads its `i1.xxxx`.
            const auto broadcast = [&](const Src& src) -> Src {
                if (src.reg.bank != Bank::Internal) {
                    return src;
                }
                const auto through = LoadedSourceOf(i, src);
                if (!through) {
                    return src;
                }
                for (std::size_t j = i; j-- > 0;) {
                    if (IsBarrier(insts[j])) {
                        break;
                    }
                    if (IsInternalLoad(insts[j]) && insts[j].dst.reg.n == src.reg.n) {
                        const Swizzle& w = insts[j].src[0].swz;
                        if (w[0] == w[1] && w[1] == w[2] && w[2] == w[3] && w[0] <= Ch::W) {
                            return Src{src.reg, {Ch::X, Ch::X, Ch::X, Ch::X}, src.mod};
                        }
                        break;
                    }
                }
                return src;
            };
            const auto a = HalfSource(inst.src[0], h, false);
            const auto b = HalfSource(is_mad ? broadcast(inst.src[1]) : inst.src[1], h, !is_mad);
            std::optional<Src> c;
            if (is_mad) {
                c = HalfSource(broadcast(inst.src[2]), h, false);
                // vmad2 first: it reads the loaded quads directly and lets the loads die.
                const bool vec3 = a && b && c && InVec3Tables(a->swz) && InVec3Tables(b->swz) &&
                                  InVec3Tables(c->swz);
                bool mad2_done = false;
                {
                    // The three-source two-wide mad (vmad2) instead: every source a
                    // pair with an xx, yy or xy swizzle. An internal source that was just
                    // loaded from a unified quad is read from that quad.
                    const auto through = [&](const Src& src) -> std::optional<Src> {
                        Src eff = src;
                        if (src.reg.bank == Bank::Internal) {
                            // The nearest earlier write of the register must be a load
                            // of a whole quad, with the quad untouched since.
                            std::optional<Src> loaded;
                            for (std::size_t j = i; j-- > 0;) {
                                if (IsBarrier(insts[j])) {
                                    break;
                                }
                                std::vector<Ref> rr;
                                Refs(insts[j], rr);
                                bool writes_reg = false;
                                for (const Ref& r : rr) {
                                    writes_reg |= r.def && r.reg->bank == Bank::Internal &&
                                                  r.reg->n == src.reg.n;
                                }
                                if (writes_reg) {
                                    const Swizzle& lw = insts[j].src[0].swz;
                                    const bool channels_only = lw[0] <= Ch::W && lw[1] <= Ch::W &&
                                                               lw[2] <= Ch::W && lw[3] <= Ch::W;
                                    if (IsInternalLoad(insts[j]) && channels_only &&
                                        insts[j].src[0].mod == Mod::None &&
                                        insts[j].src[0].reg.bank != Bank::Const) {
                                        loaded = insts[j].src[0];
                                        // The quad must not be written between.
                                        for (std::size_t k = j + 1; k < i && loaded; k++) {
                                            Refs(insts[k], rr);
                                            for (const Ref& r : rr) {
                                                if (r.def && r.reg->bank == loaded->reg.bank &&
                                                    QuadOf(*r.reg) == QuadOf(loaded->reg)) {
                                                    loaded.reset();
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                            if (!loaded) {
                                return std::nullopt;
                            }
                            // Compose the read's channels through the load.
                            Swizzle swz;
                            for (int l = 0; l < 4; l++) {
                                const Ch ch = src.swz[l];
                                swz[l] = ch <= Ch::W ? loaded->swz[static_cast<int>(ch)] : ch;
                            }
                            eff = Src{loaded->reg, swz, Compose(src.mod, loaded->mod)};
                        }
                        const auto half = HalfSource(eff, h, false);
                        uint64_t code;
                        if (!half || half->reg.bank == Bank::Const || half->reg.bank == Bank::Internal ||
                            !Encoder::Mad2Swizzle(half->swz, code)) {
                            return std::nullopt;
                        }
                        return half;
                    };
                    auto s0 = through(inst.src[0]);
                    auto s1 = through(inst.src[1]);
                    auto s2 = through(inst.src[2]);
                    // A source that is a copy of a uniform (`mov v44.xy, sa30.xy`) reads the
                    // uniform; the copies then die. src0 keeps the copy when the original
                    // is a bank src0 cannot take.
                    for (auto* x : {&s1, &s2}) {
                        if (*x) {
                            if (const auto copied = CopiedSourceOf(i, **x)) {
                                *x = copied;
                            }
                        }
                    }
                    if (s0) {
                        if (const auto copied = CopiedSourceOf(i, *s0);
                            copied && (copied->reg.bank == Bank::PrimAttr ||
                                       copied->reg.bank == Bank::Virtual ||
                                       copied->reg.bank == Bank::Temp)) {
                            s0 = copied;
                        }
                    }
                    // src0 is a temporary or a primary attribute: swap the product's
                    // operands when needed.
                    const auto src0_ok = [](const Src& x) {
                        return (x.reg.bank == Bank::Temp || x.reg.bank == Bank::PrimAttr ||
                                x.reg.bank == Bank::Virtual) &&
                               x.mod != Mod::Neg && x.mod != Mod::NegAbs;
                    };
                    if (s0 && s1 && !src0_ok(*s0) && src0_ok(*s1) && s0->mod == Mod::None) {
                        std::swap(s0, s1);
                    }
                    if (s0 && s1 && s2 && s1->mod == Mod::None && s2->mod == Mod::None &&
                        src0_ok(*s0)) {
                        Inst d;
                        d.op = Op::Mad2;
                        d.dst = move.dst;
                        d.src[0] = *s0;
                        d.src[1] = *s1;
                        d.src[2] = *s2;
                        d.pred = Pred::None;
                        direct.push_back(d);
                        lanes_read |= static_cast<uint8_t>(h == 0 ? 0x3 : 0xc);
                        j++;
                        mad2_done = true;
                    }
                }
                if (mad2_done) {
                    continue;
                }
                if (!vec3) {
                    // The four-wide mad with a pair mask, the compiler's tail of a chain
                    // (`mad.f32 sa86.xy, sa50.xx, i1.zw, i0.zw`): the internal sources
                    // keep their register and take the table entry whose first two
                    // channels are this half's lanes; the unified source reads its pair.
                    const auto pair_swz = [](const Src& src, int half) -> std::optional<Src> {
                        const Ch c0 = src.swz[2 * half], c1 = src.swz[2 * half + 1];
                        if (c0 > Ch::W || c1 > Ch::W) {
                            return std::nullopt;
                        }
                        const auto swz = Vec4SwizzleForPair(c0, c1);
                        if (!swz) {
                            return std::nullopt;
                        }
                        return Src{src.reg, *swz, src.mod};
                    };
                    const auto a4 = a ? pair_swz(Src{a->reg, {a->swz[0], a->swz[1], a->swz[0],
                                                              a->swz[1]}, a->mod}, 0)
                                      : std::nullopt;
                    const auto b4 = pair_swz(broadcast(inst.src[1]), h);
                    const auto c4 = pair_swz(broadcast(inst.src[2]), h);
                    if (!a4 || !b4 || !c4) {
                        ok = false;
                        break;
                    }
                    Inst d = inst;
                    d.dst = move.dst;
                    d.src[0] = *a4;
                    d.src[1] = *b4;
                    d.src[2] = *c4;
                    d.b[0] = true; // the four-wide form, masked to the pair
                    direct.push_back(d);
                    lanes_read |= static_cast<uint8_t>(h == 0 ? 0x3 : 0xc);
                    j++;
                    continue;
                }
            }
            if (!a || !b) {
                ok = false;
                break;
            }
            Inst d = inst;
            d.dst = move.dst;
            d.src[0] = *a;
            d.src[1] = *b;
            if (is_mad) {
                d.src[2] = *c;
                d.b[0] = false; // the two-wide form takes the three-wide tables
            }
            direct.push_back(d);
            lanes_read |= static_cast<uint8_t>(h == 0 ? 0x3 : 0xc);
            j++;
        }
        if (!ok || direct.empty() || InternalRead(j, k, 0xf)) {
            out.push_back(inst);
            i++;
            continue;
        }
        for (const Inst& d : direct) {
            out.push_back(d);
        }
        i = j;
    }
    insts.swap(out);
}

namespace {


/// A two-wide read of an internal register that a load filled from `loaded`, as the same
/// read of the load's source: the channels composed, then the pair holding them.
std::optional<Src> ReadThroughLoad(const Src& read, const Src& loaded, bool second) {
    Swizzle swz;
    for (int l = 0; l < 2; l++) {
        const Ch c = read.swz[l];
        swz[l] = c <= Ch::W ? loaded.swz[static_cast<int>(c)] : c;
    }
    swz[2] = swz[0];
    swz[3] = swz[1];
    const Src composed{loaded.reg, swz, Compose(read.mod, loaded.mod)};
    return HalfSource(composed, 0, second);
}

bool SameSrc(const Src& a, const Src& b) {
    return SameReg(a.reg, b.reg) && a.swz == b.swz && a.mod == b.mod;
}

bool Commutative(VecOp op) {
    return op == VecOp::Add || op == VecOp::Mul || op == VecOp::Max || op == VecOp::Min;
}

Cond Flipped(Cond c) {
    switch (c) {
    case Cond::Gt: return Cond::Lt;
    case Cond::Ge: return Cond::Le;
    case Cond::Lt: return Cond::Gt;
    case Cond::Le: return Cond::Ge;
    default: return c;
    }
}

bool ReadsInternal(const Src& s, uint8_t k) {
    return s.reg.bank == Bank::Internal && s.reg.n == k;
}

/// A compare operand as the single register it reads: bank and absolute index (an
/// internal register's lanes count as four registers), or a constant kept as written.
struct TestOperand {
    Src orig;
    bool is_const;
    Bank bank;
    int idx;
    Mod mod;
};

/// A compare reading internal register `k`, filled by `loaded`, rewritten to read the
/// load's source. The operands become (register, channel) pairs: the first source names a
/// pair and a channel, the second the same channel of its pair or, as a scalar, an even
/// register; the operands swap (condition flipped) when only that order encodes.
bool FoldTest(Inst& d, uint8_t k, const Src& loaded, std::vector<int>& lanes_read) {
    const uint8_t chan = d.u[2];
    const bool scalar = d.b[0];
    TestOperand ops[2];
    for (int s = 0; s < 2; s++) {
        const Src& src = d.src[s];
        const int lane = (s == 1 && scalar) ? 0 : chan;
        TestOperand& o = ops[s];
        o.orig = src;
        o.is_const = src.reg.bank == Bank::Const;
        o.bank = src.reg.bank;
        o.mod = src.mod;
        o.idx = (src.reg.bank == Bank::Internal ? src.reg.n * 4 : src.reg.n) + lane;
        if (ReadsInternal(src, k)) {
            const Ch c = loaded.swz[lane];
            if (c > Ch::W) {
                return false;
            }
            o.bank = loaded.reg.bank;
            o.is_const = o.bank == Bank::Const;
            o.idx = loaded.reg.n + static_cast<int>(c);
            o.mod = Compose(src.mod, loaded.mod);
            if (o.is_const) {
                return false; // a constant channel: not worth the table
            }
            lanes_read.push_back(o.idx);
        }
    }
    const auto as_first = [&](const TestOperand& o, Src& out, int& out_chan) {
        if (o.mod != Mod::None && o.mod != Mod::Neg) {
            return false;
        }
        if (o.is_const) {
            out = o.orig;
            out_chan = chan;
            return true;
        }
        if (o.bank == Bank::Internal) {
            out = Src{Reg::I(static_cast<uint8_t>(o.idx / 4)), XYZW, o.mod};
            out_chan = o.idx % 4;
            return true;
        }
        out = Src{Reg{o.bank, static_cast<uint8_t>(o.idx & ~1)}, XYZW, o.mod};
        out_chan = o.idx & 1;
        return true;
    };
    const auto as_second = [&](const TestOperand& o, int first_chan, Src& out, bool& out_scalar) {
        if (o.mod != Mod::None) {
            return false;
        }
        if (o.is_const) {
            // As written: scalar reads its first entry whatever the channel, a pair read
            // needs the channel unchanged.
            if (!scalar && first_chan != chan) {
                return false;
            }
            out = o.orig;
            out_scalar = scalar;
            return true;
        }
        if (o.bank == Bank::Internal) {
            if (o.idx % 4 != first_chan) {
                return false;
            }
            out = Src{Reg::I(static_cast<uint8_t>(o.idx / 4)), XYZW, Mod::None};
            out_scalar = false;
            return true;
        }
        if ((o.idx & 1) == first_chan) {
            out = Src{Reg{o.bank, static_cast<uint8_t>(o.idx - first_chan)}, XYZW, Mod::None};
            out_scalar = false;
            return true;
        }
        if ((o.idx & 1) == 0) {
            out = Src{Reg{o.bank, static_cast<uint8_t>(o.idx)}, XYZW, Mod::None};
            out_scalar = true;
            return true;
        }
        return false;
    };
    for (int order = 0; order < 2; order++) {
        const TestOperand& a = ops[order];
        const TestOperand& b = ops[1 - order];
        Src s1, s2;
        int c1;
        bool sc;
        if (!as_first(a, s1, c1) || !as_second(b, c1, s2, sc)) {
            continue;
        }
        d.src[0] = s1;
        d.src[1] = s2;
        d.u[2] = static_cast<uint8_t>(c1);
        d.b[0] = sc;
        if (order == 1) {
            d.u[0] = static_cast<uint8_t>(Flipped(static_cast<Cond>(d.u[0])));
        }
        return true;
    }
    return false;
}

} // Anonymous namespace

void Lowering::LoadSources() {
    // A load whose every reader is a two-wide op reading lanes one pair of the source
    // holds: the readers read the source, the load goes. The reader's own two-wide
    // encoding is what the compiler writes for such ops (`mul.f32 o0.xy, pa0.xy, sa12.xy`).
    std::vector<Ref> refs;
    std::vector<bool> drop(insts.size(), false);
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!IsInternalLoad(insts[i])) {
            continue;
        }
        const uint8_t k = insts[i].dst.reg.n;
        const Src loaded = insts[i].src[0];
        std::vector<std::pair<std::size_t, Inst>> rewrites;
        bool ok = true;
        std::set<int> clobbered; // registers of the source's bank written since the load
        std::vector<int> lanes_read;
        for (std::size_t j = i + 1; j < insts.size() && ok; j++) {
            Inst& inst = insts[j];
            if (IsBarrier(inst)) {
                break;
            }
            Refs(inst, refs);
            bool reads = false, writes = false;
            for (const Ref& r : refs) {
                if (r.reg->bank == Bank::Internal && r.reg->n == k) {
                    (r.def ? writes : reads) = true;
                }
            }
            if (reads) {
                Inst d = inst;
                lanes_read.clear();
                if (d.op == Op::Test) {
                    ok = FoldTest(d, k, loaded, lanes_read);
                } else if (d.op == Op::PackS16) {
                    // Reads one lane of a register: the load's source lane.
                    const Ch c = loaded.swz[d.u[0] & 3];
                    ok = loaded.mod == Mod::None && c <= Ch::W && loaded.reg.bank != Bank::Const;
                    if (ok) {
                        d.reg[1] = loaded.reg;
                        d.u[0] = static_cast<uint8_t>(c);
                        lanes_read.push_back(loaded.reg.n + static_cast<int>(c));
                    }
                } else if (d.op == Op::Mad2) {
                    // Every source that reads the register takes the load's source pair;
                    // the swizzle must stay xx, yy or xy, and modifiers are not encodable.
                    ok = true;
                    for (int s = 0; s < 3 && ok; s++) {
                        if (!ReadsInternal(d.src[s], k)) {
                            continue;
                        }
                        const auto through = ReadThroughLoad(d.src[s], loaded, false);
                        uint64_t code;
                        ok = through && through->reg.bank != Bank::Const &&
                             Encoder::Mad2Swizzle(through->swz, code) &&
                             (s == 0 ? (through->mod == Mod::None || through->mod == Mod::Abs)
                                     : through->mod == Mod::None) &&
                             (s != 0 || through->reg.bank == Bank::Temp ||
                              through->reg.bank == Bank::PrimAttr || through->reg.bank == Bank::Virtual);
                        if (ok) {
                            for (int l = 0; l < 2; l++) {
                                if (through->swz[l] <= Ch::W) {
                                    lanes_read.push_back(through->reg.n + static_cast<int>(through->swz[l]));
                                }
                            }
                            d.src[s] = *through;
                        }
                    }
                } else if (d.op == Op::Vec && d.dst.mask == 0xf && !IsInternalLoad(d)) {
                    // A four-wide vector op reads any unified-store bank in either source
                    // (the compiler's `mul.f32 i0.xyzw, sa64.xyzw, sa46.yyyy`): the load's
                    // source with the channels composed, where the swizzle is in the op's
                    // table - both tables for the first source, the standard one for the
                    // second - and the source is not a constant.
                    ok = true;
                    for (int sidx = 0; sidx < 2 && ok; sidx++) {
                        if (!ReadsInternal(d.src[sidx], k)) {
                            continue;
                        }
                        Swizzle swz;
                        for (int l = 0; l < 4; l++) {
                            const Ch ch = d.src[sidx].swz[l];
                            swz[l] = ch <= Ch::W ? loaded.swz[static_cast<int>(ch)] : ch;
                        }
                        const Mod mod = Compose(d.src[sidx].mod, loaded.mod);
                        const bool table = sidx == 0 || InVec4StdTable(swz);
                        ok = loaded.reg.bank != Bank::Const && table &&
                             (sidx == 0 || (mod != Mod::Neg && mod != Mod::NegAbs));
                        if (ok) {
                            d.src[sidx] = Src{loaded.reg, swz, mod};
                            for (int l = 0; l < 4; l++) {
                                if (swz[l] <= Ch::W) {
                                    lanes_read.push_back(loaded.reg.n + static_cast<int>(swz[l]));
                                }
                            }
                        }
                    }
                } else {
                    ok = d.op == Op::Vec && d.dst.reg.bank != Bank::Internal &&
                         (d.dst.mask & ~3u) == 0;
                    const bool r0 = ReadsInternal(d.src[0], k);
                    bool r1 = ReadsInternal(d.src[1], k);
                    std::optional<Src> t0, t1;
                    if (ok && r0) {
                        t0 = ReadThroughLoad(d.src[0], loaded, false);
                        ok = t0.has_value();
                    }
                    if (ok && r1) {
                        t1 = ReadThroughLoad(d.src[1], loaded, true);
                        if (!t1 && !r0 && Commutative(static_cast<VecOp>(d.u[0]))) {
                            // Only as the first source (a negation, a swizzle off the
                            // standard table): swap, if the other operand goes second.
                            const Src& o = d.src[0];
                            const auto other = HalfSource(
                                Src{o.reg, {o.swz[0], o.swz[1], o.swz[0], o.swz[1]}, o.mod}, 0, true);
                            const auto first = ReadThroughLoad(d.src[1], loaded, false);
                            if (other && first) {
                                d.src[1] = *other;
                                t0 = first;
                                r1 = false;
                            }
                        }
                        ok = t0.has_value() || t1.has_value();
                        if (r1 && !t1) {
                            ok = false;
                        }
                    }
                    if (ok) {
                        if (t0) {
                            d.src[0] = *t0;
                        }
                        if (t1) {
                            d.src[1] = *t1;
                        }
                        for (const Src* t : {t0 ? &*t0 : nullptr, t1 ? &*t1 : nullptr}) {
                            for (int l = 0; t && l < 2; l++) {
                                if (t->swz[l] <= Ch::W) {
                                    lanes_read.push_back(t->reg.n + static_cast<int>(t->swz[l]));
                                }
                            }
                        }
                    }
                }
                // The reader would now read the source's current registers: they must
                // hold what the load saw.
                for (const int lane : lanes_read) {
                    if (clobbered.count(lane)) {
                        ok = false;
                    }
                }
                if (ok) {
                    rewrites.emplace_back(j, d);
                }
            }
            if (writes) {
                break;
            }
            for (const Ref& r : refs) {
                if (r.def && r.reg->bank == loaded.reg.bank) {
                    const uint8_t lanes = WriteLanes(inst, *r.reg);
                    for (int l = 0; l < 4; l++) {
                        if (lanes & (1u << l)) {
                            clobbered.insert(static_cast<int>(QuadOf(*r.reg) * 4 + l));
                        }
                    }
                }
            }
        }
        if (!ok || rewrites.empty()) {
            continue;
        }
        for (auto& [j, d] : rewrites) {
            insts[j] = d;
        }
        drop[i] = true;
    }
    std::vector<Inst> kept;
    kept.reserve(insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!drop[i]) {
            kept.push_back(insts[i]);
        }
    }
    insts.swap(kept);
}

void Lowering::InternalRegisters() {
    // The emitters name i0-i2 by hand and clobber one while another sits idle, then load
    // the same operand again. Within a block, each full write starts a range that ends at
    // its last read; the ranges are placed again over the three registers, keeping a
    // loaded operand resident as long as possible (the register whose held operand is
    // loaded again soonest is spent last), and a load of an operand a register still
    // holds, unchanged since, is dropped in favour of that register.
    struct Range {
        std::size_t start;
        std::size_t last;
        uint8_t orig;
        bool pinned;               ///< read before any full write in its block, or live out
        std::optional<Src> value;  ///< the operand a load filled it with, if nothing else wrote
        std::vector<std::pair<std::size_t, int>> refs; ///< (instruction, ref index)
        int phys = -1;
        bool drop_load = false;
    };
    std::vector<Ref> refs;
    const std::size_t n = insts.size();

    // Blocks, and the registers some block reads before writing (live across a barrier:
    // those keep their names).
    std::vector<std::pair<std::size_t, std::size_t>> blocks;
    bool exposed[3] = {false, false, false};
    for (std::size_t b = 0; b < n;) {
        std::size_t e = b;
        while (e < n && !IsBarrier(insts[e])) {
            e++;
        }
        if (e > b) {
            blocks.emplace_back(b, e);
            bool written[3] = {false, false, false};
            for (std::size_t i = b; i < e; i++) {
                Refs(insts[i], refs);
                for (const Ref& r : refs) {
                    if (r.reg->bank != Bank::Internal) {
                        continue;
                    }
                    if (!r.def && !written[r.reg->n]) {
                        exposed[r.reg->n] = true;
                    }
                }
                if (insts[i].pred == Pred::None && InternalWriteLanes(insts[i]) == 0xf) {
                    for (const Ref& r : refs) {
                        if (r.def && r.reg->bank == Bank::Internal) {
                            written[r.reg->n] = true;
                        }
                    }
                }
            }
        }
        b = e + 1;
    }

    std::vector<bool> drop(n, false);
    std::vector<int> new_name(n * 8, -1); // (instruction * 8 + ref index) -> register
    for (const auto& [b, e] : blocks) {
        std::vector<Range> ranges;
        int current[3] = {-1, -1, -1};
        std::map<std::pair<std::size_t, int>, std::size_t> range_of; // (inst, ref) -> range
        for (std::size_t i = b; i < e; i++) {
            Inst& inst = insts[i];
            Refs(inst, refs);
            const bool full = inst.pred == Pred::None && InternalWriteLanes(inst) == 0xf;
            for (int x = 0; x < static_cast<int>(refs.size()); x++) {
                const Ref& r = refs[x];
                if (r.reg->bank != Bank::Internal) {
                    continue;
                }
                const uint8_t k = r.reg->n;
                if (r.def && full) {
                    Range range{i, i, k, false, std::nullopt, {}};
                    if (IsInternalLoad(inst)) {
                        range.value = inst.src[0];
                    }
                    ranges.push_back(std::move(range));
                    current[k] = static_cast<int>(ranges.size()) - 1;
                } else if (current[k] < 0) {
                    ranges.push_back(Range{b, i, k, true, std::nullopt, {}});
                    current[k] = static_cast<int>(ranges.size()) - 1;
                }
                Range& range = ranges[current[k]];
                range.refs.emplace_back(i, x);
                range_of[{i, x}] = static_cast<std::size_t>(current[k]);
                range.last = i;
                if (r.def && !full) {
                    range.value.reset();
                }
            }
        }
        for (int k = 0; k < 3; k++) {
            if (current[k] >= 0 && exposed[k]) {
                ranges[current[k]].pinned = true;
                ranges[current[k]].last = e - 1;
            }
        }
        if (ranges.empty()) {
            continue;
        }

        // The version of each unified quad: bumped by every write, so a held operand is
        // known unchanged when its quad's version still matches.
        struct Held {
            std::optional<Src> value;
            uint32_t version = 0;
            std::size_t busy_until = 0; ///< the range's last reference
            bool busy = false;
        };
        Held state[3];
        std::map<std::pair<Bank, uint32_t>, uint32_t> versions;
        const auto version_of = [&](const Src& src) -> uint32_t {
            const auto it = versions.find({src.reg.bank, QuadOf(src.reg)});
            return it == versions.end() ? 0 : it->second;
        };
        const auto next_load = [&](const Src& value, std::size_t from) -> std::size_t {
            for (std::size_t j = from; j < e; j++) {
                if (IsInternalLoad(insts[j]) && SameSrc(insts[j].src[0], value)) {
                    return j;
                }
            }
            return SIZE_MAX;
        };
        // Ranges in order of start; pinned ones hold their register over their span.
        std::vector<std::size_t> order(ranges.size());
        for (std::size_t r = 0; r < ranges.size(); r++) {
            order[r] = r;
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t x, std::size_t y) { return ranges[x].start < ranges[y].start; });
        bool failed = false;
        std::size_t cursor = b;
        // A computed value moved out to a whole virtual quad stays resident in its
        // register: a later load of that quad is served from it (stored[p] tracks the
        // lanes moved out so far).
        uint8_t stored_lanes[3] = {0, 0, 0};
        uint32_t stored_quad[3] = {0, 0, 0};
        for (const std::size_t ri : order) {
            Range& range = ranges[ri];
            // Advance the versions to the range's start.
            for (; cursor < range.start; cursor++) {
                Inst& passed = insts[cursor];
                Refs(passed, refs);
                for (const Ref& r : refs) {
                    if (r.def && r.reg->bank != Bank::Internal) {
                        versions[{r.reg->bank, QuadOf(*r.reg)}]++;
                    }
                }
                if (passed.op == Op::MoveInternal && passed.pred == Pred::None &&
                    passed.src[0].reg.bank == Bank::Internal && passed.src[0].mod == Mod::None &&
                    passed.dst.reg.bank == Bank::Virtual && passed.dst.mask == 0x3 &&
                    (passed.dst.reg.n & 1) == 0) {
                    const auto it = range_of.find({cursor, 1}); // ref 1 is the source
                    if (it != range_of.end() && ranges[it->second].phys >= 0) {
                        const int p = ranges[it->second].phys;
                        const uint32_t q = QuadOf(passed.dst.reg);
                        const bool lo = (passed.dst.reg.n & 3) == 0 && passed.src[0].swz == XYZW;
                        const bool hi = (passed.dst.reg.n & 3) == 2 && passed.src[0].swz == ZWZW;
                        if (lo || hi) {
                            if (stored_quad[p] != q) {
                                stored_lanes[p] = 0;
                                stored_quad[p] = q;
                            }
                            stored_lanes[p] |= lo ? 0x3 : 0xc;
                            if (stored_lanes[p] == 0xf) {
                                state[p].value = Src{Reg::V(static_cast<uint8_t>(q * 4)), XYZW, Mod::None};
                                state[p].version = versions[{Bank::Virtual, q}];
                                stored_lanes[p] = 0;
                            }
                        }
                    }
                }
            }
            for (Held& h : state) {
                if (h.busy && h.busy_until < range.start) {
                    h.busy = false;
                }
            }
            if (range.pinned) {
                Held& h = state[range.orig];
                if (h.busy && h.busy_until >= range.start) {
                    // The register's current occupant reaches into a pinned use: the
                    // emitter's own naming was consistent, so this cannot happen unless a
                    // renaming put it there; give the block up.
                    failed = true;
                    break;
                }
                range.phys = range.orig;
                h = Held{std::nullopt, 0, range.last, true};
                continue;
            }
            // A register already holding the operand, unchanged since it was loaded.
            int phys = -1;
            if (range.value) {
                for (int p = 0; p < 3; p++) {
                    if (state[p].value && SameSrc(*state[p].value, *range.value) &&
                        state[p].version == version_of(*range.value)) {
                        phys = p;
                        break;
                    }
                }
            }
            if (phys >= 0) {
                range.drop_load = true;
                range.phys = phys;
                Held& h = state[phys];
                h.busy = true;
                h.busy_until = std::max(h.busy_until, range.last);
                continue;
            }
            // Otherwise a free register, the one whose held operand is wanted last.
            std::size_t best_next = 0;
            for (int p = 0; p < 3; p++) {
                if (state[p].busy) {
                    continue;
                }
                std::size_t next = SIZE_MAX;
                if (state[p].value && state[p].version == version_of(*state[p].value)) {
                    next = next_load(*state[p].value, range.start + 1);
                }
                if (phys < 0 || next > best_next || (next == best_next && p == range.orig)) {
                    phys = p;
                    best_next = next;
                }
            }
            if (phys < 0) {
                failed = true;
                break;
            }
            range.phys = phys;
            Held& h = state[phys];
            h.busy = true;
            h.busy_until = range.last;
            h.value = range.value;
            h.version = range.value ? version_of(*range.value) : 0;
            stored_lanes[phys] = 0;
        }
        if (failed) {
            continue;
        }
        for (const Range& range : ranges) {
            for (const auto& [i, x] : range.refs) {
                new_name[i * 8 + x] = range.phys;
            }
            if (range.drop_load) {
                drop[range.start] = true;
            }
        }
    }

    std::vector<Inst> kept;
    kept.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Inst& inst = insts[i];
        Refs(inst, refs);
        for (int x = 0; x < static_cast<int>(refs.size()); x++) {
            const int name = new_name[i * 8 + x];
            if (name >= 0) {
                refs[x].reg->n = static_cast<uint8_t>(name);
            }
        }
        if (!drop[i]) {
            kept.push_back(inst);
        }
    }
    insts.swap(kept);
}

namespace {

/// The lanes of `src` an instruction reads, from its swizzle under the destination mask
/// (a lane the mask skips is not read); 0xf where the shape is not lane-selective.
uint8_t LanesRead(const Inst& inst, const Src& src, uint8_t mask) {
    uint8_t lanes = 0;
    for (int l = 0; l < 4; l++) {
        if ((mask >> l) & 1 && src.swz[l] <= Ch::W) {
            lanes |= static_cast<uint8_t>(1u << static_cast<int>(src.swz[l]));
        }
    }
    (void)inst;
    return lanes;
}

} // Anonymous namespace

void Lowering::Dump(const char* stage) const {
    static const bool enabled = std::getenv("USSE_IR_DUMP") != nullptr;
    if (!enabled) {
        return;
    }
    std::fprintf(stderr, "== %s (%zu)\n", stage, insts.size());
    for (std::size_t i = 0; i < insts.size(); i++) {
        std::fprintf(stderr, "%4zu: %s\n", i, Describe(const_cast<Inst&>(insts[i])).c_str());
    }
}

void Lowering::CheckInternals() const {
    std::vector<Ref> refs;
    // stale[k]: lanes of i<k> no instruction has written since the last barrier (or the
    // start); a read of one is a read of another instance's register on the hardware.
    uint8_t stale[3] = {0xf, 0xf, 0xf};
    for (std::size_t i = 0; i < insts.size(); i++) {
        Inst& inst = const_cast<Inst&>(insts[i]);
        if (IsBarrier(inst)) {
            stale[0] = stale[1] = stale[2] = 0xf;
            continue;
        }
        // Reads.
        const auto check = [&](const Reg& r, uint8_t lanes) {
            if (r.bank != Bank::Internal || (lanes & stale[r.n]) == 0) {
                return;
            }
            std::string context;
            for (std::size_t j = i >= 10 ? i - 10 : 0; j <= i; j++) {
                context += "\n  " + std::to_string(j) + ": " + Describe(const_cast<Inst&>(insts[j]));
            }
            Fail("internal register i" + std::to_string(r.n) +
                 " read across a wait, load or branch at instruction " + std::to_string(i) +
                 context);
        };
        switch (inst.op) {
        case Op::Vec:
        case Op::Vec16:
            check(inst.src[0].reg, LanesRead(inst, inst.src[0], inst.dst.mask));
            check(inst.src[1].reg, LanesRead(inst, inst.src[1], inst.dst.mask));
            break;
        case Op::Mad: // component-wise: the destination mask selects the lanes read
            for (int k = 0; k < 3; k++) {
                check(inst.src[k].reg,
                      LanesRead(inst, inst.src[k], inst.dst.mask & (inst.b[0] ? 0xf : 0x7)));
            }
            break;
        case Op::Dot:
            check(inst.src[0].reg, LanesRead(inst, inst.src[0], inst.b[0] ? 0xf : 0x7));
            check(inst.src[1].reg, LanesRead(inst, inst.src[1], inst.b[0] ? 0xf : 0x7));
            break;
        case Op::Comp:
            check(inst.src[0].reg, static_cast<uint8_t>(1u << (inst.u[1] & 3)));
            break;
        case Op::MoveInternal:
            check(inst.src[0].reg, LanesRead(inst, inst.src[0], inst.dst.mask));
            break;
        case Op::CondMove:
        case Op::Mad2:
            for (int k = 0; k < 3; k++) {
                check(inst.src[k].reg, LanesRead(inst, inst.src[k], inst.dst.mask));
            }
            break;
        case Op::Test:
            check(inst.src[0].reg, static_cast<uint8_t>(1u << (inst.u[2] & 3)));
            check(inst.src[1].reg, static_cast<uint8_t>(1u << (inst.b[0] ? 0 : (inst.u[2] & 3))));
            break;
        case Op::PackS16:
            check(inst.reg[1], static_cast<uint8_t>(1u << (inst.u[0] & 3)));
            break;
        default:
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (!r.def) {
                    check(*r.reg, 0xf);
                }
            }
            break;
        }
        // Writes: unpredicated ones make their lanes fresh.
        if (inst.pred == Pred::None) {
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (r.def && r.reg->bank == Bank::Internal) {
                    stale[r.reg->n] &= static_cast<uint8_t>(~InternalWriteLanes(inst));
                }
            }
        }
    }
}

void Lowering::ResultsIntoInternal() {
    // A quad assembled lane by lane (dots, single-lane moves) only to be loaded whole into
    // an internal register, and dead afterwards, is assembled in that register instead:
    // `dot.f32 i1.x, sa12.xyzw, i0.xyzw` is how the compiler writes a matrix multiply.
    // The register must be untouched between the first lane and the load, and no
    // barrier may lie between.
    Liveness();
    std::vector<Ref> refs;
    std::vector<bool> drop(insts.size(), false);
    for (std::size_t m = 0; m < insts.size(); m++) {
        const Inst& load = insts[m];
        if (drop[m] || !IsInternalLoad(load) || load.src[0].swz != XYZW ||
            load.src[0].mod != Mod::None || load.src[0].reg.bank != Bank::Virtual ||
            (load.src[0].reg.n & 3) != 0) {
            continue;
        }
        const uint32_t q = QuadOf(load.src[0].reg);
        const uint8_t k = load.dst.reg.n;
        if (live[m][q] != 0) {
            continue; // the quad lives on
        }
        // Walk back: the last writer of each lane, nothing else touching the quad or
        // the internal register.
        std::vector<std::size_t> defs; // instruction index per lane, or SIZE_MAX
        defs.assign(4, SIZE_MAX);
        uint8_t covered = 0;
        bool ok = true;
        std::size_t first = m;
        for (std::size_t i = m; i-- > 0 && ok && covered != 0xf;) {
            Inst& inst = insts[i];
            if (IsBarrier(inst)) {
                break;
            }
            Refs(inst, refs);
            bool touches_q = false;
            for (const Ref& r : refs) {
                if (r.reg->bank == Bank::Internal && r.reg->n == k) {
                    ok = false; // the register is in use across the span
                }
                if (r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == q) {
                    touches_q = true;
                    if (!r.def) {
                        ok = false; // a read of the quad before the load
                    }
                }
            }
            if (!ok || !touches_q) {
                continue;
            }
            // A single-lane writer of a not yet covered lane.
            const bool single_lane_dot = inst.op == Op::Dot && inst.pred == Pred::None &&
                                         (inst.dst.mask == 1 || inst.dst.mask == 2);
            const bool single_lane_move = inst.op == Op::MoveInternal && inst.pred == Pred::None &&
                                          (inst.dst.mask == 1 || inst.dst.mask == 2) &&
                                          inst.src[0].reg.bank != Bank::Internal;
            if (!single_lane_dot && !single_lane_move) {
                ok = false;
                break;
            }
            const uint8_t lanes = WriteLanes(inst, inst.dst.reg);
            if ((lanes & covered) != 0 || (lanes & (lanes - 1)) != 0) {
                ok = false; // rewritten lane, or more than one
                break;
            }
            for (int l = 0; l < 4; l++) {
                if (lanes & (1u << l)) {
                    defs[l] = i;
                }
            }
            covered |= lanes;
            first = i;
        }
        if (!ok || covered != 0xf) {
            continue;
        }
        // Between the first writer and the load nothing else may touch the register.
        for (std::size_t i = first; i < m && ok; i++) {
            Refs(insts[i], refs);
            for (const Ref& r : refs) {
                if (r.reg->bank == Bank::Internal && r.reg->n == k && !r.def &&
                    !(insts[i].op == Op::Dot || insts[i].op == Op::MoveInternal)) {
                    ok = false;
                }
            }
        }
        if (!ok) {
            continue;
        }
        for (int l = 0; l < 4; l++) {
            Inst& d = insts[defs[l]];
            const int pair_lane = static_cast<int>(d.dst.reg.n & 1) + (d.dst.mask == 2 ? 1 : 0);
            (void)pair_lane;
            if (d.op == Op::MoveInternal) {
                // The pair move's lane l reads swz[l]; as an internal lane it reads the
                // same channel at that lane.
                const int src_lane = d.dst.mask == 2 ? 1 : 0;
                Swizzle swz = d.src[0].swz;
                swz[l] = d.src[0].swz[src_lane];
                d.src[0].swz = swz;
            }
            d.dst = Dst{Reg::I(k), static_cast<uint8_t>(1u << l)};
        }
        drop[m] = true;
    }
    std::vector<Inst> kept;
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!drop[i]) {
            kept.push_back(insts[i]);
        }
    }
    insts.swap(kept);
}

void Lowering::CoalesceOutputs() {
    // An epilogue move of a whole virtual pair into an output pair, with the virtual
    // pair dead afterwards and nothing reading the quad across the pair boundary, is a
    // rename: the pair is the output pair throughout, the move goes, and every write
    // lands in place (the compiler's `mad.f32 o8.xy, pa4.xy, sa36.xy, sa38.xy`).
    std::vector<Ref> refs;
    std::vector<bool> drop(insts.size(), false);
    for (std::size_t m = 0; m < insts.size(); m++) {
        const Inst& move = insts[m];
        if (drop[m] || move.op != Op::MoveInternal || move.pred != Pred::None ||
            move.dst.reg.bank != Bank::Output || move.dst.mask != 0x3 ||
            move.src[0].reg.bank != Bank::Virtual || move.src[0].swz != XYZW ||
            move.src[0].mod != Mod::None || (move.src[0].reg.n & 1) != 0 ||
            (move.dst.reg.n & 1) != 0) {
            continue;
        }
        const uint8_t p = move.src[0].reg.n; // the virtual pair
        const uint8_t o = move.dst.reg.n;    // the output pair
        const uint32_t q = QuadOf(move.src[0].reg);
        bool ok = true;
        for (std::size_t i = 0; i < insts.size() && ok; i++) {
            Inst& inst = insts[i];
            if (i == m) {
                continue;
            }
            if (inst.op == Op::Lda32 || inst.op == Op::Sample2D || inst.op == Op::MoveF32x4 ||
                inst.op == Op::PackF16) {
                // Loads and samples cannot land in outputs; four-wide moves read pairs
                // as registers we do not retarget here.
                Refs(inst, refs);
                for (const Ref& r : refs) {
                    ok &= !(r.reg->bank == Bank::Virtual && QuadOf(*r.reg) == q);
                }
                continue;
            }
            Refs(inst, refs);
            for (const Ref& r : refs) {
                if (r.reg->bank == Bank::Output && (r.reg->n & ~1) == o && i < m) {
                    ok = false; // the output pair is in use before the move
                }
                if (r.reg->bank != Bank::Virtual || QuadOf(*r.reg) != q) {
                    continue;
                }
                if (i > m && (r.reg->n & ~1) == p) {
                    ok = false; // the pair lives on after the move
                }
                if (r.reg->n != p) {
                    // The other pair, or an odd register: only a problem when a source
                    // read from the quad base reaches into this pair (checked below).
                    continue;
                }
                if (r.def) {
                    continue; // a two-wide write to the pair
                }
                // A source at the pair base: its used channels must stay in the pair.
                const Src* src = nullptr;
                for (const Src& c : inst.src) {
                    if (&c.reg == r.reg) {
                        src = &c;
                    }
                }
                if (src == nullptr) {
                    ok = false; // a Reg operand (integer op, pack): leave it
                    continue;
                }
                uint8_t mask = 0xf;
                switch (inst.op) {
                case Op::Vec:
                case Op::Vec16:
                case Op::MoveInternal:
                case Op::CondMove:
                    mask = inst.dst.mask;
                    break;
                case Op::Mad:
                    mask = inst.dst.mask & (inst.b[0] ? 0xf : 0x7);
                    break;
                case Op::Dot:
                    mask = inst.b[0] ? 0xf : 0x7;
                    break;
                case Op::Test:
                    mask = static_cast<uint8_t>(1u << (inst.u[2] & 3));
                    break;
                default:
                    break;
                }
                for (int l = 0; l < 4; l++) {
                    if ((mask >> l) & 1 && src->swz[l] <= Ch::W && src->swz[l] > Ch::Y) {
                        ok = false;
                    }
                }
            }
        }
        if (!ok) {
            continue;
        }
        for (std::size_t i = 0; i < insts.size(); i++) {
            Refs(insts[i], refs);
            for (const Ref& r : refs) {
                if (r.reg->bank == Bank::Virtual && (r.reg->n & ~1) == p) {
                    *r.reg = Reg::O(static_cast<uint8_t>(o + (r.reg->n & 1)));
                }
            }
        }
        // The move is now o <- o.
        drop[m] = true;
        coalesced_outputs.insert(o / 4);
    }
    std::vector<Inst> kept;
    for (std::size_t i = 0; i < insts.size(); i++) {
        if (!drop[i]) {
            kept.push_back(insts[i]);
        }
    }
    insts.swap(kept);
}

void Lowering::Allocate() {
    // Linear scan over the classes in order; a value that fits nowhere in a class moves
    // to the next (the spill). Each class is a set of quads.
    // A slot is a physical quad; it holds every quad whose live positions do not
    // overlap the positions already taken in it.
    const std::size_t words = (insts.size() + 63) / 64;
    struct Slot {
        Reg base;
        std::vector<uint64_t> taken;
        bool reserved = false; ///< an output quad written in place: no spills
    };
    std::vector<std::vector<Slot>> slots(classes.size());
    for (std::size_t c = 0; c < classes.size(); c++) {
        for (uint32_t r = classes[c].first; r + 4 <= classes[c].end; r += 4) {
            Slot slot{Reg{classes[c].bank, static_cast<uint8_t>(r)}, std::vector<uint64_t>(words, 0)};
            if (classes[c].bank == Bank::Output && coalesced_outputs.count(r / 4) != 0) {
                slot.reserved = true;
            }
            slots[c].push_back(slot);
        }
    }
    std::vector<uint32_t> order;
    for (uint32_t q = 0; q < quad_count; q++) {
        if (intervals[q].start != SIZE_MAX) {
            order.push_back(q);
        }
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return intervals[a].start < intervals[b].start;
    });
    for (const uint32_t q : order) {
        Interval& iv = intervals[q];
        bool placed = false;
        for (std::size_t c = 0; c < classes.size() && !placed; c++) {
            if (iv.end >= classes[c].until) {
                continue;
            }
            if (iv.temp_or_pa && classes[c].bank != Bank::Temp && classes[c].bank != Bank::PrimAttr) {
                continue;
            }
            for (Slot& s : slots[c]) {
                if (s.reserved) {
                    continue;
                }
                bool overlap = false;
                for (std::size_t w = 0; w < words && !overlap; w++) {
                    overlap = (s.taken[w] & iv.active[w]) != 0;
                }
                if (!overlap) {
                    for (std::size_t w = 0; w < words; w++) {
                        s.taken[w] |= iv.active[w];
                    }
                    iv.home = s.base;
                    placed = true;
                    if (c != 0) {
                        spilled++;
                    }
                    break;
                }
            }
        }
        if (!placed) {
            Fail("register budget: " + std::to_string(order.size()) + " quads live");
        }
        if (iv.home->bank == Bank::Temp) {
            temp_count = std::max<uint32_t>(temp_count, iv.home->n + 4);
        } else if (iv.home->bank == Bank::PrimAttr) {
            pa_count = std::max<uint32_t>(pa_count, iv.home->n + 4);
        }
    }
}

Reg Lowering::Map(Reg r) const {
    if (!IsVirtual(r)) {
        return r;
    }
    const Interval& iv = intervals[QuadOf(r)];
    if (!iv.home) {
        Fail("virtual register without a home");
    }
    return {iv.home->bank, static_cast<uint8_t>(iv.home->n + (r.n & 3))};
}

void Lowering::Encode(Encoder& e, LowerResult& result) {
    std::map<int, std::size_t> label_word;
    std::vector<std::pair<std::size_t, int>> fixups; // (branch word, label)
    const auto S = [&](const Src& s) { return Src{Map(s.reg), s.swz, s.mod}; };
    const auto D = [&](const Dst& d) { return Dst{Map(d.reg), d.mask}; };
    const auto I = [&](const IOp& o) { return o.imm ? o : IOp::R(Map(o.reg)); };
    std::vector<Ref> refs;
    for (std::size_t n = 0; n < insts.size(); n++) {
        const Inst& i = insts[n];
        switch (i.op) {
        case Op::Phas:
            e.Phas(i.b[0]);
            break;
        case Op::Nop:
            e.Nop();
            break;
        case Op::NopAfterKill:
            e.NopAfterKill();
            break;
        case Op::Wdf:
            e.Wdf(i.u[0]);
            break;
        case Op::WdfVertex:
            e.WdfVertex(i.u[0]);
            break;
        case Op::KillUnlessP1:
            e.KillUnlessP1();
            break;
        case Op::DepthF:
            e.DepthF(Map(i.reg[0]), i.u[0]);
            break;
        case Op::EndVertex:
            e.EndVertex();
            break;
        case Op::EndSecondary:
            e.EndSecondary();
            break;
        case Op::Label: {
            label_word[i.label] = e.Count();
            // psp2cgc never sets nosched on a branch: when the instruction at a join
            // starts an internal-register live range, it puts a nop at the target to
            // carry the nosched bit that guards the boundary after that instruction
            // (`#99: nop; mov.f32 i0.xyz, r8.xyz; mad.f32 i1.xyz, ..., i0.yzx`).
            bool writes_internal = false;
            if (n + 1 < insts.size()) {
                Refs(const_cast<Inst&>(insts[n + 1]), refs);
                for (const Ref& r : refs) {
                    writes_internal |= r.def && r.reg->bank == Bank::Internal;
                }
            }
            if (writes_internal) {
                e.Nop();
            }
            break;
        }
        case Op::Br:
            // A branch to the label that follows it is a fall-through.
            if (n + 1 < insts.size() && insts[n + 1].op == Op::Label &&
                insts[n + 1].label == i.label) {
                break;
            }
            fixups.emplace_back(e.Br(i.pred), i.label);
            break;
        case Op::PhaseStart:
            result.phase_starts.push_back(static_cast<uint32_t>(e.Count()));
            break;
        case Op::Vec:
            e.Vec(static_cast<VecOp>(i.u[0]), D(i.dst), S(i.src[0]), S(i.src[1]), i.pred);
            break;
        case Op::Vec16:
            e.Vec16(static_cast<VecOp>(i.u[0]), D(i.dst), S(i.src[0]), S(i.src[1]), i.pred);
            break;
        case Op::Mad:
            e.Mad(D(i.dst), S(i.src[0]), S(i.src[1]), S(i.src[2]), i.b[0], i.pred);
            break;
        case Op::Dot:
            e.Dot(D(i.dst), S(i.src[0]), S(i.src[1]), i.b[0], i.pred);
            break;
        case Op::Mad2:
            e.Mad2(D(i.dst), S(i.src[0]), S(i.src[1]), S(i.src[2]), i.pred);
            break;
        case Op::Comp:
            e.Comp(static_cast<CompOp>(i.u[0]), D(i.dst), S(i.src[0]), i.u[1], i.pred);
            break;
        case Op::MoveF32x4:
            e.MoveF32x4(D(i.dst), Map(i.reg[0]), Map(i.reg[1]));
            break;
        case Op::PackF16:
            e.PackF16(D(i.dst), Map(i.reg[0]), Map(i.reg[1]));
            break;
        case Op::MoveInternal:
            e.MoveInternal(D(i.dst), S(i.src[0]), i.pred);
            break;
        case Op::CondMove:
            e.CondMove(D(i.dst), S(i.src[0]), S(i.src[1]), S(i.src[2]));
            break;
        case Op::Test:
            e.Test(static_cast<Cond>(i.u[0]), i.u[1], S(i.src[0]), i.u[2], S(i.src[1]), i.b[0],
                   i.pred);
            break;
        case Op::Sample2D: {
            const Reg lod = Map(i.src[0].reg);
            e.Sample2D(Map(i.reg[0]), Map(i.reg[1]), i.u[0], i.b[0] ? &lod : nullptr, i.u[1]);
            break;
        }
        case Op::Lda32: {
            const Reg off = Map(i.src[0].reg);
            e.Lda32(Map(i.reg[0]), Map(i.reg[1]), i.b[0] ? &off : nullptr, i.u[0], i.u[1], i.u[2]);
            break;
        }
        case Op::Bitwise:
            e.Bitwise(static_cast<BwOp>(i.u[0]), Map(i.reg[0]), I(i.iop[0]), I(i.iop[1]), i.pred);
            break;
        case Op::MadI16:
            e.MadI16(Map(i.reg[0]), Map(i.reg[1]), Map(i.src[0].reg), I(i.iop[0]));
            break;
        case Op::MadI32:
            e.MadI32(Map(i.reg[0]), Map(i.reg[1]), i.u[0], Map(i.src[0].reg));
            break;
        case Op::PackS16:
            e.PackS16(Map(i.reg[0]), Map(i.reg[1]), i.u[0], i.pred);
            break;
        }
    }
    for (const auto& [word, label] : fixups) {
        const auto it = label_word.find(label);
        if (it == label_word.end()) {
            Fail("branch to an unbound label");
        }
        e.PatchBr(word, it->second);
    }
}

LowerResult Lowering::Run(Encoder& e, int level) {
    std::vector<Ref> refs;
    for (Inst& i : insts) {
        Refs(i, refs);
        for (const Ref& r : refs) {
            if (IsVirtual(*r.reg)) {
                quad_count = std::max(quad_count, QuadOf(*r.reg) + 1);
            }
        }
    }
    if (quad_count != 0 && classes.empty()) {
        Fail("no register class");
    }
    // The first tier only places the registers (spilling as needed): microseconds, and
    // the program the draw waits for. The second tier, off the render thread for programs
    // that draw a lot, runs the optimising passes first. Hoisting loads lengthens live
    // ranges; when the registers then do not fit, the program is lowered again without
    // it.
    const std::vector<Inst> original = insts;
    const uint32_t original_quads = quad_count;
    for (int attempt = 0; attempt < 3; attempt++) {
        const int hoist = 3 - attempt; // groups of three, then two, then none
        if (level >= 2) {
            Liveness();
            DeadStores();
            static const bool no_hoist = GxmFlag("usse_nohoist");
            Dump("input");
            if (hoist >= 2 && !no_hoist) {
                static const bool no_global = GxmFlag("usse_noglobalhoist");
                if (hoist == 3 && !no_global) {
                    HoistLoadsGlobal();
                    Dump("hoist global");
                }
                HoistLoads(hoist);
                Dump("hoist");
            }
            CoalesceOutputs();
            Dump("coalesce");
            Reloads();
            Dump("reloads");
            TwoWideChains();
            Dump("chains");
            DirectTwoWide();
            Dump("two-wide");
            LoadSources();
            Dump("load sources");
            ResultsIntoInternal();
            Dump("results into internal");
            InternalRegisters();
            Dump("internal registers");
            // The chains that pass forms (a mad reading the previous mad's register) end
            // in a mad plus its moves out: the two-wide pass again for those tails.
            DirectTwoWide();
            Dump("two-wide again");
            InternalDeadWrites();
            Liveness();
            DeadStores();
            Dump("final");
        }
        Liveness();
        try {
            Allocate();
        } catch (const std::exception&) {
            if (level >= 2 && hoist >= 2) {
                insts = original;
                quad_count = original_quads;
                intervals.clear();
                live.clear();
                coalesced_outputs.clear();
                spilled = temp_count = pa_count = 0;
                continue;
            }
            throw;
        }
        break;
    }
    CheckInternals();
    LowerResult result;
    result.temp_count = temp_count;
    result.pa_count = pa_count;
    result.spilled = spilled;
    Encode(e, result);
    return result;
}

} // Anonymous namespace

LowerResult Builder::Lower(Encoder& e, int level) {
    e.skip_invalid = skip_invalid;
    e.nosched_override = nosched_override;
    e.secondary = secondary;
    std::vector<RegClass> cls = classes;
    if (cls.empty()) {
        cls.push_back({Bank::Temp, 0, 64});
    }
    Lowering lowering{insts, cls};
    return lowering.Run(e, level);
}

} // namespace GxmRenderer::Usse::Ir
