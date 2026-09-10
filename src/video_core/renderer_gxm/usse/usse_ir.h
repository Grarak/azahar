// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "video_core/renderer_gxm/usse/usse_encoder.h"

namespace GxmRenderer::Usse::Ir {

/**
 * The instruction list between the emitters and the encoder. An emitter writes it through
 * the same builder calls the Encoder offers, but may name registers in the virtual bank:
 * `Reg::V(n)` is register n of a virtual unified store laid out in quads (four registers,
 * pair-aligned), so a float4 is V(4q) .. V(4q + 3) and the emitters' base + 2 arithmetic
 * for the second pair holds.
 *
 * Level 1, the first tier (the program a draw waits for, microseconds): liveness by
 * dataflow over the labels and branches (a loop keeps a value live to its back edge), then
 * linear scan over the register classes the emitter offered, in order: the temporaries,
 * then the primary attributes past the inputs, then (vertex programs) the output registers
 * the epilogue writes last. Running out of one class spills the longest-lived value to the
 * next; running out of all refuses the program.
 *
 * Level 2, the second tier (gxm_shader_tier: off the render thread, for programs that
 * draw a lot), first runs the optimising passes:
 *  - dead stores: an instruction whose only effect is a write to a virtual quad nothing
 *    reads afterwards is dropped;
 *  - internal-register reloads: a four-wide load of the same operand into the same
 *    internal register, with neither changed in between, is dropped;
 *  - direct two-wide ops: a four-wide op into an internal register whose result is only
 *    moved out to unified-store pairs becomes one two-wide op per pair, when each pair's
 *    two lanes come from one source pair (the compiler's own idiom, `mul.f32 o0.xy,
 *    pa0.xy, sa12.xy`); the loads that fed only it die with it;
 *  - load sources: a load into an internal register whose readers are all two-wide ops,
 *    compares or pack reads gets folded into them (they read the source's pair or lane
 *    directly; commutative ops swap operands for a negated source, compares swap with
 *    the condition flipped) and the load goes;
 *  - internal registers placed again: within a block each full write starts a range,
 *    the ranges go over i0-i2 keeping loaded operands resident (the one loaded again
 *    soonest is spent last), and a load of an operand a register still holds, unchanged
 *    since, is dropped;
 *  - internal-register dead writes: a write to i0-i2 nothing reads before it is
 *    overwritten, or before the next label or branch, is dropped.
 *
 * The hardware's other registers (primary attributes the fetch fills, secondary
 * attributes, the internal registers, the constants, the outputs) are named physically and
 * pass through unchanged.
 */

/// Rows of the vertex uniform buffer repeated past entry 127, so a read at an immediate
/// offset from a neighbouring row's address lands on the row the wrapping index names.
constexpr uint32_t UniformMirrorRows = 4;

enum class Op : uint8_t {
    Phas,
    Nop,
    NopAfterKill,
    Wdf,
    WdfVertex,
    KillUnlessP1,
    DepthF,
    EndVertex,
    EndSecondary,
    Label,
    Br,
    PhaseStart,
    Vec,
    Vec16,
    Mad,
    Dot,
    Comp,
    MoveF32x4,
    PackF16,
    MoveInternal,
    CondMove,
    Test,
    Sample2D,
    Lda32,
    Bitwise,
    MadI16,
    MadI32,
    PackS16,
    Mad2,
};

struct Inst {
    Op op;
    Dst dst{};
    Src src[3]{};
    Reg reg[2]{};
    IOp iop[2]{};
    Pred pred = Pred::None;
    uint8_t u[3]{};
    bool b[2]{};
    int label = -1;
};

/// A range of physical registers values may live in, offered in spill order.
struct RegClass {
    Bank bank;
    uint8_t first; ///< first register (pair-aligned)
    uint8_t end;   ///< one past the last
    /// Only values dead before the instruction at this index may use the class (the
    /// vertex epilogue writes the outputs); SIZE_MAX for no limit.
    std::size_t until = SIZE_MAX;
};

struct LowerResult {
    uint32_t temp_count = 0; ///< temporaries used, r0 up
    uint32_t pa_count = 0;   ///< primary attributes used
    uint32_t spilled = 0;    ///< quads that left the first class
    std::vector<uint32_t> phase_starts{0};
};

class Builder {
public:
    bool skip_invalid = true;
    int nosched_override = 0;
    bool secondary = false;

    std::size_t Count() const {
        return insts.size();
    }
    const std::vector<Inst>& Insts() const {
        return insts;
    }

    // --- control flow ---
    int NewLabel();
    void Bind(int label);
    void Br(Pred pred, int label);
    /// The next instruction starts a new phase (the fragment programs' second phase).
    void PhaseStart();

    // --- the Encoder's builders ---
    void Phas(bool wait_all);
    void Nop();
    void NopAfterKill();
    void Wdf(uint8_t drc);
    void WdfVertex(uint8_t drc);
    void KillUnlessP1();
    void DepthF(Reg src, uint8_t control_sa);
    void EndVertex();
    void EndSecondary();
    void Vec(VecOp op, Dst dst, Src src1, Src src2, Pred pred = Pred::None, bool nosched = false);
    void Vec16(VecOp op, Dst dst, Src src1, Src src2, Pred pred = Pred::None, bool nosched = false);
    void Mad(Dst dst, Src src1, Src gpi0, Src gpi1, bool vec4, Pred pred = Pred::None,
             bool nosched = false);
    void Dot(Dst dst, Src src1, Src gpi0, bool vec4, Pred pred = Pred::None, bool nosched = false);
    void Mad2(Dst dst, Src src0, Src src1, Src src2, Pred pred = Pred::None);
    void Comp(CompOp op, Dst dst, Src src, uint8_t src_comp, Pred pred = Pred::None,
              bool nosched = false);
    void MoveF32x4(Dst dst, Reg lo, Reg hi, bool nosched = false);
    void PackF16(Dst dst, Reg lo, Reg hi, bool nosched = false);
    void MoveInternal(Dst dst, Src src, Pred pred = Pred::None, bool nosched = false);
    void CondMove(Dst dst, Src src0, Src src1, Src src2, bool nosched = false);
    void Test(Cond cond, uint8_t pred_n, Src src1, uint8_t chan, Src src2, bool src2_scalar,
              Pred pred = Pred::None);
    void Sample2D(Reg dst, Reg coord, uint8_t sampler_sa, const Reg* lod, uint8_t drc,
                  bool nosched = false);
    void Lda32(Reg dst, Reg base_sa, const Reg* offset_reg, uint8_t offset_words, uint8_t count,
               uint8_t drc);
    void Bitwise(BwOp op, Reg dst, IOp src1, IOp src2, Pred pred = Pred::None);
    void MadI16(Reg dst, Reg src0, Reg src1, IOp src2);
    void MadI32(Reg dst, Reg src0, uint8_t imm, Reg src2);
    void PackS16(Reg dst, Reg src, uint8_t comp, Pred pred = Pred::None);

    // --- lowering ---
    /// Where virtual quads may live, in spill order. Without a call, the temporaries.
    void AddClass(RegClass cls);
    /// Runs the passes (level 2 adds the optimising ones), allocates, and encodes into `e`
    /// (the caller then patches kills and places the nosched bits). Throws
    /// std::runtime_error when the values do not fit.
    LowerResult Lower(Encoder& e, int level = 1);

private:
    Inst& Push(Op op);
    std::vector<Inst> insts;
    std::vector<RegClass> classes;
    int label_count = 0;
};

} // namespace GxmRenderer::Usse::Ir
