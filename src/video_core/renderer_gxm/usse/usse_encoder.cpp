// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <cstdio>
#include <optional>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include "video_core/renderer_gxm/usse/usse_encoder.h"

namespace GxmRenderer::Usse {

namespace {

// Vita3K's decoder patterns, verbatim: 64 characters, MSB first; a run of one letter is
// one field, '0'/'1' are fixed bits, '-' is don't care (written as 0).
constexpr const char* PatV32Nmad = "00001pppsrrydcbawwwwneeeemmoiittkkllffffffzzzzzzzggghhhhhhjjjjjj";
constexpr const char* PatV16Nmad = "00010pppsrrydcbawwwwneeeemmoiittkkllffffffzzzzzzzggghhhhhhjjjjjj";
constexpr const char* PatVmad2 = "00000dpps-ry-cbawwwineeeemmookttffgghhhhhhzzjjllllllqqqqqquuuuuu";
constexpr const char* PatVmad = "00011pppsg1oderaaittnwwwwcbfhzkkjjllmmmmmmqqqquuuuvvxyAAAABBBBBB";
constexpr const char* PatVdp = "00011pppsc0oderaagttnwwwwbflllkkhhiijjjjjjzzzzmmmqqqyyyxxxuuuuuu";
constexpr const char* PatVcomp = "00110pppsddyenr-aaaaobbccmmff-ttkk--ggggggg-------hhhhhhh---wwww";
constexpr const char* PatVmov = "00111pppstrydecbmmaanoooiwwwwkllffgghhhhjjjjjjqqqqqquuuuuuvvvvvv";
constexpr const char* PatVpck = "01000pppsnuydercaaaaffftttmmmmbbkkllgggggggoohiijjqqqqqqvwwwwwwx";
// Bit 54 ('-' in Vita3K's pattern) is the compare's nosched bit, where VPCK and SMP have
// theirs; exposed as 'K'. (It looked like "set on every compare except the one feeding a
// kill" until the liveness rule explained it.)
constexpr const char* PatVtst = "01001pppsKoydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
constexpr const char* PatSmp = "11100pppsn-ymrceffaaddlltbbggkhhiijjoooooooqqqqqqquuuuuuuvvvvvvv";
constexpr const char* PatVldst = "111oopppsnmycrbakkkkddeetgffihjlqquuvvvvvvvwwwwwwwxxxxxxxzzzzzzz";
constexpr const char* PatVbw = "01ooopppsnrydecxaaaaittttthhbwkkffggjjjjjjjlllllllmmmmmmmqqqqqqq";
constexpr const char* PatI16Mad = "10100ppasnredbck-tttmmffoolhhgiijjqquuuuuuuvvvvvvvwwwwwwwxxxxxxx";
constexpr const char* PatI32Mad = "10101pps-nrcdeba0tttif00yy000kgghhjjlllllllmmmmmmmoooooooqqqqqqq";
constexpr const char* PatDepthf = "11111011s-11recb----npp---tffa--kkddggggggghhhhhhhiiiiiiijjjjjjj";
constexpr const char* PatBr = "11111ppps000e-----wynba00r----------------iloooooooooooooooooooo";

struct Field {
    char letter;
    uint64_t value;
};

[[noreturn]] void Fail(const char* what) {
    throw std::logic_error(what);
}

uint64_t Pack(const char* pattern, std::initializer_list<Field> fields) {
    uint64_t word = 0;
    int i = 0;
    while (i < 64) {
        const char c = pattern[i];
        int j = i;
        while (j < 64 && pattern[j] == c) {
            j++;
        }
        const int width = j - i;
        const int shift = 64 - j;
        uint64_t value = 0;
        if (c == '1') {
            value = (1ull << width) - 1;
        } else if (c != '0' && c != '-') {
            bool found = false;
            for (const Field& f : fields) {
                if (f.letter == c) {
                    value = f.value;
                    found = true;
                    break;
                }
            }
            if (!found) {
                Fail("usse: pattern field not supplied");
            }
            if (width < 64 && value >> width) {
                const std::string what = std::string("usse: field value does not fit: '") + c +
                                         "' = " + std::to_string(value) + " in " +
                                         std::to_string(width) + " bits";
                Fail(what.c_str());
            }
        }
        word |= value << shift;
        i = j;
    }
    return word;
}

// --- operand encodings ---

// Destination bank: (bank, ext).
std::pair<uint64_t, uint64_t> DestBank(Bank bank) {
    switch (bank) {
    case Bank::Temp:
    case Bank::Internal:
        return {0, 0};
    case Bank::Output:
        return {1, 0};
    case Bank::PrimAttr:
        return {2, 0};
    case Bank::SecAttr:
        return {0, 1};
    case Bank::Const:
        return {1, 1};
    }
    Fail("usse: bad dest bank");
}

// Source 1/2 bank: (bank, ext).
std::pair<uint64_t, uint64_t> Src12Bank(Bank bank) {
    switch (bank) {
    case Bank::Temp:
    case Bank::Internal:
        return {0, 0};
    case Bank::Output:
        return {1, 0};
    case Bank::PrimAttr:
        return {2, 0};
    case Bank::SecAttr:
        return {3, 0};
    case Bank::Const:
        return {1, 1};
    }
    Fail("usse: bad source bank");
}

// Source 0 bank (the three-source forms): (bank, ext).
std::pair<uint64_t, uint64_t> Src0Bank(Bank bank) {
    switch (bank) {
    case Bank::Temp:
    case Bank::Internal:
        return {0, 0};
    case Bank::PrimAttr:
        return {1, 0};
    case Bank::Output:
        return {0, 1};
    case Bank::SecAttr:
        return {1, 1};
    default:
        break;
    }
    Fail("usse: bad source-0 bank");
}

// A register number in a field that addresses 64-bit pairs (`is_double_regs` in the
// decoder): the index halved, and an internal register as the top of the temp bank.
// `bits` is the decoder's reg_bits: 7 for a 6-bit field, 8 for a 7-bit one.
uint64_t DoubledNum(Reg reg, int bits) {
    if (reg.bank == Bank::Internal) {
        const int limit = (1 << bits) - 8;
        return static_cast<uint64_t>((limit + reg.n * 2) >> 1);
    }
    if (reg.bank == Bank::Const) {
        return reg.n; // constant indices are not doubled
    }
    if (reg.n & 1) {
        static char message[64];
        std::snprintf(message, sizeof(message), "usse: odd register %d of bank %d cannot address a pair",
                      reg.n, static_cast<int>(reg.bank));
        Fail(message);
    }
    return reg.n >> 1;
}

// A register number in a field that addresses single 32-bit registers.
uint64_t SingleNum(Reg reg, int bits) {
    if (reg.bank == Bank::Internal) {
        return static_cast<uint64_t>((1 << bits) - 4 + reg.n);
    }
    return reg.n;
}

uint64_t ExtVecPred(Pred p) {
    switch (p) {
    case Pred::None:
        return 0;
    case Pred::P0:
        return 1;
    case Pred::P1:
        return 2;
    case Pred::P2:
        return 3;
    case Pred::NotP0:
        return 4;
    case Pred::NotP1:
        return 5;
    }
    return 0;
}

uint64_t ExtPred(Pred p) {
    switch (p) {
    case Pred::None:
        return 0;
    case Pred::P0:
        return 1;
    case Pred::P1:
        return 2;
    case Pred::P2:
        return 3;
    case Pred::NotP0:
        return 5;
    case Pred::NotP1:
        return 6;
    }
    return 0;
}

// The 12-bit swizzle of V32NMAD's first source: three bits per channel.
uint64_t Swizzle12(const Swizzle& s) {
    return static_cast<uint64_t>(s[0]) | static_cast<uint64_t>(s[1]) << 3 |
           static_cast<uint64_t>(s[2]) << 6 | static_cast<uint64_t>(s[3]) << 9;
}

#define SW(a, b, c, d) Swizzle{Ch::a, Ch::b, Ch::c, Ch::d}
constexpr Swizzle Vec4Std[16] = {
    SW(X, X, X, X), SW(Y, Y, Y, Y), SW(Z, Z, Z, Z), SW(W, W, W, W),
    SW(X, Y, Z, W), SW(Y, Z, W, W), SW(X, Y, Z, Z), SW(X, X, Y, Z),
    SW(X, Y, X, Y), SW(X, Y, W, Z), SW(Z, X, Y, W), SW(Z, W, Z, W),
    SW(Y, Z, X, Z), SW(X, X, Y, Y), SW(X, Z, W, W), SW(X, Y, Z, One),
};
constexpr Swizzle Vec4Ext[16] = {
    SW(Y, Z, X, W), SW(Z, W, X, Y), SW(X, Z, W, Y), SW(Y, Y, W, W),
    SW(W, Y, Z, W), SW(W, Z, W, Z), SW(X, Y, Z, X), SW(Z, Z, W, W),
    SW(X, W, Z, X), SW(Y, Y, Y, X), SW(Y, Y, Y, Z), SW(X, Z, Y, W),
    SW(X, X, X, Y), SW(Z, Y, X, W), SW(Y, Y, Z, Z), SW(Z, Z, Z, Y),
};
// Three-wide tables: the fourth channel is unused.
constexpr Swizzle Vec3Std[16] = {
    SW(X, X, X, X), SW(Y, Y, Y, X), SW(Z, Z, Z, X), SW(W, W, W, X),
    SW(X, Y, Z, X), SW(Y, Z, W, X), SW(X, X, Y, X), SW(X, Y, X, X),
    SW(Y, Y, X, X), SW(Y, Y, Z, X), SW(Z, X, Y, X), SW(X, Z, Y, X),
    SW(Y, Z, X, X), SW(Z, Y, X, X), SW(Z, Z, Y, X), SW(X, Y, One, X),
};
constexpr Swizzle Vec3Ext[11] = {
    SW(X, Y, Y, X),          SW(Y, X, Y, X),          SW(X, X, Z, X),
    SW(Y, X, X, X),          SW(X, Y, Zero, X),       SW(X, One, Zero, X),
    SW(Zero, Zero, Zero, X), SW(One, One, One, X),    SW(Half, Half, Half, X),
    SW(Two, Two, Two, X),    SW(X, Zero, Zero, X),
};
#undef SW

bool SameChannels(const Swizzle& a, const Swizzle& b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// A swizzle as (table index, extended) in the four-wide tables.
std::pair<uint64_t, uint64_t> Vec4Swizzle(const Swizzle& s) {
    for (int i = 0; i < 16; i++) {
        if (SameChannels(Vec4Std[i], s, 4)) {
            return {static_cast<uint64_t>(i), 0};
        }
    }
    for (int i = 0; i < 16; i++) {
        if (SameChannels(Vec4Ext[i], s, 4)) {
            return {static_cast<uint64_t>(i), 1};
        }
    }
    static const char* names = "xyzw012h";
    static char message[64];
    std::snprintf(message, sizeof(message), "usse: swizzle %c%c%c%c not in the four-wide tables",
                  names[static_cast<int>(s[0])], names[static_cast<int>(s[1])],
                  names[static_cast<int>(s[2])], names[static_cast<int>(s[3])]);
    Fail(message);
}

// The same in the three-wide tables (the fourth channel ignored).
std::pair<uint64_t, uint64_t> Vec3Swizzle(const Swizzle& s) {
    for (int i = 0; i < 16; i++) {
        if (SameChannels(Vec3Std[i], s, 3)) {
            return {static_cast<uint64_t>(i), 0};
        }
    }
    for (int i = 0; i < 11; i++) {
        if (SameChannels(Vec3Ext[i], s, 3)) {
            return {static_cast<uint64_t>(i), 1};
        }
    }
    Fail("usse: swizzle not in the three-wide tables");
}

} // Anonymous namespace

std::optional<Swizzle> Vec4SwizzleForPair(Ch c0, Ch c1) {
    for (const Swizzle* table : {Vec4Std, Vec4Ext}) {
        for (int i = 0; i < 16; i++) {
            if (table[i][0] == c0 && table[i][1] == c1) {
                return table[i];
            }
        }
    }
    return std::nullopt;
}

bool InVec4StdTable(const Swizzle& s) {
    for (const Swizzle& t : Vec4Std) {
        if (SameChannels(t, s, 4)) {
            return true;
        }
    }
    return false;
}

bool InVec4Tables(const Swizzle& s) {
    if (InVec4StdTable(s)) {
        return true;
    }
    for (const Swizzle& t : Vec4Ext) {
        if (SameChannels(t, s, 4)) {
            return true;
        }
    }
    return false;
}

bool InVec3Tables(const Swizzle& s) {
    for (int i = 0; i < 16; i++) {
        if (SameChannels(Vec3Std[i], s, 3)) {
            return true;
        }
    }
    for (int i = 0; i < 11; i++) {
        if (SameChannels(Vec3Ext[i], s, 3)) {
            return true;
        }
    }
    return false;
}

namespace {

// V32NMAD's second source only has the standard four-wide table.
uint64_t Vec4StdSwizzle(const Swizzle& s) {
    const auto [index, ext] = Vec4Swizzle(s);
    if (ext) {
        Fail("usse: swizzle needs the extended table where only the standard one exists");
    }
    return index;
}

} // Anonymous namespace

bool StdSwizzleForPair(Ch c0, Ch c1, Swizzle& out) {
    for (const Swizzle& s : Vec4Std) {
        if (s[0] == c0 && s[1] == c1) {
            out = s;
            return true;
        }
    }
    return false;
}

Reg Encoder::Map(Reg r) const {
    if (secondary && r.bank == Bank::SecAttr) {
        return {Bank::PrimAttr, r.n};
    }
    return r;
}

void Encoder::Push(uint64_t word, uint8_t nosched_bit, uint16_t reads, uint16_t writes) {
    words.push_back(word);
    meta.push_back({reads, writes, nosched_bit});
}

uint16_t Encoder::Reads(const Src& src, uint8_t lanes) {
    // Lane j of the result reads src.swz[j]; only the lanes written count, since a read
    // of a component nothing ever wrote would keep it live back to the program's start.
    if (src.reg.bank != Bank::Internal) {
        return 0;
    }
    uint16_t mask = 0;
    for (unsigned j = 0; j < 4; j++) {
        const Ch ch = src.swz[j];
        if ((lanes & (1u << j)) && ch <= Ch::W) {
            mask |= static_cast<uint16_t>(1u << (src.reg.n * 4 + static_cast<unsigned>(ch)));
        }
    }
    return mask;
}

uint16_t Encoder::ReadsOne(Reg reg, uint8_t comp) {
    return reg.bank == Bank::Internal ? static_cast<uint16_t>(1u << (reg.n * 4 + (comp & 3))) : 0;
}

uint16_t Encoder::ReadsAll(Reg reg) {
    return reg.bank == Bank::Internal ? static_cast<uint16_t>(0xfu << (reg.n * 4)) : 0;
}

uint16_t Encoder::Writes(const Dst& dst, Pred pred) {
    // A predicated write may not happen, so it ends no live range.
    if (dst.reg.bank != Bank::Internal || pred != Pred::None) {
        return 0;
    }
    return static_cast<uint16_t>((dst.mask & 0xf) << (dst.reg.n * 4));
}

void Encoder::FinishNosched() {
    const std::size_t n = words.size();
    // after[i]: the components live once word i has executed.
    std::vector<uint16_t> after(n, 0);
    uint16_t live = 0;
    for (std::size_t i = n; i-- > 0;) {
        after[i] = live;
        live = static_cast<uint16_t>((live & ~meta[i].writes) | meta[i].reads);
    }
    for (std::size_t i = 0; i < n; i++) {
        if (meta[i].bit == 0) {
            continue;
        }
        bool set = false;
        if (nosched_override == 2) {
            set = true;
        } else if (nosched_override == 0) {
            set = i + 1 < n && after[i + 1] != 0;
        }
        const uint64_t bit = 1ull << meta[i].bit;
        words[i] = set ? (words[i] | bit) : (words[i] & ~bit);
    }
}

void Encoder::EndVertex() {
    Push(0xfb275000a0200000ull, 0, 0, 0);
}

void Encoder::EndSecondary() {
    Push(0xf804014000000000ull, 0, 0, 0);
}

void Encoder::WdfVertex(uint8_t drc) {
    // Bit 43 is the wait's nosched bit (psp2cgc's `f9200800...` form before an internal
    // register write, 2026-09-06): without it the boundary after the instruction that
    // follows the wait is unguarded, and the console corrupted every dynamic uniform read.
    Push(drc ? 0xf920000100000000ull : 0xf920000000000000ull, 43, 0, 0);
}

std::size_t Encoder::Br(Pred pred) {
    const std::size_t index = words.size();
    // Letters as Vita3K's pattern names them: s syncend, e exception, w pwait, y sync_ext,
    // n nosched, b br_monitor, a save_link, r br_type, i any_inst, l all_inst.
    Push(Pack(PatBr, {
        {'p', ExtPred(pred)}, {'s', 0}, {'e', 0}, {'w', 0}, {'y', 0}, {'n', 0}, {'b', 0},
        {'a', 0}, {'r', 1}, {'i', 0}, {'l', 0}, {'o', 0},
    }), 43, 0, 0);
    return index;
}

void Encoder::PatchBr(std::size_t index, std::size_t target) {
    const int64_t off = static_cast<int64_t>(target) - static_cast<int64_t>(index);
    if (off < -(1 << 19) || off >= (1 << 19)) {
        Fail("usse: branch offset out of range");
    }
    words[index] = (words[index] & ~0xfffffull) | (static_cast<uint64_t>(off) & 0xfffff);
}

namespace {
// Integer operand banks: single 32-bit registers, 7-bit fields; an immediate is bank 2
// with the extension bit.
std::pair<uint64_t, uint64_t> IntSrc12Bank(Bank bank) {
    switch (bank) {
    case Bank::Temp:
    case Bank::Internal:
        return {0, 0};
    case Bank::Output:
        return {1, 0};
    case Bank::PrimAttr:
        return {2, 0};
    case Bank::SecAttr:
        return {3, 0};
    default:
        break;
    }
    Fail("usse: bad integer source bank");
}
} // Anonymous namespace

void Encoder::Lda32(Reg dst, Reg base_sa, const Reg* offset_reg, uint8_t offset_words,
                    uint8_t count, uint8_t drc) {
    dst = Map(dst);
    base_sa = Map(base_sa);
    if (count != 4 && count != 16) {
        Fail("usse: a load fetches 4 or 16 words");
    }
    const auto [bb, bext] = Src0Bank(base_sa.bank);
    uint64_t dest_pa = 0;
    switch (dst.bank) {
    case Bank::Temp:
        break;
    case Bank::PrimAttr:
        dest_pa = 1;
        break;
    default:
        Fail("usse: a load lands in a temporary, a primary attribute or (secondary program) a secondary attribute");
    }
    uint64_t s1b = 2, s1ext = 1, s1n = offset_words;
    if (offset_reg != nullptr) {
        const Reg r = Map(*offset_reg);
        const auto [b, e] = IntSrc12Bank(r.bank);
        s1b = b;
        s1ext = e;
        s1n = SingleNum(r, 7);
    }
    Push(Pack(PatVldst, {
        {'o', 1}, {'p', 0}, {'s', skip_invalid ? 1u : 0u}, {'n', 0}, {'m', 0}, {'y', 0}, {'c', 0},
        {'r', bext}, {'b', s1ext}, {'a', 1}, {'k', static_cast<uint64_t>(count - 1)}, {'d', 0},
        {'e', 0}, {'t', dest_pa}, {'g', 0}, {'f', 0}, {'i', 0}, {'h', bb}, {'j', 0}, {'l', drc},
        {'q', s1b}, {'u', 2}, {'v', SingleNum(dst, 7)}, {'w', SingleNum(base_sa, 7)},
        {'x', s1n}, {'z', 0},
    }), 54, 0, 0);
}

void Encoder::Bitwise(BwOp op, Reg dst, IOp src1, IOp src2, Pred pred) {
    dst = Map(dst);
    const auto [db, dext] = DestBank(dst.bank);
    const auto operand = [this](const IOp& o) -> std::array<uint64_t, 3> {
        if (o.imm) {
            if (o.value > 127) {
                Fail("usse: bitwise immediate is 7 bits");
            }
            return {2, 1, o.value};
        }
        const Reg r = Map(o.reg);
        const auto [b, e] = IntSrc12Bank(r.bank);
        return {b, e, SingleNum(r, 7)};
    };
    const auto s1 = operand(src1);
    const auto s2 = operand(src2);
    uint64_t op1 = 2, op2 = 0;
    switch (op) {
    case BwOp::And: op1 = 2; op2 = 0; break;
    case BwOp::Or: op1 = 2; op2 = 1; break;
    case BwOp::Xor: op1 = 3; op2 = 0; break;
    }
    Push(Pack(PatVbw, {
        {'o', op1}, {'p', ExtPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'n', 0}, {'r', 0}, {'y', 0},
        {'d', dext}, {'e', 0}, {'c', s1[1]}, {'x', s2[1]}, {'a', 0}, {'i', 0}, {'t', 0}, {'h', 0},
        {'b', op2}, {'w', 0}, {'k', db}, {'f', s1[0]}, {'g', s2[0]}, {'j', SingleNum(dst, 7)},
        {'l', 0}, {'m', s1[2]}, {'q', s2[2]},
    }), 54, 0, 0);
}

void Encoder::MadI16(Reg dst, Reg src0, Reg src1, IOp src2) {
    dst = Map(dst);
    src0 = Map(src0);
    src1 = Map(src1);
    const auto [db, dext] = DestBank(dst.bank);
    uint64_t s0b = 0;
    switch (src0.bank) {
    case Bank::Temp: s0b = 0; break;
    case Bank::PrimAttr: s0b = 1; break;
    default: Fail("usse: mad.i16's first source is a temporary or a primary attribute");
    }
    const auto [s1b, s1ext] = IntSrc12Bank(src1.bank);
    uint64_t s2b = 2, s2ext = 1, s2n = src2.value, s2fmt = 1;
    if (!src2.imm) {
        const Reg r = Map(src2.reg);
        const auto [b, e] = IntSrc12Bank(r.bank);
        s2b = b;
        s2ext = e;
        s2n = SingleNum(r, 7);
        s2fmt = 0;
    }
    Push(Pack(PatI16Mad, {
        {'p', 0}, {'a', 0}, {'s', skip_invalid ? 1u : 0u}, {'n', 0}, {'r', 0}, {'e', 0}, {'d', dext},
        {'b', 0}, {'c', s1ext}, {'k', s2ext}, {'t', 0}, {'m', 2}, {'f', s2fmt}, {'o', 0}, {'l', 0},
        {'h', 0}, {'g', s0b}, {'i', db}, {'j', s1b}, {'q', s2b}, {'u', SingleNum(dst, 7)},
        {'v', SingleNum(src0, 7)}, {'w', SingleNum(src1, 7)}, {'x', s2n},
    }), 54, 0, 0);
}

void Encoder::MadI32(Reg dst, Reg src0, uint8_t imm, Reg src2) {
    dst = Map(dst);
    src0 = Map(src0);
    src2 = Map(src2);
    const auto [db, dext] = DestBank(dst.bank);
    uint64_t s0b = 0;
    switch (src0.bank) {
    case Bank::Temp:
    case Bank::Internal: s0b = 0; break;
    case Bank::PrimAttr: s0b = 1; break;
    default: Fail("usse: mad.i32's first source is a temporary or a primary attribute");
    }
    if (imm > 127) {
        Fail("usse: mad.i32 immediate is 7 bits");
    }
    const auto [s2b, s2ext] = IntSrc12Bank(src2.bank);
    // The compiler's words have bit 55 set and bit 56 clear, where Vita3K's pattern puts
    // skipinv at 56 and a don't-care at 55; written as the compiler writes them.
    Push(Pack(PatI32Mad, {
        {'p', 0}, {'s', 0}, {'n', 0}, {'r', 0}, {'c', 0}, {'d', 0}, {'e', dext},
        {'b', 1}, {'a', s2ext}, {'t', 0}, {'i', 1}, {'f', 0}, {'y', 2}, {'k', s0b}, {'g', db},
        {'h', 2}, {'j', s2b}, {'l', SingleNum(dst, 7)}, {'m', SingleNum(src0, 7)}, {'o', imm},
        {'q', SingleNum(src2, 7)},
    }) | (1ull << 55), 54, 0, 0);
}

void Encoder::PackS16(Reg dst, Reg src, uint8_t comp, Pred pred) {
    dst = Map(dst);
    src = Map(src);
    const auto [db, dext] = DestBank(dst.bank);
    const auto [sb, sext] = Src12Bank(src.bank);
    Push(Pack(PatVpck, {
        {'p', ExtPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'n', 0}, {'u', 0}, {'y', 0}, {'d', dext},
        {'e', 0}, {'r', sext}, {'c', 1}, {'a', 0}, {'f', 6}, {'t', 4}, {'m', 1}, {'b', db}, {'k', sb},
        {'l', 2}, {'g', SingleNum(dst, 7)}, {'o', 0}, {'h', 0}, {'i', 0}, {'j', 0},
        {'q', DoubledNum(src, 7)}, {'v', static_cast<uint64_t>((comp >> 1) & 1)}, {'w', 0},
        {'x', static_cast<uint64_t>(comp & 1)},
    }), 54, ReadsOne(src, comp), 0);
}

void Encoder::Phas(bool wait_all) {
    Push(wait_all ? 0xfa44070000000000ull : 0xfa44010000000000ull, 0, 0, 0);
}

void Encoder::Nop() {
    Push(0xf800094000000000ull, 0, 0, 0);
}

void Encoder::NopAfterKill() {
    Push(0xf804014000000000ull, 0, 0, 0);
}

void Encoder::Wdf(uint8_t drc) {
    // The fragment form always carried bit 43 (nosched) since it was copied from a
    // compiler sequence that needed it; now placed by FinishNosched like every other.
    Push(drc ? 0xf920000100000000ull : 0xf920000000000000ull, 43, 0, 0);
}

void Encoder::KillUnlessP1() {
    kill_indices.push_back(words.size());
    Push(0xf9300406f0000000ull, 0, 0, 0); // source registers filled in by PatchKills
}

void Encoder::DepthF(Reg src, uint8_t control_sa) {
    src = Map(src);
    if (src.bank != Bank::Temp) {
        Fail("usse: depthf reads a temporary");
    }
    // psp2cgc's word for `depthf r0` with the control word at sa2: fb300000f0000102.
    Push(Pack(PatDepthf, {
        {'s', 0}, {'r', 0}, {'e', 0}, {'c', 0}, {'b', 0}, {'n', 0}, {'p', 0}, {'t', 0}, {'f', 0},
        {'a', 0}, {'k', 3}, {'d', 3}, {'g', 0}, {'h', SingleNum(src, 7)}, {'i', control_sa},
        {'j', control_sa},
    }), 43, 0, 0);
}

void Encoder::PatchKills(uint32_t control_sa) {
    // Bits 13-7 and 6-0 are the op's src1 and src2 register numbers (the DEPTHF layout,
    // which shares the visibility-test category); the bank bits above them stay as the
    // compiler writes them.
    const uint64_t regs = static_cast<uint64_t>(control_sa & 0x7f);
    for (const std::size_t index : kill_indices) {
        words[index] = (words[index] & ~0x3fffull) | (regs << 7) | regs;
    }
}

void Encoder::Vec(VecOp op, Dst dst, Src src1, Src src2, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src1.reg = Map(src1.reg);
    src2.reg = Map(src2.reg);

    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto [s2b, s2ext] = Src12Bank(src2.reg.bank);
    if (src2.mod == Mod::Neg || src2.mod == Mod::NegAbs) {
        Fail("usse: the second source of a vector operation cannot be negated");
    }
    const uint64_t swz1 = Swizzle12(src1.swz);
    Push(Pack(PatV32Nmad, {
        {'p', ExtVecPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'r', (swz1 >> 10) & 3}, {'y', 0}, {'d', dext},
        {'c', (swz1 >> 9) & 1}, {'b', s1ext}, {'a', s2ext}, {'w', Vec4StdSwizzle(src2.swz)},
        {'n', 0}, {'e', dst.mask}, {'m', static_cast<uint64_t>(src1.mod)},
        {'o', src2.mod == Mod::Abs ? 1u : 0u}, {'i', (swz1 >> 7) & 3}, {'t', db}, {'k', s1b},
        {'l', s2b}, {'f', DoubledNum(dst.reg, 7)}, {'z', swz1 & 0x7f},
        {'g', static_cast<uint64_t>(op)}, {'h', DoubledNum(src1.reg, 7)},
        {'j', DoubledNum(src2.reg, 7)},
    }), 43, Reads(src1, dst.mask) | Reads(src2, dst.mask), Writes(dst, pred));
}

void Encoder::Vec16(VecOp op, Dst dst, Src src1, Src src2, Pred pred, bool nosched) {
    Vec(op, dst, src1, src2, pred, nosched);
    // Same fields, the opcode selects the f16 unit.
    uint64_t& word = words.back();
    word = (word & ~(0x1full << 59)) | (0b00010ull << 59);
    (void)PatV16Nmad;
}

void Encoder::Mad(Dst dst, Src src1, Src gpi0, Src gpi1, bool vec4, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src1.reg = Map(src1.reg);

    if (gpi0.reg.bank != Bank::Internal || gpi1.reg.bank != Bank::Internal) {
        Fail("usse: mad's second and third sources are internal registers");
    }
    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto pick = [vec4](const Swizzle& s) { return vec4 ? Vec4Swizzle(s) : Vec3Swizzle(s); };
    const auto [sw1, sw1ext] = pick(src1.swz);
    const auto [sw0, sw0ext] = pick(gpi0.swz);
    const auto [sw2, sw2ext] = pick(gpi1.swz);
    const auto neg = [](Mod m) { return (m == Mod::Neg || m == Mod::NegAbs) ? 1u : 0u; };
    const auto abs = [](Mod m) { return (m == Mod::Abs || m == Mod::NegAbs) ? 1u : 0u; };
    Push(Pack(PatVmad, {
        {'p', ExtVecPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'g', sw2ext}, {'o', vec4 ? 1u : 0u}, {'d', dext},
        {'e', 0}, {'r', s1ext}, {'a', 3}, {'i', abs(gpi0.mod)}, {'t', 0},
        {'n', 0}, {'w', dst.mask}, {'c', neg(src1.mod)}, {'b', abs(src1.mod)},
        {'f', neg(gpi1.mod)}, {'h', abs(gpi1.mod)}, {'z', sw0ext}, {'k', db}, {'j', s1b},
        {'l', gpi0.reg.n}, {'m', DoubledNum(dst.reg, 7)}, {'q', sw0}, {'u', sw2},
        {'v', gpi1.reg.n}, {'x', neg(gpi0.mod)}, {'y', sw1ext}, {'A', sw1},
        {'B', DoubledNum(src1.reg, 7)},
    }), 43, Reads(src1, dst.mask) | Reads(gpi0, dst.mask) | Reads(gpi1, dst.mask),
         Writes(dst, pred));
}

bool Encoder::Mad2Swizzle(const Swizzle& s, uint64_t& code) {
    if (s[0] == Ch::X && s[1] == Ch::X) {
        code = 0;
    } else if (s[0] == Ch::Y && s[1] == Ch::Y) {
        code = 1;
    } else if (s[0] == Ch::X && s[1] == Ch::Y) {
        code = 4;
    } else {
        return false;
    }
    return true;
}

void Encoder::Mad2(Dst dst, Src src0, Src src1, Src src2, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src0.reg = Map(src0.reg);
    src1.reg = Map(src1.reg);
    src2.reg = Map(src2.reg);
    if (dst.mask & ~3u) {
        Fail("usse: vmad2 writes a pair");
    }
    if (pred != Pred::None) {
        Fail("usse: vmad2 predication is unverified");
    }
    uint64_t s0b = 0;
    switch (src0.reg.bank) {
    case Bank::Temp:
        s0b = 0;
        break;
    case Bank::PrimAttr:
        s0b = 1;
        break;
    default:
        Fail("usse: vmad2's first source is a temporary or a primary attribute");
    }
    if (src0.mod == Mod::Neg || src0.mod == Mod::NegAbs || src1.mod != Mod::None ||
        src2.mod != Mod::None) {
        Fail("usse: vmad2 modifier not encodable");
    }
    uint64_t sw0, sw1, sw2;
    if (!Mad2Swizzle(src0.swz, sw0) || !Mad2Swizzle(src1.swz, sw1) || !Mad2Swizzle(src2.swz, sw2)) {
        Fail("usse: vmad2 swizzle not in its tables");
    }
    const auto [db, dext] = DestBank(dst.reg.bank);
    if (dext) {
        Fail("usse: vmad2 destination bank");
    }
    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto [s2b, s2ext] = Src12Bank(src2.reg.bank);
    Push(Pack(PatVmad2, {
        {'d', 0}, {'p', 0}, {'s', skip_invalid ? 1u : 0u}, {'r', sw0 >> 2}, {'y', 0},
        {'c', src0.mod == Mod::Abs ? 1u : 0u}, {'b', s1ext}, {'a', s2ext}, {'w', sw2},
        {'i', sw1 >> 2}, {'n', 0}, {'e', dst.mask}, {'m', 0}, {'o', 0}, {'k', s0b}, {'t', db},
        {'f', s1b}, {'g', s2b}, {'h', DoubledNum(dst.reg, 6)}, {'z', sw1 & 3}, {'j', sw0 & 3},
        {'l', DoubledNum(src0.reg, 6)}, {'q', DoubledNum(src1.reg, 6)}, {'u', DoubledNum(src2.reg, 6)},
    }), 43, Reads(src0, dst.mask) | Reads(src1, dst.mask) | Reads(src2, dst.mask),
         Writes(dst, pred));
}

void Encoder::Dot(Dst dst, Src src1, Src gpi0, bool vec4, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src1.reg = Map(src1.reg);

    if (gpi0.reg.bank != Bank::Internal) {
        Fail("usse: dot's second source is an internal register");
    }
    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto [sw0, sw0ext] = vec4 ? Vec4Swizzle(gpi0.swz) : Vec3Swizzle(gpi0.swz);
    if (sw0ext) {
        Fail("usse: dot's internal source only has the standard swizzle table");
    }
    const auto neg = [](Mod m) { return (m == Mod::Neg || m == Mod::NegAbs) ? 1u : 0u; };
    const auto abs = [](Mod m) { return (m == Mod::Abs || m == Mod::NegAbs) ? 1u : 0u; };
    Push(Pack(PatVdp, {
        {'p', ExtVecPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'c', 0}, {'o', vec4 ? 1u : 0u}, {'d', dext},
        {'e', 0}, {'r', s1ext}, {'a', 3}, {'g', abs(gpi0.mod)}, {'t', 0},
        {'n', 0}, {'w', dst.mask}, {'b', neg(src1.mod)}, {'f', abs(src1.mod)},
        {'l', 0}, {'k', db}, {'h', s1b}, {'i', gpi0.reg.n}, {'j', DoubledNum(dst.reg, 7)},
        {'z', sw0}, {'m', static_cast<uint64_t>(src1.swz[3])},
        {'q', static_cast<uint64_t>(src1.swz[2])}, {'y', static_cast<uint64_t>(src1.swz[1])},
        {'x', static_cast<uint64_t>(src1.swz[0])}, {'u', DoubledNum(src1.reg, 7)},
    }), 43, Reads(src1, vec4 ? 0xf : 0x7) | Reads(gpi0, vec4 ? 0xf : 0x7), Writes(dst, pred));
}

void Encoder::Comp(CompOp op, Dst dst, Src src, uint8_t src_comp, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src.reg = Map(src.reg);

    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [sb, sext] = Src12Bank(src.reg.bank);
    Push(Pack(PatVcomp, {
        {'p', ExtPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'d', 0}, {'y', 0}, {'e', dext}, {'n', 0}, {'r', sext},
        {'a', 0}, {'o', 0}, {'b', static_cast<uint64_t>(op)}, {'c', 0},
        {'m', static_cast<uint64_t>(src.mod)}, {'f', src_comp}, {'t', db}, {'k', sb},
        {'g', DoubledNum(dst.reg, 8)}, {'h', DoubledNum(src.reg, 8)}, {'w', dst.mask},
    }), 43, ReadsOne(src.reg, src_comp), Writes(dst, pred));
}

namespace {
uint64_t PackMove(uint64_t dest_fmt, Dst dst, Reg lo, Reg hi, bool skip_invalid) {
    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [lb, lext] = Src12Bank(lo.bank);
    uint64_t hb = 0, hext = 0, hn = 0;
    if (lo.bank == Bank::Internal) {
        // An internal register is a whole 128 bits; the second source is marked absent.
        hb = 2;
        hext = 1;
        hn = 0;
    } else {
        const auto [b, e] = Src12Bank(hi.bank);
        hb = b;
        hext = e;
        hn = DoubledNum(hi, 7);
    }
    return Pack(PatVpck, {
        {'p', 0}, {'s', skip_invalid ? 1u : 0u}, {'n', 0}, {'u', 0}, {'y', 0}, {'d', dext},
        {'e', 0}, {'r', lext}, {'c', hext}, {'a', 0}, {'f', 6}, {'t', dest_fmt},
        {'m', dst.mask}, {'b', db}, {'k', lb}, {'l', hb}, {'g', SingleNum(dst.reg, 7)},
        {'o', 3}, {'h', 0}, {'i', 1}, {'j', 2}, {'q', DoubledNum(lo, 7)}, {'v', 0},
        {'w', hn}, {'x', 0},
    });
}
} // Anonymous namespace

void Encoder::MoveF32x4(Dst dst, Reg lo, Reg hi, bool nosched) {
    dst.reg = Map(dst.reg);
    lo = Map(lo);
    hi = Map(hi);

    Push(PackMove(6, dst, lo, hi, skip_invalid), 54, ReadsAll(lo) | ReadsAll(hi),
         Writes(dst, Pred::None));
}

void Encoder::PackF16(Dst dst, Reg lo, Reg hi, bool nosched) {
    dst.reg = Map(dst.reg);
    lo = Map(lo);
    hi = Map(hi);

    Push(PackMove(5, dst, lo, hi, skip_invalid), 54, ReadsAll(lo) | ReadsAll(hi),
         Writes(dst, Pred::None));
}

void Encoder::MoveInternal(Dst dst, Src src, Pred pred, bool nosched) {
    dst.reg = Map(dst.reg);
    src.reg = Map(src.reg);

    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [sb, sext] = Src12Bank(src.reg.bank);
    Push(Pack(PatVmov, {
        {'p', ExtPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'t', 0}, {'r', 0}, {'y', 0}, {'d', dext}, {'e', 0},
        {'c', sext}, {'b', 0}, {'m', 0}, {'a', 0}, {'n', 0}, {'o', 5},
        {'i', 0}, {'w', Vec4StdSwizzle(src.swz)}, {'k', 0}, {'l', db}, {'f', sb}, {'g', 0},
        {'h', dst.mask}, {'j', DoubledNum(dst.reg, 7)}, {'q', 0},
        {'u', DoubledNum(src.reg, 7)}, {'v', 0},
    }), 43, Reads(src, dst.mask), Writes(dst, pred));
}

void Encoder::CondMove(Dst dst, Src src0, Src src1, Src src2, bool nosched) {
    dst.reg = Map(dst.reg);
    src0.reg = Map(src0.reg);
    src1.reg = Map(src1.reg);
    src2.reg = Map(src2.reg);

    const auto [db, dext] = DestBank(dst.reg.bank);
    const auto [s0b, s0ext] = Src0Bank(src0.reg.bank);
    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto [s2b, s2ext] = Src12Bank(src2.reg.bank);
    Push(Pack(PatVmov, {
        {'p', 0}, {'s', skip_invalid ? 1u : 0u}, {'t', 0}, {'r', 0}, {'y', 0}, {'d', dext}, {'e', s0ext},
        {'c', s1ext}, {'b', s2ext}, {'m', 1}, {'a', 0}, {'n', 0}, {'o', 5},
        {'i', 1}, {'w', Vec4StdSwizzle(src0.swz)}, {'k', s0b}, {'l', db}, {'f', s1b},
        {'g', s2b}, {'h', dst.mask}, {'j', DoubledNum(dst.reg, 7)},
        {'q', DoubledNum(src0.reg, 7)}, {'u', DoubledNum(src1.reg, 7)},
        {'v', DoubledNum(src2.reg, 7)},
    }), 43, Reads(src0, dst.mask) | Reads(src1, dst.mask) | Reads(src2, dst.mask),
         Writes(dst, Pred::None));
}

void Encoder::Test(Cond cond, uint8_t pred_n, Src src1, uint8_t chan, Src src2,
                   bool src2_scalar, Pred pred) {
    src1.reg = Map(src1.reg);
    src2.reg = Map(src2.reg);

    const auto [s1b, s1ext] = Src12Bank(src1.reg.bank);
    const auto [s2b, s2ext] = Src12Bank(src2.reg.bank);
    // The test unit subtracts src2 from src1 and tests the sign and the zero flag of the
    // difference: sign 2 = positive, 1 = negative; zero 1 = include equal, 2 = exclude;
    // and whether the two are combined with AND (1) or OR (0).
    uint64_t sign = 0, zero = 0, andc = 0;
    switch (cond) {
    case Cond::Gt: sign = 2; zero = 2; andc = 1; break;
    case Cond::Ge: sign = 2; zero = 1; andc = 0; break;
    case Cond::Lt: sign = 1; zero = 2; andc = 1; break;
    case Cond::Le: sign = 1; zero = 1; andc = 0; break;
    case Cond::Eq: sign = 0; zero = 1; andc = 1; break; // psp2cgc's cmp.eq.f32 (2026-09-05)
    case Cond::Ne: sign = 0; zero = 2; andc = 1; break;
    }
    Push(Pack(PatVtst, {
        {'p', ExtPred(pred)}, {'s', skip_invalid ? 1u : 0u}, {'K', 0}, {'o', 0}, {'y', 0}, {'d', 1},
        {'r', (src1.mod == Mod::Neg) ? 1u : 0u}, {'c', s1ext}, {'e', s2ext}, {'a', 1},
        {'v', src2_scalar ? 1u : 0u}, {'t', 0}, {'i', sign}, {'z', zero}, {'m', andc},
        {'h', chan}, {'n', pred_n}, {'b', 1}, {'k', s1b}, {'f', s2b}, {'g', 0}, {'w', 0},
        {'l', 0}, {'u', 14}, {'j', DoubledNum(src1.reg, 8)}, {'q', DoubledNum(src2.reg, 8)},
    }), 54, ReadsOne(src1.reg, chan) | ReadsOne(src2.reg, src2_scalar ? 0 : chan), 0);
}

void Encoder::Sample2D(Reg dst, Reg coord, uint8_t sampler_sa, const Reg* lod, uint8_t drc,
                       bool nosched) {
    const auto [cb, cext] = Src0Bank(coord.bank);
    if (dst.bank != Bank::Temp && dst.bank != Bank::PrimAttr) {
        Fail("usse: a texture read lands in a temporary or a primary attribute");
    }
    Push(Pack(PatSmp, {
        {'p', 0}, {'s', lod ? 1u : 0u}, {'n', 0}, {'y', 0}, {'m', 0},
        {'r', cext}, {'c', 0}, {'e', lod ? 0u : 1u}, {'f', 3}, {'a', 0}, {'d', 1},
        {'l', lod ? 2u : 0u}, {'t', dst.bank == Bank::PrimAttr ? 1u : 0u}, {'b', 0},
        {'g', 0}, {'k', cb}, {'h', drc}, {'i', 3}, {'j', lod ? 0u : 2u}, {'o', dst.n},
        {'q', DoubledNum(coord, 8)}, {'u', static_cast<uint64_t>(sampler_sa >> 1)},
        {'v', lod ? DoubledNum(*lod, 8) : 0u},
    }), 54, static_cast<uint16_t>(ReadsAll(coord) | (lod ? ReadsAll(*lod) : 0)), 0);
}

} // namespace GxmRenderer::Usse
