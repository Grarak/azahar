#pragma once
#ifdef _MSC_VER
// nonstandard extension used: zero-sized array in struct/union
#pragma warning(disable : 4200)
#endif

#include <cstddef>
#include "common/common_types.h"

struct ARMul_State;
typedef unsigned int (*shtop_fp_t)(ARMul_State* cpu, unsigned int sht_oper);

enum class TransExtData {
    COND = (1 << 0),
    NON_BRANCH = (1 << 1),
    DIRECT_BRANCH = (1 << 2),
    INDIRECT_BRANCH = (1 << 3),
    CALL = (1 << 4),
    RET = (1 << 5),
    END_OF_PAGE = (1 << 6),
    THUMB = (1 << 7),
    SINGLE_STEP = (1 << 8)
};

struct arm_inst {
    unsigned int idx;
    unsigned int cond;
    TransExtData br;
    char component[0];
};

/**
 * How far one instruction advances the translation cache. Instructions are packed one after
 * another and stepped over by their own size, so what an allocation takes and what a step
 * takes must agree - and both are rounded to eight, because an ARM host needs every
 * instruction's fields aligned: the compiler pairs two 32-bit stores into one STRD, which
 * faults on an address that is not even, and struct sizes here are not all multiples of four.
 */
constexpr std::size_t TransInstStride(std::size_t total_size) {
    return (total_size + 7) & ~std::size_t{7};
}

struct generic_arm_inst {
    u32 Ra;
    u32 Rm;
    u32 Rn;
    u32 Rd;
    u8 op1;
    u8 op2;
};

struct adc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct add_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct orr_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct and_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct eor_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct bbl_inst {
    unsigned int L;
    int signed_immed_24;
    unsigned int next_addr;
    unsigned int jmp_addr;
};

struct bx_inst {
    unsigned int Rm;
};

struct blx_inst {
    union {
        s32 signed_immed_24;
        u32 Rm;
    } val;
    unsigned int inst;
};

struct clz_inst {
    unsigned int Rm;
    unsigned int Rd;
};

struct cps_inst {
    unsigned int imod0;
    unsigned int imod1;
    unsigned int mmod;
    unsigned int A, I, F;
    unsigned int mode;
};

struct clrex_inst {};

struct cpy_inst {
    unsigned int Rm;
    unsigned int Rd;
};

struct bic_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct sub_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct tst_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct cmn_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct teq_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct stm_inst {
    unsigned int inst;
};

struct bkpt_inst {
    u32 imm;
};

struct stc_inst {};

struct ldc_inst {};

struct swi_inst {
    unsigned int num;
};

struct cmp_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mov_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mvn_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct rev_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int op1;
    unsigned int op2;
};

struct rsb_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct rsc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct sbc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mul_inst {
    unsigned int S;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
};

struct smul_inst {
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
    unsigned int x;
    unsigned int y;
};

struct umull_inst {
    unsigned int S;
    unsigned int RdHi;
    unsigned int RdLo;
    unsigned int Rs;
    unsigned int Rm;
};

struct smlad_inst {
    unsigned int m;
    unsigned int Rm;
    unsigned int Rd;
    unsigned int Ra;
    unsigned int Rn;
    unsigned int op1;
    unsigned int op2;
};

struct smla_inst {
    unsigned int x;
    unsigned int y;
    unsigned int Rm;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rn;
};

struct smlalxy_inst {
    unsigned int x;
    unsigned int y;
    unsigned int RdLo;
    unsigned int RdHi;
    unsigned int Rm;
    unsigned int Rn;
};

struct ssat_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int imm5;
    unsigned int sat_imm;
    unsigned int shift_type;
};

struct umaal_inst {
    unsigned int Rn;
    unsigned int Rm;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct umlal_inst {
    unsigned int S;
    unsigned int Rm;
    unsigned int Rs;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct smlal_inst {
    unsigned int S;
    unsigned int Rm;
    unsigned int Rs;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct smlald_inst {
    unsigned int RdLo;
    unsigned int RdHi;
    unsigned int Rm;
    unsigned int Rn;
    unsigned int swap;
    unsigned int op1;
    unsigned int op2;
};

struct mla_inst {
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
};

struct mrc_inst {
    unsigned int opcode_1;
    unsigned int opcode_2;
    unsigned int cp_num;
    unsigned int crn;
    unsigned int crm;
    unsigned int Rd;
    unsigned int inst;
};

struct mcr_inst {
    unsigned int opcode_1;
    unsigned int opcode_2;
    unsigned int cp_num;
    unsigned int crn;
    unsigned int crm;
    unsigned int Rd;
    unsigned int inst;
};

struct mcrr_inst {
    unsigned int opcode_1;
    unsigned int cp_num;
    unsigned int crm;
    unsigned int rt;
    unsigned int rt2;
};

struct mrs_inst {
    unsigned int R;
    unsigned int Rd;
};

struct msr_inst {
    unsigned int field_mask;
    unsigned int R;
    unsigned int inst;
};

struct pld_inst {};

struct sxtb_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct sxtab_inst {
    unsigned int Rd;
    unsigned int Rn;
    unsigned int Rm;
    unsigned rotate;
};

struct sxtah_inst {
    unsigned int Rd;
    unsigned int Rn;
    unsigned int Rm;
    unsigned int rotate;
};

struct sxth_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct uxtab_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int rotate;
    unsigned int Rm;
};

struct uxtah_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int rotate;
    unsigned int Rm;
};

struct uxth_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct cdp_inst {
    unsigned int opcode_1;
    unsigned int CRn;
    unsigned int CRd;
    unsigned int cp_num;
    unsigned int opcode_2;
    unsigned int CRm;
    unsigned int inst;
};

struct uxtb_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct swp_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int Rm;
};

struct setend_inst {
    unsigned int set_bigend;
};

struct b_2_thumb {
    unsigned int imm;
};
struct b_cond_thumb {
    unsigned int imm;
    unsigned int cond;
};

struct bl_1_thumb {
    unsigned int imm;
};
struct bl_2_thumb {
    unsigned int imm;
};
struct blx_1_thumb {
    unsigned int imm;
    unsigned int instr;
};

struct pkh_inst {
    unsigned int Rm;
    unsigned int Rn;
    unsigned int Rd;
    unsigned char imm;
};

// Floating point VFPv3 structures
#define VFP_INTERPRETER_STRUCT
#include "core/arm/skyeye_common/vfp/vfpinstr.cpp"
#undef VFP_INTERPRETER_STRUCT

typedef void (*get_addr_fp_t)(ARMul_State* cpu, unsigned int inst, unsigned int& virt_addr);

struct ldst_inst {
    unsigned int inst;
    get_addr_fp_t get_addr;
};

typedef arm_inst* ARM_INST_PTR;
typedef ARM_INST_PTR (*transop_fp_t)(unsigned int, int);

extern const transop_fp_t arm_instruction_trans[];
extern const std::size_t arm_instruction_trans_len;

#ifdef __vita__
// The Vita's 512 MB user budget already holds 128 MB of guest FCRAM, the GXM heaps and the
// texture cache, so the cache is a thirty-second of the desktop one. It is cleared on every
// guest ICache invalidation, so the size bounds how much code one title can run between
// flushes, not how much it can run in total.
#define TRANS_CACHE_SIZE (32 * 1024 * 1024)
#else
#define TRANS_CACHE_SIZE (64 * 1024 * 2000)
#endif
// Allocated on first use rather than as a static array: 125 MB of .bss inflated the load image
// past what the PS Vita's loader could map alongside the newlib heap, and on that target the
// native backend is preferred so the interpreter — and this buffer — is usually never touched.
extern char* trans_cache_buf;
extern std::size_t trans_cache_buf_top;
