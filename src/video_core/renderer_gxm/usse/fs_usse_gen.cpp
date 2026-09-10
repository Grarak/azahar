// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <string>
#include <fmt/format.h>
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/usse/fs_usse_gen.h"
#include "video_core/renderer_gxm/usse/gxp_writer.h"
#include "video_core/renderer_gxm/usse/usse_encoder.h"
#include "video_core/renderer_gxm/usse/usse_ir.h"

namespace GxmRenderer::Usse {

namespace {

using Pica::FramebufferRegs;
using Pica::LightingRegs;
using Pica::TexturingRegs;
using Pica::Shader::FSConfig;
using TevStageConfig = TexturingRegs::TevStageConfig;
using Source = TevStageConfig::Source;
using Operation = TevStageConfig::Operation;
using ColorModifier = TevStageConfig::ColorModifier;
using AlphaModifier = TevStageConfig::AlphaModifier;
using TextureType = TexturingRegs::TextureConfig::TextureType;
using Sampler = LightingRegs::LightingSampler;
using LutInput = LightingRegs::LightingLutInput;

// The generator's notion of a stage that changes nothing.
bool IsPassThroughTevStage(const TevStageConfig& stage) {
    return (stage.color_op == Operation::Replace && stage.alpha_op == Operation::Replace &&
            stage.color_source1 == Source::Previous && stage.alpha_source1 == Source::Previous &&
            stage.color_modifier1 == ColorModifier::SourceColor &&
            stage.alpha_modifier1 == AlphaModifier::SourceAlpha &&
            stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1);
}

// The generator's stage-0 quirk: Previous on the first stage reads that stage's source 3.
Source ColorSource(const TevStageConfig& stage, Source source, u32 index) {
    return (index == 0 && source == Source::Previous) ? stage.color_source3.Value() : source;
}

Source AlphaSource(const TevStageConfig& stage, Source source, u32 index) {
    return (index == 0 && source == Source::Previous) ? stage.alpha_source3.Value() : source;
}

/**
 * The register plan: everything long-lived in the unified store,
 * the three internal registers only inside one pass, with nosched set for as long as one
 * of them holds a value.
 *
 * Primary attributes: pa0-3 COLOR0, then the inputs a configuration needs.
 * Temporaries (32-bit each), fixed homes:
 */
constexpr uint8_t TmpOutput = 0;      ///< combiner_output, the running result (4)
constexpr uint8_t TmpTex0 = 4;        ///< the three texture units' colours (4 each)
constexpr uint8_t TmpBuffer = 16;     ///< combiner_buffer (4)
constexpr uint8_t TmpNextBuffer = 20; ///< next_combiner_buffer (4)
constexpr uint8_t TmpPrimary = 24;    ///< the vertex colour, byte-rounded (4)
constexpr uint8_t TmpArg = 28;        ///< the three stage arguments (4 each); lighting
                                      ///< scratch before the stages run
constexpr uint8_t TmpLutCoord = 28;   ///< lighting/fog: the LUT texel coordinate (2)
constexpr uint8_t TmpLutDelta = 30;   ///< lighting/fog: the interpolation weight (1)
constexpr uint8_t TmpLutEntry = 32;   ///< lighting/fog: the fetched LUT entry (4)
constexpr uint8_t TmpHalf = 36;       ///< lighting: the normalised half vector (3)
constexpr uint8_t TmpLight = 40;      ///< lighting: the normalised light vector (3)
constexpr uint8_t TmpMisc = 44;       ///< lighting: dot product, clamp_highlights, distance,
                                      ///< the squared half vector length (4)
constexpr uint8_t TmpNormal = 48;     ///< lighting: the surface normal in view space (3)
constexpr uint8_t TmpView = 52;       ///< lighting: the normalised view vector (3)
constexpr uint8_t TmpDiffuse = 56;    ///< lighting: diffuse_sum, primary_fragment_color (4)
constexpr uint8_t TmpSpecular = 60;   ///< lighting: specular_sum, secondary_fragment_color (4)
constexpr uint8_t TmpTangent = 16;    ///< lighting: the tangent (Config7 only) (3), in the
                                      ///< combiner buffer's home, which is filled afterwards
constexpr uint8_t TmpLightAtten = 0;  ///< lighting: per-light attenuation (.x) and geometric
                                      ///< factor (.y), in combiner_output's home (2)
constexpr uint8_t TmpShadow = 64;     ///< lighting: the shadow texture's colour (4)
constexpr uint8_t TmpLightMem = 68;   ///< lighting: a light's parameters loaded from the
                                      ///< uniform buffer (24: the SA layout, six quads)
constexpr uint8_t TmpProj = 92;       ///< unit 0's projected coordinate (2), or the depth (1)
constexpr uint8_t TmpTex3 = 96;       ///< the procedural texture's colour (4)
constexpr uint8_t TmpProcUv = 100;    ///< proctex: u, v (2), the level (.z)
constexpr uint8_t TmpSnap = 104;      ///< a stage's output kept for a later stage's
                                      ///< PreviousBuffer read (4 per stage, 0-3)
constexpr uint8_t TmpLightAcc = 20;   ///< lighting: the spot direction, then the reflection
                                      ///< vector, then the specular accumulator, in
                                      ///< next_combiner_buffer's home (3). Nothing lighting
                                      ///< keeps may sit under a LUT fetch's scratch
                                      ///< (TmpLutCoord..TmpLutEntry + 3) or the half vector.

// The hardware constant bank as the disassembly names it: c<n>.x is table entry n and .y
// entry n + 1 (the bank is not pair-addressed). Entry 0 is 0, entries 2-3 are 1, entry 4
// is 2, entry 12 is 0.5 (Vita3K's usse_constant_table.h; confirmed by psp2shaderperf).
const Src ConstZero{Reg::C(0), XXXX};
const Src ConstOne{Reg::C(2), XXXX};
const Src ConstHalf{Reg::C(12), XXXX};

const Reg I0 = Reg::I(0);
const Reg I1 = Reg::I(1);
const Reg I2 = Reg::I(2);

// The literals, in the data container right after the uniforms. The first one is the
// kill's operand, not a number: psp2cgc's kill (a visibility-test op whose two source
// registers are the data container's first SA, on all 148 SM3DL programs and the
// ubershader) reads the raw word 0xE000 there, and the second literal pads it to a pair.
enum Lit : uint8_t {
    LitKillControl,
    LitKillControlPad,
    LitTwoFiftyFive,
    LitInv255,
    LitTwo,
    LitFour,
    Lit128,
    Lit127,
    LitInv256,
    LitInvLutRows,
    LitHalfInv256,
    Lit256,
    Lit255,
    LitNeg128,
    // Procedural textures: the LUT texture rows, the configuration's constants.
    LitInvRgRows,
    LitProcWidth,   ///< lut_width
    LitProcWidthM1, ///< lut_width - 1
    LitProcOff0,    ///< the colour LUT's offsets by mip level, 0-3 from the configuration
    LitProcOff1,
    LitProcOff2,
    LitProcOff3,
    LitProc240,     ///< and 4-7 fixed
    LitProc248,
    LitProc252,
    LitProc254,
    LitProcLodMin,
    LitProcLodMax,
    LitProcPad,
    LitPlusTwo,  ///< the (2, -2) pair the default normal's rotation multiplies by
    LitMinusTwo,
    // LUT texel coordinates: (column, row) * scale + bias, pairs.
    LitLutScaleX, ///< 1, 1 / LUT_LF_ROWS
    LitLutScaleY,
    LitLutBiasX,  ///< 0.5 / 256, 0.5 / LUT_LF_ROWS
    LitLutBiasY,
    LitRgScaleX,  ///< the same for the RG / RGBA LUT rows
    LitRgScaleY,
    LitRgBiasX,
    LitRgBiasY,
    LitNeg127,    ///< the clamp bounds of a negated LUT index
    LitNeg255,
    LiteralCount, // even, so the sampler words stay pair-aligned
};
static_assert(LiteralCount % 2 == 0);

// A float4 somewhere in the unified store: two 64-bit pairs at base and base + 2.
struct Vec4Loc {
    Bank bank;
    uint8_t base;
    Reg Pair(int which) const {
        return {bank, static_cast<uint8_t>(base + which * 2)};
    }
};

class FragmentEmitter {
public:
    FragmentEmitter(const FSConfig& config_, int level_) : config{config_}, level{level_} {}

    std::vector<uint8_t> Emit(std::string* refusal);
    int level = 1;

private:
    /// Null when the configuration can be emitted, else what stops it.
    const char* Unsupported() const;
    void PlanInputs();
    void PlanUniforms();
    void SampleTextures();
    void RoundPrimary();
    void Lighting();
    void LightingLutLookup(Sampler sampler, bool abs_input, bool two_sided);
    void LutLfFetch();
    /// A LUT texel fetch: i1.y holds the index, `sampler_index` names the LUT texture (256
    /// texels a row, `inv_rows` its row count's inverse); the entry lands in TmpLutEntry.
    /// Fetches the LUT entry at index i1.y (or -i1.y when `negated`) of a texture with the
    /// given row scale and bias literal pairs into TmpLutEntry, and waits.
    void LutFetch(uint32_t sampler_index, uint8_t scale_lit, uint8_t bias_lit, bool negated);
    void ProcTex();
    /// ProcTexLookupLUT: the rg LUT at i0.x (0..1) plus `offset`, interpolated, into i0.x.
    void ProcTexLookup(Src offset);
    /// The combiner of u and v (TmpProcUv) into i0.x.
    void ProcTexCombine(Pica::TexturingRegs::ProcTexCombiner combiner);
    /// The shift term for one axis from the other's coordinate, into i2.x.
    void ProcTexShift(Pica::TexturingRegs::ProcTexShift mode, Pica::TexturingRegs::ProcTexClamp clamp,
                      Src coord);
    /// One axis' clamp, in place on i0.x.
    void ProcTexClamp(Pica::TexturingRegs::ProcTexClamp mode);
    /// floor(i0.<lane>) in place, through i1.
    void FloorLane(uint8_t lane);
    void TevStage(u32 index);
    /// Loads stage argument `arg_index` into i<arg_index>, the lanes in `lanes` (0x7 colour,
    /// 0x8 alpha, 0xf both), straight from its sources.
    void LoadArg(uint8_t arg_index, const TevStageConfig& stage, u32 stage_index, Source csrc,
                 ColorModifier cmod, Source asrc, AlphaModifier amod, uint8_t lanes);
    Vec4Loc SourceLocation(Source source, u32 stage_index, bool alpha) const;
    /// Works out which stage output each PreviousBuffer read sees.
    void PlanCombinerBuffer();
    void ColorOp(Operation op, uint8_t mask);
    void AlphaTest();
    /// The nop a kill needs, then the phase marker that starts the next phase.
    void NewPhaseAfterKill();
    void Scissor();
    void Fog();
    void Byteround(Reg value, Reg scratch);
    void Clamp01(Reg value, uint8_t mask = 0xf);
    void LoadVec4(Reg internal, Vec4Loc from);
    void StoreVec4(Vec4Loc to, Reg internal);
    void CopyVec4(Vec4Loc to, Vec4Loc from, uint8_t mask = 0xf);
    /// Normalises the float3 at `from` into internal register `into` (xyz), through `scratch`.
    void Normalize3(Reg into, Vec4Loc from, Reg scratch);
    /// A scalar secondary attribute as a broadcast source.
    static Src SaScalar(uint32_t sa) {
        return {Reg::Sa(static_cast<uint8_t>(sa & ~1u)), (sa & 1) ? YYYY : XXXX};
    }
    /// A scalar anywhere in the unified store as a broadcast source.
    static Src Scalar(Bank bank, uint32_t reg) {
        return {Reg{bank, static_cast<uint8_t>(reg & ~1u)}, (reg & 1) ? YYYY : XXXX};
    }
    /// The data container starts after the resident uniforms; its first SA is the uniform
    /// buffer pointer when the program reads uniforms from memory (data_head).
    Src Lit(uint8_t index) const {
        return SaScalar(LiteralSaOffset(uniform_floats + data_head, index));
    }
    uint8_t SamplerSa(uint32_t sampler_index) const {
        return static_cast<uint8_t>(
            SamplerSaOffset(uniform_floats + data_head, LiteralCount, sampler_index));
    }
    /// A uniform in the memory part of the default uniform buffer: `components` floats at
    /// a float offset past the resident part (the light parameters of lights past the SA
    /// budget); returns the offset.
    uint32_t AllocateMemoryUniform(const std::string& name, uint8_t components);
    /// A light's parameters: the SA quad set for a resident light, else loaded into
    /// TmpLightMem here (three loads and a wait).
    Vec4Loc LightParams(uint32_t num);
    void WriteDepth();
    /// A uniform's place: `components` floats (or array_size float4s) at a pair-aligned
    /// offset.
    uint32_t AllocateUniform(const std::string& name, uint8_t components, uint32_t array_size);
    /// The LUT sampler's index among the declared samplers.
    uint32_t LutSamplerIndex() const;
    bool LightingLutEnabled(const Pica::Shader::LutConfig& lut, Sampler sampler) const;

    FSConfig config;
    Ir::Builder e; ///< the program, in virtual quads; usse_ir places them
    GxpProgram gxp;
    std::array<bool, 3> unit_used{};
    std::array<uint8_t, 3> texcoord_pa{}; ///< PA base of TEXCOORD0..2, when declared
    uint8_t position_pa = 0;              ///< PA base of WPOS (x, y, z, w), when declared
    uint8_t texcoord0_w_pa = 0;           ///< PA of TEXCOORD3 (unit 0's projective w)
    uint8_t scissor_pa = 0;               ///< scratch PA holding the scissor test's value
                                          ///< until the alpha test folds it into its kill
    bool fold_scissor = false;            ///< the scissor test kills with the alpha test
    uint8_t normquat_pa = 0;              ///< PA base of TEXCOORD4/5 (two float4s)
    uint8_t view_pa = 0;                  ///< PA base of TEXCOORD6 (float3)
    bool buffer_used = false;
    /// Where a stage's PreviousBuffer colour / alpha comes from: -2 zero (stage 0), -1 the
    /// tev_combiner_buffer_color uniform, else the stage whose output it is.
    std::array<int, 6> buffer_color_from{};
    std::array<int, 6> buffer_alpha_from{};
    /// Lanes of a stage's output a later stage reads through PreviousBuffer.
    std::array<uint8_t, 6> snapshot_lanes{};
    bool uses_kill = false;
    bool uses_scissor = false;
    bool uses_fog = false;
    bool uses_lighting = false;
    bool uses_primary_color = false;   ///< a stage reads PrimaryFragmentColor
    bool uses_secondary_color = false; ///< a stage reads SecondaryFragmentColor
    bool lut_used = false;
    // Where the uniforms landed (secondary attributes), assigned by PlanUniforms.
    uint32_t uniform_floats = 0;
    uint32_t sa_alpha_ref = 0;
    uint32_t sa_const_color = 0;
    uint32_t sa_buffer_color = 0;
    uint32_t sa_scissor = 0;
    uint32_t sa_fog_offset = 0;
    uint32_t sa_fog_color = 0;
    uint32_t sa_lut_offset = 0;
    uint32_t sa_global_ambient = 0;
    uint32_t sa_border = 0;               ///< tex_border_color float4[3]
    uint32_t data_head = 0;               ///< 1 when the data container starts with the pointer
    uint32_t memory_floats = 0;           ///< floats in the memory part of the uniform buffer
    struct MemoryUniform {
        std::string name;
        uint32_t offset;
        uint8_t components;
    };
    std::vector<MemoryUniform> memory_uniforms;
    std::array<int32_t, 8> light_memory{}; ///< per light number: float offset, or -1 (resident)
    bool uses_depth = false;              ///< W-buffering: the fragment writes its depth
    bool uses_shadow = false;             ///< lighting reads the shadow texture
    bool uses_border = false;             ///< a unit clamps to its border colour
    bool uses_proctex = false;
    uint32_t sampler_rg = 0;              ///< the rg and rgba LUT samplers' indices
    uint32_t sampler_rgba = 0;
    uint32_t sa_proctex = 0;              ///< lut_offset, diff_lut_offset, color_map_offset,
                                          ///< alpha_map_offset, bias (scalars, in that order)
    std::array<uint32_t, 8> sa_light{}; ///< per PICA light number; per light: specular_0
                                        ///< +0, dist_atten_bias +3, specular_1 +4,
                                        ///< dist_atten_scale +7, diffuse +8, ambient +12,
                                        ///< position +16, spot_direction +20
};

const char* FragmentEmitter::Unsupported() const {
    const auto& fb = config.framebuffer;
    const auto& tex = config.texture;
    const auto& lighting = config.lighting;
    if (config.proctex.enable && config.proctex.noise_enable) {
        return "proctex noise";
    }
    if (tex.fog_mode == TexturingRegs::FogMode::Gas) {
        return "gas";
    }
    if (fb.shadow_rendering) {
        return "shadow rendering";
    }
    if (fb.logic_op != FramebufferRegs::LogicOp::Copy &&
        fb.logic_op != FramebufferRegs::LogicOp::NoOp) {
        return "logic op";
    }
    auto rgb = fb.requested_rgb_blend;
    auto alpha = fb.requested_alpha_blend;
    if (rgb.RequiresMinMaxEmulation() || alpha.RequiresMinMaxEmulation()) {
        return "min/max blend";
    }
    // Cube and shadow textures: the rasterizer binds a white texture for them (P5), so a
    // program sampling them would not be the fix.
    const auto t0 = tex.texture0_type.Value();
    if (t0 != TextureType::Disabled && t0 != TextureType::Texture2D &&
        t0 != TextureType::Projection2D) {
        return "texture unit 0 type";
    }
    for (u32 i = 0; i < tex.tev_stages.size(); i++) {
        const TevStageConfig stage = tex.tev_stages[i];
        if (IsPassThroughTevStage(stage)) {
            continue;
        }
        const Source sources[] = {
            ColorSource(stage, stage.color_source1, i), ColorSource(stage, stage.color_source2, i),
            ColorSource(stage, stage.color_source3, i), AlphaSource(stage, stage.alpha_source1, i),
            AlphaSource(stage, stage.alpha_source2, i), AlphaSource(stage, stage.alpha_source3, i),
        };
        for (const Source s : sources) {
            switch (s) {
            case Source::PrimaryColor:
            case Source::Texture0:
            case Source::Texture1:
            case Source::Texture2:
            case Source::PreviousBuffer:
            case Source::Constant:
            case Source::Previous:
                break;
            case Source::PrimaryFragmentColor:
            case Source::SecondaryFragmentColor:
                break;
            case Source::Texture3: // the procedural texture, zero when it is off
                break;
            default:
                return "tev source";
            }
        }
    }
    return nullptr;
}

bool FragmentEmitter::LightingLutEnabled(const Pica::Shader::LutConfig& lut,
                                         Sampler sampler) const {
    return lut.enable &&
           LightingRegs::IsLightingSamplerSupported(config.lighting.config, sampler);
}

void FragmentEmitter::PlanInputs() {
    const auto& tex = config.texture;
    const auto& lighting = config.lighting;
    uses_lighting = lighting.enable;
    for (u32 i = 0; i < tex.tev_stages.size(); i++) {
        const TevStageConfig stage = tex.tev_stages[i];
        if (IsPassThroughTevStage(stage)) {
            continue;
        }
        const Source sources[] = {
            ColorSource(stage, stage.color_source1, i), ColorSource(stage, stage.color_source2, i),
            ColorSource(stage, stage.color_source3, i), AlphaSource(stage, stage.alpha_source1, i),
            AlphaSource(stage, stage.alpha_source2, i), AlphaSource(stage, stage.alpha_source3, i),
        };
        for (const Source s : sources) {
            if (s == Source::Texture0 && tex.texture0_type != TextureType::Disabled) {
                unit_used[0] = true;
            } else if (s == Source::Texture1) {
                unit_used[1] = true;
            } else if (s == Source::Texture2) {
                unit_used[2] = true;
            } else if (s == Source::PreviousBuffer) {
                buffer_used = true;
            } else if (s == Source::PrimaryFragmentColor) {
                uses_primary_color = true;
            } else if (s == Source::SecondaryFragmentColor) {
                uses_secondary_color = true;
            } else if (s == Source::Texture3 && config.proctex.enable) {
                uses_proctex = true;
            }
        }
    }
    if (uses_lighting && lighting.bump_mode != LightingRegs::LightingBumpMode::None) {
        const u32 unit = lighting.bump_selector;
        if (unit < 3 && (unit != 0 || tex.texture0_type != TextureType::Disabled)) {
            unit_used[unit] = true;
        }
    }
    uses_scissor = config.framebuffer.scissor_test_mode.Value() !=
                   Pica::RasterizerRegs::ScissorMode::Disabled;
    uses_fog = tex.fog_mode == TexturingRegs::FogMode::Fog;
    uses_depth = config.framebuffer.depthmap_enable ==
                 Pica::RasterizerRegs::DepthBuffering::WBuffering;
    if (uses_lighting && lighting.enable_shadow) {
        const u32 unit = lighting.shadow_selector;
        if (unit < 3 && (unit != 0 || tex.texture0_type != TextureType::Disabled)) {
            unit_used[unit] = true;
            uses_shadow = true;
        }
    }
    for (u32 unit = 0; unit < 3; unit++) {
        const auto& border = tex.texture_border_color[unit];
        if (unit_used[unit] && (border.enable_s || border.enable_t)) {
            uses_border = true;
        }
    }
    if (uses_lighting) {
        lut_used = LightingLutEnabled(lighting.lut_d0, Sampler::Distribution0) ||
                   LightingLutEnabled(lighting.lut_d1, Sampler::Distribution1) ||
                   LightingLutEnabled(lighting.lut_fr, Sampler::Fresnel) ||
                   LightingLutEnabled(lighting.lut_rr, Sampler::ReflectRed) ||
                   LightingLutEnabled(lighting.lut_rg, Sampler::ReflectGreen) ||
                   LightingLutEnabled(lighting.lut_rb, Sampler::ReflectBlue);
        for (u32 i = 0; i < lighting.src_num; i++) {
            const auto& light = lighting.lights[i];
            if (light.dist_atten_enable) {
                lut_used = true;
            }
            if (light.spot_atten_enable &&
                LightingLutEnabled(lighting.lut_sp, Sampler::SpotlightAttenuation)) {
                lut_used = true;
            }
        }
    }
    lut_used = lut_used || uses_fog;

    // pa0-3 is the vertex colour; the four-wide inputs follow, then the pairs, then the
    // one float3 (which need not be followed by anything pair-aligned).
    gxp.inputs.push_back({FragmentInput::Color0, 4});
    uint8_t pa = 4;
    if (uses_scissor || uses_fog || uses_depth) {
        gxp.inputs.push_back({FragmentInput::Position, 4});
        position_pa = pa;
        pa += 4;
    }
    if (uses_lighting) {
        gxp.inputs.push_back({FragmentInput::TexCoord4, 4});
        gxp.inputs.push_back({FragmentInput::TexCoord5, 4});
        normquat_pa = pa;
        pa += 8;
    }
    const u32 proc_coord = uses_proctex ? std::min<u32>(config.proctex.coord, 2) : 3;
    const bool coord_needed[3] = {
        unit_used[0] || proc_coord == 0,
        unit_used[1] || (unit_used[2] && tex.texture2_use_coord1) || proc_coord == 1,
        (unit_used[2] && !tex.texture2_use_coord1) || proc_coord == 2};
    const FragmentInput coord_inputs[3] = {FragmentInput::TexCoord0, FragmentInput::TexCoord1,
                                           FragmentInput::TexCoord2};
    for (int c = 0; c < 3; c++) {
        if (coord_needed[c]) {
            gxp.inputs.push_back({coord_inputs[c], 2});
            texcoord_pa[c] = pa;
            pa += 2;
        }
    }
    if (uses_lighting) {
        // Declared four wide, as psp2cgc declares a float3 varying (descriptor components
        // 4, four registers): the console, unlike Vita3K, allocates by the declaration.
        gxp.inputs.push_back({FragmentInput::TexCoord6, 4});
        view_pa = pa;
        pa += 4;
    }
    if (unit_used[0] && tex.texture0_type == TextureType::Projection2D) {
        // texcoord0's w, one register, last so nothing pair-aligned has to follow it.
        gxp.inputs.push_back({FragmentInput::TexCoord3, 1});
        texcoord0_w_pa = pa;
        pa += 1;
    }
    // One kill per program, as the compiler has it: with both tests on, the scissor
    // result waits in a scratch PA (the compiler's pa9) for the alpha test's kill.
    fold_scissor = uses_scissor &&
                   config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Always;
    if (fold_scissor) {
        pa += pa & 1;
        scissor_pa = pa;
        pa += 2;
    }
    gxp.pa_count = pa;
    for (u32 unit = 0; unit < 3; unit++) {
        if (unit_used[unit]) {
            const char* names[3] = {"tex0", "tex1", "tex2"};
            gxp.samplers.push_back({names[unit], unit});
        }
    }
    if (lut_used) {
        gxp.samplers.push_back({"texture_buffer_lut_lf", 3});
    }
    if (uses_proctex) {
        sampler_rg = static_cast<uint32_t>(gxp.samplers.size());
        gxp.samplers.push_back({"texture_buffer_lut_rg", 4});
        sampler_rgba = static_cast<uint32_t>(gxp.samplers.size());
        gxp.samplers.push_back({"texture_buffer_lut_rgba", 5});
    }
}

uint32_t FragmentEmitter::AllocateUniform(const std::string& name, uint8_t components,
                                          uint32_t array_size) {
    // Vectors and arrays start on a pair (array elements are four floats apart, as
    // sceGxmSetUniformDataF lays a float4 array out); scalars pack, so consecutive scalars
    // such as the scissor bounds form pairs the program reads as one operand.
    if ((components > 1 || array_size > 1) && (uniform_floats & 1)) {
        uniform_floats++;
    }
    const uint32_t at = uniform_floats;
    gxp.uniforms.push_back({name, at, components, array_size});
    uniform_floats += array_size > 1 ? array_size * 4 : components;
    return at;
}

void FragmentEmitter::PlanUniforms() {
    const auto& tex = config.texture;
    const auto& lighting = config.lighting;
    if (config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Always &&
        config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Never) {
        sa_alpha_ref = AllocateUniform("alphatest_ref", 1, 1);
    }
    // The constant colours as a prefix up to the last stage that reads one: the rasterizer
    // writes all six and sceGxmSetUniformDataF clips to the parameter's size.
    u32 const_stages = 0;
    for (u32 i = 0; i < tex.tev_stages.size(); i++) {
        const TevStageConfig stage = tex.tev_stages[i];
        if (IsPassThroughTevStage(stage)) {
            continue;
        }
        const Source sources[] = {
            ColorSource(stage, stage.color_source1, i), ColorSource(stage, stage.color_source2, i),
            ColorSource(stage, stage.color_source3, i), AlphaSource(stage, stage.alpha_source1, i),
            AlphaSource(stage, stage.alpha_source2, i), AlphaSource(stage, stage.alpha_source3, i),
        };
        for (const Source s : sources) {
            if (s == Source::Constant) {
                const_stages = i + 1;
            }
        }
    }
    if (const_stages != 0) {
        sa_const_color = AllocateUniform("const_color", 4, const_stages);
    }
    if (buffer_used) {
        sa_buffer_color = AllocateUniform("tev_combiner_buffer_color", 4, 1);
    }
    if (uses_scissor) {
        // Four scalars the program reads as two pairs: they start on a pair.
        uniform_floats += uniform_floats & 1;
        sa_scissor = AllocateUniform("scissor_x1", 1, 1);
        AllocateUniform("scissor_y1", 1, 1);
        AllocateUniform("scissor_x2", 1, 1);
        AllocateUniform("scissor_y2", 1, 1);
    }
    if (uses_fog) {
        sa_fog_offset = AllocateUniform("fog_lut_offset", 1, 1);
        sa_fog_color = AllocateUniform("fog_color", 3, 1);
    }
    if (uses_border) {
        sa_border = AllocateUniform("tex_border_color", 4, 3);
    }
    if (uses_proctex) {
        sa_proctex = AllocateUniform("proctex_lut_offset", 1, 1);
        AllocateUniform("proctex_diff_lut_offset", 1, 1);
        AllocateUniform("proctex_color_map_offset", 1, 1);
        AllocateUniform("proctex_alpha_map_offset", 1, 1);
        AllocateUniform("proctex_bias", 1, 1);
    }
    light_memory.fill(-1);
    if (uses_lighting) {
        // The LUT offsets as a prefix of the float4[6] up to the highest sampler used.
        u32 highest = 0;
        const auto note = [&](bool used, u32 sampler) {
            if (used) {
                highest = std::max(highest, sampler + 1);
            }
        };
        note(LightingLutEnabled(lighting.lut_d0, Sampler::Distribution0), 0);
        note(LightingLutEnabled(lighting.lut_d1, Sampler::Distribution1), 1);
        note(LightingLutEnabled(lighting.lut_fr, Sampler::Fresnel), 3);
        note(LightingLutEnabled(lighting.lut_rb, Sampler::ReflectBlue), 4);
        note(LightingLutEnabled(lighting.lut_rg, Sampler::ReflectGreen), 5);
        note(LightingLutEnabled(lighting.lut_rr, Sampler::ReflectRed), 6);
        for (u32 i = 0; i < lighting.src_num; i++) {
            const auto& light = lighting.lights[i];
            note(light.spot_atten_enable &&
                     LightingLutEnabled(lighting.lut_sp, Sampler::SpotlightAttenuation),
                 8 + light.num);
            note(light.dist_atten_enable, 16 + light.num);
        }
        if (highest != 0) {
            sa_lut_offset = AllocateUniform("lighting_lut_offset", 4, (highest + 3) / 4);
        }
        sa_global_ambient = AllocateUniform("lighting_global_ambient", 3, 1);
        // Lights are register-resident (24 SAs each) while the budget holds, counting the
        // literals, the sampler words and the pointer behind them; the rest live in the
        // memory part of the uniform buffer and are loaded when their turn comes.
        const uint32_t tail = LiteralCount + static_cast<uint32_t>(gxp.samplers.size()) * 4 + 1;
        for (u32 i = 0; i < lighting.src_num; i++) {
            const u32 num = lighting.lights[i].num;
            const char* members[8] = {"specular_0", "dist_atten_bias", "specular_1",
                                      "dist_atten_scale", "diffuse", "ambient", "position",
                                      "spot_direction"};
            const uint8_t sizes[8] = {3, 1, 3, 1, 3, 3, 3, 3};
            const bool resident_light = ((uniform_floats + 1) & ~1u) + 24 + tail <= 128;
            for (int m = 0; m < 8; m++) {
                const std::string name = fmt::format("light_src[{}].{}", num, members[m]);
                if (resident_light) {
                    const uint32_t at = AllocateUniform(name, sizes[m], 1);
                    if (m == 0) {
                        sa_light[num] = at;
                    }
                } else {
                    const uint32_t at = AllocateMemoryUniform(name, sizes[m]);
                    if (m == 0) {
                        light_memory[num] = static_cast<int32_t>(at);
                    }
                }
            }
        }
    }
    if (!memory_uniforms.empty()) {
        // The pointer takes the data container's first SA; an odd resident count keeps the
        // literals behind it on their pairs.
        uniform_floats |= 1;
        data_head = 1;
        gxp.buffer_pointer = true;
        gxp.ldst_base_value = static_cast<int32_t>(uniform_floats * 4) - 4;
        gxp.buffer_floats = uniform_floats + memory_floats;
        for (const MemoryUniform& m : memory_uniforms) {
            gxp.uniforms.push_back({m.name, uniform_floats + m.offset, m.components, 1});
        }
    } else if (uniform_floats & 1) {
        uniform_floats++;
    }
    gxp.uniform_floats = uniform_floats;
}

uint32_t FragmentEmitter::AllocateMemoryUniform(const std::string& name, uint8_t components) {
    // The SA layout, so the code addressing a light does not care where it lives: every
    // member starts on a pair, a float3 takes four floats with its scalar partner.
    if (components > 1 && (memory_floats & 1)) {
        memory_floats++;
    }
    const uint32_t at = memory_floats;
    memory_uniforms.push_back({name, at, components});
    memory_floats += components;
    return at;
}

Vec4Loc FragmentEmitter::LightParams(uint32_t num) {
    if (light_memory[num] < 0) {
        return {Bank::SecAttr, static_cast<uint8_t>(sa_light[num])};
    }
    // 24 floats from the buffer: sixteen, then two fours. The immediate offset is in
    // floats and seven bits wide; the fourth light past the budget still fits.
    const uint32_t off = static_cast<uint32_t>(light_memory[num]);
    if (off + 20 > 127) {
        throw std::runtime_error("light parameters past the load offset's reach");
    }
    const Reg ptr = Reg::Sa(static_cast<uint8_t>(uniform_floats)); // data container offset 0
    e.Lda32(Reg::V(TmpLightMem), ptr, nullptr, static_cast<uint8_t>(off), 16, 0);
    e.Lda32(Reg::V(TmpLightMem + 16), ptr, nullptr, static_cast<uint8_t>(off + 16), 4, 0);
    e.Lda32(Reg::V(TmpLightMem + 20), ptr, nullptr, static_cast<uint8_t>(off + 20), 4, 0);
    e.WdfVertex(0);
    return {Bank::Virtual, TmpLightMem};
}

void FragmentEmitter::WriteDepth() {
    // W-buffering: the fragment's depth is z / w of its window position (the vertex program
    // put the PICA depth mapping on z). Written with depthf, which ends the phase as a kill
    // does; the caller opens the next one.
    const Reg zw = Reg::Pa(static_cast<uint8_t>(position_pa + 2));
    e.Comp(CompOp::Rcp, {I0, 0x1}, {zw}, 1, Pred::None, true);
    e.Vec(VecOp::Mul, {Reg::V(TmpProj), 0x1}, {zw, XYZW}, {I0, XXXX}, Pred::None, true);
    e.DepthF(Reg::V(TmpProj), static_cast<uint8_t>(LiteralSaOffset(uniform_floats + data_head,
                                                                    LitKillControl)));
}

uint32_t FragmentEmitter::LutSamplerIndex() const {
    uint32_t index = 0;
    for (u32 u = 0; u < 3; u++) {
        index += unit_used[u] ? 1 : 0;
    }
    return index;
}

void FragmentEmitter::NewPhaseAfterKill() {
    // A kill ends its phase, and a program has one kill and two phases, never more: three
    // phases locked the console's GPU up on the title screen's alpha-tested sprites, and a
    // second kill inside phase two locked it up as well (both 2026-09-05). The scissor
    // and alpha tests therefore share one kill (fold_scissor).
    e.NopAfterKill();
    e.PhaseStart();
    e.Phas(true);
    e.Nop();
}

void FragmentEmitter::Scissor() {
    if (!uses_scissor) {
        return;
    }
    // inside = x >= x1 && y >= y1 && x < x2 && y < y2. Pixel centres sit at .5 and the
    // bounds are whole numbers, so x < x2 is x2 - x - 0.5 >= 0, and the four conditions
    // become one minimum against zero.
    const Reg pos = Reg::Pa(position_pa);
    const Reg lo = Reg::Sa(static_cast<uint8_t>(sa_scissor));
    const Reg hi = Reg::Sa(static_cast<uint8_t>(sa_scissor + 2));
    e.Vec(VecOp::Add, {I0, 0x3}, {lo, XYZW, Mod::Neg}, {pos, XYZW}, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x3}, {pos, XYZW, Mod::Neg}, {hi, XYZW}, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x3}, {Reg::C(12), XXXX, Mod::Neg}, {I1, XYZW}, Pred::None, true);
    e.Vec(VecOp::Min, {I0, 0x3}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
    e.Vec(VecOp::Min, {I0, 0x1}, {I0, XYZW}, {I0, YYYY}, Pred::None, true);
    // Exclude discards the inside (keep when the minimum is negative); Include the outside.
    const bool include = config.framebuffer.scissor_test_mode.Value() ==
                         Pica::RasterizerRegs::ScissorMode::Include;
    if (fold_scissor) {
        e.MoveInternal({Reg::Pa(scissor_pa), 0x1}, {I0, XYZW}, Pred::None, true);
        return;
    }
    e.Test(include ? Cond::Ge : Cond::Lt, 1, {I0}, 0, ConstZero, false);
    e.KillUnlessP1();
    NewPhaseAfterKill();
}

void FragmentEmitter::LutFetch(uint32_t sampler_index, uint8_t scale_lit, uint8_t bias_lit,
                               bool negated) {
    // i1.y holds a whole-number index into a LUT texture (256 texels per row), negated when
    // the caller kept it as -floor (frc's form). (column, row) = (frac, floor) of index /
    // 256, then one scale and one bias to the texel centre.
    e.Vec(VecOp::Mul, {I0, 0x2}, {I1, YYYY, negated ? Mod::Neg : Mod::None}, Lit(LitInv256),
          Pred::None, true);
    e.Vec(VecOp::Frc, {I0, 0x1}, {I0, YYYY}, {I0, YYYY}, Pred::None, true);     // column / 256
    e.Vec(VecOp::Add, {I0, 0x2}, {I0, XXXX, Mod::Neg}, {I0, YYYY}, Pred::None, true); // row
    e.Vec(VecOp::Mul, {I0, 0x3}, {I0, XYZW}, {Lit(scale_lit).reg, XYZW}, Pred::None, true);
    e.Vec(VecOp::Add, {Reg::V(TmpLutCoord), 0x3}, {I0, XYZW}, {Lit(bias_lit).reg, XYZW},
          Pred::None, true);
    e.Sample2D(Reg::V(TmpLutEntry), Reg::V(TmpLutCoord), SamplerSa(sampler_index), nullptr, 0);
    e.Wdf(0);
}

void FragmentEmitter::LutLfFetch() {
    // The lf LUT entry at i1.y interpolated with TmpLutDelta: entry.r + entry.g * delta
    // in i0.x.
    LutFetch(LutSamplerIndex(), LitLutScaleX, LitLutBiasX, true);
    e.Vec(VecOp::Mul, {I0, 0x1}, {Reg::V(TmpLutEntry), YYYY}, {Reg::V(TmpLutDelta), XXXX},
          Pred::None, true);
    e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW}, {Reg::V(TmpLutEntry), XXXX}, Pred::None, true);
}

void FragmentEmitter::FloorLane(uint8_t lane) {
    const Swizzle swz = lane == 0 ? XXXX : lane == 1 ? YYYY : lane == 2 ? ZZZZ : WWWW;
    const uint8_t mask = static_cast<uint8_t>(1u << lane);
    e.Vec(VecOp::Frc, {I1, mask}, {I0, swz}, {I0, swz}, Pred::None, true);
    e.Vec(VecOp::Add, {I0, mask}, {I1, swz, Mod::Neg}, {I0, swz}, Pred::None, true);
}

void FragmentEmitter::ProcTexLookup(Src offset) {
    // coord * 128: the whole part (clamped to the table) selects the entry, the fraction
    // interpolates with its slope, as the generator's ProcTexLookupLUT.
    e.Vec(VecOp::Mul, {I0, 0x2}, {I0, XXXX}, Lit(Lit128), Pred::None, true);
    e.Vec(VecOp::Frc, {I1, 0x2}, {I0, YYYY}, {I0, YYYY}, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x2}, {I1, YYYY, Mod::Neg}, {I0, YYYY}, Pred::None, true); // floor
    e.Vec(VecOp::Max, {I1, 0x2}, {I1, XYZW}, ConstZero, Pred::None, true);
    e.Vec(VecOp::Min, {I1, 0x2}, {I1, XYZW}, Lit(Lit127), Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x1}, {I1, YYYY, Mod::Neg}, {I0, YYYY}, Pred::None, true); // fraction
    e.MoveInternal({Reg::V(TmpLutDelta), 0x1}, {I1, XYZW}, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x2}, {I1, YYYY}, offset, Pred::None, true);
    LutFetch(sampler_rg, LitRgScaleX, LitRgBiasX, false);
    e.Vec(VecOp::Mul, {I0, 0x1}, {Reg::V(TmpLutEntry), YYYY}, {Reg::V(TmpLutDelta), XXXX},
          Pred::None, true);
    e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW}, {Reg::V(TmpLutEntry), XXXX}, Pred::None, true);
    Clamp01(I0, 0x1);
}

void FragmentEmitter::ProcTexCombine(TexturingRegs::ProcTexCombiner combiner) {
    using Combiner = TexturingRegs::ProcTexCombiner;
    const Src u{Reg::V(TmpProcUv), XXXX};
    const Src v{Reg::V(TmpProcUv), YYYY};
    // sqrt(u * u + v * v) into i0.y, zero when the sum is (rsq of zero is not finite).
    const auto length = [&] {
        e.Vec(VecOp::Mul, {I1, 0x1}, u, {Reg::V(TmpProcUv), XXXX}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x2}, v, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        e.Vec(VecOp::Add, {I1, 0x1}, {I1, XXXX}, {I1, YYYY}, Pred::None, true);
        e.Comp(CompOp::Rsq, {I1, 0x2}, {I1}, 0, Pred::None, true);
        e.Comp(CompOp::Rcp, {I0, 0x2}, {I1}, 1, Pred::None, true);
        e.Test(Cond::Gt, 2, {I1}, 0, ConstZero, true);
        e.MoveInternal({I0, 0x2}, ConstZero, Pred::None, true);
        e.Comp(CompOp::Rcp, {I0, 0x2}, {I1}, 1, Pred::P2, true);
    };
    switch (combiner) {
    case Combiner::U:
        e.MoveInternal({I0, 0x1}, u, Pred::None, true);
        break;
    case Combiner::U2:
        e.Vec(VecOp::Mul, {I0, 0x1}, u, {Reg::V(TmpProcUv), XXXX}, Pred::None, true);
        break;
    case Combiner::V:
        e.MoveInternal({I0, 0x1}, v, Pred::None, true);
        break;
    case Combiner::V2:
        e.Vec(VecOp::Mul, {I0, 0x1}, v, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        break;
    case Combiner::Add:
        e.Vec(VecOp::Add, {I0, 0x1}, u, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        break;
    case Combiner::Add2:
        e.Vec(VecOp::Mul, {I1, 0x1}, u, {Reg::V(TmpProcUv), XXXX}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x2}, v, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I1, XXXX}, {I1, YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        break;
    case Combiner::SqrtAdd2:
        length();
        e.Vec(VecOp::Min, {I0, 0x1}, {I0, YYYY}, ConstOne, Pred::None, true);
        break;
    case Combiner::Min:
        e.Vec(VecOp::Min, {I0, 0x1}, u, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        break;
    case Combiner::Max:
        e.Vec(VecOp::Max, {I0, 0x1}, u, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        break;
    case Combiner::RMax:
        length();
        e.Vec(VecOp::Add, {I0, 0x1}, u, {Reg::V(TmpProcUv), YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, XXXX}, {I0, YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        e.Vec(VecOp::Min, {I0, 0x1}, {I0, XYZW}, ConstOne, Pred::None, true);
        break;
    default:
        e.MoveInternal({I0, 0x1}, ConstZero, Pred::None, true);
        break;
    }
}

void FragmentEmitter::ProcTexShift(TexturingRegs::ProcTexShift mode,
                                   TexturingRegs::ProcTexClamp clamp, Src coord) {
    using Shift = TexturingRegs::ProcTexShift;
    if (mode != Shift::Odd && mode != Shift::Even) {
        e.MoveInternal({I2, 0x1}, ConstZero, Pred::None, true);
        return;
    }
    // offset * ((int(c) [+ 1] >> 1) & 1), the integer arithmetic in floats: the coordinate
    // is never negative here, so int(c) is floor(c); x >> 1 is floor(x / 2); x & 1 is
    // 2 * frac(x / 2).
    e.Vec(VecOp::Frc, {I1, 0x1}, coord, coord, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x1}, {I1, XXXX, Mod::Neg}, coord, Pred::None, true); // floor
    if (mode == Shift::Even) {
        e.Vec(VecOp::Add, {I1, 0x1}, {I1, XYZW}, ConstOne, Pred::None, true);
    }
    e.Vec(VecOp::Mul, {I1, 0x2}, {I1, XXXX}, ConstHalf, Pred::None, true);
    e.Vec(VecOp::Frc, {I2, 0x2}, {I1, YYYY}, {I1, YYYY}, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x2}, {I2, YYYY, Mod::Neg}, {I1, YYYY}, Pred::None, true); // >> 1
    e.Vec(VecOp::Mul, {I1, 0x2}, {I1, YYYY}, ConstHalf, Pred::None, true);
    e.Vec(VecOp::Frc, {I1, 0x2}, {I1, YYYY}, {I1, YYYY}, Pred::None, true);
    e.Vec(VecOp::Mul, {I2, 0x1}, {I1, YYYY}, Lit(LitTwo), Pred::None, true);   // & 1
    if (clamp != TexturingRegs::ProcTexClamp::MirroredRepeat) {
        e.Vec(VecOp::Mul, {I2, 0x1}, {I2, XYZW}, ConstHalf, Pred::None, true); // offset 0.5
    }
}

void FragmentEmitter::ProcTexClamp(TexturingRegs::ProcTexClamp mode) {
    using Clamp = TexturingRegs::ProcTexClamp;
    switch (mode) {
    case Clamp::ToZero:
        e.Test(Cond::Gt, 2, {I0}, 0, ConstOne, true);
        e.MoveInternal({I0, 0x1}, ConstZero, Pred::P2, true);
        break;
    case Clamp::SymmetricalRepeat:
        e.Vec(VecOp::Frc, {I0, 0x1}, {I0, XXXX}, {I0, XXXX}, Pred::None, true);
        break;
    case Clamp::MirroredRepeat: {
        // frac(c) for an even floor(c), 1 - frac(c) for an odd one: f + p * (1 - 2 f) with
        // p the parity.
        e.Vec(VecOp::Frc, {I1, 0x1}, {I0, XXXX}, {I0, XXXX}, Pred::None, true);       // f
        e.Vec(VecOp::Add, {I1, 0x2}, {I1, XXXX, Mod::Neg}, {I0, XXXX}, Pred::None, true); // floor
        e.Vec(VecOp::Mul, {I1, 0x2}, {I1, YYYY}, ConstHalf, Pred::None, true);
        e.Vec(VecOp::Frc, {I1, 0x2}, {I1, YYYY}, {I1, YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x2}, {I1, YYYY}, Lit(LitTwo), Pred::None, true);       // p
        e.Vec(VecOp::Mul, {I2, 0x1}, {Lit(LitTwo).reg, Lit(LitTwo).swz, Mod::Neg}, {I1, YYYY},
              Pred::None, true);                                                        // -2p
        e.Vec(VecOp::Add, {I2, 0x1}, {I2, XYZW}, ConstOne, Pred::None, true);           // 1 - 2p
        e.Vec(VecOp::Mul, {I2, 0x1}, {I2, XXXX}, {I1, XXXX}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I2, XXXX}, {I1, YYYY}, Pred::None, true);
        break;
    }
    case Clamp::Pulse:
        e.Test(Cond::Gt, 2, {I0}, 0, ConstHalf, true);
        e.MoveInternal({I0, 0x1}, ConstZero, Pred::None, true);
        e.MoveInternal({I0, 0x1}, ConstOne, Pred::P2, true);
        break;
    case Clamp::ToEdge:
    default:
        e.Vec(VecOp::Min, {I0, 0x1}, {I0, XYZW}, ConstOne, Pred::None, true);
        break;
    }
}

void FragmentEmitter::ProcTex() {
    using Filter = TexturingRegs::ProcTexFilter;
    const auto& pt = config.proctex;
    const Vec4Loc uv_home{Bank::Virtual, TmpProcUv};
    const Reg uv = uv_home.Pair(0);
    const Reg tc = Reg::Pa(texcoord_pa[std::min<u32>(pt.coord, 2)]);
    // uv = |texcoord|
    e.Vec(VecOp::Add, {uv, 0x3}, {tc, XYZW, Mod::Abs}, ConstZero, Pred::None, true);

    // The level, for the mipmapped filters: log2(|lut_width * bias| * (|duv.x| + |duv.y|))
    // clamped to the configuration's range, zero at a zero bias. The nearest level serves
    // both mipmap filter families.
    const auto filter = pt.lut_filter.Value();
    const bool mipmapped = filter != Filter::Linear && filter != Filter::Nearest;
    if (mipmapped) {
        // The derivatives land in the unified store, as the compiler's do (Sony's
        // disassembler rejects them aimed at an internal register).
        const Vec4Loc scratch{Bank::Virtual, TmpTex3};
        e.Vec(VecOp::Dsx, {scratch.Pair(0), 0x3}, {tc, XYZW, Mod::Abs}, {tc, XYZW, Mod::Abs});
        e.Vec(VecOp::Dsy, {scratch.Pair(1), 0x3}, {tc, XYZW, Mod::Abs}, {tc, XYZW, Mod::Abs});
        e.Vec(VecOp::Max, {I0, 0x3}, {scratch.Pair(0), XYZW, Mod::Abs},
              {scratch.Pair(1), XYZW, Mod::Abs}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, XXXX}, {I0, YYYY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x1}, Lit(LitProcWidth), SaScalar(sa_proctex + 4), Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, {I1, XXXX, Mod::Abs}, Pred::None, true);
        e.Comp(CompOp::Log, {I0, 0x1}, {I0}, 0, Pred::None, true);
        e.Test(Cond::Eq, 2, {Reg::Sa(static_cast<uint8_t>((sa_proctex + 4) & ~1u))},
               static_cast<uint8_t>((sa_proctex + 4) & 1), ConstZero, true);
        e.MoveInternal({I0, 0x1}, ConstZero, Pred::P2, true);
        e.Vec(VecOp::Max, {I0, 0x1}, {I0, XYZW}, Lit(LitProcLodMin), Pred::None, true);
        e.Vec(VecOp::Min, {I0, 0x1}, {I0, XYZW}, Lit(LitProcLodMax), Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        FloorLane(0);
        e.MoveInternal({uv_home.Pair(1), 0x1}, {I0, XXXX}, Pred::None, true); // .z = level
    }

    // u = uv.x + u_shift(uv.y), v = uv.y + v_shift(uv.x), each clamped its way.
    ProcTexShift(pt.u_shift, pt.u_clamp, {uv, YYYY});
    e.Vec(VecOp::Add, {I0, 0x1}, {uv, XXXX}, {I2, XXXX}, Pred::None, true);
    ProcTexClamp(pt.u_clamp);
    e.MoveInternal({I2, 0x2}, {I0, XXXX}, Pred::None, true); // u parked in i2.y
    ProcTexShift(pt.v_shift, pt.v_clamp, {uv, YYYY});
    e.MoveInternal({I1, 0x4}, {I2, YYYY}, Pred::None, true); // u parked in i1.z (ProcTexShift uses i2.x, i1.xy)
    e.Vec(VecOp::Add, {I0, 0x1}, {uv, YYYY}, {I2, XXXX}, Pred::None, true);
    e.MoveInternal({I2, 0x2}, {I1, ZZZZ}, Pred::None, true);
    ProcTexClamp(pt.v_clamp);
    e.MoveInternal({uv, 0x2}, {I0, XXXX}, Pred::None, true);
    e.MoveInternal({uv, 0x1}, {I2, YYYY}, Pred::None, true);

    // The colour LUT coordinate, then the colour.
    ProcTexCombine(pt.color_combiner);
    ProcTexLookup(SaScalar(sa_proctex + 2));
    // lut_coord * (width - 1) [+ 0.5 for nearest], the level's offset; the level's width is
    // lut_width >> level (the table is halved per level).
    if (mipmapped) {
        // width_level - 1 = lut_width * 2^-level - 1; offset from the per-level table.
        e.Vec(VecOp::Mul, {I1, 0x1}, {ConstOne.reg, XXXX, Mod::Neg}, {uv_home.Pair(1), XXXX},
              Pred::None, true);
        e.Comp(CompOp::Exp, {I1, 0x1}, {I1}, 0, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x1}, {I1, XYZW}, Lit(LitProcWidth), Pred::None, true);
        e.Vec(VecOp::Add, {I1, 0x1}, {ConstOne.reg, XXXX, Mod::Neg}, {I1, XYZW}, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XXXX}, {I1, XXXX}, Pred::None, true);
        // offset = offsets[level]: walk the table.
        e.MoveInternal({I1, 0x2}, Lit(LitProcOff0), Pred::None, true);
        e.MoveInternal({I1, 0x1}, {uv_home.Pair(1), XXXX}, Pred::None, true); // the level
        const uint8_t table[7] = {LitProcOff1, LitProcOff2, LitProcOff3, LitProc240,
                                  LitProc248, LitProc252, LitProc254};
        for (int level = 1; level <= 7; level++) {
            e.Vec(VecOp::Add, {I1, 0x1}, {ConstOne.reg, XXXX, Mod::Neg}, {I1, XXXX}, Pred::None,
                  true);
            e.Test(Cond::Ge, 2, {I1}, 0, ConstZero, true);
            e.MoveInternal({I1, 0x2}, Lit(table[level - 1]), Pred::P2, true);
        }
    } else {
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XXXX}, Lit(LitProcWidthM1), Pred::None, true);
        e.MoveInternal({I1, 0x2}, Lit(LitProcOff0), Pred::None, true);
    }
    const bool linear = filter == Filter::Linear || filter == Filter::LinearMipmapNearest ||
                        filter == Filter::LinearMipmapLinear;
    const Vec4Loc tex3{Bank::Virtual, TmpTex3};
    const Vec4Loc entry{Bank::Virtual, TmpLutEntry};
    if (linear) {
        // index_i = floor(coord) + offset; fraction f; colour = lut(index_i + lut_offset) +
        // f * lut(index_i + diff_offset).
        e.MoveInternal({I0, 0x4}, {I0, XXXX}, Pred::None, true);
        FloorLane(2);
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, ZZZZ, Mod::Neg}, {I0, XXXX}, Pred::None, true); // f
        e.MoveInternal({Reg::V(TmpLutDelta), 0x1}, {I0, XXXX}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x4}, {I0, ZZZZ}, {I1, YYYY}, Pred::None, true);       // + offset
        e.MoveInternal({uv_home.Pair(1), 0x2}, {I0, ZZZZ}, Pred::None, true);       // .w = index_i
        e.Vec(VecOp::Add, {I1, 0x2}, {I0, ZZZZ}, SaScalar(sa_proctex + 0), Pred::None, true);
        LutFetch(sampler_rgba, LitRgScaleX, LitRgBiasX, false);
        CopyVec4(tex3, entry);
        e.Vec(VecOp::Add, {I1, 0x2}, {uv_home.Pair(1), YYYY}, SaScalar(sa_proctex + 1),
              Pred::None, true);
        LutFetch(sampler_rgba, LitRgScaleX, LitRgBiasX, false);
        LoadVec4(I0, entry);
        e.Vec(VecOp::Mul, {I0, 0xf}, {I0, XYZW}, {Reg::V(TmpLutDelta), XXXX}, Pred::None, true);
        LoadVec4(I1, tex3);
        e.Vec(VecOp::Add, {I0, 0xf}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        StoreVec4(tex3, I0);
    } else {
        // lut(floor(coord + offset + 0.5) + lut_offset)
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, XXXX}, {I1, YYYY}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
        FloorLane(0);
        e.Vec(VecOp::Add, {I1, 0x2}, {I0, XXXX}, SaScalar(sa_proctex + 0), Pred::None, true);
        LutFetch(sampler_rgba, LitRgScaleX, LitRgBiasX, false);
        CopyVec4(tex3, entry);
    }
    if (pt.separate_alpha) {
        ProcTexCombine(pt.alpha_combiner);
        ProcTexLookup(SaScalar(sa_proctex + 3));
        e.MoveInternal({tex3.Pair(1), 0x2}, {I0, XXXX}, Pred::None, true);
    }
}

void FragmentEmitter::Fog() {
    if (!uses_fog) {
        return;
    }
    // fog_index = depth * 128 (or (1 - depth) * 128 flipped); i = clamp(floor(index), 0,
    // 127); f = index - i; entry = lut_lf(i + fog_lut_offset); factor = clamp(r + g * f).
    const Reg pos = Reg::Pa(position_pa + 2); // z, w
    if (config.texture.fog_flip) {
        e.Vec(VecOp::Add, {I0, 0x1}, {pos, XYZW, Mod::Neg}, ConstOne, Pred::None, true);
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(Lit128), Pred::None, true);
    } else {
        e.Vec(VecOp::Mul, {I0, 0x1}, {pos, XYZW}, Lit(Lit128), Pred::None, true);
    }
    // i1.y = -i (frc's floor form, see LightingLutLookup), clamped to [-127, 0].
    e.Vec(VecOp::Frc, {I1, 0x2}, ConstZero, {I0, XXXX}, Pred::None, true);
    e.Vec(VecOp::Max, {I1, 0x2}, {I1, XYZW}, Lit(LitNeg127), Pred::None, true);
    e.Vec(VecOp::Min, {I1, 0x2}, {I1, XYZW}, ConstZero, Pred::None, true);
    e.Vec(VecOp::Add, {I1, 0x1}, {I1, YYYY}, {I0, XXXX}, Pred::None, true); // f = index - i
    e.MoveInternal({Reg::V(TmpLutDelta), 0x3}, {I1, XYZW}, Pred::None, true);
    const Src fog_offset = SaScalar(sa_fog_offset);
    e.Vec(VecOp::Add, {I1, 0x2}, {fog_offset.reg, fog_offset.swz, Mod::Neg}, {I1, YYYY},
          Pred::None, true);
    LutLfFetch();
    Clamp01(I0, 0x1);
    e.MoveInternal({I0, 0xf}, {I0, XXXX}, Pred::None, true);
    LoadVec4(I1, {Bank::Virtual, TmpOutput});
    LoadVec4(I2, {Bank::SecAttr, static_cast<uint8_t>(sa_fog_color)});
    e.Vec(VecOp::Add, {I1, 0x7}, {I2, XYZW, Mod::Neg}, {I1, XYZW}, Pred::None, true);
    e.Mad({I1, 0x7}, {I1, XYZW}, {I0, XYZW}, {I2, XYZW}, true, Pred::None, true);
    StoreVec4({Bank::Virtual, TmpOutput}, I1);
}

void FragmentEmitter::SampleTextures() {
    // Every read on counter 0, one wait for all of them before the first use.
    const auto& tex = config.texture;
    bool any = false;
    uint32_t sampler_index = 0;
    Reg coords[3] = {Reg::Pa(0), Reg::Pa(0), Reg::Pa(0)};
    for (u32 unit = 0; unit < 3; unit++) {
        if (!unit_used[unit]) {
            continue;
        }
        const int coord = (unit == 2 && tex.texture2_use_coord1) ? 1 : static_cast<int>(unit);
        coords[unit] = Reg::Pa(texcoord_pa[coord]);
        if (unit == 0 && tex.texture0_type == TextureType::Projection2D) {
            // texcoord0 / texcoord0_w
            e.Comp(CompOp::Rcp, {I0, 0x1}, {Reg::Pa(static_cast<uint8_t>(texcoord0_w_pa & ~1u))},
                   static_cast<uint8_t>(texcoord0_w_pa & 1), Pred::None, true);
            e.Vec(VecOp::Mul, {Reg::V(TmpProj), 0x3}, {coords[0], XYZW}, {I0, XXXX}, Pred::None,
                  true);
            coords[0] = Reg::V(TmpProj);
        }
        e.Sample2D(Reg::V(static_cast<uint8_t>(TmpTex0 + unit * 4)), coords[unit],
                   SamplerSa(sampler_index), nullptr, 0);
        sampler_index++;
        any = true;
    }
    if (any) {
        e.Wdf(0);
    }
    // A coordinate past [0, 1] on an axis that clamps to the border colour replaces the
    // texel with it (the generator's sampleTexUnit).
    for (u32 unit = 0; unit < 3; unit++) {
        const auto& border = tex.texture_border_color[unit];
        if (!unit_used[unit] || !(border.enable_s || border.enable_t)) {
            continue;
        }
        const Vec4Loc colour{Bank::SecAttr, static_cast<uint8_t>(sa_border + unit * 4)};
        const Vec4Loc texel{Bank::Virtual, static_cast<uint8_t>(TmpTex0 + unit * 4)};
        const auto replace_when = [&](Cond cond, uint8_t chan, const Src& bound) {
            e.Test(cond, 2, {coords[unit]}, chan, bound, true);
            e.MoveInternal({texel.Pair(0), 0x3}, {colour.Pair(0), XYZW}, Pred::P2, true);
            e.MoveInternal({texel.Pair(1), 0x3}, {colour.Pair(1), XYZW}, Pred::P2, true);
        };
        for (uint8_t chan = 0; chan < 2; chan++) {
            if (chan == 0 ? border.enable_s : border.enable_t) {
                replace_when(Cond::Lt, chan, ConstZero);
                replace_when(Cond::Gt, chan, ConstOne);
            }
        }
    }
    if (uses_shadow) {
        // shadow = tex, or 1 - tex when inverted.
        const auto& lighting = config.lighting;
        const Vec4Loc texel{Bank::Virtual, static_cast<uint8_t>(TmpTex0 + lighting.shadow_selector * 4)};
        const Vec4Loc shadow{Bank::Virtual, TmpShadow};
        if (lighting.shadow_invert) {
            for (int pair = 0; pair < 2; pair++) {
                e.Vec(VecOp::Add, {shadow.Pair(pair), 0x3}, {texel.Pair(pair), XYZW, Mod::Neg},
                      ConstOne);
            }
        } else {
            CopyVec4(shadow, texel);
        }
    }
}

void FragmentEmitter::LoadVec4(Reg internal, Vec4Loc from) {
    if (from.bank == Bank::Const) {
        e.MoveInternal({internal, 0xf}, {Reg::C(0), XXXX}, Pred::None, true);
        return;
    }
    e.MoveF32x4({internal, 0xf}, from.Pair(0), from.Pair(1), true);
}

void FragmentEmitter::StoreVec4(Vec4Loc to, Reg internal) {
    e.MoveInternal({to.Pair(0), 0x3}, {internal, XYZW}, Pred::None, true);
    e.MoveInternal({to.Pair(1), 0x3}, {internal, ZWZW}, Pred::None, true);
}

void FragmentEmitter::CopyVec4(Vec4Loc to, Vec4Loc from, uint8_t mask) {
    // Two-wide adds of zero: a move between unified-store registers.
    if (mask & 0x3) {
        e.Vec(VecOp::Add, {to.Pair(0), static_cast<uint8_t>(mask & 0x3)}, {from.Pair(0), XYZW},
              ConstZero);
    }
    if (mask & 0xc) {
        e.Vec(VecOp::Add, {to.Pair(1), static_cast<uint8_t>(mask >> 2)}, {from.Pair(1), XYZW},
              ConstZero);
    }
}

void FragmentEmitter::Byteround(Reg value, Reg) {
    // floor(x * 255 + 0.5) / 255. frc computes src1 - floor(src2), so with a zero first
    // source it is -floor(x) in one instruction (psp2cgc's form), undone by the negation
    // on the last multiply.
    e.Vec(VecOp::Mul, {value, 0xf}, {value, XYZW}, Lit(LitTwoFiftyFive), Pred::None, true);
    e.Vec(VecOp::Add, {value, 0xf}, {value, XYZW}, ConstHalf, Pred::None, true);
    e.Vec(VecOp::Frc, {value, 0xf}, ConstZero, {value, XYZW}, Pred::None, true);
    e.Vec(VecOp::Mul, {value, 0xf}, {value, XYZW, Mod::Neg}, Lit(LitInv255), Pred::None, true);
}

void FragmentEmitter::Clamp01(Reg value, uint8_t mask) {
    e.Vec(VecOp::Max, {value, mask}, {value, XYZW}, ConstZero, Pred::None, true);
    e.Vec(VecOp::Min, {value, mask}, {value, XYZW}, ConstOne, Pred::None, true);
}

void FragmentEmitter::Normalize3(Reg into, Vec4Loc from, Reg scratch) {
    LoadVec4(into, from);
    e.Dot({scratch, 0x1}, {into, XYZW}, {into, XYZW}, false, Pred::None, true);
    e.Comp(CompOp::Rsq, {scratch, 0x1}, {scratch}, 0, Pred::None, true);
    e.Vec(VecOp::Mul, {into, 0x7}, {into, XYZW}, {scratch, XXXX}, Pred::None, true);
}

void FragmentEmitter::RoundPrimary() {
    LoadVec4(I0, {Bank::PrimAttr, 0});
    Byteround(I0, I1);
    StoreVec4({Bank::Virtual, TmpPrimary}, I0);
}

void FragmentEmitter::LightingLutLookup(Sampler sampler, bool abs_input, bool two_sided) {
    // The LUT input sits in i0.x. Unsigned: 256 steps over [0, 1]; signed: 128 steps over
    // [-1, 1] with negatives wrapped to the top half of the table. Leaves the value in i0.x.
    // The index is kept negated: frc with a zero first source is -floor in one
    // instruction, and every later step takes the sign in its stride.
    if (abs_input) {
        if (two_sided) {
            e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW, Mod::Abs}, ConstZero, Pred::None, true);
        } else {
            e.Vec(VecOp::Max, {I0, 0x1}, {I0, XYZW}, ConstZero, Pred::None, true);
        }
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(Lit256), Pred::None, true);
        e.Vec(VecOp::Frc, {I1, 0x2}, ConstZero, {I0, XXXX}, Pred::None, true); // -index
        e.Vec(VecOp::Max, {I1, 0x2}, {I1, XYZW}, Lit(LitNeg255), Pred::None, true);
        e.Vec(VecOp::Min, {I1, 0x2}, {I1, XYZW}, ConstZero, Pred::None, true);
    } else {
        e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(Lit128), Pred::None, true);
        e.Vec(VecOp::Frc, {I1, 0x2}, ConstZero, {I0, XXXX}, Pred::None, true); // -index
        e.Vec(VecOp::Max, {I1, 0x2}, {I1, XYZW}, Lit(LitNeg127), Pred::None, true);
        e.Vec(VecOp::Min, {I1, 0x2}, {I1, XYZW}, Lit(Lit128), Pred::None, true);
    }
    // delta = position - index, then the index wrapped and offset to its row.
    e.Vec(VecOp::Add, {I1, 0x1}, {I1, YYYY}, {I0, XXXX}, Pred::None, true);
    e.MoveInternal({Reg::V(TmpLutDelta), 0x3}, {I1, XYZW}, Pred::None, true);
    if (!abs_input) {
        e.Test(Cond::Gt, 0, {I1}, 1, ConstZero, false);
        e.Vec(VecOp::Add, {I1, 0x2}, {Lit(Lit256).reg, Lit(Lit256).swz, Mod::Neg}, {I1, YYYY},
              Pred::P0, true);
    }
    const uint32_t sampler_index = static_cast<uint32_t>(sampler);
    const Src offset = SaScalar(sa_lut_offset + sampler_index);
    e.Vec(VecOp::Add, {I1, 0x2}, {offset.reg, offset.swz, Mod::Neg}, {I1, YYYY}, Pred::None,
          true);
    LutLfFetch();
}

void FragmentEmitter::Lighting() {
    const auto& lighting = config.lighting;
    const Vec4Loc normquat{Bank::PrimAttr, normquat_pa};
    const Vec4Loc normquat_flat{Bank::PrimAttr, static_cast<uint8_t>(normquat_pa + 4)};
    const Vec4Loc view{Bank::PrimAttr, view_pa};
    const Vec4Loc diffuse_sum{Bank::Virtual, TmpDiffuse};
    const Vec4Loc specular_sum{Bank::Virtual, TmpSpecular};
    const Vec4Loc normal{Bank::Virtual, TmpNormal};
    const Vec4Loc tangent{Bank::Virtual, TmpTangent};
    const Vec4Loc nview{Bank::Virtual, TmpView};
    const Vec4Loc light_vector{Bank::Virtual, TmpLight};
    const Vec4Loc half_vector{Bank::Virtual, TmpHalf};
    const Reg dot_product = Reg::V(TmpMisc);   // .x; .y the squared half vector length
    const Reg misc_hi = Reg::V(TmpMisc + 2);   // .x clamp_highlights, .y light distance
    const bool config7 = lighting.config == LightingRegs::LightingConfig::Config7;

    // diffuse_sum = specular_sum = (0, 0, 0, 1): the alpha here, the colour by the first
    // light's store (or zero below when there is none).
    if (uses_primary_color) {
        e.Vec(VecOp::Add, {diffuse_sum.Pair(1), 0x2}, ConstZero, {Reg::C(1), XYZW});
    }
    if (uses_secondary_color) {
        e.Vec(VecOp::Add, {specular_sum.Pair(1), 0x2}, ConstZero, {Reg::C(1), XYZW});
    }
    if (lighting.src_num == 0) {
        e.Vec(VecOp::Add, {diffuse_sum.Pair(0), 0x3}, ConstZero, ConstZero);
        e.Vec(VecOp::Add, {diffuse_sum.Pair(1), 0x1}, ConstZero, ConstZero);
        e.Vec(VecOp::Add, {specular_sum.Pair(0), 0x3}, ConstZero, ConstZero);
        e.Vec(VecOp::Add, {specular_sum.Pair(1), 0x1}, ConstZero, ConstZero);
    }
    bool diffuse_empty = true;
    bool specular_empty = true;
    // The three colour lanes of an internal register into a quad.
    const auto store3 = [&](Vec4Loc to, Reg internal) {
        e.MoveInternal({to.Pair(0), 0x3}, {internal, XYZW}, Pred::None, true);
        e.MoveInternal({to.Pair(1), 0x1}, {internal, ZWZW}, Pred::None, true);
    };

    // The surface normal and tangent in the quaternion's frame, from the bump map or the
    // defaults.
    const auto perturbation = [&](Reg into) {
        // 2 * tex.rgb - 1
        const Vec4Loc tex{Bank::Virtual, static_cast<uint8_t>(TmpTex0 + lighting.bump_selector * 4)};
        LoadVec4(into, tex);
        e.Vec(VecOp::Mul, {into, 0x7}, {into, XYZW}, Lit(LitTwo), Pred::None, true);
        e.Vec(VecOp::Add, {into, 0x7}, {ConstOne.reg, XXXX, Mod::Neg}, {into, XYZW}, Pred::None,
              true);
    };
    // i0 = surface normal, i1 = surface tangent (only when Config7 wants a tangent).
    const auto default_normal = [&] { // (0, 0, 1)
        e.MoveInternal({I0, 0x7}, {Reg::C(0), XXXX}, Pred::None, true);
        e.Vec(VecOp::Add, {I0, 0x4}, ConstOne, ConstZero, Pred::None, true);
    };
    const auto default_tangent = [&] { // (1, 0, 0)
        e.MoveInternal({I1, 0x7}, {Reg::C(0), XXXX}, Pred::None, true);
        e.Vec(VecOp::Add, {I1, 0x1}, ConstOne, ConstZero, Pred::None, true);
    };
    switch (lighting.bump_mode) {
    case LightingRegs::LightingBumpMode::NormalMap:
        perturbation(I0);
        if (lighting.bump_renorm) {
            // z = sqrt(max(1 - (x*x + y*y), 0))
            e.Vec(VecOp::Mul, {I1, 0x3}, {I0, XYZW}, {I0, XYZW}, Pred::None, true);
            e.Vec(VecOp::Add, {I1, 0x1}, {I1, XYZW}, {I1, YYYY}, Pred::None, true);
            e.Vec(VecOp::Add, {I1, 0x1}, {I1, XYZW, Mod::Neg}, ConstOne, Pred::None, true);
            e.Vec(VecOp::Max, {I1, 0x1}, {I1, XYZW}, ConstZero, Pred::None, true);
            e.Comp(CompOp::Rsq, {I1, 0x2}, {I1}, 0, Pred::None, true);
            e.Vec(VecOp::Mul, {I0, 0x4}, {I1, XXXX}, {I1, YYYY}, Pred::None, true); // x * rsq(x)
            // rsq(0) is inf and 0 * inf is nan: a zero input gives a zero z.
            e.Test(Cond::Gt, 0, {I1}, 0, ConstZero, false);
            e.MoveInternal({I0, 0x4}, {Reg::C(0), XXXX}, Pred::NotP0, true);
        }
        default_tangent();
        break;
    case LightingRegs::LightingBumpMode::TangentMap:
        perturbation(I1);
        default_normal();
        break;
    default:
        default_normal();
        default_tangent();
        break;
    }
    // Park them: the quaternion work below needs all three internal registers. The
    // default normal is not parked: its rotation has a closed form (below).
    const bool default_normal_only = lighting.bump_mode != LightingRegs::LightingBumpMode::NormalMap &&
                                     lighting.bump_mode != LightingRegs::LightingBumpMode::TangentMap &&
                                     !config7;
    if (!default_normal_only) {
        StoreVec4(normal, I0);
    }
    if (config7) {
        StoreVec4(tangent, I1);
    }

    // quat = normalize(dot(normquat, normquat_flat) < 0 ? -normquat : normquat)
    LoadVec4(I0, normquat);
    LoadVec4(I1, normquat_flat);
    e.Dot({I2, 0x1}, {I0, XYZW}, {I1, XYZW}, true, Pred::None, true);
    e.Test(Cond::Lt, 0, {I2}, 0, ConstZero, false);
    e.Vec(VecOp::Mul, {I0, 0xf}, {I0, XYZW, Mod::Neg}, ConstOne, Pred::P0, true);
    e.Dot({I2, 0x1}, {I0, XYZW}, {I0, XYZW}, true, Pred::None, true);
    e.Comp(CompOp::Rsq, {I2, 0x1}, {I2}, 0, Pred::None, true);
    e.Vec(VecOp::Mul, {I0, 0xf}, {I0, XYZW}, {I2, XXXX}, Pred::None, true);
    // rotate(q, v) = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v), for the normal and
    // the tangent; i0 = q throughout.
    const auto rotate = [&](Vec4Loc v) {
        LoadVec4(I1, v);
        // i2 = cross(q, v) + q.w * v = q.yzx * v.zxy - q.zxy * v.yzx + q.w * v. The
        // second source only has the standard swizzle table, which holds zxy but not yzx,
        // so yzx always goes on the first source.
        constexpr Swizzle YZX{Ch::Y, Ch::Z, Ch::X, Ch::W};
        constexpr Swizzle ZXY{Ch::Z, Ch::X, Ch::Y, Ch::W};
        e.Vec(VecOp::Mul, {I2, 0x7}, {I0, YZX}, {I1, ZXY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x7}, {I1, YZX}, {I0, ZXY}, Pred::None, true);
        e.Vec(VecOp::Add, {I2, 0x7}, {I1, XYZW, Mod::Neg}, {I2, XYZW}, Pred::None, true);
        LoadVec4(I1, v);
        e.Mad({I2, 0x7}, {I1, XYZW}, {I0, WWWW}, {I2, XYZW}, true, Pred::None, true);
        // i1 = cross(q, i2) = q.yzx * i2.zxy - q.zxy * i2.yzx; result = v + 2 * i1
        e.Vec(VecOp::Mul, {I1, 0x7}, {I0, YZX}, {I2, ZXY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I2, 0x7}, {I2, YZX}, {I0, ZXY}, Pred::None, true);
        e.Vec(VecOp::Add, {I1, 0x7}, {I2, XYZW, Mod::Neg}, {I1, XYZW}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, Lit(LitTwo), Pred::None, true);
        LoadVec4(I2, v);
        e.Vec(VecOp::Add, {I1, 0x7}, {I1, XYZW}, {I2, XYZW}, Pred::None, true);
        StoreVec4(v, I1);
    };
    if (default_normal_only) {
        // rotate(q, (0, 0, 1)) = (2(xz + yw), 2(yz - xw), 1 - 2(x^2 + y^2)), with i0 = q:
        // P = (x^2, y^2, xz, yz) doubled, Q = (yw, xw, yw, xw), then n.xy = (2, -2) * Q.zw
        // + 2P.zw and n.z = 1 - (2P.x + 2P.y). Six instructions for the general rotation's
        // twenty-two.
        constexpr Swizzle XYZZ{Ch::X, Ch::Y, Ch::Z, Ch::Z};
        constexpr Swizzle XYXY{Ch::X, Ch::Y, Ch::X, Ch::Y};
        constexpr Swizzle YXYX{Ch::Y, Ch::X, Ch::Y, Ch::X};
        e.Vec(VecOp::Mul, {I1, 0xf}, {I0, XYZZ}, {I0, XYXY}, Pred::None, true);
        e.Vec(VecOp::Mul, {I2, 0xf}, {I0, YXYX}, {I0, WWWW}, Pred::None, true);
        e.Vec(VecOp::Mul, {I1, 0xf}, {I1, XYZW}, Lit(LitTwo), Pred::None, true);
        e.Mad({I2, 0x3}, {Lit(LitPlusTwo).reg, XYZW}, {I2, ZWZW}, {I1, ZWZW}, true, Pred::None,
              true);
        e.Vec(VecOp::Add, {I2, 0x4}, {I1, XXXX}, {I1, YYYY}, Pred::None, true);
        e.Vec(VecOp::Add, {I2, 0x4}, {I2, ZZZZ, Mod::Neg}, ConstOne, Pred::None, true);
        StoreVec4(normal, I2);
    } else {
        rotate(normal);
        if (config7) {
            rotate(tangent);
        }
    }
    // What the enabled LUTs read decides whether the view and half vectors are needed at
    // all: a light with the distribution LUT on the (light, normal) angle alone needs
    // neither.
    bool needs_half = false;
    bool needs_view = false;
    for (u32 li = 0; li < lighting.src_num; li++) {
        const auto& light = lighting.lights[li];
        if (uses_secondary_color && (light.geometric_factor_0 || light.geometric_factor_1)) {
            needs_half = true;
        }
        const auto input_of = [&](const Pica::Shader::LutConfig& lut, Sampler sampler) {
            if (!LightingLutEnabled(lut, sampler)) {
                return;
            }
            switch (lut.type.Value()) {
            case LutInput::NH:
            case LutInput::VH:
            case LutInput::CP:
                needs_half = true;
                if (lut.type.Value() == LutInput::VH) {
                    needs_view = true;
                }
                break;
            case LutInput::NV:
                needs_view = true;
                break;
            default:
                break;
            }
        };
        if (uses_secondary_color) {
            input_of(lighting.lut_d0, Sampler::Distribution0);
            input_of(lighting.lut_d1, Sampler::Distribution1);
            input_of(lighting.lut_rr, Sampler::ReflectRed);
            input_of(lighting.lut_rg, Sampler::ReflectGreen);
            input_of(lighting.lut_rb, Sampler::ReflectBlue);
        }
        if (li + 1 == lighting.src_num &&
            ((uses_primary_color && lighting.enable_primary_alpha) ||
             (uses_secondary_color && lighting.enable_secondary_alpha))) {
            input_of(lighting.lut_fr, Sampler::Fresnel);
        }
        if (light.spot_atten_enable) {
            input_of(lighting.lut_sp, Sampler::SpotlightAttenuation);
        }
    }
    if (needs_half || needs_view) {
        Normalize3(I0, view, I1);
        StoreVec4(nview, I0);
    }

    // The LUT inputs computed per light, into i0.x, from the vectors parked in temps.
    const auto lut_input = [&](LutInput input) {
        switch (input) {
        case LutInput::NH:
            LoadVec4(I1, half_vector);
            e.Dot({I0, 0x1}, {normal.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            break;
        case LutInput::VH:
            LoadVec4(I1, half_vector);
            e.Dot({I0, 0x1}, {nview.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            break;
        case LutInput::NV:
            LoadVec4(I1, nview);
            e.Dot({I0, 0x1}, {normal.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            break;
        case LutInput::LN:
            LoadVec4(I1, normal);
            e.Dot({I0, 0x1}, {light_vector.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            break;
        case LutInput::SP:
            // dot(light_vector, spot_dir), spot_dir loaded per light into TmpLightAcc.
            LoadVec4(I1, {Bank::Virtual, TmpLightAcc});
            e.Dot({I0, 0x1}, {light_vector.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            break;
        case LutInput::CP:
            if (config7) {
                // dot(half - normal * dot(normal, half), tangent)
                LoadVec4(I1, half_vector);
                e.Dot({I0, 0x1}, {normal.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
                LoadVec4(I2, normal);
                e.Vec(VecOp::Mul, {I2, 0x7}, {I2, XYZW}, {I0, XXXX}, Pred::None, true);
                e.Vec(VecOp::Add, {I1, 0x7}, {I2, XYZW, Mod::Neg}, {I1, XYZW}, Pred::None, true);
                e.Dot({I0, 0x1}, {tangent.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
            } else {
                e.MoveInternal({I0, 0x1}, {Reg::C(0), XXXX}, Pred::None, true);
            }
            break;
        default:
            e.MoveInternal({I0, 0x1}, {Reg::C(0), XXXX}, Pred::None, true);
            break;
        }
    };
    // scale * LUT(input) into i0.x, for an enabled LUT.
    const auto lut_value = [&](const Pica::Shader::LutConfig& lut, Sampler sampler,
                               bool two_sided) {
        lut_input(lut.type.Value());
        LightingLutLookup(sampler, lut.abs_input, two_sided);
        const float scale = lut.GetScale();
        if (scale != 1.0f) {
            // Scales are powers of two between 0.25 and 8: made from the constant bank.
            if (scale == 2.0f) {
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(LitTwo), Pred::None, true);
            } else if (scale == 4.0f) {
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(LitFour), Pred::None, true);
            } else if (scale == 8.0f) {
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(LitTwo), Pred::None, true);
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, Lit(LitFour), Pred::None, true);
            } else if (scale == 0.5f) {
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
            } else if (scale == 0.25f) {
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
                e.Vec(VecOp::Mul, {I0, 0x1}, {I0, XYZW}, ConstHalf, Pred::None, true);
            }
        }
    };

    for (u32 light_index = 0; light_index < lighting.src_num; light_index++) {
        const auto& light = lighting.lights[light_index];
        const Vec4Loc params = LightParams(light.num);
        const Bank pb = params.bank;
        const uint32_t sa = params.base;
        const Vec4Loc position{pb, static_cast<uint8_t>(sa + 16)};
        const Vec4Loc spot{pb, static_cast<uint8_t>(sa + 20)};
        const Vec4Loc diffuse{pb, static_cast<uint8_t>(sa + 8)};
        const Vec4Loc ambient{pb, static_cast<uint8_t>(sa + 12)};
        const Vec4Loc specular_0{pb, static_cast<uint8_t>(sa + 0)};
        const Vec4Loc specular_1{pb, static_cast<uint8_t>(sa + 4)};
        const bool two_sided = light.two_sided_diffuse;
        const bool shadow_primary = uses_shadow && lighting.shadow_primary && light.shadow_enable;
        const bool shadow_secondary =
            uses_shadow && lighting.shadow_secondary && light.shadow_enable;
        const Vec4Loc shadow{Bank::Virtual, TmpShadow};

        // light_vector = position (+ view unless directional); distance; normalised.
        // Normalising a directional light's direction once per draw in a secondary program
        // (psp2cgc does that) drew a 32-pixel tile pattern over every lit surface on the
        // console, 2026-09-06: the tiles of one fragment job do not all see the same
        // secondary attributes once the secondary program has written them.
        LoadVec4(I0, position);
        if (!light.directional) {
            LoadVec4(I1, view);
            e.Vec(VecOp::Add, {I0, 0x7}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        }
        e.Dot({I1, 0x1}, {I0, XYZW}, {I0, XYZW}, false, Pred::None, true);
        e.Comp(CompOp::Rsq, {I1, 0x2}, {I1}, 0, Pred::None, true);
        if (light.dist_atten_enable) {
            e.Vec(VecOp::Mul, {I1, 0x1}, {I1, XYZW}, {I1, YYYY}, Pred::None, true); // length
            e.MoveInternal({misc_hi, 0x2}, {I1, XXXX}, Pred::None, true);           // .y = distance
        }
        e.Vec(VecOp::Mul, {I0, 0x7}, {I0, XYZW}, {I1, YYYY}, Pred::None, true);
        StoreVec4(light_vector, I0);
        CopyVec4({Bank::Virtual, TmpLightAcc}, spot, 0x7);
        // half_vector = normalize(nview + light_vector); its squared length kept for the
        // geometric factor.
        const bool geo0 = light.geometric_factor_0;
        const bool geo1 = light.geometric_factor_1;
        if (needs_half) {
            LoadVec4(I1, nview);
            e.Vec(VecOp::Add, {I1, 0x7}, {I1, XYZW}, {I0, XYZW}, Pred::None, true);
            e.Dot({I2, 0x1}, {I1, XYZW}, {I1, XYZW}, false, Pred::None, true);
            if (geo0 || geo1) {
                e.MoveInternal({dot_product, 0x2}, {I2, XXXX}, Pred::None, true);
            }
            e.Comp(CompOp::Rsq, {I2, 0x1}, {I2}, 0, Pred::None, true);
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {I2, XXXX}, Pred::None, true);
            StoreVec4(half_vector, I1);
        }
        // dot_product = two_sided ? |dot(l, n)| : max(dot(l, n), 0)
        LoadVec4(I1, normal);
        e.Dot({I0, 0x1}, {light_vector.Pair(0), XYZW}, {I1, XYZW}, false, Pred::None, true);
        if (two_sided) {
            e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW, Mod::Abs}, ConstZero, Pred::None, true);
        } else {
            e.Vec(VecOp::Max, {I0, 0x1}, {I0, XYZW}, ConstZero, Pred::None, true);
        }
        e.MoveInternal({dot_product, 0x1}, {I0, XYZW}, Pred::None, true);
        // clamp_highlights = sign(dot_product): 1 when positive, else 0 (never negative).
        if (lighting.clamp_highlights) {
            e.Test(Cond::Gt, 0, {I0}, 0, ConstZero, false);
            e.MoveInternal({misc_hi, 0x1}, {Reg::C(0), XXXX}, Pred::None, true);
            e.MoveInternal({misc_hi, 0x1}, {Reg::C(2), XXXX}, Pred::P0, true);
        }

        // geo_factor = hh == 0 ? 0 : min(dot_product / hh, 1), in `atten`.y.
        const Reg atten = Reg::V(TmpLightAtten);
        if ((geo0 || geo1) && uses_secondary_color) {
            e.Comp(CompOp::Rcp, {I0, 0x2}, {dot_product}, 1, Pred::None, true);
            e.Vec(VecOp::Mul, {I0, 0x2}, {I0, YYYY}, {dot_product, XXXX}, Pred::None, true);
            e.Vec(VecOp::Min, {I0, 0x2}, {I0, XYZW}, ConstOne, Pred::None, true);
            e.Test(Cond::Gt, 0, {dot_product}, 1, ConstZero, false);
            e.MoveInternal({I0, 0x2}, {Reg::C(0), XXXX}, Pred::NotP0, true);
            e.MoveInternal({atten, 0x2}, {I0, XYZW}, Pred::None, true);
        }
        // Attenuation: spot and distance, multiplied into `atten`.x (starts at 1; no
        // register at all when neither is on).
        const bool spot_on = light.spot_atten_enable &&
                             LightingLutEnabled(lighting.lut_sp, Sampler::SpotlightAttenuation);
        const bool atten_on = spot_on || light.dist_atten_enable;
        if (atten_on) {
            e.MoveInternal({atten, 0x1}, {Reg::C(2), XXXX}, Pred::None, true);
        }
        if (spot_on) {
            lut_value(lighting.lut_sp, LightingRegs::SpotlightAttenuationSampler(light.num),
                      two_sided);
            e.Vec(VecOp::Mul, {atten, 0x1}, {atten, XYZW}, {I0, XXXX}, Pred::None, true);
        }
        if (light.dist_atten_enable) {
            // index = clamp(scale * distance + bias, 0, 1), unsigned lookup
            e.Vec(VecOp::Mul, {I0, 0x1}, {misc_hi, YYYY}, Scalar(pb, sa + 7), Pred::None, true);
            e.Vec(VecOp::Add, {I0, 0x1}, {I0, XYZW}, Scalar(pb, sa + 3), Pred::None, true);
            Clamp01(I0, 0x1);
            LightingLutLookup(LightingRegs::DistanceAttenuationSampler(light.num), true, false);
            e.Vec(VecOp::Mul, {atten, 0x1}, {atten, XYZW}, {I0, XXXX}, Pred::None, true);
        }

        if (uses_secondary_color) {
        // refl = (rr ? LUT : 1, rg ? LUT : refl.r, rb ? LUT : refl.r), kept in TmpLightAcc
        // (the spot direction is no longer needed by then); no register when all three
        // LUTs are off (refl = 1).
        const Vec4Loc acc{Bank::Virtual, TmpLightAcc};
        const bool rr = LightingLutEnabled(lighting.lut_rr, Sampler::ReflectRed);
        const bool rg = LightingLutEnabled(lighting.lut_rg, Sampler::ReflectGreen);
        const bool rb = LightingLutEnabled(lighting.lut_rb, Sampler::ReflectBlue);
        const bool d1_on = LightingLutEnabled(lighting.lut_d1, Sampler::Distribution1);
        const bool d0_on = LightingLutEnabled(lighting.lut_d0, Sampler::Distribution0);
        const bool refl_on = rr || rg || rb;
        if (refl_on) {
            if (rg) {
                lut_value(lighting.lut_rg, Sampler::ReflectGreen, two_sided);
                e.MoveInternal({acc.Pair(0), 0x2}, {I0, XXXX}, Pred::None, true);
            }
            if (rb) {
                lut_value(lighting.lut_rb, Sampler::ReflectBlue, two_sided);
                e.MoveInternal({acc.Pair(1), 0x1}, {I0, XXXX}, Pred::None, true);
            }
            if (rr) {
                lut_value(lighting.lut_rr, Sampler::ReflectRed, two_sided);
            } else {
                e.MoveInternal({I0, 0x1}, {Reg::C(2), XXXX}, Pred::None, true);
            }
            e.MoveInternal({acc.Pair(0), 0x1}, {I0, XXXX}, Pred::None, true);
            if (!rg) {
                e.MoveInternal({acc.Pair(0), 0x2}, {I0, XXXX}, Pred::None, true);
            }
            if (!rb) {
                e.MoveInternal({acc.Pair(1), 0x1}, {I0, XXXX}, Pred::None, true);
            }
        }
        // specular_1 = d1 * refl * light.specular_1 (each factor only when it is not 1),
        // into i2; parked in the accumulator's home when the d0 lookup runs next (it
        // needs the internal registers).
        if (d1_on) {
            lut_value(lighting.lut_d1, Sampler::Distribution1, two_sided);
            LoadVec4(I2, specular_1);
            e.Vec(VecOp::Mul, {I2, 0x7}, {I2, XYZW}, {I0, XXXX}, Pred::None, true);
        } else {
            LoadVec4(I2, specular_1);
        }
        if (refl_on) {
            LoadVec4(I1, acc);
            e.Vec(VecOp::Mul, {I2, 0x7}, {I2, XYZW}, {I1, XYZW}, Pred::None, true);
        }
        if (geo1) {
            e.Vec(VecOp::Mul, {I2, 0x7}, {I2, XYZW}, {atten, YYYY}, Pred::None, true);
        }
        if (d0_on) {
            StoreVec4(acc, I2);
        }
        // specular_0 = d0 * light.specular_0 (d0 from its LUT or 1), added on.
        if (d0_on) {
            lut_value(lighting.lut_d0, Sampler::Distribution0, two_sided);
            LoadVec4(I1, specular_0);
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {I0, XXXX}, Pred::None, true);
        } else {
            LoadVec4(I1, specular_0);
        }
        if (geo0) {
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {atten, YYYY}, Pred::None, true);
        }
        if (d0_on) {
            LoadVec4(I2, acc);
        }
        e.Vec(VecOp::Add, {I1, 0x7}, {I1, XYZW}, {I2, XYZW}, Pred::None, true);
        // * clamp_highlights * atten, into specular_sum.
        if (lighting.clamp_highlights) {
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {misc_hi, XXXX}, Pred::None, true);
        }
        if (atten_on) {
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {atten, XXXX}, Pred::None, true);
        }
        if (shadow_secondary) {
            LoadVec4(I2, shadow);
            e.Vec(VecOp::Mul, {I1, 0x7}, {I1, XYZW}, {I2, XYZW}, Pred::None, true);
        }
        if (specular_empty) {
            store3(specular_sum, I1);
            specular_empty = false;
        } else {
            LoadVec4(I2, specular_sum);
            e.Vec(VecOp::Add, {I2, 0x7}, {I2, XYZW}, {I1, XYZW}, Pred::None, true);
            store3(specular_sum, I2);
        }
        } // uses_secondary_color
        if (uses_primary_color) {
        // diffuse_sum += (diffuse * dot_product + ambient) * atten
        LoadVec4(I0, diffuse);
        LoadVec4(I1, ambient);
        e.Vec(VecOp::Mul, {I0, 0x7}, {I0, XYZW}, {dot_product, XXXX}, Pred::None, true);
        if (shadow_primary) {
            LoadVec4(I2, shadow);
            e.Vec(VecOp::Mul, {I0, 0x7}, {I0, XYZW}, {I2, XYZW}, Pred::None, true);
        }
        e.Vec(VecOp::Add, {I0, 0x7}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        if (atten_on) {
            e.Vec(VecOp::Mul, {I0, 0x7}, {I0, XYZW}, {atten, XXXX}, Pred::None, true);
        }
        if (diffuse_empty) {
            store3(diffuse_sum, I0);
            diffuse_empty = false;
        } else {
            LoadVec4(I2, diffuse_sum);
            e.Vec(VecOp::Add, {I2, 0x7}, {I2, XYZW}, {I0, XYZW}, Pred::None, true);
            store3(diffuse_sum, I2);
        }
        } // uses_primary_color

        // The fresnel term, on the last light, into the alphas asked for.
        const bool fresnel_primary = uses_primary_color && lighting.enable_primary_alpha;
        const bool fresnel_secondary = uses_secondary_color && lighting.enable_secondary_alpha;
        if (light_index + 1 == lighting.src_num && (fresnel_primary || fresnel_secondary) &&
            LightingLutEnabled(lighting.lut_fr, Sampler::Fresnel)) {
            lut_value(lighting.lut_fr, Sampler::Fresnel, two_sided);
            if (fresnel_primary) {
                e.MoveInternal({diffuse_sum.Pair(1), 0x2}, {I0, XXXX}, Pred::None, true);
            }
            if (fresnel_secondary) {
                e.MoveInternal({specular_sum.Pair(1), 0x2}, {I0, XXXX}, Pred::None, true);
            }
        }
    }

    if (uses_shadow && lighting.shadow_alpha) {
        const Vec4Loc shadow{Bank::Virtual, TmpShadow};
        if (lighting.enable_primary_alpha && uses_primary_color) {
            e.Vec(VecOp::Mul, {diffuse_sum.Pair(1), 0x2}, {diffuse_sum.Pair(1), XYZW},
                  {shadow.Pair(1), YYYY});
        }
        if (lighting.enable_secondary_alpha && uses_secondary_color) {
            e.Vec(VecOp::Mul, {specular_sum.Pair(1), 0x2}, {specular_sum.Pair(1), XYZW},
                  {shadow.Pair(1), YYYY});
        }
    }
    // diffuse_sum.rgb += global ambient; both clamped in place.
    if (uses_primary_color) {
        LoadVec4(I0, diffuse_sum);
        LoadVec4(I1, {Bank::SecAttr, static_cast<uint8_t>(sa_global_ambient)});
        e.Vec(VecOp::Add, {I0, 0x7}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        Clamp01(I0);
        StoreVec4(diffuse_sum, I0);
    }
    if (uses_secondary_color) {
        LoadVec4(I0, specular_sum);
        Clamp01(I0);
        StoreVec4(specular_sum, I0);
    }
}

void FragmentEmitter::PlanCombinerBuffer() {
    // The generator's dataflow: combiner_buffer starts at zero, next_combiner_buffer at the
    // uniform; after every stage, buffer = next, then next takes the stage's colour and/or
    // alpha when the stage updates the buffer. So stage k reads the last updating stage at
    // most k - 2 back, else the uniform, and stage 0 reads zero.
    int color_from = -1;
    int alpha_from = -1;
    for (u32 k = 0; k < 6; k++) {
        buffer_color_from[k] = k == 0 ? -2 : color_from;
        buffer_alpha_from[k] = k == 0 ? -2 : alpha_from;
        // What stage k's shuffle hands the stage after next.
        const u32 j = k >= 1 ? k - 1 : 0;
        if (k >= 1) {
            if (config.TevStageUpdatesCombinerBufferColor(j)) {
                color_from = static_cast<int>(j);
            }
            if (config.TevStageUpdatesCombinerBufferAlpha(j)) {
                alpha_from = static_cast<int>(j);
            }
        }
    }
    snapshot_lanes.fill(0);
    if (!buffer_used) {
        return;
    }
    for (u32 k = 0; k < 6; k++) {
        const TevStageConfig stage = config.texture.tev_stages[k];
        if (IsPassThroughTevStage(stage)) {
            continue;
        }
        const auto reads = [&](Source src, bool alpha) {
            if (src != Source::PreviousBuffer) {
                return;
            }
            const int from = alpha ? buffer_alpha_from[k] : buffer_color_from[k];
            if (from >= 0) {
                snapshot_lanes[from] |= alpha ? 0x8 : 0x7;
            }
        };
        reads(ColorSource(stage, stage.color_source1, k), false);
        reads(ColorSource(stage, stage.color_source2, k), false);
        reads(ColorSource(stage, stage.color_source3, k), false);
        reads(AlphaSource(stage, stage.alpha_source1, k), true);
        reads(AlphaSource(stage, stage.alpha_source2, k), true);
        reads(AlphaSource(stage, stage.alpha_source3, k), true);
    }
}

Vec4Loc FragmentEmitter::SourceLocation(Source source, u32 stage_index, bool alpha) const {
    switch (source) {
    case Source::PrimaryColor:
        return {Bank::Virtual, TmpPrimary};
    case Source::PrimaryFragmentColor:
        // Zero without lighting, as the generator has it.
        return uses_lighting ? Vec4Loc{Bank::Virtual, TmpDiffuse} : Vec4Loc{Bank::Const, 0};
    case Source::SecondaryFragmentColor:
        return uses_lighting ? Vec4Loc{Bank::Virtual, TmpSpecular} : Vec4Loc{Bank::Const, 0};
    case Source::Texture0:
        return {Bank::Virtual, TmpTex0};
    case Source::Texture1:
        return {Bank::Virtual, TmpTex0 + 4};
    case Source::Texture2:
        return {Bank::Virtual, TmpTex0 + 8};
    case Source::PreviousBuffer: {
        const int from = alpha ? buffer_alpha_from[stage_index] : buffer_color_from[stage_index];
        if (from == -2) {
            return {Bank::Const, 0};
        }
        if (from == -1) {
            return {Bank::SecAttr, static_cast<uint8_t>(sa_buffer_color)};
        }
        return {Bank::Virtual, static_cast<uint8_t>(TmpSnap + from * 4)};
    }
    case Source::Texture3:
        return uses_proctex ? Vec4Loc{Bank::Virtual, TmpTex3} : Vec4Loc{Bank::Const, 0};
    case Source::Constant:
        return {Bank::SecAttr, static_cast<uint8_t>(sa_const_color + stage_index * 4)};
    case Source::Previous:
    default:
        return {Bank::Virtual, TmpOutput};
    }
}

void FragmentEmitter::LoadArg(uint8_t arg_index, const TevStageConfig& stage, u32 stage_index,
                              Source csrc, ColorModifier cmod, Source asrc, AlphaModifier amod,
                              uint8_t lanes) {
    // An argument is built in the internal register itself: the colour source's quad in one
    // move (or one lane of it broadcast), the alpha source's lane moved into w when it
    // differs, then one add for the inverted lanes. Assembling it in temporaries first cost
    // three copies and a load per argument.
    const Reg dst = Reg::I(arg_index);
    const Vec4Loc c = SourceLocation(ColorSource(stage, csrc, stage_index), stage_index, false);
    const Vec4Loc a = SourceLocation(AlphaSource(stage, asrc, stage_index), stage_index, true);
    const bool want_c = lanes & 0x7;
    const bool want_a = lanes & 0x8;

    // A component of a float4 in the unified store: which pair, and which half.
    const auto channel = [](int comp) {
        return std::pair<int, Ch>{comp >> 1, (comp & 1) ? Ch::Y : Ch::X};
    };
    const auto same = [](const Vec4Loc& x, const Vec4Loc& y) {
        return x.bank == y.bank && x.base == y.base;
    };
    // dst.<mask> = one component of `from` broadcast.
    const auto lane = [&](uint8_t mask, const Vec4Loc& from, int comp) {
        if (from.bank == Bank::Const) {
            e.MoveInternal({dst, mask}, {Reg::C(0), XXXX}, Pred::None, true);
            return;
        }
        const auto [pair, ch] = channel(comp);
        e.MoveInternal({dst, mask}, {from.Pair(pair), Swizzle{ch, ch, ch, ch}}, Pred::None,
                       true);
    };

    const u32 cm = static_cast<u32>(cmod);
    const bool cinv = cm & 1;
    int broadcast = -1; // -1: rgb; otherwise the source component broadcast to rgb
    switch (cm & ~1u) {
    case 0x0: broadcast = -1; break; // SourceColor
    case 0x2: broadcast = 3; break;  // SourceAlpha
    case 0x4: broadcast = 0; break;  // SourceRed
    case 0x8: broadcast = 1; break;  // SourceGreen
    case 0xc: broadcast = 2; break;  // SourceBlue
    default: broadcast = -1; break;
    }
    const u32 am = static_cast<u32>(amod);
    const bool ainv = am & 1;
    const int acomp[4] = {3, 0, 1, 2}; // SourceAlpha, Red, Green, Blue
    const int alpha_comp = acomp[(am >> 1) & 3];

    bool alpha_done = false;
    if (want_c) {
        if (c.bank == Bank::Const) {
            e.MoveInternal({dst, 0x7}, {Reg::C(0), XXXX}, Pred::None, true);
        } else if (broadcast < 0) {
            // The whole quad; w is right already when the alpha comes from the same
            // source's own alpha.
            LoadVec4(dst, c);
            alpha_done = want_a && same(a, c) && alpha_comp == 3;
        } else {
            lane(0x7, c, broadcast);
        }
    }
    if (want_a && !alpha_done) {
        lane(0x8, a, alpha_comp);
    }
    const uint8_t inv = static_cast<uint8_t>((want_c && cinv ? 0x7 : 0) | (want_a && ainv ? 0x8 : 0));
    if (inv != 0) {
        e.Vec(VecOp::Add, {dst, inv}, {dst, XYZW, Mod::Neg}, ConstOne, Pred::None, true);
    }
}

void FragmentEmitter::ColorOp(Operation op, uint8_t mask) {
    // Arguments 1 and 2 are in i0 and i1, argument 3 (when the op has one) in i2 before
    // this is called; the result lands in i2 under `mask`.
    switch (op) {
    case Operation::Replace:
        e.MoveInternal({I2, mask}, {I0, XYZW}, Pred::None, true);
        break;
    case Operation::Modulate:
        e.Vec(VecOp::Mul, {I2, mask}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        break;
    case Operation::Add:
        e.Vec(VecOp::Add, {I2, mask}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        break;
    case Operation::AddSigned:
        e.Vec(VecOp::Add, {I2, mask}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        e.Vec(VecOp::Add, {I2, mask}, {Reg::C(12), XXXX, Mod::Neg}, {I2, XYZW}, Pred::None,
              true);
        break;
    case Operation::Subtract:
        e.Vec(VecOp::Add, {I2, mask}, {I1, XYZW, Mod::Neg}, {I0, XYZW}, Pred::None, true);
        break;
    case Operation::Lerp:
        // lerp(arg2, arg1, arg3) = arg2 + arg3 * (arg1 - arg2), arg3 waiting in i2.
        e.Vec(VecOp::Add, {I0, mask}, {I1, XYZW, Mod::Neg}, {I0, XYZW}, Pred::None, true);
        e.Mad({I2, mask}, {I0, XYZW}, {I2, XYZW}, {I1, XYZW}, true, Pred::None, true);
        break;
    case Operation::MultiplyThenAdd:
        e.Mad({I2, mask}, {I0, XYZW}, {I1, XYZW}, {I2, XYZW}, true, Pred::None, true);
        break;
    case Operation::AddThenMultiply:
        e.Vec(VecOp::Add, {I0, mask}, {I0, XYZW}, {I1, XYZW}, Pred::None, true);
        e.Vec(VecOp::Min, {I0, mask}, {I0, XYZW}, ConstOne, Pred::None, true);
        e.Vec(VecOp::Mul, {I2, mask}, {I0, XYZW}, {I2, XYZW}, Pred::None, true);
        break;
    case Operation::Dot3_RGB:
    case Operation::Dot3_RGBA:
        e.Vec(VecOp::Add, {I0, 0x7}, {Reg::C(12), XXXX, Mod::Neg}, {I0, XYZW}, Pred::None,
              true);
        e.Vec(VecOp::Add, {I1, 0x7}, {Reg::C(12), XXXX, Mod::Neg}, {I1, XYZW}, Pred::None,
              true);
        e.Dot({I2, 0x1}, {I0, XYZW}, {I1, XYZW}, false, Pred::None, true);
        e.Vec(VecOp::Mul, {I2, mask}, {I2, XXXX}, Lit(LitFour), Pred::None, true);
        break;
    default:
        e.MoveInternal({I2, mask}, {Reg::C(0), XXXX}, Pred::None, true);
        break;
    }
}

void FragmentEmitter::TevStage(u32 index) {
    const TevStageConfig stage = config.texture.tev_stages[index];
    if (!IsPassThroughTevStage(stage)) {
        const bool dot3_alpha = stage.color_op == Operation::Dot3_RGBA;
        const bool same_op = stage.color_op == stage.alpha_op && !dot3_alpha &&
                             stage.color_op != Operation::Dot3_RGB;
        const auto needs_three = [](Operation op) {
            return op == Operation::Lerp || op == Operation::MultiplyThenAdd ||
                   op == Operation::AddThenMultiply;
        };
        // Whether the colour operation leaves i0 and i1 as loaded (only i2 written), so
        // the alpha operation can use their w lanes without loading them again.
        const auto keeps_args = [](Operation op) {
            return op == Operation::Replace || op == Operation::Modulate ||
                   op == Operation::Add || op == Operation::AddSigned ||
                   op == Operation::Subtract || op == Operation::MultiplyThenAdd;
        };
        const bool separate_alpha = !same_op && !dot3_alpha;
        const bool alpha_three = separate_alpha && needs_three(stage.alpha_op);
        // Arguments 1 and 2 carry their alpha from the start when the colour operation
        // keeps them; argument 3's alpha only when the alpha operation is the same.
        const bool alpha_early = !separate_alpha || (keeps_args(stage.color_op) && !alpha_three);
        const uint8_t lanes01 = alpha_early ? 0xf : 0x7;
        const auto load = [&](uint8_t arg, uint8_t lanes) {
            switch (arg) {
            case 0:
                LoadArg(0, stage, index, stage.color_source1, stage.color_modifier1,
                        stage.alpha_source1, stage.alpha_modifier1, lanes);
                break;
            case 1:
                LoadArg(1, stage, index, stage.color_source2, stage.color_modifier2,
                        stage.alpha_source2, stage.alpha_modifier2, lanes);
                break;
            default:
                LoadArg(2, stage, index, stage.color_source3, stage.color_modifier3,
                        stage.alpha_source3, stage.alpha_modifier3, lanes);
                break;
            }
        };
        load(0, lanes01);
        load(1, lanes01);
        if (needs_three(stage.color_op)) {
            load(2, same_op ? 0xf : 0x7);
        }
        ColorOp(stage.color_op, same_op || dot3_alpha ? 0xf : 0x7);
        if (separate_alpha) {
            if (alpha_three) {
                // i2's colour result must survive: park it, load the alphas, restore.
                StoreVec4({Bank::Virtual, TmpOutput}, I2);
                load(0, 0x8);
                load(1, 0x8);
                load(2, 0x8);
                ColorOp(stage.alpha_op, 0x8);
                e.MoveInternal({I0, 0xf}, {I2, XYZW}, Pred::None, true);
                LoadVec4(I2, {Bank::Virtual, TmpOutput});
                e.MoveInternal({I2, 0x8}, {I0, XYZW}, Pred::None, true);
            } else {
                if (!alpha_early) {
                    load(0, 0x8);
                    load(1, 0x8);
                }
                ColorOp(stage.alpha_op, 0x8);
            }
        }
        Clamp01(I2);
        Byteround(I2, I0);
        const unsigned cscale = stage.GetColorMultiplier();
        const unsigned ascale = stage.GetAlphaMultiplier();
        if (cscale != 1) {
            e.Vec(VecOp::Mul, {I2, 0x7}, {I2, XYZW}, Lit(cscale == 2 ? LitTwo : LitFour),
                  Pred::None, true);
        }
        if (ascale != 1) {
            e.Vec(VecOp::Mul, {I2, 0x8}, {I2, XYZW}, Lit(ascale == 2 ? LitTwo : LitFour),
                  Pred::None, true);
        }
        if (cscale != 1 || ascale != 1) {
            Clamp01(I2);
        }
        StoreVec4({Bank::Virtual, TmpOutput}, I2);
    }
    if (snapshot_lanes[index] != 0) {
        // A later stage reads this output through PreviousBuffer.
        CopyVec4({Bank::Virtual, static_cast<uint8_t>(TmpSnap + index * 4)},
                 {Bank::Virtual, TmpOutput}, snapshot_lanes[index]);
    }
}

void FragmentEmitter::AlphaTest() {
    const auto func = config.framebuffer.alpha_test_func.Value();
    if (func == FramebufferRegs::CompareFunc::Always) {
        return;
    }
    if (func == FramebufferRegs::CompareFunc::Never) {
        e.Test(Cond::Gt, 1, ConstZero, 0, ConstZero, false);
        if (uses_depth) {
            WriteDepth();
        }
        e.KillUnlessP1();
        NewPhaseAfterKill();
        return;
    }
    // floor(alpha * 255) against the reference, kept when the PICA function holds.
    e.Vec(VecOp::Mul, {I0, 0x1}, {Reg::V(TmpOutput + 2), YYYY}, Lit(LitTwoFiftyFive),
          Pred::None, true);
    e.Vec(VecOp::Frc, {I1, 0x1}, {I0, XYZW}, {I0, XYZW}, Pred::None, true);
    e.Vec(VecOp::Add, {I0, 0x1}, {I1, XYZW, Mod::Neg}, {I0, XYZW}, Pred::None, true);
    Cond cond = Cond::Ge;
    switch (func) {
    case FramebufferRegs::CompareFunc::Equal: cond = Cond::Eq; break;
    case FramebufferRegs::CompareFunc::NotEqual: cond = Cond::Ne; break;
    case FramebufferRegs::CompareFunc::LessThan: cond = Cond::Lt; break;
    case FramebufferRegs::CompareFunc::LessThanOrEqual: cond = Cond::Le; break;
    case FramebufferRegs::CompareFunc::GreaterThan: cond = Cond::Gt; break;
    case FramebufferRegs::CompareFunc::GreaterThanOrEqual: cond = Cond::Ge; break;
    default: break;
    }
    e.Test(cond, 1, {I0}, 0, SaScalar(sa_alpha_ref), false);
    if (uses_depth) {
        WriteDepth();
    }
    if (fold_scissor) {
        // p1 = p1 && the scissor test kept the pixel (Scissor left its minimum in the
        // scratch PA).
        const bool include = config.framebuffer.scissor_test_mode.Value() ==
                             Pica::RasterizerRegs::ScissorMode::Include;
        e.Test(include ? Cond::Ge : Cond::Lt, 1, {Reg::Pa(scissor_pa)}, 0, ConstZero, false,
               Pred::P1);
    }
    e.KillUnlessP1();
    NewPhaseAfterKill();
}

std::vector<uint8_t> FragmentEmitter::Emit(std::string* refusal) {
    if (const char* why = Unsupported()) {
        if (refusal) {
            *refusal = why;
        }
        return {};
    }
    PlanInputs();
    PlanUniforms();
    uses_kill = config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Always ||
                uses_scissor;

    gxp.literals = {0xE000u, // the kill's control word, as every compiler program has it
                    0u,
                    std::bit_cast<uint32_t>(255.0f),
                    std::bit_cast<uint32_t>(1.0f / 255.0f),
                    std::bit_cast<uint32_t>(2.0f),
                    std::bit_cast<uint32_t>(4.0f),
                    std::bit_cast<uint32_t>(128.0f),
                    std::bit_cast<uint32_t>(127.0f),
                    std::bit_cast<uint32_t>(1.0f / 256.0f),
                    std::bit_cast<uint32_t>(1.0f / static_cast<float>(Pica::Shader::LUT_LF_ROWS)),
                    std::bit_cast<uint32_t>(0.5f / 256.0f),
                    std::bit_cast<uint32_t>(256.0f),
                    std::bit_cast<uint32_t>(255.0f),
                    std::bit_cast<uint32_t>(-128.0f),
                    std::bit_cast<uint32_t>(1.0f / static_cast<float>(Pica::Shader::LUT_RG_ROWS)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_width)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_width - 1)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_offset0)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_offset1)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_offset2)),
                    std::bit_cast<uint32_t>(static_cast<float>(config.proctex.lut_offset3)),
                    std::bit_cast<uint32_t>(240.0f),
                    std::bit_cast<uint32_t>(248.0f),
                    std::bit_cast<uint32_t>(252.0f),
                    std::bit_cast<uint32_t>(254.0f),
                    std::bit_cast<uint32_t>(std::max(0.0f, static_cast<float>(config.proctex.lod_min))),
                    std::bit_cast<uint32_t>(std::min(7.0f, static_cast<float>(config.proctex.lod_max))),
                    0u,
                    std::bit_cast<uint32_t>(2.0f),
                    std::bit_cast<uint32_t>(-2.0f),
                    std::bit_cast<uint32_t>(1.0f),
                    std::bit_cast<uint32_t>(1.0f / static_cast<float>(Pica::Shader::LUT_LF_ROWS)),
                    std::bit_cast<uint32_t>(0.5f / 256.0f),
                    std::bit_cast<uint32_t>(0.5f / static_cast<float>(Pica::Shader::LUT_LF_ROWS)),
                    std::bit_cast<uint32_t>(1.0f),
                    std::bit_cast<uint32_t>(1.0f / static_cast<float>(Pica::Shader::LUT_RG_ROWS)),
                    std::bit_cast<uint32_t>(0.5f / 256.0f),
                    std::bit_cast<uint32_t>(0.5f / static_cast<float>(Pica::Shader::LUT_RG_ROWS)),
                    std::bit_cast<uint32_t>(-127.0f),
                    std::bit_cast<uint32_t>(-255.0f)};
    static_assert(LiteralCount == 40);
    static_assert(Pica::Shader::LUT_RG_ROWS == Pica::Shader::LUT_RGBA_ROWS);
    gxp.uses_discard = uses_kill;
    gxp.writes_depth = uses_depth;
    if (uniform_floats + LiteralCount + gxp.samplers.size() * 4 > 128) {
        if (refusal) {
            *refusal = "secondary attribute budget";
        }
        return {};
    }

    // Console findings, 2026-09-04, on the sparse per-pixel noise every program with a
    // texture read drew: the phase marker must be followed by a NOP, and the nosched bits
    // the builders chose were wrong (they marked the producer of an internal register,
    // while the hardware wants the instruction before it: the bit guards the boundary
    // after the next instruction). Setting the bit everywhere cured the noise and cost
    // the GPU its instance switching; FinishNosched now places it as the compiler does.
    // skipinv made no difference on the console and stays clear, as the ubershader has it.
    e.nosched_override = 0;
    e.skip_invalid = false;

    // A kill ends its phase: psp2cgc follows every kill with its nop and then a new phase
    // marker (all 148 of SM3DL's programs, whose scissor kill comes first: phase one is the
    // scissor test alone, phase two everything else). The first marker waits with
    // condition 1 when another phase follows, 7 otherwise, as the compiler's do. Before
    // this the emitter kept sampling and shading in the kill's phase and opened its second
    // phase only before the output pack; Vita3K did not mind, the console drew noise.
    e.Phas(!(uses_kill || uses_depth));
    e.Nop();
    Scissor();
    SampleTextures();
    if (uses_proctex) {
        ProcTex();
    }
    RoundPrimary();
    // The texture unit 0 colour is zero when the unit is disabled and a stage reads it.
    if (!unit_used[0] && config.texture.texture0_type == TextureType::Disabled) {
        CopyVec4({Bank::Virtual, TmpTex0}, {Bank::Const, 0});
    }
    if (uses_lighting) {
        Lighting();
    }
    // combiner_output starts at zero; the buffers at zero and the register value.
    e.Vec(VecOp::Add, {Reg::V(TmpOutput), 0x3}, ConstZero, ConstZero);
    e.Vec(VecOp::Add, {Reg::V(TmpOutput + 2), 0x3}, ConstZero, ConstZero);
    PlanCombinerBuffer();
    for (u32 i = 0; i < config.texture.tev_stages.size(); i++) {
        TevStage(i);
    }
    AlphaTest();
    if (uses_depth && !uses_kill) {
        // No kill to share the phase end with: the depth write ends phase one by itself.
        WriteDepth();
        NewPhaseAfterKill();
    }
    Fog();
    e.PackF16({Reg::Pa(0), 0xf}, Reg::V(TmpOutput), Reg::V(TmpOutput + 2));

    // The quads go to the temporaries, then to primary attributes past the interpolants.
    e.AddClass({Bank::Temp, 0, 64});
    e.AddClass({Bank::PrimAttr, static_cast<uint8_t>((gxp.pa_count + 3) & ~3u), 64});
    Encoder encoder;
    const Ir::LowerResult lowered = e.Lower(encoder, level);
    encoder.PatchKills(LiteralSaOffset(uniform_floats + data_head, LitKillControl));
    encoder.FinishNosched();
    gxp.temp_count = lowered.temp_count;
    gxp.pa_count = std::max<uint32_t>(gxp.pa_count, lowered.pa_count);
    gxp.phase_starts = lowered.phase_starts;
    gxp.primary = encoder.Words();
    return WriteGxp(gxp);
}

} // Anonymous namespace

std::vector<uint8_t> EmitFragmentProgram(const FSConfig& config, std::string* refusal, int level) {
    try {
        FragmentEmitter emitter{config, level};
        return emitter.Emit(refusal);
    } catch (const std::exception& ex) {
        // An operand form the encoder cannot express: the compiler takes the program.
        if (refusal) {
            *refusal = ex.what();
        }
        return {};
    }
}

} // namespace GxmRenderer::Usse
