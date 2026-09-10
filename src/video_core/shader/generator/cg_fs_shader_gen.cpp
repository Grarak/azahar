// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdio>

#include "video_core/shader/generator/cg_fs_shader_gen.h"

namespace Pica::Shader::Generator::Cg {

using ProcTexClamp = TexturingRegs::ProcTexClamp;
using ProcTexShift = TexturingRegs::ProcTexShift;
using ProcTexCombiner = TexturingRegs::ProcTexCombiner;
using ProcTexFilter = TexturingRegs::ProcTexFilter;
using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

namespace {

bool IsPassThroughTevStage(const Pica::TexturingRegs::TevStageConfig& stage) {
    using TevStageConfig = Pica::TexturingRegs::TevStageConfig;
    return (stage.color_op == TevStageConfig::Operation::Replace &&
            stage.alpha_op == TevStageConfig::Operation::Replace &&
            stage.color_source1 == TevStageConfig::Source::Previous &&
            stage.alpha_source1 == TevStageConfig::Source::Previous &&
            stage.color_modifier1 == TevStageConfig::ColorModifier::SourceColor &&
            stage.alpha_modifier1 == TevStageConfig::AlphaModifier::SourceAlpha &&
            stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1);
}

// Uniforms mirror fs_data in the GLSL generator's uniform block, as flat globals: GXM has no
// uniform blocks, values land here by name through sceGxmSetUniformDataF. LUT geometry is the
// shared LUT_TEX_WIDTH x LUT_*_ROWS layout from profile.h, baked into the lut helpers below.
constexpr std::string_view FSUniformDefs = R"(
#define NUM_TEV_STAGES 6
#define NUM_LIGHTS 8
#define NUM_LIGHTING_SAMPLERS 24
struct LightSrc {
    float3 specular_0;
    float3 specular_1;
    float3 diffuse;
    float3 ambient;
    float3 position;
    float3 spot_direction;
    float dist_atten_bias;
    float dist_atten_scale;
};
// These are whole numbers, but they are declared float because that is the only type they
// are ever used as: every one of them ends up in a comparison against an interpolated value
// or in a texture-buffer coordinate. psp2cgc warns that "full precision conversions between
// 32bit floating point and 32bit integer values are very expensive" - so the way to make
// them cheap is not to narrow the integer but to never have one. The host writes them
// through sceGxmSetUniformDataF, which converts to the parameter's declared type, so
// nothing changes on that side.
uniform float alphatest_ref;
uniform float depth_scale;
uniform float depth_offset;
uniform float scissor_x1;
uniform float scissor_y1;
uniform float scissor_x2;
uniform float scissor_y2;
uniform float fog_lut_offset;
uniform float proctex_noise_lut_offset;
uniform float proctex_color_map_offset;
uniform float proctex_alpha_map_offset;
uniform float proctex_lut_offset;
uniform float proctex_diff_lut_offset;
uniform float proctex_bias;
uniform float4 lighting_lut_offset[NUM_LIGHTING_SAMPLERS / 4];
uniform float3 fog_color;
uniform float2 proctex_noise_f;
uniform float2 proctex_noise_a;
uniform float2 proctex_noise_p;
uniform float3 lighting_global_ambient;
uniform LightSrc light_src[NUM_LIGHTS];
uniform float4 const_color[NUM_TEV_STAGES];
uniform float4 tev_combiner_buffer_color;
uniform float3 tex_lod_bias;
uniform float4 tex_border_color[3];
uniform float4 blend_color;
uniform float2 tex_dims[3];
)";

// The interpolated inputs, copied into statics so every helper function can reach them the
// way the GLSL functions reach their global varyings.
constexpr std::string_view StaticInputs = R"(
static float4 primary_color;
static float2 texcoord0;
static float2 texcoord1;
static float2 texcoord2;
static float pica_texcoord0_w;
static float4 normquat;
static float4 normquat_flat;
static float3 view;
static float4 frag_coord;
static float4 color;
static float frag_depth;
)";

} // Anonymous namespace

Profile MakeGxmProfile() {
    Profile profile{};
    profile.has_texture_buffer = 0;
    profile.has_texel_fetch = 0;
    profile.has_minus_one_to_one_range = 0;
    profile.has_logic_op = 0;
    profile.is_vulkan = 0;
    return profile;
}

class FragmentModule {
public:
    explicit FragmentModule(const FSConfig& config_, const UserConfig& user_,
                            const Profile& profile_)
        : config{config_}, user{user_}, profile{profile_} {
        config.ApplyProfile(profile_);
        out.reserve(32 * 1024);
        out += FSUniformDefs;
        out += StaticInputs;
        DefineSamplers();
        DefineHelpers();
        DefineLightingHelpers();
        DefineProcTexSampler();
        for (u32 i = 0; i < 4; i++) {
            DefineTexUnitSampler(i);
        }
    }

    std::string Generate();

private:
    void WriteDepth();
    void WriteScissor();
    void WriteLighting();
    void WriteFog();
    void WriteGas();
    void WriteLogicOp();
    void WriteBlending();
    void WriteAlphaTestCondition(FramebufferRegs::CompareFunc func);
    void WriteTevStage(u32 index);
    std::string GetSource(Pica::TexturingRegs::TevStageConfig::Source source, u32 tev_index);
    void AppendColorModifier(Pica::TexturingRegs::TevStageConfig::ColorModifier modifier,
                             Pica::TexturingRegs::TevStageConfig::Source source, u32 tev_index);
    void AppendAlphaModifier(Pica::TexturingRegs::TevStageConfig::AlphaModifier modifier,
                             Pica::TexturingRegs::TevStageConfig::Source source, u32 tev_index);
    void AppendColorCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation);
    void AppendAlphaCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation);
    void AppendProcTexShiftOffset(std::string_view v, ProcTexShift mode, ProcTexClamp clamp_mode);
    void AppendProcTexClamp(std::string_view var, ProcTexClamp mode);
    void AppendProcTexCombineAndMap(ProcTexCombiner combiner, std::string_view offset);
    void DefineSamplers();
    void DefineHelpers();
    void DefineLightingHelpers();
    void DefineProcTexSampler();
    void DefineTexUnitSampler(u32 texture_unit);

    FSConfig config;
    UserConfig user;
    Profile profile;
    std::string out;
    bool use_blend_fallback{};
};

std::string FragmentModule::Generate() {
    // The body lives in a helper so 'discard' and early 'return' read exactly as they do in
    // the GLSL generator's main; the real main copies the interpolants into the statics and
    // hands the results to the output semantics.
    out += R"(
void pica_frag_main() {
float4 rounded_primary_color = byteround(primary_color);
float4 primary_fragment_color = float4(0.0);
float4 debug_tev = float4(0.0);
float4 secondary_fragment_color = float4(0.0);
)";

    const bool early_discard =
        config.framebuffer.alpha_test_func == FramebufferRegs::CompareFunc::Never;
    if (early_discard) {
        out += "discard; }";
    } else {
        WriteScissor();
        WriteDepth();
        WriteLighting();

        out += "float4 combiner_buffer = float4(0.0);\n"
               "float4 next_combiner_buffer = tev_combiner_buffer_color;\n"
               "float4 combiner_output = float4(0.0);\n";

        out += "float3 color_results_1 = float3(0.0);\n"
               "float3 color_results_2 = float3(0.0);\n"
               "float3 color_results_3 = float3(0.0);\n";

        out += "float alpha_results_1 = 0.0;\n"
               "float alpha_results_2 = 0.0;\n"
               "float alpha_results_3 = 0.0;\n";

        for (u32 index = 0; index < config.texture.tev_stages.size(); index++) {
            WriteTevStage(index);
        }

        WriteAlphaTestCondition(config.framebuffer.alpha_test_func);

        bool closed = false;
        switch (config.texture.fog_mode) {
        case TexturingRegs::FogMode::Fog:
            if (!FogDisabled()) {
                WriteFog();
            }
            break;
        case TexturingRegs::FogMode::Gas:
            WriteGas();
            closed = true;
            break;
        default:
            break;
        }

        if (!closed) {
            if (config.framebuffer.shadow_rendering) {
                // GXM has no storage images; shadow-map rendering cannot be expressed.
                out += "// UNSUPPORTED-GXM: shadow_rendering\n"
                       "color = float4(0.0);\n";
            } else {
                out += "frag_depth = depth;\n";
                out += "combiner_output = byteround(combiner_output);\n";
                WriteBlending();
                out += "color = combiner_output;\n";
                switch (GetDebugOutput()) {
                case DebugOutput::VertexColor:
                    out += "color = rounded_primary_color;\n";
                    break;
                case DebugOutput::Texture0:
                    out += "color = sampleTexUnit0();\n";
                    break;
                case DebugOutput::Lighting:
                    out += "color = float4(primary_fragment_color.rgb, 1.0);\n";
                    break;
                case DebugOutput::Normal:
                    out += "color = float4(normquat.xyz * 0.5 + 0.5, 1.0);\n";
                    break;
                case DebugOutput::View:
                    out += "color = float4(view * 0.5 + 0.5, 1.0);\n";
                    break;
                case DebugOutput::Texcoord1:
                    out += "color = float4(frac(texcoord1.x), frac(texcoord1.y), 0.0, 1.0);\n";
                    break;
                case DebugOutput::Texcoord2:
                    out += "color = float4(frac(texcoord2.x), frac(texcoord2.y), 0.0, 1.0);\n";
                    break;
                case DebugOutput::Alpha:
                    out += "color = float4(combiner_output.aaa, 1.0);\n";
                    break;
                case DebugOutput::Texcoord0:
                    // The interpolated coordinate itself, wrapped so it stays visible outside
                    // [0,1]: a picture that is sheared here is sheared in what the vertex
                    // program wrote or in how it was interpolated, not in the texture.
                    out += "color = float4(frac(texcoord0.x), frac(texcoord0.y), 0.0, 1.0);\n";
                    break;
                default:
                    if (GetDebugOutput() >= DebugOutput::TevStage0) {
                        out += "color = debug_tev;\n";
                    }
                    break;
                }
            }
            WriteLogicOp();
            out += '}';
        }
    }

    out += R"(

float4 main(
    float4 in_primary_color : COLOR0,
    float2 in_texcoord0 : TEXCOORD0,
    float2 in_texcoord1 : TEXCOORD1,
    float2 in_texcoord2 : TEXCOORD2,
    float in_texcoord0_w : TEXCOORD3,
    float4 in_normquat : TEXCOORD4,
    float4 in_normquat_flat : TEXCOORD5,
    float3 in_view : TEXCOORD6,
    float4 in_wpos : WPOS)";
    const bool writes_depth =
        config.framebuffer.depthmap_enable == RasterizerRegs::DepthBuffering::WBuffering;
    if (writes_depth) {
        out += ",\n    out float out_depth : DEPTH";
    }
    out += R"() : COLOR {
    primary_color = in_primary_color;
    texcoord0 = in_texcoord0;
    texcoord1 = in_texcoord1;
    texcoord2 = in_texcoord2;
    pica_texcoord0_w = in_texcoord0_w;
    normquat = in_normquat;
    normquat_flat = in_normquat_flat;
    view = in_view;
    frag_coord = in_wpos;
    color = float4(0.0);
    frag_depth = frag_coord.z;
    pica_frag_main();
)";
    if (writes_depth) {
        out += "    out_depth = frag_depth;\n";
    }
    out += R"(    return color;
}
)";
    return out;
}

void FragmentModule::WriteDepth() {
    // The vertex program has already applied PICA's depth mapping (z/w * depth_scale +
    // depth_offset) to clip z, so the hardware's interpolated window z is the PICA depth and
    // nothing here has to replace it - which keeps the tile-based GPU's early depth test
    // alive. Only W-buffering needs a per-fragment value, and only then is DEPTH written.
    // (Console-verified 2026-09-02: deriving depth from WPOS.z and writing it back drew
    // nothing against citro3d's GREATER test.)
    out += "float depth = frag_coord.z;\n";
    if (config.framebuffer.depthmap_enable == RasterizerRegs::DepthBuffering::WBuffering) {
        out += "depth /= frag_coord.w;\n";
    }
}

void FragmentModule::WriteScissor() {
    const auto scissor_mode = config.framebuffer.scissor_test_mode.Value();
    if (scissor_mode == RasterizerRegs::ScissorMode::Disabled) {
        return;
    }

    out += "if (";
    if (scissor_mode == RasterizerRegs::ScissorMode::Include) {
        out += '!';
    }
    out += "(frag_coord.x >= scissor_x1 && "
           "frag_coord.y >= scissor_y1 && "
           "frag_coord.x < scissor_x2 && "
           "frag_coord.y < scissor_y2)) discard;\n";
}

std::string FragmentModule::GetSource(Pica::TexturingRegs::TevStageConfig::Source source,
                                      u32 tev_index) {
    using Source = Pica::TexturingRegs::TevStageConfig::Source;
    switch (source) {
    case Source::PrimaryColor:
        return "rounded_primary_color";
    case Source::PrimaryFragmentColor:
        return "primary_fragment_color";
    case Source::SecondaryFragmentColor:
        return "secondary_fragment_color";
    case Source::Texture0:
        return "sampleTexUnit0()";
    case Source::Texture1:
        return "sampleTexUnit1()";
    case Source::Texture2:
        return "sampleTexUnit2()";
    case Source::Texture3:
        return "sampleTexUnit3()";
    case Source::PreviousBuffer:
        return "combiner_buffer";
    case Source::Constant:
        return fmt::format("const_color[{}]", tev_index);
    case Source::Previous:
        return "combiner_output";
    default:
        LOG_CRITICAL(Render, "Unknown source op {}", source);
        return "float4(0.0)";
    }
}

void FragmentModule::AppendColorModifier(
    Pica::TexturingRegs::TevStageConfig::ColorModifier modifier,
    Pica::TexturingRegs::TevStageConfig::Source source, u32 tev_index) {
    using Source = Pica::TexturingRegs::TevStageConfig::Source;
    using ColorModifier = Pica::TexturingRegs::TevStageConfig::ColorModifier;
    const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[tev_index];
    const bool force_source3 = tev_index == 0 && source == Source::Previous;
    const auto color_source =
        GetSource(force_source3 ? stage.color_source3.Value() : source, tev_index);
    switch (modifier) {
    case ColorModifier::SourceColor:
        out += fmt::format("{}.rgb", color_source);
        break;
    case ColorModifier::OneMinusSourceColor:
        out += fmt::format("float3(1.0) - {}.rgb", color_source);
        break;
    case ColorModifier::SourceAlpha:
        out += fmt::format("{}.aaa", color_source);
        break;
    case ColorModifier::OneMinusSourceAlpha:
        out += fmt::format("float3(1.0) - {}.aaa", color_source);
        break;
    case ColorModifier::SourceRed:
        out += fmt::format("{}.rrr", color_source);
        break;
    case ColorModifier::OneMinusSourceRed:
        out += fmt::format("float3(1.0) - {}.rrr", color_source);
        break;
    case ColorModifier::SourceGreen:
        out += fmt::format("{}.ggg", color_source);
        break;
    case ColorModifier::OneMinusSourceGreen:
        out += fmt::format("float3(1.0) - {}.ggg", color_source);
        break;
    case ColorModifier::SourceBlue:
        out += fmt::format("{}.bbb", color_source);
        break;
    case ColorModifier::OneMinusSourceBlue:
        out += fmt::format("float3(1.0) - {}.bbb", color_source);
        break;
    default:
        out += "float3(0.0)";
        LOG_CRITICAL(Render, "Unknown color modifier op {}", modifier);
        break;
    }
}

void FragmentModule::AppendAlphaModifier(
    Pica::TexturingRegs::TevStageConfig::AlphaModifier modifier,
    Pica::TexturingRegs::TevStageConfig::Source source, u32 tev_index) {
    using Source = Pica::TexturingRegs::TevStageConfig::Source;
    using AlphaModifier = Pica::TexturingRegs::TevStageConfig::AlphaModifier;
    const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[tev_index];
    const bool force_source3 = tev_index == 0 && source == Source::Previous;
    const auto alpha_source =
        GetSource(force_source3 ? stage.alpha_source3.Value() : source, tev_index);
    switch (modifier) {
    case AlphaModifier::SourceAlpha:
        out += fmt::format("{}.a", alpha_source);
        break;
    case AlphaModifier::OneMinusSourceAlpha:
        out += fmt::format("1.0 - {}.a", alpha_source);
        break;
    case AlphaModifier::SourceRed:
        out += fmt::format("{}.r", alpha_source);
        break;
    case AlphaModifier::OneMinusSourceRed:
        out += fmt::format("1.0 - {}.r", alpha_source);
        break;
    case AlphaModifier::SourceGreen:
        out += fmt::format("{}.g", alpha_source);
        break;
    case AlphaModifier::OneMinusSourceGreen:
        out += fmt::format("1.0 - {}.g", alpha_source);
        break;
    case AlphaModifier::SourceBlue:
        out += fmt::format("{}.b", alpha_source);
        break;
    case AlphaModifier::OneMinusSourceBlue:
        out += fmt::format("1.0 - {}.b", alpha_source);
        break;
    default:
        out += "0.0";
        LOG_CRITICAL(Render, "Unknown alpha modifier op {}", modifier);
        break;
    }
}

void FragmentModule::AppendColorCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation) {
    const auto get_combiner = [operation] {
        using Operation = Pica::TexturingRegs::TevStageConfig::Operation;
        switch (operation) {
        case Operation::Replace:
            return "color_results_1";
        case Operation::Modulate:
            return "color_results_1 * color_results_2";
        case Operation::Add:
            return "color_results_1 + color_results_2";
        case Operation::AddSigned:
            return "color_results_1 + color_results_2 - float3(0.5)";
        case Operation::Lerp:
            return "lerp(color_results_2, color_results_1, color_results_3)";
        case Operation::Subtract:
            return "color_results_1 - color_results_2";
        case Operation::MultiplyThenAdd:
            return "color_results_1 * color_results_2 + color_results_3";
        case Operation::AddThenMultiply:
            return "min(color_results_1 + color_results_2, float3(1.0)) * color_results_3";
        case Operation::Dot3_RGB:
        case Operation::Dot3_RGBA:
            return "float3(dot(color_results_1 - float3(0.5), color_results_2 - float3(0.5)) "
                   "* 4.0)";
        default:
            LOG_CRITICAL(Render, "Unknown color combiner operation: {}", operation);
            return "float3(0.0)";
        }
    };
    out += fmt::format("clamp({}, float3(0.0), float3(1.0))", get_combiner());
}

void FragmentModule::AppendAlphaCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation) {
    const auto get_combiner = [operation] {
        using Operation = Pica::TexturingRegs::TevStageConfig::Operation;
        switch (operation) {
        case Operation::Replace:
            return "alpha_results_1";
        case Operation::Modulate:
            return "alpha_results_1 * alpha_results_2";
        case Operation::Add:
            return "alpha_results_1 + alpha_results_2";
        case Operation::AddSigned:
            return "alpha_results_1 + alpha_results_2 - 0.5";
        case Operation::Lerp:
            return "lerp(alpha_results_2, alpha_results_1, alpha_results_3)";
        case Operation::Subtract:
            return "alpha_results_1 - alpha_results_2";
        case Operation::MultiplyThenAdd:
            return "alpha_results_1 * alpha_results_2 + alpha_results_3";
        case Operation::AddThenMultiply:
            return "min(alpha_results_1 + alpha_results_2, 1.0) * alpha_results_3";
        default:
            LOG_CRITICAL(Render, "Unknown alpha combiner operation: {}", operation);
            return "0.0";
        }
    };
    out += fmt::format("clamp({}, 0.0, 1.0)", get_combiner());
}

void FragmentModule::WriteAlphaTestCondition(FramebufferRegs::CompareFunc func) {
    const auto get_cond = [func]() -> std::string {
        using CompareFunc = Pica::FramebufferRegs::CompareFunc;
        switch (func) {
        case CompareFunc::Never:
            return "true";
        case CompareFunc::Always:
            return "false";
        case CompareFunc::Equal:
        case CompareFunc::NotEqual:
        case CompareFunc::LessThan:
        case CompareFunc::LessThanOrEqual:
        case CompareFunc::GreaterThan:
        case CompareFunc::GreaterThanOrEqual: {
            static constexpr std::array op{"!=", "==", ">=", ">", "<=", "<"};
            const auto index = static_cast<u32>(func) - static_cast<u32>(CompareFunc::Equal);
            // combiner_output.a is clamped to [0, 1], so the PICA's truncation to an 8-bit
            // integer is floor(), and both sides are then whole numbers a float holds
            // exactly - the comparison, equality included, is the same one.
            return fmt::format("floor(combiner_output.a * 255.0) {} alphatest_ref", op[index]);
        }
        default:
            LOG_CRITICAL(Render, "Unknown alpha test condition {}", func);
            return "false";
        }
    };
    out += fmt::format("if ({}) discard;\n", get_cond());
}

void FragmentModule::WriteTevStage(u32 index) {
    const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[index];
    if (!IsPassThroughTevStage(stage)) {
        out += "color_results_1 = ";
        AppendColorModifier(stage.color_modifier1, stage.color_source1, index);
        out += ";\ncolor_results_2 = ";
        AppendColorModifier(stage.color_modifier2, stage.color_source2, index);
        out += ";\ncolor_results_3 = ";
        AppendColorModifier(stage.color_modifier3, stage.color_source3, index);

        out += fmt::format(";\nfloat3 color_output_{} = byteround(", index);
        AppendColorCombiner(stage.color_op);
        out += ");\n";

        if (stage.color_op == Pica::TexturingRegs::TevStageConfig::Operation::Dot3_RGBA) {
            out += fmt::format("float alpha_output_{0} = color_output_{0}[0];\n", index);
        } else {
            out += "alpha_results_1 = ";
            AppendAlphaModifier(stage.alpha_modifier1, stage.alpha_source1, index);
            out += ";\nalpha_results_2 = ";
            AppendAlphaModifier(stage.alpha_modifier2, stage.alpha_source2, index);
            out += ";\nalpha_results_3 = ";
            AppendAlphaModifier(stage.alpha_modifier3, stage.alpha_source3, index);

            out += fmt::format(";\nfloat alpha_output_{} = byteround(", index);
            AppendAlphaCombiner(stage.alpha_op);
            out += ");\n";
        }

        out += fmt::format("combiner_output = float4("
                           "clamp(color_output_{} * {}.0, float3(0.0), float3(1.0)), "
                           "clamp(alpha_output_{} * {}.0, 0.0, 1.0));\n",
                           index, stage.GetColorMultiplier(), index, stage.GetAlphaMultiplier());
        // Bring-up switch: keep what this stage produced, so the stage that first goes wrong
        // can be named. Later stages still run and still write their own results; only what
        // reaches the screen is pinned here.
        const u32 debug_stage = static_cast<u32>(GetDebugOutput()) -
                                static_cast<u32>(DebugOutput::TevStage0);
        if (GetDebugOutput() >= DebugOutput::TevStage0 && debug_stage == index) {
            out += "debug_tev = combiner_output;\n";
        }
    }

    out += "combiner_buffer = next_combiner_buffer;\n";
    if (config.TevStageUpdatesCombinerBufferColor(index)) {
        out += "next_combiner_buffer.rgb = combiner_output.rgb;\n";
    }
    if (config.TevStageUpdatesCombinerBufferAlpha(index)) {
        out += "next_combiner_buffer.a = combiner_output.a;\n";
    }
}

void FragmentModule::WriteLighting() {
    if (!config.lighting.enable) {
        return;
    }

    const auto& lighting = config.lighting;

    out += "float4 diffuse_sum = float4(0.0, 0.0, 0.0, 1.0);\n"
           "float4 specular_sum = float4(0.0, 0.0, 0.0, 1.0);\n"
           "float3 light_vector = float3(0.0);\n"
           "float light_distance = 0.0;\n"
           "float3 refl_value = float3(0.0);\n"
           "float3 spot_dir = float3(0.0);\n"
           "float3 half_vector = float3(0.0);\n"
           "float dot_product = 0.0;\n"
           "float clamp_highlights = 1.0;\n"
           "float geo_factor = 1.0;\n";

    const auto perturbation = [&] {
        return fmt::format("2.0 * (sampleTexUnit{}()).rgb - 1.0", lighting.bump_selector.Value());
    };

    switch (lighting.bump_mode) {
    case LightingRegs::LightingBumpMode::NormalMap: {
        out += fmt::format("float3 surface_normal = {};\n", perturbation());
        if (lighting.bump_renorm) {
            constexpr std::string_view val = "(1.0 - (surface_normal.x*surface_normal.x + "
                                             "surface_normal.y*surface_normal.y))";
            out += fmt::format("surface_normal.z = sqrt(max({}, 0.0));\n", val);
        }
        out += "float3 surface_tangent = float3(1.0, 0.0, 0.0);\n";
        break;
    }
    case LightingRegs::LightingBumpMode::TangentMap: {
        out += fmt::format("float3 surface_tangent = {};\n", perturbation());
        out += "float3 surface_normal = float3(0.0, 0.0, 1.0);\n";
        break;
    }
    default:
        out += "float3 surface_normal = float3(0.0, 0.0, 1.0);\n"
               "float3 surface_tangent = float3(1.0, 0.0, 0.0);\n";
    }

    // The flat (provoking vertex) quaternion decides the interpolated one's hemisphere -
    // same short-arc correction as the GLSL non-barycentric path.
    out += "float4 quat_corrected = (dot(normquat, normquat_flat) < 0.0) ? -normquat : "
           "normquat;\n";

    out += "float4 normalized_normquat = normalize(quat_corrected);\n"
           "float3 normal = quaternion_rotate(normalized_normquat, surface_normal);\n"
           "float3 tangent = quaternion_rotate(normalized_normquat, surface_tangent);\n";

    if (lighting.enable_shadow) {
        std::string shadow_texture =
            fmt::format("sampleTexUnit{}()", lighting.shadow_selector.Value());
        if (lighting.shadow_invert) {
            out += fmt::format("float4 shadow = float4(1.0) - {};\n", shadow_texture);
        } else {
            out += fmt::format("float4 shadow = {};\n", shadow_texture);
        }
    } else {
        out += "float4 shadow = float4(1.0);\n";
    }

    const auto get_lut_value = [&lighting](LightingRegs::LightingSampler sampler, u32 light_num,
                                           LightingRegs::LightingLutInput input, bool abs) {
        std::string index;
        switch (input) {
        case LightingRegs::LightingLutInput::NH:
            index = "dot(normal, normalize(half_vector))";
            break;
        case LightingRegs::LightingLutInput::VH:
            index = "dot(normalize(view), normalize(half_vector))";
            break;
        case LightingRegs::LightingLutInput::NV:
            index = "dot(normal, normalize(view))";
            break;
        case LightingRegs::LightingLutInput::LN:
            index = "dot(light_vector, normal)";
            break;
        case LightingRegs::LightingLutInput::SP:
            index = "dot(light_vector, spot_dir)";
            break;
        case LightingRegs::LightingLutInput::CP:
            if (lighting.config == LightingRegs::LightingConfig::Config7) {
                constexpr std::string_view half_angle_proj =
                    "normalize(half_vector) - normal * dot(normal, normalize(half_vector))";
                index = fmt::format("dot({}, tangent)", half_angle_proj);
            } else {
                index = "0.0";
            }
            break;
        default:
            LOG_CRITICAL(HW_GPU, "Unknown lighting LUT input {}", static_cast<int>(input));
            index = "0.0";
            break;
        }

        const auto sampler_index = static_cast<u32>(sampler);

        if (abs) {
            index = lighting.lights[light_num].two_sided_diffuse
                        ? fmt::format("abs({})", index)
                        : fmt::format("max({}, 0.0)", index);
            return fmt::format("LookupLightingLUTUnsigned({}, {})", sampler_index, index);
        } else {
            return fmt::format("LookupLightingLUTSigned({}, {})", sampler_index, index);
        }
    };

    for (u32 light_index = 0; light_index < lighting.src_num; ++light_index) {
        const auto& light_config = lighting.lights[light_index];
        const std::string light_src = fmt::format("light_src[{}]", light_config.num.Value());

        if (light_config.directional) {
            out += fmt::format("light_vector = {}.position;\n", light_src);
        } else {
            out += fmt::format("light_vector = {}.position + view;\n", light_src);
        }
        out += "light_distance = length(light_vector);\n";
        out += "light_vector = normalize(light_vector);\n";

        out += fmt::format("spot_dir = {}.spot_direction;\n", light_src);
        out += "half_vector = normalize(view) + light_vector;\n";

        out += "dot_product = ";
        out += light_config.two_sided_diffuse ? "abs(dot(light_vector, normal));\n"
                                              : "max(dot(light_vector, normal), 0.0);\n";

        if (lighting.clamp_highlights) {
            out += "clamp_highlights = sign(dot_product);\n";
        }

        std::string spot_atten = "1.0";
        if (light_config.spot_atten_enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::SpotlightAttenuation)) {
            const std::string value =
                get_lut_value(LightingRegs::SpotlightAttenuationSampler(light_config.num),
                              light_config.num, lighting.lut_sp.type, lighting.lut_sp.abs_input);
            spot_atten = fmt::format("({:#} * {})", lighting.lut_sp.GetScale(), value);
        }

        std::string dist_atten = "1.0";
        if (light_config.dist_atten_enable) {
            const std::string index = fmt::format("clamp({}.dist_atten_scale * light_distance "
                                                  "+ {}.dist_atten_bias, 0.0, 1.0)",
                                                  light_src, light_src);
            const auto sampler = LightingRegs::DistanceAttenuationSampler(light_config.num);
            dist_atten = fmt::format("LookupLightingLUTUnsigned({}, {})",
                                     static_cast<u32>(sampler), index);
        }

        if (light_config.geometric_factor_0 || light_config.geometric_factor_1) {
            out += "geo_factor = dot(half_vector, half_vector);\n"
                   "geo_factor = geo_factor == 0.0 ? 0.0 : min("
                   "dot_product / geo_factor, 1.0);\n";
        }

        std::string d0_lut_value = "1.0";
        if (lighting.lut_d0.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution0)) {
            const std::string value =
                get_lut_value(LightingRegs::LightingSampler::Distribution0, light_config.num,
                              lighting.lut_d0.type, lighting.lut_d0.abs_input);
            d0_lut_value = fmt::format("({:#} * {})", lighting.lut_d0.GetScale(), value);
        }
        std::string specular_0 = fmt::format("({} * {}.specular_0)", d0_lut_value, light_src);
        if (light_config.geometric_factor_0) {
            specular_0 = fmt::format("({} * geo_factor)", specular_0);
        }

        if (lighting.lut_rr.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectRed)) {
            std::string value =
                get_lut_value(LightingRegs::LightingSampler::ReflectRed, light_config.num,
                              lighting.lut_rr.type, lighting.lut_rr.abs_input);
            value = fmt::format("({:#} * {})", lighting.lut_rr.GetScale(), value);
            out += fmt::format("refl_value.r = {};\n", value);
        } else {
            out += "refl_value.r = 1.0;\n";
        }

        if (lighting.lut_rg.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectGreen)) {
            std::string value =
                get_lut_value(LightingRegs::LightingSampler::ReflectGreen, light_config.num,
                              lighting.lut_rg.type, lighting.lut_rg.abs_input);
            value = fmt::format("({:#} * {})", lighting.lut_rg.GetScale(), value);
            out += fmt::format("refl_value.g = {};\n", value);
        } else {
            out += "refl_value.g = refl_value.r;\n";
        }

        if (lighting.lut_rb.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectBlue)) {
            std::string value =
                get_lut_value(LightingRegs::LightingSampler::ReflectBlue, light_config.num,
                              lighting.lut_rb.type, lighting.lut_rb.abs_input);
            value = fmt::format("({:#} * {})", lighting.lut_rb.GetScale(), value);
            out += fmt::format("refl_value.b = {};\n", value);
        } else {
            out += "refl_value.b = refl_value.r;\n";
        }

        std::string d1_lut_value = "1.0";
        if (lighting.lut_d1.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution1)) {
            const std::string value =
                get_lut_value(LightingRegs::LightingSampler::Distribution1, light_config.num,
                              lighting.lut_d1.type, lighting.lut_d1.abs_input);
            d1_lut_value = fmt::format("({:#} * {})", lighting.lut_d1.GetScale(), value);
        }
        std::string specular_1 =
            fmt::format("({} * refl_value * {}.specular_1)", d1_lut_value, light_src);
        if (light_config.geometric_factor_1) {
            specular_1 = fmt::format("({} * geo_factor)", specular_1);
        }

        if (light_index == lighting.src_num - 1 && lighting.lut_fr.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::Fresnel)) {
            std::string value =
                get_lut_value(LightingRegs::LightingSampler::Fresnel, light_config.num,
                              lighting.lut_fr.type, lighting.lut_fr.abs_input);
            value = fmt::format("({:#} * {})", lighting.lut_fr.GetScale(), value);

            if (lighting.enable_primary_alpha) {
                out += fmt::format("diffuse_sum.a = {};\n", value);
            }

            if (lighting.enable_secondary_alpha) {
                out += fmt::format("specular_sum.a = {};\n", value);
            }
        }

        const bool shadow_primary_enable = lighting.shadow_primary && light_config.shadow_enable;
        const bool shadow_secondary_enable =
            lighting.shadow_secondary && light_config.shadow_enable;
        const auto shadow_primary = shadow_primary_enable ? " * shadow.rgb" : "";
        const auto shadow_secondary = shadow_secondary_enable ? " * shadow.rgb" : "";

        out += fmt::format(
            "diffuse_sum.rgb += (({}.diffuse * dot_product{}) + {}.ambient) * {} * {};\n",
            light_src, shadow_primary, light_src, dist_atten, spot_atten);

        out += fmt::format("specular_sum.rgb += ({} + {}) * clamp_highlights * {} * {}{};\n",
                           specular_0, specular_1, dist_atten, spot_atten, shadow_secondary);
    }

    if (lighting.shadow_alpha) {
        if (lighting.enable_primary_alpha) {
            out += "diffuse_sum.a *= shadow.a;\n";
        }
        if (lighting.enable_secondary_alpha) {
            out += "specular_sum.a *= shadow.a;\n";
        }
    }

    out += "diffuse_sum.rgb += lighting_global_ambient;\n"
           "primary_fragment_color = clamp(diffuse_sum, float4(0.0), float4(1.0));\n"
           "secondary_fragment_color = clamp(specular_sum, float4(0.0), float4(1.0));\n";
}

void FragmentModule::WriteFog() {
    if (config.texture.fog_flip) {
        out += "float fog_index = (1.0 - float(depth)) * 128.0;\n";
    } else {
        out += "float fog_index = depth * 128.0;\n";
    }

    out += "float fog_i = clamp(floor(fog_index), 0.0, 127.0);\n"
           "float fog_f = fog_index - fog_i;\n"
           "float2 fog_lut_entry = lut_lf(fog_i + fog_lut_offset);\n"
           "float fog_factor = fog_lut_entry.r + fog_lut_entry.g * fog_f;\n"
           "fog_factor = clamp(fog_factor, 0.0, 1.0);\n";

    out += "combiner_output.rgb = lerp(fog_color.rgb, combiner_output.rgb, fog_factor);\n";
}

void FragmentModule::WriteGas() {
    LOG_CRITICAL(Render, "Unimplemented gas mode");
    out += "// UNSUPPORTED-GXM: gas mode\n";
    out += "color = float4(0.0); }";
}

void FragmentModule::WriteLogicOp() {
    const auto logic_op = config.framebuffer.logic_op.Value();
    switch (logic_op) {
    case FramebufferRegs::LogicOp::Clear:
        out += "color = float4(0.0);\n";
        break;
    case FramebufferRegs::LogicOp::Set:
        out += "color = float4(1.0);\n";
        break;
    case FramebufferRegs::LogicOp::Copy:
    case FramebufferRegs::LogicOp::NoOp:
        break;
    default:
        // Bitwise framebuffer ops have no SGX expression; recorded for the census.
        out += fmt::format("// UNSUPPORTED-GXM: logic_op {:#x}\n", static_cast<u32>(logic_op));
        break;
    }
}

void FragmentModule::WriteBlending() {
    const bool requires_rgb_minmax_emulation =
        config.framebuffer.requested_rgb_blend.RequiresMinMaxEmulation();
    const bool requires_alpha_minmax_emulation =
        config.framebuffer.requested_alpha_blend.RequiresMinMaxEmulation();
    if (!requires_rgb_minmax_emulation && !requires_alpha_minmax_emulation) [[likely]] {
        return;
    }

    using BlendFactor = Pica::FramebufferRegs::BlendFactor;
    out += "float4 source_color = combiner_output;\n";
    // Reading the destination needs the current framebuffer bound as tex_color, like the GL
    // no-texelFetch fallback.
    out += "float4 dest_color = tex2D(tex_color, frag_coord.xy / fb_dims);\n";
    const auto get_factor = [&](BlendFactor factor) -> std::string {
        switch (factor) {
        case BlendFactor::Zero:
            return "float4(0.0)";
        case BlendFactor::One:
            return "float4(1.0)";
        case BlendFactor::SourceColor:
            return "source_color";
        case BlendFactor::OneMinusSourceColor:
            return "float4(1.0) - source_color";
        case BlendFactor::DestColor:
            return "dest_color";
        case BlendFactor::OneMinusDestColor:
            return "float4(1.0) - dest_color";
        case BlendFactor::SourceAlpha:
            return "source_color.aaaa";
        case BlendFactor::OneMinusSourceAlpha:
            return "float4(1.0) - source_color.aaaa";
        case BlendFactor::DestAlpha:
            return "dest_color.aaaa";
        case BlendFactor::OneMinusDestAlpha:
            return "float4(1.0) - dest_color.aaaa";
        case BlendFactor::ConstantColor:
            return "blend_color";
        case BlendFactor::OneMinusConstantColor:
            return "float4(1.0) - blend_color";
        case BlendFactor::ConstantAlpha:
            return "blend_color.aaaa";
        case BlendFactor::OneMinusConstantAlpha:
            return "float4(1.0) - blend_color.aaaa";
        default:
            LOG_CRITICAL(Render, "Unknown blend factor {}", factor);
            return "float4(1.0)";
        }
    };

    const auto get_func = [](Pica::FramebufferRegs::BlendEquation eq) {
        return eq == Pica::FramebufferRegs::BlendEquation::Min ? "min" : "max";
    };

    if (requires_rgb_minmax_emulation) {
        out += fmt::format(
            "combiner_output.rgb = {}(source_color.rgb * ({}).rgb, dest_color.rgb * ({}).rgb);\n",
            get_func(config.framebuffer.requested_rgb_blend.eq),
            get_factor(config.framebuffer.requested_rgb_blend.src_factor),
            get_factor(config.framebuffer.requested_rgb_blend.dst_factor));
    }
    if (requires_alpha_minmax_emulation) {
        out +=
            fmt::format("combiner_output.a = {}(source_color.a * ({}).a, dest_color.a * ({}).a);\n",
                        get_func(config.framebuffer.requested_alpha_blend.eq),
                        get_factor(config.framebuffer.requested_alpha_blend.src_factor),
                        get_factor(config.framebuffer.requested_alpha_blend.dst_factor));
    }
    use_blend_fallback = true;
}

void FragmentModule::AppendProcTexShiftOffset(std::string_view v, ProcTexShift mode,
                                              ProcTexClamp clamp_mode) {
    const auto offset = (clamp_mode == ProcTexClamp::MirroredRepeat) ? "1.0" : "0.5";
    switch (mode) {
    case ProcTexShift::None:
        out += "0.0";
        break;
    case ProcTexShift::Odd:
        out += fmt::format("{} * float((int({}) >> 1) & 1)", offset, v);
        break;
    case ProcTexShift::Even:
        out += fmt::format("{} * float(((int({}) + 1) >> 1) & 1)", offset, v);
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown shift mode {}", mode);
        out += "0.0";
        break;
    }
}

void FragmentModule::AppendProcTexClamp(std::string_view var, ProcTexClamp mode) {
    switch (mode) {
    case ProcTexClamp::ToZero:
        out += fmt::format("{0} = {0} > 1.0 ? 0.0 : {0};\n", var);
        break;
    case ProcTexClamp::ToEdge:
        out += fmt::format("{0} = min({0}, 1.0);\n", var);
        break;
    case ProcTexClamp::SymmetricalRepeat:
        out += fmt::format("{0} = frac({0});\n", var);
        break;
    case ProcTexClamp::MirroredRepeat:
        out += fmt::format("{0} = (int({0}) & 1) == 0 ? frac({0}) : 1.0 - frac({0});\n", var);
        break;
    case ProcTexClamp::Pulse:
        out += fmt::format("{0} = {0} > 0.5 ? 1.0 : 0.0;\n", var);
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown clamp mode {}", mode);
        out += fmt::format("{0} = min({0}, 1.0);\n", var);
        break;
    }
}

void FragmentModule::AppendProcTexCombineAndMap(ProcTexCombiner combiner,
                                                std::string_view offset) {
    const auto combined = [combiner] {
        switch (combiner) {
        case ProcTexCombiner::U:
            return "u";
        case ProcTexCombiner::U2:
            return "(u * u)";
        case TexturingRegs::ProcTexCombiner::V:
            return "v";
        case TexturingRegs::ProcTexCombiner::V2:
            return "(v * v)";
        case TexturingRegs::ProcTexCombiner::Add:
            return "((u + v) * 0.5)";
        case TexturingRegs::ProcTexCombiner::Add2:
            return "((u * u + v * v) * 0.5)";
        case TexturingRegs::ProcTexCombiner::SqrtAdd2:
            return "min(sqrt(u * u + v * v), 1.0)";
        case TexturingRegs::ProcTexCombiner::Min:
            return "min(u, v)";
        case TexturingRegs::ProcTexCombiner::Max:
            return "max(u, v)";
        case TexturingRegs::ProcTexCombiner::RMax:
            return "min(((u + v) * 0.5 + sqrt(u * u + v * v)) * 0.5, 1.0)";
        default:
            LOG_CRITICAL(HW_GPU, "Unknown combiner {}", combiner);
            return "0.0";
        }
    }();
    out += fmt::format("ProcTexLookupLUT({}, {})", offset, combined);
}

void FragmentModule::DefineProcTexSampler() {
    if (!config.proctex.enable) {
        return;
    }

    out += R"(
float ProcTexLookupLUT(float lut_offset, float coord) {
    coord *= 128.0;
    float index_i = clamp(floor(coord), 0.0, 127.0);
    float index_f = coord - index_i;
    float2 entry = lut_rg(index_i + lut_offset);
    return clamp(entry.r + entry.g * index_f, 0.0, 1.0);
}
)";

    if (config.proctex.noise_enable) {
        // The tables are declared here rather than inside the two functions: a storage class
        // on a local is a compile error ("D5402: storage class 'static' may not be used in
        // function bodies"), and a fragment program that will not compile takes every draw of
        // the title with it - the two proctex samples drew nothing at all until this moved.
        out += R"(
const int proctex_noise_table1[16] = {0,4,10,8,4,9,7,12,5,15,13,14,11,15,2,11};
const int proctex_noise_table2[16] = {10,2,15,8,0,7,4,5,5,13,2,6,13,9,3,14};

int ProcTexNoiseRand1D(int v) {
    // The fragment profile has no integer divide with an arbitrary denominator, so the
    // division by 9 goes through float; v is small and positive here, so this is exact.
    int q = int(floor(float(v) * (1.0 / 9.0)));
    int r = v - q * 9;
    return ((r + 2) * 3 & 0xF) ^ proctex_noise_table1[q & 0xF];
}

float ProcTexNoiseRand2D(float2 point) {
    int u2 = ProcTexNoiseRand1D(int(point.x));
    int v2 = ProcTexNoiseRand1D(int(point.y));
    v2 += ((u2 & 3) == 1) ? 4 : 0;
    v2 ^= (u2 & 1) * 6;
    v2 += 10 + u2;
    v2 &= 0xF;
    v2 ^= proctex_noise_table2[u2];
    return -1.0 + float(v2) * (2.0/15.0);
}

float ProcTexNoiseCoef(float2 x) {
    float2 grid  = 9.0 * proctex_noise_f * abs(x + proctex_noise_p);
    float2 point = floor(grid);
    float2 frac_part = grid - point;

    float g0 = ProcTexNoiseRand2D(point) * (frac_part.x + frac_part.y);
    float g1 = ProcTexNoiseRand2D(point + float2(1.0, 0.0)) * (frac_part.x + frac_part.y - 1.0);
    float g2 = ProcTexNoiseRand2D(point + float2(0.0, 1.0)) * (frac_part.x + frac_part.y - 1.0);
    float g3 = ProcTexNoiseRand2D(point + float2(1.0, 1.0)) * (frac_part.x + frac_part.y - 2.0);

    float x_noise = ProcTexLookupLUT(proctex_noise_lut_offset, frac_part.x);
    float y_noise = ProcTexLookupLUT(proctex_noise_lut_offset, frac_part.y);
    float x0 = lerp(g0, g1, x_noise);
    float x1 = lerp(g2, g3, x_noise);
    return lerp(x0, x1, y_noise);
}
)";
    }

    out += "float4 SampleProcTexColor(float lut_coord, int level) {\n";
    out += fmt::format("int lut_width = {} >> level;\n", config.proctex.lut_width);
    out += fmt::format(
        "float lut_offsets[8] = {{{}.0, {}.0, {}.0, {}.0, 240.0, 248.0, 252.0, 254.0}};\n",
        config.proctex.lut_offset0, config.proctex.lut_offset1, config.proctex.lut_offset2,
        config.proctex.lut_offset3);
    out += "float lut_offset = lut_offsets[level];\n";
    out += "lut_coord *= float(lut_width - 1);\n";

    switch (config.proctex.lut_filter) {
    case ProcTexFilter::Linear:
    case ProcTexFilter::LinearMipmapLinear:
    case ProcTexFilter::LinearMipmapNearest:
        out += "float lut_index_i = floor(lut_coord) + lut_offset;\n";
        out += "float lut_index_f = frac(lut_coord);\n";
        out += "return lut_rgba(lut_index_i + proctex_lut_offset) + "
               "lut_index_f * "
               "lut_rgba(lut_index_i + proctex_diff_lut_offset);\n";
        break;
    case ProcTexFilter::Nearest:
    case ProcTexFilter::NearestMipmapLinear:
    case ProcTexFilter::NearestMipmapNearest:
        out += "lut_coord += lut_offset;\n";
        out += "return lut_rgba(floor(lut_coord + 0.5) + proctex_lut_offset);\n";
        break;
    }

    out += "}\n";

    out += "float4 ProcTex() {\n";
    if (config.proctex.coord < 3) {
        out += fmt::format("float2 uv = abs(texcoord{});\n", config.proctex.coord.Value());
    } else {
        LOG_CRITICAL(Render, "Unexpected proctex.coord >= 3");
        out += "float2 uv = abs(texcoord0);\n";
    }

    out += "float2 duv = max(abs(ddx(uv)), abs(ddy(uv)));\n";
    out += fmt::format("float lod = log2(abs(float({}) * proctex_bias) * (duv.x + duv.y));\n",
                       config.proctex.lut_width);
    out += "if (proctex_bias == 0.0) lod = 0.0;\n";
    out += fmt::format("lod = clamp(lod, {:#}, {:#});\n",
                       std::max(0.0f, static_cast<f32>(config.proctex.lod_min)),
                       std::min(7.0f, static_cast<f32>(config.proctex.lod_max)));

    out += "float u_shift = ";
    AppendProcTexShiftOffset("uv.y", config.proctex.u_shift, config.proctex.u_clamp);
    out += ";\n";
    out += "float v_shift = ";
    AppendProcTexShiftOffset("uv.x", config.proctex.v_shift, config.proctex.v_clamp);
    out += ";\n";

    if (config.proctex.noise_enable) {
        out += "uv += proctex_noise_a * ProcTexNoiseCoef(uv);\n"
               "uv = abs(uv);\n";
    }

    out += "float u = uv.x + u_shift;\n"
           "float v = uv.y + v_shift;\n";

    AppendProcTexClamp("u", config.proctex.u_clamp);
    AppendProcTexClamp("v", config.proctex.v_clamp);

    out += "float lut_coord = ";
    AppendProcTexCombineAndMap(config.proctex.color_combiner, "proctex_color_map_offset");
    out += ";\n";

    switch (config.proctex.lut_filter) {
    case ProcTexFilter::Linear:
    case ProcTexFilter::Nearest:
        out += "float4 final_color = SampleProcTexColor(lut_coord, 0);\n";
        break;
    case ProcTexFilter::NearestMipmapNearest:
    case ProcTexFilter::LinearMipmapNearest:
        out += "float4 final_color = SampleProcTexColor(lut_coord, int(floor(lod + 0.5)));\n";
        break;
    case ProcTexFilter::NearestMipmapLinear:
    case ProcTexFilter::LinearMipmapLinear:
        out += "int lod_i = int(lod);\n"
               "float lod_f = frac(lod);\n"
               "float4 final_color = lerp(SampleProcTexColor(lut_coord, lod_i), "
               "SampleProcTexColor(lut_coord, lod_i + 1), lod_f);\n";
        break;
    }

    if (config.proctex.separate_alpha) {
        out += "float final_alpha = ";
        AppendProcTexCombineAndMap(config.proctex.alpha_combiner, "proctex_alpha_map_offset");
        out += ";\n";
        out += "return float4(final_color.xyz, final_alpha);\n}\n";
    } else {
        out += "return final_color;\n}\n";
    }
}

void FragmentModule::DefineSamplers() {
    // tex0 becomes a cube sampler for the cube type; the shadow types have no GXM expression
    // and their sampler stays 2D for the stubbed helpers.
    const auto texture_type = config.texture.texture0_type.Value();
    if (texture_type == TextureType::TextureCube) {
        out += "uniform samplerCUBE tex0 : TEXUNIT0;\n";
    } else {
        out += "uniform sampler2D tex0 : TEXUNIT0;\n";
    }
    out += "uniform sampler2D tex1 : TEXUNIT1;\n";
    out += "uniform sampler2D tex2 : TEXUNIT2;\n";
    out += "uniform sampler2D texture_buffer_lut_lf : TEXUNIT3;\n";
    out += "uniform sampler2D texture_buffer_lut_rg : TEXUNIT4;\n";
    out += "uniform sampler2D texture_buffer_lut_rgba : TEXUNIT5;\n";
    if (config.framebuffer.requested_rgb_blend.RequiresMinMaxEmulation() ||
        config.framebuffer.requested_alpha_blend.RequiresMinMaxEmulation()) {
        out += "uniform sampler2D tex_color : TEXUNIT7;\n";
        out += "uniform float2 fb_dims;\n";
    }
    out += "\n";
}

void FragmentModule::DefineHelpers() {
    // The LUT helpers sample texel centres of the shared 256-wide LUT textures, the exact
    // scheme the GLSL no-texelFetch path uses; the geometry constants come from profile.h.
    out += fmt::format(R"(
float3 quaternion_rotate(float4 q, float3 v) {{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}}

float byteround(float x) {{
    return floor(x * 255.0 + 0.5) * (1.0 / 255.0);
}}

float2 byteround(float2 x) {{
    return floor(x * 255.0 + 0.5) * (1.0 / 255.0);
}}

float3 byteround(float3 x) {{
    return floor(x * 255.0 + 0.5) * (1.0 / 255.0);
}}

float4 byteround(float4 x) {{
    return floor(x * 255.0 + 0.5) * (1.0 / 255.0);
}}

float getLod(float2 coord) {{
    float2 d = max(abs(ddx(coord)), abs(ddy(coord)));
    return log2(max(d.x, d.y));
}}

// The index is a whole number in a float: row and column come out of it with a floor and a
// multiply-subtract, where the integer form needed a shift, a mask and two conversions back.
float2 lut_coordinate(float i, float inv_rows) {{
    float lut_row = floor(i * (1.0 / 256.0));
    return (float2(i - lut_row * 256.0, lut_row) + 0.5) * float2(1.0 / 256.0, inv_rows);
}}

float2 lut_lf(float i) {{ return tex2D(texture_buffer_lut_lf, lut_coordinate(i, {:.10f})).rg; }}
float2 lut_rg(float i) {{ return tex2D(texture_buffer_lut_rg, lut_coordinate(i, {:.10f})).rg; }}
float4 lut_rgba(float i) {{ return tex2D(texture_buffer_lut_rgba, lut_coordinate(i, {:.10f})); }}
)",
                       1.0 / static_cast<double>(LUT_LF_ROWS),
                       1.0 / static_cast<double>(LUT_RG_ROWS),
                       1.0 / static_cast<double>(LUT_RGBA_ROWS));
}

void FragmentModule::DefineLightingHelpers() {
    if (!config.lighting.enable) {
        return;
    }

    out += R"(
// lut_index is always a literal at the call site, so its shift and mask fold away and no
// integer survives into the generated code.
float LookupLightingLUT(int lut_index, float index, float delta) {
    float2 entry = lut_lf(lighting_lut_offset[lut_index >> 2][lut_index & 3] + index);
    return entry.r + entry.g * delta;
}

float LookupLightingLUTUnsigned(int lut_index, float pos) {
    float index = clamp(floor(pos * 256.0), 0.0, 255.0);
    float delta = pos * 256.0 - index;
    return LookupLightingLUT(lut_index, index, delta);
}

float LookupLightingLUTSigned(int lut_index, float pos) {
    float index = clamp(floor(pos * 128.0), -128.0, 127.0);
    float delta = pos * 128.0 - index;
    if (index < 0.0) index += 256.0;
    return LookupLightingLUT(lut_index, index, delta);
}
)";
}

void FragmentModule::DefineTexUnitSampler(u32 texture_unit) {
    out += fmt::format("float4 sampleTexUnit{}() {{\n", texture_unit);
    if (texture_unit == 0 &&
        config.texture.texture0_type == TexturingRegs::TextureConfig::Disabled) {
        out += "return float4(0.0);\n}\n";
        return;
    }

    if (texture_unit < 3) {
        const u32 texcoord_num =
            texture_unit == 2 && config.texture.texture2_use_coord1 ? 1 : texture_unit;
        if (config.texture.texture_border_color[texture_unit].enable_s) {
            out += fmt::format("if (texcoord{0}.x < 0.0 || texcoord{0}.x > 1.0) {{\n"
                               "    return tex_border_color[{1}];\n}}\n",
                               texcoord_num, texture_unit);
        }
        if (config.texture.texture_border_color[texture_unit].enable_t) {
            out += fmt::format("if (texcoord{0}.y < 0.0 || texcoord{0}.y > 1.0) {{\n"
                               "    return tex_border_color[{1}];\n}}\n",
                               texcoord_num, texture_unit);
        }
    }

    switch (texture_unit) {
    case 0:
        switch (config.texture.texture0_type) {
        case TexturingRegs::TextureConfig::Texture2D:
            out += "return tex2Dlod(tex0, float4(texcoord0, 0.0, "
                   "getLod(texcoord0 * tex_dims[0]) + tex_lod_bias[0]));";
            break;
        case TexturingRegs::TextureConfig::Projection2D:
            out += "return tex2Dproj(tex0, float3(texcoord0, pica_texcoord0_w));";
            break;
        case TexturingRegs::TextureConfig::TextureCube:
            out += "return texCUBE(tex0, float3(texcoord0, pica_texcoord0_w));";
            break;
        case TexturingRegs::TextureConfig::Shadow2D:
        case TexturingRegs::TextureConfig::ShadowCube:
            // Needs storage-image reads GXM does not have.
            out += "// UNSUPPORTED-GXM: shadow texture sampling\n"
                   "return float4(1.0);";
            break;
        default:
            LOG_CRITICAL(HW_GPU, "Unhandled texture type {:x}",
                         config.texture.texture0_type.Value());
            out += "return tex2D(tex0, texcoord0);";
            break;
        }
        break;
    case 1:
        out += "return tex2Dlod(tex1, float4(texcoord1, 0.0, "
               "getLod(texcoord1 * tex_dims[1]) + tex_lod_bias[1]));";
        break;
    case 2:
        if (config.texture.texture2_use_coord1) {
            out += "return tex2Dlod(tex2, float4(texcoord1, 0.0, "
                   "getLod(texcoord1 * tex_dims[2]) + tex_lod_bias[2]));";
        } else {
            out += "return tex2Dlod(tex2, float4(texcoord2, 0.0, "
                   "getLod(texcoord2 * tex_dims[2]) + tex_lod_bias[2]));";
        }
        break;
    case 3:
        if (config.proctex.enable) {
            out += "return ProcTex();";
        } else {
            out += "return float4(0.0);";
        }
        break;
    default:
        break;
    }

    out += "\n}\n";
}

std::string GenerateFragmentShader(const FSConfig& config, const UserConfig& user,
                                   const Profile& profile) {
    FragmentModule module{config, user, profile};
    return module.Generate();
}

namespace {
std::string g_dump_dir;
DebugOutput g_debug_output = DebugOutput::None;
bool g_fog_disabled = false;
} // Anonymous namespace

void SetFogDisabled(bool disabled) {
    g_fog_disabled = disabled;
}

bool FogDisabled() {
    return g_fog_disabled;
}

void SetDebugOutput(DebugOutput mode) {
    g_debug_output = mode;
}

DebugOutput GetDebugOutput() {
    return g_debug_output;
}

void SetDumpDir(std::string dir) {
    g_dump_dir = std::move(dir);
}

const std::string& DumpDir() {
    return g_dump_dir;
}

void MaybeDump(const FSConfig& config, const UserConfig& user, u64 hash) {
    if (g_dump_dir.empty()) {
        return;
    }
    const std::string source = GenerateFragmentShader(config, user, MakeGxmProfile());
    const std::string path = fmt::format("{}/fs_{:016x}.cg", g_dump_dir, hash);
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        LOG_ERROR(Render, "could not write {}", path);
        return;
    }
    std::fwrite(source.data(), 1, source.size(), file);
    std::fclose(file);
    const bool unsupported = source.find("UNSUPPORTED-GXM") != std::string::npos;
    LOG_INFO(Render, "dumped Cg fragment shader {:016x} ({} bytes{})", hash, source.size(),
             unsupported ? ", contains UNSUPPORTED-GXM markers" : "");
}

} // namespace Pica::Shader::Generator::Cg
