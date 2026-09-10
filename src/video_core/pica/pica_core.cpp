// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include <cstring>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <atomic>
#include <span>
#include "common/arch.h"
#include "common/archives.h"
#include "common/math_util.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"

namespace Pica {

// The chain segment most recently entered by a command_buffer trigger, copied into the list
// record at parse end. Emulation-thread only.
static u32 g_last_chain_addr = 0;
static u32 g_last_chain_size = 0;

MICROPROFILE_DEFINE(GPU_Drawing, "GPU", "Drawing", MP_RGB(50, 50, 240));

using namespace DebugUtils;

// Class representing implementation details of each internal register
// The set/get pattern is used instead of a bitfield union to allow
// constexpr evaluation.
class RegImplInfo {
private:
    using NeedsSpecialHandlingBF = BitField<0, 1, u16>;
    using SupportsBatchBF = BitField<1, 1, u16>;
    using RegsUntilSpecialBF = BitField<2, 14, u16>;

    u16 raw{};

public:
    constexpr bool NeedsSpecialHandling() const {
        return NeedsSpecialHandlingBF::ExtractValue(raw) != 0;
    }

    constexpr void SetNeedsSpecialHandling() {
        raw = (raw & ~NeedsSpecialHandlingBF::mask) | NeedsSpecialHandlingBF::FormatValue(1);
    }

    constexpr bool SupportsBatch() const {
        return SupportsBatchBF::ExtractValue(raw) != 0;
    }

    constexpr void SetSupportsBatch() {
        raw = (raw & ~SupportsBatchBF::mask) | SupportsBatchBF::FormatValue(1);
    }

    constexpr u16 RegsUntilSpecial() const {
        return RegsUntilSpecialBF::ExtractValue(raw);
    }

    constexpr void SetRegsUntilSpecial(u16 value) {
        raw = (raw & ~RegsUntilSpecialBF::mask) | RegsUntilSpecialBF::FormatValue(value);
    }
};

union CommandHeader {
    u32 hex;
    BitField<0, 16, u32> cmd_id;
    BitField<16, 4, u32> parameter_mask;
    BitField<20, 8, u32> extra_data_length;
    BitField<31, 1, u32> group_commands;
};
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())} {
    InitializeRegs();
    dirty_regs.SetAllDirty();

    const auto submit_vertex = [this](const AttributeBuffer& buffer) {
        const auto add_triangle = [this](const OutputVertex& v0, const OutputVertex& v1,
                                         const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
        const auto vertex = OutputVertex(output_vertex_map, buffer);
        primitive_assembler.SubmitVertex(vertex, add_triangle);
    };

    gs_unit.SetVertexHandlers(submit_vertex, [this]() { primitive_assembler.SetWinding(); });
    geometry_pipeline.SetVertexHandler(submit_vertex);

    primitive_assembler.Reconfigure(PipelineRegs::TriangleTopology::List);
}

PicaCore::~PicaCore() = default;

void PicaCore::InitializeRegs() {
    // Values initialized by GSP
    regs.internal.irq_autostop = 1;
    regs.internal.irq_mask = 0xFFFFFFF0;
    // Older versions of libctru didn't initialize this, initialize it here to avoid endless black
    // screen. Not needed on actual hardware due to previous software already having set it up
    regs.internal.irq_compare = 0x12345678;

    auto& framebuffer_top = regs.framebuffer_config[0];
    auto& framebuffer_sub = regs.framebuffer_config[1];

    // Set framebuffer defaults from nn::gx::Initialize
    framebuffer_top.address_left1 = 0x181E6000;
    framebuffer_top.address_left2 = 0x1822C800;
    framebuffer_top.address_right1 = 0x18273000;
    framebuffer_top.address_right2 = 0x182B9800;
    framebuffer_sub.address_left1 = 0x1848F000;
    framebuffer_sub.address_left2 = 0x184C7800;

    framebuffer_top.width.Assign(240);
    framebuffer_top.height.Assign(400);
    framebuffer_top.stride = 3 * 240;
    framebuffer_top.color_format.Assign(PixelFormat::RGB8);
    framebuffer_top.active_fb = 0;

    framebuffer_sub.width.Assign(240);
    framebuffer_sub.height.Assign(320);
    framebuffer_sub.stride = 3 * 240;
    framebuffer_sub.color_format.Assign(PixelFormat::RGB8);
    framebuffer_sub.active_fb = 0;

    // Tales of Abyss expects this register to have the following default values.
    auto& gs = regs.internal.gs;
    gs.max_input_attribute_index.Assign(1);
    gs.shader_mode.Assign(ShaderRegs::ShaderMode::VS);
}

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    this->rasterizer = rasterizer;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt) {
    this->signal_interrupt = signal_interrupt;
}

static bool any_byte_match(u32 a, u32 b) {
    return ((a & 0xFF) == (b & 0xFF)) || (((a >> 8) & 0xFF) == ((b >> 8) & 0xFF)) ||
           (((a >> 16) & 0xFF) == ((b >> 16) & 0xFF)) || (((a >> 24) & 0xFF) == ((b >> 24) & 0xFF));
}

// Special-register dispatch: one handler per register id, called through a table indexed by
// the id. The switch this replaces spanned ids 0x010..0x2DD with about 80 cases, too sparse
// for a jump table, so clang emitted a comparison tree that every special write walked
// (several branches for a uniform upload, the commonest special write in a command list).
// The table is one load and one indirect call; a null entry means the register is plain.
struct PicaCore::Dispatch {
    static void IrqRequest(PicaCore& p, u32 id, u32, bool& stop_requested) {
        // TODO(PabloMK7): This logic is not fully accurate, but close enough:
        // https://problemkaputt.de/gbatek-3ds-gpu-internal-registers-finalize-interrupt-registers.htm
        g_pica_probe[1].fetch_add(1, std::memory_order_relaxed);
        if (any_byte_match(p.regs.internal.reg_array[id], p.regs.internal.irq_compare))
            [[likely]] {
            g_pica_probe[2].fetch_add(1, std::memory_order_relaxed);
            // Under the software renderer the P3D must trail the shipped draws it acknowledges;
            // see RasterizerInterface::DefersInterrupts.
            if (p.rasterizer && p.rasterizer->DefersInterrupts()) [[unlikely]] {
                p.rasterizer->PostInterruptAfterQueue(
                    Service::GSP::InterruptId::P3D, p.delay_generator.CalculateAndResetDelay());
            } else {
                p.signal_interrupt(Service::GSP::InterruptId::P3D,
                                   p.delay_generator.CalculateAndResetDelay());
            }
            if (p.regs.internal.irq_autostop) [[likely]] {
                g_pica_probe[5].fetch_add(1, std::memory_order_relaxed);
                stop_requested = true;
            }
        } else {
            g_pica_probe[3].fetch_add(1, std::memory_order_relaxed);
#ifdef CITRA_TRACE_PROBES
            static const bool gx_trace = std::getenv("AZAHAR_GX_TRACE") != nullptr;
            if (gx_trace) {
                LOG_INFO(HW_GPU, "GXTRACE irq_nomatch req={:08x} cmp={:08x}",
                         p.regs.internal.reg_array[id], p.regs.internal.irq_compare);
            }
#endif // CITRA_TRACE_PROBES
        }
    }

    static void TriangleTopology(PicaCore& p, u32, u32, bool&) {
        p.primitive_assembler.Reconfigure(p.regs.internal.pipeline.triangle_topology);
        p.pa_reconfigure_events++;
    }

    static void RestartPrimitive(PicaCore& p, u32, u32, bool&) {
        p.primitive_assembler.Reset();
        p.pa_reset_events++;
    }

    static void DefaultAttributesIndex(PicaCore& p, u32, u32, bool&) {
        p.immediate.Reset();
    }

    // Load default vertex input attributes
    static void DefaultAttributesValue(PicaCore& p, u32, u32 value, bool&) {
        p.SubmitImmediate(value);
    }

    static void DefaultAttributesValueBatch(PicaCore& p, u32, const u32* values, u32 count) {
        for (u32 i = 0; i < count; i++) {
            p.SubmitImmediate(values[i]);
        }
    }

    static void CommandBufferTrigger(PicaCore& p, u32 id, u32, bool&) {
        g_pica_probe[4].fetch_add(1, std::memory_order_relaxed);
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
        const PAddr addr = p.regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
        const u32 size = p.regs.internal.pipeline.command_buffer.GetSize(index);
        g_last_chain_addr = addr;
        g_last_chain_size = size;
        const u8* head = p.memory.GetPhysicalPointer(addr);
        p.cmd_list.Reset(addr, head, size);
    }

    // It seems like these trigger vertex rendering
    static void TriggerDraw(PicaCore& p, u32 id, u32, bool&) {
        const bool is_indexed = (id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
        p.DrawArrays(is_indexed);
    }

    static bool GsMirrorsVs(const PicaCore& p) {
        return !p.regs.internal.pipeline.gs_unit_exclusive_configuration &&
               p.regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No;
    }

    static void GsBoolUniforms(PicaCore& p, u32, u32, bool&) {
        p.gs_setup.WriteUniformBoolReg(p.regs.internal.gs.bool_uniforms.Value());
    }

    static void GsIntUniform(PicaCore& p, u32 id, u32, bool&) {
        const u32 index = (id - PICA_REG_INDEX(gs.int_uniforms[0]));
        p.gs_setup.WriteUniformIntReg(index, p.regs.internal.gs.GetIntUniform(index));
    }

    static void GsUniformValue(PicaCore& p, u32, u32 value, bool&) {
        p.gs_setup.WriteUniformFloatReg(p.regs.internal.gs, value);
    }

    static void GsUniformValueBatch(PicaCore& p, u32, const u32* values, u32 count) {
        p.gs_setup.WriteUniformFloatRegRange(p.regs.internal.gs, values, count);
    }

    static void GsProgramWord(PicaCore& p, u32, u32 value, bool&) {
        u32& offset = p.regs.internal.gs.program.offset;
        if (offset >= 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {}", offset);
        } else {
            p.gs_setup.UpdateProgramCode(offset, value);
            offset++;
        }
    }

    static void GsProgramWordBatch(PicaCore& p, u32, const u32* values, u32 count) {
        u32& offset = p.regs.internal.gs.program.offset;
        if (offset + count > 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {} count {}", offset, count);
        } else {
            p.gs_setup.UpdateProgramCodeRange(offset, values, count);
            offset += count;
        }
    }

    static void GsSwizzleWord(PicaCore& p, u32, u32 value, bool&) {
        u32& offset = p.regs.internal.gs.swizzle_patterns.offset;
        if (offset >= p.gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {}", offset);
        } else {
            p.gs_setup.UpdateSwizzleData(offset, value);
            offset++;
        }
    }

    static void GsSwizzleWordBatch(PicaCore& p, u32, const u32* values, u32 count) {
        u32& offset = p.regs.internal.gs.swizzle_patterns.offset;
        if (offset + count > p.gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {} count {}", offset, count);
        } else {
            p.gs_setup.UpdateSwizzleDataRange(offset, values, count);
            offset += count;
        }
    }

    static void VsOutputMask(PicaCore& p, u32, u32 value, bool&) {
        if (GsMirrorsVs(p)) {
            p.regs.internal.gs.output_mask.Assign(value);
        }
    }

    static void VsBoolUniforms(PicaCore& p, u32, u32, bool&) {
        p.vs_setup.WriteUniformBoolReg(p.regs.internal.vs.bool_uniforms.Value());
        if (GsMirrorsVs(p)) {
            p.gs_setup.WriteUniformBoolReg(p.regs.internal.vs.bool_uniforms.Value());
        }
    }

    static void VsIntUniform(PicaCore& p, u32 id, u32, bool&) {
        const u32 index = (id - PICA_REG_INDEX(vs.int_uniforms[0]));
        p.vs_setup.WriteUniformIntReg(index, p.regs.internal.vs.GetIntUniform(index));
        if (GsMirrorsVs(p)) {
            p.gs_setup.WriteUniformIntReg(index, p.regs.internal.vs.GetIntUniform(index));
        }
    }

    static void VsUniformValue(PicaCore& p, u32, u32 value, bool&) {
        const auto index = p.vs_setup.WriteUniformFloatReg(p.regs.internal.vs, value);
        if (GsMirrorsVs(p) && index) {
            p.gs_setup.uniforms_sync_dirty = true;
            p.gs_setup.WidenFloatSyncWindow(index.value(), index.value() + 1);
            p.gs_setup.uniforms.f[index.value()] = p.vs_setup.uniforms.f[index.value()];
        }
    }

    static void VsUniformValueBatch(PicaCore& p, u32, const u32* values, u32 count) {
        const auto range = p.vs_setup.WriteUniformFloatRegRange(p.regs.internal.vs, values, count);
        if (range && GsMirrorsVs(p)) {
            p.gs_setup.uniforms_sync_dirty = true;
            p.gs_setup.WidenFloatSyncWindow(range->first_index, range->first_index + range->count);
            for (u32 i = 0; i < range->count; ++i) {
                const u32 idx = range->first_index + i;
                p.gs_setup.uniforms.f[idx] = p.vs_setup.uniforms.f[idx];
            }
        }
    }

    static void VsProgramWord(PicaCore& p, u32, u32 value, bool&) {
        u32& offset = p.regs.internal.vs.program.offset;
        if (offset >= 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {}", offset);
        } else {
            p.vs_setup.UpdateProgramCode(offset, value);
            if (GsMirrorsVs(p)) {
                p.gs_setup.UpdateProgramCode(offset, value);
            }
            offset++;
        }
    }

    static void VsProgramWordBatch(PicaCore& p, u32, const u32* values, u32 count) {
        u32& offset = p.regs.internal.vs.program.offset;
        if (offset + count > 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {} count {}", offset, count);
        } else {
            p.vs_setup.UpdateProgramCodeRange(offset, values, count);
            if (GsMirrorsVs(p)) {
                p.gs_setup.UpdateProgramCodeRange(offset, values, count);
            }
            offset += count;
        }
    }

    static void VsSwizzleWord(PicaCore& p, u32, u32 value, bool&) {
        u32& offset = p.regs.internal.vs.swizzle_patterns.offset;
        if (offset >= p.vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {}", offset);
        } else {
            p.vs_setup.UpdateSwizzleData(offset, value);
            if (GsMirrorsVs(p)) {
                p.gs_setup.UpdateSwizzleData(offset, value);
            }
            offset++;
        }
    }

    static void VsSwizzleWordBatch(PicaCore& p, u32, const u32* values, u32 count) {
        u32& offset = p.regs.internal.vs.swizzle_patterns.offset;
        if (offset + count > p.vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {} count {}", offset, count);
        } else {
            p.vs_setup.UpdateSwizzleDataRange(offset, values, count);
            if (GsMirrorsVs(p)) {
                p.gs_setup.UpdateSwizzleDataRange(offset, values, count);
            }
            offset += count;
        }
    }

    static void LightingLutData(PicaCore& p, u32, u32 value, bool&) {
        auto& lut_config = p.regs.internal.lighting.lut_config;
        const u32 prev =
            std::exchange(p.lighting.luts[lut_config.type][lut_config.index].raw, value);
        p.lighting.lut_dirty |= (prev != value) << lut_config.type;
        lut_config.index.Assign(lut_config.index + 1);
    }

    static void LightingLutDataBatch(PicaCore& p, u32, const u32* values, u32 count) {
        // SM3DL rewrites its lighting LUTs every frame, 256 words at a time: compare and copy
        // the runs up to the wrap with memcmp/memcpy instead of exchanging one word at a time
        // (2.2% of the emulation thread on the pi5 as a scalar loop).
        auto& lut_config = p.regs.internal.lighting.lut_config;
        if (lut_config.type >= p.lighting.luts.size()) [[unlikely]] {
            lut_config.index.Assign(lut_config.index + count);
            return;
        }
        auto& lut = p.lighting.luts[lut_config.type];
        static_assert(sizeof(lut[0]) == sizeof(u32));
        u32 index = lut_config.index;
        u32 done = 0;
        while (done < count) {
            const u32 run = std::min<u32>(count - done, static_cast<u32>(lut.size()) - index);
            if (std::memcmp(&lut[index], values + done, run * sizeof(u32)) != 0) {
                std::memcpy(&lut[index], values + done, run * sizeof(u32));
                p.lighting.lut_dirty |= 1u << lut_config.type;
            }
            index = (index + run) % static_cast<u32>(lut.size());
            done += run;
        }
        lut_config.index.Assign(lut_config.index + count);
    }

    static void FogLutData(PicaCore& p, u32, u32 value, bool&) {
        auto& offset = p.regs.internal.texturing.fog_lut_offset;
        const u32 prev = std::exchange(p.fog.lut[offset % 128].raw, value);
        p.fog.lut_dirty |= prev != value;
        offset.Assign(offset + 1);
    }

    static void FogLutDataBatch(PicaCore& p, u32, const u32* values, u32 count) {
        auto& offset = p.regs.internal.texturing.fog_lut_offset;
        for (u32 i = 0; i < count; i++) {
            const u32 prev = std::exchange(p.fog.lut[(offset + i) % 128].raw, values[i]);
            p.fog.lut_dirty |= prev != values[i];
        }
        offset.Assign(offset + count);
    }

    static void ProcTexLutDataBatch(PicaCore& p, u32, const u32* values, u32 count) {
        auto& index = p.regs.internal.texturing.proctex_lut_config.index;
        const auto lut_table = p.regs.internal.texturing.proctex_lut_config.ref_table.Value();
        const auto sync_lut = [&](auto& proctex_table) {
            for (u32 i = 0; i < count; i++) {
                const u32 prev =
                    std::exchange(proctex_table[(index + i) % proctex_table.size()].raw, values[i]);
                p.proctex.table_dirty |= (prev != values[i]) << u32(lut_table);
            }
        };
        switch (lut_table) {
        case TexturingRegs::ProcTexLutTable::Noise:
            sync_lut(p.proctex.noise_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            sync_lut(p.proctex.color_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            sync_lut(p.proctex.alpha_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            sync_lut(p.proctex.color_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            sync_lut(p.proctex.color_diff_table);
            break;
        }
        index.Assign(index + count);
    }

    static void ProcTexLutData(PicaCore& p, u32 id, u32 value, bool&) {
        ProcTexLutDataBatch(p, id, &value, 1);
    }

    static consteval std::array<PicaCore::SpecialFn, RegsInternal::NUM_REGS> BuildSpecialLUT() {
        std::array<PicaCore::SpecialFn, RegsInternal::NUM_REGS> table{};
        const auto set = [&table](u32 base, u32 n, PicaCore::SpecialFn fn) {
            for (u32 i = 0; i < n; ++i) {
                table[base + i] = fn;
            }
        };
        set(PICA_REG_INDEX(irq_request), 1, &IrqRequest);
        set(PICA_REG_INDEX(pipeline.triangle_topology), 1, &TriangleTopology);
        set(PICA_REG_INDEX(pipeline.restart_primitive), 1, &RestartPrimitive);
        set(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index), 1,
            &DefaultAttributesIndex);
        set(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]), 3,
            &DefaultAttributesValue);
        set(PICA_REG_INDEX(pipeline.command_buffer.trigger[0]), 2, &CommandBufferTrigger);
        set(PICA_REG_INDEX(pipeline.trigger_draw), 1, &TriggerDraw);
        set(PICA_REG_INDEX(pipeline.trigger_draw_indexed), 1, &TriggerDraw);
        set(PICA_REG_INDEX(gs.bool_uniforms), 1, &GsBoolUniforms);
        set(PICA_REG_INDEX(gs.int_uniforms[0]), 4, &GsIntUniform);
        set(PICA_REG_INDEX(gs.uniform_setup.set_value[0]), 8, &GsUniformValue);
        set(PICA_REG_INDEX(gs.program.set_word[0]), 8, &GsProgramWord);
        set(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]), 8, &GsSwizzleWord);
        set(PICA_REG_INDEX(vs.output_mask), 1, &VsOutputMask);
        set(PICA_REG_INDEX(vs.bool_uniforms), 1, &VsBoolUniforms);
        set(PICA_REG_INDEX(vs.int_uniforms[0]), 4, &VsIntUniform);
        set(PICA_REG_INDEX(vs.uniform_setup.set_value[0]), 8, &VsUniformValue);
        set(PICA_REG_INDEX(vs.program.set_word[0]), 8, &VsProgramWord);
        set(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]), 8, &VsSwizzleWord);
        set(PICA_REG_INDEX(lighting.lut_data[0]), 8, &LightingLutData);
        set(PICA_REG_INDEX(texturing.fog_lut_data[0]), 8, &FogLutData);
        set(PICA_REG_INDEX(texturing.proctex_lut_data[0]), 8, &ProcTexLutData);
        return table;
    }

    static consteval std::array<PicaCore::BatchFn, RegsInternal::NUM_REGS> BuildBatchLUT() {
        std::array<PicaCore::BatchFn, RegsInternal::NUM_REGS> table{};
        const auto set = [&table](u32 base, u32 n, PicaCore::BatchFn fn) {
            for (u32 i = 0; i < n; ++i) {
                table[base + i] = fn;
            }
        };
        set(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]), 3,
            &DefaultAttributesValueBatch);
        set(PICA_REG_INDEX(gs.uniform_setup.set_value[0]), 8, &GsUniformValueBatch);
        set(PICA_REG_INDEX(gs.program.set_word[0]), 8, &GsProgramWordBatch);
        set(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]), 8, &GsSwizzleWordBatch);
        set(PICA_REG_INDEX(vs.uniform_setup.set_value[0]), 8, &VsUniformValueBatch);
        set(PICA_REG_INDEX(vs.program.set_word[0]), 8, &VsProgramWordBatch);
        set(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]), 8, &VsSwizzleWordBatch);
        set(PICA_REG_INDEX(lighting.lut_data[0]), 8, &LightingLutDataBatch);
        set(PICA_REG_INDEX(texturing.fog_lut_data[0]), 8, &FogLutDataBatch);
        set(PICA_REG_INDEX(texturing.proctex_lut_data[0]), 8, &ProcTexLutDataBatch);
        return table;
    }
};

static constexpr std::array<PicaCore::SpecialFn, RegsInternal::NUM_REGS> special_lut =
    PicaCore::Dispatch::BuildSpecialLUT();
static constexpr std::array<PicaCore::BatchFn, RegsInternal::NUM_REGS> batch_lut =
    PicaCore::Dispatch::BuildBatchLUT();

static consteval std::array<RegImplInfo, RegsInternal::NUM_REGS> BuildRegImplFlagsLUT() {
    std::array<RegImplInfo, RegsInternal::NUM_REGS> table{};

    // Special and batch registers are the ones with a dispatch handler.
    for (u32 i = 0; i < RegsInternal::NUM_REGS; ++i) {
        if (special_lut[i] != nullptr) {
            table[i].SetNeedsSpecialHandling();
        }
        if (batch_lut[i] != nullptr) {
            table[i].SetSupportsBatch();
        }
    }

    // Build distances to next special register.
    u16 regs_since_special = std::numeric_limits<u16>::max();
    for (size_t i = RegsInternal::NUM_REGS; i-- > 0;) {
        if (table[i].NeedsSpecialHandling()) {
            regs_since_special = 0;
        }
        table[i].SetRegsUntilSpecial(regs_since_special);
        if (regs_since_special != std::numeric_limits<u16>::max()) {
            regs_since_special++;
        }
    }

    return table;
}

static constexpr std::array<RegImplInfo, RegsInternal::NUM_REGS> reg_impl_flags_lut =
    BuildRegImplFlagsLUT();

// Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF
static constexpr std::array<u32, 16> ExpandBitsToBytes = {
    0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff, 0x00ffff00, 0x00ffffff,
    0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff, 0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
};

#if CITRA_ARCH(arm32)
// Thumb-2 command-list fast path, see pica_cmdlist_a32.S. Returns the new command index and
// stores the number of applied commands in *out_count.
extern "C" u32 PicaCmdListFastLoop(const u32* head, u32 index, u32 length, const u16* flags_lut,
                                   const u32* expand_lut, u32* reg_array, u8* dirty_bytes,
                                   u32* out_count);
#endif

/**
 * This is the main loop for processing GPU command lists. On Azahar, it's the most
 * CPU expensive function (excluding the inner Draw calls) due to applications submitting
 * 10-50 command lists per frame, each with hundreds of commands in them. For this reason,
 * it is important that this function is well optimized to reduce the load on the CPU.
 *
 * Each command in the list has the following properties:
 *  - Commands come in [value (32 bit), header (32 bit)] pairs, most of the time.
 *  - The register ID that the 32 bit value should be written to is stored in the header.
 *  - The mask of bits that should be written comes in the header
 *    (to be able to write individual bytes of the 4-byte register)
 *  - Commands can have an extra length N, which means that N extra words follow
 *    after the header word.
 *    - If group_command is set in the header, N sequential registers are
 *      written to starting from the ID + 1 specified in the header.
 *    - If group_command is not set, the same register is written
 *      to with the N extra words. This is used for things like shader uploads
 *      which has a single register ID.
 *
 * Regarding implementation details, we store all register values in an array,
 * as well as a dirty array to indicate which registers have changed since
 * the last draw. Some registers need special handling, as they are "trigger"
 * registers that start the draw, or store the data in the shader units.
 *
 * To be able to determine if a register is special, we use a lookup table
 * generated by BuildRegImplFlagsLUT(). This allows determining if a register
 * is special or not in O(1). This LUT also determines if a register has
 * support for batch handling (extra_data_length != 0 && group_command == 0)
 * and the amount of commands away from the next special register, useful for
 * sequential writes (extra_data_length != 0 && group_command == 1).
 *
 * As much as possible, we want to target the following optimizations:
 *  - We should prevent branches and jumps to functions if they are not needed.
 *  - We should clearly separate special command handling from normal commands
 *    that are much cheaper to handle.
 *  - Commands with extra length should be processed in batch if possible.
 *  - Vectorization should be used as much as possible.
 *
 * On the other hand, if PICA debugging is enabled we should avoid optimizations
 * that would make debugging more complicated.
 */
void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) [[hot]] {
    g_pica_probe[0].fetch_add(1, std::memory_order_relaxed);
    const u64 irqw_before = g_pica_probe[1].load(std::memory_order_relaxed);
    const u64 p3d_before = g_pica_probe[2].load(std::memory_order_relaxed);
    const u64 chain_before = g_pica_probe[4].load(std::memory_order_relaxed);
    const auto record_list = [&](u32 end_index, u32 end_length, bool stopped) {
        auto& rec =
            g_list_ring[g_list_ring_head.fetch_add(1, std::memory_order_relaxed) % g_list_ring.size()];
        rec.addr = list;
        rec.size = size;
        rec.end_index = end_index;
        rec.end_length = end_length;
        rec.irqw = static_cast<u32>(g_pica_probe[1].load(std::memory_order_relaxed) - irqw_before);
        rec.p3d = static_cast<u32>(g_pica_probe[2].load(std::memory_order_relaxed) - p3d_before);
        rec.chains =
            static_cast<u32>(g_pica_probe[4].load(std::memory_order_relaxed) - chain_before);
        rec.stopped = stopped ? 1 : 0;
        rec.last_chain_addr = g_last_chain_addr;
        rec.last_chain_size = g_last_chain_size;
        g_last_chain_addr = 0;
        g_last_chain_size = 0;
    };
    if (ignore_list) {
        if (rasterizer && rasterizer->DefersInterrupts()) {
            rasterizer->PostInterruptAfterQueue(Service::GSP::InterruptId::P3D,
                                                delay_generator.CalculateAndResetDelay());
        } else {
            signal_interrupt(Service::GSP::InterruptId::P3D,
                             delay_generator.CalculateAndResetDelay());
        }
        record_list(0, 0, false);
        return;
    }

    const u8* head = memory.GetPhysicalPointer(list);
    bool stop_requested = false;
    cmd_list.Reset(list, head, size);

    bool skip_fast_path = false;
    while (cmd_list.current_index < cmd_list.length) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Early path that processes commands in batches of 4. If any of the commands
        // needs special handling or has extra length it stops and falls back to the
        // slower path. This pattern allows the compiler to auto-vectorize the function
        // if the current ISA allows it (that's why we process in batches of 4
        // as most SIMD operations work with 128 bit registers). MSVC is not able to
        // auto-vectorize this part with SSE4.2, due to the LUT read, instead it just
        // unrolls the loop. Other ISAs and/or compilers may be able to do it,
        // that's why it was decided to keep the structure like this.
        if (!debug_context) [[likely]] {
            if (!skip_fast_path) {
#if CITRA_ARCH(arm32)
                // Hand-written Thumb-2 loop (pica_cmdlist_a32.S): applies plain register
                // writes until list end or a special command, which falls through to the
                // slow path below with current_index already advanced past the batch.
                u32 run = 0;
                const u32 index = PicaCmdListFastLoop(
                    cmd_list.head, cmd_list.current_index, cmd_list.length,
                    reinterpret_cast<const u16*>(reg_impl_flags_lut.data()),
                    ExpandBitsToBytes.data(), regs.internal.reg_array.data(),
                    reinterpret_cast<u8*>(dirty_regs.qwords.data()), &run);
                if (run > 0) {
                    delay_generator.AddCommands(run);
                    cmd_list.current_index = index;
                    if (index + 1 >= cmd_list.length) {
                        continue;
                    }
                }
#else
                constexpr u32 batch_size = 4;
                u32 index = cmd_list.current_index;
                u32 ids[batch_size], values[batch_size], masks[batch_size];
                u32 run = 0;

                while (run < batch_size && index + 1 < cmd_list.length) {
                    const u32 value = cmd_list.head[index];
                    const CommandHeader header{cmd_list.head[index + 1]};

                    // If extra handling is needed stop and fallback to slower path.
                    if (header.extra_data_length != 0 || header.cmd_id >= RegsInternal::NUM_REGS ||
                        reg_impl_flags_lut[header.cmd_id].NeedsSpecialHandling()) {
                        skip_fast_path = true;
                        break;
                    }

                    ids[run] = header.cmd_id;
                    values[run] = value;
                    masks[run] = header.parameter_mask;
                    ++run;
                    index += 2;
                }

                // Process the commands that we have read so far (up to 4).
                if (run > 0) {
                    delay_generator.AddCommands(run);
                    for (u32 i = 0; i < run; ++i) {
                        const u32 id = ids[i];
                        const u32 write_mask = ExpandBitsToBytes[masks[i]];
                        regs.internal.reg_array[id] =
                            (regs.internal.reg_array[id] & ~write_mask) | (values[i] & write_mask);
                        dirty_regs.Set(id);
                    }
                    cmd_list.current_index = index;

                    // Continue from the while loop in case we reached the end of the list.
                    continue;
                }
#endif
            }
        }
        // Slow path, command needs special handling.

        skip_fast_path = false;

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        WriteInternalReg(header.cmd_id, value, header.parameter_mask, stop_requested);

        // Write any extra paramters as well. A header may declare more extra words than the list
        // actually has left; taking it at its word walks the parser off the end of the buffer,
        // past the `irq_request` that terminates the list, and on into whatever follows. The list
        // then never raises its completion interrupt and never chains, and the guest waits for a
        // frame that can no longer arrive.
        const u32 remaining = cmd_list.length - cmd_list.current_index;
        const u32 count = std::min(header.extra_data_length.Value(), remaining);
        if (count != header.extra_data_length) {
            g_cmdlist_overrun.fetch_add(1, std::memory_order_relaxed);
        }
        if (count == 0)
            continue;

        if (debug_context) [[unlikely]] {
            // Fallback to per word register writes if debugging is
            // enabled.
            for (u32 i = 0; i < count; ++i) {
                if (stop_requested) [[unlikely]] {
                    break;
                }
                const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
                const u32 extra_value = cmd_list.head[cmd_list.current_index++];
                WriteInternalReg(cmd, extra_value, header.parameter_mask, stop_requested);
            }
        } else {
            // Handle commands with extra length.
            const u32* extra = &cmd_list.head[cmd_list.current_index];
            cmd_list.current_index += count;

            if (!header.group_commands) {
                // Same register written count times in a row (program/swizzle upload, LUT, etc.).
                WriteInternalRegBatch(header.cmd_id, extra, count, header.parameter_mask,
                                      stop_requested);
            } else {
                // Sequential registers written from header.cmd_id+1 to header.cmd_id+count.
                WriteInternalRegSequential(header.cmd_id + 1, extra, count, header.parameter_mask,
                                           stop_requested);
            }
        }
    }
    record_list(cmd_list.current_index, cmd_list.length, stop_requested);
}


// Writes to OOB registers are no-ops. The log call carries a fmt frame; keeping it out of
// line keeps the hot writers' prologues small.
[[gnu::cold, gnu::noinline]] static void LogOutOfRangeWrite(u32 id, u32 count, u32 mask) {
    LOG_DEBUG(HW_GPU,
              "Commandlist tried to write to invalid register 0x{:03X} repeated 0x{:04X} times"
              "(mask: {:X})",
              id, count, mask);
}

// Handle batch register writes
void PicaCore::WriteInternalRegBatch(u32 id, const u32* values, u32 count, u32 mask,
                                     bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        LogOutOfRangeWrite(id, count, mask);
        return;
    }

    delay_generator.AddCommands(count);
    const u32 write_mask = ExpandBitsToBytes[mask];
    // Only write the last value to the register array.
    // Batch handlers should take this in mind.
    regs.internal.reg_array[id] =
        (regs.internal.reg_array[id] & ~write_mask) | (values[count - 1] & write_mask);
    dirty_regs.Set(id);

    if (const BatchFn batch = batch_lut[id]) {
        // If the register supports batch then call the handler.
        batch(*this, id, values, count);
    } else if (const SpecialFn special = special_lut[id]) [[unlikely]] {
        // Unlikely as all special regs that make sense to use batch mode already
        // support batch handling.
        for (u32 i = 0; i < count && !stop_requested; ++i) {
            special(*this, id, values[i], stop_requested);
        }
    }
}

// Handle sequential register writes.
void PicaCore::WriteInternalRegSequential(u32 id, const u32* __restrict values, u32 count, u32 mask,
                                          bool& stop_requested) {
    if (id + count > RegsInternal::NUM_REGS) [[unlikely]] {
        LogOutOfRangeWrite(id, count, mask);
        if (id >= RegsInternal::NUM_REGS) {
            return;
        } else {
            count = RegsInternal::NUM_REGS - id;
        }
    }
    const u32 write_mask = ExpandBitsToBytes[mask];
    u32* __restrict dst = &regs.internal.reg_array[id];
    u32 offset = 0;

    // This code is structured so that it uses the LUT to get the distance from the
    // register ID to the next special register. Then copies the range of normal registers
    // until it reaches the special register which is handled individually, and so on.
    while (offset < count) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        const u32 reg = id + offset;
        // Distance to the next special register, capped by how many writes
        // are actually left in this write. If this register is special this is 0.
        const u32 batch_count =
            std::min<u32>(reg_impl_flags_lut[reg].RegsUntilSpecial(), count - offset);
        if (batch_count > 0) {
            delay_generator.AddCommands(batch_count);

            // Allows the compiler to auto-vectorize thanks to the __restrict keywords.
            // Verified in MSVC that vectorization is happening; GCC for ARM32 did not
            // (2026-09-08, not one vector instruction in the function), so the four-wide
            // blend is written out for it. The register file is not 16-byte aligned:
            // unaligned vector loads.
            u32* __restrict batch_dst = dst + offset;
            const u32* __restrict batch_src = values + offset;
            u32 i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            const uint32x4_t mask_v = vdupq_n_u32(write_mask);
            for (; i + 4 <= batch_count; i += 4) {
                const uint32x4_t d = vld1q_u32(batch_dst + i);
                const uint32x4_t v = vld1q_u32(batch_src + i);
                vst1q_u32(batch_dst + i, vbslq_u32(mask_v, v, d));
            }
#endif
            for (; i < batch_count; ++i) {
                batch_dst[i] = (batch_dst[i] & ~write_mask) | (batch_src[i] & write_mask);
            }

            dirty_regs.SetRange(reg, batch_count);
            offset += batch_count;
        }
        // Whatever register stopped the batch (if we didn't reach the end)
        // needs individual handling, then resume batching after it.
        if (offset < count) {
            WriteInternalReg(id + offset, values[offset], mask, stop_requested);
            ++offset;
        }
    }
}

// Handle individual command write.
void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask, bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        LogOutOfRangeWrite(id, 1, mask);
        return;
    }

    delay_generator.AddCommands(1);

    // TODO: Figure out how register masking acts on e.g. vs.uniform_setup.set_value
    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    regs.internal.reg_array[id] = (old_value & ~write_mask) | (value & write_mask);

    if (debug_context) [[unlikely]] {
        // Track register write.
        DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);
        // Track events.
        debug_context->OnEvent(DebugContext::Event::PicaCommandLoaded, &id);
    }

    if (const SpecialFn special = special_lut[id]) {
        special(*this, id, value, stop_requested);
    }

    dirty_regs.Set(id);

    if (debug_context) [[unlikely]] {
        debug_context->OnEvent(DebugContext::Event::PicaCommandProcessed, &id);
    }
}

void PicaCore::SubmitImmediate(u32 value) {
    // Push to word to the queue. This returns true when a full attribute is formed.
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t IMMEDIATE_MODE_INDEX = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > IMMEDIATE_MODE_INDEX) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    // Retrieve the attribute and place it in the default attribute buffer.
    const auto attribute = immediate.queue.Get();
    if (setup.index < IMMEDIATE_MODE_INDEX) {
        input_default_attributes[setup.index] = attribute;
        default_attributes_sync_dirty = true;
        setup.index++;
        return;
    }

    // When index is 0xF the attribute is used for immediate mode drawing.
    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    // We formed a vertex, flush.
    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    if (rasterizer->ConsumesShippedDraws()) {
        // Immediate-mode input is already assembled in emulator memory; ship it by value.
        DrawPayload payload;
        payload.immediate = true;
        payload.immediate_input = immediate.input_vertex;
        payload.immediate_reset_geometry = immediate.reset_geometry_pipeline;
        immediate.reset_geometry_pipeline = false;
        rasterizer->ShipDraw(std::move(payload), nullptr);
        immediate.current_attribute = 0;
        return;
    }

    // Compile the vertex shader.
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    output_vertex_map.Build(regs.internal.rasterizer);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    // Invoke the vertex shader for the vertex.
    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    // Reconfigure geometry pipeline if needed.
    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    // Send to geometry pipeline.
    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

    // Flush the immediate triangle.
    rasterizer->DrawTriangles();
    immediate.current_attribute = 0;

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

void PicaCore::DrawArrays(bool is_indexed) {
    MICROPROFILE_SCOPE(GPU_Drawing);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
    }

    const bool accelerate_draw = [this] {
        // Geometry shaders cannot be accelerated due to register preservation.
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        // TODO (wwylele): for Strip/Fan topology, if the primitive assember is not restarted
        // after this draw call, the buffered vertex from this draw should "leak" to the next
        // draw, in which case we should buffer the vertex into the software primitive assember,
        // or disable accelerate draw completely. However, there is not game found yet that does
        // this, so this is left unimplemented for now. Revisit this when an issue is found in
        // games.

        bool accelerate_draw = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            accelerate_draw = accelerate_draw && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return accelerate_draw;
    }();

    // Add vertices to the delay generator.
    delay_generator.AddVertices(regs.internal.pipeline.num_vertices,
                                regs.internal.pipeline.triangle_topology);

    // Attempt to use hardware vertex shaders if possible. Under the offload the draw ships with
    // its arena and the GLSL path runs against the mirror; the software (NEON) pipeline stays as
    // the automatic fallback for anything the hardware path declines.
    if (accelerate_draw && rasterizer->ConsumesShippedDraws()) {
        if (TryShipDraw(is_indexed, true)) {
            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
            }
            return;
        }
    } else if (accelerate_draw && rasterizer->AccelerateDrawBatch(is_indexed)) {
        return;
    }

    // We cannot accelerate the draw, so load and execute the vertex shader for each vertex.
    LoadVertices(is_indexed);

    // Draw emitted triangles.
    rasterizer->DrawTriangles();

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

bool PicaCore::TryShipDraw(bool is_indexed, bool hw) {
    const auto& pipeline = regs.internal.pipeline;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* const index_address_8 = memory.GetPhysicalPointer(base_address + index_info.offset);
    if (index_address_8 == nullptr || pipeline.num_vertices == 0) {
        return false;
    }
    const u16* const index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    auto& loader = GetVertexLoader();
    u32 vertex_min = pipeline.vertex_offset;
    u32 vertex_max = pipeline.vertex_offset + pipeline.num_vertices - 1;
    const u32 index_bytes = is_indexed ? pipeline.num_vertices * (index_u16 ? 2 : 1) : 0;
    if (is_indexed) {
        if (index_u16) {
            const auto [min, max] = Common::FindMinMax(
                std::span<const u16>{index_address_16, pipeline.num_vertices});
            vertex_min = min;
            vertex_max = max;
        } else {
            const auto [min, max] =
                Common::FindMinMax(std::span<const u8>{index_address_8, pipeline.num_vertices});
            vertex_min = min;
            vertex_max = max;
        }
    }
    DrawArenaLayout layout = loader.ComputeArenaLayout(vertex_min, vertex_max, index_bytes);
    // The GL vertex-array path cannot express a zero-stride loader; those draws shade in
    // software.
    if (hw && layout.zero_stride) {
        return false;
    }
    // Garbage configs (boot-time junk draws) can make range x stride explode — one boot draw
    // asked for 631 MB. Real draws stay in the low megabytes; anything bigger shades locally
    // where cost scales with the vertex count instead of the index range.
    constexpr u32 MAX_ARENA_BYTES = 16 * 1024 * 1024;
    if (layout.total > MAX_ARENA_BYTES) {
        return false;
    }
    // A hardware draw would rather be written once, straight into the renderer's GPU-visible
    // ring in the layout its vertex streams read, than into an arena the render thread copies
    // into that ring; the facade grants it when the ring has room.
    const DrawArenaLayout ring_layout =
        hw ? loader.ComputeArenaLayout(vertex_min, vertex_max,
                                       is_indexed ? pipeline.num_vertices * 2 : 0, true)
           : layout;
    auto ticket = rasterizer->BeginShippedDraw(
        layout.total, regs.internal.framebuffer.framebuffer.height + 1 > 350,
        hw ? ring_layout.total : 0);
    if (!ticket) {
        return false;
    }
    if (ticket->data == nullptr) {
        return true; // Skipped before any snapshot cost.
    }
    if (ticket->in_ring) {
        layout = ring_layout;
    }
    loader.CopyIntoArena(base_address, vertex_min, vertex_max, layout, ticket->data);
    if (index_bytes != 0) {
        if (layout.ring) {
            VertexLoader::WriteRingIndices(ticket->data + layout.index_offset, index_address_8,
                                           index_u16, pipeline.num_vertices, vertex_min);
        } else {
            std::memcpy(ticket->data + layout.index_offset, index_address_8, index_bytes);
        }
    }
    DrawPayload payload;
    payload.arena = ticket->data;
    payload.layout = layout;
    payload.in_ring = ticket->in_ring;
    payload.ring_offset = ticket->ring_offset;
    payload.ring_end = ticket->ring_end;
    payload.vertex_min = vertex_min;
    payload.vertex_max = vertex_max;
    payload.num_vertices = pipeline.num_vertices;
    payload.vertex_offset = pipeline.vertex_offset;
    payload.is_indexed = is_indexed;
    payload.index_u16 = index_u16;
    payload.hw = hw;
    rasterizer->ShipDraw(std::move(payload), ticket->slot);
    return true;
}

VertexLoader& PicaCore::GetVertexLoader() {
    const auto* key = reinterpret_cast<const u8*>(&regs.internal.pipeline.vertex_attributes);
    if (!cached_vertex_loader ||
        std::memcmp(cached_loader_key.data(), key, cached_loader_key.size()) != 0) {
        std::memcpy(cached_loader_key.data(), key, cached_loader_key.size());
        cached_vertex_loader.emplace(memory, regs.internal.pipeline);
    }
    return *cached_vertex_loader;
}

void PicaCore::LoadVertices(bool is_indexed) {
    // Read and validate vertex information from the loaders
    const auto& pipeline = regs.internal.pipeline;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    auto& loader = GetVertexLoader();
    regs.internal.rasterizer.ValidateSemantics();
    output_vertex_map.Build(regs.internal.rasterizer);

    // Locate index buffer.
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GetPhysicalPointer(base_address + index_info.offset);
    if (index_address_8 == nullptr) {
        // Mario & Luigi: Superstar Saga sets an invalid base address
        // for the vertex attributes. Return early if that is the case.
        return;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    if (pipeline.num_vertices == 0) {
        return;
    }

    // Offload path: freeze this draw — index extremes, then exactly the attribute and index
    // bytes it can reference — into a render-thread arena and ship it. Shading, primitive
    // assembly and the rasterizer then all run against the mirror, and this thread is done with
    // the draw the moment the copy finishes. BeginShippedDraw is also the frameskip gate: a
    // skipped draw returns a null ticket before any copy is paid.
    if (TryShipDraw(is_indexed, false)) {
        return;
    }

    // Local path: resolve the attribute arrays to host pointers once for the whole draw;
    // fetches below are then a pointer add each.
    loader.Rebase(base_address);

    // Direct-mapped vertex cache, slot = index & 63: one probe instead of the 64-entry linear
    // scan, which alone measured 28% of the emulation thread in-match. A collision just evicts;
    // mesh indices are near-sequential, so slots rarely collide inside the reuse window anyway.
    // The sentinel doubles as the invalid flag — no index is 0xFFFFFFFF.
    constexpr std::size_t VERTEX_CACHE_SIZE = 64;
    std::array<u32, VERTEX_CACHE_SIZE> vertex_cache_ids;
    vertex_cache_ids.fill(0xFFFFFFFF);
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;

    // Compile the vertex shader for this batch.
    ShaderUnit shader_unit;
    AttributeBuffer vs_output;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Setup geometry pipeline in case we are using a geometry shader.
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        // Indexed rendering doesn't use the start offset
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);

        bool vertex_cache_hit = false;
        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            const u32 slot = vertex & (VERTEX_CACHE_SIZE - 1);
            if (vertex_cache_ids[slot] == vertex) {
                vs_output = vertex_cache[slot];
                vertex_cache_hit = true;
            }
        }

        if (!vertex_cache_hit) {
            // Initialize data for the current vertex
            AttributeBuffer input;
            loader.LoadVertex(vertex, input, input_default_attributes);

            // Record vertex processing to the debugger.
            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            // Invoke the vertex shader for this vertex.
            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            // Cache the vertex when doing indexed rendering.
            if (is_indexed) {
                const u32 slot = vertex & (VERTEX_CACHE_SIZE - 1);
                vertex_cache[slot] = vs_output;
                vertex_cache_ids[slot] = vertex;
            }
        }

        // Send to geometry pipeline
        geometry_pipeline.SubmitVertex(vs_output);
    }
}

#ifdef CITRA_TRACE_PROBES
static void DrawTraceRunProbe() {
    static const bool trace = std::getenv("AZAHAR_DRAW_TRACE") != nullptr;
    static std::atomic<u32> n{0};
    if (trace && n.fetch_add(1) < 3000) {
        LOG_INFO(HW_GPU, "DRAWTRACE run");
    }
}
#endif // CITRA_TRACE_PROBES

// Command headers that declared more extra words than the list had left; the clamp in
// ProcessCmdList keeps the parse inside the buffer, this makes the (rare) occurrences visible.
std::atomic<u64> g_cmdlist_overrun{};

// Parse-side tallies, surfaced by the debug port's `gsp` dump:
// [0] ProcessCmdList calls  [1] irq_request writes  [2] P3D raised  [3] irq compare mismatches
// [4] chain trigger writes  [5] autostop stops
// A healthy title keeps [0] and [1] in step; a list that parses without raising its interrupt is
// the fingerprint of the class of bug behind the 2026-08 software-renderer freeze.
std::array<std::atomic<u64>, 6> g_pica_probe{};

// Per-list parse summaries, newest last, read through the debug port's `lists` command. Each
// ProcessCmdList call appends one record when it finishes; the deltas of g_pica_probe across the
// call fill the flags. Writes happen only on the emulation thread; the debug port reads on the
// same thread.
std::array<ListRec, 1024> g_list_ring{};
std::atomic<u32> g_list_ring_head{};


void PicaCore::RunShippedDraw(const DrawPayload& p) {
#ifdef CITRA_TRACE_PROBES
    DrawTraceRunProbe();
#endif // CITRA_TRACE_PROBES
    if (p.immediate) {
        shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
        output_vertex_map.Build(regs.internal.rasterizer);

        ShaderUnit shader_unit;
        AttributeBuffer output{};
        shader_unit.LoadInput(regs.internal.vs, p.immediate_input);
        shader_engine->Run(vs_setup, shader_unit);
        shader_unit.WriteOutput(regs.internal.vs, output);

        if (p.immediate_reset_geometry) {
            geometry_pipeline.Reconfigure();
        }
        ASSERT(!geometry_pipeline.NeedIndexInput());
        geometry_pipeline.Setup(shader_engine.get());
        geometry_pipeline.SubmitVertex(output);

        rasterizer->DrawTriangles();
        return;
    }

    // The mirror's registers, shader setup and LUTs were applied by the shipment this payload
    // rode in on; the loader reconstructs the identical fetch configuration from them and reads
    // only the arena.
    const auto& pipeline = regs.internal.pipeline;
    auto& loader = GetVertexLoader();
    regs.internal.rasterizer.ValidateSemantics();
    output_vertex_map.Build(regs.internal.rasterizer);
    loader.AttachArena(p.arena, p.layout, p.vertex_min);

    const u8* const index_address_8 = p.arena + p.layout.index_offset;
    const u16* const index_address_16 = reinterpret_cast<const u16*>(index_address_8);

    constexpr std::size_t VERTEX_CACHE_SIZE = 64;
    std::array<u32, VERTEX_CACHE_SIZE> vertex_cache_ids;
    vertex_cache_ids.fill(0xFFFFFFFF);
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;

    ShaderUnit shader_unit;
    AttributeBuffer vs_output;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || p.is_indexed);

    for (u32 index = 0; index < p.num_vertices; ++index) {
        // A ring-layout arena holds 16-bit indices rebased to the draw's first vertex.
        const u32 vertex =
            p.is_indexed ? (p.layout.ring ? index_address_16[index] + p.vertex_min
                            : p.index_u16 ? index_address_16[index]
                                          : index_address_8[index])
                         : (index + p.vertex_offset);

        bool vertex_cache_hit = false;
        if (p.is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            const u32 slot = vertex & (VERTEX_CACHE_SIZE - 1);
            if (vertex_cache_ids[slot] == vertex) {
                vs_output = vertex_cache[slot];
                vertex_cache_hit = true;
            }
        }

        if (!vertex_cache_hit) {
            AttributeBuffer input;
            loader.LoadVertex(vertex, input, input_default_attributes);

            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);


            if (p.is_indexed) {
                const u32 slot = vertex & (VERTEX_CACHE_SIZE - 1);
                vertex_cache[slot] = vs_output;
                vertex_cache_ids[slot] = vertex;
            }
        }

        geometry_pipeline.SubmitVertex(vs_output);
    }

    rasterizer->DrawTriangles();
}

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
    // Initialize command list tracking.
    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    constexpr size_t max_iterations = 0x100;

    RenderPropertiesGuess find_info{};

    find_info.vp_height = regs.internal.rasterizer.viewport_size_y.Value();
    find_info.paddr = regs.internal.framebuffer.framebuffer.color_buffer_address.Value() * 8;

    auto process_write = [this, &find_info](u32 cmd_id, u32 value) {
        switch (cmd_id) {
        case PICA_REG_INDEX(rasterizer.viewport_size_y):
            find_info.vp_height = value;
            find_info.vp_heigh_found = true;
            break;
        case PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address):
            find_info.paddr = value * 8;
            find_info.paddr_found = true;
            break;
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
            const u32 index =
                static_cast<u32>(cmd_id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
            const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
            const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
            const u8* head = memory.GetPhysicalPointer(addr);
            cmd_list.Reset(addr, head, size);
            break;
        }
        default:
            break;
        }
        return find_info.vp_heigh_found && find_info.paddr_found;
    };

    size_t iterations = 0;
    while (cmd_list.current_index < cmd_list.length && iterations < max_iterations) {
        // Align read pointer to 8 bytes
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        if (process_write(header.cmd_id, value))
            break;

        // Write any extra paramters as well.
        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];
            if (process_write(cmd, extra_value))
                break;
        }

        iterations++;
    }

    return find_info;
}

template <class Archive>
void PicaCore::CommandList::serialize(Archive& ar, const u32 file_version) {
    ar & addr;
    ar & length;
    ar & current_index;
    if (Archive::is_loading::value) {
        const u8* ptr = Core::System::GetInstance().Memory().GetPhysicalPointer(addr);
        head = reinterpret_cast<const u32*>(ptr);
    }
}

SERIALIZE_IMPL(PicaCore::CommandList)

} // namespace Pica
