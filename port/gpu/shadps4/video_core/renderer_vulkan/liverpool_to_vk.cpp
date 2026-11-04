// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/number_utils.h"
#include "native/ps4_data_layout/pixel_format.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

#include <magic_enum/magic_enum.hpp>

#define INVALID_NUMBER_FORMAT_COMBO                                                                \
    LOG_ERROR(Render_Vulkan, "Unsupported number type {} for format {}", number_type, format);

namespace Vulkan::LiverpoolToVK {

using DepthBuffer = Core::NativeTarget::DepthTarget;

vk::StencilOp StencilOp(AmdGpu::StencilFunc op) {
    switch (op) {
    case AmdGpu::StencilFunc::Keep:
        return vk::StencilOp::eKeep;
    case AmdGpu::StencilFunc::Zero:
        return vk::StencilOp::eZero;
    case AmdGpu::StencilFunc::ReplaceTest:
        return vk::StencilOp::eReplace;
    case AmdGpu::StencilFunc::AddClamp:
        return vk::StencilOp::eIncrementAndClamp;
    case AmdGpu::StencilFunc::SubClamp:
        return vk::StencilOp::eDecrementAndClamp;
    case AmdGpu::StencilFunc::Invert:
        return vk::StencilOp::eInvert;
    case AmdGpu::StencilFunc::AddWrap:
        return vk::StencilOp::eIncrementAndWrap;
    case AmdGpu::StencilFunc::SubWrap:
        return vk::StencilOp::eDecrementAndWrap;
    case AmdGpu::StencilFunc::ReplaceOp:
        return vk::StencilOp::eReplace;
    case AmdGpu::StencilFunc::Ones:
        LOG_WARNING(Render_Vulkan, "Unsupported stencil op {}, using Replace.",
                    static_cast<u32>(op));
        return vk::StencilOp::eReplace;
    case AmdGpu::StencilFunc::And:
    case AmdGpu::StencilFunc::Or:
    case AmdGpu::StencilFunc::Xor:
    case AmdGpu::StencilFunc::Nand:
    case AmdGpu::StencilFunc::Nor:
    case AmdGpu::StencilFunc::Xnor:
        // Bitwise stencil operations have no Vulkan equivalent; eKeep is the safest fallback.
        LOG_WARNING(Render_Vulkan, "Unsupported bitwise stencil op {}, using Keep.",
                    static_cast<u32>(op));
        return vk::StencilOp::eKeep;
    default:
        UNREACHABLE();
        return vk::StencilOp::eKeep;
    }
}

vk::CompareOp CompareOp(Core::NativeTarget::CompareFunc func) {
    switch (func) {
    case Core::NativeTarget::CompareFunc::Always:
        return vk::CompareOp::eAlways;
    case Core::NativeTarget::CompareFunc::Equal:
        return vk::CompareOp::eEqual;
    case Core::NativeTarget::CompareFunc::GreaterEqual:
        return vk::CompareOp::eGreaterOrEqual;
    case Core::NativeTarget::CompareFunc::Greater:
        return vk::CompareOp::eGreater;
    case Core::NativeTarget::CompareFunc::LessEqual:
        return vk::CompareOp::eLessOrEqual;
    case Core::NativeTarget::CompareFunc::Less:
        return vk::CompareOp::eLess;
    case Core::NativeTarget::CompareFunc::NotEqual:
        return vk::CompareOp::eNotEqual;
    case Core::NativeTarget::CompareFunc::Never:
        return vk::CompareOp::eNever;
    default:
        UNREACHABLE();
        return vk::CompareOp::eAlways;
    }
}

bool IsPrimitiveCulled(Core::NativePrimitive::Kind type) {
    switch (type) {
    case Core::NativePrimitive::Kind::TriangleList:
    case Core::NativePrimitive::Kind::TriangleFan:
    case Core::NativePrimitive::Kind::TriangleStrip:
    case Core::NativePrimitive::Kind::PatchPrimitive:
    case Core::NativePrimitive::Kind::AdjTriangleList:
    case Core::NativePrimitive::Kind::AdjTriangleStrip:
    case Core::NativePrimitive::Kind::QuadList:
    case Core::NativePrimitive::Kind::QuadStrip:
    case Core::NativePrimitive::Kind::Polygon:
        return true;
    case Core::NativePrimitive::Kind::None:
    case Core::NativePrimitive::Kind::PointList:
    case Core::NativePrimitive::Kind::LineList:
    case Core::NativePrimitive::Kind::LineStrip:
    case Core::NativePrimitive::Kind::AdjLineList:
    case Core::NativePrimitive::Kind::AdjLineStrip:
    case Core::NativePrimitive::Kind::RectList: // Screen-aligned rectangles that are not culled
    case Core::NativePrimitive::Kind::LineLoop:
        return false;
    default:
        UNREACHABLE();
        return true;
    }
}

vk::PrimitiveTopology PrimitiveType(Core::NativePrimitive::Kind type) {
    switch (type) {
    case Core::NativePrimitive::Kind::PointList:
        return vk::PrimitiveTopology::ePointList;
    case Core::NativePrimitive::Kind::LineList:
        return vk::PrimitiveTopology::eLineList;
    case Core::NativePrimitive::Kind::LineStrip:
        return vk::PrimitiveTopology::eLineStrip;
    case Core::NativePrimitive::Kind::TriangleList:
        return vk::PrimitiveTopology::eTriangleList;
    case Core::NativePrimitive::Kind::TriangleFan:
    case Core::NativePrimitive::Kind::Polygon:
        return vk::PrimitiveTopology::eTriangleFan;
    case Core::NativePrimitive::Kind::TriangleStrip:
        return vk::PrimitiveTopology::eTriangleStrip;
    case Core::NativePrimitive::Kind::QuadStrip:
        LOG_WARNING(Render_Vulkan, "Unimplemented primitive type, using TriangleStrip.");
        return vk::PrimitiveTopology::eTriangleStrip;
    case Core::NativePrimitive::Kind::AdjLineList:
        return vk::PrimitiveTopology::eLineListWithAdjacency;
    case Core::NativePrimitive::Kind::AdjLineStrip:
        return vk::PrimitiveTopology::eLineStripWithAdjacency;
    case Core::NativePrimitive::Kind::AdjTriangleList:
        return vk::PrimitiveTopology::eTriangleListWithAdjacency;
    case Core::NativePrimitive::Kind::AdjTriangleStrip:
        return vk::PrimitiveTopology::eTriangleStripWithAdjacency;
    case Core::NativePrimitive::Kind::PatchPrimitive:
        return vk::PrimitiveTopology::ePatchList;
    case Core::NativePrimitive::Kind::QuadList:
    case Core::NativePrimitive::Kind::RectList:
        return vk::PrimitiveTopology::ePatchList;
    default:
        UNREACHABLE_MSG("Unimplemented primitive type: {}", static_cast<u32>(type));
        return vk::PrimitiveTopology::eTriangleList;
    }
}

vk::PolygonMode PolygonMode(AmdGpu::PolygonMode mode) {
    switch (mode) {
    case AmdGpu::PolygonMode::Point:
        return vk::PolygonMode::ePoint;
    case AmdGpu::PolygonMode::Line:
        return vk::PolygonMode::eLine;
    case AmdGpu::PolygonMode::Fill:
        return vk::PolygonMode::eFill;
    default:
        UNREACHABLE();
        return vk::PolygonMode::eFill;
    }
}

vk::CullModeFlags CullMode(AmdGpu::CullMode mode) {
    switch (mode) {
    case AmdGpu::CullMode::None:
        return vk::CullModeFlagBits::eNone;
    case AmdGpu::CullMode::Front:
        return vk::CullModeFlagBits::eFront;
    case AmdGpu::CullMode::Back:
        return vk::CullModeFlagBits::eBack;
    case AmdGpu::CullMode::FrontAndBack:
        return vk::CullModeFlagBits::eFrontAndBack;
    default:
        UNREACHABLE();
        return vk::CullModeFlagBits::eNone;
    }
}

vk::FrontFace FrontFace(AmdGpu::FrontFace face) {
    switch (face) {
    case AmdGpu::FrontFace::Clockwise:
        return vk::FrontFace::eClockwise;
    case AmdGpu::FrontFace::CounterClockwise:
        return vk::FrontFace::eCounterClockwise;
    default:
        UNREACHABLE();
        return vk::FrontFace::eClockwise;
    }
}

vk::BlendFactor BlendFactor(AmdGpu::BlendControl::BlendFactor factor) {
    using BlendFactor = AmdGpu::BlendControl::BlendFactor;
    switch (factor) {
    case BlendFactor::Zero:
        return vk::BlendFactor::eZero;
    case BlendFactor::One:
        return vk::BlendFactor::eOne;
    case BlendFactor::SrcColor:
        return vk::BlendFactor::eSrcColor;
    case BlendFactor::OneMinusSrcColor:
        return vk::BlendFactor::eOneMinusSrcColor;
    case BlendFactor::SrcAlpha:
        return vk::BlendFactor::eSrcAlpha;
    case BlendFactor::OneMinusSrcAlpha:
        return vk::BlendFactor::eOneMinusSrcAlpha;
    case BlendFactor::DstAlpha:
        return vk::BlendFactor::eDstAlpha;
    case BlendFactor::OneMinusDstAlpha:
        return vk::BlendFactor::eOneMinusDstAlpha;
    case BlendFactor::DstColor:
        return vk::BlendFactor::eDstColor;
    case BlendFactor::OneMinusDstColor:
        return vk::BlendFactor::eOneMinusDstColor;
    case BlendFactor::SrcAlphaSaturate:
        return vk::BlendFactor::eSrcAlphaSaturate;
    case BlendFactor::ConstantColor:
        return vk::BlendFactor::eConstantColor;
    case BlendFactor::OneMinusConstantColor:
        return vk::BlendFactor::eOneMinusConstantColor;
    case BlendFactor::Src1Color:
        return vk::BlendFactor::eSrc1Color;
    case BlendFactor::InvSrc1Color:
        return vk::BlendFactor::eOneMinusSrc1Color;
    case BlendFactor::Src1Alpha:
        return vk::BlendFactor::eSrc1Alpha;
    case BlendFactor::InvSrc1Alpha:
        return vk::BlendFactor::eOneMinusSrc1Alpha;
    case BlendFactor::ConstantAlpha:
        return vk::BlendFactor::eConstantAlpha;
    case BlendFactor::OneMinusConstantAlpha:
        return vk::BlendFactor::eOneMinusConstantAlpha;
    default:
        UNREACHABLE_MSG("Unknown blend factor: {}", static_cast<u32>(factor));
    }
}

bool IsDualSourceBlendFactor(AmdGpu::BlendControl::BlendFactor factor) {
    using BlendFactor = AmdGpu::BlendControl::BlendFactor;
    switch (factor) {
    case BlendFactor::Src1Color:
    case BlendFactor::Src1Alpha:
    case BlendFactor::InvSrc1Color:
    case BlendFactor::InvSrc1Alpha:
        return true;
    default:
        return false;
    }
}

vk::BlendOp BlendOp(AmdGpu::BlendControl::BlendFunc func) {
    using BlendFunc = AmdGpu::BlendControl::BlendFunc;
    switch (func) {
    case BlendFunc::Add:
        return vk::BlendOp::eAdd;
    case BlendFunc::Subtract:
        return vk::BlendOp::eSubtract;
    case BlendFunc::Min:
        return vk::BlendOp::eMin;
    case BlendFunc::Max:
        return vk::BlendOp::eMax;
    case BlendFunc::ReverseSubtract:
        return vk::BlendOp::eReverseSubtract;
    default:
        UNREACHABLE_MSG("Unknown blend op: {}", static_cast<u32>(func));
    }
}

void NormalizeMinMaxBlend(vk::BlendOp& op, vk::BlendFactor& src, vk::BlendFactor& dst,
                          Ps4Layout::NumberFormat format) {
    if (format != Ps4Layout::NumberFormat::Unorm && format != Ps4Layout::NumberFormat::Srgb) {
        return;
    }
    if (op != vk::BlendOp::eMin && op != vk::BlendOp::eMax) {
        return;
    }
    if ((src != vk::BlendFactor::eZero && src != vk::BlendFactor::eOne) ||
        (dst != vk::BlendFactor::eZero && dst != vk::BlendFactor::eOne) ||
        (src != vk::BlendFactor::eZero && dst != vk::BlendFactor::eZero)) {
        return;
    }
    if (op == vk::BlendOp::eMin) {
        src = vk::BlendFactor::eZero;
        dst = vk::BlendFactor::eZero;
    }
    op = vk::BlendOp::eAdd;
}

vk::LogicOp LogicOp(AmdGpu::ColorControl::LogicOp logic_op) {
    using LogicOp = AmdGpu::ColorControl::LogicOp;
    switch (logic_op) {
    case LogicOp::Clear:
        return vk::LogicOp::eClear;
    case LogicOp::Nor:
        return vk::LogicOp::eNor;
    case LogicOp::AndInverted:
        return vk::LogicOp::eAndInverted;
    case LogicOp::CopyInverted:
        return vk::LogicOp::eCopyInverted;
    case LogicOp::AndReverse:
        return vk::LogicOp::eAndReverse;
    case LogicOp::Invert:
        return vk::LogicOp::eInvert;
    case LogicOp::BrushXor:
        LOG_WARNING(Render_Vulkan, "Unimplemented logic op {:#x}, using closest equivalent",
                    u32(logic_op));
        [[fallthrough]];
    case LogicOp::Xor:
        return vk::LogicOp::eXor;
    case LogicOp::Nand:
        return vk::LogicOp::eNand;
    case LogicOp::And:
        return vk::LogicOp::eAnd;
    case LogicOp::Equiv:
        return vk::LogicOp::eEquivalent;
    case LogicOp::Noop:
        return vk::LogicOp::eNoOp;
    case LogicOp::OrInverted:
        return vk::LogicOp::eOrInverted;
    case LogicOp::Copy:
        return vk::LogicOp::eCopy;
    case LogicOp::OrReverse:
        return vk::LogicOp::eOrReverse;
    case LogicOp::Or:
        return vk::LogicOp::eOr;
    case LogicOp::Set:
        return vk::LogicOp::eSet;
    default:
        UNREACHABLE_MSG("Unknown logic op {}", u32(logic_op));
    }
}

// https://github.com/chaotic-cx/mesa-mirror/blob/0954afff5/src/amd/vulkan/radv_sampler.c#L21
vk::SamplerAddressMode ClampMode(AmdGpu::ClampMode mode) {
    switch (mode) {
    case AmdGpu::ClampMode::Wrap:
        return vk::SamplerAddressMode::eRepeat;
    case AmdGpu::ClampMode::Mirror:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case AmdGpu::ClampMode::ClampLastTexel:
        return vk::SamplerAddressMode::eClampToEdge;
    case AmdGpu::ClampMode::MirrorOnceHalfBorder:
    case AmdGpu::ClampMode::MirrorOnceBorder:
        LOG_WARNING(Render_Vulkan, "Unimplemented clamp mode {}, using closest equivalent.",
                    static_cast<u32>(mode));
        [[fallthrough]];
    case AmdGpu::ClampMode::MirrorOnceLastTexel:
        return vk::SamplerAddressMode::eMirrorClampToEdge;
    case AmdGpu::ClampMode::ClampHalfBorder:
        LOG_WARNING(Render_Vulkan, "Unimplemented clamp mode {}, using closest equivalent.",
                    static_cast<u32>(mode));
        [[fallthrough]];
    case AmdGpu::ClampMode::ClampBorder:
        return vk::SamplerAddressMode::eClampToBorder;
    default:
        UNREACHABLE();
    }
}

vk::CompareOp DepthCompare(AmdGpu::DepthCompare comp) {
    switch (comp) {
    case AmdGpu::DepthCompare::Never:
        return vk::CompareOp::eNever;
    case AmdGpu::DepthCompare::Less:
        return vk::CompareOp::eLess;
    case AmdGpu::DepthCompare::Equal:
        return vk::CompareOp::eEqual;
    case AmdGpu::DepthCompare::LessEqual:
        return vk::CompareOp::eLessOrEqual;
    case AmdGpu::DepthCompare::Greater:
        return vk::CompareOp::eGreater;
    case AmdGpu::DepthCompare::NotEqual:
        return vk::CompareOp::eNotEqual;
    case AmdGpu::DepthCompare::GreaterEqual:
        return vk::CompareOp::eGreaterOrEqual;
    case AmdGpu::DepthCompare::Always:
        return vk::CompareOp::eAlways;
    default:
        UNREACHABLE();
    }
}

vk::Filter Filter(AmdGpu::Filter filter) {
    switch (filter) {
    case AmdGpu::Filter::Point:
    case AmdGpu::Filter::AnisoPoint:
        return vk::Filter::eNearest;
    case AmdGpu::Filter::Bilinear:
    case AmdGpu::Filter::AnisoLinear:
        return vk::Filter::eLinear;
    default:
        UNREACHABLE();
    }
}

vk::SamplerReductionMode FilterMode(AmdGpu::FilterMode mode) {
    switch (mode) {
    case AmdGpu::FilterMode::Blend:
        return vk::SamplerReductionMode::eWeightedAverage;
    case AmdGpu::FilterMode::Min:
        return vk::SamplerReductionMode::eMin;
    case AmdGpu::FilterMode::Max:
        return vk::SamplerReductionMode::eMax;
    default:
        UNREACHABLE();
    }
}

vk::SamplerMipmapMode MipFilter(AmdGpu::MipFilter filter) {
    switch (filter) {
    case AmdGpu::MipFilter::Point:
        return vk::SamplerMipmapMode::eNearest;
    case AmdGpu::MipFilter::Linear:
        return vk::SamplerMipmapMode::eLinear;
    case AmdGpu::MipFilter::None:
        return vk::SamplerMipmapMode::eNearest;
    default:
        UNREACHABLE();
    }
}

vk::BorderColor BorderColor(AmdGpu::BorderColor color) {
    switch (color) {
    case AmdGpu::BorderColor::TransparentBlack:
        return vk::BorderColor::eFloatTransparentBlack;
    case AmdGpu::BorderColor::OpaqueBlack:
        return vk::BorderColor::eFloatOpaqueBlack;
    case AmdGpu::BorderColor::White:
        return vk::BorderColor::eFloatOpaqueWhite;
    case AmdGpu::BorderColor::Custom:
        return vk::BorderColor::eFloatCustomEXT;
    default:
        UNREACHABLE();
    }
}

vk::ComponentSwizzle ComponentSwizzle(Ps4Layout::CompSwizzle comp_swizzle) {
    switch (comp_swizzle) {
    case Ps4Layout::CompSwizzle::Zero:
        return vk::ComponentSwizzle::eZero;
    case Ps4Layout::CompSwizzle::One:
        return vk::ComponentSwizzle::eOne;
    case Ps4Layout::CompSwizzle::Red:
        return vk::ComponentSwizzle::eR;
    case Ps4Layout::CompSwizzle::Green:
        return vk::ComponentSwizzle::eG;
    case Ps4Layout::CompSwizzle::Blue:
        return vk::ComponentSwizzle::eB;
    case Ps4Layout::CompSwizzle::Alpha:
        return vk::ComponentSwizzle::eA;
    default:
        UNREACHABLE();
    }
}

vk::ComponentMapping ComponentMapping(Ps4Layout::CompMapping comp_mapping) {
    return vk::ComponentMapping{
        .r = ComponentSwizzle(comp_mapping.r),
        .g = ComponentSwizzle(comp_mapping.g),
        .b = ComponentSwizzle(comp_mapping.b),
        .a = ComponentSwizzle(comp_mapping.a),
    };
}

// Texel buffer feature flags are not needed as format is interpreted in-shader.
static constexpr vk::FormatFeatureFlags2 BufferRead = vk::FormatFeatureFlagBits2::eVertexBuffer;
static constexpr vk::FormatFeatureFlags2 BufferWrite = static_cast<vk::FormatFeatureFlags2>(0);
static constexpr vk::FormatFeatureFlags2 ImageRead = vk::FormatFeatureFlagBits2::eTransferSrc |
                                                     vk::FormatFeatureFlagBits2::eTransferDst |
                                                     vk::FormatFeatureFlagBits2::eSampledImage;
static constexpr vk::FormatFeatureFlags2 ImageWrite =
    vk::FormatFeatureFlagBits2::eStorageImage |
    vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat |
    vk::FormatFeatureFlagBits2::eStorageWriteWithoutFormat;
static constexpr vk::FormatFeatureFlags2 Mrt = vk::FormatFeatureFlagBits2::eColorAttachment;

// Table 8.13 Data and Image Formats [Sea Islands Series Instruction Set Architecture]
static constexpr vk::FormatFeatureFlags2 GetDataFormatFeatureFlags(
    const Ps4Layout::DataFormat data_format) {
    switch (data_format) {
    case Ps4Layout::DataFormat::FormatInvalid:
    case Ps4Layout::DataFormat::Format8:
    case Ps4Layout::DataFormat::Format16:
    case Ps4Layout::DataFormat::Format8_8:
    case Ps4Layout::DataFormat::Format32:
    case Ps4Layout::DataFormat::Format16_16:
    case Ps4Layout::DataFormat::Format10_11_11:
    case Ps4Layout::DataFormat::Format11_11_10:
    case Ps4Layout::DataFormat::Format10_10_10_2:
    case Ps4Layout::DataFormat::Format2_10_10_10:
    case Ps4Layout::DataFormat::Format8_8_8_8:
    case Ps4Layout::DataFormat::Format32_32:
    case Ps4Layout::DataFormat::Format16_16_16_16:
    case Ps4Layout::DataFormat::Format32_32_32_32:
        return BufferRead | BufferWrite | ImageRead | ImageWrite | Mrt;
    case Ps4Layout::DataFormat::Format32_32_32:
        return BufferRead | BufferWrite | ImageRead;
    case Ps4Layout::DataFormat::Format5_6_5:
    case Ps4Layout::DataFormat::Format1_5_5_5:
    case Ps4Layout::DataFormat::Format5_5_5_1:
    case Ps4Layout::DataFormat::Format4_4_4_4:
        return ImageRead | ImageWrite | Mrt;
    case Ps4Layout::DataFormat::Format8_24:
    case Ps4Layout::DataFormat::Format24_8:
    case Ps4Layout::DataFormat::FormatX24_8_32:
        return ImageRead | Mrt;
    case Ps4Layout::DataFormat::FormatGB_GR:
    case Ps4Layout::DataFormat::FormatBG_RG:
    case Ps4Layout::DataFormat::Format5_9_9_9:
    case Ps4Layout::DataFormat::FormatBc1:
    case Ps4Layout::DataFormat::FormatBc2:
    case Ps4Layout::DataFormat::FormatBc3:
    case Ps4Layout::DataFormat::FormatBc4:
    case Ps4Layout::DataFormat::FormatBc5:
    case Ps4Layout::DataFormat::FormatBc6:
    case Ps4Layout::DataFormat::FormatBc7:
    case Ps4Layout::DataFormat::Format4_4:
    case Ps4Layout::DataFormat::Format6_5_5:
    case Ps4Layout::DataFormat::Format1:
    case Ps4Layout::DataFormat::Format1_Reversed:
    case Ps4Layout::DataFormat::Format32_As_8:
    case Ps4Layout::DataFormat::Format32_As_8_8:
    case Ps4Layout::DataFormat::Format32_As_32_32_32_32:
        return ImageRead;
    case Ps4Layout::DataFormat::FormatFmask8_S2_F1:
    case Ps4Layout::DataFormat::FormatFmask8_S4_F1:
    case Ps4Layout::DataFormat::FormatFmask8_S8_F1:
    case Ps4Layout::DataFormat::FormatFmask8_S2_F2:
    case Ps4Layout::DataFormat::FormatFmask8_S4_F2:
    case Ps4Layout::DataFormat::FormatFmask8_S4_F4:
    case Ps4Layout::DataFormat::FormatFmask16_S16_F1:
    case Ps4Layout::DataFormat::FormatFmask16_S8_F2:
    case Ps4Layout::DataFormat::FormatFmask32_S16_F2:
    case Ps4Layout::DataFormat::FormatFmask32_S8_F4:
    case Ps4Layout::DataFormat::FormatFmask32_S8_F8:
    case Ps4Layout::DataFormat::FormatFmask64_S16_F4:
    case Ps4Layout::DataFormat::FormatFmask64_S16_F8:
        return ImageRead | ImageWrite;
    }
    UNREACHABLE_MSG("Missing feature flags for data format {}", static_cast<u32>(data_format));
}

// Table 8.13 Data and Image Formats [Sea Islands Series Instruction Set Architecture]
static constexpr vk::FormatFeatureFlags2 GetNumberFormatFeatureFlags(
    const Ps4Layout::NumberFormat number_format) {
    switch (number_format) {
    case Ps4Layout::NumberFormat::Unorm:
    case Ps4Layout::NumberFormat::Snorm:
    case Ps4Layout::NumberFormat::Uint:
    case Ps4Layout::NumberFormat::Sint:
    case Ps4Layout::NumberFormat::Float:
        return BufferRead | BufferWrite | ImageRead | ImageWrite | Mrt;
    case Ps4Layout::NumberFormat::Uscaled:
    case Ps4Layout::NumberFormat::Sscaled:
    case Ps4Layout::NumberFormat::SnormNz:
        return BufferRead | ImageRead;
    case Ps4Layout::NumberFormat::Srgb:
        return ImageRead | Mrt;
    case Ps4Layout::NumberFormat::Ubnorm:
    case Ps4Layout::NumberFormat::UbnormNz:
    case Ps4Layout::NumberFormat::Ubint:
    case Ps4Layout::NumberFormat::Ubscaled:
        return ImageRead;
    }
    UNREACHABLE_MSG("Missing feature flags for number format {}", static_cast<u32>(number_format));
}

static constexpr SurfaceFormatInfo CreateSurfaceFormatInfo(const Ps4Layout::DataFormat data_format,
                                                           const Ps4Layout::NumberFormat number_format,
                                                           const vk::Format vk_format) {
    return {
        .data_format = data_format,
        .number_format = number_format,
        .vk_format = vk_format,
        .flags =
            GetDataFormatFeatureFlags(data_format) & GetNumberFormatFeatureFlags(number_format),
    };
}

// Uscaled, Sscaled, and Ubnorm formats are automatically remapped and handled in shader.
static constexpr std::array formats{
    // Invalid
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Uscaled,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Sscaled,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::SnormNz,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Float,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Ubnorm,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::UbnormNz,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Ubint,
                            vk::Format::eUndefined),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatInvalid, Ps4Layout::NumberFormat::Ubscaled,
                            vk::Format::eUndefined),
    // 8
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR8Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR8Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR8Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR8Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eR8Srgb),
    // 16
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR16Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR16Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR16Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR16Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR16Sfloat),
    // 8_8
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR8G8Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR8G8Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR8G8Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR8G8Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eR8G8Srgb),
    // 32
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR32Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR32Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR32Sfloat),
    // 16_16
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR16G16Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR16G16Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR16G16Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR16G16Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR16G16Sfloat),
    // 10_11_11
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format10_11_11, Ps4Layout::NumberFormat::Float,
                            vk::Format::eB10G11R11UfloatPack32),
    // 11_11_10 - Remapped to 10_11_11.
    // 10_10_10_2 - Remapped to 2_10_10_10.
    // 2_10_10_10
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format2_10_10_10, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eA2B10G10R10UnormPack32),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format2_10_10_10, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eA2B10G10R10SnormPack32),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format2_10_10_10, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eA2B10G10R10UintPack32),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format2_10_10_10, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eA2B10G10R10SintPack32),
    // 8_8_8_8
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8_8_8, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR8G8B8A8Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8_8_8, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR8G8B8A8Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8_8_8, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR8G8B8A8Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8_8_8, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR8G8B8A8Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format8_8_8_8, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eR8G8B8A8Srgb),
    // 32_32
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR32G32Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR32G32Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR32G32Sfloat),
    // 16_16_16_16
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR16G16B16A16Unorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eR16G16B16A16Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR16G16B16A16Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR16G16B16A16Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::SnormNz,
                            vk::Format::eR16G16B16A16Snorm),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format16_16_16_16, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR16G16B16A16Sfloat),
    // 32_32_32
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR32G32B32Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR32G32B32Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR32G32B32Sfloat),
    // 32_32_32_32
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32_32, Ps4Layout::NumberFormat::Uint,
                            vk::Format::eR32G32B32A32Uint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32_32, Ps4Layout::NumberFormat::Sint,
                            vk::Format::eR32G32B32A32Sint),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format32_32_32_32, Ps4Layout::NumberFormat::Float,
                            vk::Format::eR32G32B32A32Sfloat),
    // 5_6_5
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format5_6_5, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR5G6B5UnormPack16),
    // 1_5_5_5
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format1_5_5_5, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eA1R5G5B5UnormPack16),
    // 5_5_5_1
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format5_5_5_1, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eR5G5B5A1UnormPack16),
    // 4_4_4_4
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format4_4_4_4, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eB4G4R4A4UnormPack16),
    // 8_24
    // 24_8
    // X24_8_32
    // GB_GR
    // BG_RG
    // 5_9_9_9
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::Format5_9_9_9, Ps4Layout::NumberFormat::Float,
                            vk::Format::eE5B9G9R9UfloatPack32),
    // BC1
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc1, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc1RgbaUnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc1, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eBc1RgbaSrgbBlock),
    // BC2
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc2, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc2UnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc2, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eBc2SrgbBlock),
    // BC3
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc3, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc3UnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc3, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eBc3SrgbBlock),
    // BC4
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc4, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc4UnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc4, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eBc4SnormBlock),
    // BC5
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc5, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc5UnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc5, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eBc5SnormBlock),
    // BC6
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc6, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc6HUfloatBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc6, Ps4Layout::NumberFormat::Snorm,
                            vk::Format::eBc6HSfloatBlock),
    // BC7
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc7, Ps4Layout::NumberFormat::Unorm,
                            vk::Format::eBc7UnormBlock),
    CreateSurfaceFormatInfo(Ps4Layout::DataFormat::FormatBc7, Ps4Layout::NumberFormat::Srgb,
                            vk::Format::eBc7SrgbBlock),
};

std::span<const SurfaceFormatInfo> SurfaceFormats() {
    return formats;
}

constexpr std::array<vk::Format, surface_format_table_size> surface_format_table = []() {
    std::array<vk::Format, surface_format_table_size> result;
    for (auto& entry : result) {
        entry = vk::Format::eUndefined;
    }
    for (const auto& supported_format : formats) {
        result[GetSurfaceFormatTableIndex(supported_format.data_format,
                                          supported_format.number_format)] =
            supported_format.vk_format;
    }
    return result;
}();

static constexpr DepthFormatInfo CreateDepthFormatInfo(
    const DepthBuffer::ZFormat z_format, const DepthBuffer::StencilFormat stencil_format,
    const vk::Format vk_format) {
    return {
        .z_format = z_format,
        .stencil_format = stencil_format,
        .vk_format = vk_format,
        .flags = vk::FormatFeatureFlagBits2::eDepthStencilAttachment,
    };
}

std::span<const DepthFormatInfo> DepthFormats() {
    using ZFormat = DepthBuffer::ZFormat;
    using StencilFormat = DepthBuffer::StencilFormat;
    static constexpr std::array formats{
        // Invalid
        CreateDepthFormatInfo(ZFormat::Invalid, StencilFormat::Invalid, vk::Format::eUndefined),
        CreateDepthFormatInfo(ZFormat::Invalid, StencilFormat::Stencil8,
                              vk::Format::eD32SfloatS8Uint),
        // 16
        CreateDepthFormatInfo(ZFormat::Z16, StencilFormat::Invalid, vk::Format::eD16Unorm),
        CreateDepthFormatInfo(ZFormat::Z16, StencilFormat::Stencil8, vk::Format::eD16UnormS8Uint),
        // 32_Float
        CreateDepthFormatInfo(ZFormat::Z32Float, StencilFormat::Invalid, vk::Format::eD32Sfloat),
        CreateDepthFormatInfo(ZFormat::Z32Float, StencilFormat::Stencil8,
                              vk::Format::eD32SfloatS8Uint),
    };
    return formats;
}

vk::Format DepthFormat(DepthBuffer::ZFormat z_format, DepthBuffer::StencilFormat stencil_format) {
    const auto& formats = DepthFormats();
    const auto format =
        std::find_if(formats.begin(), formats.end(), [&](const DepthFormatInfo& format_info) {
            return format_info.z_format == z_format && format_info.stencil_format == stencil_format;
        });
    ASSERT_MSG(format != formats.end(), "Unknown z_format={} and stencil_format={}",
               static_cast<u32>(z_format), static_cast<u32>(stencil_format));
    return format->vk_format;
}

vk::ClearValue ColorBufferClearValue(const Core::NativeTarget::ColorTarget& color_buffer) {
    const auto comp_swizzle = color_buffer.Swizzle();
    const auto format = Ps4Layout::DataFormat(color_buffer.info.format);
    const auto number_type = color_buffer.GetFixedNumberFormat();

    const auto& c0 = color_buffer.clear_word0;
    const auto& c1 = color_buffer.clear_word1;

    vk::ClearColorValue color{};

    switch (number_type) {
    case Ps4Layout::NumberFormat::Uint:
    case Ps4Layout::NumberFormat::Sint:
        color.uint32[3] = 1;
        break;
    default:
        color.float32[3] = 1.0f;
        break;
    }

    switch (format) {
    case Ps4Layout::DataFormat::Format8:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
        case Ps4Layout::NumberFormat::Srgb: // Should we handle gamma correction here?
            color.float32[0] = NumberUtils::U8ToUnorm(c0 & 0xff);
            break;
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S8ToSnorm(c0 & 0xff);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0;
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format16:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
            color.float32[0] = NumberUtils::U16ToUnorm(c0 & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S16ToSnorm(c0 & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = NumberUtils::Uf16ToF32(c0 & 0xffff);
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format8_8:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
        case Ps4Layout::NumberFormat::Srgb: // Should we handle gamma correction here?
            color.float32[0] = NumberUtils::U8ToUnorm(c0 & 0xff);
            color.float32[1] = NumberUtils::U8ToUnorm((c0 >> 8) & 0xff);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S8ToSnorm(c0 & 0xff);
            color.float32[1] = NumberUtils::S8ToSnorm((c0 >> 8) & 0xff);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0xff;
            color.uint32[1] = (c0 >> 8) & 0xff;
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format32:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = *(reinterpret_cast<const float*>(&c0));
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format16_16:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
            color.float32[0] = NumberUtils::U16ToUnorm(c0 & 0xffff);
            color.float32[1] = NumberUtils::U16ToUnorm((c0 >> 16) & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S16ToSnorm(c0 & 0xffff);
            color.float32[1] = NumberUtils::S16ToSnorm((c0 >> 16) & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0xffff;
            color.uint32[1] = (c0 >> 16) & 0xffff;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = NumberUtils::Uf16ToF32(c0 & 0xffff);
            color.float32[1] = NumberUtils::Uf16ToF32((c0 >> 16) & 0xffff);
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format10_11_11:
        color.float32[0] = NumberUtils::Uf11ToF32(c0 & 0x7ff);
        color.float32[1] = NumberUtils::Uf11ToF32((c0 >> 11) & 0x7ff);
        color.float32[2] = NumberUtils::Uf10ToF32((c0 >> 22) & 0x3ff);
        break;
    case Ps4Layout::DataFormat::Format11_11_10:
        color.float32[0] = NumberUtils::Uf10ToF32(c0 & 0x3ff);
        color.float32[1] = NumberUtils::Uf11ToF32((c0 >> 10) & 0x7ff);
        color.float32[2] = NumberUtils::Uf11ToF32((c0 >> 21) & 0x7ff);
        break;
    case Ps4Layout::DataFormat::Format5_9_9_9: {
        int exponent;
        union {
            float f;
            u32 u;
        } scale;

        exponent = (c0 >> 27) - 10;
        scale.u = (exponent + 127) << 23;

        color.float32[0] = (c0 & 0x1ff) * scale.f;
        color.float32[1] = ((c0 >> 9) & 0x1ff) * scale.f;
        color.float32[2] = ((c0 >> 18) & 0x1ff) * scale.f;
        break;
    }
    case Ps4Layout::DataFormat::Format10_10_10_2:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
            color.float32[0] = NumberUtils::U2ToUnorm(c0 & 0x3);
            color.float32[1] = NumberUtils::U10ToUnorm((c0 >> 2) & 0x3ff);
            color.float32[2] = NumberUtils::U10ToUnorm((c0 >> 12) & 0x3ff);
            color.float32[3] = NumberUtils::U10ToUnorm(c0 >> 22);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S2ToSnorm(c0 & 0x3);
            color.float32[1] = NumberUtils::S10ToSnorm((c0 >> 2) & 0x3ff);
            color.float32[2] = NumberUtils::S10ToSnorm((c0 >> 12) & 0x3ff);
            color.float32[3] = NumberUtils::S2ToSnorm(c0 >> 22);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0x3;
            color.uint32[1] = (c0 >> 2) & 0x3ff;
            color.uint32[2] = (c0 >> 12) & 0x3ff;
            color.uint32[3] = c0 >> 22;
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format2_10_10_10:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
            color.float32[0] = NumberUtils::U10ToUnorm(c0 & 0x3ff);
            color.float32[1] = NumberUtils::U10ToUnorm((c0 >> 10) & 0x3ff);
            color.float32[2] = NumberUtils::U10ToUnorm((c0 >> 20) & 0x3ff);
            color.float32[3] = NumberUtils::U2ToUnorm(c0 >> 30);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S10ToSnorm(c0 & 0x3ff);
            color.float32[1] = NumberUtils::S10ToSnorm((c0 >> 10) & 0x3ff);
            color.float32[2] = NumberUtils::S10ToSnorm((c0 >> 20) & 0x3ff);
            color.float32[3] = NumberUtils::S2ToSnorm(c0 >> 30);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0x3ff;
            color.uint32[1] = (c0 >> 10) & 0x3ff;
            color.uint32[2] = (c0 >> 20) & 0x3ff;
            color.uint32[3] = c0 >> 30;
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format8_8_8_8:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
        case Ps4Layout::NumberFormat::Srgb: // Should we handle gamma correction here?
            color.float32[0] = NumberUtils::U8ToUnorm(c0 & 0xff);
            color.float32[1] = NumberUtils::U8ToUnorm((c0 >> 8) & 0xff);
            color.float32[2] = NumberUtils::U8ToUnorm((c0 >> 16) & 0xff);
            color.float32[3] = NumberUtils::U8ToUnorm(c0 >> 24);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S8ToSnorm(c0 & 0xff);
            color.float32[1] = NumberUtils::S8ToSnorm((c0 >> 8) & 0xff);
            color.float32[2] = NumberUtils::S8ToSnorm((c0 >> 16) & 0xff);
            color.float32[3] = NumberUtils::S8ToSnorm(c0 >> 24);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0xff;
            color.uint32[1] = (c0 >> 8) & 0xff;
            color.uint32[2] = (c0 >> 16) & 0xff;
            color.uint32[3] = c0 >> 24;
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format32_32:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0;
            color.uint32[1] = c1;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = *(reinterpret_cast<const float*>(&c0));
            color.float32[1] = *(reinterpret_cast<const float*>(&c1));
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format16_16_16_16:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Unorm:
            color.float32[0] = NumberUtils::U16ToUnorm(c0 & 0xffff);
            color.float32[1] = NumberUtils::U16ToUnorm((c0 >> 16) & 0xffff);
            color.float32[2] = NumberUtils::U16ToUnorm(c1 & 0xffff);
            color.float32[3] = NumberUtils::U16ToUnorm((c1 >> 16) & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Snorm:
        case Ps4Layout::NumberFormat::SnormNz:
            color.float32[0] = NumberUtils::S16ToSnorm(c0 & 0xffff);
            color.float32[1] = NumberUtils::S16ToSnorm((c0 >> 16) & 0xffff);
            color.float32[2] = NumberUtils::S16ToSnorm(c1 & 0xffff);
            color.float32[3] = NumberUtils::S16ToSnorm((c1 >> 16) & 0xffff);
            break;
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0 & 0xffff;
            color.uint32[1] = (c0 >> 16) & 0xffff;
            color.uint32[2] = c1 & 0xffff;
            color.uint32[3] = (c1 >> 16) & 0xffff;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = NumberUtils::Uf16ToF32(c0 & 0xffff);
            color.float32[1] = NumberUtils::Uf16ToF32((c0 >> 16) & 0xffff);
            color.float32[2] = NumberUtils::Uf16ToF32(c1 & 0xffff);
            color.float32[3] = NumberUtils::Uf16ToF32((c1 >> 16) & 0xffff);
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format32_32_32_32:
        switch (number_type) {
        case Ps4Layout::NumberFormat::Uint:
        case Ps4Layout::NumberFormat::Sint:
            color.uint32[0] = c0;
            color.uint32[1] = c0;
            color.uint32[2] = c0;
            color.uint32[3] = c1;
            break;
        case Ps4Layout::NumberFormat::Float:
            color.float32[0] = *(reinterpret_cast<const float*>(&c0));
            color.float32[1] = *(reinterpret_cast<const float*>(&c0));
            color.float32[2] = *(reinterpret_cast<const float*>(&c0));
            color.float32[3] = *(reinterpret_cast<const float*>(&c1));
            break;
        default:
            INVALID_NUMBER_FORMAT_COMBO;
            break;
        }
        break;
    case Ps4Layout::DataFormat::Format5_6_5:
        color.float32[0] = NumberUtils::U5ToUnorm(c0 & 0x1f);
        color.float32[1] = NumberUtils::U6ToUnorm((c0 >> 5) & 0x3f);
        color.float32[2] = NumberUtils::U5ToUnorm(c0 >> 11);
        break;
    case Ps4Layout::DataFormat::Format1_5_5_5:
        color.float32[0] = NumberUtils::U5ToUnorm(c0 & 0x1f);
        color.float32[1] = NumberUtils::U5ToUnorm((c0 >> 5) & 0x1f);
        color.float32[2] = NumberUtils::U5ToUnorm((c0 >> 10) & 0x1f);
        color.float32[3] = (c0 >> 15) ? 1.0f : 0.0f;
        break;
    case Ps4Layout::DataFormat::Format5_5_5_1:
        color.float32[0] = (c0 & 0x1) ? 1.0f : 0.0f;
        color.float32[1] = NumberUtils::U5ToUnorm((c0 >> 1) & 0x1f);
        color.float32[2] = NumberUtils::U5ToUnorm((c0 >> 6) & 0x1f);
        color.float32[3] = NumberUtils::U5ToUnorm((c0 >> 11) & 0x1f);
        break;
    case Ps4Layout::DataFormat::Format4_4_4_4:
        color.float32[0] = NumberUtils::U4ToUnorm(c0 & 0xf);
        color.float32[1] = NumberUtils::U4ToUnorm((c0 >> 4) & 0xf);
        color.float32[2] = NumberUtils::U4ToUnorm((c0 >> 8) & 0xf);
        color.float32[3] = NumberUtils::U4ToUnorm(c0 >> 12);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unsupported color buffer format: {}", format);
        break;
    }

    color.float32 = comp_swizzle.Apply(color.float32);
    return {.color = color};
}

vk::SampleCountFlagBits RawNumSamples(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return vk::SampleCountFlagBits::e1;
    case 2:
        return vk::SampleCountFlagBits::e2;
    case 4:
        return vk::SampleCountFlagBits::e4;
    case 8:
        return vk::SampleCountFlagBits::e8;
    case 16:
        return vk::SampleCountFlagBits::e16;
    default:
        UNREACHABLE();
    }
}

vk::SampleCountFlagBits NumSamples(u32 num_samples, vk::SampleCountFlags supported_flags) {
    vk::SampleCountFlagBits flags = RawNumSamples(num_samples);
    // Half sample counts until supported, with a minimum of 1.
    while (!(supported_flags & flags) && num_samples > 1) {
        num_samples /= 2;
        flags = RawNumSamples(num_samples);
    }
    return flags;
}

} // namespace Vulkan::LiverpoolToVK
