/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace {

// Pillow's fixed-point separable resampler, as used by the slow BitImageProcessor.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);

struct PillowSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double pillow_cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<PillowSpan> make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<PillowSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("DINOv2 Pillow resize produced empty support");

        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_cubic((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("DINOv2 Pillow resize has invalid coefficients");
        span.weights.reserve(floating.size());
        for (double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kPillowScale);
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

std::uint8_t apply_pillow_span(const std::uint8_t* source, std::int32_t stride,
                               const PillowSpan& span) {
    std::int64_t sum = kPillowRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        sum += static_cast<std::int64_t>(
                   source[(span.first + static_cast<std::int32_t>(index)) * stride]) *
               span.weights[index];
    }
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPillowPrecisionBits, 255));
}

std::vector<std::uint8_t> resize_pillow_bicubic(const std::uint8_t* input, std::int32_t input_h,
                                                std::int32_t input_w, std::int32_t output_h,
                                                std::int32_t output_w) {
    // Pillow skips a pass whose size is unchanged, so an identity pass must not round twice.
    std::vector<std::uint8_t> horizontal;
    const std::uint8_t* vertical_input = input;
    if (input_w != output_w) {
        const auto plan = make_pillow_plan(input_w, output_w);
        horizontal.resize(static_cast<std::size_t>(input_h) * output_w * 3U);
        for (std::int32_t y = 0; y < input_h; ++y) {
            for (std::int32_t x = 0; x < output_w; ++x) {
                const auto& span = plan[static_cast<std::size_t>(x)];
                for (std::int32_t channel = 0; channel < 3; ++channel) {
                    const auto source = static_cast<std::size_t>(y) * input_w * 3U + channel;
                    const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                    horizontal[target] = apply_pillow_span(input + source, 3, span);
                }
            }
        }
        vertical_input = horizontal.data();
    }
    if (input_h == output_h)
        return input_w == output_w
                   ? std::vector<std::uint8_t>(input, input + static_cast<std::size_t>(input_h) *
                                                                  input_w * 3U)
                   : horizontal;

    const auto plan = make_pillow_plan(input_h, output_h);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_h) * output_w * 3U);
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto& span = plan[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < output_w; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(x) * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[target] = apply_pillow_span(vertical_input + source, output_w * 3, span);
            }
        }
    }
    return output;
}

void validate_config(const Dinov2PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0 ||
        config.resize_shortest_edge < config.input_image_h ||
        config.resize_shortest_edge < config.input_image_w)
        throw std::invalid_argument("DINOv2 crop must fit inside the resized shortest edge");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("DINOv2 image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (!std::isfinite(value) || value <= 0.0F)
            throw std::invalid_argument("DINOv2 image std must be finite and positive");
    }
}

} // namespace

Dinov2ImageGeometry compute_dinov2_image_geometry(int32_t image_height, int32_t image_width,
                                                  const Dinov2PreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("DINOv2 source image must be non-empty");
    validate_config(config);
    // transformers.image_transforms.get_resize_output_image_size(default_to_square=False):
    // the short edge becomes `shortest_edge`, the long edge int(shortest_edge * long / short).
    const auto shortest = static_cast<std::int64_t>(config.resize_shortest_edge);
    Dinov2ImageGeometry geometry;
    if (image_width <= image_height) {
        geometry.resized_w = config.resize_shortest_edge;
        geometry.resized_h = static_cast<int32_t>(shortest * image_height / image_width);
    } else {
        geometry.resized_h = config.resize_shortest_edge;
        geometry.resized_w = static_cast<int32_t>(shortest * image_width / image_height);
    }
    // transformers.image_transforms.center_crop floors the offset.
    geometry.crop_y = (geometry.resized_h - config.input_image_h) / 2;
    geometry.crop_x = (geometry.resized_w - config.input_image_w) / 2;
    return geometry;
}

std::vector<float> preprocess_dinov2_image(const uint8_t* rgb, int32_t image_height,
                                           int32_t image_width,
                                           const Dinov2PreprocessConfig& config) {
    if (rgb == nullptr)
        throw std::invalid_argument("DINOv2 source image must be non-empty");
    const auto geometry = compute_dinov2_image_geometry(image_height, image_width, config);
    const auto resized = resize_pillow_bicubic(rgb, image_height, image_width, geometry.resized_h,
                                               geometry.resized_w);

    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (int32_t y = 0; y < config.input_image_h; ++y) {
        for (int32_t x = 0; x < config.input_image_w; ++x) {
            const auto source =
                (static_cast<std::size_t>(geometry.crop_y + y) * geometry.resized_w +
                 static_cast<std::size_t>(geometry.crop_x + x)) *
                3U;
            const auto target = static_cast<std::size_t>(y) * config.input_image_w + x;
            for (std::size_t c = 0; c < 3; ++c) {
                const float value = static_cast<float>(resized[source + c]) / 255.0F;
                pixel_values[c * plane + target] =
                    (value - config.image_mean[c]) / config.image_std[c];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
