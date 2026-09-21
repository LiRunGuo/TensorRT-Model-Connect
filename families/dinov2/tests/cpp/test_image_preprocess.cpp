/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/image_preprocess.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

trtmc::Dinov2PreprocessConfig checkpoint_config() {
    return {};
}

void test_transformers_resize_and_crop_geometry() {
    const auto config = checkpoint_config();
    // Landscape 640x480 (WxH): short edge 256, long edge int(256 * 640 / 480) = 341.
    auto geometry = trtmc::compute_dinov2_image_geometry(480, 640, config);
    require(geometry.resized_h == 256 && geometry.resized_w == 341,
            "landscape resize must floor the long edge");
    require(geometry.crop_y == 16 && geometry.crop_x == 58, "center crop must floor its offset");
    // Portrait 333x500 (WxH): long edge int(256 * 500 / 333) = 384, crop offsets floor.
    geometry = trtmc::compute_dinov2_image_geometry(500, 333, config);
    require(geometry.resized_h == 384 && geometry.resized_w == 256,
            "portrait resize must fix the width");
    require(geometry.crop_y == 80 && geometry.crop_x == 16, "portrait crop offsets");
    // Odd margins: (257 - 224) / 2 floors to 16, unlike torchvision's round-half-even.
    geometry =
        trtmc::compute_dinov2_image_geometry(257, 256, {224, 224, 256, {0, 0, 0}, {1, 1, 1}});
    require(geometry.resized_h == 257 && geometry.crop_y == 16, "odd margins floor");
}

void test_constant_image_normalization() {
    const auto config = checkpoint_config();
    const std::vector<uint8_t> image(300U * 400U * 3U, 128);
    const auto pixels = trtmc::preprocess_dinov2_image(image.data(), 300, 400, config);
    require(pixels.size() == 3U * 224U * 224U, "output is one NCHW crop");
    const auto plane = pixels.size() / 3;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float expected = (static_cast<float>(128) / 255.0F - config.image_mean[channel]) /
                               config.image_std[channel];
        for (std::size_t index = 0; index < plane; ++index) {
            if (pixels[channel * plane + index] != expected)
                throw std::runtime_error("normalized bicubic resize must preserve a flat image");
        }
    }
}

void test_identity_resize_preserves_pixels() {
    // A source already at the resize size takes neither Pillow pass.
    const trtmc::Dinov2PreprocessConfig config{2, 2, 2, {0, 0, 0}, {1, 1, 1}};
    const std::vector<uint8_t> image{0, 51, 102, 153, 204, 255, 10, 20, 30, 40, 50, 60};
    const auto pixels = trtmc::preprocess_dinov2_image(image.data(), 2, 2, config);
    require(pixels[0] == 0.0F && pixels[4] == static_cast<float>(51) / 255.0F &&
                pixels[11] == static_cast<float>(60) / 255.0F,
            "identity resize must keep 8-bit values and NCHW order");
}

void test_invalid_configuration() {
    const std::vector<uint8_t> image(12, 0);
    rejects([&] { trtmc::compute_dinov2_image_geometry(0, 2, checkpoint_config()); },
            "reject empty source");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {4, 4, 2, {0, 0, 0}, {1, 1, 1}}); },
            "reject crop larger than resize");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {2, 2, 2, {0, 0}, {1, 1, 1}}); },
            "reject short mean");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {2, 2, 2, {0, 0, 0}, {1, 0, 1}}); },
            "reject zero std");
    rejects([&] { trtmc::preprocess_dinov2_image(nullptr, 2, 2, checkpoint_config()); },
            "reject null pixels");
}

} // namespace

int main() {
    try {
        test_transformers_resize_and_crop_geometry();
        test_constant_image_normalization();
        test_identity_resize_preserves_pixels();
        test_invalid_configuration();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    std::cout << "dinov2 image preprocessing tests passed\n";
    return 0;
}
