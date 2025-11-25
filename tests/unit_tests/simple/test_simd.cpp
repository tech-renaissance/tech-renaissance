// UTF-8 BOM to avoid C4828 warning
#pragma execution_character_set("utf-8")

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Simd/SimdLib.hpp"

int main()
{
    const char* input_file = WORKSPACE_PATH "/input.png";

    int width, height, channels;

    // 加载PNG图像（保留Alpha通道，强制转换为4通道RGBA）
    uint8_t* src = stbi_load(input_file, &width, &height, &channels, 4);
    if (!src)
    {
        std::cerr << "[ERROR] Failed to load input.png from: " << input_file << std::endl;
        return -1;
    }
    channels = 4; // 确保使用4通道（RGBA）

    std::cout << "[INFO] Image loaded successfully: " << width << "x" << height << " channels=" << channels << std::endl;

    // 输出尺寸（旋转后的尺寸通常与输入相同）
    const int dstWidth = width;
    const int dstHeight = height;

    // 旋转角度（15度）
    double angle = 15.0 * M_PI / 180.0;
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    // 创建仿射变换矩阵（围绕中心旋转）
    float mat[6];
    double cx = width / 2.0;
    double cy = height / 2.0;

    // 修正后的旋转变换矩阵计算
    mat[0] = (float)cos_a;
    mat[1] = (float)(-sin_a);
    mat[2] = (float)((1 - cos_a) * cx + sin_a * cy);
    mat[3] = (float)sin_a;
    mat[4] = (float)cos_a;
    mat[5] = (float)(-sin_a * cx + (1 - cos_a) * cy);

    // 创建SIMD视图对象 - 使用BGRA32格式（保留Alpha通道）
    std::cout << "[INFO] Creating SIMD view objects..." << std::endl;

    // 使用View构造函数，让SIMD库管理内存，支持BGRA32格式
    Simd::View<Simd::Allocator> src_view(width, height, Simd::View<Simd::Allocator>::Bgra32);
    Simd::View<Simd::Allocator> dst_view(dstWidth, dstHeight, Simd::View<Simd::Allocator>::Bgra32);

    // 将RGB数据复制到src_view并同时转换为BGR
    uint8_t* src_data = src_view.data;
    size_t src_stride = src_view.stride;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_index = (y * width + x) * channels;  // RGBA: 4 channels
            int dst_index = y * src_stride + x * channels; // BGRA: 4 channels

            // RGBA to BGRA conversion (保留Alpha通道)
            src_data[dst_index] = src[src_index + 2];     // B = R
            src_data[dst_index + 1] = src[src_index + 1]; // G = G
            src_data[dst_index + 2] = src[src_index];     // R = B
            src_data[dst_index + 3] = src[src_index + 3]; // A = A (保持不变)
        }
    }

    std::cout << "[INFO] RGBA to BGRA conversion completed" << std::endl;
    std::cout << "[INFO] Source view: " << src_view.width << "x" << src_view.height
              << ", stride=" << src_view.stride << std::endl;
    std::cout << "[INFO] Destination view: " << dst_view.width << "x" << dst_view.height
              << ", stride=" << dst_view.stride << std::endl;

    // 边界颜色（完全透明） - 使用4字节BGRA格式
    uint8_t border[4] = {0, 0, 0, 0};  // BGRA, Alpha=0(完全透明)
    std::cout << "[INFO] Border color: BGRA(" << (int)border[0] << "," << (int)border[1] << ","
              << (int)border[2] << "," << (int)border[3] << ") - Transparent" << std::endl;

    std::cout << "[INFO] Starting WarpAffine transformation..." << std::endl;
    
    // 使用WarpAffine进行仿射变换（旋转）
    auto flags = (SimdWarpAffineFlags)(SimdWarpAffineChannelByte | SimdWarpAffineInterpBilinear);

    Simd::WarpAffine(src_view, mat, dst_view, flags, border);

    std::cout << "[INFO] WarpAffine transformation completed successfully" << std::endl;

    // 将BGR输出转换回RGB格式（用于保存PNG）
    std::vector<uint8_t> dst_rgb(dstWidth * dstHeight * channels);
    uint8_t* dst_data = dst_view.data;
    size_t dst_stride = dst_view.stride;

    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            int src_index = y * dst_stride + x * channels;     // BGRA: 4 channels
            int dst_index = (y * dstWidth + x) * channels;     // RGBA: 4 channels

            // BGRA to RGBA conversion (保留Alpha通道)
            dst_rgb[dst_index] = dst_data[src_index + 2];     // R = B
            dst_rgb[dst_index + 1] = dst_data[src_index + 1]; // G = G
            dst_rgb[dst_index + 2] = dst_data[src_index];     // B = R
            dst_rgb[dst_index + 3] = dst_data[src_index + 3]; // A = A (保持不变)
        }
    }

    std::cout << "[INFO] BGRA to RGBA conversion completed for output" << std::endl;

    // 保存PNG图像
    const char* output_file = WORKSPACE_PATH "/output_simd_rotated.png";
    if (stbi_write_png(output_file, dstWidth, dstHeight, channels, dst_rgb.data(), dstWidth * channels))
    {
        std::cout << "[SUCCESS] Image saved successfully: " << output_file << std::endl;
    }
    else
    {
        std::cerr << "[ERROR] Failed to save output.png!" << std::endl;
    }

    // 释放图像资源
    stbi_image_free(src);

    return 0;
}