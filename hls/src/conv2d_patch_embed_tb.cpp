// conv2d_patch_embed_tb.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "conv2d_patch_embed.h"

// Load binary tensor of shape [N, C, H, W] from our input file format
bool load_input_bin(
    const char* filename,
    std::vector<float>& data,
    int& N, int& C, int& H, int& W
) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return false;
    }

    int32_t header[4];
    f.read(reinterpret_cast<char*>(header), 4 * sizeof(int32_t));
    if (!f) {
        std::cerr << "Failed to read header from input file\n";
        return false;
    }
    N = header[0];
    C = header[1];
    H = header[2];
    W = header[3];

    size_t total = static_cast<size_t>(N) * C * H * W;
    data.resize(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (!f) {
        std::cerr << "Failed to read input data\n";
        return false;
    }
    return true;
}

// Load Conv2d weight: [C_OUT, C_IN, K, K]
bool load_weight_bin(
    const char* filename,
    float weight[C_OUT][C_IN][K][K]
) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open weight file: " << filename << std::endl;
        return false;
    }

    int32_t header[4];
    f.read(reinterpret_cast<char*>(header), 4 * sizeof(int32_t));
    int oc = header[0];
    int ic = header[1];
    int kh = header[2];
    int kw = header[3];

    if (oc != C_OUT || ic != C_IN || kh != K || kw != K) {
        std::cerr << "Weight shape mismatch in " << filename << std::endl;
        return false;
    }

    size_t total = static_cast<size_t>(C_OUT) * C_IN * K * K;
    std::vector<float> buf(total);
    f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(float));
    if (!f) {
        std::cerr << "Failed to read weight data\n";
        return false;
    }

    // Copy into 4D array
    size_t idx = 0;
    for (int o = 0; o < C_OUT; ++o) {
        for (int i = 0; i < C_IN; ++i) {
            for (int kh_i = 0; kh_i < K; ++kh_i) {
                for (int kw_i = 0; kw_i < K; ++kw_i) {
                    weight[o][i][kh_i][kw_i] = buf[idx++];
                }
            }
        }
    }
    return true;
}

// Load bias: [C_OUT]
bool load_bias_bin(
    const char* filename,
    float bias[C_OUT]
) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open bias file: " << filename << std::endl;
        return false;
    }
    int32_t header[1];
    f.read(reinterpret_cast<char*>(header), sizeof(int32_t));
    int oc = header[0];
    if (oc != C_OUT) {
        std::cerr << "Bias length mismatch in " << filename << std::endl;
        return false;
    }
    std::vector<float> buf(C_OUT);
    f.read(reinterpret_cast<char*>(buf.data()), C_OUT * sizeof(float));
    if (!f) {
        std::cerr << "Failed to read bias data\n";
        return false;
    }

    for (int o = 0; o < C_OUT; ++o) {
        bias[o] = buf[o];
    }
    return true;
}

// Save output [C_OUT, H_OUT, W_OUT] as N=1 tensor
void save_output_bin(
    const char* filename,
    float output[C_OUT][H_OUT][W_OUT]
) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open output file for write: " << filename << std::endl;
        return;
    }

    int32_t header[4] = {1, C_OUT, H_OUT, W_OUT};
    f.write(reinterpret_cast<char*>(header), 4 * sizeof(int32_t));

    for (int oc = 0; oc < C_OUT; ++oc) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
                float val = output[oc][oh][ow];
                f.write(reinterpret_cast<char*>(&val), sizeof(float));
            }
        }
    }
    std::cout << "Saved output to " << filename << std::endl;
}


int main() {
    const int N_REQ = 1000;  // match your input_1000 folder

    std::string base_dir = "data/test_vectors/input_1000";
    std::string input_file = base_dir + "/input_1000.bin";
    std::string weight_file = base_dir + "/patch_embed_conv1_conv_weight.bin";
    std::string bias_file   = base_dir + "/patch_embed_conv1_conv_bias.bin";
    std::string out_file    = base_dir + "/patch_embed_conv1_conv_output_1000_hw.bin";

    // 1. Load input_N.bin
    std::vector<float> input_vec;
    int N, C, H, W;
    if (!load_input_bin(input_file.c_str(), input_vec, N, C, H, W)) {
        return 1;
    }
    std::cout << "Loaded input: N=" << N << ", C=" << C
              << ", H=" << H << ", W=" << W << std::endl;

    if (N < 1 || C != C_IN || H != H_IN || W != W_IN) {
        std::cerr << "Input shape mismatch, expected N>=1, "
                  << "C=" << C_IN << ", H=" << H_IN << ", W=" << W_IN << std::endl;
        return 1;
    }

    // 2. Move first image into [C_IN][H_IN][W_IN] array
    float input_arr[C_IN][H_IN][W_IN];
#pragma HLS ARRAY_PARTITION variable=input_arr complete dim=1
    {
        size_t idx = 0; // index in the first image: N=0
        for (int c = 0; c < C_IN; ++c) {
            for (int h = 0; h < H_IN; ++h) {
                for (int w = 0; w < W_IN; ++w) {
                    // layout in input_vec: NCHW
                    size_t offset = ((size_t)0 * C * H * W)
                                  + (size_t)c * H * W
                                  + (size_t)h * W
                                  + (size_t)w;
                    input_arr[c][h][w] = input_vec[offset];
                }
            }
        }
    }

    // 3. Load weights and bias
    float weight[C_OUT][C_IN][K][K];
    float bias[C_OUT];

    if (!load_weight_bin(weight_file.c_str(), weight)) {
        return 1;
    }
    if (!load_bias_bin(bias_file.c_str(), bias)) {
        return 1;
    }

    // 4. Run convolution
    float output[C_OUT][H_OUT][W_OUT];
    conv2d_patch_embed(input_arr, weight, bias, output);

    // 5. Save output
    save_output_bin(out_file.c_str(), output);

    std::cout << "HLS conv2d patch_embed.conv1.conv test completed.\n";
    return 0;
}
