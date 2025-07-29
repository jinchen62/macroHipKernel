#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <mutex>
#include <random>
#include <cmath>
#include <chrono>
#include "utils.h"

#define IREE_HAL_ROCM_MAX_KERNEL_ARG 128

using namespace std;

constexpr int M = 16384;
constexpr int N = 16384;
constexpr int K = 16384;
constexpr int K_f4x2 = K / 2;
constexpr int K_e8m0 = K / 32;
constexpr int recordRuns = 100;
const char* hsaco_path = "/home/jincheye/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co";
const char* kernel_name = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E";

std::vector<char> readFileIntoVector(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return std::vector<char>();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

// Simulated unpack/dequant for validation
float unpack_f4x2(uint8_t val) {
    return static_cast<float>(val);  // for now just identity
}

float scale_e8m0(uint8_t val) {
    return static_cast<float>(val) / 255.0f;
}

void reference_gemm(const uint8_t* A, const uint8_t* B,
                    const uint8_t* A_scale, const uint8_t* B_scale,
                    float* C_ref, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = unpack_f4x2(A[m * (K / 2) + k / 2]) * scale_e8m0(A_scale[m * (K / 32) + k / 32]);
                float b = unpack_f4x2(B[n * (K / 2) + k / 2]) * scale_e8m0(B_scale[n * (K / 32) + k / 32]);
                acc += a * b;
            }
            C_ref[m * N + n] = acc;
        }
    }
}

void benchmark_module() {
    std::vector<uint8_t> A(M * K_f4x2, 1);
    std::vector<uint8_t> B(N * K_f4x2, 1);
    std::vector<uint8_t> A_scale(M * K_e8m0, 1);
    std::vector<uint8_t> B_scale(N * K_e8m0, 1);
    std::vector<__bf16> C(M * N, 0);
    std::vector<__bf16> D(M * N);
    std::vector<float> C_ref(M * N);

    // Device buffers
    uint8_t *d_A, *d_B, *d_As, *d_Bs;
    __bf16 *d_C, *d_D;

    size_t bytesA = A.size() * sizeof(uint8_t);
    size_t bytesB = B.size() * sizeof(uint8_t);
    size_t bytesAs = A_scale.size() * sizeof(uint8_t);
    size_t bytesBs = B_scale.size() * sizeof(uint8_t);
    size_t bytesC = C.size() * sizeof(__bf16);
    size_t bytesD = D.size() * sizeof(__bf16);

    CHECK_HIP_ERROR(hipMalloc(&d_A, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_B, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_As, bytesAs));
    CHECK_HIP_ERROR(hipMalloc(&d_Bs, bytesBs));
    CHECK_HIP_ERROR(hipMalloc(&d_C, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_D, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_A, A.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_As, A_scale.data(), bytesAs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_Bs, B_scale.data(), bytesBs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, C.data(), bytesC, hipMemcpyHostToDevice));

    // Load kernel
    hipModule_t module;
    hipFunction_t kernel;
    auto hsacoVec = readFileIntoVector(hsaco_path);
    CHECK_HIP_ERROR(hipModuleLoadDataEx(&module, hsacoVec.data(), 0, nullptr, nullptr));
    CHECK_HIP_ERROR(hipModuleGetFunction(&kernel, module, kernel_name));

    // Kernel args
    void** kernelParam = (void**)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(void*));
    hipDeviceptr_t* device_ptrs = (hipDeviceptr_t*)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(hipDeviceptr_t));
    for (size_t i = 0; i < IREE_HAL_ROCM_MAX_KERNEL_ARG; i++) {
        kernelParam[i] = &device_ptrs[i];
    }

    *((hipDeviceptr_t*)kernelParam[0]) = (hipDeviceptr_t)d_A;
    *((hipDeviceptr_t*)kernelParam[1]) = (hipDeviceptr_t)d_B;
    *((hipDeviceptr_t*)kernelParam[2]) = (hipDeviceptr_t)d_As;
    *((hipDeviceptr_t*)kernelParam[3]) = (hipDeviceptr_t)d_Bs;
    *((hipDeviceptr_t*)kernelParam[4]) = (hipDeviceptr_t)d_D;
    *((hipDeviceptr_t*)kernelParam[5]) = (hipDeviceptr_t)kernel_name;
    *((hipDeviceptr_t*)kernelParam[6]) = (hipDeviceptr_t)d_C;

    int subm = 256, subn = 256;
    int grid_x = (N + subn - 1) / subn;
    int grid_y = (M + subm - 1) / subm;
    std::cout << "grid_x: " << grid_x << "\n";
    std::cout << "grid_y: " << grid_y << "\n";

    // Launch
    std::cout << "Launching GEMM kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    for (uint32_t i = 0; i < recordRuns; ++i) {
        assert(hipModuleLaunchKernel(kernel, grid_x, grid_y, 1,
                                     256, 1, 1,
                                     0, nullptr,
                                     kernelParam, nullptr) == 0);
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedTimeMs;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    std::cout << "Average kernel time: " << elapsedTimeMs / recordRuns << " ms\n";

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    CHECK_HIP_ERROR(hipMemcpy(D.data(), d_D, bytesD, hipMemcpyDeviceToHost));

    // Validate
    reference_gemm(A.data(), B.data(), A_scale.data(), B_scale.data(), C_ref.data(), M, N, K);

    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = __bfloat162float(C[i]);
        float ref_val = C_ref[i];
        float diff = std::abs(gpu_val - ref_val);
        if (diff > 1e-2) {
            std::cout << "Mismatch at " << i << ": GPU = " << gpu_val << ", REF = " << ref_val << "\n";
            correct = false;
        }
    }

    if (correct)
        std::cout << "GEMM kernel validated successfully!\n";
    else
        std::cerr << "GEMM kernel failed validation!\n";

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_As));
    CHECK_HIP_ERROR(hipFree(d_Bs));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipFree(d_D));
}

int main(int argc, char *argv[]) {
  benchmark_module();
  return 0;
}
