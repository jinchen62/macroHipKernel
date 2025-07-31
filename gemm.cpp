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

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
    // p3 _p23;
};

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
                    float* output_ref, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = unpack_f4x2(A[m * (K / 2) + k / 2]) * scale_e8m0(A_scale[m * (K / 32) + k / 32]);
                float b = unpack_f4x2(B[n * (K / 2) + k / 2]) * scale_e8m0(B_scale[n * (K / 32) + k / 32]);
                acc += a * b;
            }
            output_ref[m * N + n] = acc;
        }
    }
}

void benchmark_module() {
    std::vector<uint8_t> A(M * K_f4x2, 1);
    std::vector<uint8_t> B(N * K_f4x2, 1);
    std::vector<uint8_t> A_scale(M * K_e8m0, 1);
    std::vector<uint8_t> B_scale(N * K_e8m0, 1);
    std::vector<float> bias(M * N, 0);
    std::vector<__bf16> output(M * N);
    std::vector<float> output_ref(M * N);
    float alpha = 1.0;
    float beta = 0.0;
    int c0 = 0;
    int c1 = 1;

    // Device buffers
    uint8_t *d_A, *d_B, *d_As, *d_Bs;
    float *d_bias;
    __bf16 *d_output;

    size_t bytesA = A.size() * sizeof(uint8_t);
    size_t bytesB = B.size() * sizeof(uint8_t);
    size_t bytesAs = A_scale.size() * sizeof(uint8_t);
    size_t bytesBs = B_scale.size() * sizeof(uint8_t);
    size_t bytesBias = bias.size() * sizeof(float);
    size_t bytesOutput = output.size() * sizeof(__bf16);

    CHECK_HIP_ERROR(hipMalloc(&d_A, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_B, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_As, bytesAs));
    CHECK_HIP_ERROR(hipMalloc(&d_Bs, bytesBs));
    CHECK_HIP_ERROR(hipMalloc(&d_bias, bytesBias));
    CHECK_HIP_ERROR(hipMalloc(&d_output, bytesOutput));

    CHECK_HIP_ERROR(hipMemcpy(d_A, A.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_As, A_scale.data(), bytesAs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_Bs, B_scale.data(), bytesBs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_bias, bias.data(), bytesBias, hipMemcpyHostToDevice));

    // Load kernel
    hipModule_t module;
    hipFunction_t kernel;
    auto hsacoVec = readFileIntoVector(hsaco_path);
    CHECK_HIP_ERROR(hipModuleLoadDataEx(&module, hsacoVec.data(), 0, nullptr, nullptr));
    CHECK_HIP_ERROR(hipModuleGetFunction(&kernel, module, kernel_name));

    KernelArgs args;
    args.ptr_D          = d_output;
    args.ptr_C          = d_bias;
    args.ptr_A          = d_A;
    args.ptr_B          = d_B;
    args.alpha          = alpha;
    args.beta           = beta;
    args.stride_C0      = N;
    args.stride_A0      = K;
    args.stride_B0      = K;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.ptr_ScaleA     = d_As;
    args.ptr_ScaleB     = d_Bs;
    args.stride_ScaleA0 = K_e8m0;
    args.stride_ScaleB0 = K_e8m0;
    args.log2_k_split   = 0;

    void* d_args;
    CHECK_HIP_ERROR(hipMalloc(&d_args, sizeof(args)));
    CHECK_HIP_ERROR(hipMemcpy(d_args, &args, sizeof(args), hipMemcpyHostToDevice));
    void* kernelParam[] = { &d_args };

    // Kernel args
    // void** kernelParam = (void**)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(void*));
    // hipDeviceptr_t* device_ptrs = (hipDeviceptr_t*)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(hipDeviceptr_t));
    // for (size_t i = 0; i < IREE_HAL_ROCM_MAX_KERNEL_ARG; i++) {
    //     kernelParam[i] = &device_ptrs[i];
    // }

    // *((hipDeviceptr_t*)kernelParam[0]) = (hipDeviceptr_t)d_args;
    // *((hipDeviceptr_t*)kernelParam[0]) = (hipDeviceptr_t)d_output;
    // *((hipDeviceptr_t*)kernelParam[2]) = (hipDeviceptr_t)d_bias;
    // *((hipDeviceptr_t*)kernelParam[4]) = (hipDeviceptr_t)d_A;
    // *((hipDeviceptr_t*)kernelParam[6]) = (hipDeviceptr_t)d_B;
    // *((float*)kernelParam[8]) = alpha;
    // *((float*)kernelParam[10]) = beta;
    // *((uint32_t*)kernelParam[12]) = static_cast<uint32_t>(N);
    // *((uint32_t*)kernelParam[14]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[16]) = static_cast<uint32_t>(N);
    // *((uint32_t*)kernelParam[18]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[20]) = static_cast<uint32_t>(K);
    // *((uint32_t*)kernelParam[22]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[24]) = static_cast<uint32_t>(K);
    // *((uint32_t*)kernelParam[26]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[28]) = static_cast<uint32_t>(M);
    // *((uint32_t*)kernelParam[30]) = static_cast<uint32_t>(N);
    // *((uint32_t*)kernelParam[32]) = static_cast<uint32_t>(K);
    // *((hipDeviceptr_t*)kernelParam[34]) = (hipDeviceptr_t)d_As;
    // *((hipDeviceptr_t*)kernelParam[36]) = (hipDeviceptr_t)d_Bs;
    // *((uint32_t*)kernelParam[38]) = static_cast<uint32_t>(K_e8m0);
    // *((uint32_t*)kernelParam[40]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[42]) = static_cast<uint32_t>(K_e8m0);
    // *((uint32_t*)kernelParam[44]) = static_cast<uint32_t>(c1);
    // *((uint32_t*)kernelParam[46]) = static_cast<uint32_t>(c0);

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
                                     0, 0,
                                     kernelParam, nullptr) == 0);
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedTimeMs;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    std::cout << "Average kernel time: " << elapsedTimeMs / recordRuns << " ms\n";

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    CHECK_HIP_ERROR(hipMemcpy(output.data(), d_output, bytesOutput, hipMemcpyDeviceToHost));

    // Validate
    reference_gemm(A.data(), B.data(), A_scale.data(), B_scale.data(), output_ref.data(), M, N, K);

    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = __bfloat162float(output[i]);
        float ref_val = output_ref[i];
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
    CHECK_HIP_ERROR(hipFree(d_bias));
    CHECK_HIP_ERROR(hipFree(d_output));
}

int main(int argc, char *argv[]) {
  benchmark_module();
  return 0;
}
