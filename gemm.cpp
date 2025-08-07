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

#define IREE_HAL_ROCM_MAX_KERNEL_ARG 96

using namespace std;

constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;
constexpr int K_f4x2 = K / 2;
constexpr int K_e8m0 = K / 32;
constexpr int recordRuns = 100;
const char* hsaco_path = "f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co";
const char* kernel_name = "_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E";
constexpr int SUBM = 256;
constexpr int SUBN = 256;

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
struct AiterAsmKernelArgs
{
    void *args_ptr;
    void *arg_size_ptr;
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

float decode_e8m0(uint8_t byte) {
    uint32_t bits = ((uint32_t)byte) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

__bf16 float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(f));
    uint16_t upper = static_cast<uint16_t>(u >> 16);
    return *reinterpret_cast<__bf16*>(&upper);
}

void reference_gemm(const vector<uint8_t> &A, const vector<uint8_t> &B,
                    const vector<uint8_t> &A_scale, const vector<uint8_t> &B_scale,
                    vector<float> &output_ref) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Index into f4x2
                int k_half = k / 2;
                int is_hi = (k % 2 == 0) ? 1 : 0;

                // Get 4-bit A value
                uint8_t A_byte = A[m * K_f4x2 + k_half];
                uint8_t A_val = is_hi ? (A_byte >> 4) & 0xF : A_byte & 0xF;

                // Get 4-bit B value
                uint8_t B_byte = B[n * K_f4x2 + k_half];  // B: N x K/2
                uint8_t B_val = is_hi ? (B_byte >> 4) & 0xF : B_byte & 0xF;

                // Get scale
                int scale_idx = k / 32;
                float A_s = decode_e8m0(A_scale[m * K_e8m0 + scale_idx]);
                float B_s = decode_e8m0(B_scale[n * K_e8m0 + scale_idx]);

                // Dequantize and accumulate
                float A_f = A_s * static_cast<float>(A_val);
                float B_f = B_s * static_cast<float>(B_val);
                acc += A_f * B_f;
            }
            output_ref[m * N + n] = float_to_bf16(acc);
        }
    }
}

void benchmark_module() {
    vector<uint8_t> A(M * K_f4x2, 34); // 00100010 -> 2,2
    vector<uint8_t> B(N * K_f4x2, 17); // 00010001 -> 1,1
    vector<uint8_t> A_scale(M * K_e8m0, 63); // 1.0
    vector<uint8_t> B_scale(N * K_e8m0, 63); // 1.0
    vector<float> bias(M * N, 0);
    vector<__bf16> output(M * N);
    float alpha = 1.0;
    float beta = 0.0;
    int c0 = 0;
    int c1 = 1;

    // Device buffers
    std::cout << "Initializing device data..." << std::endl;
    uint8_t *d_A, *d_B, *d_As, *d_Bs;
    float *d_bias;
    __bf16 *d_output;

    const size_t bytesA = A.size() * sizeof(uint8_t);
    const size_t bytesB = B.size() * sizeof(uint8_t);
    const size_t bytesAs = A_scale.size() * sizeof(uint8_t);
    const size_t bytesBs = B_scale.size() * sizeof(uint8_t);
    const size_t bytesBias = bias.size() * sizeof(float);
    const size_t bytesOutput = output.size() * sizeof(__bf16);

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
    vector<char> hsacoVec = readFileIntoVector(hsaco_path);
    CHECK_HIP_ERROR(hipModuleLoadDataEx(&module, hsacoVec.data(), 0, nullptr, nullptr));
    CHECK_HIP_ERROR(hipModuleGetFunction(&kernel, module, kernel_name));

    // Set up args
    void** kernelParam = (void**)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(void*));
    hipDeviceptr_t* device_ptrs = (hipDeviceptr_t*)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(hipDeviceptr_t));
    for (size_t i = 0; i < IREE_HAL_ROCM_MAX_KERNEL_ARG; i++) {
        kernelParam[i] = &device_ptrs[i];
    }

    *((hipDeviceptr_t*)kernelParam[0]) = d_output;
    *((hipDeviceptr_t*)kernelParam[2]) = d_bias;
    *((hipDeviceptr_t*)kernelParam[4]) = d_A;
    *((hipDeviceptr_t*)kernelParam[6]) = d_B;
    *((float*)kernelParam[8]) = static_cast<float>(alpha);
    *((float*)kernelParam[10]) = static_cast<float>(beta);
    *((uint32_t*)kernelParam[12]) = static_cast<uint32_t>(N);
    *((uint32_t*)kernelParam[14]) = static_cast<uint32_t>(c1);
    *((uint32_t*)kernelParam[16]) = static_cast<uint32_t>(N);
    *((uint32_t*)kernelParam[18]) = static_cast<uint32_t>(c1);
    *((uint32_t*)kernelParam[20]) = static_cast<uint32_t>(K);
    *((uint32_t*)kernelParam[22]) = static_cast<uint32_t>(c1);
    *((uint32_t*)kernelParam[24]) = static_cast<uint32_t>(K);
    *((uint32_t*)kernelParam[26]) = static_cast<uint32_t>(c1);
    *((uint32_t*)kernelParam[28]) = static_cast<uint32_t>(M);
    *((uint32_t*)kernelParam[30]) = static_cast<uint32_t>(N);
    *((uint32_t*)kernelParam[32]) = static_cast<uint32_t>(K);
    *((hipDeviceptr_t*)kernelParam[34]) = d_As;
    *((hipDeviceptr_t*)kernelParam[36]) = d_Bs;
    *((uint32_t*)kernelParam[38]) = static_cast<uint32_t>(K_e8m0);
    *((uint32_t*)kernelParam[40]) = static_cast<uint32_t>(c1);
    *((uint32_t*)kernelParam[42]) = static_cast<uint32_t>(K_e8m0);
    *((uint32_t*)kernelParam[44]) = static_cast<uint32_t>(c1);
    *((int*)kernelParam[46]) = static_cast<int>(c0);

    int bdx = 256, bdy = 1;
    int gdx = (N + SUBN - 1) / SUBN;
    int gdy = (M + SUBM - 1) / SUBM;

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Launch
    std::cout << "Launching GEMM kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    for (uint32_t i = 0; i < recordRuns; ++i) {
        assert(hipModuleLaunchKernel(
            kernel,
            gdx, gdy, 1,
            bdx, bdy, 1,
            0, stream, kernelParam, nullptr) == 0);
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    CHECK_HIP_ERROR(hipMemcpy(output.data(), d_output, bytesOutput, hipMemcpyDeviceToHost));

    // Validate
    std::cout << "Validating..." << std::endl;
    vector<float> output_ref(M * N);
    reference_gemm(A, B, A_scale, B_scale, output_ref);
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

    std::cout << "Average kernel time: " << elapsedTimeMs / recordRuns << " ms\n";
    std::cout << "Finished!" << std::endl;
}

int main(int argc, char *argv[]) {
  benchmark_module();
  return 0;
}
