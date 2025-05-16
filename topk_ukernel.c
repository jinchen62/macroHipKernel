// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define MAX_K 32  // Upper limit for K, safe for stack on GPU

extern "C" __device__ __attribute__((const)) half __ockl_wfred_max_f16(half);
extern "C" __device__ __attribute__((const))
int64_t __ockl_wfred_min_i64(int64_t);
extern "C" __device__ __attribute__((const))
int32_t __ockl_wfred_min_i32(int32_t);

/*
Constraint/Tiling note:
For simplicity, we distribute all parallel dim across different workgroup, and
only use single subgroup/warp per workgroup. This constraint is also set during
tiling phase in KernelConfig.
*/

extern "C" __global__ void topk_F16I32(const float* __restrict__ inputBuffer,
                                       float* __restrict__ outputValues,
                                       int64_t* __restrict__ outputIndices,
                                       int reductionSize) {
  int k = 4;
  uint laneID = __builtin_amdgcn_workitem_id_x();
  // Set identity value to handle problem non divisible by subgroupSize.
  float topk_vals[MAX_K];
  int64_t topk_indices[MAX_K];
  // Initialize topk values to identity (-FLT_MAX for max)
  for (int i = 0; i < k; ++i) {
    topk_vals[i] = -FLT_MAX;
    topk_indices[i] = -1;
  }

  uint numBatches = (reductionSize + warpSize - 1) / warpSize;
  for (int i = 0; i < numBatches; ++i) {
    uint idx = warpSize * i + laneID;
    float val = idx < reductionSize ? inputBuffer[idx] : -FLT_MAX;

    // Insert into local top-k buffer
    for (int j = 0; j < k; ++j) {
      if (val > topk_vals[j]) {
        // Shift down
        for (int m = k - 1; m > j; --m) {
          topk_vals[m] = topk_vals[m - 1];
          topk_indices[m] = topk_indices[m - 1];
        }
        topk_vals[j] = val;
        topk_indices[j] = idx;
        break;
      }
    }
  }

  // Collect and merge top-k from all lanes
  __shared__ float warp_topk_vals[warpSize * MAX_K];
  __shared__ int64_t warp_topk_indices[warpSize * MAX_K];

  for (int i = 0; i < k; ++i) {
    warp_topk_vals[laneID * k + i] = topk_vals[i];
    warp_topk_indices[laneID * k + i] = topk_indices[i];
  }

  __syncthreads();

  // Merge in lane 0
  if (laneID == 0) {
    // Naive partial sort of k * warpSize
    for (int i = 0; i < warpSize * k; ++i) {
      for (int j = i + 1; j < warpSize * k; ++j) {
        if (warp_topk_vals[j] > warp_topk_vals[i]) {
          float tmp_v = warp_topk_vals[i];
          int64_t tmp_i = warp_topk_indices[i];
          warp_topk_vals[i] = warp_topk_vals[j];
          warp_topk_indices[i] = warp_topk_indices[j];
          warp_topk_vals[j] = tmp_v;
          warp_topk_indices[j] = tmp_i;
        }
      }
    }
    for (int i = 0; i < k; ++i) {
      outputValues[i] = warp_topk_vals[i];
      outputIndices[i] = warp_topk_indices[i];
    }
  }
}
