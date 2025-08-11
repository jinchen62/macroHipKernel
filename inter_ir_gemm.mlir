#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], scaled_mma = [<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 163840, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#executable_target_rocm_hsaco_fb1 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx950", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<constants = 10, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module @module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hip
  util.func public @test_gemm_a4w4(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @test_gemm_a4w4(%input0: tensor<16384x8192xui8>, %input1: tensor<16384x8192xui8>) -> (%output0: tensor<16384x16384xbf16>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<16384x8192xui8>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<16384x8192xui8>
    %2 = util.call @_test_gemm_a4w4(%0, %1) : (tensor<16384x8192xui8>, tensor<16384x8192xui8>) -> tensor<16384x16384xbf16>
    %3 = hal.tensor.export %2 "output0" : tensor<16384x16384xbf16> -> !hal.buffer_view
    util.return %3 : !hal.buffer_view
  }
  util.func private @_test_gemm_a4w4(%arg0: tensor<16384x8192xui8>, %arg1: tensor<16384x8192xui8>) -> tensor<16384x16384xbf16> {
    %c512_i32 = arith.constant 512 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c512 = arith.constant 512 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16384x16384xf32>
    %cst_0 = arith.constant dense<6.400000e+01> : tensor<16384x512xf8E8M0FNU>
    %0 = util.optimization_barrier %cst_0 : tensor<16384x512xf8E8M0FNU>
    %1 = util.optimization_barrier %cst_0 : tensor<16384x512xf8E8M0FNU>
    %2 = util.optimization_barrier %cst : tensor<16384x16384xf32>
    %cast = tensor.cast %arg0 : tensor<16384x8192xui8> to tensor<?x?xui8>
    %cast_1 = tensor.cast %arg1 : tensor<16384x8192xui8> to tensor<?x?xui8>
    %cast_2 = tensor.cast %0 : tensor<16384x512xf8E8M0FNU> to tensor<?x?xf8E8M0FNU>
    %cast_3 = tensor.cast %1 : tensor<16384x512xf8E8M0FNU> to tensor<?x?xf8E8M0FNU>
    %cast_4 = tensor.cast %2 : tensor<16384x16384xf32> to tensor<?x?xf32>
    %3 = hal.dispatch.extern "f4gemm_kernel_func"[%c16384, %c16384](%c1_i32, %c0_i32, %c16384_i32, %c16384_i32, %c16384_i32, %c16384_i32, %c16384_i32, %c16384_i32, %c512_i32, %c512_i32, %cast, %cast_1, %cast_2, %cast_3, %cast_4) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?xui8>{%c16384, %c8192}, tensor<?x?xui8>{%c16384, %c8192}, tensor<?x?xf8E8M0FNU>{%c16384, %c512}, tensor<?x?xf8E8M0FNU>{%c16384, %c512}, tensor<?x?xf32>{%c16384, %c16384}) -> tensor<?x?xbf16>{%c16384, %c16384} count(%arg2: !hal.device, %arg3: index, %arg4: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c255 = arith.constant 255 : index
      %5 = arith.addi %arg4, %c255 : index
      %6 = arith.divui %5, %c256 : index
      %7 = arith.addi %arg3, %c255 : index
      %8 = arith.divui %7, %c256 : index
      hal.return %6, %8, %c1 : index, index, index
    } layout(#pipeline_layout) objects({
      #executable_target_rocm_hsaco_fb1 ordinal(0) = [#hal.executable.object<{path = "/home/jincheye/macroHipKernel/f4gemm_outBF16_tn_256x256_scale_ordered_grouped_8bytes.s.co"}>]
    }) attributes {subgroupSize = 64 : i64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %cast_5 = tensor.cast %3 : tensor<?x?xbf16> to tensor<16384x16384xbf16>
    util.return %cast_5 : tensor<16384x16384xbf16>
  }
}