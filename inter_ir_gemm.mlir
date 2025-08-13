#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], scaled_mma = [<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 163840, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#executable_target_rocm_hsaco_fb1 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx950", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<constants = 10, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module @module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hip
  util.func public @test_gemm_a4w4(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @test_gemm_a4w4(%input0: tensor<256x512xui8>, %input1: tensor<256x512xui8>, %input2: tensor<256x32xf8E8M0FNU>, %input3: tensor<256x32xf8E8M0FNU>, %input4: tensor<256x256xf32>) -> (%output0: tensor<256x256xbf16>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<256x512xui8>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<256x512xui8>
    %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<256x32xf8E8M0FNU>
    %3 = hal.tensor.import %arg3 "input3" : !hal.buffer_view -> tensor<256x32xf8E8M0FNU>
    %4 = hal.tensor.import %arg4 "input4" : !hal.buffer_view -> tensor<256x256xf32>
    %5 = util.call @_test_gemm_a4w4(%0, %1, %2, %3, %4) : (tensor<256x512xui8>, tensor<256x512xui8>, tensor<256x32xf8E8M0FNU>, tensor<256x32xf8E8M0FNU>, tensor<256x256xf32>) -> tensor<256x256xbf16>
    %6 = hal.tensor.export %5 "output0" : tensor<256x256xbf16> -> !hal.buffer_view
    util.return %6 : !hal.buffer_view
  }
  util.func private @_test_gemm_a4w4(%arg0: tensor<256x512xui8>, %arg1: tensor<256x512xui8>, %arg2: tensor<256x32xf8E8M0FNU>, %arg3: tensor<256x32xf8E8M0FNU>, %arg4: tensor<256x256xf32>) -> tensor<256x256xbf16> {
    %c32_i32 = arith.constant 32 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %c0_i32 = arith.constant 0 : i32
    %cast = tensor.cast %arg0 : tensor<256x512xui8> to tensor<?x?xui8>
    %cast_0 = tensor.cast %arg1 : tensor<256x512xui8> to tensor<?x?xui8>
    %cast_1 = tensor.cast %arg2 : tensor<256x32xf8E8M0FNU> to tensor<?x?xf8E8M0FNU>
    %cast_2 = tensor.cast %arg3 : tensor<256x32xf8E8M0FNU> to tensor<?x?xf8E8M0FNU>
    %cast_3 = tensor.cast %arg4 : tensor<256x256xf32> to tensor<?x?xf32>
    %0 = hal.dispatch.extern "f4gemm_kernel_func"[%c256, %c256](%c1065353216_i32, %c0_i32, %c1024_i32, %c1024_i32, %c256_i32, %c256_i32, %c256_i32, %c1024_i32, %c32_i32, %c32_i32, %cast, %cast_0, %cast_1, %cast_2, %cast_3) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?xui8>{%c256, %c512}, tensor<?x?xui8>{%c256, %c512}, tensor<?x?xf8E8M0FNU>{%c256, %c32}, tensor<?x?xf8E8M0FNU>{%c256, %c32}, tensor<?x?xf32>{%c256, %c256}) -> tensor<?x?xbf16>{%c256, %c256} count(%arg5: !hal.device, %arg6: index, %arg7: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c256_5 = arith.constant 256 : index
      %c255 = arith.constant 255 : index
      %2 = arith.addi %arg7, %c255 : index
      %3 = arith.divui %2, %c256_5 : index
      %4 = arith.addi %arg6, %c255 : index
      %5 = arith.divui %4, %c256_5 : index
      hal.return %3, %5, %c1 : index, index, index
    } layout(#pipeline_layout) objects({
      #executable_target_rocm_hsaco_fb1 ordinal(0) = [#hal.executable.object<{path = "f4gemm_outBF16_tn_256x256_scale_ordered_8bytes.s.co"}>]
    }) attributes {subgroupSize = 64 : i64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %cast_4 = tensor.cast %0 : tensor<?x?xbf16> to tensor<256x256xbf16>
    util.return %cast_4 : tensor<256x256xbf16>
  }
}