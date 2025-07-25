#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 163840, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#executable_target_rocm_hsaco_fb1 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx950", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module @module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hip
  util.func public @topk_k4(%arg0: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @topk_k4(%input0: tensor<8x2x131072xf16>) -> (%output0: tensor<8x2x4xf16>, %output1: tensor<8x2x4xi32>)"}} {
    %c131072_i32 = arith.constant 131072 : i32
    %c131072 = arith.constant 131072 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<8x2x131072xf16>
    %cast = tensor.cast %0 : tensor<8x2x131072xf16> to tensor<?x?x?xf16>
    %1:2 = hal.dispatch.extern "topk_F16I32"[%c8, %c2](%c131072_i32, %cast) : (i32, tensor<?x?x?xf16>{%c8, %c2, %c131072}) -> (tensor<?x?x4xf16>{%c8, %c2}, tensor<?x?x4xi32>{%c8, %c2}) count(%arg1: !hal.device, %arg2: index, %arg3: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %arg2, %arg3, %c1 : index, index, index
    } layout(#pipeline_layout) objects({
      #executable_target_rocm_hsaco_fb1 ordinal(0) = [#hal.executable.object<{path = "/home/jincheye/macroHipKernel/topk_ukernel_f16i32.c.hsaco"}>]
    }) attributes {subgroupSize = 64 : i64, workgroup_size = [64 : index, 1 : index, 1 : index]}
    %cast_0 = tensor.cast %1#0 : tensor<?x?x4xf16> to tensor<8x2x4xf16>
    %cast_1 = tensor.cast %1#1 : tensor<?x?x4xi32> to tensor<8x2x4xi32>
    %2 = hal.tensor.export %cast_0 "output0" : tensor<8x2x4xf16> -> !hal.buffer_view
    %3 = hal.tensor.export %cast_1 "output1" : tensor<8x2x4xi32> -> !hal.buffer_view
    util.return %2, %3 : !hal.buffer_view, !hal.buffer_view
  }
}