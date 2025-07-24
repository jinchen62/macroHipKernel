// // Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx950", ukernels = "none"}>

module attributes {transform.with_named_sequence} {
  util.func public @gemm_a4w4_asm_entry_point(%arg0: tensor<16384x8192xui8>, %arg1: tensor<16384x8192xui8>, %arg2: tensor<16384x512xf8E8M0FNU>, %arg3: tensor<16384x512xf8E8M0FNU>, %arg4: tensor<16384x16384xbf16>) -> (tensor<16384x16384xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = tensor.dim %arg0, %c0 : tensor<16384x8192xui8>
    %N = tensor.dim %arg1, %c0 : tensor<16384x8192xui8>
    %k_half = tensor.dim %arg0, %c1 : tensor<16384x8192xui8>
    %k_half_16 = tensor.dim %arg2, %c1 : tensor<16384x512xf8E8M0FNU>
    // %m_256 = (%M + 255) // 256 * 256
    %c255 = arith.constant 255 : index
    %c256 = arith.constant 256 : index
    %add = arith.addi %M, %c255 : index
    %div = arith.divui %add, %c256 : index
    %m_256 = arith.muli %div, %c256 : index
    %out = hal.dispatch.extern "gemm_a4w4_asm"[%M, %N](%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16384x8192xui8>, tensor<16384x8192xui8>, tensor<16384x512xf8E8M0FNU>, tensor<16384x512xf8E8M0FNU>, tensor<16384x16384xbf16>) -> tensor<16384x16384xbf16>
      count(%device: !hal.device, %m: index, %n: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        %subm = arith.constant 256 : index
        %subn = arith.constant 256 : index
        // int gdx = (Ndim + SUBN - 1) / SUBN;
        // int gdy = (Mdim + SUBM - 1) / SUBM;
        %subn_sub_1 = arith.subi %subn, %c1_0 : index
        %n_add = arith.addi %n, %subn_sub_1 : index
        %gdx = arith.divui %n_add, %subn : index
        %subm_sub_1 = arith.subi %subm, %c1_0 : index
        %m_add = arith.addi %m, %subm_sub_1 : index
        %gdy = arith.divui %m_add, %subm : index
        hal.return %gdx, %gdy, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/jincheye/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_tn_256x256.co"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [64 : index, 1 : index, 1 : index]}
    util.return %out : tensor<16384x16384xbf16>
  }

  transform.named_sequence @match_gemm_a4w4(%linalg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %linalg ["custom_op.gemm_a4w4"] : !transform.any_op
    transform.yield %linalg : !transform.any_op
  }

  transform.named_sequence @cast_and_call_gemm_a4w4(%gemm_a4w4: !transform.any_op {transform.readonly}) {
    %module = transform.util.get_nearest_symbol_table %gemm_a4w4 : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @gemm_a4w4_asm_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.print %gemm_a4w4 : !transform.any_op
    // transform.print %func : !transform.any_op
    %ins = transform.get_operand %gemm_a4w4[all] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %gemm_a4w4[all] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %gemm_a4w4 {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_gemm_a4w4 -> @cast_and_call_gemm_a4w4
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    // transform.print %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
