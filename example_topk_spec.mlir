// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>

module attributes {transform.with_named_sequence} {
  util.func private @topk_3d_f16_entry_point(%arg0: tensor<1x1x?xf16>) -> (tensor<1x1x4xf16>, tensor<1x1x4xi32>) {
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x1x?xf16>
    %dim_i32 = arith.index_cast %dim : index to i32
    %4:2 = hal.dispatch.extern "topk_F16I32"[%dim](%dim_i32, %arg0) : (i32, tensor<1x1x?xf16>{%dim}) -> tensor<1x1x4xf16>, tensor<1x1x4xi32>
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        hal.return %c1_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<constants = 2, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/jinchen/macroHipKernel/topk_ukernel.c.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 32, workgroup_size = [32 : index, 1 : index, 1 : index]}
    util.return %4#0, %4#1 : tensor<1x1x4xf16>, tensor<1x1x4xi32>
  }

  transform.named_sequence @match_topk(%linalg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %linalg ["iree_linalg_ext.topk"] : !transform.any_op
    // transform.print %linalg {name="matched op"} : !transform.any_op
    // %matched = transform.match.structured failures(propagate) %linalg : (!transform.any_op) -> (!transform.any_op) {
    // ^bb1(%topk: !transform.any_op):
    //   // %c2 = transform.param.constant 2 : i32 -> !transform.param<i32>
    //   // %c3 = transform.param.constant 3 : i32 -> !transform.param<i32>
    //   // %rank = transform.match.structured.rank %topk : (!transform.any_op) -> !transform.param<i32>
    //   // transform.match.param.cmpi eq %rank, %c3 : !transform.param<i32>
    //   // %n_inputs = transform.match.structured.num_inputs %topk : (!transform.any_op) -> !transform.param<i32>
    //   // transform.match.param.cmpi eq %n_inputs, %c2 : !transform.param<i32>
    //   // %n_outputs = transform.match.structured.num_inits %topk : (!transform.any_op) -> !transform.param<i32>
    //   // transform.match.param.cmpi eq %n_outputs, %c2 : !transform.param<i32>
    //   transform.match.structured.yield %topk : !transform.any_op 
    // }

    %in0 = transform.get_operand %linalg[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<1x1x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[2], 64 : !transform.any_value
    // %out0 = transform.get_result %linalg[0] : (!transform.any_op) -> !transform.any_value
    // transform.iree.match.cast_compatible_type %out0 = tensor<1x1x4xf16> : !transform.any_value
    // %out1 = transform.get_result %linalg[1] : (!transform.any_op) -> !transform.any_value
    // transform.iree.match.cast_compatible_type %out1 = tensor<1x1x4xi32> : !transform.any_value

     // transform.iree.match.regions %linalg : !transform.any_op {
    //   ^bb0(%input: tensor<1x1x?xf16>, %indices: tensor<1x1x?xi32>, %empty_value: tensor<1x1x4xf16>, %empty_idx: tensor<1x1x4xi32>):
    //     %6:2 = iree_linalg_ext.topk dimension(2)
    //                                 ins(%input, %indices : tensor<1x1x?xf16>, tensor<1x1x?xi32>)
    //                                 outs(%empty_value, %empty_idx : tensor<1x1x4xf16>, tensor<1x1x4xi32>) {
    //     ^bb0(%arg1: f16, %arg2: f16):
    //       %7 = arith.cmpf ogt, %arg1, %arg2 : f16
    //       iree_linalg_ext.yield %7 : i1
    //     } -> tensor<1x1x4xf16>, tensor<1x1x4xi32>
    // }
    transform.yield %linalg : !transform.any_op
  }

  transform.named_sequence @cast_and_call_topk(%topk: !transform.any_op {transform.readonly}) {
    %module = transform.util.get_nearest_symbol_table %topk : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @topk_3d_f16_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %topk[0] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %topk[all] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %topk {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_topk -> @cast_and_call_topk
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
