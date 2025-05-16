// iree-compile --iree-hal-target-device=hip --iree-hip-target=gfx942 torch_example_topk.mlir -o torch_example_topk.vmfb
// iree-benchmark-module --module=torch_example_topk.vmfb --function=topk --device=hip://4 --input=1x32000xf32

func.func @topk(%arg0: !torch.vtensor<[1,32000],f32>) -> (!torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],si64>) {
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %values, %indices = torch.aten.topk %arg0, %int4, %int1, %true, %true : !torch.vtensor<[1,32000],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],si64>
    return %values, %indices : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],si64>
}

// module {
//   func.func @topk(%arg0: !torch.vtensor<[1,32000],f32>) -> (!torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],si64>) {
//     %int0 = torch.constant.int 0
//     %true = torch.constant.bool true
//     %int1 = torch.constant.int 1
//     %int4 = torch.constant.int 4
//     %values, %indices = torch.aten.sort %arg0, %int1, %true : !torch.vtensor<[1,32000],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,32000],f32>, !torch.vtensor<[1,32000],si64>
//     %0 = torch.aten.slice.Tensor %values, %int1, %int0, %int4, %int1 : !torch.vtensor<[1,32000],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
//     %1 = torch.aten.slice.Tensor %indices, %int1, %int0, %int4, %int1 : !torch.vtensor<[1,32000],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],si64>
//     return %0, %1 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],si64>
//   }
// }

// #map = affine_map<(d0, d1) -> (d0, d1)>
// module {
//   func.func @topk(%arg0: tensor<1x32000xf32>) -> (tensor<1x4xf32>, tensor<1x4xi64>) {
//     %0 = tensor.empty() : tensor<1x32000xi64>
//     %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<1x32000xi64>) {
//     ^bb0(%out: i64):
//       %3 = linalg.index 1 : index
//       %4 = arith.index_cast %3 : index to i64
//       linalg.yield %4 : i64
//     } -> tensor<1x32000xi64>
//     %2:2 = tm_tensor.sort dimension(1) outs(%arg0, %1 : tensor<1x32000xf32>, tensor<1x32000xi64>) {
//     ^bb0(%arg1: f32, %arg2: f32, %arg3: i64, %arg4: i64):
//       %3 = arith.cmpf oge, %arg1, %arg2 : f32
//       tm_tensor.yield %3 : i1
//     } -> tensor<1x32000xf32>, tensor<1x32000xi64>
//     %extracted_slice = tensor.extract_slice %2#0[0, 0] [1, 4] [1, 1] : tensor<1x32000xf32> to tensor<1x4xf32>
//     %extracted_slice_0 = tensor.extract_slice %2#1[0, 0] [1, 4] [1, 1] : tensor<1x32000xi64> to tensor<1x4xi64>
//     return %extracted_slice, %extracted_slice_0 : tensor<1x4xf32>, tensor<1x4xi64>
//   }
// }
