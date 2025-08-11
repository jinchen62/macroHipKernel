// iree-opt linalg_example_gemm.mlir -allow-unregistered-dialect

module @module {
  util.func public @test_gemm_a4w4(%arg0: tensor<16384x8192xui8>, %arg1: tensor<16384x8192xui8>) -> (tensor<16384x16384xbf16>) {
    %a_scale = util.unfoldable_constant dense<63.00> : tensor<16384x512xf8E8M0FNU>
    %b_scale = util.unfoldable_constant dense<63.00> : tensor<16384x512xf8E8M0FNU>
    %c = util.unfoldable_constant dense<0.0> : tensor<16384x16384xf32>
    %d = "custom_op.gemm_a4w4" (%arg0, %arg1, %a_scale, %b_scale, %c): (tensor<16384x8192xui8>, tensor<16384x8192xui8>, tensor<16384x512xf8E8M0FNU>, tensor<16384x512xf8E8M0FNU>, tensor<16384x16384xf32>) -> tensor<16384x16384xbf16>
    util.return %d : tensor<16384x16384xbf16>
  }
}
