// iree-opt linalg_example_gemm.mlir -allow-unregistered-dialect

module @module {
  util.func public @test_gemm_a4w4(%arg0: tensor<16384x8192xui8>, %arg1: tensor<16384x8192xui8>) -> (tensor<16384x16384xbf16>) {
    %x_scale = util.unfoldable_constant dense<1.00> : tensor<16384x512xf8E8M0FNU>
    %w_scale = util.unfoldable_constant dense<1.00> : tensor<16384x512xf8E8M0FNU>
    %bias = util.unfoldable_constant dense<0.0> : tensor<16384x16384xbf16>
    %out = "custom_op.gemm_a4w4" (%arg0, %arg1, %x_scale, %w_scale, %bias): (tensor<16384x8192xui8>, tensor<16384x8192xui8>, tensor<16384x512xf8E8M0FNU>, tensor<16384x512xf8E8M0FNU>, tensor<16384x16384xbf16>) -> tensor<16384x16384xbf16>
    util.return %out : tensor<16384x16384xbf16>
  }
}
