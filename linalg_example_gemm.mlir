// iree-opt linalg_example_gemm.mlir -allow-unregistered-dialect

module @module {
  util.func public @test_gemm_a4w4(%arg0: tensor<256x512xui8>, %arg1: tensor<256x512xui8>, %arg2: tensor<256x32xf8E8M0FNU>, %arg3: tensor<256x32xf8E8M0FNU>, %arg4: tensor<256x256xf32>) -> (tensor<256x256xbf16>) {
    %out = "custom_op.gemm_a4w4" (%arg0, %arg1, %arg2, %arg3, %arg4): (tensor<256x512xui8>, tensor<256x512xui8>, tensor<256x32xf8E8M0FNU>, tensor<256x32xf8E8M0FNU>, tensor<256x256xf32>) -> tensor<256x256xbf16>
    util.return %out : tensor<256x256xbf16>
  }
}
