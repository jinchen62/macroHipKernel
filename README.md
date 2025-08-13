# macroHipKernel

## Instructions to compile and test

C++ test

```sh
hipcc gemm.cpp -o gemm_test
./gemm_test
```

IREE test

```sh
iree-compile inter_ir_gemm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx950 -o test_gemm.vmfb
iree-benchmark-module --module=test_gemm.vmfb --function=test_gemm_a4w4 --device=hip://4 --input=256x512xui8 --input=256x512xui8 --input=256x32xf8E8M0FNU --input=256x32xf8E8M0FNU --input=256x256xf32
iree-run-module --module=test_gemm.vmfb --function=test_gemm_a4w4 --device=hip://4 --input=256x512xui8=@input_x.bin --input=256x512xui8=@input_w.bin --input=256x32xf8E8M0FNU=@input_x_scale.bin --input=256x32xf8E8M0FNU=@input_w_scale.bin --input=256x256xf32=@input_bias.bin --expected_output=256x256xbf16=@output_asm.bin
```
