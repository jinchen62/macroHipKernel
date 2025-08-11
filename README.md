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
iree-benchmark-module --module=test_gemm.vmfb --function=test_gemm_a4w4 --device=hip://4 --input=16384x8192xui8 --input=16384x8192xui8
```
