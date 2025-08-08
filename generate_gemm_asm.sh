rm $1.co
~/iree-build/llvm-project/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -c f4gemm_outBF16_tn_256x256_scale.s -o f4gemm_outBF16_tn_256x256_scale.s.o
~/iree-build/llvm-project/bin/lld -flavor gnu -shared f4gemm_outBF16_tn_256x256_scale.s.o -o f4gemm_outBF16_tn_256x256_scale.s.co
rm $1.o
