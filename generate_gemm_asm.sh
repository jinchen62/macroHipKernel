~/iree-build/llvm-project/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -c $1 -o $1.o
~/iree-build/llvm-project/bin/lld -flavor gnu -shared $1.o -o $1.co
xxd -p -c 1000000 $1.co > $1.hex
rm $1.o $1.co
