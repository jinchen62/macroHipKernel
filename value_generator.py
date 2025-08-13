import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, perftest
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]

@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    if log2_k_split is not None:
        out_reset = torch.zeros(
            (out.shape[0] + 255) // 256 * 256, out.shape[1], dtype=dtype
        )
        out = out_reset

    return aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )

@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx
    if get_gfx() not in ["gfx950"]:
        return

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    # wshuffle = shuffle_weight(w, layout=(16, 16))
    out = torch.empty((M + 255) // 256 * 256, N, dtype=dtype)
    bias_f32 = torch.zeros((M, N), dtype=torch.float32)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)

    # x = torch.full((M, 512), 34, dtype=torch.uint8)
    # w = torch.full((M, 512), 17, dtype=torch.uint8)
    # x_scales_shuffle = torch.full((M, 32), 2.0, dtype=torch.float8_e8m0fnu)
    # w_scales_shuffle = torch.full((M, 32), 1.0, dtype=torch.float8_e8m0fnu)

    a.detach().contiguous().to(torch.bfloat16).cpu().view(torch.uint16).numpy().tofile("output_torch.bin")

    x.detach().contiguous().to(torch.uint8).cpu().numpy().tofile("input_x.bin")
    w.detach().contiguous().to(torch.uint8).cpu().numpy().tofile("input_w.bin")
    x_scales_shuffle.detach().contiguous().to(torch.uint8).cpu().numpy().tofile("input_x_scale.bin")
    w_scales_shuffle.detach().contiguous().to(torch.uint8).cpu().numpy().tofile("input_w_scale.bin")
    bias_f32.detach().contiguous().to(torch.float32).cpu().numpy().tofile("input_bias.bin")

    err_c = None
    avg_c = None
    tflops_c = None
    tbs_c = None
    c, avg_c = run_gemm_asm(
        x,
        w,
        x_scales_shuffle,
        w_scales_shuffle,
        out,
        "_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E",  # kernelName
        bias_f32,
        bpreshuffle=False,
        log2_k_split=0,
    )

    c.detach().contiguous().to(torch.bfloat16).cpu().view(torch.uint16).numpy().tofile("output_asm.bin")

    err_c = checkAllclose(a, c[:M], msg="asm no splitK  ")
    tflops_c = M * N * K * 2 / avg_c / 1e6
    tbs_c = (x.nbytes + w.nbytes) / avg_c / 1e6

    return {
        "asm no splitK": avg_c,
        "asm no splitK err": err_c,
        "asm no splitK TFLOPS": tflops_c,
        "asm no splitK TB/s": tbs_c,
    }


l_dtype = ["bf16"]
l_mnk = [
    # pure_compute
    (256, 2048, 8192),
    (2048, 8192, 8192),
    (16384, 16384, 16384),
    (32768, 106496, 16384),
    (32768, 16384, 53248),
    (32768, 18432, 16384),
    (32768, 16384, 16384),
    (128, 106496, 16384),
    (128, 16384, 53248),
    (128, 18432, 16384),
    (128, 16384, 16384),
    (64, 106496, 16384),
    (64, 16384, 53248),
    (64, 18432, 16384),
    (64, 16384, 16384),
    (64, 106496, 16384),
    (32, 106496, 16384),
    (32, 16384, 53248),
    (32, 18432, 16384),
    (32, 16384, 16384),
    # qkv_proj
    (1, 1280, 8192),
    (64, 1280, 8192),
    (127, 1280, 8192),
    (129, 1280, 8192),
    (65, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_mnk = [args.shape]

    for dtype in l_dtype:
        for m, n, k in l_mnk:
            ret = test_gemm(dtype, m, n, k)
            print(ret)
