import numpy as np

# batch = 1
# reductionSize = 32000 # tried 32000, 32, 10
# label = 98
# data = np.ones([batch, reductionSize]).astype(np.float32)*7
# # data = np.random.normal(size=[batch, reductionSize]).astype(np.float32)
# data[0, label] = 53.0
# np.save("input0.npy", data)
# data_fp16 = data.astype(np.float16)
# np.save("input0_f16.npy", data_fp16)

# batch0 = 1
# batch1 = 1
# reductionSize = 32000 # tried 32000, 32, 10
# argmax_input = np.random.normal(size=[batch0, batch1, reductionSize]).astype(np.float32)
# argmax_output = np.argmax(argmax_input,axis=-1).astype(np.float32)
# np.save("argmax_3d_input_f32.npy", argmax_input)
# np.save("argmax_3d_output_f32.npy", argmax_output)


# argmax_input = np.random.normal(size=[batch0, batch1, reductionSize]).astype(np.float16)
# argmax_output = np.argmax(argmax_input,axis=-1).astype(np.float32)
# np.save("argmax_3d_input_f16.npy", argmax_input)
# np.save("argmax_3d_output_f16.npy", argmax_output)


batch = 8
reductionSize = 131072
inputs  = np.zeros([batch, 1, reductionSize]).astype(np.float32)
inputs[0, 0, 1001:1005] = [250, 249, 248, 247]
inputs[1, 0, 2001:2005] = [250, 249, 248, 247]
inputs[2, 0, 3001:3005] = [250, 249, 248, 247]
inputs[3, 0, 4001:4005] = [250, 249, 248, 247]
inputs[4, 0, 5001:5005] = [250, 249, 248, 247]
inputs[5, 0, 6001:6005] = [250, 249, 248, 247]
inputs[6, 0, 7001:7005] = [250, 249, 248, 247]
inputs[7, 0, 8001:8005] = [250, 249, 248, 247]
np.save("input0.npy", inputs)

inputs_fp16 = inputs.astype(np.float16)
np.save("input0_f16.npy", inputs_fp16)

import torch
inputs_torch = torch.tensor(inputs, dtype=torch.float32)
inputs_torch_f16 = torch.tensor(inputs_fp16, dtype=torch.float16)
values, indices = torch.topk(inputs_torch, k=4, dim=2)
# print(values)
# print(indices)
# tensor([[[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]],
#         [[250., 249., 248., 247.]]])
# tensor([[[1001, 1002, 1003, 1004]],
#         [[2001, 2002, 2003, 2004]],
#         [[3001, 3002, 3003, 3004]],
#         [[4001, 4002, 4003, 4004]],
#         [[5001, 5002, 5003, 5004]],
#         [[6001, 6002, 6003, 6004]],
#         [[7001, 7002, 7003, 7004]],
#         [[8001, 8002, 8003, 8004]]])
