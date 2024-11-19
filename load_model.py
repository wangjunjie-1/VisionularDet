import torch
import numpy as np
# load ckpt
# ...
input_numpy =  np.random.random((1,3,320,320)).astype(np.float32)
# load pt
pt_path = 'workspace/v5_1/model_best/nanodet_model_best.pt'
torch_pt = torch.load(pt_path,map_location='cpu')
out_numpy = torch_pt(torch.from_numpy(input_numpy))
print(out_numpy.shape)
torch.jit.save(torch.jit.script(torch_pt), 'model_jit.pth')

