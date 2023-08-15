import torch 
from thop import profile 

from models import * 

model = dehazeformer_b()

input = torch.randn(1,3,256,256)

macs, params = profile(model, inputs = (input,))

print(macs, params)