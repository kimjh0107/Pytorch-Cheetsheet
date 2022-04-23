# %% 
# ================================================================== #
#                        Initializing Tensor                         #
# ================================================================== #
import torch
from zmq import device
# %%
torch.__version__

# %%
# Initializing Tensor 
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], 
                          [4,5,6]], 
                          dtype = torch.float32, device = device, requires_grad = True)
# %%
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)
print(my_tensor.shape)
# %%
# Other common initialization methods 
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))  # values 0~1
x = torch.ones((3,3))
x = torch.eye(5,5)     # identity matrix 형성 
x = torch.arange(start = 0, end = 5, step = 1)  # result = tensor([0, 1, 2, 3, 4])
x = torch.linspace(start = 0.1, end = 1, steps =10) # result tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000,1.0000])
x = torch.empty(size = (1,5)).normal_(mean=0, std=1)
x = torch.empty(size = (1,5)).uniform_(1,2)    # torch.rand랑 유사, 하지만 범위를 지정가능함 
x = torch.diag(torch.ones(3))   # create a diagnonal matirx of one 
print(x)
# %%
# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
tensor
print(tensor.bool())
print(tensor.short())   # int16 으로 change 
print(tensor.long())    # int64로 change, 사용많이함
print(tensor.half())    # float16으로 change, 이에 적합한 GPU series가 있는데 그게 아니면 사용할 일 없음 
print(tensor.float())   # float 32 
print(tensor.double())  # float 64 
# %%
# Numpy arrary -> tensor conversion and vice-versa 
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)   # np array -> torch 
np_array_back = tensor.numpy()        # torch -> np array
print(tensor)
print(np_array_back)





#  %%
# ================================================================== #
#                  Tensor Math & Comparison Operations               #
# ================================================================== #
