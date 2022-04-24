# %% 
# ================================================================== #
#                        Initializing Tensor                         #
# ================================================================== #
from tkinter import E
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
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# 더하기 -> 마지막 방법이 제일 간단 명료
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z = x + y   # 가장 간단한 방법 
# %%
# 빼기
z = x - y

# Division 
z = torch.true_divide(x,y)
z
# %%
# inplace operations 
t = torch.zeros(3)
t.add_(x)  # _ under score가 있는 경우 inplace가 된다는 것으로 기억해두기 
# %%
# Exponentitation
z = x.pow(2)  # 안에 tensor 값들 제곱 
z = x ** 2    # pow와 동일한 방법 
# %%
# Simple comparison 
z = x > 0 
print(z)   # result: tensor([True, True, True])
# %%
# Matrix Multiplication 
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)  # 2x3
x3 = x1.mm(x2)        # 동일한 곱셈 방법 mm 
# %%
# Matrix exponentitation 
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)  # 거듭제곱 해주는 함수 = matrix_power 
# %%
# element wise multiplication 
z = x * y 

# dot product 
z = torch.dot(x,y)
z   # result: tensor(46)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))  # 3dimensions of tensor -> batch matrix multiplication을 적용 (차원이 하나더 있으니)
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
print(out_bmm.shape)      # batch의 dimensions을 맞춰야지 계산이 가능함 
# %%
# Example of Broadcasting - 차원이 일치하지 않아도 계산이 가능 
# broadcastable하고 같은 요소 수를 가진 경우 broadcasting의 도입은 backwards incompatible 변화를 야기가능
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2 
z = x1 ** x2
# %%
# Other useful tensor operators 
x = torch.tensor([1,2,3])
sum_x = torch.sum(x, dim = 0)  # tensor(6)
values, indices = torch.max(x, dim = 0)  # 순서대로 안의 list처럼 이제 max값을 순서대로 출력 가능 
valeus = x.max(dim = 0)  # 동일하게 max값을 확인 가능 
values, indices = torch.min(x, dim = 0)  # values = tensor(1), indices = tensor(2)
abs_x = torch.abs(x)  # take absolute value element wise for each in X 
z = torch.argmax(x, dim = 0)  # index 값중에 이제 max인것을 나타내주는 것, result : tensor(2)
z = torch.argmin(x, dim = 0)
mean_x = torch.mean(x.float(), dim = 0)  # pytorch requires ot to be a float!!
z = torch.eq(x, y)  # 2 tensor사이에 이제 일치하는 값을 출력해주는 함수, result: tensor([False, False, False])
z = torch.clamp(x, min = 0)  # going to check all elements of X that are less than zero, and set to zero
z = torch.clamp(x, max = 10) # if any value is greater than 10, it's going to set it to 10






# %%
# ================================================================== #
#                  Tensor Indexing               #
# ================================================================== #
batch_size = 10
features = 25
x = torch.rand((batch_size, features))
x.shape    # torch.Size([10, 25])
print(x[0].shape)  # x[0, :]  앞의 index 값 torch.Size([25])
print(x[:,0].shape)  # torch.Size([10])
print(x[2, 0:10])  # 해당 원하는 값만 출력 

x[0,0] = 200
print(x[0,:])  # x[0:0] 값 200으로 변경 
# %%
# Fancy indexing 
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])  # 내가 원하는 list에 속하는 values만 선택해서 뽑아오기 가능 

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])

print(x)
"""tensor([[0.8487, 0.6415, 0.2435, 0.9468, 0.2696],
        [0.4921, 0.1021, 0.6248, 0.5871, 0.2024],
        [0.3376, 0.7142, 0.5681, 0.5263, 0.3916]])"""
print(x[rows, cols])  #tensor([0.2024, 0.8487])
# -> rows, cols 를 조합해서 첫번째 값이 (1,4) 값, (0,0)값을 가져오는 것 (엄청 햇갈림)
# %%
# More advanced indexing 
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # 더 큰 차원의 tensor에도 적용할 경우는 어떻게 할지 
print(x[x.remainder(2) == 0])  # 나누었을 때 0인 값들을 출력 
# %%
# Useful operations 
x = torch.arange(10)
print(x)
print(torch.where(x > 5, x, x*2))  # where 함수를 통해서 원하는 값 변경을 좀더 세분화 가능 
print(torch.tensor([0,0,1,2,2,3,4]).unique())  # 공통된 값들 중 unique한 값들만 찾기 
print(x.ndimension())    # dimension 확인! , ex) 5x5x 의 경우 3차원임을 출력해줌 
print(x.numel())  # count elements number -> 복잡한 경우에 사용 






# %%
# ================================================================== #
#                  Tensor Reshaping                                  #
# ================================================================== #
x = torch.arange(9)

x_3x3 = x.view(3,3)
print(x_3x3)

x_3x3 = x.reshape(3,3)  # view, reshape 유사기능 -> 하지만 view는 memory block이 contiguous한 경우 사용가능하다고함. -> reshape사용!
print(x_3x3)

y = x_3x3.t()  # transpose 기능 

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1, x2), dim = 0).shape)  # torch.Size([4, 5])
print(torch.cat((x1, x2), dim = 1).shape)  # torch.Size([2, 10])
print(torch.cat((x1,x2)).shape)            # torch.Size([4,5]) -> default dim = 0 잡는 것 확인 가능 

z = x1.view(-1)
print(z.shape)  # torch.Size([10]) , size([2,5]) -> size([10]) , -1 통해 하나로 나열 

# %%
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) # 앞의 dimension은 일정하게 유지하고, 나머지에 대해서만 지정해서 가능 
z.shape  # torch.Size([64, 10])
# %%
z = x.permute(0, 2, 1)  # permute = 내가 원하는 차원을 이제 변환하는 방법 
z.shape  # torch.Size([64, 5, 2])
# %%
x = torch.arange(10)        # Unsqueeze 기능을 통해서 원하는 차원을 추가 
print(x.shape)              # torch.Size([10])
print(x.unsqueeze(0).shape) # torch.Size([1,10])
print(x.unsqueeze(1).shape) # torch.Size([10,1])

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)              # torch.Size([1, 1, 10])

z = x.squeeze(1)            # squeeze을 통해서 이제 차원을 확장 
print(z.shape)              # torch.Size([1, 10])
z = x.squeeze(2)
print(z.shape)              # torch.Size([1, 1, 10])
