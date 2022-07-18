import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import (DataLoader,)  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# load pretrain model 
model = torchvision.models.vgg16(pretrained=True)

# 만약 앞에 layer들은 학습을 하지 않고 뒤에 새롭게 추가한 부분에 대해서만 학습을 진행해야 되는 경우는 requires_grad 를 false로 지정해주면 가능 
for param in model.parameters():
    param.requires_grad = False 

model.avgpool = Identity() # 이렇게 설정을 해줌으로 우리가 새롭게 설정 가능 
model.classifier = nn.Sequential( # classifier 부분에 추가적으로 한층더 만들거나 classes만 변경해주는 식으로 진행 
    nn.Linear(512,100),
    nn.ReLU(),
    nn.Linear(100, num_classes)
)

# 추가적으로 특정 부분, 예를 들어 classifier 부분에 대해서 전부 변경을 하고 싶은 경우는 아래와 같이 진행 가능 
# for i in range(1,7):
#     model.classifier[i] = Identity()

