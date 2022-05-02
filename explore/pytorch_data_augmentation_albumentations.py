# %%
import torch 
import torch.nn as nn #
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os 
import albumentations as A
from tqdm import tqdm
 
# %%
# After build up model - model 과정은 생략 
# Load data에 적용 
# Numpy array 데이터에 대해 각각에 대해서 4배로 양을 증폭 

train_x = np.load('data/lungT_sepsis/processed/lung_tcell_x_train.npy')
images_list = []
for p1, p2 in [[0,0], [1,0], [0,1], [1,1]]:
    transform = A.Compose([
        A.VerticalFlip(p = p1),
        A.HorizontalFlip(p = p2)
    ])
    augmentation = transform(image=train_x)
    augmented_img = augmentations['image']
    images_list.append(augmented_img)

train_x = np.concatenate(images_list, axis=0) # 현재 데이터를 0번째 리스트에 맞춰 concat
train_y = np.load('data/lungT_sepsis/processed/lung_tcell_y_train.npy')
train_y = np.concat([train_y]*4) # train_x의 shape을 확인해보면 len가 4배가 된것을 확인가능 

train_x = torch.Tensor(train_x).unsqueeze(1)
train_y = torch.LongTensor(train_y)
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
