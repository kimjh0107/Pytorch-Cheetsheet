# %% 
import torch 
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import os
import pandas as pd
from customDataset import CatsAndDogsDataset
os.chdir("/Users/kimjh/Documents/pytorch_cheetsheat/")
# %%
# Load Data 
#my_transforms = transforms.ToTensor() # numpy -> tensor 
my_transforms = transform.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5), # flip horizontally
    transforms.ColorJitter(brightness=0.5), # randomly change the brightness, contrast and saturation of an image
    transforms.Resize(256,256),
    transforms.RandomCrop(224,224),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.05), 
    transforms.RandomGrayScale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.0, 0.0,0.0], std = [1.0, 1.0, 1.0])  # note: 실제로 진행할때는 안에 값들을 저런식으로 하는 것은 아님 
    # tensor 로 변형을 해 준 이후 적용을 하면 traning 결과 향상시키는데 도움이 된다고 함 
    # transform.Normalize -> 이제 각각의 채널들에 대해서 따로따로 진행을 해주는 것 한번에 묶어서 진행을 하는 것이 아닌 
])

train_dataset = CatsAndDogsDataset(csv_file= 'data/customdataset/cat_dogs.csv', 
                             root_dir='data/customdataset/cats_dogs_resized', 
                             transform=my_transforms)

# dataloader안에 이제 dataset을 적용하고 싶을 경우, 위에 train_dataset 부분에 transform에 내가 원하는 부분만 추가해서 진행을 해주면 됨 -> 적용해보기 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# %%
img_num = 0
for img, label in dataset:
    save_image(img, 'img'+ str(img_num)+ 'png')
    img_num += 1