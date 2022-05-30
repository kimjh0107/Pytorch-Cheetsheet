from config import * 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms 
import cv2
import numpy as np

# np.array 형태로 구성되어 있는 데이터에서 바로 적용 -> 데이터 양이 늘어나는 것은 확인했지만 3D img였기에 적용은 못함 .. 

class CustomDataset(Dataset):
    def __init__(self, images : np.array, 
                        label_list : np.array, 
                        train_mode=True, 
                        transforms=None): #필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        # self.img_path_list = img_path_list
        self.images = images
        self.label_list = label_list

    def __getitem__(self, index): #index번째 data를 return
        image = self.images[index]
        # Get image data
        # image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image
    
    def __len__(self): #길이 return
        return len(self.images)

train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
   # transforms.RandomPerspective(distortion_scale=.15,p=.15,interpolation=transforms.InterpolationMode.NEAREST),
])

test_transform = transforms.Compose([
                   # transforms.ToPILImage(),
                    transforms.ToTensor(),
                    ])


def get_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    train_dataset = CustomDataset(np.load(x_train_path), np.load(y_train_path), train_mode=True, transforms=train_transform)

    for _ in range(3):
        train_dataset += CustomDataset(np.load(x_train_path), np.load(y_train_path), train_mode=True, transforms=train_transform)
    
    valid_dataset = CustomDataset(np.load(x_valid_path), np.load(y_valid_path), train_mode=False, transforms=test_transform)
    test_dataset = CustomDataset(np.load(x_test_path) , np.load(y_test_path), train_mode=False, transforms=test_transform)
    
    return train_dataset, valid_dataset, test_dataset




def get_augmentation_loader(train, valid, test, batch_size):
    loader = {}

    loader['train'] = DataLoader(train, batch_size = batch_size, shuffle = True, pin_memory = True)
    loader['valid'] = DataLoader(valid, batch_size = batch_size, shuffle = False ,pin_memory = True)
    loader['test'] = DataLoader(test, batch_size = batch_size, shuffle = False ,pin_memory = True)
    return loader