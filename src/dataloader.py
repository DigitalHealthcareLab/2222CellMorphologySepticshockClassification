import os
os.chdir("/home/kimjh/2022_gigascience/")

import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from config import * 
from torch.utils.data import DataLoader, Dataset
import cv2
import torchvision.transforms as transforms
from src.normalization import normalize_individual_image




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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(13370.405,143.7316)
])

test_transform = transforms.Compose([
    transforms.Normalize(13370.405,143.7316)
                ])

def get_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    
    
    train_df = torch.Tensor(np.float64(np.load(x_train_path))).unsqueeze(1)
    train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)

    valid_df = torch.Tensor(np.float64(np.load(x_valid_path))).unsqueeze(1)
    valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

    test_df = torch.Tensor(np.float64(np.load(x_test_path))).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)

    return train_dataset, valid_dataset, test_dataset


def get_augmentation_loader(train, valid, test, batch_size):
    loader = {}

    loader['train'] = DataLoader(train, batch_size = batch_size, shuffle = True,  pin_memory = True, num_workers=8)
    loader['valid'] = DataLoader(valid, batch_size = batch_size, shuffle = False,  pin_memory = True, num_workers=8)
    loader['test'] = DataLoader(test, batch_size = batch_size,  shuffle = False, pin_memory = True, num_workers=8)
    
    return loader


def get_test_augmentation_dataset(x_test_path, y_test_path):
    test_df = torch.Tensor(normalize_individual_image(np.load(x_test_path).astype('float64'))).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)
    return test_dataset



def get_augmentation_blind_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    
    train_df = torch.Tensor(normalize_individual_image(np.load(x_train_path))).unsqueeze(1)
    train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)

    valid_df = torch.Tensor(normalize_individual_image(np.load(x_valid_path))).unsqueeze(1)
    valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

    test_df = torch.Tensor(normalize_individual_image(np.load(x_test_path))).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)

    return train_dataset, valid_dataset, test_dataset