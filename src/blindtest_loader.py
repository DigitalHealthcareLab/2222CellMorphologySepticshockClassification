import os
os.chdir("/home/kimjh/2022_gigascience/")
import torchvision.transforms as transforms
import torch 
import numpy as np 
from src.normalize import normalize_individual_image
from src.dataloader import CustomDataset


train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

test_transform = transforms.Compose([
                ])


def get_blind_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    
    train_df = torch.Tensor(normalize_individual_image(np.load(x_train_path))).unsqueeze(1)
    train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)

    valid_df = torch.Tensor(normalize_individual_image(np.load(x_valid_path))).unsqueeze(1)
    valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

    test_df = torch.Tensor(normalize_individual_image(np.load(x_test_path))).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)

    return train_dataset, valid_dataset, test_dataset

def get_blind_test_augmentation_dataset(x_test_path, y_test_path):
    test_df = torch.Tensor(normalize_individual_image(np.load(x_test_path).astype('float64'))).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)
    return test_dataset