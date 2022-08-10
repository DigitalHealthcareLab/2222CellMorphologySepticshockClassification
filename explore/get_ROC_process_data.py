import torch 
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np 
from src.device import get_device
from sklearn.metrics import roc_curve
from config import *
from src.blindtest_loader import get_blind_augmentation_dataset
from src.path import get_blind_pathes
from src.dataloader import get_augmentation_loader
from src.accuracy import *


def get_labels_and_outputs(test_num:int, cell_type:str):
      device = get_device()
      criterion = nn.CrossEntropyLoss()
      cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path = get_blind_pathes(test_num, cell_type)
      train_dataset, valid_dataset, test_dataset = get_blind_augmentation_dataset(cd8_x_train_path, cd8_y_train_path, 
                                                                                  cd8_x_valid_path, cd8_y_valid_path, 
                                                                                  cd8_x_test_path, cd8_y_test_path)


      batch_size  = int(1)
      dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, batch_size)
      best_model = torch.load(f'model/blind_test/blindtest_{test_num}_{cell_type}_model_1.pt')
      labels, output_probs, outputs  = roc_test_model(dataloaders, best_model, criterion,device)

      return labels, output_probs, outputs

def get_roc_sources(labels, output):
    fpr, tpr, _ = roc_curve(labels, output)
    roc_auc = roc_auc_score(labels, output)
    return fpr, tpr, roc_auc

def calcuate_roc_sources(test_num, cell_type):
    labels, output_probs, outputs = get_labels_and_outputs(test_num, cell_type)
    fpr,tpr, roc_auc  = get_roc_sources(labels, output_probs)
    return fpr,tpr, roc_auc  


def main():

    fpr = []
    tpr = []
    roc_auc = []

    for i in range(8):
        print(f'get {i+1} test result')
        fpr_i, tpr_i, roc_auc_i = calcuate_roc_sources(i+1, 'cd4')
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(roc_auc_i)
    
    np_fpr = np.array(fpr)
    np_tpr = np.array(tpr)
    np_roc_auc = np.array(roc_auc)

    print(f'get save test result')
    np.save('data/roc_data/cd4/np_fpr.npy', np_fpr)
    np.save('data/roc_data/cd4/np_tpr.npy', np_tpr)
    np.save('data/roc_data/cd4/np_roc_auc.npy', np_roc_auc)


if __name__ == '__main__':
    main()

