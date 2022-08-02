import torch 
from sklearn.metrics import roc_auc_score
from src.dataloader import *
from src.path import get_blind_pathes
import torch.nn.functional as F



# calculate roc 
def calculate_roc(loader , model, device):
    num_correct = 0
    num_samples = 0
    answers = []
    preds = []
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0) 

            answers.extend(y.detach().cpu().numpy())
            preds.extend(predictions.detach().cpu().numpy())

       
    
    return roc_auc_score(answers, preds) , answers, preds

def get_roc_loader(remove_patient:int, celltype:str):
    x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path = get_blind_pathes(remove_patient,celltype)
    train_dataset, valid_dataset, test_dataset = get_augmentation_dataset(x_train_path, y_train_path, 
                                                                          x_valid_path, y_valid_path, 
                                                                          x_test_path, y_test_path)
    batch_size = 1
    dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, batch_size)
    return dataloaders 