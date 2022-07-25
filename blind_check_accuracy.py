import numpy as np 
from config import * 
from src.dataloader import get_augmentation_loader, get_test_augmentation_loader
from src.device import get_device
import torch 
from src.accuracy import *
from src.seed import seed_everything
from src.model_resnet import * 
from src.blindtest_loader import get_blind_augmentation_dataset, get_blind_test_augmentation_dataset
from src.path import get_blind_pathes, get_blind_test_pathes

seed_everything(42)




def main(test_num:int, cell_type:str):
      device = get_device()
      criterion = nn.CrossEntropyLoss()
      cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path = get_blind_pathes(test_num, cell_type)

      train_dataset, valid_dataset, test_dataset = get_blind_augmentation_dataset(cd8_x_train_path, cd8_y_train_path, 
                                                                                  cd8_x_valid_path, cd8_y_valid_path, 
                                                                                  cd8_x_test_path, cd8_y_test_path)
      batch_size  = int(1)
      dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, batch_size)

      best_model = torch.load(f'model/blind_test/blindtest_{test_num}_{cell_type}_model_1.pt')

      loss_sum, acc, auroc, aupr, conf_matrix, labels , outputs = test_model(dataloaders, best_model, criterion,device)
    #  roc_scores, answers, preds = calculate_roc(dataloaders['test'], best_model, device)
    #  f1_score = f1_score(answers,preds,average='macro')

      print({f"AUROC on test set: {auroc*100:.2f}"})
      print({f"AUPR on test set: {aupr*100:.2f}"})
      print({f"ACC on test set: {acc*100:.2f}"})
    #  print({f"F1 Score on test set: {f1_score*100:.2f}"})
      print({f"loss on test set: {loss_sum}"})
      print(conf_matrix)



def main_test(test_num:int, cell_type:str):
      device = get_device()
      criterion = nn.CrossEntropyLoss()
      test_cd8_x_test_path, test_cd8_y_test_path = get_blind_test_pathes(test_num, cell_type)  

      blind_test_dataset = get_blind_test_augmentation_dataset(test_cd8_x_test_path, test_cd8_y_test_path)
      batch_size  = int(1)

      dataloaders = get_test_augmentation_loader(blind_test_dataset, batch_size)

      best_model = torch.load(f'model/blind_test/blindtest_{test_num}_{cell_type}_model_1.pt')

      loss_sum, acc, auroc, aupr, conf_matrix, labels , outputs = test_model(dataloaders, best_model, criterion,device)
    #  roc_scores, answers, preds = calculate_roc(dataloaders['test'], best_model, device)
    #  f1_score = f1_score(answers,preds,average='macro')

      print({f"AUROC on test set: {auroc*100:.2f}"})
      print({f"AUPR on test set: {aupr*100:.2f}"})
      print({f"ACC on test set: {acc*100:.2f}"})
    #  print({f"F1 Score on test set: {f1_score*100:.2f}"})
      print({f"loss on test set: {loss_sum}"})
      print(conf_matrix)


if __name__ == '__main__':

    print('test 7 - cd8')
   # main(2, 'cd8')

    print('---------------------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------')

    print('blind test 7 - cd8')
    main_test(2, 'cd8')



