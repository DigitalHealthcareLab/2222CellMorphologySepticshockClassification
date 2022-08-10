import os 
from config import * 
from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model_resnet import *
from src.model_densenet import * 
from src.optimizer import *
from src.dataloader import get_augmentation_loader
import torchvision.transforms as transforms 
from src.earlystopping import EarlyStopping
from src.train import *
from src.path import get_blind_pathes
from src.blindtest_loader import get_blind_augmentation_dataset

seed_everything(42)

 
def main(remove_patient:int, celltype:str):

    logger = set_logger()
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    print('get model')
    model = create_classification_ANN(device)
    model = nn.DataParallel(model)

    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    print('get dataset')
    x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path = get_blind_pathes(remove_patient, celltype)
    train_dataset, valid_dataset, test_dataset = get_blind_augmentation_dataset(x_train_path, 
                                                                                y_train_path, 
                                                                                x_valid_path, 
                                                                                y_valid_path, 
                                                                                x_test_path, 
                                                                                y_test_path)
    dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, BATCH_SIZE)


    logger.info("Start Training") 
    logger.info(f"learning_rate: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY},  epoch: {NUM_EPOCH}, batch_size: {BATCH_SIZE},dropout: {DROPOUT}")
    
    early_stopping = EarlyStopping(
        metric=EARLYSTOPPING_METRIC,
        mode=EARLYSTOPPING_MODE,     
        patience=PATIENCE,           
        path=MODEL_PATH,             
        verbose=False,
        )

    best_model, train_loss_history, val_loss_history =  train_model_v2(model, NUM_EPOCH, dataloaders, criterion, optimizer, device, scheduler, early_stopping)

    torch.save(best_model, f'model/blind_test/blindtest_{remove_patient}_{celltype}_model_1.pt') 
    LOSS_PATH = f'model/plot/blindtest_{remove_patient}_{celltype}'
    save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)
    plt.clf()

if __name__ == '__main__':
    
    main(1, 'cd8')


