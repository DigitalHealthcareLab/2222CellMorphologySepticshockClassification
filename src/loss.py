# %%
import torch.nn as nn
import torch 
from sklearn.utils import class_weight
import numpy as np 

def getClassWeight(y_data):
    weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes = np.unique(y_data),
                                                y = y_data)
    return weights

y_test = np.load('data/survived_died/cd8_y_train.npy')
weights = getClassWeight(y_test)

def get_loss():
    return nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).cuda())


