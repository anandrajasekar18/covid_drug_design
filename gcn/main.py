import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

import mol2graph
from model import Model
from utils import myDataLoader,getProbs,getAUC,testModel

num_folds = 10
n_features = 75

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mydataloader = myDataLoader(num_folds=num_folds) 

prc_auc_arr = []
roc_auc_arr = []
prc_prob_auc_arr = []
roc_prob_auc_arr = []

Loss_arr = []
print('Starting Folds')
print('#'*50)
# k=0
num_epochs = 51
for k in range(num_folds):
# for k in range(8,10):
    train_df,test_df,val_df = mydataloader.CVData(k)   # Get pandas dataframe for train,test and val 
    train_X, train_y = mydataloader.makeData(train_df) # Get features and labels
    test_X,  test_y  = mydataloader.makeData(test_df)  # Get features and labels
    val_X,   val_y   = mydataloader.makeData(val_df)   # Get features and labels

    train_loader= mydataloader.makeDataLoader(train_X,train_y) # Convert to PyTorch Dataloader
    test_loader = mydataloader.makeDataLoader(test_X,test_y)   # Convert to PyTorch Dataloader
    val_loader  = mydataloader.makeDataLoader(val_X,val_y)     # Convert to PyTorch Dataloader

    model = Model(n_features,device)
    hist = {"loss":[], "acc":[], "test_acc":[]}
    for epoch in tqdm(range(1, num_epochs)):
        train_loss = model.train(train_loader)
        train_acc = model.test(train_loader)
        test_acc = model.test(val_loader)
        hist["loss"].append(train_loss)
        hist["acc"].append(train_acc)
        hist["test_acc"].append(test_acc)
        print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')
    Loss_arr.append(hist)
        
    y_pred, y_true, y_pred_prob = testModel(model,test_loader)
    roc_auc,prc_auc = getAUC(y_pred,y_true)
    roc_auc_prob,prc_auc_prob = getAUC(y_pred_prob,y_true)

    prc_auc_arr.append(prc_auc)
    roc_auc_arr.append(roc_auc)
    prc_prob_auc_arr.append(prc_auc_prob)
    roc_prob_auc_arr.append(roc_auc_prob)


    print(f'Fold:{k} \t ROC_AUC:{roc_auc} \t PRC_AUC:{prc_auc}')
    print(f'Fold:{k} \t ROC_AUC_PROB:{roc_auc_prob} \t PRC_AUC_PROB:{prc_auc_prob}')
    print('#'*50)
    # ax = plt.subplot(1,1,1)
    # ax.plot([e for e in range(1,101)], hist["loss"], label="train_loss")
    # ax.plot([e for e in range(1,101)], hist["acc"], label="train_acc")
    # ax.plot([e for e in range(1,101)], hist["test_acc"], label="test_acc")
    # plt.xlabel("epoch")
    # ax.legend()