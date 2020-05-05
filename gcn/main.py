import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
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
from config import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makeData(X):
        smiles = [Chem.MolFromSmiles(mol) for mol in X]
        X  = [mol2graph.mol2vec(m) for m in smiles]
        return X

def trainModels(data_type):
    for k in range(num_folds):
        if data_type == 'folds':
            if k == 7:
                # removing the exceptional dataset case where roc,prc evaluation fail
                continue
            train_df,test_df,val_df = mydataloader.CVData(k,1)   # Get pandas dataframe for train,test and val 
            X_train, y_train = mydataloader.makeData(train_df) # Get features and labels
            X_test,  y_test  = mydataloader.makeData(test_df)  # Get features and labels

        
        elif data_type == 'psuedo':
            df = mydataloader.singleData()
            X_train, X_test, y_train, y_test = train_test_split(df['smiles'], np.array(df['activity']), test_size=0.2)
            X_train = makeData(X_train)
            X_test  = makeData(X_test)


        train_loader= mydataloader.makeDataLoader(X_train,y_train,True,batch_size) # Convert to PyTorch Dataloader
        test_loader = mydataloader.makeDataLoader(X_test,y_test,True,batch_size)   # Convert to PyTorch Dataloader

        model = Model(n_features,device,net_type)
        for epoch in tqdm(range(1, num_epochs)):
            train_loss = model.train(train_loader)
        

        y_pred, y_true, acc, y_pred_prob = testModel(model,test_loader)
        # roc_auc,prc_auc,recall,precision = getAUC(y_pred,y_true)
        y_pred_arr.append(y_pred)
        y_pred_prob_arr.append(y_pred_prob)
        y_true_arr.append(y_true)
        roc_auc,prc_auc,recall,precision = getAUC(y_pred_prob,y_true)

        prc_auc_arr.append(prc_auc)
        roc_auc_arr.append(roc_auc)
        acc_arr.append(acc)
        # prc_prob_auc_arr.append(prc_auc_prob)
        # roc_prob_auc_arr.append(roc_auc_prob)


        print(f'Fold:{k} \t ROC_AUC:{roc_auc:.3} \t PRC_AUC:{prc_auc:.3}\t Accuracy:{acc:.3}')
        # print(f'Accuracy:{acc} \t Precision:{precision} \t Recall:{recall}')
        torch.save(model.net.state_dict(),base_path+extra_path+'test_'+str(k)+'.pt')
        # print(f'Fold:{k} \t ROC_AUC_PROB:{roc_auc_prob} \t PRC_AUC_PROB:{prc_auc_prob}')
        print('#'*50)
        # return roc_auc_arr,prc_auc_arr,acc_arr



data_type = args.data_type
num_folds = args.num_folds
base_path = args.base_path
n_features = args.n_features
num_epochs= args.num_epochs
batch_size = args.batch_size
net_type = args.net_type 
extra_path = args.extra_path + str(data_type)+ '_' + str(net_type) + '/'

mydataloader = myDataLoader(num_folds=num_folds) 


print('Starting Folds')
print('#'*50)
print(extra_path)
if not os.path.exists(base_path):
    os.makedirs(base_path)
if not os.path.exists(base_path+extra_path):
    os.makedirs(base_path+extra_path)


y_true_arr = []
y_pred_arr = []
y_pred_prob_arr = []
Loss_arr = []
acc_arr = []
prc_auc_arr = []
roc_auc_arr = []



trainModels(data_type)
print('ROC:\t',np.mean(roc_auc_arr),'+-',np.std(roc_auc_arr))
print('PRC:\t',np.mean(prc_auc_arr),'+-',np.std(prc_auc_arr))
print('ACC:\t',np.mean(acc_arr),'+-',np.std(acc_arr))




# base_path = 'Model_weights/'
# extra_path = '64_dropout_psued/'
# if not os.path.exists(base_path):
#     os.makedirs(base_path)
# if not os.path.exists(base_path+extra_path):
#     os.makedirs(base_path+extra_path)
# y_true_arr = []
# y_pred_arr = []
# y_pred_prob_arr = []
# Loss_arr = []
# acc_arr = []
# prc_auc_arr = []
# roc_auc_arr = []


# df = pd.read_csv('Data/pseudomonas/test.csv')
# X_test = np.array(df['smiles'])
# X_test = makeData(X_test)
# test_loader = DataLoader(X_test, batch_size=1, shuffle=False, drop_last=True)

# model = Model(n_features,device)
# model.net.load_state_dict(torch.load(base_path+extra_path+'test_'+str(7)+'.pt'))
# model.net.eval()
# outputs = []
# y_pred = []
# for data in test_loader:
#     data = data.to(device)
#     output = model.net(data)
#     pred = output.max(dim=1)[1]
#     y_pred.extend(np.array(pred))
#     outputs.append(output.detach().numpy())
# ops = np.array(outputs).reshape((int(len(y_pred)),2))  
# y_pred_prob = getProbs(ops)[:,1]
