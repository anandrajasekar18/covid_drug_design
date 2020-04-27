import numpy as np
import pandas as pd
import mol2graph
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getProbs(out):
    num = np.exp(out)
    den = np.sum(num,1)
    den = np.stack((den,) * 2, axis=-1) 
    probs = num/den 
    return probs 

def getAUC(y_pred,y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = auc(recall,precision)
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc,prc_auc

def testModel(model,loader):
    y_pred = []
    y_true = []
    outputs = []
    model.net.eval()
    for data in loader:
        data = data.to(device)
        output = model.net(data)
        pred = output.max(dim=1)[1]
        y_pred.extend(np.array(pred))
        y_true.extend(np.array(data.y))
        outputs.append(output.detach().numpy())
    ops = np.array(outputs).reshape((int(64*len(y_pred)//64),2))  
    y_pred_prob = getProbs(ops)[:,1]
    return y_pred,y_true,y_pred_prob

class myDataLoader():
    def __init__(self,path=None, num_folds=10):
        self.path = path
        self.num_folds = num_folds

    def singleData(self):
        """
        Give pandas dataframe
        """
        # if self.path is not None:
        path = 'Data/pseudomonas/train.csv'
        df = pd.read_csv(path)
        return df

    def CVDataMega(self,k):
        """
        Get pandas dataframe for train,test and val combined
        """
        path = 'Data/pseudomonas/train_cv/fold_'
        column_names = ['smiles','activity']
        train_df = pd.DataFrame(columns = column_names)
        test_df  = pd.DataFrame(columns = column_names)
        val_df   = pd.DataFrame(columns = column_names)

        for i in range(self.num_folds):
            if i!=k:
                print(i)
                temp_train_df = pd.read_csv(path+str(i)+'/train.csv')
                temp_test_df  = pd.read_csv(path+str(i)+'/test.csv')
                temp_val_df   = pd.read_csv(path+str(i)+'/dev.csv')
                train_df      = train_df.append(temp_train_df)
                test_df       = test_df.append(temp_test_df)
                val_df        = val_df.append(temp_val_df)

        return train_df,test_df,val_df
    
    def CVData(self,k,folds=1,path = 'Data/pseudomonas/train_cv/fold_'):
        """
        Get pandas dataframe for train,test and val 
        """
        if folds == 1:
            path = path + str(k)
        train_df = pd.read_csv(path+'/train.csv')
        test_df  = pd.read_csv(path+'/test.csv')
        val_df   = pd.read_csv(path+'/dev.csv')

        return train_df,test_df,val_df
    

    def makeData(self,df):
        """
        Get features and labels
        """
        mols = np.array(df['smiles'])
        y = np.array(df['activity'])
        smiles = [Chem.MolFromSmiles(mol) for mol in mols]
        X  = [mol2graph.mol2vec(m) for m in smiles]
        return X,y
    
    def MorganFeats(self,df):
        mols = np.array(df['smiles'])
        y = np.array(df['activity'])
        smiles = [Chem.MolFromSmiles(mol) for mol in mols]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in smiles]
        X = []
        for fp in fps:
            arr = np.zeros(0,)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X.append(arr)
        return np.asarray(X),y


    def makeDataLoader(self,X,y):
        """
        Convert to PyTorch Dataloader
        """
        for i, data in enumerate(X):
            data.y = torch.tensor([y[i]],dtype=torch.long)

        return DataLoader(X, batch_size=64, shuffle=True, drop_last=True)



                