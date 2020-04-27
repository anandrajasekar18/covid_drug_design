import numpy as np
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class Pred():
    def __init__(self,train_path,test_path,radius=2,nbits=2048,test_flag=1):
        self.radius = radius
        self.nbits  = nbits
        self.train_data = pd.read_csv(train_path)
        self.test_data  = pd.read_csv(test_path)
        self.train_data['fp'] = self.train_data.smiles.map(lambda x: self.fp_arr(x))
        self.test_data['fp'] = self.test_data.smiles.map(lambda x: self.fp_arr(x))
        self.X_train, self.y_train = np.concatenate(self.train_data.fp), self.train_data.activity
        self.X_test,  self.y_test  = np.concatenate(self.test_data.fp),  self.test_data.activity
        self.test_flag = test_flag # if test data has labels then test_flag=1 else test_flag=0


    def fp_arr(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.nbits)
        arr = np.empty((1,), dtype = np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape((1,-1))

    def randomForest(self,n):
        # parameter_candidates = [{'n_estimators':[100, 300]}]
        # self.gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid = parameter_candidates, n_jobs=-1, cv = 10, scoring = 'roc_auc', return_train_score=True)
        # self.gs.fit(self.X_train, self.y_train)
        # print (self.gs.best_score_, self.gs.cv_results_['mean_train_score'], self.gs.cv_results_['mean_test_score'])
        self.gs = RandomForestClassifier(n_estimators=n)
        self.gs.fit(self.X_train,self.y_train)
        y_pred = self.gs.predict(self.X_test)
        print('#'*50)
        print('RandomForest:',n)
        print('ROC_AUC:{} \t PRC_AUC:{}'.format(self.roc_auc(self.y_test,y_pred),self.prc_auc(self.y_test,y_pred)))


    def gradBoosting(self,n):
        # parameter_candidates = [{'n_estimators':[100, 300]}]
        # self.gs = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid = parameter_candidates, n_jobs=-1, cv = 10, scoring = 'roc_auc', return_train_score=True)
        # self.gs.fit(self.X_train, self.y_train)
        # # self.gs = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(self.X_train, self.y_train)
        # print (self.gs.best_score_, self.gs.cv_results_['mean_train_score'], self.gs.cv_results_['mean_test_score'])
        self.gs = GradientBoostingClassifier(n_estimators=n)
        self.gs.fit(self.X_train,self.y_train)
        y_pred = self.gs.predict(self.X_test)
        print('#'*50)
        print('GradBoost:',n)
        print('ROC_AUC:{} \t PRC_AUC:{}'.format(self.roc_auc(self.y_test,y_pred),self.prc_auc(self.y_test,y_pred)))

    def MLP(self):
        self.gs = MLPClassifier(hidden_layer_sizes=(int(self.nbits/4),int(self.nbits/16),int(self.nbits/64),))
        self.gs.fit(self.X_train,self.y_train)
        y_pred = self.gs.predict(self.X_test)
        print('#'*50)
        print('MLP:',n)
        print('ROC_AUC:{} \t PRC_AUC:{}'.format(self.roc_auc(self.y_test,y_pred),self.prc_auc(self.y_test,y_pred)))


    def test(self):
        y_pred = self.gs.predict(self.X_test)
        print('ROC_AUC:{} \t PRC_AUC:{}'.format(self.roc_auc(self.y_test,y_pred),self.prc_auc(self.y_test,y_pred)))

    def roc_auc(self,y_test,y_pred):
        return roc_auc_score(y_test,y_pred)  

    def prc_auc(self,y_test,y_pred):
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        return auc(recall,precision)

        


train_path = 'Data/splits/ecoli_scaffold/train.csv' 
test_path = 'Data/splits/ecoli_scaffold/test.csv' 
model = Pred(train_path,test_path)
# print('#'*50)
# print('RandomForest')
model.gradBoosting(100)
model.randomForest(100)
# model.test()
# print('#'*50)
# print('MLP')
model.MLP()
# model.test()
# print('#'*50)
# print('GradientBoosting')

# model.test()
# print('#'*50)