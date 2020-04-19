import pandas as pd 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


##Hyper_params
radius = 2
nbits = 2048


def fp_arr(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nbits)
    arr = np.empty((1,), dtype = np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape((1,-1))





train_data = pd.read_csv('pseudomonas/train.csv')
test_data = pd.read_csv('pseudomonas/test.csv')



train_data['fp'] = train_data.smiles.map(lambda x: fp_arr(x))
test_data['fp'] = test_data.smiles.map(lambda x: fp_arr(x))

X_train, y_train = np.concatenate(train_data.fp), train_data.activity
X_test = np.concatenate(test_data.fp)

# parameter_candidates = [{'hidden_layer_sizes':[(100, )], 'activation':['relu']}]
parameter_candidates = [{'n_estimators':[300, 500, 1200]}]
gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid = parameter_candidates, n_jobs=-1, cv = 10, scoring = 'roc_auc', return_train_score=True)
gs.fit(X_train, y_train)





print (gs.best_score_, gs.cv_results_['mean_train_score'], gs.cv_results_['mean_test_score'])



