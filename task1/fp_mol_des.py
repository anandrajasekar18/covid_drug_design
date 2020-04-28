import sys 
sys.path.append('/usr/local/lib/python3.7/site-packages/')
import pandas as pd 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, make_scorer
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE

descriptors = list(Chem.rdMolDescriptors.Properties().GetAvailableProperties())

cls = Chem.rdMolDescriptors.Properties(descriptors)



##Hyper_params
radius = 2
nbits = 1024


def fp_arr(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nbits)
    arr = np.empty((1,), dtype = np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    moldes = np.array(cls.ComputeProperties(mol)).reshape((1,-1))

    return np.concatenate([arr.reshape((1,-1)), moldes], axis = -1)



def auprc_scorer(y_true, y_pred_proba):
  pre, rec, thre = precision_recall_curve(y_true, y_pred_proba)
  return auc(rec, pre)


auprc = make_scorer(auprc_scorer, needs_proba= True)

train_data = pd.read_csv('pseudomonas/train.csv')
test_data = pd.read_csv('pseudomonas/test.csv')

# X_train = np.load('pseudomonas/X_train.npy')    
# X_test = np.load('pseudomonas/X_test.npy')
# y_train = train_data.activity
train_data['fp'] = train_data.smiles.map(lambda x: fp_arr(x))
test_data['fp'] = test_data.smiles.map(lambda x: fp_arr(x))

X_train, y_train = np.concatenate(train_data.fp), train_data.activity
X_test = np.concatenate(test_data.fp)

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]# Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,object of too small depth for desired array gridsearch cv
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rsv = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring = 'f1')
# rsv.fit(X_train, y_train)

# print (rsv.best_score_, rsv.cv_results_['mean_test_score'], rsv.best_params_)
# 1/y_train.mean()
# !pip install imblearn sklearn
# y_train.shape
np.concatenate(train_data.fp).shape

# parameter_candidates = [{'hidden_layer_sizes':[(100, )], 'activation':['relu']}]
parameter_candidates = {'rf__n_estimators': [200, 300,400,500]}
# parameter_candidates = {'rf__n_estimators': [20]}('over', SMOTE())
# parameter_candidates = {'hidden_layer_sizes': [(100,), (500,), (1200,), (1000,)]}
# parameter_candidates = {
#     'bootstrap': [False],
#     'max_depth': [None],
#     'min_samples_split': [1,2,3],
#     'n_estimators': [200, 300,400,500]
# }
from imblearn.under_sampling import RandomUnderSampler
pipeline = Pipeline([('over', SMOTE(sampling_strategy=0.1, k_neighbors = 7)), ('under', RandomUnderSampler(sampling_strategy = 0.5)) ,('rf', RandomForestClassifier())])
gs = GridSearchCV(estimator=pipeline, param_grid = parameter_candidates, n_jobs=-1, cv = 10, scoring = auprc, return_train_score=True)
gs.fit(X_train, y_train)
test_data['activity'] = gs.predict(X_test)

test_data.to_csv('pseudomonas/test_pred.csv', index= False, columns =['smiles', 'activity'])


print (gs.best_score_, gs.cv_results_['mean_train_score'], gs.cv_results_['mean_test_score'])

gs = GridSearchCV(estimator=pipeline, param_grid = parameter_candidates, n_jobs=-1, cv = 10, scoring = 'average_precision', return_train_score=True)
gs.fit(X_train, y_train)
test_data['activity'] = gs.predict(X_test)

test_data.to_csv('pseudomonas/test_pred.csv', index= False, columns =['smiles', 'activity'])




print (gs.best_score_, gs.cv_results_['mean_train_score'], gs.cv_results_['mean_test_score'])

