from gcn_model import *
from molecular_graph import mol_graph
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

descriptors = list(Chem.rdMolDescriptors.Properties().GetAvailableProperties())
cls = Chem.rdMolDescriptors.Properties(descriptors)


def get_descriptors(smiles):
	mol = Chem.MolFromSmiles(smiles)
	fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
	arr = np.empty((1,), dtype=np.int8)
	AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
	moldes = np.array(cls.ComputeProperties(mol)).reshape((1, -1))
	return np.concatenate([arr.reshape((1,-1)), moldes], axis = -1)


def get_max_atom(train_data):
	smiles_ls = train_data.smiles.values
	atoms_ls = []
	for smiles in smiles_ls:
		mol = Chem.MolFromSmiles(smiles)
		atoms_ls.append(mol.GetNumAtoms())
	return max(atoms_ls)


def preprocess_dataset(data):
	data_A, data_F = [], []
	data_desc = []

	max_atoms = get_max_atom(train_data)
	graph_gen = mol_graph(max_atoms)

	for smiles in data.smiles:
		A, F = graph_gen.get_graph(smiles)
		D = get_descriptors(smiles)
		data_A += [A.copy()[0]]
		data_F += [F.copy()[0]]
		data_desc += [D.copy()[0]]
	return np.array(data_A).sum(axis=3), np.array(data_F), np.array(data_desc), data.activity


if __name__ == '__main__':
	train_data = pd.read_csv('train.csv')
	test_data = pd.read_csv('test.csv')
	hidden_layers=[30, 200, 500]
	val_ratio=0.1
	batch_size=64
	lr=1e-2
	early_stop=5

	print('Data imported...', train_data.shape, test_data.shape)

	train_A, train_F, train_D, train_Y = preprocess_dataset(train_data)
	print('Preprocessed...',train_A.shape, train_F.shape, train_D.shape, train_Y.shape)

	g, features, descriptors, labels = gcn_data_ready(train_A, train_F, train_D, train_Y)

	print('\nData ready for training...')
	model = train_gcn(g, features, descriptors, labels, hidden_layers, val_ratio, batch_size, lr, early_stop)

	print('\nModel trained')

	test_A, test_F, test_D, test_Y = preprocess_dataset(test_data)
	test_g, test_features, test_descriptors, test_labels = gcn_data_ready(test_A, test_F, test_D, test_Y)
	model = load_model(model)

	print('\nData ready for testing...', test_A.shape, test_F.shape, test_D.shape, test_Y.shape)
	test_acc, test_auc, test_loss = evaluate(model, test_g, test_features, test_descriptors, test_labels)

	print("Test Loss ", np.round(test_loss, 4), "| Test Acc ", np.round(test_acc, 4), "| Test AUC ", np.round(test_auc, 4))
