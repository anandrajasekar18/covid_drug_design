from fold_exp import *
import pandas as pd
import sys
import pickle



def save_folds(path):
	# train_data = pd.read_csv('task1_data/pseudomonas/train.csv')
	# test_data = pd.read_csv('task1_data/pseudomonas/test.csv')
	cluster = None
	i = 0
	datasets = []
	for i in range(10):
		train_file = path + "/pseudomonas/train_cv/fold_" + str(i) + "/train.csv"
		test_file = path + "/pseudomonas/train_cv/fold_" + str(i) + "/test.csv"
		train_data = pd.read_csv(train_file)
		test_data = pd.read_csv(test_file)

		print('Data imported...', train_data.shape, test_data.shape)

		train_A, train_F, train_D, train_Y, clus = preprocess_dataset(train_data, cluster)
		if i == 0:
			cluster = clus
		test_A, test_F, test_D, test_Y, _ = preprocess_dataset(test_data, cluster)
		train_D, test_D = norm(train_D, test_D)
		g, features, descriptors, labels = gcn_data_ready(train_A, train_F, train_D, train_Y)
		test_g, test_features, test_descriptors, test_labels = gcn_data_ready(test_A, test_F, test_D, test_Y)
		print('Preprocessed...', train_A.shape, train_F.shape, train_D.shape, train_Y.shape, test_features.shape)
		datasets += [[g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels]]

	p_dump(datasets, 'pseudomonas_folds')


def p_dump(data, name='pseudomonas_data'):
	file = open(name, 'wb')
	pickle.dump(data, file)
	print(name)
	file.close()


def save_data(path):
	from molecular_graph import preprocess_dataset

	# train_data = pd.read_csv(path + '/train.csv').sample(frac=0.3)
	# test_data = pd.read_csv(path + '/test.csv')
	# cluster = None
	# i = 0
	datasets = []
	# print('Data imported...', train_data.shape, test_data.shape)
	#
	# train_A, train_F, train_D, train_Y, clus = preprocess_dataset(train_data, cluster)
	# if i == 0:
	# 	cluster = clus
	# print('Train Preprocessed...',train_A.shape, train_F.shape, train_D.shape, train_Y.shape)
	# p_dump([train_A, train_F, train_D, train_Y, clus], path+'/sars_data_tr')
	# test_A, test_F, test_D, test_Y, _ = preprocess_dataset(test_data, cluster)
	# print('Test Preprocessed...',test_A.shape, test_F.shape, test_D.shape, test_Y.shape)
	# p_dump([test_A, test_F, test_D, test_Y], path+'/sars_data_te')

	file = open(path+"/sars_data_tr", 'rb')
	train_A, train_F, train_D, train_Y, clus = pickle.load(file)
	file.close()
	file = open(path+"/sars_data_te", 'rb')
	test_A, test_F, test_D, test_Y = pickle.load(file)
	file.close()
	train_D, test_D = norm(train_D, test_D)
	g, features, descriptors, labels = gcn_data_ready(train_A, train_F, train_D, train_Y)
	p_dump([g, features, descriptors, labels], path+'/sars_data_tr2')
	test_g, test_features, test_descriptors, test_labels = gcn_data_ready(test_A, test_F, test_D, test_Y)
	p_dump([test_g, test_features, test_descriptors, test_labels], path+'/sars_data_tr2')
	print('Ready...', train_A.shape, train_F.shape, train_D.shape, train_Y.shape, test_features.shape)
	datasets += [[g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels]]

	p_dump(datasets[0], path+'/sars_data')


def pretrain(path):
	file = open(path, 'rb')
	g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels = pickle.load(file)
	file.close()
	train_gcn(RGCN, g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels, hidden_layers, batch_size, 0.01, early_stop, name='pretrained_model.pt')


hidden_layers = [15, 15, 15, 15, 15, 10]
val_ratio = 0.2
batch_size = 256
lr = 1e-3
early_stop = 5
if __name__ == '__main__':
	# import matplotlib.pyplot as plt
	# vect = pd.read_csv('task1_data/pseudomonas/test_vector.csv')
	# train_vect = pd.read_csv('task1_data/pseudomonas/train_vector.csv')
	# print(vect)
	# vect.hist()
	# train_vect.hist()
	# plt.show()

	# save_data('covid_data/sars')

	path = sys.argv[1]
	run = sys.argv[2]
	pretrain(path+"/sars_data")
	if run == 'fold':
		from fold_exp import *
		run_exp(path, hidden_layers, batch_size, lr, early_stop)
	elif run == 'skf':
		from skf_exp import *
		run_exp(path, hidden_layers, batch_size, lr, early_stop)
	else:
		from train_sub import *
		train_final(path, hidden_layers, batch_size, lr, early_stop)
