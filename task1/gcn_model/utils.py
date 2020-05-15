import pickle

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# from molecular_graph import mol_graph

criterion = nn.BCEWithLogitsLoss()



def get_dgl_graph(graph_data):
	data = []
	for mat in graph_data:
		G = nx.from_numpy_matrix(mat)
		data += [dgl.DGLGraph(G)]
	return np.array(data)


def auprc_scorer(y_true, y_pred_proba):
	pre, rec, thre = precision_recall_curve(y_true.detach().cpu(), y_pred_proba.detach().cpu())
	return auc(rec, pre)


def predict(model, test_g, test_features, test_descriptors):
	model.eval()
	sh = test_features.shape
	with torch.no_grad():
		logits = model(dgl.batch(test_g), test_features.reshape((sh[0] * sh[1], sh[-1])), test_descriptors)
		return F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy(), logits


def evaluate(net, g, features, descriptors, labels):
	net.eval()
	sh = features.shape
	with torch.no_grad():
		features = features.reshape((sh[0] * sh[1], sh[-1]))
		logits = net(dgl.batch(g), features, descriptors)
		correct = torch.sum(torch.max(logits, dim=1)[1] == labels)
		y_hot = F.one_hot(labels, 2)
		loss = criterion(logits, y_hot.type_as(logits))
		auprc = auprc_scorer(labels, F.softmax(logits, dim=1)[:, 1])
		auc = roc_auc_score(y_hot.detach().cpu(), F.softmax(logits, dim=1).detach().cpu())
		acc = correct.item() * 1.0 / len(labels)
		return acc, auprc, auc, loss.detach().item()


def gcn_data_ready(graph_data, features, descriptors, labels):
	features = torch.FloatTensor(features)
	try:
		labels = torch.LongTensor(labels)
	except:
		labels = None
	descriptors = torch.FloatTensor(descriptors)
	g = get_dgl_graph(graph_data)
	return g, features, descriptors, labels


def get_mask(labels, val_ratio):
	val_mask = []
	sampl = torch.sum(labels == 1.0) * val_ratio
	while len(val_mask) <= sampl:
		z = np.random.choice(range(len(labels)))
		if labels[z] == 1.0 and z not in val_mask:
			val_mask.append(z)
	while len(val_mask) <= val_ratio * len(labels):
		z = np.random.choice(range(len(labels)))
		if labels[z] == 0.0 and z not in val_mask:
			val_mask.append(z)
	train_mask = [z for z in range(len(labels)) if not z in val_mask]
	return train_mask, val_mask


def load_model(model, model_path='ecoli_model.pt'):
	model.load_state_dict(torch.load(model_path))
	model.eval()
	return model


def get_data(path):
	file = open(path+"/pseudomonas_data", 'rb')
	g, features, descriptors, labels, test_g, test_features, test_descriptors, _ = pickle.load(file)
	file.close()
	return g, features, descriptors, labels, test_g, test_features, test_descriptors


def p_dump(data, name='pseudomonas_data'):
	file = open(name, 'wb')
	pickle.dump(data, file)
	print(name)
	file.close()


def norm(train_D, test_D):
	# data = np.append(train_D, test_D, axis=0)
	mean, std = train_D.mean(axis=0), train_D.std(axis=0)
	train_D = (train_D - mean) / std
	test_D = (test_D - mean) / std
	train_D[np.abs(train_D) == np.inf] = 0.0
	test_D[np.abs(test_D) == np.inf] = 0.0
	return np.nan_to_num(train_D), np.nan_to_num(test_D)
