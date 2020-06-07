import pickle
from tqdm import tqdm
import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from molecular_graph import mol_graph

ratio = 0.0005
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio, 1-ratio]).cuda())

def gcn_data_ready(graph_data, features, descriptors, labels):
	features = torch.FloatTensor(features)
	try:
		labels = torch.LongTensor(labels)
	except:
		labels = None
	descriptors = torch.FloatTensor(descriptors)
	g = get_dgl_graph(graph_data)
	return g, features, descriptors, labels


def get_dgl_graph(graph_data):
	data = []
	for i, mat in enumerate(graph_data):
		G = nx.from_numpy_matrix(mat)
		g = dgl.DGLGraph(G)
		g.edata.update({'rel_type': mat.flatten()[mat.flatten() != 0.]})
		if i%int(0.1*len(graph_data))==0:
			print(100.*i/len(graph_data), '% Done from', len(graph_data))
		data += [g]
	return np.array(data)


def auprc_scorer(y_true, y_pred_proba):
	pre, rec, thre = precision_recall_curve(y_true.detach().cpu(), np.round(y_pred_proba, 4))
	return auc(rec, pre)


def predict(model, test_g, test_features, test_descriptors):
	model.eval()
	sh = test_features.shape
	with torch.no_grad():
		logits, vector = model(dgl.batch(test_g), test_features.reshape((sh[0] * sh[1], sh[-1])).cuda(), test_descriptors.cuda())
		return F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy(), logits, vector.detach().cpu().numpy()


def evaluate(net, g, features, descriptors, labels, batch_size=64):
	net.eval()
	_, natoms, nfeatures = features.shape
	with torch.no_grad():
		steps = int(len(features) / batch_size)
		correct, loss = 0, 0
		softs = []
		for step in range(steps+1):
			batch_mask = np.arange(len(labels))[step * batch_size:(step + 1) * batch_size]
			batch_feat, batch_desc = features[batch_mask].reshape((len(batch_mask) * natoms, nfeatures)).cuda(), descriptors[batch_mask].cuda()
			logits, _ = net(dgl.batch(g[batch_mask]), batch_feat, batch_desc)
			correct += torch.sum(torch.max(logits, dim=1)[1] == labels[batch_mask].cuda())
			softs += list(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
			loss += criterion(logits, F.one_hot(labels[batch_mask], 2).type_as(logits)).detach().item()/steps
		softs = np.array(softs).reshape(-1, 1)
		auprc = auprc_scorer(labels, softs)
		auc = roc_auc_score(F.one_hot(labels, 2).detach().cpu(), softs)
		acc = correct.item() * 1.0 / len(labels)
		return acc, auprc, auc, loss


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




def norm(train_D, test_D):
	# data = np.append(train_D, test_D, axis=0)
	mean, std = train_D.mean(axis=0), train_D.std(axis=0)
	train_D = (train_D - mean) / std
	test_D = (test_D - mean) / std
	train_D[np.abs(train_D) == np.inf] = 0.0
	test_D[np.abs(test_D) == np.inf] = 0.0
	return np.nan_to_num(train_D), np.nan_to_num(test_D)
