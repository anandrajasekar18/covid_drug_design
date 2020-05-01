import numpy as np
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import roc_auc_score
import math

criterion = nn.BCEWithLogitsLoss()

model_path = 'gcn_model.pt'


class GCNLayer(nn.Module):
	def __init__(self, in_feats, out_feats, activation=F.relu, dropout=0.1, bias=True):
		super(GCNLayer, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_feats))
		else:
			self.bias = None
		self.activation = activation
		if dropout:
			self.dropout = nn.Dropout(p=dropout)
		else:
			self.dropout = 0.
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, g, h):
		if self.dropout:
			h = self.dropout(h)
		h = torch.mm(h, self.weight)
		g.ndata['h'] = h
		g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
		h = g.ndata.pop('h')
		if self.bias is not None:
			h = h + self.bias
		if self.activation:
			h = self.activation(h)
		return h


class Classifier(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_desc, n_classes):
		super(Classifier, self).__init__()
		self.layers = nn.ModuleList()
		self.layers.append(GCNLayer(in_dim, hidden_dim[0], F.relu, 0.))
		for i, hid in enumerate(hidden_dim[:-1]):
			self.layers.append(GCNLayer(hid, hidden_dim[i + 1], F.relu, 0.2))

		self.lin1 = nn.Linear(hidden_dim[-1]+n_desc, 500)
		self.lin2 = nn.Linear(500, 100)
		self.classify = nn.Linear(100, n_classes)

	def forward(self, g, features, descriptors):
		# GCN
		h = features
		for layer in self.layers:
			h = layer(g, h)
		g.ndata['h'] = h
		h = dgl.mean_nodes(g, 'h')

		# Concat (GCN_feat, descriptors)
		h = torch.cat((h, descriptors), 1)

		# Classify
		h = F.relu(self.lin1(h))
		h = F.relu(self.lin2(h))
		return self.classify(h)


def get_dgl_graph(graph_data):
	data = []
	for mat in graph_data:
		G = nx.from_numpy_matrix(mat)
		data += [dgl.DGLGraph(G)]
	return np.array(data)


def evaluate(net, g, features, descriptors, labels):
	net.eval()
	sh = features.shape
	with torch.no_grad():
		features = features.reshape((sh[0] * sh[1], sh[-1]))
		logits = net(dgl.batch(g), features, descriptors)
		_, indices = torch.max(logits, dim=1)
		correct = torch.sum(indices == labels)
		y_hot = F.one_hot(labels, 2)
		loss = criterion(logits, y_hot.type_as(logits))
		auc = roc_auc_score(labels, indices)
		acc = correct.item() * 1.0 / len(labels)
		return acc, auc, loss.detach().item()


def gcn_data_ready(graph_data, features, descriptors, labels):
	features = torch.FloatTensor(features)
	labels = torch.LongTensor(labels)
	descriptors = torch.FloatTensor(descriptors)
	g = get_dgl_graph(graph_data)
	return g, features, descriptors, labels


def load_model(model):
	model.load_state_dict(torch.load(model_path))
	model.eval()
	return model


def train_gcn(graphs, node_feat, descriptors, labels, hidden_layers=[30, 200, 500], val_ratio=0.2, batch_size=64, lr=1e-2, early=7):
	train_mask = np.random.choice(range(len(graphs)), int((1 - val_ratio) * len(graphs)))
	val_mask = [z for z in range(len(graphs)) if not z in train_mask]

	_, natoms, nfeatures = node_feat.shape

	net = Classifier(nfeatures, hidden_layers, descriptors.shape[1], 2)
	print(net)
	try:
		net = load_model(net)
	except:
		pass

	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	overfit, best_loss, best_epoch, best_net = 0, np.inf, 0, None

	for epoch in range(1000):
		epoch_loss = 0
		np.random.shuffle(train_mask)
		for step in range(int(len(train_mask) / batch_size)):
			train_batch_mask = train_mask[step * batch_size:(step + 1) * batch_size]
			batch_features = node_feat[train_batch_mask].reshape((batch_size * natoms, nfeatures))
			logits = net(dgl.batch(graphs[train_batch_mask]), batch_features, descriptors[train_batch_mask])
			y_hot = F.one_hot(labels[train_batch_mask], 2)
			loss = criterion(logits, y_hot.type_as(logits))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss += loss.detach().item()
		# if step % int(0.2 * len(train_mask) / batch_size) == 0:
		# 	print('\tStep:', step, loss.detach().item())
		epoch_loss /= (step + 1)
		val_acc, val_auc, val_loss = evaluate(net, graphs[val_mask], node_feat[val_mask], descriptors[val_mask], labels[val_mask])

		print("Epoch", epoch,
		      "| Train Loss ", np.round(epoch_loss, 4),
		      "| Val Loss ", np.round(val_loss, 4),
		      "| Val Acc ", np.round(val_acc, 4),
		      "| Val AUC ", np.round(val_auc, 4), end=' ')

		if val_loss > best_loss:
			if epoch > 10:
				overfit += 1
			print('Overfit:', overfit)
			if overfit == early:
				break
		else:
			best_loss = val_loss
			torch.save(net.state_dict(), model_path)
			overfit=0
			print('<-- Saved')
	return net
