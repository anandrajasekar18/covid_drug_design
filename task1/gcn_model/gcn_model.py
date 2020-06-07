from utils import *

import dgl.function as fn
import math


class GCNLayer(nn.Module):
	def __init__(self, in_feats, out_feats, activation=F.relu, dropout=0.1, bias=True):
		super(GCNLayer, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats)).cuda()
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_feats)).cuda()
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
		h = torch.mm(h, self.weight.cuda())
		g.ndata['h'] = h
		g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
		h = g.ndata.pop('h')
		if self.bias is not None:
			h = h + self.bias
		if self.activation:
			h = self.activation(h)
		return h


class GCN(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_desc, n_classes):
		super(GCN, self).__init__()
		self.layers = nn.ModuleList()
		self.layers.append(GCNLayer(in_dim, hidden_dim[0], F.relu, 0.))
		for i, hid in enumerate(hidden_dim[:-1]):
			self.layers.append(GCNLayer(hid, hidden_dim[i + 1], F.relu, 0.2))

		self.lin1 = nn.Linear(hidden_dim[-1] + n_desc, 1000)
		self.lin2 = nn.Linear(1000, 500)
		self.lin3 = nn.Linear(500, 100)
		self.lin4 = nn.Linear(100, 20)
		self.drop = nn.Dropout(0.2)
		self.classify = nn.Linear(20, n_classes)

	def forward(self, g, features, descriptors):
		# GCN
		h = features
		for layer in self.layers:
			h = layer(g, h)
		g.ndata['h'] = h
		feats = dgl.mean_nodes(g, 'h')

		# Concat (GCN_feat, descriptors)
		h = torch.cat((feats, descriptors), 1)

		# Classify
		h = self.drop(F.relu(self.lin1(h)))
		h = self.drop(F.relu(self.lin2(h)))
		h = self.drop(F.relu(self.lin3(h)))
		h = self.drop(F.relu(self.lin4(h)))
		return self.classify(h), feats


class GCN_des(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_desc, n_classes):
		super(GCN_des, self).__init__()
		self.layers = nn.ModuleList()
		self.layers.append(GCNLayer(in_dim, hidden_dim[0], F.relu, 0.))
		for i, hid in enumerate(hidden_dim[:-1]):
			self.layers.append(GCNLayer(hid, hidden_dim[i + 1], F.relu, 0.2))

		self.lin1 = nn.Linear(hidden_dim[-1], 10)
		self.lin2 = nn.Linear(10, 5)
		self.drop = nn.Dropout(0.2)
		self.classify = nn.Linear(5, n_classes)

	def forward(self, g, features, descriptors):
		# GCN
		h = features
		for layer in self.layers:
			h = layer(g, h)
		g.ndata['h'] = h
		feats = dgl.mean_nodes(g, 'h')

		# Concat (GCN_feat, descriptors)
		# h = torch.cat((h, descriptors), 1)

		# Classify
		h = self.drop(F.relu(self.lin1(feats)))
		h = self.drop(F.relu(self.lin2(h)))
		return self.classify(h), feats