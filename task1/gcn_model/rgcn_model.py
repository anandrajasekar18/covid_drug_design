from utils import *
import dgl.function as fn

class RGCNLayer(nn.Module):
	def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None, is_input_layer=False):
		super(RGCNLayer, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.num_rels = num_rels
		self.num_bases = num_bases
		self.bias = bias
		self.activation = activation
		self.is_input_layer = is_input_layer

		# sanity check
		if self.num_bases <= 0 or self.num_bases > self.num_rels:
			self.num_bases = self.num_rels

		# weight bases in equation (3)
		self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
		if self.num_bases < self.num_rels:
			# linear combination coefficients in equation (3)
			self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

		# add bias
		if self.bias:
			self.bias = nn.Parameter(torch.Tensor(out_feat))

		# init trainable parameters
		nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
		if self.num_bases < self.num_rels:
			nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
		if self.bias:
			nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

	def forward(self, g, features):
		if self.num_bases < self.num_rels:
			# generate all weights from bases (equation (3))
			weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
			weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)
		else:
			weight = self.weight

		def message_func(edges):
			w = weight[edges.data['rel_type'].type(torch.LongTensor).cuda()]
			msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
			# msg = msg * edges.data['norm'].cuda()
			return {'msg': msg}

		def apply_func(nodes):
			h = nodes.data['h']
			if self.bias:
				h = h + self.bias
			if self.activation:
				h = self.activation(h)
			return {'h': h}

		g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCN(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_desc, n_classes, num_rels=11):
		super(RGCN, self).__init__()
		self.num_nodes = in_dim
		self.layers = nn.ModuleList()
		self.layers.append(RGCNLayer(in_dim, hidden_dim[0], num_rels, 11, activation=F.relu))
		for i, hid in enumerate(hidden_dim[:-1]):
			self.layers.append(RGCNLayer(hid, hidden_dim[i + 1], num_rels, 11, activation=F.relu))
		self.lin1 = nn.Linear(hidden_dim[-1] + n_desc, 1000)
		self.lin2 = nn.Linear(1000, 500)
		self.lin3 = nn.Linear(500, 100)
		self.lin4 = nn.Linear(100, 20)
		self.classify = nn.Linear(20, n_classes)

	def forward(self, g, features, descriptors):
		g.ndata['h'] = features
		for layer in self.layers:
			layer(g, features)

		h = dgl.mean_nodes(g, 'h')

		# Concat (GCN_feat, descriptors)
		h = torch.cat((h, descriptors), 1)

		# Classify
		h = F.relu(self.lin1(h))
		h = F.relu(self.lin2(h))
		h = F.relu(self.lin3(h))
		h = F.relu(self.lin4(h))
		return F.relu(self.classify(h))


class RGCN_des(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_desc, n_classes, num_rels=11):
		super(RGCN_des, self).__init__()
		self.num_nodes = in_dim
		self.layers = nn.ModuleList()
		self.layers.append(RGCNLayer(in_dim, hidden_dim[0], num_rels, 11, activation=F.relu))
		for i, hid in enumerate(hidden_dim[:-1]):
			self.layers.append(RGCNLayer(hid, hidden_dim[i + 1], num_rels, 11, activation=F.relu))
		self.lin1 = nn.Linear(hidden_dim[-1], 10)
		self.lin2 = nn.Linear(10, 5)
		self.classify = nn.Linear(5, n_classes)

	def forward(self, g, features, descriptors):
		g.ndata['h'] = features
		for layer in self.layers:
			layer(g, features)

		h = dgl.mean_nodes(g, 'h')

		# Concat (GCN_feat, descriptors)
		# h = torch.cat((h, descriptors), 1)

		# Classify
		h = F.relu(self.lin1(h))
		h = F.relu(self.lin2(h))
		return F.relu(self.classify(h))