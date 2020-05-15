from gcn_model import *
from rgcn_model import *


def train_gcn(clf, graphs, node_feat, descriptor, labels, val_g, val_features, val_descriptors, val_labels, hidden_layers, batch_size=64, lr=1e-2, early=7):
	train_mask = np.arange(len(node_feat))
	_, natoms, nfeatures = node_feat.shape
	net = clf(nfeatures, hidden_layers, descriptor.shape[1], 2)
	# print(net)
	net.cuda()
	# net = load_model(net, 'ecoli_model.pt')
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	overfit, best_loss, best_epoch, best_net = 0, np.inf, 0, None
	best_stat = []
	for epoch in range(30):
		np.random.shuffle(train_mask)
		for step in range(int(len(train_mask) / batch_size)):
			train_batch_mask = train_mask[step * batch_size:(step + 1) * batch_size]
			batch_features = node_feat[train_batch_mask].reshape((batch_size * natoms, nfeatures))
			logits = net(dgl.batch(graphs[train_batch_mask]), batch_features, descriptor[train_batch_mask])
			y_hot = F.one_hot(labels[train_batch_mask], 2)
			loss = criterion(logits, y_hot.type_as(logits))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# if step % int(0.1 * len(train_mask) / batch_size) == 0:
			# 	print('\tStep:', step, loss.detach().item())
		train_acc, train_auprc, train_auc, train_loss = evaluate(net, graphs, node_feat, descriptor, labels)
		val_acc, val_auprc, val_auc, val_loss = evaluate(net, val_g, val_features, val_descriptors, val_labels)

		print("Epoch", epoch,
		      "| Train Loss ", np.round(train_loss, 4),
		      "| Train Acc ", np.round(train_acc, 4),
		      "| Train AUC ", np.round(train_auc, 4),
		      "| Train AUPRC ", np.round(train_auprc, 4), end=' ')

		print(
		      "| Val Loss ", np.round(val_loss, 4),
		      "| Val Acc ", np.round(val_acc, 4),
		      "| Val AUC ", np.round(val_auc, 4),
		      "| Val AUPRC ", np.round(val_auprc, 4), end=' ')

		if val_loss > best_loss:
			overfit += 1
			print('Overfit:', overfit)
			if overfit == early:
				break
		else:
			best_loss = val_loss
			best_model = net
			best_stat = [train_loss, train_acc, train_auc, train_auprc, val_loss, val_acc, val_auc, val_auprc]
			# torch.save(net.state_dict(), 'folds_model.pt')
			overfit=0
			print('<-- Saved')
	return best_stat


def get_folds_data(path):
	file = open(path+"/pseudomonas_folds", 'rb')
	datasets = pickle.load(file)
	file.close()
	return datasets


def run_exp(path, hidden_layers, batch_size=128, lr=1e-3, early_stop=5):
	datasets = get_folds_data(path)
	folds_data = []
	for g, features, descriptors, labels, val_g, val_features, val_descriptors, val_labels in datasets:
		best_stat_gcn = train_gcn(GCN, g, features.cuda(), descriptors.cuda(), labels.cuda(), val_g, val_features.cuda(), val_descriptors.cuda(), val_labels.cuda(), hidden_layers, batch_size, lr, early_stop)
		print('best_stat_gcn',len(folds_data), best_stat_gcn)
		best_stat_gcn_des = train_gcn(GCN_des, g, features.cuda(), descriptors.cuda(), labels.cuda(), val_g, val_features.cuda(), val_descriptors.cuda(), val_labels.cuda(), hidden_layers, batch_size, lr, early_stop)
		print('best_stat_gcn_des',len(folds_data), best_stat_gcn_des)
		best_stat_rgcn = train_gcn(RGCN, g, features.cuda(), descriptors.cuda(), labels.cuda(), val_g, val_features.cuda(), val_descriptors.cuda(), val_labels.cuda(), hidden_layers, batch_size, lr, early_stop)
		print('best_stat_rgcn',len(folds_data), best_stat_rgcn)
		best_stat_rgcn_des = train_gcn(RGCN_des, g, features.cuda(), descriptors.cuda(), labels.cuda(), val_g, val_features.cuda(), val_descriptors.cuda(), val_labels.cuda(), hidden_layers, batch_size, lr, early_stop)
		print('best_stat_rgcn_des',len(folds_data), best_stat_rgcn_des,  '\n')
		folds_data+=[[best_stat_rgcn[-1], best_stat_gcn_des[-1], best_stat_rgcn[-1], best_stat_rgcn_des[-2]]]
		torch.cuda.empty_cache()
	return np.array(folds_data).mean(axis=0)