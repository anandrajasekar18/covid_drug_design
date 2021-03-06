from gcn_model import *
from rgcn_model import *
from sklearn.model_selection import StratifiedKFold

def train_gcn_mask(clf, graphs, node_feat, descriptor, labels, train_mask, val_mask, hidden_layers, batch_size=64, lr=1e-2, early=7):
	_, natoms, nfeatures = node_feat.shape
	net = clf(nfeatures, hidden_layers, descriptor.shape[1], 2)
	# print(net)
	net = load_model(net, 'pretrained_model.pt')
	net.cuda()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	overfit, best_loss, best_epoch, best_net = 0, np.inf, 0, None

	for epoch in tqdm(range(30)):
		np.random.shuffle(train_mask)
		np.random.shuffle(val_mask)
		train_acc, train_auprc, train_auc, train_loss = 0,0,0,0
		for step in range(int(len(train_mask) / batch_size)):
			train_batch_mask = train_mask[step * batch_size:(step + 1) * batch_size]
			batch_features = node_feat[train_batch_mask].reshape((batch_size * natoms, nfeatures))
			logits, vector = net(dgl.batch(graphs[train_batch_mask]), batch_features, descriptor[train_batch_mask])
			y_hot = F.one_hot(labels[train_batch_mask], 2)
			loss = criterion(logits, y_hot.type_as(logits))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.detach().cpu()
			# if step % int(0.05 * len(train_mask) / batch_size) == 0:
			# 	print('\tStep:', step, loss.detach().item())
		train_acc, train_auprc, train_auc, train_loss = evaluate(net, graphs[train_mask], node_feat[train_mask], descriptor[train_mask], labels[train_mask])
		val_acc, val_auprc, val_auc, val_loss = evaluate(net, graphs[val_mask], node_feat[val_mask], descriptor[val_mask], labels[val_mask])

		# print("Epoch", epoch,
		#       "| Train Loss ", np.round(train_loss, 4),
		#       "| Train Acc ", np.round(train_acc, 4),
		#       "| Train AUC ", np.round(train_auc, 4),
		#       "| Train AUPRC ", np.round(train_auprc, 4), end=' ')
		#
		# print("| Val Loss ", np.round(val_loss, 4),
		#       "| Val Acc ", np.round(val_acc, 4),
		#       "| Val AUC ", np.round(val_auc, 4),
		#       "| Val AUPRC ", np.round(val_auprc, 4), end=' ')

		if val_loss > best_loss:
			overfit += 1
			# print('Overfit:', overfit)
			if overfit == early:
				break
		else:
			best_loss = val_loss
			best_model = net
			best_stat = [train_loss, train_acc, train_auc, train_auprc, val_loss, val_acc, val_auc, val_auprc]
			# torch.save(net.state_dict(), 'skf_model.pt')
			overfit = 0
			# print('<-- Saved')
	return best_model, best_stat


def cross_val(clf, g, features, descriptors, labels, hidden_layers, batch_size=128, lr=1e-3, early_stop=5):
	final_test = [0]
	while len(final_test)<int(0.1*len(descriptors)) and torch.sum(labels[final_test])!=5:
		ind = np.random.choice(np.arange(len(descriptors)))
		if ind not in final_test:
			final_test+=[ind]
		final_test = final_test[-int(0.1*len(descriptors)):]
	final_train = list(set(np.arange(len(descriptors))).difference(final_test))
	np.random.shuffle(final_train)
	skf = StratifiedKFold(n_splits=10, shuffle=True)
	skf_data = []
	skf_models = []
	best_auprc = 0.0
	for train_mask, val_mask in skf.split(descriptors[final_train], labels[final_train]):
		model, best_stat = train_gcn_mask(clf, g, features.cuda(), descriptors.cuda(), labels.cuda(), train_mask, val_mask, hidden_layers, batch_size, lr, early_stop)
		if best_auprc < best_stat[-1]:
			best_model = model
			best_auprc = best_stat[-1]
		skf_data += [best_stat]
		skf_models += [model]
		print('Model trained', len(skf_data), best_stat)
		del model
		torch.cuda.empty_cache()
	val_acc, val_auprc, val_auc, val_loss = evaluate(best_model, g[final_test], features[final_test].cuda(), descriptors[final_test].cuda(), labels[final_test].cuda())
	print('Best model', val_acc, val_auprc, val_auc, val_loss)
	return np.array(skf_data)


def run_exp(path, hidden_layers, batch_size, lr, early_stop):
	g, features, descriptors, labels, test_g, test_features, test_descriptors = get_data(path)
	# for hid in [[15, 15, 15, 10]]:
	score_rgcn = cross_val(RGCN, g, features, descriptors, labels, hidden_layers, batch_size, lr, early_stop)
	print('RGCN_des', hidden_layers, score_rgcn.mean(axis=0), score_rgcn.std(axis=0), '\n')
	# score_gcn = cross_val(GCN, g, features, descriptors, labels, hid, batch_size, lr, early_stop)
	# print('GCN_des', hid, score_gcn, '\n')
	# score_gcn_des = cross_val(GCN_des, g, features, descriptors, labels, hid, batch_size, lr, early_stop)
	# print('GCN', hid, score_gcn_des, '\n')
	# score_rgcn_des = cross_val(RGCN_des, g, features, descriptors, labels, hid, batch_size, lr, early_stop)
	# print('RGCN', hid, score_rgcn_des, '\n\n')
