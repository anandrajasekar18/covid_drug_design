from gcn_model import *
from rgcn_model import *
import pandas as pd

def train_gcn_mask_final(clf, graphs, node_feat, descriptors, labels, hidden_layers, batch_size=64, lr=1e-2, early=7):
	train_mask = np.arange(len(node_feat))

	_, natoms, nfeatures = node_feat.shape
	net = clf(nfeatures, hidden_layers, descriptors.shape[1], 2)
	print(net)
	net = load_model(net, 'pretrained_model.pt')
	net.cuda()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	overfit, best_loss, best_epoch, best_net = 0, np.inf, 0, None
	best_stat = []
	for epoch in range(20):
		np.random.shuffle(train_mask)
		for step in tqdm(range(int(len(train_mask) / batch_size))):
			train_batch_mask = train_mask[step * batch_size:(step + 1) * batch_size]
			batch_features = node_feat[train_batch_mask].reshape((batch_size * natoms, nfeatures))
			logits, vector = net(dgl.batch(graphs[train_batch_mask]), batch_features, descriptors[train_batch_mask])
			y_hot = F.one_hot(labels[train_batch_mask], 2)
			loss = criterion(logits, y_hot.type_as(logits))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# if step % int(0.1 * len(train_mask) / batch_size) == 0:
			# 	print('\tStep:', step, loss.detach().item())
		train_acc, train_auprc, train_auc, train_loss = evaluate(net, graphs, node_feat, descriptors, labels)

		print("Epoch", epoch,
			"| Train Loss ", np.round(train_loss, 4),
			"| Train Acc ", np.round(train_acc, 4),
			"| Train AUC ", np.round(train_auc, 4),
			"| Train AUPRC ", np.round(train_auprc, 4), end=' ')

		if train_loss > best_loss:
			overfit += 1
			print('Overfit:', overfit)
			if overfit == early:
				break
		else:
			best_loss = train_loss
			best_model = net
			best_stat = [train_loss, train_acc, train_auc, train_auprc]
			torch.save(net.state_dict(), 'gcn_model_final.pt')
			overfit=0
			print('<-- Saved')
	return best_model, best_stat


def train_final(path, hidden_layer, batch_size, lr, early_stop):
	torch.cuda.empty_cache()
	g, features, descriptors, labels, test_g, test_features, test_descriptors = get_data(path)
	final_model, best_stat = train_gcn_mask_final(RGCN, g, features.cuda(), descriptors.cuda(), labels.cuda(), hidden_layer, batch_size, lr, early_stop)
	torch.cuda.empty_cache()
	print(best_stat)
	np.set_printoptions(suppress=True)
	pred, logits, vector = predict(final_model, test_g, test_features, test_descriptors)
	test_data = pd.read_csv(path+'/test.csv')
	print(test_data, pred, len(pred))
	test_data['activity']=np.array(pred).round(5)
	prev = pd.read_csv(path+'/test_pred2.csv')
	print('Old:\t',list(prev['activity'].values.round(3)))
	print('New:\t',list(test_data['activity'].values.round(3)))
	print('Old:\t',sorted(prev['activity'].values.round(3))[::-1][:25])
	print('New:\t',sorted(test_data['activity'].values)[::-1][:25])

	pred, logits, train_vector = predict(final_model, g, features, descriptors)
	pd.DataFrame(train_vector).to_csv(path+'/train_vectors.csv', index= False)
	pd.DataFrame(vector).to_csv(path+'/vectors.csv', index= False)
	test_data.to_csv(path+'/test_pred3.csv', index= False, columns =['smiles', 'activity'])
