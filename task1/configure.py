from featurizers import BaseAtomFeaturizer, CanonicalAtomFeaturizer, ConcatFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot, BaseBondFeaturizer
from functools import partial

from utils import chirality

GCN_pseud = {
    'train_path': 'Data/pseudomonas/train_cv/fold_0/train.csv',
    'test_path': 'Data/pseudomonas/train_cv/fold_0/test.csv',
    'val_path': 'Data/pseudomonas/train_cv/fold_0/dev.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_pseud = {
    'train_path': 'Data/pseudomonas/train_cv/fold_0/train.csv',
    'test_path': 'Data/pseudomonas/train_cv/fold_0/test.csv',
    'val_path': 'Data/pseudomonas/train_cv/fold_0/dev.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}


GCN_Ecoli_MIT = {
    'train_path': 'Data/splits/ecoli_scaffold/train.csv',
    'test_path': 'Data/splits/ecoli_scaffold/test.csv',
    'val_path': 'Data/splits/ecoli_scaffold/dev.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_Ecoli_MIT = {
    'train_path': 'Data/splits/ecoli_scaffold/train.csv',
    'test_path': 'Data/splits/ecoli_scaffold/test.csv',
    'val_path': 'Data/splits/ecoli_scaffold/dev.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GCN_Ecoli = {
    'path': 'Data/ecoli.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_Ecoli = {
    'path': 'Data/ecoli.csv',
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}


GCN_Tox21 = {
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_Tox21 = {
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

MPNN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'node_in_feats': 15,
    'edge_in_feats': 5,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1',
    'weight_decay': 0
}

SCHNET_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'norm': True,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1',
    'weight_decay': 0
}

MGCN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'norm': True,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1',
    'weight_decay': 0
}

AttentiveFP_Aromaticity = {
    'random_seed': 8,
    'graph_feat_size': 200,
    'num_layers': 2,
    'num_timesteps': 2,
    'node_feat_size': 39,
    'edge_feat_size': 10,
    'output_size': 1,
    'dropout': 0.2,
    'weight_decay': 10 ** (-5.0),
    'lr': 10 ** (-2.5),
    'batch_size': 128,
    'num_epochs': 800,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 80,
    'metric_name': 'rmse',
    # Follow the atom featurization in the original work
    'atom_featurizer': BaseAtomFeaturizer(
        featurizer_funcs={'hv': ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=[
                'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                    encode_unknown=True),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            atom_formal_charge, atom_num_radical_electrons,
            partial(atom_hybridization_one_hot, encode_unknown=True),
            lambda atom: [0], # A placeholder for aromatic information,
            atom_total_num_H_one_hot, chirality
        ],
        )}
    ),
    'bond_featurizer': BaseBondFeaturizer({
        'he': lambda bond: [0 for _ in range(10)]
    })
}

experiment_configures = {
    'GCN_pseud':     GCN_pseud,
    'GAT_pseud':     GAT_pseud,
    'GCN_Ecoli_MIT': GCN_Ecoli_MIT,
    'GAT_Ecoli_MIT': GAT_Ecoli_MIT,
    'GCN_Ecoli':     GCN_Ecoli,
    'GAT_Ecoli':     GAT_Ecoli,
    'GCN_Tox21':     GCN_Tox21,
    'GAT_Tox21':     GAT_Tox21,
    'MPNN_Alchemy':  MPNN_Alchemy,
    'SCHNET_Alchemy':SCHNET_Alchemy,
    'MGCN_Alchemy':  MGCN_Alchemy,
    'AttentiveFP_Aromaticity': AttentiveFP_Aromaticity
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]