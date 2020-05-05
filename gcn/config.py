import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]


parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--base_path', type=str, default='Model_weights/',
                    help='Base Path for storing weights')
parser.add_argument('--extra_path', type=str, default='64_dropout/',
                    help='Extra Path for storing weights')
parser.add_argument('--n_features', type=int, default=75,
                    help='Number of features for nodes')

parser.add_argument('--data_type', type=str, default='psuedo',choices=['psuedo','folds'],
                    help='Whether to use psudonomas/train.csv or the 10 fold cv splits')
parser.add_argument('--num_folds', type=int, default=10,
                    help='Number of folds for CV')
parser.add_argument('--num_epochs', type=int, default=51,
                    help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch Size')
parser.add_argument('--net_type', type=str, default='Edge',choices=['Edge','Conv'],
                    help='EdgeGraphConv or GraphConv')









args = parser.parse_args()