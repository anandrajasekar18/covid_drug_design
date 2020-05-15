from fold_exp import *
import pandas as pd
import sys


def save_folds(path):
    # train_data = pd.read_csv('task1_data/pseudomonas/train.csv')
    # test_data = pd.read_csv('task1_data/pseudomonas/test.csv')
    cluster = None
    i = 0
    datasets = []
    for i in range(10):
        train_file = path+"/pseudomonas/train_cv/fold_" + str(i) + "/train.csv"
        test_file = path+"/pseudomonas/train_cv/fold_" + str(i) + "/test.csv"
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        print('Data imported...', train_data.shape, test_data.shape)

        train_A, train_F, train_D, train_Y, clus = preprocess_dataset(train_data, cluster)
        if i==0:
            cluster = clus
        test_A, test_F, test_D, test_Y, _ = preprocess_dataset(test_data, cluster)
        train_D, test_D = norm(train_D, test_D)
        g, features, descriptors, labels = gcn_data_ready(train_A, train_F, train_D, train_Y)
        test_g, test_features, test_descriptors, test_labels = gcn_data_ready(test_A, test_F, test_D, test_Y)
        print('Preprocessed...', train_A.shape, train_F.shape, train_D.shape, train_Y.shape, test_features.shape)
        datasets+=[[g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels]]

    p_dump(datasets, 'pseudomonas_folds')


def save_data(path):
    train_data = pd.read_csv(path+'/pseudomonas/train.csv')
    test_data = pd.read_csv(path+'/pseudomonas/test.csv')
    cluster = None
    i = 0
    datasets = []
    print('Data imported...', train_data.shape, test_data.shape)

    train_A, train_F, train_D, train_Y, clus = preprocess_dataset(train_data, cluster)
    if i==0:
        cluster = clus
    test_A, test_F, test_D, test_Y, _ = preprocess_dataset(test_data, cluster)
    train_D, test_D = norm(train_D, test_D)
    g, features, descriptors, labels = gcn_data_ready(train_A, train_F, train_D, train_Y)
    test_g, test_features, test_descriptors, test_labels = gcn_data_ready(test_A, test_F, test_D, test_Y)
    print('Preprocessed...', train_A.shape, train_F.shape, train_D.shape, train_Y.shape, test_features.shape)
    datasets+=[[g, features, descriptors, labels, test_g, test_features, test_descriptors, test_labels]]

    p_dump(datasets[0], 'pseudomonas_data')


hidden_layers = [15, 15, 15, 15]
val_ratio = 0.2
batch_size = 128
lr = 1e-2
early_stop = 5

if __name__ == '__main__':
    path = sys.argv[1]
    run = sys.argv[2]
    if run == 'fold':
        from fold_exp import *
        run_exp(path, hidden_layers, batch_size, lr, early_stop)
    elif run == 'skf':
        from skf_exp import *
        run_exp(path, hidden_layers, batch_size, lr, early_stop)
    else:
        from train_sub import *
        train_final(path, hidden_layers, batch_size, lr, early_stop)