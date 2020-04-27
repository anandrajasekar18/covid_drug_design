# All codes related to task1

To run the model:
`python classification.py --dataset=Ecoli --model=GCN`

Choices with `--dataset`:
* Ecoli: The data given on aicures.mit.edu with random train-test-validation split [source](https://github.com/yangkevin2/coronavirus_data/blob/master/data/ecoli.csv)
* Ecoli_MIT: The data given with separate csvs for train-test-validation [source](https://github.com/yangkevin2/coronavirus_data/blob/master/splits.zip)

Choices with graph network algo `--model`:
* GCN: Graph Convolutional Network [arxiv](https://arxiv.org/abs/1609.02907)
* GAT: Graph Attention Network [arxiv](https://arxiv.org/abs/1710.10903)

## Results:
* `python classification.py --dataset=Ecoli --model=GCN`

validation roc_auc 0.7857; best validation roc_auc 0.9340; test roc_auc 0.8822

* `python classification.py --dataset=Ecoli_MIT --model=GCN`

validation roc_auc 0.8679; best validation roc_auc 0.8725; test roc_auc 0.8479

* `python classification.py --dataset=Ecoli --model=GAT`

validation roc_auc 0.9077; best validation roc_auc 0.9420; test roc_auc 0.8363

* `python classification.py --dataset=Ecoli_MIT --model=GAT`

validation roc_auc 0.8301; best validation roc_auc 0.8644; test roc_auc 0.7913
