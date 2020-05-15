## Graph classification using GCN

##### Prerequisites

1. numpy
2. pandas
3. sklearn
4. pytorch
5. rdkit
6. dgl
7. networkx

##### Usage
Specify the arguments `<data_directory>` and `<experiment>` and execute the following command:

  `python3 main_.py <data_directory> <experiment>` 

Possible `<experiment>` values are: 
* `skf`: Run cross validation with k=10 on Pseudomonas data
* `fold`: Train model on their Folds data
* `final`: Train best model on Pseudomonas data and make predictions on their Test data
