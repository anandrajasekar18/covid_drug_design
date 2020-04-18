import sys

from dataset import MoleculeCSVDataset
from dgl_mol import smiles_to_bigraph
import dgl.backend as F

try:
    import pandas as pd
except ImportError:
    pass

class Ecoli(MoleculeCSVDataset):
    """Ecoli dataset.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    Parameters
    ----------
    data_path: str
        Path to the csv file containing the smiles and labels
    smiles_to_graph: callable, str -> DGLGraph
        A function turning smiles into a DGLGraph.
        Default to :func:`dgl.data.chem.smiles_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    """
    def __init__(self, data_path, smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False):

        df = pd.read_csv(data_path)
        # NOTE uncomment next two lines for other datasets
        # self.id = df['mol_id']

        # df = df.drop(columns=['mol_id'])

        super(Ecoli, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles",cache_file_path='useless_file_path', load=load)
        self._weight_balancing()

    def _weight_balancing(self):
        """Perform re-balancing for each task.

        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        If weight balancing is performed, one attribute will be affected:

        * self._task_pos_weights is set, which is a list of positive sample weights
          for each task.
        """
        num_pos = F.sum(self.labels, dim=0)
        num_indices = F.sum(self.mask, dim=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos

    @property
    def task_pos_weights(self):
        """Get weights for positive samples on each task

        Returns
        -------
        numpy.ndarray
            numpy array gives the weight of positive samples on all tasks
        """
        return self._task_pos_weights