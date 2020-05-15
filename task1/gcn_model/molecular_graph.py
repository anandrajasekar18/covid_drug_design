import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


descriptors = list(Chem.rdMolDescriptors.Properties().GetAvailableProperties())
cls = Chem.rdMolDescriptors.Properties(descriptors)

def one_of_k_encoding_unk(x, allowable_set):
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))


class mol_graph:

	def __init__(self, max_atoms):
		self.graphs_ls = []
		self.n_bonds = 4  # single, double, triple
		self.n_feat = 15
		self.max_atoms = max_atoms
		self.bond_types = np.array(
			[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
		self.possible_hybridization = np.array(
			[Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
			 Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])

		self.A = np.zeros((1, self.max_atoms, self.max_atoms, 10))
		self.F = np.zeros((1, self.max_atoms, self.n_feat))

	def get_node_feat(self, atom):
		atomid = atom.GetIdx()
		charge = atom.GetFormalCharge()
		ring = atom.IsInRing()
		degree = atom.GetDegree()
		valence = atom.GetImplicitValence()
		hybridization = (atom.GetHybridization() == self.possible_hybridization).tolist()
		ring_size = [not atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5), atom.IsInRingSize(6), (atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) and ( not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6)))]
		return atomid, [charge, ring, degree, valence] + hybridization + ring_size

	def bond_features(self, bond, use_chirality=True):
		bt = bond.GetBondType()
		bond_feats = [
			bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
			bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
			bond.GetIsConjugated(),
			bond.IsInRing()
		]
		if use_chirality:
			bond_feats = bond_feats + one_of_k_encoding_unk(
				str(bond.GetStereo()),
				["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
		return np.array(bond_feats)

	def get_graph(self, smiles):
		mol = Chem.MolFromSmiles(smiles)
		Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
		for atom in mol.GetAtoms():
			feat_arr = self.get_node_feat(atom)
			self.F[0, feat_arr[0], :] = feat_arr[1]
		n_atoms = mol.GetNumAtoms()
		for i in range(self.n_bonds):
			self.A[0, :n_atoms, :n_atoms, i] = np.eye(n_atoms)

		for bond in mol.GetBonds():
			beginidx = bond.GetBeginAtomIdx()
			endidx = bond.GetEndAtomIdx()
			self.A[0, beginidx, endidx] = self.A[0, endidx, beginidx] = self.bond_features(bond)

		return self.A, self.F

	def get_flatten_graph(self, smiles):
		A, F = self.get_graph(smiles)
		return np.append(A.flatten(), F.flatten(), axis=0)



def get_descriptors(smiles):
	mol = Chem.MolFromSmiles(smiles)
	fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
	arr = np.empty((1,), dtype=np.int8)
	AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
	moldes = np.array(cls.ComputeProperties(mol)).reshape((1, -1))
	return np.concatenate([arr.reshape((1, -1)), moldes], axis=-1)


def get_max_atom(train_data):
	smiles_ls = train_data.smiles.values
	atoms_ls = []
	for smiles in smiles_ls:
		mol = Chem.MolFromSmiles(smiles)
		atoms_ls.append(mol.GetNumAtoms())
	return max(atoms_ls)


def preprocess_dataset(data, cluster=None):
	data_A, data_F = [], []
	data_desc = []

	max_atoms = get_max_atom(data)
	graph_gen = mol_graph(max_atoms)

	for smiles in data.smiles:
		A, F = graph_gen.get_graph(smiles)
		D = get_descriptors(smiles)
		data_A += [A.copy()[0]]
		data_F += [F.copy()[0]]
		data_desc += [D.copy()[0]]
	data_A = np.array(data_A)
	sh = data_A.shape
	if cluster is None:
		dataset = data_A.reshape((sh[0] * sh[1] * sh[2], sh[3]))
		dataset = dataset[~np.all(dataset == 0, axis=1)]
		cluster = np.unique(dataset, axis=0)
		print(cluster, len(cluster))
	new_data = np.zeros((sh[0], sh[1], sh[2]))
	for i, row in enumerate(cluster):
		new_data[np.where((data_A == row).all(axis=3))] = i + 1
	return new_data, np.array(data_F), np.array(data_desc), data.activity, cluster
