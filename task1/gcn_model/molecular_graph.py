import numpy as np
from rdkit import Chem


class mol_graph:

	def __init__(self, max_atoms):
		self.graphs_ls = []
		self.n_bonds = 3  # single, double, triple
		self.n_feat = 15
		self.max_atoms = max_atoms
		self.bond_types = np.array(
			[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
		self.possible_hybridization = np.array(
			[Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
			 Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])

		self.A = np.zeros((1, self.max_atoms, self.max_atoms, self.n_bonds))
		self.F = np.zeros((1, self.max_atoms, self.n_feat))

	def get_node_feat(self, atom):
		atomid = atom.GetIdx()
		charge = atom.GetFormalCharge()
		ring = atom.IsInRing()
		degree = atom.GetDegree()
		valence = atom.GetImplicitValence()
		hybridization = (atom.GetHybridization() == self.possible_hybridization).tolist()
		ring_size = [not atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5), atom.IsInRingSize(6), (atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) and ( not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6)))]
		return (atomid, [charge, ring, degree, valence] + hybridization + ring_size)

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
			bond_type = (bond.GetBondType() == self.bond_types)
			self.A[0, beginidx, endidx] = self.A[0, endidx, beginidx] = bond_type

		return self.A, self.F

	def get_flatten_graph(self, smiles):
		A, F = self.get_graph(smiles)
		return np.append(A.flatten(), F.flatten(), axis=0)
