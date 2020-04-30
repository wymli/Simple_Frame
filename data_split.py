from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts

from tqdm import tqdm
import numpy as np
import pandas as pd

import os
import math
import json
import random
import argparse

from sklearn.model_selection import KFold, StratifiedKFold


def generate_scaffold(smiles, include_chirality=False):
	"""
	Compute the Bemis-Murcko scaffold for a SMILES string.
	:param smiles: A smiles string.
	:param include_chirality: Whether to include chirality.
	:return:
	"""
	mol = Chem.MolFromSmiles(smiles)
	scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
	return scaffold


def scaffold_to_smiles(mols, use_indices=False):
	"""
	Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.
	:param mols: A list of smiles strings or RDKit molecules.
	:param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
	to the smiles string itself. This is necessary if there are duplicate smiles.
	:return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
	"""
	scaffolds = defaultdict(set)
	for i, mol in tqdm(enumerate(mols), total=len(mols)):
		scaffold = generate_scaffold(mol)
		if use_indices:
			scaffolds[scaffold].add(i)
		else:
			scaffolds[scaffold].add(mol)
	return scaffolds

def scaffold_split(data, n_splits=5, balanced=True, seed=0):
	"""
	Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
	:param data: Pandas DataFrame.
	:param n_splits: Number of folds. Must be at least 2.
	:param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
	:param seed: Seed for shuffling when doing balanced splitting.
	:return: A dictionary containing N-fold splits of the data.
	"""
	fold_size = math.ceil(len(data) / n_splits)
	scaffold_to_indices = scaffold_to_smiles(list(data['SMILES']), use_indices=True)
	if balanced:
	# Put stuff that's bigger than half the val/test size into train, rest just order randomly
		index_sets = list(scaffold_to_indices.values())
		big_index_sets = []
		small_index_sets = []
		for index_set in index_sets:
			if len(index_set) > fold_size / 2:
				big_index_sets.append(index_set)
			else:
				small_index_sets.append(index_set)
		random.seed(seed)
		random.shuffle(big_index_sets)
		random.shuffle(small_index_sets)
		index_sets = big_index_sets + small_index_sets
	else:
	# Sort from largest to smallest scaffold sets
		index_sets = sorted(list(scaffold_to_indices.values()),
							key=lambda index_set: len(index_set),
							reverse=True)

	n_splits_data, fold_index = [], {}
	for k in range(n_splits):
		fold_index['fold_%d'%(k+1)] = []

	for index_set in index_sets:
		for k in range(n_splits):
			if len(fold_index['fold_%d'%(k+1)]) + len(index_set) <= fold_size:
				fold_index['fold_%d'%(k+1)] += index_set
				break
			if k == (n_splits - 1):
				fold_index['fold_%d'%(k+1)] += index_set

	for k in range(n_splits):
		test_index = fold_index['fold_%d'%(k+1)]
		train_index = [train_index + fold_index['fold_%d'%(i+1)] for i in range(n_splits) if i != k]
		n_splits_data.appen({'train':train_index, 'test':test_index})
	return n_splits_data


def stratified_kfold(data, target_name, n_splits=5, shuffle=True, seed=0):
	"""
	Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
	:param data: Pandas DataFrame.
	:param target_name: Name of label columns in the data.
	:param n_splits: Number of folds. Must be at least 2.
	:param seed: Seed for shuffling when doing balanced splitting.
	:return: A dictionary containing N-fold splits of the data.
	"""
	skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
	n_splits_data, fold_index = [], {}
	for idx, (train_index, test_index) in enumerate(skf.split(data['smiles'], data[target_name])):
		fold_index['train'] = list(train_index)
		fold_index['test'] = list(test_index)
		n_splits_data.append(fold_index)
	return n_splits_data


def kfold(data, n_splits=5, shuffle=True, seed=0):
	"""
	Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
	:param data: Pandas DataFrame.
	:param n_splits: Number of folds. Must be at least 2.
	:param seed: Seed for shuffling when doing balanced splitting.
	:return: A dictionary containing N-fold splits of the data.
	"""
	kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
	n_splits_data, fold_index = [], {}
	for idx, (train_index, test_index) in enumerate(kf.split(data['smiles'])):
		fold_index['train'] = list(train_index)
		fold_index['test'] = list(test_index)
		n_splits_data.append(fold_index)
	return n_splits_data


def write_to_json(n_splits_data, split_path):
	with open(split_path, 'w') as fp:
		json.dump(n_splits_data, fp)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_path', type=str, required=True,
		default='', help='')
	parser.add_argument('--split_path', type=str, required=True,
		default='', help='')

	parser.add_argument('--kfold', action='store_true', default=True, help='')
	parser.add_argument('--skfold', action='store_true', default=False, help='')
	parser.add_argument('--scaffold', action='store_true', default=False, help='')
	parser.add_argument('--folds_num', str=int, default=5, help='')

	config = parser.parse_args()

	data_df = pd.read_csv(config.dataset_path)

	target_name_list = data_df.columns.tolist()
	target_name_list.remove('smiles')

	if config.kfold:
		output_path = os.path.join(config.split_path, 'KFold')
		n_splits_data = kfold(data_df, n_splits=config.folds_num)
		write_to_json(n_splits_data, output_path)
	elif config.skfold:
		output_path = os.path.join(config.split_path, 'Stratified_KFold')
		for target_name in target_name_list:
			n_splits_data = stratified_kfold(data_df, target_name, n_splits=config.folds_num)
			target_name_output_path = os.path.join(output_path, target_name)
			write_to_json(n_splits_data, target_name_output_path)
	elif config.scaffold:
		output_path = os.path.join(config.split_path, 'Scaffold')
		n_splits_data = scaffold_split(data_df, n_splits=config.folds_num)
		write_to_json(n_splits_data, output_path)
	else:
		print('Warning!')

	print('Done.')