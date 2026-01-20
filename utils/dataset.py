import os
import json
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import dgl
from typing import List, Union, Optional


class AffinityDataset(Dataset):
    """
    Unified AffinityDataset for both standard training and cross-validation.
    
    Data format:
    - Graph: g.bin, g3.bin, keys.bin (pickle format)
    - Embedding: {pdbid}.pt files (torch format)
    
    Args:
        csv_path: CSV file path with columns (pdbid, protein_length, ligand_length, total_length, pK)
        embedding_dir: Directory(ies) containing embedding .pt files (str or List[str])
        graph_bin_dir: Directory containing g.bin, g3.bin, keys.bin
        pdbid_list: Optional list of pdbids to use (for CV splits).If None, use all available data.
    """
    
    def __init__(self, 
                 csv_path: str, 
                 embedding_dir:  Union[str, List[str]], 
                 graph_bin_dir: str,
                 pdbid_list: Optional[List[str]] = None):
        
        self.df_initial = pd.read_csv(csv_path)
        
        if isinstance(embedding_dir, str):
            self.embedding_dirs = [embedding_dir]
        else:
            self.embedding_dirs = embedding_dir
        
        print(f"Loading graph bins from: {graph_bin_dir} using pickle.load()")
        with open(os.path.join(graph_bin_dir, 'g.bin'), 'rb') as f:
            self.graphs = pickle.load(f)
        with open(os.path.join(graph_bin_dir, 'g3.bin'), 'rb') as f:
            self.graphs3 = pickle.load(f)
        with open(os.path.join(graph_bin_dir, 'keys.bin'), 'rb') as f:
            self.keys = pickle.load(f)
        
        # Create key -> graph index mapping
        self.key_to_graph_idx = {key: i for i, key in enumerate(self.keys)}
        available_keys = set(self.keys)
        
        # Filter by pdbid_list if provided (for CV splits)
        initial_size = len(self.df_initial)
        
        if pdbid_list is not None:
            # CV mode: filter by pdbid_list
            pdbid_set = set(pdbid_list)
            self.df = self.df_initial[
                self.df_initial['pdbid'].isin(pdbid_set) & 
                self.df_initial['pdbid'].isin(available_keys)
            ].reset_index(drop=True)
            print(f" Dataset initialized (CV mode)")
            print(f"   Total graphs: {len(self.keys)}")
            print(f"   Requested:  {len(pdbid_list)}, Available: {len(self.df)}")
        else:
            # Standard mode: use all available data
            self.df = self.df_initial[
                self.df_initial['pdbid'].isin(available_keys)
            ].reset_index(drop=True)
            print(f" Dataset initialized (Standard mode)")
            print(f"   Total graphs:  {len(self.keys)}")
            print(f"   Dataset size: {len(self.df)}")
        
        final_size = len(self.df)
        
        if initial_size != final_size: 
            dropped = initial_size - final_size
            print(f"  Warning: {dropped} entries from CSV were dropped (graphs not found)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdbid = row['pdbid']
        
        # Load graph
        graph_idx = self.key_to_graph_idx[pdbid]
        g = self.graphs[graph_idx]
        g3 = self.graphs3[graph_idx]
        
        # Load labels and metadata
        binding_affinity = torch.tensor(row['pK'], dtype=torch.float32)
        total_length = torch.tensor(row['total_length'], dtype=torch.long)
        protein_length = torch.tensor(row['protein_length'], dtype=torch.long)
        
        # Load embedding (.pt file)
        single_embedding, pair_embedding = None, None
        for dir_path in self.embedding_dirs:
            embedding_file_path = os.path.join(dir_path, f"{pdbid}.pt")
            if os.path.exists(embedding_file_path):
                embeddings = torch.load(embedding_file_path, map_location='cpu')
                single_embedding = embeddings['single']
                pair_embedding = embeddings['pair']
                break
        
        if single_embedding is None: 
            raise FileNotFoundError(
                f"Graph for {pdbid} exists, but embedding file not found in {self.embedding_dirs}"
            )
        
        return (pdbid, g, g3, single_embedding, pair_embedding, binding_affinity,
                total_length, protein_length)



def collate_fn(batch):
    """
    Collate function for AffinityDataset.
    
    Returns:
        tuple: (pdbids, batched_graph, batched_graph3, single_embeddings_list, 
                pair_embeddings_list, affinities, token_lengths, protein_lengths)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    (pdbids, graphs, graphs3, single_embeddings, pair_embeddings,
     affinities, token_lengths, protein_lengths) = zip(*batch)

    # Batch DGL graphs
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)

    # Stack tensors
    affinities = torch.stack(affinities)
    token_lengths = torch.stack(token_lengths)
    protein_lengths = torch.stack(protein_lengths)

    # Keep embeddings as lists (padding happens on GPU during training)
    return (
        pdbids,
        bg,
        bg3,
        list(single_embeddings),
        list(pair_embeddings),
        affinities,
        token_lengths,
        protein_lengths,
    )


def load_cv_split(json_path: str):
    """
    Load cross-validation split from JSON file.
    
    Expected JSON format:
    {
        "train": ["pdbid1", "pdbid2", ...],
        "validation": ["pdbid3", "pdbid4", ...]
    }
    
    Args:
        json_path: Path to split JSON file
        
    Returns: 
        tuple: (train_ids, val_ids)
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    return split_data['train'], split_data['validation']


def create_cv_datasets(csv_path: str,
                       embedding_dir: Union[str, List[str]],
                       graph_bin_dir: str,
                       split_json_path: str):
    """
    Create train/val datasets using CV split JSON.
    
    Args:
        csv_path: Path to full data CSV
        embedding_dir:  Embedding directory path(s)
        graph_bin_dir: Graph bin directory path
        split_json_path: Path to CV split JSON file
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_ids, val_ids = load_cv_split(split_json_path)
    
    print(f"\nCreating CV datasets from split: {split_json_path}")
    print(f"  Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
    
    train_dataset = AffinityDataset(
        csv_path=csv_path,
        embedding_dir=embedding_dir,
        graph_bin_dir=graph_bin_dir,
        pdbid_list=train_ids
    )
    
    val_dataset = AffinityDataset(
        csv_path=csv_path,
        embedding_dir=embedding_dir,
        graph_bin_dir=graph_bin_dir,
        pdbid_list=val_ids
    )
    
    return train_dataset, val_dataset

