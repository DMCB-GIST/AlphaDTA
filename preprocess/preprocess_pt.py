import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from Bio.PDB import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X"
}

def get_heavy_atom_count(smiles: str) -> int:
    try:
        from rdkit import Chem
    except ImportError:
        logger.error("RDkit not installed.")
        return 0
    
    if not smiles:
        return 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    except Exception:
        return 0

def parse_structure_and_find_pocket(cif_path, pdb_id, threshold=10.0):
    if not os.path.exists(cif_path):
        return None, f"CIF file missing: {cif_path}"

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception as e:
        return None, f"CIF parsing error: {e}"

    ligand_atoms = []
    protein_ca_atoms = []

    for model in structure:
        for chain in model:
            if not chain: continue
            
            is_ligand_chain = True
            is_protein_chain = False
            
            for residue in chain:
                if residue.get_id()[0] == ' ': 
                    is_ligand_chain = False
                    is_protein_chain = True
                    break 
            
            if is_ligand_chain:
                for residue in chain:
                    if residue.get_resname() in ['HOH', 'WAT']: continue
                    for atom in residue:
                        if atom.element != 'H': 
                            ligand_atoms.append(atom)
            elif is_protein_chain:
                for residue in chain:
                    if 'CA' in residue:
                        protein_ca_atoms.append(residue['CA'])

    if not ligand_atoms:
        return None, "No ligand atoms found"
    if not protein_ca_atoms:
        return None, "No protein CA atoms found"

    # Neighbor Search
    ns = NeighborSearch(protein_ca_atoms)
    nearby_residues = set()

    for lig_atom in ligand_atoms:
        nearby_ca_atoms = ns.search(lig_atom.coord, threshold, level='A')
        for ca_atom in nearby_ca_atoms:
            residue = ca_atom.get_parent()
            # (ChainID, ResidueIndex_1based, ResName)
            nearby_residues.add((residue.get_parent().get_id(), residue.get_id()[1], residue.get_resname()))
            
    return nearby_residues, None

def process_single_pdb(pdb_id, dataset_name, base_af_input, base_af_output, output_pt_dir):    
    npz_path  = os.path.join(base_af_output, pdb_id, "seed-42_embeddings", "embeddings.npz")
    json_path = os.path.join(base_af_input, f"{pdb_id}.json")
    cif_path  = os.path.join(base_af_output, pdb_id, f"{pdb_id}_model.cif")
    
    if not os.path.exists(npz_path):
        return {"status": "SKIP", "reason": "NPZ not found"}
    if not os.path.exists(json_path):
        return {"status": "SKIP", "reason": "JSON not found"}

    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            
        output_name = json_data.get("name", pdb_id)
        
        protein_len_sum = 0
        ligand_len = 0
        chain_lengths = {} 
        chain_order = []   

        if 'sequences' in json_data:
            for item in json_data['sequences']:
                if 'protein' in item:
                    p_info = item['protein']
                    p_seq = p_info.get('sequence', '')
                    p_ids = p_info.get('id', [])
                    seq_len = len(p_seq)
                    
                    for cid in p_ids:
                        chain_lengths[cid] = seq_len
                        chain_order.append(cid)
                        protein_len_sum += seq_len
                
                elif 'ligand' in item:
                    l_smiles = item['ligand'].get('smiles', '')
                    ligand_len += get_heavy_atom_count(l_smiles)

        expected_total = protein_len_sum + ligand_len

    except Exception as e:
        return {"status": "ERROR", "reason": f"JSON parse error: {e}"}

    try:
        data = np.load(npz_path)
        single_np = data["single_embeddings"]
        pair_np = data["pair_embeddings"]
        
        trimmed_len = single_np.shape[0]
        
        if trimmed_len != expected_total:
            return {
                "status": "FAIL", 
                "reason": f"Length mismatch: NPZ({trimmed_len}) != JSON({expected_total})"
            }
            
    except Exception as e:
        return {"status": "ERROR", "reason": f"NPZ load failed: {e}"}

    nearby_res_set, err_msg = parse_structure_and_find_pocket(cif_path, pdb_id, threshold=10.0)
    
    if nearby_res_set is None:
        return {"status": "ERROR", "reason": f"Structure Analysis failed: {err_msg}"}

    chain_offsets = {}
    curr_offset = 0
    for cid in chain_order:
        chain_offsets[cid] = curr_offset
        curr_offset += chain_lengths[cid]

    indices_to_keep = set()
    
    for (chain_id, res_idx, res_name) in nearby_res_set:
        if chain_id not in chain_offsets:
            continue
        
        # 0-based token index
        token_idx = chain_offsets[chain_id] + res_idx - 1
        
        if token_idx < protein_len_sum:
            indices_to_keep.add(token_idx)
    
    ligand_indices = list(range(protein_len_sum, trimmed_len))
    final_indices = sorted(list(indices_to_keep) + ligand_indices)
    
    if len(final_indices) == 0:
        return {"status": "FAIL", "reason": "No tokens extracted (empty pocket?)"}

    try:
        single_t = torch.from_numpy(single_np).half()
        pair_t = torch.from_numpy(pair_np).half()
        
        idx_tensor = torch.tensor(final_indices, dtype=torch.long)
        
        pocket_single = single_t[idx_tensor, :]
        pocket_pair = pair_t[idx_tensor[:, None], idx_tensor, :]
        
        out_file = os.path.join(output_pt_dir, f"{output_name}.pt")
        torch.save({'single': pocket_single, 'pair': pocket_pair}, out_file)
        
    except Exception as e:
        return {"status": "ERROR", "reason": f"Tensor slicing/save error: {e}"}

    return {
        "status": "SUCCESS",
        "pdbid": pdb_id,           
        "output_name": output_name, 
        "original_protein_len": protein_len_sum,
        "ligand_length": ligand_len,
        "original_total_len": trimmed_len,
        "protein_length": len(indices_to_keep),
        "total_length": len(final_indices)
    }

def main():
    parser = argparse.ArgumentParser(description="AlphaFold3 Post-processing Pipeline (No Padding Trim)")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help='Dataset root dir'
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.rstrip("/") 
    dataset_name = os.path.basename(dataset_root) 

    af_input_root  = os.path.join(dataset_root, "af_input")
    af_output_root = os.path.join(dataset_root, "af_output")
    pt_save_dir    = os.path.join(dataset_root, "processed_emb")
    csv_save_dir   = os.path.join(dataset_root, "csv")

    os.makedirs(pt_save_dir, exist_ok=True)
    os.makedirs(csv_save_dir, exist_ok=True)

    target_dir = af_output_root
    if not os.path.exists(target_dir):
        logger.error(f"Input directory does not exist: {target_dir}")
        sys.exit(1)

    pdb_ids = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])

    logger.info(f"Dataset Root: {dataset_root}")
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Found {len(pdb_ids)} directories")
    logger.info(f"Saving .pt files to: {pt_save_dir}")
    logger.info(f"Saving CSV to: {csv_save_dir}")

    results = []
    for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
        res = process_single_pdb(
            pdb_id,
            dataset_name,       
            af_input_root,      
            af_output_root,    
            pt_save_dir       
        )

        if res["status"] == "SUCCESS":
            results.append(res)
        else:
            tqdm.write(f"[{res['status']}] {pdb_id}: {res['reason']}")

    if results:
        csv_path = os.path.join(csv_save_dir, f"{dataset_name}.csv")
        df = pd.DataFrame(results)
        cols = ["pdbid", "output_name", "original_protein_len", "ligand_length", "original_total_len",
                "protein_length", "total_length"]
        df = df[cols]
        df.to_csv(csv_path, index=False)
        logger.info(f"Processing Complete. CSV saved to {csv_path}")
        logger.info(f"Success count: {len(df)}")
    else:
        logger.warning("No PDBs were successfully processed.")


if __name__ == "__main__":
    main()