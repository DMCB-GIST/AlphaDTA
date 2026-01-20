#!/usr/bin/env python3
"""
AF3 CIF → IGN Graph Preprocessing Automation Pipeline

Overall Process:
1. CIF → PDB conversion
2. PDB → protein.pdb + ligand.mol2 separation
3. MOL2 → SDF conversion (with pdbid folder structure)
4. Execute select_residues.py → generate ign_input
5. Graph generation
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import multiprocessing
from typing import Optional, List, Dict
import pymol2
from IGN_codes.graph_constructor import GraphDatasetV2MulPro, collate_fn_v2_MulPro
from IGN_codes.utils import set_random_seed


class AF3PreprocessingPipeline:
    """Complete pipeline to convert AF3 CIF files to IGN graphs"""
    
    def __init__(self, dataset_dir: str, label_csv: Optional[str] = None, 
                 num_process: int = 12, verbose: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.label_csv = label_csv
        self.num_process = num_process
        self.verbose = verbose
        
        # Path configuration
        self.af_output_dir = self.dataset_dir / "af_output"
        self.processed_dir = self.dataset_dir / "processed_structure"
        
        # Intermediate and final output directories
        self.temp_pdb_dir = self.processed_dir / "temp_pdb"
        self.protein_dir = self.processed_dir / "protein"
        self.mol2_dir = self.processed_dir / "mol2"
        self.sdf_dir = self.processed_dir / "sdf"
        self.ign_input_dir = self.processed_dir / "ign_input"
        self.graph_ls_dir = self.processed_dir / "graph_ls"
        self.graph_dic_dir = self.processed_dir / "graph_dic"
        
        # Create directories
        for dir_path in [self.temp_pdb_dir, self.protein_dir, self.mol2_dir, 
                         self.sdf_dir, self.ign_input_dir, self.graph_ls_dir, 
                         self.graph_dic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load label information
        self.labels = self._load_labels() if label_csv else None
        
        # Get PDB ID list
        self.pdb_ids = self._get_pdb_ids()
        
    def _load_labels(self) -> Dict[str, float]:
        """Load label CSV"""
        if not os.path.exists(self.label_csv):
            print(f"  Warning: Label CSV not found: {self.label_csv}")
            return None

        df = pd.read_csv(self.label_csv)

        if 'pdbid' not in df.columns or 'pK' not in df.columns:
            print(f"  Warning: CSV must have 'pdbid' and 'pK' columns")
            return None

        return dict(zip(df['pdbid'], df['pK']))
        
    def _get_pdb_ids(self) -> List[str]:
        """Extract PDB ID list from af_output directory"""
        if not self.af_output_dir.exists():
            raise FileNotFoundError(f"af_output directory not found: {self.af_output_dir}")
        
        pdb_ids = []
        for item in self.af_output_dir.iterdir():
            if item.is_dir():
                cif_file = item / f"{item.name}_model.cif"
                if cif_file.exists():
                    pdb_ids.append(item.name)
        
        return sorted(pdb_ids)
    
    def run(self):
        """Execute the entire pipeline"""
        print("=" * 80)
        print("AF3 CIF → IGN Graph Preprocessing Pipeline")
        print("=" * 80)
        print(f"Dataset: {self.dataset_dir}")
        print(f"Found {len(self.pdb_ids)} PDB IDs")
        print(f"Labels: {'Loaded' if self.labels else 'Not provided'}")
        print("=" * 80)
        
        # Step 1: CIF → PDB conversion
        print("\n[Step 1/5] CIF → PDB Conversion")
        self._step1_cif_to_pdb()
        
        # Step 2: PDB → Protein + Ligand separation
        print("\n[Step 2/5] PDB → Protein + Ligand (MOL2) Separation")
        self._step2_split_protein_ligand()
        
        # Step 3: MOL2 → SDF conversion
        print("\n[Step 3/5] MOL2 → SDF Conversion")
        self._step3_mol2_to_sdf()
        
        # Step 4: Execute select_residues.py
        print("\n[Step 4/5] IGN Input Generation (select_residues.py)")
        self._step4_create_ign_input()
        
        # Step 5: Graph generation
        print("\n[Step 5/5] Graph Generation")
        self._step5_create_graphs()
        
        print("\n" + "=" * 80)
        print("Pipeline Completed Successfully!")
        print("=" * 80)
        print(f"Output Location: {self.processed_dir}")
        print(f"  - Protein: {self.protein_dir}")
        print(f"  - MOL2: {self.mol2_dir}")
        print(f"  - SDF: {self.sdf_dir}")
        print(f"  - IGN Input: {self.ign_input_dir}")
        print(f"  - Graphs: {self.graph_ls_dir}, {self.graph_dic_dir}")
        print("=" * 80)
    
    def _step1_cif_to_pdb(self):
        """Step 1: CIF → PDB conversion"""
        success_count = 0
        
        for i, pdbid in enumerate(self.pdb_ids, 1):
            cif_file = self.af_output_dir / pdbid / f"{pdbid}_model.cif"
            pdb_file = self.temp_pdb_dir / f"{pdbid}.pdb"
            
            if pdb_file.exists():
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: Already exists")
                success_count += 1
                continue
            
            try:
                with pymol2.PyMOL() as pymol:
                    pymol.cmd.load(str(cif_file), 'structure')
                    pymol.cmd.save(str(pdb_file), selection='structure')
                
                success_count += 1
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✓")
                elif i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.pdb_ids)}")
            
            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✗ {e}")
        
        print(f"✓ Completed: {success_count}/{len(self.pdb_ids)}")
    
    def _step2_split_protein_ligand(self):
        """Step 2: PDB → Protein + Ligand separation"""
        success_count = 0
        
        for i, pdbid in enumerate(self.pdb_ids, 1):
            pdb_file = self.temp_pdb_dir / f"{pdbid}.pdb"
            protein_file = self.protein_dir / f"{pdbid}_protein.pdb"
            ligand_pdb = self.processed_dir / f"{pdbid}_ligand_temp.pdb"
            ligand_sdf = self.processed_dir / f"{pdbid}_ligand_temp.sdf"
            ligand_mol2 = self.mol2_dir / f"{pdbid}_ligand.mol2"
            
            if protein_file.exists() and ligand_mol2.exists():
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: Already exists")
                success_count += 1
                continue
            
            try:
                # Extract protein
                self._extract_protein(pdb_file, protein_file)
                
                # Extract and convert ligand: PDB → SDF → MOL2
                self._extract_ligand_pdb(pdb_file, ligand_pdb)
                self._convert_pdb_to_sdf(ligand_pdb, ligand_sdf)
                self._convert_sdf_to_mol2(ligand_sdf, ligand_mol2)
                
                # Delete temporary files
                if ligand_pdb.exists():
                    ligand_pdb.unlink()
                if ligand_sdf.exists():
                    ligand_sdf.unlink()
                
                success_count += 1
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✓")
                elif i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.pdb_ids)}")
            
            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✗ {e}")
        
        print(f"✓ Completed: {success_count}/{len(self.pdb_ids)}")
    
    def _extract_protein(self, pdb_file: Path, output_file: Path):
        """Extract only protein from PDB"""
        protein_lines = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    protein_lines.append(line)
        
        if not protein_lines:
            raise ValueError("No protein atoms found")
        
        with open(output_file, 'w') as f:
            for line in protein_lines:
                f.write(line)
            f.write("END\n")
    
    def _extract_ligand_pdb(self, pdb_file: Path, output_file: Path):
        """Extract only ligand from PDB"""
        ligand_lines = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('HETATM') or line.startswith('CONECT'):
                    ligand_lines.append(line)
        
        if not ligand_lines:
            raise ValueError("No ligand atoms found")
        
        with open(output_file, 'w') as f:
            for line in ligand_lines:
                f.write(line)
            f.write("END\n")
    
    def _convert_pdb_to_sdf(self, pdb_file: Path, sdf_file: Path):
        """OpenBabel: PDB → SDF"""
        cmd = ['obabel', str(pdb_file), '-O', str(sdf_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 or not sdf_file.exists():
            raise RuntimeError(f"OpenBabel PDB→SDF failed: {result.stderr[:100]}")
    
    def _convert_sdf_to_mol2(self, sdf_file: Path, mol2_file: Path):
        """OpenBabel: SDF → MOL2"""
        cmd = ['babel', '-isdf', str(sdf_file), '-omol2', '-O', str(mol2_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 or not mol2_file.exists():
            raise RuntimeError(f"OpenBabel SDF→MOL2 failed: {result.stderr[:100]}")
    
    def _step3_mol2_to_sdf(self):
        """Step 3: MOL2 → SDF conversion (Improved: save directly to pdbid folder)"""
        mol2_files = [(self.mol2_dir / f"{pdbid}_ligand.mol2", pdbid) 
                      for pdbid in self.pdb_ids]
        
        # Parallel processing
        with multiprocessing.Pool(self.num_process) as pool:
            results = pool.starmap(self._convert_single_mol2_to_sdf, mol2_files)
        
        success_count = sum(results)
        print(f"✓ Completed: {success_count}/{len(self.pdb_ids)}")
    
    def _convert_single_mol2_to_sdf(self, mol2_file: Path, pdbid: str) -> bool:
        """Single MOL2 → SDF conversion (with pdbid folder structure)"""
        try:
            # Create pdbid folder
            pdbid_dir = self.sdf_dir / pdbid
            pdbid_dir.mkdir(exist_ok=True)
            
            # SDF file path
            sdf_file = pdbid_dir / f"{pdbid}_ligand.sdf"
            
            if sdf_file.exists():
                return True
            
            # MOL2 → SDF conversion
            cmd = f'babel -imol2 {mol2_file} -osdf {sdf_file} -h'
            result = os.system(cmd)
            
            return result == 0 and sdf_file.exists()
        
        except Exception as e:
            print(f"  ✗ {pdbid}: {e}")
            return False
    
    def _step4_create_ign_input(self):
        """Step 4: Execute select_residues.py"""
        select_residues_script = "preprocess/IGN_codes/select_residues.py"
        
        if not os.path.exists(select_residues_script):
            raise FileNotFoundError(f"select_residues.py not found: {select_residues_script}")
        
        success_count = 0
        
        for i, pdbid in enumerate(self.pdb_ids, 1):
            protein_file = self.protein_dir / f"{pdbid}_protein.pdb"
            sdf_path = self.sdf_dir / pdbid
            
            # Check if already exists
            output_dir = self.ign_input_dir / f"{pdbid}_ligand"
            if output_dir.exists():
                # Rename to remove _ligand suffix
                final_dir = self.ign_input_dir / pdbid
                if not final_dir.exists():
                    os.rename(output_dir, final_dir)
                
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: Already exists")
                success_count += 1
                continue
            
            try:
                command = [
                    "python3", select_residues_script,
                    "--proteinfile", str(protein_file),
                    "--sdfpath", str(sdf_path),
                    "--finalpath", str(self.ign_input_dir),
                    "--num_process", str(self.num_process)
                ]
                
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                
                # Remove _ligand suffix
                output_dir = self.ign_input_dir / f"{pdbid}_ligand"
                final_dir = self.ign_input_dir / pdbid
                if output_dir.exists() and not final_dir.exists():
                    os.rename(output_dir, final_dir)
                
                success_count += 1
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✓")
                elif i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.pdb_ids)}")
            
            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: ✗ {e}")
        
        print(f"✓ Completed: {success_count}/{len(self.pdb_ids)}")
    
    def _step5_create_graphs(self):
        """Step 5: Graph generation (without training)"""
        print("  Generating graphs...")
        
        # Prepare PDB IDs and labels
        keys = []
        labels = []
        data_dirs = []
        
        for pdbid in self.pdb_ids:
            ign_dir = self.ign_input_dir / pdbid
            if ign_dir.exists():
                keys.append(pdbid)
                # Use label if available, otherwise use 0.0
                label = self.labels.get(pdbid, 0.0) if self.labels else 0.0
                labels.append(label)
                data_dirs.append(str(ign_dir))
        
        if not keys:
            print("    Warning: No valid IGN input directories found")
            return
        
        print(f"  Valid data: {len(keys)} samples")
        
        # Create graphs (using GraphDatasetV2MulPro)
        try:
            dataset = GraphDatasetV2MulPro(
                keys=keys,
                labels=labels,
                data_dirs=data_dirs,
                graph_ls_path=str(self.graph_ls_dir),
                graph_dic_path=str(self.graph_dic_dir),
                dis_threshold=12.0,
                num_process=self.num_process,
                path_marker='/'
            )
            
            print(f"✓ Completed: {len(dataset)} graphs generated")
            print(f"  - graph_ls: {self.graph_ls_dir}")
            print(f"  - graph_dic: {self.graph_dic_dir}")
        
        except Exception as e:
            print(f"  ✗ Graph generation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='AF3 CIF → IGN Graph Preprocessing Automation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory (containing af_output)')
    parser.add_argument('--label_csv', type=str, default=None,
                        help='Label CSV file path (optional)')
    parser.add_argument('--num_process', type=int, default=12,
                        help='Number of parallel processes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Execute pipeline
    pipeline = AF3PreprocessingPipeline(
        dataset_dir=args.dataset_dir,
        label_csv=args.label_csv,
        num_process=args.num_process,
        verbose=args.verbose
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()