```python
#!/usr/bin/env python3
"""
AF3 CIF -> IGN Graph Preprocessing Pipeline

Pipeline steps:
1. CIF -> PDB conversion
2. PDB -> protein PDB + ligand MOL2 separation
3. MOL2 -> SDF conversion
4. IGN input generation with select_residues.py
5. Interaction graph generation

The --start_from option allows users to resume from intermediate stages.

Supported --start_from values:
- all:
    Run the full pipeline from dataset_dir/af_output.
- pdb:
    Start from processed_structure/temp_pdb and run steps 2-5.
- split:
    Start from processed_structure/protein and processed_structure/mol2 and run steps 3-5.
- mol2:
    Alias of split.
- sdf:
    Start from processed_structure/protein and processed_structure/sdf and run steps 4-5.
- ign_input:
    Start from processed_structure/ign_input and run graph generation only.
- graph:
    Alias of ign_input. This is the recommended mode when using released IGN inputs.

Example for released PDBbind2020 IGN inputs:

python preprocess/preprocess_structure.py \
    --dataset_dir /path/to/custom_dataset \
    --label_csv /path/to/custom_split.csv \
    --start_from graph \
    --num_process 12 \
    --verbose
"""

import argparse
import multiprocessing
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from IGN_codes.graph_constructor import GraphDatasetV2MulPro


class AF3PreprocessingPipeline:
    """Pipeline to convert AF3 outputs or released IGN inputs into IGN graphs."""

    VALID_START_FROM = {
        "all",
        "pdb",
        "split",
        "mol2",
        "sdf",
        "ign_input",
        "graph",
    }

    def __init__(
        self,
        dataset_dir: str,
        label_csv: Optional[str] = None,
        num_process: int = 12,
        verbose: bool = False,
        start_from: str = "all",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.label_csv = label_csv
        self.num_process = num_process
        self.verbose = verbose
        self.start_from = start_from

        if self.start_from not in self.VALID_START_FROM:
            raise ValueError(
                f"Invalid start_from={self.start_from}. "
                f"Valid values are: {sorted(self.VALID_START_FROM)}"
            )

        # Path configuration
        self.af_output_dir = self.dataset_dir / "af_output"
        self.processed_dir = self.dataset_dir / "processed_structure"

        self.temp_pdb_dir = self.processed_dir / "temp_pdb"
        self.protein_dir = self.processed_dir / "protein"
        self.mol2_dir = self.processed_dir / "mol2"
        self.sdf_dir = self.processed_dir / "sdf"
        self.ign_input_dir = self.processed_dir / "ign_input"
        self.graph_ls_dir = self.processed_dir / "graph_ls"
        self.graph_dic_dir = self.processed_dir / "graph_dic"

        # Create only the standard output directories.
        # Existing input directories are not overwritten.
        for dir_path in [
            self.temp_pdb_dir,
            self.protein_dir,
            self.mol2_dir,
            self.sdf_dir,
            self.ign_input_dir,
            self.graph_ls_dir,
            self.graph_dic_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.labels = self._load_labels() if self.label_csv else None
        self.pdb_ids = self._resolve_pdb_ids()

    @staticmethod
    def _normalize_pdbid(value: str) -> str:
        """Normalize PDB IDs for matching labels and file names."""
        return str(value).strip().lower()

    def _load_labels(self) -> Optional[Dict[str, float]]:
        """Load labels from a CSV file containing pdbid and pK columns."""
        if not self.label_csv:
            return None

        if not os.path.exists(self.label_csv):
            print(f"  Warning: Label CSV not found: {self.label_csv}")
            return None

        df = pd.read_csv(self.label_csv)

        if "pdbid" not in df.columns or "pK" not in df.columns:
            print("  Warning: label_csv must contain 'pdbid' and 'pK' columns")
            return None

        labels = {}
        for _, row in df.iterrows():
            pdbid = self._normalize_pdbid(row["pdbid"])
            labels[pdbid] = row["pK"]

        return labels

    def _resolve_pdb_ids(self) -> List[str]:
        """
        Determine the PDB ID list based on start_from.

        If label_csv is provided for graph-like modes, the CSV is used as the
        source of the requested split. Otherwise, available files/directories
        under the relevant processed_structure folder are used.
        """

        if self.start_from == "all":
            return self._get_pdb_ids_from_af_output()

        if self.start_from == "pdb":
            return self._get_pdb_ids_from_temp_pdb()

        if self.start_from in {"split", "mol2"}:
            if self.label_csv:
                return self._get_pdb_ids_from_csv()
            return self._get_pdb_ids_from_protein_mol2()

        if self.start_from == "sdf":
            if self.label_csv:
                return self._get_pdb_ids_from_csv()
            return self._get_pdb_ids_from_sdf()

        if self.start_from in {"ign_input", "graph"}:
            if self.label_csv:
                return self._get_pdb_ids_from_csv()
            return self._get_pdb_ids_from_ign_input()

        raise ValueError(f"Unsupported start_from: {self.start_from}")

    def _get_pdb_ids_from_csv(self) -> List[str]:
        """Read the PDB ID list from label_csv."""
        if not self.label_csv:
            raise FileNotFoundError("label_csv is required to read PDB IDs from CSV")

        if not os.path.exists(self.label_csv):
            raise FileNotFoundError(f"Label CSV not found: {self.label_csv}")

        df = pd.read_csv(self.label_csv)

        if "pdbid" not in df.columns:
            raise ValueError("label_csv must contain a 'pdbid' column")

        pdb_ids = [
            self._normalize_pdbid(x)
            for x in df["pdbid"].astype(str).tolist()
            if str(x).strip()
        ]

        if not pdb_ids:
            raise ValueError(f"No PDB IDs found in label_csv: {self.label_csv}")

        return sorted(set(pdb_ids))

    def _get_pdb_ids_from_af_output(self) -> List[str]:
        """Read PDB IDs from dataset_dir/af_output/*/{pdbid}_model.cif."""
        if not self.af_output_dir.exists():
            raise FileNotFoundError(f"af_output directory not found: {self.af_output_dir}")

        pdb_ids = []
        for item in self.af_output_dir.iterdir():
            if not item.is_dir():
                continue

            pdbid = self._normalize_pdbid(item.name)
            cif_file = item / f"{item.name}_model.cif"
            if cif_file.exists():
                pdb_ids.append(pdbid)

        if not pdb_ids:
            raise FileNotFoundError(f"No valid AF3 CIF files found in: {self.af_output_dir}")

        return sorted(set(pdb_ids))

    def _get_pdb_ids_from_temp_pdb(self) -> List[str]:
        """Read PDB IDs from processed_structure/temp_pdb/*.pdb."""
        if not self.temp_pdb_dir.exists():
            raise FileNotFoundError(f"temp_pdb directory not found: {self.temp_pdb_dir}")

        pdb_ids = [
            self._normalize_pdbid(p.stem)
            for p in self.temp_pdb_dir.glob("*.pdb")
            if p.is_file()
        ]

        if not pdb_ids:
            raise FileNotFoundError(f"No PDB files found in: {self.temp_pdb_dir}")

        return sorted(set(pdb_ids))

    def _get_pdb_ids_from_protein_mol2(self) -> List[str]:
        """
        Read PDB IDs from processed_structure/protein and processed_structure/mol2.

        Only IDs with both protein and ligand MOL2 files are returned.
        """
        if not self.protein_dir.exists():
            raise FileNotFoundError(f"protein directory not found: {self.protein_dir}")

        if not self.mol2_dir.exists():
            raise FileNotFoundError(f"mol2 directory not found: {self.mol2_dir}")

        protein_ids: Set[str] = set()
        for p in self.protein_dir.glob("*_protein.pdb"):
            pdbid = p.name[: -len("_protein.pdb")]
            protein_ids.add(self._normalize_pdbid(pdbid))

        mol2_ids: Set[str] = set()
        for p in self.mol2_dir.glob("*_ligand.mol2"):
            pdbid = p.name[: -len("_ligand.mol2")]
            mol2_ids.add(self._normalize_pdbid(pdbid))

        pdb_ids = sorted(protein_ids & mol2_ids)

        missing_mol2 = sorted(protein_ids - mol2_ids)
        missing_protein = sorted(mol2_ids - protein_ids)

        if missing_mol2:
            print(f"  Warning: {len(missing_mol2)} protein files have no matching MOL2 file")

        if missing_protein:
            print(f"  Warning: {len(missing_protein)} MOL2 files have no matching protein file")

        if not pdb_ids:
            raise FileNotFoundError(
                f"No matching protein/MOL2 pairs found in {self.protein_dir} and {self.mol2_dir}"
            )

        return pdb_ids

    def _get_pdb_ids_from_sdf(self) -> List[str]:
        """Read PDB IDs from processed_structure/sdf."""
        if not self.sdf_dir.exists():
            raise FileNotFoundError(f"sdf directory not found: {self.sdf_dir}")

        pdb_ids = set()

        for item in self.sdf_dir.iterdir():
            if item.is_dir():
                pdb_ids.add(self._normalize_pdbid(item.name))
            elif item.is_file():
                name = item.stem
                if name.lower().endswith("_ligand"):
                    name = name[: -len("_ligand")]
                pdb_ids.add(self._normalize_pdbid(name))

        if not pdb_ids:
            raise FileNotFoundError(f"No SDF entries found in: {self.sdf_dir}")

        return sorted(pdb_ids)

    def _normalize_ign_input_name(self, item: Path) -> str:
        """
        Convert an IGN input entry name to a PDB ID.

        Supported examples:
        - ign_input/1abc
        - ign_input/1abc_ligand
        - ign_input/1abc.pkl
        - ign_input/1abc_ligand.pkl
        """
        name = item.name

        if item.is_file() and item.suffix:
            name = item.stem

        if name.lower().endswith("_ligand"):
            name = name[: -len("_ligand")]

        return self._normalize_pdbid(name)

    def _get_pdb_ids_from_ign_input(self) -> List[str]:
        """Read PDB IDs from processed_structure/ign_input."""
        if not self.ign_input_dir.exists():
            raise FileNotFoundError(f"ign_input directory not found: {self.ign_input_dir}")

        pdb_ids = sorted(
            {
                self._normalize_ign_input_name(item)
                for item in self.ign_input_dir.iterdir()
                if item.is_file() or item.is_dir()
            }
        )

        if not pdb_ids:
            raise FileNotFoundError(f"No IGN input files found in: {self.ign_input_dir}")

        return pdb_ids

    def _steps_to_run(self) -> List[str]:
        """Return the list of pipeline steps that should be executed."""
        if self.start_from == "all":
            return ["cif_to_pdb", "split_protein_ligand", "mol2_to_sdf", "ign_input", "graph"]

        if self.start_from == "pdb":
            return ["split_protein_ligand", "mol2_to_sdf", "ign_input", "graph"]

        if self.start_from in {"split", "mol2"}:
            return ["mol2_to_sdf", "ign_input", "graph"]

        if self.start_from == "sdf":
            return ["ign_input", "graph"]

        if self.start_from in {"ign_input", "graph"}:
            return ["graph"]

        raise ValueError(f"Unsupported start_from: {self.start_from}")

    def run(self):
        """Run the pipeline from the requested stage."""
        steps = self._steps_to_run()

        print("=" * 80)
        print("AF3 / IGN Graph Preprocessing Pipeline")
        print("=" * 80)
        print(f"Dataset:     {self.dataset_dir}")
        print(f"start_from:  {self.start_from}")
        print(f"PDB IDs:     {len(self.pdb_ids)}")
        print(f"Labels:      {'Loaded' if self.labels else 'Not provided'}")
        print("=" * 80)

        if "cif_to_pdb" in steps:
            print("\n[Step 1/5] CIF -> PDB Conversion")
            self._step1_cif_to_pdb()

        if "split_protein_ligand" in steps:
            print("\n[Step 2/5] PDB -> Protein + Ligand MOL2 Separation")
            self._step2_split_protein_ligand()

        if "mol2_to_sdf" in steps:
            print("\n[Step 3/5] MOL2 -> SDF Conversion")
            self._step3_mol2_to_sdf()

        if "ign_input" in steps:
            print("\n[Step 4/5] IGN Input Generation")
            self._step4_create_ign_input()

        if "graph" in steps:
            print("\n[Step 5/5] Graph Generation")
            self._step5_create_graphs()

        print("\n" + "=" * 80)
        print("Pipeline Completed")
        print("=" * 80)
        print(f"Output Location: {self.processed_dir}")
        print(f"  - temp_pdb:  {self.temp_pdb_dir}")
        print(f"  - protein:   {self.protein_dir}")
        print(f"  - mol2:      {self.mol2_dir}")
        print(f"  - sdf:       {self.sdf_dir}")
        print(f"  - ign_input: {self.ign_input_dir}")
        print(f"  - graph_ls:  {self.graph_ls_dir}")
        print(f"  - graph_dic: {self.graph_dic_dir}")
        print("=" * 80)

    def _step1_cif_to_pdb(self):
        """Step 1: Convert AF3 CIF files to PDB files."""
        try:
            import pymol2
        except ImportError as e:
            raise ImportError(
                "pymol2 is required for CIF -> PDB conversion. "
                "Use --start_from graph if you already have IGN inputs."
            ) from e

        success_count = 0

        for i, pdbid in enumerate(self.pdb_ids, 1):
            cif_dir = self.af_output_dir / pdbid
            cif_file = cif_dir / f"{pdbid}_model.cif"

            if not cif_file.exists():
                # Fall back to the original directory name if case differs.
                candidates = list(cif_dir.glob("*_model.cif")) if cif_dir.exists() else []
                cif_file = candidates[0] if candidates else cif_file

            pdb_file = self.temp_pdb_dir / f"{pdbid}.pdb"

            if pdb_file.exists():
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: Already exists")
                success_count += 1
                continue

            try:
                with pymol2.PyMOL() as pymol:
                    pymol.cmd.load(str(cif_file), "structure")
                    pymol.cmd.save(str(pdb_file), selection="structure")

                success_count += 1
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: OK")
                elif i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.pdb_ids)}")

            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - {e}")

        print(f"Completed: {success_count}/{len(self.pdb_ids)}")

    def _step2_split_protein_ligand(self):
        """Step 2: Split PDB files into protein PDB and ligand MOL2 files."""
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

            if not pdb_file.exists():
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - missing PDB file {pdb_file}")
                continue

            try:
                self._extract_protein(pdb_file, protein_file)

                self._extract_ligand_pdb(pdb_file, ligand_pdb)
                self._convert_pdb_to_sdf(ligand_pdb, ligand_sdf)
                self._convert_sdf_to_mol2(ligand_sdf, ligand_mol2)

                if ligand_pdb.exists():
                    ligand_pdb.unlink()
                if ligand_sdf.exists():
                    ligand_sdf.unlink()

                success_count += 1
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: OK")
                elif i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.pdb_ids)}")

            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - {e}")

        print(f"Completed: {success_count}/{len(self.pdb_ids)}")

    def _extract_protein(self, pdb_file: Path, output_file: Path):
        """Extract protein ATOM records from a PDB file."""
        protein_lines = []

        with pdb_file.open("r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    protein_lines.append(line)

        if not protein_lines:
            raise ValueError("No protein ATOM records found")

        with output_file.open("w") as f:
            for line in protein_lines:
                f.write(line)
            f.write("END\n")

    def _extract_ligand_pdb(self, pdb_file: Path, output_file: Path):
        """Extract ligand HETATM and CONECT records from a PDB file."""
        ligand_lines = []

        with pdb_file.open("r") as f:
            for line in f:
                if line.startswith("HETATM") or line.startswith("CONECT"):
                    ligand_lines.append(line)

        if not ligand_lines:
            raise ValueError("No ligand HETATM/CONECT records found")

        with output_file.open("w") as f:
            for line in ligand_lines:
                f.write(line)
            f.write("END\n")

    def _convert_pdb_to_sdf(self, pdb_file: Path, sdf_file: Path):
        """Convert ligand PDB to SDF with OpenBabel."""
        cmd = ["obabel", str(pdb_file), "-O", str(sdf_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0 or not sdf_file.exists():
            raise RuntimeError(f"OpenBabel PDB -> SDF failed: {result.stderr[:300]}")

    def _convert_sdf_to_mol2(self, sdf_file: Path, mol2_file: Path):
        """Convert ligand SDF to MOL2 with OpenBabel."""
        cmd = ["babel", "-isdf", str(sdf_file), "-omol2", "-O", str(mol2_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0 or not mol2_file.exists():
            raise RuntimeError(f"OpenBabel SDF -> MOL2 failed: {result.stderr[:300]}")

    def _step3_mol2_to_sdf(self):
        """Step 3: Convert MOL2 files into SDF files under sdf/<pdbid>/."""
        mol2_files = [
            (self.mol2_dir / f"{pdbid}_ligand.mol2", pdbid)
            for pdbid in self.pdb_ids
        ]

        with multiprocessing.Pool(self.num_process) as pool:
            results = pool.starmap(self._convert_single_mol2_to_sdf, mol2_files)

        success_count = sum(results)
        print(f"Completed: {success_count}/{len(self.pdb_ids)}")

    def _convert_single_mol2_to_sdf(self, mol2_file: Path, pdbid: str) -> bool:
        """Convert a single MOL2 file to SDF."""
        try:
            if not mol2_file.exists():
                print(f"  {pdbid}: missing MOL2 file {mol2_file}")
                return False

            pdbid_dir = self.sdf_dir / pdbid
            pdbid_dir.mkdir(parents=True, exist_ok=True)

            sdf_file = pdbid_dir / f"{pdbid}_ligand.sdf"

            if sdf_file.exists():
                return True

            cmd = ["babel", "-imol2", str(mol2_file), "-osdf", str(sdf_file), "-h"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0 or not sdf_file.exists():
                print(f"  {pdbid}: MOL2 -> SDF failed: {result.stderr[:200]}")
                return False

            return True

        except Exception as e:
            print(f"  {pdbid}: FAILED - {e}")
            return False

    def _step4_create_ign_input(self):
        """Step 4: Generate IGN input files with select_residues.py."""
        select_residues_script = Path("preprocess/IGN_codes/select_residues.py")

        if not select_residues_script.exists():
            raise FileNotFoundError(
                f"select_residues.py not found: {select_residues_script}. "
                "Run this script from the AlphaDTA repository root."
            )

        success_count = 0

        for i, pdbid in enumerate(self.pdb_ids, 1):
            protein_file = self.protein_dir / f"{pdbid}_protein.pdb"
            sdf_path = self.sdf_dir / pdbid

            output_path = self.ign_input_dir / f"{pdbid}_ligand"
            final_path = self.ign_input_dir / pdbid

            if final_path.exists():
                if self.verbose:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: Already exists")
                success_count += 1
                continue

            if output_path.exists():
                if not final_path.exists():
                    os.rename(output_path, final_path)
                success_count += 1
                continue

            if not protein_file.exists():
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - missing protein file")
                continue

            if not sdf_path.exists():
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - missing SDF directory")
                continue

            try:
                command = [
                    "python3",
                    str(select_residues_script),
                    "--proteinfile",
                    str(protein_file),
                    "--sdfpath",
                    str(sdf_path),
                    "--finalpath",
                    str(self.ign_input_dir),
                    "--num_process",
                    str(self.num_process),
                ]

                subprocess.run(command, check=True, capture_output=True, text=True)

                output_path = self.ign_input_dir / f"{pdbid}_ligand"
                final_path = self.ign_input_dir / pdbid

                if output_path.exists() and not final_path.exists():
                    os.rename(output_path, final_path)

                if final_path.exists():
                    success_count += 1
                    if self.verbose:
                        print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: OK")
                    elif i % 10 == 0:
                        print(f"  Progress: {i}/{len(self.pdb_ids)}")
                else:
                    print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - output not found")

            except Exception as e:
                print(f"  [{i}/{len(self.pdb_ids)}] {pdbid}: FAILED - {e}")

        print(f"Completed: {success_count}/{len(self.pdb_ids)}")

    def _find_ign_input_path(self, pdbid: str) -> Optional[Path]:
        """
        Find the IGN input path for a PDB ID.

        The matched path can be either a file or a directory.
        """
        pdbid = self._normalize_pdbid(pdbid)
        pdbid_stem = Path(pdbid).stem

        candidate_names = {
            pdbid,
            f"{pdbid}_ligand",
            pdbid.upper(),
            f"{pdbid.upper()}_ligand",
            pdbid_stem,
            f"{pdbid_stem}_ligand",
            pdbid_stem.upper(),
            f"{pdbid_stem.upper()}_ligand",
        }

        for name in candidate_names:
            candidate = self.ign_input_dir / name
            if candidate.exists():
                return candidate

        if self.ign_input_dir.exists():
            target_names = {pdbid.lower(), pdbid_stem.lower()}

            for item in self.ign_input_dir.iterdir():
                if not (item.is_file() or item.is_dir()):
                    continue

                normalized = self._normalize_ign_input_name(item)
                raw_stem = item.stem.lower() if item.is_file() else item.name.lower()

                if normalized in target_names or raw_stem in target_names:
                    return item

        return None

    def _step5_create_graphs(self):
        """Step 5: Generate interaction graphs from IGN inputs."""
        print("  Generating graphs...")

        keys = []
        labels = []
        data_dirs = []
        missing_ign_inputs = []

        for pdbid in self.pdb_ids:
            pdbid = self._normalize_pdbid(pdbid)
            ign_input_path = self._find_ign_input_path(pdbid)

            if ign_input_path is None:
                missing_ign_inputs.append(pdbid)
                if self.verbose:
                    print(f"  Skipping {pdbid}: IGN input not found")
                continue

            keys.append(pdbid)

            label = 0.0
            if self.labels is not None:
                label = self.labels.get(pdbid, 0.0)

            labels.append(label)
            data_dirs.append(str(ign_input_path))

            if self.verbose:
                path_type = "dir" if ign_input_path.is_dir() else "file"
                print(f"  Found {pdbid}: {ign_input_path} ({path_type})")

        if missing_ign_inputs:
            print(
                f"  Warning: skipped {len(missing_ign_inputs)} samples "
                "because IGN input files were not found"
            )
            if self.verbose:
                for pdbid in missing_ign_inputs[:20]:
                    print(f"    missing: {pdbid}")
                if len(missing_ign_inputs) > 20:
                    print(f"    ... and {len(missing_ign_inputs) - 20} more")

        if not keys:
            print("  Warning: no valid IGN input files found")
            print(f"  Expected files or directories under: {self.ign_input_dir}")
            return

        print(f"  Valid data: {len(keys)} samples")

        try:
            dataset = GraphDatasetV2MulPro(
                keys=keys,
                labels=labels,
                data_dirs=data_dirs,
                graph_ls_path=str(self.graph_ls_dir),
                graph_dic_path=str(self.graph_dic_dir),
                dis_threshold=12.0,
                num_process=self.num_process,
                path_marker="/",
            )

            print(f"Completed: {len(dataset)} graphs generated")
            print(f"  graph_ls:  {self.graph_ls_dir}")
            print(f"  graph_dic: {self.graph_dic_dir}")

        except Exception as e:
            print(f"  Graph generation failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="AF3 / IGN graph preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help=(
            "Dataset directory. For --start_from graph, this should contain "
            "processed_structure/ign_input."
        ),
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default=None,
        help="Optional CSV file with pdbid and pK columns.",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=12,
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress messages.",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default="all",
        choices=["all", "pdb", "split", "mol2", "sdf", "ign_input", "graph"],
        help=(
            "Pipeline entry point. "
            "'all' starts from AF3 CIF files; "
            "'pdb' starts from processed_structure/temp_pdb; "
            "'split' or 'mol2' starts from processed_structure/protein and mol2; "
            "'sdf' starts from processed_structure/sdf; "
            "'ign_input' or 'graph' starts from processed_structure/ign_input."
        ),
    )

    args = parser.parse_args()

    pipeline = AF3PreprocessingPipeline(
        dataset_dir=args.dataset_dir,
        label_csv=args.label_csv,
        num_process=args.num_process,
        verbose=args.verbose,
        start_from=args.start_from,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
```
