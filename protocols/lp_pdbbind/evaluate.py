import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import json
from models.AlphaDTA import create_AlphaDTA
from models.AlphaDTA_baseline import create_AlphaDTA_baseline
from utils.dataset import AffinityDataset, collate_fn
from utils.util import (
    set_random_seed, 
    compute_metrics,
    pad_pairformer_embeddings_to_device)

    
TEST_DATASETS = {
    "crystal": [
        {
            "name": "LP-PDB",
            "csv_path":  "data/csv/lp_pdbbind/test.csv",
            "embedding_dirs": ["data/af3_embedding/shared", "data/af3_embedding/lp_only"],
            "graph_bin_dir": "data/interaction_graph/test/crystal/lp_test_graph_ls"
        },
        {
            "name": "BDB2020+",
            "csv_path":  "data/csv/lp_pdbbind/bdb2020+.csv",
            "embedding_dirs": ["data/af3_embedding/bdb2020+"],
            "graph_bin_dir": "data/interaction_graph/test/crystal/bdb2020+_graph_ls"
        },
        {
            "name": "EGFR",
            "csv_path": "data/csv/lp_pdbbind/egfr.csv",
            "embedding_dirs": ["data/af3_embedding/egfr"],
            "graph_bin_dir": "data/interaction_graph/test/crystal/egfr_graph_ls"
        },
        {
            "name": "Mpro",
            "csv_path": "data/csv/lp_pdbbind/mpro.csv",
            "embedding_dirs": ["data/af3_embedding/mpro"],
            "graph_bin_dir": "data/interaction_graph/test/crystal/mpro_graph_ls"
        },
    ],
    "af3": [
        {
            "name": "LP-PDB",
            "csv_path":  "data/csv/lp_pdbbind/test.csv",
            "embedding_dirs": ["data/af3_embedding/shared", "data/af3_embedding/lp_only"],
            "graph_bin_dir": "data/interaction_graph/test/af3/lp_test_graph_ls"
        },
        {
            "name": "BDB2020+",
            "csv_path":  "data/csv/lp_pdbbind/bdb2020+.csv",
            "embedding_dirs": ["data/af3_embedding/bdb2020+"],
            "graph_bin_dir": "data/interaction_graph/test/af3/bdb2020+_graph_ls"
        },
        {
            "name": "EGFR",
            "csv_path": "data/csv/lp_pdbbind/egfr.csv",
            "embedding_dirs": ["data/af3_embedding/egfr"],
            "graph_bin_dir": "data/interaction_graph/test/af3/egfr_graph_ls"
        },
        {
            "name": "Mpro",
            "csv_path": "data/csv/lp_pdbbind/mpro.csv",
            "embedding_dirs": ["data/af3_embedding/mpro"],
            "graph_bin_dir": "data/interaction_graph/test/af3/mpro_graph_ls"
        },
    ],
}


def evaluate_with_predictions(model, dataloader, device, criterion, dataset_name, output_dir):
    model.eval()
    all_preds = []
    all_targets = []
    all_pdbids = []
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            (pdbids, bg, bg3, single_list, pair_list, 
             targets, token_length, protein_length) = batch
            
            bg = bg.to(device, non_blocking=True)
            bg3 = bg3.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            token_length = token_length.to(device, non_blocking=True)
            protein_length = protein_length.to(device, non_blocking=True)
            
            single_emb, pair_emb = pad_pairformer_embeddings_to_device(
                single_list, pair_list, device
            )
            
            batch_preds = model(bg, bg3, single_emb, pair_emb, token_length, protein_length)
            
            loss = criterion(batch_preds, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(batch_preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_pdbids.extend(list(pdbids))
    
    if len(all_preds) > 0 and total_samples > 0:
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse, rmse, mae, r, rho = compute_metrics(predictions, targets)
        avg_loss = total_loss / total_samples
        
        predictions_np = predictions.numpy().flatten()
        targets_np = targets.numpy().flatten()
        abs_diff = np.abs(predictions_np - targets_np)
        
        df_pred = pd.DataFrame({
            'pdbid': all_pdbids,
            'pred':  predictions_np,
            'true': targets_np,
            'absolute_difference': abs_diff
        })
        
        csv_path = os.path.join(output_dir, f"predictions_{dataset_name}.csv")
        df_pred.to_csv(csv_path, index=False)
        print(f"   Predictions saved:  {csv_path}")
        
        return avg_loss, mse, rmse, mae, r, rho, len(predictions)
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0


def evaluate_all_datasets(model, device, criterion, config, output_dir, test_datasets_info):
    
    all_results = {}
    all_pearson_r_list = []
    key_pearson_r_list = []
    
    print("\n" + "="*80)
    print("Evaluating on Multiple Test Datasets")
    print("="*80 + "\n")
    
    for dataset_info in test_datasets_info:
        dataset_name = dataset_info["name"]
        try:
            print(f"Evaluating on {dataset_name}...")
            
            dataset = AffinityDataset(
                csv_path=dataset_info["csv_path"],
                embedding_dir=dataset_info["embedding_dirs"],
                graph_bin_dir=dataset_info["graph_bin_dir"]
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=64,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            loss, mse, rmse, mae, r, rho, sample_count = evaluate_with_predictions(
                model, dataloader, device, criterion, dataset_name, output_dir
            )
            
            all_results[dataset_name] = {
                'loss': loss, 
                'mse': mse, 
                'rmse': rmse, 
                'mae': mae, 
                'r': r, 
                'rho': rho,
                'sample_count': sample_count
            }
            all_pearson_r_list.append(r)
            
            if dataset_name in ['LP-PDB', 'BDB2020+']: 
                key_pearson_r_list.append(r)
            
            print(f"  Loss={loss:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            print(f"  Pearson R={r:.4f}, Spearman Rho={rho:.4f}")
            print(f"  Samples={sample_count}\n")
            
        except Exception as e:
            print(f"   Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            print()
            
            all_results[dataset_name] = {
                'loss': 0.0, 'mse': 0.0, 'rmse': 0.0, 
                'mae': 0.0, 'r': 0.0, 'rho': 0.0,
                'sample_count': 0
            }
            all_pearson_r_list.append(0.0)
            if dataset_name in ['LP-PDB', 'BDB2020+']:
                key_pearson_r_list.append(0.0)
    
    detailed_results = {
        'datasets': all_results
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(detailed_results, f, indent=4)
    print(f"✓ Detailed results saved:  {results_path}")

    summary_data = []
    for name, metrics in all_results.items():
        summary_data.append({
            'Dataset': name,
            'Samples': metrics['sample_count'],
            'Loss': f"{metrics['loss']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'Pearson R': f"{metrics['r']:.4f}",
            'Spearman Rho': f"{metrics['rho']:.4f}"
        })

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✓ Summary table saved: {summary_csv_path}\n")

    return all_results


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model_from_config(config, device):
    model_type = config.get("model_type", "alphadta")
    
    if model_type == "alphadta": 
        model = create_AlphaDTA(
            fusion_output_dim=config["ign_config"]["outdim_g3"] * 2,
            init_ign_weight=config.get("init_ign_weight", 0.5),
            fusion_dropout=config.get("fusion_dropout", 0.2),

            emb_encoder_single_dim=config["emb_encoder_config"]["single_in_dim"],
            emb_encoder_pair_dim=config["emb_encoder_config"]["pair_in_dim"],
            emb_encoder_hidden_dim=config["emb_encoder_config"]["hidden_dim"],
            emb_encoder_num_heads=config["emb_encoder_config"]["num_heads"],
            emb_encoder_num_protein_layers=config["emb_encoder_config"]["num_protein_layers"],
            emb_encoder_num_ligand_layers=config["emb_encoder_config"]["num_ligand_layers"],
            emb_encoder_ff_mult=4,
            emb_encoder_dropout=config["emb_encoder_config"]["dropout"],

            graph_node_feat_size=config["ign_config"]["node_feat_size"],
            graph_edge_feat_size=config["ign_config"]["edge_feat_size"],
            graph_hidden_dim=config["ign_config"]["graph_feat_size"],
            graph_num_layers=config["ign_config"]["num_layers"],
            graph_dropout=config.get("graph_dropout", 0.2),

            fc_hidden_dim=config.get("fc_hidden_dim", 128),
            fc_num_layers=config.get("fc_num_layers", 2)
        ).to(device)
        
    elif model_type == "baseline":  
        model = create_AlphaDTA_baseline(
            fusion_output_dim=config["ign_config"]["outdim_g3"] * 2,
            init_ign_weight=config.get("init_ign_weight", 0.6),
            fusion_dropout=config.get("fusion_dropout", 0.2),

            emb_encoder_single_dim=config["emb_encoder_config"]["single_in_dim"],
            emb_encoder_pair_dim=config["emb_encoder_config"]["pair_in_dim"],
            emb_encoder_hidden_dim=config["emb_encoder_config"]["hidden_dim"],
            emb_encoder_dropout=config["emb_encoder_config"]["dropout"],

            graph_node_feat_size=config["ign_config"]["node_feat_size"],
            graph_edge_feat_size=config["ign_config"]["edge_feat_size"],
            graph_hidden_dim=config["ign_config"]["graph_feat_size"],
            graph_num_layers=config["ign_config"]["num_layers"],
            graph_dropout=config.get("graph_dropout", 0.35),

            fc_hidden_dim=config.get("fc_hidden_dim", 128),
            fc_num_layers=config.get("fc_num_layers", 2)
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}.Choose 'alphadta' or 'baseline'")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Models (AlphaDTA or Baseline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
"""
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config.json (default: same directory as model)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: model_dir/evaluation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    set_random_seed(args.seed)
    print(f"\n🎲 Random seed set to: {args.seed}")

    if args.config_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(args.config_path):
        print(f"❌ Config file not found: {args.config_path}")
        print("Please specify --config_path or ensure config.json exists in the model directory.")
        return

    model_dir = os.path.dirname(os.path.abspath(args.model_path))

    if args.output_dir is None:
        args.output_dir = os.path.join(model_dir, "evaluation")
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Base output directory: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config_path}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    print("Loading configuration...")
    config = load_config(args.config_path)

    model_type = config.get("model_type", "alphadta")
    model_name = "AlphaDTA" if model_type == "alphadta" else "Baseline"

    print(f"  Model Type: {model_name}")
    print(f"  Init IGN Weight: {config.get('init_ign_weight', 0.5)}")
    print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"  Batch Size: {config.get('batch_size', 'N/A')}")

    emb_encoder_config = config.get("emb_encoder_config", {})
    if model_type == "alphadta":
        print(f"  Protein Layers: {emb_encoder_config.get('num_protein_layers', 'N/A')}")
        print(f"  Ligand Layers: {emb_encoder_config.get('num_ligand_layers', 'N/A')}")
    else:
        print(f"  Simple Hidden Dim: {emb_encoder_config.get('hidden_dim', 'N/A')}")
        print(f"  Simple Symmetrize: {emb_encoder_config.get('symmetrize', 'N/A')}")

    ign_config = config.get("ign_config", {})
    print(f"  IGN Layers: {ign_config.get('num_layers', 'N/A')}")
    print(f"  IGN Hidden Dim: {ign_config.get('graph_feat_size', 'N/A')}")
    print()

    # Create model from config
    print(f"Creating {model_name} from config...")
    model = create_model_from_config(config, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    fusion_info = model.get_fusion_info()
    print(f"  Fusion info:  {fusion_info}")

    # Load model weights
    print("\nLoading model weights...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("  ✅ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading model:  {e}")
        import traceback
        traceback.print_exc()
        return

    # Create loss criterion
    criterion = nn.MSELoss()
    print(f"\nLoss:  MSE")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base_eval_dir = args.output_dir

    for split_name, datasets in TEST_DATASETS.items():
        split_out_dir = os.path.join(base_eval_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        print("\n" + "#"*80)
        print(f"Running evaluation split: {split_name}")
        print(f"Saving to: {split_out_dir}")
        print("#"*80 + "\n")

        evaluate_all_datasets(
            model=model,
            device=device,
            criterion=criterion,
            config=config,
            output_dir=split_out_dir,
            test_datasets_info=datasets
        )

    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {base_eval_dir}")


if __name__ == "__main__":
    main()