import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.AlphaDTA import create_AlphaDTA
from utils.dataset import AffinityDataset, collate_fn
from utils.util import (set_random_seed, 
                  compute_metrics, 
                  pad_pairformer_embeddings_to_device)


def load_model(model_path, config, device):
    """Load a trained AlphaDTA model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config
    
    # Create AlphaDTA model
    model = create_AlphaDTA(
        fusion_output_dim=model_config["ign_config"]["outdim_g3"] * 2,
        init_ign_weight=model_config.get("init_ign_weight", 0.5),
        fusion_dropout=model_config.get("fusion_dropout", 0.2),
        
        emb_encoder_single_dim=model_config["emb_encoder_config"]["single_in_dim"],
        emb_encoder_pair_dim=model_config["emb_encoder_config"]["pair_in_dim"],
        emb_encoder_hidden_dim=model_config["emb_encoder_config"]["hidden_dim"],
        emb_encoder_num_heads=model_config["emb_encoder_config"]["num_heads"],
        emb_encoder_num_protein_layers=model_config["emb_encoder_config"]["num_protein_layers"],
        emb_encoder_num_ligand_layers=model_config["emb_encoder_config"]["num_ligand_layers"],
        emb_encoder_ff_mult=4,
        emb_encoder_dropout=model_config["emb_encoder_config"]["dropout"],
        
        graph_node_feat_size=model_config["ign_config"]["node_feat_size"],
        graph_edge_feat_size=model_config["ign_config"]["edge_feat_size"],
        graph_hidden_dim=model_config["ign_config"]["graph_feat_size"],
        graph_num_layers=model_config["ign_config"]["num_layers"],
        graph_dropout=model_config.get("graph_dropout", 0.2),
        
        fc_hidden_dim=model_config.get("fc_hidden_dim", 128),
        fc_num_layers=model_config.get("fc_num_layers", 2)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_single_model(model, dataloader, device):
    """Get predictions from a single model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_pdbids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            (pdbids, bg, bg3, single_list, pair_list,
             targets, token_lengths, protein_lengths) = batch
            
            bg = bg.to(device, non_blocking=True)
            bg3 = bg3.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            token_lengths = token_lengths.to(device, non_blocking=True)
            protein_lengths = protein_lengths.to(device, non_blocking=True)
            
            single_emb, pair_emb = pad_pairformer_embeddings_to_device(
                single_list, pair_list, device
            )
            
            predictions = model(bg, bg3, single_emb, pair_emb, token_lengths, protein_lengths)
            
            all_preds.append(predictions.cpu())
            all_labels.append(targets.cpu())
            all_pdbids.extend(list(pdbids))
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_preds, all_labels, all_pdbids


def ensemble_predict(models, dataloader, device):
    """Get ensemble predictions from multiple models"""
    all_model_preds = []
    all_labels = None
    all_pdbids = None
    
    for i, model in enumerate(models):
        print(f"  Predicting with model {i+1}/{len(models)}...")
        preds, labels, pdbids = predict_single_model(model, dataloader, device)
        all_model_preds.append(preds)
        
        if all_labels is None:
            all_labels = labels
            all_pdbids = pdbids
    
    # Stack and average predictions
    stacked_preds = torch.stack(all_model_preds, dim=0)  # (num_models, num_samples)
    ensemble_preds = stacked_preds.mean(dim=0)  # (num_samples,)
    
    return ensemble_preds, all_labels, all_pdbids, all_model_preds


def plot_predictions(true_values, pred_values, title, save_path, axislim=14):
    """Plot predictions vs true values"""
    true_np = true_values.numpy().flatten() if torch.is_tensor(true_values) else np.array(true_values)
    pred_np = pred_values.numpy().flatten() if torch.is_tensor(pred_values) else np.array(pred_values)
    
    # Compute metrics
    mse, rmse, mae, r, rho = compute_metrics(
        torch.tensor(pred_np), torch.tensor(true_np)
    )
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_np, pred_np, alpha=0.5, c='blue', s=50)
    
    # Display metrics
    metrics_text = f"Pearson R = {r:.3f}\nRMSE = {rmse:.3f}\nSpearman ρ = {rho:.3f}"
    plt.text(0.05, 0.95, metrics_text, fontsize=14, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add diagonal line
    plt.plot([0, axislim], [0, axislim], color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('True pKa Values', fontsize=12)
    plt.ylabel('Predicted pKa Values', fontsize=12)
    plt.ylim(0, axislim)
    plt.xlim(0, axislim)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='AlphaDTA CASF2016 Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # Required
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CASF2016 CSV file')
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Directory containing graph bin files')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing embedding files')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained CV models (cv_run_XXXXXX)')
    
    # Optional
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: model_dir)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.abspath(args.model_dir), "evaluation")
    else:
        args.output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Load config from model directory
    config_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f" Config file not found at {config_path}\n"
            f"Please ensure the model directory contains 'config.json' from training."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f" Loaded config from {config_path}")
    print(f"   Model Type: {config.get('model_type', 'alphadta')}")
    print(f"   Init IGN Weight: {config.get('init_ign_weight', 0.5)}")
    
    # Create test dataset
    print(f"\nLoading CASF2016 test data...")
    print(f"  CSV: {args.csv_path}")
    print(f"  Graph: {args.graph_dir}")
    print(f"  Embedding: {args.embedding_dir}")
    
    test_dataset = AffinityDataset(
        csv_path=args.csv_path,
        embedding_dir=args.embedding_dir,
        graph_bin_dir=args.graph_dir,
        pdbid_list=None  # Use all samples
    )
    
    print(f"  Total samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Load models (.pth files)
    print(f"\nLoading AlphaDTA models from {args.model_dir}...")
    models = []
    fold_info = []
    
    for fold_idx in range(args.num_folds):
        # Try different possible paths
        possible_paths = [
            os.path.join(args.model_dir, f'fold_{fold_idx}', 'best_model.pth'),
            os.path.join(args.model_dir, f'fold_{fold_idx}', 'best_model.pt'),
            os.path.join(args.model_dir, f'fold{fold_idx}', 'best_model.pth'),
            os.path.join(args.model_dir, f'fold{fold_idx}', 'best_model.pt'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"   Model for fold {fold_idx} not found")
            continue
        
        print(f"  Loading fold {fold_idx}:  {model_path}")
        model, checkpoint = load_model(model_path, config, device)
        models.append(model)
        
        fold_info.append({
            'fold': fold_idx,
            'model_path': model_path,
            'best_epoch': checkpoint.get('best_epoch', 'N/A'),
            'best_val_rmse': checkpoint.get('best_val_rmse', 'N/A'),
            'final_metrics': checkpoint.get('final_metrics', {})
        })
    
    if len(models) == 0:
        raise RuntimeError(
            f" No models found in {args.model_dir}\n"
            f"Expected model paths:\n"
            f"  - fold_0/best_model.pth ~ fold_{args.num_folds-1}/best_model.pth"
        )
    
    print(f"\n Loaded {len(models)} AlphaDTA models for ensemble evaluation")
    
    # Get ensemble predictions
    print("\nEvaluating on CASF2016...")
    ensemble_preds, true_labels, pdbids, all_model_preds = ensemble_predict(
        models, test_loader, device
    )
    
    # Compute ensemble metrics
    ens_mse, ens_rmse, ens_mae, ens_r, ens_rho = compute_metrics(ensemble_preds, true_labels)
    
    # Compute per-model metrics
    per_model_metrics = []
    for i, model_preds in enumerate(all_model_preds):
        mse, rmse, mae, r, rho = compute_metrics(model_preds, true_labels)
        per_model_metrics.append({
            'fold': i,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_r': float(r),
            'spearman_rho': float(rho)
        })
    
    # Print results
    print("\n" + "="*70)
    print("CASF2016 Evaluation Results (AlphaDTA)")
    print("="*70)
    
    print("\nPer-Model Results:")
    print("-"*70)
    print(f"{'Fold':<8} {'Pearson R': >12} {'Spearman ρ':>12} {'RMSE':>12} {'MAE':>12}")
    print("-"*70)
    
    for metrics in per_model_metrics: 
        print(f"Fold {metrics['fold']:<4} {metrics['pearson_r']:>12.4f} {metrics['spearman_rho']:>12.4f} "
              f"{metrics['rmse']:>12.4f} {metrics['mae']:>12.4f}")
    
    print("-"*70)
    
    # Statistics across folds
    avg_r = np.mean([m['pearson_r'] for m in per_model_metrics])
    std_r = np.std([m['pearson_r'] for m in per_model_metrics])
    avg_rho = np.mean([m['spearman_rho'] for m in per_model_metrics])
    std_rho = np.std([m['spearman_rho'] for m in per_model_metrics])
    avg_rmse = np.mean([m['rmse'] for m in per_model_metrics])
    std_rmse = np.std([m['rmse'] for m in per_model_metrics])
    avg_mae = np.mean([m['mae'] for m in per_model_metrics])
    std_mae = np.std([m['mae'] for m in per_model_metrics])
    
    print(f"{'Mean':<8} {avg_r:>12.4f} {avg_rho: >12.4f} {avg_rmse:>12.4f} {avg_mae:>12.4f}")
    print(f"{'Std': <8} {std_r:>12.4f} {std_rho:>12.4f} {std_rmse:>12.4f} {std_mae:>12.4f}")
    print("-"*70)
    
    # Ensemble results
    print(f"\n{'='*70}")
    print(f"Ensemble Results ({len(models)} models)")
    print(f"{'='*70}")
    print(f"  Pearson R:      {ens_r:.4f}")
    print(f"  Spearman ρ:    {ens_rho:.4f}")
    print(f"  RMSE:          {ens_rmse:.4f}")
    print(f"  MAE:           {ens_mae:.4f}")
    print(f"{'='*70}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'pdbid': pdbids,
        'true_pKa': true_labels.numpy().flatten(),
        'pred_pKa_ensemble': ensemble_preds.numpy().flatten()
    })
    
    # Add per-model predictions
    for i, model_preds in enumerate(all_model_preds):
        predictions_df[f'pred_pKa_fold{i}'] = model_preds.numpy().flatten()
    
    # Add absolute error
    predictions_df['abs_error'] = np.abs(
        predictions_df['true_pKa'] - predictions_df['pred_pKa_ensemble']
    )
    
    predictions_path = os.path.join(args.output_dir, 'casf2016_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n Predictions saved to {predictions_path}")
    
    # Save detailed results
    results = {
        'model_type': 'AlphaDTA',
        'ensemble_metrics': {
            'mse': float(ens_mse),
            'rmse': float(ens_rmse),
            'mae': float(ens_mae),
            'pearson_r': float(ens_r),
            'spearman_rho': float(ens_rho)
        },
        'per_model_metrics': per_model_metrics,
        'fold_statistics': {
            'pearson_r': {'mean': float(avg_r), 'std': float(std_r)},
            'spearman_rho': {'mean':  float(avg_rho), 'std': float(std_rho)},
            'rmse': {'mean': float(avg_rmse), 'std': float(std_rmse)},
            'mae': {'mean': float(avg_mae), 'std': float(std_mae)}
        },
        'fold_info': fold_info,
        'num_models': len(models),
        'num_samples': len(pdbids),
        'config': config,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(args.output_dir, 'casf2016_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to {results_path}")
    
    # Plot predictions
    plot_path = os.path.join(args.output_dir, 'casf2016_predictions.png')
    plot_predictions(
        true_labels, ensemble_preds,
        f'CASF2016 Ensemble Prediction (AlphaDTA, {len(models)} folds)',
        plot_path
    )
    
    print("\n Evaluation completed!")


if __name__ == '__main__':
    main()