import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import yaml
import argparse
import time
import math
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.AlphaDTA import create_AlphaDTA
from utils.dataset import AffinityDataset, collate_fn, load_cv_split
from utils.util import (
    set_random_seed, 
    compute_metrics,
    pad_pairformer_embeddings_to_device,
    WarmupScheduler
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config, args):
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.batch_size is not None: 
        config['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.patience is not None:
        config['patience'] = args.patience
    if args.seed is not None:
        config['seed'] = args.seed
    
    return config


class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', buffering=1, encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_rmse = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0
    
    def __call__(self, val_rmse, model, epoch):
        if val_rmse < self.best_val_rmse - self.min_delta:
            self.best_val_rmse = val_rmse
            self.best_state_dict = deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            return False, True  # (should_stop, is_best)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
            return False, False


def train_one_epoch(model, dataloader, optimizer, criterion, device, config, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for step, batch in enumerate(dataloader):
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
        
        if step % config["accum_steps"] == 0:
            optimizer.zero_grad(set_to_none=True)
        
        predictions = model(bg, bg3, single_emb, pair_emb, token_lengths, protein_lengths)
        loss = criterion(predictions, targets) / config["accum_steps"]
        
        loss.backward()
        
        if ((step + 1) % config["accum_steps"] == 0) or ((step + 1) == len(dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        batch_size = targets.size(0)
        total_samples += batch_size
        
        with torch.no_grad():
            total_loss += criterion(predictions.detach(), targets).item() * batch_size
        
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mse, rmse, mae, r, rho = compute_metrics(all_preds, all_targets)
    else:
        mse, rmse, mae, r, rho = 0.0, 0.0, 0.0, 0.0, 0.0
    
    return avg_loss, rmse, r, rho


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
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
            loss = criterion(predictions, targets)
            
            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_pdbids.extend(list(pdbids))
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mse, rmse, mae, r, rho = compute_metrics(all_preds, all_targets)
    else:
        mse, rmse, mae, r, rho = 0.0, 0.0, 0.0, 0.0, 0.0
    
    return avg_loss, rmse, r, rho, all_preds, all_targets, all_pdbids


def train_fold(fold_idx, config, csv_path, embedding_dir, graph_bin_dir, 
               split_json_path, output_dir, device):
    """Train a single fold"""
    
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    log_path = os.path.join(fold_output_dir, 'training.log')
    logger = Logger(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"\n{'='*70}")
        print(f"Training Fold {fold_idx}")
        print(f"Log file: {log_path}")
        print(f"{'='*70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load split
        train_ids, val_ids = load_cv_split(split_json_path)
        print(f"Split loaded: Train={len(train_ids)}, Val={len(val_ids)}")
        
        # Create datasets
        print(f"\nCreating training dataset...")
        train_dataset = AffinityDataset(
            csv_path=csv_path,
            embedding_dir=embedding_dir,
            graph_bin_dir=graph_bin_dir,
            pdbid_list=train_ids
        )
        
        print(f"\nCreating validation dataset...")
        val_dataset = AffinityDataset(
            csv_path=csv_path,
            embedding_dir=embedding_dir,
            graph_bin_dir=graph_bin_dir,
            pdbid_list=val_ids
        )
        
        print(f"\nDataset sizes:  Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Create AlphaDTA model
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
                
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        fusion_info = model.get_fusion_info()
        print(f"\nModel:  AlphaDTA")
        print(f"Parameters: {total_params:,}")
        print(f"Fusion info: {fusion_info}")

        # Print config
        print(f"\nConfiguration:")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Weight Decay: {config['weight_decay']}")
        print(f"  Max Epochs: {config['max_epochs']}")
        print(f"  Early Stop Patience: {config['patience']}")
        print(f"  Warmup Epochs: {config.get('warmup_epochs', 3)}")
        print(f"  Init IGN Weight: {config.get('init_ign_weight', 0.5)}")
        
        # Optimizer and criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Scheduler
        updates_per_epoch = math.ceil(len(train_loader) / config["accum_steps"])
        if config.get("use_warmup", True):
            scheduler = WarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=config.get("warmup_epochs", 3),
                base_lr=config["learning_rate"],
                updates_per_epoch=updates_per_epoch
            )
        else:
            scheduler = None
        
        # Early stopping
        early_stopper = EarlyStopper(patience=config['patience'])
        
        # Training history
        train_loss_history = []
        val_loss_history = []
        train_rmse_history = []
        val_rmse_history = []
        train_r_history = []
        val_r_history = []
        epoch_list = []
        
        # Training loop
        print(f"\nStarting training...")
        print(f"Max epochs: {config['max_epochs']}, Early stop patience: {config['patience']}")
        print("-" * 110)
        print(f"{'Epoch': >6} | {'Train Loss':>10} {'Train RMSE':>11} {'Train R':>8} | "
              f"{'Val Loss': >10} {'Val RMSE':>9} {'Val R':>7} | {'LR':>10} | {'Time':>6} | Status")
        print("-" * 110)
        
        for epoch in range(1, config['max_epochs'] + 1):
            start_time = time.time()
            
            train_loss, train_rmse, train_r, train_rho = train_one_epoch(
                model, train_loader, optimizer, criterion, device, config, scheduler
            )
            
            val_loss, val_rmse, val_r, val_rho, _, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            epoch_list.append(epoch)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_rmse_history.append(train_rmse)
            val_rmse_history.append(val_rmse)
            train_r_history.append(train_r)
            val_r_history.append(val_r)
            
            # Early stopping check
            should_stop, is_best = early_stopper(val_rmse, model, epoch)
            
            if is_best:
                status = "✓ Best"
            else:
                status = f"  ({early_stopper.counter}/{config['patience']})"
            
            print(f"{epoch:>6} | {train_loss:>10.4f} {train_rmse:>11.4f} {train_r: >8.4f} | "
                  f"{val_loss:>10.4f} {val_rmse:>9.4f} {val_r: >7.4f} | "
                  f"{current_lr:>10.6f} | {epoch_time: >5.1f}s | {status}")
            
            if should_stop:
                print("-" * 110)
                print(f"\n✋ Early stopping at epoch {epoch}")
                break
        
        print("-" * 110)
        
        # Load best model
        if early_stopper.best_state_dict is not None:
            model.load_state_dict(early_stopper.best_state_dict)
        
        # Final evaluation
        val_loss, val_rmse, val_r, val_rho, val_preds, val_targets, val_pdbids = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx} Best Results (Epoch {early_stopper.best_epoch}):")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")
        print(f"  Pearson R: {val_r:.4f}")
        print(f"  Spearman Rho: {val_rho:.4f}")
        print(f"{'='*50}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save model (.pth)
        model_path = os.path.join(fold_output_dir, "best_model.pth")
        torch.save({
            'fold':  fold_idx,
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_epoch': early_stopper.best_epoch,
            'best_val_rmse': early_stopper.best_val_rmse,
            'final_metrics': {
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'pearson_r': val_r,
                'spearman_rho': val_rho
            }
        }, model_path)
        print(f"Model saved to {model_path}")
        
        # Save training history
        history = {
            'epochs': epoch_list,
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_rmse': train_rmse_history,
            'val_rmse': val_rmse_history,
            'train_r': train_r_history,
            'val_r': val_r_history,
            'best_epoch': early_stopper.best_epoch,
            'total_epochs': len(epoch_list)
        }
        with open(os.path.join(fold_output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save training curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(epoch_list, train_loss_history, label='Train Loss', alpha=0.8)
        axes[0].plot(epoch_list, val_loss_history, label='Val Loss', alpha=0.8)
        axes[0].axvline(x=early_stopper.best_epoch, color='red', linestyle='--', 
                        label=f'Best (epoch {early_stopper.best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Fold {fold_idx} - Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epoch_list, train_rmse_history, label='Train RMSE', alpha=0.8)
        axes[1].plot(epoch_list, val_rmse_history, label='Val RMSE', alpha=0.8)
        axes[1].axvline(x=early_stopper.best_epoch, color='red', linestyle='--', 
                        label=f'Best (epoch {early_stopper.best_epoch})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title(f'Fold {fold_idx} - RMSE Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(epoch_list, train_r_history, label='Train R', alpha=0.8)
        axes[2].plot(epoch_list, val_r_history, label='Val R', alpha=0.8)
        axes[2].axvline(x=early_stopper.best_epoch, color='red', linestyle='--', 
                        label=f'Best (epoch {early_stopper.best_epoch})')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Pearson R')
        axes[2].set_title(f'Fold {fold_idx} - Pearson R Curve')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_output_dir, 'training_curves.png'), dpi=150)
        plt.close()
        
        # Save predictions
        predictions_df = {
            'pdbid': val_pdbids,
            'true':  val_targets.numpy().flatten().tolist(),
            'pred': val_preds.numpy().flatten().tolist()
        }
        with open(os.path.join(fold_output_dir, 'val_predictions.json'), 'w') as f:
            json.dump(predictions_df, f, indent=2)
        
        fold_metrics = {
            'fold': fold_idx,
            'best_epoch': early_stopper.best_epoch,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'pearson_r': val_r,
            'spearman_rho': val_rho
        }
        
        return fold_metrics
    
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"Fold {fold_idx} log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description='AlphaDTA Cross-Validation Training (CleanSplit Protocol)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to training/validation CSV file')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='Directory containing CV split JSON files')
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Directory containing graph bin files')
    parser.add_argument(
        '--embedding_dir',
        nargs='+',
        required=True,
        help='One or more embedding directories (space-separated). '
            'Example: --embedding_dir /path/a /path/b'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/cleansplit',
        help='Directory to save trained models (default: ./output/cleansplit)'
    )
    
    # Optional CV settings
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (None = train all folds)')
    
    # Optional overrides
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs (overrides config)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config, default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:X or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override config with args
    config = override_config(config, args)
    
    # Default seed
    if 'seed' not in config: 
        config['seed'] = 42
    
    # Set random seed
    set_random_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cv_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Print configuration
    print(f"\n{'='*70}")
    print("AlphaDTA CleanSplit CV Training")
    print(f"{'='*70}")
    print(f"CSV Path: {args.csv_path}")
    print(f"Split Dir: {args.split_dir}")
    print(f"Graph Dir: {args.graph_dir}")
    print(f"Embedding Dir: {args.embedding_dir}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Max Epochs: {config['max_epochs']}")
    print(f"Early Stop Patience: {config['patience']}")
    print(f"Seed: {config['seed']}")
    print(f"Init IGN Weight: {config['init_ign_weight']}")
    print(f"{'='*70}")
    
    # Determine folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.num_folds))
    
    print(f"\nFolds to train: {folds_to_train}")
    
    # Cross-validation training
    all_fold_metrics = []
    
    for fold_idx in folds_to_train:
        split_json_path = os.path.join(
            args.split_dir, 
            f'PDBbind_cleansplit_train_val_split_f{fold_idx}.json'
        )
        
        if not os.path.exists(split_json_path):
            print(f" Split file not found: {split_json_path}, skipping fold {fold_idx}")
            continue
        
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            config=config,
            csv_path=args.csv_path,
            embedding_dir=args.embedding_dir,
            graph_bin_dir=args.graph_dir,
            split_json_path=split_json_path,
            output_dir=output_dir,
            device=device
        )
        
        all_fold_metrics.append(fold_metrics)
    
    # Print summary
    if len(all_fold_metrics) > 0:
        print(f"\n{'='*70}")
        print("Cross-Validation Summary")
        print(f"{'='*70}")
        
        print(f"\n{'Fold':<6} {'Best Epoch':<12} {'Val RMSE':<12} {'Pearson R':<12} {'Spearman Rho':<12}")
        print("-" * 60)
        
        for m in all_fold_metrics: 
            print(f"{m['fold']:<6} {m['best_epoch']:<12} {m['val_rmse']:<12.4f} {m['pearson_r']:<12.4f} {m['spearman_rho']:<12.4f}")
        
        print("-" * 60)
        
        # Calculate averages
        avg_rmse = np.mean([m['val_rmse'] for m in all_fold_metrics])
        std_rmse = np.std([m['val_rmse'] for m in all_fold_metrics])
        avg_r = np.mean([m['pearson_r'] for m in all_fold_metrics])
        std_r = np.std([m['pearson_r'] for m in all_fold_metrics])
        avg_rho = np.mean([m['spearman_rho'] for m in all_fold_metrics])
        std_rho = np.std([m['spearman_rho'] for m in all_fold_metrics])
        
        print(f"{'Mean':<6} {'':<12} {avg_rmse: <12.4f} {avg_r:<12.4f} {avg_rho:<12.4f}")
        print(f"{'Std':<6} {'':<12} {std_rmse:<12.4f} {std_r:<12.4f} {std_rho:<12.4f}")
        
        print(f"\n Final Results:")
        print(f"   RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"   Pearson R: {avg_r:.4f} ± {std_r:.4f}")
        print(f"   Spearman Rho: {avg_rho:.4f} ± {std_rho:.4f}")
        
        # Save summary
        summary = {
            'fold_metrics': all_fold_metrics,
            'mean_metrics': {
                'val_rmse': float(avg_rmse),
                'pearson_r': float(avg_r),
                'spearman_rho': float(avg_rho)
            },
            'std_metrics': {
                'val_rmse': float(std_rmse),
                'pearson_r': float(std_r),
                'spearman_rho': float(std_rho)
            },
            'config': config,
            'args': vars(args),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(output_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n Summary saved to {summary_path}")
    
    print("\n🎉 Training completed!")


if __name__ == '__main__':
    main()