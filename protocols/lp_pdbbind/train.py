import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import math
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from models.AlphaDTA import create_AlphaDTA
from models.AlphaDTA_baseline import create_AlphaDTA_baseline
from utils.dataset import AffinityDataset, collate_fn
from utils.util import (
    set_random_seed, 
    compute_metrics,
    pad_pairformer_embeddings_to_device,
    WarmupScheduler,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config, args):
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seed is not None:
        config['seed'] = args.seed
    if args.patience is not None:
        config['patience'] = args.patience
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    return config

def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    
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

            all_preds.append(batch_preds.detach())
            all_targets.append(targets.detach())
    
    if len(all_preds) > 0 and total_samples > 0:
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse, rmse, mae, r, rho = compute_metrics(predictions, targets)
        avg_loss = total_loss / total_samples
        return avg_loss, mse, rmse, mae, r, rho
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def evaluate_test_set(model, config, device, criterion, output_dir, log_file):
    print("\n" + "="*60)
    print("Evaluating on LP-PDBBind Test Set...")
    print("="*60)
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("Evaluating on LP-PDBBind Test Set...\n")
    log_file.write("="*60 + "\n")
    
    try:
        test_dataset = AffinityDataset(
            csv_path=os.path.join(config["csv_dir"], "test.csv"),
            embedding_dir=config["embedding_dirs"],
            graph_bin_dir=config["test_graph_bin_dir"]
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        model.eval()
        all_preds = []
        all_targets = []
        all_pdbids = []
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
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
        
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse, rmse, mae, r, rho = compute_metrics(predictions, targets)
        avg_loss = total_loss / total_samples
        
        predictions_np = predictions.numpy().flatten()
        targets_np = targets.numpy().flatten()
        abs_diff = np.abs(predictions_np - targets_np)
        
        df_pred = pd.DataFrame({
            'pdbid': all_pdbids,
            'pred': predictions_np,
            'true': targets_np,
            'absolute_difference': abs_diff
        })
        
        csv_path = os.path.join(output_dir, "test_predictions.csv")
        df_pred.to_csv(csv_path, index=False)
        
        result_msg = (
            f"\nLP-PDBBind Test Set Results:\n"
            f"  Samples: {len(test_dataset)}\n"
            f"  Loss: {avg_loss:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  MAE: {mae:.4f}\n"
            f"  Pearson R: {r:.4f}\n"
            f"  Spearman Rho: {rho:.4f}\n"
            f"  Predictions saved to: {csv_path}\n"
        )
        print(result_msg)
        log_file.write(result_msg)
        
        test_results = {
            'loss': float(avg_loss),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_r': float(r),
            'spearman_rho': float(rho),
            'sample_count': len(test_dataset)
        }
        
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)
        
        return test_results
        
    except Exception as e:
        error_msg = f"Error evaluating test set: {e}\n"
        print(error_msg)
        log_file.write(error_msg)
        import traceback
        traceback.print_exc()
        return None


def create_model(config, device):
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
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'alphadta' or 'baseline'")
    
    return model


def train_model(config, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_seed = config.get('seed', 42)
    set_random_seed(training_seed)
    torch.use_deterministic_algorithms(True)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    log_file = open(os.path.join(output_dir, "training.log"), "w", buffering=1)

    try:
        csv_dir = config["csv_dir"]
        embedding_dirs = config["embedding_dirs"]
        train_graph_bin_dir = config["train_graph_bin_dir"]
        valid_graph_bin_dir = config["valid_graph_bin_dir"]

        train_dataset = AffinityDataset(
            csv_path=os.path.join(csv_dir, "train.csv"),
            embedding_dir=embedding_dirs,
            graph_bin_dir=train_graph_bin_dir
        )
        valid_dataset = AffinityDataset(
            csv_path=os.path.join(csv_dir, "valid.csv"),
            embedding_dir=embedding_dirs,
            graph_bin_dir=valid_graph_bin_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )

        model = create_model(config, device)

        total_params = sum(p.numel() for p in model.parameters())
        fusion_info = model.get_fusion_info()
        
        model_type = config.get("model_type", "alphadta")
        model_name = "AlphaDTA" if model_type == "alphadta" else "Baseline"
        
        info_msg = (
            f"Model: {model_name}\n"
            f"Total parameters: {total_params:,}\n"
            f"Fusion info: {fusion_info}\n"
            f"Precision: float32 (no AMP)\n"
            f"Seed: {training_seed}\n"
            f"Patience: {config['patience']}\n\n"
        )
        print(info_msg)
        log_file.write(info_msg)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        criterion = nn.MSELoss()
        updates_per_epoch = math.ceil(len(train_loader) / config["accum_steps"])

        if config.get("use_warmup", True):
            warmup_epochs = config.get("warmup_epochs", 3)
            scheduler = WarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                base_lr=config["learning_rate"],
                updates_per_epoch=updates_per_epoch
            )
        else:
            scheduler = None

        train_loss_history, valid_loss_history = [], []
        train_r_history, valid_r_history = [], []
        train_rmse_history, valid_rmse_history = [], []

        # Early stopping
        patience = config["patience"]
        best_val_rmse = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state_dict = None

        max_epochs = config.get("max_epochs", 150)

        print(f"\n=== Training {model_name} ===")
        print(f"LR: {config['learning_rate']:.2e}, Seed: {training_seed}")
        print(f"Patience: {patience}, Max Epochs: {max_epochs}")
        print(f"Precision: float32 (no AMP)\n")

        log_file.write(
            f"\nTraining:  LR={config['learning_rate']:.2e}, Seed={training_seed}\n"
            f"Patience:  {patience}, Max Epochs: {max_epochs}\n\n"
        )

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            all_train_preds, all_train_targets = [], []
            epoch_loss, total_samples = 0.0, 0

            for global_step, batch in enumerate(train_loader):
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

                # Gradient accumulation
                if global_step % config["accum_steps"] == 0:
                    optimizer.zero_grad(set_to_none=True)

                batch_preds = model(
                    bg, bg3, single_emb, pair_emb, token_length, protein_length
                )
                loss = criterion(batch_preds, targets) / config["accum_steps"]
                loss.backward()

                if ((global_step + 1) % config["accum_steps"] == 0) or ((global_step + 1) == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                batch_size_val = targets.size(0)
                total_samples += batch_size_val

                with torch.no_grad():
                    epoch_loss += criterion(batch_preds.detach(), targets).item() * batch_size_val

                all_train_preds.append(batch_preds.detach())
                all_train_targets.append(targets.detach())

            # Train metrics
            train_predictions = torch.cat(all_train_preds, dim=0)
            train_targets = torch.cat(all_train_targets, dim=0)
            avg_train_loss = epoch_loss / total_samples
            train_mse, train_rmse, train_mae, train_r, train_rho = compute_metrics(
                train_predictions, train_targets
            )

            # Validation
            valid_loss, valid_mse, valid_rmse, valid_mae, valid_r, valid_rho = evaluate(
                model, valid_loader, device, criterion
            )

            # History
            train_loss_history.append(avg_train_loss)
            valid_loss_history.append(valid_loss)
            train_r_history.append(train_r)
            valid_r_history.append(valid_r)
            train_rmse_history.append(train_rmse)
            valid_rmse_history.append(valid_rmse)

            current_lr = optimizer.param_groups[0]["lr"]

            # Early stopping check
            if valid_rmse < best_val_rmse:
                best_val_rmse = valid_rmse
                best_epoch = epoch + 1
                best_state_dict = deepcopy(model.state_dict())
                patience_counter = 0
                status = "✓"
            else:
                patience_counter += 1
                status = f"{patience_counter}/{patience}"

            log_message = (
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train:  Loss={avg_train_loss:.4f}, RMSE={train_rmse:.4f}, R={train_r:.4f} | "
                f"Valid:  Loss={valid_loss:.4f}, RMSE={valid_rmse:.4f}, R={valid_r:.4f} | "
                f"LR:  {current_lr:.3e} | Status: {status}"
            )

            print(log_message)
            log_file.write(log_message + "\n")

            # Early stopping
            if patience_counter >= patience:
                stop_msg = f"\nEarly stopping at epoch {epoch + 1} (patience {patience})"
                print(stop_msg)
                log_file.write(stop_msg + "\n")
                break

        # Save best model
        if best_state_dict is not None:
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(best_state_dict, model_path)
            
            summary_msg = (
                f"\n{'='*60}\n"
                f"Training completed!\n"
                f"Best epoch: {best_epoch}\n"
                f"Best validation RMSE: {best_val_rmse:.4f}\n"
                f"Model saved:  {model_path}\n"
                f"{'='*60}\n"
            )
            print(summary_msg)
            log_file.write(summary_msg)
            
            model.load_state_dict(best_state_dict)
            test_results = evaluate_test_set(model, config, device, criterion, output_dir, log_file)

        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Loss curve
        ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history,
                 marker='o', markersize=3, alpha=0.7, label="Train Loss")
        ax1.plot(range(1, len(valid_loss_history) + 1), valid_loss_history,
                 marker='o', markersize=3, alpha=0.7, label="Valid Loss")
        ax1.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2,
                    label=f'Best (ep={best_epoch})')
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title("Loss Curve", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # RMSE curve
        ax2.plot(range(1, len(train_rmse_history) + 1), train_rmse_history,
                 marker='o', markersize=3, alpha=0.7, color='green', label="Train RMSE")
        ax2.plot(range(1, len(valid_rmse_history) + 1), valid_rmse_history,
                 marker='o', markersize=3, alpha=0.7, color='orange', label="Valid RMSE")
        ax2.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2,
                    label=f'Best (ep={best_epoch})')
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("RMSE", fontsize=11)
        ax2.set_title("RMSE Curve", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"{model_name} Training - LR={config['learning_rate']:.2e}, Seed={training_seed}",
            fontsize=13,
            y=0.99
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    finally:
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Train Models')
    
    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    
    # Optional
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config, default: 42)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (overrides config)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides default)')
    
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    config = override_config(config, args)
    
    if 'seed' not in config: 
        config['seed'] = 42
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Precision: float32 (no AMP)\n")
    
    if 'output_dir' not in config or config['output_dir'] is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.get('model_type', 'unknown')

        lr = config.get('learning_rate', None)
        bs = config.get('batch_size', None)
        seed = config.get('seed', None)

        lr_str = f"{lr:.0e}" if isinstance(lr, (int, float)) else "na"
        bs_str = str(bs) if bs is not None else "na"

        output_dir = f"./output/lp_pdbbind/{model_type}_lr{lr_str}_bs{bs_str}_seed{seed}"
    else:
        output_dir = config['output_dir']
    
    model_name = "AlphaDTA" if config.get('model_type') == "alphadta" else "Baseline (Simple)"
    
    print(f"{'='*80}")
    print(f"Model Training - {model_name}")
    print(f"{'='*80}")
    print(f"Model Type: {config.get('model_type')}")
    print(f"Learning Rate: {config['learning_rate']:.2e}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Seed: {config['seed']}")
    print(f"Patience: {config['patience']}")
    print(f"Max Epochs: {config['max_epochs']}")
    print(f"Init IGN Weight: {config.get('init_ign_weight', 0.5)}")
    print(f"Output:  {output_dir}\n")
    
    train_model(config, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Training Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()