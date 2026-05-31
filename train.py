import os
import sys
import math
import argparse
import random
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from shared_config import get_dataset_config
from datasets.dataset_model_config import get_models_for_dataset, get_num_models

from cp_model import CoverageTransformer, MultiMetricMLP, MultiMetricDataset
from cp_model import LatentSpaceClusterCoverage, train_latent_space_clusters

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).astype(np.int32)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0,
                label_smoothing=0.0, criterion_drop_prob=0.0, gate_entropy_lambda=0.0):
    model.train()
    total_loss = 0.0
    total_ent_loss = 0.0
    all_preds = []
    all_labels = []

    for features, model_ids, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        model_ids = model_ids.to(device)
        labels = labels.to(device)

        if label_smoothing > 0:
            smoothed_labels = labels * (1 - label_smoothing) + label_smoothing / 2
        else:
            smoothed_labels = labels

        optimizer.zero_grad()
        logits = model(features, model_ids, criterion_drop_prob=criterion_drop_prob)
        loss = criterion(logits, smoothed_labels)

        if gate_entropy_lambda > 0 and hasattr(model, 'get_gate_weights'):
            gate_weights = model.get_gate_weights(features, model_ids)
            gate_probs = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=1).mean()
            max_entropy = math.log(gate_weights.shape[1])
            ent_loss = gate_entropy_lambda * (1.0 - entropy / max_entropy)
            loss = loss + ent_loss
            total_ent_loss += ent_loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return avg_loss, all_preds, all_labels

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, model_ids, labels in tqdm(dataloader, desc="Validation", leave=False):
            features = features.to(device)
            model_ids = model_ids.to(device)
            labels = labels.to(device)

            logits = model(features, model_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return avg_loss, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "MNIST", "Fashion-MNIST", "TinyImageNet"],
                        help="Dataset name (default: CIFAR10)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override coverage matrices dir (default: auto from dataset)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override checkpoint dir (default: auto from dataset)")
    parser.add_argument("--pca_dim", type=int, default=64, help="PCA output dimension")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=30, help="Number of clusters")
    parser.add_argument("--criterion_drop_prob", type=float, default=0.1,
                        help="Probability of dropping an entire criterion during training")
    parser.add_argument("--gate_entropy_lambda", type=float, default=0.02,
                        help="CAG gate entropy regularization coefficient")
    args = parser.parse_args()

    dataset_cfg = get_dataset_config(args.dataset)
    dataset_tag = dataset_cfg['weight_tag']
    num_classes = dataset_cfg['num_classes']

    if args.data_dir is None:
        if args.dataset == "CIFAR10":
            args.data_dir = "./coverage_matrices"
        else:
            args.data_dir = "./coverage_matrices"
    if args.output_dir is None:
        args.output_dir = f"./cp_checkpoints/{args.dataset}"

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} (tag={dataset_tag}, num_classes={num_classes})")

    criterion_names = ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']

    dataset_dir_map = {
        'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST',
        'Fashion-MNIST': 'Fashion-MNIST', 'TinyImageNet': 'TinyImageNet',
    }
    dataset_dir = dataset_dir_map.get(args.dataset, args.dataset)

    model_names = get_models_for_dataset(args.dataset)
    print(f"Models for {args.dataset}: {model_names}")
    data_infos = []

    pca_base_dir = os.path.join(args.data_dir, dataset_dir)

    for model_id, model_name in enumerate(model_names):
        candidate_dirs = [
            os.path.join(args.data_dir, dataset_dir, "multi_metric_coverage"),
            os.path.join(args.data_dir, "multi_metric_coverage"),
            os.path.join(args.data_dir, dataset_dir, f"activation_coverage_{model_name}"),
            os.path.join(args.data_dir, dataset_dir, f"activation_coverage_{model_name}_{dataset_tag}"),
            os.path.join(args.data_dir, f"activation_coverage_{model_name}_{dataset_tag}"),
            os.path.join(args.data_dir, f"activation_coverage_{model_name}"),
            args.data_dir,
        ]

        base_dir = args.data_dir
        for cdir in candidate_dirs:
            if os.path.exists(os.path.join(cdir, f"{model_name}_meta.npz")):
                base_dir = cdir
                break

        coverage_paths = {}
        for criterion in criterion_names:
            coverage_paths[criterion] = os.path.join(base_dir, f"{model_name}_{criterion}_coverage.npy")
        fault_path = os.path.join(base_dir, f"{model_name}_fault_flags.npy")

        meta_path = os.path.join(base_dir, f"{model_name}_meta.npz")
        num_samples = None
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            num_samples = int(meta["num_samples"])
            print(f"  {model_name}: meta.num_samples={num_samples}, "
                  f"filter_ratio={float(meta.get('filter_ratio', 0)):.4f}")

        data_infos.append({
            "model_name": model_name,
            "model_id": model_id,
            "coverage_paths": coverage_paths,
            "fault_path": fault_path,
            "num_samples": num_samples,
        })

    print("\n" + "="*60)
    print("Creating multi-metric dataset")
    print("="*60)

    dataset = MultiMetricDataset(
        data_infos=data_infos,
        criterion_names=criterion_names,
        pca_dim=args.pca_dim,
        seed=args.seed,
        dataset_tag=dataset_tag,
        pca_base_dir=pca_base_dir,
    )

    n_mutants = 15
    n_seeds_per_model = min(dataset.lengths) // n_mutants
    n_models = len(data_infos)

    rng = np.random.RandomState(args.seed)
    all_seed_ids = np.arange(n_seeds_per_model)
    rng.shuffle(all_seed_ids)
    n_train_seeds = int(0.8 * n_seeds_per_model)
    train_seed_set = set(all_seed_ids[:n_train_seeds].tolist())

    train_indices = []
    val_indices = []
    for m_idx in range(n_models):
        model_offset = int(dataset.cum[m_idx])
        for sid in range(n_seeds_per_model):
            start = model_offset + sid * n_mutants
            indices = list(range(start, start + n_mutants))
            if sid in train_seed_set:
                train_indices.extend(indices)
            else:
                val_indices.extend(indices)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    print(f"  Seed-level split: {n_train_seeds}/{n_seeds_per_model} seeds -> "
          f"train={len(train_indices)}, val={len(val_indices)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    input_dim = args.pca_dim * len(criterion_names)

    if args.model_type == "mlp":
        model = MultiMetricMLP(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            dropout=args.dropout,
            num_models=len(model_names)
        ).to(device)
    else:
        model = CoverageTransformer(
            pca_dim=args.pca_dim,
            num_criteria=len(criterion_names),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_models=len(model_names)
        ).to(device)

    print(f"\nModel type: {args.model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_auc = 0.0
    best_epoch = 0
    patience = 0
    max_patience = 15

    log_file = os.path.join(args.output_dir, "train_log.txt")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,train_auc,val_auc,best_val_loss,patience\n")

    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device,
            label_smoothing=args.label_smoothing,
            criterion_drop_prob=args.criterion_drop_prob,
            gate_entropy_lambda=args.gate_entropy_lambda
        )

        val_loss, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion, device
        )

        val_auc = auc_score(val_preds, val_labels)
        train_auc = auc_score(train_preds, train_labels)

        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_auc:.6f},{val_auc:.6f},{best_val_loss:.6f},{patience}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            best_epoch = epoch
            patience = 0

            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "args": vars(args),
                "pca_dim": args.pca_dim,
                "criterion_names": criterion_names,
                "dataset": args.dataset,
                "num_classes": num_classes,
            }, checkpoint_path)
            print(f"  -> Saved best model (Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f})")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch} (val_loss did not improve for {max_patience} epochs)")
            break

    print(f"\nTraining complete!")
    print(f"Best val Loss: {best_val_loss:.4f}, AUC: {best_auc:.4f} (epoch {best_epoch})")

    print("\n" + "="*60)
    print("Training latent space clusters (Latent Space Coverage)")
    print("="*60)

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_features = dataset.precomputed_features
    all_faults = dataset.precomputed_faults

    print("\nComputing CP scores and CLS token representations for all samples...")
    all_cp_scores = []
    all_cls_representations = []

    batch_size = 256
    for i in tqdm(range(0, len(all_features), batch_size), desc="Computing CP scores"):
        batch_features = torch.from_numpy(all_features[i:i+batch_size]).to(device)
        batch_model_ids = torch.from_numpy(dataset.precomputed_model_ids[i:i+batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch_features, batch_model_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            if hasattr(model, 'get_cls_representation'):
                cls_repr = model.get_cls_representation(batch_features, batch_model_ids).cpu().numpy()
            else:
                cls_repr = batch_features.cpu().numpy()
        all_cp_scores.append(probs)
        all_cls_representations.append(cls_repr)
    all_cp_scores = np.concatenate(all_cp_scores)
    all_cls_representations = np.concatenate(all_cls_representations)

    cluster_centers, cluster_cp_scores, cluster_stats = train_latent_space_clusters(
        features=all_cls_representations,
        cp_scores=all_cp_scores,
        n_clusters=args.n_clusters,
        random_state=args.seed
    )

    cluster_path = os.path.join(args.output_dir, "latent_space_clusters.npz")
    exclude_keys = {'n_clusters', 'cluster_cp_scores'}
    cluster_stats_copy = {k: v for k, v in cluster_stats.items() if k not in exclude_keys}
    np.savez(
        cluster_path,
        cluster_centers=cluster_centers,
        cluster_cp_scores=cluster_cp_scores,
        n_clusters=args.n_clusters,
        **cluster_stats_copy
    )
    print(f"\nCluster results saved to {cluster_path}")

    print("\n" + "="*60)
    print("Demonstrating latent space coverage mechanism")
    print("="*60)

    lsc = LatentSpaceClusterCoverage(
        n_clusters=args.n_clusters,
        cluster_centers=cluster_centers,
        cluster_cp_scores=cluster_cp_scores,
        lambda_param=0.7,
    )

    lscc_coverage_history = []
    for i in range(min(1000, len(all_cls_representations))):
        cls_repr = all_cls_representations[i]
        cp_score = all_cp_scores[i]
        lsc.update(cls_repr, cp_score)
        lscc_coverage_history.append(lsc.get_coverage())

    stats = lsc.get_cluster_stats()
    print(f"After 1000 samples:")
    print(f"  Covered clusters: {stats['covered_clusters']}/{stats['n_clusters']}")
    print(f"  LSCC (pure coverage): {stats['lsc']:.4f}")
    print(f"  WLSCC (weighted coverage): {stats['wlsc']:.4f}")

    lsc_results_path = os.path.join(args.output_dir, "latent_space_coverage_demo.npz")
    np.savez(
        lsc_results_path,
        coverage_history=lscc_coverage_history,
        covered_clusters=stats['covered_cluster_ids'],
        cluster_centers=cluster_centers,
        cluster_cp_scores=cluster_cp_scores
    )
    print(f"Latent space coverage demo results saved to {lsc_results_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()

