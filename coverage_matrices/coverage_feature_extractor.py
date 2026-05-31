"""
Shared Coverage Feature Extractor

This module provides a unified interface for extracting per-layer summary
coverage features and applying PCA, ensuring consistency between training
(extract_multi_metric_coverage.py) and inference (fuzz/experiment scripts).

The feature extraction logic here is identical to
coverage_matrices/extract_multi_metric_coverage.py:extract_coverage_features().
"""

import os
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tool

def extract_coverage_features(data_batch, model, criteria_dict, seed_label=None):
    """
    Extract per-sample coverage feature vectors using per-layer summaries.

    This is the SAME logic as extract_multi_metric_coverage.py lines 222-408.
    Each criterion produces one scalar per layer (not per-neuron).

    Args:
        data_batch: torch.Tensor [B, C, H, W] on device
        model: the DNN model (eval mode, on device)
        criteria_dict: dict {criterion_name: criterion_object}
        seed_label: int, ground-truth label (needed for LSC/DSC)

    Returns:
        dict: {criterion_name -> np.array [B, D]}
              where D = number of layers (per-layer summary)
    """
    batch_size = data_batch.shape[0]
    features = {}

    with torch.no_grad():
        layer_output_dict = tool.get_layer_output(model, data_batch)
    layer_names = sorted(layer_output_dict.keys())

    if 'NC' in criteria_dict:
        nc = criteria_dict['NC']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            scaled = tool.scale(lo)
            frac = (scaled > nc.threshold).float().mean(dim=1)
            vals.append(frac.cpu().numpy().reshape(-1, 1))
        features['NC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'KMNC' in criteria_dict:
        kmnc = criteria_dict['KMNC']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            if ln in kmnc.range_dict:
                l_bound, u_bound = kmnc.range_dict[ln]
                div = (u_bound - l_bound).clamp(min=1e-6)
                normalized = ((lo - l_bound) / div).clamp(0, 1)
                mean_pos = normalized.mean(dim=1)
                vals.append(mean_pos.cpu().numpy().reshape(-1, 1))
        features['KMNC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'NBC' in criteria_dict:
        nbc = criteria_dict['NBC']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            if ln in nbc.range_dict:
                l_bound, u_bound = nbc.range_dict[ln]
                below = (lo < l_bound).float().mean(dim=1)
                above = (lo > u_bound).float().mean(dim=1)
                boundary_frac = (below + above) / 2
                vals.append(boundary_frac.cpu().numpy().reshape(-1, 1))
        features['NBC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'SNAC' in criteria_dict:
        snac = criteria_dict['SNAC']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            if ln in snac.range_dict:
                _, u_bound = snac.range_dict[ln]
                above = (lo > u_bound).float().mean(dim=1)
                vals.append(above.cpu().numpy().reshape(-1, 1))
        features['SNAC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'TKNC' in criteria_dict:
        tknc = criteria_dict['TKNC']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            num_neuron = lo.size(1)
            k = min(tknc.k, num_neuron)
            topk_vals, _ = lo.topk(k, dim=1, largest=True)
            mean_topk = topk_vals.mean(dim=1)
            vals.append(mean_topk.cpu().numpy().reshape(-1, 1))
        features['TKNC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'TKNP' in criteria_dict:
        tknp = criteria_dict['TKNP']
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            num_neuron = lo.size(1)
            k = min(int(tknp.k), num_neuron)
            topk_vals, _ = lo.topk(k, dim=1, largest=True)
            std_topk = topk_vals.std(dim=1)
            vals.append(std_topk.cpu().numpy().reshape(-1, 1))
        features['TKNP'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'CC' in criteria_dict:
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            l2_norm = lo.norm(dim=1)
            vals.append(l2_norm.cpu().numpy().reshape(-1, 1))
        features['CC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'NLC' in criteria_dict:
        vals = []
        for ln in layer_names:
            lo = layer_output_dict[ln]
            l1_norm = lo.abs().mean(dim=1)
            vals.append(l1_norm.cpu().numpy().reshape(-1, 1))
        features['NLC'] = np.concatenate(vals, axis=1) if vals else np.zeros((batch_size, 1))

    if 'LSC' in criteria_dict and seed_label is not None:
        lsc = criteria_dict['LSC']
        if lsc.kde_cache:
            try:
                SA_batch = []
                for ln in layer_names:
                    lo = layer_output_dict[ln]
                    if ln in lsc.mask_index_dict:
                        SA_batch.append(lo[:, lsc.mask_index_dict[ln]].view(batch_size, -1))
                if SA_batch:
                    SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()
                    lsa_vals = np.zeros(batch_size, dtype=np.float32)
                    if seed_label in lsc.kde_cache:
                        for i in range(batch_size):
                            SA = SA_batch[i]
                            if lsc.num_class <= 1:
                                lsa = float(-lsc.kde_cache[seed_label].logpdf(
                                    np.expand_dims(SA, 1)))
                            else:
                                lsa = float(-lsc.kde_cache[seed_label].score_samples(
                                    SA.reshape(1, -1)))
                            if not (np.isnan(lsa) or np.isinf(lsa)):
                                lsa_vals[i] = lsa
                    features['LSC'] = lsa_vals.reshape(-1, 1)
                else:
                    features['LSC'] = np.zeros((batch_size, 1), dtype=np.float32)
            except Exception:
                features['LSC'] = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            features['LSC'] = np.zeros((batch_size, 1), dtype=np.float32)

    if 'DSC' in criteria_dict and seed_label is not None:
        dsc = criteria_dict['DSC']
        if dsc.SA_cache:
            try:
                SA_batch = []
                for ln in layer_names:
                    lo = layer_output_dict[ln]
                    if ln in dsc.mask_index_dict:
                        SA_batch.append(lo[:, dsc.mask_index_dict[ln]].view(batch_size, -1))
                if SA_batch:
                    SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()
                    dsa_vals = np.zeros(batch_size, dtype=np.float32)
                    if seed_label in dsc.SA_cache:
                        sa_same = dsc.SA_cache[seed_label]
                        sa_others = []
                        for j in range(dsc.num_class):
                            if j != seed_label and j in dsc.SA_cache:
                                sa_others.append(dsc.SA_cache[j])
                        sa_other = np.concatenate(sa_others, axis=0) if sa_others else None
                        for i in range(batch_size):
                            SA = SA_batch[i]
                            dist_a = np.min(np.linalg.norm(SA - sa_same, axis=1))
                            if sa_other is not None:
                                dist_b = np.min(np.linalg.norm(SA - sa_other, axis=1))
                                dsa = dist_a / max(dist_b, 1e-6)
                            else:
                                dsa = 0.0
                            if not (np.isnan(dsa) or np.isinf(dsa)):
                                dsa_vals[i] = dsa
                    features['DSC'] = dsa_vals.reshape(-1, 1)
                else:
                    features['DSC'] = np.zeros((batch_size, 1), dtype=np.float32)
            except Exception:
                features['DSC'] = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            features['DSC'] = np.zeros((batch_size, 1), dtype=np.float32)

    if 'MDSC' in criteria_dict and seed_label is not None:
        mdsc = criteria_dict['MDSC']
        if hasattr(mdsc, 'estimator') and hasattr(mdsc.estimator, 'CoVarianceInv') and mdsc.estimator.CoVarianceInv is not None:
            try:
                SA_batch = []
                for ln in layer_names:
                    lo = layer_output_dict[ln]
                    if ln in mdsc.mask_index_dict:
                        SA_batch.append(lo[:, mdsc.mask_index_dict[ln]].view(batch_size, -1))
                if SA_batch:
                    SA_batch = torch.cat(SA_batch, 1)
                    label_tensor = torch.tensor([seed_label] * batch_size, device=SA_batch.device)
                    mu = mdsc.estimator.Ave[label_tensor]
                    covar_inv = mdsc.estimator.CoVarianceInv[label_tensor]
                    diff = (SA_batch - mu).unsqueeze(1)
                    mdsa = torch.bmm(torch.bmm(diff, covar_inv), diff.transpose(1, 2))
                    mdsa = mdsa.squeeze().sqrt().detach().cpu().numpy()
                    mdsa = np.nan_to_num(mdsa, nan=0.0, posinf=0.0, neginf=0.0)
                    features['MDSC'] = mdsa.reshape(-1, 1).astype(np.float32)
                else:
                    features['MDSC'] = np.zeros((batch_size, 1), dtype=np.float32)
            except Exception:
                features['MDSC'] = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            features['MDSC'] = np.zeros((batch_size, 1), dtype=np.float32)

    return features

def load_pca_models(model_name, data_dir="./coverage_matrices"):
    """
    Load PCA models and scalers saved during training (train_cp_pca.py).

    These are the EXACT same PCA models used to produce the training features,
    ensuring perfect consistency between training and inference.

    Args:
        model_name: e.g. "resnet50", "vgg16_bn", "mobilenet_v2"
        data_dir: base directory containing activation_coverage_* folders

    Returns:
        pca_models: dict {criterion_name: PCA object}
        scalers: dict {criterion_name: StandardScaler object}
    """
    possible_paths = [
        os.path.join(data_dir, f"activation_coverage_{model_name}_cifar10", "pca_models.pkl"),
        os.path.join(data_dir, f"activation_coverage_{model_name}", "pca_models.pkl"),
    ]

    pca_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pca_path = path
            break

    assert pca_path is not None, (
        f"PCA models not found. Tried:\n" + "\n".join(f"  - {p}" for p in possible_paths) +
        f"\nRun train_cp_pca.py first to generate them."
    )

    with open(pca_path, 'rb') as f:
        saved = pickle.load(f)

    pca_models = {}
    scalers = {}
    for key, pca_obj in saved['pca_models'].items():
        criterion = key.replace(f"{model_name}_", "", 1)
        pca_models[criterion] = pca_obj

    for key, scaler_obj in saved['scalers'].items():
        criterion = key.replace(f"{model_name}_", "", 1)
        scalers[criterion] = scaler_obj

    print(f"  Loaded PCA models for {model_name}: {sorted(pca_models.keys())}")
    return pca_models, scalers

def extract_pca_coverage(model, data_batch, pca_models, scalers,
                         criteria_dict, criterion_names, pca_dim,
                         seed_label=None):
    """
    Extract per-layer summary features and apply PCA reduction.

    This is the inference-time equivalent of what train_cp_pca.py does
    during precomputation.

    Args:
        model: DNN model (eval mode, on device)
        data_batch: torch.Tensor [B, C, H, W]
        pca_models: dict {criterion_name: PCA} from load_pca_models()
        scalers: dict {criterion_name: StandardScaler} from load_pca_models()
        criteria_dict: dict {criterion_name: criterion_object}
        criterion_names: list of criterion names in order
        pca_dim: target PCA dimension (must match training)
        seed_label: int, for LSC/DSC (can be None if not using those)

    Returns:
        torch.Tensor of shape (B, num_criteria * pca_dim)
    """
    batch_size = data_batch.shape[0]

    raw_features = extract_coverage_features(
        data_batch, model, criteria_dict, seed_label=seed_label
    )

    all_reduced = []
    for criterion in criterion_names:
        assert criterion in pca_models, (
            f"PCA model missing for criterion '{criterion}'. "
            f"Available: {sorted(pca_models.keys())}"
        )
        assert criterion in scalers, (
            f"Scaler missing for criterion '{criterion}'. "
            f"Available: {sorted(scalers.keys())}"
        )

        scaler = scalers[criterion]
        pca = pca_models[criterion]
        expected_dim = scaler.n_features_in_

        if criterion in raw_features:
            cov_vec = raw_features[criterion]
        else:
            cov_vec = np.zeros((batch_size, expected_dim), dtype=np.float32)

        cov_vec = np.nan_to_num(cov_vec, nan=0.0, posinf=1e6, neginf=-1e6)

        actual_dim = cov_vec.shape[1]
        assert actual_dim == expected_dim, (
            f"Feature dimension mismatch for {criterion}: "
            f"got {actual_dim}, expected {expected_dim} (from training). "
            f"This means the feature extraction is inconsistent with training."
        )

        scaled = scaler.transform(cov_vec)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        reduced = pca.transform(scaled)

        if reduced.shape[-1] < pca_dim:
            padded = np.zeros((batch_size, pca_dim), dtype=np.float32)
            padded[:, :reduced.shape[-1]] = reduced
            reduced = padded
        else:
            reduced = reduced[:, :pca_dim]

        all_reduced.append(reduced)

    features = np.concatenate(all_reduced, axis=1)
    return torch.from_numpy(features).float()

def init_coverage_criteria(model, layer_size_dict, train_loader=None, device=None, num_classes=10, skip_sa_criteria=False):
    """
    Initialize 10 coverage criteria. Mirrors the logic in
    extract_multi_metric_coverage.py:init_coverage_criteria().

    Args:
        model: DNN model
        layer_size_dict: from tool.get_layer_output_sizes()
        train_loader: DataLoader for building criteria statistics
        device: torch device
        num_classes: number of classes in the dataset
        skip_sa_criteria: if True, skip LSC and DSC (SA-based, very slow)

    Returns:
        dict: {criterion_name: criterion_object}
    """
    import coverage

    if device is None:
        device = next(model.parameters()).device

    criteria = {}
    criteria['NC'] = coverage.NC(model, layer_size_dict, hyper=0.5)
    criteria['NBC'] = coverage.NBC(model, layer_size_dict, hyper=None)
    criteria['SNAC'] = coverage.SNAC(model, layer_size_dict, hyper=None)
    criteria['NLC'] = coverage.NLC(model, layer_size_dict, hyper=None)
    criteria['KMNC'] = coverage.KMNC(model, layer_size_dict, hyper=100)
    criteria['TKNC'] = coverage.TKNC(model, layer_size_dict, hyper=3)
    criteria['TKNP'] = coverage.TKNP(model, layer_size_dict, hyper=3)
    criteria['CC'] = coverage.CC(model, layer_size_dict, hyper=10)

    if not skip_sa_criteria:
        criteria['LSC'] = coverage.LSC(model, layer_size_dict, hyper=2000,
                                        min_var=1e-5, num_class=num_classes)
        criteria['DSC'] = coverage.DSC(model, layer_size_dict, hyper=1000,
                                        min_var=1e-5, num_class=num_classes)

    if train_loader is not None:
        print("Building criteria that require training data statistics...")
        for name in ['KMNC', 'NBC', 'SNAC']:
            print(f"  Building {name}...")
            criteria[name].build(train_loader)
            print(f"  {name} build completed")
        if not skip_sa_criteria:
            for name in ['LSC', 'DSC']:
                try:
                    print(f"  Building {name} (SA-based, may be slow/OOM)...")
                    criteria[name].build(train_loader)
                    print(f"  {name} build completed")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"  WARNING: {name} build failed: {e}")
                    torch.cuda.empty_cache()

    print("  All criteria built, returning...")
    return criteria
