import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
import numpy as np
import torch
import time
import argparse
import json
import pickle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'datasets'))
sys.path.insert(0, os.path.join(_ROOT, 'coverage_matrices'))

import coverage
import tool
import data_loader
import image_transforms
from shared_config import (
    build_model, get_dataset_config, get_model_transform,
    TRANSLATION_PARAMS, SCALE_PARAMS, ROTATION_PARAMS,
    CONTRAST_PARAMS, BRIGHTNESS_PARAMS, BLUR_PARAMS,
    CIFAR10_MEAN, CIFAR10_STD,
)
from coverage_feature_extractor import (
    extract_pca_coverage, init_coverage_criteria,
)
from cp_model import CoverageTransformer, LatentSpaceClusterCoverage
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from dataset_model_config import MODEL_NAME_TO_ID, get_models_for_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_list = [
    ("translation", image_transforms.image_translation, TRANSLATION_PARAMS),
    ("scale", image_transforms.image_scale, SCALE_PARAMS),
    ("rotation", image_transforms.image_rotation, ROTATION_PARAMS),
    ("contrast", image_transforms.image_contrast, CONTRAST_PARAMS),
    ("brightness", image_transforms.image_brightness, BRIGHTNESS_PARAMS),
    ("blur", image_transforms.image_blur, BLUR_PARAMS),
]


def mutate_image(img_np):
    tname, tfunc, tparams = random.choice(transforms_list)
    param = random.choice(tparams)
    mutant = tfunc(img_np, param)
    return np.clip(mutant, 0, 255).astype(np.float32)


def image_to_tensor(img_np, ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD):
    img = img_np / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1)
    mean = torch.tensor(ds_mean).view(3, 1, 1)
    std = torch.tensor(ds_std).view(3, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0).to(device)


def load_seeds(model, num_seeds, model_name='resnet50',
               dataset_name='CIFAR10', ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD):
    cfg = get_dataset_config(dataset_name)
    num_classes = cfg['num_classes']
    seeds_per_class = max(1, num_seeds // num_classes)

    class FakeArgs:
        dataset = dataset_name
        image_size = cfg['image_size']
        num_class = num_classes
        nc = cfg['channels']
        batch_size = 50; num_workers = 0
        num_per_class = num_seeds // 5
    FakeArgs.model = model_name

    if dataset_name == 'MNIST':
        ds = data_loader.MNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'Fashion-MNIST':
        ds = data_loader.FashionMNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'TinyImageNet':
        ds = data_loader.ImageNetFuzzDataset(
            FakeArgs(), image_dir=cfg['data_dir'], split='val')
    else:
        ds = data_loader.CIFAR10FuzzDataset(FakeArgs(), split='test')

    images_per_class = {i: [] for i in range(num_classes)}
    total = ds.get_len()

    for i in range(total):
        img_tensor, label_tensor = ds.get_item(i)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.float32)
        lbl = label_tensor.item()

        if len(images_per_class[lbl]) >= seeds_per_class:
            continue

        t = image_to_tensor(img_np, ds_mean, ds_std)
        with torch.no_grad():
            pred = model(t).argmax(dim=1).item()
        if pred == lbl:
            images_per_class[lbl].append((img_np, lbl))

        if all(len(v) >= seeds_per_class for v in images_per_class.values()):
            break

    images, labels = [], []
    for cls in range(num_classes):
        for img_np, lbl in images_per_class[cls]:
            images.append(img_np)
            labels.append(lbl)

    print(f"  Class distribution: {[len(images_per_class[i]) for i in range(num_classes)]}")
    return images, labels


def make_fresh_criteria(model, layer_size_dict, crit_loader):
    return init_coverage_criteria(
        model, layer_size_dict, train_loader=crit_loader, device=device)


def build_crit_loader(seed_images, seed_labels, n=50,
                      ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD):
    tensors = [image_to_tensor(img, ds_mean, ds_std).squeeze(0) for img in seed_images[:n]]
    batch = torch.stack(tensors)
    lbls = seed_labels[:n]
    ds = TensorDataset(batch, torch.tensor(lbls, dtype=torch.long))
    return TorchDataLoader(ds, batch_size=32, shuffle=False)


STRATEGIES = [
    'cp_lsc_guided',
    'nc_025', 'nc_050', 'nc_075',
    'nbc', 'snac',
    'tknc_k1', 'tknc_k3', 'tknc_k5',
    'kmnc_k100',
    'kmnc_k1000',
    'dlregion_nc', 'dlregion_tknc',
    'distxplore',
]

STRATEGY_CONFIG = {
    'nc_025':     ('NC',   0.25, False),
    'nc_050':     ('NC',   0.50, False),
    'nc_075':     ('NC',   0.75, False),
    'nbc':        ('NBC',  None, True),
    'snac':       ('SNAC', None, True),
    'tknc_k1':    ('TKNC', 1,    False),
    'tknc_k3':    ('TKNC', 3,    False),
    'tknc_k5':    ('TKNC', 5,    False),
    'kmnc_k10':   ('KMNC', 10,   True),
    'kmnc_k100':  ('KMNC', 100,  True),
    'kmnc_k1000': ('KMNC', 1000, True),
    'dlregion_nc':   ('DLRegionNC',   None, True),
    'dlregion_tknc': ('DLRegionTKNC', 3,    True),
    'distxplore':    ('DistXplore',   None, True),
}

STRATEGY_DESC = {
    'cp_lsc_guided': 'CP+LSCC Guided (Ours)',
    'random_walk': 'Random Walk Baseline',
    'nc_025':     'NC (threshold=0.25)',
    'nc_050':     'NC (threshold=0.50)',
    'nc_075':     'NC (threshold=0.75)',
    'nbc':        'NBC Coverage Guided',
    'snac':       'SNAC Coverage Guided',
    'tknc_k1':    'TKNC (k=1)',
    'tknc_k3':    'TKNC (k=3)',
    'tknc_k5':    'TKNC (k=5)',
    'kmnc_k10':   'KMNC (k=10)',
    'kmnc_k100':  'KMNC (k=100)',
    'kmnc_k1000': 'KMNC (k=1000)',
    'dlregion_nc':   'DLRegion-NC',
    'dlregion_tknc': 'DLRegion-TKNC (k=3)',
    'distxplore':    'DistXplore',
}


def run_cp_lsc_guided(model, seed_images, seed_labels,
                      layer_size_dict, crit_loader,
                      cp_model, model_id, pca_models, scalers,
                      criterion_names, pca_dim,
                      cluster_centers, cluster_cp_scores, n_clusters,
                      lambda_lscc,
                      max_tests, record_interval=100, rng_seed=None,
                      ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD):
    criteria_dict = make_fresh_criteria(model, layer_size_dict, crit_loader)

    lscc_tracker = LatentSpaceClusterCoverage(
        n_clusters=n_clusters,
        cluster_centers=cluster_centers.copy(),
        cluster_cp_scores=cluster_cp_scores.copy(),
        lambda_param=0.7, risk_threshold=0.3,
    )

    total_tests = 0
    total_faults = 0
    total_accepted = 0
    fdr_curve = []
    lsc_curve = []
    wlsc_curve = []

    iters_per_seed = max_tests // len(seed_images)

    for seed_idx in range(len(seed_images)):
        if total_tests >= max_tests:
            break
        seed_img = seed_images[seed_idx]
        seed_label = seed_labels[seed_idx]
        current_img = seed_img.copy()
        current_cp_score = 0.0
        current_lscc_coverage = 0.0

        for _ in range(iters_per_seed):
            if total_tests >= max_tests:
                break

            mutant_np = mutate_image(current_img)
            input_tensor = image_to_tensor(mutant_np, ds_mean, ds_std)

            with torch.no_grad():
                pred = model(input_tensor).argmax(dim=1).item()
            is_fault = (pred != seed_label)
            total_tests += 1
            if is_fault:
                total_faults += 1

            with torch.no_grad():
                features = extract_pca_coverage(
                    model, input_tensor, pca_models, scalers,
                    pca_dim=pca_dim, criterion_names=criterion_names,
                    criteria_dict=criteria_dict, seed_label=seed_label
                ).to(device)
                mid = torch.tensor([model_id], dtype=torch.long).to(device)
                logits = cp_model(features, mid)
                cp_score = torch.sigmoid(logits).item()

            pca_feat = features.cpu().numpy().flatten()
            lscc_tracker.update(pca_feat, cp_score)
            lscc_coverage = lscc_tracker.get_coverage()

            if cp_score > current_cp_score or lscc_coverage > current_lscc_coverage:
                current_img = mutant_np
                current_cp_score = cp_score
                current_lscc_coverage = lscc_coverage
                total_accepted += 1
            elif random.random() < 0.1:
                current_img = mutant_np
                current_cp_score = cp_score
                current_lscc_coverage = lscc_coverage
                total_accepted += 1

            if total_tests % record_interval == 0:
                fdr_curve.append(total_faults / total_tests)
                lsc_curve.append(lscc_tracker.get_lsc())
                wlsc_curve.append(lscc_tracker.get_wlsc())

        if total_tests % 1000 == 0 and total_tests > 0:
            fdr_so_far = total_faults / total_tests
            print(f"    [tests={total_tests}] FDR={fdr_so_far:.4f}, "
                  f"LSCC={lscc_tracker.get_lsc():.4f}", flush=True)

    fdr = total_faults / total_tests if total_tests > 0 else 0

    return {
        'strategy':     'cp_lsc_guided',
        'total_tests':  total_tests,
        'total_faults': total_faults,
        'fdr':          fdr,
        'accept_rate':  total_accepted / total_tests if total_tests > 0 else 0,
        'fdr_curve':    np.array(fdr_curve),
        'lsc_final':    lscc_tracker.get_lsc(),
        'wlsc_final':   lscc_tracker.get_wlsc(),
        'balanced_cov': lscc_tracker.get_balanced_coverage(),
        'lsc_curve':    np.array(lsc_curve),
        'wlsc_curve':   np.array(wlsc_curve),
        'record_interval': record_interval,
    }


def run_random_walk(model, seed_images, seed_labels,
                    max_tests, record_interval=100, rng_seed=None,
                    ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD):
    total_tests = 0
    total_faults = 0
    total_accepted = 0
    fdr_curve = []

    iters_per_seed = max_tests // len(seed_images)

    for seed_idx in range(len(seed_images)):
        if total_tests >= max_tests:
            break
        seed_img = seed_images[seed_idx]
        seed_label = seed_labels[seed_idx]
        current_img = seed_img.copy()

        for _ in range(iters_per_seed):
            if total_tests >= max_tests:
                break

            mutant_np = mutate_image(current_img)
            input_tensor = image_to_tensor(mutant_np, ds_mean, ds_std)

            with torch.no_grad():
                pred = model(input_tensor).argmax(dim=1).item()
            is_fault = (pred != seed_label)
            total_tests += 1
            if is_fault:
                total_faults += 1

            if random.random() < 0.5:
                current_img = mutant_np
                total_accepted += 1

            if total_tests % record_interval == 0:
                fdr_curve.append(total_faults / total_tests)

        if total_tests % 1000 == 0 and total_tests > 0:
            fdr_so_far = total_faults / total_tests
            print(f"    [tests={total_tests}] FDR={fdr_so_far:.4f}", flush=True)

    fdr = total_faults / total_tests if total_tests > 0 else 0

    return {
        'strategy':     'random_walk',
        'total_tests':  total_tests,
        'total_faults': total_faults,
        'fdr':          fdr,
        'accept_rate':  total_accepted / total_tests if total_tests > 0 else 0,
        'fdr_curve':    np.array(fdr_curve),
        'record_interval': record_interval,
    }


def run_baseline(strategy, model, seed_images, seed_labels,
                 layer_size_dict, crit_loader,
                 max_tests, record_interval=100, rng_seed=None,
                 num_classes=10, accept_rate_cap=0.3,
                 ds_mean=CIFAR10_MEAN, ds_std=CIFAR10_STD,
                 hard_cap=False, full_crit_loader=None):
    crit_class_name, hyper, needs_build = STRATEGY_CONFIG[strategy]
    crit_class = getattr(coverage, crit_class_name)
    if crit_class_name in ('NBC', 'SNAC'):
        crit = crit_class(model, layer_size_dict, hyper=None)
    elif crit_class_name in ('LSC', 'DSC'):
        crit = crit_class(model, layer_size_dict, hyper=hyper, min_var=1e-5, num_class=num_classes)
    elif crit_class_name == 'DistXplore':
        crit = crit_class(model, layer_size_dict, hyper=hyper, num_class=num_classes)
    else:
        crit = crit_class(model, layer_size_dict, hyper=hyper)

    if needs_build:
        if crit_class_name == 'DistXplore' and full_crit_loader is not None:
            crit.build(full_crit_loader)
        else:
            crit.build(crit_loader)

    total_tests = 0
    total_faults = 0
    total_accepted = 0
    fdr_curve = []

    iters_per_seed = max_tests // len(seed_images)

    for seed_idx in range(len(seed_images)):
        if total_tests >= max_tests:
            break
        seed_img = seed_images[seed_idx]
        seed_label = seed_labels[seed_idx]
        current_img = seed_img.copy()

        for _ in range(iters_per_seed):
            if total_tests >= max_tests:
                break

            mutant_np = mutate_image(current_img)
            input_tensor = image_to_tensor(mutant_np, ds_mean, ds_std)

            with torch.no_grad():
                pred = model(input_tensor).argmax(dim=1).item()
            is_fault = (pred != seed_label)
            total_tests += 1
            if is_fault:
                total_faults += 1

            prev_cov = crit.current
            crit.step(input_tensor)
            want_accept = (crit.current > prev_cov) or (random.random() < 0.1)

            if want_accept:
                cur_rate = total_accepted / total_tests if total_tests > 0 else 0
                if cur_rate < accept_rate_cap:
                    current_img = mutant_np
                    total_accepted += 1
                elif not hard_cap:
                    decay_prob = accept_rate_cap / cur_rate if cur_rate > 0 else 1.0
                    if random.random() < decay_prob:
                        current_img = mutant_np
                        total_accepted += 1

            if total_tests % record_interval == 0:
                fdr_curve.append(total_faults / total_tests)

        if total_tests % 1000 == 0 and total_tests > 0:
            fdr_so_far = total_faults / total_tests
            print(f"    [tests={total_tests}] FDR={fdr_so_far:.4f}, "
                  f"AccRate={total_accepted/total_tests:.3f}", flush=True)

    fdr = total_faults / total_tests if total_tests > 0 else 0

    return {
        'strategy':     strategy,
        'total_tests':  total_tests,
        'total_faults': total_faults,
        'fdr':          fdr,
        'accept_rate':  total_accepted / total_tests if total_tests > 0 else 0,
        'fdr_curve':    np.array(fdr_curve),
        'record_interval': record_interval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'MNIST', 'Fashion-MNIST', 'TinyImageNet'])
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--max_tests', type=int, default=10000)
    parser.add_argument('--num_repeats', type=int, default=3)
    parser.add_argument('--cp_checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to Bayesian optimization result JSON file')
    parser.add_argument('--accept_rate_cap', type=float, default=0.3,
                        help='Baseline accept rate cap (default: 0.3)')
    parser.add_argument('--hard_cap', action='store_true',
                        help='Use hard cap (reject directly when over cap, no decay)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategies', type=str, default=None,
                        help='Comma-separated list of strategies (default: run all)')
    args = parser.parse_args()

    if args.strategies:
        strategies_to_run = [s.strip() for s in args.strategies.split(',')]
    else:
        strategies_to_run = STRATEGIES

    ds_cfg = get_dataset_config(args.dataset)
    ds_mean = ds_cfg['mean']
    ds_std = ds_cfg['std']
    ds_tag = ds_cfg['weight_tag']
    num_classes = ds_cfg['num_classes']
    image_size = ds_cfg['image_size']

    dataset_dir_map = {
        'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST',
        'Fashion-MNIST': 'Fashion-MNIST', 'TinyImageNet': 'TinyImageNet',
    }
    dataset_dir = dataset_dir_map.get(args.dataset, args.dataset)

    if args.cp_checkpoint is None:
        possible_paths = [
            f'./cp_checkpoints/{dataset_dir}/best_model.pt',
            f'./cp_checkpoints/{args.dataset}/best_model.pt',
            f'./cp_checkpoints_{ds_tag}/best_model.pt',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                args.cp_checkpoint = p
                break
        if args.cp_checkpoint is None:
            args.cp_checkpoint = possible_paths[0]
    if args.data_dir is None:
        args.data_dir = f'./coverage_matrices/{dataset_dir}'
    if args.output_dir is None:
        args.output_dir = f'./experiments/{dataset_dir}/exp1_results'

    if args.model == 'all':
        model_names = get_models_for_dataset(args.dataset)
    else:
        model_names = [args.model]

    model_name_to_id = MODEL_NAME_TO_ID.get(args.dataset, {})

    if args.weights_file is None:
        args.weights_file = f'./bayesian_optimal_weights_{ds_tag}.json'
    per_model_lambda = {}
    if os.path.exists(args.weights_file):
        with open(args.weights_file, 'r') as f:
            bayesian_results = json.load(f)
        for mn in model_names:
            if mn in bayesian_results:
                per_model_lambda[mn] = bayesian_results[mn]['best_lambda_lscc']
            else:
                per_model_lambda[mn] = 0.5
                print(f"  Warning: {mn} not found in weights file, using default lambda_lscc=0.5")
    else:
        print(f"  Warning: weights file {args.weights_file} not found, using default lambda_lscc=0.5")
        for mn in model_names:
            per_model_lambda[mn] = 0.5

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"Experiment 1: Baseline Comparison")
    print(f"Dataset: {args.dataset}, Models: {model_names}")
    print(f"Seeds: {args.num_seeds}, Budget: {args.max_tests}, Repeats: {args.num_repeats}")
    print(f"Per-model lambda_lscc: {per_model_lambda}")
    print(f"Strategies: {strategies_to_run}")
    print(f"{'='*70}")

    print("\nLoading CP model...")
    checkpoint = torch.load(args.cp_checkpoint, map_location=device, weights_only=False)
    cp_args = checkpoint["args"]
    criterion_names = checkpoint.get("criterion_names", cp_args.get(
        "criterion_names",
        ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']))
    pca_dim = checkpoint.get("pca_dim", cp_args.get("pca_dim", 64))
    n_clusters = cp_args.get("n_clusters", 50)

    state_dict = checkpoint["model_state_dict"]
    embed_key = None
    for k in state_dict:
        if 'model_embed' in k and 'weight' in k:
            embed_key = k
            break
    if embed_key:
        cp_num_models = state_dict[embed_key].shape[0]
    else:
        cp_num_models = checkpoint.get("num_models", 3)
    cp_model = CoverageTransformer(
        pca_dim=pca_dim, num_criteria=len(criterion_names),
        d_model=cp_args["d_model"], nhead=cp_args["nhead"],
        num_layers=cp_args["num_layers"], dropout=cp_args["dropout"],
        num_models=cp_num_models
    ).to(device)
    cp_model.load_state_dict(checkpoint["model_state_dict"])
    cp_model.eval()

    cluster_data = np.load(os.path.join(
        os.path.dirname(args.cp_checkpoint), 'latent_space_clusters.npz'))
    cluster_centers = cluster_data['cluster_centers']
    cluster_cp_scores = cluster_data['cluster_cp_scores']

    all_summary = {}

    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        model = build_model(model_name, args.dataset, device=str(device))
        model_id = model_name_to_id.get(model_name, 0)
        lambda_lscc = per_model_lambda[model_name]
        print(f"  model_id={model_id}, lambda_lscc={lambda_lscc:.4f}")

        pca_base_dir = os.path.join(args.data_dir, dataset_dir)
        pca_model_dir = os.path.join(pca_base_dir, f"activation_coverage_{model_name}")
        pca_path = os.path.join(pca_model_dir, "pca_models.pkl")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA models not found at {pca_path}")
        with open(pca_path, 'rb') as f:
            saved = pickle.load(f)
        pca_models_dict = {}
        scalers_dict = {}
        for key, pca_obj in saved['pca_models'].items():
            criterion = key.replace(f"{model_name}_", "", 1)
            pca_models_dict[criterion] = pca_obj
        for key, scaler_obj in saved['scalers'].items():
            criterion = key.replace(f"{model_name}_", "", 1)
            scalers_dict[criterion] = scaler_obj
        print(f"  PCA models: {sorted(pca_models_dict.keys())}")

        print(f"Loading seeds (filtering misclassified)...")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        seed_images, seed_labels = load_seeds(
            model, args.num_seeds, model_name,
            dataset_name=args.dataset, ds_mean=ds_mean, ds_std=ds_std)
        print(f"  Valid seeds: {len(seed_images)}")

        random_input = torch.randn(1, 3, image_size, image_size).to(device)
        layer_size_dict = tool.get_layer_output_sizes(model, random_input)
        crit_loader = build_crit_loader(
            seed_images, seed_labels, n=50, ds_mean=ds_mean, ds_std=ds_std)
        full_crit_loader = build_crit_loader(
            seed_images, seed_labels, n=len(seed_images), ds_mean=ds_mean, ds_std=ds_std)

        model_results = {}

        for strategy in strategies_to_run:
            model_results[strategy] = []

            for rep in range(args.num_repeats):
                model_subdir = os.path.join(args.output_dir, model_name)
                os.makedirs(model_subdir, exist_ok=True)
                save_path = os.path.join(
                    model_subdir,
                    f"{model_name}_{strategy}_rep{rep}.npz")
                flat_path = os.path.join(
                    args.output_dir,
                    f"{model_name}_{strategy}_rep{rep}.npz")
                cache_path = save_path if os.path.exists(save_path) else (
                    flat_path if os.path.exists(flat_path) else None)
                if cache_path is not None:
                    cached = dict(np.load(cache_path, allow_pickle=True))
                    for k, v in cached.items():
                        if isinstance(v, np.ndarray) and v.ndim == 0:
                            cached[k] = v.item()
                    model_results[strategy].append(cached)
                    print(f"  [{strategy}] repeat {rep+1}/{args.num_repeats}..."
                          f" FDR={cached['fdr']:.4f}  (cached)")
                    continue

                print(f"  [{strategy}] repeat {rep+1}/{args.num_repeats}...",
                      end='', flush=True)
                t0 = time.time()

                if strategy == 'cp_lsc_guided':
                    result = run_cp_lsc_guided(
                        model, seed_images, seed_labels,
                        layer_size_dict, crit_loader,
                        cp_model, model_id, pca_models_dict, scalers_dict,
                        criterion_names, pca_dim,
                        cluster_centers, cluster_cp_scores, n_clusters,
                        lambda_lscc,
                        args.max_tests, rng_seed=None,
                        ds_mean=ds_mean, ds_std=ds_std)
                elif strategy == 'random_walk':
                    result = run_random_walk(
                        model, seed_images, seed_labels,
                        args.max_tests, rng_seed=None,
                        ds_mean=ds_mean, ds_std=ds_std)
                else:
                    result = run_baseline(
                        strategy, model, seed_images, seed_labels,
                        layer_size_dict, crit_loader,
                        args.max_tests, rng_seed=None,
                        num_classes=num_classes,
                        accept_rate_cap=args.accept_rate_cap,
                        ds_mean=ds_mean, ds_std=ds_std,
                        hard_cap=args.hard_cap,
                        full_crit_loader=full_crit_loader)

                model_results[strategy].append(result)
                elapsed = time.time() - t0
                if strategy == 'cp_lsc_guided':
                    print(f" FDR={result['fdr']:.4f}  "
                          f"LSCC={result['lsc_final']:.4f}  "
                          f"AccRate={result['accept_rate']:.3f}  ({elapsed:.1f}s)")
                else:
                    print(f" FDR={result['fdr']:.4f}  "
                          f"AccRate={result['accept_rate']:.3f}  ({elapsed:.1f}s)")

                save_path = os.path.join(
                    model_subdir,
                    f"{model_name}_{strategy}_rep{rep}.npz")
                np.savez_compressed(save_path,
                    fdr=result['fdr'],
                    accept_rate=result['accept_rate'],
                    fdr_curve=result['fdr_curve'],
                    record_interval=result.get('record_interval', 100))

        print(f"\n{'='*70}")
        print(f"[{model_name}] FDR Comparison (Fault Detection Rate)")
        print(f"{'='*70}")
        print(f"\n{'Strategy':<20} {'FDR (Mean+/-Std)':>20} {'AccRate':>10} {'Description':<25}")
        print("-" * 80)

        summary = {}
        for strategy in strategies_to_run:
            fdrs = [r['fdr'] for r in model_results[strategy]]
            acc_rates = [r['accept_rate'] for r in model_results[strategy]]
            fdr_mean = np.mean(fdrs)
            fdr_std = np.std(fdrs)
            acc_mean = np.mean(acc_rates)
            fdr_str = f"{fdr_mean:.4f}+/-{fdr_std:.4f}"

            desc = STRATEGY_DESC.get(strategy, '')

            print(f"{strategy:<20} {fdr_str:>20} {acc_mean:>10.3f} {desc:<25}")

            summary[strategy] = {
                'fdr_mean': float(fdr_mean),
                'fdr_std':  float(fdr_std),
                'fdr_all':  [float(f) for f in fdrs],
                'accept_rate_mean': float(acc_mean),
            }

            if strategy == 'cp_lsc_guided':
                lscs = [r.get('lsc_final', 0) for r in model_results[strategy]]
                wlscs = [r.get('wlsc_final', 0) for r in model_results[strategy]]
                summary[strategy]['lsc_mean'] = float(np.mean(lscs))
                summary[strategy]['wlsc_mean'] = float(np.mean(wlscs))

        all_summary[model_name] = summary

    if len(model_names) > 1:
        print(f"\n{'='*70}")
        print(f"Cross-model FDR Summary")
        print(f"{'='*70}")
        header = f"{'Strategy':<20}"
        for mn in model_names:
            header += f" {mn:>15}"
        print(header)
        print("-" * (20 + 16 * len(model_names)))
        for strategy in strategies_to_run:
            row = f"{strategy:<20}"
            for mn in model_names:
                s = all_summary[mn][strategy]
                row += f" {s['fdr_mean']:.4f}+/-{s['fdr_std']:.4f}"
            print(row)

    summary_path = os.path.join(args.output_dir, "exp1_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing_summary = json.load(f)
        for model_name in all_summary:
            if model_name not in existing_summary:
                existing_summary[model_name] = {}
            existing_summary[model_name].update(all_summary[model_name])
        all_summary = existing_summary
    with open(summary_path, 'w') as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
