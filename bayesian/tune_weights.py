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

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import coverage
import tool
import data_loader
from shared_config import build_model, get_dataset_config, build_cifar10_model
from coverage_feature_extractor import (
    extract_pca_coverage, load_pca_models, init_coverage_criteria
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import image_transforms
from shared_config import (
    TRANSLATION_PARAMS, SCALE_PARAMS, ROTATION_PARAMS,
    CONTRAST_PARAMS, BRIGHTNESS_PARAMS, BLUR_PARAMS,
    CIFAR10_MEAN, CIFAR10_STD, get_dataset_config,
)

transforms_list = [
    ("translation", image_transforms.image_translation, TRANSLATION_PARAMS),
    ("scale", image_transforms.image_scale, SCALE_PARAMS),
    ("rotation", image_transforms.image_rotation, ROTATION_PARAMS),
    ("contrast", image_transforms.image_contrast, CONTRAST_PARAMS),
    ("brightness", image_transforms.image_brightness, BRIGHTNESS_PARAMS),
    ("blur", image_transforms.image_blur, BLUR_PARAMS),
]

VALIDITY_ALPHA = 0.4
VALIDITY_BETA = 0.8
VALIDITY_TRY_NUM = 50


def mutate_image(img_np):
    tname, tfunc, tparams = random.choice(transforms_list)
    param = random.choice(tparams)
    mutant = tfunc(img_np, param)
    return np.clip(mutant, 0, 255).astype(np.float32)


def is_valid_mutation(I0, I_new):
    changed = np.sum((I0 - I_new) != 0)
    nonzero = np.sum(I0 > 0)
    max_diff = np.max(np.abs(I0 - I_new))
    if changed < VALIDITY_ALPHA * nonzero:
        return max_diff <= 255
    return max_diff <= VALIDITY_BETA * 255


def mutate_with_validity(current_img, seed_img):
    for _ in range(VALIDITY_TRY_NUM):
        mutant_np = mutate_image(current_img)
        if is_valid_mutation(seed_img, mutant_np):
            return mutant_np, True
    return current_img.copy(), False


def image_to_tensor(img_np, dataset_name='CIFAR10'):
    cfg = get_dataset_config(dataset_name)
    img = img_np / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1)
    mean = torch.tensor(cfg['mean']).view(3, 1, 1)
    std = torch.tensor(cfg['std']).view(3, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0).to(device)


def load_seeds(model_name, num_seeds, model, dataset_name='CIFAR10'):
    cfg = get_dataset_config(dataset_name)
    num_classes = cfg['num_classes']
    seeds_per_class = num_seeds // num_classes

    class FakeArgs:
        dataset = dataset_name
        image_size = cfg['image_size']
        num_class = num_classes
        num_per_class = num_seeds // 5
        nc = 3
        batch_size = 50
        num_workers = 0
    FakeArgs.model = model_name

    if dataset_name == 'MNIST':
        data_set = data_loader.MNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'Fashion-MNIST':
        data_set = data_loader.FashionMNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'TinyImageNet':
        data_set = data_loader.ImageNetFuzzDataset(FakeArgs(), image_dir=cfg['data_dir'], split='val')
    else:
        data_set = data_loader.CIFAR10FuzzDataset(FakeArgs(), split='test')

    images_per_class = {i: [] for i in range(num_classes)}
    total = data_set.get_len()

    for i in range(total):
        img_tensor, label_tensor = data_set.get_item(i)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.float32)
        lbl = label_tensor.item()

        if len(images_per_class[lbl]) >= seeds_per_class:
            continue

        t = image_to_tensor(img_np, dataset_name)
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

    print(f"  Seed distribution (expected {seeds_per_class} per class):")
    for cls in range(num_classes):
        count = len(images_per_class[cls])
        status = "ok" if count == seeds_per_class else f"missing ({count})"
        print(f"    class {cls}: {count:3d} {status}")

    total_collected = sum(len(images_per_class[cls]) for cls in range(num_classes))
    print(f"  Total: {total_collected}/{num_seeds} seeds")

    return images, labels


def evaluate_weights(lambda_lscc,
                     model, seed_images, seed_labels,
                     cp_model, model_id, pca_models, scalers,
                     criterion_names, pca_dim, criteria_dict,
                     make_tracker, max_tests, rng_seed=42,
                     dataset_name='CIFAR10'):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    tracker = make_tracker()
    total_tests = 0
    total_faults = 0
    iters_per_seed = max_tests // len(seed_images)

    for seed_idx in range(len(seed_images)):
        if total_tests >= max_tests:
            break
        seed_img = seed_images[seed_idx]
        seed_label = seed_labels[seed_idx]
        current_img = seed_img.copy()
        current_objective = 0.0
        current_cp_score = 0.0
        current_lscc_coverage = 0.0

        for _ in range(iters_per_seed):
            if total_tests >= max_tests:
                break
            mutant_np = mutate_image(current_img)
            input_tensor = image_to_tensor(mutant_np, dataset_name)
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
            tracker.update(pca_feat, cp_score)
            lscc_coverage = tracker.get_coverage()

            objective = cp_score + lambda_lscc * lscc_coverage

            if cp_score > current_cp_score or lscc_coverage > current_lscc_coverage:
                current_img = mutant_np
                current_objective = objective
                current_cp_score = cp_score
                current_lscc_coverage = lscc_coverage
            elif random.random() < 0.1:
                current_img = mutant_np
                current_objective = objective
                current_cp_score = cp_score
                current_lscc_coverage = lscc_coverage

    fdr = total_faults / total_tests if total_tests > 0 else 0
    lsc = tracker.get_coverage()
    return fdr, lsc


def main():
    parser = argparse.ArgumentParser(description='Bayesian weight optimization search')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'MNIST', 'Fashion-MNIST', 'TinyImageNet'])
    parser.add_argument('--model', type=str, default='all',
                        help='Model name or all (auto-fetched from dataset_model_config)')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--max_tests', type=int, default=3000)
    parser.add_argument('--n_calls', type=int, default=40,
                        help='Number of Bayesian optimization iterations')
    parser.add_argument('--n_initial', type=int, default=10,
                        help='Number of initial random exploration steps')
    parser.add_argument('--cp_checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset_name = args.dataset
    cfg = get_dataset_config(dataset_name)
    tag = cfg['weight_tag']

    if args.num_seeds == 100 and dataset_name == 'TinyImageNet':
        args.num_seeds = 400
        print(f"[Auto-adjust] TinyImageNet seed count: 100 -> 400 (200 classes x 2)")

    dataset_dir_map = {
        'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST', 'Fashion-MNIST': 'Fashion-MNIST', 'TinyImageNet': 'TinyImageNet',
    }
    dataset_dir = dataset_dir_map.get(dataset_name, dataset_name)

    if args.cp_checkpoint is None:
        possible_paths = [
            f'./cp_checkpoints/{dataset_dir}/best_model.pt',
            f'./cp_checkpoints/{dataset_name}/best_model.pt',
            f'./cp_checkpoints_{tag}/best_model.pt',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                args.cp_checkpoint = p
                break
        if args.cp_checkpoint is None:
            args.cp_checkpoint = possible_paths[0]
    if args.data_dir is None:
        args.data_dir = './coverage_matrices'

    from cp_model import CoverageTransformer, LatentSpaceClusterCoverage
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    from dataset_model_config import get_models_for_dataset

    if args.model == 'all':
        model_names = get_models_for_dataset(dataset_name)
    else:
        model_names = [args.model]

    print(f"{'='*70}")
    print(f"Bayesian weight optimization (simplified: CP-dominant + LSCC auxiliary)")
    print(f"Models: {model_names}")
    print(f"Iterations: {args.n_calls} (initial random: {args.n_initial})")
    print(f"Budget: {args.max_tests}, seeds: {args.num_seeds}")
    print(f"{'='*70}")

    print("\nLoading CP model...")
    checkpoint = torch.load(args.cp_checkpoint, map_location=device, weights_only=False)
    cp_args = checkpoint["args"]
    criterion_names = checkpoint.get("criterion_names", cp_args.get(
        "criterion_names",
        ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC',
         'LSC', 'DSC']))
    pca_dim = cp_args["pca_dim"]
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

    cluster_path = os.path.join(os.path.dirname(args.cp_checkpoint),
                                "latent_space_clusters.npz")
    cluster_data = np.load(cluster_path)
    cluster_centers = cluster_data["cluster_centers"]
    cluster_cp_scores = cluster_data["cluster_cp_scores"]
    n_clusters = int(cluster_data["n_clusters"])
    print(f"  pca_dim={pca_dim}, clusters={n_clusters}, num_models={cp_num_models}")

    from dataset_model_config import MODEL_NAME_TO_ID
    model_name_to_id = MODEL_NAME_TO_ID.get(dataset_name, {})
    if not model_name_to_id:
        model_name_to_id = {name: idx for idx, name in enumerate(model_names)}

    space = [
        Real(0.0, 2.0, name='lambda_lscc', prior='uniform'),
    ]

    all_results = {}

    for model_name in model_names:
        print(f"\n{'#'*70}")
        print(f"# Optimizing model: {model_name}")
        print(f"{'#'*70}")

        model = build_model(model_name, dataset_name, device=str(device))
        seed_images, seed_labels = load_seeds(
            model_name, args.num_seeds, model, dataset_name)
        print(f"  Seeds: {len(seed_images)}")

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

        random_input = torch.randn(1, 3, cfg['image_size'], cfg['image_size']).to(device)
        layer_size_dict = tool.get_layer_output_sizes(model, random_input)
        crit_tensors, crit_labels_list = [], []
        for img_np, lbl in zip(seed_images[:50], seed_labels[:50]):
            crit_tensors.append(image_to_tensor(img_np).squeeze(0))
            crit_labels_list.append(lbl)
        crit_batch = torch.stack(crit_tensors)
        crit_dataset = TensorDataset(
            crit_batch, torch.tensor(crit_labels_list, dtype=torch.long))
        crit_loader = TorchDataLoader(
            crit_dataset, batch_size=32, shuffle=False)
        criteria_dict = init_coverage_criteria(
            model, layer_size_dict, train_loader=None,
            device=device, num_classes=cfg['num_classes'],
            skip_sa_criteria=True)
        print(f"  Coverage criteria initialized: {list(criteria_dict.keys())}")

        model_id = model_name_to_id[model_name]
        print(f"  Model ID: {model_id}")

        def make_tracker():
            return LatentSpaceClusterCoverage(
                n_clusters=n_clusters,
                cluster_centers=cluster_centers.copy(),
                cluster_cp_scores=cluster_cp_scores.copy(),
                lambda_param=0.7, risk_threshold=0.3,
            )
        print(f"  Tracker factory created")

        iteration_log = []
        call_count = [0]
        print(f"  Iteration log initialized")

        print(f"  Defining objective function...")
        @use_named_args(space)
        def objective(lambda_lscc):
            call_count[0] += 1
            t0 = time.time()
            iter_seed = args.seed + call_count[0]
            fdr, lsc = evaluate_weights(
                lambda_lscc,
                model, seed_images, seed_labels,
                cp_model, model_id, pca_models_dict, scalers_dict,
                criterion_names, pca_dim, criteria_dict,
                make_tracker, args.max_tests, rng_seed=iter_seed,
                dataset_name=dataset_name)
            elapsed = time.time() - t0
            iteration_log.append({
                'iter': call_count[0],
                'lambda_lscc': float(lambda_lscc),
                'fdr': fdr, 'lsc': lsc, 'time': elapsed,
            })
            print(f"  [{call_count[0]:3d}/{args.n_calls}] "
                  f"lambda={lambda_lscc:.4f} -> FDR={fdr:.4f}  LSCC={lsc:.4f}  ({elapsed:.1f}s)")
            return -fdr
        print(f"  Objective function defined")

        print(f"\nStarting Bayesian optimization ({args.n_calls} iterations)...")
        result = gp_minimize(
            objective, space,
            n_calls=args.n_calls,
            n_initial_points=args.n_initial,
            random_state=args.seed,
            verbose=False,
            acq_func='EI',
        )

        best_lambda = result.x[0]
        best_fdr = -result.fun

        print(f"\n{'='*70}")
        print(f"[{model_name}] Best weights:")
        print(f"  lambda_LSCC = {best_lambda:.4f}")
        print(f"  FDR = {best_fdr:.4f}")

        print(f"\nValidating best weights (3 different random seeds)...")
        val_fdrs = []
        val_lscs = []
        for s in [42, 123, 456]:
            fdr_v, lsc_v = evaluate_weights(
                best_lambda,
                model, seed_images, seed_labels,
                cp_model, model_id, pca_models_dict, scalers_dict,
                criterion_names, pca_dim, criteria_dict,
                make_tracker, args.max_tests, rng_seed=s,
                dataset_name=dataset_name)
            val_fdrs.append(fdr_v)
            val_lscs.append(lsc_v)
            print(f"  seed={s}: FDR={fdr_v:.4f}, LSC={lsc_v:.4f}")
        mean_fdr = np.mean(val_fdrs)
        std_fdr = np.std(val_fdrs)
        mean_lsc = np.mean(val_lscs)
        print(f"  Mean FDR = {mean_fdr:.4f} +/- {std_fdr:.4f}")
        print(f"  Mean LSCC = {mean_lsc:.4f}")

        all_results[model_name] = {
            'best_lambda_lscc': float(best_lambda),
            'best_fdr': float(best_fdr),
            'val_fdr_mean': float(mean_fdr),
            'val_fdr_std': float(std_fdr),
            'val_lsc_mean': float(mean_lsc),
            'iteration_log': iteration_log,
        }

    print(f"\n{'='*70}")
    print(f"Cross-model best weights summary")
    print(f"{'='*70}")
    print(f"\n{'Model':<15} {'lambda_LSCC':>10} {'FDR':>8} {'Val FDR':>12}")
    print("-" * 50)
    for mn in model_names:
        r = all_results[mn]
        print(f"{mn:<15} {r['best_lambda_lscc']:>10.4f} "
              f"{r['best_fdr']:>8.4f} "
              f"{r['val_fdr_mean']:.4f}+/-{r['val_fdr_std']:.4f}")

    output_path = f"./bayesian_optimal_weights_{tag}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

