import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'datasets'))
sys.path.insert(0, os.path.join(_ROOT, 'coverage_matrices'))

from shared_config import build_model, get_dataset_config
from coverage_feature_extractor import extract_pca_coverage, load_pca_models
from cp_model import CoverageTransformer
from dataset_model_config import MODEL_NAME_TO_ID
import data_loader
import coverage
import tool
import image_transforms
from shared_config import (
    TRANSLATION_PARAMS, SCALE_PARAMS, ROTATION_PARAMS,
    CONTRAST_PARAMS, BRIGHTNESS_PARAMS, BLUR_PARAMS,
)

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


def image_to_tensor(img_np, ds_mean, ds_std):
    img = img_np / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1)
    mean = torch.tensor(ds_mean).view(3, 1, 1)
    std = torch.tensor(ds_std).view(3, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0).to(device)


def load_seeds(model_name, num_seeds, model, dataset_name='CIFAR10'):
    cfg = get_dataset_config(dataset_name)
    num_classes = cfg['num_classes']
    seeds_per_class = max(1, num_seeds // num_classes)
    ds_mean = cfg['mean']
    ds_std = cfg['std']

    class FakeArgs:
        dataset = dataset_name
        image_size = cfg['image_size']
        num_class = num_classes
        num_per_class = num_seeds // 5
        nc = cfg['channels']
        batch_size = 50
        num_workers = 0
    FakeArgs.model = model_name

    if dataset_name == 'MNIST':
        data_set = data_loader.MNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'Fashion-MNIST':
        data_set = data_loader.FashionMNISTFuzzDataset(FakeArgs(), split='test')
    elif dataset_name == 'TinyImageNet':
        data_set = data_loader.ImageNetFuzzDataset(
            FakeArgs(), image_dir=cfg['data_dir'], split='val')
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

        t = image_to_tensor(img_np, ds_mean, ds_std)
        with torch.no_grad():
            pred = model(t).argmax(dim=1).item()
        if pred == lbl:
            images_per_class[lbl].append((img_np, lbl))

        del t
        if i % 100 == 0:
            torch.cuda.empty_cache()

        if all(len(v) >= seeds_per_class for v in images_per_class.values()):
            break

    images, labels = [], []
    for cls_id in range(num_classes):
        for img_np, lbl in images_per_class[cls_id]:
            images.append(img_np)
            labels.append(lbl)

    print(f"Loaded {len(images)} seed samples")
    return images, labels


def run_correlation_analysis(dataset_name, model_name, num_seeds=100, max_tests=3000):

    print(f"\n{'='*60}")
    print(f"Experiment 4: Correlation Analysis")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Seeds: {num_seeds}, Max tests: {max_tests}")
    print(f"{'='*60}\n")

    cfg = get_dataset_config(dataset_name)
    ds_mean = cfg['mean']
    ds_std = cfg['std']

    print("1. Loading DNN model...")
    model = build_model(model_name, dataset_name)
    model.to(device)
    model.eval()

    print("2. Loading CP predictor...")
    cp_dataset_name = 'CIFAR-10' if dataset_name == 'CIFAR10' else dataset_name
    cp_checkpoint = f'./cp_checkpoints/{cp_dataset_name}/best_model.pt'

    ckpt = torch.load(cp_checkpoint, map_location=device)

    state_dict = ckpt['model_state_dict']
    if 'model_config' in ckpt:
        cfg_cp = ckpt['model_config']
        d_model = cfg_cp.get('d_model', 128)
        pca_dim = cfg_cp.get('pca_dim', 64)
        num_layers = cfg_cp.get('num_layers', 3)
        nhead = cfg_cp.get('nhead', 4)
        num_criteria = cfg_cp.get('num_criteria', 10)
        num_models = cfg_cp.get('num_models', 6)
    else:
        d_model = state_dict['criterion_proj.weight'].shape[0]
        pca_dim = state_dict['criterion_proj.weight'].shape[1]
        num_criteria = state_dict['criterion_embed.weight'].shape[0]
        num_models = state_dict['model_embed.weight'].shape[0]
        nhead = 4
        num_layers = 0
        for key in state_dict.keys():
            if 'transformer.layers.' in key:
                layer_num = int(key.split('.')[2])
                num_layers = max(num_layers, layer_num + 1)

    print(f"  CP model config: d_model={d_model}, pca_dim={pca_dim}, num_layers={num_layers}, num_criteria={num_criteria}")

    cp_model = CoverageTransformer(
        pca_dim=pca_dim,
        num_criteria=num_criteria,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1,
        num_models=num_models
    ).to(device)
    cp_model.load_state_dict(ckpt['model_state_dict'])
    cp_model.eval()

    print("3. Loading PCA models...")
    cp_dataset_name = 'CIFAR-10' if dataset_name == 'CIFAR10' else dataset_name
    pca_data_dir = f'./coverage_matrices/{cp_dataset_name}'
    pca_models, scalers = load_pca_models(model_name, pca_data_dir)

    print("4. Initializing coverage criteria...")
    random_input = torch.randn(1, cfg['channels'], cfg['image_size'], cfg['image_size']).to(device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_input)

    criteria_dict = {
        'NC': coverage.NC(model, layer_size_dict, hyper=0.5),
        'NBC': coverage.NBC(model, layer_size_dict, hyper=None),
        'SNAC': coverage.SNAC(model, layer_size_dict, hyper=None),
        'KMNC': coverage.KMNC(model, layer_size_dict, hyper=100),
        'TKNC': coverage.TKNC(model, layer_size_dict, hyper=3),
        'TKNP': coverage.TKNP(model, layer_size_dict, hyper=3),
        'CC': coverage.CC(model, layer_size_dict, hyper=10),
        'NLC': coverage.NLC(model, layer_size_dict, hyper=None),
        'LSC': coverage.LSC(model, layer_size_dict, hyper=2000, min_var=1e-5, num_class=cfg['num_classes']),
        'DSC': coverage.DSC(model, layer_size_dict, hyper=1000, min_var=1e-5, num_class=cfg['num_classes']),
    }

    print("  Building KMNC, NBC, SNAC, LSC, DSC...")
    temp_seeds, temp_labels = load_seeds(model_name, 50, model, dataset_name)
    build_tensors = []
    for img_np in temp_seeds:
        t = image_to_tensor(img_np, ds_mean, ds_std)
        build_tensors.append(t.squeeze(0))
    build_batch = torch.stack(build_tensors)
    build_dataset = torch.utils.data.TensorDataset(build_batch, torch.tensor(temp_labels, dtype=torch.long))
    build_loader = torch.utils.data.DataLoader(build_dataset, batch_size=32, shuffle=False)
    criteria_dict['KMNC'].build(build_loader)
    criteria_dict['NBC'].build(build_loader)
    criteria_dict['SNAC'].build(build_loader)
    criteria_dict['LSC'].build(build_loader)
    criteria_dict['DSC'].build(build_loader)
    del temp_seeds, temp_labels, build_tensors, build_batch, build_dataset, build_loader
    torch.cuda.empty_cache()

    print("5. Loading seed samples...")
    seed_images, seed_labels = load_seeds(model_name, num_seeds, model, dataset_name)

    print("6. Starting mutation testing...")
    sample_records = []

    max_per_seed = max_tests // len(seed_images)

    for seed_idx, (seed_img, seed_label) in enumerate(tqdm(zip(seed_images, seed_labels),
                                                             total=len(seed_images),
                                                             desc="Mutating seeds")):
        for mut_idx in range(max_per_seed):
            mutant_np = mutate_image(seed_img)
            mutant_tensor = image_to_tensor(mutant_np, ds_mean, ds_std)

            with torch.no_grad():
                pred = model(mutant_tensor).argmax(dim=1).item()

            is_fault = (pred != seed_label)

            with torch.no_grad():
                criterion_names = ['NC', 'NBC', 'SNAC', 'KMNC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']
                cov_tensor = extract_pca_coverage(
                    model, mutant_tensor, pca_models, scalers,
                    criteria_dict, criterion_names, pca_dim=pca_dim,
                    seed_label=seed_label
                )
                cov_tensor = cov_tensor.to(device)
                model_id = MODEL_NAME_TO_ID.get(dataset_name, {}).get(model_name, 0)
                model_ids = torch.tensor([model_id], dtype=torch.long, device=device)
                logits = cp_model(cov_tensor, model_ids)
                cp_score = torch.sigmoid(logits).item()

            coverage_scores = {}
            for crit_name, crit in criteria_dict.items():
                if crit_name in ['LSC', 'DSC']:
                    cove_dict = crit.calculate(mutant_tensor, torch.tensor([pred]).to(device))
                else:
                    cove_dict = crit.calculate(mutant_tensor)

                gain = crit.gain(cove_dict)
                score = gain if gain is not None else 0.0

                if gain is not None:
                    crit.update(cove_dict, gain)

                coverage_scores[crit_name] = score

            record = {
                'seed_idx': seed_idx,
                'mut_idx': mut_idx,
                'is_fault': int(is_fault),
                'cp_score': cp_score,
                **coverage_scores
            }
            sample_records.append(record)

    print(f"\nCollected {len(sample_records)} sample records")

    print("\n" + "="*60)
    print("Experiment A: Sample-level Fault Relevance (AUC-ROC, AUC-PR)")
    print("="*60)

    y_true = np.array([r['is_fault'] for r in sample_records])

    results_a = {}

    cp_scores = np.array([r['cp_score'] for r in sample_records])
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, cp_scores)
        auc_pr = average_precision_score(y_true, cp_scores)
        results_a['CP'] = {'AUC-ROC': auc_roc, 'AUC-PR': auc_pr}
        print(f"CP:    AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

    for crit_name in ['NC', 'NBC', 'SNAC', 'KMNC', 'TKNC', 'TKNP', 'CC']:
        scores = np.array([r[crit_name] for r in sample_records])
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, scores)
            auc_pr = average_precision_score(y_true, scores)
            results_a[crit_name] = {'AUC-ROC': auc_roc, 'AUC-PR': auc_pr}
            print(f"{crit_name:6s} AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

    print("\n" + "="*60)
    print("Experiment B: Cluster-level Spearman Correlation with Fault Rate")
    print("="*60)

    cp_scores_array = np.array([r['cp_score'] for r in sample_records]).reshape(-1, 1)
    n_clusters = 30
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cp_scores_array)

    cluster_stats = defaultdict(lambda: {
        'cp_scores': [],
        'faults': [],
        'nc_scores': [],
        'nbc_scores': [],
        'snac_scores': [],
        'kmnc_scores': [],
        'tknc_scores': [],
        'tknp_scores': [],
        'cc_scores': [],
    })

    for i, record in enumerate(sample_records):
        cluster_id = cluster_labels[i]
        cluster_stats[cluster_id]['cp_scores'].append(record['cp_score'])
        cluster_stats[cluster_id]['faults'].append(record['is_fault'])
        cluster_stats[cluster_id]['nc_scores'].append(record['NC'])
        cluster_stats[cluster_id]['nbc_scores'].append(record['NBC'])
        cluster_stats[cluster_id]['snac_scores'].append(record['SNAC'])
        cluster_stats[cluster_id]['kmnc_scores'].append(record['KMNC'])
        cluster_stats[cluster_id]['tknc_scores'].append(record['TKNC'])
        cluster_stats[cluster_id]['tknp_scores'].append(record['TKNP'])
        cluster_stats[cluster_id]['cc_scores'].append(record['CC'])

    cluster_data = []
    for cluster_id in range(n_clusters):
        stats = cluster_stats[cluster_id]
        if len(stats['faults']) > 0:
            cluster_data.append({
                'cluster_id': cluster_id,
                'cp_mean': np.mean(stats['cp_scores']),
                'fault_rate': np.mean(stats['faults']),
                'nc_mean': np.mean(stats['nc_scores']),
                'nbc_mean': np.mean(stats['nbc_scores']),
                'snac_mean': np.mean(stats['snac_scores']),
                'kmnc_mean': np.mean(stats['kmnc_scores']),
                'tknc_mean': np.mean(stats['tknc_scores']),
                'tknp_mean': np.mean(stats['tknp_scores']),
                'cc_mean': np.mean(stats['cc_scores']),
            })

    fault_rates = np.array([c['fault_rate'] for c in cluster_data])

    results_b = {}

    cp_means = np.array([c['cp_mean'] for c in cluster_data])
    rho, pval = spearmanr(cp_means, fault_rates)
    results_b['CP'] = {'Spearman': rho, 'p-value': pval}
    print(f"CP:    Spearman rho={rho:.4f}, p={pval:.4f}")

    for crit_name in ['nc', 'nbc', 'snac', 'kmnc', 'tknc', 'tknp', 'cc']:
        means = np.array([c[f'{crit_name}_mean'] for c in cluster_data])
        rho, pval = spearmanr(means, fault_rates)
        results_b[crit_name.upper()] = {'Spearman': rho, 'p-value': pval}
        print(f"{crit_name.upper():6s} Spearman rho={rho:.4f}, p={pval:.4f}")

    output_dir = f'./exp4_results/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_file = f'{output_dir}/{model_name}_correlation.json'
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'num_seeds': num_seeds,
        'max_tests': max_tests,
        'num_samples': len(sample_records),
        'fault_rate': float(np.mean(y_true)),
        'experiment_a': results_a,
        'experiment_b': results_b,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    debug_file = f'{output_dir}/{model_name}_samples_debug.npz'
    np.savez(debug_file,
             cp_scores=np.array([r['cp_score'] for r in sample_records]),
             is_fault=np.array([r['is_fault'] for r in sample_records]),
             nc_scores=np.array([r['NC'] for r in sample_records]),
             cc_scores=np.array([r['CC'] for r in sample_records]))
    print(f"Debug data saved to: {debug_file}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--max_tests', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_correlation_analysis(
        dataset_name=args.dataset,
        model_name=args.model,
        num_seeds=args.num_seeds,
        max_tests=args.max_tests
    )
