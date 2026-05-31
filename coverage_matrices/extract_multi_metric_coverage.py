import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'datasets'))
sys.path.insert(0, os.path.join(_ROOT, 'coverage_matrices'))

from image_transforms import (
    image_translation, image_scale, image_rotation,
    image_contrast, image_brightness, image_blur
)
import coverage
import tool

from shared_config import (
    build_model, get_dataset_config, get_model_transform,
    TRANSLATION_PARAMS, SCALE_PARAMS, ROTATION_PARAMS,
    CONTRAST_PARAMS, BRIGHTNESS_PARAMS, BLUR_PARAMS,
    build_cifar10_model, CIFAR10_MEAN, CIFAR10_STD,
)
from dataset_model_config import get_models_for_dataset
from coverage_feature_extractor import extract_coverage_features, init_coverage_criteria

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

translation_params = TRANSLATION_PARAMS
scale_params = SCALE_PARAMS
rotation_params = ROTATION_PARAMS
contrast_params = CONTRAST_PARAMS
brightness_params = BRIGHTNESS_PARAMS
blur_params = BLUR_PARAMS

transforms_list = [
    ("translation", image_translation, translation_params),
    ("scale",       image_scale,       scale_params),
    ("rotation",    image_rotation,    rotation_params),
    ("contrast",    image_contrast,    contrast_params),
    ("brightness",  image_brightness,  brightness_params),
    ("blur",        image_blur,        blur_params),
]

model_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

mutant_preprocess = model_transform

_current_dataset = 'CIFAR10'

def load_model(model_name, dataset_name='CIFAR10'):
    weight_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'models', dataset_name if dataset_name != 'CIFAR10' else 'CIFAR-10')
    return build_model(model_name, dataset_name, weight_dir=weight_dir, device=str(device))

def generate_mutants(image_pil, k, dataset_name='CIFAR10'):
    mutants = []
    img_np = np.array(image_pil).astype(np.float32)

    for _ in range(k):
        tname, tfunc, tparams = random.choice(transforms_list)
        param = random.choice(tparams)

        if tname == "translation":
            mutant_np = tfunc(img_np, param)
        else:
            mutant_np = tfunc(img_np, param)

        mutant_np = np.clip(mutant_np, 0, 255).astype(np.uint8)
        mutant_pil = Image.fromarray(mutant_np)
        mutants.append(mutant_preprocess(mutant_pil))

    return torch.stack(mutants, dim=0)

def main():
    global model_transform, mutant_preprocess, _current_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "MNIST", "Fashion-MNIST", "TinyImageNet"],
                        help="Dataset to extract coverage from")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specify model(s) or leave empty to use dataset's default models")
    parser.add_argument("--num_seeds", type=int, default=10000)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    dataset_name = args.dataset
    _current_dataset = dataset_name
    cfg = get_dataset_config(dataset_name)

    model_transform = get_model_transform(dataset_name)
    if dataset_name == 'MNIST':
        mutant_preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(cfg['image_size']),
            transforms.CenterCrop(cfg['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
    else:
        mutant_preprocess = model_transform

    output_dir = f"./coverage_matrices/{dataset_name}/multi_metric_coverage"
    os.makedirs(output_dir, exist_ok=True)

    model_names = get_models_for_dataset(dataset_name)
    if args.models is not None and len(args.models) > 0:
        model_names = args.models

    print(f"Models to process for {dataset_name}: {model_names}")

    image_dir = os.path.join("..", cfg['data_dir'], "train")
    class_list = sorted(os.listdir(image_dir))[:cfg['num_classes']]

    all_image_paths = []
    for cls in class_list:
        cls_dir = os.path.join(image_dir, cls)
        img_subdir = os.path.join(cls_dir, 'images') if os.path.isdir(os.path.join(cls_dir, 'images')) else cls_dir
        imgs = sorted([f for f in os.listdir(img_subdir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        all_image_paths += [os.path.join(img_subdir, x) for x in imgs]

    per_class_paths = {}
    for p in all_image_paths:
        cls = os.path.basename(os.path.dirname(os.path.dirname(p))) if 'images' in p else os.path.basename(os.path.dirname(p))
        per_class_paths.setdefault(cls, []).append(p)

    num_classes = len(per_class_paths)
    per_class_n = args.num_seeds // num_classes
    remainder = args.num_seeds % num_classes
    sampled = []
    for i, (cls, paths) in enumerate(sorted(per_class_paths.items())):
        random.shuffle(paths)
        n = per_class_n + (1 if i < remainder else 0)
        sampled.extend(paths[:n])
    random.shuffle(sampled)
    all_image_paths = sampled

    img_convert_mode = 'L' if dataset_name == 'MNIST' else 'RGB'

    for model_name in model_names:
        print(f"\n========== Processing {model_name} ({dataset_name}) ==========")
        model = load_model(model_name, dataset_name)

        input_size = (1, 3, cfg['image_size'], cfg['image_size'])
        random_input = torch.randn(input_size).to(device)
        layer_size_dict = tool.get_layer_output_sizes(model, random_input)

        print("Building train_loader for coverage criteria initialization...")
        train_image_dir = os.path.join("..", cfg['data_dir'], "train")
        train_class_list = sorted(os.listdir(train_image_dir))[:cfg['num_classes']]
        train_tensors = []
        train_labels = []
        for cls_idx, cls in enumerate(train_class_list):
            cls_dir = os.path.join(train_image_dir, cls)
            img_subdir = os.path.join(cls_dir, 'images') if os.path.isdir(os.path.join(cls_dir, 'images')) else cls_dir
            imgs = sorted([f for f in os.listdir(img_subdir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])[:200]
            for img_name in imgs:
                img_path = os.path.join(img_subdir, img_name)
                img = Image.open(img_path).convert(img_convert_mode)
                if dataset_name == 'MNIST':
                    img_tensor = mutant_preprocess(img)
                else:
                    img_tensor = model_transform(img)
                train_tensors.append(img_tensor)
                train_labels.append(cls_idx)
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack(train_tensors),
            torch.tensor(train_labels, dtype=torch.long)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=False
        )
        print(f"  Train loader: {len(train_tensors)} samples")

        criteria_dict = init_coverage_criteria(model, layer_size_dict,
                                                train_loader=train_loader,
                                                num_classes=cfg['num_classes'])

        print(f"Initialized {len(criteria_dict)} coverage criteria")

        first_seed_path = all_image_paths[0]
        try:
            seed_pil = Image.open(first_seed_path).convert(img_convert_mode)
            if dataset_name == 'MNIST':
                seed_pil_rgb = seed_pil.convert('RGB')
            else:
                seed_pil_rgb = seed_pil
        except Exception as e:
            raise RuntimeError(f"Failed to open first seed image: {first_seed_path}, err={e}")

        mutants_batch = generate_mutants(seed_pil_rgb, 1, dataset_name).to(device)
        sample_coverage = extract_coverage_features(mutants_batch, model, criteria_dict, seed_label=0)

        criterion_dims = {name: vec.shape[1] for name, vec in sample_coverage.items()}
        print(f"Coverage dimensions: {criterion_dims}")

        total_samples = len(all_image_paths) * args.K

        coverage_arrays = {}
        for name, dim in criterion_dims.items():
            coverage_arrays[name] = np.zeros((total_samples, dim), dtype=np.float32)

        labels_arr = np.zeros(total_samples, dtype=np.int64)
        faults_arr = np.zeros(total_samples, dtype=np.uint8)

        idx = 0
        skipped_wrong_pred = 0
        start_time = time.time()

        with torch.inference_mode():
            for seed_path in tqdm(all_image_paths, desc=f"{model_name} seeds"):
                try:
                    seed_pil = Image.open(seed_path).convert(img_convert_mode)
                    if dataset_name == 'MNIST':
                        seed_pil_rgb = seed_pil.convert('RGB')
                    else:
                        seed_pil_rgb = seed_pil
                except Exception as e:
                    print(f"[WARN] Skip corrupted image: {seed_path}, err={e}")
                    continue

                _parent = os.path.dirname(seed_path)
                label_name = os.path.basename(_parent)
                if label_name == 'images':
                    label_name = os.path.basename(os.path.dirname(_parent))
                seed_label = class_list.index(label_name)
                seed_label_tensor = torch.tensor([seed_label], device=device)

                if dataset_name == 'MNIST':
                    seed_tensor = mutant_preprocess(seed_pil).unsqueeze(0).to(device)
                else:
                    seed_tensor = model_transform(seed_pil_rgb).unsqueeze(0).to(device)
                seed_out = model(seed_tensor)
                seed_pred = seed_out.argmax(dim=1).item()

                if seed_pred != seed_label:
                    skipped_wrong_pred += 1
                    continue

                mutants = generate_mutants(seed_pil_rgb, args.K, dataset_name).to(device)
                mutant_out = model(mutants)
                mutant_pred = mutant_out.argmax(dim=1)

                fault_flags = (mutant_pred != seed_label_tensor.repeat(args.K)).cpu().numpy()

                multi_cov = extract_coverage_features(mutants, model, criteria_dict, seed_label=seed_label)

                for name, cov_vec in multi_cov.items():
                    coverage_arrays[name][idx:idx+args.K] = cov_vec

                labels_arr[idx:idx+args.K] = seed_label
                faults_arr[idx:idx+args.K] = fault_flags

                idx += args.K

        elapsed = time.time() - start_time
        total_seeds_processed = idx // args.K if idx > 0 else 0
        total_seeds_attempted = len(all_image_paths)
        filter_ratio = skipped_wrong_pred / total_seeds_attempted if total_seeds_attempted > 0 else 0

        print(f"\n{model_name} completed in {elapsed:.2f}s")
        print(f"Total mutant samples: {idx}")
        print(f"Seeds processed (model correct): {total_seeds_processed}")
        print(f"Seeds skipped (model already wrong): {skipped_wrong_pred}")
        print(f"Filter ratio: {filter_ratio:.4f} ({skipped_wrong_pred}/{total_seeds_attempted})")

        print(f"Saving {idx} valid samples...")
        for name in coverage_arrays:
            save_path = os.path.join(output_dir, f"{model_name}_{name}_coverage.npy")
            np.save(save_path, coverage_arrays[name][:idx])
        labels_path = os.path.join(output_dir, f"{model_name}_labels.npy")
        fault_path = os.path.join(output_dir, f"{model_name}_fault_flags.npy")
        np.save(labels_path, labels_arr[:idx])
        np.save(fault_path, faults_arr[:idx])

        meta_path = os.path.join(output_dir, f"{model_name}_meta.npz")
        np.savez(
            meta_path,
            model_name=model_name,
            num_samples=idx,
            num_seeds_processed=total_seeds_processed,
            num_seeds_skipped=skipped_wrong_pred,
            num_seeds_attempted=total_seeds_attempted,
            filter_ratio=filter_ratio,
            criterion_dims=criterion_dims,
            criterion_names=list(criterion_dims.keys())
        )

        print(f"Saved metadata to {meta_path}")

        import pickle
        criteria_state_path = os.path.join(output_dir, f"{model_name}_criteria_state.pkl")
        criteria_state = {}
        for name in ['KMNC', 'NBC', 'SNAC']:
            if name in criteria_dict:
                criteria_state[name] = {
                    'range_dict': criteria_dict[name].range_dict
                }
        with open(criteria_state_path, 'wb') as f:
            pickle.dump(criteria_state, f)
        print(f"Saved criteria states to {criteria_state_path}")

if __name__ == "__main__":
    main()
