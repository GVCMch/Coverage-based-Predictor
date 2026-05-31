import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'datasets'))
sys.path.insert(0, os.path.join(_ROOT, 'coverage_matrices'))

from image_transforms import (
    image_translation, image_scale, image_rotation,
    image_contrast, image_brightness, image_blur
)
import tool

from cp_model import CoverageTransformer, LatentSpaceClusterCoverage

from coverage_feature_extractor import (
    extract_pca_coverage as _extract_pca_coverage_shared,
    load_pca_models,
    init_coverage_criteria,
)

from shared_config import (
    build_model, get_dataset_config, get_model_transform,
    TRANSLATION_PARAMS, SCALE_PARAMS, ROTATION_PARAMS,
    CONTRAST_PARAMS, BRIGHTNESS_PARAMS, BLUR_PARAMS,
    CIFAR10_MEAN, CIFAR10_STD,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_transform = None
mutant_preprocess = None

translation_params = TRANSLATION_PARAMS
scale_params = SCALE_PARAMS
rotation_params = ROTATION_PARAMS
contrast_params = CONTRAST_PARAMS
brightness_params = BRIGHTNESS_PARAMS
blur_params = BLUR_PARAMS

transforms_list = [
    ("translation", image_translation, translation_params),
    ("scale", image_scale, scale_params),
    ("rotation", image_rotation, rotation_params),
    ("contrast", image_contrast, contrast_params),
    ("brightness", image_brightness, brightness_params),
    ("blur", image_blur, blur_params),
]

def load_model(model_name, dataset_name='CIFAR10'):
    return build_model(model_name, dataset_name, device=str(device))

def extract_pca_coverage(model, data_batch, pca_models, scalers, pca_dim=256,
                         criterion_names=None, criteria_dict=None, seed_label=None):
    if criterion_names is None:
        criterion_names = ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']
    if criteria_dict is None:
        raise ValueError(
            "criteria_dict is required for extract_pca_coverage. "
            "Pass the initialized coverage criteria objects."
        )
    return _extract_pca_coverage_shared(
        model, data_batch, pca_models, scalers,
        criteria_dict, criterion_names, pca_dim,
        seed_label=seed_label
    )

def mutate_image(image_pil):
    img_np = np.array(image_pil).astype(np.float32)

    tname, tfunc, tparams = random.choice(transforms_list)
    param = random.choice(tparams)

    if tname == "translation":
        dx, dy = random.choice(translation_params)
        mutant_np = tfunc(img_np, (dx, dy))
    else:
        mutant_np = tfunc(img_np, param)

    mutant_np = np.clip(mutant_np, 0, 255).astype(np.uint8)

    mutant_pil = Image.fromarray(mutant_np)
    return mutant_pil

VALIDITY_ALPHA = 0.4
VALIDITY_BETA = 0.8
VALIDITY_TRY_NUM = 50

def is_valid_mutation_pil(seed_pil, mutant_pil, alpha=VALIDITY_ALPHA, beta=VALIDITY_BETA):
    I0 = np.array(seed_pil).astype(np.float32)
    I_new = np.array(mutant_pil).astype(np.float32)

    changed_count = np.sum((I0 - I_new) != 0)
    nonzero_count = np.sum(I0 > 0)
    max_abs_diff = np.max(np.abs(I0 - I_new))

    if changed_count < alpha * nonzero_count:
        return max_abs_diff <= 255
    else:
        return max_abs_diff <= beta * 255

def mutate_with_validity_pil(current_pil, seed_pil, max_tries=VALIDITY_TRY_NUM):
    for _ in range(max_tries):
        mutant_pil = mutate_image(current_pil)
        if is_valid_mutation_pil(seed_pil, mutant_pil):
            return mutant_pil, True
    return current_pil.copy(), False

class CPPCAFuzzer:

    def __init__(
        self,
        model,
        cp_model,
        pca_models,
        scalers,
        model_id,
        criteria_dict,
        pca_dim=256,
        criterion_names=None,
        calibration_interval=100,
        cluster_centers=None,
        cluster_cp_scores=None,
        n_clusters=50,
        max_queue_size=1000,
        max_exploration_seeds=5,
        risk_threshold=0.5,
        lambda_param=0.7,
    ):
        self.model = model
        self.cp_model = cp_model
        self.pca_models = pca_models
        self.scalers = scalers
        self.model_id = model_id
        self.criteria_dict = criteria_dict
        self.pca_dim = pca_dim
        self.criterion_names = criterion_names or ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']
        self.lambda_param = lambda_param

        if cluster_centers is None:
            raise ValueError("cluster_centers is required for Latent Space Coverage")

        print(f"Using balanced coverage framework (k={n_clusters}, lambda={lambda_param})")
        self.coverage_tracker = LatentSpaceClusterCoverage(
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_cp_scores=cluster_cp_scores,
            risk_threshold=risk_threshold,
            lambda_param=lambda_param,
            weight_transform='none'
        )

        self.calibration_interval = calibration_interval
        self.calibration_scores = []

        self.lambda_lscc = lambda_param  # λ_LSCC in paper Eq.3: U = scp + λ_LSCC · ΔC

        self.max_queue_size = max_queue_size
        self.max_exploration_seeds = max_exploration_seeds

        self.total_tests = 0
        self.total_faults = 0
        self.coverage_history = []
        self.fault_history = []

        self.coverage_gain_history = []
        self.soft_gain_history = []
        self.novelty_history = []
        self.cp_score_history = []
        self.new_cluster_count = 0

        self.seed_queue = []
        self.seed_energies = []

    def add_seed(self, image_pil, label):
        self.seed_queue.append((image_pil, label))
        self.seed_energies.append(1.0)

    def prune_queue(self):
        if len(self.seed_queue) > self.max_queue_size:
            indices = np.argsort(self.seed_energies)[::-1][:self.max_queue_size]
            self.seed_queue = [self.seed_queue[i] for i in indices]
            self.seed_energies = [self.seed_energies[i] for i in indices]

    def get_cp_score_and_features(self, image_tensor, seed_label=None):
        with torch.no_grad():
            features = extract_pca_coverage(
                self.model,
                image_tensor.unsqueeze(0),
                self.pca_models,
                self.scalers,
                pca_dim=self.pca_dim,
                criterion_names=self.criterion_names,
                criteria_dict=self.criteria_dict,
                seed_label=seed_label
            ).to(device)

            model_id_tensor = torch.tensor([self.model_id], dtype=torch.long).to(device)

            logits = self.cp_model(features, model_id_tensor)
            score = torch.sigmoid(logits).item()

            # Extract h_CLS for LSCC cluster assignment (paper Section 3.3)
            if hasattr(self.cp_model, 'get_cls_representation'):
                cls_repr = self.cp_model.get_cls_representation(features, model_id_tensor)
                cls_repr = cls_repr.cpu().numpy().flatten()
            else:
                cls_repr = features.cpu().numpy().flatten()

        return score, cls_repr

    def get_cp_score(self, image_tensor, seed_label=None):
        score, _ = self.get_cp_score_and_features(image_tensor, seed_label=seed_label)
        return score

    def power_schedule(self, seed_idx):
        energy = self.seed_energies[seed_idx]
        return max(1, int(energy * 10))

    def fuzz_random_walk(self, num_iterations=500, max_tests=None):
        print(f"\nUsing random walk strategy for fuzzing")
        print(f"Seeds: {len(self.seed_queue)}, mutations per seed: {num_iterations}")
        if max_tests:
            print(f"Max tests: {max_tests}")

        stopped_early = False

        for seed_idx in tqdm(range(len(self.seed_queue)), desc="Random Walk Fuzzing"):
            if stopped_early:
                break

            seed_pil, seed_label = self.seed_queue[seed_idx]
            current_img = seed_pil.copy()

            for _ in range(num_iterations):
                if max_tests and self.total_tests >= max_tests:
                    stopped_early = True
                    break

                mutant_pil, valid = mutate_with_validity_pil(current_img, seed_pil)
                if not valid:
                    continue

                mutant_tensor = mutant_preprocess(mutant_pil).to(device)

                with torch.no_grad():
                    output = self.model(mutant_tensor.unsqueeze(0))
                    pred = output.argmax(dim=1).item()

                is_fault = (pred != seed_label)
                self.total_tests += 1

                if is_fault:
                    self.total_faults += 1

                cp_score, cls_repr = self.get_cp_score_and_features(mutant_tensor, seed_label=seed_label)

                coverage_gain, novelty, soft_gain = self.coverage_tracker.update(cls_repr, cp_score)

                self.coverage_gain_history.append(coverage_gain)
                self.soft_gain_history.append(soft_gain)
                self.novelty_history.append(novelty)
                self.cp_score_history.append(cp_score)
                if coverage_gain > 0:
                    self.new_cluster_count += 1

                if random.random() < 0.5:
                    current_img = mutant_pil

                self.coverage_history.append(self.coverage_tracker.get_coverage())
                self.fault_history.append(self.total_faults)

        print(f"\nFuzzing complete!")
        print(f"Total tests: {self.total_tests}")
        print(f"Total faults: {self.total_faults}")
        fault_rate = self.total_faults / self.total_tests if self.total_tests > 0 else 0
        print(f"Fault detection rate: {fault_rate:.2%}")

        stats = self.coverage_tracker.get_cluster_stats()
        print(f"Final balanced coverage: {stats['balanced_coverage']:.4f}")
        print(f"Final LSCC: {stats['lsc']:.4f}")

        return {
            "total_tests": self.total_tests,
            "total_faults": self.total_faults,
            "fault_rate": fault_rate,
            "final_lsc": stats['lsc'],
            "final_wlsc": stats['wlsc'],
            "final_balanced_coverage": stats['balanced_coverage'],
            "final_c_h": stats['c_h'],
            "final_c_l": stats['c_l'],
            "lambda_param": stats['lambda_param'],
            "n_high_risk": stats['n_high_risk'],
            "n_low_risk": stats['n_low_risk'],
            "covered_high_risk": stats['covered_high_risk'],
            "covered_low_risk": stats['covered_low_risk'],
            "final_ra_lsc": stats['ra_lsc'],
            "final_ra_wlsc": stats['ra_wlsc'],
            "n_risk_clusters": stats['n_risk_clusters'],
            "covered_risk_clusters": stats['covered_risk_clusters'],
            "coverage_history": self.coverage_history,
            "fault_history": self.fault_history,
        }

    def fuzz_guided_random_walk(self, num_iterations=500, max_tests=None):
        print(f"\nUsing guided random walk strategy for fuzzing")
        print(f"Seeds: {len(self.seed_queue)}, mutations per seed: {num_iterations}")
        print(f"Objective: U = scp(x) + lambda_LSCC * delta_C  (lambda_LSCC={self.lambda_lscc})")
        if max_tests:
            print(f"Max tests: {max_tests}")

        stopped_early = False

        for seed_idx in tqdm(range(len(self.seed_queue)), desc="Guided Random Walk"):
            if stopped_early:
                break

            seed_pil, seed_label = self.seed_queue[seed_idx]
            current_img = seed_pil.copy()
            current_objective = 0.0

            for _ in range(num_iterations):
                if max_tests and self.total_tests >= max_tests:
                    stopped_early = True
                    break

                mutant_pil, valid = mutate_with_validity_pil(current_img, seed_pil)
                if not valid:
                    continue
                mutant_tensor = mutant_preprocess(mutant_pil).to(device)

                with torch.no_grad():
                    output = self.model(mutant_tensor.unsqueeze(0))
                    pred = output.argmax(dim=1).item()

                is_fault = (pred != seed_label)
                self.total_tests += 1

                if is_fault:
                    self.total_faults += 1

                cp_score, cls_repr = self.get_cp_score_and_features(mutant_tensor, seed_label=seed_label)

                # Compute gain without updating Covered yet (paper Algorithm 1 lines 6-8)
                cluster_id, coverage_gain = self.coverage_tracker.compute_gain(cls_repr)

                self.coverage_gain_history.append(coverage_gain)
                self.cp_score_history.append(cp_score)
                if coverage_gain > 0:
                    self.new_cluster_count += 1

                # U(x,t) = scp(x) + λ_LSCC · ΔC(x,t)  (paper Eq.3)
                objective = cp_score + self.lambda_lscc * coverage_gain

                if objective > current_objective:
                    current_img = mutant_pil
                    current_objective = objective
                    self.coverage_tracker.commit(cluster_id, cp_score)  # update only on acceptance
                elif random.random() < 0.1:
                    current_img = mutant_pil
                    current_objective = objective
                    self.coverage_tracker.commit(cluster_id, cp_score)

                self.coverage_history.append(self.coverage_tracker.get_coverage())
                self.fault_history.append(self.total_faults)

        print(f"\nFuzzing complete!")
        print(f"Total tests: {self.total_tests}")
        print(f"Total faults: {self.total_faults}")
        fault_rate = self.total_faults / self.total_tests if self.total_tests > 0 else 0
        print(f"Fault detection rate: {fault_rate:.2%}")

        stats = self.coverage_tracker.get_cluster_stats()
        print(f"Final balanced coverage: {stats['balanced_coverage']:.4f}")
        print(f"Final LSCC: {stats['lsc']:.4f}")

        return {
            "total_tests": self.total_tests,
            "total_faults": self.total_faults,
            "fault_rate": fault_rate,
            "final_lsc": stats['lsc'],
            "final_wlsc": stats['wlsc'],
            "final_balanced_coverage": stats['balanced_coverage'],
            "final_c_h": stats['c_h'],
            "final_c_l": stats['c_l'],
            "lambda_param": stats['lambda_param'],
            "n_high_risk": stats['n_high_risk'],
            "n_low_risk": stats['n_low_risk'],
            "covered_high_risk": stats['covered_high_risk'],
            "covered_low_risk": stats['covered_low_risk'],
            "final_ra_lsc": stats['ra_lsc'],
            "final_ra_wlsc": stats['ra_wlsc'],
            "n_risk_clusters": stats['n_risk_clusters'],
            "covered_risk_clusters": stats['covered_risk_clusters'],
            "coverage_history": self.coverage_history,
            "fault_history": self.fault_history,
        }

    def fuzz_high_risk_guided(self, num_iterations=500, max_tests=None, gamma_hr=0.5):
        print(f"\nUsing high-risk cluster proximity guided strategy for fuzzing")
        print(f"Seeds: {len(self.seed_queue)}, mutations per seed: {num_iterations}")
        print(f"Objective: U = scp(x) + lambda_LSCC*delta_C + gamma_hr*prox_H  (lambda_LSCC={self.lambda_lscc}, gamma_hr={gamma_hr})")
        if max_tests:
            print(f"Max tests: {max_tests}")

        stopped_early = False

        for seed_idx in tqdm(range(len(self.seed_queue)), desc="High-Risk Guided"):
            if stopped_early:
                break

            seed_pil, seed_label = self.seed_queue[seed_idx]
            current_img = seed_pil.copy()
            current_objective = 0.0

            for _ in range(num_iterations):
                if max_tests and self.total_tests >= max_tests:
                    stopped_early = True
                    break

                mutant_pil, valid = mutate_with_validity_pil(current_img, seed_pil)
                if not valid:
                    continue
                mutant_tensor = mutant_preprocess(mutant_pil).to(device)

                with torch.no_grad():
                    output = self.model(mutant_tensor.unsqueeze(0))
                    pred = output.argmax(dim=1).item()

                is_fault = (pred != seed_label)
                self.total_tests += 1

                if is_fault:
                    self.total_faults += 1

                cp_score, cls_repr = self.get_cp_score_and_features(mutant_tensor, seed_label=seed_label)

                cluster_id, coverage_gain = self.coverage_tracker.compute_gain(cls_repr)

                prox_h = self.coverage_tracker.get_high_risk_proximity(cls_repr)

                self.coverage_gain_history.append(coverage_gain)
                self.cp_score_history.append(cp_score)
                if coverage_gain > 0:
                    self.new_cluster_count += 1

                # U = scp(x) + λ_LSCC · ΔC + γ_hr · prox_H  (high_risk_guided variant)
                objective = cp_score + self.lambda_lscc * coverage_gain + gamma_hr * prox_h

                if objective > current_objective:
                    current_img = mutant_pil
                    current_objective = objective
                    self.coverage_tracker.commit(cluster_id, cp_score)
                elif random.random() < 0.1:
                    current_img = mutant_pil
                    current_objective = objective
                    self.coverage_tracker.commit(cluster_id, cp_score)

                self.coverage_history.append(self.coverage_tracker.get_coverage())
                self.fault_history.append(self.total_faults)

        print(f"\nFuzzing complete!")
        print(f"Total tests: {self.total_tests}")
        print(f"Total faults: {self.total_faults}")
        fault_rate = self.total_faults / self.total_tests if self.total_tests > 0 else 0
        print(f"Fault detection rate: {fault_rate:.2%}")

        stats = self.coverage_tracker.get_cluster_stats()
        print(f"Final balanced coverage: {stats['balanced_coverage']:.4f}")
        print(f"Final LSCC: {stats['lsc']:.4f}")

        return {
            "total_tests": self.total_tests,
            "total_faults": self.total_faults,
            "fault_rate": fault_rate,
            "final_lsc": stats['lsc'],
            "final_wlsc": stats['wlsc'],
            "final_balanced_coverage": stats['balanced_coverage'],
            "final_c_h": stats['c_h'],
            "final_c_l": stats['c_l'],
            "lambda_param": stats['lambda_param'],
            "n_high_risk": stats['n_high_risk'],
            "n_low_risk": stats['n_low_risk'],
            "covered_high_risk": stats['covered_high_risk'],
            "covered_low_risk": stats['covered_low_risk'],
            "final_ra_lsc": stats['ra_lsc'],
            "final_ra_wlsc": stats['ra_wlsc'],
            "n_risk_clusters": stats['n_risk_clusters'],
            "covered_risk_clusters": stats['covered_risk_clusters'],
            "coverage_history": self.coverage_history,
            "fault_history": self.fault_history,
        }

    def fuzz(self, num_iterations=1000, max_tests=None):
        if max_tests:
            print(f"\nStarting fuzzing with max {max_tests} tests...")
        else:
            print(f"\nStarting fuzzing for {num_iterations} iterations...")

        iteration = 0
        while iteration < num_iterations:
            if max_tests and self.total_tests >= max_tests:
                print(f"\nReached max tests limit: {max_tests}")
                break

            if not self.seed_queue:
                print("No seeds in queue!")
                break

            seed_idx = random.randint(0, len(self.seed_queue) - 1)
            seed_pil, seed_label = self.seed_queue[seed_idx]

            num_mutations = self.power_schedule(seed_idx)

            batch_objectives = []

            for _ in range(num_mutations):
                if max_tests and self.total_tests >= max_tests:
                    break

                mutant_pil, valid = mutate_with_validity_pil(seed_pil, seed_pil)
                if not valid:
                    continue
                mutant_tensor = mutant_preprocess(mutant_pil).to(device)

                with torch.no_grad():
                    output = self.model(mutant_tensor.unsqueeze(0))
                    pred = output.argmax(dim=1).item()

                is_fault = (pred != seed_label)
                self.total_tests += 1

                if is_fault:
                    self.total_faults += 1

                cp_score, cls_repr = self.get_cp_score_and_features(mutant_tensor, seed_label=seed_label)

                cluster_id, coverage_gain = self.coverage_tracker.compute_gain(cls_repr)

                self.coverage_gain_history.append(coverage_gain)
                self.cp_score_history.append(cp_score)
                if coverage_gain > 0:
                    self.new_cluster_count += 1

                # U(x,t) = scp(x) + λ_LSCC · ΔC(x,t)  (paper Eq.3)
                objective = cp_score + self.lambda_lscc * coverage_gain

                batch_objectives.append((objective, mutant_pil, seed_label, coverage_gain, cluster_id, cp_score))

                if objective > 0:
                    self.seed_energies[seed_idx] = min(10.0, self.seed_energies[seed_idx] + objective)

                self.calibration_scores.append(cp_score)

            if batch_objectives:
                added_seeds = set()

                exploration_count = 0
                for obj, mutant_pil, seed_label, cov_gain, cluster_id, cp_score in batch_objectives:
                    if cov_gain > 0 and exploration_count < self.max_exploration_seeds:
                        self.add_seed(mutant_pil, seed_label)
                        self.coverage_tracker.commit(cluster_id, cp_score)
                        added_seeds.add(id(mutant_pil))
                        exploration_count += 1

                batch_objectives.sort(key=lambda x: x[0], reverse=True)
                top_k = max(1, len(batch_objectives) // 10)

                exploitation_count = 0
                for obj, mutant_pil, seed_label, cov_gain, cluster_id, cp_score in batch_objectives[:top_k]:
                    if id(mutant_pil) not in added_seeds:
                        self.add_seed(mutant_pil, seed_label)
                        self.coverage_tracker.commit(cluster_id, cp_score)
                        exploitation_count += 1

                self.prune_queue()

            self.coverage_history.append(self.coverage_tracker.get_coverage())
            self.fault_history.append(self.total_faults)

            if (iteration + 1) % self.calibration_interval == 0:
                print(f"\n[Calibration at iteration {iteration + 1}]")
                stats = self.coverage_tracker.get_cluster_stats()
                print(f"  Balanced Coverage: {stats['balanced_coverage']:.4f} (lambda={stats['lambda_param']:.2f})")
                print(f"  C_H (high-risk): {stats['c_h']:.4f} ({stats['covered_high_risk']}/{stats['n_high_risk']})")
                print(f"  C_L (low-risk): {stats['c_l']:.4f} ({stats['covered_low_risk']}/{stats['n_low_risk']})")
                print(f"  LSCC (all clusters): {stats['lsc']:.4f} ({stats['covered_clusters']}/{stats['n_clusters']})")
                print(f"  Faults found: {self.total_faults}")
                print(f"  Total tests: {self.total_tests}")
                print(f"  Fault rate: {self.total_faults / self.total_tests:.4f}")
                print(f"  Seeds in queue: {len(self.seed_queue)}")
                recent_n = min(100, len(self.coverage_gain_history))
                if recent_n > 0:
                    recent_cg = self.coverage_gain_history[-recent_n:]
                    recent_sg = self.soft_gain_history[-recent_n:]
                    recent_nov = self.novelty_history[-recent_n:]
                    recent_cp = self.cp_score_history[-recent_n:]
                    cg_nonzero_ratio = sum(1 for x in recent_cg if x > 0) / recent_n
                    print(f"  [Monitoring] coverage_gain>0 ratio: {cg_nonzero_ratio:.2%} (last {recent_n})")
                    print(f"  [Monitoring] soft_gain: mean={np.mean(recent_sg):.4f}, std={np.std(recent_sg):.4f}")
                    print(f"  [Monitoring] novelty: mean={np.mean(recent_nov):.4f}, std={np.std(recent_nov):.4f}")
                    print(f"  [Monitoring] cp_score: mean={np.mean(recent_cp):.4f}, std={np.std(recent_cp):.4f}")
                    print(f"  [Monitoring] new_cluster_count: {self.new_cluster_count} total")

                dist_stats = self.coverage_tracker.get_distance_stats()
                if dist_stats.get('margin_mean') is not None:
                    print(f"  [Distance Stats] margin: mean={dist_stats['margin_mean']:.4f}, std={dist_stats['margin_std']:.4f}")
                    print(f"  [Distance Stats] margin P10/P50/P90: {dist_stats['margin_p10']:.4f}/{dist_stats['margin_p50']:.4f}/{dist_stats['margin_p90']:.4f}")
                    print(f"  [Distance Stats] tau={dist_stats['tau']:.4f}")

            iteration += 1

        print(f"\nFuzzing completed!")
        print(f"Total tests: {self.total_tests}")
        print(f"Total faults: {self.total_faults}")
        stats = self.coverage_tracker.get_cluster_stats()
        print(f"Final Balanced Coverage: {stats['balanced_coverage']:.4f} (lambda={stats['lambda_param']:.2f})")
        print(f"Final C_H (high-risk): {stats['c_h']:.4f} ({stats['covered_high_risk']}/{stats['n_high_risk']})")
        print(f"Final C_L (low-risk): {stats['c_l']:.4f} ({stats['covered_low_risk']}/{stats['n_low_risk']})")
        print(f"Final LSCC: {stats['lsc']:.4f}")
        print(f"Final WLSCC: {stats['wlsc']:.4f}")
        print(f"Covered clusters: {stats['covered_clusters']}/{stats['n_clusters']}")
        print(f"Fault rate: {self.total_faults / self.total_tests:.4f}")
        if len(self.coverage_gain_history) > 0:
            cg_nonzero_ratio = sum(1 for x in self.coverage_gain_history if x > 0) / len(self.coverage_gain_history)
            print(f"\n[Final Monitoring]")
            print(f"  coverage_gain>0 ratio: {cg_nonzero_ratio:.2%} ({self.new_cluster_count}/{len(self.coverage_gain_history)})")
            print(f"  soft_gain: mean={np.mean(self.soft_gain_history):.4f}, std={np.std(self.soft_gain_history):.4f}")
            print(f"  novelty: mean={np.mean(self.novelty_history):.4f}, std={np.std(self.novelty_history):.4f}")
            print(f"  cp_score: mean={np.mean(self.cp_score_history):.4f}, std={np.std(self.cp_score_history):.4f}")

        return {
            "total_tests": self.total_tests,
            "total_faults": self.total_faults,
            "final_lsc": stats['lsc'],
            "final_wlsc": stats['wlsc'],
            "final_balanced_coverage": stats['balanced_coverage'],
            "final_c_h": stats['c_h'],
            "final_c_l": stats['c_l'],
            "lambda_param": stats['lambda_param'],
            "n_high_risk": stats['n_high_risk'],
            "n_low_risk": stats['n_low_risk'],
            "covered_high_risk": stats['covered_high_risk'],
            "covered_low_risk": stats['covered_low_risk'],
            "new_cluster_count": self.new_cluster_count,
            "coverage_gain_history": self.coverage_gain_history,
            "soft_gain_history": self.soft_gain_history,
            "novelty_history": self.novelty_history,
            "cp_score_history": self.cp_score_history,
            "final_ra_lsc": stats['ra_lsc'],
            "final_ra_wlsc": stats['ra_wlsc'],
            "n_risk_clusters": stats['n_risk_clusters'],
            "covered_risk_clusters": stats['covered_risk_clusters'],
            "coverage_history": self.coverage_history,
            "fault_history": self.fault_history
        }

def load_pca_models_from_data(model_name, data_dir, pca_dim=256, seed=42,
                              criterion_names=None):
    pca_path = os.path.join(data_dir, f"activation_coverage_{model_name}", "pca_models.pkl")
    if os.path.exists(pca_path):
        print(f"\nLoading saved PCA models for {model_name} from {pca_path}")
        return load_pca_models(model_name, data_dir)

    print(f"\nWARNING: Saved PCA models not found at {pca_path}")
    print(f"Re-fitting PCA from raw data. This may produce different results than training!")
    if criterion_names is None:
        criterion_names = ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']

    model_data_dir = os.path.join(data_dir, f"activation_coverage_{model_name}")

    pca_models = {}
    scalers = {}

    for criterion in tqdm(criterion_names, desc="PCA fitting"):
        cov_path = os.path.join(model_data_dir, f"{criterion}_coverage.npy")
        cov_data = np.load(cov_path, mmap_mode="r")

        n_samples = min(1000, len(cov_data))
        indices = np.random.choice(len(cov_data), n_samples, replace=False)
        samples = np.array([cov_data[i] for i in indices])

        scaler = StandardScaler()
        samples_scaled = scaler.fit_transform(samples)

        actual_dim = min(pca_dim, samples_scaled.shape[1], samples_scaled.shape[0])
        pca = PCA(n_components=actual_dim, random_state=seed)
        pca.fit(samples_scaled)

        pca_models[criterion] = pca
        scalers[criterion] = scaler

        explained_var = sum(pca.explained_variance_ratio_) * 100
        print(f"  {criterion}: {samples.shape[1]} -> {actual_dim} (explained variance: {explained_var:.1f}%)")

    return pca_models, scalers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "MNIST", "TinyImageNet"],
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16_bn", "mobilenet_v2"])
    parser.add_argument("--cp_checkpoint", type=str, default=None,
                        help='CP model checkpoint path (default: ./cp_checkpoints_{dataset_tag}/best_model.pt)')
    parser.add_argument("--data_dir", type=str, default=None,
                        help='Coverage matrices directory (default: ./coverage_matrices_{dataset_tag})')
    parser.add_argument("--num_seeds", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--max_tests", type=int, default=None,
                        help='Maximum total tests (overrides num_iterations if set)')
    parser.add_argument("--n_clusters", type=int, default=50,
                        help='Number of clusters for Latent Space Coverage')
    parser.add_argument("--risk_threshold", type=float, default=0.5,
                        help='Proportion of H set (top risk_threshold are high-risk clusters, default: 0.5)')
    parser.add_argument("--lambda_param", type=float, default=0.7,
                        help='H vs L balance parameter, C(t) = lambda*C_H + (1-lambda)*C_L (default: 0.7)')
    parser.add_argument("--calibration_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./fuzz_results_cp_pca")
    parser.add_argument("--strategy", type=str, default="random_walk",
                        choices=["random_walk", "guided_random_walk", "energy_based", "high_risk_guided"],
                        help='Fuzzing strategy: random_walk, guided_random_walk, energy_based, high_risk_guided')
    parser.add_argument("--gamma_hr", type=float, default=0.5,
                        help='High-risk proximity weight (for high_risk_guided strategy, default: 0.5)')
    parser.add_argument("--seed", type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument("--ablate_cp", action="store_true",
                        help='Ablate CP: replace cluster_cp_scores with uniform values '
                             'so H/L partitioning is not CP-informed')
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_name = args.dataset
    ds_cfg = get_dataset_config(dataset_name)
    dataset_tag = ds_cfg['weight_tag']
    num_classes = ds_cfg['num_classes']
    image_size = ds_cfg['image_size']
    ds_mean = ds_cfg['mean']
    ds_std = ds_cfg['std']
    ds_data_dir = ds_cfg['data_dir']

    if args.cp_checkpoint is None:
        dataset_dir_map = {
            'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST',
            'Fashion-MNIST': 'Fashion-MNIST', 'TinyImageNet': 'TinyImageNet',
        }
        dataset_dir = dataset_dir_map.get(dataset_name, dataset_name)
        possible = [
            f"./cp_checkpoints/{dataset_dir}/best_model.pt",
            f"./cp_checkpoints/{dataset_name}/best_model.pt",
            f"./cp_checkpoints_{dataset_tag}/best_model.pt",
        ]
        for p in possible:
            if os.path.exists(p):
                args.cp_checkpoint = p
                break
        if args.cp_checkpoint is None:
            args.cp_checkpoint = possible[0]
    if args.data_dir is None:
        dataset_dir_map = {
            'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST',
            'Fashion-MNIST': 'Fashion-MNIST', 'TinyImageNet': 'TinyImageNet',
        }
        dataset_dir = dataset_dir_map.get(dataset_name, dataset_name)
        args.data_dir = f"./coverage_matrices/{dataset_dir}"

    global model_transform, mutant_preprocess
    model_transform = get_model_transform(dataset_name)

    if dataset_name == 'MNIST':
        base_transform = get_model_transform(dataset_name)
        mutant_preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            base_transform,
        ])
    else:
        mutant_preprocess = model_transform

    print(f"Dataset: {dataset_name} (tag={dataset_tag}, classes={num_classes}, size={image_size})")

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model, dataset_name)

    print(f"Loading CP model from {args.cp_checkpoint}...")
    checkpoint = torch.load(args.cp_checkpoint, map_location=device)
    cp_args = checkpoint["args"]

    cp_criterion_names = checkpoint.get("criterion_names", cp_args.get("criterion_names",
        ['NC', 'KMNC', 'NBC', 'SNAC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC']))
    cp_pca_dim = cp_args["pca_dim"]
    print(f"  CP model criteria ({len(cp_criterion_names)}): {cp_criterion_names}")
    print(f"  CP model pca_dim: {cp_pca_dim}")

    model_name_to_id = {"resnet50": 0, "vgg16_bn": 1, "mobilenet_v2": 2}
    model_id = model_name_to_id[args.model]

    print(f"Loading PCA models for {args.model}...")
    pca_models, scalers = load_pca_models_from_data(
        args.model,
        args.data_dir,
        pca_dim=cp_pca_dim,
        seed=42,
        criterion_names=cp_criterion_names
    )

    cp_model = CoverageTransformer(
        pca_dim=cp_pca_dim,
        num_criteria=len(cp_criterion_names),
        d_model=cp_args["d_model"],
        nhead=cp_args["nhead"],
        num_layers=cp_args["num_layers"],
        dropout=cp_args["dropout"],
        num_models=3
    ).to(device)

    cp_model.load_state_dict(checkpoint["model_state_dict"])
    cp_model.eval()

    print(f"CP model loaded successfully!")

    cluster_path = os.path.join(os.path.dirname(args.cp_checkpoint), "latent_space_clusters.npz")
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Cluster file not found at {cluster_path}. Please run train_cp_pca.py first.")

    print(f"Loading Latent Space Clusters from {cluster_path}...")
    cluster_data = np.load(cluster_path)
    cluster_centers = cluster_data["cluster_centers"]
    cluster_cp_scores = cluster_data["cluster_cp_scores"]
    args.n_clusters = int(cluster_data["n_clusters"])
    print(f"  Loaded {args.n_clusters} clusters")
    print(f"  Avg cluster CP score: {np.mean(cluster_cp_scores):.4f}")

    if args.ablate_cp:
        print(f"  [ABLATION] Replacing cluster_cp_scores with uniform values (--ablate_cp)")
        cluster_cp_scores = np.ones_like(cluster_cp_scores) * 0.5

    print(f"Loading {args.num_seeds} seeds from {dataset_name}...")
    if dataset_name == 'CIFAR10':
        from torchvision.datasets import CIFAR10
        test_dataset = CIFAR10(root='./datasets', train=False, download=False)
    elif dataset_name == 'MNIST':
        from torchvision.datasets import MNIST
        test_dataset = MNIST(root='./datasets', train=False, download=False)
    elif dataset_name == 'TinyImageNet':
        from torchvision.datasets import ImageFolder
        tiny_test_dir = os.path.join(ds_data_dir, 'val')
        if not os.path.exists(tiny_test_dir):
            tiny_test_dir = os.path.join(ds_data_dir, 'test')
        test_dataset = ImageFolder(root=tiny_test_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"\nInitializing coverage criteria for {args.model} on {dataset_name}...")
    input_size = (1, 3, image_size, image_size)
    random_input = torch.randn(input_size).to(device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_input)

    print("Building train_loader for coverage criteria...")
    import torchvision.transforms as crit_transforms
    from PIL import Image as PILImage
    crit_transform_list = [
        crit_transforms.Resize(image_size),
        crit_transforms.CenterCrop(image_size),
    ]
    if dataset_name == 'MNIST':
        crit_transform_list.append(crit_transforms.Grayscale(num_output_channels=3))
    crit_transform_list += [
        crit_transforms.ToTensor(),
        crit_transforms.Normalize(ds_mean, ds_std),
    ]
    crit_transform = crit_transforms.Compose(crit_transform_list)

    train_image_dir = os.path.join(ds_data_dir, "train")
    max_classes = num_classes if num_classes <= 200 else 200
    if os.path.exists(train_image_dir):
        train_class_list = sorted(os.listdir(train_image_dir))[:max_classes]
        train_tensors = []
        train_labels = []
        for cls_idx, cls in enumerate(train_class_list):
            cls_dir = os.path.join(train_image_dir, cls)
            imgs = sorted(os.listdir(cls_dir))[:200]
            for img_name in imgs:
                img_path = os.path.join(cls_dir, img_name)
                if dataset_name == 'MNIST':
                    img = PILImage.open(img_path).convert('L')
                    img = img.convert('RGB')
                else:
                    img = PILImage.open(img_path).convert('RGB')
                train_tensors.append(crit_transform(img))
                train_labels.append(cls_idx)
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack(train_tensors),
            torch.tensor(train_labels, dtype=torch.long)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=False
        )
        print(f"  Train loader: {len(train_tensors)} samples")
    else:
        print(f"  Train image dir not found, using torchvision {dataset_name}...")
        if dataset_name == 'CIFAR10':
            from torchvision.datasets import CIFAR10 as TVDataset
            tv_train = TVDataset(root='./datasets', train=True, download=False,
                                 transform=crit_transform)
        elif dataset_name == 'MNIST':
            from torchvision.datasets import MNIST as TVDataset
            tv_train = TVDataset(root='./datasets', train=True, download=False,
                                 transform=crit_transform)
    criteria_dict = init_coverage_criteria(model, layer_size_dict, train_loader=train_loader, device=device)
    print(f"Initialized {len(criteria_dict)} coverage criteria: {sorted(criteria_dict.keys())}")

    fuzzer = CPPCAFuzzer(
        model=model,
        cp_model=cp_model,
        pca_models=pca_models,
        scalers=scalers,
        model_id=model_id,
        criteria_dict=criteria_dict,
        pca_dim=cp_pca_dim,
        criterion_names=cp_criterion_names,
        calibration_interval=args.calibration_interval,
        cluster_centers=cluster_centers,
        cluster_cp_scores=cluster_cp_scores,
        n_clusters=args.n_clusters,
        risk_threshold=args.risk_threshold,
        lambda_param=args.lambda_param
    )

    skipped_wrong = 0
    added = 0
    for i in range(min(args.num_seeds * 2, len(test_dataset))):
        img_pil, label = test_dataset[i]
        if dataset_name == 'MNIST':
            if img_pil.mode != 'L':
                img_pil = img_pil.convert('L')
            img_pil_rgb = img_pil.convert('RGB')
        else:
            img_pil_rgb = img_pil.convert('RGB') if img_pil.mode != 'RGB' else img_pil
        seed_tensor = model_transform(img_pil_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(seed_tensor).argmax(dim=1).item()
        if pred == label:
            fuzzer.add_seed(img_pil_rgb, label)
            added += 1
            if added >= args.num_seeds:
                break
        else:
            skipped_wrong += 1

    print(f"Added {added} seeds to fuzzer (skipped {skipped_wrong} misclassified)")

    print(f"\nUsing strategy: {args.strategy}")
    if args.strategy == "random_walk":
        results = fuzzer.fuzz_random_walk(num_iterations=args.num_iterations, max_tests=args.max_tests)
    elif args.strategy == "guided_random_walk":
        results = fuzzer.fuzz_guided_random_walk(num_iterations=args.num_iterations, max_tests=args.max_tests)
    elif args.strategy == "high_risk_guided":
        results = fuzzer.fuzz_high_risk_guided(num_iterations=args.num_iterations, max_tests=args.max_tests, gamma_hr=args.gamma_hr)
    else:
        results = fuzzer.fuzz(num_iterations=args.num_iterations, max_tests=args.max_tests)

    output_path = os.path.join(args.output_dir, f"{dataset_tag}_{args.model}_results.npz")
    np.savez(
        output_path,
        total_tests=results["total_tests"],
        total_faults=results["total_faults"],
        final_lsc=results["final_lsc"],
        final_wlsc=results["final_wlsc"],
        final_balanced_coverage=results["final_balanced_coverage"],
        final_c_h=results["final_c_h"],
        final_c_l=results["final_c_l"],
        lambda_param=results["lambda_param"],
        n_high_risk=results["n_high_risk"],
        n_low_risk=results["n_low_risk"],
        covered_high_risk=results["covered_high_risk"],
        covered_low_risk=results["covered_low_risk"],
        final_ra_lsc=results["final_ra_lsc"],
        final_ra_wlsc=results["final_ra_wlsc"],
        n_risk_clusters=results["n_risk_clusters"],
        covered_risk_clusters=results["covered_risk_clusters"],
        coverage_history=results["coverage_history"],
        fault_history=results["fault_history"]
    )
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
