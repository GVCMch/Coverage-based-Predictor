import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MultiMetricDataset(Dataset):

    def __init__(
        self,
        data_infos: List[Dict],
        criterion_names: List[str],
        pca_dim: int = 256,
        seed: int = 42,
        pca_sample_ratio: float = 0.1,
        dataset_tag: str = "",
        pca_base_dir: str = None,
    ):
        self.data_infos = data_infos
        self.criterion_names = criterion_names
        self.pca_dim = pca_dim
        self.seed = seed
        self.dataset_tag = dataset_tag
        self.pca_base_dir = pca_base_dir
        self.num_criteria = len(criterion_names)

        print(f"Initializing multi-metric dataset...")
        print(f"  Feature dimension per criterion: {pca_dim}")
        print(f"  Number of criteria: {self.num_criteria}")
        print(f"  Total feature dimension: {pca_dim * self.num_criteria}")

        self.coverage_maps = {}
        self.fault_maps = []
        self.model_ids = []
        self.lengths = []

        for info in self.data_infos:
            model_idx = len(self.model_ids)
            self.coverage_maps[model_idx] = {}

            num_samples = info.get("num_samples", None)

            for criterion in criterion_names:
                cov_path = info["coverage_paths"][criterion]
                cov = np.load(cov_path, mmap_mode="r")
                if num_samples is not None:
                    cov = cov[:num_samples]
                self.coverage_maps[model_idx][criterion] = cov

            fault = np.load(info["fault_path"], mmap_mode="r")
            if num_samples is not None:
                fault = fault[:num_samples]
            self.fault_maps.append(fault)
            self.model_ids.append(int(info["model_id"]))
            actual_len = int(len(fault))
            self.lengths.append(actual_len)

            if num_samples is not None:
                assert actual_len == num_samples, \
                    f"Length mismatch: fault has {actual_len} but meta says {num_samples}"
                if actual_len > 0:
                    last_row = self.coverage_maps[model_idx][criterion_names[0]][actual_len - 1]
                    assert not np.all(last_row == 0), \
                        f"Last row is all zeros for {info['model_name']} — truncation may be wrong"
                print(f"  {info['model_name']}: truncated to {actual_len} samples (verified)")

        self.cum = np.cumsum([0] + self.lengths)
        self.total_len = int(self.cum[-1])

        print(f"\nDataset summary:")
        for i, info in enumerate(self.data_infos):
            print(f"  - {info['model_name']}: N={self.lengths[i]}")
        print(f"  -> Total samples: {self.total_len}")

        self.pca_models = {}
        self.scalers = {}

        loaded_from_disk = self._try_load_pca_models()

        if not loaded_from_disk:
            print(f"\nNo pre-saved PCA models found, fitting from scratch (per model)...")
            for model_idx in range(len(self.data_infos)):
                model_name = self.data_infos[model_idx]["model_name"]
                print(f"\n  Model {model_name}:")

                for criterion in criterion_names:
                    cov_data = self.coverage_maps[model_idx][criterion]
                    n_samples = min(int(len(cov_data) * pca_sample_ratio), 5000)
                    indices = np.random.choice(len(cov_data), n_samples, replace=False)
                    samples = np.array([cov_data[i] for i in indices])

                    scaler = StandardScaler()
                    samples_scaled = scaler.fit_transform(samples)

                    actual_dim = min(pca_dim, samples_scaled.shape[1], samples_scaled.shape[0])
                    pca = PCA(n_components=actual_dim, random_state=seed)
                    pca.fit(samples_scaled)

                    self.pca_models[(model_idx, criterion)] = pca
                    self.scalers[(model_idx, criterion)] = scaler

                    explained_var = sum(pca.explained_variance_ratio_) * 100
                    print(f"    {criterion}: {samples.shape[1]} -> {actual_dim} (explained variance: {explained_var:.1f}%)")

        print(f"\nFeature preprocessing ready!")

        print(f"\nPre-computing PCA features...")
        self.precomputed_features = []
        self.precomputed_faults = []
        self.precomputed_model_ids = []

        for model_idx in range(len(self.data_infos)):
            model_name = self.data_infos[model_idx]["model_name"]
            n_samples = self.lengths[model_idx]

            print(f"  Processing {model_name} ({n_samples} samples)...")

            for local_idx in range(n_samples):
                features = []
                for criterion in self.criterion_names:
                    cov_row = self.coverage_maps[model_idx][criterion][local_idx]
                    cov_row = np.array(cov_row).reshape(1, -1)

                    scaled = self.scalers[(model_idx, criterion)].transform(cov_row)
                    reduced = self.pca_models[(model_idx, criterion)].transform(scaled)

                    reduced_flat = reduced.flatten()
                    if len(reduced_flat) < self.pca_dim:
                        reduced_flat = np.pad(reduced_flat, (0, self.pca_dim - len(reduced_flat)))
                    features.append(reduced_flat[:self.pca_dim])

                features = np.concatenate(features).astype(np.float32)
                fault = float(self.fault_maps[model_idx][local_idx])
                model_id = self.model_ids[model_idx]

                self.precomputed_features.append(features)
                self.precomputed_faults.append(fault)
                self.precomputed_model_ids.append(model_id)

        self.precomputed_features = np.array(self.precomputed_features)
        self.precomputed_faults = np.array(self.precomputed_faults)
        self.precomputed_model_ids = np.array(self.precomputed_model_ids)

        self._save_pca_models()

        del self.coverage_maps
        del self.fault_maps
        del self.pca_models
        del self.scalers

        print(f"\nPre-computation done! Feature shape: {self.precomputed_features.shape}")

    def _try_load_pca_models(self) -> bool:
        import pickle

        if self.pca_base_dir is None:
            return False

        print(f"\nTrying to load pre-saved PCA models from {self.pca_base_dir}...")

        loaded_all = True
        for model_idx, info in enumerate(self.data_infos):
            model_name = info['model_name']
            pca_dir = os.path.join(self.pca_base_dir, f"activation_coverage_{model_name}")
            pca_path = os.path.join(pca_dir, "pca_models.pkl")

            if not os.path.exists(pca_path):
                print(f"  [WARN] PCA not found: {pca_path}")
                loaded_all = False
                break

            with open(pca_path, 'rb') as f:
                saved = pickle.load(f)

            saved_pca = saved['pca_models']
            saved_scalers = saved['scalers']

            print(f"\n  Model {model_name} (loaded from {pca_path}):")
            for criterion in self.criterion_names:
                key_prefixed = f"{model_name}_{criterion}"
                pca_obj = saved_pca.get(key_prefixed, saved_pca.get(criterion, None))
                scaler_obj = saved_scalers.get(key_prefixed, saved_scalers.get(criterion, None))

                if pca_obj is None or scaler_obj is None:
                    print(f"    [WARN] Missing PCA/scaler for {criterion}")
                    loaded_all = False
                    break

                self.pca_models[(model_idx, criterion)] = pca_obj
                self.scalers[(model_idx, criterion)] = scaler_obj

                n_components = pca_obj.n_components_
                orig_dim = pca_obj.components_.shape[1]
                explained_var = sum(pca_obj.explained_variance_ratio_) * 100
                print(f"    {criterion}: {orig_dim} -> {n_components} (explained variance: {explained_var:.1f}%)")

            if not loaded_all:
                self.pca_models.clear()
                self.scalers.clear()
                break

        if loaded_all:
            print(f"\n  Successfully loaded all pre-saved PCA models!")
        return loaded_all

    def _save_pca_models(self):
        import pickle

        for model_idx, info in enumerate(self.data_infos):
            model_name = info['model_name']
            if self.pca_base_dir:
                save_dir = os.path.join(self.pca_base_dir, f"activation_coverage_{model_name}")
            elif self.dataset_tag:
                save_dir = f'./coverage_matrices/activation_coverage_{model_name}_{self.dataset_tag}'
            else:
                save_dir = f'./coverage_matrices/activation_coverage_{model_name}'
            os.makedirs(save_dir, exist_ok=True)

            model_pca = {}
            model_scalers = {}

            for criterion in self.criterion_names:
                key = (model_idx, criterion)
                if key in self.pca_models:
                    model_pca[f'{model_name}_{criterion}'] = self.pca_models[key]
                    model_scalers[f'{model_name}_{criterion}'] = self.scalers[key]

            pca_path = os.path.join(save_dir, 'pca_models.pkl')
            with open(pca_path, 'wb') as f:
                pickle.dump({'pca_models': model_pca, 'scalers': model_scalers, 'pca_dim': self.pca_dim}, f)

            print(f"  PCA models saved: {pca_path}")

    def __len__(self):
        return self.total_len

    def _locate(self, global_idx: int):
        m = int(np.searchsorted(self.cum, global_idx, side="right") - 1)
        local = int(global_idx - self.cum[m])
        return m, local

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.precomputed_features[idx].copy()),
            torch.tensor(self.precomputed_model_ids[idx], dtype=torch.long),
            torch.tensor(self.precomputed_faults[idx], dtype=torch.float32)
        )
