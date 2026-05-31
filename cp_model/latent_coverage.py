import numpy as np
from typing import Dict, Tuple
from sklearn.cluster import KMeans


class LatentSpaceClusterCoverage:

    def __init__(
        self,
        n_clusters: int = 50,
        cluster_centers: np.ndarray = None,
        cluster_cp_scores: np.ndarray = None,
        risk_threshold: float = 0.5,
        lambda_param: float = 0.7,
        weight_transform: str = 'none',
    ):
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.cluster_cp_scores = cluster_cp_scores
        self.risk_threshold = risk_threshold
        self.lambda_param = lambda_param
        self.weight_transform = weight_transform

        self.covered_clusters = set()

        if cluster_cp_scores is not None:
            n_high_risk = max(1, int(n_clusters * risk_threshold))
            sorted_indices = np.argsort(cluster_cp_scores)[::-1]

            self.high_risk_clusters = set(sorted_indices[:n_high_risk].tolist())
            self.low_risk_clusters = set(sorted_indices[n_high_risk:].tolist())

            self.n_high_risk = len(self.high_risk_clusters)
            self.n_low_risk = len(self.low_risk_clusters)

            avg_cp_high = np.mean([cluster_cp_scores[j] for j in self.high_risk_clusters])
            avg_cp_low = np.mean([cluster_cp_scores[j] for j in self.low_risk_clusters]) if self.n_low_risk > 0 else 0

            print(f"  Balanced Coverage Framework")
            print(f"    lambda_param (alpha in paper) = {lambda_param:.2f}")
            print(f"    H (high-risk clusters): {self.n_high_risk}, avg CP={avg_cp_high:.4f}")
            print(f"    L (low-risk clusters): {self.n_low_risk}, avg CP={avg_cp_low:.4f}")
            print(f"    Coverage gain: delta_C_H = lambda/|H| = {lambda_param/self.n_high_risk:.4f}")
            print(f"    Coverage gain: delta_C_L = (1-lambda)/|L| = {(1-lambda_param)/self.n_low_risk:.4f}" if self.n_low_risk > 0 else "    delta_C_L = 0 (no L clusters)")

            self.risk_clusters = self.high_risk_clusters
            self.n_risk_clusters = self.n_high_risk
            self.total_weight_risk = sum(cluster_cp_scores[j] for j in self.high_risk_clusters)

            if weight_transform == 'sqrt':
                self.transformed_weights = np.sqrt(cluster_cp_scores + 0.01)
            elif weight_transform == 'log':
                self.transformed_weights = np.log1p(cluster_cp_scores * 10) / np.log1p(10)
            elif weight_transform == 'clip':
                p20, p80 = np.percentile(cluster_cp_scores, [20, 80])
                self.transformed_weights = np.clip(cluster_cp_scores, p20, p80)
            else:
                self.transformed_weights = cluster_cp_scores.copy()
        else:
            self.high_risk_clusters = set(range(n_clusters // 2))
            self.low_risk_clusters = set(range(n_clusters // 2, n_clusters))
            self.n_high_risk = len(self.high_risk_clusters)
            self.n_low_risk = len(self.low_risk_clusters)
            self.risk_clusters = self.high_risk_clusters
            self.n_risk_clusters = self.n_high_risk
            self.total_weight_risk = self.n_high_risk * 0.5
            self.transformed_weights = np.ones(n_clusters) * 0.5

        self.covered_high_risk = set()
        self.covered_low_risk = set()

        self.total_weight_all = np.sum(self.transformed_weights)
        self.total_weight_covered = 0.0
        self.total_weight_risk_covered = 0.0

    def find_nearest_cluster(self, features: np.ndarray) -> int:
        if self.cluster_centers is None:
            raise ValueError("Cluster centers not initialized!")

        features = np.array(features).reshape(1, -1)
        distances = np.linalg.norm(self.cluster_centers - features, axis=1)
        return int(np.argmin(distances))

    def get_distance_to_nearest_center(self, features: np.ndarray) -> float:
        if self.cluster_centers is None:
            return 0.0
        features = np.array(features).reshape(1, -1)
        distances = np.linalg.norm(self.cluster_centers - features, axis=1)
        return float(np.min(distances))

    def get_novelty_score(self, features: np.ndarray) -> float:
        distance = self.get_distance_to_nearest_center(features)

        if not hasattr(self, 'distance_history'):
            self.distance_history = []

        self.distance_history.append(distance)

        if len(self.distance_history) > 100:
            distances = np.array(self.distance_history[-1000:])
            p10, p90 = np.percentile(distances, [10, 90])
            if p90 > p10:
                novelty = np.clip((distance - p10) / (p90 - p10), 0, 1)
            else:
                novelty = 0.5
        else:
            novelty = min(1.0, distance / (np.mean(self.distance_history) + 1e-6))

        return float(novelty)

    def get_soft_coverage_gain(self, features: np.ndarray) -> float:
        if self.cluster_centers is None:
            return 0.0

        features = np.array(features).reshape(1, -1)

        uncovered_high_risk = self.high_risk_clusters - self.covered_high_risk

        if len(uncovered_high_risk) > 0:
            target_uncovered = uncovered_high_risk
        else:
            all_uncovered = set(range(self.n_clusters)) - self.covered_clusters
            if len(all_uncovered) == 0:
                return 0.5
            target_uncovered = all_uncovered

        uncovered_indices = list(target_uncovered)
        uncovered_centers = self.cluster_centers[uncovered_indices]
        distances_u = np.linalg.norm(uncovered_centers - features, axis=1)
        d_u = np.min(distances_u)

        if len(self.covered_clusters) == 0:
            if not hasattr(self, 'soft_gain_tau'):
                self._calibrate_temperature()
            soft_gain = np.exp(-d_u / self.soft_gain_tau)
            return float(np.clip(soft_gain, 0, 1))

        covered_indices = list(self.covered_clusters)
        covered_centers = self.cluster_centers[covered_indices]
        distances_c = np.linalg.norm(covered_centers - features, axis=1)
        d_c = np.min(distances_c)

        margin = d_c - d_u

        if not hasattr(self, 'margin_history'):
            self.margin_history = []
        self.margin_history.append(margin)

        if not hasattr(self, 'soft_gain_tau'):
            self._calibrate_temperature()

        if len(self.margin_history) % 500 == 0 and len(self.margin_history) >= 100:
            self._calibrate_temperature()

        soft_gain = 1.0 / (1.0 + np.exp(-margin / self.soft_gain_tau))

        return float(soft_gain)

    def _calibrate_temperature(self):
        if hasattr(self, 'margin_history') and len(self.margin_history) >= 50:
            margins = np.array(self.margin_history[-500:])
            p10, p90 = np.percentile(margins, [10, 90])
            self.soft_gain_tau = max(0.1, (p90 - p10) / 4.0)
        else:
            if not hasattr(self, 'avg_cluster_distance'):
                all_distances = []
                for i in range(min(10, self.n_clusters)):
                    for j in range(i+1, min(10, self.n_clusters)):
                        d = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                        all_distances.append(d)
                self.avg_cluster_distance = np.mean(all_distances) if all_distances else 1.0
            self.soft_gain_tau = self.avg_cluster_distance / 2.0

    def get_high_risk_proximity(self, features: np.ndarray) -> float:
        if self.cluster_centers is None or len(self.high_risk_clusters) == 0:
            return 0.5

        features = np.array(features).reshape(1, -1)

        high_risk_indices = list(self.high_risk_clusters)
        high_risk_centers = self.cluster_centers[high_risk_indices]
        distances = np.linalg.norm(high_risk_centers - features, axis=1)
        d_H = np.min(distances)

        if not hasattr(self, 'high_risk_distance_history'):
            self.high_risk_distance_history = []
        self.high_risk_distance_history.append(d_H)

        window_size = 500
        recent_distances = self.high_risk_distance_history[-window_size:]

        if len(recent_distances) < 10:
            if not hasattr(self, 'avg_cluster_distance'):
                all_distances = []
                for i in range(min(10, self.n_clusters)):
                    for j in range(i+1, min(10, self.n_clusters)):
                        d = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                        all_distances.append(d)
                self.avg_cluster_distance = np.mean(all_distances) if all_distances else 1.0
            prox = 1.0 / (1.0 + np.exp(d_H / self.avg_cluster_distance - 1))
        else:
            percentile = np.sum(np.array(recent_distances) <= d_H) / len(recent_distances)
            prox = 1.0 - percentile

        return float(np.clip(prox, 0, 1))

    def get_distance_stats(self) -> dict:
        stats = {
            'tau': getattr(self, 'soft_gain_tau', None),
            'avg_cluster_distance': getattr(self, 'avg_cluster_distance', None),
        }

        if hasattr(self, 'margin_history') and len(self.margin_history) > 0:
            margins = np.array(self.margin_history)
            stats['margin_mean'] = float(np.mean(margins))
            stats['margin_std'] = float(np.std(margins))
            stats['margin_p10'] = float(np.percentile(margins, 10))
            stats['margin_p50'] = float(np.percentile(margins, 50))
            stats['margin_p90'] = float(np.percentile(margins, 90))

        if hasattr(self, 'distance_history') and len(self.distance_history) > 0:
            distances = np.array(self.distance_history)
            stats['distance_mean'] = float(np.mean(distances))
            stats['distance_std'] = float(np.std(distances))
            stats['distance_p10'] = float(np.percentile(distances, 10))
            stats['distance_p50'] = float(np.percentile(distances, 50))
            stats['distance_p90'] = float(np.percentile(distances, 90))

        return stats

    def compute_gain(self, features: np.ndarray) -> Tuple[int, float]:
        """Compute coverage gain without updating state (paper Algorithm 1 lines 6-8)."""
        cluster_id = self.find_nearest_cluster(features)
        coverage_gain = 0.0
        if cluster_id not in self.covered_clusters:
            if cluster_id in self.high_risk_clusters:
                coverage_gain = self.lambda_param / self.n_high_risk
            elif cluster_id in self.low_risk_clusters:
                if self.n_low_risk > 0:
                    coverage_gain = (1 - self.lambda_param) / self.n_low_risk
        return cluster_id, coverage_gain

    def commit(self, cluster_id: int, cp_score: float = None):
        """Update covered set after acceptance (paper Algorithm 1 line 11)."""
        if cluster_id not in self.covered_clusters:
            self.covered_clusters.add(cluster_id)
            if cluster_id in self.high_risk_clusters:
                self.covered_high_risk.add(cluster_id)
            elif cluster_id in self.low_risk_clusters:
                self.covered_low_risk.add(cluster_id)
            transformed_weight = self.transformed_weights[cluster_id]
            self.total_weight_covered += transformed_weight
            if cluster_id in self.risk_clusters:
                if self.cluster_cp_scores is not None:
                    self.total_weight_risk_covered += self.cluster_cp_scores[cluster_id]

    def update(self, features: np.ndarray, cp_score: float = None) -> Tuple[float, float, float]:
        cluster_id = self.find_nearest_cluster(features)

        novelty = self.get_novelty_score(features)

        soft_gain = self.get_soft_coverage_gain(features)

        coverage_gain = 0.0
        if cluster_id not in self.covered_clusters:
            self.covered_clusters.add(cluster_id)

            if cluster_id in self.high_risk_clusters:
                coverage_gain = self.lambda_param / self.n_high_risk
                self.covered_high_risk.add(cluster_id)
            elif cluster_id in self.low_risk_clusters:
                if self.n_low_risk > 0:
                    coverage_gain = (1 - self.lambda_param) / self.n_low_risk
                self.covered_low_risk.add(cluster_id)

            transformed_weight = self.transformed_weights[cluster_id]
            self.total_weight_covered += transformed_weight

            if cluster_id in self.risk_clusters:
                if self.cluster_cp_scores is not None:
                    self.total_weight_risk_covered += self.cluster_cp_scores[cluster_id]

        return coverage_gain, novelty, soft_gain

    def get_lsc(self) -> float:
        return len(self.covered_clusters) / self.n_clusters

    def get_balanced_coverage(self) -> float:
        c_h = len(self.covered_high_risk) / self.n_high_risk if self.n_high_risk > 0 else 0
        c_l = len(self.covered_low_risk) / self.n_low_risk if self.n_low_risk > 0 else 0
        return self.lambda_param * c_h + (1 - self.lambda_param) * c_l

    def get_c_h(self) -> float:
        return len(self.covered_high_risk) / self.n_high_risk if self.n_high_risk > 0 else 0

    def get_c_l(self) -> float:
        return len(self.covered_low_risk) / self.n_low_risk if self.n_low_risk > 0 else 0

    def get_wlsc(self) -> float:
        if self.total_weight_all > 0:
            return self.total_weight_covered / self.total_weight_all
        return 0.0

    def get_ra_lsc(self) -> float:
        covered_risk = len(self.covered_clusters & self.risk_clusters)
        return covered_risk / self.n_risk_clusters if self.n_risk_clusters > 0 else 0.0

    def get_ra_wlsc(self) -> float:
        if self.total_weight_risk > 0:
            return self.total_weight_risk_covered / self.total_weight_risk
        return 0.0

    def get_coverage(self) -> float:
        return self.get_balanced_coverage()

    def get_covered_ratio(self) -> float:
        return self.get_lsc()

    def reset(self):
        self.covered_clusters = set()
        self.covered_high_risk = set()
        self.covered_low_risk = set()
        self.total_weight_covered = 0.0
        self.total_weight_risk_covered = 0.0

    def get_cluster_stats(self) -> Dict:
        covered_risk = len(self.covered_clusters & self.risk_clusters)
        return {
            'n_clusters': self.n_clusters,
            'covered_clusters': len(self.covered_clusters),
            'lsc': self.get_lsc(),
            'wlsc': self.get_wlsc(),
            'balanced_coverage': self.get_balanced_coverage(),
            'lambda_param': self.lambda_param,
            'n_high_risk': self.n_high_risk,
            'n_low_risk': self.n_low_risk,
            'covered_high_risk': len(self.covered_high_risk),
            'covered_low_risk': len(self.covered_low_risk),
            'c_h': self.get_c_h(),
            'c_l': self.get_c_l(),
            'n_risk_clusters': self.n_risk_clusters,
            'covered_risk_clusters': covered_risk,
            'ra_lsc': self.get_ra_lsc(),
            'ra_wlsc': self.get_ra_wlsc(),
            'covered_cluster_ids': list(self.covered_clusters),
            'risk_cluster_ids': list(self.risk_clusters)
        }


def train_latent_space_clusters(
    features: np.ndarray,
    cp_scores: np.ndarray,
    n_clusters: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    print(f"\nTraining latent space clusters (k={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    cluster_centers = kmeans.cluster_centers_

    cluster_cp_scores = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters)

    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_sizes[i] = np.sum(mask)
        if cluster_sizes[i] > 0:
            cluster_cp_scores[i] = np.mean(cp_scores[mask])

    cluster_stats = {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'cluster_cp_scores': cluster_cp_scores,
        'avg_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'avg_cp_score': np.mean(cluster_cp_scores),
        'high_cp_clusters': np.sum(cluster_cp_scores > 0.5),
        'low_cp_clusters': np.sum(cluster_cp_scores <= 0.5),
    }

    print(f"  Clustering done!")
    print(f"  Avg cluster size: {cluster_stats['avg_cluster_size']:.1f} +/- {cluster_stats['std_cluster_size']:.1f}")
    print(f"  High CP clusters (>0.5): {cluster_stats['high_cp_clusters']}")
    print(f"  Low CP clusters (<=0.5): {cluster_stats['low_cp_clusters']}")
    print(f"  Avg CP score: {cluster_stats['avg_cp_score']:.4f}")

    return cluster_centers, cluster_cp_scores, cluster_stats
