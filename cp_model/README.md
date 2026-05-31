# cp_model/

CP model implementation. Contains four files:

- `cp_model.py` — `CoverageTransformer` (main model) and `MultiMetricMLP` (baseline)
- `cag.py` — Criterion-Aware Gating module
- `latent_coverage.py` — `LatentSpaceClusterCoverage` tracker and `train_latent_space_clusters()`
- `dataset.py` — `MultiMetricDataset` for loading precomputed coverage features

Trained checkpoints are saved to `cp_checkpoints/`, not here.

## Example

```python
from cp_model import CoverageTransformer, LatentSpaceClusterCoverage

checkpoint = torch.load('cp_checkpoints/CIFAR10/best_model.pt')
cp_model = CoverageTransformer(pca_dim=64, num_criteria=10, d_model=128, nhead=4, num_layers=3)
cp_model.load_state_dict(checkpoint['model_state_dict'])

lscc = LatentSpaceClusterCoverage(
    n_clusters=30,
    cluster_centers=checkpoint['cluster_centers'],
    cluster_cp_scores=checkpoint['cluster_cp_scores'],
    lambda_param=0.7
)
```
