# bayesian/

Stores output of `tune_weights.py`. Gitignored.

`tune_weights.py` uses Gaussian Process optimization to search for the optimal λ_LSCC weight. Output is saved as `bayesian_optimal_weights_{dataset}.json`.

```json
{
  "resnet50": {
    "best_lambda_lscc": 0.72,
    "best_fdr": 0.7377,
    "val_fdr_mean": 0.6978
  }
}
```

```bash
cd bayesian
python tune_weights.py --dataset CIFAR10 --n_calls 40
```
