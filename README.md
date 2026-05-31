# Coverage-based Predictor (CP)

Code for the paper: ***One Model to Rule Them All: Unifying Test Coverage via a Transformer-based Fuzzing for Deep Neural Networks***

## Overview

Coverage-guided fuzzing (CGF) for DNNs typically relies on scalar coverage values that correlate weakly with model faults. This work proposes a learning-based CGF approach centered on a Coverage-based Predictor (CP). CP treats statistical features derived from multiple coverage criteria as structured representations of DNN behavior and uses a Transformer with Criterion-Aware Gating (CAG) to learn their association with failures. We further introduce Latent Space Clustering Coverage (LSCC) to characterize behavioral-cluster exploration, and combine CP risk scores with LSCC coverage gains as dual feedback for test generation:

$$U(x, t) = \text{scp}(x) + \lambda_{\text{LSCC}} \cdot \Delta C(x, t)$$

## Coverage Criteria

Ten criteria implemented in `coverage.py`, extending [NeuraL-Coverage](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage):

| Criterion | Reference |
|-----------|-----------|
| NC | DeepXplore, SOSP 2017 |
| KMNC, NBC, SNAC, TKNC, TKNP | DeepGauge, ASE 2018 |
| CC | TensorFuzz, ICML 2019 |
| LSC, DSC | Surprise Adequacy, ICSE/FSE 2019 |
| NLC | NeuraL-Coverage, ICSE 2023 |

## Installation

```bash
git clone https://github.com/GVCMch/Coverage-based-Predictor
cd Coverage-based-Predictor
pip install -r requirements.txt
```

Python 3.8+, PyTorch 2.4.1, CUDA 11.8+

## Usage

**Step 1 — Extract coverage matrices**

```bash
cd coverage_matrices
python extract_multi_metric_coverage.py --dataset CIFAR10 --num_seeds 10000 --K 15
```

**Step 2 — Train CP**

```bash
python train.py --dataset CIFAR10 --pca_dim 32 --epochs 100
```

**Step 3 — (Optional) Tune λ_LSCC via Bayesian optimization**

```bash
cd bayesian && python tune_weights.py --dataset CIFAR10 --n_calls 40
```

**Step 4 — Run fuzzing**

```bash
python fuzz.py --dataset CIFAR10 --model resnet50 --num_seeds 100 --num_iterations 1000 --strategy guided_random_walk
```

**Step 5 — Evaluate**

```bash
python eval/fuzz_eval_fdr_baseline.py --dataset CIFAR10 --model resnet50
python eval/fuzz_eval_coverage_correlation.py --dataset CIFAR10 --model resnet50
```

## Datasets and Models

Evaluated on four datasets (MNIST, Fashion-MNIST, CIFAR-10, TinyImageNet), 14 distinct DNN architectures, and 20 model-dataset combinations. See Table 1 in the paper for details.

## Acknowledgements

Coverage criteria extended from [NeuraL-Coverage](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage).

## License

MIT — see [LICENSE](LICENSE).
