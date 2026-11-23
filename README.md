# Sparse Stochastic Digital Twins (PyTorch)

A PyTorch reimplementation of the dependence-discovery method from the paper
**"Sparse Stochastic Digital Twins Can Outperform Exact Ones"** by Zongren Zou, Théo Bourdais,
Ricardo Baptista and Houman Owhadi ([arXiv:2606.16219](https://arxiv.org/abs/2606.16219)).

The original JAX codebase is at [ZongrenZou/sparse-digital-twins](https://github.com/ZongrenZou/sparse-digital-twins).
This repository is an independent rewrite, not the authors' code. It currently covers the
economic-data example (Section 2.4: oil returns and geopolitical risk).

## Method

1. **Conditional generative model.** Train a flow-matching model that samples `y | x`.
2. **Distillation.** Draw triples `(x, z, ŷ)` from the trained flow, where `z` is the input noise.
3. **Kernel ANOVA.** Fit a Gaussian process with the product kernel
   `K = Π_z (1 + k_z) · Π_i (1 + k_x,i)` and score each input by the share of RKHS energy
   in the kernel components that involve it.
4. **Greedy pruning.** Remove the lowest-scoring input, refit, and record the noise-to-signal
   ratio (NSR). The selected inputs are those present just before the largest NSR increase.

## Files

| File | Contents |
|---|---|
| `dataset.py` | `DiscoverySettings` (all settings) and `OilGprDataset` (loads the `.mat` file and builds lagged inputs) |
| `models.py` | `ConditionalFlow` (flow matching) and `KernelANOVAPruner` (GP contribution scores and pruning) |
| `pipeline.py` | `DependenceDiscovery`, which runs the full method and returns a `DiscoveryResult` |
| `metrics.py` | Probabilistic scores for the generator |
| `results.py` | `DiscoveryResult` and `ResultExporter` (figures, CSV tables, JSON summary) |
| `run.py` | Dataset instances for each period and the calls that run them |

## Setup

```bash
pip install torch numpy scipy matplotlib
```

Place the data at `data/data_oil_gpr.mat`, an `(n, 2)` array named `data` with columns
`[oil price, GPR index]`. The preprocessed file is provided in the
[original repository](https://github.com/ZongrenZou/sparse-digital-twins). The raw sources are
WTI prices from the Federal Reserve Bank of St. Louis and the GPR index from
[Caldara and Iacoviello](https://www.matteoiacoviello.com/gpr.htm).

## Usage

```bash
python run.py
```

`QUICK_TEST = True` in `run.py` runs one short job to check the pipeline. Set it to `False`
for the full settings and all periods (the 4 main periods and 8 five-year periods from the paper).

To run a single configuration:

```python
from dataset import OilGprDataset
from pipeline import DependenceDiscovery
from results import ResultExporter

dataset = OilGprDataset(name="oil_gpr_2000_2009", rows=(3550, 6056))
result = DependenceDiscovery(dataset).run()
ResultExporter("results").export(result)
```

To add a dataset, subclass `DiscoverySettings` and implement `load()` returning
`x, y, names, prunable`, where `prunable` lists the column indices that may be removed.

## Outputs

Each run writes to `results/<name>/`:

- `nsr.png`: NSR after each removal, with the stopping point marked
- `contribution_scores.png`: first-iteration scores and scores along the pruning path
- `generator_diagnostics.png`: PIT histogram and flow training loss
- `pruning_history.csv`: NSR, GP train/validation error and scores at each step
- `summary.json`: selected inputs, last-pruned input, pruning order and metrics

`export_comparison` writes a cross-period bar chart and `comparison.csv`.

## Evaluation metrics

Computed on the held-out final 20% of each period:

- **CRPS** and **CRPS skill** against an unconditional baseline from the training returns
- **50% and 90% interval coverage** (targets 0.5 and 0.9)
- **5% tail exceedance** (target 0.05)
- **PIT Kolmogorov–Smirnov statistic** (lower means better calibrated)
- **Wasserstein-1 distance** between generated and observed returns

## Differences from the original code

- The flow trains on the first 80% of each period so metrics are out of sample.
  Set `holdout_fraction=0` to train on all data as the original does.
- Only the samples used by the GP are generated, rather than 100 per row followed by subsampling.
- Sampling uses fixed-step RK4 (`ode_steps=50`) instead of adaptive Tsit5.
- The largest-NSR-jump selection rule from the paper is applied and reported.
- The velocity network has 2 hidden layers of 64 units, matching the original code
  (the paper's appendix states four).

## Notes

The full GP step builds `n_distill × n_distill` float64 matrices (10,000 by default),
which needs a few GB of memory. Lower `n_distill` on machines with limited RAM.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{zou2026sparse,
  title   = {Sparse Stochastic Digital Twins Can Outperform Exact Ones},
  author  = {Zou, Zongren and Bourdais, Th{\'e}o and Baptista, Ricardo and Owhadi, Houman},
  journal = {arXiv preprint arXiv:2606.16219},
  year    = {2026}
}
```
