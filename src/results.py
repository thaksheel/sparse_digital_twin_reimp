import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DiscoveryResult:
    """Output of one DependenceDiscovery run.

    Attributes
    ----------
    name : str
        Run label (used as the output folder name).
    feature_names, prunable_names : list of str
        All conditioning inputs, and the subset that was allowed to be pruned.
    history : list of dict
        One record per pruning step: active inputs, NSR, GP train/val MSE,
        contribution scores of the active prunable inputs, and the input pruned.
    selected : list of str
        Prunable inputs kept immediately before the largest NSR increase.
    last_pruned : str
        Prunable input that survived longest (the dominant one).
    metrics : dict
        Probabilistic scores of the generator on held-out rows.
    pit : numpy.ndarray
        Probability integral transform values of the held-out outcomes.
    flow_losses : list of float
        Flow-matching loss per epoch.
    """

    name: str
    feature_names: list
    prunable_names: list
    history: list
    selected: list
    last_pruned: str
    metrics: dict
    pit: np.ndarray
    flow_losses: list

    @property
    def nsr(self):
        return np.array([h["nsr"] for h in self.history])

    @property
    def pruning_order(self):
        return [h["pruned"] for h in self.history[:-1]]

    @property
    def first_scores(self):
        return np.array([self.history[0]["scores"][v] for v in self.prunable_names])


class ResultExporter:
    """Writes figures, CSV tables and a JSON summary for DiscoveryResult objects.

    Parameters
    ----------
    out_dir : str
        Root folder; each result gets its own sub-folder.
    """

    def __init__(self, out_dir="results"):
        self.out_dir = Path(out_dir)

    def export(self, result):
        folder = self.out_dir / result.name
        folder.mkdir(parents=True, exist_ok=True)
        self.plot_nsr(result, folder / "nsr.png")
        self.plot_scores(result, folder / "contribution_scores.png")
        self.plot_generator(result, folder / "generator_diagnostics.png")
        self.write_history(result, folder / "pruning_history.csv")
        self.write_summary(result, folder / "summary.json")
        return folder

    def export_comparison(self, results, label):
        folder = self.out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        self.plot_comparison(results, folder / "first_iteration_scores.png")
        keys = ["crps_skill", "coverage_50", "coverage_90", "var05_exceedance", "pit_ks"]
        with open(folder / "comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "last_pruned", "selected", *keys])
            for r in results:
                w.writerow([r.name, r.last_pruned, "; ".join(r.selected)] + [round(r.metrics[k], 4) for k in keys])
        return folder

    def plot_nsr(self, result, path):
        nsr = result.nsr
        steps = ["all"] + [f"-{v}" for v in result.pruning_order]
        jump = int(np.argmax(np.diff(nsr)))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(steps, nsr, "k-o")
        ax[0].set_title("Noise-to-signal ratio after each removal")
        ax[1].bar(steps[1:], np.diff(nsr), color=["C3" if i == jump else "0.6" for i in range(len(nsr) - 1)])
        ax[1].set_title("NSR increment (red = stopping point)")
        for a in ax:
            a.tick_params(axis="x", rotation=30)
        fig.suptitle(result.name)
        self._save(fig, path)

    def plot_scores(self, result, path):
        names = result.prunable_names
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = ["C1" if v == result.last_pruned else "0.6" for v in names]
        ax[0].bar(names, result.first_scores, color=colors)
        ax[0].set_title("First-iteration contribution (orange = last pruned)")
        path_scores = np.array([[h["scores"].get(v, np.nan) for v in names] for h in result.history[:-1]])
        im = ax[1].imshow(path_scores, aspect="auto", cmap="viridis")
        ax[1].set_xticks(range(len(names)), names)
        ax[1].set_yticks(range(len(path_scores)), [f"step {i}" for i in range(len(path_scores))])
        ax[1].set_title("Contribution score along the pruning path")
        fig.colorbar(im, ax=ax[1])
        for a in ax:
            a.tick_params(axis="x", rotation=30)
        self._save(fig, path)

    def plot_generator(self, result, path):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].hist(result.pit, bins=20, density=True, color="0.6")
        ax[0].axhline(1, color="C3", ls="--")
        ax[0].set_title(f"PIT histogram (flat = calibrated), KS = {result.metrics['pit_ks']:.3f}")
        ax[1].plot(result.flow_losses, "k-")
        ax[1].set_yscale("log")
        ax[1].set_title("Flow-matching training loss")
        ax[1].set_xlabel("epoch")
        self._save(fig, path)

    def plot_comparison(self, results, path):
        fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.5), sharey=True, squeeze=False)
        for ax, r in zip(axes[0], results):
            scores = r.first_scores / r.first_scores.sum()
            ax.bar(range(len(scores)), scores, color=["C1" if v == r.last_pruned else "0.6" for v in r.prunable_names])
            ax.set_xticks(range(len(scores)), [v.split()[-1] for v in r.prunable_names])
            ax.set_title(r.name, fontsize=9)
        axes[0, 0].set_ylabel("relative contribution")
        self._save(fig, path)

    def write_history(self, result, path):
        nsr_inc = np.r_[np.diff(result.nsr), np.nan]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "pruned", "nsr", "nsr_increment", "gp_train_mse", "gp_val_mse",
                        *[f"score[{v}]" for v in result.prunable_names]])
            for i, h in enumerate(result.history):
                w.writerow([i, h["pruned"], h["nsr"], nsr_inc[i], h["train_mse"], h["val_mse"],
                            *[h["scores"].get(v, "") for v in result.prunable_names]])

    def write_summary(self, result, path):
        summary = {
            "name": result.name,
            "selected": result.selected,
            "last_pruned": result.last_pruned,
            "pruning_order": result.pruning_order,
            "first_iteration_scores": dict(zip(result.prunable_names, result.first_scores.tolist())),
            "generator_metrics": result.metrics,
        }
        path.write_text(json.dumps(summary, indent=2))

    def _save(self, fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
