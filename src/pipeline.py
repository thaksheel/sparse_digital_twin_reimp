import numpy as np
import torch

from metrics import forecast_metrics
from models import ConditionalFlow, KernelANOVAPruner
from results import DiscoveryResult


def standardize(a, ref):
    return (a - ref.mean(0)) / ref.std(0)


class DependenceDiscovery:
    """Fit a conditional generator, distil it into a GP and prune inputs by contribution score.

    Steps: (1) train flow matching on the training rows, (2) score its samples on the
    held-out rows, (3) draw (x, z, y_hat) triples from the flow, (4) run kernel-ANOVA
    pruning on those triples and pick the inputs kept before the largest NSR jump.

    Parameters
    ----------
    dataset : DiscoverySettings
        Dataset object that supplies the data and all settings.
    device : str or None
        Torch device; defaults to CUDA when available.
    """

    def __init__(self, dataset, device=None):
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        ds = self.dataset
        torch.manual_seed(ds.seed)
        rng = np.random.default_rng(ds.seed)
        x, y, names, prunable = ds.load()
        n_train = int(len(x) * (1 - ds.holdout_fraction))
        test = slice(n_train, None) if n_train < len(x) else slice(None)

        xs = torch.tensor(standardize(x, x[:n_train]), dtype=torch.float32)
        ys = torch.tensor(standardize(y, y[:n_train]), dtype=torch.float32)
        y_mu, y_sd = y[:n_train].mean(0), y[:n_train].std(0)

        flow = ConditionalFlow(x.shape[1], 1, ds.hidden, ds.depth, ds.lr, self.device)
        flow.fit(xs[:n_train], ys[:n_train], ds.epochs, ds.batch_size)

        x_test = xs[test].repeat_interleave(ds.eval_samples, dim=0)
        samples = flow.sample(x_test, steps=ds.ode_steps).view(-1, ds.eval_samples).numpy()
        metrics, pit = forecast_metrics(samples * y_sd + y_mu, y[test, 0], y[:n_train, 0], rng)

        m = ds.n_distill
        rows = torch.tensor(rng.integers(0, len(x), 2 * m))
        z = torch.randn(2 * m, 1)
        y_hat = flow.sample(xs[rows], z, ds.ode_steps)
        X, Y, Z = (t.double().to(self.device) for t in (xs[rows], y_hat, z))
        X, Y = standardize(X, X[:m]), standardize(Y, Y[:m])

        pruner = KernelANOVAPruner(ds.length_x, ds.length_z, float(np.exp(ds.log_nugget)))
        history = pruner.prune(X[:m], Z[:m], Y[:m], X[m:], Z[m:], Y[m:], prunable, names)

        prunable_names = [names[j] for j in prunable]
        jump = int(np.argmax(np.diff([h["nsr"] for h in history])))
        return DiscoveryResult(
            name=ds.name,
            feature_names=names,
            prunable_names=prunable_names,
            history=history,
            selected=[v for v in history[jump]["active"] if v in prunable_names],
            last_pruned=history[-2]["pruned"],
            metrics=metrics,
            pit=pit,
            flow_losses=flow.losses,
        )
