import numpy as np
from scipy import stats


def crps(samples, y):
    """Sample-based CRPS per row: E|X - y| - 0.5 E|X - X'| (lower is better)."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    weights = 2 * np.arange(1, m + 1) - m - 1
    half_spread = (s * weights).sum(1) / m**2
    return np.abs(samples - y[:, None]).mean(1) - half_spread


def forecast_metrics(samples, y, y_reference, rng):
    """Score conditional samples (n, s) against observed outcomes y (n,).

    The unconditional baseline draws from the training targets y_reference, so
    crps_skill > 0 means conditioning on the inputs helps.
    """
    baseline = rng.choice(y_reference, size=samples.shape)
    score, score_base = crps(samples, y).mean(), crps(baseline, y).mean()
    pit = (samples < y[:, None]).mean(1)
    q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95], axis=1)
    metrics = {
        "crps": score,
        "crps_unconditional": score_base,
        "crps_skill": 1 - score / score_base,
        "coverage_50": np.mean((y >= q25) & (y <= q75)),
        "coverage_90": np.mean((y >= q05) & (y <= q95)),
        "var05_exceedance": np.mean(y < q05),
        "pit_ks": stats.kstest(pit, "uniform").statistic,
        "wasserstein_marginal": stats.wasserstein_distance(samples.ravel(), y),
    }
    return {k: float(v) for k, v in metrics.items()}, pit
