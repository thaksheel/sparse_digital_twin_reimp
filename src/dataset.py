from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class DiscoverySettings:
    """Settings shared by every dataset: flow matching, distillation and the GP.

    Parameters
    ----------
    name : str
        Label used for output folders.
    hidden, depth : int
        Width and number of hidden layers of the velocity network.
    epochs, batch_size : int
        Flow-matching training passes over the data and minibatch size.
    lr : float
        Adam learning rate.
    ode_steps : int
        RK4 steps used to push noise through the learned flow.
    eval_samples : int
        Generated samples per held-out input when scoring the generator.
    n_distill : int
        Size of the GP training set (an equal-sized validation set is also drawn).
    length_x, length_z : float
        RBF lengthscales for the inputs and the latent noise.
    log_nugget : float
        Log of the GP regularisation (nugget) gamma.
    holdout_fraction : float
        Final fraction of the rows held out to evaluate the generator.
        Set to 0 to train on everything, as in the original code.
    seed : int
        Random seed.
    """

    name: str = "run"
    hidden: int = 64
    depth: int = 2
    epochs: int = 5000
    batch_size: int = 1000
    lr: float = 1e-4
    ode_steps: int = 50
    eval_samples: int = 200
    n_distill: int = 10_000
    length_x: float = 5.0
    length_z: float = 5.0
    log_nugget: float = 1.0
    holdout_fraction: float = 0.2
    seed: int = 1234

    def load(self):
        """Return x (n, d), y (n, 1), feature names and the column indices that may be pruned."""
        raise NotImplementedError


@dataclass
class OilGprDataset(DiscoverySettings):
    """Daily oil log-returns conditioned on lagged returns and lagged log-GPR.

    Parameters
    ----------
    path : str
        .mat file holding an (n, 2) array with columns [oil price, GPR index].
    mat_key : str
        Variable name inside the .mat file.
    rows : tuple of int or None
        (start, end) slice of the raw series; None keeps all rows.
    return_lags, gpr_lags : tuple of int
        Lags, in retained trading days, for each input group. Only GPR lags are pruned.
    """

    path: str = "data/data_oil_gpr.mat"
    mat_key: str = "data"
    rows: tuple | None = None
    return_lags: tuple = (1, 5, 10, 15, 20)
    gpr_lags: tuple = (1, 5, 10, 15, 20)

    def load(self):
        data = sio.loadmat(self.path)[self.mat_key]
        if self.rows is not None:
            data = data[slice(*self.rows)]
        returns = np.diff(np.log(data[:, 0]))
        log_gpr = np.log(data[:-1, 1])
        m, n = max(self.return_lags + self.gpr_lags), len(returns)

        columns = [log_gpr[m - lag + 1 : n - lag + 1] for lag in self.gpr_lags]
        columns += [returns[m - lag : n - lag] for lag in self.return_lags]
        names = [f"GPR lag {lag}" for lag in self.gpr_lags]
        names += [f"return lag {lag}" for lag in self.return_lags]
        prunable = list(range(len(self.gpr_lags)))
        return np.stack(columns, axis=1), returns[m:, None], names, prunable
