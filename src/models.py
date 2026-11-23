import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    def __init__(self, dim_y, dim_x, hidden, depth):
        super().__init__()
        layers, width = [], dim_y + 1 + dim_x
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, dim_y))

    def forward(self, t, z, x):
        return self.net(torch.cat([z, t, x], dim=1))


class ConditionalFlow:
    """Flow-matching generator for the conditional law y | x.

    Learns a velocity field v(t, z, x) that carries Gaussian noise z ~ N(0, I)
    at t = 0 to a sample of y | x at t = 1.

    Parameters
    ----------
    dim_x, dim_y : int
        Input and output dimensions.
    hidden, depth : int
        Width and number of hidden layers of the velocity network.
    lr : float
        Adam learning rate.
    device : str
        Torch device.
    """

    def __init__(self, dim_x, dim_y, hidden=64, depth=2, lr=1e-4, device="cpu"):
        self.dim_y, self.device = dim_y, device
        self.net = VelocityNet(dim_y, dim_x, hidden, depth).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.losses = []

    def fit(self, x, y, epochs, batch_size, log_every=100):
        x, y = x.to(self.device), y.to(self.device)
        for epoch in range(epochs):
            total = 0.0
            for idx in torch.randperm(len(x), device=self.device).split(batch_size):
                z0 = torch.randn(len(idx), self.dim_y, device=self.device)
                t = torch.rand(len(idx), 1, device=self.device)
                z_t = (1 - t) * z0 + t * y[idx]
                loss = ((self.net(t, z_t, x[idx]) - (y[idx] - z0)) ** 2).sum(1).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += loss.item() * len(idx)
            self.losses.append(total / len(x))
            if (epoch + 1) % log_every == 0:
                print(f"epoch {epoch + 1}: flow loss {self.losses[-1]:.4f}", flush=True)
        return self

    @torch.no_grad()
    def sample(self, x, z0=None, steps=50, chunk=200_000):
        """Integrate the flow with RK4 from z0 (drawn if None); returns samples on the CPU."""
        if z0 is None:
            z0 = torch.randn(len(x), self.dim_y)
        out = [self._integrate(z.to(self.device), xc.to(self.device), steps).cpu()
               for z, xc in zip(z0.split(chunk), x.split(chunk))]
        return torch.cat(out)

    def _integrate(self, z, x, steps):
        h = 1.0 / steps
        for k in range(steps):
            t = torch.full((len(z), 1), k * h, device=z.device)
            k1 = self.net(t, z, x)
            k2 = self.net(t + h / 2, z + h / 2 * k1, x)
            k3 = self.net(t + h / 2, z + h / 2 * k2, x)
            k4 = self.net(t + h, z + h * k3, x)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z


def rbf(a, b, length):
    return torch.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * length**2))


class KernelANOVAPruner:
    """Greedy input pruning of a distilled generator with a product-kernel GP.

    The GP regresses generated samples y on (x, z) with
    K = prod_z (1 + k_z) * prod_{active i} (1 + k_x,i). The part of K that involves
    input j is K_j = K * k_j / (1 + k_j), and its contribution score is
    ||f_j||^2 / ||f||^2 = alpha' K_j alpha / alpha' K alpha (kernel mode decomposition).

    Parameters
    ----------
    length_x, length_z : float
        RBF lengthscales for inputs and latent noise.
    nugget : float
        Regularisation gamma of the kernel ridge regression.
    """

    def __init__(self, length_x, length_z, nugget):
        self.length_x, self.length_z, self.nugget = length_x, length_z, nugget

    def gram(self, xa, za, xb, zb, active):
        K = torch.ones(len(xa), len(xb), dtype=xa.dtype, device=xa.device)
        for i in range(za.shape[1]):
            K *= 1 + rbf(za[:, i], zb[:, i], self.length_z)
        for i in range(xa.shape[1]):
            if active[i]:
                K *= 1 + rbf(xa[:, i], xb[:, i], self.length_x)
        return K

    @torch.no_grad()
    def prune(self, x, z, y, x_val, z_val, y_val, prunable, names):
        """Remove the lowest-scoring prunable input until none is left; return one record per step."""
        active = [True] * x.shape[1]
        history = []
        for _ in range(len(prunable) + 1):
            K = self.gram(x, z, x, z, active)
            A = K.clone()
            A.diagonal().add_(self.nugget)
            alpha = torch.linalg.solve(A, y)
            del A
            signal = (alpha.T @ K @ alpha).item()
            noise = self.nugget * (alpha**2).sum().item()

            scores = {}
            for j in prunable:
                if active[j]:
                    kj = rbf(x[:, j], x[:, j], self.length_x)
                    scores[names[j]] = (alpha.T @ ((K * kj / (1 + kj)) @ alpha)).item() / signal

            val_pred = self.gram(x_val, z_val, x, z, active) @ alpha
            step = {
                "active": [n for n, a in zip(names, active) if a],
                "nsr": noise / (noise + signal),
                "train_mse": ((K @ alpha - y) ** 2).mean().item(),
                "val_mse": ((val_pred - y_val) ** 2).mean().item(),
                "scores": scores,
                "pruned": None,
            }
            if scores:
                step["pruned"] = min(scores, key=scores.get)
                active[names.index(step["pruned"])] = False
            history.append(step)
            print(f"NSR {step['nsr']:.4f}, pruned {step['pruned']}", flush=True)
        return history
