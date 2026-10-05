"""
Experiment 1: One-dimensional consolidation problem solved with a PINN.

Dimensionless PDE (Eq. 4.4 of the paper), for 0 < x < 1 and 0 < t <= T_MAX:

    dp/dt - d2p/dx2 = 0
    p = 0        at x = 0          (enforced exactly: output is multiplied by x)
    dp/dx = 0    at x = 1          (Neumann, enforced in the loss)
    p = 1        at t = 0, x > 0   (initial condition, enforced in the loss)

Reference: analytical series solution (Eq. 4.5).

Run from the repository root:
    python experiments/exp1_consolidation.py                 # full run
    python experiments/exp1_consolidation.py --epochs 2000 --lbfgs-iters 500   # quick test
    python experiments/exp1_consolidation.py --load          # reuse saved weights, only plot

Outputs are saved in results/ (figure + trained weights).
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Hyper-parameters (values taken from Section 4.1 of the paper)
# ----------------------------------------------------------------------------
SEED = 42
T_MAX = 0.5                # dimensionless time horizon
N_RESIDUAL = 500           # collocation points inside the domain
N_NEUMANN = 250            # boundary points at x = 1
N_INITIAL = 125            # initial-condition points at t = 0
N_TEST = 10011             # test points
W_F = 1 / 4                # weight of PDE loss
W_B = 1 / 4                # weight of boundary/initial loss
HIDDEN_LAYERS = 5
WIDTH = 50
ADAM_EPOCHS = 50000
ADAM_LR = 1e-3
LBFGS_ITERS = 5000         # upper bound on L-BFGS iterations ("until convergence")
LOG_EVERY = 500

torch.set_default_dtype(torch.float64)
OUT_DIR = Path.cwd() / "results"


# ----------------------------------------------------------------------------
# Analytical reference solution (Eq. 4.5)
# ----------------------------------------------------------------------------
def analytical(x, t, n_terms=1000):
    """p_hat(x, t) = sum over odd m of 4/(m*pi) * sin(m*pi*x/2) * exp(-m^2 pi^2 t / 4)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    m = np.arange(1, 2 * n_terms, 2)
    series = (4 / (m * np.pi)) * np.sin(m * np.pi * x[..., None] / 2) \
        * np.exp(-(m ** 2) * np.pi ** 2 * t[..., None] / 4)
    p = series.sum(axis=-1)
    # The series is not reliable exactly at t = 0, so use the initial condition there.
    return np.where(t <= 0, np.where(x > 0, 1.0, 0.0), p)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class PINN(nn.Module):
    """MLP (x, t) -> p_hat. Output is multiplied by x so that p_hat(0, t) = 0 exactly."""

    def __init__(self, hidden_layers=HIDDEN_LAYERS, width=WIDTH):
        super().__init__()
        dims = [2] + [width] * hidden_layers + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)   # Glorot normal
            nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return x * self.layers[-1](h)              # linear output, times x


# ----------------------------------------------------------------------------
# Loss terms (automatic differentiation)
# ----------------------------------------------------------------------------
def pde_residual(model, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    p = model(x, t)
    p_t = torch.autograd.grad(p, t, torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, torch.ones_like(p_x), create_graph=True)[0]
    return p_t - p_xx


def neumann_residual(model, x, t):
    x = x.clone().requires_grad_(True)
    p = model(x, t)
    return torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]


def sample_points():
    """Randomly sample residual, boundary and test points."""
    pts = {
        "x_f": torch.rand(N_RESIDUAL, 1),
        "t_f": T_MAX * torch.rand(N_RESIDUAL, 1),
        "x_n": torch.ones(N_NEUMANN, 1),
        "t_n": T_MAX * torch.rand(N_NEUMANN, 1),
        "x_i": torch.rand(N_INITIAL, 1),
        "t_i": torch.zeros(N_INITIAL, 1),
    }
    test = (torch.rand(N_TEST, 1), T_MAX * torch.rand(N_TEST, 1))
    return pts, test


def total_loss(model, pts):
    l_f = (pde_residual(model, pts["x_f"], pts["t_f"]) ** 2).mean()
    r_neumann = neumann_residual(model, pts["x_n"], pts["t_n"])
    r_initial = model(pts["x_i"], pts["t_i"]) - 1.0
    l_b = (torch.cat([r_neumann, r_initial]) ** 2).mean()
    return W_F * l_f + W_B * l_b


def evaluate(model, test, test_ref):
    x, t = test
    test_pde_loss = (pde_residual(model, x, t) ** 2).mean().item()
    with torch.no_grad():
        p = model(x, t).clamp(0, 1)
    mse = ((p - test_ref) ** 2).mean().item()
    return test_pde_loss, mse


# ----------------------------------------------------------------------------
# Training: Adam, then L-BFGS
# ----------------------------------------------------------------------------
def train(model, pts, test, test_ref, adam_epochs, lbfgs_iters):
    hist = {"iter": [], "train": [], "test": [], "mse": []}

    def log(it, train_loss):
        test_loss, mse = evaluate(model, test, test_ref)
        hist["iter"].append(it)
        hist["train"].append(train_loss)
        hist["test"].append(test_loss)
        hist["mse"].append(mse)

    start = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    for ep in range(adam_epochs + 1):
        opt.zero_grad()
        loss = total_loss(model, pts)
        loss.backward()
        opt.step()
        if ep % LOG_EVERY == 0:
            log(ep, loss.item())
            if ep % (LOG_EVERY * 2) == 0:
                print(f"[Adam] epoch {ep:6d} | train loss {loss.item():.3e} "
                      f"| test PDE loss {hist['test'][-1]:.3e} "
                      f"| MSE vs analytical {hist['mse'][-1]:.3e} "
                      f"| {time.time() - start:.0f}s")

    if lbfgs_iters > 0:
        print("Switching to L-BFGS ...")
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_iters,
            max_eval=int(lbfgs_iters * 1.25), history_size=50,
            tolerance_grad=1e-10, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            l = total_loss(model, pts)
            l.backward()
            return l

        lbfgs.step(closure)
        n_iter = lbfgs.state[lbfgs._params[0]]["n_iter"]
        log(adam_epochs + n_iter, total_loss(model, pts).item())
        print(f"[L-BFGS] {n_iter} iterations | train loss {hist['train'][-1]:.3e} "
              f"| MSE vs analytical {hist['mse'][-1]:.3e}")

    print(f"Training finished in {time.time() - start:.0f}s")
    return hist


# ----------------------------------------------------------------------------
# Plots and error metrics
# ----------------------------------------------------------------------------
def make_plots(model, hist, out_path):
    xs = np.linspace(0, 1, 101)
    ts = np.linspace(0, T_MAX, 101)
    X, T = np.meshgrid(xs, ts)                       # shape (n_t, n_x)
    ref = analytical(X, T)
    with torch.no_grad():
        pred = model(torch.tensor(X.reshape(-1, 1)), torch.tensor(T.reshape(-1, 1)))
    pred = pred.clamp(0, 1).numpy().reshape(X.shape)  # outputs limited to [0, 1]
    err = np.abs(pred - ref)

    print("\nResults on a 101 x 101 grid:")
    print(f"  max absolute error : {err.max():.3e}")
    print(f"  mean absolute error: {err.mean():.3e}")
    print(f"  relative L2 error  : {np.linalg.norm(pred - ref) / np.linalg.norm(ref):.3e}")

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    s = ax.plot_surface(X, T, ref, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Reference solution (analytical)")
    ax.set_xlabel(r"$\hat{x}$"); ax.set_ylabel(r"$\hat{t}$"); ax.set_zlabel(r"$\hat{p}$")
    fig.colorbar(s, ax=ax, shrink=0.6)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    s = ax.plot_surface(X, T, pred, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Network solution (PINN)")
    ax.set_xlabel(r"$\hat{x}$"); ax.set_ylabel(r"$\hat{t}$"); ax.set_zlabel(r"$\hat{p}$")
    fig.colorbar(s, ax=ax, shrink=0.6)

    ax = fig.add_subplot(2, 2, 3)
    c = ax.pcolormesh(X, T, err, cmap="jet", shading="auto")
    ax.set_title("Absolute error")
    ax.set_xlabel(r"$\hat{x}$"); ax.set_ylabel(r"$\hat{t}$")
    fig.colorbar(c, ax=ax)

    ax = fig.add_subplot(2, 2, 4)
    ax.semilogy(hist["iter"], hist["train"], label="PDE train loss")
    ax.semilogy(hist["iter"], hist["test"], label="PDE test loss")
    ax.semilogy(hist["iter"], hist["mse"], label="MSE vs analytical")
    ax.set_title("Convergence history")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("PINN: 1D consolidation problem", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Figure saved to {out_path}")
    plt.show()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ADAM_EPOCHS, help="Adam epochs")
    parser.add_argument("--lbfgs-iters", type=int, default=LBFGS_ITERS, help="max L-BFGS iterations")
    parser.add_argument("--load", action="store_true", help="load saved weights instead of training")
    args, _ = parser.parse_known_args()      # parse_known_args keeps this Colab-safe

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    weights_path = OUT_DIR / "exp1_model.pt"

    model = PINN()
    pts, test = sample_points()
    test_ref = torch.tensor(analytical(test[0].numpy(), test[1].numpy()))

    if args.load and weights_path.exists():
        saved = torch.load(weights_path)
        model.load_state_dict(saved["state"])
        hist = saved["history"]
        print(f"Loaded weights from {weights_path}")
    else:
        hist = train(model, pts, test, test_ref, args.epochs, args.lbfgs_iters)
        torch.save({"state": model.state_dict(), "history": hist}, weights_path)
        print(f"Weights saved to {weights_path}")

    make_plots(model, hist, OUT_DIR / "exp1_results.png")


if __name__ == "__main__":
    main()
