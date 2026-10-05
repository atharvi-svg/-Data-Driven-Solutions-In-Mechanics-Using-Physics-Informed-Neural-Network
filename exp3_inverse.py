"""
Experiment 3: Inverse problem for a damped spring-mass system, solved with a PINN.

ODE (Eq. 4.8 of the paper), with m = 1, u(0) = 1, u'(0) = 1:

    m u'' + c u' + k u = 0

The damping coefficient c and stiffness k are UNKNOWN. The PINN learns u(t) and
finds c and k from 50 observations of the displacement. The observations are
generated from the analytical solution (Eq. 4.9) with the true values c = 0.4, k = 4.

Run from the repository root:
    python experiments/exp3_inverse.py                        # full run (100,000 epochs)
    python experiments/exp3_inverse.py --epochs 5000          # quick test
    python experiments/exp3_inverse.py --noise 0.02           # noisy observations (extra)
    python experiments/exp3_inverse.py --c0 1.0 --k0 6.0      # different initial guesses (extra)
    python experiments/exp3_inverse.py --load                 # reuse saved weights, only plot

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
# Problem setup and hyper-parameters (values from Section 4.3 of the paper)
# ----------------------------------------------------------------------------
SEED = 42
M, U0, V0 = 1.0, 1.0, 1.0          # mass, initial displacement, initial velocity
C_TRUE, K_TRUE = 0.4, 4.0          # true parameters (used only to make the data)
C_INIT, K_INIT = 0.2, 2.0          # initial guesses (our choice, not given in the paper)
T_MAX = 15.0                       # time horizon
N_DATA = 50                        # equispaced observations
N_RESIDUAL = 100                   # random collocation points
N_TEST = 10000                     # random test points
W_F = W_B = W_D = 1 / 4            # loss weights
HIDDEN_LAYERS = 3
WIDTH = 32
ADAM_EPOCHS = 100000
ADAM_LR = 5e-4
LOG_EVERY = 1000                   # loss history (includes a test-set evaluation)
PARAM_EVERY = 100                  # c and k history

OUT_DIR = Path.cwd() / "results"


# ----------------------------------------------------------------------------
# Analytical solution (Eq. 4.9) -- underdamped case (xi < 1)
# ----------------------------------------------------------------------------
def analytical(t, c=C_TRUE, k=K_TRUE):
    t = np.asarray(t, dtype=float)
    wn = np.sqrt(k / M)
    xi = c / (2 * M * wn)
    wd = wn * np.sqrt(1 - xi ** 2)
    return np.exp(-xi * wn * t) * (U0 * np.cos(wd * t) + (V0 + xi * wn * U0) / wd * np.sin(wd * t))


# ----------------------------------------------------------------------------
# Network with two trainable physical parameters, c and k
# ----------------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, c0=C_INIT, k0=K_INIT):
        super().__init__()
        dims = [1] + [WIDTH] * HIDDEN_LAYERS + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)   # Glorot uniform
            nn.init.zeros_(layer.bias)
        self.c = nn.Parameter(torch.tensor(float(c0)))   # unknown damping coefficient
        self.k = nn.Parameter(torch.tensor(float(k0)))   # unknown stiffness

    def forward(self, t):
        h = t
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)                   # no output transform


# ----------------------------------------------------------------------------
# Loss terms (automatic differentiation)
# ----------------------------------------------------------------------------
def ode_residual(model, t):
    t = t.clone().requires_grad_(True)
    u = model(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    return M * u_tt + model.c * u_t + model.k * u


def initial_condition_loss(model):
    """u(0) = 1 and u'(0) = 1. All initial-condition points sit at t = 0, so one point is enough."""
    t0 = torch.zeros(1, 1, requires_grad=True)
    u = model(t0)
    u_t = torch.autograd.grad(u, t0, torch.ones_like(u), create_graph=True)[0]
    return (torch.cat([u - U0, u_t - V0]) ** 2).mean()


def total_loss(model, t_f, t_d, u_d):
    l_f = (ode_residual(model, t_f) ** 2).mean()
    l_b = initial_condition_loss(model)
    l_d = ((model(t_d) - u_d) ** 2).mean()
    return W_F * l_f + W_B * l_b + W_D * l_d, l_f, l_d


# ----------------------------------------------------------------------------
# Training (Adam only, as in the paper)
# ----------------------------------------------------------------------------
def train(model, t_f, t_d, u_d, t_test, epochs):
    hist = {"iter": [], "train": [], "test": [], "mse": [],
            "p_iter": [], "c": [], "k": []}
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    start = time.time()

    for ep in range(epochs + 1):
        opt.zero_grad()
        loss, l_f, l_d = total_loss(model, t_f, t_d, u_d)
        loss.backward()
        opt.step()

        if ep % PARAM_EVERY == 0:
            hist["p_iter"].append(ep)
            hist["c"].append(model.c.item())
            hist["k"].append(model.k.item())
        if ep % LOG_EVERY == 0:
            test_loss = (ode_residual(model, t_test) ** 2).mean().item()
            hist["iter"].append(ep)
            hist["train"].append(l_f.item())
            hist["test"].append(test_loss)
            hist["mse"].append(l_d.item())
            if ep % (LOG_EVERY * 10) == 0:
                print(f"epoch {ep:6d} | loss {loss.item():.3e} | c = {model.c.item():.4f} "
                      f"| k = {model.k.item():.4f} | {time.time() - start:.0f}s")

    print(f"Training finished in {time.time() - start:.0f}s")
    return hist


# ----------------------------------------------------------------------------
# Plots and results
# ----------------------------------------------------------------------------
def make_plots(model, hist, t_d, u_d, out_path):
    c_id, k_id = model.c.item(), model.k.item()
    print("\nIdentified parameters:")
    print(f"  k = {k_id:.6f}   (true {K_TRUE}, error {abs(k_id - K_TRUE) / K_TRUE * 100:.3f} %)")
    print(f"  c = {c_id:.6f}   (true {C_TRUE}, error {abs(c_id - C_TRUE) / C_TRUE * 100:.3f} %)")

    t_grid = np.linspace(0, T_MAX, 1000)
    with torch.no_grad():
        u_pred = model(torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)).numpy().ravel()
    u_true = analytical(t_grid)
    print(f"  max |u_pred - u_true| on [0, {T_MAX:g}]: {np.abs(u_pred - u_true).max():.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.plot(t_d.numpy(), u_d.numpy(), "bo", markersize=4, mfc="none", label="Observations")
    ax.plot(t_grid, u_true, "k-", label="True solution")
    ax.plot(t_grid, u_pred, "r--", label="Prediction")
    ax.set_xlabel("t"); ax.set_ylabel("u"); ax.set_title("Displacement")
    ax.legend()

    ax = axes[1]
    ax.plot(hist["p_iter"], hist["k"], "r-", label="Identified k")
    ax.axhline(K_TRUE, color="r", linestyle="-.", label="True k")
    ax.plot(hist["p_iter"], hist["c"], "b-", label="Identified c")
    ax.axhline(C_TRUE, color="b", linestyle="-.", label="True c")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Parameter values")
    ax.set_title("Parameter convergence"); ax.legend()

    ax = axes[2]
    ax.semilogy(hist["iter"], hist["train"], label="ODE train loss")
    ax.semilogy(hist["iter"], hist["test"], label="ODE test loss")
    ax.semilogy(hist["iter"], hist["mse"], label="Observations MSE")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Loss")
    ax.set_title("Convergence history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("PINN: parameter identification for a damped oscillator", fontsize=14)
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
    parser.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to observations")
    parser.add_argument("--c0", type=float, default=C_INIT, help="initial guess for c")
    parser.add_argument("--k0", type=float, default=K_INIT, help="initial guess for k")
    parser.add_argument("--load", action="store_true", help="load saved weights instead of training")
    args, _ = parser.parse_known_args()      # parse_known_args keeps this Colab-safe

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    weights_path = OUT_DIR / "exp3_model.pt"

    # Data: 50 equispaced observations, 100 random residual points, 10000 test points
    t_d_np = np.linspace(0, T_MAX, N_DATA)
    u_d_np = analytical(t_d_np) + args.noise * np.random.randn(N_DATA)
    t_d = torch.tensor(t_d_np, dtype=torch.float32).reshape(-1, 1)
    u_d = torch.tensor(u_d_np, dtype=torch.float32).reshape(-1, 1)
    t_f = T_MAX * torch.rand(N_RESIDUAL, 1)
    t_test = T_MAX * torch.rand(N_TEST, 1)

    model = PINN(args.c0, args.k0)

    if args.load and weights_path.exists():
        saved = torch.load(weights_path)
        model.load_state_dict(saved["state"])
        hist = saved["history"]
        print(f"Loaded weights from {weights_path}")
    else:
        hist = train(model, t_f, t_d, u_d, t_test, args.epochs)
        torch.save({"state": model.state_dict(), "history": hist}, weights_path)
        print(f"Weights saved to {weights_path}")

    make_plots(model, hist, t_d, u_d, OUT_DIR / "exp3_results.png")


if __name__ == "__main__":
    main()
