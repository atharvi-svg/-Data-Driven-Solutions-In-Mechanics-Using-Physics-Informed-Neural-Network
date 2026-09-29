# Physics-Informed Neural Networks for Mechanics Problems

Machine Learning mini-project, Team 13.

This project implements Physics-Informed Neural Networks (PINNs) to solve differential equations from mechanics. A PINN embeds the governing equation into the loss function of a neural network, using automatic differentiation to compute the derivatives. We reproduce three experiments from the Stanford CS229 report *"Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network"* (Zhang, Chen, Yang).

## Problem Statement

Many mechanics phenomena are governed by differential equations. Instead of a mesh-based numerical method, we train a neural network `u(x; θ)` so that it satisfies the equation, the boundary and initial conditions, and any available data. The total loss is:

```
L = w_f * L_f  +  w_b * L_b  +  w_d * L_d
```

- `L_f`: mismatch with the governing equation at residual (collocation) points
- `L_b`: mismatch with boundary and initial conditions
- `L_d`: mismatch with observed data (used only in the inverse problem)

## Experiments

| # | Problem | Type | Compared against |
|---|---------|------|------------------|
| 1 | 1D consolidation (diffusion PDE, non-dimensionalised) | Forward | Analytical series solution |
| 2 | 2D steady-state heat conduction (Poisson equation) | Forward | Finite-difference reference solution |
| 3 | Damped spring-mass system, identify `c` and `k` | Inverse | True values `c = 0.4`, `k = 4` |

## Dataset Details

There is no external dataset. All data is generated in code:

- **Collocation points:** randomly sampled residual, boundary and test points inside the domain
- **Experiment 3 observations:** 50 equispaced points generated from the analytical solution of the damped oscillator with `c = 0.4`, `k = 4`, `m = u0 = v0 = 1`

## Repository Structure

```
.
├── src/
│   ├── model.py              # MLP network
│   ├── train.py              # Adam + L-BFGS training loop
│   └── utils.py              # sampling, plotting, seeds
├── experiments/
│   ├── exp1_consolidation.py # Experiment 1
│   ├── exp2_heat.py          # Experiment 2
│   └── exp3_inverse.py       # Experiment 3
├── results/                  # saved figures and outputs
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9 or newer.

```bash
git clone <your-repo-url>
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
torch
numpy
scipy
matplotlib
```

## How to Run

Run each experiment from the repository root. Figures are saved to `results/`.

```bash
python experiments/exp1_consolidation.py
python experiments/exp2_heat.py
python experiments/exp3_inverse.py
```

Full training is slow on a CPU. For a quick demo run, lower the number of Adam epochs at the top of each script (the `EPOCHS` variable).

## Hyperparameters

| | Exp 1: Consolidation | Exp 2: Heat conduction | Exp 3: Inverse problem |
|---|---|---|---|
| Hidden layers x neurons | 5 x 50 | 4 x 50 | 3 x 32 |
| Activation | tanh | tanh | tanh |
| Initializer | Glorot normal | Glorot uniform | Glorot uniform |
| Optimizer | Adam, then L-BFGS | Adam, then L-BFGS | Adam only |
| Adam epochs | 50,000 | 50,000 | 100,000 |
| Learning rate | 0.001 | 0.0005 | 0.0005 |
| Residual / boundary points | 500 / 250 + 125 | 1500 / 500 | 100 / 50 |
| Data points | none | none | 50 |
| Test points | 10,011 | 10,000 | 10,000 |
| Loss weights | w_f = w_b = 1/4 | w_f = w_b = 1/2 | w_f = w_b = w_d = 1/4 |

Notes:
- Experiment 1 multiplies the network output by `x̂`, so the boundary condition at `x̂ = 0` is satisfied exactly.
- Experiment 1 uses the dimensionless variables `x̂ = x/L`, `p̂ = p/p_u`, `t̂ = ct/L²`, with `t̂_max = 0.5`.

## Results

Add your own numbers and figures here after running the experiments.

| Experiment | Metric | Value |
|---|---|---|
| 1. Consolidation | Max absolute error vs analytical | _fill in_ |
| 2. Heat conduction | Max absolute error vs reference | _fill in_ |
| 3. Inverse problem | Identified `k` (true 4.0) | _fill in_ |
| 3. Inverse problem | Identified `c` (true 0.4) | _fill in_ |

Example figure:

```
![Exp 1 results](results/exp1_results.png)
```

## Team

| Name | GitHub | Contribution |
|---|---|---|
| Person 1 | @username | Repo, shared code, Experiments 1 and 3, slides, write-up |
| Person 2 | @username | Experiment 2, reference solver, write-up sections |

## Reference

Q. Zhang, Y. Chen, Z. Yang, *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network*, Stanford CS229 project report.

M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, Journal of Computational Physics, 378 (2019).
