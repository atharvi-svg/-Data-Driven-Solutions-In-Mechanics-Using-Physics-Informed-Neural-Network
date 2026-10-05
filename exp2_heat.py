import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_finite_difference(nx=51, ny=51):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    n_interior_x = nx - 2
    n_interior_y = ny - 2
    n_unknowns = n_interior_x * n_interior_y

    A = lil_matrix((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)

    def index(i, j):
        return (j - 1) * n_interior_x + (i - 1)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = index(i, j)

            A[k, k] = -2 / dx**2 - 2 / dy**2

            if i > 1:
                A[k, index(i - 1, j)] = 1 / dx**2

            if i < nx - 2:
                A[k, index(i + 1, j)] = 1 / dx**2

            if j > 1:
                A[k, index(i, j - 1)] = 1 / dy**2

            if j < ny - 2:
                A[k, index(i, j + 1)] = 1 / dy**2

            b[k] = -1

    T_interior = spsolve(A.tocsr(), b)

    T = np.zeros((ny, nx))

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            T[j, i] = T_interior[index(i, j)]

    return x, y, T


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.network(inputs)


def generate_training_points(n_interior=1500, n_boundary=500):
    x_interior = 2 * torch.rand(n_interior, 1) - 1
    y_interior = 2 * torch.rand(n_interior, 1) - 1

    x_interior.requires_grad_(True)
    y_interior.requires_grad_(True)

    n_side = n_boundary // 4

    y_left = 2 * torch.rand(n_side, 1) - 1
    x_left = -torch.ones_like(y_left)

    y_right = 2 * torch.rand(n_side, 1) - 1
    x_right = torch.ones_like(y_right)

    x_bottom = 2 * torch.rand(n_side, 1) - 1
    y_bottom = -torch.ones_like(x_bottom)

    x_top = 2 * torch.rand(n_side, 1) - 1
    y_top = torch.ones_like(x_top)

    x_boundary = torch.cat(
        [x_left, x_right, x_bottom, x_top],
        dim=0
    )

    y_boundary = torch.cat(
        [y_left, y_right, y_bottom, y_top],
        dim=0
    )

    return (
        x_interior,
        y_interior,
        x_boundary,
        y_boundary
    )


def physics_loss(model, x, y):
    T = model(x, y)

    T_x = torch.autograd.grad(
        T,
        x,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_xx = torch.autograd.grad(
        T_x,
        x,
        grad_outputs=torch.ones_like(T_x),
        create_graph=True
    )[0]

    T_y = torch.autograd.grad(
        T,
        y,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_yy = torch.autograd.grad(
        T_y,
        y,
        grad_outputs=torch.ones_like(T_y),
        create_graph=True
    )[0]

    residual = T_xx + T_yy + 1

    return torch.mean(residual ** 2)


def boundary_loss(model, x_boundary, y_boundary):
    T_boundary = model(
        x_boundary,
        y_boundary
    )

    return torch.mean(T_boundary ** 2)


def train_pinn(epochs=10000):
    model = PINN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    (
        x_interior,
        y_interior,
        x_boundary,
        y_boundary
    ) = generate_training_points()

    loss_history = []

    print()
    print("Starting PINN training...")
    print("----------------------------------------")

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_pde = physics_loss(
            model,
            x_interior,
            y_interior
        )

        loss_bc = boundary_loss(
            model,
            x_boundary,
            y_boundary
        )

        loss = 0.5 * loss_pde + 0.5 * loss_bc

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch + 1:5d} | "
                f"Total Loss: {loss.item():.6e} | "
                f"PDE Loss: {loss_pde.item():.6e} | "
                f"BC Loss: {loss_bc.item():.6e}"
            )

    print("----------------------------------------")
    print("PINN training completed.")

    return model, loss_history


def predict_pinn(model, x, y):
    X, Y = np.meshgrid(x, y)

    x_tensor = torch.tensor(
        X.reshape(-1, 1),
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        Y.reshape(-1, 1),
        dtype=torch.float32
    )

    with torch.no_grad():
        T_prediction = model(
            x_tensor,
            y_tensor
        )

    T_prediction = T_prediction.numpy()

    T_prediction = T_prediction.reshape(
        len(y),
        len(x)
    )

    return T_prediction


def plot_reference_solution(x, y, T):
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        T,
        levels=50
    )

    plt.colorbar(
        contour,
        label="Temperature T(x,y)"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Finite Difference Reference Solution")

    plt.tight_layout()

    plt.savefig(
        "figures/exp2_reference.png",
        dpi=300
    )

    plt.show()


def plot_pinn_solution(x, y, T):
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        T,
        levels=50
    )

    plt.colorbar(
        contour,
        label="Temperature T(x,y)"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN Temperature Solution")

    plt.tight_layout()

    plt.savefig(
        "figures/exp2_pinn.png",
        dpi=300
    )

    plt.show()


def plot_error(x, y, error):
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        error,
        levels=50
    )

    plt.colorbar(
        contour,
        label="Absolute Error"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute Error: |FDM - PINN|")

    plt.tight_layout()

    plt.savefig(
        "figures/exp2_error.png",
        dpi=300
    )

    plt.show()


def plot_loss(loss_history):
    plt.figure(figsize=(7, 5))

    plt.semilogy(
        loss_history
    )

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("PINN Training Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "figures/exp2_loss.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    print()
    print("========================================")
    print("Experiment 2: 2D Steady-State Heat")
    print("Conduction using a Physics-Informed")
    print("Neural Network")
    print("========================================")

    print()
    print("Creating finite-difference reference...")

    x, y, T_reference = solve_finite_difference()

    print()
    print("Finite Difference Solution")
    print("--------------------------")
    print(f"Grid size: {len(x)} x {len(y)}")
    print(
        f"Maximum temperature: "
        f"{T_reference.max():.6f}"
    )
    print(
        f"Minimum temperature: "
        f"{T_reference.min():.6f}"
    )

    plot_reference_solution(
        x,
        y,
        T_reference
    )

    model, loss_history = train_pinn(
        epochs=10000
    )

    print()
    print("Generating PINN prediction...")

    T_pinn = predict_pinn(
        model,
        x,
        y
    )

    print("PINN prediction generated.")

    plot_pinn_solution(
        x,
        y,
        T_pinn
    )

    absolute_error = np.abs(
        T_reference - T_pinn
    )

    mae = np.mean(
        absolute_error
    )

    rmse = np.sqrt(
        np.mean(
            (T_reference - T_pinn) ** 2
        )
    )

    max_error = np.max(
        absolute_error
    )

    print()
    print("========================================")
    print("PINN vs FDM Results")
    print("========================================")
    print(f"Mean Absolute Error : {mae:.6e}")
    print(f"RMSE                : {rmse:.6e}")
    print(f"Maximum Absolute Error: {max_error:.6e}")

    plot_error(
        x,
        y,
        absolute_error
    )

    plot_loss(
        loss_history
    )

    print()
    print("========================================")
    print("Experiment 2 Completed")
    print("========================================")
    print()
    print("Generated files:")
    print("1. figures/exp2_reference.png")
    print("2. figures/exp2_pinn.png")
    print("3. figures/exp2_error.png")
    print("4. figures/exp2_loss.png")
    print()