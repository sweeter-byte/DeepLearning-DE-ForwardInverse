import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import os
import pandas as pd

os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Parameters
a1 = 1.0
a2 = 3.0
pi = np.pi
lambda_eig = (a1**2 + a2**2) * pi**2  

# Analytical solution u_true (only for evaluation, not used in training)
def u_true(x, y, k):
    """
    Analytical solution for the Helmholtz equation.
    Used only for computing error after training.
    """
    c = - (lambda_eig + k**2) / (lambda_eig - k**2)
    return c * np.sin(a1 * pi * x) * np.sin(a2 * pi * y)

# SIREN initialization function
def siren_init_(tensor, omega=30.0, is_first=False):
    """
    Custom SIREN initialization for sin activation.
    """
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    if is_first:
        bound = 1.0 / fan_in  # For first layer
    else:
        bound = math.sqrt(6.0 / fan_in) / omega
    with torch.no_grad():
        tensor.uniform_(-bound, bound)

# Define the neural network (MLP) with SIREN style
class Net(nn.Module):
    """
    PINN model with SIREN initialization.
    Multi-layer perceptron with sin activation functions.
    Input: (x, y), Output: u(x, y)
    """
    def __init__(self, layers=[2, 100, 100, 100, 100, 1], omega_0=30.0):
        super(Net, self).__init__()
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])
            if i == 0:
                siren_init_(linear.weight, omega=1.0, is_first=True)
            else:
                siren_init_(linear.weight, omega=omega_0)
            nn.init.zeros_(linear.bias)  # Biases to zero
            self.layers.append(linear)

    def forward(self, xy):
        """
        Forward pass with SIREN scaling.
        xy: tensor of shape (N, 2)
        """
        h = xy * self.omega_0  # Scale input for first layer
        for i in range(len(self.layers) - 1):
            h = torch.sin(self.layers[i](h))
        return self.layers[-1](h)

# Define the source term q(x, y)
def q(x, y, a1=a1, a2=a2, k=4.0):
    """
    Source term q(x, y) as given in the problem.
    """
    pi_t = torch.pi
    term1 = - (a1 * pi_t)**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    term2 = - (a2 * pi_t)**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    term3 = - k**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    return term1 + term2 + term3

# Sampling functions
def sample_interior(N_interior, device):
    """
    Sample interior points uniformly in [-1, 1] x [-1, 1].
    Returns xy_i: (N_interior, 2) tensor with requires_grad=True for derivatives.
    """
    x_i = torch.rand(N_interior, device=device) * 2 - 1
    y_i = torch.rand(N_interior, device=device) * 2 - 1
    xy_i = torch.stack([x_i, y_i], dim=1)
    xy_i.requires_grad_(True)
    return xy_i

def sample_boundary(N_per_side, device):
    """
    Sample boundary points on the four sides of the square [-1, 1] x [-1, 1].
    Returns xy_b: (4 * N_per_side, 2) tensor.
    """
    # Left: x = -1, y in [-1,1]
    x_left = -torch.ones(N_per_side, device=device)
    y_left = torch.rand(N_per_side, device=device) * 2 - 1
    # Right: x = 1, y in [-1,1]
    x_right = torch.ones(N_per_side, device=device)
    y_right = torch.rand(N_per_side, device=device) * 2 - 1
    # Bottom: y = -1, x in [-1,1]
    x_bottom = torch.rand(N_per_side, device=device) * 2 - 1
    y_bottom = -torch.ones(N_per_side, device=device)
    # Top: y = 1, x in [-1,1]
    x_top = torch.rand(N_per_side, device=device) * 2 - 1
    y_top = torch.ones(N_per_side, device=device)
    
    xy_b = torch.cat([
        torch.stack([x_left, y_left], dim=1),
        torch.stack([x_right, y_right], dim=1),
        torch.stack([x_bottom, y_bottom], dim=1),
        torch.stack([x_top, y_top], dim=1)
    ])
    xy_b.requires_grad_(True)  # Though not needed for BC loss, but for consistency
    return xy_b

# Loss function computation
def compute_loss(net, xy_i, xy_b, k, lambda_pde=1.0, lambda_bc=100.0):
    """
    Compute the total loss: lambda_pde * L_pde + lambda_bc * L_bc
    L_pde: MSE of PDE residual in interior points.
    L_bc: MSE of u on boundary (should be 0).
    """
    # Interior: PDE loss
    u_i = net(xy_i)
    # First derivatives
    du = torch.autograd.grad(u_i, xy_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0]
    du_x = du[:, 0]
    du_y = du[:, 1]
    # Second derivatives
    du_xx = torch.autograd.grad(du_x, xy_i, grad_outputs=torch.ones_like(du_x), create_graph=True)[0][:, 0]
    du_yy = torch.autograd.grad(du_y, xy_i, grad_outputs=torch.ones_like(du_y), create_graph=True)[0][:, 1]
    delta_u = du_xx + du_yy
    pde_res = -delta_u - (k**2) * u_i.squeeze() - q(xy_i[:, 0], xy_i[:, 1], k=k)
    L_pde = torch.mean(pde_res**2)
    
    # Boundary: BC loss
    u_b = net(xy_b)
    L_bc = torch.mean(u_b**2)
    
    # Total loss
    loss = lambda_pde * L_pde + lambda_bc * L_bc
    return loss, L_pde, L_bc

# Training loop with Adam and cosine scheduler
def train(net, device, epochs=80000, N_interior=8000, N_per_side=500, lr=0.001, k=4.0):
    """
    Training loop for the PINN using Adam with cosine annealing scheduler.
    Resamples points every epoch for better generalization.
    Prints loss every 2000 epochs.
    Returns loss history for plotting convergence.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2, eta_min=1e-6)
    
    history = {'iteration': [], 'loss': [], 'pde_loss': [], 'bc_loss': []}
    
    best_loss = float('inf')
    
    pbar = tqdm(range(epochs), desc='Training')
    
    for epoch in pbar:
        xy_i = sample_interior(N_interior, device)
        xy_b = sample_boundary(N_per_side, device)
        
        loss, pde_loss, bc_loss = compute_loss(net, xy_i, xy_b, k)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        
        if epoch % 10 == 0:
            history['iteration'].append(epoch)
            history['loss'].append(current_loss)
            history['pde_loss'].append(pde_loss.item())
            history['bc_loss'].append(bc_loss.item())
        
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(net.state_dict(), f'results/models/best_k_{int(k)}.pth')
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.2e}',
            'Best': f'{best_loss:.2e}'
        })
        
        if (epoch + 1) % 2000 == 0:
            print(f'\n{"="*80}')
            print(f'Iteration {epoch+1}: Loss={current_loss:.6e}, Best={best_loss:.6e}')
            print(f'{"="*80}\n')
    
    pd.DataFrame(history).to_csv(f'results/data/history_k_{int(k)}.csv', index=False)
    
    print(f"\n训练完成！最佳Loss: {best_loss:.6e}")
    return history

# Evaluation and visualization
def evaluate_and_plot(net, device, history, k):
    """
    Evaluate the trained model:
    - Compute relative L2 error on a grid.
    - Plot true u, predicted u, absolute error.
    - Plot loss convergence curve.
    """
    # Test grid
    nx, ny = 100, 100
    x_test = np.linspace(-1, 1, nx)
    y_test = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x_test, y_test)
    xy_test_np = np.c_[X.ravel(), Y.ravel()]
    xy_test = torch.tensor(xy_test_np, dtype=torch.float32, device=device)
    
    # Predictions
    with torch.no_grad():
        u_pred_flat = net(xy_test).cpu().numpy()
    u_pred = u_pred_flat.reshape((nx, ny))
    
    # True values
    u_true_val = u_true(X, Y, k)
    
    # Relative L2 error
    rel_l2 = np.sqrt(np.mean((u_pred - u_true_val)**2) / np.mean(u_true_val**2))
    print(f"Relative L2 Error: {rel_l2:.4e}")
    
    # Save evaluation data
    np.savez(f'results/data/eval_k_{int(k)}.npz', X=X, Y=Y, u_pred=u_pred, u_true=u_true_val)
    
    # Plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # True solution
    cp0 = axs[0, 0].contourf(X, Y, u_true_val, 50, cmap='jet')
    fig.colorbar(cp0, ax=axs[0, 0])
    axs[0, 0].set_title('True Solution u(x,y)')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    
    # Predicted solution
    cp1 = axs[0, 1].contourf(X, Y, u_pred, 50, cmap='jet')
    fig.colorbar(cp1, ax=axs[0, 1])
    axs[0, 1].set_title('Predicted Solution u(x,y)')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    
    # Absolute error
    cp2 = axs[1, 0].contourf(X, Y, np.abs(u_pred - u_true_val), 50, cmap='jet')
    fig.colorbar(cp2, ax=axs[1, 0])
    axs[1, 0].set_title('Absolute Error |u_pred - u_true|')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    
    # Loss convergence
    axs[1, 1].plot(history['loss'])
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Loss Convergence')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f'results/figures/helmholtz_pinn_results_k_{int(k)}.png')
    plt.savefig(f'results/data/results_k_{int(k)}.png')

# Main execution
def main():
    start = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    
    for k in [100, 500, 1000]:
        print(f"\n{'='*80}")
        print(f"k = {k} Start")
        print(f"{'='*80}\n")
        
        net = Net(omega_0=30.0).to(device)
        history = train(net, device, epochs=30000, N_interior=8000, N_per_side=500, lr=0.001, k=k)
        
        net.load_state_dict(torch.load(f'results/models/best_k_{int(k)}.pth'))
        evaluate_and_plot(net, device, history, k)
    
    total_time = time.time() - start
    print(f"\n{'='*80}")
    print(f"Total Time: {total_time / 60:.2f} Mins")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()