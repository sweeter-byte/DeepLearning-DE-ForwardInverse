"""
Task 2: Helmholtz Equation with High Wave Numbers (k=100, 500, 1000)
Training script - focuses on training and saving results
Visualization is done separately in task2_visualize.py
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
from tqdm import tqdm
import os
import pandas as pd
import json

# Create directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# Parameters
a1 = 1.0
a2 = 3.0
pi = np.pi
lambda_eig = (a1**2 + a2**2) * pi**2  

def u_true(x, y, k):
    """Analytical solution for the Helmholtz equation."""
    c = - (lambda_eig + k**2) / (lambda_eig - k**2)
    return c * np.sin(a1 * pi * x) * np.sin(a2 * pi * y)

def siren_init_(tensor, omega=30.0, is_first=False):
    """Custom SIREN initialization."""
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega
    with torch.no_grad():
        tensor.uniform_(-bound, bound)

class Net(nn.Module):
    """PINN model with SIREN initialization."""
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
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)

    def forward(self, xy):
        h = xy * self.omega_0
        for i in range(len(self.layers) - 1):
            h = torch.sin(self.layers[i](h))
        return self.layers[-1](h)

def q(x, y, k, a1=a1, a2=a2):
    """Source term q(x, y)."""
    pi_t = torch.pi
    term1 = - (a1 * pi_t)**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    term2 = - (a2 * pi_t)**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    term3 = - k**2 * torch.sin(a1 * pi_t * x) * torch.sin(a2 * pi_t * y)
    return term1 + term2 + term3

def sample_interior(N_interior, device):
    """Sample interior points."""
    x_i = torch.rand(N_interior, device=device) * 2 - 1
    y_i = torch.rand(N_interior, device=device) * 2 - 1
    xy_i = torch.stack([x_i, y_i], dim=1)
    xy_i.requires_grad_(True)
    return xy_i

def sample_boundary(N_per_side, device):
    """Sample boundary points."""
    x_left = -torch.ones(N_per_side, device=device)
    y_left = torch.rand(N_per_side, device=device) * 2 - 1
    x_right = torch.ones(N_per_side, device=device)
    y_right = torch.rand(N_per_side, device=device) * 2 - 1
    x_bottom = torch.rand(N_per_side, device=device) * 2 - 1
    y_bottom = -torch.ones(N_per_side, device=device)
    x_top = torch.rand(N_per_side, device=device) * 2 - 1
    y_top = torch.ones(N_per_side, device=device)
    
    xy_b = torch.cat([
        torch.stack([x_left, y_left], dim=1),
        torch.stack([x_right, y_right], dim=1),
        torch.stack([x_bottom, y_bottom], dim=1),
        torch.stack([x_top, y_top], dim=1)
    ])
    xy_b.requires_grad_(True)
    return xy_b

def compute_loss(net, xy_i, xy_b, k, lambda_pde=1.0, lambda_bc=100.0):
    """Compute total loss."""
    # Interior: PDE loss
    u_i = net(xy_i)
    du = torch.autograd.grad(u_i, xy_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0]
    du_x = du[:, 0]
    du_y = du[:, 1]
    du_xx = torch.autograd.grad(du_x, xy_i, grad_outputs=torch.ones_like(du_x), create_graph=True)[0][:, 0]
    du_yy = torch.autograd.grad(du_y, xy_i, grad_outputs=torch.ones_like(du_y), create_graph=True)[0][:, 1]
    delta_u = du_xx + du_yy
    pde_res = -delta_u - (k**2) * u_i.squeeze() - q(xy_i[:, 0], xy_i[:, 1], k)
    L_pde = torch.mean(pde_res**2)
    
    # Boundary: BC loss
    u_b = net(xy_b)
    L_bc = torch.mean(u_b**2)
    
    loss = lambda_pde * L_pde + lambda_bc * L_bc
    return loss, L_pde, L_bc

def train(net, device, k, epochs=30000, N_interior=8000, N_per_side=500, lr=0.001):
    """Training loop for a specific k value."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10000, T_mult=2, eta_min=1e-6
    )
    
    loss_history = {
        'epoch': [], 
        'total_loss': [], 
        'pde_loss': [], 
        'bc_loss': [],
        'learning_rate': []
    }
    best_loss = float('inf')
    
    pbar = tqdm(range(epochs), desc=f'Training k={k}')
    
    for epoch in pbar:
        xy_i = sample_interior(N_interior, device)
        xy_b = sample_boundary(N_per_side, device)
        
        loss, L_pde, L_bc = compute_loss(net, xy_i, xy_b, k)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        
        # Record history
        if epoch % 100 == 0:
            loss_history['epoch'].append(epoch)
            loss_history['total_loss'].append(current_loss)
            loss_history['pde_loss'].append(L_pde.item())
            loss_history['bc_loss'].append(L_bc.item())
            loss_history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(net.state_dict(), f'results/models/best_k_{int(k)}.pth')
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.2e}',
            'Best': f'{best_loss:.2e}'
        })
        
        if (epoch + 1) % 5000 == 0:
            print(f'\nEpoch {epoch+1}: Loss={current_loss:.6e}, Best={best_loss:.6e}')
    
    # Save loss history
    pd.DataFrame(loss_history).to_csv(f'results/data/loss_history_k_{int(k)}.csv', index=False)
    
    print(f"\nTraining Complete for k={k}! Best Loss: {best_loss:.6e}")
    return loss_history, best_loss

def evaluate(net, device, k, grid_size=100):
    """Evaluate the trained model."""
    # Create test grid
    x_test = np.linspace(-1, 1, grid_size)
    y_test = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_test, y_test)
    xy_test_np = np.c_[X.ravel(), Y.ravel()]
    xy_test = torch.tensor(xy_test_np, dtype=torch.float32, device=device)
    
    # Get predictions
    with torch.no_grad():
        u_pred_flat = net(xy_test).cpu().numpy()
    u_pred = u_pred_flat.reshape((grid_size, grid_size))
    
    # Compute true solution
    u_true_val = u_true(X, Y, k)
    
    # Compute errors
    error = np.abs(u_pred - u_true_val)
    rel_l2 = np.sqrt(np.mean((u_pred - u_true_val)**2) / np.mean(u_true_val**2))
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"\nEvaluation Results for k={k}:")
    print(f"  Relative L2 Error: {rel_l2:.6e} ({rel_l2*100:.4f}%)")
    print(f"  Maximum Error: {max_error:.6e}")
    print(f"  Mean Error: {mean_error:.6e}")
    
    # Save evaluation data
    np.savez(
        f'results/data/evaluation_k_{int(k)}.npz',
        X=X, Y=Y,
        u_pred=u_pred,
        u_true=u_true_val,
        error=error,
        k=k
    )
    
    # Save metrics
    metrics = {
        'k': int(k),
        'rel_l2_error': float(rel_l2),
        'rel_l2_error_percent': float(rel_l2 * 100),
        'max_error': float(max_error),
        'mean_error': float(mean_error),
        'grid_size': grid_size
    }
    
    with open(f'results/data/metrics_k_{int(k)}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    start_time = time.time()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Task 2: Helmholtz Equation with High Wave Numbers")
    print("="*80)
    
    k_values = [100, 500, 1000]
    all_results = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"Training for k = {k}")
        print(f"{'='*80}")
        
        # Initialize network
        net = Net(layers=[2, 100, 100, 100, 100, 1], omega_0=30.0)
        net.to(device)
        
        # Train
        task_start = time.time()
        loss_history, best_loss = train(
            net, device, k,
            epochs=30000,
            N_interior=8000,
            N_per_side=500,
            lr=0.001
        )
        task_time = time.time() - task_start
        
        # Load best model
        net.load_state_dict(torch.load(f'results/models/best_k_{int(k)}.pth'))
        
        # Evaluate
        metrics = evaluate(net, device, k, grid_size=100)
        
        # Store results
        all_results[f'k_{int(k)}'] = {
            'best_loss': float(best_loss),
            'training_time_minutes': float(task_time / 60),
            'metrics': metrics
        }
    
    # Total time
    total_time = time.time() - start_time
    
    # Save overall summary
    summary = {
        'task': 'Task 2: High Wave Number Helmholtz',
        'k_values': k_values,
        'total_training_time_minutes': float(total_time / 60),
        'device': str(device),
        'results': all_results
    }
    
    with open('results/data/task2_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*80)
    print("Task 2 Training Complete!")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print("="*80)
    print("\nSummary:")
    for k in k_values:
        result = all_results[f'k_{int(k)}']
        print(f"  k={k}: L2 Error = {result['metrics']['rel_l2_error_percent']:.4f}%, "
              f"Time = {result['training_time_minutes']:.2f} min")
    
    print("\nResults saved to results/data/")
    print("Run task2_visualize.py to generate paper figures.")

if __name__ == "__main__":
    main()
