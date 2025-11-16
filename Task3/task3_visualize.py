"""
Subtask 3: Visualization script
Generate all figures required for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import json
from pathlib import Path
import torch
import sys

# Import model definition
try:
    from task3_train import PINN_Task3
except:
    print("Warning: Failed to import task3_train_complete, trying task3_train")
    try:
        from task3_train import PINN_Task3
    except:
        print("Error: Cannot import PINN_Task3 model definition")
        sys.exit(1)

# Set font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    """Load trained model"""
    print(f"Loading model: {checkpoint_path}")
    
    model = PINN_Task3().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.6e})")
    return model, checkpoint


def plot_solution_and_parameter(save_dir='results', output_dir='paper_figures'):
    """
    Plot solution field u(x,y) and parameter field k(x,y)
    """
    print("\nGenerating Figure 1: Solution and parameter fields...")
    
    # Load data
    data = np.load(Path(save_dir) / 'grid_predictions.npz')
    x = data['x']
    y = data['y']
    u_pred = data['u_pred']
    k_pred = data['k_pred']
    
    # Load observation points
    df_obs = pd.read_csv(Path(save_dir) / 'training_data_predictions.csv')
    
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Parameter field k(x,y)
    ax = axes[0]
    c1 = ax.contourf(X, Y, k_pred, levels=20, cmap='viridis')
    ax.scatter(df_obs['x'], df_obs['y'], c='red', s=10, alpha=0.5, label='Observation points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(a) Identified parameter field $k(x,y)$')
    ax.legend()
    ax.set_aspect('equal')
    plt.colorbar(c1, ax=ax, label='$k(x,y)$')
    
    # (b) Solution field u(x,y)
    ax = axes[1]
    c2 = ax.contourf(X, Y, u_pred, levels=20, cmap='RdBu_r')
    ax.scatter(df_obs['x'], df_obs['y'], c='black', s=10, alpha=0.3, label='Observation points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(b) Reconstructed solution field $u(x,y)$')
    ax.legend()
    ax.set_aspect('equal')
    plt.colorbar(c2, ax=ax, label='$u(x,y)$')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'task3_solution_and_parameter.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'task3_solution_and_parameter.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path / 'task3_solution_and_parameter.pdf'}")
    plt.close()


def plot_3d_surfaces(save_dir='results', output_dir='paper_figures'):
    """
    Plot 3D surface plots
    """
    print("\nGenerating Figure 2: 3D surfaces...")
    
    data = np.load(Path(save_dir) / 'grid_predictions.npz')
    x = data['x']
    y = data['y']
    u_pred = data['u_pred']
    k_pred = data['k_pred']
    
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(14, 6))
    
    # (a) 3D k(x,y)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, k_pred, cmap='viridis', 
                             linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('$k(x,y)$')
    ax1.set_title('(a) Parameter field $k(x,y)$ (3D)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # (b) 3D u(x,y)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_pred, cmap='RdBu_r',
                             linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('$u(x,y)$')
    ax2.set_title('(b) Solution field $u(x,y)$ (3D)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'task3_3d_surfaces.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'task3_3d_surfaces.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path / 'task3_3d_surfaces.pdf'}")
    plt.close()


def plot_training_history(save_dir='results', output_dir='paper_figures'):
    """
    Plot training history
    """
    print("\nGenerating Figure 3: Training history...")
    
    df = pd.read_csv(Path(save_dir) / 'training_history.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Total loss
    ax = axes[0, 0]
    ax.semilogy(df['epoch'], df['total_loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('(a) Total Loss')
    ax.grid(True, alpha=0.3)
    
    # (b) Component losses
    ax = axes[0, 1]
    ax.semilogy(df['epoch'], df['data_loss'], label='Data', linewidth=1.5)
    ax.semilogy(df['epoch'], df['pde_loss'], label='PDE', linewidth=1.5)
    ax.semilogy(df['epoch'], df['bc_loss'], label='BC', linewidth=1.5)
    ax.semilogy(df['epoch'], df['reg_loss'], label='Reg', linewidth=1.5)
    ax.semilogy(df['epoch'], df['smooth_loss'], label='Smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Component Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Learning rate
    ax = axes[1, 0]
    ax.semilogy(df['epoch'], df['lr'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(c) Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # (d) Data fitting vs PDE loss
    ax = axes[1, 1]
    ax.semilogy(df['epoch'], df['data_loss'], label='Data Loss', linewidth=2)
    ax.semilogy(df['epoch'], df['pde_loss'], label='PDE Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(d) Data Fitting vs PDE Constraints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'task3_training_history.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'task3_training_history.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path / 'task3_training_history.pdf'}")
    plt.close()


def plot_k_distribution(save_dir='results', output_dir='paper_figures'):
    """
    Plot k(x,y) statistical distribution
    """
    print("\nGenerating Figure 4: k(x,y) distribution...")
    
    data = np.load(Path(save_dir) / 'grid_predictions.npz')
    k_pred = data['k_pred'].flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Histogram
    ax = axes[0]
    ax.hist(k_pred, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(k_pred.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {k_pred.mean():.3f}')
    ax.axvline(k_pred.mean() - k_pred.std(), color='orange', linestyle=':', linewidth=2, label=f'±1σ')
    ax.axvline(k_pred.mean() + k_pred.std(), color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('$k(x,y)$ value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'(a) $k(x,y)$ Histogram\n(Mean={k_pred.mean():.3f}, Std={k_pred.std():.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Cumulative distribution
    ax = axes[1]
    sorted_k = np.sort(k_pred)
    cumulative = np.arange(1, len(sorted_k) + 1) / len(sorted_k)
    ax.plot(sorted_k, cumulative, linewidth=2)
    ax.axvline(k_pred.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(np.median(k_pred), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(k_pred):.3f}')
    ax.set_xlabel('$k(x,y)$ value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('(b) Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'task3_k_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'task3_k_distribution.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path / 'task3_k_distribution.pdf'}")
    plt.close()


def plot_data_fitting(save_dir='results', output_dir='paper_figures'):
    """
    Plot data fitting quality
    """
    print("\nGenerating Figure 5: Data fitting quality...")
    
    df = pd.read_csv(Path(save_dir) / 'training_data_predictions.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Predicted vs True
    ax = axes[0]
    ax.scatter(df['u_true'], df['u_pred'], alpha=0.6, s=20)
    min_val = min(df['u_true'].min(), df['u_pred'].min())
    max_val = max(df['u_true'].max(), df['u_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')
    ax.set_xlabel('True Value $u_{\\rm obs}$')
    ax.set_ylabel('Predicted Value $u_{\\rm pred}$')
    ax.set_title('(a) Predicted vs Observed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # (b) Error distribution
    ax = axes[1]
    ax.scatter(df['x'], df['y'], c=df['u_error'], cmap='hot', s=30, alpha=0.8)
    plt.colorbar(ax.collections[0], ax=ax, label='Absolute Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(b) Spatial Error Distribution')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'task3_data_fitting.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'task3_data_fitting.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path / 'task3_data_fitting.pdf'}")
    plt.close()


def print_summary_metrics(save_dir='results'):
    """
    Print summary metrics (for paper)
    """
    print("\n" + "="*70)
    print("TASK 3 Summary Metrics (for paper)")
    print("="*70)
    
    # Load evaluation results
    with open(Path(save_dir) / 'evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    # Load training history
    df_history = pd.read_csv(Path(save_dir) / 'training_history.csv')
    
    # Load configuration
    with open(Path(save_dir) / 'config.json', 'r') as f:
        config = json.load(f)
    
    print(f"\nNetwork Architecture:")
    print(f"  u_net: {config['u_layers']}")
    print(f"  k_net: {config['k_layers']}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']:,}")
    print(f"  Collocation points: {config['n_collocation']:,} (interior) + {config['n_boundary']:,} (boundary)")
    print(f"  Observation points: 200")
    
    print(f"\nLoss Weights:")
    for key, value in config['loss_weights'].items():
        print(f"  λ_{key}: {value}")
    
    print(f"\nEvaluation Results:")
    print(f"  Data fitting MSE: {eval_results['data_mse']:.6e}")
    print(f"  Data relative error: {eval_results['data_rel_error']:.4%}")
    print(f"  PDE residual MSE: {eval_results['pde_mse']:.6e}")
    print(f"  PDE relative error: {eval_results['pde_rel_error']:.4%}")
    
    print(f"\nIdentified k(x,y) statistics:")
    print(f"  Range: [{eval_results['k_min']:.4f}, {eval_results['k_max']:.4f}]")
    print(f"  Mean: {eval_results['k_mean']:.4f}")
    print(f"  Std: {eval_results['k_std']:.4f}")
    
    print(f"\nTraining statistics:")
    print(f"  Final total loss: {df_history['total_loss'].iloc[-1]:.6e}")
    print(f"  Minimum total loss: {df_history['total_loss'].min():.6e}")
    print(f"  Final learning rate: {df_history['lr'].iloc[-1]:.6e}")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    save_dir = 'results'
    output_dir = 'paper_figures'
    
    print("="*70)
    print("Task 3: Generate Paper Figures")
    print("="*70)
    
    # Check necessary files
    required_files = [
        'grid_predictions.npz',
        'training_history.csv',
        'training_data_predictions.csv',
        'evaluation_results.json',
        'config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (Path(save_dir) / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\nPlease run task3_train.py to train the model first")
        return
    
    # Generate all figures
    plot_solution_and_parameter(save_dir, output_dir)
    plot_3d_surfaces(save_dir, output_dir)
    plot_training_history(save_dir, output_dir)
    plot_k_distribution(save_dir, output_dir)
    plot_data_fitting(save_dir, output_dir)
    
    # Print summary
    print_summary_metrics(save_dir)
    
    print("\n" + "="*70)
    print("✅ All figures generated successfully!")
    print("="*70)
    print(f"\nFigures saved in: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(Path(output_dir).glob('task3_*')):
        print(f"  - {file.name}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
