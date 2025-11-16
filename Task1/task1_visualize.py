"""
Task 1 Visualization Script
Generates publication-quality figures for the conference paper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Set publication-quality plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create output directory
os.makedirs('paper_figures', exist_ok=True)

def load_data():
    """Load all saved data from training."""
    print("Loading data...")
    
    # Load evaluation data
    eval_data = np.load('results/data/evaluation.npz')
    X = eval_data['X']
    Y = eval_data['Y']
    u_pred = eval_data['u_pred']
    u_true = eval_data['u_true']
    error = eval_data['error']
    
    # Load loss history
    loss_df = pd.read_csv('results/data/loss_history.csv')
    
    # Load metrics
    with open('results/data/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    return X, Y, u_pred, u_true, error, loss_df, metrics

def plot_true_solution(X, Y, u_true):
    """Figure 1: True analytical solution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    levels = 50
    contour = ax.contourf(X, Y, u_true, levels=levels, cmap='RdBu_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$u(x,y)$', rotation=0, labelpad=15)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('True Solution of Helmholtz Equation ($k=4$)')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task1_true_solution.pdf')
    plt.savefig('paper_figures/task1_true_solution.png')
    plt.close()
    print("✓ Generated: task1_true_solution.pdf")

def plot_predicted_solution(X, Y, u_pred):
    """Figure 2: PINN predicted solution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    levels = 50
    contour = ax.contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$u_{\\rm pred}(x,y)$', rotation=0, labelpad=20)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('PINN Predicted Solution ($k=4$)')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task1_predicted_solution.pdf')
    plt.savefig('paper_figures/task1_predicted_solution.png')
    plt.close()
    print("✓ Generated: task1_predicted_solution.pdf")

def plot_error_distribution(X, Y, error, metrics):
    """Figure 3: Absolute error distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    levels = 50
    contour = ax.contourf(X, Y, error, levels=levels, cmap='hot_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$|u_{\\rm pred} - u_{\\rm true}|$', rotation=0, labelpad=35)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Absolute Error Distribution (Max: {metrics["max_error"]:.2e})')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task1_error_distribution.pdf')
    plt.savefig('paper_figures/task1_error_distribution.png')
    plt.close()
    print("✓ Generated: task1_error_distribution.pdf")

def plot_loss_convergence(loss_df):
    """Figure 4: Training loss convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(loss_df['epoch'], loss_df['total_loss'], 
           linewidth=2, color='#1f77b4', label='Total Loss')
    ax.plot(loss_df['epoch'], loss_df['pde_loss'], 
           linewidth=1.5, color='#ff7f0e', alpha=0.7, label='PDE Loss')
    ax.plot(loss_df['epoch'], loss_df['bc_loss'], 
           linewidth=1.5, color='#2ca02c', alpha=0.7, label='BC Loss')
    
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Convergence (Task 1, $k=4$)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig('paper_figures/task1_loss_convergence.pdf')
    plt.savefig('paper_figures/task1_loss_convergence.png')
    plt.close()
    print("✓ Generated: task1_loss_convergence.pdf")

def plot_cross_sections(X, Y, u_pred, u_true):
    """Figure 5: Cross-sectional comparison (optional, for detailed analysis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Cross section at y=0
    mid_idx = Y.shape[0] // 2
    x_line = X[mid_idx, :]
    u_true_line = u_true[mid_idx, :]
    u_pred_line = u_pred[mid_idx, :]
    
    ax1.plot(x_line, u_true_line, 'k-', linewidth=2, label='True')
    ax1.plot(x_line, u_pred_line, 'r--', linewidth=2, label='Predicted')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$u(x, y=0)$')
    ax1.set_title('Cross Section at $y=0$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cross section at x=0
    mid_idx = X.shape[1] // 2
    y_line = Y[:, mid_idx]
    u_true_line = u_true[:, mid_idx]
    u_pred_line = u_pred[:, mid_idx]
    
    ax2.plot(y_line, u_true_line, 'k-', linewidth=2, label='True')
    ax2.plot(y_line, u_pred_line, 'r--', linewidth=2, label='Predicted')
    ax2.set_xlabel('$y$')
    ax2.set_ylabel('$u(x=0, y)$')
    ax2.set_title('Cross Section at $x=0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_figures/task1_cross_sections.pdf')
    plt.savefig('paper_figures/task1_cross_sections.png')
    plt.close()
    print("✓ Generated: task1_cross_sections.pdf")

def plot_comparison_grid(X, Y, u_true, u_pred, error, metrics):
    """Figure 6: Comprehensive comparison (3-panel figure for paper)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # True solution
    levels = 50
    im1 = axes[0].contourf(X, Y, u_true, levels=levels, cmap='RdBu_r')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_title('(a) True Solution')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted solution
    im2 = axes[1].contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    axes[1].set_title('(b) PINN Prediction')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Error
    im3 = axes[2].contourf(X, Y, error, levels=levels, cmap='hot_r')
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    axes[2].set_title(f'(c) Error (L2: {metrics["rel_l2_error_percent"]:.3f}%)')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('paper_figures/task1_comparison_grid.pdf')
    plt.savefig('paper_figures/task1_comparison_grid.png')
    plt.close()
    print("✓ Generated: task1_comparison_grid.pdf")

def generate_summary_table(metrics):
    """Generate a summary table of metrics."""
    print("\n" + "="*60)
    print("TASK 1 SUMMARY METRICS (for LaTeX table)")
    print("="*60)
    print(f"Relative L2 Error: {metrics['rel_l2_error_percent']:.4f}%")
    print(f"Maximum Error: {metrics['max_error']:.6e}")
    print(f"Mean Error: {metrics['mean_error']:.6e}")
    print("="*60)
    
    # LaTeX table format
    print("\nLaTeX Table Entry:")
    print(f"k=4 & {metrics['rel_l2_error_percent']:.4f}\\% & "
          f"{metrics['max_error']:.2e} & [Training time] \\\\")

def main():
    print("="*60)
    print("Task 1 Visualization - Generating Paper Figures")
    print("="*60)
    
    # Load data
    X, Y, u_pred, u_true, error, loss_df, metrics = load_data()
    
    print("\nGenerating figures...")
    
    # Generate all figures
    plot_true_solution(X, Y, u_true)
    plot_predicted_solution(X, Y, u_pred)
    plot_error_distribution(X, Y, error, metrics)
    plot_loss_convergence(loss_df)
    plot_cross_sections(X, Y, u_pred, u_true)
    plot_comparison_grid(X, Y, u_true, u_pred, error, metrics)
    
    # Print summary
    generate_summary_table(metrics)
    
    print("\n" + "="*60)
    print("All figures saved to paper_figures/")
    print("="*60)
    
    print("\nGenerated figures:")
    print("  1. task1_true_solution.pdf - True analytical solution")
    print("  2. task1_predicted_solution.pdf - PINN prediction")
    print("  3. task1_error_distribution.pdf - Error distribution")
    print("  4. task1_loss_convergence.pdf - Training convergence")
    print("  5. task1_cross_sections.pdf - 1D cross sections")
    print("  6. task1_comparison_grid.pdf - 3-panel comparison")
    print("\nRecommended for paper: Use figures 1-4 or figure 6")

if __name__ == "__main__":
    main()
