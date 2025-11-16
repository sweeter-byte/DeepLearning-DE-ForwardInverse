"""
Task 2 Visualization Script
Generates publication-quality figures for high wave number results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Set publication-quality plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create output directory
os.makedirs('paper_figures', exist_ok=True)

def load_all_data():
    """Load data for all k values."""
    print("Loading data for all k values...")
    
    k_values = [100, 500, 1000]
    data_dict = {}
    
    for k in k_values:
        # Load evaluation data
        eval_data = np.load(f'results/data/evaluation_k_{k}.npz')
        
        # Load loss history
        loss_df = pd.read_csv(f'results/data/loss_history_k_{k}.csv')
        
        # Load metrics
        with open(f'results/data/metrics_k_{k}.json', 'r') as f:
            metrics = json.load(f)
        
        data_dict[k] = {
            'X': eval_data['X'],
            'Y': eval_data['Y'],
            'u_pred': eval_data['u_pred'],
            'u_true': eval_data['u_true'],
            'error': eval_data['error'],
            'loss_df': loss_df,
            'metrics': metrics
        }
    
    # Load summary
    with open('results/data/task2_summary.json', 'r') as f:
        summary = json.load(f)
    
    return data_dict, summary

def plot_solution_k100(data):
    """Figure 1: Solution field for k=100."""
    k = 100
    fig, ax = plt.subplots(figsize=(6, 5))
    
    X = data[k]['X']
    Y = data[k]['Y']
    u_pred = data[k]['u_pred']
    
    levels = 50
    contour = ax.contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$u(x,y)$', rotation=0, labelpad=15)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'PINN Solution for $k={k}$')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task2_solution_k100.pdf')
    plt.savefig('paper_figures/task2_solution_k100.png')
    plt.close()
    print("✓ Generated: task2_solution_k100.pdf")

def plot_solution_k500(data):
    """Figure 2: Solution field for k=500."""
    k = 500
    fig, ax = plt.subplots(figsize=(6, 5))
    
    X = data[k]['X']
    Y = data[k]['Y']
    u_pred = data[k]['u_pred']
    
    levels = 50
    contour = ax.contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$u(x,y)$', rotation=0, labelpad=15)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'PINN Solution for $k={k}$')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task2_solution_k500.pdf')
    plt.savefig('paper_figures/task2_solution_k500.png')
    plt.close()
    print("✓ Generated: task2_solution_k500.pdf")

def plot_solution_k1000(data):
    """Figure 3: Solution field for k=1000."""
    k = 1000
    fig, ax = plt.subplots(figsize=(6, 5))
    
    X = data[k]['X']
    Y = data[k]['Y']
    u_pred = data[k]['u_pred']
    
    levels = 50
    contour = ax.contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$u(x,y)$', rotation=0, labelpad=15)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'PINN Solution for $k={k}$')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task2_solution_k1000.pdf')
    plt.savefig('paper_figures/task2_solution_k1000.png')
    plt.close()
    print("✓ Generated: task2_solution_k1000.pdf")

def plot_solutions_comparison(data):
    """Figure 4: Three-panel comparison of all k values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    k_values = [100, 500, 1000]
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        X = data[k]['X']
        Y = data[k]['Y']
        u_pred = data[k]['u_pred']
        metrics = data[k]['metrics']
        
        levels = 50
        im = ax.contourf(X, Y, u_pred, levels=levels, cmap='RdBu_r')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'({"abc"[idx]}) $k={k}$ (L2: {metrics["rel_l2_error_percent"]:.3f}%)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('paper_figures/task2_solutions_comparison.pdf')
    plt.savefig('paper_figures/task2_solutions_comparison.png')
    plt.close()
    print("✓ Generated: task2_solutions_comparison.pdf")

def plot_loss_comparison(data):
    """Figure 5: Training loss comparison for different k values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = [100, 500, 1000]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for k, color in zip(k_values, colors):
        loss_df = data[k]['loss_df']
        ax.plot(loss_df['epoch'], loss_df['total_loss'], 
               linewidth=2, color=color, label=f'$k={k}$')
    
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Convergence: High Wave Numbers')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig('paper_figures/task2_loss_comparison.pdf')
    plt.savefig('paper_figures/task2_loss_comparison.png')
    plt.close()
    print("✓ Generated: task2_loss_comparison.pdf")

def plot_error_vs_k(data, summary):
    """Figure 6: Relative L2 error vs wave number k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = [100, 500, 1000]
    errors = [data[k]['metrics']['rel_l2_error_percent'] for k in k_values]
    
    ax.plot(k_values, errors, 'o-', linewidth=2, markersize=10, color='#d62728')
    
    # Add value labels
    for k, err in zip(k_values, errors):
        ax.text(k, err, f'{err:.3f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Wave Number $k$')
    ax.set_ylabel('Relative $L_2$ Error (%)')
    ax.set_title('Error Scaling with Wave Number')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    plt.savefig('paper_figures/task2_error_vs_k.pdf')
    plt.savefig('paper_figures/task2_error_vs_k.png')
    plt.close()
    print("✓ Generated: task2_error_vs_k.pdf")

def plot_error_distribution_k1000(data):
    """Figure 7: Error distribution for k=1000 (most challenging)."""
    k = 1000
    fig, ax = plt.subplots(figsize=(6, 5))
    
    X = data[k]['X']
    Y = data[k]['Y']
    error = data[k]['error']
    metrics = data[k]['metrics']
    
    levels = 50
    contour = ax.contourf(X, Y, error, levels=levels, cmap='hot_r')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error', rotation=0, labelpad=15)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Error Distribution for $k={k}$ (Max: {metrics["max_error"]:.2e})')
    ax.set_aspect('equal')
    
    plt.savefig('paper_figures/task2_error_k1000.pdf')
    plt.savefig('paper_figures/task2_error_k1000.png')
    plt.close()
    print("✓ Generated: task2_error_k1000.pdf")

def generate_summary_table(data, summary):
    """Generate summary table for LaTeX."""
    print("\n" + "="*70)
    print("TASK 2 SUMMARY METRICS (for LaTeX table)")
    print("="*70)
    
    k_values = [100, 500, 1000]
    
    print(f"{'k':<10} {'L2 Error (%)':<15} {'Max Error':<15} {'Time (min)':<15}")
    print("-" * 70)
    
    for k in k_values:
        metrics = data[k]['metrics']
        result = summary['results'][f'k_{k}']
        print(f"{k:<10} {metrics['rel_l2_error_percent']:<15.4f} "
              f"{metrics['max_error']:<15.2e} {result['training_time_minutes']:<15.2f}")
    
    print("="*70)
    
    # LaTeX table format
    print("\nLaTeX Table Entries:")
    for k in k_values:
        metrics = data[k]['metrics']
        result = summary['results'][f'k_{k}']
        print(f"{k} & {metrics['rel_l2_error_percent']:.4f}\\% & "
              f"{metrics['max_error']:.2e} & {result['training_time_minutes']:.2f} \\\\")

def main():
    print("="*70)
    print("Task 2 Visualization - Generating Paper Figures")
    print("="*70)
    
    # Load all data
    data, summary = load_all_data()
    
    print("\nGenerating figures...")
    
    # Individual solution plots
    plot_solution_k100(data)
    plot_solution_k500(data)
    plot_solution_k1000(data)
    
    # Comparison plots
    plot_solutions_comparison(data)
    plot_loss_comparison(data)
    plot_error_vs_k(data, summary)
    plot_error_distribution_k1000(data)
    
    # Summary table
    generate_summary_table(data, summary)
    
    print("\n" + "="*70)
    print("All figures saved to paper_figures/")
    print("="*70)
    
    print("\nGenerated figures:")
    print("  1. task2_solution_k100.pdf - Solution for k=100")
    print("  2. task2_solution_k500.pdf - Solution for k=500")
    print("  3. task2_solution_k1000.pdf - Solution for k=1000")
    print("  4. task2_solutions_comparison.pdf - 3-panel comparison")
    print("  5. task2_loss_comparison.pdf - Training loss comparison")
    print("  6. task2_error_vs_k.pdf - Error scaling analysis")
    print("  7. task2_error_k1000.pdf - Error distribution for k=1000")
    print("\nRecommended for paper: Use figures 4, 5, and 6")

if __name__ == "__main__":
    main()
