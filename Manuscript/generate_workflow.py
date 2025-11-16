"""
Generate all architecture and workflow diagrams for the paper
Unified script to create:
1. PINN principle diagram
2. SIREN architecture diagram  
3. Dual-network architecture (Task 3)
4. Technical workflow diagram
5. Training procedure flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import numpy as np
from pathlib import Path

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
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

def add_arrow(ax, x1, y1, x2, y2, label='', color='black', lw=2):
    """Add a fancy arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        mutation_scale=30, 
        lw=lw,
        color=color,
        zorder=1
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='bottom', 
                fontsize=14, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='none'))

def add_box(ax, x, y, width, height, text, color='lightblue', textcolor='black'):
    """Add a fancy box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", 
        edgecolor='black',
        facecolor=color,
        linewidth=2,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=16, weight='bold', color=textcolor, zorder=3)


def plot_pinn_principle():
    """
    Figure 1: PINN Principle Diagram
    Shows the fundamental concept of PINNs
    """
    print("Generating PINN principle diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.7, 'Physics-Informed Neural Network Framework', 
            ha='center', fontsize=20, weight='bold')
    
    # Input layer
    add_box(ax, 1.5, 3, 1.2, 0.8, 'Input\n$(x, y)$', 'lightgreen')
    
    # Neural network
    add_box(ax, 3.5, 3, 1.5, 1.2, 'Neural\nNetwork\n$u_\\theta(x,y)$', 'lightblue')
    
    # Output
    add_box(ax, 5.8, 3, 1.2, 0.8, 'Output\n$u(x,y)$', 'lightcoral')
    
    # Automatic differentiation
    add_box(ax, 3.5, 1, 1.8, 0.8, 'Auto-Diff\n$\\nabla u, \\nabla^2 u$', 'plum')
    
    # Loss components
    add_box(ax, 7.5, 4.5, 1.4, 0.7, 'Data Loss\n$\\mathcal{L}_{data}$', 'lightyellow')
    add_box(ax, 7.5, 3.3, 1.4, 0.7, 'PDE Loss\n$\\mathcal{L}_{PDE}$', 'lightyellow')
    add_box(ax, 7.5, 2.1, 1.4, 0.7, 'BC Loss\n$\\mathcal{L}_{BC}$', 'lightyellow')
    
    # Total loss
    add_box(ax, 7.5, 0.5, 1.6, 0.7, 'Total Loss\n$\\mathcal{L}_{total}$', 'orange')
    
    # Optimizer
    add_box(ax, 5, 0.5, 1.2, 0.7, 'Optimizer\n(Adam)', 'lightgreen')
    
    # Arrows
    add_arrow(ax, 2.1, 3, 2.75, 3, '', 'black')
    add_arrow(ax, 4.25, 3, 5.2, 3, '', 'black')
    add_arrow(ax, 3.5, 2.4, 3.5, 1.4, '', 'blue')
    add_arrow(ax, 4.4, 1, 6.8, 3.3, '', 'blue')
    add_arrow(ax, 6.4, 3, 6.8, 4.5, '', 'black')
    add_arrow(ax, 7.5, 1.45, 7.5, 0.85, '', 'black')
    add_arrow(ax, 6.9, 0.5, 5.6, 0.5, '', 'red')
    add_arrow(ax, 4.4, 0.5, 3.5, 2.2, 'Update $\\theta$', 'red')
    
    # Annotations
    ax.text(5, 4.8, 'Forward Pass', ha='center', fontsize=14, 
            style='italic', color='blue')
    ax.text(8.5, 1.2, 'Loss\nComponents', ha='center', fontsize=14,
            style='italic', color='darkred')
    ax.text(3, 0.2, 'Backpropagation', ha='center', fontsize=14,
            style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pinn_principle.pdf')
    plt.savefig(output_dir / 'pinn_principle.png')
    plt.close()
    print("  ✓ Saved: pinn_principle.pdf")


def plot_siren_architecture():
    """
    Figure 2: SIREN Network Architecture
    Detailed layer-by-layer structure
    """
    print("\nGenerating SIREN architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(8, 7.6, 'SIREN Architecture for Forward Problems (Tasks 1 & 2)', 
            ha='center', fontsize=20, weight='bold')
    
    # Input layer
    y_start = 4
    x_pos = 1.5
    
    # Input
    add_box(ax, x_pos, y_start, 1.2, 1.5, 'Input\nLayer\n\n$(x, y)$\n\n2 neurons', 
            'lightgreen', 'black')
    
    # First scaling
    x_pos += 2
    add_arrow(ax, x_pos-0.9, y_start, x_pos-0.3, y_start, '', 'black', 2)
    add_box(ax, x_pos, y_start, 1.2, 1, 'Scale\n$\\omega_0=30$', 'lightyellow')
    ax.text(x_pos, y_start-1.2, 'First Layer\nInitialization:\n$W_0 \\sim \\mathcal{U}(-1/2, 1/2)$',
            ha='center', fontsize=12, style='italic')
    
    # Hidden layers
    for i in range(4):
        x_pos += 2.2
        add_arrow(ax, x_pos-1.0, y_start, x_pos-0.7, y_start, '', 'black', 2)
        
        add_box(ax, x_pos, y_start, 1.4, 1.8, 
                f'Hidden\nLayer {i+1}\n\n100 neurons\n\n$\\sin(\\cdot)$', 
                'lightblue', 'black')
        
        if i == 0:
            ax.text(x_pos, y_start-1.8, 'Hidden Layers\nInitialization:\n$W_i \\sim \\mathcal{U}(-\\sqrt{6/n_{in}}/\\omega_0, \\sqrt{6/n_{in}}/\\omega_0)$',
                    ha='center', fontsize=11, style='italic')
    
    # Output layer
    x_pos += 2.2
    add_arrow(ax, x_pos-1.0, y_start, x_pos-0.7, y_start, '', 'black', 2)
    add_box(ax, x_pos, y_start, 1.2, 1.5, 'Output\nLayer\n\n$u(x,y)$\n\n1 neuron',
            'lightcoral', 'black')
    ax.text(x_pos, y_start-1.2, 'Linear\nActivation',
            ha='center', fontsize=12, style='italic')
    
    # Network info box
    info_text = ('Network Parameters:\n'
                 '• Input: 2D coordinates\n'
                 '• Hidden: 4 × 100 neurons\n'
                 '• Output: 1D solution\n'
                 '• Total Parameters: 30,701\n'
                 '• Activation: $\\sin(\\cdot)$')
    ax.text(8, 1.2, info_text, ha='center', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                     edgecolor='black', linewidth=2))
    
    # Key features
    ax.text(1, 6.8, 'Key Features:', fontsize=16, weight='bold')
    features = [
        '1. Sinusoidal activations for oscillatory solutions',
        '2. Specialized initialization for stable training',
        '3. Frequency parameter $\\omega_0$ controls receptive field'
    ]
    for i, feature in enumerate(features):
        ax.text(1, 6.3 - i*0.4, feature, fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'siren_architecture.pdf')
    plt.savefig(output_dir / 'siren_architecture.png')
    plt.close()
    print("  ✓ Saved: siren_architecture.pdf")


def plot_dual_network_architecture():
    """
    Figure 3: Dual-Network Architecture for Inverse Problem (Task 3)
    """
    print("\nGenerating dual-network architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Dual-Network Architecture for Inverse Problem (Task 3)', 
            ha='center', fontsize=20, weight='bold')
    
    # Shared input
    add_box(ax, 2, 5, 1.5, 1, 'Input\n$(x, y)$', 'lightgreen')
    
    # Upper branch - Solution Network
    y_u = 7
    ax.text(8, y_u + 1.5, 'Solution Network $u_{\\theta_u}(x,y)$', 
            ha='center', fontsize=18, weight='bold', color='blue')
    
    add_box(ax, 4.5, y_u, 1.2, 0.8, 'Input\n2', 'lightblue')
    add_box(ax, 6.5, y_u, 1.2, 0.8, 'Hidden\n64', 'lightblue')
    add_box(ax, 8.5, y_u, 1.2, 0.8, 'Hidden\n64', 'lightblue')
    add_box(ax, 10.5, y_u, 1.2, 0.8, 'Hidden\n64', 'lightblue')
    add_box(ax, 12.5, y_u, 1.2, 0.8, 'Output\n1', 'lightcoral')
    
    # Arrows for upper branch
    for i in range(4):
        add_arrow(ax, 4.5 + i*2 + 0.6, y_u, 4.5 + (i+1)*2 - 0.6, y_u, '', 'blue')
    
    # Lower branch - Parameter Network
    y_k = 3
    ax.text(8, y_k + 1.5, 'Parameter Network $k_{\\theta_k}(x,y)$', 
            ha='center', fontsize=18, weight='bold', color='darkgreen')
    
    add_box(ax, 4.5, y_k, 1.2, 0.8, 'Input\n2', 'lightgreen')
    add_box(ax, 6.5, y_k, 1.2, 0.8, 'Hidden\n32', 'lightgreen')
    add_box(ax, 8.5, y_k, 1.2, 0.8, 'Hidden\n32', 'lightgreen')
    add_box(ax, 10.5, y_k, 1.2, 0.8, 'Hidden\n32', 'lightgreen')
    add_box(ax, 12.5, y_k, 1.3, 0.8, 'Softplus\n$k>0$', 'orange')
    
    # Arrows for lower branch
    for i in range(4):
        add_arrow(ax, 4.5 + i*2 + 0.6, y_k, 4.5 + (i+1)*2 - 0.6, y_k, '', 'darkgreen')
    
    # Input splitting arrows
    add_arrow(ax, 2.75, 5.3, 3.9, y_u-0.3, '', 'black', 2)
    add_arrow(ax, 2.75, 4.7, 3.9, y_k+0.3, '', 'black', 2)
    
    # Output combining
    add_box(ax, 14.5, 5, 1.2, 1.5, 'PDE\nResidual', 'plum')
    add_arrow(ax, 13.1, y_u, 13.9, 5.5, '$u$', 'blue')
    add_arrow(ax, 13.1, y_k, 13.9, 4.5, '$k$', 'darkgreen')
    
    # Network specifications
    spec_u = ('$u$-Network:\n'
             '• Architecture: [2,64,64,64,1]\n'
             '• Activation: Tanh\n'
             '• Parameters: 12,481')
    ax.text(13, y_u - 1.2, spec_u, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='blue', linewidth=1.5))
    
    spec_k = ('$k$-Network:\n'
             '• Architecture: [2,32,32,32,1]\n'
             '• Activation: Tanh + Softplus\n'
             '• Parameters: 3,329')
    ax.text(13, y_k - 1.2, spec_k, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                     edgecolor='darkgreen', linewidth=1.5))
    
    # Key advantages
    ax.text(1, 1.5, 'Key Advantages:', fontsize=14, weight='bold')
    advantages = [
        '• Separate specialization for solution and parameter',
        '• Ensured positivity: $k(x,y) > 0$ via Softplus',
        '• Reduced parameter count: 15,810 total'
    ]
    for i, adv in enumerate(advantages):
        ax.text(1, 1.0 - i*0.35, adv, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dual_network_architecture.pdf')
    plt.savefig(output_dir / 'dual_network_architecture.png')
    plt.close()
    print("  ✓ Saved: dual_network_architecture.pdf")


def plot_workflow_diagram():
    """
    Figure 4: Complete Technical Workflow
    Shows the entire process from problem to solution
    """
    print("\nGenerating technical workflow diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15.5, 'PINN Technical Workflow', 
            ha='center', fontsize=22, weight='bold')
    
    y = 14
    
    # Step 1: Problem Definition
    add_box(ax, 5, y, 4, 1, 'Step 1: Problem Definition', 'lightblue')
    ax.text(5, y-0.7, 'PDE: $-\\nabla \\cdot (k\\nabla u) = f$ in $\\Omega$\nBC: $u = 0$ on $\\partial\\Omega$',
            ha='center', fontsize=12)
    add_arrow(ax, 5, y-0.5, 5, y-1.3, '', 'black', 3)
    y -= 2.5
    
    # Step 2: Network Design
    add_box(ax, 5, y, 4, 1, 'Step 2: Network Architecture', 'lightgreen')
    ax.text(5, y-0.7, 'Forward: SIREN with $\\sin$ activation\nInverse: Dual networks $(u_{\\theta_u}, k_{\\theta_k})$',
            ha='center', fontsize=12)
    add_arrow(ax, 5, y-0.5, 5, y-1.3, '', 'black', 3)
    y -= 2.5
    
    # Step 3: Loss Function
    add_box(ax, 5, y, 4, 1, 'Step 3: Loss Function Design', 'lightyellow')
    ax.text(5, y-0.8, '$\\mathcal{L} = \\lambda_{PDE}\\mathcal{L}_{PDE} + \\lambda_{BC}\\mathcal{L}_{BC} + \\lambda_{data}\\mathcal{L}_{data}$\n$+ \\lambda_{reg}\\mathcal{L}_{reg} + \\lambda_{smooth}\\mathcal{L}_{smooth}$',
            ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.5, 5, y-1.5, '', 'black', 3)
    y -= 2.8
    
    # Step 4: Collocation Sampling
    add_box(ax, 5, y, 4, 1, 'Step 4: Dynamic Sampling', 'plum')
    ax.text(5, y-0.7, 'Interior: $N_i$ points from $\\Omega$\nBoundary: $N_b$ points from $\\partial\\Omega$',
            ha='center', fontsize=12)
    add_arrow(ax, 5, y-0.5, 5, y-1.3, '', 'black', 3)
    y -= 2.5
    
    # Step 5: Training Loop
    add_box(ax, 5, y, 4, 1.2, 'Step 5: Training Loop', 'lightcoral')
    ax.text(5, y-0.8, 'Adam optimizer + Cosine annealing\nGradient clipping\nAuto-differentiation',
            ha='center', fontsize=12)
    
    # Iteration arrow
    add_arrow(ax, 7.3, y, 7.8, y+5, 'Iterate', 'red', 2)
    add_arrow(ax, 7.8, y+5, 7.3, y+5.5, '', 'red', 2)
    
    add_arrow(ax, 5, y-0.6, 5, y-1.5, 'Converged?', 'black', 3)
    y -= 2.8
    
    # Step 6: Evaluation
    add_box(ax, 5, y, 4, 1, 'Step 6: Evaluation', 'lightgreen')
    ax.text(5, y-0.7, 'Compute errors on test grid\nVisualize solution fields',
            ha='center', fontsize=12)
    add_arrow(ax, 5, y-0.5, 5, y-1.3, '', 'black', 3)
    y -= 2.5
    
    # Step 7: Results
    add_box(ax, 5, y, 4, 1, 'Step 7: Results & Analysis', 'orange')
    ax.text(5, y-0.7, 'Error metrics\nParameter identification\nPhysical validation',
            ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'workflow_diagram.pdf')
    plt.savefig(output_dir / 'workflow_diagram.png')
    plt.close()
    print("  ✓ Saved: workflow_diagram.pdf")


def plot_training_procedure():
    """
    Figure 5: Detailed Training Procedure Flowchart
    """
    print("\nGenerating training procedure flowchart...")
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Training Procedure Flowchart', 
            ha='center', fontsize=20, weight='bold')
    
    y = 12.5
    
    # Initialize
    add_box(ax, 5, y, 3, 0.8, 'Initialize Networks', 'lightgreen')
    add_arrow(ax, 5, y-0.4, 5, y-1, '', 'black', 2)
    y -= 1.5
    
    # Epoch loop start
    add_box(ax, 5, y, 3.5, 0.8, 'For each epoch $e = 1, ..., E$', 'lightyellow')
    add_arrow(ax, 5, y-0.4, 5, y-1, '', 'black', 2)
    y -= 1.5
    
    # Sample points
    add_box(ax, 5, y, 3.2, 0.8, 'Sample Collocation Points', 'lightblue')
    ax.text(5, y-0.6, '$\\{x_i, y_i\\}_{i=1}^{N_i+N_b}$', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Forward pass
    add_box(ax, 5, y, 3, 0.8, 'Forward Pass', 'plum')
    ax.text(5, y-0.6, 'Compute $u_\\theta$, derivatives', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Compute losses
    add_box(ax, 5, y, 3, 0.8, 'Compute Losses', 'lightcoral')
    ax.text(5, y-0.6, '$\\mathcal{L}_{total} = \\sum_i \\lambda_i \\mathcal{L}_i$', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Backprop
    add_box(ax, 5, y, 3, 0.8, 'Backpropagation', 'orange')
    ax.text(5, y-0.6, 'Compute gradients $\\nabla_\\theta \\mathcal{L}$', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Gradient clipping (optional)
    add_box(ax, 5, y, 3, 0.8, 'Gradient Clipping', 'plum')
    ax.text(5, y-0.6, 'If $\\|\\nabla\\| > g_{max}$', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Update parameters
    add_box(ax, 5, y, 3, 0.8, 'Update Parameters', 'lightgreen')
    ax.text(5, y-0.6, '$\\theta \\leftarrow \\theta - \\alpha \\nabla_\\theta \\mathcal{L}$', ha='center', fontsize=11)
    add_arrow(ax, 5, y-0.4, 5, y-1.1, '', 'black', 2)
    y -= 1.6
    
    # Learning rate update
    add_box(ax, 5, y, 3, 0.8, 'Update Learning Rate', 'lightyellow')
    ax.text(5, y-0.6, 'Cosine annealing schedule', ha='center', fontsize=11)
    
    # Loop back
    add_arrow(ax, 6.8, y, 8.5, 11, 'Next\nepoch', 'red', 2)
    add_arrow(ax, 8.5, 11.5, 6.8, 12.1, '', 'red', 2)
    
    # Converged
    add_arrow(ax, 5, y-0.4, 5, y-1.1, 'Converged', 'darkgreen', 2)
    y -= 1.6
    
    # End
    add_box(ax, 5, y, 3, 0.8, 'Training Complete', 'lightgreen')
    
    # Side notes
    ax.text(0.5, 10, 'Key Components:', fontsize=14, weight='bold')
    notes = [
        '• Dynamic sampling',
        '• Auto-differentiation',
        '• Adaptive learning rate',
        '• Gradient stabilization'
    ]
    for i, note in enumerate(notes):
        ax.text(0.5, 9.5 - i*0.4, note, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_procedure.pdf')
    plt.savefig(output_dir / 'training_procedure.png')
    plt.close()
    print("  ✓ Saved: training_procedure.pdf")


def main():
    """Generate all diagrams"""
    print("="*70)
    print("Generating Architecture and Workflow Diagrams")
    print("="*70)
    
    # Generate all diagrams
    plot_pinn_principle()
    plot_siren_architecture()
    plot_dual_network_architecture()
    plot_workflow_diagram()
    plot_training_procedure()
    
    print("\n" + "="*70)
    print("✅ All diagrams generated successfully!")
    print("="*70)
    print(f"\nFiles saved in: {output_dir}/")
    print("\nGenerated diagrams:")
    print("  1. pinn_principle.pdf - PINN fundamental concept")
    print("  2. siren_architecture.pdf - SIREN network structure")
    print("  3. dual_network_architecture.pdf - Dual-network for Task 3")
    print("  4. workflow_diagram.pdf - Complete technical workflow")
    print("  5. training_procedure.pdf - Detailed training flowchart")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()