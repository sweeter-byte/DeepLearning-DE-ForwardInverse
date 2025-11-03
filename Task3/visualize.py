"""
子任务3 - 可视化脚本

从保存的数据文件读取并生成所有图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 设置matplotlib字体以支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# 数据加载
# ============================================================================

def load_all_data(results_dir):
    """加载所有保存的数据"""
    results_dir = Path(results_dir)
    data = {}
    
    print("="*70)
    print("加载数据")
    print("="*70)
    
    # 训练历史
    history_file = results_dir / 'training_history.csv'
    if history_file.exists():
        data['history'] = pd.read_csv(history_file)
        print(f" 训练历史: {len(data['history'])} 条记录")
    
    # 训练数据预测
    train_pred_file = results_dir / 'training_data_predictions.csv'
    if train_pred_file.exists():
        data['train_pred'] = pd.read_csv(train_pred_file)
        print(f" 训练数据预测: {len(data['train_pred'])} 个点")
    
    # 网格预测
    grid_file = results_dir / 'grid_predictions.json'
    if grid_file.exists():
        with open(grid_file, 'r') as f:
            data['grid'] = json.load(f)
        grid_size = len(data['grid']['x'])
        print(f" 网格预测: {grid_size}x{grid_size}")
    
    # 评估结果
    eval_file = results_dir / 'evaluation_results.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            data['eval'] = json.load(f)
        print(f" 评估结果")
    
    # 配置
    config_file = results_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            data['config'] = json.load(f)
        print(f" 配置")
    
    return data

# 绘图函数
def plot_training_history(data, save_path):
    """绘制训练历史"""
    history = data['history']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 总损失
    axes[0, 0].semilogy(history['epoch'], history['total_loss'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数据拟合
    axes[0, 1].semilogy(history['epoch'], history['data_loss'], 'g-', lw=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Data Fitting Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PDE残差
    axes[0, 2].semilogy(history['epoch'], history['pde_loss'], 'r-', lw=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss (log scale)')
    axes[0, 2].set_title('PDE Residual Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 边界条件
    axes[1, 0].semilogy(history['epoch'], history['bc_loss'], 'orange', lw=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Boundary Condition Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # k正则化
    axes[1, 1].semilogy(history['epoch'], history['reg_loss'], 'purple', lw=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('k Regularization Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 学习率
    axes[1, 2].semilogy(history['epoch'], history['lr'], 'brown', lw=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate (log scale)')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {save_path.name}")
    plt.close()


def plot_k_field(data, save_path):
    """绘制k(x,y)场"""
    grid = data['grid']
    
    x = np.array(grid['x'])
    y = np.array(grid['y'])
    k = np.array(grid['k_pred'])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(18, 5))
    
    # 填充等高线
    ax1 = plt.subplot(131)
    cf1 = ax1.contourf(X, Y, k, levels=50, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Identified Parameter Field k(x,y)')
    ax1.set_aspect('equal')
    plt.colorbar(cf1, ax=ax1, label='k(x,y)')
    
    # 线条等高线
    ax2 = plt.subplot(132)
    cf2 = ax2.contourf(X, Y, k, levels=50, cmap='RdYlBu_r', alpha=0.7)
    c2 = ax2.contour(X, Y, k, levels=15, colors='black', linewidths=0.5, alpha=0.6)
    ax2.clabel(c2, inline=True, fontsize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('k(x,y) Contour Plot')
    ax2.set_aspect('equal')
    plt.colorbar(cf2, ax=ax2, label='k(x,y)')
    
    # 3D曲面
    ax3 = plt.subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, k, cmap='coolwarm', alpha=0.8, 
                            linewidth=0, antialiased=True)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('k(x,y)')
    ax3.set_title('k(x,y) 3D View')
    plt.colorbar(surf, ax=ax3, shrink=0.5, label='k(x,y)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" {save_path.name}")
    plt.close()


def plot_u_field(data, save_path):
    """绘制u(x,y)场"""
    grid = data['grid']
    train = data['train_pred']
    
    x = np.array(grid['x'])
    y = np.array(grid['y'])
    u = np.array(grid['u_pred'])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(18, 6))
    
    # u场
    ax1 = plt.subplot(131)
    cf1 = ax1.contourf(X, Y, u, levels=50, cmap='seismic')
    ax1.scatter(train['x'], train['y'], c=train['u_true'], 
                s=30, cmap='seismic', edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Solution Field u(x,y)')
    ax1.set_aspect('equal')
    plt.colorbar(cf1, ax=ax1, label='u(x,y)')
    
    # 预测vs真值
    ax2 = plt.subplot(132)
    scatter = ax2.scatter(train['u_true'], train['u_pred'], 
                         c=train['u_error'], s=50, cmap='hot', alpha=0.6)
    min_u = min(train['u_true'].min(), train['u_pred'].min())
    max_u = max(train['u_true'].max(), train['u_pred'].max())
    ax2.plot([min_u, max_u], [min_u, max_u], 'k--', lw=2, label='Perfect Fit')
    ax2.set_xlabel('u_true')
    ax2.set_ylabel('u_pred')
    ax2.set_title('Prediction Quality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Error')
    
    # 误差分布
    ax3 = plt.subplot(133)
    scatter3 = ax3.scatter(train['x'], train['y'], 
                          c=train['u_error'], s=100, cmap='hot', alpha=0.7)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Spatial Error Distribution')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, label='|u_pred - u_true|')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" {save_path.name}")
    plt.close()


def plot_k_analysis(data, save_path):
    """绘制k的统计分析"""
    train = data['train_pred']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Analysis of k(x,y)', fontsize=16, fontweight='bold')
    
    # 直方图
    axes[0, 0].hist(train['k_pred'], bins=30, color='skyblue', 
                    edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(train['k_pred'].mean(), color='red', 
                       linestyle='--', lw=2, 
                       label=f'Mean: {train["k_pred"].mean():.4f}')
    axes[0, 0].set_xlabel('k value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of k')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # k vs x
    scatter1 = axes[0, 1].scatter(train['x'], train['k_pred'], 
                                  c=train['y'], s=50, cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('k')
    axes[0, 1].set_title('k vs x')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 1], label='y coordinate')
    
    # k vs y
    scatter2 = axes[1, 0].scatter(train['y'], train['k_pred'],
                                  c=train['x'], s=50, cmap='plasma', alpha=0.6)
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('k')
    axes[1, 0].set_title('k vs y')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 0], label='x coordinate')
    
    # 空间分布
    scatter3 = axes[1, 1].scatter(train['x'], train['y'], 
                                  c=train['k_pred'], s=100, 
                                  cmap='coolwarm', alpha=0.8)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Spatial Distribution of k')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(scatter3, ax=axes[1, 1], label='k(x,y)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {save_path.name}")
    plt.close()


def plot_comparison(data, save_path):
    """绘制f, u, k的对比"""
    grid = data['grid']
    
    x = np.array(grid['x'])
    y = np.array(grid['y'])
    u = np.array(grid['u_pred'])
    k = np.array(grid['k_pred'])
    
    X, Y = np.meshgrid(x, y)
    
    # 计算f
    pi = np.pi
    term1 = (pi**2/2) * (1+X**2+Y**2) * np.sin(pi*X/2) * np.cos(pi*Y/2)
    term2 = pi * X * np.cos(pi*X/2) * np.cos(pi*Y/2)
    term3 = pi * Y * np.sin(pi*X/2) * np.sin(pi*Y/2)
    f = term1 - term2 + term3
    
    fig = plt.figure(figsize=(18, 5))
    
    # 源项f
    ax1 = plt.subplot(131)
    cf1 = ax1.contourf(X, Y, f, levels=50, cmap='coolwarm')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Source Term f(x,y)')
    ax1.set_aspect('equal')
    plt.colorbar(cf1, ax=ax1, label='f(x,y)')
    
    # 解u
    ax2 = plt.subplot(132)
    cf2 = ax2.contourf(X, Y, u, levels=50, cmap='seismic')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Solution u(x,y)')
    ax2.set_aspect('equal')
    plt.colorbar(cf2, ax=ax2, label='u(x,y)')
    
    # 参数k
    ax3 = plt.subplot(133)
    cf3 = ax3.contourf(X, Y, k, levels=50, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Parameter k(x,y)')
    ax3.set_aspect('equal')
    plt.colorbar(cf3, ax=ax3, label='k(x,y)')
    
    fig.suptitle('Poisson Equation: -∇·(k∇u) = f', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {save_path.name}")
    plt.close()


def create_summary_report(data, save_path):
    """创建摘要报告"""
    lines = []
    lines.append("="*70)
    lines.append("子任务3: Poisson方程参数识别 - 结果摘要")
    lines.append("="*70)
    lines.append("")
    
    # 配置
    if 'config' in data:
        config = data['config']
        lines.append("训练配置:")
        lines.append(f"  轮数: {config.get('epochs', 'N/A')}")
        lines.append(f"  学习率: {config.get('lr', 'N/A')}")
        lines.append(f"  配点数: {config.get('n_collocation', 'N/A')}")
        lines.append(f"  边界点数: {config.get('n_boundary', 'N/A')}")
        lines.append("")
    
    # 评估
    if 'eval' in data:
        eval_res = data['eval']
        lines.append("模型性能:")
        lines.append(f"  数据拟合MSE: {eval_res.get('data_mse', 'N/A'):.6e}")
        lines.append(f"  数据拟合相对误差: {eval_res.get('data_rel_error', 'N/A'):.4%}")
        lines.append(f"  PDE残差MSE: {eval_res.get('pde_mse', 'N/A'):.6e}")
        lines.append(f"  PDE残差相对误差: {eval_res.get('pde_rel_error', 'N/A'):.4%}")
        lines.append("")
        lines.append("识别的k(x,y):")
        lines.append(f"  最小值: {eval_res.get('k_min', 'N/A'):.6f}")
        lines.append(f"  最大值: {eval_res.get('k_max', 'N/A'):.6f}")
        lines.append(f"  平均值: {eval_res.get('k_mean', 'N/A'):.6f}")
        lines.append(f"  标准差: {eval_res.get('k_std', 'N/A'):.6f}")
        lines.append("")
    
    # 最终损失
    if 'history' in data:
        history = data['history']
        final = history.iloc[-1]
        lines.append("最终损失:")
        lines.append(f"  总损失: {final['total_loss']:.6e}")
        lines.append(f"  数据: {final['data_loss']:.6e}")
        lines.append(f"  PDE: {final['pde_loss']:.6e}")
        lines.append(f"  边界: {final['bc_loss']:.6e}")
        lines.append("")
    
    lines.append("="*70)
    
    report = "\n".join(lines)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ {save_path.name}")
    print("\n" + report)


# 主函数
def main():
    """主函数"""
    results_dir = Path('results')
    
    print("="*70)
    print("子任务3 - 可视化")
    print("="*70)
    print(f"数据目录: {results_dir}\n")
    
    # 加载数据
    data = load_all_data(results_dir)
    
    if not data:
        print("\n错误: 未找到数据文件！")
        print("请先运行: python train.py")
        return
    
    # 创建图表目录
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("生成图表")
    print("="*70)
    
    # 生成各类图表
    if 'history' in data:
        plot_training_history(data, plots_dir / '1_training_history.png')
    
    if 'grid' in data:
        plot_k_field(data, plots_dir / '2_k_field.png')
        plot_comparison(data, plots_dir / '5_comparison.png')
    
    if 'train_pred' in data:
        plot_u_field(data, plots_dir / '3_u_field.png')
        plot_k_analysis(data, plots_dir / '4_k_analysis.png')
    
    # 生成摘要
    create_summary_report(data, results_dir / 'summary_report.txt')
    
    print(f"\n{'='*70}")
    print("完成！")
    print("="*70)
    print(f"图表保存在: {plots_dir}")
    print("\n生成的图表:")
    for file in sorted(plots_dir.glob('*.png')):
        print(f"  - {file.name}")
    print("="*70)


if __name__ == '__main__':
    main()
