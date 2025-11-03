"""
Helmholtz方程求解器 - 可视化脚本
从训练脚本生成的数据文件中读取数据并生成图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 创建图表输出目录
os.makedirs('results/figures', exist_ok=True)

def load_data():
    """加载所有数据文件"""
    print("="*80)
    print(" "*20 + "Helmholtz方程求解器 - 可视化脚本")
    print("="*80)
    print("\n加载数据文件...")
    
    # 加载训练历史
    df_history = pd.read_csv('results/data/training_history.csv')
    print(f"训练历史: {len(df_history)} 条记录")
    
    # 加载评估结果
    eval_data = np.load('results/data/evaluation_results.npz')
    X = eval_data['X']
    Y = eval_data['Y']
    u_pred = eval_data['u_pred']
    u_true = eval_data['u_true']
    error = eval_data['error']
    q_test = eval_data['q_test']
    print(f"评估结果: {u_pred.shape[0]}×{u_pred.shape[1]} 网格")
    
    # 加载误差统计
    df_stats = pd.read_csv('results/data/error_statistics.csv')
    stats = df_stats.iloc[0].to_dict()
    print(f"误差统计")
    
    # 加载截面数据
    df_cross_y0 = pd.read_csv('results/data/cross_section_y0.csv')
    df_cross_x0 = pd.read_csv('results/data/cross_section_x0.csv')
    print(f"截面数据")
    
    print("\n数据加载完成！\n")
    
    return {
        'history': df_history,
        'X': X, 'Y': Y,
        'u_pred': u_pred,
        'u_true': u_true,
        'error': error,
        'q_test': q_test,
        'stats': stats,
        'cross_y0': df_cross_y0,
        'cross_x0': df_cross_x0
    }

def plot_training_history(df_history):
    """绘制训练历史"""
    print("生成训练历史图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1 = axes[0]
    ax1.plot(df_history['epoch'], df_history['train_loss'], 
             'b-', linewidth=1.5, alpha=0.7, label='Train Loss')
    ax1.plot(df_history['epoch'], df_history['val_loss'], 
             'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Training History', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # 学习率曲线
    ax2 = axes[1]
    ax2.plot(df_history['epoch'], df_history['lr'], 
             'orange', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_history.png', dpi=300, bbox_inches='tight')
    print("已保存: training_history.png")
    plt.close()

def plot_comprehensive_results(data):
    """生成综合结果图"""
    print("生成综合结果图...")
    
    X = data['X']
    Y = data['Y']
    u_pred = data['u_pred']
    u_true = data['u_true']
    error = data['error']
    q_test = data['q_test']
    stats = data['stats']
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: 主要对比
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
    ax1.set_title('Predicted Solution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    ax1.set_aspect('equal')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, u_true, levels=50, cmap='RdBu_r')
    ax2.set_title('True Solution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    ax2.set_aspect('equal')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(X, Y, error, levels=50, cmap='hot')
    ax3.set_title(f'Error (Rel L2: {stats["relative_l2_error_percent"]:.2f}%)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, format='%.2e')
    ax3.set_aspect('equal')
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.contourf(X, Y, q_test, levels=50, cmap='viridis')
    ax4.set_title('Source Term q(x,y)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4)
    ax4.set_aspect('equal')
    
    # Row 2: 3D视图
    ax5 = fig.add_subplot(gs[1, 0], projection='3d')
    ax5.plot_surface(X, Y, u_pred, cmap='RdBu_r', alpha=0.9)
    ax5.set_title('3D Predicted', fontweight='bold')
    ax5.view_init(25, 45)
    
    ax6 = fig.add_subplot(gs[1, 1], projection='3d')
    ax6.plot_surface(X, Y, u_true, cmap='RdBu_r', alpha=0.9)
    ax6.set_title('3D True', fontweight='bold')
    ax6.view_init(25, 45)
    
    ax7 = fig.add_subplot(gs[1, 2], projection='3d')
    ax7.plot_surface(X, Y, error, cmap='hot', alpha=0.9)
    ax7.set_title('3D Error', fontweight='bold')
    ax7.view_init(25, 45)
    
    # 误差分布
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(error.flatten(), bins=50, color='orangered', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Absolute Error')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Error Distribution', fontweight='bold')
    ax8.axvline(stats['mean_error'], color='red', linestyle='--', linewidth=2)
    ax8.grid(alpha=0.3)
    
    # Row 3: 截面对比
    df_cross_y0 = data['cross_y0']
    df_cross_x0 = data['cross_x0']
    
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.plot(df_cross_y0['x_coord'], df_cross_y0['u_true_at_y0'], 
             'r--', linewidth=3, label='True', alpha=0.8)
    ax9.plot(df_cross_y0['x_coord'], df_cross_y0['u_pred_at_y0'], 
             'b-', linewidth=2, label='Predicted')
    ax9.fill_between(df_cross_y0['x_coord'], 
                      df_cross_y0['u_true_at_y0'], 
                      df_cross_y0['u_pred_at_y0'], 
                      alpha=0.2)
    ax9.set_xlabel('x', fontsize=12)
    ax9.set_ylabel('u(x, 0)', fontsize=12)
    ax9.set_title('Cross-section at y=0', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.plot(df_cross_x0['y_coord'], df_cross_x0['u_true_at_x0'], 
              'r--', linewidth=3, label='True', alpha=0.8)
    ax10.plot(df_cross_x0['y_coord'], df_cross_x0['u_pred_at_x0'], 
              'b-', linewidth=2, label='Predicted')
    ax10.fill_between(df_cross_x0['y_coord'], 
                       df_cross_x0['u_true_at_x0'], 
                       df_cross_x0['u_pred_at_x0'], 
                       alpha=0.2)
    ax10.set_xlabel('y', fontsize=12)
    ax10.set_ylabel('u(0, y)', fontsize=12)
    ax10.set_title('Cross-section at x=0', fontsize=14, fontweight='bold')
    ax10.legend()
    ax10.grid(alpha=0.3)
    
    plt.savefig('results/figures/comprehensive_results.png', dpi=300, bbox_inches='tight')
    print("已保存: comprehensive_results.png")
    plt.close()

def plot_solution_comparison(data):
    """绘制解的对比图（单独的大图）"""
    print("生成解场对比图...")
    
    X = data['X']
    Y = data['Y']
    u_pred = data['u_pred']
    u_true = data['u_true']
    error = data['error']
    stats = data['stats']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 预测解
    im1 = axes[0].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
    axes[0].set_title('Predicted Solution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # 真实解
    im2 = axes[1].contourf(X, Y, u_true, levels=50, cmap='RdBu_r')
    axes[1].set_title('True Solution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # 误差
    im3 = axes[2].contourf(X, Y, error, levels=50, cmap='hot')
    axes[2].set_title(f'Absolute Error\n(Rel L2: {stats["relative_l2_error_percent"]:.3f}%)', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], format='%.2e')
    
    plt.tight_layout()
    plt.savefig('results/figures/solution_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: solution_comparison.png")
    plt.close()

def plot_cross_sections(data):
    """绘制截面对比图"""
    print("生成截面对比图...")
    
    df_cross_y0 = data['cross_y0']
    df_cross_x0 = data['cross_x0']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # y=0截面
    axes[0].plot(df_cross_y0['x_coord'], df_cross_y0['u_true_at_y0'], 
                 'r--', linewidth=3, label='True Solution', alpha=0.8)
    axes[0].plot(df_cross_y0['x_coord'], df_cross_y0['u_pred_at_y0'], 
                 'b-', linewidth=2, label='Predicted Solution')
    axes[0].fill_between(df_cross_y0['x_coord'], 
                         df_cross_y0['u_true_at_y0'], 
                         df_cross_y0['u_pred_at_y0'], 
                         alpha=0.2, color='gray')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('u(x, 0)', fontsize=12)
    axes[0].set_title('Cross-section at y=0', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # x=0截面
    axes[1].plot(df_cross_x0['y_coord'], df_cross_x0['u_true_at_x0'], 
                 'r--', linewidth=3, label='True Solution', alpha=0.8)
    axes[1].plot(df_cross_x0['y_coord'], df_cross_x0['u_pred_at_x0'], 
                 'b-', linewidth=2, label='Predicted Solution')
    axes[1].fill_between(df_cross_x0['y_coord'], 
                         df_cross_x0['u_true_at_x0'], 
                         df_cross_x0['u_pred_at_x0'], 
                         alpha=0.2, color='gray')
    axes[1].set_xlabel('y', fontsize=12)
    axes[1].set_ylabel('u(0, y)', fontsize=12)
    axes[1].set_title('Cross-section at x=0', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/cross_sections.png', dpi=300, bbox_inches='tight')
    print("已保存: cross_sections.png")
    plt.close()

def plot_3d_views(data):
    """绘制3D视图"""
    print("生成3D视图...")
    
    X = data['X']
    Y = data['Y']
    u_pred = data['u_pred']
    u_true = data['u_true']
    error = data['error']
    
    fig = plt.figure(figsize=(18, 5))
    
    # 预测解3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, u_pred, cmap='RdBu_r', alpha=0.9)
    ax1.set_title('Predicted Solution (3D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.view_init(25, 45)
    
    # 真实解3D
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_true, cmap='RdBu_r', alpha=0.9)
    ax2.set_title('True Solution (3D)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.view_init(25, 45)
    
    # 误差3D
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.9)
    ax3.set_title('Absolute Error (3D)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('error')
    ax3.view_init(25, 45)
    
    plt.tight_layout()
    plt.savefig('results/figures/3d_views.png', dpi=300, bbox_inches='tight')
    print("已保存: 3d_views.png")
    plt.close()

def print_summary(data):
    """打印结果摘要"""
    stats = data['stats']
    
    print("\n" + "="*80)
    print(" "*30 + "结果摘要")
    print("="*80)
    print(f"\n模型性能:")
    print(f"  相对L2误差:  {stats['relative_l2_error']:.6e} ({stats['relative_l2_error_percent']:.4f}%)")
    print(f"  最大误差:    {stats['max_error']:.6e}")
    print(f"  平均误差:    {stats['mean_error']:.6e}")
    print(f"  边界误差:    {stats['boundary_error']:.6e}")
    print(f"  测试分辨率:  {stats['resolution']}×{stats['resolution']}")
    
    if stats['relative_l2_error'] < 0.01:
        print(f"\n 优秀! 相对L2误差 < 1%")
    elif stats['relative_l2_error'] < 0.05:
        print(f"\n 良好! 相对L2误差 < 5%")
    else:
        print(f"\n○ 可接受")
    
    print(f"\n生成的图表:")
    print(f"  - training_history.png      (训练历史)")
    print(f"  - comprehensive_results.png (综合结果)")
    print(f"  - solution_comparison.png   (解场对比)")
    print(f"  - cross_sections.png        (截面对比)")
    print(f"  - 3d_views.png              (3D视图)")
    print(f"\n所有图表保存在: results/figures/")
    print("="*80 + "\n")

def main():
    """主函数"""
    # 加载数据
    data = load_data()
    
    # 生成各类图表
    print("="*80)
    print("开始生成图表...")
    print("="*80 + "\n")
    
    plot_training_history(data['history'])
    plot_comprehensive_results(data)
    plot_solution_comparison(data)
    plot_cross_sections(data)
    plot_3d_views(data)
    
    # 打印摘要
    print_summary(data)
    
    print(" 所有图表生成完成！")

if __name__ == "__main__":
    main()
