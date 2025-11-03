"""
K-Conditioned FNO 
从训练脚本生成的数据文件中读取数据并生成图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import os

# 创建图表输出目录
os.makedirs('results/figures', exist_ok=True)

def load_data():
    """加载所有数据文件"""
    print("\n加载数据文件...")
    
    # 加载训练历史
    df_history = pd.read_csv('results/data/training_history.csv')
    print(f" 训练历史: {len(df_history)} 条记录")
    
    # 加载完整结果
    with open('results/analysis/full_results.pkl', 'rb') as f:
        results = pickle.load(f)
    print(f" 完整结果: {len(results)} 个波数")
    
    # 加载误差摘要
    df_summary = pd.read_csv('results/analysis/error_summary.csv')
    print(f" 误差摘要")
    
    # 加载归一化参数
    with open('results/data/norm_params.json', 'r') as f:
        norm_params = json.load(f)
    print(f" 归一化参数")
    
    print("\n数据加载完成！\n")
    
    return {
        'history': df_history,
        'results': results,
        'summary': df_summary,
        'norm_params': norm_params
    }

def plot_training_summary(data):
    """生成训练总结图（4子图）"""
    print("生成训练总结图...")
    
    df_train = data['history']
    results = data['results']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 训练历史
    axes[0, 0].plot(df_train['epoch'], df_train['train_loss'], 
                    'b-', linewidth=1.5, alpha=0.7, label='Train')
    valid_val = df_train[df_train['val_loss'] > 0]
    axes[0, 0].plot(valid_val['epoch'], valid_val['val_loss'], 
                    'r-', linewidth=2, label='Validation')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Training History', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, which='both')
    
    # 子图2: 学习率调度
    axes[0, 1].plot(df_train['epoch'], df_train['lr'], 'orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 误差vs波数
    k_values = sorted([int(k) for k in results.keys()])
    rel_errors = [results[k]['relative_l2_error'] * 100 for k in k_values]
    
    axes[1, 0].plot(k_values, rel_errors, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[1, 0].set_xlabel('Wavenumber k', fontsize=12)
    axes[1, 0].set_ylabel('Relative L2 Error (%)', fontsize=12)
    axes[1, 0].set_title('Error vs Wavenumber', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    for k, err in zip(k_values, rel_errors):
        axes[1, 0].text(k, err, f'{err:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 子图4: 误差对比柱状图
    axes[1, 1].bar(range(len(k_values)), rel_errors, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(k_values)))
    axes[1, 1].set_xticklabels([f'k={k}' for k in k_values])
    axes[1, 1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1, 1].set_title('Error Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(rel_errors):
        axes[1, 1].text(i, val, f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_summary.png', dpi=300, bbox_inches='tight')
    print(" 已保存: training_summary.png")
    plt.close()

def plot_solution_for_each_k(data):
    """为每个k生成解场对比图（3子图）"""
    print("生成各波数的解场对比图...")
    
    results = data['results']
    k_values = sorted([int(k) for k in results.keys()])
    
    for k in k_values:
        res = results[k]
        X, Y = res['X'], res['Y']
        u_true = res['u_true']
        u_pred = res['u_pred']
        error = res['error']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 真实解
        im1 = axes[0].contourf(X, Y, u_true, levels=50, cmap='RdBu_r')
        axes[0].set_title(f'True Solution (k={k})', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])
        
        # 预测解
        im2 = axes[1].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
        axes[1].set_title(f'Predicted Solution (k={k})', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('y', fontsize=12)
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        # 误差
        im3 = axes[2].contourf(X, Y, error, levels=50, cmap='hot')
        axes[2].set_title(f'Error (k={k})\nRel L2: {res["relative_l2_error"]*100:.4f}%', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel('y', fontsize=12)
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'results/figures/solution_k{k:04d}.png', dpi=300, bbox_inches='tight')
        print(f" k={k}解图")
        plt.close()

def plot_training_history_detailed(data):
    """生成详细的训练历史图"""
    print("生成详细训练历史图...")
    
    df_history = data['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练和验证损失
    axes[0].plot(df_history['epoch'], df_history['train_loss'], 
                 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
    valid_val = df_history[df_history['val_loss'] > 0]
    axes[0].plot(valid_val['epoch'], valid_val['val_loss'], 
                 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # 学习率
    axes[1].plot(df_history['epoch'], df_history['lr'], 
                 'orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_history_detailed.png', dpi=300, bbox_inches='tight')
    print(" 已保存: training_history_detailed.png")
    plt.close()

def plot_error_analysis(data):
    """生成误差分析图"""
    print("生成误差分析图...")
    
    results = data['results']
    summary = data['summary']
    
    k_values = summary['k'].values
    rel_errors = summary['Relative L2 Error (%)'].values
    max_errors = summary['Max Error'].values
    mean_errors = summary['Mean Error'].values
    inference_times = summary['Inference Time (ms)'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 相对L2误差
    axes[0, 0].plot(k_values, rel_errors, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].set_xlabel('Wavenumber k', fontsize=12)
    axes[0, 0].set_ylabel('Relative L2 Error (%)', fontsize=12)
    axes[0, 0].set_title('Relative L2 Error vs k', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    for k, err in zip(k_values, rel_errors):
        axes[0, 0].text(k, err, f'{err:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 最大误差
    axes[0, 1].plot(k_values, max_errors, 's-', linewidth=2, markersize=8, color='crimson')
    axes[0, 1].set_xlabel('Wavenumber k', fontsize=12)
    axes[0, 1].set_ylabel('Max Absolute Error', fontsize=12)
    axes[0, 1].set_title('Maximum Error vs k', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 平均误差
    axes[1, 0].plot(k_values, mean_errors, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Wavenumber k', fontsize=12)
    axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1, 0].set_title('Mean Error vs k', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 推理时间
    axes[1, 1].bar(range(len(k_values)), inference_times, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(k_values)))
    axes[1, 1].set_xticklabels([f'k={k}' for k in k_values])
    axes[1, 1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1, 1].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(inference_times):
        axes[1, 1].text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/error_analysis.png', dpi=300, bbox_inches='tight')
    print(" 已保存: error_analysis.png")
    plt.close()

def plot_combined_solutions(data):
    """生成组合解场图（展示多个k）"""
    print("生成组合解场对比图...")
    
    results = data['results']
    k_values = sorted([int(k) for k in results.keys()])
    
    # 选择3个代表性的k值进行展示
    selected_k = [k_values[0], k_values[len(k_values)//2], k_values[-1]]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    for i, k in enumerate(selected_k):
        res = results[k]
        X, Y = res['X'], res['Y']
        u_true = res['u_true']
        u_pred = res['u_pred']
        error = res['error']
        
        # 真实解
        im1 = axes[i, 0].contourf(X, Y, u_true, levels=50, cmap='RdBu_r')
        axes[i, 0].set_title(f'True (k={k})', fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel('y', fontsize=11)
        if i == 2:
            axes[i, 0].set_xlabel('x', fontsize=11)
        axes[i, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 预测解
        im2 = axes[i, 1].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
        axes[i, 1].set_title(f'Predicted (k={k})', fontsize=12, fontweight='bold')
        if i == 2:
            axes[i, 1].set_xlabel('x', fontsize=11)
        axes[i, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 误差
        im3 = axes[i, 2].contourf(X, Y, error, levels=50, cmap='hot')
        axes[i, 2].set_title(f'Error (k={k})\n{res["relative_l2_error"]*100:.3f}%', 
                            fontsize=12, fontweight='bold')
        if i == 2:
            axes[i, 2].set_xlabel('x', fontsize=11)
        axes[i, 2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[i, 2], format='%.2e')
    
    plt.tight_layout()
    plt.savefig('results/figures/combined_solutions.png', dpi=300, bbox_inches='tight')
    print(" 已保存: combined_solutions.png")
    plt.close()

def print_summary(data):
    """打印结果摘要"""
    summary = data['summary']
    
    print("\n" + "="*70)
    print("结果摘要")
    print("="*70)
    
    print("\n各波数性能:")
    print("-" * 70)
    print(f"{'k':>8} | {'Rel L2 (%)':>12} | {'Max Err':>12} | {'Mean Err':>12} | {'Time (ms)':>10}")
    print("-" * 70)
    
    for _, row in summary.iterrows():
        print(f"{row['k']:>8.0f} | {row['Relative L2 Error (%)']:>12.4f} | "
              f"{row['Max Error']:>12.6e} | {row['Mean Error']:>12.6e} | "
              f"{row['Inference Time (ms)']:>10.2f}")
    
    avg_error = summary['Relative L2 Error (%)'].mean()
    avg_time = summary['Inference Time (ms)'].mean()
    
    print("-" * 70)
    print(f"{'平均':>8} | {avg_error:>12.4f} | {'-':>12} | {'-':>12} | {avg_time:>10.2f}")
    print("-" * 70)
    
    print(f"\n生成的图表:")
    print(f"  - training_summary.png          (训练总结，4子图)")
    print(f"  - training_history_detailed.png (详细训练历史)")
    print(f"  - error_analysis.png            (误差分析，4子图)")
    print(f"  - combined_solutions.png        (组合解场对比)")
    print(f"  - solution_k*.png               (各波数解场，{len(summary)}张)")
    
    print(f"\n所有图表保存在: results/figures/")
    print("="*70 + "\n")

def main():
    """主函数"""
    # 加载数据
    data = load_data()
    
    # 生成各类图表
    print("="*70)
    print("开始生成图表...")
    print("="*70 + "\n")
    
    plot_training_summary(data)
    plot_training_history_detailed(data)
    plot_error_analysis(data)
    plot_combined_solutions(data)
    plot_solution_for_each_k(data)
    
    # 打印摘要
    print_summary(data)
    
    print(" 所有图表生成完成！")

if __name__ == "__main__":
    main()
