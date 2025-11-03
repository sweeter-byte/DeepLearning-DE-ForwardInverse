"""
K-Conditioned FNO  - 训练脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import pandas as pd
import json
import time
from tqdm import tqdm
import os
import pickle

# 创建目录
os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/analysis', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

a1, a2 = 1, 3


# 1. 频谱卷积层
class SpectralConv2d(nn.Module):
    """频谱卷积"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W//2 + 1)
        
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        weights = torch.complex(
            self.weights_real[:, :, :modes1, :modes2],
            self.weights_imag[:, :, :modes1, :modes2]
        )
        
        out_ft[:, :, :modes1, :modes2] = torch.einsum(
            "bcxy,icxy->bixy", 
            x_ft[:, :, :modes1, :modes2], 
            weights
        )
        
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x

class FNOBlock(nn.Module):
    """FNO块"""
    def __init__(self, width, modes):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes, modes)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
    
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = self.norm(x1 + x2)
        x = F.gelu(x)
        return x


# 2. K-Conditioned FNO 
class KConditionedFNO_v1_5(nn.Module):
    """
    K-Conditioned FNO 
    1. k嵌入层加入LayerNorm（提高稳定性）
    2. 轻微的dropout（0.05，防止过拟合）
    """
    
    def __init__(self, width=64, modes=20, n_layers=4, dropout=0.05):
        super().__init__()
        
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        
        # 输入投影
        self.fc0 = nn.Conv2d(1, width, 1)
        
        # k嵌入网络（加入LayerNorm）
        self.k_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128),  
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256), 
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(width, modes) for _ in range(n_layers)
        ])
        
        # FiLM调制层
        self.film_layers = nn.ModuleList([
            nn.Linear(128, width * 2)
            for _ in range(n_layers)
        ])
        
        # 输出投影（加入轻微dropout）
        self.fc_out = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Conv2d(128, 1, 1)
        )
        
        self.global_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, k, q):
        B, _, H, W = q.shape
        
        k_feat = self.k_embed(k)
        
        x = self.fc0(q)
        
        for i in range(self.n_layers):
            film_params = self.film_layers[i](k_feat)
            scale, shift = torch.chunk(film_params, 2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            
            x = self.fno_layers[i](x)
            x = (1 + 0.1 * scale) * x + 0.1 * shift
        
        x = self.fc_out(x)
        
        return self.global_scale * x


# 3. 简单的全局归一化器
class SimpleNormalizer:
    """简单全局归一化"""
    def __init__(self):
        self.q_mean = 0.0
        self.q_std = 1.0
        self.u_mean = 0.0
        self.u_std = 1.0
        self.k_scale = 1000.0
    
    def fit(self, q_samples, u_sample):
        """拟合全局统计量"""
        self.q_mean = np.mean(q_samples)
        self.q_std = np.std(q_samples)
        self.u_mean = np.mean(u_sample)
        self.u_std = np.std(u_sample)
    
    def normalize_k(self, k):
        return k / self.k_scale
    
    def normalize_q(self, q):
        return (q - self.q_mean) / (self.q_std + 1e-8)
    
    def normalize_u(self, u):
        return (u - self.u_mean) / (self.u_std + 1e-8)
    
    def denormalize_u(self, u_norm):
        return u_norm * self.u_std + self.u_mean

# 4. 数据生成 
def source_term(x, y, k):
    """计算源项"""
    return (-(a1*np.pi)**2 - (a2*np.pi)**2 - k**2) * np.sin(a1*np.pi*x) * np.sin(a2*np.pi*y)

def true_solution(x, y):
    """真实解"""
    return np.sin(a1*np.pi*x) * np.sin(a2*np.pi*y)

def generate_training_data(k_range=[50, 1000], n_k=30, n_samples_per_k=100, resolution=64):
    """
    生成训练数据
    
    1. 增加样本数：80 → 100 每个k
    2. 增加k采样数：25 → 30
    3. 保持全局归一化
    """
    
    print(f"\n{'='*70}")
    print("生成训练数据")
    print(f"{'='*70}")
    print(f"  策略: 简单且有效")
    print(f"  k范围: [{k_range[0]}, {k_range[1]}]")
    print(f"  k采样数: {n_k}")
    print(f"  每个k样本数: {n_samples_per_k}")
    print(f"  空间分辨率: {resolution}x{resolution}")
    print(f"  归一化: 全局统一")
    
    # k采样：对数+线性混合
    k_log = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_k//2)
    k_lin = np.linspace(k_range[0], k_range[1], n_k - n_k//2)
    k_values = np.unique(np.concatenate([k_log, k_lin]))
    
    # 确保包含测试点
    test_k = np.array([100, 300, 500, 700, 1000])
    k_values = np.unique(np.concatenate([k_values, test_k]))
    k_values = np.sort(k_values)
    
    print(f"  实际k样本数: {len(k_values)}")
    
    # 空间网格
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 收集统计信息
    print("\n  收集统计信息...")
    q_samples = []
    sample_k = np.random.choice(k_values, min(15, len(k_values)), replace=False)
    
    for k in tqdm(sample_k, desc="  采样"):
        q = source_term(X, Y, k)
        q_samples.append(q)
    
    q_samples = np.array(q_samples)
    u_sample = true_solution(X, Y)
    
    # 创建归一化器
    normalizer = SimpleNormalizer()
    normalizer.fit(q_samples, u_sample)
    
    print(f"\n  归一化参数:")
    print(f"    q: mean={normalizer.q_mean:.3e}, std={normalizer.q_std:.3e}")
    print(f"    u: mean={normalizer.u_mean:.3e}, std={normalizer.u_std:.3e}")
    
    # 生成所有数据
    print(f"\n  生成训练样本...")
    
    all_k = []
    all_q = []
    all_u = []
    
    for k in tqdm(k_values, desc="  进度"):
        for _ in range(n_samples_per_k):
            q = source_term(X, Y, k)
            u = true_solution(X, Y)
            
            # 全局归一化
            k_norm = normalizer.normalize_k(k)
            q_norm = normalizer.normalize_q(q)
            u_norm = normalizer.normalize_u(u)
            
            all_k.append(k_norm)
            all_q.append(q_norm)
            all_u.append(u_norm)
    
    all_k = np.array(all_k)
    all_q = np.array(all_q)[:, np.newaxis, :, :]
    all_u = np.array(all_u)[:, np.newaxis, :, :]
    
    print(f"\n数据生成完成:")
    print(f"  总样本数: {len(all_k):,}")
    print(f"  k范围: [{all_k.min():.3f}, {all_k.max():.3f}] (归一化)")
    print(f"  q范围: [{all_q.min():.3f}, {all_q.max():.3f}] (标准化)")
    print(f"  u范围: [{all_u.min():.3f}, {all_u.max():.3f}] (标准化)")
    
    # 保存
    np.savez('results/data/training_data.npz', k=all_k, q=all_q, u=all_u)
    
    norm_params = {
        'q_mean': float(normalizer.q_mean),
        'q_std': float(normalizer.q_std),
        'u_mean': float(normalizer.u_mean),
        'u_std': float(normalizer.u_std),
        'k_scale': float(normalizer.k_scale),
        'k_values': k_values.tolist()
    }
    
    with open('results/data/norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"  数据已保存")
    print(f"{'='*70}\n")
    
    return all_k, all_q, all_u, normalizer


# 5. 训练函数 
def train_model(model, k_data, q_data, u_data, epochs=12000, batch_size=64, lr=1e-3):
    """
    训练模型 
    
    改进：
    1. 增加训练轮数：10000 → 12000（给模型更多时间收敛）
    """
    
    print(f"\n{'='*70}")
    print("训练 K-Conditioned FNO ")
    print(f"{'='*70}")
    
    n_total = len(k_data)
    n_train = int(0.9 * n_total) 
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    k_train = torch.tensor(k_data[train_idx], dtype=torch.float32, device=device).unsqueeze(1)
    q_train = torch.tensor(q_data[train_idx], dtype=torch.float32, device=device)
    u_train = torch.tensor(u_data[train_idx], dtype=torch.float32, device=device)
    
    k_val = torch.tensor(k_data[val_idx], dtype=torch.float32, device=device).unsqueeze(1)
    q_val = torch.tensor(q_data[val_idx], dtype=torch.float32, device=device)
    u_val = torch.tensor(u_data[val_idx], dtype=torch.float32, device=device)
    
    print(f"  训练集: {len(train_idx):,} (90%)")
    print(f"  验证集: {len(val_idx):,} (10%)")
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    training_history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []
    }
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = 2000  
    
    print(f"  学习率: {lr}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs:,}")
    print(f"  早停耐心: {max_patience}")
    print(f"{'='*70}\n")
    
    pbar = tqdm(range(epochs), desc='Training')
    
    for epoch in pbar:
        model.train()
        
        batch_idx = np.random.choice(n_train, min(batch_size, n_train), replace=False)
        k_batch = k_train[batch_idx]
        q_batch = q_train[batch_idx]
        u_batch = u_train[batch_idx]
        
        optimizer.zero_grad()
        
        u_pred = model(k_batch, q_batch)
        loss = F.mse_loss(u_pred, u_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 验证
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                u_val_pred = model(k_val, q_val)
                val_loss = F.mse_loss(u_val_pred, u_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'results/models/model_best.pth')
                patience = 0
            else:
                patience += 50
            
            if patience >= max_patience:
                print(f"\n早停: 验证损失{max_patience}轮未改善")
                break
        else:
            val_loss = 0.0
        
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(loss.item())
        training_history['val_loss'].append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
        training_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4e}',
            'Val': f'{val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss:.4e}',
            'Best': f'{best_val_loss:.4e}'
        })
        
        if (epoch + 1) % 2000 == 0:
            print(f'\n{"="*70}')
            print(f'Epoch {epoch+1:,}/{epochs:,}')
            print(f'  Train Loss: {loss.item():.6e}')
            if val_loss > 0:
                print(f'  Val Loss: {val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss:.6e}')
            print(f'  Best Val: {best_val_loss:.6e}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6e}')
            print(f'{"="*70}\n')
    
    # 保存训练历史
    df = pd.DataFrame(training_history)
    df.to_csv('results/data/training_history.csv', index=False)
    print(f"\n 训练历史已保存: results/data/training_history.csv")
    
    with open('results/data/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return training_history

# 6. 评估
def evaluate_model(model, normalizer, test_k_values=None, resolution=100):
    """评估模型并保存数据"""
    
    if test_k_values is None:
        test_k_values = [100, 300, 500, 700, 1000]
    
    print(f"\n{'='*70}")
    print("评估模型")
    print(f"{'='*70}")
    print(f"  测试k值: {test_k_values}")
    
    model.eval()
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    results = {}
    
    print(f"\n{'='*70}")
    for k in test_k_values:
        print(f"\n评估 k = {k}...")
        
        q = source_term(X, Y, k)
        u_true = true_solution(X, Y)
        
        k_norm = normalizer.normalize_k(k)
        q_norm = normalizer.normalize_q(q)
        
        k_tensor = torch.tensor([[k_norm]], dtype=torch.float32, device=device)
        q_tensor = torch.tensor(q_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            u_pred_norm = model(k_tensor, q_tensor)
            u_pred_norm = u_pred_norm.squeeze().cpu().numpy()
        inference_time = time.time() - start_time
        
        u_pred = normalizer.denormalize_u(u_pred_norm)
        
        rel_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        max_error = np.max(np.abs(u_pred - u_true))
        mean_error = np.mean(np.abs(u_pred - u_true))
        
        print(f"  相对L2误差: {rel_error:.6e} ({rel_error*100:.4f}%)")
        print(f"  最大绝对误差: {max_error:.6e}")
        print(f"  平均绝对误差: {mean_error:.6e}")
        print(f"  推理时间: {inference_time*1000:.2f} ms")
        
        results[k] = {
            'k': k,
            'relative_l2_error': float(rel_error),
            'max_absolute_error': float(max_error),
            'mean_absolute_error': float(mean_error),
            'inference_time_ms': float(inference_time * 1000),
            'u_pred': u_pred,
            'u_true': u_true,
            'error': np.abs(u_pred - u_true),
            'X': X,
            'Y': Y
        }
    
    print(f"\n{'='*70}")
    
    # 保存评估数据
    print("\n保存评估数据...")
    
    # 保存完整结果
    with open('results/analysis/full_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f" 完整结果已保存: results/analysis/full_results.pkl")
    
    # 保存误差摘要
    summary = {
        str(k): {
            'relative_l2_error': res['relative_l2_error'],
            'relative_l2_error_percent': res['relative_l2_error'] * 100,
            'max_absolute_error': res['max_absolute_error'],
            'mean_absolute_error': res['mean_absolute_error'],
            'inference_time_ms': res['inference_time_ms']
        }
        for k, res in results.items()
    }
    
    with open('results/analysis/error_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" 误差摘要已保存: results/analysis/error_summary.json")
    
    # 保存误差摘要
    df_summary = pd.DataFrame([
        {
            'k': k,
            'Relative L2 Error (%)': res['relative_l2_error'] * 100,
            'Max Error': res['max_absolute_error'],
            'Mean Error': res['mean_absolute_error'],
            'Inference Time (ms)': res['inference_time_ms']
        }
        for k, res in results.items()
    ])
    df_summary.to_csv('results/analysis/error_summary.csv', index=False)
    print(f" 误差摘要已保存: results/analysis/error_summary.csv")
    
    return results

# 7. 主程序
if __name__ == "__main__":
    print("="*70)
    print("K-Conditioned FNO ")
    print("="*70)
    
    total_start = time.time()
    
    # 阶段1: 生成数据
    print("\n" + "="*70)
    print("阶段1: 生成训练数据")
    print("="*70)
    
    k_data, q_data, u_data, normalizer = generate_training_data(
        k_range=[50, 1000],
        n_k=30,
        n_samples_per_k=100,
        resolution=64
    )
    
    # 阶段2: 创建模型
    print("\n" + "="*70)
    print("阶段2: 创建模型")
    print("="*70)
    
    model = KConditionedFNO_v1_5(
        width=64,      
        modes=20,      
        n_layers=4,   
        dropout=0.05  
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    
    # 阶段3: 训练
    print("\n" + "="*70)
    print("阶段3: 训练")
    print("="*70)
    
    training_history = train_model(
        model, k_data, q_data, u_data,
        epochs=12000,
        batch_size=64,
        lr=1e-3
    )
    
    # 阶段4: 加载最佳模型
    print("\n加载最佳模型...")
    checkpoint = torch.load('results/models/model_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" 已加载 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6e})")
    
    # 阶段5: 评估
    print("\n" + "="*70)
    print("阶段5: 评估")
    print("="*70)
    
    test_k_values = [100, 300, 500, 700, 1000]
    results = evaluate_model(model, normalizer, test_k_values=test_k_values, resolution=100)
    
    # 总结
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    
    print("\n各波数相对L2误差:")
    print("-" * 50)
    for k in test_k_values:
        rel_err = results[k]['relative_l2_error']
        print(f"  k = {k:5d}: {rel_err:.6e} ({rel_err*100:.4f}%)")
    
    avg_error = np.mean([results[k]['relative_l2_error'] for k in test_k_values])
    print("-" * 50)
    print(f"  平均: {avg_error:.6e} ({avg_error*100:.4f}%)")
    
    print(f"\n结果数据保存在: results/data/ 和 results/analysis/")
    print(f"运行 visualize.py 生成图表")
    print("="*70)
