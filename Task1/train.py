"""
Helmholtz方程求解器 - 训练脚本
1. 训练和测试使用相同分辨率
2. 更简单但更鲁棒的架构
3. 正则化防止过拟合
4. 更多数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from torch.optim import Adam
import os

os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

a1, a2, k = 1, 3, 4

class SimpleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class RobustHelmholtzNet(nn.Module):
    """
    UNet - 适合小数据集
    关键: 防止过拟合
    """
    def __init__(self):
        super().__init__()
        
        # 编码器 - 更少的层以防止过拟合
        self.enc1 = SimpleConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = SimpleConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = SimpleConvBlock(64, 128)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = SimpleConvBlock(128, 64)  # 64 + 64
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = SimpleConvBlock(64, 32)   # 32 + 32
        
        # 输出
        self.out = nn.Conv2d(32, 1, 1)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, q, x_coords, y_coords):
        # 输入
        x = torch.cat([q, x_coords, y_coords], dim=1)
        
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e3 = self.dropout(e3)
        
        # 解码
        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        u = self.out(d2)
        
        # 硬约束边界条件
        u = self.apply_boundary_condition(u, x_coords, y_coords)
        
        return u
    
    def apply_boundary_condition(self, u, x_coords, y_coords):
        """强制边界条件 u=0"""
        # 计算到边界的距离
        dist_x = torch.min(1 + x_coords, 1 - x_coords)
        dist_y = torch.min(1 + y_coords, 1 - y_coords)
        dist = torch.min(dist_x, dist_y)
        
        # 平滑掩码
        mask = torch.tanh(10 * dist)  # 使用tanh更平滑
        
        return u * mask


# 数据生成
def source_term(x, y):
    return (-(a1*np.pi)**2 - (a2*np.pi)**2 - k**2) * np.sin(a1*np.pi*x) * np.sin(a2*np.pi*y)

def true_solution(x, y):
    return np.sin(a1*np.pi*x) * np.sin(a2*np.pi*y)

def generate_diverse_samples(n_samples=500, resolution=128):
    """
    生成多样化训练数据
    关键: 使用中等分辨率(128)作为训练和测试的统一分辨率
    """
    print(f"生成训练数据...")
    print(f"  样本数: {n_samples}")
    print(f"  分辨率: {resolution}×{resolution} (训练和测试统一)")
    
    all_q, all_u, all_x, all_y = [], [], [], []
    
    for i in range(n_samples):
        # 数据增强: 随机扰动网格
        noise_scale = 0.02
        x_offset = np.random.uniform(-noise_scale, noise_scale)
        y_offset = np.random.uniform(-noise_scale, noise_scale)
        
        x = np.linspace(-1 + x_offset, 1 + x_offset, resolution)
        y = np.linspace(-1 + y_offset, 1 + y_offset, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 计算
        q = source_term(X, Y)
        u = true_solution(X, Y)
        
        # 添加小噪声增强鲁棒性
        if i > n_samples // 2:  # 后半部分加噪声
            q += np.random.randn(*q.shape) * 0.01 * np.abs(q).max()
        
        all_q.append(q)
        all_u.append(u)
        all_x.append(X)
        all_y.append(Y)
    
    all_q = np.array(all_q)[:, np.newaxis, :, :]
    all_u = np.array(all_u)[:, np.newaxis, :, :]
    all_x = np.array(all_x)[:, np.newaxis, :, :]
    all_y = np.array(all_y)[:, np.newaxis, :, :]
    
    print(f"  生成完成: q={all_q.shape}")
    print(f"  q范围: [{all_q.min():.2f}, {all_q.max():.2f}]")
    print(f"  u范围: [{all_u.min():.3f}, {all_u.max():.3f}]")
    
    return all_q, all_u, all_x, all_y


# 训练
def train_model(model, train_loader, val_loader, epochs=8000, lr=2e-4):
    """训练模型"""
    print(f"\n{'='*80}")
    print(" "*30 + "开始训练")
    print(f"{'='*80}\n")
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 500
    min_improvement = 0.001
    
    print(f"训练配置:")
    print(f"  训练batch数: {len(train_loader)}")
    print(f"  验证batch数: {len(val_loader)}")
    print(f"  初始学习率: {lr}")
    print(f"  最大轮数: {epochs}")
    print(f"  早停耐心: {max_patience}轮")
    print(f"  最小改进: {min_improvement*100}%")  
    print(f"{'-'*80}\n")
    
    pbar = tqdm(range(epochs), desc='Training')
    
    for epoch in pbar:
        # 训练阶段
        model.train()
        train_losses = []
        
        for q_batch, u_batch, x_batch, y_batch in train_loader:
            q_batch = q_batch.to(device)
            u_batch = u_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            u_pred = model(q_batch, x_batch, y_batch)
            loss = F.mse_loss(u_pred, u_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # 验证阶段
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for q_batch, u_batch, x_batch, y_batch in val_loader:
                q_batch = q_batch.to(device)
                u_batch = u_batch.to(device)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                u_pred = model(q_batch, x_batch, y_batch)
                loss = F.mse_loss(u_pred, u_batch)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 改进的早停逻辑
        if val_loss < best_val_loss:
            # 计算相对改进
            relative_improvement = (best_val_loss - val_loss) / best_val_loss
            
            if relative_improvement >= min_improvement:
                # 改进足够大，重置计数器
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'results/models/best_model.pth')
            else:
                # 改进太小，增加计数器
                patience_counter += 1
                # 但仍然更新最佳损失
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'results/models/best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"\n早停触发:")
            print(f"  原因: {max_patience}轮内改进 < {min_improvement*100}%")
            print(f"  最佳验证损失: {best_val_loss:.6e}")
            break
        
        # 更新进度条
        pbar.set_postfix({
            'TrLoss': f'{train_loss:.6e}',
            'VaLoss': f'{val_loss:.6e}',
            'Best': f'{best_val_loss:.6e}',
            'Pat': f'{patience_counter}/{max_patience}'  # 显示耐心计数
        })
        
        # 定期打印
        if (epoch + 1) % 500 == 0:
            print(f'\n{"="*80}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'  Train Loss: {train_loss:.6e}')
            print(f'  Val Loss:   {val_loss:.6e}')
            print(f'  Best Val:   {best_val_loss:.6e}')
            print(f'  LR:         {optimizer.param_groups[0]["lr"]:.6e}')
            print(f'  Patience:   {patience_counter}/{max_patience}')
            print(f'{"="*80}\n')
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6e}")
    
    # 保存训练历史为CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv('results/data/training_history.csv', index=False)
    print(f" 训练历史已保存: results/data/training_history.csv")
    
    return history

# 评估
def evaluate_model(model, resolution=128):
    """评估模型并保存结果数据"""
    print(f"\n{'='*80}")
    print(" "*30 + "模型评估")
    print(f"{'='*80}\n")
    
    model.eval()
    
    # 使用相同分辨率测试
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    q_test = source_term(X, Y)
    u_true = true_solution(X, Y)
    
    # 准备输入
    q_t = torch.tensor(q_test, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    x_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        u_pred = model(q_t, x_t, y_t).squeeze().cpu().numpy()
    
    # 误差
    error = np.abs(u_pred - u_true)
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(u_true)
    max_err = np.max(error)
    mean_err = np.mean(error)
    
    # 边界误差
    bc_err = np.mean([
        np.mean(np.abs(u_pred[0, :])),
        np.mean(np.abs(u_pred[-1, :])),
        np.mean(np.abs(u_pred[:, 0])),
        np.mean(np.abs(u_pred[:, -1]))
    ])
    
    print(f"误差统计:")
    print(f"  相对L2误差: {rel_l2:.6e} ({rel_l2*100:.4f}%)")
    print(f"  最大误差:   {max_err:.6e}")
    print(f"  平均误差:   {mean_err:.6e}")
    print(f"  边界误差:   {bc_err:.6e}")
    print(f"{'='*80}\n")
    
    # 保存评估结果数据
    # 1. 保存网格数据和结果
    np.savez('results/data/evaluation_results.npz',
             X=X, Y=Y,
             u_pred=u_pred,
             u_true=u_true,
             error=error,
             q_test=q_test)
    print(f"评估数据已保存: results/data/evaluation_results.npz")
    
    # 2. 保存误差统计
    error_stats = {
        'relative_l2_error': rel_l2,
        'relative_l2_error_percent': rel_l2 * 100,
        'max_error': max_err,
        'mean_error': mean_err,
        'boundary_error': bc_err,
        'resolution': resolution
    }
    df_stats = pd.DataFrame([error_stats])
    df_stats.to_csv('results/data/error_statistics.csv', index=False)
    print(f"误差统计已保存: results/data/error_statistics.csv")
    
    # 3. 保存截面数据（用于绘图）
    mid = resolution // 2
    cross_section_data = {
        'x_coord': X[mid, :],
        'u_true_y0': u_true[mid, :],
        'u_pred_y0': u_pred[mid, :],
        'y_coord': Y[:, mid],
        'u_true_x0': u_true[:, mid],
        'u_pred_x0': u_pred[:, mid]
    }
    df_cross = pd.DataFrame({
        'x_coord': X[mid, :],
        'u_true_at_y0': u_true[mid, :],
        'u_pred_at_y0': u_pred[mid, :]
    })
    df_cross.to_csv('results/data/cross_section_y0.csv', index=False)
    
    df_cross2 = pd.DataFrame({
        'y_coord': Y[:, mid],
        'u_true_at_x0': u_true[:, mid],
        'u_pred_at_x0': u_pred[:, mid]
    })
    df_cross2.to_csv('results/data/cross_section_x0.csv', index=False)
    print(f"✓ 截面数据已保存: results/data/cross_section_*.csv")
    
    return {'rel_l2': rel_l2, 'max_err': max_err, 'mean_err': mean_err, 'bc_err': bc_err}


# 主程序
def main():
    print("="*80)
    print(" "*15 + "Helmholtz方程求解器 - 训练脚本")
    print("="*80)
    print(f"\n问题: -Δu - {k}²u = q, u=0 on ∂Ω")
    print(f"参数: a₁={a1}, a₂={a2}, k={k}")
    print(f"真解: u = sin({a1}πx)sin({a2}πy)")
    print(f"\n{'='*80}\n")
    
    start_time = time.time()
    
    # 1. 生成数据 (使用128分辨率)
    RESOLUTION = 128
    q_data, u_data, x_data, y_data = generate_diverse_samples(n_samples=500, resolution=RESOLUTION)
    
    # 2. 划分训练/验证集
    n_total = len(q_data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # 创建DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.tensor(q_data[train_idx], dtype=torch.float32),
        torch.tensor(u_data[train_idx], dtype=torch.float32),
        torch.tensor(x_data[train_idx], dtype=torch.float32),
        torch.tensor(y_data[train_idx], dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(q_data[val_idx], dtype=torch.float32),
        torch.tensor(u_data[val_idx], dtype=torch.float32),
        torch.tensor(x_data[val_idx], dtype=torch.float32),
        torch.tensor(y_data[val_idx], dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"数据划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
    # 3. 创建模型
    model = RobustHelmholtzNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {n_params:,}\n")
    
    # 4. 训练
    history = train_model(model, train_loader, val_loader, epochs=8000, lr=2e-4)
    
    # 5. 加载最佳模型
    print("\n加载最佳模型...")
    model.load_state_dict(torch.load('results/models/best_model.pth', weights_only=True))
    
    # 6. 评估 (使用相同分辨率)
    results = evaluate_model(model, resolution=RESOLUTION)
    
    # 7. 总结
    print(f"\n{'='*80}")
    print(" "*30 + "训练完成")
    print(f"{'='*80}")
    print(f"\n总用时: {(time.time()-start_time)/60:.2f} 分钟")
    print(f"\n最终结果:")
    print(f"  相对L2误差: {results['rel_l2']:.6e} ({results['rel_l2']*100:.3f}%)")
    print(f"  最大误差:   {results['max_err']:.6e}")
    print(f"  平均误差:   {results['mean_err']:.6e}")
    print(f"  边界误差:   {results['bc_err']:.6e}")
    
    if results['rel_l2'] < 0.01:
        print(f"\n✓ 优秀! L2误差 < 1%")
    elif results['rel_l2'] < 0.05:
        print(f"\n✓ 良好! L2误差 < 5%")
    else:
        print(f"\n○ 可接受")
    
    print(f"\n结果数据保存在: results/data/")
    print(f"运行 visualize_results.py 生成图表")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
