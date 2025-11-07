"""
子任务3：Poisson方程参数k(x,y)识别反问题 - PINN求解器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")



# 神经网络定义
class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, layers, activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class PINN_Task3(nn.Module):
    """双网络PINN"""
    def __init__(self, u_layers=[2, 64, 64, 64, 1], k_layers=[2, 32, 32, 32, 1]):
        super(PINN_Task3, self).__init__()
        
        self.u_net = MLP(u_layers, activation=nn.Tanh())
        self.k_net = MLP(k_layers, activation=nn.Tanh())
        
        u_params = sum(p.numel() for p in self.u_net.parameters())
        k_params = sum(p.numel() for p in self.k_net.parameters())
        print(f"u_net 参数量: {u_params:,}")
        print(f"k_net 参数量: {k_params:,}")
        print(f"总参数量: {u_params + k_params:,}")
    
    def forward_u(self, x, y):
        xy = torch.stack([x, y], dim=-1)
        u = self.u_net(xy)
        return u.squeeze()
    
    def forward_k(self, x, y):
        xy = torch.stack([x, y], dim=-1)
        k_raw = self.k_net(xy)
        k = nn.functional.softplus(k_raw.squeeze()) + 1e-6
        return k


# 物理方程
def compute_source_term(x, y):
    """计算源项f(x,y)"""
    pi = np.pi
    
    term1 = (pi**2 / 2) * (1 + x**2 + y**2) * torch.sin(pi*x/2) * torch.cos(pi*y/2)
    term2 = pi * x * torch.cos(pi*x/2) * torch.cos(pi*y/2)
    term3 = pi * y * torch.sin(pi*x/2) * torch.sin(pi*y/2)
    
    f = term1 - term2 + term3
    return f


def compute_pde_residual(model, x, y):
    """计算PDE残差"""
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    
    u = model.forward_u(x, y)
    k = model.forward_k(x, y)
    
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    
    ku_x = k * u_x
    ku_y = k * u_y
    
    div_ku_x = torch.autograd.grad(
        ku_x, x, grad_outputs=torch.ones_like(ku_x),
        create_graph=True, retain_graph=True)[0]
    
    div_ku_y = torch.autograd.grad(
        ku_y, y, grad_outputs=torch.ones_like(ku_y),
        create_graph=True, retain_graph=True)[0]
    
    div_k_grad_u = div_ku_x + div_ku_y
    f = compute_source_term(x, y)
    residual = -div_k_grad_u - f
    
    return residual



# 损失函数
def compute_loss(model, data_dict, weights):
    """计算总损失"""
    losses = {}
    
    # 数据拟合
    x_data = data_dict['x_data']
    y_data = data_dict['y_data']
    u_data = data_dict['u_data']
    
    u_pred = model.forward_u(x_data, y_data)
    L_data = torch.mean((u_pred - u_data)**2)
    losses['data'] = L_data.item()
    
    # PDE残差
    x_col = data_dict['x_col']
    y_col = data_dict['y_col']
    
    pde_residual = compute_pde_residual(model, x_col, y_col)
    L_pde = torch.mean(pde_residual**2)
    losses['pde'] = L_pde.item()
    
    # 边界条件
    x_bc = data_dict['x_bc']
    y_bc = data_dict['y_bc']
    
    u_bc = model.forward_u(x_bc, y_bc)
    L_bc = torch.mean(u_bc**2)
    losses['bc'] = L_bc.item()
    
    # k正则化
    k_data = model.forward_k(x_data, y_data)
    L_reg = torch.mean((k_data - 1.0)**2)
    losses['reg'] = L_reg.item()
    
    # k光滑性
    x_smooth = x_data.clone().detach().requires_grad_(True)
    y_smooth = y_data.clone().detach().requires_grad_(True)
    k_smooth = model.forward_k(x_smooth, y_smooth)
    
    k_x = torch.autograd.grad(
        k_smooth, x_smooth, grad_outputs=torch.ones_like(k_smooth),
        create_graph=True, retain_graph=True)[0]
    
    k_y = torch.autograd.grad(
        k_smooth, y_smooth, grad_outputs=torch.ones_like(k_smooth),
        create_graph=True, retain_graph=True)[0]
    
    L_smooth = torch.mean(k_x**2 + k_y**2)
    losses['smooth'] = L_smooth.item()
    
    # 总损失
    total_loss = (
        weights['data'] * L_data + 
        weights['pde'] * L_pde + 
        weights['bc'] * L_bc +
        weights['reg'] * L_reg +
        weights['smooth'] * L_smooth
    )
    
    losses['total'] = total_loss.item()
    
    return total_loss, losses


# 数据准备
def prepare_data(data_file, n_collocation=5000, n_boundary=400, device='cpu'):
    """准备训练数据"""
    print("\n" + "="*70)
    print("准备训练数据")
    print("="*70)
    
    df = pd.read_excel(data_file)
    x_train = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32, device=device)
    y_train = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32, device=device)
    u_train = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32, device=device)
    
    print(f"\n训练数据: {len(x_train)} 个点")
    print(f"  x 范围: [{x_train.min():.4f}, {x_train.max():.4f}]")
    print(f"  y 范围: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  u 范围: [{u_train.min():.4f}, {u_train.max():.4f}]")
    
    # 配点
    x_col = torch.rand(n_collocation, device=device) * 2 - 1
    y_col = torch.rand(n_collocation, device=device) * 2 - 1
    print(f"\n配点 (PDE残差): {n_collocation} 个")
    
    # 边界点
    n_per_edge = n_boundary // 4
    
    x_left = torch.ones(n_per_edge, device=device) * -1
    y_left = torch.rand(n_per_edge, device=device) * 2 - 1
    
    x_right = torch.ones(n_per_edge, device=device) * 1
    y_right = torch.rand(n_per_edge, device=device) * 2 - 1
    
    x_bottom = torch.rand(n_per_edge, device=device) * 2 - 1
    y_bottom = torch.ones(n_per_edge, device=device) * -1
    
    x_top = torch.rand(n_per_edge, device=device) * 2 - 1
    y_top = torch.ones(n_per_edge, device=device) * 1
    
    x_bc = torch.cat([x_left, x_right, x_bottom, x_top])
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top])
    print(f"边界点: {len(x_bc)} 个")
    
    data_dict = {
        'x_data': x_train,
        'y_data': y_train,
        'u_data': u_train,
        'x_col': x_col,
        'y_col': y_col,
        'x_bc': x_bc,
        'y_bc': y_bc
    }
    
    return data_dict


# 训练
def train_model(model, data_dict, config, save_dir):
    """训练PINN模型"""
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=200,  
        verbose=True,
        min_lr=1e-6
    )
    
    weights = config['loss_weights']
    
    history = {
        'epoch': [],
        'total_loss': [],
        'data_loss': [],
        'pde_loss': [],
        'bc_loss': [],
        'reg_loss': [],
        'smooth_loss': [],
        'lr': []
    }
    
    best_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    pbar = tqdm(range(config['epochs']), desc="训练进度")
    
    for epoch in pbar:
        model.train()
        
        optimizer.zero_grad()
        total_loss, losses = compute_loss(model, data_dict, weights)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        pbar.set_postfix({
            'loss': f'{losses["total"]:.2e}',
            'data': f'{losses["data"]:.2e}',
            'pde': f'{losses["pde"]:.2e}'
        })
        
        if epoch % config['log_interval'] == 0:
            history['epoch'].append(epoch)
            history['total_loss'].append(losses['total'])
            history['data_loss'].append(losses['data'])
            history['pde_loss'].append(losses['pde'])
            history['bc_loss'].append(losses['bc'])
            history['reg_loss'].append(losses['reg'])
            history['smooth_loss'].append(losses['smooth'])
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            if epoch % (config['log_interval'] * 10) == 0:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch}/{config['epochs']}")
                print(f"{'='*70}")
                print(f"Total Loss:  {losses['total']:.6e}")
                print(f"Data Loss:   {losses['data']:.6e}")
                print(f"PDE Loss:    {losses['pde']:.6e}")
                print(f"BC Loss:     {losses['bc']:.6e}")
                print(f"Reg Loss:    {losses['reg']:.6e}")
                print(f"Smooth Loss: {losses['smooth']:.6e}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, save_dir / 'best_model.pth')
        
        if epoch % config['save_interval'] == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}")
    print(f"总用时: {elapsed_time/60:.2f} 分钟")
    print(f"最佳损失: {best_loss:.6e} (Epoch {best_epoch})")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / 'training_history.csv', index=False)
    print(f"训练历史已保存: training_history.csv")
    
    return history



# 评估
def evaluate_model(model, data_dict, save_dir):
    """评估模型"""
    print("\n" + "="*70)
    print("模型评估")
    print("="*70)
    
    model.eval()
    
    with torch.no_grad():
        x_data = data_dict['x_data']
        y_data = data_dict['y_data']
        u_data = data_dict['u_data']
        
        u_pred = model.forward_u(x_data, y_data)
        k_pred = model.forward_k(x_data, y_data)
        
        u_error = torch.mean((u_pred - u_data)**2).item()
        u_rel_error = (torch.sqrt(torch.mean((u_pred - u_data)**2)) / 
                       torch.std(u_data)).item()
        
        print(f"\n数据拟合:")
        print(f"  MSE: {u_error:.6e}")
        print(f"  相对误差: {u_rel_error:.4%}")
        
        k_min = k_pred.min().item()
        k_max = k_pred.max().item()
        k_mean = k_pred.mean().item()
        k_std = k_pred.std().item()
        
        print(f"\n识别的k(x,y):")
        print(f"  范围: [{k_min:.4f}, {k_max:.4f}]")
        print(f"  均值: {k_mean:.4f}")
        print(f"  标准差: {k_std:.4f}")
    
    # PDE残差
    model.train()
    pde_residual = compute_pde_residual(model, x_data, y_data)
    f_values = compute_source_term(x_data, y_data)
    
    pde_error = torch.mean(pde_residual**2).item()
    pde_rel_error = (torch.sqrt(torch.mean(pde_residual**2)) / 
                     torch.std(f_values)).item()
    
    print(f"\nPDE残差:")
    print(f"  MSE: {pde_error:.6e}")
    print(f"  相对误差: {pde_rel_error:.4%}")
    
    eval_results = {
        'data_mse': u_error,
        'data_rel_error': u_rel_error,
        'pde_mse': pde_error,
        'pde_rel_error': pde_rel_error,
        'k_min': k_min,
        'k_max': k_max,
        'k_mean': k_mean,
        'k_std': k_std
    }
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\n评估结果已保存: evaluation_results.json")
    
    model.eval()
    return eval_results


def generate_predictions(model, save_dir, device='cpu', grid_size=100):
    """生成网格预测"""
    print(f"\n生成网格预测 ({grid_size}x{grid_size})...")
    
    model.eval()
    
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred = model.forward_u(X_flat, Y_flat).cpu().numpy()
        k_pred = model.forward_k(X_flat, Y_flat).cpu().numpy()
    
    u_pred = u_pred.reshape(grid_size, grid_size)
    k_pred = k_pred.reshape(grid_size, grid_size)
    
    predictions = {
        'x': x.tolist(),
        'y': y.tolist(),
        'u_pred': u_pred.tolist(),
        'k_pred': k_pred.tolist()
    }
    
    with open(save_dir / 'grid_predictions.json', 'w') as f:
        json.dump(predictions, f)
    
    print(f"网格预测已保存: grid_predictions.json")
    
    return predictions


def save_training_predictions(model, data_dict, save_dir):
    """保存训练数据预测 - 修复版"""
    print("\n保存训练数据预测...")
    
    model.eval()
    
    with torch.no_grad():
        x = data_dict['x_data'].cpu().numpy()
        y = data_dict['y_data'].cpu().numpy()
        u_true = data_dict['u_data'].cpu().numpy()
        
        u_pred = model.forward_u(data_dict['x_data'], data_dict['y_data']).cpu().numpy()
        k_pred = model.forward_k(data_dict['x_data'], data_dict['y_data']).cpu().numpy()
        
        # 修复: 在no_grad内计算f
        f = compute_source_term(data_dict['x_data'], data_dict['y_data']).cpu().numpy()
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'u_true': u_true,
        'u_pred': u_pred,
        'k_pred': k_pred,
        'f': f,
        'u_error': np.abs(u_pred - u_true)
    })
    
    df.to_csv(save_dir / 'training_data_predictions.csv', index=False)
    print("训练数据预测已保存: training_data_predictions.csv")



# 主函数
def main():
    """主函数"""
    
    # 配置参数
    config = {
        'data_file': '子任务3数据.xlsx',
        'save_dir': 'results',
        
        # 网络结构
        'u_layers': [2, 64, 64, 64, 1],
        'k_layers': [2, 32, 32, 32, 1],
        
        # 训练
        'epochs': 20000,
        'lr': 1e-3,
        'log_interval': 100,
        'save_interval': 2000,
        
        # 采样
        'n_collocation': 5000,
        'n_boundary': 400,
        
        # 损失权重 - 优化建议
        'loss_weights': {
            'data': 1.0,       # 数据拟合
            'pde': 0.1,        # 修改: 从0.01增加到0.1，加强物理约束
            'bc': 0.1,         # 边界条件
            'reg': 1e-4,       # 修改: 从1e-5增加到1e-4，防止k过大
            'smooth': 1e-4     # 修改: 从1e-5增加到1e-4，增加光滑性
        },
        
        'grid_size': 100
    }
    
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*70)
    print("子任务3：Poisson方程参数识别反问题")
    print("="*70)
    print(f"保存目录: {save_dir}")
    
    # 准备数据
    data_dict = prepare_data(
        config['data_file'],
        n_collocation=config['n_collocation'],
        n_boundary=config['n_boundary'],
        device=DEVICE
    )
    
    # 创建模型
    print("\n" + "="*70)
    print("创建PINN模型")
    print("="*70)
    
    model = PINN_Task3(
        u_layers=config['u_layers'],
        k_layers=config['k_layers']
    ).to(DEVICE)
    
    # 训练
    history = train_model(model, data_dict, config, save_dir)
    
    # 加载最佳模型
    print("\n加载最佳模型...")
    checkpoint = torch.load(save_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳模型: Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.6e}")
    
    # 评估
    eval_results = evaluate_model(model, data_dict, save_dir)
    
    # 生成预测
    predictions = generate_predictions(model, save_dir, DEVICE, config['grid_size'])
    save_training_predictions(model, data_dict, save_dir)
    
    print("\n" + "="*70)
    print("任务完成！")
    print("="*70)
    print(f"所有结果已保存到: {save_dir}")
    print("\n生成的文件:")
    for file in sorted(save_dir.glob('*')):
        print(f"  - {file.name}")
    print("\n下一步: 运行 python visualize.py 生成图表")
    print("="*70)


if __name__ == '__main__':
    main()
