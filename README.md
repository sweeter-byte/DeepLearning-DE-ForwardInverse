# 基于物理信息神经网络（PINN）的偏微分方程求解竞赛报告

## 摘要

本文档记录了本人参加的一项基于**物理信息神经网络（Physics-Informed Neural Networks, PINN）**的偏微分方程求解竞赛的全过程。比赛包含三个子任务：

1. **子任务1**：求解固定波数（k=4）的亥姆霍兹（Helmholtz）方程正问题；
2. **子任务2**：求解高波数（k=100, 500, 1000）的亥姆霍兹方程，考察模型对高频解的捕捉能力；
3. **子任务3**：泊松（Poisson）方程参数识别反问题，从200个采样点数据中反演系数场 k(x,y)。

所有代码均使用 **PyTorch** 实现，文件名分别为：
- `task1_train.py`
- `task2_train.py`
- `task3_train.py`

本报告按子任务逐一详细说明：**训练数据来源、模型结构、训练策略、损失函数组成、数据流向、评估方法**等内容。

---

## 子任务1：亥姆霍兹方程正问题求解（k=4）

### 训练数据来源

本任务为**无数据正问题**，完全依赖物理约束训练，无需观测数据。训练点在每个epoch动态采样：

| 类型 | 数量 | 采样方式 | 说明 |
|------|------|----------|------|
| 内部点 | 8000 | `torch.rand()` 均匀采样 | 域 Ω = [-1,1]×[-1,1]，`requires_grad=True` |
| 边界点 | 2000（每边500） | 均匀随机分布于四条边 | 满足 Dirichlet 边界条件 u=0 |

源项 q(x,y) 由解析式给定：

$$
q(x,y) = -(a_1\pi)^2 \sin(a_1\pi x)\sin(a_2\pi y) - (a_2\pi)^2 \sin(a_1\pi x)\sin(a_2\pi y) - k^2 \sin(a_1\pi x)\sin(a_2\pi y)
$$

其中 $ a_1=1, a_2=3, k=4 $。

### 模型结构

采用 **SIREN**（Sinusoidal Representation Networks）架构，适合高频函数拟合：

- **输入**：(x, y) ∈ ℝ²
- **网络层**：[2, 100, 100, 100, 100, 1]
- **激活函数**：`torch.sin`
- **初始化**：
    - 第一层：$\omega = 1.0$，均匀初始化界限 $1 / \text{fan\_in}$
    - 后续层：$\omega_0 = 30.0$，界限 $\sqrt{6}/\text{fan\_in}/\omega_0$
    - 偏置全为0
- **输出**：$u(x,y)$

**SIREN（Sinusoidal Representation Networks）** 的**标准权重初始化方法**，来自论文：
*Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020*

模型支持 GPU 加速。

### 训练策略

| 项目 | 设置 |
|------|------|
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | `CosineAnnealingWarmRestarts(T_0=10000, T_mult=2, eta_min=1e-6)` |
| 训练轮数 | 30,000 |
| 梯度裁剪 | `max_norm=0.5` |
| 模型保存 | 每轮保存最低 loss 模型 |
| 日志记录 | 每2000轮打印 loss；每100轮记录历史 |

训练时间7分钟（GPU）。

### 损失函数组成

$$
\mathcal{L} = \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}
$$

- **PDE 损失**（内部点）：
  $$
  \mathcal{L}_{\text{pde}} = \frac{1}{N_i} \sum \left( -\Delta u - k^2 u - q(x,y) \right)^2
  $$
  使用 `torch.autograd` 计算二阶导数。

- **边界损失**（边界点）：
  $$
  \mathcal{L}_{\text{bc}} = \frac{1}{N_b} \sum u(x,y)^2, \quad (x,y)\in \partial\Omega
  $$

- **权重**：$λ_{pde}$ = 1.0，$λ_{bc}$ = 100.0（强约束边界）

### 评估方法

- **评估网格**：100×100 均匀网格
- **指标**：相对 L2 误差
  $$
  \text{Rel L2} = \sqrt{\frac{\sum_{i=1}^{N} (u_{\text{pred}}^{(i)}-u_{\text{true}}^{(i)})^2}{\sum_{i=1}^{N} (u_{\text{true}}^{(i)})^2}}
  $$
- **真解**：
  $$
  u_{\text{true}}(x,y) = c \cdot \sin(\pi x)\sin(3\pi y), \quad c = -\frac{\lambda + k^2}{\lambda - k^2}
  $$
  其中 $ \lambda = (1^2 + 3^2)\pi^2 = 10\pi^2 $

---

## 子任务2：高波数亥姆霍兹方程求解（k=100, 500, 1000）

### 训练数据来源

与子任务1完全一致：
- 内部点：8000
- 边界点：2000（每边500）
- 源项 q(x,y) 相同，**k 为变量**

**对每个 k 值独立训练一个模型**，共三个模型。

### 模型结构

**完全复用子任务1的 SIREN 网络**，无需修改。

### 训练策略

| 项目 | 设置 |
|------|------|
| 优化器 | Adam (lr=0.001) |
| 调度器 | CosineAnnealingWarmRestarts |
| 每 k 训练轮数 | 30,000 |
| 模型保存 | `best_k_{k}.pth` |
| 历史记录 | `history_k_{k}.csv` |

### 损失函数组成

与子任务1**完全相同**，仅 k 值不同：

$$
\mathcal{L} = \mathcal{L}_{\text{pde}}(k) + 100 \cdot \mathcal{L}_{\text{bc}}
$$

### 数据流向

与子任务1一致，仅在计算 q 和 PDE 残差时传入当前 k。

### 评估方法

- **逐 k 评估**：100×100 网格
- **真解**：
  $$
  c_k = -\frac{\lambda + k^2}{\lambda - k^2}, \quad u_{\text{true}} = c_k \sin(\pi x)\sin(3\pi y)
  $$
- **输出**：
  - 每 k 一套图：`helmholtz_pinn_results_k_{k}.png`
  - 数据：`eval_k_{k}.npz`
  - 模型：`best_k_{k}.pth`

---

## 子任务3：泊松方程参数识别反问题

### 训练数据来源

| 类型 | 数量 | 来源 |
|------|------|------|
| 观测数据 | 200 | `子任务3数据.xlsx`（x, y, u） |
| 配点（PDE） | 5000 | 域内均匀随机采样 |
| 边界点 | 400（每边100） | 四条边界 |

源项 f(x,y) 解析给定：

$$
f(x,y) = \frac{\pi^2}{2}(1+x^2+y^2)\sin\frac{\pi x}{2}\cos\frac{\pi y}{2} - \pi x \cos\frac{\pi x}{2}\cos\frac{\pi y}{2} + \pi y \sin\frac{\pi x}{2}\sin\frac{\pi y}{2}
$$

### 模型结构

**双网络 PINN**：

| 网络 | 结构 | 激活 | 输出 | 参数量 |
|------|------|------|------|--------|
| u_net | [2,64,64,64,1] | Tanh | u(x,y) | ~10,000 |
| k_net | [2,32,32,32,1] | Tanh → softplus + 1e-6 | k(x,y) > 0 | ~3,000 |

总参数量约 14,000。

### 训练策略

| 项目 | 设置 |
|------|------|
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | `ReduceLROnPlateau(patience=200, factor=0.5, min_lr=1e-6)` |
| 训练轮数 | 20,000 |
| 梯度裁剪 | max_norm=1.0 |
| 最佳模型 | 自动保存最低 loss |
| 日志 | 每100轮记录，每1000轮详细打印 |

### 损失函数组成

$$
\mathcal{L} = w_d\mathcal{L}_d + w_p\mathcal{L}_p + w_b\mathcal{L}_b + w_r\mathcal{L}_r + w_s\mathcal{L}_s
$$

| 损失项 | 公式 | 权重 |
|--------|------|------|
| 数据损失 | $\frac{1}{N_d}\sum (u_{\text{pred}}-u_{\text{data}})^2$ | 1.0 |
| PDE损失 | $\frac{1}{N_c}\sum \left(-\nabla\cdot(k\nabla u) - f\right)^2$ | 0.1 |
| 边界损失 | $\frac{1}{N_b}\sum u_b^2$ | 0.1 |
| 正则损失 | $\frac{1}{N_d}\sum (k-1)^2$ | 1e-4 |
| 光滑损失 | $\frac{1}{N_d}\sum (\partial_x k)^2 + (\partial_y k)^2$ | 1e-4 |

**权重调整建议**：增加 PDE 和正则权重可提升物理一致性。


### 评估方法

| 指标 | 计算方式 |
|------|----------|
| 数据拟合误差 | MSE(u_pred, u_data), 相对误差 |
| PDE 残差误差 | MSE(res), 相对于 f 的相对误差 |
| k 场统计 | min, max, mean, std |


