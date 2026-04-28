# Perceptually-Aware σ-zero: 视觉感知约束的稀疏对抗攻击

## 1. 引言

### 1.1 背景

对抗攻击通过在输入图像上添加微小扰动来欺骗深度学习模型。在众多攻击范式中，**ℓ₀ 范数攻击**旨在最小化被修改的像素数量，产生稀疏的对抗扰动。

### 1.2 σ-zero 算法

σ-zero (Cinà et al., ICLR 2025) 是一种基于梯度优化的 ℓ₀ 范数对抗攻击方法。其核心创新在于：

1. **ℓ₀ 范数的可微近似**：使用 `σ(x) = x² / (x² + σ)` 近似不可微的 ℓ₀ 范数
2. **动态阈值机制**：自适应调整阈值，逐步稀疏化扰动
3. **Difference of Logits (DL) Loss**：使用 logits 差值作为攻击损失

### 1.3 痛点分析

原始 σ-zero 算法优化的是 ℓ₀ 范数（扰动像素数量），但存在以下问题：

- **单个像素变化幅度不受控**：即使只修改了少量像素，这些像素的变化幅度可能很大（接近 1.0），导致对抗样本肉眼可见
- **无结构感知**：没有考虑图像的结构性，扰动可能破坏图像的视觉连贯性

## 2. 方法

### 2.1 Perceptually-Aware σ-zero

我们在原始 σ-zero 的 Loss 函数中加入视觉感知约束：

**原始 Loss**：
```
adv_loss = is_not_adv + dl_loss + l0_approx_normalized
```

**改进 Loss**：
```
adv_loss = is_not_adv + dl_loss + l0_approx_normalized 
           + λ_linf * L∞_penalty 
           + λ_ssim * SSIM_loss
```

其中：
- **L∞ penalty**：`max(|δ_i|)`，限制单个像素的最大变化幅度
- **SSIM loss**：`1 - SSIM(x + δ, x)`，保持图像的结构相似性
- **λ_linf, λ_ssim**：超参数，控制各项权重

### 2.2 SSIM 实现

我们使用 PyTorch 实现了结构相似性（SSIM）指标：

```python
def compute_ssim(img1, img2, window_size=11):
    """计算两张图像之间的 SSIM"""
    window_1d = gaussian_kernel(window_size)
    window = window_1d.view(1, 1, 1, -1) * window_1d.view(1, 1, -1, 1)
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[1]
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    
    # ... 计算 SSIM map
    return ssim_map.mean()
```

### 2.3 代码改动

改进算法仅需在 [`sigma_zero_attack.py`](sigma-zero-adversarial-attack/sigma_zero_attack.py:63) 的 Loss 计算部分添加 3 行代码：

```python
# 新增：视觉感知约束
linf_penalty = active_delta.flatten(1).abs().max(dim=1)[0]
ssim_loss = 1 - compute_ssim(adv_inputs, active_inputs)

# 组合 Loss
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized 
            + lambda_linf * linf_penalty 
            + lambda_ssim * ssim_loss).mean()
```

## 3. 实验

### 3.1 实验设置

| 项目 | 配置 |
|------|------|
| 数据集 | MNIST |
| 模型 | SmallCNN (DDN 鲁棒训练) |
| 样本数 | 50 |
| Batch Size | 16 |
| 迭代步数 | 100 |
| 设备 | CPU |

### 3.2 对比方法

| 方法 | 描述 |
|------|------|
| σ-zero (original) | 原始算法 |
| σ-zero + L∞ | 仅加入 L∞ 惩罚 (λ=0.2) |
| σ-zero + SSIM | 仅加入 SSIM 损失 (λ=0.8) |
| σ-zero + L∞ + SSIM | 同时加入两项 (λ_linf=0.1, λ_ssim=0.5) |

### 3.3 评估指标

| 指标 | 说明 | 方向 |
|------|------|------|
| ASR | 攻击成功率 | ↑ 越高越好 |
| ℓ₀ | 扰动像素数量（中位数） | ↓ 越低越好 |
| ℓ∞ | 最大单像素变化 | ↓ 越低越好 |
| SSIM | 结构相似性 | ↑ 越高越好 |

### 3.4 实验结果

| 方法 | ASR ↑ | ℓ₀ ↓ | ℓ∞ ↓ | SSIM ↑ |
|------|-------|------|------|--------|
| σ-zero (original) | 100.0% | 23.0 | 1.0000 | 0.7939 |
| σ-zero + L∞ | 100.0% | 45.0 | 0.9984 | 0.6282 |
| **σ-zero + SSIM** | **100.0%** | **22.0** | 0.9997 | **0.8549** |
| σ-zero + L∞ + SSIM | 100.0% | 28.0 | 0.9947 | 0.7896 |

### 3.5 结果分析

1. **所有方法均达到 100% ASR**：改进算法在攻击成功率上没有妥协

2. **SSIM 版本表现最佳**：
   - SSIM 从 0.7939 提升到 **0.8549**（**+7.7%**）
   - ℓ₀ 范数从 23.0 降低到 **22.0**（更稀疏）
   - 说明 SSIM 约束有效保持了图像的结构信息

3. **L∞ 惩罚效果有限**：
   - MNIST 是黑白图像，像素值在 [0, 1] 范围内
   - 最优扰动倾向于将黑色像素变为白色（或反之），导致 ℓ∞ 接近 1.0
   - 在彩色图像（如 CIFAR-10、ImageNet）上预期会有更好的效果

4. **L∞ + SSIM 组合**：
   - 两项约束存在一定冲突
   - 需要更精细的超参数调优

## 4. 结论

我们提出了 **Perceptually-Aware σ-zero**，通过在原始 σ-zero 的 Loss 函数中加入视觉感知约束，显著提升了生成对抗样本的隐蔽性：

- **SSIM 损失**有效保持了图像的结构相似性（+7.7%）
- **攻击成功率**保持在 100%
- **ℓ₀ 范数**进一步降低（更稀疏的扰动）

改进算法仅需在核心代码中添加 **3 行代码**，是一种简单而有效的优化策略。

## 5. 未来工作

1. **扩展到彩色图像**：在 CIFAR-10 和 ImageNet 上验证 L∞ 约束的效果
2. **自适应权重**：动态调整 λ_linf 和 λ_ssim
3. **LPIPS 损失**：使用深度特征距离代替 SSIM，更好地模拟人类感知
4. **目标攻击**：扩展到 targeted attack 场景

## 附录：复现指南

### 环境配置

```bash
cd sigma-zero-adversarial-attack
conda env create -f env_china.yml
conda activate sigmazero
```

### 运行对比实验

```bash
python run_comparison.py --config=configs/config_comparison.json --device=cpu
```

### 代码结构

```
sigma-zero-adversarial-attack/
├── sigma_zero_attack.py        # 原始 σ-zero 实现
├── sigma_zero_perceptual.py    # 改进的 Perceptually-Aware σ-zero
├── attacks.py                  # 攻击注册（已添加新方法）
├── run_comparison.py             # 对比实验脚本
└── configs/
    └── config_comparison.json    # 对比实验配置
```
