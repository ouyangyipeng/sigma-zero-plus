# σ-zero 复现与优化 - ROADMAP

## 项目概述

复现 ICLR 2025 论文 **σ-zero: Gradient-based Optimization of ℓ₀-norm Adversarial Examples**，并提出改进算法 **Perceptually-Aware σ-zero**（视觉感知约束的 σ-zero）。

---

## Phase 1: 环境配置与复现

### 1.1 修改 conda 镜像源

创建修改后的 `env_china.yml`，使用国内镜像加速：

```yaml
name: sigmazero
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - defaults
dependencies:
  - pip=23.2.1
  - python=3.11.5
  - pytorch=2.0.1
  - torchvision=0.15.2
  - tqdm=4.66.1
  - pip:
      - git+https://ghproxy.com/https://github.com/RobustBench/robustbench.git
      - git+https://ghproxy.com/https://github.com/jeromerony/adversarial-library
      - git+https://ghproxy.com/https://github.com/Harry24k/adversarial-attacks-pytorch.git
      - eagerpy==0.30.0
      - foolbox==3.3.3
      - scipy==1.11.2
      - numba==0.58.0
```

### 1.2 安装环境

```bash
cd sigma-zero-adversarial-attack
conda env create -f env_china.yml
conda activate sigmazero
```

### 1.3 验证环境

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from adv_lib.utils.losses import difference_of_logits; print('adv-lib OK')"
python -c "from robustbench import load_model; print('robustbench OK')"
```

### 1.4 运行复现实验

创建最小测试配置 `configs/config_reproduce.json`：

```json
{
  "seed": 1233,
  "experiments": [
    {
      "attack": {
        "name": "sigma_zero",
        "params": {
          "steps": 100
        }
      },
      "dataset": "mnist",
      "model": "smallcnn_ddn",
      "n_samples": 20,
      "batch_size": 16
    }
  ]
}
```

运行：
```bash
python main.py --device=cuda:0 --config=configs/config_reproduce.json
```

---

## Phase 2: 改进算法设计

### 2.1 痛点分析

原始 σ-zero 算法优化的是 ℓ₀ 范数（扰动像素数量），但存在以下问题：

- **单个像素变化幅度不受控**：即使只修改了少量像素，这些像素的变化幅度可能很大，导致肉眼可见
- **无结构感知**：没有考虑图像的结构性，扰动可能破坏图像的视觉连贯性

### 2.2 改进方案：Perceptually-Aware σ-zero

在原有 Loss 基础上加入视觉感知约束：

```python
# 原始 Loss
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

# 改进方案 A: 加入 L∞ 范数惩罚
linf_penalty = active_delta.flatten(1).abs().max(dim=1)[0]
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized + lambda_linf * linf_penalty).mean()

# 改进方案 B: 加入 SSIM 损失
ssim_loss = 1 - ssim(adv_inputs, active_inputs)
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized + lambda_ssim * ssim_loss).mean()

# 改进方案 C: 同时加入 L∞ + SSIM
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized 
            + lambda_linf * linf_penalty 
            + lambda_ssim * ssim_loss).mean()
```

### 2.3 实现位置

修改文件：[`sigma_zero_attack.py`](sigma-zero-adversarial-attack/sigma_zero_attack.py:63)

需要修改的核心代码段（第 46-63 行）：

```python
# compute loss
logits = model(adv_inputs)
dl_loss = difference_of_logits(logits, active_labels).clip(0)
l0_approx = l0_approximation(active_delta.flatten(1), sigma)
l0_approx_normalized = l0_approx / active_delta.data.flatten(1).shape[1]

# ===== 新增：视觉感知约束 =====
# L∞ penalty: 限制单个像素的最大变化
linf_penalty = active_delta.flatten(1).abs().max(dim=1)[0]

# SSIM loss: 保持结构相似性
ssim_loss = 1 - compute_ssim(adv_inputs, active_inputs)

# 组合 Loss
adv_loss = (is_not_adv + dl_loss + l0_approx_normalized 
            + lambda_linf * linf_penalty 
            + lambda_ssim * ssim_loss).mean()
```

### 2.4 SSIM 实现

使用 PyTorch 实现 SSIM（不依赖外部库）：

```python
import torch.nn.functional as F

def gaussian_kernel(size: int = 11, sigma: float = 1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def compute_ssim(img1, img2, window_size=11, window=None):
    if window is None:
        window_1d = gaussian_kernel(window_size)
        window = window_1d.view(1, 1, 1, -1) * window_1d.view(1, 1, -1, 1)
        window = window.to(img1.device)
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])
    
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
```

---

## Phase 3: 对比实验

### 3.1 实验配置

创建对比实验配置 `configs/config_comparison.json`：

```json
{
  "seed": 1233,
  "experiments": [
    {
      "attack": {
        "name": "sigma_zero",
        "params": {"steps": 100}
      },
      "dataset": "mnist",
      "model": "smallcnn_ddn",
      "n_samples": 100,
      "batch_size": 32
    },
    {
      "attack": {
        "name": "sigma_zero_perceptual",
        "params": {
          "steps": 100,
          "lambda_linf": 0.1,
          "lambda_ssim": 0.5
        }
      },
      "dataset": "mnist",
      "model": "smallcnn_ddn",
      "n_samples": 100,
      "batch_size": 32
    }
  ]
}
```

### 3.2 评估指标

| 指标 | 说明 |
|------|------|
| Attack Success Rate (ASR) | 攻击成功率 |
| ℓ₀ norm | 扰动像素数量 |
| ℓ∞ norm | 最大单像素变化 |
| SSIM | 结构相似性 |
| LPIPS | 感知相似度（可选） |

### 3.3 预期结果

| 方法 | ASR ↑ | ℓ₀ ↓ | ℓ∞ ↓ | SSIM ↑ |
|------|-------|------|------|--------|
| σ-zero (original) | ~95% | ~50 | ~0.8 | ~0.75 |
| σ-zero + L∞ | ~93% | ~55 | ~0.3 | ~0.82 |
| σ-zero + SSIM | ~92% | ~60 | ~0.5 | ~0.88 |
| σ-zero + L∞ + SSIM | ~90% | ~65 | ~0.3 | ~0.90 |

---

## Phase 4: 报告撰写

### 报告结构

1. **Introduction**
   - 对抗攻击背景
   - ℓ₀ 范数攻击的挑战
   - σ-zero 的贡献与局限

2. **痛点分析**
   - 原始 σ-zero 生成的对抗样本容易被肉眼察觉
   - 缺乏对视觉感知质量的约束

3. **方法**
   - 提出 Perceptually-Aware σ-zero
   - L∞ 范数裁剪：限制单像素最大变化
   - SSIM 损失：保持结构相似性

4. **实验**
   - 实验设置（数据集、模型、超参数）
   - 定量对比（ASR、ℓ₀、ℓ∞、SSIM）
   - 定性对比（可视化对抗样本）

5. **结论**
   - 改进算法在保持攻击成功率的同时显著提升隐蔽性

---

## 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-04-28 | 使用 conda + 清华镜像配置环境 | 用户偏好，原项目使用 conda |
| 2026-04-28 | 使用 ghproxy.com 加速 GitHub 依赖 | 国内网络限制 |

---

## 风险与应对

| 风险 | 应对措施 |
|------|---------|
| ghproxy.com 不可用 | 尝试 gitclone.com 或手动下载 zip |
| MNIST 下载失败 | 使用 torchvision 镜像源或手动下载 |
| CUDA OOM | 减小 batch_size 到 8 或 4 |
| robustbench 模型下载失败 | 手动下载模型权重到 models/ 目录 |
