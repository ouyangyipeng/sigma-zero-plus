# 扩展实验与论文撰写计划

## 项目概述

在已完成 σ-zero 复现和基础改进（L∞ + SSIM 约束）的基础上，进行更深入的扩展实验，并撰写符合顶会标准的正式论文。

---

## Phase 1: 扩展算法实现

### 1.1 自适应权重 (Adaptive Weighting)

**动机**：固定的 λ_linf 和 λ_ssim 可能在不同优化阶段不是最优的。早期需要更强的攻击性（低约束），后期需要更好的视觉质量（高约束）。

**实现**：
```python
# 在 sigma_zero_perceptual.py 中添加
def get_adaptive_weights(i, steps, lambda_linf_init, lambda_ssim_init, mode='linear'):
    """
    根据优化进度自适应调整权重
    
    Args:
        i: 当前迭代步数
        steps: 总迭代步数
        lambda_linf_init: L∞ 初始权重
        lambda_ssim_init: SSIM 初始权重
        mode: 'linear', 'cosine', 'exponential'
    """
    progress = i / steps
    
    if mode == 'linear':
        factor = progress
    elif mode == 'cosine':
        factor = (1 - math.cos(math.pi * progress)) / 2
    elif mode == 'exponential':
        factor = 1 - math.exp(-3 * progress)
    else:
        factor = progress
    
    return lambda_linf_init * factor, lambda_ssim_init * factor
```

**文件**：修改 `sigma_zero_perceptual.py`，添加 `sigma_zero_perceptual_adaptive` 函数

### 1.2 多尺度 SSIM (Multi-Scale SSIM)

**动机**：单尺度 SSIM 可能无法捕捉不同分辨率下的结构信息。多尺度 SSIM 在多个下采样级别计算 SSIM，更全面地评估视觉质量。

**实现**：
```python
def compute_multiscale_ssim(img1, img2, scales=[1, 0.5, 0.25], weights=[0.5, 0.3, 0.2]):
    """
    多尺度 SSIM
    
    Args:
        img1, img2: 输入图像 [B, C, H, W]
        scales: 下采样比例列表
        weights: 各尺度权重
    """
    total_ssim = 0
    for scale, weight in zip(scales, weights):
        if scale < 1:
            size = (int(img1.shape[2] * scale), int(img1.shape[3] * scale))
            img1_s = F.interpolate(img1, size=size, mode='bilinear')
            img2_s = F.interpolate(img2, size=size, mode='bilinear')
        else:
            img1_s, img2_s = img1, img2
        total_ssim += weight * compute_ssim(img1_s, img2_s)
    return total_ssim
```

### 1.3 TV 正则化 (Total Variation Regularization)

**动机**：TV 正则化鼓励空间平滑的扰动，减少视觉伪影，使对抗样本更加自然。

**实现**：
```python
def compute_tv_loss(delta):
    """
    Total Variation 损失
    
    Args:
        delta: 扰动 [B, C, H, W]
    """
    tv_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]).sum()
    tv_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / delta.shape[0]
```

### 1.4 组合 Loss 函数

最终 Loss 函数：
```
adv_loss = is_not_adv + dl_loss + l0_approx_normalized
           + λ_linf * L∞_penalty
           + λ_ssim * SSIM_loss
           + λ_tv * TV_loss
```

---

## Phase 2: CIFAR-10 数据集实验

### 2.1 数据集与模型

| 项目 | 配置 |
|------|------|
| 数据集 | CIFAR-10 (32×32×3 彩色图像) |
| 模型 | Carmon2019Unlabeled (WideResNet-28-10) |
| 来源 | RobustBench |
| 样本数 | 100 |
| Batch Size | 16 (受内存限制可能调整为 8) |

### 2.2 配置

创建 `configs/config_cifar10_comparison.json`：
```json
{
  "seed": 1233,
  "experiments": [
    {
      "attack": {
        "name": "sigma_zero",
        "params": {"steps": 100}
      },
      "dataset": "cifar10",
      "model": "carmon2019",
      "n_samples": 100,
      "batch_size": 16
    },
    {
      "attack": {
        "name": "sigma_zero_perceptual_ssim",
        "params": {"steps": 100, "lambda_linf": 0.0, "lambda_ssim": 0.8}
      },
      "dataset": "cifar10",
      "model": "carmon2019",
      "n_samples": 100,
      "batch_size": 16
    },
    {
      "attack": {
        "name": "sigma_zero_perceptual_both",
        "params": {"steps": 100, "lambda_linf": 0.1, "lambda_ssim": 0.5}
      },
      "dataset": "cifar10",
      "model": "carmon2019",
      "n_samples": 100,
      "batch_size": 16
    }
  ]
}
```

### 2.3 预期差异

| 指标 | MNIST (黑白) | CIFAR-10 (彩色) |
|------|-------------|-----------------|
| L∞ 效果 | 有限（像素值 0 或 1） | 显著（连续像素值） |
| SSIM 效果 | 中等 | 显著（更多结构信息） |
| TV 效果 | 有限 | 显著（彩色通道间相关性） |
| 计算时间 | 快 | 慢（3 通道，更大模型） |

---

## Phase 3: 超参数敏感性分析

### 3.1 网格搜索配置

| 超参数 | 测试值 |
|--------|--------|
| λ_linf | [0.0, 0.05, 0.1, 0.2, 0.5] |
| λ_ssim | [0.0, 0.2, 0.5, 0.8, 1.0] |
| σ (sigma) | [1e-4, 1e-3, 1e-2, 1e-1] |
| threshold | [0.1, 0.2, 0.3, 0.5] |

### 3.2 实验设计

**实验 A：λ_linf × λ_ssim 网格搜索**
- 固定 σ=1e-3, threshold=0.3
- 5 × 5 = 25 个配置
- 每个配置运行 50 个 MNIST 样本
- 记录 ASR, L0, L∞, SSIM

**实验 B：σ 敏感性**
- 固定 threshold=0.3, λ_linf=0.1, λ_ssim=0.5
- 4 个 σ 值
- 每个配置运行 50 个样本

**实验 C：threshold 敏感性**
- 固定 σ=1e-3, λ_linf=0.1, λ_ssim=0.5
- 4 个 threshold 值
- 每个配置运行 50 个样本

### 3.3 脚本

创建 `run_hyperparameter_search.py`：
```python
"""
超参数敏感性分析
"""
import itertools
import json
# ... 运行网格搜索并保存结果
```

---

## Phase 4: 数据可视化

### 4.1 图表清单

| 图表 | 类型 | 数据 | 文件 |
|------|------|------|------|
| Fig 1: 超参数热力图 | 热力图 | λ_linf × λ_ssim 对 ASR/SSIM 的影响 | `figures/heatmap_params.png` |
| Fig 2: 优化曲线 | 折线图 | SSIM/L0/ASR vs 迭代步数 | `figures/optimization_curves.png` |
| Fig 3: 方法对比 | 柱状图+误差线 | 各方法 ASR/L0/L∞/SSIM | `figures/method_comparison.png` |
| Fig 4: 样本散点图 | 散点图 | 每样本 L0 vs SSIM | `figures/scatter_l0_ssim.png` |
| Fig 5: 超参数拟合 | 拟合图 | λ_ssim 与 SSIM 提升关系 | `figures/fitting_ssim.png` |
| Fig 6: 可视化对比 | 图像网格 | 原始/原始σ-zero/改进 对比 | `figures/visual_comparison.png` |
| Fig 7: CIFAR-10 对比 | 柱状图 | MNIST vs CIFAR-10 结果 | `figures/cifar10_comparison.png` |
| Fig 8: σ 敏感性 | 折线图 | σ 值对各项指标的影响 | `figures/sigma_sensitivity.png` |

### 4.2 绘图脚本

创建 `plot_results.py`，使用 matplotlib + seaborn：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# 设置顶会风格
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': False,
    'font.family': 'serif',
})

def plot_heatmap(data, xlabel, ylabel, title, save_path):
    """绘制热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_optimization_curves(...):
    """绘制优化曲线"""
    # ...

def plot_method_comparison(...):
    """绘制方法对比柱状图（带误差线）"""
    # ...

def plot_scatter_l0_ssim(...):
    """绘制散点图"""
    # ...

def plot_fitting(...):
    """绘制拟合图"""
    # ...

def plot_visual_comparison(...):
    """绘制可视化对比"""
    # ...
```

---

## Phase 5: 论文撰写

### 5.1 论文结构

```
┌─────────────────────────────────────────────────────────┐
│ Page 1                                                   │
│ - Abstract (150-200 words)                               │
│ - CCS Concepts                                           │
│ - Keywords                                               │
│ - 1. Introduction (part 1)                               │
├─────────────────────────────────────────────────────────┤
│ Page 2                                                   │
│ - 1. Introduction (part 2: contributions)                │
│ - 2. Background and Related Work                         │
│   - 2.1 Adversarial Attacks                              │
│   - 2.2 ℓ₀-norm Attacks                                  │
│   - 2.3 Perceptual Quality Metrics                       │
├─────────────────────────────────────────────────────────┤
│ Page 3                                                   │
│ - 3. Methodology                                         │
│   - 3.1 σ-zero: Preliminaries                            │
│   - 3.2 Perceptually-Aware σ-zero                        │
│   - 3.3 Loss Function Design                             │
├─────────────────────────────────────────────────────────┤
│ Page 4                                                   │
│   - 3.4 Extensions                                       │
│     - Adaptive Weighting                                 │
│     - Multi-Scale SSIM                                   │
│     - Total Variation Regularization                     │
│   - [Figure: System Architecture / Method Overview]      │
├─────────────────────────────────────────────────────────┤
│ Page 5                                                   │
│ - 4. Experimental Setup                                  │
│   - 4.1 Datasets and Models                              │
│   - 4.2 Evaluation Metrics                               │
│   - 4.3 Baselines                                        │
│   - 4.4 Implementation Details                           │
│   - [Table: Experimental Configuration]                  │
├─────────────────────────────────────────────────────────┤
│ Page 6                                                   │
│ - 5. Results                                             │
│   - 5.1 MNIST Results                                    │
│   - [Table: MNIST comparison results]                    │
│   - [Figure: Method comparison bar chart]                │
│   - [Figure: Optimization curves]                        │
├─────────────────────────────────────────────────────────┤
│ Page 7                                                   │
│   - 5.2 CIFAR-10 Results                                 │
│   - [Table: CIFAR-10 comparison results]                 │
│   - [Figure: CIFAR-10 comparison]                        │
│   - [Figure: Visual comparison examples]                 │
├─────────────────────────────────────────────────────────┤
│ Page 8                                                   │
│ - 6. Analysis                                            │
│   - 6.1 Hyperparameter Sensitivity                       │
│   - [Figure: Heatmap of λ_linf × λ_ssim]                 │
│   - [Figure: σ sensitivity]                              │
│   - 6.2 Ablation Study                                   │
│   - [Figure: Scatter plot L0 vs SSIM]                    │
│   - [Figure: Fitting curve]                              │
├─────────────────────────────────────────────────────────┤
│ Page 9                                                   │
│ - 7. Discussion                                          │
│   - 7.1 Why SSIM Works Better Than L∞                    │
│   - 7.2 Limitations                                      │
│   - 7.3 Future Work                                      │
│ - 8. Conclusion                                          │
├─────────────────────────────────────────────────────────┤
│ Page 10                                                  │
│ - References (20-30 real papers)                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 论文文件结构

```
mypaper/
├── main.tex                    # 主论文文件
├── sigplan/                    # SIGPLAN 模板
│   ├── sigplanconf.cls
│   ├── sigplanconf-template.tex
│   └── sigplanconf-guide.pdf
├── figures/                    # 图表
│   ├── heatmap_params.png
│   ├── optimization_curves.png
│   ├── method_comparison.png
│   ├── scatter_l0_ssim.png
│   ├── fitting_ssim.png
│   ├── visual_comparison.png
│   ├── cifar10_comparison.png
│   └── sigma_sensitivity.png
└── tables/                     # 表格数据
    ├── mnist_results.tex
    └── cifar10_results.tex
```

### 5.3 参考文献列表（真实文献）

```latex
\bibliographystyle{abbrvnat}
\begin{thebibliography}{}

% σ-zero 原论文
\bibitem[Cin{\`a} et~al.(2025)]{cina2025sigma}
A.~E. Cin{\`a}, F.~Villani, M.~Pintor, L.~Sch{\"o}nherr, B.~Biggio, and
  M.~Pelillo, ``$\sigma$-zero: Gradient-based optimization of $\ell_0$-norm
  adversarial examples,'' in \emph{ICLR}, 2025.

% 对抗攻击经典论文
\bibitem[Szegedy et~al.(2014)]{szegedy2014intriguing}
C.~Szegedy et~al., ``Intriguing properties of neural networks,'' in \emph{ICLR}, 2014.

\bibitem[Goodfellow et~al.(2015)]{goodfellow2015explaining}
I.~J. Goodfellow, J.~Shlens, and C.~Szegedy, ``Explaining and harnessing
  adversarial examples,'' in \emph{ICLR}, 2015.

\bibitem[Madry et~al.(2018)]{madry2018towards}
A.~Madry et~al., ``Towards deep learning models resistant to adversarial
  attacks,'' in \emph{ICLR}, 2018.

% ℓ₀ 攻击相关
\bibitem[Carlini and Wagner(2017)]{carlini2017towards}
N.~Carlini and D.~Wagner, ``Towards evaluating the robustness of neural
  networks,'' in \emph{IEEE S\&P}, 2017.

\bibitem[Modas et~al.(2019)]{modas2019sparsefool}
A.~Modas, S.-M. Moosavi-Dezfooli, and P.~Frossard, ``SparseFool: a few pixels
  make a big difference,'' in \emph{CVPR}, 2019.

% 感知质量相关
\bibitem[Wang et~al.(2004)]{wang2004image}
Z.~Wang, A.~C. Bovik, H.~R. Sheikh, and E.~P. Simoncelli, ``Image quality
  assessment: from error visibility to structural similarity,'' \emph{IEEE TIP},
  vol.~13, no.~4, pp. 600--612, 2004.

\bibitem[Zhang et~al.(2018)]{zhang2018unreasonable}
R.~Zhang, P.~Isola, A.~A. Efros, E.~Shechtman, and O.~Wang, ``The
  unreasonable effectiveness of deep features as a perceptual metric,'' in
  \emph{CVPR}, 2018.

% 鲁棒基准
\bibitem[Croce et~al.(2021)]{croce2021robustbench}
F.~Croce et~al., ``RobustBench: a standardized adversarial robustness
  benchmark,'' in \emph{NeurIPS Datasets and Benchmarks}, 2021.

% 更多相关论文...
\end{thebibliography}
```

---

## 执行顺序

```
Step 1: 扩展算法实现 (sigma_zero_perceptual.py)
    ├── 自适应权重
    ├── 多尺度 SSIM
    └── TV 正则化

Step 2: MNIST 扩展实验
    ├── 运行自适应权重实验
    ├── 运行多尺度 SSIM 实验
    └── 运行 TV 正则化实验

Step 3: CIFAR-10 实验
    ├── 下载 CIFAR-10 模型
    └── 运行对比实验

Step 4: 超参数敏感性分析
    ├── λ_linf × λ_ssim 网格搜索
    ├── σ 敏感性
    └── threshold 敏感性

Step 5: 数据可视化
    ├── 生成所有图表
    └── 保存到 mypaper/figures/

Step 6: 论文撰写
    ├── 编写 main.tex
    ├── 插入图表
    └── 添加参考文献
```

---

## 预期产出

| 产出 | 描述 |
|------|------|
| `sigma_zero_perceptual.py` | 扩展的改进算法实现 |
| `run_hyperparameter_search.py` | 超参数搜索脚本 |
| `plot_results.py` | 数据可视化脚本 |
| `mypaper/figures/` | 8 张高质量图表 |
| `mypaper/main.tex` | 9页+1页引用的完整论文 |
| `results/comparison/` | 所有实验结果 JSON |

---

## 风险与应对

| 风险 | 应对措施 |
|------|---------|
| CIFAR-10 模型下载失败 | 使用本地预训练模型或手动下载 |
| CUDA OOM (CIFAR-10) | 减小 batch_size 到 8 或 4 |
| 超参数搜索时间过长 | 减少网格搜索粒度或并行运行 |
| LaTeX 编译问题 | 使用 pdflatex 或 xelatex，确保所有宏包已安装 |
| 图表质量不够 | 使用 dpi=300，调整字体大小和线条宽度 |

---

## 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-04-28 | 选择 sigplan 模板而非 IEEE | 用户指定使用 mypaper/sigplan 模板 |
| 2026-04-28 | 添加 TV 正则化 | 简单且有效的空间平滑约束 |
| 2026-04-28 | 添加多尺度 SSIM | 更全面的结构相似性评估 |
| 2026-04-28 | 添加自适应权重 | 动态平衡攻击强度和视觉质量 |
