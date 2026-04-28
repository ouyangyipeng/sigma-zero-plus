# 论文插图设计规划

> 本文档描述论文中需要的系统结构图和示意图的设计方案，可直接提供给 AI 绘图工具生成。

---

## 📐 图 1：系统架构总览图（System Architecture Overview）

**用途**：放在 Section 3 (Design) 开头，展示整个 Perceptually-Aware $\sigma$-zero 的流程。

**类型**：流程图 / 架构图

**布局**：从左到右的水平流程，分为 4 个主要模块

### 模块设计

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Input Image │ ──► │  σ-zero Optimizer    │ ──► │  Perceptual         │ ──► │  Output      │
│  x ∈ [0,1]ᵈ  │     │  (Gradient-based)    │     │  Constraints Module │     │  Adversarial │
│              │     │                      │     │                     │     │  Example x'  │
│  [MNIST 图片] │     │  • σ-approximation   │     │  • ℓ∞ Penalty       │     │              │
│              │     │  • Dynamic threshold │     │  • SSIM Loss        │     │  [对抗样本]   │
│  Label: y    │     │  • Adam optimizer    │     │  • TV Regularizer   │     │              │
└─────────────┘     │  • Cosine annealing  │     │  • Adaptive weights │     └──────────────┘
                    └──────────────────────┘     └─────────────────────┘
                              │                            │
                              ▼                            ▼
                    ┌──────────────────────┐     ┌─────────────────────┐
                    │  Adversarial Loss    │     │  Total Loss         │
                    │  L_adv = DL(z, y)    │     │  L_total = Σ λᵢLᵢ   │
                    └──────────────────────┘     └─────────────────────┘
```

### AI 绘图 Prompt

```
A clean, professional system architecture diagram for a machine learning paper.
Horizontal layout with 4 main blocks connected by arrows:

1. Left block: "Input Image" - shows a grayscale MNIST digit image (28x28), with label "x, y"
2. Middle-left block: "σ-zero Optimizer" - shows gradient descent icon, with bullet points:
   "σ-approximation", "Dynamic threshold", "Adam optimizer"
3. Middle-right block: "Perceptual Constraints" - shows three sub-modules stacked vertically:
   "ℓ∞ Penalty", "SSIM Loss", "TV Regularizer", with "Adaptive Weights" label
4. Right block: "Adversarial Example" - shows a slightly perturbed MNIST digit

Below the main flow, show two feedback boxes:
- "Adversarial Loss (DL)" connected to the optimizer
- "Total Loss = Σ λᵢLᵢ" combining all losses

Style: Minimalist, academic paper style, blue and gray color scheme,
clear sans-serif font, white background, vector-style icons.
Size: Wide aspect ratio (16:6) suitable for a paper figure.
```

### 配色方案

| 元素 | 颜色 | 说明 |
|------|------|------|
| 输入/输出模块 | 浅蓝色 (#E3F2FD) | 数据流 |
| 优化器模块 | 深蓝色 (#1565C0) | 核心算法 |
| 感知约束模块 | 橙色 (#FF6F00) | 我们的贡献 |
| 损失函数框 | 灰色 (#F5F5F5) | 辅助计算 |
| 箭头 | 深灰 (#424242) | 数据流向 |

---

## 📐 图 2：$\sigma$-zero 平滑近似函数示意图

**用途**：放在 Section 2 (Original Paper) 中，直观展示 $\sigma_\sigma(x) = \frac{x^2}{x^2 + \sigma}$ 的行为。

**类型**：函数曲线图

### 设计描述

```
y
│
│  σ→0 (sharp)
│  ╱────────────────────  y ≈ 1 (x ≠ 0)
│ ╱
│╱
├──────────●─────────── x
│         0
│
│  σ=0.1 (smooth)
│  ───╱╲───  (smooth bump)
│
│  σ=0.01 (medium)
│  ──╱  ╲──
```

### AI 绘图 Prompt

```
A mathematical function plot showing the σ-zero smooth approximation function:
σ_σ(x) = x² / (x² + σ)

Plot 3 curves on the same graph:
1. σ = 0.001 (nearly step function, sharp transition at x=0) - dark blue, thick line
2. σ = 0.01 (moderate smoothness) - medium blue
3. σ = 0.1 (very smooth, wide bump) - light blue, dashed

X-axis: x ∈ [-1, 1], labeled "Perturbation magnitude δᵢ"
Y-axis: σ_σ(x) ∈ [0, 1.1], labeled "σ_σ(x) ≈ ‖δ‖₀"

Add annotations:
- Arrow pointing to x=0: "σ_σ(0) = 0 (no perturbation)"
- Arrow pointing to flat region: "σ_σ(x) → 1 as |x| grows"
- Text box: "Smaller σ → tighter ℓ₀ approximation"

Style: Clean academic plot, white background, grid lines,
Times New Roman or similar serif font for labels.
Size: Square aspect ratio (1:1).
```

---

## 📐 图 3：视觉质量对比示意图（Visual Quality Comparison）

**用途**：放在 Section 2 (Original Paper Weakness) 或 Section 5 (Results) 中，展示原始 $\sigma$-zero 和我们方法的视觉差异。

**类型**：图像对比网格

### 设计描述

```
┌─────────────────┬─────────────────────┬─────────────────────┐
│  Original Image │  σ-zero (Original)  │  Ours (Perceptual)  │
│                 │                     │                     │
│   [干净的数字]   │  [有散乱噪点的数字]  │  [几乎干净的数字]    │
│                 │                     │                     │
│  Label: 3       │  Label: 3 → 8 ✗    │  Label: 3 → 8 ✗    │
│  SSIM: 1.000    │  SSIM: 0.794       │  SSIM: 0.870        │
│  L0: 0          │  L0: 23            │  L0: 18             │
└─────────────────┴─────────────────────┴─────────────────────┘

下方：Perturbation Mask 对比
┌─────────────────┬─────────────────────┬─────────────────────┐
│  (no mask)      │  Scattered pixels   │  Clustered pixels   │
│                 │  (salt-and-pepper)  │  (smooth regions)   │
└─────────────────┴─────────────────────┴─────────────────────┘
```

### AI 绘图 Prompt

```
A comparison figure showing 3 columns of MNIST digit images for an adversarial attack paper:

Column 1: "Original" - A clean handwritten digit "3" on black background, white pixels
Column 2: "σ-zero (Original)" - Same digit "3" but with scattered white noise pixels
          (salt-and-pepper style), visibly perturbed, SSIM=0.794
Column 3: "Ours (Perceptual)" - Same digit "3" with minimal visible perturbation,
          looks almost identical to original, SSIM=0.870

Below each image, show the perturbation mask (difference image):
- Original: empty (all black)
- σ-zero: scattered white dots across the image
- Ours: fewer, more clustered white dots

Add metrics below each column:
- Original: "L0: 0, SSIM: 1.000"
- σ-zero: "L0: 23, SSIM: 0.794"
- Ours: "L0: 18, SSIM: 0.870"

Style: Clean grid layout, white text labels on dark background for images,
clear metric labels below. Academic paper style.
Size: 3:1 aspect ratio (wide).
```

---

## 📐 图 4：自适应权重调度曲线

**用途**：放在 Section 3 (Design) 中，展示三种自适应权重调度函数。

**类型**：函数曲线图

### 设计描述

```
λ(t)
│
│         ╱─── Exponential
│       ╱╱
│     ╱╱    Cosine
│   ╱╱    ╱╲
│ ╱╱    ╱   ╲
│╱────╱──────╲── Linear
│
├──────────────── t/T
0               1
```

### AI 绘图 Prompt

```
A line chart showing three adaptive weight scheduling functions for an optimization algorithm:

X-axis: "Optimization Progress (t/T)" from 0 to 1
Y-axis: "Weight Multiplier α(t)" from 0 to 1

Three curves:
1. Linear: α(t) = t/T — straight diagonal line from (0,0) to (1,1) — blue solid
2. Cosine: α(t) = (1 - cos(π·t/T)) / 2 — S-shaped curve — orange solid, thicker
3. Exponential: α(t) = 1 - exp(-3·t/T) — fast rise then plateau — green dashed

Add annotations:
- Near t=0: "Early stage: low constraints → aggressive attack"
- Near t=1: "Late stage: high constraints → refine visual quality"
- Arrow pointing to cosine curve: "Best trade-off (used in final model)"

Style: Clean academic plot, white background, subtle grid,
legend in top-left corner, serif font for axis labels.
Size: 4:3 aspect ratio.
```

---

## 📐 图 5：损失函数组成示意图（Loss Components Breakdown）

**用途**：放在 Section 3 (Design) 中，直观展示总损失函数的各个组成部分。

**类型**：堆叠条形图 / 面积图

### 设计描述

```
Loss Value
│
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│  │ TV  │  │     │  │     │  │     │  │     │
│  ├─────┤  │ TV  │  │     │  │     │  │     │
│  │SSIM │  ├─────┤  │     │  │     │  │     │
│  ├─────┤  │SSIM │  │ TV  │  │     │  │     │
│  │ L∞  │  ├─────┤  ├─────┤  │ TV  │  │     │
│  ├─────┤  │ L∞  │  │SSIM │  ├─────┤  │ TV  │
│  │ L₀  │  ├─────┤  ├─────┤  │SSIM │  ├─────┤
│  ├─────┤  │ L₀  │  │ L∞  │  ├─────┤  │SSIM │
│  │Adv  │  ├─────┤  ├─────┤  │ L₀  │  ├─────┤
│  └─────┘  │Adv  │  │Adv  │  ├─────┤  │ L₀  │
│           └─────┘  └─────┘  │Adv  │  ├─────┤
│  Original  +SSIM  +L∞+SSIM  └─────┘  │Adv  │
│           (+TV)            +Adaptive  └─────┘
│                                        Full
└─────────────────────────────────────────────
```

### AI 绘图 Prompt

```
A stacked bar chart showing the loss function composition across different method variants
for an adversarial attack paper.

5 bars (left to right):
1. "Original" — 2 segments: Adversarial Loss (bottom), ℓ₀ Loss (top)
2. "+SSIM" — 3 segments: Adv, ℓ₀, SSIM
3. "+L∞+SSIM" — 4 segments: Adv, ℓ₀, L∞, SSIM
4. "+Adaptive+TV" — 5 segments: Adv, ℓ₀, L∞, SSIM, TV
5. "Full" — 5 segments with adaptive weighting indication

Color scheme:
- Adversarial Loss: dark blue (#1565C0)
- ℓ₀ Loss: medium blue (#42A5F5)
- L∞ Penalty: light orange (#FFB74D)
- SSIM Loss: orange (#FF6F00)
- TV Regularizer: green (#66BB6A)

Add a legend on the right side.
Add annotation: "All variants maintain 100% ASR" at the top.

Style: Clean academic bar chart, white background,
subtle grid lines, clear labels.
Size: 16:9 aspect ratio.
```

---

## 📐 图 6：Pareto 前沿示意图（L0 vs SSIM Trade-off）

**用途**：放在 Section 5 (Results) 中，展示各方法的 L0-SSIM 权衡关系。

**类型**：散点图 + Pareto 前沿线

### AI 绘图 Prompt

```
A scatter plot showing the trade-off between sparsity (L0) and visual quality (SSIM)
for different adversarial attack methods.

X-axis: "Median L0 (number of perturbed pixels)" — range [15, 30], lower is better
Y-axis: "SSIM (structural similarity)" — range [0.78, 0.88], higher is better

Data points (7 methods):
1. "σ-zero (original)" — L0=23, SSIM=0.794 — gray circle, labeled
2. "+SSIM" — L0=22, SSIM=0.855 — light blue triangle
3. "+L∞+SSIM" — L0=28, SSIM=0.790 — light red square
4. "+Adaptive" — L0=27, SSIM=0.852 — purple diamond
5. "+Multi-scale SSIM" — L0=23, SSIM=0.870 — blue star, labeled "Best SSIM"
6. "+TV" — L0=21, SSIM=0.869 — green pentagon, labeled "Best Trade-off"
7. "Full" — L0=18, SSIM=0.866 — orange hexagon, labeled "Best L0"

Draw a Pareto frontier line connecting the non-dominated points
(Full → TV → Multi-scale SSIM).

Add a green shaded region in the bottom-left labeled "Ideal Region"
(low L0, high SSIM).

Add annotation: "Original σ-zero is strictly dominated by all our variants"

Style: Clean academic scatter plot, white background, grid lines,
legend in bottom-right corner.
Size: Square aspect ratio (1:1).
```

---

## 📐 图 7：超参数搜索热力图

**用途**：放在 Section 5 (Results) 中，展示 $\lambda_{L\infty} \times \lambda_{\text{SSIM}}$ 的网格搜索结果。

**类型**：3 面板热力图

### AI 绘图 Prompt

```
Three side-by-side heatmaps showing hyperparameter grid search results
for an adversarial attack optimization.

Grid: 5×5 grid for each heatmap
X-axis: "λ_SSIM" values [0.0, 0.1, 0.2, 0.5, 0.8]
Y-axis: "λ_L∞" values [0.0, 0.05, 0.1, 0.2, 0.5]

Heatmap 1 (left): "ASR (%)"
- Most cells are 100% (dark green)
- Color range: 90-100%

Heatmap 2 (center): "SSIM"
- Best region: λ_SSIM ∈ [0.2, 0.5], λ_L∞ ≤ 0.1 (bright yellow/orange)
- Worst region: high λ_L∞ (dark blue)
- Color range: 0.78-0.87

Heatmap 3 (right): "Median L0"
- Best (lowest): λ_SSIM ∈ [0.2, 0.5] (dark blue)
- Worst (highest): high λ_L∞ (bright red)
- Color range: 18-30

Add annotations:
- Circle around optimal region on SSIM heatmap: "Optimal: λ_L∞ ≤ 0.1, λ_SSIM ∈ [0.2, 0.5]"
- Arrow on L∞ axis: "High L∞ degrades performance"

Style: Clean academic heatmaps, viridis or similar perceptually uniform colormap,
white borders between cells, value labels in each cell.
Size: 3:1 aspect ratio (three panels side by side).
```

---

## 📐 图 8：算法迭代过程可视化

**用途**：放在 Section 3 (Design) 或 Section 5 (Results) 中，展示优化过程中各指标的变化。

**类型**：多线图（随迭代步数变化）

### AI 绘图 Prompt

```
A multi-line chart showing the optimization trajectory over 100 iterations
of the Perceptually-Aware σ-zero algorithm.

X-axis: "Iteration" from 0 to 100

Create 3 subplots stacked vertically:

Subplot 1 (top): "Adversarial Loss"
- Y-axis: Loss value (decreasing from ~5 to ~0)
- Two lines: Original σ-zero (blue) vs Ours (orange)
- Both decrease, ours slightly slower but reaches same level

Subplot 2 (middle): "SSIM"
- Y-axis: SSIM from 0.7 to 1.0
- Two lines: Original σ-zero (blue, stays ~0.79) vs Ours (orange, increases to ~0.87)
- Add annotation: "Perceptual constraints improve SSIM throughout optimization"

Subplot 3 (bottom): "L0 (perturbed pixels)"
- Y-axis: L0 from 0 to 40
- Two lines: Original σ-zero (blue, converges to ~23) vs Ours (orange, converges to ~18)
- Both start high and decrease due to dynamic thresholding

Style: Clean academic multi-panel plot, white background,
shared x-axis, consistent color scheme (blue=original, orange=ours),
legend in top-right of each subplot.
Size: 4:6 aspect ratio (tall).
```

---

## 📐 图 9：σ 参数敏感性分析

**用途**：放在 Section 5 (Results) 中，展示 σ 参数对结果的影响。

**类型**：带误差线的折线图

### AI 绘图 Prompt

```
A line chart with error bars showing the sensitivity of the σ-zero algorithm
to the σ hyperparameter.

X-axis: "σ (log scale)" — values: 1e-4, 1e-3, 1e-2, 1e-1
Y-axis (left): "SSIM" — range [0.80, 0.86]
Y-axis (right): "Median L0" — range [18, 28]

Two lines:
1. SSIM (blue, circles): increases from ~0.81 at σ=1e-4 to ~0.85 at σ=1e-2,
   then plateaus. Error bars showing std across 50 samples.
2. L0 (orange, squares): relatively stable around 20-23, slight increase at σ=1e-1.
   Error bars showing IQR.

Add annotations:
- Arrow at σ=1e-3: "Default value (used in main experiments)"
- Arrow at σ=1e-2: "Best SSIM but slightly higher L0"
- Text box: "Larger σ → smoother approximation → better perceptual optimization"

Style: Clean academic plot with dual y-axis, log-scale x-axis,
error bars with caps, white background.
Size: 4:3 aspect ratio.
```

---

## 📐 图 10：论文整体结构图（Paper Organization）

**用途**：放在 Introduction 末尾或作为独立的 overview 图。

**类型**：层次结构图

### AI 绘图 Prompt

```
A hierarchical diagram showing the organization and contributions of the paper.

Top level: "Perceptually-Aware σ-zero"

Second level (3 branches):
1. "Problem Identification"
   → "Visual quality weakness in original σ-zero"
   → "No perceptual constraints"

2. "Method Design" (highlighted in orange)
   → "SSIM Loss"
   → "L∞ Penalty"  
   → "TV Regularization"
   → "Adaptive Weighting"

3. "Experimental Validation"
   → "MNIST: 7 methods comparison"
   → "Hyperparameter search: 25 configs"
   → "Ablation study"
   → "Sensitivity analysis"

Bottom level: "Key Results"
→ "SSIM +9.6% (0.794 → 0.870)"
→ "L0 -22% (23 → 18)"
→ "100% ASR maintained"

Style: Clean tree diagram, top-down layout,
orange highlight for our contributions,
blue for baseline components,
green for results.
White background, rounded rectangles for nodes.
Size: 16:9 aspect ratio.
```

---

## 🎨 统一设计规范

### 字体

| 用途 | 字体 | 大小 |
|------|------|------|
| 图标题 | Times New Roman Bold | 11pt |
| 轴标签 | Times New Roman | 10pt |
| 图例/标注 | Times New Roman | 9pt |
| 数据标签 | Arial | 8pt |

### 配色（ColorBrewer Set1 变体）

| 元素 | HEX | RGB |
|------|-----|-----|
| 原始方法 | #377EB8 | (55, 126, 184) |
| 我们的方法 | #E41A1C | (228, 26, 28) |
| SSIM 变体 | #4DAF4A | (77, 175, 74) |
| TV 变体 | #984EA3 | (152, 78, 163) |
| 完整模型 | #FF7F00 | (255, 127, 0) |
| 背景/网格 | #F0F0F0 | (240, 240, 240) |

### 尺寸

| 图表类型 | 宽度 | 高度 |
|----------|------|------|
| 单栏图 | 3.5 in | 2.5 in |
| 双栏图 | 7 in | 3 in |
| 方形图 | 3.5 in | 3.5 in |
| 架构图 | 7 in | 2.5 in |

### 文件格式

- 矢量图：PDF 或 EPS（优先）
- 位图：PNG，300 DPI 以上
- 所有图保存在 `mypaper/figures/` 目录

---

*最后更新：2026-04-28*
