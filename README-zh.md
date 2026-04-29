# Spectrum — 扩散模型采样加速节点

[English](README.md) | **中文**

**扩散模型采样提速 1~4.79 倍** — [Spectrum (CVPR 2026)](https://github.com/hanjq17/Spectrum) 的 ComfyUI 非官方实现。无需训练，即插即用。

---

## 介绍

Spectrum 是一种**训练无关**的扩散采样加速技术。它将去噪网络内部特征视为时间函数，用切比雪夫多项式在谱域做全局拟合，从而预测并跳过大量冗余的网络前向计算。相比传统方法（如局部泰勒展开），Spectrum 的近似误差不随跳过步长增长，因此在高加速比下仍能保持画质。

节点当前支持的模型：

| 模型 | 检测类型 | 加速质量 |
|------|----------|----------|
| Klein 9b | Flux-like | 优秀 |
| Longcat Image | Flux-like | 优秀 |
| FLUX.1 | Flux-like | 优秀 |
| Qwen Image (T2I) | MMDiT | 良好 |
| Z Image Turbo | Lumina2 | 略有降低（noise_refiner 结构特殊） |
| ErnieImage | Ernie | 正常 |
| Wan2.2 | Wan | 加速效果偏弱（双采样，每轮步数少） |
| HunyuanVideo 1.5 | Hunyuan | 正常 |
| Qwen Image Edit | MMDiT | **较差**（60层全带分离调制，不建议开启） |
| LTX2.3 | LTX | 未测试（硬件受限） |

节点需 `warmup_steps` 步（默认 3）收集初始缓存，之后逐步提速。**总步数越多，加速效果越明显**。对于 Z Image Turbo 或 Klein 等轻量模型，可将 `warmup_steps` 设为 1。

### 示例

**文生图**（Qwen image）：

![文生图示例](assets/example1.png)

**图像编辑**（Klein base 9b）：

![图像编辑示例](assets/example2.png)

### 速度对比

以下测试均在 RTX 4090 上使用默认参数（w=0.5, M=4, window_size=2, flex_window=0.75）完成。

| | Klein 9b | Z Image Turbo | Qwen Image | ErnieImage |
|---|---|---|---|---|
| 未加速 | ![klein](assets/klein.png) | ![zimage](assets/zimage.png) | ![qwenimage](assets/qwenimage.png) | ![ernie](assets/ernie.png) |

---

## 节点原理

### 第一步：把特征看作时间的函数

将去噪网络中**最后一个 attention 块**输出的每个特征通道，看作沿去噪时间 $t$ 变化的标量函数 $h_i(t)$。

### 第二步：用切比雪夫多项式做全局拟合

用 $M+1$ 个切比雪夫正交基函数的加权和来逼近这个函数：

$$h_i(t) = \sum_{m=0}^{M} c_{m,i} \cdot T_m(\tau), \quad \tau = 2t - 1 \in [-1, 1]$$

其中 $T_m$ 是第 $m$ 阶切比雪夫多项式（$T_0=1,\; T_1=\tau,\; T_m = 2\tau\cdot T_{m-1} - T_{m-2}$）。

**为什么选切比雪夫？** 它的近似误差上界只取决于阶数 $M$，不与预测步长（跳过多少步）挂钩（论文定理 3.3）。而传统方法（如局部泰勒展开）的误差随步长幂次增长，大步长时画质崩溃。

### 第三步：在线岭回归拟合系数

每次实际网络前向时，收集块输出特征 $\mathbf{H}$ 和对应时间 $\Phi$。对所有已缓存点做岭回归：

$$\mathbf{C} = \arg\min_{\mathbf{C}} \|\Phi\mathbf{C} - \mathbf{H}\|_F^2 + \lambda \|\mathbf{C}\|_F^2$$

解得 $\mathbf{C} = (\Phi^T\Phi + \lambda I)^{-1}\Phi^T\mathbf{H}$（$M$ 很小，计算量可忽略）。

### 第四步：混合预测

最终特征由两部分混合：

$$h_{\text{mix}} = (\underbrace{1 - w}_{\text{局部泰勒}}) \cdot h_{\text{taylor}} \;+\; \underbrace{w}_{\text{全局切比雪夫}} \cdot h_{\text{cheb}}$$

- **泰勒项**：用最近几个缓存点的离散差分做局部外推，捕捉高频细节
- **切比雪夫项**：用所有缓存点做谱域全局拟合，捕捉长程趋势
- **$w$ 控制两者的比例**：大步长跳过多用切比雪夫，小步长多用泰勒

> 直观理解：泰勒预测就像只看前车的**尾灯距离**来推测下一秒位置——近距离很准，远距离误差爆炸。切比雪夫预测则像是看清了前车**行驶轨迹的波谱**——你能预判 5 步之后的位置，和 1 步之后的预测误差差不多。

---

## 参数说明

### `w` — 切比雪夫/泰勒混合权重

- **公式**：$h_{\text{mix}} = (1-w) \cdot h_{\text{taylor}} + w \cdot h_{\text{cheb}}$
- **范围**：0.0 ~ 1.0，**默认** 0.5，**推荐** 0.3 ~ 0.8
- **w=0**：纯泰勒。近距离准，大步长画质差
- **w=1**：纯切比雪夫。大步长稳定，细节可能丢失
- 节点内部 w 会**动态调整**：窗口变大时自动增加 w，但不超过 `max_w`

### `M` — 切比雪夫多项式阶数

- **公式**：$\sum_{m=0}^{M} c_{m,i} \cdot T_m(\tau)$
- **范围**：1 ~ 10，**默认** 4，**推荐** 3 ~ 6
- M=2 太粗糙，M=4 是甜点，M=6+ 提升微弱

### `lam` (λ) — 岭回归正则化强度

- **公式**：$(\Phi^T\Phi + \lambda I)^{-1}\Phi^T\mathbf{H}$
- **范围**：0.001 ~ 10.0，**默认** 0.1，**推荐** 0.01 ~ 1.0
- 太小→数值不稳定；太大→欠拟合。0.1 是论文最优值

### `warmup_steps` — 预热步数

- **范围**：0 ~ 20，**默认** 3，**推荐** 2 ~ 5
- 前 N 步始终实际前向，收集初始缓存。轻量模型（Klein、Z Image Turbo）可设为 1
- 设为总步数 = 禁用加速

### `window_size` — 初始跳过间隔

- **公式**：$\mathcal{N}$（论文初始窗口）
- **范围**：1.0 ~ 16.0，**默认** 2.0，**推荐** 1.5 ~ 4.0
- window=1 不加速，window=2 每步交替，越大初始越激进

### `flex_window` (α) — 窗口增长速度

- **公式**：$\alpha$（自适应调度斜率）
- **范围**：0.0 ~ 4.0，**默认** 0.75，**推荐** 0.3 ~ 2.0
- 每次实际前向后窗口增加 α：`window, window+α, window+2α, ...`
- α=0 固定间隔；α=0.75 逐渐加速；α=3.0 极限加速
- **为什么后期可以跳过更多？** 早期步定布局（误差敏感），后期步调细节（误差不敏感）
- **步距 0.01**，可精确调节

> 类比：flex_window 是你的**加速油门**。α=0 是定速巡航，α=0.75 是缓慢加速，α=3.0 是地板油。"先慢后快"最优。

### `max_w` — 最大切比雪夫权重

- **范围**：0.0 ~ 1.0，**默认** 0.8，**推荐** 0.6 ~ 0.9
- w 动态调整的上限。极端加速时可调至 0.9，一般不需要改

### `verbose` — 调试日志

- 开启后打印每步的 FWD/SKIP 决策和窗口大小，方便调参

---

## 推荐参数组合

| 场景 | 参数 | 预期加速 |
|------|------|----------|
| 保守（质量优先） | w=0.3, M=4, warmup=4, window=2, flex=0.3, max_w=0.6 | ≈2x |
| 均衡（默认） | w=0.5, M=4, warmup=3, window=2, flex=0.75, max_w=0.8 | ≈3x |
| 激进（速度优先） | w=0.7, M=6, warmup=2, window=2, flex=2.0, max_w=0.9 | ≈4-5x |
| 图像编辑 | w=0.5, M=4, warmup=4, window=2, flex=0.5, max_w=0.8 | ≈2x |

---

## 图像编辑模型须知

- **Klein / Longcat Edit**：单 blocks 统一调制平滑了主/参差异，加速质量与 T2I 持平
- **Qwen Image Edit**：60 层全带 timestep_zero 分离调制，质量下降明显。建议用保守参数或关闭加速

---

## 补充说明

此节点由 Claude Code 与 DeepSeek 辅助开发，许可证与原项目一致（MIT License）。欢迎自由使用和贡献。

---

## 引用

```
@article{han2026adaptive,
  title={Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration},
  author={Han, Jiaqi and Shi, Juntong and Li, Puheng and Ye, Haotian and Guo, Qiushan and Ermon, Stefano},
  journal={arXiv preprint arXiv:2603.01623},
  year={2026}
}
```
