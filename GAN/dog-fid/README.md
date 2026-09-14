# Kaggle · Generative Dog Images（DCGAN）

用 **DCGAN** 生成 64×64 的狗图，包含**自实现的 FID / MiFID 评估系统**、**训练稳定性调试**与**完整迭代记录**。

> 参考比赛：[Kaggle Generative Dog Images](https://www.kaggle.com/competitions/generative-dog-images)
> 数据集：Stanford Dogs（120 品种，20580 张）

---

## 成果

从"完全学不到"到"能生成可辨识的不同狗狗"，FID 从 **433 降到 91**：

| 迭代 | 最佳 FID | 关键动作 |
| :--: | :--: | :-- |
| 0 → 1 | 433 | 标准 DCGAN 基线（失败，卡死） |
| 1 → 2 | 145.0 | 修正标签（真=0.9 / 假=0.0） |
| 2 → 3 | 138.7 | 加实例噪声 σ=0.1 |
| 3 → 4 | **94.6** | 训练 300 epoch |
| 4 → 5 | **91.2** | 判别器加谱归一化（SN） |

> 参考水平：随机噪声基线 ~355，DCGAN 标准 55~59，顶尖方案 18~30。
> 完整迭代过程与踩坑记录见 [`PROGRESS.md`](PROGRESS.md)。

---

## 目录结构

```
.
├── README.md              # 本文件
├── API.md                 # 各模块 API 参考
├── PROGRESS.md            # 学习精进记录（迭代格式）
├── requirements.txt
│
├── MainFunc.ipynb         # 主函数（训练 + 评估）
├── gan_models.py          # DCGAN 模型（Generator / Discriminator）
├── gan_data.py            # 训练数据接口（.pt / 文件夹）
├── fid_plot.py            # FID 逐 epoch 折线图
├── flow_diagram.py        # 全流程图
│
├── gan_eval/              # 评估库（自实现，不依赖 GitHub 权重）
│   ├── inception.py       # InceptionV3 特征提取
│   ├── metrics.py         # FID / MiFID / KID / IS / Precision-Recall
│   ├── evaluator.py       # 高层 API：GANEvaluator
│   ├── tracker.py         # 训练时 FID 曲线：FIDTracker
│   ├── data.py            # 图片加载
│   └── viz.py             # 样本网格 / 最近邻可视化
│
├── scripts/               # 命令行脚本
│   ├── download_data.py   # 下载 Stanford Dogs（无需验证）
│   ├── evaluate.py        # 命令行评估
│   ├── train_dcgan.py     # DCGAN 训练脚本
│   └── selftest.py        # 环境自检
│
├── notebooks/             # 辅助 notebook
│   ├── 00_data_preprocess.ipynb
│   ├── 01_gan_evaluation.ipynb
│   └── 02_data_usage_template.ipynb
│
└── results/               # 结果图
    ├── fid_curve.png
    ├── flow_diagram.png
    └── sample_grids/      # 各迭代最优 epoch 的生成图
        ├── iter4_noise_300ep_epoch295.png
        ├── iter4_noise_300ep_epoch300.png
        ├── iter5_SN_noise_300ep_epoch295.png
        └── iter5_SN_noise_300ep_epoch300.png
```

---

## 环境

```bash
pip install -r requirements.txt
# 核心：torch / torchvision / numpy / scipy / pillow / matplotlib / tqdm
```

**说明**：`gan_eval` 使用 **torchvision 的 InceptionV3 权重**（download.pytorch.org），
不依赖 torchmetrics / clean-fid 的 GitHub Release 权重，国内网络可直接运行。

自检：

```bash
python scripts/selftest.py     # 看到 SELFTEST OK 即成功
```

---

## 快速开始

### 1. 准备数据

```bash
python scripts/download_data.py          # 下载 Stanford Dogs 到 data/
```

### 2. 训练（见 `MainFunc.ipynb`）

```python
from gan_data import dog_dataloader
from gan_models import Generator, Discriminator, weights_init
from gan_eval import GANEvaluator, FIDTracker

# 数据（输出 float [-1,1]）
loader = dog_dataloader(batch_size=64)

# 模型
G = Generator(z_dim=100).apply(weights_init).to(device)
D = Discriminator().apply(weights_init).to(device)

# 评估器 + 训练时 FID 跟踪
ev = GANEvaluator(real_dir="data/all-dogs", image_size=64)
tracker = FIDTracker(ev, out_dir="outputs/dcgan", every=5, num_samples=2000)
```

### 3. 画 FID 曲线

```python
from fid_plot import plot_fid
plot_fid(tracker, out_path="outputs/fid_curve.png")
```

---

## 核心模块

| 模块 | 作用 |
| :-- | :-- |
| `gan_models.py` | DCGAN 生成器/判别器（判别器输出概率分数 + Sigmoid） |
| `gan_data.py` | 训练数据接口，输出已归一化到 `[-1,1]` 的张量 |
| `gan_eval` | FID / MiFID / KID / IS / Precision-Recall 评估 |
| `fid_plot.py` | FID 逐 epoch 折线图（标注最佳 epoch） |

**详细 API 见 [`API.md`](API.md)。**

---

## 关键经验（踩坑总结）

1. **FID 一条直线 = 完全没学到**：不要用 loss 判断 GAN，要看 FID 曲线
2. **判别器过强会导致梯度消失**：D_loss→0、G_loss 爆炸时，用**实例噪声**或**谱归一化**
3. **标签要正确**：单边标签平滑（真=0.9、假=0.0）
4. **"模式坍塌"可能是训练不足**：训久后 Recall 从 0.05 涨到 0.38
5. **技巧会功能重叠**：SN 和实例噪声都在稳定 D，叠加收益有限
6. **单 batch 过拟合测试**：判断"是 bug 还是训练不够"的黄金工具

---

## 评估指标说明

| 指标 | 方向 | 含义 |
| :-- | :--: | :-- |
| **FID** | 越低越好 | 生成图与真实图的分布距离 |
| **MiFID** | 越低越好 | FID × 记忆惩罚（防抄训练图） |
| **KID** | 越低越好 | FID 的无偏版，小样本更稳 |
| **Inception Score** | 越高越好 | 类别可分性 |
| **Precision** | 越高越好 | 生成图的质量 |
| **Recall** | 越高越好 | 生成图的多样性 |
