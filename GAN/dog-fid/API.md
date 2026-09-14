# API 参考 · GAN 评估与训练模块

> 位置：`C:\Users\moneyforever\Desktop\Deep-Learning\Kaggle\`
> 说明：每条 API 都标注了**来源模块**和**在流程中的使用阶段**。

---

## 模块总览（哪个文件提供什么、什么时候需要）

| 文件（模块） | 提供 | 使用阶段 | 你需要它吗 |
|---|---|---|---|
| `gan_data.py` | **`dog_dataloader` / `make_dataloader`** / `PTImageDataset` / `FolderImageDataset` | ① 数据 | 训练读数据 ⭐ |
| `gan_models.py` | `Generator` / `Discriminator` / `weights_init` | ② 模型 | 训练必用 |
| `gan_eval/data.py` | `list_images` / `ImageFolderDataset` / `make_loader` | ① 数据 | 评估器内部用 |
| `gan_eval/inception.py` | `InceptionFeatureExtractor` | ③ 评估底层 | 一般不用直接调 |
| `gan_eval/metrics.py` | FID/KID/IS/MiFID/Precision-Recall 纯函数 | ③ 评估底层 | 想自管特征时才用 |
| `gan_eval/evaluator.py` | **`GANEvaluator`**（高层） | ③ 评估 | 评估生成图 ⭐ |
| `gan_eval/tracker.py` | **`FIDTracker`** | ③ 训练时评估 | 画 FID 曲线 ⭐ |
| `gan_eval/viz.py` | 样本网格 / 最近邻可视化 | ③ 可视化 | 看效果 |
| `gan_eval/__init__.py` | 汇总导出上面所有公开 API | — | 统一导入入口 |

**流程与模块对应关系**

```
① 数据 ──────► ② 模型 ──────► 训练循环 ──────► ③ 评估
gan_data.py     gan_models.py   (你自己写)      gan_eval.evaluator (GANEvaluator)
                                                 gan_eval.tracker   (FIDTracker)
                                                 gan_eval.metrics   (纯函数)
                                                 gan_eval.viz       (可视化)
```

---

## 0. 统一导入入口 —— `gan_eval/__init__.py`

```python
from gan_eval import (
    # —— evaluator.py ——
    GANEvaluator, ALL_METRICS,
    # —— tracker.py ——
    FIDTracker,
    # —— data.py ——
    list_images, make_loader, ImageFolderDataset,
    # —— inception.py ——
    InceptionFeatureExtractor,
    # —— metrics.py ——
    frechet_distance, kernel_inception_distance, inception_score,
    memorization_distance, memorization_informed_fid,
    improved_precision_recall, format_results,
)
from gan_models import Generator, Discriminator, weights_init
from gan_data import dog_dataloader, make_dataloader, PTImageDataset, FolderImageDataset
```

---

## 1. 模型 —— `gan_models.py`　【阶段 ② 模型】

```python
Generator(z_dim=100, ngf=64, nc=3)
# forward(z: (N, z_dim, 1, 1)) -> (N, 3, 64, 64)，取值 [-1, 1]
# ⚠️ 输入必须是 4D (N, z_dim, 1, 1)，不是 (N, z_dim)

Discriminator(ndf=64, nc=3)
# forward(x: (N, 3, 64, 64)) -> (N,) 的概率分数 (0,1)（末尾：FC + Sigmoid）
# ⚠️ 因此用 nn.BCELoss()，不要 BCEWithLogitsLoss（会二次 sigmoid）

weights_init(module)
# 用法：G.apply(weights_init); D.apply(weights_init)
```

---

## 2. 训练数据 —— `gan_data.py`　【阶段 ① 数据】⭐

训练读数据用这个模块。两种来源、一套接口，输出**已归一化到 [-1,1]** 的 float 张量。

```python
# 最简：用项目默认路径（优先 data/stanford_dogs_64.pt，回退 data/all-dogs）
dog_dataloader(batch_size=64, val_split=0.0, ...) -> DataLoader | (train, val)
#   batch 形状 (B, 3, 64, 64)，float32，范围 [-1,1]

# 显式指定来源
make_dataloader(pt=..., folder=..., batch_size=64, image_size=64,
                shuffle=True, drop_last=True, num_workers=0,
                pin_memory=None, val_split=0.0, seed=42,
                normalize=True, return_labels=False) -> DataLoader | (train, val)

# 底层类
PTImageDataset(pt_path, normalize=True, return_labels=False)      # 读 .pt（内存）
FolderImageDataset(folder, image_size=64, normalize=True, ...)    # 流式读文件夹
load_pt(pt_path) -> dict                                           # {'X','Y','classes'}
to_minus_one_one(x) -> Tensor                                      # uint8->[-1,1]
```

**关键约定**

| 项 | 说明 |
|---|---|
| 输出 | float `[-1,1]`，与生成器 tanh 对齐（训练循环**不用再归一化**） |
| `drop_last=True` | 默认开，GAN 训练**必须**（否则 BatchNorm 遇单样本 batch 报错） |
| `num_workers=0` | `.pt` 已在内存，无需 worker |
| `val_split>0` | 返回 `(train_loader, val_loader)`；val 为 `shuffle=False, drop_last=False` |
| 固定种子 | `seed=42`，保证每次划分一致 |

用法见下方「端到端调用骨架」。

> 评估时加载真实图由 `GANEvaluator` 内部处理，不需要用这个模块。

### 附：评估内部加载器 —— `gan_eval/data.py`

评估器内部使用，一般不用直接调：

```python
list_images(folder) -> list[str]          # 递归扫描图片路径
ImageFolderDataset(paths, size=64)        # 返回 uint8 CHW
make_loader(paths, size=64, ...) -> DataLoader
```

---

## 3. 特征提取（底层）—— `gan_eval/inception.py`　【阶段 ③ 评估底层】

> 一般不用直接调，`GANEvaluator` 已封装。

```python
InceptionFeatureExtractor(device="cpu", resize=299)
# 基于 torchvision 的 ImageNet InceptionV3（权重已缓存本地）

extractor.extract(images) -> (features, logits)
# images: uint8 [0,255] 或 float [0,1]，形状 (N,3,H,W) —— 内部自动 resize 到 299 并归一化
# features: (N, 2048) float CPU tensor  → FID / KID / MiFID / PR
# logits:   (N, 1000) float CPU tensor  → Inception Score
```

---

## 4. 指标纯函数（底层）—— `gan_eval/metrics.py`　【阶段 ③ 评估底层】

> 输入都是**特征矩阵 (N, D)**（numpy 或 torch 均可）。只有你想自管特征提取时才用。

```python
frechet_distance(feats_real, feats_fake, eps=1e-6) -> float
# FID，越低越好

kernel_inception_distance(feats_real, feats_fake, degree=3, coef=1.0,
                          subsets=50, subset_size=1000, seed=0, return_std=False)
# KID，越低越好（可为负，正常）

inception_score(logits, splits=10) -> (mean, std)
# IS，输入是 logits，越高越好

memorization_distance(feats_gen, feats_real, chunk=2048) -> float
# 记忆距离，越大越安全（越小=越像抄训练图）

memorization_informed_fid(feats_real, feats_fake, tau=0.1, eps=1e-8) -> dict
# MiFID（比赛官方指标），-> {"mifid","fid","memorization_distance","penalty"}

improved_precision_recall(feats_real, feats_fake, k=3) -> dict
# -> {"precision","recall"}   precision=质量，recall=多样性

format_results(results: dict) -> str
# 把结果 dict 格式化成表格文本
```

---

## 5. 高层评估器 —— `gan_eval/evaluator.py`　【阶段 ③ 评估】⭐推荐

```python
GANEvaluator(real_dir=None, image_size=64, batch_size=64, num_workers=0, device=None)

ev.real_features      # property：真实图特征 (N,2048)，首次访问时计算并缓存
ev.real_paths         # property：真实图路径列表（最近邻可视化用）
ev.set_real_dir(dir)  # 换真实集（会清缓存）

# 评估：fake 可以是【文件夹路径】或【uint8 张量 (N,3,64,64)】
ev.evaluate(fake, metrics=ALL_METRICS) -> dict
#   metrics 可选子集：("fid","mifid","kid","is","pr")；ALL_METRICS 默认全部
#   返回 dict 键：fid, mifid, memorization_distance, penalty,
#               kid, kid_std, is, is_std, precision, recall（按请求出现）

ev.report(results) -> str      # 格式化打印
ev.save_grid(images, path, nrow=8) -> path
ev.nearest_neighbours(fake_dir, path, n=8, size=None) -> path   # 只能传文件夹
```

**最小评估三行：**
```python
ev = GANEvaluator(real_dir=r"C:\Users\moneyforever\Desktop\Deep-Learning\Kaggle\data\all-dogs", image_size=64)
scores = ev.evaluate(r"C:\Users\moneyforever\Desktop\Deep-Learning\Kaggle\outputs\dcgan\samples")   # 或传生成图张量
print(ev.report(scores))
```

---

## 6. 训练时跟踪 —— `gan_eval/tracker.py`　【阶段 ③ 训练时评估】⭐

```python
FIDTracker(evaluator, out_dir="outputs", every=5, num_samples=2000,
           seed=0, metrics=("fid","mifid","kid","is"), save_grid=True)

tracker.step(epoch, sample_fn) -> dict
#   sample_fn(n, seed) -> uint8 张量 (n,3,64,64)   ← 你的生成器采样函数
#   自动：存样本网格 + 算分 + 写 fid_history.json + 返回结果
#   在 epoch % tracker.every == 0 时调用

tracker.history             # list[dict]，每步结果
tracker.best_epoch("fid")   # 返回 FID 最低的那条记录
tracker.plot(path=None, keys=("fid","mifid")) -> path   # 画曲线
tracker.every / tracker.num_samples                     # 属性
```

**`sample_fn` 写法约定（关键）：**
```python
def sample_fn(n, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, 1, 1, generator=g, device=device)
        imgs = G(z)                                    # [-1,1], (n,3,64,64)
    G.train()
    return ((imgs.clamp(-1,1) + 1) * 127.5).round().to(torch.uint8).cpu()
```

---

## 7. 可视化 —— `gan_eval/viz.py`　【阶段 ③ 可视化】

```python
save_sample_grid(images: uint8 (N,3,H,W), path, nrow=8, padding=2) -> path
nearest_neighbour_indices(feats_fake, feats_real, chunk=2048) -> np.ndarray
save_nearest_neighbour_grid(fake_paths, real_paths, nn_indices, path, size=64, n=8) -> path
```

---

## 端到端调用骨架（每步标注来源模块）

```python
import torch, torch.nn as nn
from gan_data import dog_dataloader                                # ① 数据
from gan_models import Generator, Discriminator, weights_init      # ② 模型
from gan_eval import GANEvaluator, FIDTracker                      # ③ 评估

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100

# ── ① 数据（gan_data.py）──────────────────────────────────
loader = dog_dataloader(batch_size=64)     # 默认用 data/stanford_dogs_64.pt
# 产出 float [-1,1] 的 (B,3,64,64) 批次，drop_last=True 已设好

# ── ② 模型（gan_models.py）────────────────────────────────
G = Generator(z_dim=z_dim).to(device); G.apply(weights_init)
D = Discriminator().to(device);        D.apply(weights_init)
optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
bce  = nn.BCELoss()                # Discriminator 输出概率分数 (0,1)

# ── ③ 评估器 + 跟踪器（gan_eval.evaluator / tracker）───────
ev = GANEvaluator(real_dir=r"C:\Users\moneyforever\Desktop\Deep-Learning\Kaggle\data\all-dogs",
                  image_size=64, device=device)          # 真实特征只算一次并缓存
tracker = FIDTracker(ev, out_dir="outputs/dcgan", every=5,
                     num_samples=2000, metrics=("fid","mifid","kid","is"))

def sample_fn(n, seed):            # 供 FIDTracker 调用
    g = torch.Generator(device=device).manual_seed(seed)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, 1, 1, generator=g, device=device)
        imgs = G(z)
    G.train()
    return ((imgs.clamp(-1,1)+1)*127.5).round().to(torch.uint8).cpu()

# ── 训练循环（你写）────────────────────────────────────────
for epoch in range(1, epochs+1):
    for real in loader:
        real = real.to(device)
        b = real.size(0)
        # ... D step / G step（bce + D(real) / D(fake)）...

    if epoch % tracker.every == 0:               # ← 评估接入点
        tracker.step(epoch, sample_fn)           # gan_eval.tracker

tracker.plot()                                   # FID 曲线
print("best:", tracker.best_epoch("fid"))
```

---

## 约定速查（最容易踩的坑）

| 项 | 约定 | 来源模块 |
|---|---|---|
| 喂给评估的生成图 | **uint8 [0,255]**，形状 `(N,3,64,64)`（与 `image_size` 一致） | evaluator |
| 生成器输出 | float **[-1,1]**；转 uint8：`((x.clamp(-1,1)+1)*127.5).round().to(torch.uint8)` | gan_models |
| `Discriminator` 输出 | **概率分数 (0,1)**（带 Sigmoid）→ 用 `BCELoss` | gan_models |
| `Generator` 输入 | 必须 4D `(N, z_dim, 1, 1)` | gan_models |
| 特征维度 | `(N, 2048)`；logits `(N, 1000)` | inception |
| `nearest_neighbours` | 只能传**文件夹**（需要文件路径） | evaluator |
| `evaluate` | 文件夹或张量都行；文件夹会 resize 到 `image_size`，张量请自己保证是 `image_size` | evaluator |
| Windows / Jupyter | `num_workers=0` | data / evaluator |
