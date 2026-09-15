# GAN 研究与练习

本分支 `GAN-research` 是 **GAN（生成对抗网络）系列研究与练习的母分支**。
所有 GAN 项目都从这里切出独立的子分支，母分支只维护总览索引、目录约定与已完成项目作为基线。

## 目录约定

每个 GAN 项目占一个顶层目录 `GAN/<project-name>/`，目录名与子分支名保持一致，便于对照：

```
GAN/
└── DCGAN-anime-faces/      # DCGAN 动漫头像生成
    ├── MAIN.ipynb          # 训练脚本
    ├── EVAL.ipynb          # 推理脚本
    ├── README.md           # 项目说明
    ├── model/              # 训练好的模型权重
    └── sample/             # 生成效果样例
```

## 项目列表

| 项目 | 子分支 | 说明 | 状态 |
|---|---|---|---|
| [DCGAN 动漫头像生成](GAN/DCGAN-anime-faces/) | `DCGAN-anime-faces` | 基于 Kaggle 动漫人脸数据集（63,565 张）训练 DCGAN，从 100 维噪声生成 64×64 动漫头像 | 已完成 |

## 分支约定

| 分支 | 作用 |
|---|---|
| `GAN-research` | 母分支（本分支）。只放总览索引与已完成项目的基线，不放实验过程 |
| `GAN-<project>` | 单个 GAN 项目的开发分支，从 `GAN-research` 切出 |

**新建一个 GAN 项目：**

```bash
git switch GAN-research
git pull
git switch -c GAN-<project-name>
# ... 开发 ...
git push -u origin GAN-<project-name>
```

**项目完成后按需合回母分支：**

```bash
git switch GAN-research
git merge --no-ff GAN-<project-name>
```

> 本分支为孤儿分支，与仓库 `main` 没有共同历史，两者互不影响。

## 路线图

- [x] **DCGAN** —— 无条件生成，动漫头像
- [ ] **WGAN / WGAN-GP** —— 用 Wasserstein 距离与梯度惩罚改善训练稳定性
- [ ] **CGAN** —— 条件生成，按类别或标签控制输出
- [ ] **潜空间探索** —— 插值、属性方向、DCGAN 特征可视化
- [ ] **高分辨率生成** —— ProGAN / StyleGAN 的核心思路
- [ ] **图像翻译** —— Pix2Pix / CycleGAN

> 路线图只是占位计划，会随学习进度调整。

## 环境

- Python 3.10
- PyTorch + torchvision
- CUDA（可选；CPU 也能跑，只是慢很多）

```
torch
torchvision
numpy
matplotlib
```

## 说明

- 数据集体积较大，不纳入版本管理；各项目 README 会说明数据来源与放置位置。
- 单个模型权重目前直接提交（约十几 MB）；当项目数量增多、仓库明显膨胀时，再迁移到 Git LFS。
- 所有项目仅用于学习与研究，数据集版权归原作者所有。
