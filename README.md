# 动手学深度学习（PyTorch）学习笔记与实践

基于《动手学深度学习》（Dive into Deep Learning，PyTorch 版）的系统学习记录。包含**章节知识点梳理、学习心得、综合题实战笔记、练习代码（Jupyter Notebook）**以及**自建工具函数**。

## 仓库总览（分支导航）

本仓库用**分支**组织不同的学习方向与项目，各分支独立演进、互不影响：

| 分支 | 主题 | 内容概览 |
| :--- | :--- | :--- |
| **`main`**（当前） | 动手学深度学习 | `notes/` + `practice/` + `utils/`，覆盖 ch02–ch06 与 ch17 |
| [`kNN`](https://github.com/lucky-money-account/repository_for_machine_learning/tree/kNN) | kNN 手写数字 OCR | OpenCV `digits.png`（92.56%）与 MNIST（97.71%），含多种预处理对照实验 |
| [`Neural-Network`](https://github.com/lucky-money-account/repository_for_machine_learning/tree/Neural-Network) | 神经网络基础知识梳理 | `神经骨架.md`：激活函数 / 优化器 / 损失函数 / 正则化 / 评价指标 |
| [`PBLF-Image-Processing`](https://github.com/lucky-money-account/repository_for_machine_learning/tree/PBLF-Image-Processing) | 人工智能在图像处理 | Git 入门、视频逐帧处理、YOLOv8n 人物识别、图像分类延伸实验 |
| [`GAN-research`](https://github.com/lucky-money-account/repository_for_machine_learning/tree/GAN-research) | GAN 系列研究母分支 | 系列索引 + 路线图；当前含 DCGAN 动漫头像生成 |
| [`GAN-dog-fid`](https://github.com/lucky-money-account/repository_for_machine_learning/tree/GAN-dog-fid) | DCGAN 生成狗图 | 自实现的 FID / MiFID 评估系统，FID 从 433 降到 91 |

> **关于「没有共同历史」**：以上分支各自独立起步（root commit 不同），这是有意为之 —— 每个学习方向可以自由演进、互不干扰，也不会互相拖累。

### GAN 系列的分支约定

GAN 相关内容遵循「**一项目一分支**」：

- `GAN-research` 是**母分支**，只放系列索引与路线图，不放具体实验过程
- 每个 GAN 项目从它切出独立分支（如 `GAN-dog-fid`），完成后按需合回

新建一个 GAN 项目：

```bash
git switch GAN-research
git pull
git switch -c GAN-<project-name>
```

> 注：`practice/ch17-生成对抗网络/` 属于《动手学深度学习》的课程练习，保留在本分支；独立 GAN 项目见 `GAN-research` 系列。

## 目录结构

```
.
├── notes/                          # 笔记
│   ├── 知识点/                     # 章节知识点梳理（概念、公式、易错点）
│   ├── 学习心得/                   # 各章学习后的反思与总结
│   ├── 综合题实战笔记/             # 综合题完整解题过程（对话式整理）
│   └── 特定模块深入讲解/           # 公式推导 / 逐行精讲 PDF
├── practice/                       # 练习（Jupyter Notebook）
│   ├── ch02-预备知识/
│   ├── ch03-线性神经网络/
│   ├── ch04-多层感知机/
│   ├── ch05-深度学习计算/
│   ├── ch06-卷积神经网络/
│   └── ch17-生成对抗网络/
├── utils/                          # 自建工具函数
│   ├── preprocessing.py            # 通用数据预处理
│   ├── train.py                    # 训练流程封装
│   ├── visualization.py            # 可视化工具
│   └── README.md
└── README.md
```

## 学习进度

| 章节 | 主题 | 知识点 | 学习心得 | 练习 |
| :--: | :-- | :--: | :--: | :--: |
| ch02 | 预备知识（张量、广播、自动微分等） | ✅ | ✅ | ✅ |
| ch03 | 线性神经网络（线性回归、softmax） | ✅ | ✅ | ✅ |
| ch04 | 多层感知机（MLP、过拟合、正则化） | ✅ | ✅ | ✅ |
| ch05 | 深度学习计算（参数管理、模型保存） | ✅ | ✅ | ✅ |
| ch06 | 卷积神经网络（卷积、池化、LeNet） | — | — | ✅ |
| ch17 | 生成对抗网络 GAN（专题） | ✅ | — | ✅ |

> 已完成 ch02–ch06 的核心内容学习，并额外完成 ch17 生成对抗网络专题（含 DCGAN 拓展）。

## 各模块说明

### notes/知识点
章节核心知识点的系统梳理，覆盖概念、关键公式与易错点，按章节编号命名（如 `ch03-线性神经网络.md`）。

### notes/学习心得
每章学习结束后的反思总结，记录理解难点与收获。

### notes/综合题实战笔记
综合题的完整解题过程整理，包含思路推导与踩坑记录（如成都房价回归综合题）。

### notes/特定模块深入讲解
针对特定模块的深入材料（PDF）：
- `softmax回归-公式推导.pdf`：softmax 回归的数学推导
- `softmax从零实现-逐行精讲.pdf`：从零实现 softmax 的逐行讲解
- `成都房价-完整流程与预处理分析.pdf`：完整流程与数据预处理分析

### practice
按章节组织的练习 notebook，每章包含若干分节小练习（`exNN-*.ipynb`）与综合题（`综合题-*.ipynb`）。练习均在 Jupyter 中完成，保留运行结果。

### utils
自建工具函数，供各章练习复用：
- `preprocessing.py` — 通用数据预处理函数
- `train.py` — 训练流程封装（设备选择、网络构建、损失、评估、训练循环）
- `visualization.py` — 可视化工具（loss 曲线、网格预测、3D 曲面、Animator 动画等）

## 环境说明

- Python 3.10
- PyTorch（CUDA 版，支持 GPU 训练）
- Jupyter Notebook
- 依赖：`torch`、`torchvision`、`d2l`、`matplotlib`、`numpy`、`pandas` 等

## 说明

- 本仓库仅包含学习相关的有效文件（笔记、练习、工具函数），不包含数据集、模型权重与缓存文件。
- 练习中的 `solutions/` 答案、`.ipynb_checkpoints/`、`__pycache__/` 等均已排除。
