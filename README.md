# kNN 手写数字识别（OCR）

用 **k-近邻（kNN）** 算法实现手写数字光学字符识别，包含两个数据集上的完整实验：OpenCV 官方 `digits.png` 与 MNIST，以及多种图像预处理对准确率影响的对照实验。

> 本分支是 PBLF《人工智能在图像处理方面的应用》课程的课外自主延伸练习，也是进入机器学习领域的第一步实践。

## 数据集与结果

| 数据集 | 规模 | 图像尺寸 | 最佳准确率 | 关键做法 |
| :--- | :--- | :---: | :---: | :--- |
| `digits.png`（OpenCV 官方） | 5,000 张（10 个数字 × 500） | 20×20 | **92.56%** | k=4；均衡化 / 高斯 / 二值化均无提升 |
| MNIST | 60,000 训练 / 10,000 测试 | 28×28 | **97.71%** | 高斯模糊 + 直方图均衡化 + 归一化，k=3 |

### digits.png 上 k 值的影响

| k | 1 | 2 | 3 | 4 | 5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy | 92.2 | 91.04 | 92.36 | **92.56** | 92.0 |

k 值小 → 模型复杂、易过拟合；k 值大 → 模型简单、易欠拟合。合适的 k 值需通过实验确定。

### MNIST 上预处理的影响

| 方案 | 准确率 |
| :--- | :---: |
| 基线（仅归一化） | 97.05% |
| + 高斯模糊 | 97.56% |
| + 高斯模糊 + 直方图均衡化 | **97.71%** |

**为什么组合有效？** 高斯模糊去掉孤立噪点与笔画锯齿，使同类数字更一致；直方图均衡化拉伸灰度分布，增强笔画与背景的对比度。先模糊后均衡化，避免均衡化放大噪声。

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `main.py` | digits.png 主程序：读图 → 灰度 → 切分 → 扁平化 → 划分训练/测试 → 打标签 → kNN 训练预测 → 输出准确率 |
| `learning_notes.md` | digits.png 实验完整笔记（原理、逐函数讲解、k 值对比、与图像处理技术的交叉思考） |
| `mnist_learning_notes.md` | MNIST 实验完整笔记（预处理对比、关键代码解释、可视化、反思） |
| `extension_mnist.py` | MNIST 基础版（无预处理） |
| `extension_GaussianBlur.py` | 单独尝试高斯模糊 |
| `extension_equalize.py` | 单独尝试直方图均衡化 |
| `extension_binary.py` | 单独尝试二值化 |
| `extension_mnist_pro.py` | 高斯模糊 + 直方图均衡化组合（最佳） |
| `extension_mnist_reflection.py` | 增加正确/错误样本的可视化对比 |
| `requirement.ipynb` | 题目与要求 |
| `digits.png` | OpenCV 官方手写数字图，5000 个样本按 0~9 整齐排列，可直接切分 |

## 核心代码

```python
import cv2 as cv
import numpy as np

OriginPhoto = cv.imread('digits.png')
image = cv.cvtColor(OriginPhoto, cv.COLOR_BGR2GRAY)

# 切分：50 行 × 100 列 = 5000 个 20×20 小块
cells = [np.hsplit(row, 100) for row in np.vsplit(image, 50)]
x = np.array([block.ravel() for row in cells for block in row])

# 每个数字前 250 个作训练、后 250 个作测试
x_reshaped = x.reshape(10, 500, 400)
train = x_reshaped[:, :250, :].reshape(-1, 400).astype(np.float32)
test  = x_reshaped[:, 250:, :].reshape(-1, 400).astype(np.float32)

# 标签：250 个 0、250 个 1 ……
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)

matches = result == test_labels
accuracy = np.count_nonzero(matches) * 100.0 / result.size
print(accuracy)
```

## 运行

```bash
pip install opencv-python numpy
python main.py                # digits.png 实验

pip install tensorflow        # MNIST 扩展实验需要
python extension_mnist.py
python extension_mnist_pro.py
python extension_mnist_reflection.py
```

`extension_mnist*.py` 通过 `tensorflow.keras.datasets.mnist.load_data()` 加载数据，首次运行会自动下载。

## 学习收获

- 理解 kNN 的「懒惰学习」本质：训练阶段只是记住样本，预测时才计算距离。
- 体会预处理的作用与边界：MNIST 上组合预处理带来 0.66% 提升；而 `digits.png` 原图过于清晰，均衡化 / 高斯 / 二值化均无提升。
- 掌握 OpenCV kNN 接口：`KNearest_create()` / `train(ROW_SAMPLE)` / `findNearest(k)` 及其四个返回值。
- 认识到准确率在样本不均衡时会「骗人」，需要 Precision / Recall / F1 配合判断。

## 参考

- [OpenCV：使用 kNN 进行手写数据光学字符识别（OCR）](https://docs.opencv.ac.cn/4.11.0/d8/d4b/tutorial_py_knn_opencv.html)
