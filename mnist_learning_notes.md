# MNIST 手写数字识别学习笔记

## 1. MNIST 数据集简介

MNIST 是计算机视觉领域的“Hello World”数据集，包含 0~9 的手写数字灰度图像。

- **图像尺寸**：28 × 28 像素
- **图像类型**：单通道灰度图，像素值范围 0（黑）~ 255（白）
- **样本数量**：
  - 训练集：60,000 张
  - 测试集：10,000 张
- **类别**：10 个数字（0~9），各类别样本基本均衡
- **加载方式**：`tensorflow.keras.datasets.mnist.load_data()` 直接返回 (train_images, train_labels), (test_images, test_labels)，数据已随机打乱并划分好。

## 2. 基础代码实现（无预处理）

python

```python
import cv2 as cv
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 扁平化 (28x28 -> 784)
X_train = train_images.reshape(-1, 784).astype(np.float32)
X_test  = test_images.reshape(-1, 784).astype(np.float32)

# 归一化 (0~255 -> 0~1)
X_train /= 255.0
X_test  /= 255.0

# 标签整理为列向量 (kNN要求)
y_train = train_labels.astype(np.float32).reshape(-1, 1)
y_test  = test_labels.astype(np.float32).reshape(-1, 1)

# 创建并训练 kNN 模型
knn = cv.ml.KNearest_create()
knn.train(X_train, cv.ml.ROW_SAMPLE, y_train)

# 预测 (k=3)
ret, result, neighbours, dist = knn.findNearest(X_test, k=3)
accuracy = np.mean(result == y_test) * 100
print(f"准确率: {accuracy:.2f}%")
```



**结果**：准确率约 **97.05%**。

## 3. 预处理尝试及效果

### 3.1 高斯模糊（单独使用）

- 目的：平滑图像，去除微小噪声和边缘锯齿。
- 实现：对每张图 `cv2.GaussianBlur(img, (3,3), 0)`。
- 结果：准确率 **97.56%**（比基础提升约 0.5%）。

### 3.2 高斯模糊 + 直方图均衡化（组合）

- 目的：先平滑再增强对比度，使数字笔画更突出。
- 实现：循环处理每张图：先模糊，再 `cv2.equalizeHist()`。
- 结果：准确率 **97.71%**（进一步提升 0.15%）。

**为什么组合有效？**

- 高斯模糊去除了孤立噪点和笔画锯齿，使同类数字更一致。
- 直方图均衡化拉伸了灰度分布，增强了笔画与背景的对比度。
- 先模糊后均衡化，避免了均衡化放大噪声。

## 4. 关键代码解释

### 4.1 为什么用 `reshape(-1, 784)` 扁平化？

- `train_images` 形状为 (60000, 28, 28)，需要转换为 (60000, 784) 才能输入 kNN。
- `reshape(-1, 784)` 中 `-1` 自动计算样本数，简洁高效，返回视图（不复制数据）。
- 相比循环 `ravel()`，直接 `reshape` 速度更快，代码更简洁。

### 4.2 为什么要除以 255.0（归一化）？

- 原始像素值 0~255，归一化到 [0,1] 可以：
  - 避免大数值对距离计算产生过大影响。
  - 提高数值稳定性（浮点运算更精确）。
  - 为后续使用其他模型（如 SVM、神经网络）打下基础。
- 虽然不是 kNN 的绝对必需，但强烈推荐，且不会降低准确率。

### 4.3 标签为什么要 `reshape(-1, 1)`？

- kNN 的 `train()` 要求标签为二维列向量，形状 (样本数, 1)。
- 直接使用一维标签会报错，因此需要转换。

### 4.4 kNN 训练与预测

- `cv.ml.KNearest_create()` 创建模型。
- `knn.train(train, cv.ml.ROW_SAMPLE, labels)` 训练（kNN 只是存储数据）。
- `knn.findNearest(test, k)` 预测，返回结果、邻居标签、距离等。

## 5. 可视化对比（原始 vs 预处理）

通过随机选取正确/错误分类的样本，并排显示原始图像和预处理后图像，可以直观感受预处理的效果。

**正确样本**：预处理后数字更清晰，但原图本身也易识别。
**错误样本**：多为书写潦草或形状相近的数字（如 4 和 9、7 和 1），预处理后可能仍难以区分。

（可视化代码略，见附件 `extension_mnist_reflection.py`）

## 6. 总结与反思

- **预处理是机器学习的重要环节**，合适的预处理能提升模型性能。本例中高斯模糊+均衡化使准确率从 97.05% 提升到 97.71%。
- **单一预处理未必有效**，组合使用可能产生协同效果。
- **参数调优**：高斯核大小、k 值等需要实验确定，可进一步探索。
- **后续方向**：尝试其他特征（如 HOG）、更强分类器（SVM、简单 CNN），或挑战更复杂的数据集（Fashion-MNIST、HWD-V1）。

通过本次实践，我不仅学会了 MNIST 的基本处理流程，还亲身体验了预处理对模型的影响，为后续学习更复杂的视觉任务打下了基础。

------

**附件**：

- `extension_mnist.py`：基础版本
- `extension_mnist_pro.py`：加入高斯模糊+均衡化
- `extension_mnist_reflection.py`：增加可视化对比