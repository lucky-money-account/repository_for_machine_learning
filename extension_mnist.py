import cv2 as cv
import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. 数据扁平化 (28x28 -> 784)
# 为什么选择用reshape（）来扁平化处理？
# 对于单个二维数组（图像），可以用 ravel() 或 flatten() 将其变为一维。
# 对于批量图像（如 MNIST），最标准、最高效的方式是直接对整个数组使用 reshape(-1, 特征数)，它会自动对每个样本进行展平。
# 用 ravel() 或 flatten() 逐个处理也是可行的，但代码冗长且效率低，通常只在处理非 NumPy 数组（如 Python 列表）时才需要。
X_train = train_images.reshape(-1, 784).astype(np.float32)
X_test  = test_images.reshape(-1, 784).astype(np.float32)

# 3. 归一化（将像素值缩放到 [0,1]，强烈推荐）
X_train /= 255.0
X_test  /= 255.0

# 4. 标签整理：确保是二维列向量（kNN要求）
y_train = train_labels.astype(np.float32).reshape(-1, 1)
y_test  = test_labels.astype(np.float32).reshape(-1, 1)

# 5. 训练 kNN 模型
knn = cv.ml.KNearest_create()
knn.train(X_train, cv.ml.ROW_SAMPLE, y_train)

# 6. 预测测试集（以 k=3 为例）
ret, result, neighbours, dist = knn.findNearest(X_test, k=1)

# 7. 计算准确率
matches = result == y_test
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(f"准确率: {accuracy:.2f}%")