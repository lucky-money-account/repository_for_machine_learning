import cv2 as cv
import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. 预处理函数：先高斯模糊，再直方图均衡化
def preprocess_images(images, blur_ksize=(3,3), blur_sigma=0):
    """
    对批量灰度图像依次进行高斯模糊和直方图均衡化
    参数:
        images: 输入图像数组，形状 (N, H, W)，值范围 0-255，uint8类型
        blur_ksize: 高斯核大小
        blur_sigma: 高斯核标准差
    返回:
        处理后的图像数组，形状相同，值范围 0-255，uint8类型
    """
    processed = np.empty_like(images)
    for i in range(images.shape[0]):    ##images 是一个 NumPy 数组，通过image.shape()获取形状，形状通常是 (N, H, W)，其中 N 是图像的数量，H 和 W 分别是图像的高度和宽度。
        # 高斯模糊
        blurred = cv.GaussianBlur(images[i], blur_ksize, blur_sigma)
        # 直方图均衡化
        equalized = cv.equalizeHist(blurred)
        processed[i] = equalized
    return processed

# 应用预处理（对训练集和测试集）
train_images = preprocess_images(train_images)
test_images  = preprocess_images(test_images)

# 3. 扁平化 (28x28 -> 784)
X_train = train_images.reshape(-1, 784).astype(np.float32)
X_test  = test_images.reshape(-1, 784).astype(np.float32)

# 4. 归一化到 [0,1]
X_train /= 255.0
X_test  /= 255.0

# 5. 标签整理
y_train = train_labels.astype(np.float32).reshape(-1, 1)
y_test  = test_labels.astype(np.float32).reshape(-1, 1)

# 6. 训练 kNN
knn = cv.ml.KNearest_create()
knn.train(X_train, cv.ml.ROW_SAMPLE, y_train)

# 7. 预测（以 k=3 为例）
ret, result, neighbours, dist = knn.findNearest(X_test, k=3)

# 8. 准确率
matches = result == y_test
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(f"准确率: {accuracy:.2f}%")
