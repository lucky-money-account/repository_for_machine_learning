import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 1. 加载数据 --------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 备份原始测试图像（用于可视化对比）
test_images_original = test_images.copy()

# -------------------- 2. 定义预处理函数 --------------------
def preprocess_images(images, blur_ksize=(3,3), blur_sigma=0):
    processed = np.empty_like(images)
    for i in range(images.shape[0]):
        blurred = cv.GaussianBlur(images[i], blur_ksize, blur_sigma)
        equalized = cv.equalizeHist(blurred)
        processed[i] = equalized
    return processed

# 对训练集和测试集进行预处理
train_images = preprocess_images(train_images)
test_images  = preprocess_images(test_images)

# -------------------- 3. 扁平化 + 归一化 --------------------
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0
X_test  = test_images.reshape(-1, 784).astype(np.float32) / 255.0

y_train = train_labels.astype(np.float32).reshape(-1, 1)
y_test  = test_labels.astype(np.float32).reshape(-1, 1)

# -------------------- 4. 训练 kNN 模型 --------------------
knn = cv.ml.KNearest_create()
knn.train(X_train, cv.ml.ROW_SAMPLE, y_train)

# -------------------- 5. 预测测试集 --------------------
k = 3
ret, result, neighbours, dist = knn.findNearest(X_test, k=k)
result = result.flatten()

# -------------------- 6. 正确/错误分类索引 --------------------
true_labels = test_labels.flatten()
correct_idx = np.where(result == true_labels)[0]
wrong_idx   = np.where(result != true_labels)[0]

print(f"正确分类样本数: {len(correct_idx)}")
print(f"错误分类样本数: {len(wrong_idx)}")

# -------------------- 7. 可视化对比 --------------------
n_samples = 5
np.random.seed(42)
selected_correct = np.random.choice(correct_idx, n_samples, replace=False)
selected_wrong   = np.random.choice(wrong_idx, n_samples, replace=False)

# 第一组：原始图像
fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
for i, idx in enumerate(selected_correct):
    ax = axes[0, i]
    ax.imshow(test_images_original[idx], cmap='gray')
    ax.set_title(f"真实: {true_labels[idx]}\n预测: {int(result[idx])}")
    ax.axis('off')
    ax.text(0.5, -0.1, "原始", transform=ax.transAxes, ha='center', fontsize=10)

for i, idx in enumerate(selected_wrong):
    ax = axes[1, i]
    ax.imshow(test_images_original[idx], cmap='gray')
    ax.set_title(f"真实: {true_labels[idx]}\n预测: {int(result[idx])}", color='red')
    ax.axis('off')
    ax.text(0.5, -0.1, "原始", transform=ax.transAxes, ha='center', fontsize=10)

plt.suptitle("原始图像（预处理前）", y=1.02)
plt.tight_layout()
plt.show()

# 第二组：预处理后图像
fig2, axes2 = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
for i, idx in enumerate(selected_correct):
    ax = axes2[0, i]
    ax.imshow(test_images[idx], cmap='gray')
    ax.set_title(f"真实: {true_labels[idx]}\n预测: {int(result[idx])}")
    ax.axis('off')
    ax.text(0.5, -0.1, "预处理后", transform=ax.transAxes, ha='center', fontsize=10)

for i, idx in enumerate(selected_wrong):
    ax = axes2[1, i]
    ax.imshow(test_images[idx], cmap='gray')
    ax.set_title(f"真实: {true_labels[idx]}\n预测: {int(result[idx])}", color='red')
    ax.axis('off')
    ax.text(0.5, -0.1, "预处理后", transform=ax.transAxes, ha='center', fontsize=10)

plt.suptitle("预处理后图像（高斯模糊+均衡化）", y=1.02)
plt.tight_layout()
plt.show()