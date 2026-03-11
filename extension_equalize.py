import cv2 as cv
import numpy as np

def equalize_hist(im, nbr_bins=256):
    """对一幅灰度图像进行直方图均衡化"""
    # 图像直方图统计
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    # 累积分布函数
    cdf = imhist.cumsum()    # 计算累积和，并归一化到 0~255。
    cdf = 255.0 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)  # 分段线性插值函数
    return im2.reshape(im.shape), cdf

# image pre_processing & data loading
OriginPhoto = cv.imread('digits.png')
image = cv.cvtColor(OriginPhoto, cv.COLOR_BGR2GRAY)
img_equalize_hist, cdf = equalize_hist(image)

# image split
cells = [np.hsplit(row,100) for row in np.vsplit(image,50)]

# data flattening
x = np.array([block.ravel() for row in cells for block in row])

# data division
    # 将 x 重塑为 (10, 500, 400) → 10个数字，每个数字500个样本，每个400特征
x_reshaped = x.reshape(10, 500, 400)

    # 每个数字前250个训练，后250个测试
train = x_reshaped[:, :250, :].reshape(-1, 400).astype(np.float32)   # (2500,400)
test  = x_reshaped[:, 250:, :].reshape(-1, 400).astype(np.float32)   # (2500,400)

# labeling
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# 初始化kNN，在训练数据上进行训练，然后使用k=1的测试数据进行测试
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=1)

# 现在我们检查分类的准确性
# 为此，将结果与test_labels进行比较，并检查哪些是错误的
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(accuracy)