# 使用kNN进行手写数据光学字符识别(OCR)

## *原理：*

### k近邻算法:smile:：

（故事化）通过比较数量和距离，来确定新来者应该归类到哪一类家族中。

（理论化）上述故事中的距离代表权重，距离越近权重越大，通过数量+权重的组合实现特征归类，从而达到特征识别的效果。实际处理中，两幅图之间的像素差异抽象为距离，将距离从小到大排序，选出前 k 个（即最相似的 k 个训练样本）。查看这 k 个邻居的标签，分别统计每个标签出现了多少次。例如 k=3 时，三个邻居的标签分别是 3、5、3，那么数字 3 出现了 2 次，数字 5 出现了 1 次。数量最多的那个标签（数字 3）就是预测结果。



## *代码分析​*:computer:：

### 所需要的库:statue_of_liberty::

cv2（包含所需的knn模型代码等）

numpy（包含图像处理，数据处理的函数）



### 涉及的函数:camera_flash:：

```python
cv.imread()
```

**读取图像**



```python
cv.cvtColor(PHOTOFILE, cv.COLOR_BGR2GRAY)
```

**灰度化处理图像**



```python
np.hsplit(ary, indices_or_sections)
```

**水平切割**（按列分割，eg. |1|2|3| ）

形参1为传入的数组，形参2为分割的块数or断点位置：

- 如果是一个**整数**，比如 `N`，则要求数组的行数必须能被 `N` 整除，此时会将数组**等分成 N 个**子数组（每个子数组行数相同）。

- 如果是一个**列表/元组**，比如 `[i, j]`，则会在指定的行索引位置进行分割。索引从0开始，分割后得到 `[ary[:i], ary[i:j], ary[j:]]`。

  

```
np.vsplit(ary, indices_or_sections)
```

**竖直分割**（按行分割）。



```python
np.array([block.ravel() for row in cells for block in row])
```

**数据扁平化：**

（`flatten()`、`ravel()`、`reshape(-1)`。）并生成特征矩阵（一维数组）

此处采用推导式遍历，写法更简单。



```python
x_reshaped = x.reshape(10, 500, 400)
```

**重塑化：**

依据内存顺序将原数组重塑为（10，500，400）->10个数字，每个数字500个样本，每个400特征（像素）

（ps.能够采用这种简单的方式分割，主要是因为原图digits.png的顺序极其有序：

![digits](https://raw.githubusercontent.com/lucky-money-account/picgo_resources_photos/main/digits.png)



```python
train = x_reshaped[:, :250, :].reshape(-1, 400).astype(np.float32)   # (2500,400)
test  = x_reshaped[:, 250:, :].reshape(-1, 400).astype(np.float32) 
```

**划分训练集&测试集：**

每个数字前250个用作训练，后250个用作测试。

reshape将得到的三维数组转化成knn能够处理的二维数组

（-1，400）：-1代表自动计算，400保留像素特征及数量不变

astype(np.float32)：OpenCV 的机器学习模块要求输入数据为 `float32` 类型（单精度浮点数）。虽然原始像素值是 `uint8`（0-255），但转换为浮点数可以避免后续计算中的精度损失和类型不匹配错误。



```python
k = np.arange(10)		#等差数列生成，创建一个包含数字0-9的一维数组。
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()
```

**图像标签化：**

先生成一个一维数组，重复250个0，250个1......，然后[:, np.newaxis]强制转化为二维数组（符合knn要求）



```
knn = cv.ml.KNearest_create()
```

**创建knn模型：**

- **作用**：调用 OpenCV 机器学习模块的 `KNearest_create()` 函数，创建一个空的 k-近邻分类器对象。

- **返回**：一个 `cv2.ml_KNearest` 类型的实例，后续可通过该实例调用训练和预测方法。

  

```
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
```

**训练模型：**

- **参数**：

  - `train`：训练数据，形状为 `(2500, 400)` 的二维数组，每行是一个样本（手写数字图像），每列是一个特征（像素值）。
  - `cv.ml.ROW_SAMPLE`：指定训练数据的布局方式，表示每行是一个样本（Row Sample），这是最常用的格式。
  - `train_labels`：训练标签，形状为 `(2500, 1)` 的二维数组，每行对应一个样本的真实数字（0~9）。

- **作用**：将训练数据和标签“喂”给模型，kNN 会**记住**所有训练样本及其标签（kNN 属于懒惰学习，训练阶段实际上只存储数据，不进行参数拟合）。

  

```
ret, result, neighbours, dist = knn.findNearest(test, k=1)
```

**预测测试集：**

- **参数**：
  - `test`：测试数据，形状为 `(2500, 400)` 的二维数组。
  - `k=1`：指定找 **1 个最近邻**（即只考虑最相似的一个训练样本）。
- **返回值**（四个）：
  - **`ret`**（或 `retval`）：预测结果矩阵，与 `result` 完全相同，通常直接忽略或作为冗余输出。
  - **`result`**：**主要预测结果**，形状为 `(2500, 1)` 的二维数组，每个元素是模型对测试样本的预测数字。
  - **`neighbours`**：最近邻的标签，形状为 `(2500, 1)`（因为 `k=1`，只有一个邻居），表示每个测试样本最相似的那个训练样本的类别。
  - **`dist`**：到最近邻的距离，形状为 `(2500, 1)`，表示每个测试样本与其最近邻的欧氏距离。



```python
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(accuracy)
```

**分析准确度：**

调用np.count_nonzero()函数检测matches数组中的非零项个数（匹配的个数），然后计算accuracy。



## *结果：*:artificial_satellite:

## ![屏幕截图 2026-03-11 201219](https://raw.githubusercontent.com/lucky-money-account/picgo_resources_photos/main/knn_result.png)



## *与其他图像处理技术的交叉思考:thought_balloon:*

### +图像均衡化技术：

无提升，准确率仍为92.2%



### +高斯去噪：

无提升



### +二值化处理：

无提升



## *k值的改变带来的影响*：

K										Accuracy

1											92.2

2											91.04											

3											92.36

4											92.56

5											92.0

...

k值在故事中扮演“有资格投票的人数”的“角色”，k值越大，考虑的图像个数越多，可能导致不相关的类别混入干扰结果。

- k 值小 → 模型复杂，容易过拟合。
- k 值大 → 模型简单，容易欠拟合。
- 合适 k 值需通过实验确定。



## *kNN复现总结:*:smiley:


本次实验复现了经典机器学习模型kNN，作为本人进入该领域的第一步练习，在其他大佬制作的文档的帮助下，我成功编写出整个框架（鸣谢：[OpenCV：使用kNN进行手写数据光学字符识别(OCR) - OpenCV 计算机视觉库](https://docs.opencv.ac.cn/4.11.0/d8/d4b/tutorial_py_knn_opencv.html)），在deepseek帮助下我理解了每个函数的用法并制作出了整份学习笔记。同时，作为课内PBLF“人工智能在图像处理方面的应用”的课外自主延伸内容，我将课内学到的几个处理技术融入到代码中，但是很可惜，官方图太过清楚了，不需要过多的处理:sob:。kNN作为一门老思想，仍能做到较高的准确度，足以见机器学习的高准确度和极大美丽。下一步可能就是引入新的数据集进行模型改进or学习新的模型方法~:kissing_heart:
