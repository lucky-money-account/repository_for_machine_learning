# 利用yolov8n模型实现人物识别

## :question:yolov8n模型的基本介绍及输入输出：

### 1. 输入

#### 1.1支持的输入类型

YOLOv8 模型可以接受多种输入：

- 单张图像文件路径（如 `"image.jpg"`）
- 图像数组（numpy 数组，形状 `(H,W,3)`，BGR 或 RGB 均可）
- 视频文件路径
- 视频流（如摄像头）
- 图像目录
- 网页 URL
- PIL 图像对象

在代码中，你只需要将输入传给模型：

```python
results = model('image.jpg')
# 或
results = model(img_array)   # img_array 为 numpy 数组
```

#### 1.2 预处理

模型会自动完成预处理：

- 将输入缩放到模型要求的尺寸（默认为 640×640）

- 归一化（除以 255）

- 颜色通道转换（BGR → RGB，如果输入是 BGR 的话）

- 转换为 PyTorch 张量

  

### 2. 输出

YOLOv8 的输出是一个 `results` 对象（实际上是一个 `Results` 对象列表，每个元素对应一张输入图像）。对于单张图像，通常取 `results[0]`。

#### 2.1 主要属性

`Results` 对象包含以下重要属性和方法：

| 属性/方法    | 说明                                 |
| :----------- | :----------------------------------- |
| `orig_img`   | 原始图像（numpy 数组，BGR 格式）     |
| `orig_shape` | 原始图像尺寸 `(height, width)`       |
| `boxes`      | 检测框对象，包含所有检测到的目标信息 |
| `masks`      | 实例分割掩码（如果模型是分割模型）   |
| `keypoints`  | 关键点（如果模型是姿态估计模型）     |
| `probs`      | 分类概率（如果模型是分类模型）       |
| `save()`     | 保存结果图像到文件                   |
| `plot()`     | 返回带标注的图像数组（RGB）          |
| `verbose`    | 打印检测信息                         |

#### 2.2 `boxes` 对象详解

`results[0].boxes` 是一个 `Boxes` 对象，它包含了所有检测框的信息，可以通过以下属性获取：

| 属性         | 说明                                        | 形状     |
| :----------- | :------------------------------------------ | :------- |
| `boxes.xyxy` | 边界框坐标 `[x1, y1, x2, y2]`（左上、右下） | `(N, 4)` |
| `boxes.xywh` | 边界框坐标 `[x, y, w, h]`（中心点+宽高）    | `(N, 4)` |
| `boxes.conf` | 置信度（每个框的得分）                      | `(N,)`   |
| `boxes.cls`  | 类别 ID（整数，从0开始）                    | `(N,)`   |
| `boxes.id`   | 跟踪 ID（如果启用了跟踪）                   | `(N,)`   |

注意：这些属性返回的是 PyTorch 张量（在 GPU 上），通常需要转换到 CPU 并转为 numpy 或 Python 数值：



```python
xyxy = box.xyxy[0].cpu().numpy()  # 第一个框的坐标
#如果需要整数坐标，通常采用以下写法：
#xyxy = box.xyxy[0].cpu().numpy().astype(int)
#xyxy = map(int, box.xyxy[0].cpu().numpy())
conf = float(box.conf[0])          # 第一个框的置信度
cls_id = int(box.cls[0])           # 第一个框的类别ID
```

#### 2.3 类别名称

模型内置了 COCO 数据集的 80 个类别名称，可通过 `model.names` 获取一个字典，如 `{0: 'person', 1: 'bicycle', ...}`。



## :fearful:作业实现要求：

![631fb0c4e75868276ea6eea69cc10fea](https://raw.githubusercontent.com/lucky-money-account/picgo_resources_photos/main/pblfhomework2.png)



## :cheese:基本实现代码：

```python
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
img = cv2.imread('picture.jpg')  

# 加载模型
model = YOLO("yolov8n.pt")
# 进行检测
results = model("picture.jpg")
# 接收结果
result = results[0]

# 提取检测结果
person_boxes = []
for box in result.boxes:
    if int(box.cls) == 0:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        w, h = x2 - x1, y2 - y1
        person_boxes.append({'conf': conf, 'size':(w,h), 'location':(x1, y1, x2, y2)})

# 输出
for i, p in enumerate(person_boxes, 1):		#枚举类型遍历
    x1, y1, x2, y2 = p['location']    #注意注意！解包后才能正确读入（qwq）
    print(f"person{i}: 置信度{p['conf']:.2f}, 大小：({p['size'][0]}, {p['size'][1]}, 3)| 位置：左上点({x1}, {y1}), 右下点({x2}, {y2})")
print()	#输出一行空白行
print(f"总检测出 {len(person_boxes)} 个人") 

# 保存图像
if not person_boxes:
    print("未检测到人")
else:
    lowest_person = person_boxes[0]   
    for p in person_boxes[1:]:        
        if p['conf'] < lowest_person['conf']:
            lowest_person = p
x1, y1, x2, y2 = lowest_person['location']
crop = img[y1:y2, x1:x2]    #依据坐标进行裁剪
_ =cv2.imwrite("thelowest.jpg", crop)
# 申明：如果不用一个变量接收返回值，因为jupyter的输出特性，从而单元格最后一个表达式的值会被打印
# 有可能会在最后一行输出一个“True”
# （当然也可以调用内置的min（）函数来查找置信度最低的人）
'''
lowest_person = min(person_boxes, key=lambda p: p['conf'])
x1, y1, x2, y2 = lowest_person['location']
crop = img[y1:y2, x1:x2]   # 注意切片顺序 [行, 列]
'''

## 运行结果
<div style="display: flex; justify-content: center; gap: 30px;">
    <div style="text-align: center;">
        <img src="attachment:63b97112-82b2-45f6-a6ff-d22bcc83e3e9.jpg" style="max-height: 200px;">
        <br><b> (a) 原图 </b>
    </div>
    <div style="text-align: center;">
        <img src="attachment:ba704318-dbde-4e0e-9ca3-227abcafae39.jpg" style="max-height: 200px;">
        <br><b> (b) person_NULL(置信度NULL） </b>
    </div>
</div>
```



## :smile:预期输出结果示例：

```
image 1/1 C:\Users\moneyforever\Desktop\PBLF\  code\picture.jpg: 448x640 5 persons, 13.3ms
Speed: 4.2ms preprocess, 13.3ms inference, 1.3ms postprocess per image at shape (1, 3, 448, 640)
person1: 置信度0.90, 大小：(185, 538, 3)| 位置：左上点(129, 304), 右下点(314, 842)
person2: 置信度0.88, 大小：(173, 497, 3)| 位置：左上点(360, 327), 右下点(533, 824)
person3: 置信度0.88, 大小：(153, 436, 3)| 位置：左上点(584, 373), 右下点(737, 809)
person4: 置信度0.87, 大小：(128, 395, 3)| 位置：左上点(745, 402), 右下点(873, 797)
person5: 置信度0.86, 大小：(117, 371, 3)| 位置：左上点(893, 413), 右下点(1010, 784)

总检测出 5 个人
```



## :thought_balloon:一些思考：

该模型很全面，对各种物体都有一定的识别能力，eg.

![9a7f2cae143fed0966a0d13782156803](https://raw.githubusercontent.com/lucky-money-account/picgo_resources_photos/main/pblfyolodemo.png)

虽然这种能力有时也会产生神奇的效果：（莫名其妙把背景幕布里的人形识别上了哈哈哈:laughing:）

![c3af2e285a9ba62934cf45219afa9474](https://raw.githubusercontent.com/lucky-money-account/picgo_resources_photos/main/pblfyolofunnydemo.png)

每个方框的数据都被切割且尽数保存，对于图像识别这个领域影响甚大，个人认为这项技术太有意思了，而且上手简单，且准确率很高，是一门不可多得的优秀技术。



## 总结：

​		通过本次作业，我学习了如何调用yolov8n模型进行人物识别以及置信度检测的方法，并对照老师提供的代码以及网上的一些资料对该模型进行学习，学习了该模型的输入以及输出的参数，并了解到新的语法知识，同时复习了python语法中的字典和列表部分的知识。同时对于代码中出现的异常也进行了原因分析，eg.莫名其妙多输出了一个“True”。

