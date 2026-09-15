# PBLF · 人工智能在图像处理方面的应用

PBLF（Project Based Learning Framework）课程《人工智能在图像处理方面的应用》的学习记录。包含 Git/GitHub 入门、视频逐帧图像处理、YOLOv8n 目标检测作业与图像分类延伸实验。

## 内容一览

| 主题 | 文件 | 说明 |
| :--- | :--- | :--- |
| **Git & GitHub** | `Git_and _Github.md` | 从零开始的学习记录：配置用户名/邮箱、生成 SSH 密钥、`init` / 分支 / 合并、SourceTree 可视化工具。含大量操作截图与踩坑反思 |
| **视频图像处理** | `video_processing.ipynb` | 读取视频逐帧处理：灰度化 → 缩放（350×240）→ 高斯滤波 → 直方图均衡化（手写 `equalize_hist`，基于累积分布函数与 `np.interp`） |
| | `video_processing_modified.ipynb` | 改进版：修正 `cv2.imshow` 显示浮点图过白的问题（需 `/255.0`），并对比了 `imshow` 与 `imwrite` 对浮点图的处理差异 |
| **目标检测作业** | `yolov8_learning_notes.md` | YOLOv8n 人物识别完整笔记：输入类型与自动预处理、`Results` 对象结构、`boxes` 各属性（`xyxy` / `xywh` / `conf` / `cls`）、置信度统计与按最低置信度裁剪保存 |
| | `homework.pdf` | 课程作业要求 |
| **图像分类延伸** | `图像分类延伸实验总结.docx` | 图像分类方向的延伸实验总结 |
| **模型权重** | `yolov8n.pt` | YOLOv8 nano 预训练权重（约 6 MB）。**已不再纳入版本管理** —— `ultralytics` 在首次运行 `YOLO("yolov8n.pt")` 时会自动下载，本地缺失不影响运行 |

## 关键代码：YOLOv8n 人物识别

```python
from ultralytics import YOLO
import cv2

img = cv2.imread('picture.jpg')
model = YOLO("yolov8n.pt")          # 权重缺失时自动下载
results = model("picture.jpg")
result = results[0]

person_boxes = []
for box in result.boxes:
    if int(box.cls) == 0:                                    # COCO 类别 0 = person
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        person_boxes.append({'conf': conf, 'size': (x2 - x1, y2 - y1),
                             'location': (x1, y1, x2, y2)})

for i, p in enumerate(person_boxes, 1):
    x1, y1, x2, y2 = p['location']
    print(f"person{i}: 置信度{p['conf']:.2f}, 大小：({p['size'][0]}, {p['size'][1]}, 3)"
          f"| 位置：左上点({x1}, {y1}), 右下点({x2}, {y2})")

print(f"总检测出 {len(person_boxes)} 个人")

# 裁剪置信度最低的人并保存
if person_boxes:
    lowest = min(person_boxes, key=lambda p: p['conf'])
    x1, y1, x2, y2 = lowest['location']
    cv2.imwrite("thelowest.jpg", img[y1:y2, x1:x2])
```

## 关键代码：视频逐帧直方图均衡化

```python
def equalize_hist(im, nbr_bins=256):
    """对一幅灰度图像进行直方图均衡化"""
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum()                             # 累积分布函数
    cdf = 255.0 * cdf / cdf[-1]                       # 归一化到 0~255
    im2 = np.interp(im.flatten(), bins[:-1], cdf)     # 分段线性插值
    return im2.reshape(im.shape).astype(np.uint8), cdf

v = cv2.VideoCapture('video/v.mp4')
while True:
    re, frame = v.read()
    if not re:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 灰度化
    image = cv2.resize(image, (350, 240))                 # 缩放
    filtered_img = cv2.GaussianBlur(image, (3, 3), 1.5)   # 高斯去噪
    img_eq, cdf = equalize_hist(filtered_img)             # 均衡化
    cv2.imshow('result', img_eq)
    if cv2.waitKey(25) & 0xFF == 27:                      # ESC 退出
        break
```

## 环境

```bash
pip install opencv-python numpy matplotlib
pip install ultralytics          # YOLOv8 作业需要
```

## 说明

- 视频文件（`video/`）与图片素材体积较大，未纳入版本管理；运行 notebook 前需自备对应素材。
- 本分支是 PBLF 课程的学习记录分支，与仓库 `main`（《动手学深度学习》学习笔记）无共同历史。
