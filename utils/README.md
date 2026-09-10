# utils 工具库

> 把各章反复出现的「固定套路代码」封装成可复用函数，分门别类存放。
> 以后写 notebook 时直接 `import` 调用，不用重复手写模板。

## 目录结构

| 文件 | 内容 | 状态 |
|---|---|---|
| `visualization.py` | 画 3D 曲面、损失曲线、增量绘图 Animator | ✅ 已添加 |
| `preprocessing.py` | 数据预处理套路（正则/one-hot/缺失/划分/标准化/转张量） | ✅ 已添加 |
| `training.py` | 训练循环封装 | ⬜ 待加 |
| `metrics.py` | 评估指标（RMSE/MAE/准确率） | ⬜ 待加 |

## 使用方式

在 notebook 开头加入主目录路径，然后 import：

```python
import sys
sys.path.append('C:/Users/moneyforever/Desktop/Deep-Learning')  # 主目录
from utils.visualization import plot_3d_surface, plot_true_vs_pred, plot_loss_history
```

（如果 notebook 已经能直接访问主目录，可省略 `sys.path.append` 那行。）

## visualization.py 现有函数

| 函数 | 作用 |
|---|---|
| `make_grid(x1_lim, x2_lim, n)` | 生成网格坐标 + 样本列表 |
| `predict_on_grid(model, grid, n)` | 喂模型预测并还原网格 |
| `plot_3d_surface(model, x1_lim, x2_lim)` | 一站式画模型 3D 曲面 |
| `plot_true_vs_pred(true_fn, model, ...)` | 对比真实 vs 预测 |
| `plot_loss_history(histories)` | 画损失曲线（单条/多条） |
| `Animator`（类） | 训练中增量画曲线（像 d2l 的 Animator，`add(x, y)` 逐点添加） |

## preprocessing.py 现有函数

| 函数 | 作用 |
|---|---|
| `regex_num(s, pattern, default)` | 正则提取数字，匹配不到填 default |
| `one_hot(df, cols)` | 类别列 one-hot |
| `fill_missing(df, cols, method)` | 数值列填缺失（中位数/均值/指定值） |
| `split_train_test(n, ratio)` | 随机划分 0..n-1，返回 (train_idx, test_idx) |
| `standardize(train, test, num_cols)` | 用训练集统计量 z-score（测试集复用，防泄漏） |
| `to_tensor(df, dtype)` | DataFrame → torch 张量 |
