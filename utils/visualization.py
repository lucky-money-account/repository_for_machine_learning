"""可视化套路函数库（utils/visualization.py）

把第 4 章反复出现的「画 3D 曲面 / 损失曲线」固定模板封装成函数，
以后直接 import 调用，不用再手写 meshgrid / stack / reshape / plot_surface。

用法示例：
    import sys
    sys.path.append('C:/Users/moneyforever/Desktop/Deep-Learning')  # 主目录
    from utils.visualization import plot_3d_surface, plot_loss_history

    net = ...           # 训练好的模型（输入 2 维）
    plot_3d_surface(net, (-3, 3), (-3, 3), title='模型拟合')

    plot_loss_history(history)   # 画单条损失曲线
    plot_loss_history({'lr=0.1': h1, 'lr=0.01': h2})  # 多条对比
"""

import torch
import matplotlib.pyplot as plt


def make_grid(x1_lim, x2_lim, n=100):
    """在 [x1_lim] × [x2_lim] 区域生成 n×n 网格。

    参数:
        x1_lim, x2_lim: 范围元组，如 (-3, 3)
        n: 每条轴取多少个点

    返回:
        X1, X2: 形状 (n, n) 的坐标网格（画图用）
        grid:   形状 (n*n, 2) 的样本列表（喂模型用，每行一个 (x1, x2)）
    """
    x1 = torch.linspace(x1_lim[0], x1_lim[1], n)
    x2 = torch.linspace(x2_lim[0], x2_lim[1], n)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
    return X1, X2, grid


def predict_on_grid(model, grid, n=100):
    """把 (n*n, 2) 的样本列表喂给模型，预测后还原成 (n, n) 网格。

    封装了「预测 → reshape 回网格」这一步，自动关梯度。
    """
    with torch.no_grad():
        return model(grid).reshape(n, n)


def plot_3d_surface(model, x1_lim, x2_lim, n=100, title=''):
    """一站式：喂模型画 3D 曲面。

    封装了 meshgrid → 预测 → reshape → plot_surface 全流程。
    """
    X1, X2, grid = make_grid(x1_lim, x2_lim, n)
    Z = predict_on_grid(model, grid, n)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(), cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if title:
        ax.set_title(title)
    plt.show()
    return fig, ax


def plot_true_vs_pred(true_fn, model, x1_lim, x2_lim, n=100):
    """并排对比「真实函数」和「模型预测」的两张 3D 曲面。

    true_fn: 真实函数，如 lambda x1, x2: torch.sin(x1) * torch.cos(x2)
    model:   训练好的模型（输入 2 维、输出 1 维）
    """
    X1, X2, grid = make_grid(x1_lim, x2_lim, n)
    Z_true = true_fn(X1, X2)
    Z_pred = predict_on_grid(model, grid, n)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1.numpy(), X2.numpy(), Z_true.numpy(), cmap='viridis')
    ax1.set_title('真实函数')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X1.numpy(), X2.numpy(), Z_pred.numpy(), cmap='viridis')
    ax2.set_title('模型预测')
    plt.show()
    return fig


def plot_loss_history(histories, title='', yscale='linear'):
    """画损失曲线。

    参数:
        histories: list（单条曲线）或 dict（多条，key 自动做图例）
        yscale: 'linear'（普通）或 'log'（对数，loss 跨数量级时用）
    """
    plt.figure(figsize=(8, 5))
    if isinstance(histories, dict):
        for label, hist in histories.items():
            plt.plot(hist, label=str(label))
    else:
        plt.plot(histories)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale(yscale)
    if title:
        plt.title(title)
    if isinstance(histories, dict):
        plt.legend()
    plt.show()


class Animator:
    """增量绘制训练曲线（类似 d2l 的 Animator）。

    用法：
        anim = Animator(xlabel='epoch', ylabel='loss',
                        legend=['train loss', 'test acc'], yscale='log')
        for epoch in range(num_epochs):
            # ... 训练 ...
            anim.add(epoch, [train_loss, test_acc])   # y 可以是标量或列表（多曲线）
        anim.show()   # 脚本环境显式显示；Jupyter 里 add 完会自动显示

    参数:
        xlabel, ylabel: 坐标轴标签
        legend: 曲线名列表（和 add 的 y 长度对应）
        xlim, ylim: 坐标范围（可选）
        yscale: 'linear' / 'log'
        figsize: 画布大小
    """

    def __init__(self, xlabel='epoch', ylabel=None, legend=None,
                 xlim=None, ylim=None, yscale='linear', figsize=(8, 5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend if legend is not None else []
        self.xlim = xlim
        self.ylim = ylim
        self.yscale = yscale
        self.X = []   # 存 x 值（每个数据点的横坐标）
        self.Y = []   # 存 y 值（list of list，每条曲线一列）

    def add(self, x, y):
        """添加一个数据点。y 是标量 → 单条曲线；y 是列表 → 多条曲线。"""
        if not hasattr(y, '__len__'):   # 标量 → 包装成单元素列表
            y = [y]
        self.X.append(x)
        if not self.Y:
            self.Y = [[] for _ in range(len(y))]
        for i, yi in enumerate(y):
            self.Y[i].append(yi)
        self._redraw()

    def _redraw(self):
        """清空并重绘（增量更新）。"""
        self.ax.clear()
        for i, ys in enumerate(self.Y):
            label = self.legend[i] if i < len(self.legend) else None
            self.ax.plot(self.X[:len(ys)], ys, label=label)
        self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        self.ax.set_yscale(self.yscale)
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
        if self.legend:
            self.ax.legend()
        self.fig.canvas.draw()
        # Jupyter 动态刷新：清旧图 + 显示新图（否则 %matplotlib inline 只在 cell 结束显示一次）
        try:
            from IPython import display
            display.clear_output(wait=True)
            display.display(self.fig)
        except Exception:
            pass   # 非 notebook 环境（纯脚本）则跳过，最后用 show() 显示

    def show(self):
        """脚本/非交互环境显式显示图。"""
        plt.show()
        return self.fig
