"""生成「GAN 训练与评估 · 全流程图」。

在 notebook 里用法
------------------
    import sys, os
    PROJECT = r"C:\\Users\\moneyforever\\Desktop\\Deep-Learning\\Kaggle"
    if PROJECT not in sys.path:
        sys.path.insert(0, PROJECT)

    from flow_diagram import draw_flow
    from IPython.display import Image, display

    path = draw_flow()              # 默认存到 <项目>/outputs/flow.png
    display(Image(path))            # 在 notebook 里直接显示

也可自定义路径与分辨率：
    draw_flow("outputs/my_flow.png", dpi=150)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def draw_flow(out_path: str | None = None, dpi: int = 130) -> str:
    """画出全流程图并保存，返回图片路径。

    Parameters
    ----------
    out_path: 保存路径；默认 ``<项目>/outputs/flow.png``。
    dpi: 输出分辨率。
    """
    # 中文字体（Windows）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 13)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                     linewidth=2, edgecolor="#333", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#333", style="->"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=22, linewidth=2.2, color=color))

    def label(x, y, text, color="#555", fs=9):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)

    # ===== 标题 =====
    ax.text(6, 12.5, "GAN 训练与评估 · 全流程图",
            ha="center", fontsize=17, fontweight="bold")

    # ===== 训练部分 =====
    box(0.5, 10.6, 2.6, 1.0, "随机噪声 z\n(100维)", "#E3F2FD")
    box(4.7, 10.6, 2.6, 1.0, "真实狗图\n.pt (20580张)", "#FFF3E0")
    box(0.5, 8.6, 2.6, 1.1, "① 生成器 G\n噪声 → 假图", "#BBDEFB")
    box(4.7, 8.6, 2.6, 1.1, "② 判别器 D\n图 → 真假概率", "#FFE0B2")
    box(8.8, 8.6, 2.6, 1.1, "输出: 0~1\n概率分数", "#F0F0F0", fs=10)

    arrow(1.8, 10.6, 1.8, 9.7)               # z -> G
    arrow(4.7, 10.6, 5.3, 9.7)               # 真实图 -> D
    arrow(3.1, 9.15, 4.7, 9.15, "#2E7D32")   # G -> D 假图
    label(3.9, 9.45, "假图", "#2E7D32")
    arrow(7.3, 9.15, 8.8, 9.15, "#333")      # D -> 输出
    arrow(5.0, 8.6, 5.0, 7.85, "#D32F2F")    # 反向传播
    label(6.6, 8.2, "反向传播更新 G 和 D\n(交替训练)", "#D32F2F", fs=9)

    # ===== 训练循环框 =====
    ax.add_patch(FancyBboxPatch((0.3, 7.6), 7.3, 2.3, boxstyle="round,pad=0.15",
                 linewidth=2, edgecolor="#D32F2F", facecolor="none", linestyle="--"))
    ax.text(0.5, 9.62, "训练循环 (每个 batch)", fontsize=9, color="#D32F2F",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    # ===== 评估部分 =====
    arrow(3.0, 7.6, 3.0, 6.6, "#1565C0")
    label(4.3, 7.1, "每 N 个 epoch", "#1565C0")
    box(0.5, 5.3, 5.0, 1.2, "③ FID 评估\n真实图特征(缓存) vs 生成图特征", "#C8E6C9")
    box(0.5, 3.5, 5.0, 1.2, "④ FID 曲线\nFID / MiFID 随 epoch 变化", "#A5D6A7")
    arrow(3.0, 5.3, 3.0, 4.7)
    box(0.5, 1.7, 5.0, 1.2, "⑤ 选曲线最低点 = 最佳模型\n(保存 checkpoint)", "#81C784")
    arrow(3.0, 3.5, 3.0, 2.9)

    # ===== 指标说明 =====
    box(7.0, 5.3, 4.5, 2.4,
        "指标怎么读\n\nFID:  越小越好\nMiFID: FID × 记忆惩罚\n"
        "Precision: 质量\nRecall: 多样性\n\n参考: DCGAN 约 55~59",
        "#FFF9C4", fs=10)

    # ===== 比赛流程 =====
    ax.text(6, 1.25, "比赛流程 (本地验证 vs 提交)",
            ha="center", fontsize=12, fontweight="bold")
    box(0.5, 0.15, 2.6, 0.8, "本地试验\n(无限次)", "#B3E5FC", fs=10)
    box(3.6, 0.15, 2.6, 0.8, "提交\n(有限次)", "#FFCCBC", fs=10)
    box(6.7, 0.15, 2.6, 0.8, "排行榜\n(公开/私榜)", "#D1C4E9", fs=10)
    box(9.8, 0.15, 1.8, 0.8, "回本地\n迭代", "#C8E6C9", fs=10)
    arrow(3.1, 0.55, 3.6, 0.55)
    arrow(6.2, 0.55, 6.7, 0.55)
    arrow(9.3, 0.55, 9.8, 0.55)

    # 回环箭头
    ax.add_patch(FancyArrowPatch((10.7, 0.15), (10.7, -0.35), arrowstyle="-",
                 linewidth=2, color="#333"))
    ax.add_patch(FancyArrowPatch((10.7, -0.35), (1.8, -0.35), arrowstyle="-",
                 linewidth=2, color="#333"))
    ax.add_patch(FancyArrowPatch((1.8, -0.35), (1.8, 0.15), arrowstyle="->",
                 mutation_scale=20, linewidth=2, color="#333"))

    # ===== 保存 =====
    if out_path is None:
        out_path = os.path.join(_PROJECT_ROOT, "outputs", "flow.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("saved:", draw_flow())
