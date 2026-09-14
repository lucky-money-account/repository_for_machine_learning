"""画 FID / MiFID 的「逐 epoch 折线图」。

在 notebook 里用法
------------------
    import sys, os
    PROJECT = r"C:\\Users\\moneyforever\\Desktop\\Deep-Learning\\Kaggle"
    if PROJECT not in sys.path:
        sys.path.insert(0, PROJECT)

    from fid_plot import plot_fid
    from IPython.display import Image, display

    # 三种输入都行：
    path = plot_fid(tracker)                              # 1) FIDTracker 对象
    path = plot_fid(tracker.history)                      # 2) history 列表
    path = plot_fid("outputs/dcgan/fid_history.json")     # 3) json 文件路径

    display(Image(path))
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _to_history(source) -> list[dict]:
    """把三种输入统一成 history 列表：FIDTracker / list[dict] / json 路径。"""
    if hasattr(source, "history"):            # FIDTracker 对象
        return list(source.history)
    if isinstance(source, (str, os.PathLike)):  # json 路径
        with open(source, encoding="utf-8") as fh:
            return json.load(fh)
    if isinstance(source, list):              # history 列表
        return source
    raise TypeError("source 必须是 FIDTracker / history 列表 / json 路径")


def plot_fid(
    source,
    out_path: str | None = None,
    keys: tuple[str, ...] = ("fid", "mifid"),
    dpi: int = 130,
    title: str = "FID 逐 epoch 曲线（越低越好）",
) -> str:
    """画折线图并保存，返回图片路径。

    Parameters
    ----------
    source: FIDTracker 对象 / history 列表 / json 路径。
    out_path: 保存路径；默认 ``<项目>/outputs/fid_curve.png``。
    keys: 要画的指标，如 ``("fid", "mifid")``。
    dpi: 分辨率。
    """
    history = _to_history(source)
    if not history:
        raise ValueError("history 为空，没有可画的数据")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(keys):
        xs = [h["epoch"] for h in history if k in h]
        ys = [h[k] for h in history if k in h]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2, label=k.upper())

        # 标出最低点（最佳 epoch）
        best_i = min(range(len(ys)), key=lambda j: ys[j])
        ax.scatter([xs[best_i]], [ys[best_i]], s=130, facecolor="none",
                   edgecolor="red", linewidth=2, zorder=5)
        ax.annotate(f"最佳 {k.upper()}={ys[best_i]:.2f}  @epoch {xs[best_i]}",
                    (xs[best_i], ys[best_i]),
                    textcoords="offset points", xytext=(-12, 12 + i * 20),
                    ha="right", fontsize=9, color="red")

    ax.set_xlabel("epoch")
    ax.set_ylabel("score（越低越好）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(_PROJECT_ROOT, "outputs", "fid_curve.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # 自测：用假数据画一张
    demo = [{"epoch": e, "fid": 400 / (1 + e * 0.3) + 50, "mifid": 420 / (1 + e * 0.3) + 55}
            for e in range(1, 11)]
    print("saved:", plot_fid(demo, out_path=os.path.join(_PROJECT_ROOT, "outputs", "fid_demo.png")))
