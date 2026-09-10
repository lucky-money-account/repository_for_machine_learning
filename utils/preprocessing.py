"""数据预处理通用函数库（utils/preprocessing.py）

只收录「通用常用」的预处理套路，与具体业务（户型/房价等）解耦。
特定业务的解析函数在自己 notebook 里写，不要堆进这里。

用法示例：
    import sys
    sys.path.append('C:/Users/moneyforever/Desktop/Deep-Learning')
    from utils.preprocessing import one_hot, fill_missing, split_train_test, standardize, to_tensor
"""

import re
import torch
import pandas as pd


def regex_num(s, pattern, default=0):
    """用正则从字符串中提取第一个数字，匹配不到返回 default。

    例：regex_num('价格 3 万', r'(\\d+)') -> 3
    """
    m = re.search(pattern, str(s))
    return int(m.group(1)) if m else default


# ---------- 类别与缺失 ----------

def one_hot(df, cols, fillna='未知'):
    """对指定类别列做 one-hot，返回拼接后的新 DataFrame（不改原 df）。

    参数:
        df: DataFrame
        cols: 列名列表
        fillna: 缺失类别先填成该值（默认「未知」），避免整行全 0
    """
    out = df.copy()
    for col in cols:
        filled = out[col].fillna(fillna)
        dummies = pd.get_dummies(filled, prefix=col)
        out = pd.concat([out, dummies], axis=1)
    return out


def fill_missing(df, cols, method='median'):
    """对数值列填缺失值（默认中位数）。返回新 DataFrame（不改原 df）。

    method: 'median' / 'mean' / 0（填指定值）
    """
    out = df.copy()
    for col in cols:
        # 先把非数字（如「暂无」）转成 NaN
        out[col] = pd.to_numeric(out[col], errors='coerce')
        if method == 'median':
            out[col] = out[col].fillna(out[col].median())
        elif method == 'mean':
            out[col] = out[col].fillna(out[col].mean())
        else:
            out[col] = out[col].fillna(method)
    return out


# ---------- 划分与标准化 ----------

def split_train_test(n, ratio=0.8, seed=None):
    """返回 (train_idx, test_idx)，按 ratio 随机划分 0..n-1。"""
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.randperm(n).tolist()
    n_train = int(ratio * n)
    return idx[:n_train], idx[n_train:]


def standardize(train, test=None, num_cols=None):
    """用「训练集」的均值/标准差做 z-score 标准化（测试集复用训练统计量，防信息泄漏）。

    参数:
        train: 训练集 DataFrame
        test:  测试集 DataFrame（可选；不给则只标准化训练集）
        num_cols: 要标准化的数值列；不给则对 train 所有数值列标准化
    返回:
        (train_std, test_std, (mean, std))，test_std 在 test=None 时为 None
    """
    if num_cols is None:
        num_cols = train.select_dtypes(include='number').columns.tolist()
    mean = train[num_cols].mean()
    std = train[num_cols].std()

    train_std = train.copy()
    train_std[num_cols] = (train_std[num_cols] - mean) / std

    test_std = None
    if test is not None:
        test_std = test.copy()
        test_std[num_cols] = (test_std[num_cols] - mean) / std
    return train_std, test_std, (mean, std)


# ---------- 转张量 ----------

def to_tensor(df, dtype='float32'):
    """DataFrame -> torch.FloatTensor（用 .values 取底层 numpy）。"""
    return torch.tensor(df.values.astype(dtype))
