"""CycleGAN 数据准备与加载（gan_data.py）

把 CelebA 按头发颜色属性拆成 CycleGAN 标准的两个域，并提供可直接调用的 DataLoader。
预处理只做一次，训练时反复调用 get_dataloaders() 即可。

用法示例：
    import sys
    sys.path.append(r'C:\\Users\\moneyforever\\Desktop\\Deep-Learning\\GANs\\CycleGAN')
    from gan_data import prepare_celeba, get_dataloaders, unpaired, denorm

    # 1) 一次性：把 CelebA 拆成 trainA/trainB/testA/testB（拷贝，原图不动）
    stats = prepare_celeba()

    # 2) 训练时：拿 Black / Blond 两个非配对的 loader
    dl = get_dataloaders(batch_size=1)
    for real_Black, real_Blond in unpaired(dl.train_Black, dl.train_Blond):
        ...  # real_Black: (N,3,H,W) in [-1,1]，real_Blond 同理

关于 ImageFolder：
    torchvision.datasets.ImageFolder 要求 root 下有一层「类别子文件夹」，
    而 data/trainA 是扁平目录，直接用它会在 torchvision 0.26 上报
    FileNotFoundError: Couldn't find any class folder in ...
    所以这里用 FlatImageDataset 直接吃扁平目录。
"""

import random
import shutil
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

__all__ = [
    'ROOT',
    'prepare_celeba',
    'FlatImageDataset',
    'build_transform',
    'CycleGANData',
    'get_dataloaders',
    'unpaired',
    'denorm',
]

# 项目根目录取本文件所在目录，因此和 notebook 的工作目录无关
ROOT = Path(__file__).resolve().parent

# 认作图片的扩展名（小写比较）
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# 默认的 A 域 / B 域属性列
ATTR_A = 'Black_Hair'
ATTR_B = 'Blond_Hair'

# 归一化参数：ToTensor 后是 [0,1]，用 (0.5,0.5,0.5) 拉到 [-1,1]，配合 tanh 输出
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------- 预处理


def prepare_celeba(
    root=ROOT,
    attr_a=ATTR_A,
    attr_b=ATTR_B,
    ratio=0.8,
    seed=42,
    max_per_domain=None,
    reset=True,
    verbose=True,
):
    """把 CelebA 按属性拆成 trainA/trainB/testA/testB（拷贝，不动原图）。

    参数:
        root: 项目根目录，默认取本文件所在目录
        attr_a / attr_b: A 域、B 域对应的属性列名
        ratio: 训练集比例，其余进测试集
        seed: 随机种子，保证划分可复现
        max_per_domain: 每个域最多取多少张；None = 全部。想快速跑通可设 2000
        reset: True 则先清空四个目标目录（只删生成目录，原图不受影响）
        verbose: 是否打印进度
    返回:
        dict: 各集合数量、被丢弃的双标签样本数等统计信息
    """
    root = Path(root)
    data_dir = root / 'data'
    src = data_dir / 'img_align_celeba' / 'img_align_celeba'
    csv = data_dir / 'list_attr_celeba.csv'

    # 先校验再干活，避免跑到一半才发现路径或列名不对
    if not csv.exists():
        raise FileNotFoundError(
            f'找不到属性表：{csv}\n'
            f'请从 Kaggle 下载 CelebA 的 list_attr_celeba.csv，放到 {data_dir} 下。'
        )
    if not src.is_dir():
        raise FileNotFoundError(f'找不到原始图片目录：{src}')

    df = pd.read_csv(csv)
    need = {'image_id', attr_a, attr_b}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(
            f'CSV 缺少列：{sorted(missing)}\n实际前 10 列：{list(df.columns[:10])}'
        )

    df = df[['image_id', attr_a, attr_b]]

    # CSV 取值是 1（有此属性）/ -1（无）
    is_a = df[attr_a].eq(1)
    is_b = df[attr_b].eq(1)
    both = is_a & is_b          # 双标签：放哪个域都会污染配对，丢弃

    ids_a = df.loc[is_a & ~both, 'image_id'].tolist()
    ids_b = df.loc[is_b & ~both, 'image_id'].tolist()

    rng = random.Random(seed)

    def split(ids):
        """打乱后按 ratio 切成 (train, test)。"""
        ids = sorted(ids)
        rng.shuffle(ids)
        if max_per_domain is not None:
            ids = ids[:max_per_domain]
        n = int(len(ids) * ratio)
        return ids[:n], ids[n:]

    train_a, test_a = split(ids_a)
    train_b, test_b = split(ids_b)
    splits = {'trainA': train_a, 'testA': test_a, 'trainB': train_b, 'testB': test_b}

    if reset:
        for name in splits:
            d = data_dir / name
            if d.exists():
                if verbose:
                    print('清空', d)
                shutil.rmtree(d)

    for name, ids in splits.items():
        dst = data_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        total = len(ids)
        for i, fname in enumerate(ids, 1):
            shutil.copy2(src / fname, dst / fname)   # 拷贝，源图全程不动
            if verbose and (i % 500 == 0 or i == total):
                print(f'\r{name}: {i}/{total}', end='')
        if verbose:
            print()

    stats = {
        'attr_a': attr_a,
        'attr_b': attr_b,
        'total_a': len(ids_a),
        'total_b': len(ids_b),
        'dropped_double_label': int(both.sum()),
        'trainA': len(train_a),
        'testA': len(test_a),
        'trainB': len(train_b),
        'testB': len(test_b),
    }
    stats['total'] = (
        stats['trainA'] + stats['testA'] + stats['trainB'] + stats['testB']
    )
    return stats


# ---------------------------------------------------------------- 数据集与加载


class FlatImageDataset(Dataset):
    """扁平目录图片数据集（不需要类别子文件夹）。

    torchvision 的 ImageFolder 要求 root 下有一层类别子文件夹，
    而 data/trainA 是扁平的，所以这里自己实现一个最小的 Dataset：
    只负责「列文件 + 读图 + 变换」，分批、打乱、多进程都交给 DataLoader。

    参数:
        root: 图片目录，如 data/trainA
        transform: torchvision 变换；为 None 时返回 PIL.Image
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f'找不到图片目录：{self.root}\n'
                f'请先运行 prepare_celeba() 生成 trainA/trainB/testA/testB。'
            )
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f'{self.root} 下没有找到任何图片')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = default_loader(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_transform(size=256, train=True):
    """构造图像变换。

    参数:
        size: 输出边长（CycleGAN 常用 256，快速实验可用 128）
        train: True 用随机裁剪 + 随机水平翻转；False 用中心裁剪，结果确定
    返回:
        torchvision.transforms.Compose，输出 (3, size, size) 且落在 [-1, 1]
    """
    # Resize 传单个整数表示「短边缩到这个值」，保持长宽比，再裁剪成正方形
    if train:
        return transforms.Compose([
            transforms.Resize(
                int(size * 1.12),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


@dataclass
class CycleGANData:
    """Black / Blond 两个域的训练与测试 DataLoader。

    字段名用域的名字（Black / Blond），对应的目录仍然是 CycleGAN 惯例的
    trainA / trainB / testA / testB，其中 A = Black，B = Blond。
    """

    train_Black: DataLoader
    train_Blond: DataLoader
    test_Black: DataLoader
    test_Blond: DataLoader


def get_dataloaders(
    root=ROOT,
    size=256,
    batch_size=1,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
):
    """构造 A/B 两个域的训练与测试 DataLoader。

    参数:
        root: 项目根目录
        size: 输入边长
        batch_size: CycleGAN 常规用 1（配合 InstanceNorm；配 BatchNorm 会报错）
        num_workers: Windows + Jupyter 建议先用 0，能跑通再调大
        shuffle: 训练集是否打乱（测试集恒为 False）
        pin_memory: 用 GPU 时可开
    返回:
        CycleGANData(train_Black, train_Blond, test_Black, test_Blond)
    """
    data_dir = Path(root) / 'data'
    tf_train = build_transform(size, train=True)
    tf_test = build_transform(size, train=False)

    def make(domain_dir, train):
        ds = FlatImageDataset(data_dir / domain_dir, tf_train if train else tf_test)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if train else False,
            num_workers=num_workers,
            drop_last=train,        # 测试集不丢最后一批
            pin_memory=pin_memory,
        )

    return CycleGANData(
        train_Black=make('trainA', True),      # 目录 trainA  ->  字段 train_Black
        train_Blond=make('trainB', True),      # 目录 trainB  ->  字段 train_Blond
        test_Black=make('testA', False),
        test_Blond=make('testB', False),
    )


def unpaired(loader_a, loader_b):
    """产出「一轮」非配对的 (batch_A, batch_B)，共 len(loader_a) 对，取完即停。

    内部用 cycle 而不是裸 zip：A、B 数量不等（48469 vs 29980），
    裸 zip 会截断到短的那边，等于每轮白扔 18489 张 A 域图片。

    用法（和普通 DataLoader 的 epoch 循环写法完全一致）:
        for epoch in range(n_epochs):
            for real_Black, real_Blond in unpaired(dl.train_Black, dl.train_Blond):
                ...   # 一轮 = len(dl.train_Black) 对

    想要无限流，自己包一层 cycle(unpaired(...)) 即可。
    """
    return zip(loader_a, cycle(loader_b))


def denorm(t):
    """把 Normalize((0.5,)*3) 之后的张量还原到 [0, 1]，方便 imshow/保存。"""
    return (t * 0.5 + 0.5).clamp(0, 1)
