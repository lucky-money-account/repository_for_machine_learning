"""CycleGAN 模型定义（model.py）

按官方 junyanz/pytorch-CycleGAN-and-pix2pix 的默认配置，展开成扁平的 nn.Sequential。

    生成器 netG = resnet_9blocks, ngf=64
    判别器 netD = basic (PatchGAN), ndf=64, n_layers_D=3
    norm  = InstanceNorm2d, G/D 都用
    初始化 = normal, gain=0.02

用法示例：
    import sys
    sys.path.append(r'C:\\Users\\moneyforever\\Desktop\\Deep-Learning\\GANs\\CycleGAN')
    from model import (GeneratorBLOND, GeneratorBLACK,
                       DiscriminatorBLOND, DiscriminatorBLACK, weights_init)

    G_AB = GeneratorBLOND().cuda()        # 黑发 -> 金发
    G_BA = GeneratorBLACK().cuda()        # 金发 -> 黑发
    D_A  = DiscriminatorBLACK().cuda()    # 判黑发真假
    D_B  = DiscriminatorBLOND().cuda()    # 判金发真假

    for net in (G_AB, G_BA, D_A, D_B):
        net.apply(weights_init)

尺寸：
    生成器  (n, 3, 256, 256) -> (n, 3, 256, 256)   值域 [-1, 1]（Tanh）
    判别器  (n, 3, 256, 256) -> (n, 1, 30, 30)     每个点判一块 70x70 感受野

为什么残差块要单独写一个类：
    nn.Sequential 只能顺序串联，表达不了 x + conv_block(x) 这个加法。
    所以 ResnetBlock 必须是 nn.Module 子类。
"""

import torch.nn as nn

__all__ = [
    'ResnetBlock',
    'GeneratorBLOND',
    'GeneratorBLACK',
    'DiscriminatorBLOND',
    'DiscriminatorBLACK',
    'weights_init',
]


class ResnetBlock(nn.Module):
    """残差块：out = x + conv_block(x)

    结构（两层 3x3 卷积，通道数不变）：
        ReflectionPad2d(1) + Conv2d(dim,dim,3) + InstanceNorm2d + ReLU
        ReflectionPad2d(1) + Conv2d(dim,dim,3) + InstanceNorm2d

    三个要点：
        1. 第二个 Conv 后面不加激活，加法后面也不加，否则就不是残差了
        2. padding 用 ReflectionPad2d 写在外面，Conv 的 padding=0
        3. 两个 Conv 通道数都不变，所以可以串联任意多个，形状永远不变
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),           # (n, dim, H, W)
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),           # (n, dim, H, W)
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


def GeneratorBLOND():
    """金发生成器 G_AB：黑发 -> 金发。"""
    return nn.Sequential(
        nn.ReflectionPad2d(3),                      # 镜面 padding，图像生成常用
        nn.Conv2d(3, 64, 7, 1, 0),                  # (n, 64, 256, 256)
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, 3, 2, 1),                # (n, 128, 128, 128)
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, 3, 2, 1),               # (n, 256, 64, 64)
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),

        # ---------- 9 个残差块：尺寸和通道都不变 ----------
        ResnetBlock(256),                           # (n, 256, 64, 64)
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),

        # ---------- 上采样 x2 ----------
        # output_padding=1 不能省，否则尺寸会变成 63 / 255
        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),   # (n, 128, 128, 128)
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),    # (n, 64, 256, 256)
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 3, 7, 1, 0),                  # (n, 3, 256, 256)
        nn.Tanh(),                                  # 值域 [-1, 1]，和数据的归一化对齐
    )


def GeneratorBLACK():
    """黑发生成器 G_BA：金发 -> 黑发。

    和 GeneratorBLOND 结构完全相同，只是独立初始化的一份权重。
    CycleGAN 的两个生成器是「同架构、不同参数」，不是两种网络。
    """
    return nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, 64, 7, 1, 0),                  # (n, 64, 256, 256)
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, 3, 2, 1),                # (n, 128, 128, 128)
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, 3, 2, 1),               # (n, 256, 64, 64)
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),

        ResnetBlock(256),                           # (n, 256, 64, 64)
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),
        ResnetBlock(256),

        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),   # (n, 128, 128, 128)
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),    # (n, 64, 256, 256)
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 3, 7, 1, 0),                  # (n, 3, 256, 256)
        nn.Tanh(),
    )


def DiscriminatorBLOND():
    """金发判别器 D_B：判断输入是不是「真金发」。输出 (n, 1, 30, 30)。"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),                  # (n, 64, 128, 128)
        nn.LeakyReLU(0.2, inplace=True),            # 第一层后面不接 norm

        nn.Conv2d(64, 128, 4, 2, 1),                # (n, 128, 64, 64)
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 256, 4, 2, 1),               # (n, 256, 32, 32)
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(256, 512, 4, 1, 1),               # (n, 512, 31, 31)  stride 改成 1
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(512, 1, 4, 1, 1),                 # (n, 1, 30, 30)
    )


def DiscriminatorBLACK():
    """黑发判别器 D_A：判断输入是不是「真黑发」。结构同 DiscriminatorBLOND。"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),                  # (n, 64, 128, 128)
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, 4, 2, 1),                # (n, 128, 64, 64)
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 256, 4, 2, 1),               # (n, 256, 32, 32)
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(256, 512, 4, 1, 1),               # (n, 512, 31, 31)
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(512, 1, 4, 1, 1),                 # (n, 1, 30, 30)
    )


def weights_init(m, init_type='normal', init_gain=0.02):
    """官方权重初始化，配合 net.apply(weights_init) 使用。

    参数:
        init_type: 'normal' | 'xavier' | 'kaiming' | 'orthogonal'
        init_gain: 标准差/增益，官方 0.02
    说明:
        init_gain 调小可以减少训练早期生成器「把颜色翻转」的问题。
    """
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('不支持的 init_type: %s' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
