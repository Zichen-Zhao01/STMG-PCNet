# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : SCTransNet.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class Attention_org(nn.Module):
    '''
    该注意力模块的核心功能是：
    将多层特征图与全局特征通过注意力机制进行交互，使每一层特征能够借助 emb_all 的 Key 和 Value 得到更全局感知的表示。
    '''
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        ##################  构建多头 Query、Key、Value
        ############### 说明：虽然这里实现了多头结构的框架，但目前只设置了 num_attention_heads=1，等价于 单头注意力。

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.mhead1~4: 把不同通道映射为多头格式（尽管这里 num_attention_heads=1）
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        # self.q1~q4：分别给每个 Query 分支加了 grouped 3x3 卷积
        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        # self.k 和 self.v：同样给 Key、Value 加了 3x3 的 depthwise 卷积
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)


        # ****************** useless ***************************************
        # 这些参数似乎是打算手动控制不同 Query 分支的注意力融合权重，但在 forward 里没有用到，当前状态是 无效代码，可能是遗留设计。
        self.q1_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q2_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q3_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q4_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        '''
        emb1 到 emb4: 分别来自不同尺度的特征层
        emb_all: 全局特征（可能是某个融合后的输出）
        它们的通道数C不一致，但是H,W一致
        [1, 32, 16, 16]
        [1, 64, 16, 16]
        [1, 128, 16, 16]
        [1, 256, 16, 16]
        [1, 480, 16, 16]
        '''
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))  # [1, 32, 16, 16]
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))  # [1, 480, 16, 16]
        v = self.v(self.mheadv(emb_all))
        # k, v = kv.chunk(2, dim=1)

        # 使用 einops.rearrange 将 (B, C, H, W) 变成 (B, head, C', H*W)
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)  # [1, 1, 32, 256]
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)  #
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        # 再通过 F.normalize 对 Query 和 Key 进行 L2 归一化，方便后续内积注意力
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape # [1, 1, 32, 256]
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape # [1, 1, 480, 256]

        #  计算注意力得分：这一步计算了每一个 Query 特征与所有 Key 的相似性。
        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)  # [1, 1, 32, 480]
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        # 做归一化。
        # 注意：这里对注意力使用了 InstanceNorm2d 作为先验调整，可以看作是对每个 head 的响应强度进行进一步控制。
        attention_probs1 = self.softmax(self.psi(attn1))  # [1, 1, 32, 480]
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        #  应用注意力
        #  即标准的 dot-product attention。
        #  v: [1, 1, 480, 256]
        out1 = (attention_probs1 @ v)  # [1, 1, 32, 256]
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)
        out4 = (attention_probs4 @ v)
        # 再对不同 head 取 mean(dim=1)
        out_1 = out1.mean(dim=1)  # [1, 32, 256]
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)
        out_4 = out4.mean(dim=1)
        # 最后还原为 (B, C, H, W)
        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)  # [1, 32, 16, 16]
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)
        out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)  # [1, 32, 16, 16]
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        O4 = self.project_out4(out_4)
        weights = None
        # 输出为 4 个增强后的特征层 O1 到 O4，分别对 emb1~4 进行了全局交互式增强
        return O1, O2, O3, O4, weights


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)  #  B, C, 1, 1
        out = out.view(x.size(0), 1, x.size(1)) # B, 1, C
        out = self.conv(out)  # 对通道做卷积
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    '''
    改进版的 FeedForward 网络，结合了 多尺度卷积（3×3 + 5×5）、通道注意力（ECA），结构灵活且具备轻量化和感受野增强的优势
    '''
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)  # 表示中间层通道数扩展为 dim × 扩展因子。比如输入通道是 64，扩展因子为 2.66，则 hidden_features = 170
        # 轻量线性映射，类似于 Transformer 里的 Linear(dim, 2 × hidden_features)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # Depthwise卷积分支
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        # 将输出在通道维度平均分成两个分支：1. x_3：用于 3×3 depthwise 卷积; 2. x_5：用于 5×5 depthwise 卷积
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        # 特征融合
        # 将两个尺度的卷积输出拼接，得到融合多尺度信息的特征图，通道数恢复为 2 × hidden_features
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)  # 输出通道恢复
        x = self.eca(x)  # 使用 Efficient Channel Attention (ECA)，为输出特征加上轻量化的通道注意力。
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    '''
    用于融合多尺度特征的模块 Block_ViT，
    结合了 LayerNorm、注意力机制（Attention_org）和前馈网络（FeedForward），
    用于提取红外图像中的小目标，具有 Vision Transformer 的思想
    '''
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        # LayerNorm 初始化
        # 对输入做标准化处理，提高训练稳定性。
        # 用于注意力模块前的归一化。
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')
        # 多尺度注意力机制模块， 核心注意力模块
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        # FFN 前馈网络部分
        # 这个结构模仿了 Transformer 中的 MLP 部分，常用于增强特征表达
        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)  #  var_name = 'emb1', 'emb2', 'emb3', 'emb4'
            tmp_var = locals()[var_name]  # # 通过变量名字符串从当前局部变量中拿变量
            if tmp_var is not None:
                embcat.append(tmp_var)  # # 如果不是 None，就加入 embcat
        emb_all = torch.cat(embcat, dim=1)  # 将 emb1~4 在 通道维度 上拼接，形成 emb_all：全局融合特征图，供 K/V 生成
        # 归一化
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        # 进行注意力融合
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        # 残差连接原始输入
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        # 再接 LayerNorm + FeedForward 模块（Transformer 结构中的 MLP）
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None  # [1, 32, 16, 16]
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        # 残差连接
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3, emb4)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, x4, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class SCTransNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)  # 64 224 224
        x2 = self.down_encoder1(self.pool(x1))  # 128 112 112
        x3 = self.down_encoder2(self.pool(x2))  # 256 56  56
        x4 = self.down_encoder3(self.pool(x3))  # 512 28  28
        d5 = self.down_encoder4(self.pool(x4))  # 512 14  14
        #  CCT
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        #  CCT
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        #  Feature fusion
        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))
        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))  # 训练集和测试集的逻辑不一样
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    config_vit = get_CTranS_config()
    model = SCTransNet(config_vit, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
