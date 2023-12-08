# UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning

> "UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning" ICLR+TPAMI, 2022 Jan
> [paper](http://arxiv.org/abs/2201.04676v3) [code](https://github.com/Sense-X/UniFormer) [blog](https://zhuanlan.zhihu.com/p/461361343) :star:
> [pdf](./2022_01_ICLR_UniFormer--Unified-Transformer-for-Efficient-Spatiotemporal-Representation-Learning.pdf)
> Authors: Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, Yu Qiao

## Key-point

- Task: video backbone

- Problems

  1. local redundancy: 视觉数据在局部空间/时间/时空邻域具有相似性，这种局部性质容易引入大量低效的计算

  2. global dependency: 要实现准确的识别，需要动态地将不同区域中的目标关联，建模长时依赖

      3D convolution can efficiently aggregate local context to suppress local redundancy from a small 3D neighborhood, it lacks the capability to capture global dependency because of the limited receptive field

- :label: Label:

**浅层 Conv 深层 Transformer**



### Motivation

> [Attention heatmap visualization](https://github.com/facebookresearch/dinov2/issues/90)
> [diffusion attn map](https://github.com/wooyeolBaek/attention-map)
> [visualizer github repo](https://github.com/luo3300612/Visualizer)
> [[FEATURE] Visualizing activation maps on efficientnet model.](https://github.com/huggingface/pytorch-image-models/issues/560) >> 可视化只有 outliers 参考 https://arxiv.org/pdf/2309.16588.pdf

但如下可视化可以发现，ViT 在浅层编码局部特征十分低效；无论是 spatial attention 抑或 temporal attention，在 ViT 的浅层，都仅会倾向于关注 query token 的邻近 token；

![](https://pic2.zhimg.com/80/v2-bbd997ab62431df641564bd7fcbf6c01_720w.webp)

相较而言，convolution在提取这些浅层特征时，无论是在效果上还是计算量上都具有显著的优势

> - :question: 验证可视化？

UniFormer (**Uni**fied trans**Former**)，便旨在以Transformer的风格，有机地统一convolution和self-attention

## Contributions



## Introduction





## methods

`UniFormer` 在网络的浅层采用local MHRA节省了冗余计算量，使得网络在深层可以轻松使用联合时空注意力

![](https://picx.zhimg.com/70/v2-3436b41cf0a20ba753f21a67ddf5c7c6_1440w.avis?source=172ae18b&biz_tag=Post)



浅层用卷积，深层用 Transformer

`class CBlock` [code](https://github.com/Sense-X/UniFormer/blob/f92e423f7360b0026b83362311a4d85e448264d7/video_classification/slowfast/models/uniformer.py#L117)

```python
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # FFN

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   
```

使用 Conv3d, k3p1s1 可学习参数作为 position embedding



`class SABlock`

joint spatiotemporal learning to generate more discriminative video representation.
adopt dynamic position embedding (DPE, see Section 3.3) to overcome permutation-invariance

```python
class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B C L -> B L C
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x  
```

 our UniFormer with LLGG can cooperatively learn local and global contexts in a joint manner

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```



在每个 stage 前面做 `PatchEmbedding`，过 Conv + Norm 还原回 `(b c t h w)`

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
```





## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

