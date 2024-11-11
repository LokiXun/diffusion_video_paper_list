# Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer

> "Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer" ECCV, 2024 May 7
> [paper](http://arxiv.org/abs/2405.04312v2) [code](https://github.com/THUDM/Inf-DiT) [pdf](./2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer.pdf) [note](./2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note.md)
> Authors: Zhuoyi Yang, Heyang Jiang, Wenyi Hong, Jiayan Teng, Wendi Zheng, Yuxiao Dong, Ming Ding, Jie Tang

## Key-point

- Task: `ultra-high-resolution SR`

- Problems

  - 大分辨率图显存爆掉

    > quadratic increase in memory during generating ultra-high-resolution images(e.g. 4096 × 4096), the resolution of generated images is often limited to 1024 ×1024 .

  - global dependencies 各个 patch 全局一致性

- :label: Label: `ultra-high-resolution SR` 



## Contributions

- propose Unidirectional Block Attention (UniBA) 实现全局一致性 & 降低推理时候的显存占用

  4096x4096 每次只预测 1/16 居然可以保持全局一致性很牛

- 训练了一个 DiT 实现多分辨率（没说是任意分辨率很严谨）& SOTA

  >  adopt the DiT structure for upsampling and develop an infinite super-resolution model capable of upsampling images of **various shapes and resolutions**.

- 设计了多种 tricks 维持局部 & 全局一致性

  > design multiple techniques to further enhance local and global consistency, and offer a zero-shot ability for flexible text control





## Introduction

**强调最大的问题是显存占用！**

- Q：ultra-high-resolution images 显存限制不好生成？

> 行文：说问题，再说一下衍生出来的应用上的受限

先前方法 **cascaded generation**, DALLE2, Imagen 通过此方法生成 1024 分辨率的图像

> first produces a low-resolution image, then applies multiple upsampling models to increase the image’s resolution step by step. This approach breaks down the generation of high-resolution images into multiple tasks. Based on the results generated in the previous stage, the models in the later stages only need to perform local generation.

- "Cascaded diffusion models for high fidelity image generation"

- Q：什么是 cascaded generation？

![fig1](docs/2024_06_CVPR_Hierarchical-Patch-Diffusion-Models-for-High-Resolution-Video-Generation_Note/fig1.png)





- Q：最大的问题是显存限制，没法搞超大分辨率图像

> The biggest challenge for upsampling to much higher resolution images is the significant GPU memory demands

推理都要 80G 不好搞，训练要存储梯度，需要更大显存

> Specifically, generating a 4096×4096 resolution image, which comprises over 16 million pixels requires more than 80GB of memory, exceeding the capacities of standard RTX 4090 or A100 graphics cards. Furthermore, the process of training models for high-resolution image generation exacerbates these demands, as it necessitates additional memory for storing gradients, optimizer states, etc

- Q：降低显存方式？

LDM 使用 VAE 转换到 latent 处理，但**过高的压缩比导致信息丢失很严重**；

如果为了显存继续提升 f 压缩比，质量会更烂





- Q：先前方式都是整张图去训练，推理为啥一定就要这么搞？:star:

提出 UniBA 保持多个 block 的一致性，**维持 global consistency**

> propose a Unidirectional Block Attention (UniBA) algorithm that can dramatically reduce the space complexity of gen
>
> 分析下时间复杂度 $O(N^2) \to O(N)$

还有一些 tricks **维持 local consistency**

>  design several techniques including providing **global image embedding** to enhance the global semantic consistency and offer zero-shot text control ability, and provision of all neighboring low-resolution(LR) blocks through cross-attention mechanisms





###  block-based generation methods

- "Multidiffusion: Fusing diffusion paths for controlled image generation"
- "Mixture of diffusers for scene composition and high resolution image generation"
- "Exploiting diffusion prior for real-world image super-resolution"





## methods

- Q：How to avoid storing the entire image’s hidden state in memory becomes the key issue？

类似 RVRT 的方式，划分滑动窗口；这里是指空间上的（h,w）上拆分 patch。以 patch=4x4 像素为最小单位，32x32 个patch 为一个 block (128x128 像素)，每次把一个 block 的图像特征输入到 DiT 中；

> 具体实现见后面的 code

1. 加上整个图的 clip image embedding 1x768 + MLP 映射为 1x1028 特征，加到 time-step embedding 上 **（全局一致性）**
2. 在 Self-Attn 中把**当前 patch 和前几个 patch 的 K，V 一起 concat ，加上相对位置编码（block 的 id）**，融合实现局部一致性；
3. **在 DiT 的第一层**， cross-attn 模块不用文本，换成 LR 图像中取一个更大的范围（包含当前 block 信息）的局部图，一层 Conv + ViT patchify 提取图像特征，作为 KV 融合；（"全局"一致性，这里可以理解为还是局部一致性。。没用全图啊）



### Unidirectional block attention

![fig3.png](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig3.png)

- Q：在 layer 里面对特征划分 patch？:star:

> When the image is fed into the network, the channel size and resolution of a block may change, but the layout and the relative positional relationships between blocks will remain unchanged.

只输入一部分 patch 的特征到模型里面降低显存

> If there is a way to apply sequential batch generation of blocks where each batch simultaneously produces a subset of the blocks, only a small number of block hidden states have to be kept in memory simultaneously, making it possible to generate ultra-high-resolution images.

- Q：relative positional relationships between blocks will remain unchanged？设个 pos 怎么设计？







- Q：如何定义两个 block 是相关的？

> define that blockA is dependent on blockB if the generation of blockA involves the hidden state of blockB in computation

之间的关系是双向的，两个block 放到 UNet Conv Layer 都会在一起处理，所以**理解为双向（彼此都要用到对方信息），因此先前 UNet 多个 block 是要一起生成的**

> dependencies between blocks are bidirectional in most previous structures

- Q：怎么理解双向？

**相邻的 patch** 的特征在 CNN 里面会一起处理，来做到双向；这里也只是**相邻的 patch 也有个局部性**

> Take UNet as an example: two adjacent elements in neighboring blocks use each other’s hidden state in the convolution operation, therefore all pairs of neighboring blocks must be generated simultaneously

- Q：分 patch 后，这个 bidirectional 怎么办？:star:





本工作目标设计一个算法，可以把各个 blocks 拆分成一个个 batch 进行处理（提升效率？）

> Given the aim to save the memory of blocks’ hidden states, we hope to devise an algorithm that allows the blocks in the same image to be divided into several batches for generation

定义一下情况可以把 blocks 按顺序分别推理，**做了一些假设限定**

> Generally, an image generation algorithm can perform such a sequential batch generation among blocks if it meets the following conditions
>
> 1. The generative dependency between blocks are **unidirectional**, and can form a directed acyclic graph (DAG). 
> 2. Each block has only **a few direct (1st-order) dependencies on other blocks**, since the hidden states of the block and its direct dependencies should simultaneously be kept in the memory

- Q：ensure consistency across the whole image？（**多次推理的全局一致性！**）

首先保证 block 有足够的信息，能理解到是啥

> the blocks have a large enough receptive field to manage long-range dependencies.

设计 UniBA 模块

>  For each layer, every block directly depends on three 1st-order neighboring blocks

![fig3.png](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig3.png)



- Q：如何加上这个 relative pos embedding?

![eq1](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/eq1.png)





- Q：针对全图，边缘的块没有相邻块咋搞？

**在 Appendix A 实现了一个更加高效的 UniBA 模块** :star:

> We also implement an efficient approach to apply UniBA for full image in pytorch style

类似 SwinTransformer 复制了边缘的块，高效在于边缘的块不用单独写模块处理。。

<img src="docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig9.png" alt="fig9" style="zoom:50%;" />



**时间复杂度变成 O(N)，推理时候不需要的 KV 就丢掉了**

> we implement a simple but effective inference process. As illustrated in Fig. 3, we generate n × n blocks at once, from the top-left to bottom-right. After generating a set of blocks, we discard hidden states i.e. KV-cache that are no longer used and append newly generated KV-cache to the memory.

一次生成 nxn 个 blocks**，实验发现这个 n 越大越好，显存节省越多**

> it’s optimal to choose the largest n allowed by the memory limitation





#### pseudocode

![fig10](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig10.png)



#### Unstable Train

- Q：大部分 diffusion 模型使用 FP16 训练的，**训练不稳定咋搞？**

> many large diffusion models are trained under FP16 (16-bit precision) to reduce GPU memory and computation time. In practice, however, **a sudden increase in loss** often occurs after training for several thousand steps.

发现当 block 数量太多，QKV 数值贼大，导致了训练不稳定 :star:

> We observe that when the number of patches in a single attention window is too many (e.g. there are 4096 patches with block size = 128 and patch size = 4), the attention score QT K/√ d can become very large, leading to unstable gradients or even overflow.

参考 ViT 中训练 trick 对 Q，K 分别都做 Norm

![appendix_attn_eq](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/appendix_attn_eq.png)



#### Initial Noise

初始噪声影响很大的。。用 LR 作为 prior

> Previous experience [8] shows that the diffusion modal tends to generate images that are closely aligned with the initial noise, which may lead to color mismatching with the original image during upsampling, e.g. Fig. 11. 

xLR 是 resized 的 LR 图像，从而得到 XT

![initial_noise](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/initial_noise.png)

- Q：这里替换均值是几个意思？

在 xLR 上再加噪声。。。:face_with_head_bandage:

- Q：如何验证上面初始噪声的影响？？:star:

直接拿纯黑图像来做 SR 看看，设计的 toy example 很巧妙

<img src="docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig11.png" alt="fig11.png" style="zoom:50%;" />





- Q： 发现这么搞输出图像有可能变得很模糊

> However, this method can sometimes lead to the generated images appearing somewhat blurred just like LR images.

没说咋解决。。



### Framework

![fig4.png](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig4.png)



DiT 验证了使用 ViT 方式分 patch 再处理更高效，可扩展性更好

> DiT [17], which applies Vision Transformer (ViT) [6] to diffusion models and proves its efficacy and scalability

**Model input**

在 RGB 空间分 patch

> Inf-DiT first partitions input images into multiple non-overlapping blocks, which are further divided into patches with a side length equal to the patch size

模型没有用类似 VAE 的东西对输入下采样，类似 SDv2 x4 SR 直接把 LR 作为输入，最后只对输出过 VAE 上采样得到 x4

> Unlike DiT, considering the compression loss such as color shifting and detail loss, the patchifying of Inf-DiT is conducted in RGB pixel space instead of latent space.





#### Position Encoding

UNet 的 pos embedding 在 Conv，Attn 里面了

> UNet-based diffusion models [21] that can perceive positional relationships through convolution operations, all operations including self-attention, FFN in transformers are permutation invariant functions.

LLM 一些工作发现相对 pos embedding 更有效，**因此使用 Rotary Positional Encoding (RoPE)** :star:

> relative positional encoding is more effective in capturing word position relevance compared to absolute positional encoding



- Q：具体使用 RoPE 方式？

和 StableSR 一样，没区别

> Specifically, we divide channels of hidden states in half, one for encoding the x-coordinate and the other for the y-coordinate, and apply RoPE in both halves.

**随机取 LR 的 patch 来学习到完整的 relative pos table** :star:

> To ensure all parts of the positional encoding table can be seen by the model during training, we employ the Random Starting Point: For each training image, we randomly assign a position (x, y) for the top-left corner of the image, instead of the default (0, 0).



- Q：block 的 relative pos embedding 咋搞？:star: :warning:

> we additionally introduce block-level relative position encoding P1∼4, which assigns a distinct learnable embedding based on the relative position before attention.

如何获取 T 维度的 embedding？？



### Global  Consistency :star:

> The global semantic information within low-resolution (LR) images, such as artistic style and object material, plays a crucial role during upsampling.

加入全局图，减轻模型理解图像内容的负担？？

> However, compared to text-to-image generation models, **the upsampling model has an additional task understanding and analyzing the semantic information of LR images,** which significantly increases the model’s burden

**是因为没有 paired text 导致不好理解**？所以用 CLIP image embedding 作为全局特征

> This is particularly challenging when training without text data, as high-resolution images rarely have high-quality paired texts, making these aspects difficult for the model.



- Q：CLIP image embedding 放到哪里？:star:

![fig4.png](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig4.png)

对 LR 图提取特征，**加到输入的 t-embedding 上**

> Inspired by DALL·E2 [20], we utilize the image encoder from pre-trained CLIP [19] to extract image embedding ILR from low-resolution images, which we refer to as Semantic Input.
>
> We **add the global semantic embedding to the time embedding of the diffusion transformer** and **input it into each layer,** enabling the model to learn directly from high-level semantic information.





- Q：与文本冲突？？对 image embedding 更新一下？

使用 CLIP 还有个好处是能让模型通过 text CLIP embedding 引导 :star:

> using the aligned image-text latent space in CLIP, we can **use text to guide the direction of generation**, even if our model has not been trained on any image-text pairs.

**使用 Pos & Negative Prompt** 去更新一下 image embedding，也没说一定有用！:star: :star:

![eq5](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/eq5.png)

> Cpos = “clear” and Cneg = “blur” sometimes help.





### Local Consistency

- Q：把 LR 和 noise concat 作为 prior 有用，但还是会有局部不连续的问题

**猜测分析**了一下为啥，因为设计的 UniBA 模块是单向的，右边的 block 没考虑；**如果右边的 block 信息更丰富，会导致左边 block 差异**

> The reason is that, there are several possibilities of upsampling for a given LR block, which require analysis in conjunction with several nearby LR blocks to select one solution. Assume that the upsampling is only performed based on the LR blocks to its left and above, it may select an HR generation solution that conflicts with the LR block to the right and below. Then when upsampling the LR block to the right, if the model considers conforming to its corresponding LR block more important than being continuous with the block to the left, a HR block discontinuous with previous blocks would be generated.

在 transformer 第一层，加上 3x3 领域做 cross-attn

> we **introduce Nearby LR Cross Attention**. In the first layer of the transformer, each block conducts cross-attention on the surrounding 3 × 3 LR blocks to capture nearby LR information.

实验发现能够明显降低不一致

![fig7](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig7.png)







## Exp Setting

-  LAION-5B [25] with a resolution higher than 1024×1024 and aesthetic score higher than 5, and 100 thousand high-resolution wallpapers from the Internet

- we use fixed-size image crops of 512×512 resolution during training.

- 评估使用 FID

  > [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)

  **分数越低代表两组图像越相似**，或者说二者的统计量越相似，FID 在最佳情况下的得分为 0.0，表示**两组图像**相同

  用 torchvision 中**预训练好的 InceptionV3 模型**（修改了几层），提取第几个 block 的输出。对每个图像弄成 dimens 尺寸的 embedding。对这个 embedding 提取均值和方差

  ```python
  def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                      device='cpu', num_workers=1):
      """Calculation of the statistics used by the FID"""
      mu = np.mean(act, axis=0)
      sigma = np.cov(act, rowvar=False)
      return mu, sigma
  ```

  $ d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).$




**Data Processing** 》》Real-ESRGAN

1. rescale 512
2. random crop 512

训练时候随机取哪种方式

> When processing training images with a resolution higher than 512, there are two alternative methods: directly performing a random crop, or resizing the shorter side to 512 before performing a random crop. While the direct cropping method preserves high-frequency features in high-resolution images, the resizethen-crop method avoids frequently cropping out areas with a single color background, which is detrimental to the model’s convergence. Therefore, in practice, we randomly select from these two processing methods to crop training images.

**Training settings**

block size = 128 and patch size = 4, which means every training image is divided into 4 × 4 blocks and every block has 32×32 patches.

- mean and std of training noise distribution to −1.0 and 1.4.
- BF16 format due to its broader numerical range
- we first resize LR images to 224 × 224 and then input them to CLIP.



## Experiment

> ablation study 看那个模块有效，总结一下

说明显存爆炸的问题（完全可以用 Swin3D & Mamba 展示一下）

![fig2](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig2.png)

**HPDV2 dataset**

![tb1](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/tb1.png)

主观结果，基于 patch 的方式细节好，但是 patch 之间 blocking 太明显

![fig5](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig5.png)



LR=1024 数据

![fig6](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig6.png)







对比 RealSR 方法，效果居然接近 StableSR

![fig16](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig16.png)

效果比 SDv2 x4SR 好太多了，看起来纹理扭曲得问题得到了解决？？纹理细节更加清晰了



### Iterative SR

![fig8](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig8.png)

- Q：细节还是很渣渣?

没有搞文本，没理解内容是啥？

**迭代的搞，一开始错了后面都挂了**

> However, it is hard for the model to correct inaccuracies generated in earlier stages

### ablation

FID 看不出来差异

![tb3](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/tb3.png)



### Semantic Input

整图 LR 的 Semantic Input 影响很大的！！！:star:

![fig12](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig12.png)



### base-model

除了 DiT 试试看别的

**SDv2 也是可以这么搞得！**

![fig13](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig13.png)

![fig14](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig14.png)

![fig15](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig15.png)



## Code

`pos_embed_config` 

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L481

```
{'target': 'dit.embeddings.RotaryPositionEmbedding', 'params': {'num_patches': 1024, 'hidden_size': 1024, 'hidden_size_head': 64, 'pix2struct': True}}
```



- qk 上加 layerNorm

```
        if qk_ln:
            print("--------use qk_ln--------")
            self.q_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])
            self.k_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])
```



- pos_embedding

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L145

```
        if random_position:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 8, hidden_size=hidden_size, hidden_size_head=hidden_size//num_head)
        else:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 2, hidden_size=hidden_size,
                                                hidden_size_head=hidden_size // num_head)
```





- LR-image cross-attention :star:

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L166

```
self.proj_lr = nn.Conv2d(in_channels, hidden_size, kernel_size=self.lr_patch_size, stride=self.lr_patch_size, bias=bias)
# Conv2d(3, 1280, kernel_size=(2, 2), stride=(2, 2))
```

这里也加了 LayerNorm ！！



- image-patch-embedding

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/embeddings.py#L114



- time-embedding

```python
        if args.ddpm_time_emb:
            self.time_embed = DDPMTimeEmbedding(args.hidden_size)  # Create sinusoidal timestep embeddings.
        else:
            self.time_embed = TimeEmbedding(args.hidden_size)
```





### Image-embedding

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/encoders.py#L244

加载权重

```python
def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys
```

details

```python
# set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD
# OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
# OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
```

freeze

```python
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
```





#### download

```
{'embed_dim': 768, 'vision_cfg': {'image_size': 224, 'layers': 24, 'width': 1024, 'patch_size': 14}, 'text_cfg': {'context_length': 77, 'vocab_size': 49408, 'width': 768, 'heads': 12, 'layers': 12}}
```

预训练模型权重 `'datacomp_xl_s13b_b90k'`，去全局变量读取 url

> https://vscode.dev/github/THUDM/SwissArmyTransformer/blob/main_clip/pretrained.py#L235 
>
> ```python
> _VITL14 = dict(
>     openai=_pcfg(
>         "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
>     laion400m_e31=_pcfg(
>         "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
>     laion400m_e32=_pcfg(
>         "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
>     laion2b_s32b_b82k=_pcfg(
>         hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
>         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
>     # DataComp-XL models
>     datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/'),
>     commonpool_xl_clip_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K/'),
>     commonpool_xl_laion_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K/'),
>     commonpool_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/'),
> )
> ```

````
pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
# pretrained_cfg={'url': '', 'hf_hub': 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/', 'mean': None, 'std': None}

````

​	如果 `model_name` 是本地路径返回 `{}`, 去本地读取



- download from huggingface :construction_worker:

> https://vscode.dev/github/THUDM/SwissArmyTransformer/blob/main_clip/pretrained.py#L397

```python
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
```

可以设置 `resume_download=True`

> https://docs.python.org/3/library/stat.html  `os.stat(incomplete_path).st_size` 读取 byte size

```python
        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> Generator[io.BufferedWriter, None, None]:
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
```



`cache_dir=None` 默认保存位置

```python
HF_HOME = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
    )
)
default_cache_path = os.path.join(HF_HOME, "hub")
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)  # /home/ps/.cache/huggingface/hub
```

`repo_folder_name(repo_id=repo_id, repo_type=repo_type)='models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'`

`storage_folder='/home/ps/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'`

> https://hf-mirror.com/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin

```python
headers = {'user-agent': 'open_clip/2.22.0; hf_hub/0.22.2; python/3.10.9; torch/2.1.1+cu118'}

# check python version
import sys
_PY_VERSION: str = sys.version.split()[0].rstrip("+")
# sys.version  # '3.10.9 (main, Mar  8 2023, 10:47:38) [GCC 11.2.0]'
# check package version
import importlib.metadata
_package_versions[candidate_name] = importlib.metadata.version(name)
_package_versions.get(package_name, "N/A")
```

本地文件实际保存在 `blob` 目录，**文件名是 sha256sum 值** `6509f07e6fc0da68f8e1ee881bf90803f0b053d2f7ed2013cc7c3a49ac4dd3db`

`pointer_path` 的 hash 值是对应仓库的提交 `commit=84c9828e63dc9a9351d1fe637c346d4c1c4db341` 

```python
blob_path = os.path.join(storage_folder, "blobs", etag)
# '/home/ps/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/blobs/6509f07e6fc0da68f8e1ee881bf90803f0b053d2f7ed2013cc7c3a49ac4dd3db'
pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
# '/home/ps/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/open_clip_pytorch_model.bin'  # metadata.commit_hash
```

会检查 `pointer_path` & `blob_path` 如果本地文件存在就直接返回；下载到 `blob_path` 后面造一个软链接到 `pointer_path`

> 下载模块代码 https://vscode.dev/github/THUDM/SwissArmyTransformer/blob/mainingface_hub/file_download.py#L1492
>
> ```python
>     if _support_symlinks:
>         src_rel_or_abs = relative_src or abs_src
>         logger.debug(f"Creating pointer from {src_rel_or_abs} to {abs_dst}")
>         try:
>             os.symlink(src_rel_or_abs, abs_dst)
>             return
>         except FileExistsError:
>             if os.path.islink(abs_dst) and os.path.realpath(abs_dst) == os.path.realpath(abs_src):
>                 # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
>                 # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
>                 return
>             else:
>                 # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
>                 # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
>                 raise
>         except PermissionError:
>             # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
>             # is supported on both volumes but not between them. Let's just make a hard copy in that case.
>             pass
> 
> ```

```
            is_big_file = os.stat(temp_file.name).st_size > constants.HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD
            if local_dir_use_symlinks is True or (local_dir_use_symlinks == "auto" and is_big_file):
                logger.debug(f"Storing {url} in cache at {blob_path}")
                _chmod_and_replace(temp_file.name, blob_path)
                logger.debug("Create symlink to local dir")
                _create_symlink(blob_path, local_dir_filepath, new_blob=False)
```





- Image-embedding + MLP

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/embeddings.py#L80

```python
class ConditionEmbedding(nn.Module):
    def __init__(self, hidden_size, label_dim, augment_dim, vector_dim, label_dropout=0):
        super().__init__()
        self.map_vector = nn.Sequential(
            nn.Linear(vector_dim, hidden_size),  # vector_dim=768, hidden_size=1280
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ) if vector_dim != 0 else None
        if self.map_vector:
            #zero output init
            nn.init.constant_(self.map_vector[2].weight, 0)
            nn.init.constant_(self.map_vector[2].bias, 0)
    
    def forward(self, emb, **kwargs):
        # emb: time-step-embedding
        if self.map_vector is not None and 'vector' in kwargs:
            emb = emb + self.map_vector(kwargs['vector'])
        return emb
```



### sample prepare

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/generate_t2i_sr.py#L163

- `concat_lr_image` 把 LR 图像 resize 到 SR 大小

```
tmp_lr_image = transforms.functional.resize(lr_image, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
concat_lr_image = torch.clip(tmp_lr_image, -1, 1).to(device).to(inference_type)
```



- 在 4h x 4w 图上，按 patch_size = 4x4 划分 patch，计算 `rope_position_ids`

```python
num_patches = h * w // (self.patch_size ** 2)  # (4h=1920 * 4w=3328)// 16 = 399360 = h*w
position_ids = torch.zeros(num_patches, 2, device=images.device)
position_ids[:, 0] = torch.arange(num_patches, device=images.device) // (w // self.patch_size)  # row id
position_ids[:, 1] = torch.arange(num_patches, device=images.device) % (w // self.patch_size)  # column id

position_ids = position_ids % (8 * self.image_size // self.patch_size)  # % 1024
# if patch_id in h or w dim < 1024, nothing change

position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), images.shape[0], dim=0).long()
position_ids[position_ids == -1] = 0
rope_position_ids = position_ids
```



- sampler `ConcatSRHeunEDMSampler` :star:

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/sampling/samplers.py#L188
>
> - `dit.sampling.discretizers.EDMDiscretization` ??? 
>   https://blog.csdn.net/weixin_44966641/article/details/135181485

加噪到 xT，获取噪声表

```python
images, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            images, cond, uc, num_steps, init_noise=init_noise
        )
```

- `self.num_steps=20` 的噪声表，长度 = 21，最后一步是噪声 = 0

```python
sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        ).to(x.dtype)
"""
tensor([4.0000e+01, 3.0125e+01, 2.2375e+01, 1.6375e+01, 1.1875e+01, 8.4375e+00,
        5.9062e+00, 4.0625e+00, 2.7188e+00, 1.7891e+00, 1.1406e+00, 7.0703e-01,
        4.2188e-01, 2.4219e-01, 1.3184e-01, 6.8359e-02, 3.2959e-02, 1.4587e-02,
        5.7983e-03, 1.9989e-03, 0.0000e+00], device='cuda:0',
       dtype=torch.bfloat16), len(sigmas)==21
"""
if init_noise:
	x = x + torch.randn_like(x) * sigmas[0]
```





- 迭代去噪

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/sampling/samplers.py#L188

```python
        for i in tqdm(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            images = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                images,
                cond,
                uc,
                gamma,
                rope_position_ids,
                i,
                return_attention_map
            )

        return images
```



`def sampler_step ` 去噪一步，**使用 `HeunEDMSampler`** ：每次 DiT 出来，**去噪一步还要再修正一步**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/sampling/samplers.py#L97

```python
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0, rope_position_ids=None, sample_step=None, return_attention_map=None):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, rope_position_ids, sample_step)  # DiT
        
        # correction
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids, sample_step
        )
        return x
```



### DiT infer

![fig4.png](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig4.png)

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L612

```python
denoiser = lambda images, sigmas, rope_position_ids, cond, sample_step: \
	self.precond_forward(images=images, sigmas=sigmas, rope_position_ids=rope_position_ids, inference=True, sample_step=sample_step, do_concat=do_concat, ar=ar, ar2=ar2, block_batch=block_batch, **cond)
    
def precond_forward(self, inference, rope_position_ids, concat_lr_imgs, lr_imgs=None, ar=False,
                    ar2=False, sample_step=None, block_batch=1, *args, **kwargs):
    images, sigmas = kwargs["images"], kwargs["sigmas"]
```

input params

- `kwargs["images"]` 输入的 resized LR image **加噪后的 xT**

- `kwargs["sigmas"]` 噪声强度

- `kwargs['do_concat'] = True` 

- `concat_lr_image` 为 resized LR image

- `cond`

  ```python
  # use lr_imgs extract fea
  cond["vector"] = image_embedding  # [1, 768]
  uncond["vector"] = image_embedding
  
  cond["lr_imgs"] = lr_imgs  # origin size LR image
  cond["concat"] = torch.ones(1, dtype=torch.bool, device=images.device)
  uncond["concat"] = torch.zeros(1, dtype=torch.bool, device=images.device)
  
  cond["concat_lr_imgs"] = images  # resized LR image
  uncond["concat_lr_imgs"] = images
  ```



- images -> resized LR image **加噪后的 xT**  && concat_lr_imgs -> resized LR image，concat 起来输入 DiT

```
images = torch.cat((images, concat_lr_imgs), dim=1)  # [1, 6, 4h 4w]
```





**DiT prepare**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/model/base_model.py#L93
>
> - DiT forward
>
> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/model/transformer.py 
>
> 模型函数都保存在 HOOKS 全局变量里面。。pdb 直接看就行
>
> ```python
>         # embedding part
>         if 'word_embedding_forward' in self.hooks:
>             hidden_states = self.hooks['word_embedding_forward'](input_ids, output_cross_layer=output_cross_layer, **kw_args)
>         else:  # default
>             hidden_states = HOOKS_DEFAULT['word_embedding_forward'](self, input_ids, output_cross_layer=output_cross_layer,**kw_args)
> ```
>

- Q：`rope_position_ids >> [1, 32*32, 2]` 取 `vit_block_size=32` ？？`tmp_images >> [1, 6, 128, 128]` 取 `block_size=128` 啥关系？

concat x4 LR 图像，取 block=128 像素，以 patch=4 拆开, 就是 32 个 patch；

rope_position_ids  是在 x4 LR 上**以 patch=4** 算的，大小和 LR 一样。取 vit_block_size=32 和上面 x4 LR 图像取得 block 对应

- Q：`rope_position_ids ` 在哪里用？ :star:

Self-attn 里面对 QK 加上 RotaryPositionEmbedding



- Q：cache 存得是啥，哪里用到？

之前的 patch 过当前 timestep 的 KV tensor 存下来，在后面 patch 过 Self-attn 将 K, V 和 memory 里的 KV concat 起来

- Q：训练时候怎么分 patch 咋办？

整个图放入训练，不分多个 patch



- Q：怎么把整张图 resize 放入？？

  1. reize 224

- Q：先前的 patch 怎么 concat 做 QKV？

- Plan1: 存好每个 scale 的 KV tensor :star:

  1. diffusion 去噪 25 步，每一步都有 4 个 scale，**16 个 Self-Attn block 咋搞？**

     $16 \times (b t c h w)$

- Plan2: 类似 CLIP text encoder **单独弄个新的特征**？？咋训练呢。。

  - Cross-LR ??
  - 1x768







**输入层提取特征**

- embedding part >> `class ImagePatchEmbeddingMixin(BaseMixin):` 

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/embeddings.py#L127

对 resized-LR one-block 图像 128x128，按 4x4 patch 提取特征，转为 `(b h*w c)`

```python
    def word_embedding_forward(self, input_ids, **kwargs):
        images = kwargs["images"]  # [1, 6, 128, 128] one block in resized-LR
        emb = self.proj(images)  # [1, 1280, 32, 32]
        emb = emb.flatten(2).transpose(1, 2)  # ([1, 1024, 1280]
        if self.append_emb:
            emb = torch.cat((kwargs["emb"][:, None, :], emb), dim=1)
        if self.reg_token_num > 0:
            emb = torch.cat((self.reg_token_emb[None, ...].repeat(emb.shape[0], 1, 1), emb), dim=1)
        if self.add_emb:
            emb = emb + kwargs["emb"][:, None, :]
        return emb
```

- position_embeddings=None
- dropout(0)





**`BaseTransformerLayer `**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/model/transformer.py#L314
>
> 结构可以参考 SD3 [pdf](./2024_03_Arxiv_Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-Synthesis.pdf)
>
> `layer_forward`  https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L383

#### **Self-Attn** (memory-concat)

DiT 结构如图，在这里 KV concat 起来之前的 memory

![fig2](docs/2024_03_Arxiv_Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-Synthesis_Note/fig2.png)

- timestep 特征 `emb` 计算 AdaIN 调制参数

```python
self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, out_times * hidden_size)  # 1280 -> 4*1280
            ) for _ in range(num_layers)
            ])


# ...
pass
elif self.nogate and not self.cross_adaln:
    shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(4, dim=1)  # [1, 4*1280] -> chunk
    gate_msa = gate_mlp = 1
```

- xT 做 layerNorm + 调制

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

attention_input = layer.input_layernorm(hidden_states)
attention_input = modulate(attention_input, shift_msa, scale_msa)  # [1, 1024, 1280]
```

实际 `attention_forward` 位置

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L196 运行的 forward，但好多模块都用的是 对 https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/model/transformer.py#L34

MLP + 拆为 QKV

> 用 cuda 算子并行 https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/mpu/layers.py#L232

- 对 QK 做 LayerNorm

```python
if origin.qk_ln:
    query_layernorm = origin.q_layer_norm[kw_args['layer_id']]
    key_layernorm = origin.q_layer_norm[kw_args['layer_id']]
    query_layer = query_layernorm(query_layer)
    key_layer = key_layernorm(key_layer)
```

- QK 加上 RotaryPositionEmbedding :star:

加上 patch 的相对位置信息

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/embeddings.py#L433

```python
    def forward(self, t, **kwargs):
        if self.pix2struct:
            x_coords = kwargs['rope_position_ids'][:, :, 0]  # [1, 1024]
            y_coords = kwargs['rope_position_ids'][:, :, 1]
            freqs_cos = self.freqs_cos[x_coords, y_coords].unsqueeze(2)
            freqs_sin = self.freqs_sin[x_coords, y_coords].unsqueeze(2)
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        return t * freqs_cos + rotate_half(t) * freqs_sin

if origin.rope:
    query_layer = origin.rope(query_layer, rope_position_ids=rope_position_ids)  # [1, 1024, 16, 80]
    key_layer = origin.rope(key_layer, rope_position_ids=rope_position_ids)  # [1, 1024, 16, 80]
```

- 和 memory 保存的 K, V concat 起来，这里的 KV 已经加上了 RotaryPositionEmbedding :star:

```python
            elif inference == 1:
                kw_args['output_this_layer']['mem_kv'] = [key_layer, value_layer]  # save this k,v
                k_stack = [key_layer]
                v_stack = [value_layer]
                for mem in mems:
                    k_stack.append(mem[kw_args['layer_id']]['mem_kv'][0])  # [1, 1, 1, 1024, 16, 80]
                    v_stack.append(mem[kw_args['layer_id']]['mem_kv'][1])
                key_layer = torch.cat(k_stack, dim=3) # [1, 1, 1, 2048, 16, 80]
                value_layer = torch.cat(v_stack, dim=3)
```

- Q：训练时候咋办？

不用 memory



- K 加上 re_position ?? 论文里面的 P

![eq1](docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/eq1.png)

```python
if re_position:
	self.re_pos_embed = nn.Parameter(torch.zeros(num_layers, 4, num_head, hidden_size_head))  # [28, 4, 16, 80]
if origin.re_position and do_concat:
    re_pos_embed = origin.re_pos_embed[kw_args['layer_id']].repeat_interleave(key_layer.shape[1] // origin.re_pos_embed.shape[1], dim=0).unsqueeze(0)
    # origin.re_pos_embed[kw_args['layer_id']].shape >> [4, 16, 80]
    key_layer = key_layer + re_pos_embed  # [1, 2048, 16, 80]
```

- FFN: MLP 输出 `attention_output`
- 在外面加上 residual

```python
hidden_states = hidden_states + gate_msa * attention_output  # gate_msa = 1
```





#### cross-LR :star:

在 28 个 LayerForward block 里面，**只有第 0 个 block用 Cross-attn!** 其他 block 只做 layernorm :warning:

```python
        if layer_id == 0 and self.cross_lr:
            cross_attention_input = layer.post_attention_layernorm(hidden_states)

            # do cross attention here
            cross_attention_output = self.cross_attention_forward(cross_attention_input, **kwargs)

            hidden_states = hidden_states + cross_attention_output
            mlp_input = self.post_cross_attention_layernorm(hidden_states)
        else:
            mlp_input = layer.post_attention_layernorm(hidden_states)
```



对于每个 block 想只用大一圈的 LR 图像，不用全图（看 code 里面 TODO 写了降低显存）；
因此LR 图像先用一层卷积转为特征，再用**ViT 方式划分 patch，作为 LR-image feature**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L325

- Conv2d 提取特征，转为 c=1280，h/2, w/2

```python
lr_imgs = self.proj_lr(lr_imgs)  # Conv2d(3, 1280, kernel_size=(2, 2), stride=(2, 2)) [1, 3, 480, 832] -> [1, 1280, 240, 416]
```

- Unfold

> https://blog.csdn.net/ViatorSun/article/details/119940759

**resized LR 上面取 128x128 区域**，sr=4, 相当于原始 LR 32x32 区域，由于 LR image 用了 stride = 2 的 conv，感受野相当于原始的 LR 图像中 **16x16 区域** :star: :star:

后面 LR-image **取包括当前 resized block 的局部图时候**，stride 设置为 16x16 能够保证每个 LR-image-block 能包括当前 resized-block :star:

```python
self.block_size = 32  # 32 patches
self.sr_scale = 4
self.patch_size = 4
self.lr_patch_size = 2  # using stride=2 conv
self.lr_block_size = self.block_size * self.patch_size // self.sr_scale // self.lr_patch_size
```



**在 LR feature 上以 block=48x48 区域**，unfold，每个 fold 特征 `C=1280*48*48`；调整每个 block 的 image feature 为 `(b h*w c)`

```python
unFold = torch.nn.Unfold(kernel_size=3 * self.lr_block_size, stride=self.lr_block_size,  # self.lr_block_size=16
                                 padding=self.lr_block_size)
lr_imgs = unFold(lr_imgs)  # [1, 1280, 240, 416] -> [1, 2949120=1280*48*48, 390=((240 + 32 - 48) / 16 + 1) * (416 + 32 - 48) / 16 + 1]

lr_imgs = lr_imgs.view(lr_imgs.shape[0], lr_hidden_size, self.lr_block_size * 3, self.lr_block_size * 3, -1)
lr_imgs = lr_imgs.permute(0, 4, 2, 3, 1).contiguous()  # b n h w c [1, 390, 48, 48, 1280]
lr_imgs = lr_imgs.view(lr_imgs.shape[0] * lr_imgs.shape[1], -1, lr_imgs.shape[-1])  # [390, 2304, 1280]
```

**Block =48x48 区域提取的特征，patch 为 4x4 像素区域，所以每个 block 有 3x3 个 patch (Fig3 图里面也画得是这样，后面有些不一致，但在 Cross-LR 是这样的）**

1. 以 patch=4个基本单元（像素or特征点） 为最小的 patch

2. block_size = 128 个基本单元（像素or特征点），就是单边 128/4=32 个 patch，组成一个 block

3. **cross-LR-image 没有用全图**，在LR 图像中取 48x48 patch，**相当于 resized LR 中 96x96 patch**

   > :warning: PS：**这里 cross-LR-image 放的不是整张图，是一个比当前 block 对应到原始 LR 图像中更大一丢的区域，code 里面用 resized LR block=32x32, 原始LR 用的 48x48 区域，对应 resized LR 上面 96x96 patch；**
   >
   > resized LR 上面取 block=32x32 个patch，对应 128x128 区域，sr=4, 相当于原始 LR 32x32 区域，由于 LR image 用了 stride = 2 的 conv，感受野相当于原始 LR 的 16x16 区域 :star: ；这里 cross 用的 LR-image block 是 48x48 block，相当于 resized LR 上面 96x96 block 大了 3 倍。
   >
   > ```python
   > lr_id = i * block_w + j
   > output, *output_per_layers = self.model_forward(*args, hw=[vit_block_size, vit_block_size], mems=mems, inference=1, lr_imgs=lr_imgs[lr_id:lr_id+1], **kwargs)
   > ```

<img src="docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig3.png" alt="fig3.png" style="zoom:50%;" />



**`def cross_attention_forward(self, hidden_states, lr_imgs, **kw_args)`**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L338
>
> init https://vscode.dev/github/THUDM/Inf-DiT/blob/main/sat/model/transformer.py

这里的 `lr_imgs` 是包含当前 32x32 block 信息在 LR 中一个大 3 倍的 block;

- MLP 分为 QKV，LayerNorm
- 加 `lr_query_position_embedding`

```python
self.lr_query_position_embedding = nn.Parameter(torch.zeros(1, self.block_size ** 2, head, hidden_size_per_attention_head))  # [1, 1024, 16, 80]
self.lr_key_position_embedding = nn.Parameter(torch.zeros(1, (self.lr_block_size * 3) ** 2, head, hidden_size_per_attention_head))  # [1, 2304, 16, 80]

query_layer = query_layer + origin.lr_query_position_embedding
key_layer = key_layer + origin.lr_key_position_embedding
```

- attn + MLP

- 在外面和输入相加 + LayerNorm

```python
cross_attention_output = self.cross_attention_forward(cross_attention_input, **kwargs)

hidden_states = hidden_states + cross_attention_output  # hidden_states 是 self-attn 出来的特征
mlp_input = self.post_cross_attention_layernorm(hidden_states)

mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
mlp_output = layer.mlp(mlp_input, **kwargs)
```

- FFN：Cross attention 模型的结果再要过一次 MLP，再和 self-attn 出来的特征相加

```
if self.transformer.layernorm_order == 'sandwich':
	mlp_output = layer.fourth_layernorm(mlp_output)
hidden_states = hidden_states + gate_mlp * mlp_output
```



#### Final-Forward

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L73

![fig2](docs/2024_03_Arxiv_Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-Synthesis_Note/fig2.png)

- 28 个 DiT block 出来的特征 Norm 一下，和 timestep embedding **调制一下**
- MLP 把每个 block 的特征转换为 pixel，每个 block 有 4x4 个 c=3 的像素 :star:
- 初始化很重要！
  - MLP 初始化 bias =0 ！
  - 调制权重为 0

```python
class FinalLayerMixin(BaseMixin):
    def __init__(self, hidden_size, patch_size, num_patches, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def final_forward(self, logits, **kwargs):
        x, emb = logits, kwargs['emb']  # [1, 1024, 1280], [1, 1280]
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # [1, 1024=32*32, 1280] -> [1, 1024, 48]
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None), hw=kwargs.get('hw', None))

    def reinit(self, parent_model=None):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
```

- unpatch

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L35
>
> - https://pytorch.org/docs/stable/generated/torch.einsum.html#torch-einsum

```python
def unpatchify(x, c, p, rope_position_ids=None, hw=None):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    if False:
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum('nlpqc->ncplq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    
    return imgs
```





#### EDM-Sampler :question:

> 比 DDPM，DDIM 更优的采样方式

- "Elucidating the Design Space of Diffusion-Based Generative Models" NIPS, 2022 Jun 1
  [paper](http://arxiv.org/abs/2206.00364v2) [code](https://github.com/NVlabs/edm) [pdf](./2022_06_NIPS_Elucidating-the-Design-Space-of-Diffusion-Based-Generative-Models.pdf) [note](./2022_06_NIPS_Elucidating-the-Design-Space-of-Diffusion-Based-Generative-Models_Note.md)
  Authors: Tero Karras, Miika Aittala, Timo Aila, Samuli Laine

![tb1](docs/2022_06_NIPS_Elucidating-the-Design-Space-of-Diffusion-Based-Generative-Models_Note/tb1.png)

对照上面公式

```python
class EDMPrecond:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise
```



![ag1](docs/2022_06_NIPS_Elucidating-the-Design-Space-of-Diffusion-Based-Generative-Models_Note/ag1.png)



### training

**Loss**

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/loss.py#L74

**模型直接输出 SR_img_pred** 和 GT_img 做 L2 Loss

```python
    def __call__(self, model, images, text_inputs=None, rope_position_ids=None):
        concat_lr_imgs, lr_imgs, sr_imgs = images
        kwargs = {}
        if model.image_encoder:
            image_embedding = model.image_encoder(lr_imgs)
            kwargs["vector"] = image_embedding
        images = sr_imgs
        sigmas = self.sigma_sampler(images.shape[0]).to(images.dtype).to(images.device)
        noise = torch.randn_like(images).to(images.dtype)
        noised_images = images + noise * append_dims(sigmas, images.ndim)  # xT
        model_output = model(images=noised_images, sigmas=sigmas, text_inputs=text_inputs, concat_lr_imgs=concat_lr_imgs, lr_imgs=lr_imgs, **kwargs)
        w = append_dims(self.weighting(sigmas), images.ndim)
        return self.get_loss(model_output, images, w)
```



- Q：多个 patch 怎么训练？？

> https://vscode.dev/github/THUDM/Inf-DiT/blob/main/dit/model.py#L731

直接把一个图（单个patch）去优化，没有用 memory

```python
images = torch.cat((images, concat_lr_imgs), dim=1)

kwargs["images"] = images * c_in
kwargs["sigmas"] = c_noise.reshape(-1)
direction = "lt"
if self.random_direction and torch.rand(1) > 0.5 and not inference:
    direction = "rb"
    if self.random_direction and sample_step is not None and sample_step % 2 == 1:
        direction = "rb"
        # switch direction
        # if direction == "rb":
        #     direction = "lt"
        # else:
        #     direction = "rb"
        kwargs["direction"] = direction
        output, *output_per_layers = self.model_forward(*args, hw=[h//self.patch_size, w//self.patch_size], rope_position_ids=rope_position_ids, lr_imgs=lr_imgs, **kwargs)
        output = output * c_out + images[:, :self.out_channels] * c_skip

        #calc attention score
        if self.collect_attention is not None:
            self.collect_attention.append(output_per_layers)

            # output = output.to(in_type)
            return 1. / self.scale_factor * output
```



- Q：训练数据？

使用 LAION-5B 子集 $1024^2$ 大小图像 & 网上的高分辨率图像

> Our dataset comprises a subset of LAION-5B [25] with a resolution higher than 1024×1024 and aesthetic score higher than 5

对大图 crop or (crop+resize) **用 512x512 crop 出来的图进行训练**

> Following the previous works [20,23,30], we use fixed-size image crops of 512×512 resolution during training.

> When processing training images with a resolution higher than 512, there are two alternative methods: directly performing a random crop, or resizing the shorter side to 512 before performing a random crop
>
> Therefore, in practice, we randomly select from these two processing methods to crop training images.





## Limitations

- Q： 发现 LR 加噪声作为 xT 输出图像有可能变得很模糊

> However, this method can sometimes lead to the generated images appearing somewhat blurred just like LR images.

没说咋解决。。目前大家都这么搞





## Summary :star2:

> learn what

- 主打显存更低

  - save more than 5× memory when generating 4096 × 4096 images

    对比下直接过

- 主观可视化，搞一个窗口可以看 LQ，生成结果

  https://imgsli.com/MjYyMTU5

- 初始噪声的影响很大的:star:

直接拿纯黑图像来做 SR 看看，设计的 toy example 很巧妙

<img src="docs/2024_05_Arxiv_Inf-DiT--Upsampling-Any-Resolution-Image-with-Memory-Efficient-Diffusion-Transformer_Note/fig11.png" alt="fig11.png" style="zoom:50%;" />





### how to apply to our task

- 和其他 patch 做 cross-attn？在 self-attn 里面的 KV concat 起来就好
- 可以把整张图 resize 到 224 计算 CLIP-image-embedding，加到 timestep-embedding 里面
  - 在 28 个 LayerForward block 里面，**只有第 0 个 block用 Cross-attn!** 其他 block 只做 layernorm :warning:


![fig2](docs/2024_03_Arxiv_Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-Synthesis_Note/fig2.png)

