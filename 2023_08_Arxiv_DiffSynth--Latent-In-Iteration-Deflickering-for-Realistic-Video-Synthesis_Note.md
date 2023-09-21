# DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis

> "DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis" Arxiv, 2023 Aug :star:
> [paper](https://arxiv.org/abs/2308.03463) [website](https://anonymous456852.github.io/) [code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth)
> [paper local pdf](./2023_08_Arxiv_DiffSynth--Latent-In-Iteration-Deflickering-for-Realistic-Video-Synthesis.pdf)

## **Key-point**

直接用 stable diffusion 生成每帧图像，去噪迭代过程中帧与帧之间差异越来越大，存在累计误差使得生成的视频一致性较差。

DiffSynth 主要提出两个模块 （`Latent In-Iteration Deflickering` ，`Patch Blending Algorithm` ）对 diffusion 去噪时候得到的 latent code 进行处理，并借鉴 patch mapping 实现视频去闪烁 & 融合前后帧信息。
具体来说，`Latent In-Iteration Deflickering` 对 Diffusion latent code 进行去闪烁 ，中间步骤 latent code $\hat{Z_0}$ decode 的得到的 local window 各帧，结合 Patch Blending Algorithm 融合前后帧信息，进一步融合信息 + 去闪烁，得到去闪烁后的图像。再 encode 为图像得到更新后的 $\hat{Z_0}$  去预测 $Z_{t-1}$ 

- DiffSynth 优势在于能够应用于**多种视频合成的下游任务**

  text-guided 视频风格转换，视频服饰替换，视频修复，3D渲染，不需要 cherry picking



**Contributions**

- 提出增强一致性的视频合成框架，能够用于多种下游任务
- 提出的两个模块，对 diffusion latent code 进行处理实现视频去闪烁 & 使用 patch matching 调整



## **Related Work**

- "Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks."

直接用 stable diffusion 生成每帧图像，迭代过程中帧与帧之间差异越来越大，累计误差。



### Patch Matching

> [paper](https://www.cs.princeton.edu/courses/archive/fall12/cos526/papers/barnes09.pdf) [blog_explanation](https://zhuanlan.zhihu.com/p/377230002)

提出的PatchMatch算法在寻找最近邻的速度上又是一次质的飞跃（比之前的SOTA还要快1-2个数量级），用户交互场景的实时图像编辑工具提供了强有力的算法支持







## **methods**

### Latent In-Iteration Deflickering

$x_{t-1} = a * \hat{x_0} + b *\epsilon$

利用 existing 去闪烁方法对 $x_0$ 修正，带入原来公式，预测 $\epsilon$



### Patch Blending Algorithm

> 第 i 帧出现的内容，在隔了很多帧的 j 帧也出现

**本文用 patch matching 匹配的角度去做**，本文为了实现简单，在 local window (8帧) 做 remap，原始计算复杂度 $O(n^2)$ >> 本文提出一个近似的方法降低复杂度

构造一个 mapping 表格，每隔两帧融合一次存入 row1，用上次的融合结果间隔两帧和此次融合结果再融合存入 row2

$[j \to i]$ 表示 remapping operator 操作符



### Other Modificiation

- fixed noise

  diffusion 去噪，各帧使用同样的高斯噪声开始

- 修改 U-net 实现输入的多尺度，针对下游任务，**实现可变分辨率**

- Cross frame attn

  将前后的 4 帧 concat 过 self-attn

  可以替换为 **ControlNet 的 **self-attn layer

- 修改 VAE 采样为确定方式采样，采样 $\mu, but~\sigma=0$ 
  Stable Diffusion VAE 出来的特征为高斯分布 >> 存在随机性，希望视频各帧出来的特征相似





## **Experiment**

> ablation study 看那个模块有效，总结一下

- Video restoration

  先 SR 方法处理，再用 Stable Diffusion 映射到 latent code，用 ControlNet 修复

  不从头开始



## **Limitations**

- 推理慢

  A10 GPU 1 帧图像要推理 1min

- blending operator 文章内直接去平均，优化为更好的 operator



## **Summary :star2:**

> learn what & how to apply to our task

- Patch Matching 方式提取相邻帧之间的特征

- 如何用 diffusion 实现视频增强，对中间步骤的 latent code $\hat{Z_0}$ 处理，再继续去噪。实现 diffusion 各帧的一致性，去闪烁

- Stable Diffusion 中间 layer 替换为 ControlNet 

  **文章中列出了许多 package 直接提取 edge，深度，额外的信息**