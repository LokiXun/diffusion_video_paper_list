# Align and Prompt: Video-and-Language Pre-training with Entity Prompts

> "Align and Prompt: Video-and-Language Pre-training with Entity Prompts" CVPR, 2021 Dec
> [paper](https://arxiv.org/abs/2112.09583) [code](https://github.com/salesforce/ALPRO) [blog_explanation](https://zhuanlan.zhihu.com/p/453020334)
> [paper local pdf](./2021_12_CVPR_Align-and-Prompt--Video-and-Language-Pre-training-with-Entity-Prompts.pdf)

## Key-point

先前大多数方法都使用基于标准Transformer的多模态编码器捕获跨模态交互，没有完全解决**视觉特征和文本特征的未对齐问题**。

提出了一个利用 sparse frames 做 video-language 预训练的框架 ALPRO，效果在 VQA，video-text 检索上 SOTA。**主要是提出了 VTC 和 PEM 这两个 loss**，VTC loss来提升模型对于视频整体的特征，PEM loss 用于提升视频局部区域的特征，PEM模块里面设计了一个 **prompter 生成一堆 prompt 作为伪标签去训练模型，避免去认为标注数据。**



**Contributions**

- 只用稀疏几帧就能提升性能的 video-text 预训练框架
- VTC, PEM loss 实现不用 detector 就能提升局部区域特征与文本的对应能力
- prompter 生成伪标签去以监督方式训练模型



## **Related Work**

- "ActBERT: Learning Global-Local Video-Text Representations" CVPR, 2020 Nov
  [paper](https://arxiv.org/abs/2011.07231)
- "Is Space-Time Attention All You Need for Video Understanding?" ICML, 2021 Feb, **TimeSformer**
  [paper](https://arxiv.org/abs/2102.05095)



**视频通常在连续帧中包含更多冗余，容量和计算效率有挑战**

1. 用**预训练的模型，离线提取的特征**存下来，来规避昂贵的计算开销

   pretrained visual backbone, 在图像 or 没有文本匹配的视频数据上训练，用到 Visual-language 存在 feature gap; 并且对于下游任务，这些预训练模型没有 finetune，特征很难用于下游任务。

2. 用 sparse frames 去 finetune 预训练模型

   - ClipBERT
   - FiT



**Video & 文本联合预训练模型**

1. 离线用对比学习，不改动视觉模型，提取到特征对齐程度有限
2. 目标检测器去获取物体对应标签

ALPRO 对单模态，模态融合 Encoder 一起训练，进一步解决对齐问题 && detector free



**Zero-shot Visual Recognition with Prompts**

CLIP 用大量固定的 prompt 和视觉特征计算相似度

本文参考 CLIP 设计 Prompter 模块



## **methods**

![](https://pic2.zhimg.com/80/v2-143ad9fa49e024a16a75301806cc20fd_1440w.webp)



ALPRO consists of two main modules, a **video-language pre-training model** and a **prompter**

模型框架：TimeSformer，Transformer

使用4 个 Loss: MLM, VTM, **VTC, PEM Loss** 训练模型，主要是提出了 **VTC, PEM Loss** 这两个 loss，用于增强 cross-modal alignment.

- VTC loss 强调视频实例级别的对齐 video-text
- PEM loss 强调视频局部区域与文本实例的对齐



### VTC  loss


$$
Similarity:~S(V,T) = g(V_{cls}) * g(T_{cls})\\
\cal{L_{t2v}} = -\log{(s(V,T) 取指数平均)}
\\
\cal{L_{VTC}} = 1/2(\cal{L}_{x2t} + \cal{L_{t2v}})
$$

- PEM Loss

  用 prompter 模块生成伪标签 $q_{\hat{V}}$ ，例如 "dog, ..."

  > cropped video 和 语料库中实体组成的句子 （预先选 M 个文本）计算相似度



### prompting entity modeling(PEM)

improves the models’ capabilities in capturing local regional information and strengthening cross-modal alignment between video regions and textual entities.

视频帧里面的各个物体没有对应的文本标注，导致模型输出的特征对于局部实体的信息不足。作者构造了一个 prompter 模块去生成伪标签（随机 cropped 后视频中实体的 soft pseudo labels），后续用于监督训练（设计 loss）。

Prompter 模块含有单独的视觉，文本 encoder，用前面的 VTC Loss 在 video-text pairs 上先训练，之后冻住 prompter 去训练模型。
**Prompter 每次输出伪标签与 cropped video 相似的概率。**具体来说，M 个 text prompts, `A video of {Entity}` 里面的 entity 用语料库里面频繁出现的实体（例如 dog, ...）。视频先随机选个区域整体 crop 出来，过 video encoder 得到的特征，和 M 个文本得到的特征计算相似度，得到和每个文本匹配的概率 $q_{\hat{V},m}$

> 对相似度低于 0.2 的实体，直接把这个实体从  cropped video 对应的M个实体中丢弃

$$
q_{\hat{V},m} = \frac{exp(S(\hat{V}, T_m) / \tau)}{\sum_{i=1}^{M}\exp(s(\hat{V}, T_i) /\tau)}
$$



- PEM Loss

  **视频随机 crop** 后用 prompter 输出的与 M 个文本匹配的概率 $q_{\hat{V},m}$。

  要训练的模型，视频整体输入 multimodal encoder 输出特征，**没有 cropped 的区域对应的特征做 `max-pooling`**，得到特征 $e_{\hat{V}}$ ，之后**用一个分类器 MLP** 预测 prompter 生成的 M 个伪标签对应的概率 $p_{\hat{V},m}$。

  **两个概率做 cross-entropy。实现不需要人为地，去标注视频局部实例-文本对数据** :star:

$$
\cal{L}_{pem} = -\sum_{m=1}^{M}{ q_{\hat{V},m} * \log{p_{\hat{V},m}}}
$$




## **Experiment**

> ablation study 看那个模块有效，总结一下

在 text-video retrieval and video question answering tasks 任务上测试，验证提出的 VTC，PEM loss 对于模型特征增强的有效性。

- 输入视频不同宽长比，直接先缩放到 224x224 输入网络，实验发现效果没有明显下降
- pretrain:  10 epochs, batchsize=256, 在 16 个 A100 训练。finetune 8*A100, 1-5h



- 与现有 text2video 检索任务上，两个数据集上，对比已有 finetune 和 zero-shot 方法，效果 SOTA
- 现有 VQA 方法，效果 SOTA top1 acc 提升 2.6%

### ablation study

- 加入不同 loss 的比较

  加了 PEM,VTC 比只用 MLM+VTM 效果提升明显

- **prompt design & ensemble**

  ensembling: 用 12 个文本模板 A video of, A fotage of... 对同一实例编码后，对 $cls~token$ 取平均，效果和不做 ensemble 差接近 1%

  **prompt ensembling** demonstrates its importance in generating high-quality pseudo-labels

- prompter 实例个数（M 的个数）

  发现加到 1000 一直在升，**到 2000 会加入许多不频繁的实例，产生噪声反而降低效果。**

- frame num

  对于 VQA，text2video 检索只取稀疏的几帧输入模型，例如随机取 8 帧，发现帧数越多越好



## **Limitations**

- 用于训练的数据有私人数据，有隐私问题？



## **Summary :star2:**

> learn what & how to apply to our task

- prompter 用语料库生成大量 prompt 去预训练一个局部视频的特征和文本的对应，生成得到的伪标签，去监督地训练模型。避免人工标注数据。

  监督方式训练，可以通过 cross-entropy 体现。

