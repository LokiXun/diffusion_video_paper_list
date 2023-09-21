# Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

> "Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP" CVPR, 2022 Oct, **ov-seg**
> [paper](https://arxiv.org/abs/2210.04150) [code](https://jeff-liangf.github.io/projects/ovseg/?utm_source=catalyzex.com) [website](https://jeff-liangf.github.io/projects/ovseg/?utm_source=catalyzex.com) 
> [blog_explanation](https://mp.weixin.qq.com/s/sfW966KpFMa5M0oh3D3EpQ)
> [paper local pdf](./2022_10_CVPR_Open-Vocabulary-Semantic-Segmentation-with-Mask-adapted-CLIP.pdf)

## **Key-point**

> **Background**
> 开放词汇语义分割，拓展到模型没有见过的类别。目的是根据文本描述将图像分割成语义区域，这些区域在训练中可能没有看到

实现 open-vocabulary segmentation with **user-defined arbitrary queries**.

最近的两阶段方法首先生成类未知的mask proposal，然后利用预训练的视觉语言模型(例如CLIP)对mask区域进行分类。作者认为**这种范式的性能瓶颈是预训练的CLIP模型，因为它在 mask 图像上表现不佳**。提出了一种 **mask prompt tuning** 的方法，最大程度的减少了mask图像和自然图像的差距，最大程度发挥CLIP的能力

- **finetune CLIP** on a collection of masked image regions and their corresponding text descriptions

  从 COCO Image Caption 数据集，collect training data

- mask prompt tuning



**Contributions**

- 验证了假设，发现 CLIP 对 mask image 分类是当前问题的瓶颈
- 收集了一个数据集，用于微调 CLIP: mask-category pairs
- mask prompt tuning，不需要更新 CLIP 的参数
- open-vocabulary 的效果，第一次和 2017 年监督的分割任务匹配



## Related Work

利用预训练的CLIP来做 open-vocabulary 图像分割，这些工作的成功**依赖于两个假设**：1) 模型可以生成与类别无关的 mask proposal；2) 预训练的 CLIP 可以将其分类性能转移到 mask proposal上

> 发现直接用 CLIP 效果不好，实验验证假设

1. 假设有一个**完美的 mask proposal生成器 + 预训练的 CLIP 分类器**。
   作者使用ground-truth mask作为proposal mask，并将被mask的图像提供给预训练的CLIP进行分类。该模型在ADE20K-150数据集上的 mIoU 仅达到 20.1%。

2. 用一个一般的 mask 生成器 + GT 标签（mask 和 GT 区域计算 overlap）

   higher mIoU of 66.5%

**发现是预训练的CLIP不能对被mask的图像进行令人满意的分类**，mask 图像和 CLIP 的训练的 natural image 之间有 domain gap 引起的。

> 因此，解决方案是 **finetune  CLIP + 收集数据**

- collect training data by mining COCO Captions

  提取 captions 里面的名词，对 best-match proposal 区域，用 pretrained CLIP 去匹配名词作为标签

- 如何微调 CLIP？



- "Visual Prompt Tuning" ECCV, 2022 Mar
  [paper](https://arxiv.org/abs/2203.12119) [code](https://github.com/kmnp/vpt?utm_source=catalyzex.com)



## methods

> - [ ] MaskFormer

分为 3 个部分： two-stage open vocabulary segmentation 方法；如何收集数据集去 finetune CLIP；提出的 mask prompt tuning technique

### Two-stage models

![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfrfzeQEo5egjYsjtWv8fXXOTeJ0X8mVhJYhCAYZcK5xaZe3cCoFZhdZTw43C9cUy18hu4sYPduTicA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

![ov_seg_two_stage_proposal.png](./docs/ov_seg_two_stage_proposal.png)

输入图像大小 HxW，MaskFormer 预测 N 个 mask proposal 二值 mask && C-dimensional proposal embedding,

> C=512 for ViT-B/16 and 768 for ViT-L/14

训练时的 text prompt, 缺失的文本用 `COCO-Stuff` 数据集的 171 个类别。这样训练得到的 mask generator (MaskFormer) 限制了类别，但任意物体 mask 生成不是本文的研究内容。

> 框架的区别是在于 mask 图像 background 部分，使用可学习的 token 代替之前效果不好的 zero-token. 参考下面 Mask prompt tuning



### Mask prompt tuning

![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfrfzeQEo5egjYsjtWv8fXXOZOaBTjX5EWic76ssnglZOmZnGqSbPmfWa8x4s1kdaPlTialQrB51bPjA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图像打成 $N_p$ 个 patch (CLIP 里面 ViT backbone)，提取 E 维的特征 $T \in R^{N_p \times E}$
**learnable tensor** representing prompt tokens as $P \in R^{N_p \times E}$
MaskFormer 预测出来 $Mp \in \set{0, 1}^{N_p}$
$$
\text{Final input to Transformer}=T ⊗ M_p + P ⊗ (1 − M_p),
$$






### Dataset: self-labeling strategy

![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfrfzeQEo5egjYsjtWv8fXXOUbyejFBbB9P4AYxeJJibxJ8IddL4vWdUEbQukBo3xvKniazHh05DibvdQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

用预训练的 MaskFormer 获取 mask proposal 候选区域。用现成的文本分词器，把 captions 里面的名词取出来作为可能的类别。用 CLIP 去匹配 proposal 和待选的名词。

> 1.3M mask-category pairs with 27K unique nouns using 5 captions per image, or 440K pairs with 12K nouns using 1 caption per image





## Experiment

> ablation study 看那个模块有效，总结一下

- Datset

  train: COCO-Stuff 171 类别

  val: ADE20K >> one with 150 frequently used categories **(A-150)** and one with more diverse 847 categories **(A-847).**



- open-vocabulary models 比较

- Ablation Study

  - 数据的影响，COCO 原始类别，captions 名词

  - mask prompt tuning && full model finetune

    验证 mask prompt tuning 的有效性



## **Limitations**



## **Summary :star2:**

> learn what & how to apply to our task

- 行文思路，发现效果不好，去验证假设
- CLIP finetune 方式，对输入增加可学习的内容，不改变 CLIP 权重