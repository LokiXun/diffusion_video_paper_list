# Memory-Augmented Non-Local Attention for Video Super-Resolution

> "Memory-Augmented Non-Local Attention for Video Super-Resolution" CVPR, 2021 Aug
> [paper](https://arxiv.org/abs/2108.11048) [code](https://github.com/jiy173/MANA) [blog_explanation](https://zhuanlan.zhihu.com/p/552844253)
> [paper local pdf](./2021_08_CVPR_Memory-Augmented-Non-Local-Attention-for-Video-Super-Resolution.pdf)

## **Key-point**

**memory mechanism**；视频帧间存在较大的运动时，对齐的准确性很难保证。另外，相似相邻帧可利用的有用信息有限，这在某种程度上限制了视频超分辨方法的性能。
**提出更加鲁棒的多帧信息融合的方法**；受reference-based和general image prior-based 的单图像超分方法，引入 general video prior 来辅助视频超分辨

对于当前帧 $I_t$，**输入为 neighbor frames** >> 更好挖掘 neighbor 信息，并且**用一个 memory 机制记录下对其他视频做 SR 时候的经验**



**Contributions**

- 提出cross-frame non-local attention 模块
- memory-augmented attention模块，引入 general video prior
- 提出Parkour 数据集，该数据集中视频帧间存在大的位移



## **Related Work**

## **methods**

### Cross-Frame Non-local Attention

引入高斯加权来减轻像素匹配错误的影响



### Memory-Augmented Attention :question:

global memory bank 是**作为网络参数来学习**，提供一种视频的先验信息







## **Experiment**

> ablation study 看那个模块有效，总结一下

## **Limitations**

- 三阶段训练



## **Summary :star2:**

> learn what & how to apply to our task