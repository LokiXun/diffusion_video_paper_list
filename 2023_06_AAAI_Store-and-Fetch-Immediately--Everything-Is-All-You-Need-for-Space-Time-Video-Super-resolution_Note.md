# Store and Fetch Immediately

> "Store and Fetch Immediately: Everything Is All You Need for Space-Time Video Super-resolution" AAAI, 2023 Jun :warning:
> [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25165)
> [paper local pdf](./2023_06_AAAI_Store-and-Fetch-Immediately--Everything-Is-All-You-Need-for-Space-Time-Video-Super-resolution.pdf)

## **Key-point**

- Task: Spatial-Temporal VSR
- Background

- motivation

  recurrent setup 只用一个 hidden state 来代表全部视频，明显不合理，只是目前针对短视频还能凑活用。**整个视频内的信息没有用全**

Store and Fetch Network





## **Contributions**



## **Related Work**

> ST-VSR methods are roughly divided into **three categories: two-stage, one-stage and compact one-stage based methods**

### ST-VSR methods

![image-20231001115044303](.\docs\2023_06_AAAI_Store-and-Fetch-Immediately--Everything-Is-All-You-Need-for-Space-Time-Video-Super-resolution_Note\image-20231001115044303.png)

- compact 1 stage

  - "You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution"
    [paper](https://arxiv.org/abs/2207.06345)

    temporal consistency:  spatial-temporal feature aggregation

    缺陷：only short-range spatial-temporal correlations
    **some complementary spatial and temporal information from distant frames also matter for ST-VSR under large motion and occlusion scenarios**



### Store&Fetch

- "Multi-Scale Memory-Based Video Deblurring" CVPR, 2022 Oct :star:
  [code](https://github.com/jibo27/MemDeblur)
- Reconstruction details: progressive fusion and reconstruction (F&R)
  - "Omniscient video super-resolution"
  - "You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution"




## **methods**

### Fetch block

Motivation：从其他帧抽取与当前帧有关的信息

#### AF

参考其他方法的现有模块

> - "Zooming slow-mo: Fast and accurate one-stage space-time video super-resolution"
>   [code](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)
>
>   explore spatial information from adjacent frames, ConvLSTM + Deformable CNN
>
> - "Temporal Modulation Network for Controllable Space-Time Video Super-Resolution" CVPR, 2021 Apr
>   [paper](https://arxiv.org/pdf/2104.10642.pdf) [code](https://github.com/CS-GangXu/TMNet)
>
>   有效融合 Adjacent frame 相邻帧的运动信息

参考 EDVR 发现**当与参考帧有较大运动，deformable CNN 训练很不稳定**，overflow 的 offset 降低生成效果。论文中按帧间距分类，**对不同 motion 程度的帧用不同方法去融合特征**。认为相邻帧 motions 较小用 AF，相隔较远的帧用提出的 DF。

> "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks" CVPR workshop 1st, 2019 May
> [paper](https://arxiv.org/abs/1905.02716) [code](https://github.com/xinntao/EDVR)
>
> *对应到视频修复，STDAN 每两帧之间进行融合，认为所有参考帧 motions 类似，对存在较大 motions 的帧没有特别处理，这些帧 DCN 预测的 offset 偏差较大，还进行加权融合降低了效果*。 :star:









## **Experiment**

> ablation study 看那个模块有效，总结一下

**Limitations**

**Summary :star2:**

> learn what & how to apply to our task