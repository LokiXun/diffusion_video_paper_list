# 2023_05_CVPR_Blind-Video-Deflickering-by-Neural-Filtering-with-a-Flawed-Atlas

> "Blind Video Deflickering by Neural Filtering with a Flawed Atlas" CVPR, 2023 Mar
> [paper](https://arxiv.org/abs/2303.08120) [code](https://github.com/ChenyangLEI/All-In-One-Deflicker?utm_source=catalyzex.com) [website](https://chenyanglei.github.io/deflicker/)
> [paper local pdf](./2023_05_CVPR_Blind-Video-Deflickering-by-Neural-Filtering-with-a-Flawed-Atlas.pdf)



## **Key-point**

The core of our approach is utilizing **the neural atlas** in cooperation with a neural filtering strategy. 
**The neural atlas** is a unified representation for all frames in a video that provides temporal consistency guidance but is flawed in many cases. To this end, **a neural network is trained to mimic a filter** to learn the consistent features (e.g., color, brightness) and **avoid introducing the artifacts in the atlas**.

引入**整个视频共享的单个 atlas 作为时间一致性信息 **，认为 atlas 颜色是较为准确的，但可能 structure 保存不好。作者就构造了一个 filter (U-net) 融合原始帧的 structure 信息和 atlas 较好的颜色信息，输出第 t 帧的过滤后的信息。之后又发现有 local flickering，作者参考已有方法，一个轻量 ConvLSTM 结合过滤后的 t, t-1 帧和之前预测 t-1 时刻的预测结果，融合得到当前帧结果

- 能够用于解决：flickering video；生成的视频 or 上色等，算法处理后存在 flickering 的问题；**老视频闪烁问题**

- Potential application

  all evaluated types of flickering videos;
  novel view synthesis
  **old videos artifacts 作者说后续工作要做** :warning:



**Contributions**

- 引入 atlas 视频一致性标准到 video flickering 任务
- 构造了一个过滤网络，修复 atlas 结果扭曲的缺陷



## **Related Work**

- [deep-video-prior (DVP)](https://github.com/ChenyangLEI/deep-video-prior)
  Blind Video Temporal Consistency via Deep Video Prior
- "Layered Neural Atlases for Consistent Video Editing" SIGGRAPH, 2021 Sep :star:
  [paper](https://arxiv.org/abs/2109.11418) [website](https://layered-neural-atlases.github.io/) [code](https://github.com/ykasten/layered-neural-atlases)

- Baseline

  As our approach is the first method for **blind deflickering**, no existing public method can be used

  - ConvLSTM

    > "Learning Blind Video Temporal Consistency" ECCV, 2018 Aug
    > [paper](https://arxiv.org/abs/1808.00449v1)

  - "Blind video temporal consistency"
    [paper](https://dl.acm.org/doi/abs/10.1145/2816795.2818107)

  - DVP "Blind video temporal consistency via deep video prior"
    [paper](https://proceedings.neurips.cc/paper/2020/hash/0c0a7566915f4f24853fc4192689aa7e-Abstract.html)



## **methods**

![blind_flickering_structure.png](./docs/blind_flickering_structure.png)

### Flawed atlas generation

两个 implicit neural network MLP 训练 atlas，得到类似 Nerf 的建图信息来表示视频。从 atlas 里面获取物体全局的颜色，避免闪烁



### Neural Filtering and Refinement

filters 对 atlas 输出的结果进行过滤
**认为 altas 信息颜色准确，structure 失真，因此和原始帧融合一下送入 filter 进行过滤**

- 训练方式
  - 用 COCO 数据，每单张图构造颜色失真 -> I_t 和结构扭曲 -> A_t 的数据对
  - 训练一个 U-net, 把 $I_t$ & $A_t$ 通道 concat 起来输入得到 $O_t^f$



### local flickering 调整

类似 “ReBotNet: Fast Real-time Video Enhancement” AeXiv, 2023 Mar 视频实时去模糊，里面的 recurrent-setup，就是**用上一帧的预测结果，一起作为当前帧预测的输入，实现 temporal consistency**。

结合 $O_{t-1}, O^f_{t}, O^f_{t-1}$ 过 Conv & Res & ConvLSTM 得到 $O_{t}$



## Experiment

> - [ ] why use MaskRCNN to have mask?
>
>   help learn atlas better>> separate human foreground 
>
> - [ ] how IMLP  work?

- Comparison with baseline
  ConvLSTM, ours atlas only, ours

- Comparisons to blind temporal consistency methods

- **Ablation Study**

  - Neural filtering

  - Local refinement

- Comparison with human Expert

  人工用商业软件 `RE:Vision DE:Flicker` 去修复，进行对比



### Dataset

作者说创建的是第一个公开的 blind flickering 数据集

- Real-world data **没有 GT >> evaluation**
  contain various types of flickering artifacts

  - **old_movie，60 * video 存储为 `%05d.jpg` 大多为 350 帧图像，若 fps=25，约为 10-14s的视频。**


  - **old_cartoon，21 * video 图像格式存储，大多为 50-100 帧，约为 1 - 4s 视频**
  - Time-lapse videos capture a scene for a long time, and the environment illumination usually changes a lot
  - Slow-motion videos can capture high-frequency changes in lighting
  - algorithms processed result

  

- 合成 flickering 的视频数据

  - human expert use commercial software for de-flickering

    [RE:VISION. De:flicker](https://revisionfx.com/products/deflicker/)

- metrics

  - DVP

  - warping error

    预测结果与原始 GT，乘上制造模糊的 mask ，最后计算 L1 loss, 





## **Summary :star2:**

> learn what & how to apply to our task

- 用类似 Nerf Implicit Neural Network 对整个视频得到一个共享的 atlas 表征，来表示视频颜色的一致性特征。之后每帧计算时，直接用当前帧的位置输入 atlas 去查询得到 $A_t$ 
  - 其他方法拿过来用发现效果不好，找到是 atlas structure 不行，但颜色还行的特点，构造了一个 filter 融合原始帧一起处理
  - local flickering 有现成 pipline 可以拿过来用

- **提供的 old_movie, old_cartoon 也有闪烁问题**，可以借鉴这里修复的思路（本文作者也说后需要做 old_film 修复，解决 video consistency 问题）


