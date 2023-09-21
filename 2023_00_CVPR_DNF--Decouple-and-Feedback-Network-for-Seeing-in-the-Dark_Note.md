# DNF: Decouple and Feedback Network for Seeing in the Dark

> "DNF: Decouple and Feedback Network for Seeing in the Dark" CVPR Highlight, 2023
> [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.html) [code](https://github.com/Srameo/DNF)
> [paper local pdf](./2023_00_CVPR_DNF--Decouple-and-Feedback-Network-for-Seeing-in-the-Dark.pdf)

Q: Single stage(Mixed mapping across RAW and sRGB domains, RAW space 的噪声映射到位置分布的 color space) & multi-stage (多阶段 pipeline 存在累积 loss ) 效果不好

**domain-specific decoupled & feedback info** 分解为 noisy-to-clean and RAW-to-sRGB 任务

- CID block for RAW denoise, MCC block for color, GFM encoder 接收 decoupled feature

  - MCC global matrix transformation（CxC 矩阵相乘调整 C）**实现调整颜色**
  - incorporate a Denoising Prior Feedback mechanism to avoid error accumulation across stages

- multi-scale features from the RAW decoder as denoising prior. **代替使用 denoise 不准确的 RAW image** :thinking:

  - 用去噪 Decoder 里面的特征，融合到 Encoder 里面。**怎么学 Decoder ？？**

    Decoder 特征 make noises more distinguished, serve as guidance for further denoising >> 存入多个任务共享的 Encoder 继续 denoise

  - Gated Fusion Module (GFM) 模块：融合 Decoder 的 feedback，用 a point-wise convolution and a depth-wise convolution 过滤一下 :+1:

  - Residual Switch Mechanism (RSM) **调节 CID 里面残差跳连是否开启，来区分 noise & signal** :star:

    local shortcut of the CID block during the color restoration provides more information about the image content, thus resulting in higher performance.

- Dataset

  SID 有 normal light reference; MCR no GT in RAW >> select images with the longest exposure time of each scene as GT