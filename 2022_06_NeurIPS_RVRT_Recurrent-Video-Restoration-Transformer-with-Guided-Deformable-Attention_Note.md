# RVRT

> "Recurrent Video Restoration Transformer with Guided Deformable Attention" NeurlPS, 2022 June, **RVRT** :statue_of_liberty:
> [paper](https://arxiv.org/abs/2206.02146) [code](https://github.com/JingyunLiang/RVRT?utm_source=catalyzex.com)
> [paper local pdf](./2022_06_NeurIPS_RVRT_Recurrent-Video-Restoration-Transformer-with-Guided-Deformable-Attention.pdf)

## **Key-point**

> Introduction 总结了现有 video restoration 两类框架：Parallel & recurrent



## **Contributions**

- RVRT将视频分成多个片段，**以 clip 为单位提取特征**，利用先前的片段特征来估计后续的片段特征。通过减小视频序列长度并且以更大的隐藏状态传递信息
- 使用引导变形注意（GDA）从整个推断片段中预测多个相关位置，然后通过注意机制聚合它们的特征来进行片段间对齐。



## **Related Work**

现有的视频恢复方法主要有两种 ：

1. 并行恢复所有帧，但是模型尺寸大，内存消耗大

   Parallel methods estimate all frames simultaneously, as the refinement of one frame feature is not dependent on the update of other frame features

   - "Spatio-Temporal Filter Adaptive Network for Video Deblurring" STFAN
     [paper](https://arxiv.org/abs/1904.12257) [code](https://www.github.com/sczhou/STFAN)

2. 循环逐帧恢复，它跨帧共享参数所以模型尺寸较小，但是缺乏长期建模能力和并行性



## **methods**

> [RVRT code](https://github.com/JingyunLiang/RVRT/blob/main/models/network_rvrt.py#L742)

将 T 帧的视频按 `window_size=N` 分为 $T/N$ 段，每段里面





### Guided Deformable Attention :star:





## **Experiment**

### Dataset

randomly crop 256 × 256 HQ patches and use different video lengths for different datasets: **30 frames for REDS** [53], 14 frames for Vimeo-90K [87], and 16 frames for DVD [63], GoPro [54] as well as DAVIS [31]

- Vimeo-90K

  > "Video Enhancement with Task-Oriented Flow" IJCV 2019
  > [website](http://toflow.csail.mit.edu/) [sample-clip-from-viemo-90K](https://data.csail.mit.edu/tofu/dataset.html)

  build a large-scale, high-quality video dataset, Vimeo90K. This dataset consists of **89,800 video clips downloaded from [vimeo.com](http://toflow.csail.mit.edu/vimeo.com),** which covers large variaty of scenes and actions. It is designed for the following four video processing tasks: temporal frame interpolation, video denoising, video deblocking, and video super-resolution.

**SR Task**

- BI degradation

  train the model on two different datasets: REDS [53] and Vimeo-90K, and then test the model on their corresponding testsets: REDS4 and Vimeo-90K-T

- BD degradation

  train on Vimeo-90K and test it on Vimeo-90K-T, Vid4, and UDM10 [89]

**Deblurring**

DVD [63] and GoPro [54], with their official training/testing splits

**Denoising**

training set of DAVIS [31], test on DAVIS-test-set & Set8 

- Set8 (usually used as test set)

  *Set8* is composed of 8 sequences: 4 sequences from the *Derf 480p* testset ("tractor", "touchdown", "park_joy", "sunflower") plus other 4 540p sequences. You can find these under the *test_sequences* folder [here](https://drive.google.com/drive/folders/11chLkbcX-oKGLOLONuDpXZM2-vujn_KD?usp=sharing).

## Code

### **RSTB**

先转到 `(b c t h w)` 用 3D 卷积

```python
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]
```

> In MRSTB, we upgrade the original 2D h × w attention window to the 3D N × h × w attention window





**Limitations**

## **Summary :star2:**

> learn what & how to apply to our task