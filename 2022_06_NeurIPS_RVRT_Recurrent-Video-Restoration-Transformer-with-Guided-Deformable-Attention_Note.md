# RVRT

> "Recurrent Video Restoration Transformer with Guided Deformable Attention" NeurlPS, 2022 June, **RVRT** :statue_of_liberty:
> [paper](https://arxiv.org/abs/2206.02146) [code](https://github.com/JingyunLiang/RVRT?utm_source=catalyzex.com)
> [paper local pdf](./2022_06_NeurIPS_RVRT_Recurrent-Video-Restoration-Transformer-with-Guided-Deformable-Attention.pdf)

## **Key-point**

**Contributions**

- RVRT将视频分成多个片段，利用先前的片段特征来估计后续的片段特征。通过减小视频序列长度并且以更大的隐藏状态传递信息'
- 使用引导变形注意（GDA）从整个推断片段中预测多个相关位置，然后通过注意机制聚合它们的特征来进行片段间对齐。



## **Related Work**

现有的视频恢复方法主要有两种 ：

1. 并行恢复所有帧，它具有时间信息融合的优势，但是模型尺寸大，内存消耗大
2. 循环逐帧恢复，它跨帧共享参数所以模型尺寸较小，但是缺乏长期建模能力和并行性

## **methods**

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



**Limitations**

## **Summary :star2:**

> learn what & how to apply to our task