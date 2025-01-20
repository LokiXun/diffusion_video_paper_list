# SVFR: A Unified Framework for Generalized Video Face Restoration

> "SVFR: A Unified Framework for Generalized Video Face Restoration" Arxiv, 2025 Jan 2
> [paper](http://arxiv.org/abs/2501.01235v2) [code](https://github.com/wangzhiyaoo/SVFR.git) [pdf](./2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration.pdf) [note](./2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note.md)
> Authors: Zhiyao Wang, Xu Chen, Chengming Xu, Junwei Zhu, Xiaobin Hu, Jiangning Zhang, Chengjie Wang, Yuqi Liu, Yiyi Zhou, Rongrong Ji

## Key-point

- Task: 多任务 Face Restoration(有 Film), **同时做 Blur, colorization and inpainting**
- Problems
  - 之前工作只做 SR 没做 inpaint。。。看起来是硬凑多个任务。。

- :label: Label:



## Contributions

- SVD 同时做 BFR, inpainting, and colorization tasks，搞了一个 learnable embedding 指定任务

> In this paper, we propose a novel approach for the Generalized Video Face Restoration (GVFR) task, which integrates video BFR, inpainting, and colorization tasks that we empirically show to benefit each other. We present a unified framework, termed as stable video face restoration (SVFR),

> We present a unified framework, termed as stable video face restoration (SVFR), which leverages the generative and motion priors of Stable Video Diffusion (SVD) and incorporates task-specific information through a unified face restoration framework.
>
> A learnable task embedding is introduced to enhance task identification

- Unified Latent Regularization (ULR)

> Meanwhile, a novel Unified Latent Regularization (ULR) is employed to encourage the shared feature representation learning among different subtasks

- 针对人脸，搞了特征点对齐进行训练 :star:

>  To further enhance the restoration quality and temporal stability, we introduce the facial prior learning and the self-referred refinement as auxiliary strategies used for both training and inference.

- SOTA



## Introduction

![fig1](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/fig1.png)



- Q: Can training on multiple video FR tasks help?



### SVD v2v

逐帧 Decode

> Given a video x ∈ R N×3×H×W , we first encode each frame into the latent space, represented as z = E(x). In this latent space, we perform forward and reverse processes. The final generated video, x, is then obtained by decoding from this latent representation



## methods

![fig2](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/fig2.png)



- Q：LQ video 怎么输入？

LQ 直接过 VAE 提取 latent 和 noise concat -> finetune UNet `conv_in` layer，改输入通道数，验证了可以的

> a simple approach is to encode Vd into latent space and directly concatenate E(Vd) with the noise z
>
> Therefore, simply adopting pretrained VAE cannot encode all source videos into a proper and consistent latent space which can then be used to guide the diffusion U-Net.



### multi-task

多个任务使用 Task Embed 指定任务，通过提出的 Latent Regularization 实现共享一个 latent space

> To this end, we propose a unified face restoration framework comprising two key modules: Task Embedding and the Unified Latent Regularization.

搞一个 binary vector -> 也就三个任务

> While the source video Vd can indicate the required task to some extent, directly relying on Vd for task information can lead to confusion. 
>
> Given a task set T = {T1, T2, T3}, where T1, T2 and T3 represent BFR, colorization, and inpainting tasks respectively
>
> To enhance the model’s ability to recognize tasks, each task is represented by a binary indicator ti(i ∈ [1, 3]) (0 for absence and 1 for presence). The task prompt is represented as a binary vector γ = [t1, t2, t3]. For example, [0, 1, 1] indicates that colorization and inpainting tasks are active while BFR is inactive. 



提取 UNet middle layer 特征，认为有很重要的特征。。。拿这个特征做 contrastive loss，搞同一个视频另一种 DA 作为正样本，其他视频作为负样本

> To further align features across different tasks and enable the model to leverage shared knowledge within a unified learning framework, we propose Unified Latent Regularization (ULR).
>
> we implement cross-task alignment using features from the middle block of the UNet. We first extract the output from the intermediate layers of the UNet, as these layers capture essential structural and semantic information crucial for modeling nuanced video details.

![eq2](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/eq2.png)



### Facial Prior

只用 Diffusion Loss 没法优化特定人脸区域，loss 太泛化没有针对性

> While the pretrained SVD can be directly adapted to our task by optimizing the noise prediction loss as in Eq. 1, such an objective cannot help inject the structure priors of human faces into the model.

拿 UNet mid features 过几层 MLP 预测 68 个人脸关键点，在训练时候搞一个新的 Loss

> Concretely, we leverage the features xd output from the **U-Net middle block and pass them through a landmark predictor Plm composed of an average pooling layer followed by five layers of MLP.** This predictor is trained to estimate a 68-points facial landmark set, obtained from ground truth frames using a pretrained landmark detection model [16]
>
> - "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks" 
>   https://arxiv.org/pdf/1711.06753

![eq3](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/eq3.png)



完整 loss

![eq4](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/eq4.png)



### self-refine

先生成一波，找一个 HQ 的一帧作为 reference 再修复一次

- Q：训练细节？

> In the training phase, we extract features from the reference frame Iref using a VAE encoder zref = E(Iref ). These features are then injected into the initial noise of the U-Net model
>
> l. Additionally, identity features fid are extracted by Eid and passed through a mapping network Mid before being injected into the cross-attention layers of the U-Net

实现 long consistency





### Code

> https://github.com/wangzhiyaoo/SVFR/blob/3a4539054281b1d3765e427c2294dd1776f0a57e/src/pipelines/pipeline.py#L75





## setting

- SVD

- We selected VoxCeleb2 [11], CelebV-Text [47], and VFHQ [39] as our training datasets

- we utilize ARNIQA [1] score to **filtere a highquality video dataset** comprising 20,000 clips

- 对比 5 个方法，3个数据集

- PSNR, SSIM, LPIPS

- IDS

- VIDD [10] measures temporal consistency by calculating identity feature differences between adjacent frames using ArcFace

- FVD [31] assesses overall video quality, reflecting spatial and temporal coherence. 

  https://github.com/ragor114/PyTorch-Frechet-Video-Distance?tab=readme-ov-file





## Experiment

> ablation study 看那个模块有效，总结一下

![tb2](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/tb2.png)



- 这么模糊。。。只看主观了。。。没关注纹理一致性，**看生成的文字超级垃圾**
- 时序也不太行。。看挑出来的图，**那个文字在多帧内很闪啊。。。。**

![fig3](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/fig3.png)





- 时序一致性，划条线看看
- 加了 refer 特征。。人脸细节看起来没啥变化，**颜色变化倒是明显一点**

![fig5](docs/2025_01_Arxiv_SVFR--A-Unified-Framework-for-Generalized-Video-Face-Restoration_Note/fig5.png)



## Limitations

- 同一个 scene 从修复结果找 reference，**一致性和 GT 没法保证**



## Summary :star2:

> learn what

### how to apply to our task

- Q：SVD 怎么 video2video?
- Q：condition 怎么引入？
- 数据集

