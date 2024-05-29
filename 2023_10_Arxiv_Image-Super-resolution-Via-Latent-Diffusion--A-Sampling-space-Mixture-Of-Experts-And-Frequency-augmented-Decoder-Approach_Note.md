# Image Super-resolution Via Latent Diffusion: A Sampling-space Mixture Of Experts And Frequency-augmented Decoder Approach

> "Image Super-resolution Via Latent Diffusion: A Sampling-space Mixture Of Experts And Frequency-augmented Decoder Approach" Arxiv, 2023 Oct
> [paper](http://arxiv.org/abs/2310.12004v3) [code](https://github.com/amandaluof/moe_sr.) 
> [pdf](./2023_10_Arxiv_Image-Super-resolution-Via-Latent-Diffusion--A-Sampling-space-Mixture-Of-Experts-And-Frequency-augmented-Decoder-Approach.pdf) [note](./2023_10_Arxiv_Image-Super-resolution-Via-Latent-Diffusion--A-Sampling-space-Mixture-Of-Experts-And-Frequency-augmented-Decoder-Approach_Note.md)
> Authors: （Tencent AI Lab）Feng Luo, Jinxi Xiang, Jun Zhang, Xiao Han, Wei Yang

## Key-point

- Task: Image SR (x4, x8)

- Problems

  1. 发现 the compression of latent space usually causes reconstruction distortion

     propose a frequency compensation module that enhances the frequency components from latent space to pixel space

  2.  huge computational cost constrains the parameter scale of the diffusion model

      use Sample-Space Mixture of Experts (SS-MoE) to achieve more powerful latent-based SR

- :label: Label: diffusion prior



## Contributions

## Introduction

## methods



### Sampling MoE

 It is important to note that the complexity of these denoising steps fluctuates based on the level of noise present in the image

低质图像噪声程度不同，去噪步数理论上也要不同

divide all timesteps uniformly into N stages consisting of consecutive timesteps and assign single Sampling Expert to one stage

4 个模型，only a single expert network is activated at each step 计算资源和只用一个模型一样



## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

