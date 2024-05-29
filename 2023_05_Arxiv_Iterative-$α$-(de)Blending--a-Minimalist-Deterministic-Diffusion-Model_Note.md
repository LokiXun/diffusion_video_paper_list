# Iterative $α$-(de)Blending: a Minimalist Deterministic Diffusion Model

> "Iterative $α$-(de)Blending: a Minimalist Deterministic Diffusion Model" Arxiv, 2023 May
> [paper](http://arxiv.org/abs/2305.03486v1) [code]() 
> [pdf](./2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model.pdf)
> Authors: Eric Heitz, Laurent Belcour, Thomas Chambon

## Key-point

- Task
- Problems
- :label: Label:



## Contributions

## Introduction

- 为什么要用两个样本？

  为了推广原始 diffusion，希望针对不是高斯噪声的分布也满足

  黑白图像到彩色图像的映射

![image-20240201181313024](docs/2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model_Note/image-20240201181313024.png)

想证明两个样本的关系，在某种变换下可以变得确定

![image-20240201181836712](docs/2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model_Note/image-20240201181836712.png)

模型 $D_\theta$ 根据两个样本学习两个样本期望的差值。





## methods



## Experiment

> ablation study 看那个模块有效，总结一下
>
> 学习如何发现问题，论证问题

在不同分布上验证效果

![image-20240201182121369](docs/2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model_Note/image-20240201182121369.png)

1. 一维分布上，验证 L2 范数更好



从不同分布开始的生成结构

![image-20240201182528582](docs/2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model_Note/image-20240201182528582.png)

非高斯分布

![image-20240201183000998](docs/2023_05_Arxiv_Iterative-$α$-(de)Blending--a-Minimalist-Deterministic-Diffusion-Model_Note/image-20240201183000998.png)



## Limitations



## Summary :star2:

> learn what & how to apply to our task

1. 学习一下实验设计，不同分布上效果的可视化
2. 以图到图直接的映射



