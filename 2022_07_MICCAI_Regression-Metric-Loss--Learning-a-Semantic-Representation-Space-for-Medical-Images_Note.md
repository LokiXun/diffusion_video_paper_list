# Regression Metric Loss: Learning a Semantic Representation Space for Medical Images

> "Regression Metric Loss: Learning a Semantic Representation Space for Medical Images" MICCAI, 2022 Jul
> [paper](http://arxiv.org/abs/2207.05231v1) [code](https://github.com/DIAL-RPI/Regression-Metric-Loss) 
> [pdf](./2022_07_MICCAI_Regression-Metric-Loss--Learning-a-Semantic-Representation-Space-for-Medical-Images.pdf) [note](./2022_07_MICCAI_Regression-Metric-Loss--Learning-a-Semantic-Representation-Space-for-Medical-Images_Note.md)
> Authors: Hanqing Chao, Jiajin Zhang, Pingkun Yan

## Key-point

- Task: regression task (calcium score estimation and bone age assessment)

- Problems

  high-dimensional feature representation learned by existing popular loss functions like Mean Squared Error or L1 loss is hard to interpret

- :label: Label:

 propose a novel **Regression Metric Loss (RM-Loss)**, which endows the representation space with the semantic meaning of the label space by finding a representation manifold that is isometric to the label space



## Contributions

## Introduction

## methods

> - Riemannian manifold (黎曼流形) 为一个空间
>
>   manifold (流形) 是局部具有欧几里得空间性质的空间，是**高维空间中曲线、曲面概念的拓广**：
>
>   可以在低维上直观理解这个概念，比如我们说三维空间中的一个曲面是一个二维流形，因为它的本质维度（intrinsic dimension）只有2，**一个点在这个二维流形上移动只有两个方向的自由度**。同理，三维空间或者二维空间中的一条曲线都是一个一维流形。欧几里得空间就是最简单的流形的实例。

assume that the label space 𝑌 is a Euclidean space



## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

