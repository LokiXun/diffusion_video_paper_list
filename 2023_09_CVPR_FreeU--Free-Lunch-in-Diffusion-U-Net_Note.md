# FreeU: Free Lunch in Diffusion U-Net

> "FreeU: Free Lunch in Diffusion U-Net" CVPR-oral, 2023 Sep 20 :star:
> [paper](http://arxiv.org/abs/2309.11497v2) [code](https://chenyangsi.top/FreeU/) [pdf](./2023_09_CVPR_FreeU--Free-Lunch-in-Diffusion-U-Net.pdf) [note](./2023_09_CVPR_FreeU--Free-Lunch-in-Diffusion-U-Net_Note.md)
> Authors: Chenyang Si, Ziqi Huang, Yuming Jiang, Ziwei Liu

## Key-point

- Task:  improves diffusion model sample quality
- Problems
- :label: Label:

## Contributions

> **improves diffusion model sample quality** at no costs: no training, no additional parameter introduced, and no increase in memory or sampling time.

可视化发现 U-Net Encoder 的残差主要是高频信息，含有较多噪声。因此先用 FFT 和 IFFT 变换降低高频信息，将 UNet decoder 特征乘个系数（加大权重）再 concat

## Introduction

发现低频信息变化更慢

![fig2](docs/2023_09_CVPR_FreeU--Free-Lunch-in-Diffusion-U-Net_Note/fig2.png)

## methods

![fig4](docs/2023_09_CVPR_FreeU--Free-Lunch-in-Diffusion-U-Net_Note/fig4.png)



原始 UNet 特征乘上一个权值 b，高频 residual 权重 s

1. 如果低频少的太多 b=0.6 s=1 ，OR 低频加多了结构都会变。b 权重不要动
2. 就去补足 diffusion denoising 过程中的高频特征就好

![fig5-6](docs/2023_09_CVPR_FreeU--Free-Lunch-in-Diffusion-U-Net_Note/fig5-6.png)



## setting

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what

### how to apply to our task

