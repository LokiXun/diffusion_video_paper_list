# Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

> "Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models" NIPS, 2023 May
> [paper](http://arxiv.org/abs/2305.16322v3) [code](https://github.com/ShihaoZhaoZSH/Uni-ControlNet) 
> [pdf](./2023_05_NIPS_Uni-ControlNet--All-in-One-Control-to-Text-to-Image-Diffusion-Models.pdf) [note](2023_05_NIPS_Uni-ControlNet--All-in-One-Control-to-Text-to-Image-Diffusion-Models_Note.md)
> Authors: Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong

## Key-point

- Task
- Problems
- :label: Label:

实现多种 condition 同时控制

 allows for the simultaneous utilization of different local controls (e.g., edge maps, depth map, segmentation masks) and global controls (e.g., CLIP image embeddings) in a flexible and composable manner within one model. 

只需要训练两个 adapter，训练量小

only requires the fine-tuning of two additional adapters upon frozen pre-trained text-to-image diffusion models, eliminating the huge cost of training from scratch



## Contributions

## Introduction

## methods

![image-20240202213357705](docs/2023_05_NIPS_Uni-ControlNet--All-in-One-Control-to-Text-to-Image-Diffusion-Models_Note/Uni-ControlNet_framework.png)





## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

