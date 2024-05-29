# CLIP-DINOiser: Teaching CLIP a few DINO tricks

> "CLIP-DINOiser: Teaching CLIP a few DINO tricks" Arxiv, 2023 Dec
> [paper](http://arxiv.org/abs/2312.12359v1) [code](https://github.com/wysoczanska/clip_dinoiser.) 
> [pdf](./2023_12_Arxiv_CLIP-DINOiser--Teaching-CLIP-a-few-DINO-tricks.pdf)
> Authors: Monika Wysoczańska, Oriane Siméoni, Michaël Ramamonjisoa, Andrei Bursuc, Tomasz Trzciński, Patrick Pérez

## Key-point

- Task
- Problems
- :label: Label:



- 发现 CLIP 特征 lack of spatial awareness makes it unsuitable for dense computer vision tasks, 但自监督方法得到的特征对空间信息很好 self-supervised representation methods have demonstrated good localization properties without human-made annotations nor explicit supervision

propose a **zero-shot open-vocabulary semantic segmentation method**, which does not require any annotations. We propose to locally improve dense MaskCLIP features, computed with a simple modification of CLIP’s last pooling layer, by integrating localization priors extracted from self-supervised features.

效果：CLIP-DINOiser needs only a single forward pass of CLIP and two light convolutional layers at inference, no extra supervision nor extra memory and reaches state-ofthe-art results



## Contributions

## Introduction

## methods

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

