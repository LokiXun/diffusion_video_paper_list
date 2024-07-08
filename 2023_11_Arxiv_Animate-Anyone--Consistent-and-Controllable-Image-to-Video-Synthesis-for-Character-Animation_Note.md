# Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation

> "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation" Arxiv, 2023 Nov 28
> [paper](http://arxiv.org/abs/2311.17117v2) [code]() [pdf](./2023_11_Arxiv_Animate-Anyone--Consistent-and-Controllable-Image-to-Video-Synthesis-for-Character-Animation.pdf) [note](./2023_11_Arxiv_Animate-Anyone--Consistent-and-Controllable-Image-to-Video-Synthesis-for-Character-Animation_Note.md)
> Authors: Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, Liefeng Bo

## Key-point

- Task
- Problems
- :label: Label:

## Contributions

## Introduction

### DWPose

> https://github.com/IDEA-Research/DWPose.git

- "Effective Whole-body Pose Estimation with Two-stages Distillation" ICCV 2023
  [paper](https://arxiv.org/abs/2307.15880) [code](https://github.com/IDEA-Research/DWPose) [demo](https://openxlab.org.cn/apps/detail/mmpose/RTMPose)

使用 DWPose 提取骨架再输入网络，支持部分身体

- Q：训练数据？

Prepare [COCO](https://cocodataset.org/#download) in mmpose/data/coco and [UBody](https://github.com/IDEA-Research/OSX) in mmpose/data/UBody.

UBody 是视频数据！





## methods

## Setting

We employ DWPose[52] to extract the pose sequence of characters in the video, including the body and hands, rendering it as pose skeleton images following OpenPose[5]



**Dataset**





## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

