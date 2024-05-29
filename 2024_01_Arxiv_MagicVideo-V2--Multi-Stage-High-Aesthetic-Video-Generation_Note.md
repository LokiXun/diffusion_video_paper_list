# MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation

> "MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation" Arxiv, 2024 Jan
> [paper](http://arxiv.org/abs/2401.04468v1) [code]() 
> [pdf](./2024_01_Arxiv_MagicVideo-V2--Multi-Stage-High-Aesthetic-Video-Generation.pdf)
> Authors: Weimin Wang, Jiawei Liu, Zhijie Lin, Jiangqiao Yan, Shuo Chen, Chetwin Low, Tuyen Hoang, Jie Wu, Jun Hao Liew, Hanshu Yan, Daquan Zhou, Jiashi Feng

## Key-point

- Task
- Problems
- :label: Label:

## Contributions

## Introduction

## methods

- motion module 使用 Animatediff 里面的
- deploy a ControlNet [14] module to directly extract RGB information from the reference image and apply it to all frames. These techniques align the the frames with the reference image well while allowing the model to generate clear motion



### V2V Module

It shares the same backbone and spatial layers as in I2V module. Its motion module is separately finetuned using a high-resolution video subset for video super-resolution

The image apperance encoder and ControlNet module are also used here. 

### VFI

VFI module uses an internally trained GAN based VFI model.



## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

