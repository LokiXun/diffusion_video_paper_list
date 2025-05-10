# How Far is Video Generation from World Model: A Physical Law Perspective

> "How Far is Video Generation from World Model: A Physical Law Perspective" Arxiv, 2024 Nov 4
> [paper](http://arxiv.org/abs/2411.02385v1) [code](https://phyworld.github.io) [pdf](./2024_11_Arxiv_How-Far-is-Video-Generation-from-World-Model--A-Physical-Law-Perspective.pdf) [note](./2024_11_Arxiv_How-Far-is-Video-Generation-from-World-Model--A-Physical-Law-Perspective_Note.md)
> Authors: Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, Jiashi Feng

## Key-point

- Task: 探索 Sora 是否能学习到物理规律
- Problems
- :label: Label:

## Contributions

- The model generalises **perfectly** for in-distribution data, but **fails** to do out-of-distribution generalization. For combinatorial scenarios, **scaling law** is observed. 
- The models **fail to abstract general rules** and instead tries to mimic the closest training example.
- The model prioritizes different attributes when referencing training data: **color > size > velocity > shape.**



## Introduction

## methods

## setting

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what

### how to apply to our task

- 验证 world model 发现 “失败” 了，如何去观察结果分析，得到只是在拟合训练数据的结论
  - 要对训练数据非常熟悉，知道哪些样例好
  - 学习下可视化
