# Video Swin Transformers for Egocentric Video Understanding @ Ego4D Challenges 2022

> "Video Swin Transformers for Egocentric Video Understanding @ Ego4D Challenges 2022" Arxiv, 2022 Jul
> [paper](http://arxiv.org/abs/2207.11329v1) [code]() 
> [pdf](./2022_07_Arxiv_Video-Swin-Transformers-for-Egocentric-Video-Understanding-@-Ego4D-Challenges-2022.pdf)
> Authors: Maria Escobar, Laura Daza, Cristina González, Jordi Pont-Tuset, Pablo Arbeláez

## Key-point

- Task
- Problems
- :label: Label:



## Contributions



## Introduction



## methods

instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off.

在 Temporal 维度增加一个 window，T=2

![image-20231128173441880](docs/2022_07_CVPR_Video-Swin-Transformers_Note/video_swin_transformer.png)

输入 `(b t c h w)` , PatchPartition 按 `(2, 4, 4)` 拉直得到 `b, 96, t/2, h/4, w/4` 特征 （`96 == (2*4*4*3)`）

整个过程模仿了 CNN 模块中的下采样过程，这也是为了让模型可以针对不同尺度生成特征。**浅层可以看到小物体**，深层则着重关注大物体。

```python
depths=[2, 2, 6, 2],
num_heads=[8, 8, 8, 8],
window_size=(2, 8, 8),  >> shift_size = tuple(i // 2 for i in window_size)
mlp_ratio=4.,
drop_path_rate=0.2
```

1. cycle shift

2. window partition  

   在 T 维度窗口 window=2, 移动一帧，等于只有前后帧信息



## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

