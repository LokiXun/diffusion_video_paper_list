# TransPixar: Advancing Text-to-Video Generation with Transparency

> "TransPixar: Advancing Text-to-Video Generation with Transparency" Arxiv, 2025 Jan 6
> [paper](http://arxiv.org/abs/2501.03006v1) [code](https://github.com/wileewang/TransPixar) [web](https://wileewang.github.io/TransPixar/) [pdf](./2025_01_Arxiv_TransPixar--Advancing-Text-to-Video-Generation-with-Transparency.pdf) [note](./2025_01_Arxiv_TransPixar--Advancing-Text-to-Video-Generation-with-Transparency_Note.md)
> Authors: Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, Yingcong Chen

## Key-point

- Task: 生成 PNG 视频，加个透明通道
- Problems
- :label: Label:

## Contributions

- 用 DiT 在少量数据上做 RGBA 生成

> We propose an RGBA video generation framework using DiT models that requires limited data and training parameters, achieving diverse generation with strong alignment.

- 分析 Attention Modules 的作用

> We analyze the role of each attention component in the generation process, optimize their interactions, and introduce necessary modifications to improve RGBA generation quality

- effectiveness

> Our method is validated through extensive experiments, demonstrating its effectiveness across a variety of challenging scenarios.





## Introduction

想要把生成的内容，作为一个组件加入其他场景；烟雾的透明通道不好搞

> However, generating RGBA video, which includes alpha channels for transparency, remains a challenge due to limited datasets and the difficulty of adapting existing models.
>
> Alpha channels are crucial for visual effects (VFX), allowing transparent elements like smoke and reflections to blend seamlessly into scenes.



## methods

## setting

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what

### how to apply to our task

